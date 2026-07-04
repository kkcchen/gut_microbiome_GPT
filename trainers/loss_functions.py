"""
Implementation of the necessary loss functions and the overall compute_loss
"""
import torch
from typing import Dict, Optional
from trainers import logger
import torch.nn.functional as F
from skbio.stats.composition import closure


def masked_mse_loss_counts(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, log_transform: bool = False
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    # take the log of the raw counts
    if log_transform:
        target = torch.log(target + 1e-6)
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / (mask.sum() + 1e-4)

def masked_binary_ce_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute masked binary cross entropy loss.
    Target should be continuous counts (will be binarized) or already binary.
    """
    target_binary = (target > 0).float()
    mask = mask.float()
    # input are logits
    loss = F.binary_cross_entropy_with_logits(input * mask, target_binary * mask, reduction="sum")
    return loss / (mask.sum() + 1e-6)

def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, log_transform: bool = False
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    if log_transform:
        target = torch.log(target + 1e-6)
    else:
        # convert counts to relative abundance
        target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)
    
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / (mask.sum() + 1e-6)

def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-4)
    return loss.mean()
  

# from chatGPT 
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Standard SimCLR / NT-Xent loss
    z1, z2: (B, D)
    """
    assert z1.ndim == 2 and z2.ndim == 2 and z1.shape == z2.shape
    B, _ = z1.shape
    device = z1.device

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    out = torch.cat([z1, z2], dim=0)  # (2B, D)

    # logits: (2B, 2B)
    logits = (out @ out.T) / temperature

    # mask self-similarity
    logits.fill_diagonal_(-torch.inf)

    # positives: for i in [0..B-1], pos is i+B; for i in [B..2B-1], pos is i-B
    labels = torch.arange(2 * B, device=device)
    labels = (labels + B) % (2 * B)

    return F.cross_entropy(logits, labels)


def mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    relative: bool = False,
) -> torch.Tensor:
    """
    Compute the MSE loss between input and target.
    """
    if relative:
        weight = torch.sqrt(1/(target + 1e-4))
        loss = F.mse_loss(weight * input, weight * target, reduction="mean")
    else:
        loss = F.mse_loss(input, target, reduction="mean")
    return loss

def denoising_reconstruction_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    relative: bool = True
) -> torch.Tensor:
    """
    Cross-entropy between output logits and target counts (relative abundance if relative==True).
    If relative is True, make target compositional and use softmax, otherwise raw counts (and softplus (?)).
    """
    if relative:
        # Avoid skbio.closure and round-trip to CPU/NumPy
        target_comp = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        loss = F.cross_entropy(input, target_comp)
    else:
        print("have not implemented denoising loss for raw counts yet.")
    return loss


def dm_nll_loss(
    scale: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    '''
    Negative log-likelihood of the data under dirichlet multinomial parameterized
    by the model outputs, optimized for numerical stability.
    '''
    # scale is (B,), logits is (B, L)
    p = F.softmax(logits, dim=-1)
    alpha = scale.unsqueeze(-1) * p + 1e-7  
    N = target.sum(dim=-1, keepdim=True)  # total counts per sample
    alpha_0 = alpha.sum(dim=-1, keepdim=True)
    
    # log_gamma(N+1) - sum(log_gamma(target+1)) is the multinomial coefficient part
    # lgamma(alpha_0) - lgamma(N + alpha_0) + sum(lgamma(target + alpha) - lgamma(alpha))
    
    logp = torch.lgamma(alpha_0) - torch.lgamma(N + alpha_0) + \
           (torch.lgamma(target + alpha) - torch.lgamma(alpha)).sum(dim=-1, keepdim=True)
    
    # The multinomial coefficient is constant wrt parameters if we are just doing ML on DM
    # but for completeness:
    # log_multinomial = torch.lgamma(N + 1) - torch.lgamma(target + 1).sum(dim=-1, keepdim=True)
    # logp = logp + log_multinomial

    # Weighting per sample or per count? Per count (normalized by N) is often more stable.
    # Clip N to avoid division by zero
    N_clipped = N.clamp(min=1.0)
    return (-logp / N_clipped).mean()

def xe_smoothed_loss(logits, target_probs, positions_to_count=None, T=2.0, eps=1e-8):
    # Smooth target with temperature
    if not positions_to_count:
        positions_to_count = torch.ones_like(logits)
    out_probs = F.softmax(logits, dim=-1)
    log_out_probs = torch.log(out_probs) - torch.log((out_probs * positions_to_count).sum(dim=-1, keepdim=True).clamp_min(1e-8))
    
    target_probs_mask = (target_probs * positions_to_count) ** (1/T)
    target_probs_norm = target_probs_mask / target_probs_mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    
    return -(log_out_probs * target_probs_norm).sum()
    

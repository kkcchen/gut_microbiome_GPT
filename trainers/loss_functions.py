"""
Implementation of the necessary loss functions and the overall compute_loss
"""
import torch
from typing import Dict, Optional
from trainers import logger
import torch.nn.functional as F
from skbio.stats.composition import closure


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, log_transform: bool = True
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

def zinb_nll_loss(
    mean: torch.Tensor,
    disp: torch.Tensor,
    pi: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ZINB negative log-likelihood loss.
    mean, disp, pi: predicted parameters from the model, shape batch_size x n_taxa
    target: original counts
    """
    eps = 1e-8
    mean = mean + eps
    disp = disp + eps

    # Log likelihood for NB
    t1 = torch.lgamma(disp + target) - torch.lgamma(disp) - torch.lgamma(target + 1)
    t2 = disp * (torch.log(disp) - torch.log(disp + mean))
    t3 = target * (torch.log(mean) - torch.log(disp + mean))
    nb_case = t1 + t2 + t3

    # Log likelihood for zero inflation
    zero_nb = torch.pow(disp / (disp + mean), disp)
    zero_case = torch.log(pi + (1.0 - pi) * zero_nb + eps)

    # Combine cases
    result = torch.where(target < 1e-8, zero_case, torch.log(1.0 - pi + eps) + nb_case)

    loss = -result
    return loss.sum() / target.shape[1]  # average over taxa

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
) -> torch.tensor:
    """
    Cross-entropy between output logits and target counts (relative abundance if relative==True).
    If relative is True, make target compositional and use softmax, otherwise raw counts (and softplus (?)).
    """
    if relative:
        target_comp = torch.tensor(closure(target.cpu().numpy()), device=target.device)
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
    by the model outputs
    '''
    p = F.softmax(logits, dim=-1)
    alpha = scale.unsqueeze(-1) * p + 1e-8  
    N = target.sum(dim=-1)  # total counts per sample
    alpha_0 = alpha.sum(dim=-1)
    logp = torch.lgamma(N + 1) + torch.lgamma(alpha_0) - torch.lgamma(N + alpha_0) + \
              torch.lgamma(target + alpha).sum(dim=-1) - torch.lgamma(target + 1).sum(dim=-1) - torch.lgamma(alpha).sum(dim=-1)

    # Weighting this by N to avoid over-emphasizing high-depth samples
    # TODO: consider using the average log-likelihood per count instead of per sample
    # TODO make the normalization optional
    return (-logp/N).mean()

def xe_smoothed_loss(logits, target_probs, positions_to_count=None, T=2.0, eps=1e-8):
    # Smooth target with temperature
    if not positions_to_count:
        positions_to_count = torch.ones_like(logits)
    out_probs = F.softmax(logits, dim=-1)
    log_out_probs = torch.log(out_probs) - torch.log((out_probs * positions_to_count).sum(dim=-1, keepdim=True).clamp_min(1e-8))
    
    target_probs_mask = (target_probs * positions_to_count) ** (1/T)
    target_probs_norm = target_probs_mask / target_probs_mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    
    return -(log_out_probs * target_probs_norm).sum()
    

"""
Implementation of the necessary loss functions and the overall compute_loss
"""
import torch
from typing import Dict, Optional
from trainers import logger
import torch.nn.functional as F


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
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
    loss = F.mse_loss(input, target, reduction="mean")
    return loss
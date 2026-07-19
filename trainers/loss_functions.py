"""
Implementation of the necessary loss functions and the overall compute_loss
"""
import torch
from typing import Dict, Optional
from trainers import logger
import torch.nn.functional as F


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
    loss = F.binary_cross_entropy_with_logits(input, target_binary, reduction="none")
    loss = (loss * mask).sum()
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

def masked_ce_loss(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute cross-entropy loss over a vocabulary, restricted to masked positions.
    logits: (B, L, V), target: (B, L) integer class ids, mask: (B, L) bool.
    """
    mask_flat = mask.reshape(-1)
    if not mask_flat.any():
        # no masked positions in this batch; keep the value differentiable and zero
        return logits.sum() * 0.0
    logits_flat = logits.reshape(-1, logits.size(-1))[mask_flat]
    target_flat = target.reshape(-1)[mask_flat].long()
    return F.cross_entropy(logits_flat, target_flat)

def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-4)
    return loss.mean()
  

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

def xe_smoothed_loss(logits, target_probs, positions_to_count=None, T=2.0, eps=1e-8):
    # Smooth target with temperature
    if not positions_to_count:
        positions_to_count = torch.ones_like(logits)
    out_probs = F.softmax(logits, dim=-1)
    log_out_probs = torch.log(out_probs) - torch.log((out_probs * positions_to_count).sum(dim=-1, keepdim=True).clamp_min(1e-8))
    
    target_probs_mask = (target_probs * positions_to_count) ** (1/T)
    target_probs_norm = target_probs_mask / target_probs_mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    
    return -(log_out_probs * target_probs_norm).sum()
    

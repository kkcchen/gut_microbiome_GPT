"""
Implementation of the necessary loss functions and the overall compute_loss
"""
import torch
from typing import Dict, Optional
from trainers import logger
import torch.nn.functional as F


def compute_loss(
                    outputs,
                    targets,
                    cfg,
                    vocab
                 ) -> tuple[torch.Tensor, Dict[str, float]]:
    '''
    Compute the loss for microbiome representation learning.
    :param outputs: Model outputs, dict containing the various different outputs
    :param targets: Ground truth targets, dict containing the various different targets
    :param cfg: Configuration object.
    :param vocab: Vocabulary object for taxa 
    :return: tuple of (loss tensor, metrics dictionary)
    '''
    loss = 0.0
    metrics = {}
    
    # Expression reconstruction loss 
    if cfg.tasks.do_reconstruction:
        recon_output = outputs['abundance']
        recon_target = targets['original_counts']
        recon_mask = targets['expressed_mask']
        
        recon_loss = masked_mse_loss(
            recon_output,
            recon_target,
            recon_mask
        )
        metrics['reconstruction_loss'] = recon_loss.item()
        loss += cfg.params.reconstruction_loss_weight * recon_loss
    
    
    # Contrastive loss
    if cfg.tasks.do_contrastive:
        z1 = outputs['view1']['cell_emb_proj']
        z2 = outputs['view2']['cell_emb_proj']  
        contrastive_loss = nt_xent(
            z1, z2
        )
        metrics['contrastive_loss'] = contrastive_loss.item()
        loss += cfg.params.contrastive_loss_weight * contrastive_loss
    
    
    return loss, metrics
        




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
def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Standard SimCLR / NT-Xent for a single process (no cross-process gather).
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
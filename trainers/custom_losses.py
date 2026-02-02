# From scGPT
import torch
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


def env_contrastive_loss(
    input1: torch.Tensor, input2: torch.Tensor, sim_temp: float
) -> torch.Tensor:
    """Compute the environment contrastive loss between two inputs.
    
    Args:
        input1 (torch.Tensor): First input tensor. (batch_size, embedding_dim)
        input2 (torch.Tensor): Second input tensor. (batch_size, embedding_dim)
        sim_temp (float): Temperature for the similarity computation.
    """
    assert input1.shape == input2.shape, "Input tensors must have the same shape."
    
    # Normalize the inputs
    input1 = F.normalize(input1, dim=-1)
    input2 = F.normalize(input2, dim=-1)
    
    # Compute cosine similarity
    sim_matrix = torch.mm(input1, input2.T) / sim_temp
    
    # Create labels for contrastive loss
    batch_size = sim_matrix.size(0)
    labels = torch.arange(batch_size, device=sim_matrix.device)
    
    # Compute contrastive loss
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-4))
    return loss.mean()

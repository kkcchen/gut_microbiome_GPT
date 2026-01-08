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
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-4)
    return loss.mean()


# from chatgpt
def nt_xent_loss_accelerate(
    z1_local: torch.Tensor,
    z2_local: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    NT-Xent / InfoNCE for paired views.

    z1_local, z2_local: (B, D) embeddings for view1/view2 on THIS process.
    Positives are aligned by row index within each process/batch.
    Negatives are all other samples across all processes (and batch).

    Returns a scalar loss (average over local batch).
    """
    assert z1_local.ndim == 2 and z2_local.ndim == 2
    assert z1_local.shape == z2_local.shape

    # normalize
    z1_local = F.normalize(z1_local, dim=1)
    z2_local = F.normalize(z2_local, dim=1)

    # Gather across processes (includes local). Accelerate returns a tensor where
    # batch dim = world_size * local_batch (assuming equal batch per rank; drop_last=True helps).
    # z1_all = accelerator.gather(z1_local)
    # z2_all = accelerator.gather(z2_local)

    # logits for local anchors against all candidates in the other view
    logits_12 = (z1_local @ z2_local.t()) / temperature
    logits_21 = (z2_local @ z1_local.t()) / temperature

    # Determine where THIS rank's samples sit in the gathered tensor
    # local batch size:
    B = z1_local.shape[0]
    # rank index:
    # rank = accelerator.process_index
    # positive indices in gathered tensors for local samples
    labels = torch.arange(B, device=z1_local.device)# + rank * B

    loss_12 = F.cross_entropy(logits_12, labels)
    loss_21 = F.cross_entropy(logits_21, labels)

    return 0.5 * (loss_12 + loss_21)

# From Gemini
def nt_xent_intra_view(z1, z2, temperature=0.2):
    batch_size = z1.shape[0]
    
    # 1. Normalize and Concatenate
    # out shape: (2 * B, D)
    out = torch.cat([z1, z2], dim=0)
    out = F.normalize(out, dim=1)

    # 2. Compute full similarity matrix (2B, 2B)
    sim_matrix = torch.exp(torch.mm(out, out.t()) / temperature)
    
    # 3. Create a mask to remove self-similarity (diagonal)
    # We don't want a sample to be its own negative
    mask = ~torch.eye(2 * batch_size, device=z1.device).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # 4. Extract positives
    # For z1_i, the positive is z2_i (which is at index i + B)
    # For z2_i, the positive is z1_i (which is at index i)
    pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    # We concatenate twice because we are calculating loss for both directions
    positives = torch.cat([pos_sim, pos_sim], dim=0)

    # 5. Loss calculation: -log( pos / sum(all_others) )
    loss = -torch.log(positives / sim_matrix.sum(dim=-1))
    
    return loss.mean()


# different chatgpt version
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
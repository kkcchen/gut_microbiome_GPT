# TODO: do proper attribution from scGPT
import torch
import numpy as np
from torch import nn, Tensor
from typing import Optional

#TODO: try starting with embeddings of taxa. evo2? word2vec?
class TaxaEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_vocab_path: Optional[str] = None,
        freeze_vocab: bool = False,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        if init_vocab_path is not None:
            assert init_vocab_path.endswith('.npy'), "init_vocab_path must be a .npy file"
            print("Loading initial vocab from ", init_vocab_path)
            vocab = np.load(init_vocab_path)
            n, d_vocab = vocab.shape
            print(f"Loaded vocab of shape {vocab.shape}")
            # Now create embedding matrix of shape (n+3, d_vocab)
            self.embedding = nn.Embedding(num_embeddings, d_vocab, padding_idx=padding_idx)
            with torch.no_grad():
                self.embedding.weight[:n].copy_(torch.from_numpy(vocab))
            self.embedding.weight.requires_grad = not freeze_vocab
            self.proj = nn.Linear(d_vocab, embedding_dim)
            self.enc_norm = nn.LayerNorm(embedding_dim)
        else:
            self.embedding = nn.Embedding(
                num_embeddings, embedding_dim, padding_idx=padding_idx
            )
            self.proj = None
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        if self.proj is not None:
            x = self.proj(x)
        x = self.enc_norm(x)
        return x
    

# Positional encoding in the original code, but not used anywhere, so skipped


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, mask_value, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value
        self.mask_value = mask_value
        self.mask_embedding = nn.Embedding(1, d_model)  # Embedding for mask_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)

        # Handle mask_value
        mask_indices = (x == self.mask_value).squeeze(-1)
        if mask_indices.any():
            x[mask_indices] = 0  # Temporarily set mask_value to 0 for processing

        # Ensure values are within range
        assert torch.max(x) <= self.max_value, "Input values exceed max_value. too many bins?"

        # Process non-mask values
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)

        # Replace mask_value positions with mask embedding
        if mask_indices.any():
            x[mask_indices] = self.mask_embedding(torch.zeros(mask_indices.sum(), dtype=torch.long, device=x.device))

        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mask_value,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)
        self.mask_value = mask_value
        self.mask_embedding = nn.Embedding(1, embedding_dim)  # Embedding for mask_value

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()

        # Handle mask_value
        mask_indices = (x == self.mask_value)
        if mask_indices.any():
            x[mask_indices] = 0  # Temporarily set mask_value to 0 for processing

        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)

        # Replace mask_value positions with mask embedding
        if mask_indices.any():
            x[mask_indices] = self.mask_embedding(torch.zeros(mask_indices.sum(), dtype=torch.long, device=x.device))

        return x


class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


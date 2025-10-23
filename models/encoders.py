# TODO: do proper attribution from scGPT
import torch
import numpy as np
from torch import nn, Tensor
from typing import Optional

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from data_utils.graph_helpers import build_tg_data_from_taxon_df

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
            # Now create embedding matrix of shape (n, d_vocab)
            self.embedding = nn.Embedding(num_embeddings, d_vocab, padding_idx=padding_idx)
            with torch.no_grad():
                self.embedding.weight[:n].copy_(torch.from_numpy(vocab))
            self.embedding.weight.requires_grad = not freeze_vocab
            self.proj = nn.Linear(d_vocab, embedding_dim)
        else:
            self.embedding = nn.Embedding(
                num_embeddings, embedding_dim, padding_idx=padding_idx
            )
            self.proj = None
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  
        if self.proj is not None:
            x = self.proj(x) # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class TaxaGraphEncoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_special_tokens: int,
        num_taxa: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.node_embs = nn.Embedding(num_nodes, embedding_dim)
        self.num_taxa = num_taxa
        
        # 1. deal with taxa embeddings
        self.conv1 = GCNConv(embedding_dim, embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)
        
        # 2. special token embeddings
        assert padding_idx >= self.num_taxa, "Padding idx should be in special tokens range"
        self.special_embedding = nn.Embedding(
            num_special_tokens, embedding_dim, padding_idx=padding_idx-self.num_taxa
        )
            
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, graph_data: Data) -> torch.Tensor:
        """
        x: Tensor of shape (batch_size, seq_len)
        containing vocab indices (taxon + special tokens)
        edge_list: Tensor of shape (2, num_edges)
            containing edges of the graph
        vocabindex_to_nodeindex: Tensor of shape (num_taxa,)

        Returns:
            Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        edge_list = graph_data.edge_index  # (2, num_edges)
        vocabindex_to_nodeindex = graph_data.vocabindex_to_nodeindex

        # ---- 1. Run (or reuse) GCN on the graph ----
        if (not hasattr(self, "cached_node_embs")) or self.training:
            node_embs = self.conv1(self.node_embs.weight, edge_list)
            node_embs = torch.relu(node_embs)
            node_embs = self.conv2(node_embs, edge_list)
            node_embs = self.enc_norm(node_embs)  # (num_nodes, emb_dim)
            self.cached_node_embs = node_embs.detach() if not self.training else node_embs
        else:
            node_embs = self.cached_node_embs

        batch_size, seq_len = x.shape
        out = torch.zeros(batch_size, seq_len, node_embs.size(-1), device=x.device)

        # ---- 2. Split taxa vs special tokens ----
        taxa_mask = x < self.num_taxa
        special_mask = ~taxa_mask

        # ---- 3. Taxa embeddings (from graph nodes) ----
        if taxa_mask.any():
            taxa_indices = x[taxa_mask]
            node_indices = vocabindex_to_nodeindex[taxa_indices]
            out[taxa_mask] = node_embs[node_indices]

        # ---- 4. Special tokens ----
        if special_mask.any():
            special_indices = x[special_mask] - self.num_taxa
            out[special_mask] = self.special_embedding(special_indices)

        return out



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


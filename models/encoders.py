# TODO: do proper attribution from scGPT
import torch
import numpy as np
from torch import nn, Tensor
from typing import Optional
from torch_geometric.nn import GCN, GAT
from torch_geometric.data import Data
from sklearn.preprocessing import normalize

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
            # normalize the initialized embeddings
            vocab_normalized = normalize(vocab, norm='l2', axis=1)
            # Now create embedding matrix of shape (n, d_vocab)
            self.embedding = nn.Embedding(num_embeddings, d_vocab, padding_idx=padding_idx)
            with torch.no_grad():
                self.embedding.weight[:n].copy_(torch.from_numpy(vocab_normalized))
            # self.embedding.weight.requires_grad = not freeze_vocab
            self.n_initialized = n
            self.freeze_vocab = freeze_vocab
            self.proj = nn.Sequential(
                nn.Linear(d_vocab, embedding_dim),  # expand hidden layer width
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings, embedding_dim, padding_idx=padding_idx
            )
            self.embedding.weight.requires_grad = not freeze_vocab
            self.proj = None
            self.n_initialized = 0
            self.freeze_vocab = False
        self.enc_norm = nn.LayerNorm(embedding_dim)
        if self.freeze_vocab and self.n_initialized > 0:
            self._register_freeze_hook()

    def _register_freeze_hook(self):
        """
        Attaches a gradient hook to zero out gradients for the frozen portion.
        This works under Accelerate and DDP transparently.
        """
        # if self.embedding.weight.grad is not None:
        #     with torch.no_grad():
        #         # Zero out gradients for the first n_initialized embeddings
        #         self.embedding.weight.grad[:self.n_initialized] = 0
        def _mask_grad(grad):
            grad = grad.clone()
            grad[:self.n_initialized] = 0
            return grad
        self.embedding.weight.register_hook(_mask_grad)
        print(f"Hook registered: first {self.n_initialized} embeddings frozen")

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
        graph_type: str = "gat",
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.node_embs = nn.Embedding(num_nodes, embedding_dim)
        self.num_taxa = num_taxa
        
        # 1. deal with taxa embeddings
        self.graph_type = graph_type
        if graph_type == "gcn":
            self.conv_model = GCN(embedding_dim, embedding_dim, num_layers=2, norm="layer")
        elif graph_type == "gat":
            self.conv_model = GAT(embedding_dim, embedding_dim, num_layers=2, heads=4, norm="layer")
        
        # 2. special token embeddings
        assert padding_idx >= self.num_taxa, "Padding idx should be in special tokens range"
        self.special_embedding = nn.Embedding(
            num_special_tokens, embedding_dim, padding_idx=padding_idx-self.num_taxa
        )
               
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
            edge_attr = graph_data.edge_attr
            edge_weight = 2 / (2 ** edge_attr) # greater distance should mean less weight
            node_embs = self.conv_model(self.node_embs.weight, edge_list, edge_weight=edge_weight)
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

    def __init__(self, d_model: int, mask_value, dropout: float = 0.1, max_value: int = 512, freeze:bool=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value
        self.mask_value = mask_value
        self.mask_embedding = nn.Embedding(1, d_model)  # Embedding for mask_value
        if freeze:
            self._freeze_parameters()

    def _freeze_parameters(self):
        """Freeze all parameters in this module."""
        for param in self.parameters():
            param.requires_grad = False
        print("ContinuousValueEncoder: all parameters frozen")

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
        freeze: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)
        self.mask_value = mask_value
        self.mask_embedding = nn.Embedding(1, embedding_dim)  # Embedding for mask_value
        if freeze:
            self._freeze_parameters()
    
    def _freeze_parameters(self):
        """Freeze all parameters in this module."""
        for param in self.parameters():
            param.requires_grad = False
        print("CategoryValueEncoder: all parameters frozen")

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


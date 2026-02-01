from typing import Dict, Mapping, Optional, Tuple, Any, Union, List

import torch
import numpy as np
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli

from torch_geometric.data import Data
# from tqdm import trange

from functools import lru_cache

from .encoders import (
    TaxaEncoder,
    TaxaGraphEncoder,
    ContinuousValueEncoder,
    BatchLabelEncoder
)

from .decoders import (
    ExprDecoder,
)

# PLACEHOLDER FOR NOW

class hgmGPT(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        dropout: float = 0.5,
        abundance_emb_style: str = "continuous",
        use_gnn: bool = False,
        num_gnn_nodes: Optional[int] = None,
        tasks: List[str] = ["denoising"],
    ):
        """
        The base model for the human gut microbiome. This initializes the transformer based architecture for encoding abundance tables. 
        
        :param d_model: The number of expected features in the embeddings.
        :type d_model: int
        :param nhead: The number of heads in the attention mechanism.
        :type nhead: int
        :param d_hid: The dimension of the feedforward network model in attention. 
        :type d_hid: int
        :param nlayers: The number of transformer encoder layers.
        :type nlayers: int
        :param use_batch_labels: Whether to use batch labels for batch effect correction.
        :type use_batch_labels: bool
        :param num_batch_labels: The number of batch labels.
        :type num_batch_labels: Optional[int]
        :param dropout: The dropout rate.
        :type dropout: float
        :param abundance_emb_style: The style of abundance embedding to use.
        :type abundance_emb_style: str
        :param use_gnn: Whether to use a graph neural network for taxa encoding.
        :type use_gnn: bool
        :param num_gnn_nodes: The number of nodes in the graph neural network.
        :type num_gnn_nodes: Optional[int]
        :param tasks: The list of tasks to perform, e.g., denoising, bottleneck, contrastive
        :type tasks: List[str]
        """
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.use_batch_labels = use_batch_labels
        self.abundance_emb_style = abundance_emb_style # default, continuous, mentioned in paper. could try using category encoding but this is likely less expressive
        self.nhead = nhead
        self.tasks = tasks
        if self.abundance_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"abundance_emb_style should be one of category, continuous, scaling, "
                f"got {abundance_emb_style}"
            )
        
        # ================================ BUILD ENCODERS ================================
        self.use_gnn = use_gnn
        if use_gnn:
            assert num_gnn_nodes is not None, "num_gnn_nodes must be provided when use_gnn is True"
            self.taxa_encoder = TaxaGraphEncoder(num_gnn_nodes, 
                                            vocab_num_special_tokens, 
                                            vocab_len - vocab_num_special_tokens, 
                                            d_model, 
                                            padding_idx=vocab_pad_index)
        else:
            self.taxa_encoder = TaxaEncoder(vocab_len, d_model, init_vocab_path, freeze_vocab, padding_idx=vocab_pad_index)

        if self.abundance_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        else:
            print("Using scaling style for input embedding, just identity for now")
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)

        # Batch Encoder
        if use_batch_labels:
            assert num_batch_labels is not None, "num_batch_labels must be provided when use_batch_labels is True"
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)

        # TODO: also probably need some metadata encoder, not implemented yet
        # ================================================================================

        # ================================ BUILD TRANSFORMER =============================
        # TODO: potentially try cross attention
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # ================================================================================


        # ================================ BUILD DECODERS ================================

        # output shape: B, num_tokens (max_seq_len + sample_token + batch_id_token + ...), d_model
        # 1. denoising
        # expression decoder, this operates on all the taxa tokens
        if "denoising" in tasks:
            self.abundance_decoder = ExprDecoder(
                d_model=d_model,
                nfirst_tokens_to_skip=...,
                dropout=dropout,
                distribution=...,
                use_depth=...,
            )
        
        # 2. bottleneck 
        # TODO: zero out all the taxa tokens, leave only sample_token and batch_id_token, and essentially recreate the distribution for each taxa

    def encode(
        self,
        taxa_ids: Tensor,
        taxa_abundances: Tensor,
        batch_ids: Optional[Tensor] = None,
        graph_data: Optional[Data] = None,
    ) -> Tensor:
        """
        This is the function that runs the encoder part of the model. Includes taxa/value encoding and transformer encoding. 
        Args:
            taxa_ids (Tensor): The taxa ids tensor of shape (batch, seq_len).
            taxa_values (Tensor): The taxa values tensor of shape (batch, seq_len).
            batch_ids (Optional[Tensor]): The batch ids tensor of shape (batch,).
            graph_data (Optional[Data]): The graph data for GNN encoding, if applicable.
        Output: 
            tensor of shape (batch, seq_len, d_model)
        """
        B, num_taxa = taxa_ids.shape
        if self.use_gnn:
            assert graph_data is not None, "graph_data should not be None when use_gnn is True"
            taxa_ids_embeds = self.taxa_encoder(taxa_ids, graph_data)
        else:
            taxa_ids_embeds = self.taxa_encoder(taxa_ids)  # (batch, seq_len, d_model)

        taxa_abundances_embeds = self.value_encoder(taxa_abundances)  # (batch, seq_len, d_model)
        if self.abundance_emb_style == "scaling":
            taxa_abundances_embeds = taxa_abundances_embeds.unsqueeze(2)
            total_embs = taxa_ids_embeds * taxa_abundances_embeds
        else:
            total_embs = taxa_ids_embeds + taxa_abundances_embeds

        # add special token embeddings before feeding to transformer
        # 1. batch labels
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_ids)  # (batch, d_model)
        
        # 2. sample token embedding, learned
        sample_token_emb = nn.Parameter(torch.randn(B, 1, self.d_model))  # (1, 1, d_model)
        # concat all special tokens, sample token first
        if self.use_batch_labels:
            total_embs = torch.cat(
                [sample_token_emb, batch_emb.unsqueeze(1), total_embs], dim=1
            )  # (batch, seq_len + 2, d_model)
        else:
            total_embs = torch.cat(
                [sample_token_emb, total_embs], dim=1
            )  # (batch, seq_len + 1, d_model)
        
        # TODO: need to build attention mask

        output = self.transformer_encoder(total_embs)
        return output

    def decode(
        self,
        transformer_output: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Runs the decoder part of the model. Returns a dictionary of outputs depending on tasks.
        
        For denoising task: Decodes all taxa tokens to predict denoised abundances.
        For bottleneck task: Uses only special tokens (sample embeddings) to reconstruct full profile.
        
        Args:
            transformer_output: Output from transformer encoder, shape (batch, num_special_tokens + num_taxa, d_model)
        
        Returns:
            Dictionary containing task-specific predictions:
                - For denoising with ZINB: {"denoising_mean", "denoising_disp", "denoising_pi"}
                - For denoising without dist: {"denoising_pred"}
                - For bottleneck with ZINB: {"bottleneck_mean", "bottleneck_disp", "bottleneck_pi"}
                - For bottleneck without dist: {"bottleneck_pred"}
        """
        output = {}
        
        # Denoising task: decode all taxa tokens
        if 'denoising' in self.tasks and hasattr(self, 'denoising_decoder'):
            denoising_output = self.denoising_decoder(transformer_output)
            
            # Add predictions with task prefix
            if self.denoising_decoder.distribution == "zinb":
                output["denoising_mean"] = denoising_output["mean"]
                output["denoising_disp"] = denoising_output["disp"]
                output["denoising_pi"] = denoising_output["pi"]
            else:
                output["denoising_pred"] = denoising_output["pred"]
        
        # Bottleneck task: decode from special tokens only TODO not yet implemented
        # if 'bottleneck' in self.tasks and hasattr(self, 'bottleneck_decoder'):
        #     # Extract only the first special token (sample embedding)
        #     sample_embedding = transformer_output[:, 0, :]  # (batch, d_model)
            
        #     # Decode from sample embedding alone
        #     bottleneck_output = self.bottleneck_decoder(sample_embedding)
            
        #     # Add predictions with task prefix
        #     if self.bottleneck_decoder.distribution == "zinb":
        #         output["bottleneck_mean"] = bottleneck_output["mean"]
        #         output["bottleneck_disp"] = bottleneck_output["disp"]
        #         output["bottleneck_pi"] = bottleneck_output["pi"]
        #     else:
        #         output["bottleneck_pred"] = bottleneck_output["pred"]
        
        return output



    def _get_sample_embedding(
        self, transformer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        helper function for retrieving sample embedding vector from the transformer output.
        depends on self.sample_emb_style. 
        IF self.sample_emb_style == "cls", then take the first token output as the sample embedding.
        IF self.sample_emb_style == "avg-pool", then take the average of all token outputs as the sample embedding.
        IF self.sample_emb_style == "w-pool", then take the weighted average of all token outputs as the sample embedding,
        with weights provided as input.

        Args:
            transformer_output(:obj:`Tensor`): shape (batch, seq_len + number of special tokens, d_model)
            weights(:obj:`Tensor`): shape (batch, seq_len + number of special tokens), optional and only used
                when :attr:`self.sample_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.sample_emb_style == "cls":
            sample_emb = transformer_output[:, 0, :]  # (batch, embsize)
        elif self.sample_emb_style == "avg-pool":
            sample_emb = torch.mean(transformer_output, dim=1)
        elif self.sample_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when sample_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            sample_emb = torch.sum(transformer_output * weights.unsqueeze(2), dim=1)
            sample_emb = F.normalize(sample_emb, p=2, dim=1)  # (batch, embsize)

        return sample_emb
    
    def make_attention_mask(known_positions: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Create a custom attention mask. TODO: need fixing

        Args:
            known_positions (torch.Tensor): (B, T) boolean tensor. True where the position is "known".
            device (torch.device): Device for output tensor.

        Returns:
            torch.Tensor: (B, T, T) boolean attention mask. True means attend, False means block.
        """
        B, T = known_positions.shape

        # Expand known_positions for broadcasting
        known_q = known_positions.unsqueeze(2)  # (B, T, 1) – whether the query is known
        known_k = known_positions.unsqueeze(1)  # (B, 1, T) – whether the key is known

        # known queries attend only to known keys
        known_to_known = known_q & known_k  # (B, T, T)

        # unknown queries attend to known keys and self
        # ~known_q = queries that are unknown
        eye = torch.eye(T, dtype=torch.bool, device=device).unsqueeze(0)  # (1, T, T)
        unknown_to_known_and_self = (~known_q) & (known_k | eye)

        # Combine both cases
        attention_mask = ~(known_to_known | unknown_to_known_and_self)

        return attention_mask # (B, T, T)
    
    def forward(
        self,
        taxa_ids: Tensor,
        abundance_values: Tensor,
        depth: Tensor,
        batch_ids: Optional[Tensor] = None,
        graph_data: Optional[Data] = None,
    ) -> Mapping[str, Tensor]:
        """
        Forward pass of the model.
            taxa_ids (:obj:`Tensor`): Token IDs representing taxa, shape [batch_size, seq_len].
            abundance_values (:obj:`Tensor`): Token values corresponding to taxa, shape [batch_size, seq_len].
            depth (:obj:`Tensor`): Depth information, shape [batch_size].
            batch_ids (:obj:`Optional[Tensor]`): Batch IDs for encoding, shape [batch_size]. 
                Required if `use_batch_labels` is True.
        """
        if self.use_batch_labels:
            assert batch_ids is not None, "batch_ids should not be None when use_batch_labels is True"
        else:
            assert batch_ids is None, "batch_ids should be None when use_batch_labels is False"
        
        if self.use_gnn:
            assert graph_data is not None, "graph_data should not be None when use_gnn is True"

        # 1. encode
        transformer_output = self.encode(
            taxa_ids,
            abundance_values,
            graph_data,
        )  # (batch, seq_len + number of special tokens, d_model)

        assert not torch.isnan(transformer_output).any(), "NaN in transformer output"
        # 2. decode
        output = self.decode(transformer_output)
        return output
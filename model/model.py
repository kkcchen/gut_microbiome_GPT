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
    AbundanceDecoder,
    SampleProjection,
)
from trainers import logger

# PLACEHOLDER FOR NOW

# tasks that rely on the masked-position machinery (mask token embedding, do_mask sampling)
MASKING_TASKS = {"masking", "masking_binary", "masking_from_cls"}


class hgmGPT(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        num_taxa: int,
        seq_len: Optional[int] = None,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        dropout: float = 0.5,
        abundance_emb_style: str = "continuous",
        sample_emb_style: str = "cls",
        preinitialized_taxa_embedding_path: Optional[str] = None,
        freeze_preinitialized_embeddings: bool = False,
        preinitialized_embedding_projection: str = "linear",  # "linear" or "mlp"
        use_gnn: bool = False,
        num_gnn_nodes: Optional[int] = None,
        tasks: List[str] = [],
        model_distribution: Optional[str] = None,
        masking_prob: Optional[float] = None,
        finetune_mode: Optional[str] = None,
        finetune_task: str = "classification",
        finetune_num_classes: Optional[int] = None,
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
        :param abundance_emb_style: The style of abundance embedding to use. One of "continuous"
            (additive), "scaling" (elementwise multiplicative), "category" (unimplemented), or
            "concatenation" (concat taxa-id and abundance embeddings, then project back to d_model).
        :type abundance_emb_style: str
        :param use_gnn: Whether to use a graph neural network for taxa encoding.
        :type use_gnn: bool
        :param num_gnn_nodes: The number of nodes in the graph neural network.
        :type num_gnn_nodes: Optional[int]
        :param tasks: The list of tasks to perform, e.g., denoising, bottleneck
        :type tasks: List[str]
        :param model_distribution: The distribution model to use, e.g., "zinb"
        :type model_distribution: Optional[str]
        :param finetune_mode: The finetuning mode to use, e.g., "none", "full", "partial"
        :type finetune_mode: str
        :param finetune_task: The downstream task for finetuning, e.g., "classification", "regression"
        :type finetune_task: str
        :param finetune_num_classes: The number of classes for classification finetuning, required if finetune_task is "classification"
        :type finetune_num_classes: Optional[int]
        """
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.use_batch_labels = use_batch_labels
        self.num_batch_labels = num_batch_labels
        self.abundance_emb_style = abundance_emb_style # default, continuous, mentioned in paper. could try using category encoding but this is likely less expressive. "concatenation" combines via concat+linear projection instead of add/multiply
        self.nhead = nhead
        self.tasks = tasks
        self.model_distribution = model_distribution
        self.num_taxa = num_taxa
        self.sample_emb_style = sample_emb_style
        self.dropout = dropout
        self.finetune_mode = finetune_mode
        self.finetune_task = finetune_task
        self.finetune_num_classes = finetune_num_classes
        self.preinitialized_taxa_embedding_path = preinitialized_taxa_embedding_path
        self.freeze_preinitialized_embeddings = freeze_preinitialized_embeddings
        self.preinitialized_embedding_projection = preinitialized_embedding_projection
        self.use_gnn = use_gnn
        if self.abundance_emb_style not in ["category", "continuous", "scaling", "concatenation"]:
            raise ValueError(
                f"abundance_emb_style should be one of category, continuous, scaling, concatenation, "
                f"got {abundance_emb_style}"
            )
        self.seq_len = seq_len
        
        # ================================ BUILD ENCODERS ================================
        self.use_gnn = use_gnn
        if use_gnn:
            assert num_gnn_nodes is not None, "num_gnn_nodes must be provided when use_gnn is True"
            self.taxa_encoder = TaxaGraphEncoder(num_nodes=num_gnn_nodes, 
                                            num_special_tokens=0, 
                                            num_taxa=self.num_taxa, 
                                            embedding_dim=d_model)
        else:
            self.taxa_encoder = TaxaEncoder(num_taxa=self.num_taxa, 
                                            embedding_dim=d_model, 
                                            init_taxa_embedding_path=preinitialized_taxa_embedding_path,
                                            freeze_preinitialized=freeze_preinitialized_embeddings,
                                            preinitialized_embedding_projection=preinitialized_embedding_projection
                                            )

        if self.abundance_emb_style in ("continuous", "concatenation"):
            self.value_encoder = ContinuousValueEncoder(d_model, self.dropout)
        else:
            print("Using scaling style for input embedding, just identity for now")
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)

        if self.abundance_emb_style == "concatenation":
            # Combines taxa-identity and abundance embeddings by concatenation followed by a
            # linear projection back to d_model, rather than addition ("continuous") or an
            # elementwise product ("scaling").
            self.abundance_concat_proj = nn.Linear(d_model * 2, d_model)

        # Batch Encoder
        if use_batch_labels:
            assert self.num_batch_labels is not None, "num_batch_labels must be provided when use_batch_labels is True"
            self.batch_encoder = BatchLabelEncoder(num_embeddings=self.num_batch_labels, 
                                                   embedding_dim=d_model)

        # TODO: also probably need some metadata encoder, not implemented yet
        # ================================================================================

        # ================================ BUILD TRANSFORMER =============================
        # build special tokens
        # sample token embedding, learned
        self.sample_token_emb = nn.Parameter(torch.randn(1, self.d_model))  # (1, d_model)

        if MASKING_TASKS.intersection(self.tasks):
            # mask token embedding, learned
            self.masking_prob = masking_prob
            self.mask_token_emb = nn.Parameter(torch.randn(1, self.d_model))  # (1, d_model)
        
        # TODO: potentially try cross attention
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, self.dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # ================================================================================


        # ================================ BUILD DECODERS ================================

        # output shape: B, num_tokens (max_seq_len + sample_token + batch_id_token + ...), d_model
        # 1. denoising
        # expression decoder, this operates on all the taxa tokens
        if "denoising" in self.tasks:
            self.abundance_decoder = AbundanceDecoder(
                d_model=d_model,
                num_special_tokens=2 if use_batch_labels else 1,
                output_format=self.model_distribution,
                dropout=self.dropout
            )
            # not checked
        # TODO: make this into bottleneck
        if "denoising_from_token" in self.tasks:
            # should project to the sequence length of the input to the whole model
            # note, this is not num_taxa, which is the vocab size, but the actual input sequence length
            self.sample_level_denoising_head = SampleProjection(d_model=d_model, 
                                                                projection_dim=self.seq_len)
        if "denoising" in self.tasks and self.model_distribution == "dm":
            # project the scale parameter from the sample embedding
            self.dirichlet_scale_head = SampleProjection(d_model=d_model,
                                                                projection_dim=1)
        
        
        # 2. bottleneck
        # TODO: zero out all the taxa tokens, leave only sample_token and batch_id_token, and essentially recreate the distribution for each taxa

        # 3. masking
        if "masking" in tasks:
            self.masking_decoder = AbundanceDecoder(
                d_model=d_model,
                num_special_tokens=2 if use_batch_labels else 1,
                output_format="logits",
                dropout=self.dropout
            )

        # 4. masking from cls: predict each masked position's abundance from the sample
        # (cls) token combined with that position's own taxon-identity embedding. Reuses
        # TaxaEncoder's weights (the same embedding used in encode()) for taxon identity
        # rather than learning a new (d_model -> num_taxa) matrix, and needs neither
        # seq_len nor a full-vocab gather since it's computed directly per position.
        if "masking_from_cls" in tasks:
            self.cls_masking_head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(d_model, 1),
            )

        # if finetune, add heads
        if self.finetune_mode is not None:
            logger.info("finetune mode enabled! adding finetuning head...")
            if self.finetune_task == "classification":
                assert self.finetune_num_classes is not None, "finetune_num_classes must be provided for classification finetuning"
                self.finetune_head = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, self.finetune_num_classes)
                )
            elif self.finetune_task == "regression":
                self.finetune_head = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1)  # Single output for regression
                )
            else:
                raise ValueError(f"Unsupported finetune_task: {self.finetune_task}")
            self._configure_finetuning(self.finetune_mode)
        # ================================================================================
        # =============================== PRINT MODEL INFO ===============================
        logger.info(f"Initialized hgmGPT model with {sum(p.numel() for p in self.parameters() if p.requires_grad)} trainable parameters")
        self._log_arguments()
        # print model architecture
        logger.info(self)

    def _log_arguments(self):
        """helper function to log all model arguments."""
        logger.info("Model arguments:")
        logger.info(f"\t d_model: {self.d_model}")
        logger.info(f"\t use_batch_labels: {self.use_batch_labels}")
        if self.use_batch_labels:
            logger.info(f"\t num_batch_labels: {self.num_batch_labels}")
        logger.info(f"\t abundance_emb_style: {self.abundance_emb_style}")
        logger.info(f"\t nhead: {self.nhead}")
        logger.info(f"\t tasks: {self.tasks}")
        logger.info(f"\t sample_emb_style: {self.sample_emb_style}")
        logger.info(f"\t dropout: {self.dropout}")
        logger.info(f"\t model_distribution: {self.model_distribution}")
        logger.info(f"\t use_gnn: {self.use_gnn}")
        if self.use_gnn:
            logger.info(f"\t num_taxa (vocab size): {self.num_taxa}")
    
    def _configure_finetuning(self, finetune_mode: str):
        """
        Configures the model for finetuning based on the specified mode. This involves freezing or unfreezing certain layers of the model depending on whether we are doing full finetuning, partial finetuning, or no finetuning.

        Args:
            finetune_mode (str): The finetuning mode to use. Should be one of "none", "full", "partial".
                - "none": No finetuning, all layers are frozen.
                - "full": Full finetuning, all layers are trainable.
                - "partial": Partial finetuning, only the transformer encoder and finetuning head are trainable, all other layers are frozen.
        """
        if finetune_mode == "none":
            # Freeze all layers
            for param in self.parameters():
                param.requires_grad = False
            logger.info("Finetuning mode: NONE. All layers frozen.")
        
        elif finetune_mode == "full":
            # Unfreeze all layers
            for param in self.parameters():
                param.requires_grad = True
            logger.info("Finetuning mode: FULL. All layers trainable.")
        elif finetune_mode == "partial":
            # Freeze encoders and decoders, unfreeze transformer encoder and finetuning head
            for name, param in self.named_parameters():
                if 'finetune_head' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            logger.info("Finetuning mode: PARTIAL. Transformer encoder and finetuning head are trainable, all other layers frozen.")
        
        else:
            raise ValueError(f"Unsupported finetune_mode: {finetune_mode}")
        

    def encode(
        self,
        taxa_ids: Tensor,
        taxa_abundances: Tensor,
        batch_ids: Optional[Tensor] = None,
        graph_data: Optional[Data] = None,
        do_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        This is the function that runs the encoder part of the model. Includes taxa/value encoding and transformer encoding. 
        Args:
            taxa_ids (Tensor): The taxa ids tensor of shape (batch, seq_len).
            taxa_values (Tensor): The taxa values tensor of shape (batch, seq_len).
            batch_ids (Optional[Tensor]): The batch ids tensor of shape (batch,).
            graph_data (Optional[Data]): The graph data for GNN encoding, if applicable.
            do_mask (Optional[Tensor]): Boolean tensor indicating which positions to mask, shape (batch, seq_len). Only used if "masking" in tasks.
            attention_mask (Optional[Tensor]): Attention mask for padding, shape (batch, num_tokens).
        Output: 
            tensor of shape (batch, num_tokens, d_model)
        """
        assert torch.isfinite(taxa_ids).all()
        assert torch.isfinite(taxa_abundances).all(), f"Non-finite abundances: {taxa_abundances}"

        B, num_taxa = taxa_ids.shape
        if self.use_gnn:
            assert graph_data is not None, "graph_data should not be None when use_gnn is True"
            taxa_ids_embeds = self.taxa_encoder(taxa_ids, graph_data)
        else:
            taxa_ids_embeds = self.taxa_encoder(taxa_ids)  # (batch, seq_len, d_model)
        assert torch.isfinite(taxa_ids_embeds).all(), "NaN/inf in taxa id embeddings"

        taxa_abundances_embeds = self.value_encoder(taxa_abundances)  # (batch, seq_len, d_model)
        # mask labels, replacing taxa_abundance_embeds
        if MASKING_TASKS.intersection(self.tasks) and do_mask is not None:
            mask_emb = self.mask_token_emb.unsqueeze(0)  # (1, 1, d_model)
            # Apply masking to taxa_ids_embeds
            taxa_abundances_embeds = torch.where(
                do_mask.unsqueeze(2),  # (batch, seq_len, 1)
                mask_emb,  # (1, 1, d_model)
                taxa_abundances_embeds  # (batch, seq_len, d_model)
            )  # (batch, seq_len, d_model)
            
        if self.abundance_emb_style == "scaling":
            taxa_abundances_embeds = taxa_abundances_embeds.unsqueeze(2)
            total_embs = taxa_ids_embeds * taxa_abundances_embeds
        elif self.abundance_emb_style == "concatenation":
            total_embs = self.abundance_concat_proj(
                torch.cat([taxa_ids_embeds, taxa_abundances_embeds], dim=-1)
            )  # (batch, seq_len, d_model)
        else:
            total_embs = taxa_ids_embeds + taxa_abundances_embeds
        assert torch.isfinite(taxa_abundances_embeds).all(), "NaN/inf in total embeddings"

        # add special token embeddings before feeding to transformer
        # 1. batch labels
        if self.use_batch_labels:
            if batch_ids is None:
                batch_ids = torch.zeros(B, dtype=torch.long, device=taxa_ids.device)
            batch_emb = self.batch_encoder(batch_ids)  # (batch, d_model)
        
        # # 2. mask labels
        # if 'masking' in self.tasks and do_mask is not None: 
        #     # do_mask ^ hack to get this to work in eval mode. TODO: Fix this properly
        #     mask_emb = self.mask_token_emb.unsqueeze(0)  # (1, 1, d_model)
        #     # Apply masking to taxa_ids_embeds
        #     total_embs = torch.where(
        #         do_mask.unsqueeze(2),  # (batch, seq_len, 1)
        #         mask_emb,  # (1, 1, d_model)
        #         total_embs  # (batch, seq_len, d_model)
        #     )  # (batch, seq_len, d_model)
        
        # concat all special tokens, sample token first
        sample_token_emb = self.sample_token_emb.unsqueeze(0).expand(B, -1, -1)
        if self.use_batch_labels:
            total_embs = torch.cat(
                [sample_token_emb, batch_emb.unsqueeze(1), total_embs], dim=1
            )  # (batch, seq_len + 2, d_model)
        else:
            total_embs = torch.cat(
                [sample_token_emb, total_embs], dim=1
            )  # (batch, seq_len + 1, d_model)

        output = self.transformer_encoder(total_embs, src_key_padding_mask=attention_mask)
        assert torch.isfinite(output).all(), "NaN/inf inside transformer"

        return output

    def decode(
        self,
        transformer_output: Tensor,
        do_mask: Optional[Tensor] = None,
        taxa_ids: Optional[Tensor] = None,
        graph_data: Optional[Data] = None,
    ) -> Dict[str, Tensor]:
        """
        Runs the decoder part of the model. Returns a dictionary of outputs depending on tasks.

        For denoising task: Decodes all taxa tokens to predict denoised abundances.
        For bottleneck task: Uses only special tokens (sample embeddings) to reconstruct full profile.

        Args:
            transformer_output: Output from transformer encoder, shape (batch, num_special_tokens + num_taxa, d_model)
            do_mask: Optional boolean tensor indicating which positions were masked during encoding. Only used if "masking" is in tasks, shape (batch, seq_len)
            taxa_ids: Optional taxa id tensor of shape (batch, seq_len). Required when "masking_from_cls" is in tasks, used to recompute each position's taxon-identity embedding via TaxaEncoder.
            graph_data: Optional graph data for GNN taxa encoding. Required when "masking_from_cls" is in tasks and use_gnn is True.

        Returns:
            Dictionary containing task-specific predictions:
                - For denoising with ZINB: {"denoising_mean", "denoising_disp", "denoising_pi"}
                - For denoising without dist: {"denoising_pred"}
                - For bottleneck with ZINB: {"bottleneck_mean", "bottleneck_disp", "bottleneck_pi"}
                - For bottleneck without dist: {"bottleneck_pred"}
        """
        output = {}
        
        # Denoising task: decode all taxa tokens
        if 'denoising' in self.tasks and hasattr(self, 'abundance_decoder'):
            
            denoising_output = self.abundance_decoder(transformer_output)
            
            # Add predictions with task prefix
            if self.abundance_decoder.output_format == "dm":
                output["denoising_mean"] = denoising_output["mean_logits"]
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
        if "denoising_from_token" in self.tasks and hasattr(self, 'sample_level_denoising_head'):
            # Get sample embedding
            sample_embedding = self._get_sample_embedding(transformer_output)
            output["denoising_projected"] = self.sample_level_denoising_head(sample_embedding)
        # if "denoising" in self.tasks and hasattr(self, 'dirichlet_scale_head'):
        if 'denoising' in self.tasks and hasattr(self, 'dirichlet_scale_head'):
            # Get sample embedding
            sample_embedding = self._get_sample_embedding(transformer_output)
            raw_scale = self.dirichlet_scale_head(sample_embedding).squeeze(-1)
            output["dirichlet_scale"] = F.softplus(raw_scale) + 1e-4
        # if "denoising_dm" in self.tasks and hasattr(self, 'sample_level_denoising_head'):
        #     sample_embedding = self._get_sample_embedding(transformer_output)
        #     output["denoising_projected"] = self.sample_level_denoising_head(sample_embedding)
        if "masking" in self.tasks:
            masking_output = self.masking_decoder(transformer_output)
            output["masking_logits"] = masking_output["logits"]
        if "masking_from_cls" in self.tasks and hasattr(self, 'cls_masking_head'):
            # Taxon-identity embedding, using the same TaxaEncoder weights as encode() --
            # NOT looked up from transformer_output, so it carries no contextual leakage
            # from neighboring positions, only "which taxon is this".
            if self.use_gnn:
                taxa_ids_embeds = self.taxa_encoder(taxa_ids, graph_data)
            else:
                taxa_ids_embeds = self.taxa_encoder(taxa_ids)  # (B, L, d_model)
            sample_embedding = self._get_sample_embedding(transformer_output)  # (B, d_model)
            sample_embedding_expanded = sample_embedding.unsqueeze(1).expand(-1, taxa_ids_embeds.size(1), -1)
            combined = torch.cat([sample_embedding_expanded, taxa_ids_embeds], dim=-1)  # (B, L, 2*d_model)
            output["cls_masking_logits"] = self.cls_masking_head(combined).squeeze(-1)  # (B, L)
        if MASKING_TASKS.intersection(self.tasks):
            output["masking_mask"] = do_mask

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
        # number of special tokens at the beginning of the sequence: 
        # Sample Token always exists. Batch Token exists if use_batch_labels is True.
        n_special = 2 if self.use_batch_labels else 1
            
        if self.sample_emb_style == "cls":
            sample_emb = transformer_output[:, 0, :]  # (batch, embsize)
        elif self.sample_emb_style == "avg-pool":
            sample_emb = torch.mean(transformer_output[:, n_special:, :], dim=1)
        elif self.sample_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when sample_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            sample_emb = torch.sum((transformer_output * weights.unsqueeze(2))[:, n_special:, :], dim=1)
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
        expressed_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:
        """
        Forward pass of the model.
            taxa_ids (:obj:`Tensor`): Token IDs representing taxa, shape [batch_size, seq_len].
            abundance_values (:obj:`Tensor`): Token values corresponding to taxa, shape [batch_size, seq_len].
            abundance_values_original (:obj:`Optional[Tensor]`): Original token values for denoising from token task, shape [batch_size, seq_len]. Only used if "denoising_from_token" in tasks.
            depth (:obj:`Tensor`): Depth information, shape [batch_size].
            batch_ids (:obj:`Optional[Tensor]`): Batch IDs for encoding, shape [batch_size]. 
                Required if `use_batch_labels` is True.
            attention_mask (:obj:`Optional[Tensor]`): Attention mask for padding, shape [batch_size, seq_len].
        """
        if self.use_batch_labels:
            assert batch_ids is not None, "batch_ids should not be None when use_batch_labels is True"
        else:
            assert batch_ids is None, "batch_ids should be None when use_batch_labels is False"
        
        if self.use_gnn:
            assert graph_data is not None, "graph_data should not be None when use_gnn is True"

        # Update attention mask for special tokens
        if attention_mask is not None:
            B = taxa_ids.size(0)
            n_special = 2 if self.use_batch_labels else 1
            special_mask = torch.zeros((B, n_special), dtype=torch.bool, device=attention_mask.device)
            full_attention_mask = torch.cat([special_mask, attention_mask], dim=1)
        else:
            full_attention_mask = None
        
        if MASKING_TASKS.intersection(self.tasks):
            do_mask = (torch.rand_like(taxa_ids.float()) < self.masking_prob)
            if attention_mask is not None:
                do_mask = do_mask & (~attention_mask)
        else:
            do_mask = None

        # 1. encode
        transformer_output = self.encode(
            taxa_ids,
            abundance_values,
            batch_ids,
            graph_data,
            do_mask=do_mask,
            attention_mask=full_attention_mask
        )  # (batch, seq_len + number of special tokens, d_model)

        assert not torch.isnan(transformer_output).any(), "NaN in transformer output"
        # 2. decode
        output = self.decode(transformer_output, do_mask=do_mask, taxa_ids=taxa_ids, graph_data=graph_data)
        return output

    def inference(
        self,
        taxa_ids: Tensor,
        abundance_values: Tensor,
        depth: Tensor,
        batch_ids: Optional[Tensor] = None,
        graph_data: Optional[Data] = None,
    ) -> Mapping[str, Tensor]:
        """
        Forward pass of the model, used for inference. Returns only the sample embeddings. 
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
            batch_ids=None, # TODO: pass in None for now, until we figure out what to do
            graph_data=graph_data,
        )  # (batch, seq_len + number of special tokens, d_model)

        assert not torch.isnan(transformer_output).any(), "NaN in transformer output"
        # 2. decode not needed for inference
        # output = self.decode(transformer_output)
        # 3. get sample embeddings
        sample_embeddings = self._get_sample_embedding(transformer_output)
        return sample_embeddings

    def finetune_forward(
        self,
        taxa_ids: Tensor,
        abundance_values: Tensor,
        depth: Tensor,
        batch_ids: Optional[Tensor] = None,
        graph_data: Optional[Data] = None,
    ) -> Tensor:
        """
        Forward pass specifically for finetuning. Returns predictions from the finetuning head.
        
        :param taxa_ids: Token IDs representing taxa, shape [batch_size, seq_len].
        :param abundance_values: Token values corresponding to taxa, shape [batch_size, seq_len].
        :param depth: Depth information, shape [batch_size].
        :param batch_ids: Batch IDs for encoding, shape [batch_size].
            Required if `use_batch_labels` is True.
        :param graph_data: Graph data for GNN encoding, if applicable.
        :return: Predictions tensor.
            - For classification: logits of shape [batch_size, num_classes]
            - For regression: values of shape [batch_size]
        """
        if self.finetune_mode == "none":
            raise RuntimeError("Model not in finetune mode. Set finetune_mode to 'full' or 'partial' during initialization.")
        
        if not hasattr(self, 'finetune_head'):
            raise RuntimeError("Finetuning head not initialized.")
        
        # Validate inputs
        if self.use_batch_labels:
            assert batch_ids is not None, "batch_ids should not be None when use_batch_labels is True"
        else:
            assert batch_ids is None, "batch_ids should be None when use_batch_labels is False"
        
        if self.use_gnn:
            assert graph_data is not None, "graph_data should not be None when use_gnn is True"
        
        # 1. Encode through transformer
        transformer_output = self.encode(
            taxa_ids,
            abundance_values,
            batch_ids=None, # TODO: pass in None for now, until we figure out what to do
            graph_data=graph_data,
            do_mask=None  # No masking during finetuning
        )  # (batch, seq_len + number of special tokens, d_model)
        
        assert not torch.isnan(transformer_output).any(), "NaN in transformer output"
        
        # 2. Extract sample embedding
        sample_embedding = self._get_sample_embedding(transformer_output)  # (batch, d_model)
        
        # 3. Pass through finetuning head
        predictions = self.finetune_head(sample_embedding)
        
        # 4. Format output based on task
        if self.finetune_task == "regression":
            predictions = predictions.squeeze(-1)  # (batch,)
        # For classification, return logits as-is: (batch, num_classes)
        
        return predictions

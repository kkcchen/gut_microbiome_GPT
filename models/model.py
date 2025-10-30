# TODO: do proper attribution from scGPT

from typing import Dict, Mapping, Optional, Tuple, Any, Union

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
    CategoryValueEncoder,
    BatchLabelEncoder
)

from .decoders import (
    OutputMulticlassDecoder,
    MVCDecoder,
)


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        # nlayers_cls: int = 3,
        # n_cls: int = 1,
        vocab_len: int,
        vocab_pad_index: int,
        vocab_pad_value: int,
        vocab_mask_value: int,
        n_input_bins: int,
        do_attn_mask: bool,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        dropout: float = 0.5,
        input_emb_style: str = "continuous",
        cell_emb_style: str = "cls",
        explicit_zero_prob: bool = False,
        do_mvc: bool = False,
        do_taxa_decoder: bool = False,
        mvc_decoder_style: str = "inner product",
        pre_norm: bool = False,
        vocab_num_special_tokens: int = 3,
        init_vocab_path: str = None,
        freeze_vocab: bool = False,
        use_gnn: bool = False,
        num_gnn_nodes: Optional[int] = None,
        gnn_type: str = "gat",
        gnn_num_layers: int = 2,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.use_batch_labels = use_batch_labels
        self.input_emb_style = input_emb_style # default, continuous, mentioned in paper. could try using category encoding but this is likely less expressive
        self.cell_emb_style = cell_emb_style # default: cls, but can also be avg-pool, w-pool. refers to how to encode the cell embeddings
        self.explicit_zero_prob = explicit_zero_prob # use a separate NN to predict the probability of zero for expression. Not mentioned in the paper, so off for now
        self.norm_scheme = "pre" if pre_norm else "post" # hyperparameter for the transformer encoder.
        self.do_mvc = do_mvc
        self.n_input_bins = n_input_bins
        self.mvc_decoder_style = mvc_decoder_style
        self.nhead = nhead
        self.do_attn_mask = do_attn_mask
        self.do_taxa_decoder = do_taxa_decoder
        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {input_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        # TODO: add dropout in the TaxaEncoder
        # self.flag_encoder = nn.Embedding(2, d_model)
        self.use_gnn = use_gnn
        if use_gnn:
            assert num_gnn_nodes is not None, "num_gnn_nodes must be provided when use_gnn is True"
            self.encoder = TaxaGraphEncoder(num_gnn_nodes, 
                                            vocab_num_special_tokens, 
                                            vocab_len - vocab_num_special_tokens, 
                                            d_model,
                                            num_layers=gnn_num_layers,
                                            graph_type=gnn_type,
                                            padding_idx=vocab_pad_index)
        else:
            self.encoder = TaxaEncoder(vocab_len, d_model, init_vocab_path, freeze_vocab, padding_idx=vocab_pad_index)

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model, vocab_mask_value, dropout)
        elif input_emb_style == "category":
            assert n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(
                n_input_bins, d_model, vocab_mask_value, padding_idx=vocab_pad_value
            )
        else: # input_emb_style == "scaling"
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling

        # Batch Encoder
        if use_batch_labels:
            assert num_batch_labels is not None, "num_batch_labels must be provided when use_batch_labels is True"
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)

        # masked value classification decoder
        # always init mvc, even if it isn't used for model init compatibility purposes
        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )
        
        # use unified TransformerEncoder
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # MLP to get from abundance embedding (not just the <cls> token) to quantity prediction
        self.abundance_decoder = OutputMulticlassDecoder(
            d_model,
            d_out=1,
            explicit_zero_prob=explicit_zero_prob,
            use_batch_labels=use_batch_labels,
        )
        
        if do_taxa_decoder:
            self.taxa_decoder = OutputMulticlassDecoder(
                d_model,
                d_out=vocab_len - vocab_num_special_tokens,  # exclude special tokens
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )

        # self.init_weights()

    def encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        graph_data: Optional[Data] = None,
        # batch_labels: Optional[Tensor] = None,  # (batch,)
    ) -> Tensor:
        # self._check_batch_labels(batch_labels)

        if self.use_gnn:
            assert graph_data is not None, "graph_data should not be None when use_gnn is True"
            src = self.encoder(src, graph_data)
        else:
            src = self.encoder(src)  # (batch, seq_len, embsize)
        cur_taxa_token_embs = src

        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output, cur_taxa_token_embs # (batch, seq_len, embsize), (batch, seq_len, embsize)
    
    # # this only initializes the taxa embedding layer
    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     # TODO: check if this initialization is helpful and shall we apply to all?
    #     self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    # for the <cls> embedding this will be the first token in the sequence
    def get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb
    
    def make_mask(known_positions: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Create a custom attention mask.

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
    
    
    def transformer_generate(
        self,
        taxa: Tensor,
        values: Tensor,
        key_padding_mask: Tensor,
        known_positions: Optional[Tensor] = None, # (batch, seq_len)
        # batch_labels: Optional[Tensor] = None,  # (batch,)
        input_cell_emb: Optional[Tensor] = None,  # (batch, embsize)
        graph_data: Optional[Data] = None,
    ) -> Tuple[Tensor, Tensor]:
        # self._check_batch_labels(batch_labels)

        if self.use_gnn:
            token_embs = self.encoder(taxa, graph_data)
        else:
            token_embs = self.encoder(taxa)  # (batch, seq_len, embsize)
        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        total_embs = token_embs + values

        assert self.input_emb_style != "scaling"

        if input_cell_emb is not None:
            # this is for the second step of pretraining, where we replace the cls token with the cell embedding
            total_embs[:, 0, :] = input_cell_emb

        if self.do_attn_mask:
            assert known_positions is not None, "known_positions should not be None when do_attn_mask is True"
            attn_mask = TransformerModel.make_mask(known_positions, device=taxa.device)
            B, T1, T2 = attn_mask.shape

            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1)
            attn_mask = attn_mask.reshape(B * self.nhead, T1, T2)
        else:
            attn_mask = None
        
        all_output = self.transformer_encoder(
            total_embs,
            src_key_padding_mask=key_padding_mask,
            mask=attn_mask,
        )
        
        return all_output, token_embs  # (batch, seq_len, embsize)
    
    def forward(
        self,
        taxa: Tensor,
        values: Tensor,
        key_padding_mask: Tensor,
        known_positions: Optional[Tensor] = None,
        batch_labels: Optional[Tensor] = None,
        # CLS: bool = False,
        MVC: bool = False,
        TCS: bool = False,  # Taxa Classification, i.e. taxa decoder
        # ECS: bool = False,
        # do_sample: bool = False,
        input_cell_emb: Optional[Tensor] = None,
        graph_data: Optional[Data] = None,
    ) -> Mapping[str, Tensor]:
        """
        Forward pass of the model.
            taxa (:obj:`Tensor`): Token IDs representing taxa, shape [batch_size, seq_len].
            values (:obj:`Tensor`): Token values corresponding to taxa, shape [batch_size, seq_len].
            key_padding_mask (:obj:`Tensor`): Mask for taxa tokens, shape [batch_size, seq_len].
            known_positions (:obj:`Optional[Tensor]`): Known positions for the taxa, shape [batch_size, seq_len].
            batch_labels (:obj:`Optional[Tensor]`): Batch labels for encoding, shape [batch_size]. 
                Required if `use_batch_labels` is True.
            MVC (:obj:`bool`): If True, perform masked value prediction for cell embedding (MVC).
            input_cell_emb (:obj:`Optional[Tensor]`): Precomputed cell embeddings, shape [batch_size, embsize].
            Mapping[str, Tensor]: A dictionary containing the following keys:
                - "preds": Predictions for values, shape [batch_size, seq_len].
                - "cell_emb": Cell embeddings, shape [batch_size, embsize].
                - "mvc_preds" (optional): MVC predictions for values, shape [batch_size, seq_len].
        """
        if self.use_batch_labels:
            assert batch_labels is not None, "batch_labels should not be None when use_batch_labels is True"
            batch_emb = self.batch_encoder(batch_labels) # (batch, embsize)
        else:
            assert batch_labels is None, "batch_labels should be None when use_batch_labels is False"
        
        if self.use_gnn:
            assert graph_data is not None, "graph_data should not be None when use_gnn is True"
        transformer_output, cur_taxa_token_embs = self.transformer_generate(
            taxa,
            values,
            key_padding_mask,
            known_positions,
            # batch_labels,
            input_cell_emb=input_cell_emb,
            graph_data=graph_data,
        )

        output = {}
        decoder_output = self.abundance_decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # # else transformer_output + batch_emb.unsqueeze(1),
        )
        # if self.explicit_zero_prob and do_sample:
        #     bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
        #     output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        # else:
        full_preds = decoder_output["pred"]  # (batch, seq_len)

        output["preds"] = full_preds
        # if self.explicit_zero_prob:
        #     output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = self.get_cell_emb_from_layer(transformer_output)
        output["cell_emb"] = cell_emb

        # if CLS: # GEP
        #     raise NotImplementedError(
        #         "CLS is not implemented yet. Please set CLS=False to avoid this error."
        #     )
        #     output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if MVC: # GEPC
            if not self.do_mvc:
                raise ValueError("MVC is not enabled for this model, so do not call MVC in the forward pass!")
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                cur_taxa_token_embs,
            )
            # if self.explicit_zero_prob and do_sample:
            #     bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
            #     output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            # else:
            output["mvc_preds"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                raise NotImplementedError(
                        "Explicit zero prob is not implemented for MVC decoder"
                    )
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if TCS:
            if not self.do_taxa_decoder:
                raise ValueError("TCS is not enabled for this model, so do not call TCS in the forward pass!")
            taxa_output = self.taxa_decoder(
                transformer_output
                if not self.use_batch_labels
                else torch.cat(
                    [
                        transformer_output,
                        batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                    ],
                    dim=2,
                ),
            )
            # if self.explicit_zero_prob and do_sample:
            #     bernoulli = Bernoulli(probs=taxa_output["zero_probs"])
            #     output["taxa_output"] = bernoulli.sample() * taxa_output["pred"]
            # else:
            output["taxa_preds"] = taxa_output["pred"]
        # if ECS:
        #     raise NotImplementedError(
        #         "Elastic cell similarity is not implemented yet. "
        #         "Please set ECS=False to avoid this error."
        #     )
        #     # Here using customized cosine similarity instead of F.cosine_similarity
        #     # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
        #     # normalize the embedding
        #     cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
        #     cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

            # # mask out diagnal elements
            # mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            # cos_sim = cos_sim.masked_fill(mask, 0.0)
            # # only optimize positive similarities
            # cos_sim = F.relu(cos_sim)

            # output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        # if self.do_dab:
        #     raise NotImplementedError(
        #         "DAB is not implemented yet. Please set do_dab=False to avoid this error."
        #     )
        #     output["dab_output"] = self.grad_reverse_discriminator(cell_emb)

        return output
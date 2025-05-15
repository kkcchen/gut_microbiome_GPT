# TODO: do proper attribution from scGPT

import gc
import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union

import torch
import numpy as np
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli
from tqdm import trange

try:
    from flash_attn.flash_attention import FlashMHA

    flash_attn_available = True
except ImportError:
    import warnings

    warnings.warn("flash_attn is not installed")
    flash_attn_available = False

# from .dsbn import DomainSpecificBatchNorm1d
# from .grad_reverse import grad_reverse

from .encoders import (
    TaxaEncoder,
    ContinuousValueEncoder,
    CategoryValueEncoder,
)

from .decoders import (
    AbundanceDecoder,
    MVCDecoder,
    AdversarialDiscriminator
)


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int = 3,
        n_cls: int = 1,
        vocab: Any = None,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        do_mvc: bool = False,
        do_dab: bool = False,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        domain_spec_batchnorm: Union[bool, str] = False,
        input_emb_style: str = "continuous",
        n_input_bins: Optional[int] = None,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        # ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        # fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.do_dab = do_dab # Domain adaptation via reverse back propagation
        # self.ecs_threshold = ecs_threshold # Elastic Cell Similarity could be added later
        self.use_batch_labels = use_batch_labels # batch labels could be added label "Representation for batch and modality"
        self.domain_spec_batchnorm = domain_spec_batchnorm # look into DSBN later, but it is false for now
        self.input_emb_style = input_emb_style # default, continuous, mentioned in paper. could try using category encoding but this is likely less expressive
        self.cell_emb_style = cell_emb_style # default: cls, but can also be avg-pool, w-pool. refers to how to encode the cell embeddings
        # self.explicit_zero_prob = explicit_zero_prob # use a separate NN to predict the probability of zero for expression. Not mentioned in the paper, so off for now
        self.norm_scheme = "pre" if pre_norm else "post" # hyperparameter for the transformer encoder.
        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {input_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        # if use_fast_transformer:
        #     if not flash_attn_available:
        #         warnings.warn(
        #             "flash-attn is not installed, using pytorch transformer instead. "
        #             "Set use_fast_transformer=False to avoid this warning. "
        #             "Installing flash-attn is highly recommended."
        #         )
        #         use_fast_transformer = False
        # self.use_fast_transformer = use_fast_transformer

        # TODO: add dropout in the TaxaEncoder
        self.encoder = TaxaEncoder(ntoken, d_model, padding_idx=vocab[pad_token])

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        elif input_emb_style == "category":
            assert n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(
                n_input_bins, d_model, padding_idx=pad_value
            )
        else: # input_emb_style == "scaling"
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling

        # # Batch Encoder
        # if use_batch_labels:
        #     self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)

        # if domain_spec_batchnorm is True or domain_spec_batchnorm == "dsbn":
        #     use_affine = True if domain_spec_batchnorm == "do_affine" else False
        #     print(f"Use domain specific batchnorm with affine={use_affine}")
        #     self.dsbn = DomainSpecificBatchNorm1d(
        #         d_model, num_batch_labels, eps=6.1e-5, affine=use_affine
        #     )
        # elif domain_spec_batchnorm == "batchnorm":
        #     print("Using simple batchnorm instead of domain specific batchnorm")
        #     self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        # dependency issues here. will need to be fixed later
        # if use_fast_transformer:
        #     if fast_transformer_backend == "linear":
        #         self.transformer_encoder = FastTransformerEncoderWrapper(
        #             d_model, nhead, d_hid, nlayers, dropout
        #         )
        #     elif fast_transformer_backend == "flash":
        #         encoder_layers = FlashTransformerEncoderLayer(
        #             d_model,
        #             nhead,
        #             d_hid,
        #             dropout,
        #             batch_first=True,
        #             norm_scheme=self.norm_scheme,
        #         )
        #         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # else:
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # MLP to get from abundance embedding (not just the <cls> token) to quantity prediction
        self.decoder = AbundanceDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
            use_batch_labels=use_batch_labels,
        )

        # not sure if there is an analogous decoder here... will ask about it
        # self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)

        # masked value classification decoder
        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )

        if do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                d_model,
                n_cls=num_batch_labels,
                reverse_grad=True,
            )

        # self.sim only used for CCE (or)
        # self.sim = Similarity(temp=0.5)  # TODO: auto set temp
        # self.criterion_cce = nn.CrossEntropyLoss()

        self.init_weights()

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,  # (batch,)
    ) -> Tensor:
        # self._check_batch_labels(batch_labels)

        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src

        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        # to do with dsbn. ignore for now
        # if getattr(self, "dsbn", None) is not None:
        #     batch_label = int(batch_labels[0].item())
        #     total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
        #         0, 2, 1
        #     )  # the batch norm always works on dim 1
        # elif getattr(self, "bn", None) is not None:
        #     total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (batch, seq_len, embsize)
    
    # this only initializes the taxa embedding layer
    def init_weights(self) -> None:
        initrange = 0.1
        # TODO: check if this initialization is helpful and shall we apply to all?
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    # for the <cls> embedding this will be the first token in the sequence
    def _get_cell_emb_from_layer(
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
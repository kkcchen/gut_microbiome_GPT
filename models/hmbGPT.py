# TODO: do proper attribution from scGPT

from typing import Dict, Mapping, Optional, Tuple, Any, Union

import torch
import numpy as np
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli
# from tqdm import trange

from data_utils.vocab import MicrobiomeVocab
from functools import lru_cache

# from .flash_layers import (
#     FlashscGPTLayer,
#     FlashscGPTGenerator,
# )


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
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        # nlayers_cls: int = 3,
        # n_cls: int = 1,
        vocab: MicrobiomeVocab,
        dropout: float = 0.5,
        input_emb_style: str = "continuous",
        n_input_bins: Optional[int] = None,
        cell_emb_style: str = "cls",
        explicit_zero_prob: bool = False,
        do_mvc: bool = False,
        mvc_decoder_style: str = "inner product",
        pre_norm: bool = False,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.input_emb_style = input_emb_style # default, continuous, mentioned in paper. could try using category encoding but this is likely less expressive
        self.cell_emb_style = cell_emb_style # default: cls, but can also be avg-pool, w-pool. refers to how to encode the cell embeddings
        self.explicit_zero_prob = explicit_zero_prob # use a separate NN to predict the probability of zero for expression. Not mentioned in the paper, so off for now
        self.norm_scheme = "pre" if pre_norm else "post" # hyperparameter for the transformer encoder.
        self.do_mvc = do_mvc
        self.mvc_decoder_style = mvc_decoder_style
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
        self.flag_encoder = nn.Embedding(2, d_model)
        self.encoder = TaxaEncoder(len(vocab), d_model, padding_idx=vocab.pad_index)

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        elif input_emb_style == "category":
            assert n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(
                n_input_bins, d_model, padding_idx=vocab.pad_value
            )
        else: # input_emb_style == "scaling"
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling

        # # masked value classification decoder
        # if do_mvc:
        #     self.mvc_decoder = MVCDecoder(
        #         d_model,
        #         arch_style=mvc_decoder_style,
        #         explicit_zero_prob=explicit_zero_prob,
        #         use_batch_labels=False,  # do not use batch labels for MVC during pretraining
        #     )
        # if domain_spec_batchnorm is True or domain_spec_batchnorm == "dsbn":
        #     use_affine = True if domain_spec_batchnorm == "do_affine" else False
        #     print(f"Use domain specific batchnorm with affine={use_affine}")
        #     self.dsbn = DomainSpecificBatchNorm1d(
        #         d_model, num_batch_labels, eps=6.1e-5, affine=use_affine
        #     )
        # elif domain_spec_batchnorm == "batchnorm":
        #     print("Using simple batchnorm instead of domain specific batchnorm")
        #     self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        # if use_generative_training:
        #     encoder_layers = FlashscGPTLayer(
        #         d_model,
        #         nhead,
        #         d_hid,
        #         dropout,
        #         batch_first=True,
        #         norm_scheme=self.norm_scheme,
        #     )
        #     self.transformer_encoder = FlashscGPTGenerator(encoder_layers, nlayers)
        # elif use_fast_transformer:
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
        #     NotImplementedError(
        #         "use_generative_training should be true!"
        #     )
        #     encoder_layers = TransformerEncoderLayer(
        #         d_model, nhead, d_hid, dropout, batch_first=True
        #     )
        #     self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        # use unified TransformerEncoder
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # MLP to get from abundance embedding (not just the <cls> token) to quantity prediction
        self.decoder = AbundanceDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
        )



        self.init_weights()

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        # batch_labels: Optional[Tensor] = None,  # (batch,)
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
    
    @lru_cache(maxsize=1)
    @staticmethod
    def make_mask(mask_len, seq_len, device):
        assert mask_len <= seq_len, "mask_len should be less than or equal to seq_len"
        attention_mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.bool)

        split = seq_len - mask_len
        # Top part: mask right mask_len columns
        if split > 0:
            attention_mask[:split, split:] = True

        # Bottom mask_len x mask_len block: mask everything except diagonal
        if mask_len > 0:
            attention_mask[split:, split:] = ~torch.eye(mask_len, device=device, dtype=torch.bool)

        return attention_mask
    
    def transformer_generate(
        self,
        pcpt_taxa: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_taxa: Tensor,
        gen_key_padding_mask: Tensor,
        # batch_labels: Optional[Tensor] = None,  # (batch,)
        input_cell_emb: Optional[Tensor] = None,  # (batch, embsize)
    ) -> Tuple[Tensor, Tensor]:
        # self._check_batch_labels(batch_labels)

        pcpt_token_embs = self.encoder(pcpt_taxa)  # (batch, pcpt_len, embsize)
        pcpt_values = self.value_encoder(pcpt_values)  # (batch, pcpt_len, embsize)
        pcpt_total_embs = pcpt_token_embs + pcpt_values

        assert self.input_emb_style != "scaling"
        if gen_taxa is not None:
            gen_token_embs = self.encoder(gen_taxa)  # (batch, gen_len, embsize)
            # self.cur_gene_token_embs = torch.cat(
            #     [pcpt_token_embs, gen_token_embs], dim=1
            # )
            # this is a flag to let the model know that this is a generative training
            gen_flags = self.flag_encoder(
                torch.tensor(1).to(pcpt_values.device)
            ).expand(gen_taxa.shape[0], gen_taxa.shape[1], -1)

            gen_total_embs = gen_token_embs + gen_flags
        else:
            raise NotImplementedError(
                "gen_taxa should not be none..."
            )
            # self.cur_gene_token_embs = pcpt_token_embs
            # gen_total_embs = None

        # if self.domain_spec_batchnorm:
        #     batch_label = int(batch_labels[0].item())
        #     pcpt_total_embs = self.dsbn(
        #         pcpt_total_embs.permute(0, 2, 1), batch_label
        #     ).permute(0, 2, 1)
        #     if gen_taxa is not None:
        #         gen_total_embs = self.dsbn(
        #             gen_total_embs.permute(0, 2, 1), batch_label
        #         ).permute(0, 2, 1)
        # else:
        #     pcpt_total_embs = self.bn(pcpt_total_embs.permute(0, 2, 1)).permute(0, 2, 1)
        #     if gen_taxa is not None:
        #         gen_total_embs = self.bn(gen_total_embs.permute(0, 2, 1)).permute(
        #             0, 2, 1
        #         )

        if input_cell_emb is not None:
            # this is for the second step of pretraining, where we replace the cls token with the cell embedding
            pcpt_total_embs[:, 0, :] = input_cell_emb
            
        all_embs = torch.cat(
            [pcpt_total_embs, gen_total_embs], dim=1
        ) # (batch, pcpt_len + gen_len, embsize)
        all_key_padding_mask = torch.cat(
            [pcpt_key_padding_mask, gen_key_padding_mask], dim=1
        )

        attn_mask = TransformerModel.make_mask(gen_total_embs.shape[1], all_embs.shape[1], device=pcpt_total_embs.device)
        
        # pcpt_output, gen_output = self.transformer_encoder(
        #     pcpt_total_embs,
        #     gen_total_embs,
        #     pcpt_key_padding_mask=pcpt_key_padding_mask,
        #     gen_key_padding_mask=gen_key_padding_mask,
        # )
        
        all_output = self.transformer_encoder(
            all_embs,
            src_key_padding_mask=all_key_padding_mask,
            mask=attn_mask,
        )
        
        # the first part of all_output refers to the perceptual part (pcpt_total_embs.shape[1])
        # the rest of all_output refers to the generative part
        
        return all_output
    
    def forward(
        self,
        pcpt_taxa: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_taxa: Tensor = None,
        gen_key_padding_mask: Tensor = None,
        # batch_labels: Optional[Tensor] = None,
        # CLS: bool = False,
        # CCE: bool = False,
        # MVC: bool = False,
        # ECS: bool = False,
        # do_sample: bool = False,
        input_cell_emb: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            pcpt_taxa (:obj:`Tensor`): token ids of the perceptual part, shape
                [batch_size, seq_len]
            pcpt_values (:obj:`Tensor`): token values of the perceptual part, shape
                [batch_size, seq_len]
            pcpt_key_padding_mask (:obj:`Tensor`): mask for pcpt_taxa, shape
                [batch_size, seq_len]
            gen_taxa (:obj:`Tensor`): token ids of the generative part, shape
                [batch_size, seq_len]
            gen_key_padding_mask (:obj:`Tensor`): mask for gen_taxa, shape
                [batch_size, seq_len]

            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.

            input_cell_emb (:obj:`Tensor`): cell embeddings, shape [batch_size, embsize]

        Returns:
            dict of output Tensors.
        """
        assert gen_taxa is not None and gen_key_padding_mask is not None, "gen_taxa and gen_key_padding_mask should not be None during pretraining"
        transformer_output = self.transformer_generate(
            pcpt_taxa,
            pcpt_values,
            pcpt_key_padding_mask,
            gen_taxa,
            gen_key_padding_mask,
            # batch_labels,
            input_cell_emb=input_cell_emb,
        )
        # else: if not pretraining
        #     transformer_output = self._encode(
        #         pcpt_taxa,
        #         pcpt_values,
        #         pcpt_key_padding_mask,
        #         # batch_labels,
        #     )

        output = {}
        decoder_output = self.decoder(
            transformer_output
            # if not self.use_batch_labels
            # else torch.cat(
            #     [
            #         transformer_output,
            #         batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
            #     ],
            #     dim=2,
            # ),
            # # else transformer_output + batch_emb.unsqueeze(1),
        )
        # if self.explicit_zero_prob and do_sample:
        #     bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
        #     output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        # else:
        full_preds = decoder_output["pred"]  # (batch, seq_len)

        # separate pcpt and gen predictions
        output["pcpt_preds"] = full_preds[:, : pcpt_taxa.shape[1]]
        output["gen_preds"] = full_preds[:, pcpt_taxa.shape[1] :]
        # if self.explicit_zero_prob:
        #     output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = self._get_cell_emb_from_layer(transformer_output)
        output["cell_emb"] = cell_emb

        # if CLS: # GEP
        #     raise NotImplementedError(
        #         "CLS is not implemented yet. Please set CLS=False to avoid this error."
        #     )
        #     output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        # if CCE:
        #     raise NotImplementedError(
        #         "Contrastive cell embedding objective is not implemented yet. "
        #         "Please set CCE=False to avoid this error."
        #     )
        #     cell1 = cell_emb
        #     transformer_output2 = self._encode(
        #         src, values, src_key_padding_mask, batch_labels
        #     )
        #     cell2 = self._get_cell_emb_from_layer(transformer_output2)

        #     # Gather embeddings from all devices if distributed training
        #     if dist.is_initialized() and self.training:
        #         cls1_list = [
        #             torch.zeros_like(cell1) for _ in range(dist.get_world_size())
        #         ]
        #         cls2_list = [
        #             torch.zeros_like(cell2) for _ in range(dist.get_world_size())
        #         ]
        #         dist.all_gather(tensor_list=cls1_list, tensor=cell1.contiguous())
        #         dist.all_gather(tensor_list=cls2_list, tensor=cell2.contiguous())

        #         # NOTE: all_gather results have no gradients, so replace the item
        #         # of the current rank with the original tensor to keep gradients.
        #         # See https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L186
        #         cls1_list[dist.get_rank()] = cell1
        #         cls2_list[dist.get_rank()] = cell2

        #         cell1 = torch.cat(cls1_list, dim=0)
        #         cell2 = torch.cat(cls2_list, dim=0)
        #     # TODO: should detach the second run cls2? Can have a try
        #     cos_sim = self.sim(cell1.unsqueeze(1), cell2.unsqueeze(0))  # (batch, batch)
        #     labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
        #     output["loss_cce"] = self.creterion_cce(cos_sim, labels)
        # if MVC: # GEPC
        #     mvc_output = self.mvc_decoder(
        #         cell_emb
        #         # if not self.use_batch_labels
        #         # else torch.cat([cell_emb, batch_emb], dim=1),
        #         # # else cell_emb + batch_emb,
        #         # self.cur_gene_token_embs,
        #     )
        #     # if self.explicit_zero_prob and do_sample:
        #     #     bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
        #     #     output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
        #     # else:
        #     output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
        #     if self.explicit_zero_prob:
        #         raise NotImplementedError(
        #                 "Explicit zero prob is not implemented for MVC decoder"
        #             )
        #         output["mvc_zero_probs"] = mvc_output["zero_probs"]
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
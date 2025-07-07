# TODO: do proper attribution from scGPT

import torch
from torch import nn, Tensor
from typing import Dict, Mapping, Optional, Tuple, Any, Union

# MLP to get from encoder output to abundance prediction
class OutputMulticlassDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_out: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_out),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_out),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.


class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for environment embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.taxon2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.taxon2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.taxon2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, env_emb: Tensor, taxon_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            env_emb: Tensor, shape (batch, embsize=d_model)
            taxon_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        taxon_embs = taxon_embs.detach() if self.do_detach else taxon_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.taxon2query(taxon_embs))
            env_emb = env_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), env_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the env_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), env_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.taxon2query(taxon_embs))
            # expand env_emb to (batch, seq_len, embsize)
            env_emb = env_emb.unsqueeze(1).expand(-1, taxon_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([env_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.taxon2query(taxon_embs))
            env_emb = env_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(env_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)


# class AdversarialDiscriminator(nn.Module):
#     """
#     Discriminator for the adversarial training for batch correction.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         n_cls: int,
#         nlayers: int = 3,
#         activation: callable = nn.LeakyReLU,
#         reverse_grad: bool = False,
#     ):
#         super().__init__()
#         # module list
#         self._decoder = nn.ModuleList()
#         for i in range(nlayers - 1):
#             self._decoder.append(nn.Linear(d_model, d_model))
#             self._decoder.append(activation())
#             self._decoder.append(nn.LayerNorm(d_model))
#         self.out_layer = nn.Linear(d_model, n_cls)
#         self.reverse_grad = reverse_grad

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [batch_size, embsize]
#         """
#         if self.reverse_grad:
#             x = grad_reverse(x, lambd=1.0)
#         for layer in self._decoder:
#             x = layer(x)
#         return self.out_layer(x)
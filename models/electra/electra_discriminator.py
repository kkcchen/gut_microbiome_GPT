"""
original code from
Learning a deep language model for microbiomes: the power of large scale unlabeled microbiome data
Quintin Pope, Rohan Varma, Chritine Tataru, Maude David, Xiaoli Fern
bioRxiv 2023.07.17.549267; doi: https://doi.org/10.1101/2023.07.17.549267

edited by Haoze Deng
"""
from transformers import ElectraConfig, ElectraForPreTraining
import torch.nn as nn
import torch


class ElectraDiscriminator(nn.Module):

    def __init__(self, config: ElectraConfig, embeddings: torch.Tensor, load_disc=None, load_embed=None):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size,
                                        padding_idx=config.vocab_size - 1)
        if load_embed is not None:
            self.embed_layer.load_state_dict(torch.load(load_embed, weights_only=False))
        else:
            self.embed_layer.weight = nn.Parameter(embeddings)

        if load_disc is not None:
            self.discriminator = ElectraForPreTraining.from_pretrained(load_disc, config=config)
        else:
            self.discriminator = ElectraForPreTraining(config)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, attention_mask, labels):
        #pdb.set_trace()
        data = self.embed_layer(data)
        output = self.discriminator(attention_mask=attention_mask, inputs_embeds=data, labels=labels)
        loss = output.loss
        scores = output.logits
        scores = self.sigmoid(scores)
        return loss, scores

    def forward_embeddings(self, data, attn_mask):
        data = self.embed_layer(data)
        output = self.discriminator(inputs_embeds=data, attention_mask=attn_mask,
                                    output_hidden_states=True)
        return output

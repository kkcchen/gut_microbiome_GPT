"""
original code from
Learning a deep language model for microbiomes: the power of large scale unlabeled microbiome data
Quintin Pope, Rohan Varma, Chritine Tataru, Maude David, Xiaoli Fern
bioRxiv 2023.07.17.549267; doi: https://doi.org/10.1101/2023.07.17.549267

edited by Haoze Deng
"""
from transformers import ElectraConfig,ElectraForMaskedLM
import torch.nn as nn
import torch

class ElectraGenerator(nn.Module):

    def __init__(self,config: ElectraConfig,embeddings,generator=None,embed_layer=None):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.embedding_size, padding_idx= config.vocab_size-1)
        if embed_layer:
            self.embed_layer.load_state_dict(torch.load(embed_layer))
        else:
            self.embed_layer.weight = nn.Parameter(embeddings)
        if generator:
            self.generator = ElectraForMaskedLM.from_pretrained(generator,config=config)
        else:
            self.generator = ElectraForMaskedLM(config)
        self.softmax = nn.Softmax(dim=2)

    def forward(self,data,attention_mask,labels):
        #pdb.set_trace()
        data = self.embed_layer(data)
        output = self.generator(attention_mask=attention_mask,inputs_embeds=data,labels=labels)
        loss = output.loss
        scores = output.logits
        scores = self.softmax(scores)
        return loss, scores

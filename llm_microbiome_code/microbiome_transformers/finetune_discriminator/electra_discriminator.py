from transformers import ElectraConfig,ElectraForSequenceClassification
from transformers.models.electra.modeling_electra import ElectraClassificationHead
import torch.nn as nn
import torch
import pdb

class ElectraDiscriminator(nn.Module):

    def __init__(self,config:ElectraConfig,embeddings,discriminator = None, embed_layer = None, use_static_embeddings=False):
        super().__init__()
        self.use_static_embeddings = use_static_embeddings

        if use_static_embeddings:
            self.embed_layer = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=config.vocab_size-1)
            self.vocab_embed_dim = embeddings.shape[1]
            if self.vocab_embed_dim != config.hidden_size:
                self.projection = nn.Linear(self.vocab_embed_dim, config.hidden_size)
            self.classification_head = ElectraClassificationHead(config)
        else:
            self.embed_layer = nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.embedding_size,padding_idx = config.vocab_size-1)

            if embed_layer:
                self.embed_layer.load_state_dict(torch.load(embed_layer))
            else:
                self.embed_layer.weight = nn.Parameter(embeddings)
            if discriminator:
                self.discriminator = ElectraForSequenceClassification.from_pretrained(discriminator,config=config)
            else:
                self.discriminator = ElectraForSequenceClassification(config)
        self.softmax = nn.Softmax(1)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self,data,attention_mask,labels,frequencies=None,return_embeddings=False):
        #pdb.set_trace()
        data = self.embed_layer(data)
        if self.use_static_embeddings:
            if frequencies is None:
                # If no frequencies are provided, average over the sequence dimension
                data = data.mean(dim=1)
            else:
                # If frequencies are provided, weight the data by the normalized frequencies
                frequency_weights = frequencies / frequencies.sum(dim=1, keepdim=True)
                # Compute weighted average
                data = torch.sum(data * frequency_weights.unsqueeze(-1), dim=1)
            if hasattr(self,'projection'):
                data = self.projection(data)
            data = data.unsqueeze(1)
            logits = self.classification_head(data)
            loss = self.loss_fn(logits,labels)
            scores = self.softmax(logits)
        else:
            output = self.discriminator(attention_mask=attention_mask,inputs_embeds=data,labels=labels,output_hidden_states=return_embeddings)
            scores = self.softmax(output['logits'])
            loss = output['loss']
        if return_embeddings:
            if self.use_static_embeddings:
                return loss, scores, data
            else:
                return loss, scores, output['hidden_states'][-1]
        else:
            return loss, scores
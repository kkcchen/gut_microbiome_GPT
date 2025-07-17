from models import TransformerModel

from torch import nn, Tensor

class FinetunedTransformer(nn.Module):
    def __init__(self, model_config):
        super(FinetunedTransformer, self).__init__()
        base_model_config = model_config['base_model_config']
        self.base_model = TransformerModel(**base_model_config)
        self.classification_head = ClsDecoder(
            d_model=base_model_config['d_model'],
            n_cls=model_config['num_classes'],
        )
        
    def load_base_state_dict(self, state_dict):
        """
        Load the state dict into the base model.
        """
        self.base_model.load_state_dict(state_dict)
        
    
    def set_base_model_trainable(self, trainable: bool) -> list:
        """
        Freeze or unfreeze the base model weights and return the list of trainable parameters.
        
        Args:
            trainable (bool): If True, unfreeze the base model weights; if False, freeze them.
        
        Returns:
            list: A list of trainable parameters in the base model.
        """
        for param in self.base_model.parameters():
            param.requires_grad = False
        if trainable:
            encoder_blocks = [self.base_model.encoder, self.base_model.value_encoder, self.base_model.transformer_encoder]
            for block in encoder_blocks:
                for name, param in block.named_parameters():
                    if "mask_embedding" in name:
                        param.requires_grad = False  # Always freeze unused embedding
                    else:
                        param.requires_grad = trainable
        
        return [param for param in self.parameters() if param.requires_grad]
        

    def forward(
        self,
        src: Tensor, # (batch, seq_len)
        values: Tensor, # (batch, seq_len)
        src_key_padding_mask: Tensor, # (batch, seq_len)
    ) -> Tensor:
        output_dict = {}
        encoded_output, _ = self.base_model.encode(
            src,
            values,
            src_key_padding_mask,
        )
        env_emb = self.base_model.get_cell_emb_from_layer(encoded_output)  # (batch, embsize)
        output_dict['logits'] = self.classification_head(env_emb)
        return output_dict

class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
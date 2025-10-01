from models import TransformerModel
from torch import nn, Tensor

class RawMLP(nn.Module):
    """
    Decoder for classification or regression task, for raw MLP input.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        current_dim = input_dim
        for _ in range(nlayers - 1):
            self._decoder.append(nn.Linear(current_dim, hidden_dim))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim
        self.out_layer = nn.Linear(current_dim, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, input_dim]
        """
        output_dict = {}
        for layer in self._decoder:
            x = layer(x)
        output_dict['logits'] = self.out_layer(x)
        return output_dict
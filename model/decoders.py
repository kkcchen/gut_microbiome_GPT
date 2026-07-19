from torch import nn, Tensor
from typing import Dict

class AbundanceDecoder(nn.Module):
    """
    Decoder for per-taxon abundance logits, used by the masking reconstruction task.

    Predicts a logit per taxon from the contextual transformer output.
    """

    def __init__(
        self,
        d_model: int,
        num_special_tokens: int = 1,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Dimension of transformer hidden states
            num_special_tokens: Number of special tokens to skip at sequence start (default: 1 for sample embedding)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.num_special_tokens = num_special_tokens

        # Shared MLP backbone (512-512 style from scPRINT)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.pred_head = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: Transformer output of shape (batch_size, num_special_tokens + num_taxa, d_model)

        Returns:
            {"logits": Tensor} of shape (batch_size, num_taxa)
        """
        # Skip special tokens (e.g., sample embedding token)
        x_taxa = x[:, self.num_special_tokens:, :]  # (batch, num_taxa, d_model)

        # Apply decoder MLP
        h = self.decoder_mlp(x_taxa)  # (batch, num_taxa, d_model)

        # Predict per-taxon logits
        pred = self.pred_head(h)  # (batch, num_taxa, 1)

        return {
            "logits": pred.squeeze(-1)  # (batch, num_taxa)
        }


class TaxaIdentityDecoder(nn.Module):
    """
    Decoder for the masked-taxon-identity task.

    Predicts which taxon occupied a masked position from the contextual
    transformer output (i.e. from its neighbors and its still-visible
    abundance value).
    """

    def __init__(
        self,
        d_model: int,
        num_taxa: int,
        num_special_tokens: int = 1,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Dimension of transformer hidden states
            num_taxa: Size of the taxon vocabulary (classification output dimension)
            num_special_tokens: Number of special tokens to skip at sequence start (default: 1 for sample embedding)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.num_special_tokens = num_special_tokens

        # Shared MLP backbone (same shape as AbundanceDecoder)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.pred_head = nn.Linear(d_model, num_taxa)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: Transformer output of shape (batch_size, num_special_tokens + num_taxa, d_model)

        Returns:
            Dictionary with {"logits": Tensor} of shape (batch_size, num_taxa_positions, num_taxa_vocab)
        """
        x_taxa = x[:, self.num_special_tokens:, :]  # (batch, num_taxa, d_model)
        h = self.decoder_mlp(x_taxa)  # (batch, num_taxa, d_model)
        return {"logits": self.pred_head(h)}  # (batch, num_taxa, num_taxa_vocab)
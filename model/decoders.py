import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Dict

class AbundanceDecoder(nn.Module):
    """
    Decoder for taxa abundance denoising task.
    
    Predicts original taxa counts from noisy/downsampled input.
    Supports ZINB distribution or direct count prediction.
    """
    
    def __init__(
        self,
        d_model: int,
        num_special_tokens: int = 1,
        distribution: Optional[str] = "zinb",
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Dimension of transformer hidden states
            num_special_tokens: Number of special tokens to skip at sequence start (default: 1 for sample embedding)
            distribution: Distribution type - "zinb" for Zero-Inflated Negative Binomial, None for direct count prediction
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.num_special_tokens = num_special_tokens
        self.distribution = distribution
        
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
        
        # Distribution-specific prediction heads
        if self.distribution == "zinb":
            # Predict 3 ZINB parameters: mean, dispersion, zero-inflation probability
            self.pred_head = nn.Linear(d_model, 3)
        else:
            # Direct count prediction
            self.pred_head = nn.Linear(d_model, 1)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: Transformer output of shape (batch_size, num_special_tokens + num_taxa, d_model)
        
        Returns:
            Dictionary with distribution parameters:
                - If ZINB: {"mean": Tensor, "disp": Tensor, "pi": Tensor}
                - If None: {"pred": Tensor}
            All output tensors have shape (batch_size, num_taxa)
        """
        # Skip special tokens (e.g., sample embedding token)
        x_taxa = x[:, self.num_special_tokens:, :]  # (batch, num_taxa, d_model)
        
        # Apply decoder MLP
        h = self.decoder_mlp(x_taxa)  # (batch, num_taxa, d_model)
        
        # Predict distribution parameters
        pred = self.pred_head(h)  # (batch, num_taxa, 3) or (batch, num_taxa, 1)
        
        if self.distribution == "zinb":
            # Split into ZINB parameters
            mean_logits, disp_logits, pi_logits = pred.split(1, dim=-1)
            
            return {
                "mean": F.softplus(mean_logits.squeeze(-1)),  # Ensure positive mean
                "disp": torch.exp(torch.clamp(disp_logits.squeeze(-1), max=15)),  # Positive dispersion, clamp for stability
                "pi": torch.sigmoid(pi_logits.squeeze(-1)),  # Zero-inflation probability in [0, 1]
            }
        else:
            # Direct count prediction (non-negative)
            return {
                "pred": F.softplus(pred.squeeze(-1))  # (batch, num_taxa)
            }


class SampleProjection(nn.Module):
    """
    Projects sample-level embeddings to a lower-dimensional space.
    
    Useful for tasks like contrastive learning or batch effect correction.
    """
    
    def __init__(
        self,
        d_model: int,
        projection_dim: int,
    ):
        """
        Args:
            d_model: Dimension of transformer hidden states
            projection_dim: Dimension of projected sample embeddings
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Transformer cell_embedding of shape (batch_size, d_model)
        
        Returns:
            Projected sample embeddings of shape (batch_size, projection_dim)
        """
        # Assume the first token is the sample embedding token
        
        # Apply projection MLP
        projected = self.projection(x)  # (batch_size, projection_dim)
        
        return {"projected": projected}
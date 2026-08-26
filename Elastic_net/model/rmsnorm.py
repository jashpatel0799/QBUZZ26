"""
RMSNorm — LLaMA 3.2 style Root Mean Square Layer Normalization.

Unlike LayerNorm, RMSNorm:
- Does NOT subtract the mean (no re-centering)
- Does NOT have bias
- Only scales by learned weight after RMS normalization
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA 3.2 style).

    Computes: x * weight / sqrt(mean(x^2) + eps)

    Args:
        dim: Hidden dimension size.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization without the learned scale."""
        # x: (..., dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of same shape.
        """
        # Cast to float32 for numerical stability, then back to input dtype
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

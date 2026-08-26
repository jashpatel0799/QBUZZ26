"""
Dense Checkpoint-to-Checkpoint Skip Connections — NOVEL CONTRIBUTION.

This module implements DenseNet-style skip connections between designated
exit checkpoints. Unlike standard residual connections (within each block),
these connections let shallow checkpoint representations directly inform
deeper exit points, bypassing all intermediate blocks.

For 4 exit checkpoints at blocks {8, 12, 16, 18}, we create 6 projections:
    8->12, 8->16, 8->18, 12->16, 12->18, 16->18

Each projection is a learned linear map (no bias, small init scale).

The skip connections are additive:
    h12_eff = h12_raw + Proj(8->12)(h8)
    h16_eff = h16_raw + Proj(8->16)(h8) + Proj(12->16)(h12_eff)
    h18_eff = h18_raw + Proj(8->18)(h8) + Proj(12->18)(h12_eff) + Proj(16->18)(h16_eff)
"""

import torch
import torch.nn as nn


class DenseSkipConnections(nn.Module):
    """Dense checkpoint-to-checkpoint skip connections.

    Creates a learned linear projection for every pair of exit checkpoints
    (source -> target where source < target).

    Args:
        hidden_size: Model hidden dimension.
        exit_checkpoints: Sorted list of exit block indices (e.g., [8, 12, 16, 18]).
        skip_pairs: List of (source, target) pairs for skip connections.
        init_scale: Standard deviation for weight initialization.
    """

    def __init__(
        self,
        hidden_size: int,
        exit_checkpoints: list[int],
        skip_pairs: list[tuple[int, int]],
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.exit_checkpoints = exit_checkpoints
        self.skip_pairs = skip_pairs

        # Create one Linear(d, d, bias=False) per skip pair
        self.projections = nn.ModuleDict()
        for src, tgt in skip_pairs:
            key = f"{src}_to_{tgt}"
            proj = nn.Linear(hidden_size, hidden_size, bias=False)
            # Small init to avoid blowing up residuals at the start of training
            nn.init.normal_(proj.weight, mean=0.0, std=init_scale)
            self.projections[key] = proj

    def forward(
        self, hidden_states: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Apply dense skip connections to checkpoint hidden states.

        The first exit checkpoint is unchanged (no earlier checkpoints to skip from).
        Each subsequent checkpoint accumulates skip contributions from all
        earlier checkpoints in order.

        Args:
            hidden_states: Dict mapping exit block index to its hidden state.
                Example: {8: h8, 12: h12_raw, 16: h16_raw, 18: h18_raw}
                Each tensor has shape (batch, seq_len, hidden_size).

        Returns:
            Dict with same keys, but values updated with skip contributions.
            The update is done in-order so that h16 gets the already-updated h12.
        """
        # Work on a copy to avoid mutating input
        enhanced = dict(hidden_states)

        # Process in checkpoint order (so later checkpoints see updated earlier ones)
        for tgt in self.exit_checkpoints:
            for src in self.exit_checkpoints:
                if src >= tgt:
                    break  # Only earlier checkpoints can skip to later ones
                key = f"{src}_to_{tgt}"
                if key in self.projections:
                    enhanced[tgt] = enhanced[tgt] + self.projections[key](enhanced[src])

        return enhanced

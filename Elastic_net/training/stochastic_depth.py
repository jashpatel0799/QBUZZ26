"""
Stochastic Depth (LayerDrop) implementation.
"""

from typing import List


def get_survival_probs(n_layers: int, drop_rate: float, use_stochastic_depth: bool) -> List[float]:
    """Calculate survival probability for each layer.
    
    Args:
        n_layers: Total number of transformer blocks.
        drop_rate: Maximum drop rate for the deepest block.
        use_stochastic_depth: If False, returns all 1.0 (disabled).
        
    Returns:
        List of survival probabilities for each block (index 0 to n_layers-1).
    """
    if not use_stochastic_depth or drop_rate <= 0.0:
        return [1.0] * n_layers
        
    # Linearly decrease survival probability with depth
    # Layer 1 has prob ~1.0, deepest layer has prob 1.0 - drop_rate
    probs = []
    for i in range(n_layers):
        # i is 0-indexed, so we use i / (n_layers - 1) to scale from 0 to 1
        scale = i / max(1, n_layers - 1)
        layer_drop = drop_rate * scale
        probs.append(1.0 - layer_drop)
        
    return probs

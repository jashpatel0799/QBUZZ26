"""
Loss functions.
Computes weighted multi-exit cross-entropy loss.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


def multi_exit_loss(
    logits_dict: Dict[int, torch.Tensor],
    targets: torch.Tensor,
    weights: List[float],
    exit_checkpoints: List[int],
) -> Tuple[torch.Tensor, Dict[int, float]]:
    """Compute weighted sum of CrossEntropy loss across all exits.
    
    Args:
        logits_dict: Dict mapping exit block index to logits tensor of shape (batch, seq_len, vocab_size)
        targets: Target tokens tensor of shape (batch, seq_len)
        weights: List of weights corresponding to each exit in exit_checkpoints
        exit_checkpoints: List of exit block indices
        
    Returns:
        total_loss: Scalar tensor representing the weighted sum of losses
        exit_losses: Dict mapping exit block index to its individual loss value (float)
    """
    total_loss = 0.0
    exit_losses = {}
    
    for i, exit_k in enumerate(exit_checkpoints):
        if exit_k not in logits_dict:
            continue
            
        logits = logits_dict[exit_k]
        weight = weights[i]
        
        # Shift so that tokens < n predict n
        # logits: (batch, seq_len - 1, vocab_size)
        # targets: (batch, seq_len - 1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        
        # Flatten the tokens
        # We ignore padding index -100 if present
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        exit_losses[exit_k] = loss.item()
        total_loss += weight * loss
        
    return total_loss, exit_losses

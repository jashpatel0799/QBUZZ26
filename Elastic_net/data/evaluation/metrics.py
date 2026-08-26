"""
Evaluation metrics.
Computes PPL, BPC, Top-1/Top-5 accuracy, and handles multi-exit evaluation.
"""

import torch
import torch.nn.functional as F
import math
import logging
from typing import Dict, Any

from config import EXIT_CHECKPOINTS

logger = logging.getLogger(__name__)


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from CrossEntropy loss."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')


def compute_bpc(loss: float) -> float:
    """Compute Bits Per Character (BPC) from CrossEntropy loss."""
    return loss / math.log(2)


def compute_accuracies(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """Compute Top-1 and Top-5 accuracy.
    
    Args:
        logits: Shape (batch, seq_len, vocab_size)
        targets: Shape (batch, seq_len)
        
    Returns:
        top1_acc, top5_acc (as percentages 0-100)
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    
    # Flatten
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    
    # Ignore padding
    mask = flat_labels != -100
    valid_logits = flat_logits[mask]
    valid_labels = flat_labels[mask]
    
    if valid_labels.numel() == 0:
        return 0.0, 0.0
        
    # Top-1
    _, pred_top1 = valid_logits.topk(1, dim=-1)
    top1_correct = pred_top1.squeeze(-1).eq(valid_labels).float().sum()
    top1_acc = (top1_correct / valid_labels.numel()).item() * 100.0
    
    # Top-5
    _, pred_top5 = valid_logits.topk(5, dim=-1)
    # Check if target is in top-5 predictions
    valid_labels_expanded = valid_labels.unsqueeze(-1).expand_as(pred_top5)
    top5_correct = pred_top5.eq(valid_labels_expanded).float().sum()
    top5_acc = (top5_correct / valid_labels.numel()).item() * 100.0
    
    return top1_acc, top5_acc


def evaluate_all_exits(model: torch.nn.Module, dataloader, device: torch.device) -> Dict[int, Dict[str, float]]:
    """Force-exit at each checkpoint and compute all metrics.
    
    Returns:
        Dict mapping exit_k to dict of metrics (PPL, BPC, Top1, Top5, Loss)
    """
    model.eval()
    
    results = {k: {"loss": 0.0, "top1": 0.0, "top5": 0.0} for k in EXIT_CHECKPOINTS}
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=(device.type == "cuda")):
                logits_dict = model(batch)
                
            for k in EXIT_CHECKPOINTS:
                if k not in logits_dict:
                    continue
                    
                logits = logits_dict[k]
                
                # Loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                ).item()
                
                # Accuracies
                top1, top5 = compute_accuracies(logits, batch)
                
                results[k]["loss"] += loss
                results[k]["top1"] += top1
                results[k]["top5"] += top5
                
            num_batches += 1
            if num_batches >= 20:  # Limit for PoC speed
                break
                
    # Finalize metrics
    final_results = {}
    for k in EXIT_CHECKPOINTS:
        avg_loss = results[k]["loss"] / num_batches
        final_results[k] = {
            "loss": avg_loss,
            "ppl": compute_perplexity(avg_loss),
            "bpc": compute_bpc(avg_loss),
            "top1_acc": results[k]["top1"] / num_batches,
            "top5_acc": results[k]["top5"] / num_batches,
        }
        
    return final_results

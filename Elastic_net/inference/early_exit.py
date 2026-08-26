"""
Early exit routing logic.
Implements entropy-based threshold routing during generation.
"""

import torch
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm

from config import EXIT_CHECKPOINTS, ENTROPY_THRESHOLDS
from evaluation.metrics import compute_perplexity, compute_accuracies

logger = logging.getLogger(__name__)


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy of the logits distribution.
    
    Args:
        logits: Shape (batch, vocab_size)
        
    Returns:
        Entropy tensor of shape (batch,)
    """
    probs = torch.softmax(logits, dim=-1)
    # Add small epsilon to avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def generate_with_early_exit(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    tau: float,
    max_new_tokens: int = 50,
) -> Tuple[torch.Tensor, List[int]]:
    """Generate tokens using entropy-based early exit.
    
    Args:
        model: ElasticDepthLM
        input_ids: Shape (1, seq_len)
        tau: Entropy threshold. Lower = exits earlier.
        max_new_tokens: Max tokens to generate
        
    Returns:
        Tuple of (generated_ids, list_of_exit_blocks_used)
    """
    model.eval()
    device = input_ids.device
    
    current_ids = input_ids.clone()
    exit_log = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Embeddings
            x = model.embed_tokens(current_ids)
            
            raw_hidden_states = {}
            exit_chosen = EXIT_CHECKPOINTS[-1]
            
            # Run blocks sequentially
            for i, layer in enumerate(model.layers):
                x = layer(x)
                block_num = i + 1
                
                if block_num in model.exit_checkpoints:
                    raw_hidden_states[block_num] = x
                    
                    # Apply skip connections up to this block
                    enhanced = model.skip_connections(raw_hidden_states)
                    
                    # Compute logits for this exit (only for the last token)
                    h_norm = model.shared_lm_head_norm(enhanced[block_num][:, -1:, :])
                    logits = model.lm_head(h_norm).squeeze(1)  # (1, vocab_size)
                    
                    entropy = compute_entropy(logits).item()
                    
                    if entropy < tau or block_num == model.exit_checkpoints[-1]:
                        # Confident enough OR reached the end
                        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                        exit_chosen = block_num
                        break
                        
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            exit_log.append(exit_chosen)
            
    return current_ids, exit_log


def sweep_tau(model: torch.nn.Module, dataloader, device: torch.device) -> Dict[float, Dict[str, float]]:
    """Evaluate speed-quality tradeoff across different tau values."""
    model.eval()
    results = {}
    
    logger.info("Starting tau sweep for early exit...")
    
    for tau in ENTROPY_THRESHOLDS:
        total_loss = 0.0
        total_top1 = 0.0
        total_exit_layer = 0.0
        num_tokens = 0
        
        # We process token by token to simulate generation/scoring
        # For PoC, we just score a few batches to build the curve
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                seq_len = batch.size(1)
                
                # We process the batch fully, but calculate entropy for every token
                # This is a vectorized approximation of the token-by-token generation for evaluation
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=(device.type == "cuda")):
                    logits_dict = model(batch)
                
                # For each token position, find the first exit where H < tau
                batch_loss = 0.0
                batch_top1 = 0.0
                batch_exit_sum = 0
                
                # Shape tracking: logits_dict[k] is (batch, seq_len, vocab_size)
                # We want to select the logits from the earliest exit where H < tau
                batch_size = batch.size(0)
                active_tokens = seq_len - 1 # predicting next token
                
                # Build matrices
                entropies = torch.zeros(len(EXIT_CHECKPOINTS), batch_size, active_tokens, device=device)
                all_logits = torch.zeros(len(EXIT_CHECKPOINTS), batch_size, active_tokens, model.vocab_size, device=device)
                
                for idx, k in enumerate(EXIT_CHECKPOINTS):
                    logits = logits_dict[k][:, :-1, :] # (batch, seq_len-1, vocab_size)
                    all_logits[idx] = logits
                    entropies[idx] = compute_entropy(logits)
                    
                # Find first exit where entropy < tau, or default to last exit
                exit_mask = entropies < tau
                
                # We want the argmax of the cumsum to find the FIRST true value
                # Or just iterate since it's only 4 exits
                selected_logits = all_logits[-1].clone() # Default to deepest
                selected_exits = torch.full((batch_size, active_tokens), EXIT_CHECKPOINTS[-1], device=device)
                
                for idx in reversed(range(len(EXIT_CHECKPOINTS))):
                    k = EXIT_CHECKPOINTS[idx]
                    mask = exit_mask[idx]
                    selected_logits[mask] = all_logits[idx][mask]
                    selected_exits[mask] = k
                    
                # Compute metrics on selected logits
                shift_labels = batch[:, 1:].contiguous()
                
                flat_logits = selected_logits.view(-1, model.vocab_size)
                flat_labels = shift_labels.view(-1)
                
                valid_mask = flat_labels != -100
                if valid_mask.any():
                    valid_logits = flat_logits[valid_mask]
                    valid_labels = flat_labels[valid_mask]
                    valid_exits = selected_exits.view(-1)[valid_mask]
                    
                    loss = torch.nn.functional.cross_entropy(valid_logits, valid_labels, reduction='sum').item()
                    _, pred = valid_logits.topk(1, dim=-1)
                    top1 = pred.squeeze(-1).eq(valid_labels).float().sum().item()
                    
                    total_loss += loss
                    total_top1 += top1
                    total_exit_layer += valid_exits.float().sum().item()
                    num_tokens += valid_labels.numel()
                    
                num_batches += 1
                if num_batches >= 10: # Limit for speed
                    break
                    
        avg_loss = total_loss / num_tokens
        results[tau] = {
            "loss": avg_loss,
            "ppl": compute_perplexity(avg_loss),
            "top1_acc": (total_top1 / num_tokens) * 100.0,
            "avg_exit_layer": total_exit_layer / num_tokens,
        }
        logger.info(f"Tau {tau}: PPL={results[tau]['ppl']:.2f}, Avg Exit={results[tau]['avg_exit_layer']:.2f}")
        
    return results

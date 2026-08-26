%%writefile evaluate.py
import time
import torch
import torch.nn.functional as F
from config import Config

@torch.no_grad()
def evaluate_metrics(model, val_loader, cfg: Config):
    model.eval()
    perplexities = {k: [] for k in cfg.exits}
    latencies = {k: [] for k in cfg.exits}
    match_rates = {k: [] for k in cfg.exits[:-1]}

    device_type = "cuda" if "cuda" in cfg.device else "cpu"

    for batch in val_loader:
        input_ids = batch["input_ids"].to(cfg.device)
        targets = input_ids.clone()
        shift_targets = targets[:, 1:].contiguous()

        with torch.amp.autocast(device_type=device_type, dtype=cfg.dtype):
            for k in cfg.exits:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_t = time.perf_counter()
                single_exit_logits = model(input_ids, target_exit=k)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latency = (time.perf_counter() - start_t) * 1000 / cfg.batch_size
                latencies[k].append(latency)

            exit_logits = model(input_ids)

        teacher_preds = exit_logits[cfg.exits[-1]][:, :-1, :].argmax(dim=-1)

        for k in cfg.exits:
            shift_logits = exit_logits[k][:, :-1, :].contiguous().float()
            loss = F.cross_entropy(
                shift_logits.view(-1, cfg.vocab_size), 
                shift_targets.view(-1),
                ignore_index=cfg.pad_token_id
            )
            perplexities[k].append(torch.exp(loss).item())

            if k != cfg.exits[-1]:
                preds = shift_logits.argmax(dim=-1)
                valid_mask = (shift_targets != cfg.pad_token_id)
                agreement = ((preds == teacher_preds) & valid_mask).float().sum() / valid_mask.sum().clamp(min=1)
                match_rates[k].append(agreement.item())

    results = {}
    for k in cfg.exits:
        results[f"eval/ppl_exit_{k}"] = sum(perplexities[k]) / len(perplexities[k])
        results[f"eval/latency_ms_token_exit_{k}"] = sum(latencies[k]) / len(latencies[k])
        if k in match_rates:
            results[f"eval/agreement_with_exit18_{k}"] = sum(match_rates[k]) / len(match_rates[k])

    return results
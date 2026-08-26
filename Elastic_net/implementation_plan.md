# Elastic-Depth Decoder LM — Final Approved Plan

> Status: ✅ **APPROVED — Ready for code generation**

---

## All Decisions — Final

| # | Question | Answer | Decision |
|---|---|---|---|
| Q1 | Batch size | ✅ OK | batch=8, grad_accum=4, effective batch=32. Auto-fallback to batch=4 on OOM |
| Q2 | Tokenizer fallback | ✅ OK | Try `meta-llama/Llama-3.2-1B` first; fallback to `NousResearch/Meta-Llama-3-8B` |
| Q3 | File list changes | ✅ No changes | All 27 files as listed — confirmed |
| Q4 | W&B token handling | ❌ No silent skip | **Always require WANDB_TOKEN** — raise clear error if empty, never skip silently |
| Q5 | Stochastic depth default | `False` | `USE_STOCHASTIC_DEPTH = False` in config; toggle via CLI flag at runtime |

---

## Batch Size Decision

**Q: Can we increase batch from 4 to 8/12/16?**

Memory breakdown with hidden=1024, seq=512, gradient checkpointing ON:

| Batch | Logits memory (1 exit, bf16) | All 4 exits peak | Safe? |
|---|---|---|---|
| 4 | ~131 MB | ~524 MB | ✅ |
| 8 | ~262 MB | ~1.05 GB | ✅ |
| 12 | ~393 MB | ~1.57 GB | ✅ |
| 16 | ~524 MB | ~2.1 GB | ⚠️ tight |

> [!IMPORTANT]
> **Recommendation: batch=8, grad_accum=4 → effective batch=32.**
> This keeps peak memory safe and doubles throughput vs batch=4.
> Vocab=128,256 makes logits the dominant memory cost — not activations.
> We will compute exits sequentially (not simultaneously) to keep peak memory low.
> If OOM occurs at batch=8, script auto-falls back to batch=4.

---

## Repo Structure (Files to Generate)

```
/home/jash/Workspace/QBUZZ26/Elastic_net/
│
├── config.py                     [1]  All configs + HF_TOKEN + WANDB_TOKEN placeholders
├── requirements.txt              [2]  All pip dependencies
├── kaggle_train.py               [3]  Single entry point — runs everything
│
├── model/
│   ├── __init__.py               [4]
│   ├── rmsnorm.py                [5]  LLaMA-style RMSNorm
│   ├── rope.py                   [6]  RoPE with base=500,000
│   ├── attention.py              [7]  GQA (16 heads, 4 kv-heads)
│   ├── ffn.py                    [8]  SwiGLU FFN
│   ├── block.py                  [9]  Full transformer block
│   ├── skip_connections.py       [10] 6 dense checkpoint projections
│   └── elastic_model.py          [11] Full 18-block model + shared LM head
│
├── data/
│   ├── __init__.py               [12]
│   ├── tokenizer.py              [13] LLaMA 3.2 tokenizer loader + HF login
│   ├── datasets.py               [14] Load + stream all 5 datasets
│   └── dataloader.py             [15] Sequence packing + curriculum sampler
│
├── training/
│   ├── __init__.py               [16]
│   ├── loss.py                   [17] Weighted multi-exit cross-entropy
│   ├── stochastic_depth.py       [18] LayerDrop per block
│   └── trainer.py                [19] Full training loop + checkpointing
│
├── inference/
│   ├── __init__.py               [20]
│   └── early_exit.py             [21] Entropy-threshold routing at inference
│
├── evaluation/
│   ├── __init__.py               [22]
│   └── metrics.py                [23] PPL, BPC, Top-1, Top-5, Avg-exit-layer
│
├── utils/
│   ├── __init__.py               [24]
│   ├── plotting.py               [25] All 6 training/eval plots (matplotlib+seaborn)
│   └── wandb_logger.py           [26] W&B login + all logging helpers
│
└── scripts/
    └── dry_run.py                [27] Shape/memory check — run before full training
```

**Total: 27 files**

---

## Component Checklist

### Phase 0 — Setup
- [ ] **[1] `config.py`** — master config file
  - `HF_TOKEN = ""` placeholder
  - `WANDB_TOKEN = ""` placeholder
  - `WANDB_PROJECT = "elastic-depth-lm"`
  - `WANDB_RUN_NAME = "poc-18block-exits-8-12-16-18"`
  - Model config: `hidden=1024, n_layers=18, n_heads=16, n_kv_heads=4, intermediate=4096, vocab=128256, rope_base=500000, max_seq_len=512`
  - Exit config: `EXIT_CHECKPOINTS = [8, 12, 16, 18]`
  - Skip conn config: `SKIP_PAIRS = [(8,12),(8,16),(8,18),(12,16),(12,18),(16,18)]`
  - Loss weights: `PHASE1_WEIGHTS = [0.5,0.6,0.8,1.0]`, `PHASE2_WEIGHTS = [0.3,0.5,0.7,1.0]`
  - Train config: `batch=8, grad_accum=4, lr=3e-4, total_steps=20000, phase1_steps=12000, warmup=500`
  - Inference: `ENTROPY_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0]`
  - Paths: `OUTPUT_DIR, CHECKPOINT_DIR, PLOT_DIR`
  - **`USE_STOCHASTIC_DEPTH = False`** ← default OFF; enable via `--use-stochastic-depth` CLI flag
  - `STOCHASTIC_DEPTH_RATE = 0.1` ← only used when `USE_STOCHASTIC_DEPTH = True`

- [ ] **[2] `requirements.txt`**
  - `torch>=2.1.0`, `transformers>=4.40.0`, `datasets>=2.18.0`
  - `wandb`, `matplotlib`, `seaborn`, `tqdm`, `numpy`, `huggingface_hub`

---

### Phase 1 — Model Components

- [ ] **[5] `rmsnorm.py`**
  - `class RMSNorm(nn.Module)`: `x * weight / sqrt(mean(x²) + eps)`
  - No bias, no mean subtraction — exact LLaMA 3.2 style

- [ ] **[6] `rope.py`**
  - `class RotaryEmbedding`: precompute cos/sin cache
  - Base frequency = 500,000 (LLaMA 3.2 value)
  - `apply_rotary_emb(q, k, cos, sin)` function

- [ ] **[7] `attention.py`**
  - `class GroupedQueryAttention`: 16 Q heads, 4 KV heads
  - KV repetition: `repeat_kv(k, n_rep=4)`
  - Scaled dot-product attention (use `F.scaled_dot_product_attention` for flash-attn)
  - No bias in Q/K/V/O projections
  - Causal mask via `is_causal=True`

- [ ] **[8] `ffn.py`**
  - `class SwiGLUFFN`: `FFN(x) = down(SiLU(gate(x)) * up(x))`
  - No bias, intermediate=4096

- [ ] **[9] `block.py`**
  - `class TransformerBlock`: RMSNorm → GQA → residual → RMSNorm → FFN → residual
  - `stochastic_depth_prob` parameter (float, 0.0 = disabled)
  - If `stochastic_depth_prob == 0.0` (i.e., `USE_STOCHASTIC_DEPTH=False`): block always runs, zero overhead

- [ ] **[10] `skip_connections.py`**
  - `class DenseSkipConnections(nn.Module)`
  - Dict of 6 `nn.Linear(1024, 1024, bias=False)` projections
  - Keys: `"8->12"`, `"8->16"`, `"8->18"`, `"12->16"`, `"12->18"`, `"16->18"`
  - `forward(hidden_states_dict)` → returns updated dict with skip-enhanced states
  - Init: small normal init (scale=0.02) to avoid blowing up residuals at start

- [ ] **[11] `elastic_model.py`** — core novel model
  - `class ElasticDepthLM(nn.Module)`
  - `nn.Embedding(128256, 1024)` — token embedding
  - 18 `TransformerBlock` instances
  - `DenseSkipConnections` module
  - `shared_lm_head_norm = RMSNorm(1024)` — applied before shared head
  - `lm_head` weights **tied** to embedding weights
  - `forward()` returns dict: `{"logits_8": ..., "logits_12": ..., "logits_16": ..., "logits_18": ...}`
  - `forward_to_exit(x, exit_k)` — for inference, stops at block k
  - Gradient checkpointing: wrap each block segment between exits

---

### Phase 2 — Data Pipeline

- [ ] **[13] `tokenizer.py`**
  - `load_tokenizer()`:
    - `huggingface_hub.login(token=HF_TOKEN)` if `HF_TOKEN != ""`
    - Try `AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")`
    - Fallback: `AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")`
    - Set `tokenizer.pad_token = tokenizer.eos_token`

- [ ] **[14] `datasets.py`** — 5 dataset sources
  - Stream + tokenize each, then concatenate:
    1. `allenai/c4` en — 100K samples (streaming)
    2. `Skylion007/openwebtext` — 50K samples (streaming)
    3. `wikitext` / `wikitext-103-raw-v1` — full dataset
    4. `bookcorpus` — 50K samples (streaming)
    5. `monology/pile-uncopyrighted` — 50K samples (streaming)
  - Tokenize: `tokenizer(text, add_special_tokens=True)`
  - Pack into chunks of exactly 512 tokens (no padding — concatenate and split)
  - 80/10/10 split; WikiText-103 uses its own official val/test

- [ ] **[15] `dataloader.py`**
  - `PackedDataset(torch.utils.data.Dataset)` — serves packed 512-token sequences
  - `CurriculumSampler` — Phase 1: sort by rough difficulty (seq entropy), Phase 2: random
  - `get_dataloaders(phase)` → train/val DataLoader with `batch=8, num_workers=2`

---

### Phase 3 — Training Utilities

- [ ] **[17] `loss.py`**
  - `multi_exit_loss(logits_dict, targets, weights, phase)`:
    - Shift logits/targets for causal LM (predict next token)
    - CrossEntropy per exit (ignore padding index = -100)
    - Weighted sum: `L = Σ λ_k × L_k`
    - Return: `total_loss, {loss_8, loss_12, loss_16, loss_18}`
  - Note: compute losses **sequentially per exit**, free logits after each to save memory

- [ ] **[18] `stochastic_depth.py`**
  - `stochastic_depth_drop(block, x, survival_prob, training)`:
    - If `survival_prob >= 1.0` (disabled): pass-through, no randomness, no overhead
    - During training + enabled: skip block with prob `1 - survival_prob`
    - During eval: always run block (deterministic, regardless of flag)
    - `get_survival_probs(n_layers, drop_rate, use_stochastic_depth)` helper:
      - If `use_stochastic_depth=False`: returns `[1.0] * n_layers` (all blocks always active)
      - If `use_stochastic_depth=True`: `survival_prob(i) = 1 - (i/n_layers) * drop_rate`
  - **Effect of flag:**
    - `USE_STOCHASTIC_DEPTH=True` → stochastic depth active, model learns robustness to missing blocks
    - `USE_STOCHASTIC_DEPTH=False` → all 18 blocks always run, deterministic training (cleaner ablation baseline)

- [ ] **[19] `trainer.py`**
  - `class Trainer`:
    - `setup_optimizer()`: AdamW, lr=3e-4, betas=(0.9,0.95), wd=0.1
    - `setup_scheduler()`: cosine decay + warmup (500 steps)
    - `train_step()`: forward → multi-exit loss → backward → grad clip → step
    - `val_step()`: no grad, all 4 exit losses
    - `save_checkpoint(step)`: save model + optimizer + scheduler state
    - `load_checkpoint(path)`: restore full state for resumption
    - Logs every 50 steps: all exit losses to W&B + console
    - Validates every 500 steps: val loss + all 5 metrics
    - Auto batch-size fallback: catch OOM, halve batch, retry

---

### Phase 4 — W&B Logging

- [ ] **[26] `wandb_logger.py`**
  - `init_wandb()`:
    - If `WANDB_TOKEN == ""`: **raise `ValueError`** with clear message: `"WANDB_TOKEN is empty. Please set your W&B token in config.py before running."`
    - `wandb.login(key=WANDB_TOKEN)` — fails loudly if token invalid
    - `wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config=ALL_CONFIGS)`
  - `log_train_step(step, loss_dict, lr)`: logs per-exit train losses + LR
  - `log_val_step(step, loss_dict, metrics_dict)`: logs val losses + all 5 metrics
  - `log_plot(name, figure)`: logs matplotlib figures as W&B images
  - `log_exit_distribution(step, dist_dict)`: bar chart of exit distribution
  - `finish()`: `wandb.finish()`
  - **No silent fallback** — W&B is mandatory, always required

---

### Phase 5 — Evaluation & Inference

- [ ] **[21] `early_exit.py`**
  - `generate_with_early_exit(model, input_ids, tau, max_new_tokens)`:
    - At each new token, run blocks up to exit k
    - Compute entropy of logits_k; if `H < tau`: exit, take argmax as next token
    - Else: continue to next exit
    - Track which exit was used per token
  - `sweep_tau(model, val_loader, tau_values)`: returns `{tau: (avg_exit_layer, ppl)}`

- [ ] **[23] `metrics.py`**
  - `compute_perplexity(logits, targets)`: `exp(mean(CrossEntropy))`
  - `compute_bpc(logits, targets, char_counts)`: `loss / log(2)` per character
  - `compute_top1_accuracy(logits, targets)`: exact match rate
  - `compute_top5_accuracy(logits, targets)`: in-top-5 rate
  - `compute_avg_exit_layer(exit_log)`: mean block depth used across all tokens
  - `evaluate_all_exits(model, dataloader)`: force-exit at k ∈ {8,12,16,18}, compute all 5 metrics per exit

---

### Phase 6 — Plotting

- [ ] **[25] `plotting.py`** — 6 plots, all saved as PNG + logged to W&B

  | # | Function | Plot |
  |---|---|---|
  | 1 | `plot_train_losses(history)` | Step vs train loss — 4 exits + total |
  | 2 | `plot_val_losses(history)` | Step vs val loss — 4 exits (shows overfit/underfit) |
  | 3 | `plot_perplexity(history)` | Step vs PPL — 4 exits |
  | 4 | `plot_exit_distribution(dist)` | Bar chart: % tokens per exit at each τ |
  | 5 | `plot_tradeoff_curve(tau_results)` | τ vs (PPL, avg exit layer) — the "elastic" curve |
  | 6 | `plot_phase_curriculum(history)` | Loss weight λ_k per step — shows phase transition |

---

### Phase 7 — Entry Points

- [ ] **[27] `scripts/dry_run.py`**
  - Instantiate model, print param count, run 1 forward pass (batch=1, seq=16)
  - Print shape of all 4 exit logits
  - Print estimated VRAM usage
  - Run without any dataset loading — quick sanity check

- [ ] **[3] `kaggle_train.py`** — single script to rule all
  ```
  1. Parse args:
       --dry-run              → shape/memory check only, no training
       --resume PATH          → resume from checkpoint
       --eval-only            → skip training, run evaluation only
       --tau FLOAT            → entropy threshold for inference (default: 1.5)
       --no-stochastic-depth  → override USE_STOCHASTIC_DEPTH=False at runtime
       --use-stochastic-depth → override USE_STOCHASTIC_DEPTH=True at runtime
  2. Load config (runtime flag overrides config.py flag if provided)
  3. Init W&B (skip gracefully if WANDB_TOKEN is empty)
  4. Load tokenizer (HF login if HF_TOKEN provided)
  5. Load datasets + dataloaders
  6. Build model (bf16, gradient checkpointing ON)
  7. Print: stochastic depth = ON/OFF, all hyperparams
  8. If --dry-run: print shapes + VRAM estimate, exit
  9. If --resume: load checkpoint
  10. Phase 1 training loop (12,000 steps)
  11. Phase 2 training loop (8,000 steps)
  12. Final evaluation (all 5 metrics × 4 exits)
  13. τ sweep (early exit efficiency curve)
  14. Generate all 6 plots
  15. Save final checkpoint
  16. W&B finish
  ```

---

## Full Dependency Graph (Build Order)

```
config.py
    ↓
rmsnorm → rope → attention → ffn → block
                                      ↓
                          skip_connections → elastic_model
                                                  ↓
tokenizer → datasets → dataloader          loss + stochastic_depth
                           ↓                        ↓
                        trainer ←───────────────────┘
                           ↓
                    wandb_logger
                           ↓
              metrics + early_exit + plotting
                           ↓
                     kaggle_train.py
```

---

## What Is NOT in Scope (Future Work)
- Flash Attention 2 (use `F.scaled_dot_product_attention` instead — available in PyTorch 2.1+)
- Quantization / GGUF export
- Instruction fine-tuning / RLHF
- Full 1.38B scale training
- Distributed training (multi-GPU)

---

---

## Generation Order (27 files, strict dependency sequence)

```
Batch 1 — Foundation
  [1] config.py
  [2] requirements.txt
  [4] model/__init__.py
  [12] data/__init__.py
  [16] training/__init__.py
  [20] inference/__init__.py
  [22] evaluation/__init__.py
  [24] utils/__init__.py

Batch 2 — Model Components (build bottom-up)
  [5]  model/rmsnorm.py
  [6]  model/rope.py
  [7]  model/attention.py
  [8]  model/ffn.py
  [9]  model/block.py
  [10] model/skip_connections.py
  [11] model/elastic_model.py

Batch 3 — Data Pipeline
  [13] data/tokenizer.py
  [14] data/datasets.py
  [15] data/dataloader.py

Batch 4 — Training Utilities
  [17] training/loss.py
  [18] training/stochastic_depth.py
  [19] training/trainer.py

Batch 5 — Logging + Eval + Inference
  [26] utils/wandb_logger.py
  [23] evaluation/metrics.py
  [21] inference/early_exit.py
  [25] utils/plotting.py

Batch 6 — Entry Points
  [27] scripts/dry_run.py
  [3]  kaggle_train.py
```

---
*Status: ✅ APPROVED — Awaiting start signal to generate code*

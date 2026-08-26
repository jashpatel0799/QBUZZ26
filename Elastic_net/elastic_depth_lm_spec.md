# Elastic-Depth Decoder LM — Complete Technical Specification

---

## AGENDA (PPT Structure)
1. Problem Statement
2. Proposed Solution (Architecture + Training + Inference)
3. What Is Novel
4. Prior Work Comparison
5. Results (PoC targets — no real results yet)
6. Summary
7. Additional Info (PoC Setup for Kaggle)

---

---

# 1. PROBLEM STATEMENT

## Core Observation
Production NLP systems need models of **different sizes** for different use-cases:
- A **small fast model** (e.g., 9-block) for low-latency APIs
- A **medium model** (e.g., 12-block) for moderate tasks
- A **large model** (e.g., 24-block) for complex reasoning
- A **full model** (e.g., 36-block) for maximum quality

## Current Cost
Today, each size variant = a **completely separate training run**:
- Separate pretraining
- Separate fine-tuning
- Separate distillation (if used)
- Separate serving infrastructure

This multiplies compute cost, storage cost, and engineering effort by the number of variants.

## The Problem in One Sentence
> "We waste enormous compute training N separate models of different sizes when we could train one model that contains all N sizes simultaneously."

---

---

# 2. PROPOSED SOLUTION

## Core Idea (One Sentence)
> Train one large decoder-only transformer such that its intermediate block outputs at designated checkpoints are already good enough to produce outputs — eliminating the need to train separate smaller models.

---

## 2A. Architecture

### Base Model
- A standard **decoder-only transformer** with **N blocks total**
- (In PoC: N=12; in full paper: N=36 is illustrative, actual number tunable)
- Standard components per block: Multi-Head Self-Attention + FFN + RMSNorm + Residual

### Exit Checkpoints
Designate specific block depths as **exit points**:

```
N=12 PoC:   exits at blocks  {3, 6, 9, 12}
N=36 full:  exits at blocks  {9, 12, 24, 36}
```

At each exit checkpoint k, the hidden state is called **hₖ**.

### Dense Checkpoint-to-Checkpoint Skip Connections [NOVEL]
Every earlier checkpoint feeds **directly** into every later checkpoint via a learned linear projection:

```
h₃ ──────────────────────────────────────► (+) h₆'  ──► (+) h₉'  ──► (+) h₁₂'
     ↘ (proj₃→₆)                                ↘ (proj₆→₉)
      ────────────────────────────────────────────────────
     ↘ (proj₃→₉)
      ──────────────────────────────────────────────────────────────────
     ↘ (proj₃→₁₂)
```

More precisely, for exit k, the input is:

```
h_k_effective = h_k_raw + Σ_{j < k, j is checkpoint} Proj_{j→k}(h_j)
```

Where `Proj_{j→k}` is a linear layer `ℝ^d → ℝ^d`.

**Why this matters:** Shallow representations (h₃, h₆) can directly inform deeper exits without being filtered through all intermediate blocks. This helps resolve the "representation conflict" — earlier blocks know their output will directly help deeper exits too.

**Skip connections that exist (for 4 exits at 3,6,9,12):**
- 3→6, 3→9, 3→12
- 6→9, 6→12
- 9→12

Total skip connections = C(4,2) = **6 learned projections**

---

### Single Shared LM Head [NOVEL]
Instead of 4 separate output heads (one per exit), **one shared LM head** handles all exits:

```
Logits_k = W_head · RMSNorm(h_k_effective)

where:
  W_head  ∈ ℝ^{d × |V|}   (single shared matrix, |V| = vocab size)
  h_k     ∈ ℝ^{d}          (hidden state at exit k, after dense skip accumulation)
  k ∈ {3, 6, 9, 12}        (exit checkpoints)
```

**Why this matters:**
- Forces all exit hidden states to live in a **common representational space** — implicit alignment pressure across depths
- Model trained at exit-3 and exit-12 must both produce hidden states that the same W_head can decode
- Reduces total parameters (1 head vs 4 heads)
- **Deployment advantage:** any exit checkpoint works without swapping or loading a different head — just point to a different block's output

---

## 2B. Training

### Loss Function
Compute cross-entropy loss at **every exit checkpoint** simultaneously, weighted by depth:

```
L_total = λ₃·L₃ + λ₆·L₆ + λ₉·L₉ + λ₁₂·L₁₂

where:
  L_k    = CrossEntropy(Logits_k, target_tokens)
  λ_k    = depth weight (deeper exits get higher weight, e.g., 0.3, 0.5, 0.7, 1.0)
```

**No routing gate during training.** All exits are active every forward pass. All blocks remain trainable throughout the entire training (no freezing).

### Stochastic Depth (LayerDrop)
Apply **layer dropout** during training — randomly drop individual blocks with a survival probability that decreases for deeper blocks:

```
P(keep block i) = 1 - (i/N) * drop_rate

Example (N=12, drop_rate=0.2):
  Block 1:  P(keep) = 1.0
  Block 6:  P(keep) = 0.9
  Block 12: P(keep) = 0.8
```

This prevents the model from becoming over-reliant on any single block being present, and implicitly trains robustness for early exits.

### Two-Phase Curriculum
**Phase 1 — Homogeneous batches:**
- Group training samples roughly by "expected difficulty" (e.g., sentence length, perplexity estimate)
- Short/simple samples → trained with stronger early-exit loss weight
- Long/complex samples → trained with stronger deep-exit loss weight
- Stabilizes early-stage training

**Phase 2 — Mixed batches:**
- Standard random batching
- All exit losses equally active
- Fine-tunes and harmonizes all exit points together

---

## 2C. Inference

### Confidence-Based Threshold Routing (No Learned Gate)
At inference time, process the input token-by-token. At each exit checkpoint k, compute the **entropy** of the predicted next-token distribution:

```
H_k = -Σ_v  p_k(v) · log p_k(v)

where p_k = softmax(Logits_k)
```

Decision rule:
```
if H_k < τ:    EXIT HERE  →  use Logits_k as the prediction
else:           CONTINUE  →  run the next segment of blocks
```

- `τ` (tau) is a **single scalar hyperparameter** set at deployment time
- **Low τ** = exit early often = faster, slightly lower quality
- **High τ** = rarely exit early = slower, higher quality
- Same model, different τ → effectively different "model sizes"

### Why Entropy Works Here
- Low entropy = peaked distribution = model is confident about the next token → safe to exit
- High entropy = flat distribution = model is uncertain → needs more computation

No gradient, no learned parameters, no collapse risk.

---

---

# 3. WHAT IS NOVEL

| Component | Exists in Prior Work? | Your Contribution |
|---|---|---|
| Train-once, deploy-many | MatFormer (width), LayerSkip (depth) | Not novel alone |
| Stochastic depth / LayerDrop | Fan et al. 2019, LayerSkip | Not novel |
| Early-exit loss at checkpoints | LayerSkip, DeeBERT | Not novel |
| Confidence/entropy-based exit | CALM, DeeBERT | Not novel |
| **Dense checkpoint-to-checkpoint skip connections** | ❌ Not in any prior work | ✅ **YOUR NOVEL CONTRIBUTION #1** |
| **Single shared LM head across all exits** | ❌ Not in any prior work | ✅ **YOUR NOVEL CONTRIBUTION #2** |

### Novel Contribution #1 — Dense Checkpoint Skip Connections
Prior early-exit methods (LayerSkip, DeeBERT) treat each exit as independent — block k only sees information that passed through blocks 1→k sequentially. Your model lets h₃ **directly inform** h₁₂ via a skip projection, bypassing intermediate blocks. Hypothesis: this makes earlier exits significantly better because they benefit from "hints" from the training dynamics of the full path.

### Novel Contribution #2 — Single Shared LM Head as Representational Aligner
Prior work either uses separate heads per exit (4 separate W matrices) or shares a head without motivation. Your shared head is motivated as a **representational alignment mechanism**: all exits are forced to produce hidden states that are decodable by the same linear map, which creates implicit training pressure for all exit representations to be in the same semantic space. This is a testable hypothesis via representational similarity analysis (RSA / CKA).

---

---

# 4. PRIOR WORK COMPARISON

| Paper | Goal | Depth-based? | Shared Head? | Dense Skips? | Routing |
|---|---|---|---|---|---|
| **MatFormer** (Google, 2023) | Train once, extract sizes | ❌ Width (neurons) | N/A | ❌ | Manual |
| **LayerSkip** (Meta, 2024) | Early exit via layer dropout | ✅ | ❌ Separate | ❌ | Confidence |
| **LayerDrop** (Fan et al., 2019) | Structured dropout for pruning | ✅ | ❌ | ❌ | None |
| **Mixture-of-Depths** (DeepMind, 2024) | Per-token depth routing | ✅ | ❌ | ❌ | Learned (per-token) |
| **CALM** (Schuster et al., 2022) | Token-level early exit | ✅ | ❌ | ❌ | Confidence |
| **Your Idea** | Train once, extract sizes | ✅ | ✅ Single shared | ✅ | Confidence (entropy) |

**Your idea = LayerSkip + dense checkpoint skips + single shared head**

The delta over LayerSkip is the two novel components above. A paper needs to ablate each component against a clean LayerSkip baseline to demonstrate their individual contributions.

---

---

# 5. RESULTS (Proof-of-Concept Targets — No Results Yet)

## What to Measure
1. **Perplexity at each exit** vs. a standalone model trained only to that depth (baseline)
   - Hypothesis: your exit-6 perplexity < standalone 6-block model perplexity
2. **Perplexity at each exit** vs. LayerSkip at same exit depth (ablation baseline)
   - Hypothesis: dense skip connections improve early exits over LayerSkip
3. **Exit distribution under entropy threshold τ** — what % of tokens exit at each depth
4. **Training curve** — do all exit losses decrease smoothly?

## Success Criteria for PoC
- Exit-k perplexity is within 5-10% of a standalone k-block model trained from scratch
- Dense skip ablation shows measurable improvement over LayerSkip at early exits (even 0.5 perplexity points on WikiText-2 counts)

---

---

# 6. SUMMARY

## One-Paragraph Summary
Elastic-Depth Decoder LM trains a single large decoder-only transformer such that its intermediate outputs at designated block depths (called exit checkpoints) are already usable for next-token prediction. This is achieved by: (a) computing a weighted early-exit loss at every checkpoint during training — so all exits are trained simultaneously; (b) connecting every earlier checkpoint directly to every later checkpoint via learned linear projections (dense checkpoint skip connections) — so shallow representations can inform deeper exits without going through all intermediate blocks; and (c) using a single shared LM head with RMSNorm for all exits — forcing representational alignment across depths. At inference, a simple entropy threshold on the predicted token distribution decides whether to exit at the current checkpoint or continue deeper. The result is one model that can run as a fast small model (exit early) or a high-quality large model (exit late) depending on the entropy threshold, with no retraining required for each variant.

## Key Claims
1. Training one elastic model is cheaper than training N separate size variants
2. Dense checkpoint skip connections improve early-exit quality over LayerSkip
3. A single shared LM head creates implicit representational alignment across exit depths
4. Entropy-based routing is stable, simple, and effective — no learned gate needed

---

---

# 7. ADDITIONAL INFO — KAGGLE POC SETUP

## Hardware Budget
- **GPU:** Single T4/P100 (14–16 GB VRAM)
- **Dataset:** WikiText-2 (small, standard LM benchmark, freely available)
- **Training time target:** Under 3 hours on T4

## Model Config (Fits in 14 GB)
```python
MODEL_CONFIG = {
    "vocab_size":     50257,      # GPT-2 tokenizer
    "n_blocks":       12,         # total transformer blocks
    "d_model":        512,        # hidden dimension
    "n_heads":        8,          # attention heads
    "d_ff":           2048,       # feedforward dimension
    "max_seq_len":    256,        # context window (keep short for PoC)
    "dropout":        0.1,
    "exit_checkpoints": [3, 6, 9, 12],  # exit at these block indices
    "exit_loss_weights": [0.3, 0.5, 0.7, 1.0],  # λ per exit
    "stochastic_depth_rate": 0.2,
}
```

Estimated parameter count: ~85M params → fits comfortably in 14 GB

## Skip Connection Config
```python
# 6 skip projections for 4 exit points
SKIP_CONNECTIONS = [
    (3, 6),   # proj from exit-3 hidden to exit-6 input
    (3, 9),
    (3, 12),
    (6, 9),
    (6, 12),
    (9, 12),
]
# Each is nn.Linear(d_model, d_model, bias=False)
```

## Training Config
```python
TRAIN_CONFIG = {
    "batch_size":     16,
    "grad_accumulation": 4,       # effective batch = 64
    "lr":             3e-4,
    "warmup_steps":   500,
    "total_steps":    10000,
    "phase1_steps":   6000,       # homogeneous batches
    "phase2_steps":   4000,       # mixed batches
    "optimizer":      "AdamW",
    "weight_decay":   0.1,
    "grad_clip":      1.0,
}
```

## Inference Config
```python
INFERENCE_CONFIG = {
    "entropy_threshold": 1.5,   # τ — tune this to trade speed vs quality
    # τ=0.5  → exits very early (fastest, lowest quality)
    # τ=1.5  → moderate early exit
    # τ=3.0  → rarely exits early (near full model quality)
}
```

## Files to Implement (Repo Structure)
```
elastic_depth_lm/
├── README.md
├── requirements.txt
├── config.py               # MODEL_CONFIG, TRAIN_CONFIG, INFERENCE_CONFIG
├── model/
│   ├── __init__.py
│   ├── attention.py        # Multi-head self-attention block
│   ├── ffn.py              # Feed-forward network block
│   ├── transformer_block.py # Single transformer block (attn + ffn + norms)
│   ├── skip_connections.py  # Dense checkpoint-to-checkpoint projections
│   └── elastic_lm.py        # Full model: blocks + skips + shared LM head
├── training/
│   ├── __init__.py
│   ├── dataset.py          # WikiText-2 loader + tokenizer
│   ├── loss.py             # Weighted multi-exit cross-entropy loss
│   ├── stochastic_depth.py # LayerDrop implementation
│   └── trainer.py          # Training loop (phase 1 + phase 2)
├── inference/
│   ├── __init__.py
│   └── generate.py         # Entropy-based early exit generation
├── evaluation/
│   ├── __init__.py
│   └── eval_perplexity.py  # Per-exit perplexity on WikiText-2 test set
└── train_kaggle.py         # Single entry-point script for Kaggle notebook
```

## Evaluation Script Logic
```
For each exit k in {3, 6, 9, 12}:
    Force all tokens to exit at block k (ignore entropy threshold)
    Compute perplexity on WikiText-2 test set
    Report as "Exit-k Perplexity"

Compare against:
    Baseline A: Standalone k-block model trained from scratch
    Baseline B: LayerSkip (same depth, no dense skips, separate heads)
```

## Ablations to Run (Even on PoC Scale)
| Ablation | What It Tests |
|---|---|
| Remove all dense skip connections | Is skip connection contribution real? |
| Use 4 separate LM heads instead of shared | Does shared head help? |
| Combine both removals | Reduces to LayerSkip baseline |

---

## Key Citations (For PPT References Slide)
1. **LayerSkip** — Elhoushi et al., Meta, 2024. arXiv:2404.16710
2. **MatFormer** — Devvrit et al., Google, 2023. arXiv:2310.07707
3. **LayerDrop** — Fan et al., Facebook AI, 2019. arXiv:1909.11556
4. **Mixture-of-Depths** — Raposo et al., DeepMind, 2024. arXiv:2404.02258
5. **CALM** — Schuster et al., Google, 2022. arXiv:2207.07061
6. **DenseNet** — Huang et al., 2017. arXiv:1608.06993 (inspiration for dense skip connections)

---

*Document Version: 1.0 | Status: Pre-experiment (no empirical results yet)*

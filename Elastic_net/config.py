"""
Elastic-Depth Decoder LM — Master Configuration
All hyperparameters, tokens, and paths in one place.
"""

import os

# ============================================================================
# API TOKENS — Fill these before running
# ============================================================================
HF_TOKEN = ""        # HuggingFace token for LLaMA 3.2 tokenizer access
WANDB_TOKEN = ""     # Weights & Biases token — REQUIRED, will error if empty

# ============================================================================
# W&B Configuration
# ============================================================================
WANDB_PROJECT = "elastic-depth-lm"
WANDB_RUN_NAME = "poc-18block-exits-8-12-16-18"

# ============================================================================
# Model Architecture (LLaMA 3.2 style, scaled down to ~405M params)
# ============================================================================
MODEL_CONFIG = {
    "vocab_size": 128256,          # LLaMA 3.2 tokenizer vocab
    "hidden_size": 1024,           # d_model (scaled from 2048)
    "intermediate_size": 4096,     # FFN intermediate (4× hidden)
    "num_heads": 16,               # Query heads (scaled from 32)
    "num_kv_heads": 4,             # KV heads for GQA (4:1 ratio)
    "head_dim": 64,                # Per-head dimension
    "num_layers": 18,              # Total transformer blocks
    "max_seq_len": 512,            # Context window for PoC
    "rope_base": 500_000.0,        # RoPE base frequency (LLaMA 3.2 value)
    "rms_norm_eps": 1e-5,          # RMSNorm epsilon
    "dropout": 0.0,                # No standard dropout (using stochastic depth instead)
}

# ============================================================================
# Exit Checkpoints & Skip Connections (NOVEL)
# ============================================================================
EXIT_CHECKPOINTS = [8, 12, 16, 18]   # Block indices where early exit is possible

# Dense checkpoint-to-checkpoint skip connection pairs (NOVEL)
# Every earlier checkpoint feeds directly into every later checkpoint
SKIP_PAIRS = [
    (8, 12),
    (8, 16),
    (8, 18),
    (12, 16),
    (12, 18),
    (16, 18),
]

SKIP_INIT_SCALE = 0.02  # Small init for skip projections to avoid blowing up residuals

# ============================================================================
# Loss Weights (per exit, per training phase)
# ============================================================================
# Phase 1: favor early exits more for stable staged training
PHASE1_EXIT_WEIGHTS = [0.5, 0.6, 0.8, 1.0]  # for exits [8, 12, 16, 18]

# Phase 2: standard weights for joint harmonization
PHASE2_EXIT_WEIGHTS = [0.3, 0.5, 0.7, 1.0]  # for exits [8, 12, 16, 18]

# ============================================================================
# Stochastic Depth (LayerDrop)
# ============================================================================
USE_STOCHASTIC_DEPTH = False    # Default OFF; enable via --use-stochastic-depth CLI flag
STOCHASTIC_DEPTH_RATE = 0.1     # Max drop rate for deepest layer (only used if enabled)

# ============================================================================
# Training Configuration
# ============================================================================
TRAIN_CONFIG = {
    "max_seq_len": MODEL_CONFIG["max_seq_len"],
    "batch_size": 4,               # Per-GPU batch size
    "grad_accumulation_steps": 2,  # Effective batch = 8 × 4 = 32
    "lr": 1e-3,                    # Peak learning rate
    "lr_scheduler": "cosine",      # Cosine decay with warmup
    "warmup_steps": 500,           # Linear warmup steps
    "total_steps": 20_000,         # Total training steps
    "phase1_steps": 12_000,        # Phase 1 (homogeneous curriculum)
    "phase2_steps": 8_000,         # Phase 2 (mixed batches)
    "betas": (0.9, 0.95),          # AdamW betas
    "weight_decay": 0.1,           # AdamW weight decay
    "grad_clip": 1.0,              # Gradient clipping max norm
    "precision": "bf16",           # Training precision
    "gradient_checkpointing": True,  # Activation checkpointing to save memory
    "log_interval": 50,            # Log every N steps
    "val_interval": 500,           # Validate every N steps
    "save_interval": 2000,         # Save checkpoint every N steps
    "fallback_batch_size": 4,      # Fallback if OOM with primary batch size
}

# ============================================================================
# Inference Configuration
# ============================================================================
ENTROPY_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0]  # τ values for speed-quality sweep
DEFAULT_TAU = 1.5                                  # Default entropy threshold

# ============================================================================
# Dataset Configuration
# ============================================================================
DATASET_CONFIG = {
    "datasets": {
        "c4": {
            "path": "allenai/c4",
            "name": "en",
            "split": "train",
            "streaming": True,
            "num_samples": 100_000,
            "text_field": "text",
        },
        "openwebtext": {
            "path": "Skylion007/openwebtext",
            "name": None,
            "split": "train",
            "streaming": True,
            "num_samples": 50_000,
            "text_field": "text",
        },
        "wikitext": {
            "path": "wikitext",
            "name": "wikitext-103-raw-v1",
            "split": "train",
            "streaming": True,
            "num_samples": None,  # Use full dataset
            "text_field": "text",
        },
        "bookcorpus": {
            "path": "bookcorpus/bookcorpus",
            "name": None,
            "split": "train",
            "streaming": True,
            "num_samples": 50_000,
            "text_field": "text",
        },
        "pile": {
            "path": "monology/pile-uncopyrighted",
            "name": None,
            "split": "train",
            "streaming": True,
            "num_samples": 50_000,
            "text_field": "text",
        },
    },
    "val_dataset": {
        "path": "wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "validation",
        "text_field": "text",
    },
    "test_dataset": {
        "path": "wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "test",
        "text_field": "text",
    },
}

# ============================================================================
# Tokenizer Configuration
# ============================================================================
TOKENIZER_CONFIG = {
    "primary": "meta-llama/Llama-3.2-1B",              # Requires HF approval
    "fallback": "NousResearch/Meta-Llama-3-8B",         # Open access, same tokenizer
}

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Create directories
for d in [OUTPUT_DIR, CHECKPOINT_DIR, PLOT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

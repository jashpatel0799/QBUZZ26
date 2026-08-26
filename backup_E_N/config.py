%%writefile config.py
import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Your Custom Model Architecture
    vocab_size: int = 128256  # Dynamically synced from Llama 3.2 tokenizer
    pad_token_id: int = 128004
    dim: int = 768
    intermediate_size: int = 2048
    n_layers: int = 18
    n_heads: int = 12
    n_kv_heads: int = 4  # Grouped-Query Attention
    head_dim: int = 64   # dim // n_heads
    max_seq_len: int = 512
    norm_eps: float = 1e-5
    
    # Checkpoint Exits & Highways (Your Original Setup)
    exits: List[int] = field(default_factory=lambda: [8, 12, 16, 18])
    
    # Precision
    dtype: torch.dtype = torch.bfloat16
    
    # Training Hyperparameters
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3e-4
    max_steps: int = 1000
    eval_interval: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configurable Loss Switch
    use_kl_distillation: bool = False
    ce_weight: float = 1.0
    kl_weight: float = 0.5
    distill_temp: float = 2.0
    
    # Tracking
    wandb_project: str = "elastic-depth-decoder-poc"
    wandb_run_name: str = "dense-highway-llama-18b"
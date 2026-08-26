#!/usr/bin/env python3
"""
Test script to execute kaggle_train.py directly on CPU using mocks
for Weights & Biases and HuggingFace dataset downloads.
"""

import sys
import os
import torch
import logging
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_kaggle_train")

# Add workspace directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1. Modify config before importing project modules that bind config values
import config

logger.info("Scaling down configuration for kaggle_train.py test...")
config.MODEL_CONFIG.update({
    "vocab_size": 2000,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_heads": 4,
    "num_kv_heads": 2,
    "head_dim": 16,
    "num_layers": 6,
    "max_seq_len": 64,
})

config.EXIT_CHECKPOINTS = [2, 4, 6]
config.SKIP_PAIRS = [
    (2, 4),
    (2, 6),
    (4, 6),
]

config.PHASE1_EXIT_WEIGHTS = [0.5, 0.8, 1.0]
config.PHASE2_EXIT_WEIGHTS = [0.3, 0.7, 1.0]

config.TRAIN_CONFIG.update({
    "batch_size": 2,
    "grad_accumulation_steps": 2, # Effective batch = 4
    "total_steps": 5,
    "phase1_steps": 3,
    "phase2_steps": 2,
    "log_interval": 1,
    "val_interval": 2,
    "save_interval": 2,
    "precision": "fp32",
})

# Mock dataset class yielding synthetic token lists
class MockDataset:
    def __init__(self, num_samples=10, seq_len=64, vocab_size=2000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __iter__(self):
        for _ in range(self.num_samples):
            # Yield structure expected by PackedDataset: dict with "input_ids"
            yield {"input_ids": torch.randint(0, self.vocab_size, (self.seq_len + 1,)).tolist()}

def main():
    # Force argv to contain only script name (simulating no command line overrides)
    sys.argv = ["kaggle_train.py"]
    
    # We define mocks to replace HuggingFace dataset downloads and W&B initialization
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "[PAD]"
    mock_tokenizer.eos_token = "[EOS]"
    
    mock_train_dataset = MockDataset(num_samples=20)
    mock_eval_dataset = MockDataset(num_samples=10)

    # Patch the data loaders/tokenizers at the root source
    with patch("data.tokenizer.load_tokenizer", return_value=mock_tokenizer), \
         patch("data.datasets.load_training_data", return_value=mock_train_dataset), \
         patch("data.datasets.load_eval_data", return_value=mock_eval_dataset), \
         patch("utils.wandb_logger.WandbLogger.init_wandb") as mock_init_wandb, \
         patch("utils.wandb_logger.WandbLogger.finish") as mock_finish_wandb, \
         patch("torch.cuda.is_available", return_value=False): # Force CPU mode
         
         logger.info("Mocks applied. Importing kaggle_train...")
         import kaggle_train
         
         logger.info("Executing kaggle_train.main()...")
         kaggle_train.main()
         logger.info("Execution finished successfully!")

if __name__ == "__main__":
    main()

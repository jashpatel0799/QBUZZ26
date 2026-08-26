#!/usr/bin/env python3
"""
Test script to run 5 steps of training and evaluation on CPU with synthetic data.
Scales down the model size dynamically for extremely fast CPU testing.
"""

import os
import torch
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_cpu_run")

# Add the workspace directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1. Import config and scale down parameters
import config

logger.info("Scaling down configuration for CPU testing...")
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
    "grad_accumulation_steps": 2, # Effective batch size = 4
    "total_steps": 5,
    "phase1_steps": 3,
    "phase2_steps": 2,
    "log_interval": 1,
    "val_interval": 2,
    "save_interval": 2,
    "precision": "fp32", # Use standard fp32 for CPU stability
})

# 2. Import project modules (they will now read the scaled-down config values)
from model.elastic_model import ElasticDepthLM
from training.trainer import Trainer
from utils.wandb_logger import WandbLogger
from evaluation.metrics import evaluate_all_exits
from inference.early_exit import sweep_tau
from utils.plotting import plot_all
from data.dataloader import get_dataloaders

class MockDataset:
    def __init__(self, num_samples=100, seq_len=64, vocab_size=2000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __iter__(self):
        for _ in range(self.num_samples):
            # yield dictionary with "input_ids"
            # PackedDataset needs seq_len + 1 tokens
            yield {"input_ids": torch.randint(0, self.vocab_size, (self.seq_len + 1,)).tolist()}

def main():
    logger.info("Starting CPU test run (5 steps)...")
    
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # 3. Create synthetic datasets
    vocab_size = config.MODEL_CONFIG["vocab_size"]
    seq_len = config.MODEL_CONFIG["max_seq_len"]
    
    train_dataset = MockDataset(num_samples=20, seq_len=seq_len, vocab_size=vocab_size)
    val_dataset = MockDataset(num_samples=10, seq_len=seq_len, vocab_size=vocab_size)
    
    # 4. Load data loaders
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, config.TRAIN_CONFIG)
    
    # 5. Initialize model
    logger.info("Initializing Model...")
    model = ElasticDepthLM().to(device)
    logger.info(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 6. Initialize (Mocked) Wandb Logger
    # W&B is not initialized with WANDB_TOKEN so we don't call init_wandb.
    # W&B calls in logging methods check is_initialized, so they will be skipped safely.
    wandb_logger = WandbLogger()
    
    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        wandb_logger=wandb_logger,
        device=device,
    )
    
    # Track metrics history to test plotting
    train_history = []
    val_history = []
    
    # We will override trainer's logging to capture history
    original_log_train_step = trainer.wandb_logger.log_train_step
    original_log_val_step = trainer.wandb_logger.log_val_step
    
    def mock_log_train_step(step, loss_dict, lr):
        original_log_train_step(step, loss_dict, lr)
        metrics = {"step": step, "train_loss_total": loss_dict["train_loss_total"]}
        for k, v in loss_dict.items():
            metrics[k] = v
        train_history.append(metrics)
        
    def mock_log_val_step(step, val_dict):
        original_log_val_step(step, val_dict)
        metrics = {"step": step, "val_loss_total": val_dict["val_loss_total"]}
        for k, v in val_dict.items():
            metrics[k] = v
        val_history.append(metrics)
        
    trainer.wandb_logger.log_train_step = mock_log_train_step
    trainer.wandb_logger.log_val_step = mock_log_val_step
    
    # 8. Run training loop
    logger.info("Starting training loop...")
    trainer.train()
    logger.info("Training loop completed successfully!")
    
    # 9. Run final evaluation
    logger.info("Running evaluation on all exits...")
    eval_metrics = evaluate_all_exits(model, val_loader, device)
    for k, metrics in eval_metrics.items():
        logger.info(f"Exit {k}: PPL={metrics['ppl']:.2f}, Top1={metrics['top1_acc']:.2f}%")
        
    # 10. Run tau sweep
    logger.info("Running Tau sweep...")
    tau_results = sweep_tau(model, val_loader, device)
    
    # 11. Plotting
    logger.info("Generating and saving plots...")
    # Fill in some dummy history if too short
    if not train_history:
        train_history = [{"step": i, "train_loss_total": 2.0} for i in range(1, 6)]
    if not val_history:
        val_history = [{"step": i, "val_loss_total": 2.0} for i in [2, 4]]
        
    plot_all(train_history, val_history, tau_results, config.__dict__)
    logger.info(f"Plots saved to {config.PLOT_DIR}")
    
    logger.info("Test run completed successfully!")

if __name__ == "__main__":
    main()

"""
Weights & Biases logger.
Handles W&B initialization, logging, and error handling for missing tokens.
"""

import wandb
import logging
from typing import Dict, Any

from config import WANDB_TOKEN, WANDB_PROJECT, WANDB_RUN_NAME, MODEL_CONFIG, TRAIN_CONFIG, EXIT_CHECKPOINTS, USE_STOCHASTIC_DEPTH, STOCHASTIC_DEPTH_RATE

logger = logging.getLogger(__name__)


class WandbLogger:
    def __init__(self):
        self.is_initialized = False

    def init_wandb(self):
        """Initialize W&B run. Raises ValueError if token is missing."""
        if not WANDB_TOKEN:
            raise ValueError(
                "WANDB_TOKEN is empty. Please set your W&B token in config.py before running."
            )
            
        try:
            wandb.login(key=WANDB_TOKEN)
            
            all_configs = {
                "model": MODEL_CONFIG,
                "train": TRAIN_CONFIG,
                "exit_checkpoints": EXIT_CHECKPOINTS,
                "use_stochastic_depth": USE_STOCHASTIC_DEPTH,
                "stochastic_depth_rate": STOCHASTIC_DEPTH_RATE if USE_STOCHASTIC_DEPTH else 0.0,
            }
            
            wandb.init(
                project=WANDB_PROJECT,
                name=WANDB_RUN_NAME,
                config=all_configs,
            )
            self.is_initialized = True
            logger.info("W&B initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            raise

    def log_train_step(self, step: int, loss_dict: Dict[str, float], lr: float):
        """Log training step metrics."""
        if not self.is_initialized:
            return
            
        metrics = {"train/lr": lr, **{f"train/{k}": v for k, v in loss_dict.items()}}
        wandb.log(metrics, step=step)

    def log_val_step(self, step: int, metrics_dict: Dict[str, float]):
        """Log validation metrics."""
        if not self.is_initialized:
            return
            
        metrics = {f"val/{k}": v for k, v in metrics_dict.items()}
        wandb.log(metrics, step=step)

    def log_plot(self, name: str, figure: Any, step: int = None):
        """Log a matplotlib figure."""
        if not self.is_initialized:
            return
            
        try:
            wandb.log({name: wandb.Image(figure)}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log plot {name} to W&B: {e}")

    def log_exit_distribution(self, step: int, dist_dict: Dict[int, float], tau: float):
        """Log exit distribution for a specific tau."""
        if not self.is_initialized:
            return
            
        data = [[exit_k, pct] for exit_k, pct in dist_dict.items()]
        table = wandb.Table(data=data, columns=["Exit Checkpoint", "Percentage"])
        wandb.log(
            {f"eval/exit_dist_tau_{tau}": wandb.plot.bar(table, "Exit Checkpoint", "Percentage", title=f"Exit Distribution (tau={tau})")},
            step=step
        )

    def finish(self):
        """Finish W&B run."""
        if self.is_initialized:
            wandb.finish()
            self.is_initialized = False
            logger.info("W&B run finished.")

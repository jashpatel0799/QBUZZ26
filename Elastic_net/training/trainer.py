"""
Trainer class.
Handles the training loop, phases, checkpointing, and evaluation.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
import logging
from typing import Dict, Optional

from config import TRAIN_CONFIG, PHASE1_EXIT_WEIGHTS, PHASE2_EXIT_WEIGHTS, EXIT_CHECKPOINTS, CHECKPOINT_DIR
from training.loss import multi_exit_loss
from utils.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5
):
    """Create a schedule with a learning rate that decreases following the values of the cosine function."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        wandb_logger: WandbLogger,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wandb_logger = wandb_logger
        self.device = device
        
        self.config = TRAIN_CONFIG
        self.global_step = 0
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            betas=self.config["betas"],
            weight_decay=self.config["weight_decay"],
        )
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=self.config["total_steps"],
        )
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler(device.type, enabled=(self.config["precision"] == "bf16" and device.type == "cuda"))
        self.dtype = torch.bfloat16 if self.config["precision"] == "bf16" else torch.float32

    def get_phase_weights(self) -> list[float]:
        """Return the loss weights for the current training phase."""
        if self.global_step < self.config["phase1_steps"]:
            return PHASE1_EXIT_WEIGHTS
        return PHASE2_EXIT_WEIGHTS

    def save_checkpoint(self, path: str):
        """Save model and optimizer state."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model and optimizer state."""
        if not os.path.exists(path):
            logger.warning(f"Checkpoint {path} does not exist.")
            return
            
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Perform one training step."""
        self.model.train()
        batch = batch.to(self.device)
        
        weights = self.get_phase_weights()
        
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.device.type == "cuda")):
            logits_dict = self.model(batch)
            loss, exit_losses = multi_exit_loss(logits_dict, batch, weights, EXIT_CHECKPOINTS)
            
            # Normalize loss for gradient accumulation
            loss = loss / self.config["grad_accumulation_steps"]
            
        self.scaler.scale(loss).backward()
        
        # We step the optimizer only after grad_accumulation_steps
        # This logic is handled in the main training loop to keep this method simple
        
        metrics = {"train_loss_total": loss.item() * self.config["grad_accumulation_steps"]}
        for k, v in exit_losses.items():
            metrics[f"train_loss_exit_{k}"] = v
            
        return metrics

    def val_step(self) -> Dict[str, float]:
        """Evaluate the model on the validation set."""
        self.model.eval()
        
        total_val_loss = 0.0
        exit_val_losses = {k: 0.0 for k in EXIT_CHECKPOINTS}
        num_batches = 0
        
        weights = self.get_phase_weights()
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.device.type == "cuda")):
                    logits_dict = self.model(batch)
                    loss, exit_losses = multi_exit_loss(logits_dict, batch, weights, EXIT_CHECKPOINTS)
                    
                total_val_loss += loss.item()
                for k, v in exit_losses.items():
                    exit_val_losses[k] += v
                num_batches += 1
                
                # Limit validation batches for speed during PoC
                if num_batches >= 20:
                    break
                    
        metrics = {"val_loss_total": total_val_loss / num_batches}
        for k, v in exit_val_losses.items():
            metrics[f"val_loss_exit_{k}"] = v / num_batches
            
        return metrics

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training from step {self.global_step}")
        
        train_iter = iter(self.train_loader)
        
        while self.global_step < self.config["total_steps"]:
            self.optimizer.zero_grad()
            
            step_metrics = {}
            for _ in range(self.config["grad_accumulation_steps"]):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)
                    
                batch_metrics = self.train_step(batch)
                
                # Accumulate metrics for logging
                if not step_metrics:
                    step_metrics = batch_metrics
                else:
                    for k, v in batch_metrics.items():
                        step_metrics[k] += v
                        
            # Average metrics over accumulation steps
            for k in step_metrics:
                step_metrics[k] /= self.config["grad_accumulation_steps"]
                
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config["log_interval"] == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.wandb_logger.log_train_step(self.global_step, step_metrics, lr)
                logger.info(f"Step {self.global_step} | LR: {lr:.2e} | Loss: {step_metrics['train_loss_total']:.4f}")
                
            # Validation
            if self.global_step % self.config["val_interval"] == 0:
                val_metrics = self.val_step()
                self.wandb_logger.log_val_step(self.global_step, val_metrics)
                logger.info(f"Validation Step {self.global_step} | Val Loss: {val_metrics['val_loss_total']:.4f}")
                
            # Checkpointing
            if self.global_step % self.config["save_interval"] == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{self.global_step}.pt")
                self.save_checkpoint(ckpt_path)
                
        # Save final checkpoint
        final_path = os.path.join(CHECKPOINT_DIR, "checkpoint_final.pt")
        self.save_checkpoint(final_path)
        logger.info("Training complete.")

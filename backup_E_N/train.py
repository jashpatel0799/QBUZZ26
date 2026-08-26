%%writefile train.py
import argparse
from config import Config
from dataset import get_dataloaders
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train Elastic-Depth Decoder")
    parser.add_argument("--use_kl_distillation", action="store_true", help="Enable KL Self-Distillation")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    args = parser.parse_args()

    cfg = Config()
    cfg.use_kl_distillation = args.use_kl_distillation
    cfg.max_steps = args.max_steps
    cfg.wandb_run_name += "-kl" if cfg.use_kl_distillation else "-ce-only"

    train_loader, val_loader, tokenizer = get_dataloaders(cfg)
    
    trainer = Trainer(cfg, train_loader, val_loader)
    trainer.train()

if __name__ == "__main__":
    main()
%%writefile trainer.py
import torch
import wandb
from config import Config
from model import DenseHighwayElasticDecoder
from loss import MultiExitLoss
from evaluate import evaluate_metrics

class Trainer:
    def __init__(self, cfg: Config, train_loader, val_loader):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.model = DenseHighwayElasticDecoder(cfg).to(device=cfg.device, dtype=cfg.dtype)
        self.loss_fn = MultiExitLoss(cfg)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)

    def train(self):
        wandb.init(project=self.cfg.wandb_project, name=self.cfg.wandb_run_name, config=self.cfg.__dict__)
        data_iter = iter(self.train_loader)
        
        self.model.train()
        device_type = "cuda" if "cuda" in self.cfg.device else "cpu"
        print(f"Starting bfloat16 Training on {self.cfg.device}...")

        for step in range(1, self.cfg.max_steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(self.cfg.device)
            targets = input_ids.clone()

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=device_type, dtype=self.cfg.dtype):
                exit_logits = self.model(input_ids)
                loss, loss_logs = self.loss_fn(exit_logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if step % 10 == 0:
                wandb.log({"step": step, **loss_logs})
                print(f"Step {step}/{self.cfg.max_steps} | Total Loss: {loss.item():.4f}")

            if step % self.cfg.eval_interval == 0:
                print(f"\n--- Running Evaluation at Step {step} ---")
                eval_logs = evaluate_metrics(self.model, self.val_loader, self.cfg)
                
                # Explicitly recover training mode after evaluation loop
                self.model.train()
                
                wandb.log({"step": step, **eval_logs})
                for k, v in eval_logs.items():
                    print(f"  {k}: {v:.4f}")
                print("-------------------------------------------\n")

        wandb.finish()
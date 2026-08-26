%%writefile loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class MultiExitLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, exit_logits, targets):
        total_loss = 0.0
        logs = {}
        
        shift_targets = targets[:, 1:].contiguous()
        
        # Teacher logits from Exit 18
        teacher_logits = exit_logits[self.cfg.exits[-1]][:, :-1, :].detach().float()
        teacher_p = F.softmax(teacher_logits / self.cfg.distill_temp, dim=-1)

        for exit_k in self.cfg.exits:
            shift_logits = exit_logits[exit_k][:, :-1, :].contiguous().float()
            
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size), 
                shift_targets.view(-1),
                ignore_index=self.cfg.pad_token_id
            )
            
            kl_loss = torch.tensor(0.0, device=targets.device)
            if self.cfg.use_kl_distillation and exit_k != self.cfg.exits[-1]:
                student_log_p = F.log_softmax(shift_logits / self.cfg.distill_temp, dim=-1)
                kl_loss = F.kl_div(student_log_p, teacher_p, reduction="batchmean") * (self.cfg.distill_temp ** 2)

            exit_loss = (self.cfg.ce_weight * ce_loss) + (self.cfg.kl_weight * kl_loss)
            total_loss += exit_loss

            logs[f"loss/ce_exit_{exit_k}"] = ce_loss.item()
            logs[f"loss/kl_exit_{exit_k}"] = kl_loss.item()

        logs["loss/total_combined"] = total_loss.item()
        return total_loss, logs
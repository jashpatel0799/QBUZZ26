%%writefile model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        return (x_fp32 * torch.rsqrt(variance + self.eps)).to(input_dtype) * self.weight

def precompute_rope_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq.shape[1]].unsqueeze(0).unsqueeze(2).to(xq_.device)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.num_key_value_groups = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.head_dim

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xk = xk.repeat_interleave(self.num_key_value_groups, dim=2)
        xv = xv.repeat_interleave(self.num_key_value_groups, dim=2)

        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask

        output = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(output, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attention = Attention(cfg)
        self.feed_forward = FeedForward(cfg)
        self.attention_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)

    def forward(self, x, freqs_cis):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h))

class DenseHighwayElasticDecoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        
        self.exit_norms = nn.ModuleDict({
            str(k): RMSNorm(cfg.dim, eps=cfg.norm_eps) for k in cfg.exits
        })
        
        self.output_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output_head.weight  # Weight Tying
        
        self.highway_adapters = nn.ModuleDict()
        for i, src in enumerate(cfg.exits):
            for dst in cfg.exits[i+1:]:
                self.highway_adapters[f"{src}_to_{dst}"] = nn.Linear(cfg.dim, cfg.dim, bias=False)

        freqs_cis = precompute_rope_freqs_cis(cfg.head_dim, cfg.max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, input_ids, target_exit=None):
        bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:seqlen]

        checkpoint_states = {}
        exit_logits = {}

        for layer_idx, layer in enumerate(self.layers, start=1):
            h = layer(h, freqs_cis)

            if layer_idx in self.cfg.exits:
                highway_signal = torch.zeros_like(h)
                for prev_exit in self.cfg.exits:
                    if prev_exit < layer_idx:
                        adapter_key = f"{prev_exit}_to_{layer_idx}"
                        highway_signal = highway_signal + self.highway_adapters[adapter_key](checkpoint_states[prev_exit])
                
                h = h + highway_signal
                checkpoint_states[layer_idx] = h

                normed_h = self.exit_norms[str(layer_idx)](h)
                exit_logits[layer_idx] = self.output_head(normed_h)

                if target_exit == layer_idx:
                    return {layer_idx: exit_logits[layer_idx]}

        return exit_logits
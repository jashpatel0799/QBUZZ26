"""
Elastic-Depth Decoder LM — Full Architecture.

Combines LLaMA 3.2-style transformer blocks, dense checkpoint skip connections,
and a single shared LM head across all exit depths.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from config import MODEL_CONFIG, EXIT_CHECKPOINTS, SKIP_PAIRS, SKIP_INIT_SCALE, USE_STOCHASTIC_DEPTH, STOCHASTIC_DEPTH_RATE
from model.block import TransformerBlock
from model.skip_connections import DenseSkipConnections
from model.rmsnorm import RMSNorm
from training.stochastic_depth import get_survival_probs


class ElasticDepthLM(nn.Module):
    """Elastic-Depth Language Model.

    A single decoder-only transformer that can exit at multiple depths.
    Features dense skip connections between checkpoints and a shared LM head.
    """

    def __init__(self):
        super().__init__()
        self.config = MODEL_CONFIG
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]
        self.vocab_size = self.config["vocab_size"]
        self.exit_checkpoints = sorted(EXIT_CHECKPOINTS)

        # 1. Token Embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # 2. Transformer Blocks
        survival_probs = get_survival_probs(
            n_layers=self.num_layers,
            drop_rate=STOCHASTIC_DEPTH_RATE,
            use_stochastic_depth=USE_STOCHASTIC_DEPTH
        )
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                intermediate_size=self.config["intermediate_size"],
                num_heads=self.config["num_heads"],
                num_kv_heads=self.config["num_kv_heads"],
                head_dim=self.config["head_dim"],
                max_seq_len=self.config["max_seq_len"],
                rope_base=self.config["rope_base"],
                rms_norm_eps=self.config["rms_norm_eps"],
                stochastic_depth_prob=1.0 - survival_probs[i],
            )
            for i in range(self.num_layers)
        ])

        # 3. Dense Skip Connections
        self.skip_connections = DenseSkipConnections(
            hidden_size=self.hidden_size,
            exit_checkpoints=self.exit_checkpoints,
            skip_pairs=SKIP_PAIRS,
            init_scale=SKIP_INIT_SCALE,
        )

        # 4. Shared LM Head (applied to all exits)
        self.shared_lm_head_norm = RMSNorm(self.hidden_size, eps=self.config["rms_norm_eps"])
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Tie embeddings and LM head weights (standard practice)
        self.lm_head.weight = self.embed_tokens.weight

    def forward_blocks(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Forward pass through all blocks, saving states at checkpoints."""
        hidden_states = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 1-indexed for block numbers (layer 0 is block 1)
            block_num = i + 1
            if block_num in self.exit_checkpoints:
                hidden_states[block_num] = x
        return hidden_states

    def forward(self, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Forward pass during training (returns logits for all exits).
        
        Args:
            input_ids: Shape (batch, seq_len)
            
        Returns:
            Dict mapping exit_checkpoint -> logits tensor of shape (batch, seq_len, vocab_size)
        """
        # Embeddings
        x = self.embed_tokens(input_ids)
        
        # Get raw checkpoint hidden states
        raw_hidden_states = self.forward_blocks(x)
        
        # Apply dense skip connections
        enhanced_hidden_states = self.skip_connections(raw_hidden_states)
        
        # Compute logits for each exit using the shared head
        logits_dict = {}
        for k in self.exit_checkpoints:
            # We must normalize before the final linear head
            h_norm = self.shared_lm_head_norm(enhanced_hidden_states[k])
            logits_dict[k] = self.lm_head(h_norm)
            
        return logits_dict

    def forward_to_exit(self, input_ids: torch.Tensor, target_exit: int) -> torch.Tensor:
        """Forward pass stopping at a specific exit (for inference).
        
        Args:
            input_ids: Shape (batch, seq_len)
            target_exit: Block index to stop at (must be in EXIT_CHECKPOINTS)
            
        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
        """
        if target_exit not in self.exit_checkpoints:
            raise ValueError(f"target_exit {target_exit} not in {self.exit_checkpoints}")
            
        x = self.embed_tokens(input_ids)
        
        raw_hidden_states = {}
        for i in range(target_exit):
            x = self.layers[i](x)
            block_num = i + 1
            if block_num in self.exit_checkpoints:
                raw_hidden_states[block_num] = x
                
        # Apply skip connections (will only compute up to target_exit)
        enhanced_hidden_states = self.skip_connections(raw_hidden_states)
        
        # Compute logits for the target exit
        h_norm = self.shared_lm_head_norm(enhanced_hidden_states[target_exit])
        logits = self.lm_head(h_norm)
        
        return logits

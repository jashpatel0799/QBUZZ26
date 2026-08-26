#!/usr/bin/env python3
"""
Dry run script.
Checks shapes, parameter counts, and VRAM estimation without loading datasets.
"""

import torch
import logging
from config import MODEL_CONFIG, USE_STOCHASTIC_DEPTH
from model.elastic_model import ElasticDepthLM

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=== DRY RUN: Elastic-Depth Decoder LM ===\n")
    
    logger.info("1. Model Configuration:")
    for k, v in MODEL_CONFIG.items():
        logger.info(f"   {k}: {v}")
    logger.info(f"   USE_STOCHASTIC_DEPTH: {USE_STOCHASTIC_DEPTH}")
        
    logger.info("\n2. Initializing Model...")
    model = ElasticDepthLM()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable parameters: {trainable_params:,}")
    
    logger.info("\n3. Testing Forward Pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"   Using device: {device}")
    
    model = model.to(device)
    
    # Dummy input (batch=2, seq_len=16)
    batch_size = 2
    seq_len = 16
    dummy_input = torch.randint(0, MODEL_CONFIG["vocab_size"], (batch_size, seq_len)).to(device)
    
    try:
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, enabled=(device.type == "cuda")):
                logits_dict = model(dummy_input)
                
        logger.info("   Forward pass SUCCESS!")
        logger.info("   Output shapes:")
        for k, logits in logits_dict.items():
            logger.info(f"     Exit {k}: {logits.shape}")
            
    except Exception as e:
        logger.error(f"   Forward pass FAILED: {e}")
        return
        
    logger.info("\n4. Memory Estimation (bf16 training, batch=8, seq=512):")
    # Rule of thumb for bf16 AdamW: weights (2 bytes) + grads (2 bytes) + AdamW (8 bytes) = 12 bytes per param
    # Plus activations
    bytes_per_param = 12
    model_mem = total_params * bytes_per_param / (1024**3)
    
    logger.info(f"   Model + Optimizer States: ~{model_mem:.2f} GB")
    logger.info("   Note: Activations will add ~2-4 GB depending on batch size and gradient checkpointing.")
    
    if model_mem > 14.0:
        logger.warning("   WARNING: Estimated memory exceeds 14 GB (Kaggle T4 limit)!")
    else:
        logger.info("   Estimated memory fits comfortably within 14 GB T4 limit.")
        
    logger.info("\nDry run completed successfully.")

if __name__ == "__main__":
    main()

"""
Dataloader and Curriculum Sampler.
Packs tokenized text into fixed-length sequences (no padding).
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class PackedDataset(IterableDataset):
    """Packs tokenized sequences into fixed-length chunks.
    
    Reads from a streaming HF dataset, concatenates `input_ids`, 
    and yields chunks of exactly `max_seq_len` + 1 tokens.
    (+1 is for the language modeling target shifted by 1).
    """
    def __init__(self, hf_dataset, max_seq_len: int):
        self.dataset = hf_dataset
        # We need max_seq_len + 1 tokens to have targets for all positions
        self.chunk_size = max_seq_len + 1

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            buffer.extend(sample["input_ids"])
            
            while len(buffer) >= self.chunk_size:
                chunk = buffer[:self.chunk_size]
                buffer = buffer[self.chunk_size:]
                
                # Convert to tensor and yield
                # Shape: (max_seq_len + 1,)
                yield torch.tensor(chunk, dtype=torch.long)


def get_dataloaders(train_dataset, val_dataset, config: dict):
    """Create PyTorch DataLoaders from HF datasets."""
    seq_len = config["max_seq_len"]
    batch_size = config["batch_size"]
    
    logger.info(f"Creating dataloaders with seq_len={seq_len}, batch_size={batch_size}")
    
    packed_train = PackedDataset(train_dataset, max_seq_len=seq_len)
    
    # We use a standard DataLoader for the iterable dataset
    train_loader = DataLoader(
        packed_train,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )
    
    # For validation (not streaming), we can also use PackedDataset
    # but we need to convert it to an iterable or list
    packed_val = list(PackedDataset(val_dataset, max_seq_len=seq_len))
    
    val_loader = DataLoader(
        packed_val,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )
    
    return train_loader, val_loader

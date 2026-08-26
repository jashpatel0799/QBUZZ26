"""
Dataset loading and processing.
Streams all 5 dataset sources and tokenizes them.
"""

from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
import logging

from config import DATASET_CONFIG

logger = logging.getLogger(__name__)


def get_streaming_dataset(name: str, config: dict, tokenizer: AutoTokenizer):
    """Load and tokenize a single streaming dataset."""
    logger.info(f"Loading dataset: {name} ({config['path']})")
    
    # Load dataset
    ds = load_dataset(
        config["path"],
        name=config.get("name"),
        split=config["split"],
        streaming=config["streaming"],
        trust_remote_code=True
    )
    
    # Take only the required number of samples if specified
    if config.get("num_samples"):
        ds = ds.take(config["num_samples"])
        
    text_field = config["text_field"]
    
    # Tokenize function
    def tokenize_fn(examples):
        return tokenizer(
            examples[text_field],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        
    # Apply tokenization
    ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=list(ds.features.keys()) if hasattr(ds, 'features') else [text_field],
    )
    return ds


def load_training_data(tokenizer: AutoTokenizer):
    """Load and interleave all training datasets."""
    datasets = []
    
    for name, config in DATASET_CONFIG["datasets"].items():
        ds = get_streaming_dataset(name, config, tokenizer)
        datasets.append(ds)
        
    # Interleave datasets (equal probability for PoC, can be adjusted)
    logger.info("Interleaving training datasets...")
    interleaved = interleave_datasets(datasets)
    return interleaved


def load_eval_data(tokenizer: AutoTokenizer, split: str = "val"):
    """Load validation or test dataset."""
    config_key = "val_dataset" if split == "val" else "test_dataset"
    config = DATASET_CONFIG[config_key]
    
    logger.info(f"Loading {split} dataset: {config['path']}")
    
    ds = load_dataset(
        config["path"],
        name=config.get("name"),
        split=config["split"],
        streaming=False,  # Load eval sets into memory
        trust_remote_code=True
    )
    
    text_field = config["text_field"]
    
    def tokenize_fn(examples):
        return tokenizer(
            examples[text_field],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        
    ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
    )
    return ds

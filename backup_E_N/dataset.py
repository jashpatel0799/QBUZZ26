%%writefile dataset.py
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from config import Config

def get_dataloaders(cfg: Config):
    # Load Llama 3.2 1B Tokenizer (Ungated mirror)
    # tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dynamically bind vocabulary size and pad token to config
    cfg.vocab_size = len(tokenizer)
    cfg.pad_token_id = tokenizer.pad_token_id

    raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cfg.max_seq_len, padding="max_length")

    tokenized = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids"])

    train_loader = DataLoader(tokenized["train"], batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(tokenized["validation"], batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, tokenizer
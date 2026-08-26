"""
Tokenizer loader.
Supports HuggingFace login and fallback tokenizer logic.
"""

from transformers import AutoTokenizer
from huggingface_hub import login
import logging

from config import HF_TOKEN, TOKENIZER_CONFIG

logger = logging.getLogger(__name__)


def load_tokenizer() -> AutoTokenizer:
    """Load the tokenizer, handling authentication and fallback."""
    if HF_TOKEN:
        logger.info("Logging into HuggingFace Hub...")
        login(token=HF_TOKEN)
    else:
        logger.warning("No HF_TOKEN provided. LLaMA 3.2 access may fail.")

    primary_name = TOKENIZER_CONFIG["primary"]
    fallback_name = TOKENIZER_CONFIG["fallback"]

    try:
        logger.info(f"Attempting to load primary tokenizer: {primary_name}")
        tokenizer = AutoTokenizer.from_pretrained(primary_name)
    except Exception as e:
        logger.warning(f"Failed to load primary tokenizer: {e}")
        logger.info(f"Loading fallback tokenizer: {fallback_name}")
        tokenizer = AutoTokenizer.from_pretrained(fallback_name)

    # Set pad_token to eos_token since LLaMA 3 doesn't have a default pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer

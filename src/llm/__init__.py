"""Core LLM implementation modules."""

from .attention import (
    MultiHeadAttention,
    combine_masks,
    create_causal_mask,
    create_padding_mask,
)
from .config import ModelConfig, TrainingConfig
from .tokenizer import Tokenizer

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "Tokenizer",
    "MultiHeadAttention",
    "create_causal_mask",
    "create_padding_mask",
    "combine_masks",
]

# For type checking compatibility
__version__ = "0.1.0"

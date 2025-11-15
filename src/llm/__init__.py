"""Core LLM implementation modules."""

from .attention import (
    MultiHeadAttention,
    combine_masks,
    create_causal_mask,
    create_padding_mask,
)
from .config import ModelConfig, TrainingConfig
from .model import GPTModel
from .tokenizer import Tokenizer
from .trainer import Trainer
from .transformer import FeedForward, TransformerBlock

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "Tokenizer",
    "MultiHeadAttention",
    "create_causal_mask",
    "create_padding_mask",
    "combine_masks",
    "TransformerBlock",
    "FeedForward",
    "GPTModel",
    "Trainer",
]

# For type checking compatibility
__version__ = "0.1.0"

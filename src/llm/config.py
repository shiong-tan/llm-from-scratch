"""Configuration classes for the LLM model and training."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the GPT-style transformer model.

    Attributes:
        vocab_size: Size of the vocabulary (number of unique tokens)
        max_seq_len: Maximum sequence length the model can process
        d_model: Dimension of the model embeddings and hidden states
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads in multi-head attention
        d_ff: Dimension of the feedforward network (typically 4 * d_model)
        dropout: Dropout probability for regularization
        use_bias: Whether to use bias terms in linear layers
    """

    vocab_size: int = 50257  # GPT-2 tokenizer vocabulary size
    max_seq_len: int = 256  # Maximum context length
    d_model: int = 384  # Embedding dimension
    n_layers: int = 6  # Number of transformer blocks
    n_heads: int = 6  # Number of attention heads
    d_ff: int = 1536  # Feedforward dimension (4 * d_model)
    dropout: float = 0.1  # Dropout rate
    use_bias: bool = True  # Use bias in linear layers

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_ff < self.d_model:
            raise ValueError(f"d_ff ({self.d_ff}) should be >= d_model ({self.d_model})")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout ({self.dropout}) must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads

    def num_parameters(self) -> int:
        """Estimate the total number of parameters in the model.

        Returns:
            Approximate number of trainable parameters
        """
        # Token embeddings + position embeddings
        embeddings = self.vocab_size * self.d_model + self.max_seq_len * self.d_model

        # Per transformer block:
        # - Multi-head attention: 4 * d_model * d_model (Q, K, V, output projection)
        # - Layer norm 1: 2 * d_model
        # - Feedforward: d_model * d_ff + d_ff * d_model
        # - Layer norm 2: 2 * d_model
        per_block = (
            4 * self.d_model * self.d_model  # Attention
            + 2 * self.d_model  # LN1
            + 2 * self.d_model * self.d_ff  # FFN
            + 2 * self.d_model  # LN2
        )

        total_blocks = self.n_layers * per_block

        # Final layer norm + output projection
        final = 2 * self.d_model + self.vocab_size * self.d_model

        return embeddings + total_blocks + final


@dataclass
class TrainingConfig:
    """Configuration for training the model.

    Attributes:
        batch_size: Number of sequences per batch
        learning_rate: Initial learning rate for optimizer
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        max_steps: Maximum number of training steps (None = train for num_epochs)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping (None = no clipping)
        eval_interval: Evaluate every N steps
        save_interval: Save checkpoint every N steps
        log_interval: Log metrics every N steps
        device: Device to train on ('cpu', 'cuda', 'mps')
    """

    batch_size: int = 16
    learning_rate: float = 3e-4
    num_epochs: int = 10
    warmup_steps: int = 100
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 4
    max_grad_norm: Optional[float] = 1.0
    eval_interval: int = 100
    save_interval: int = 500
    log_interval: int = 10
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate training configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, "
                f"got {self.gradient_accumulation_steps}"
            )
        if self.device not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"device must be 'cpu', 'cuda', or 'mps', got {self.device}")

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size accounting for gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

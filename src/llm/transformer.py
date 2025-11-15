"""Transformer block implementation for GPT-style models."""

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feedforward network.

    This is a simple 2-layer MLP that processes each position independently.
    The standard transformer uses GELU activation and projects from d_model
    to d_ff (typically 4 * d_model) and back to d_model.

    Args:
        d_model: Model embedding dimension
        d_ff: Feedforward hidden dimension (typically 4 * d_model)
        dropout: Dropout probability
        use_bias: Whether to use bias in linear layers

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        use_bias: bool = True,
    ) -> None:
        """Initialize feedforward network.

        Args:
            d_model: Model embedding dimension
            d_ff: Feedforward hidden dimension
            dropout: Dropout probability
            use_bias: Whether to use bias in linear layers

        Warns:
            UserWarning: If d_ff < d_model (unusual configuration that may limit capacity)
        """
        super().__init__()

        if d_ff < d_model:
            warnings.warn(
                f"d_ff ({d_ff}) is less than d_model ({d_model}). "
                f"Standard is 4 * d_model = {4 * d_model}. "
                f"This may limit model capacity.",
                UserWarning,
                stacklevel=2,
            )

        self.fc1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of feedforward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)

        Raises:
            ValueError: If input dimensions are invalid
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, d_model), got shape {x.shape}")

        # First layer with GELU activation
        # GELU (Gaussian Error Linear Unit) is smoother than ReLU and works
        # better for language models. It's defined as: x * Φ(x) where Φ is
        # the cumulative distribution function of the standard normal distribution.
        # See: Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)", 2016
        x = self.fc1(x)
        x = F.gelu(x)

        # Second layer back to d_model
        # Dropout is applied only once after the second projection, following
        # the standard transformer architecture (Vaswani et al., 2017)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    """Single transformer decoder block.

    This implements a standard transformer decoder block with:
    1. Pre-LayerNorm multi-head self-attention
    2. Residual connection
    3. Pre-LayerNorm feedforward network
    4. Residual connection

    The pre-norm architecture (LayerNorm before attention/FFN) is used instead
    of post-norm as it provides better gradient flow and training stability.
    This is the architecture used in GPT-2, GPT-3, and modern LLMs.

    Args:
        d_model: Model embedding dimension
        n_heads: Number of attention heads
        d_ff: Feedforward hidden dimension
        dropout: Dropout probability
        use_bias: Whether to use bias in linear layers

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)

    Examples:
        Basic usage with causal masking:
        >>> block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048)
        >>> x = torch.randn(2, 10, 512)
        >>> mask = create_causal_mask(10)
        >>> output = block(x, mask=mask)

        Stack multiple blocks for full transformer:
        >>> blocks = nn.ModuleList([
        ...     TransformerBlock(d_model=512, n_heads=8, d_ff=2048)
        ...     for _ in range(6)
        ... ])
        >>> x = embeddings
        >>> for block in blocks:
        ...     x = block(x, mask=mask)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_bias: bool = True,
    ) -> None:
        """Initialize transformer block.

        Args:
            d_model: Model embedding dimension
            n_heads: Number of attention heads
            d_ff: Feedforward hidden dimension
            dropout: Dropout probability
            use_bias: Whether to use bias in linear layers
        """
        super().__init__()

        # Pre-norm layer normalization
        # Layer normalization normalizes across the feature dimension for each
        # example independently, unlike batch norm which normalizes across the batch.
        # This makes it suitable for sequence models where batch sizes and sequence
        # lengths can vary.
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Multi-head self-attention
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_bias=use_bias,
        )

        # Feedforward network
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            use_bias=use_bias,
        )

        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of transformer block.

        Implements the following computation:
        1. x = x + attention(layer_norm(x))
        2. x = x + ffn(layer_norm(x))

        This is the pre-norm architecture which provides better gradient flow
        compared to post-norm (x = layer_norm(x + sublayer(x))).

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
                  or (seq_len, seq_len). Boolean mask where True indicates
                  positions to mask.
            return_attention: Whether to return attention weights

        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
                  if return_attention=True, else None

        Raises:
            ValueError: If input dimensions are invalid

        Shape:
            - Input: (batch_size, seq_len, d_model)
            - Output: (batch_size, seq_len, d_model)
            - Attention: (batch_size, n_heads, seq_len, seq_len)
        """
        # Validate input
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, d_model), got shape {x.shape}")

        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {x.size(-1)}")

        # Self-attention with residual connection (pre-norm)
        # Apply layer norm before attention (this is the "pre" in pre-norm)
        normed = self.ln1(x)
        attn_output, attn_weights = self.attn(normed, mask=mask, return_attention=return_attention)
        # Add residual connection
        x = x + attn_output

        # Feedforward with residual connection (pre-norm)
        # Apply layer norm before feedforward
        normed = self.ln2(x)
        ffn_output = self.ffn(normed)
        # Add residual connection
        x = x + ffn_output

        if return_attention:
            return x, attn_weights
        return x, None

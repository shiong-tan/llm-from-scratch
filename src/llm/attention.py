"""Multi-head self-attention mechanism for transformer models."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    This implements the scaled dot-product attention with multiple heads as described
    in "Attention Is All You Need" (Vaswani et al., 2017). The mechanism allows the
    model to jointly attend to information from different representation subspaces.

    The attention mechanism computes:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    With multiple heads, we:
    1. Project input to Q, K, V for each head
    2. Compute attention for each head in parallel
    3. Concatenate head outputs and project back

    Note: This implementation performs self-attention only (Q, K, V all come from
    the same input). For cross-attention in encoder-decoder models, a separate
    implementation is needed where keys and values come from a different source.

    Args:
        d_model: Model embedding dimension
        n_heads: Number of attention heads
        dropout: Dropout probability for attention weights
        use_bias: Whether to use bias in linear projections

    Attributes:
        head_dim: Dimension of each attention head (d_model // n_heads)
        scale: Scaling factor for attention scores (1 / sqrt(head_dim))

    Examples:
        Basic usage with causal masking:
        >>> attn = MultiHeadAttention(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 10, 512)
        >>> mask = create_causal_mask(10)
        >>> output, _ = attn(x, mask=mask)

        Combining causal and padding masks:
        >>> causal = create_causal_mask(seq_len)
        >>> padding = create_padding_mask(input_ids, pad_token_id=0)
        >>> combined = combine_masks(causal, padding, batch_size=2)
        >>> output, _ = attn(x, mask=combined)
    """

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.1, use_bias: bool = True
    ) -> None:
        """Initialize multi-head attention.

        Args:
            d_model: Model embedding dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            use_bias: Whether to use bias in linear layers

        Raises:
            ValueError: If d_model is not divisible by n_heads
        """
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Scaling factor for attention scores: 1/sqrt(d_k)
        # This prevents the dot product from growing too large in magnitude,
        # which would push softmax into regions with extremely small gradients.
        # See "Attention Is All You Need" section 3.2.1
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        # Using a single linear layer for efficiency, will split into heads later
        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

        # Dropout for attention weights and output
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
                  or (seq_len, seq_len). Boolean or integer mask where True/1
                  values indicate positions to mask.
            return_attention: Whether to return attention weights

        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
                  if return_attention=True, else None

        Raises:
            ValueError: If input dimensions are invalid or mask shape is incompatible

        Shape:
            - Input: (batch_size, seq_len, d_model)
            - Output: (batch_size, seq_len, d_model)
            - Attention: (batch_size, n_heads, seq_len, seq_len)
        """
        # Validate input
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, d_model), got shape {x.shape}")

        batch_size, seq_len, d_model = x.shape

        if seq_len == 0:
            raise ValueError("Cannot apply attention to empty sequence (seq_len=0)")

        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")

        # Project to Q, K, V and reshape to separate heads
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, head_dim)
        q = self._split_heads(self.q_proj(x), batch_size)
        k = self._split_heads(self.k_proj(x), batch_size)
        v = self._split_heads(self.v_proj(x), batch_size)

        # Compute attention scores
        # Shape: (batch_size, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            # Validate mask dimensions
            if mask.dim() not in [2, 3, 4]:
                raise ValueError(f"Mask must be 2D, 3D, or 4D, got shape {mask.shape}")

            # Validate mask matches sequence length
            if mask.size(-1) != seq_len or mask.size(-2) != seq_len:
                raise ValueError(
                    f"Mask shape {mask.shape} incompatible with sequence length {seq_len}"
                )

            # Ensure mask has correct shape for broadcasting
            if mask.dim() == 2:
                # Broadcast mask to (batch_size, 1, seq_len, seq_len)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # Add head dimension
                mask = mask.unsqueeze(1)

            # Convert to boolean if needed and apply mask
            if mask.dtype != torch.bool:
                mask = mask.bool()

            # Apply mask by setting masked positions to large negative value
            # Using -1e9 instead of -inf to avoid NaN issues
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Handle NaN values that may arise from fully masked rows
        # This can occur when an entire row is masked (e.g., padding tokens)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Apply dropout to attention weights
        # Note: This breaks the sum-to-1 property during training, but empirically
        # works well as it acts as a regularizer on the attention distribution
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # Shape: (batch_size, n_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # Merge heads
        # Shape: (batch_size, seq_len, d_model)
        attn_output = self._merge_heads(attn_output, batch_size)

        # Final output projection
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)

        if return_attention:
            return output, attn_weights
        return output, None

    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Split the last dimension into (n_heads, head_dim).

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            batch_size: Batch size

        Returns:
            Tensor of shape (batch_size, n_heads, seq_len, head_dim)
        """
        # Reshape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, head_dim)
        x = x.view(batch_size, -1, self.n_heads, self.head_dim)
        # Transpose: (batch_size, seq_len, n_heads, head_dim) -> (batch_size, n_heads, seq_len, head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Merge attention heads back to d_model dimension.

        Args:
            x: Input tensor of shape (batch_size, n_heads, seq_len, head_dim)
            batch_size: Batch size

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Transpose: (batch_size, n_heads, seq_len, head_dim) -> (batch_size, seq_len, n_heads, head_dim)
        x = x.transpose(1, 2)
        # Reshape: (batch_size, seq_len, n_heads, head_dim) -> (batch_size, seq_len, d_model)
        return x.contiguous().view(batch_size, -1, self.d_model)


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create a causal (autoregressive) attention mask.

    This mask prevents positions from attending to subsequent positions, which is
    essential for autoregressive models like GPT where we can only attend to previous
    tokens when predicting the next token.

    Args:
        seq_len: Sequence length
        device: Device to create the tensor on

    Returns:
        Boolean mask of shape (seq_len, seq_len) where True indicates positions to mask

    Example:
        For seq_len=4, creates:
        [[0, 1, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 0]]

        Where 1 indicates positions that should be masked (cannot attend).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def create_padding_mask(seq: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Create padding mask for sequences with padding tokens.

    Args:
        seq: Input sequence of shape (batch_size, seq_len)
        pad_token_id: ID of the padding token

    Returns:
        Boolean mask of shape (batch_size, seq_len, seq_len) where True indicates
        positions that should be masked. Each row i has True where key positions
        are padding tokens.

    Example:
        For seq = [[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]] with pad_token_id=0:
        Creates mask indicating which key positions are padding for each query position.
    """
    # Shape: (batch_size, seq_len)
    is_pad = seq == pad_token_id
    # Expand to (batch_size, seq_len, seq_len) - broadcast across query dimension
    # Each query position cannot attend to padding key positions
    mask = is_pad.unsqueeze(1).expand(-1, seq.size(1), -1)
    return mask


def combine_masks(
    causal_mask: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    batch_size: int = 1,
) -> Optional[torch.Tensor]:
    """Combine causal and padding masks into a single attention mask.

    This utility helps properly combine different mask types for use in attention.
    Both causal masking (preventing future token attention) and padding masking
    (preventing attention to padding tokens) are common in sequence models.

    Args:
        causal_mask: Optional causal mask of shape (seq_len, seq_len)
        padding_mask: Optional padding mask of shape (batch_size, seq_len, seq_len)
                     or (batch_size, 1, 1, seq_len) from create_padding_mask
        batch_size: Batch size for expanding causal mask

    Returns:
        Combined mask of shape (batch_size, seq_len, seq_len) where True indicates
        positions to mask, or None if both inputs are None

    Example:
        >>> causal = create_causal_mask(seq_len=5)
        >>> padding = create_padding_mask(input_ids, pad_token_id=0)
        >>> combined = combine_masks(causal, padding, batch_size=2)
        >>> output, _ = attn(x, mask=combined)
    """
    if causal_mask is None and padding_mask is None:
        return None

    mask = None

    if causal_mask is not None:
        # Expand causal to (batch_size, seq_len, seq_len)
        mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

    if padding_mask is not None:
        # Handle old-style padding mask shape (batch, 1, 1, seq_len)
        if padding_mask.dim() == 4 and padding_mask.size(1) == 1 and padding_mask.size(2) == 1:
            # Expand to (batch_size, seq_len, seq_len)
            seq_len = padding_mask.size(-1)
            pad_mask = padding_mask.squeeze(1).squeeze(1).unsqueeze(1).expand(-1, seq_len, -1)
        else:
            pad_mask = padding_mask

        mask = pad_mask if mask is None else (mask | pad_mask)

    return mask

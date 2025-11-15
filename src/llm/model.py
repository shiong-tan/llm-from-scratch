"""GPT-style decoder-only transformer language model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .transformer import TransformerBlock


class GPTModel(nn.Module):
    """GPT-style decoder-only transformer language model.

    This implements a complete autoregressive language model following the GPT
    architecture (Radford et al., 2018, 2019). The model uses:
    - Token embeddings to convert input IDs to vectors
    - Learned positional embeddings to encode sequence position
    - Stack of transformer decoder blocks
    - Final layer normalization
    - Output projection to vocabulary for next-token prediction

    The model is trained to predict the next token given previous tokens, making
    it suitable for text generation tasks.

    Args:
        config: Model configuration specifying architecture hyperparameters

    Attributes:
        config: Stored configuration
        token_embedding: Embedding layer for token IDs
        position_embedding: Learned positional embeddings
        blocks: Stack of transformer decoder blocks
        ln_final: Final layer normalization before output projection
        lm_head: Output projection to vocabulary (shares weights with token_embedding)

    Shape:
        - Input: (batch_size, seq_len) - token IDs
        - Output: (batch_size, seq_len, vocab_size) - next-token logits

    Examples:
        Create and use a GPT model:
        >>> config = ModelConfig(vocab_size=50257, max_seq_len=256, d_model=384, n_layers=6)
        >>> model = GPTModel(config)
        >>> input_ids = torch.randint(0, 50257, (2, 10))  # Batch of 2, length 10
        >>> logits = model(input_ids)
        >>> logits.shape
        torch.Size([2, 10, 50257])

        Generate next token probabilities:
        >>> logits = model(input_ids)
        >>> next_token_logits = logits[:, -1, :]  # Get last position
        >>> probs = torch.softmax(next_token_logits, dim=-1)
        >>> next_token = torch.argmax(probs, dim=-1)
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize GPT model.

        Args:
            config: Model configuration with architecture hyperparameters

        Raises:
            ValueError: If config parameters are invalid
        """
        super().__init__()

        # Validate config
        if not isinstance(config, ModelConfig):
            raise TypeError(f"Expected ModelConfig, got {type(config)}")

        self.config = config

        # Token embeddings: convert token IDs to d_model dimensional vectors
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embeddings: learned position encodings
        # Unlike sinusoidal encodings, learned embeddings can adapt to the data
        # GPT-2 and GPT-3 use learned positional embeddings
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Dropout for embeddings
        self.emb_dropout = nn.Dropout(config.dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    use_bias=config.use_bias,
                )
                for _ in range(config.n_layers)
            ]
        )

        # Final layer normalization
        # Applied before the output projection to stabilize logits
        self.ln_final = nn.LayerNorm(config.d_model)

        # Language modeling head: project to vocabulary
        # Note: This layer's weights are tied with token_embedding (see below)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: Share weights between token embedding and output projection
        # This reduces parameters and often improves performance
        # See "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017)
        self.lm_head.weight = self.token_embedding.weight

        # Mark residual projections BEFORE initialization
        # These need scaled initialization for training stability in deep networks
        for block in self.blocks:
            block.attn.out_proj._is_residual_projection = True  # type: ignore
            block.ffn.fc2._is_residual_projection = True  # type: ignore

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights following GPT-2/GPT-3 initialization scheme.

        Uses:
        - Normal distribution N(0, 0.02) for most weights
        - Scaled initialization for residual projections
        - Zero initialization for biases

        This initialization scheme is critical for stable training of deep transformers.
        """
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Initialize all linear layers and layer norms
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Normal initialization for most linear layers
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

                # Scaled initialization for residual projections
                # In GPT-2, attention output and FFN output projections are scaled by 1/sqrt(2*n_layers)
                # This prevents activation explosion in deep networks
                if hasattr(module, "_is_residual_projection"):
                    nn.init.normal_(
                        module.weight,
                        mean=0.0,
                        std=0.02 / torch.sqrt(torch.tensor(2.0 * self.config.n_layers)),
                    )

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of GPT model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            targets: Target token IDs for loss computation, shape (batch_size, seq_len)
                     If provided with return_loss=True, will compute cross-entropy loss
            return_loss: Whether to compute and return loss (requires targets)
            ignore_index: Token ID to ignore in loss computation (e.g., padding tokens).
                         Default is -100, which is PyTorch's default for cross_entropy.
                         Set to your padding token ID to exclude padding from loss.

        Returns:
            Tuple of:
                - logits: Next-token prediction logits of shape (batch_size, seq_len, vocab_size)
                - loss: Cross-entropy loss if return_loss=True and targets provided, else None

        Raises:
            ValueError: If input dimensions are invalid or exceed max_seq_len
            RuntimeError: If return_loss=True but targets not provided

        Shape:
            - input_ids: (batch_size, seq_len)
            - targets: (batch_size, seq_len)
            - logits: (batch_size, seq_len, vocab_size)
            - loss: scalar tensor or None
        """
        # Validate input
        if input_ids.dim() != 2:
            raise ValueError(
                f"Expected 2D input_ids (batch, seq_len), got shape {input_ids.shape}"
            )

        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds maximum ({self.config.max_seq_len})"
            )

        if seq_len == 0:
            raise ValueError("Cannot process empty sequence (seq_len=0)")

        if return_loss and targets is None:
            raise RuntimeError("targets must be provided when return_loss=True")

        # Create position indices [0, 1, 2, ..., seq_len-1]
        # Shape: (seq_len,)
        positions = torch.arange(seq_len, device=input_ids.device)

        # Get embeddings
        # Token embeddings: (batch_size, seq_len, d_model)
        token_emb = self.token_embedding(input_ids)
        # Position embeddings: (seq_len, d_model) -> broadcast to (batch_size, seq_len, d_model)
        pos_emb = self.position_embedding(positions)

        # Combine embeddings
        # Shape: (batch_size, seq_len, d_model)
        x = token_emb + pos_emb
        x = self.emb_dropout(x)

        # Apply transformer blocks sequentially
        # Each block has shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        # We use causal masking to prevent attending to future tokens
        from .attention import create_causal_mask

        causal_mask = create_causal_mask(seq_len, device=input_ids.device)

        for block in self.blocks:
            x, _ = block(x, mask=causal_mask)

        # Apply final layer normalization
        # Shape: (batch_size, seq_len, d_model)
        x = self.ln_final(x)

        # Project to vocabulary
        # Shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)

        # Compute loss if requested
        loss = None
        if return_loss and targets is not None:
            # Reshape for cross entropy: (batch_size * seq_len, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)

            # Compute cross-entropy loss
            # This automatically handles the softmax and negative log likelihood
            # ignore_index allows excluding padding tokens from loss computation
            loss = nn.functional.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate new tokens autoregressively.

        This is a simple greedy/sampling generation method. More sophisticated
        methods (beam search, nucleus sampling) can be implemented separately.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
                        temperature=1.0 uses raw logits
                        temperature<1.0 makes distribution sharper (more confident)
                        temperature>1.0 makes distribution softer (more diverse)
            top_k: If set, only sample from top-k most likely tokens

        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)

        Raises:
            ValueError: If temperature <= 0 or top_k < 1

        Examples:
            >>> model = GPTModel(config)
            >>> input_ids = torch.tensor([[1, 2, 3]])
            >>> generated = model.generate(input_ids, max_new_tokens=10)
            >>> generated.shape
            torch.Size([1, 13])  # Original 3 + 10 new tokens
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        # Save training mode and set to eval
        training_mode = self.training
        self.eval()

        # Start with input_ids
        # Shape: (batch_size, seq_len)
        generated = input_ids.clone()

        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # If sequence is longer than max_seq_len, use only the last max_seq_len tokens
                    input_context = generated[:, -self.config.max_seq_len :]

                    # Get logits for next token
                    # Shape: (batch_size, context_len, vocab_size)
                    logits, _ = self(input_context)

                    # Get logits for last position (next token prediction)
                    # Shape: (batch_size, vocab_size)
                    next_token_logits = logits[:, -1, :]

                    # Apply temperature
                    next_token_logits = next_token_logits / temperature

                    # Apply top-k filtering if specified
                    if top_k is not None:
                        # Get top-k logits and indices
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)

                        # Set all non-top-k logits to -inf
                        next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                    # Convert logits to probabilities
                    probs = torch.softmax(next_token_logits, dim=-1)

                    # Sample next token
                    # Shape: (batch_size, 1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Append to generated sequence
                    # Shape: (batch_size, seq_len + 1)
                    generated = torch.cat([generated, next_token], dim=1)

            return generated
        finally:
            # Restore original training mode
            self.train(training_mode)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count the number of trainable parameters.

        Args:
            exclude_embeddings: If True, exclude embedding parameters from count.
                              Useful for comparing model sizes, as embeddings scale
                              with vocabulary size rather than model depth.

        Returns:
            Number of trainable parameters

        Examples:
            >>> model = GPTModel(config)
            >>> total_params = model.num_parameters()
            >>> non_emb_params = model.num_parameters(exclude_embeddings=True)
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)

        if exclude_embeddings:
            # Subtract embedding parameters
            # Note: lm_head shares weights with token_embedding, so only count once
            emb_params = sum(
                p.numel()
                for p in [self.token_embedding.weight, self.position_embedding.weight]
            )
            return total - emb_params

        return total

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  max_seq_len={self.config.max_seq_len},\n"
            f"  d_model={self.config.d_model},\n"
            f"  n_layers={self.config.n_layers},\n"
            f"  n_heads={self.config.n_heads},\n"
            f"  parameters={self.num_parameters():,}\n"
            f")"
        )

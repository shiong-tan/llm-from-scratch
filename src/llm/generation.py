"""Advanced text generation utilities for GPT model."""

from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn.functional as F

from .model import GPTModel


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
                    temperature=1.0 uses raw logits
                    temperature<1.0 makes distribution sharper
                    temperature>1.0 makes distribution softer
        top_k: If set, only sample from top-k most likely tokens
        top_p: If set, sample from smallest set of tokens with cumulative
               probability >= top_p (nucleus sampling)
        repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
        no_repeat_ngram_size: If set, prevent repeating n-grams of this size
        min_length: Minimum generation length (prevents early EOS)
        eos_token_id: Token ID that signals end of sequence
        pad_token_id: Token ID for padding (default: same as eos_token_id)
        do_sample: Whether to use sampling (True) or greedy decoding (False)
        num_beams: Number of beams for beam search (1 = no beam search)
        length_penalty: Exponential penalty to length for beam search
                       >1.0 encourages longer sequences
                       <1.0 encourages shorter sequences

    Examples:
        Greedy decoding (deterministic):
        >>> config = GenerationConfig(max_new_tokens=50, do_sample=False)

        Nucleus sampling (high quality, diverse):
        >>> config = GenerationConfig(
        ...     max_new_tokens=100,
        ...     temperature=0.8,
        ...     top_p=0.9,
        ...     repetition_penalty=1.2
        ... )

        Beam search (coherent, focused):
        >>> config = GenerationConfig(
        ...     max_new_tokens=50,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     length_penalty=1.0
        ... )
    """

    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: Optional[int] = None
    min_length: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    do_sample: bool = True
    num_beams: int = 1
    length_penalty: float = 1.0

    def __post_init__(self) -> None:
        """Validate generation configuration."""
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be positive, got {self.repetition_penalty}"
            )
        if self.num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {self.num_beams}")
        if self.num_beams > 1 and self.do_sample:
            raise ValueError("Beam search is not compatible with sampling (do_sample=True)")

        # Set pad_token_id to eos_token_id if not specified
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id


class TextGenerator:
    """Advanced text generation for GPT models.

    Provides various generation strategies:
    - Greedy decoding (deterministic, picks most likely token)
    - Top-k sampling (sample from k most likely tokens)
    - Top-p/nucleus sampling (sample from smallest set with cumulative prob >= p)
    - Beam search (explore multiple hypotheses, return best)
    - Temperature scaling (control randomness)
    - Repetition penalty (discourage repeating tokens/n-grams)

    Args:
        model: GPT model to use for generation
        config: Generation configuration

    Examples:
        Greedy decoding:
        >>> generator = TextGenerator(model, GenerationConfig(do_sample=False))
        >>> output = generator.generate(input_ids)

        Nucleus sampling:
        >>> config = GenerationConfig(temperature=0.8, top_p=0.9)
        >>> generator = TextGenerator(model, config)
        >>> output = generator.generate(input_ids)

        Beam search:
        >>> config = GenerationConfig(do_sample=False, num_beams=5)
        >>> generator = TextGenerator(model, config)
        >>> output = generator.generate(input_ids)
    """

    def __init__(self, model: GPTModel, config: GenerationConfig) -> None:
        """Initialize text generator.

        Args:
            model: GPT model for generation
            config: Generation configuration
        """
        self.model = model
        self.config = config

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stopping_criteria: Optional[Callable[[torch.Tensor, int], bool]] = None,
    ) -> torch.Tensor:
        """Generate text using configured strategy.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask (1 for real tokens, 0 for padding)
            stopping_criteria: Optional function(generated_ids, step) -> bool
                             to stop generation early

        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
            or (batch_size, seq_len + actual_length) if stopped early

        Raises:
            ValueError: If configuration is invalid for the generation strategy

        Examples:
            >>> generator = TextGenerator(model, config)
            >>> input_ids = torch.tensor([[1, 2, 3]])
            >>> output = generator.generate(input_ids)
            >>> output.shape  # (1, 3 + max_new_tokens)
        """
        # Save and set eval mode
        was_training = self.model.training
        self.model.eval()

        try:
            if self.config.num_beams > 1:
                return self._beam_search(input_ids, attention_mask, stopping_criteria)
            else:
                return self._sample(input_ids, attention_mask, stopping_criteria)
        finally:
            # Restore training mode
            if was_training:
                self.model.train()

    def _sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        stopping_criteria: Optional[Callable[[torch.Tensor, int], bool]],
    ) -> torch.Tensor:
        """Generate using sampling or greedy decoding.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            stopping_criteria: Optional early stopping function

        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()

        # Track which sequences have finished (encountered EOS)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        for step in range(self.config.max_new_tokens):
            # Check stopping criteria
            if stopping_criteria is not None and stopping_criteria(generated, step):
                break

            # Check if all sequences finished
            if unfinished_sequences.max() == 0:
                break

            # Get context window (last max_seq_len tokens)
            input_context = generated[:, -self.model.config.max_seq_len :]

            # Forward pass
            logits, _ = self.model(input_context)

            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :]

            # Apply repetition penalty
            if self.config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated
                )

            # Enforce minimum length (prevent EOS before min_length)
            if self.config.min_length is not None and generated.shape[1] < self.config.min_length:
                if self.config.eos_token_id is not None:
                    next_token_logits[:, self.config.eos_token_id] = float("-inf")

            # Prevent n-gram repetition
            if self.config.no_repeat_ngram_size is not None:
                next_token_logits = self._prevent_ngram_repetition(
                    next_token_logits, generated
                )

            # Sample or decode next token
            if self.config.do_sample:
                next_token = self._sample_next_token(next_token_logits)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Mark finished sequences (those that generated EOS)
            if self.config.eos_token_id is not None:
                next_token = next_token * unfinished_sequences.unsqueeze(-1) + (
                    self.config.pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))
                )
                unfinished_sequences = unfinished_sequences * (
                    next_token.squeeze(-1) != self.config.eos_token_id
                ).long()

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def _sample_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next token from logits with temperature, top-k, and top-p.

        Args:
            logits: Logits of shape (batch_size, vocab_size)

        Returns:
            Sampled token IDs of shape (batch_size, 1)
        """
        # Apply temperature
        logits = logits / self.config.temperature

        # Apply top-k filtering
        if self.config.top_k is not None:
            indices_to_remove = logits < torch.topk(logits, self.config.top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus) filtering
        if self.config.top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            # Shift to keep first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Scatter to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Check for all -inf logits (no valid tokens) - numerical stability safeguard
        all_inf = torch.isinf(logits).all(dim=-1)
        if all_inf.any():
            # Fallback: uniform distribution for affected batch items
            logits = torch.where(
                all_inf.unsqueeze(-1),
                torch.zeros_like(logits),
                logits
            )

        # Convert to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, generated: torch.Tensor
    ) -> torch.Tensor:
        """Apply repetition penalty to discourage repeating tokens.

        Args:
            logits: Logits of shape (batch_size, vocab_size)
            generated: Previously generated tokens of shape (batch_size, seq_len)

        Returns:
            Modified logits with repetition penalty applied
        """
        batch_size, vocab_size = logits.shape

        # For each token in vocabulary, check if it appears in generated sequence
        for i in range(batch_size):
            for token_id in set(generated[i].tolist()):
                # If logit is positive, divide by penalty (makes it less likely)
                # If logit is negative, multiply by penalty (makes it even less likely)
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= self.config.repetition_penalty
                else:
                    logits[i, token_id] *= self.config.repetition_penalty

        return logits

    def _prevent_ngram_repetition(
        self, logits: torch.Tensor, generated: torch.Tensor
    ) -> torch.Tensor:
        """Prevent repeating n-grams by setting their logits to -inf.

        Args:
            logits: Logits of shape (batch_size, vocab_size)
            generated: Previously generated tokens of shape (batch_size, seq_len)

        Returns:
            Modified logits with banned n-grams set to -inf
        """
        batch_size = generated.shape[0]
        ngram_size = self.config.no_repeat_ngram_size

        if generated.shape[1] < ngram_size:
            return logits  # Not enough tokens to form n-gram

        for i in range(batch_size):
            # Get last (n-1) tokens
            context_ngram = generated[i, -(ngram_size - 1) :].tolist()

            # Find all n-grams in history
            banned_tokens = set()
            for j in range(len(generated[i]) - ngram_size + 1):
                ngram = generated[i, j : j + ngram_size].tolist()
                # If current context matches first (n-1) tokens, ban the last token
                if ngram[:-1] == context_ngram:
                    banned_tokens.add(ngram[-1])

            # Ban these tokens
            for token_id in banned_tokens:
                logits[i, token_id] = float("-inf")

        return logits

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        stopping_criteria: Optional[Callable[[torch.Tensor, int], bool]],
    ) -> torch.Tensor:
        """Generate using beam search.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            stopping_criteria: Optional early stopping function

        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        batch_size, seq_len = input_ids.shape
        num_beams = self.config.num_beams
        device = input_ids.device

        # Expand input_ids to beam size
        # Shape: (batch_size * num_beams, seq_len)
        input_ids = input_ids.unsqueeze(1).expand(-1, num_beams, -1)
        input_ids = input_ids.reshape(batch_size * num_beams, seq_len)

        # Initialize beam scores (log probabilities)
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = float("-inf")  # Only first beam is active initially
        beam_scores = beam_scores.view(-1)  # Shape: (batch_size * num_beams,)

        # Track finished beams
        done = torch.zeros(batch_size, num_beams, dtype=torch.bool, device=device)

        for step in range(self.config.max_new_tokens):
            # Check stopping criteria
            if stopping_criteria is not None and stopping_criteria(input_ids, step):
                break

            # Check if all beams finished
            if done.all():
                break

            # Get context window
            input_context = input_ids[:, -self.model.config.max_seq_len :]

            # Forward pass
            logits, _ = self.model(input_context)
            next_token_logits = logits[:, -1, :]  # Shape: (batch_size * num_beams, vocab_size)

            # Convert to log probabilities
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

            # Add beam scores
            # Shape: (batch_size * num_beams, vocab_size)
            next_token_scores = next_token_log_probs + beam_scores.unsqueeze(-1)

            vocab_size = next_token_scores.shape[-1]

            # Reshape to (batch_size, num_beams * vocab_size)
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Get top 2 * num_beams scores (we'll filter later)
            # Note: Length penalty is applied later when storing beam scores
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            # Convert flattened indices to beam and token indices
            next_beam_indices = next_tokens // vocab_size  # Which beam
            next_tokens = next_tokens % vocab_size  # Which token

            # Select top num_beams beams per batch
            beam_outputs = []
            for batch_idx in range(batch_size):
                batch_beams = []
                for beam_token_rank, (beam_idx, token_id, score) in enumerate(
                    zip(next_beam_indices[batch_idx], next_tokens[batch_idx], next_token_scores[batch_idx])
                ):
                    # Get original beam index
                    beam_id = batch_idx * num_beams + beam_idx.item()

                    # Skip if this beam is already done
                    if done[batch_idx, beam_idx]:
                        continue

                    # Check for EOS token
                    is_eos = (
                        self.config.eos_token_id is not None
                        and token_id.item() == self.config.eos_token_id
                    )

                    batch_beams.append(
                        {
                            "score": score.item(),
                            "token_id": token_id.item(),
                            "beam_id": beam_id,
                            "beam_idx": len(batch_beams),  # Store index for done tracking
                            "is_eos": is_eos,
                        }
                    )

                    if len(batch_beams) >= num_beams:
                        break

                beam_outputs.append(batch_beams)

            # Collect new beams
            new_beam_scores = []
            new_beam_tokens = []
            new_beam_indices = []

            for batch_idx, batch_beams in enumerate(beam_outputs):
                for beam_data in batch_beams:
                    # Apply length penalty when storing scores
                    current_length = seq_len + step + 1
                    length_penalty_factor = (current_length / (seq_len + 1)) ** self.config.length_penalty
                    penalized_score = beam_data["score"] * length_penalty_factor

                    new_beam_scores.append(penalized_score)
                    new_beam_tokens.append(beam_data["token_id"])
                    new_beam_indices.append(beam_data["beam_id"])

                    # Mark as done if EOS (using correct beam index)
                    if beam_data["is_eos"]:
                        done[batch_idx, beam_data["beam_idx"]] = True

            # Convert to tensors
            beam_scores = torch.tensor(new_beam_scores, device=device)
            beam_tokens = torch.tensor(new_beam_tokens, device=device).unsqueeze(-1)
            beam_indices = torch.tensor(new_beam_indices, device=device)

            # Reorder input_ids based on selected beams
            input_ids = input_ids[beam_indices]

            # For finished beams, replace new token with pad token
            if self.config.pad_token_id is not None:
                for batch_idx in range(batch_size):
                    for beam_idx in range(num_beams):
                        global_beam_idx = batch_idx * num_beams + beam_idx
                        if done[batch_idx, beam_idx]:
                            beam_tokens[global_beam_idx] = self.config.pad_token_id

            # Append new tokens
            input_ids = torch.cat([input_ids, beam_tokens], dim=-1)

        # Return best beam for each batch
        # Reshape input_ids: (batch_size, num_beams, seq_len)
        final_length = input_ids.shape[1]
        input_ids = input_ids.view(batch_size, num_beams, final_length)

        # Select beam with highest score
        best_beams = beam_scores.view(batch_size, num_beams).argmax(dim=1)
        output = input_ids[torch.arange(batch_size), best_beams]

        return output

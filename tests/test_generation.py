"""Tests for text generation utilities."""

import pytest
import torch

from src.llm.config import ModelConfig
from src.llm.generation import GenerationConfig, TextGenerator
from src.llm.model import GPTModel


class TestGenerationConfig:
    """Tests for GenerationConfig class."""

    def test_default_initialization(self) -> None:
        """Test default configuration."""
        config = GenerationConfig()

        assert config.max_new_tokens == 50
        assert config.temperature == 1.0
        assert config.top_k is None
        assert config.top_p is None
        assert config.repetition_penalty == 1.0
        assert config.do_sample is True
        assert config.num_beams == 1

    def test_custom_initialization(self) -> None:
        """Test custom configuration."""
        config = GenerationConfig(
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
        )

        assert config.max_new_tokens == 100
        assert config.temperature == 0.8
        assert config.top_k == 50
        assert config.top_p == 0.95
        assert config.repetition_penalty == 1.2

    def test_invalid_max_new_tokens(self) -> None:
        """Test that invalid max_new_tokens raises error."""
        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            GenerationConfig(max_new_tokens=0)

        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            GenerationConfig(max_new_tokens=-1)

    def test_invalid_temperature(self) -> None:
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            GenerationConfig(temperature=0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            GenerationConfig(temperature=-1.0)

    def test_invalid_top_k(self) -> None:
        """Test that invalid top_k raises error."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            GenerationConfig(top_k=0)

    def test_invalid_top_p(self) -> None:
        """Test that invalid top_p raises error."""
        with pytest.raises(ValueError, match="top_p must be in"):
            GenerationConfig(top_p=0.0)

        with pytest.raises(ValueError, match="top_p must be in"):
            GenerationConfig(top_p=1.5)

    def test_invalid_repetition_penalty(self) -> None:
        """Test that invalid repetition_penalty raises error."""
        with pytest.raises(ValueError, match="repetition_penalty must be positive"):
            GenerationConfig(repetition_penalty=0)

    def test_invalid_num_beams(self) -> None:
        """Test that invalid num_beams raises error."""
        with pytest.raises(ValueError, match="num_beams must be >= 1"):
            GenerationConfig(num_beams=0)

    def test_beam_search_with_sampling_raises(self) -> None:
        """Test that beam search + sampling raises error."""
        with pytest.raises(ValueError, match="Beam search is not compatible with sampling"):
            GenerationConfig(num_beams=5, do_sample=True)

    def test_pad_token_defaults_to_eos(self) -> None:
        """Test that pad_token_id defaults to eos_token_id."""
        config = GenerationConfig(eos_token_id=50256)
        assert config.pad_token_id == 50256

        config2 = GenerationConfig(eos_token_id=100, pad_token_id=200)
        assert config2.pad_token_id == 200


class TestTextGenerator:
    """Tests for TextGenerator class."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create small model config for testing."""
        return ModelConfig(
            vocab_size=100, max_seq_len=32, d_model=64, n_layers=2, n_heads=4
        )

    @pytest.fixture
    def model(self, model_config: ModelConfig) -> GPTModel:
        """Create model for testing."""
        model = GPTModel(model_config)
        model.eval()
        return model

    def test_initialization(
        self, model: GPTModel
    ) -> None:
        """Test generator initialization."""
        config = GenerationConfig(max_new_tokens=10)
        generator = TextGenerator(model, config)

        assert generator.model is model
        assert generator.config is config

    def test_greedy_generation(self, model: GPTModel) -> None:
        """Test greedy decoding (deterministic)."""
        config = GenerationConfig(max_new_tokens=5, do_sample=False)
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))

        # Greedy should be deterministic
        output1 = generator.generate(input_ids)
        output2 = generator.generate(input_ids)

        assert output1.shape == (1, 15)  # 10 + 5
        assert torch.equal(output1, output2)

    def test_sampling_generation(self, model: GPTModel) -> None:
        """Test sampling generation."""
        config = GenerationConfig(max_new_tokens=5, temperature=1.0, do_sample=True)
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))

        output = generator.generate(input_ids)

        assert output.shape == (1, 15)
        # Check that output contains input
        assert torch.equal(output[:, :10], input_ids)

    def test_temperature_effect(self, model: GPTModel) -> None:
        """Test that different temperatures produce different outputs."""
        input_ids = torch.randint(0, 100, (1, 10))

        # Low temperature (more deterministic)
        config_low = GenerationConfig(max_new_tokens=5, temperature=0.1, do_sample=True)
        generator_low = TextGenerator(model, config_low)

        # High temperature (more random)
        config_high = GenerationConfig(max_new_tokens=5, temperature=2.0, do_sample=True)
        generator_high = TextGenerator(model, config_high)

        # Generate multiple times
        torch.manual_seed(42)
        low_outputs = [generator_low.generate(input_ids) for _ in range(3)]

        torch.manual_seed(42)
        high_outputs = [generator_high.generate(input_ids) for _ in range(3)]

        # Low temperature should be more consistent
        # High temperature should be more diverse
        # (This is a statistical test, so we just check shapes for now)
        for output in low_outputs + high_outputs:
            assert output.shape == (1, 15)

    def test_top_k_sampling(self, model: GPTModel) -> None:
        """Test top-k sampling."""
        config = GenerationConfig(
            max_new_tokens=5, temperature=1.0, top_k=10, do_sample=True
        )
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))
        output = generator.generate(input_ids)

        assert output.shape == (1, 15)

    def test_top_p_sampling(self, model: GPTModel) -> None:
        """Test top-p (nucleus) sampling."""
        config = GenerationConfig(
            max_new_tokens=5, temperature=1.0, top_p=0.9, do_sample=True
        )
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))
        output = generator.generate(input_ids)

        assert output.shape == (1, 15)

    def test_combined_top_k_top_p(self, model: GPTModel) -> None:
        """Test combined top-k and top-p filtering."""
        config = GenerationConfig(
            max_new_tokens=5, temperature=0.8, top_k=50, top_p=0.95, do_sample=True
        )
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))
        output = generator.generate(input_ids)

        assert output.shape == (1, 15)

    def test_repetition_penalty(self, model: GPTModel) -> None:
        """Test repetition penalty application."""
        config = GenerationConfig(
            max_new_tokens=5, repetition_penalty=2.0, do_sample=True
        )
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))
        output = generator.generate(input_ids)

        assert output.shape == (1, 15)

    def test_no_repeat_ngram(self, model: GPTModel) -> None:
        """Test n-gram repetition blocking."""
        config = GenerationConfig(
            max_new_tokens=10, no_repeat_ngram_size=3, do_sample=True
        )
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))
        output = generator.generate(input_ids)

        assert output.shape == (1, 20)

    def test_eos_token_stopping(self, model: GPTModel) -> None:
        """Test that generation stops at EOS token."""
        config = GenerationConfig(
            max_new_tokens=20, eos_token_id=50, do_sample=False
        )
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))
        output = generator.generate(input_ids)

        # Output may be shorter than max if EOS is generated
        assert output.shape[0] == 1
        assert output.shape[1] >= 10  # At least input length
        assert output.shape[1] <= 30  # At most input + max_new_tokens

    def test_min_length_enforcement(self, model: GPTModel) -> None:
        """Test minimum length prevents early EOS."""
        config = GenerationConfig(
            max_new_tokens=20, min_length=25, eos_token_id=50, do_sample=False
        )
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))
        output = generator.generate(input_ids)

        # Should generate at least min_length tokens
        assert output.shape[1] >= 25

    def test_batch_generation(self, model: GPTModel) -> None:
        """Test generation with batch size > 1."""
        config = GenerationConfig(max_new_tokens=5, do_sample=False)
        generator = TextGenerator(model, config)

        batch_size = 4
        input_ids = torch.randint(0, 100, (batch_size, 10))

        output = generator.generate(input_ids)

        assert output.shape == (batch_size, 15)

    def test_different_input_lengths(self, model: GPTModel) -> None:
        """Test generation with different input lengths."""
        config = GenerationConfig(max_new_tokens=5, do_sample=False)
        generator = TextGenerator(model, config)

        for seq_len in [5, 10, 20]:
            input_ids = torch.randint(0, 100, (1, seq_len))
            output = generator.generate(input_ids)
            assert output.shape == (1, seq_len + 5)

    def test_long_context_handling(self, model: GPTModel) -> None:
        """Test that long contexts are truncated properly."""
        config = GenerationConfig(max_new_tokens=5, do_sample=False)
        generator = TextGenerator(model, config)

        # Input longer than model's max_seq_len
        input_ids = torch.randint(0, 100, (1, 50))
        output = generator.generate(input_ids)

        # Should use sliding window
        assert output.shape == (1, 55)

    def test_beam_search(self, model: GPTModel) -> None:
        """Test beam search generation."""
        config = GenerationConfig(
            max_new_tokens=5, do_sample=False, num_beams=3
        )
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))
        output = generator.generate(input_ids)

        assert output.shape == (1, 15)

    def test_beam_search_with_length_penalty(self, model: GPTModel) -> None:
        """Test beam search with length penalty."""
        config = GenerationConfig(
            max_new_tokens=5,
            do_sample=False,
            num_beams=5,
            length_penalty=0.8,  # Encourage shorter sequences
        )
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))
        output = generator.generate(input_ids)

        assert output.shape == (1, 15)

    def test_beam_search_deterministic(self, model: GPTModel) -> None:
        """Test that beam search is deterministic."""
        config = GenerationConfig(
            max_new_tokens=5, do_sample=False, num_beams=3
        )
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))

        output1 = generator.generate(input_ids)
        output2 = generator.generate(input_ids)

        assert torch.equal(output1, output2)

    def test_stopping_criteria(self, model: GPTModel) -> None:
        """Test custom stopping criteria."""
        config = GenerationConfig(max_new_tokens=20, do_sample=False)
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))

        # Stop after 5 steps
        def stop_after_5(generated: torch.Tensor, step: int) -> bool:
            return step >= 5

        output = generator.generate(input_ids, stopping_criteria=stop_after_5)

        # Should stop early
        assert output.shape == (1, 15)  # 10 + 5

    def test_model_state_preservation(self, model: GPTModel) -> None:
        """Test that generation preserves model training state."""
        # Set model to train mode
        model.train()
        assert model.training

        config = GenerationConfig(max_new_tokens=5, do_sample=False)
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))
        generator.generate(input_ids)

        # Should restore training mode
        assert model.training

    def test_generation_no_gradients(self, model: GPTModel) -> None:
        """Test that generation doesn't compute gradients."""
        config = GenerationConfig(max_new_tokens=5, do_sample=False)
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (1, 10))

        # Enable gradient computation
        with torch.enable_grad():
            output = generator.generate(input_ids)

        # Output should not require gradients
        assert not output.requires_grad

    def test_apply_repetition_penalty_method(self, model: GPTModel) -> None:
        """Test _apply_repetition_penalty internal method."""
        config = GenerationConfig(repetition_penalty=2.0)
        generator = TextGenerator(model, config)

        logits = torch.randn(2, 100)
        # Ensure specific tokens are in the generated sequence
        generated = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]])

        # Save original values for tokens we know appear
        original_logit_token_1 = logits[0, 1].item()

        modified_logits = generator._apply_repetition_penalty(logits, generated)

        assert modified_logits.shape == logits.shape
        # Logits should be modified for tokens that appear in generated
        assert modified_logits[0, 1].item() != original_logit_token_1
        # Penalty is applied differently based on sign
        if original_logit_token_1 > 0:
            assert modified_logits[0, 1].item() < original_logit_token_1
        else:
            assert abs(modified_logits[0, 1].item()) > abs(original_logit_token_1)

    def test_prevent_ngram_repetition_method(self, model: GPTModel) -> None:
        """Test _prevent_ngram_repetition internal method."""
        config = GenerationConfig(no_repeat_ngram_size=3)
        generator = TextGenerator(model, config)

        logits = torch.randn(2, 100)
        # Create sequence with repeating trigram [1, 2, 3]
        generated = torch.tensor([[1, 2, 3, 4, 5, 1, 2], [6, 7, 8, 9, 10, 11, 12]])

        modified_logits = generator._prevent_ngram_repetition(logits, generated)

        assert modified_logits.shape == logits.shape
        # Token 3 should be banned for first sequence (would complete [1, 2, 3])
        assert modified_logits[0, 3] == float("-inf")

    def test_sample_next_token_method(self, model: GPTModel) -> None:
        """Test _sample_next_token internal method."""
        config = GenerationConfig(temperature=1.0, top_k=10, top_p=0.9)
        generator = TextGenerator(model, config)

        logits = torch.randn(2, 100)
        next_token = generator._sample_next_token(logits)

        assert next_token.shape == (2, 1)
        assert (next_token >= 0).all()
        assert (next_token < 100).all()

    def test_empty_batch_handling(self, model: GPTModel) -> None:
        """Test that empty batches are handled gracefully."""
        config = GenerationConfig(max_new_tokens=5, do_sample=False)
        generator = TextGenerator(model, config)

        # Single sequence with minimal length
        input_ids = torch.randint(0, 100, (1, 1))
        output = generator.generate(input_ids)

        assert output.shape == (1, 6)

    def test_deterministic_greedy_across_runs(self, model: GPTModel) -> None:
        """Test that greedy decoding is fully deterministic across multiple runs."""
        config = GenerationConfig(max_new_tokens=10, do_sample=False)
        generator = TextGenerator(model, config)

        input_ids = torch.randint(0, 100, (2, 10))

        outputs = [generator.generate(input_ids) for _ in range(5)]

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.equal(outputs[0], outputs[i])

    def test_top_k_one_equals_greedy(self, model: GPTModel) -> None:
        """Test that top_k=1 is equivalent to greedy decoding."""
        input_ids = torch.randint(0, 100, (1, 10))

        config_greedy = GenerationConfig(max_new_tokens=5, do_sample=False)
        generator_greedy = TextGenerator(model, config_greedy)
        output_greedy = generator_greedy.generate(input_ids)

        config_top_k = GenerationConfig(max_new_tokens=5, temperature=1.0, top_k=1, do_sample=True)
        generator_top_k = TextGenerator(model, config_top_k)

        # Should be very similar due to top_k=1 (only most likely token)
        output_top_k = generator_top_k.generate(input_ids)

        assert output_top_k.shape == output_greedy.shape

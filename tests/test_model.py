"""Tests for GPT model."""

import pytest
import torch

from src.llm.config import ModelConfig
from src.llm.model import GPTModel


class TestGPTModel:
    """Tests for GPTModel class."""

    def test_initialization(self) -> None:
        """Test model initialization."""
        config = ModelConfig(
            vocab_size=1000, max_seq_len=128, d_model=256, n_layers=4, n_heads=4
        )
        model = GPTModel(config)

        assert model.config == config
        assert isinstance(model.token_embedding, torch.nn.Embedding)
        assert isinstance(model.position_embedding, torch.nn.Embedding)
        assert len(model.blocks) == 4
        assert isinstance(model.ln_final, torch.nn.LayerNorm)
        assert isinstance(model.lm_head, torch.nn.Linear)

    def test_weight_tying(self) -> None:
        """Test that token embedding and lm_head share weights."""
        config = ModelConfig()
        model = GPTModel(config)

        # Check that weights are the same object (not just equal)
        assert model.lm_head.weight is model.token_embedding.weight

    def test_invalid_config_type(self) -> None:
        """Test initialization with invalid config type."""
        with pytest.raises(TypeError, match="Expected ModelConfig"):
            GPTModel({"vocab_size": 1000})  # type: ignore

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        config = ModelConfig(vocab_size=1000, max_seq_len=128, d_model=384, n_heads=6)
        model = GPTModel(config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        logits, loss = model(input_ids)

        assert logits.shape == (batch_size, seq_len, 1000)
        assert loss is None

    def test_forward_with_loss(self) -> None:
        """Test forward pass with loss computation."""
        config = ModelConfig(vocab_size=1000, max_seq_len=128, d_model=384, n_heads=6)
        model = GPTModel(config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        targets = torch.randint(0, 1000, (batch_size, seq_len))

        logits, loss = model(input_ids, targets=targets, return_loss=True)

        assert logits.shape == (batch_size, seq_len, 1000)
        assert loss is not None
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Cross-entropy is non-negative

    def test_forward_without_targets_raises(self) -> None:
        """Test that return_loss=True without targets raises error."""
        config = ModelConfig()
        model = GPTModel(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 10))

        with pytest.raises(RuntimeError, match="targets must be provided"):
            model(input_ids, return_loss=True)

    def test_forward_different_batch_sizes(self) -> None:
        """Test forward pass with different batch sizes."""
        config = ModelConfig(vocab_size=1000)
        model = GPTModel(config)

        for batch_size in [1, 4, 8]:
            input_ids = torch.randint(0, 1000, (batch_size, 10))
            logits, _ = model(input_ids)
            assert logits.shape == (batch_size, 10, 1000)

    def test_forward_different_seq_lengths(self) -> None:
        """Test forward pass with different sequence lengths."""
        config = ModelConfig(vocab_size=1000, max_seq_len=128)
        model = GPTModel(config)

        for seq_len in [1, 5, 20, 50, 128]:
            input_ids = torch.randint(0, 1000, (2, seq_len))
            logits, _ = model(input_ids)
            assert logits.shape == (2, seq_len, 1000)

    def test_forward_max_seq_len_exceeded(self) -> None:
        """Test that exceeding max_seq_len raises error."""
        config = ModelConfig(max_seq_len=128)
        model = GPTModel(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 129))  # Exceeds max

        with pytest.raises(ValueError, match="exceeds maximum"):
            model(input_ids)

    def test_forward_empty_sequence(self) -> None:
        """Test that empty sequence raises error."""
        config = ModelConfig()
        model = GPTModel(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 0))  # Empty

        with pytest.raises(ValueError, match="Cannot process empty sequence"):
            model(input_ids)

    def test_forward_invalid_input_shape(self) -> None:
        """Test that invalid input shape raises error."""
        config = ModelConfig()
        model = GPTModel(config)

        # 1D input
        input_ids_1d = torch.randint(0, config.vocab_size, (10,))
        with pytest.raises(ValueError, match="Expected 2D input_ids"):
            model(input_ids_1d)

        # 3D input
        input_ids_3d = torch.randint(0, config.vocab_size, (2, 10, 5))
        with pytest.raises(ValueError, match="Expected 2D input_ids"):
            model(input_ids_3d)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        config = ModelConfig(vocab_size=1000, d_model=384, n_heads=6)
        model = GPTModel(config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        targets = torch.randint(0, 1000, (batch_size, seq_len))

        logits, loss = model(input_ids, targets=targets, return_loss=True)
        assert loss is not None

        loss.backward()

        # Check that gradients exist and are valid
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_generate_shape(self) -> None:
        """Test generation output shape."""
        config = ModelConfig(vocab_size=1000, max_seq_len=128)
        model = GPTModel(config)
        model.eval()

        batch_size, seq_len = 2, 10
        max_new_tokens = 5
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        generated = model.generate(input_ids, max_new_tokens=max_new_tokens)

        assert generated.shape == (batch_size, seq_len + max_new_tokens)

    def test_generate_with_temperature(self) -> None:
        """Test generation with different temperatures."""
        config = ModelConfig(vocab_size=1000, max_seq_len=128)
        model = GPTModel(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 10))

        # Should work with different temperatures
        for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
            generated = model.generate(input_ids, max_new_tokens=5, temperature=temp)
            assert generated.shape == (2, 15)

    def test_generate_with_top_k(self) -> None:
        """Test generation with top-k sampling."""
        config = ModelConfig(vocab_size=1000, max_seq_len=128)
        model = GPTModel(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 10))

        # Should work with different top_k values
        for top_k in [1, 10, 50]:
            generated = model.generate(input_ids, max_new_tokens=5, top_k=top_k)
            assert generated.shape == (2, 15)

    def test_generate_invalid_temperature(self) -> None:
        """Test that invalid temperature raises error."""
        config = ModelConfig()
        model = GPTModel(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 10))

        with pytest.raises(ValueError, match="temperature must be positive"):
            model.generate(input_ids, max_new_tokens=5, temperature=0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            model.generate(input_ids, max_new_tokens=5, temperature=-1.0)

    def test_generate_invalid_top_k(self) -> None:
        """Test that invalid top_k raises error."""
        config = ModelConfig()
        model = GPTModel(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 10))

        with pytest.raises(ValueError, match="top_k must be >= 1"):
            model.generate(input_ids, max_new_tokens=5, top_k=0)

    def test_generate_handles_long_context(self) -> None:
        """Test that generation handles sequences longer than max_seq_len."""
        config = ModelConfig(max_seq_len=50)
        model = GPTModel(config)
        model.eval()

        # Start with a long sequence
        input_ids = torch.randint(0, config.vocab_size, (1, 60))

        # Should use only last max_seq_len tokens
        generated = model.generate(input_ids, max_new_tokens=5)
        assert generated.shape == (1, 65)

    def test_num_parameters(self) -> None:
        """Test parameter counting."""
        config = ModelConfig(vocab_size=1000, d_model=384, n_heads=6, n_layers=4)
        model = GPTModel(config)

        total_params = model.num_parameters()
        non_emb_params = model.num_parameters(exclude_embeddings=True)

        assert total_params > 0
        assert non_emb_params > 0
        assert total_params > non_emb_params  # Embeddings contribute to total

        # Embeddings should account for the difference
        emb_params = (
            model.token_embedding.weight.numel() + model.position_embedding.weight.numel()
        )
        assert total_params == non_emb_params + emb_params

    def test_weight_initialization(self) -> None:
        """Test that weights are properly initialized."""
        config = ModelConfig()
        model = GPTModel(config)

        # Check that weights are not all zero (properly initialized)
        assert not torch.allclose(model.token_embedding.weight, torch.zeros_like(model.token_embedding.weight))
        assert not torch.allclose(model.position_embedding.weight, torch.zeros_like(model.position_embedding.weight))

        # Check layer norm initialized correctly
        assert torch.allclose(model.ln_final.weight, torch.ones_like(model.ln_final.weight))
        assert torch.allclose(model.ln_final.bias, torch.zeros_like(model.ln_final.bias))

    def test_residual_projection_markers(self) -> None:
        """Test that residual projections are marked for scaled initialization."""
        config = ModelConfig(n_layers=4)
        model = GPTModel(config)

        # Check that residual projections have the marker attribute
        for block in model.blocks:
            assert hasattr(block.attn.out_proj, "_is_residual_projection")
            assert hasattr(block.ffn.fc2, "_is_residual_projection")

    def test_deterministic_forward(self) -> None:
        """Test that forward pass is deterministic in eval mode."""
        config = ModelConfig(vocab_size=1000, dropout=0.0)
        model = GPTModel(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 10))

        logits1, _ = model(input_ids)
        logits2, _ = model(input_ids)

        assert torch.allclose(logits1, logits2)

    def test_different_positions_different_outputs(self) -> None:
        """Test that positional embeddings affect output."""
        config = ModelConfig(vocab_size=1000, dropout=0.0)
        model = GPTModel(config)
        model.eval()

        # Same tokens in different positions
        input_ids = torch.tensor([[1, 2], [2, 1]])

        logits, _ = model(input_ids)

        # Outputs should be different due to positional encoding
        assert not torch.allclose(logits[0, 0], logits[1, 1])
        assert not torch.allclose(logits[0, 1], logits[1, 0])

    def test_causal_masking(self) -> None:
        """Test that model uses causal masking (cannot attend to future)."""
        config = ModelConfig(vocab_size=1000, dropout=0.0)
        model = GPTModel(config)
        model.eval()

        # Two sequences: one with extra tokens at the end
        input_ids_short = torch.randint(0, 1000, (1, 5))
        input_ids_long = torch.cat([input_ids_short, torch.randint(0, 1000, (1, 3))], dim=1)

        logits_short, _ = model(input_ids_short)
        logits_long, _ = model(input_ids_long)

        # Due to causal masking, predictions for first 5 tokens should be same
        # (future tokens shouldn't affect past predictions)
        assert torch.allclose(logits_short, logits_long[:, :5, :], atol=1e-5)

    def test_repr(self) -> None:
        """Test string representation."""
        config = ModelConfig(vocab_size=1000, d_model=384, n_heads=6, n_layers=4)
        model = GPTModel(config)

        repr_str = repr(model)

        assert "GPTModel" in repr_str
        assert "vocab_size=1000" in repr_str
        assert "d_model=384" in repr_str
        assert "n_layers=4" in repr_str
        assert "parameters=" in repr_str

    def test_model_modes(self) -> None:
        """Test that model can switch between train and eval modes."""
        config = ModelConfig()
        model = GPTModel(config)

        # Default is train mode
        assert model.training

        # Switch to eval
        model.eval()
        assert not model.training

        # Switch back to train
        model.train()
        assert model.training

    def test_generate_deterministic_with_top_k_1(self) -> None:
        """Test that top_k=1 generates deterministically."""
        config = ModelConfig(vocab_size=1000)
        model = GPTModel(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 10))

        # With top_k=1, should always pick the most likely token (deterministic)
        generated1 = model.generate(input_ids, max_new_tokens=5, top_k=1, temperature=1.0)
        generated2 = model.generate(input_ids, max_new_tokens=5, top_k=1, temperature=1.0)

        assert torch.equal(generated1, generated2)

    def test_loss_decreases_with_training(self) -> None:
        """Test that loss decreases with a training step (sanity check)."""
        config = ModelConfig(vocab_size=100, d_model=384, n_heads=6, n_layers=2)
        model = GPTModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Simple training loop
        input_ids = torch.randint(0, 100, (4, 10))
        targets = torch.randint(0, 100, (4, 10))

        # Initial loss
        _, loss_initial = model(input_ids, targets=targets, return_loss=True)
        assert loss_initial is not None
        initial_loss_value = loss_initial.item()

        # Training step
        for _ in range(5):
            optimizer.zero_grad()
            _, loss = model(input_ids, targets=targets, return_loss=True)
            assert loss is not None
            loss.backward()
            optimizer.step()

        # Loss after training
        _, loss_final = model(input_ids, targets=targets, return_loss=True)
        assert loss_final is not None
        final_loss_value = loss_final.item()

        # Loss should decrease (this is a simple overfitting test)
        assert final_loss_value < initial_loss_value

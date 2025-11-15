"""Tests for configuration modules."""

import pytest

from src.llm.config import ModelConfig, TrainingConfig


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self) -> None:
        """Test that default configuration is valid."""
        config = ModelConfig()
        assert config.vocab_size == 50257
        assert config.max_seq_len == 256
        assert config.d_model == 384
        assert config.n_layers == 6
        assert config.n_heads == 6
        assert config.d_ff == 1536
        assert config.dropout == 0.1

    def test_head_dim_property(self) -> None:
        """Test head_dim calculation."""
        config = ModelConfig(d_model=384, n_heads=6)
        assert config.head_dim == 64

        config = ModelConfig(d_model=768, n_heads=12)
        assert config.head_dim == 64

    def test_d_model_divisible_by_n_heads(self) -> None:
        """Test that d_model must be divisible by n_heads."""
        with pytest.raises(ValueError, match="d_model.*must be divisible by n_heads"):
            ModelConfig(d_model=385, n_heads=6)

    def test_d_ff_validation(self) -> None:
        """Test that d_ff should be >= d_model."""
        with pytest.raises(ValueError, match="d_ff.*should be >= d_model"):
            ModelConfig(d_model=384, d_ff=100)

    def test_dropout_validation(self) -> None:
        """Test dropout must be in valid range."""
        with pytest.raises(ValueError, match="dropout.*must be in"):
            ModelConfig(dropout=-0.1)

        with pytest.raises(ValueError, match="dropout.*must be in"):
            ModelConfig(dropout=1.5)

        # Valid edge cases
        config = ModelConfig(dropout=0.0)
        assert config.dropout == 0.0

        config = ModelConfig(dropout=0.99)
        assert config.dropout == 0.99

    def test_num_parameters(self) -> None:
        """Test parameter count estimation."""
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=128,
            d_model=256,
            n_layers=4,
            n_heads=4,
            d_ff=1024,
        )

        num_params = config.num_parameters()
        assert num_params > 0
        assert isinstance(num_params, int)

        # Should scale with model size
        larger_config = ModelConfig(
            vocab_size=1000,
            max_seq_len=128,
            d_model=512,  # Double the size
            n_layers=4,
            n_heads=8,
            d_ff=2048,
        )
        assert larger_config.num_parameters() > num_params

    def test_custom_config(self) -> None:
        """Test creating custom configuration."""
        config = ModelConfig(
            vocab_size=10000,
            max_seq_len=512,
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=3072,
            dropout=0.2,
            use_bias=False,
        )

        assert config.vocab_size == 10000
        assert config.max_seq_len == 512
        assert config.d_model == 768
        assert config.n_layers == 12
        assert config.n_heads == 12
        assert config.d_ff == 3072
        assert config.dropout == 0.2
        assert config.use_bias is False


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self) -> None:
        """Test that default configuration is valid."""
        config = TrainingConfig()
        assert config.batch_size == 16
        assert config.learning_rate == 3e-4
        assert config.num_epochs == 10
        assert config.warmup_steps == 100
        assert config.max_steps is None
        assert config.gradient_accumulation_steps == 4
        assert config.max_grad_norm == 1.0
        assert config.eval_interval == 100
        assert config.save_interval == 500
        assert config.log_interval == 10
        assert config.device == "cpu"

    def test_effective_batch_size(self) -> None:
        """Test effective batch size calculation."""
        config = TrainingConfig(batch_size=16, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 64

        config = TrainingConfig(batch_size=8, gradient_accumulation_steps=1)
        assert config.effective_batch_size == 8

    def test_batch_size_validation(self) -> None:
        """Test batch_size validation."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=-1)

    def test_learning_rate_validation(self) -> None:
        """Test learning_rate validation."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=0)

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-0.001)

    def test_num_epochs_validation(self) -> None:
        """Test num_epochs validation."""
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            TrainingConfig(num_epochs=0)

        with pytest.raises(ValueError, match="num_epochs must be positive"):
            TrainingConfig(num_epochs=-1)

    def test_gradient_accumulation_validation(self) -> None:
        """Test gradient_accumulation_steps validation."""
        with pytest.raises(ValueError, match="gradient_accumulation_steps must be positive"):
            TrainingConfig(gradient_accumulation_steps=0)

    def test_device_validation(self) -> None:
        """Test device validation."""
        # Valid devices
        for device in ["cpu", "cuda", "mps"]:
            config = TrainingConfig(device=device)
            assert config.device == device

        # Invalid device
        with pytest.raises(ValueError, match="device must be"):
            TrainingConfig(device="tpu")

    def test_custom_config(self) -> None:
        """Test creating custom configuration."""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=1e-3,
            num_epochs=20,
            warmup_steps=500,
            max_steps=10000,
            gradient_accumulation_steps=2,
            max_grad_norm=0.5,
            eval_interval=200,
            save_interval=1000,
            log_interval=50,
            device="cuda",
        )

        assert config.batch_size == 32
        assert config.learning_rate == 1e-3
        assert config.num_epochs == 20
        assert config.warmup_steps == 500
        assert config.max_steps == 10000
        assert config.gradient_accumulation_steps == 2
        assert config.max_grad_norm == 0.5
        assert config.eval_interval == 200
        assert config.save_interval == 1000
        assert config.log_interval == 50
        assert config.device == "cuda"
        assert config.effective_batch_size == 64

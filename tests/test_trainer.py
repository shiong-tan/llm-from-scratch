"""Tests for training utilities."""

import math
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.llm.config import ModelConfig, TrainingConfig
from src.llm.model import GPTModel
from src.llm.trainer import Trainer


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create small model config for testing."""
        return ModelConfig(
            vocab_size=100, max_seq_len=32, d_model=64, n_layers=2, n_heads=4
        )

    @pytest.fixture
    def train_config(self) -> TrainingConfig:
        """Create training config for testing."""
        return TrainingConfig(
            batch_size=2,
            learning_rate=1e-3,
            num_epochs=2,
            warmup_steps=10,
            max_steps=50,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            eval_interval=10,
            save_interval=20,
            log_interval=5,
            device="cpu",
        )

    @pytest.fixture
    def model(self, model_config: ModelConfig) -> GPTModel:
        """Create model for testing."""
        return GPTModel(model_config)

    @pytest.fixture
    def optimizer(self, model: GPTModel) -> torch.optim.Optimizer:
        """Create optimizer for testing."""
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    @pytest.fixture
    def train_loader(self) -> DataLoader:
        """Create dummy training data loader."""
        # Create dummy data: (batch_size, seq_len)
        input_ids = torch.randint(0, 100, (20, 32))
        targets = torch.randint(0, 100, (20, 32))

        dataset = TensorDataset(input_ids, targets)

        # Custom collate function to match expected format
        def collate_fn(batch):
            input_ids = torch.stack([item[0] for item in batch])
            targets = torch.stack([item[1] for item in batch])
            return {"input_ids": input_ids, "targets": targets}

        return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    @pytest.fixture
    def val_loader(self) -> DataLoader:
        """Create dummy validation data loader."""
        input_ids = torch.randint(0, 100, (10, 32))
        targets = torch.randint(0, 100, (10, 32))

        dataset = TensorDataset(input_ids, targets)

        def collate_fn(batch):
            input_ids = torch.stack([item[0] for item in batch])
            targets = torch.stack([item[1] for item in batch])
            return {"input_ids": input_ids, "targets": targets}

        return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    def test_initialization(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Test trainer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model,
                optimizer,
                train_config,
                train_loader,
                val_loader,
                checkpoint_dir=tmpdir,
            )

            assert trainer.model is model
            assert trainer.optimizer is optimizer
            assert trainer.config is train_config
            assert trainer.step == 0
            assert trainer.epoch == 0
            assert trainer.best_val_loss == float("inf")
            assert Path(tmpdir).exists()

    def test_checkpoint_directory_creation(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
    ) -> None:
        """Test that checkpoint directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            trainer = Trainer(
                model, optimizer, train_config, train_loader, checkpoint_dir=str(checkpoint_dir)
            )

            assert checkpoint_dir.exists()
            assert checkpoint_dir.is_dir()

    def test_get_lr_warmup(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
    ) -> None:
        """Test learning rate during warmup phase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model, optimizer, train_config, train_loader, checkpoint_dir=tmpdir
            )

            # Step 0: lr should be lr/warmup_steps (not 0, to avoid wasting first batch)
            trainer.step = 0
            expected_lr = train_config.learning_rate * (1 / train_config.warmup_steps)
            assert trainer.get_lr() == pytest.approx(expected_lr)

            # Step 5: lr should be ~0.6 * base_lr
            trainer.step = 5
            expected_lr = train_config.learning_rate * (6 / train_config.warmup_steps)
            assert trainer.get_lr() == pytest.approx(expected_lr)

            # Step 9: lr should be close to base_lr (10/10 = 1.0)
            trainer.step = 9
            expected_lr = train_config.learning_rate * (10 / train_config.warmup_steps)
            assert trainer.get_lr() == pytest.approx(expected_lr)

    def test_get_lr_cosine_decay(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
    ) -> None:
        """Test learning rate during cosine decay phase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model, optimizer, train_config, train_loader, checkpoint_dir=tmpdir
            )

            # After warmup, should start cosine decay
            trainer.step = 20  # Warmup is 10, max_steps is 50
            lr = trainer.get_lr()
            assert lr < train_config.learning_rate  # Should be decaying

            # At end of training, should approach 0
            trainer.step = 50
            lr_end = trainer.get_lr()
            assert lr_end < lr  # Further decayed
            assert lr_end < train_config.learning_rate * 0.1  # Close to 0

    def test_update_lr(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
    ) -> None:
        """Test that update_lr correctly updates optimizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model, optimizer, train_config, train_loader, checkpoint_dir=tmpdir
            )

            trainer.step = 5
            expected_lr = trainer.get_lr()
            trainer.update_lr()

            for param_group in optimizer.param_groups:
                assert param_group["lr"] == pytest.approx(expected_lr)

    def test_train_step(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
    ) -> None:
        """Test single training step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model, optimizer, train_config, train_loader, checkpoint_dir=tmpdir
            )

            input_ids = torch.randint(0, 100, (2, 32))
            targets = torch.randint(0, 100, (2, 32))

            logits, loss = trainer.train_step(input_ids, targets)

            assert logits.shape == (2, 32, 100)
            assert loss.dim() == 0  # Scalar
            assert loss.item() >= 0  # Non-negative

    def test_backward_step(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
    ) -> None:
        """Test backward pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model, optimizer, train_config, train_loader, checkpoint_dir=tmpdir
            )

            input_ids = torch.randint(0, 100, (2, 32))
            targets = torch.randint(0, 100, (2, 32))

            _, loss = trainer.train_step(input_ids, targets)
            trainer.backward_step(loss)

            # Check that gradients exist
            for param in model.parameters():
                if param.requires_grad:
                    assert param.grad is not None

    def test_optimizer_step(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
    ) -> None:
        """Test optimizer step with gradient clipping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model, optimizer, train_config, train_loader, checkpoint_dir=tmpdir
            )

            input_ids = torch.randint(0, 100, (2, 32))
            targets = torch.randint(0, 100, (2, 32))

            # Save initial parameters
            initial_params = [p.clone() for p in model.parameters()]

            _, loss = trainer.train_step(input_ids, targets)
            trainer.backward_step(loss)
            trainer.optimizer_step()

            # Check that parameters changed
            for initial_param, current_param in zip(initial_params, model.parameters()):
                if current_param.requires_grad:
                    assert not torch.allclose(initial_param, current_param)

            # Check that gradients are cleared
            for param in model.parameters():
                assert param.grad is None or torch.allclose(
                    param.grad, torch.zeros_like(param.grad)
                )

    def test_validate(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Test validation loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model,
                optimizer,
                train_config,
                train_loader,
                val_loader,
                checkpoint_dir=tmpdir,
            )

            metrics = trainer.validate()

            assert "loss" in metrics
            assert "perplexity" in metrics
            assert metrics["loss"] >= 0
            assert metrics["perplexity"] >= 1  # Perplexity is always >= 1

    def test_validate_without_val_loader_raises(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
    ) -> None:
        """Test that validate raises error without val_loader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model, optimizer, train_config, train_loader, checkpoint_dir=tmpdir
            )

            with pytest.raises(ValueError, match="Validation loader not provided"):
                trainer.validate()

    def test_save_checkpoint(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
    ) -> None:
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model, optimizer, train_config, train_loader, checkpoint_dir=tmpdir
            )

            trainer.step = 100
            trainer.epoch = 5
            trainer.best_val_loss = 2.5

            checkpoint_path = trainer.save_checkpoint()

            assert Path(checkpoint_path).exists()
            assert "step_100.pt" in checkpoint_path

    def test_save_checkpoint_custom_name(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
    ) -> None:
        """Test checkpoint saving with custom name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model, optimizer, train_config, train_loader, checkpoint_dir=tmpdir
            )

            checkpoint_path = trainer.save_checkpoint("custom_model.pt")

            assert Path(checkpoint_path).exists()
            assert "custom_model.pt" in checkpoint_path

    def test_load_checkpoint(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
    ) -> None:
        """Test checkpoint loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save checkpoint
            trainer1 = Trainer(
                model, optimizer, train_config, train_loader, checkpoint_dir=tmpdir
            )
            trainer1.step = 100
            trainer1.epoch = 5
            trainer1.best_val_loss = 2.5
            checkpoint_path = trainer1.save_checkpoint()

            # Create new trainer and load checkpoint
            new_model = GPTModel(model.config)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
            trainer2 = Trainer(
                new_model,
                new_optimizer,
                train_config,
                train_loader,
                checkpoint_dir=tmpdir,
            )

            trainer2.load_checkpoint(checkpoint_path)

            assert trainer2.step == 100
            assert trainer2.epoch == 5
            assert trainer2.best_val_loss == 2.5

            # Check that model weights match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)

    def test_load_nonexistent_checkpoint_raises(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
    ) -> None:
        """Test that loading nonexistent checkpoint raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model, optimizer, train_config, train_loader, checkpoint_dir=tmpdir
            )

            with pytest.raises(FileNotFoundError):
                trainer.load_checkpoint("nonexistent.pt")

    def test_train_basic(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
        train_loader: DataLoader,
    ) -> None:
        """Test basic training loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use very small config for fast testing
            small_config = TrainingConfig(
                batch_size=2,
                learning_rate=1e-3,
                num_epochs=1,
                warmup_steps=2,
                max_steps=5,
                gradient_accumulation_steps=1,
                eval_interval=1000,  # Disable eval
                save_interval=1000,  # Disable save
                log_interval=2,
                device="cpu",
            )

            trainer = Trainer(
                model, optimizer, small_config, train_loader, checkpoint_dir=tmpdir
            )

            stats = trainer.train()

            assert trainer.step == 5
            assert "final_step" in stats
            assert "final_checkpoint" in stats
            assert Path(stats["final_checkpoint"]).exists()

    def test_train_with_validation(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Test training with validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            small_config = TrainingConfig(
                batch_size=2,
                learning_rate=1e-3,
                num_epochs=1,
                warmup_steps=2,
                max_steps=6,
                gradient_accumulation_steps=1,
                eval_interval=3,  # Validate at step 3 and 6
                save_interval=1000,
                log_interval=2,
                device="cpu",
            )

            trainer = Trainer(
                model,
                optimizer,
                small_config,
                train_loader,
                val_loader,
                checkpoint_dir=tmpdir,
            )

            stats = trainer.train()

            assert trainer.step == 6
            assert len(trainer.metrics_history["val_loss"]) >= 1
            assert stats["best_val_loss"] < float("inf")

    def test_gradient_accumulation(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
    ) -> None:
        """Test gradient accumulation works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                batch_size=2,
                learning_rate=1e-3,
                num_epochs=1,
                warmup_steps=1,
                max_steps=2,
                gradient_accumulation_steps=4,  # Accumulate over 4 batches
                eval_interval=1000,
                save_interval=1000,
                log_interval=1,
                device="cpu",
            )

            trainer = Trainer(
                model, optimizer, config, train_loader, checkpoint_dir=tmpdir
            )

            initial_params = [p.clone() for p in model.parameters()]

            stats = trainer.train()

            # After gradient accumulation, parameters should have changed
            for initial_param, current_param in zip(initial_params, model.parameters()):
                if current_param.requires_grad:
                    assert not torch.allclose(initial_param, current_param)

            assert trainer.step == 2

    def test_metrics_tracking(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
    ) -> None:
        """Test that metrics are tracked correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                batch_size=2,
                learning_rate=1e-3,
                num_epochs=1,
                warmup_steps=2,
                max_steps=10,
                gradient_accumulation_steps=1,
                eval_interval=1000,
                save_interval=1000,
                log_interval=2,
                device="cpu",
            )

            trainer = Trainer(
                model, optimizer, config, train_loader, checkpoint_dir=tmpdir
            )

            trainer.train()

            assert len(trainer.metrics_history["train_loss"]) > 0
            assert len(trainer.metrics_history["learning_rate"]) > 0
            assert len(trainer.metrics_history["step"]) > 0

            # Check that metrics match
            assert len(trainer.metrics_history["train_loss"]) == len(
                trainer.metrics_history["step"]
            )

    def test_loss_decreases_with_training(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
    ) -> None:
        """Test that loss decreases during training (overfitting test)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                batch_size=2,
                learning_rate=5e-3,  # Higher LR for faster convergence
                num_epochs=3,  # Need 3 epochs to achieve 30 steps with 10 batches
                warmup_steps=5,
                max_steps=30,
                gradient_accumulation_steps=1,
                eval_interval=1000,
                save_interval=1000,
                log_interval=5,
                device="cpu",
            )

            trainer = Trainer(
                model, optimizer, config, train_loader, checkpoint_dir=tmpdir
            )

            trainer.train()

            # Loss should generally decrease
            losses = trainer.metrics_history["train_loss"]
            assert len(losses) >= 2

            # Check that final loss is lower than initial loss
            initial_loss = losses[0]
            final_loss = losses[-1]
            assert final_loss < initial_loss

    def test_checkpoint_saves_at_interval(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
    ) -> None:
        """Test that checkpoints are saved at specified intervals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                batch_size=2,
                learning_rate=1e-3,
                num_epochs=1,
                warmup_steps=1,
                max_steps=10,
                gradient_accumulation_steps=1,
                eval_interval=1000,
                save_interval=5,  # Save every 5 steps
                log_interval=2,
                device="cpu",
            )

            trainer = Trainer(
                model, optimizer, config, train_loader, checkpoint_dir=tmpdir
            )

            trainer.train()

            # Check that checkpoints exist
            checkpoint_dir = Path(tmpdir)
            checkpoints = list(checkpoint_dir.glob("step_*.pt"))

            # Should have at least one checkpoint (step_5.pt)
            assert len(checkpoints) >= 1

    def test_best_model_saved(
        self,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Test that best model is saved during validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                batch_size=2,
                learning_rate=1e-3,
                num_epochs=1,
                warmup_steps=1,
                max_steps=6,
                gradient_accumulation_steps=1,
                eval_interval=3,
                save_interval=1000,
                log_interval=2,
                device="cpu",
            )

            trainer = Trainer(
                model,
                optimizer,
                config,
                train_loader,
                val_loader,
                checkpoint_dir=tmpdir,
            )

            trainer.train()

            # Check that best_model.pt exists
            best_model_path = Path(tmpdir) / "best_model.pt"
            assert best_model_path.exists()

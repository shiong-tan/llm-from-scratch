"""Training utilities for GPT model."""

import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .model import GPTModel


class Trainer:
    """Trainer for GPT model with gradient accumulation and checkpointing.

    This implements a complete training loop with:
    - Gradient accumulation for large effective batch sizes
    - Learning rate warmup and cosine decay scheduling
    - Gradient clipping for training stability
    - Checkpointing with optimizer state
    - Validation loop with perplexity calculation
    - Mixed precision training support (optional)
    - Metrics tracking and logging

    Args:
        model: GPT model to train
        optimizer: Optimizer instance (e.g., AdamW)
        config: Training configuration
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        checkpoint_dir: Directory to save checkpoints (default: checkpoints/)

    Attributes:
        model: The model being trained
        optimizer: Optimizer for parameter updates
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        checkpoint_dir: Checkpoint save directory
        step: Current training step
        epoch: Current epoch
        best_val_loss: Best validation loss seen
        metrics_history: Dictionary of tracked metrics over time

    Examples:
        Basic training loop:
        >>> model = GPTModel(model_config)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        >>> trainer = Trainer(model, optimizer, train_config, train_loader, val_loader)
        >>> trainer.train()

        Resume from checkpoint:
        >>> trainer.load_checkpoint("checkpoints/step_1000.pt")
        >>> trainer.train()
    """

    def __init__(
        self,
        model: GPTModel,
        optimizer: Optimizer,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Initialize trainer.

        Args:
            model: GPT model to train
            optimizer: Optimizer instance
            config: Training configuration
            train_loader: Training data loader
            val_loader: Optional validation data loader
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.accumulation_counter = 0  # Persistent counter for gradient accumulation

        # Metrics history
        self.metrics_history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "step": [],
        }

        # Mixed precision training (optional)
        self.use_amp = config.device == "cuda"
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def get_lr(self) -> float:
        """Get current learning rate with warmup and cosine decay.

        Implements:
        1. Linear warmup for first warmup_steps
        2. Cosine decay after warmup

        Returns:
            Current learning rate
        """
        if self.step < self.config.warmup_steps:
            # Linear warmup - start from lr/warmup_steps instead of 0
            return self.config.learning_rate * ((self.step + 1) / self.config.warmup_steps)
        else:
            # Cosine decay after warmup
            if self.config.max_steps is not None:
                progress = (self.step - self.config.warmup_steps) / (
                    self.config.max_steps - self.config.warmup_steps
                )
            else:
                # Estimate total steps from epochs
                total_steps = len(self.train_loader) * self.config.num_epochs
                progress = (self.step - self.config.warmup_steps) / (
                    total_steps - self.config.warmup_steps
                )

            progress = min(1.0, max(0.0, progress))
            return self.config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def update_lr(self) -> None:
        """Update optimizer learning rate based on schedule."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train_step(
        self, input_ids: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute single training step.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            targets: Target token IDs of shape (batch_size, seq_len)

        Returns:
            Tuple of (logits, loss)
        """
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits, loss = self.model(input_ids, targets=targets, return_loss=True)
        else:
            logits, loss = self.model(input_ids, targets=targets, return_loss=True)

        # Scale loss by gradient accumulation steps
        loss = loss / self.config.gradient_accumulation_steps

        return logits, loss

    def backward_step(self, loss: torch.Tensor) -> None:
        """Execute backward pass with optional mixed precision.

        Args:
            loss: Loss tensor to backpropagate
        """
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self) -> None:
        """Execute optimizer step with gradient clipping."""
        if self.config.max_grad_norm is not None:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation loop and compute metrics.

        Returns:
            Dictionary with validation metrics (loss, perplexity)

        Raises:
            ValueError: If val_loader is not provided
        """
        if self.val_loader is None:
            raise ValueError("Validation loader not provided")

        was_training = self.model.training
        try:
            self.model.eval()

            total_loss = 0.0
            total_tokens = 0

            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)

                logits, loss = self.model(input_ids, targets=targets, return_loss=True)

                # Accumulate loss weighted by number of tokens
                batch_tokens = (targets != -100).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

            # Use higher threshold to prevent premature infinity
            try:
                perplexity = math.exp(min(avg_loss, 100))
            except OverflowError:
                perplexity = float("inf")

            return {"loss": avg_loss, "perplexity": perplexity}
        finally:
            # Restore original training state
            if was_training:
                self.model.train()

    def save_checkpoint(self, filename: Optional[str] = None) -> str:
        """Save training checkpoint.

        Args:
            filename: Optional custom filename. If None, uses step_<step>.pt

        Returns:
            Path to saved checkpoint

        Examples:
            >>> trainer.save_checkpoint()  # Saves to checkpoints/step_1000.pt
            >>> trainer.save_checkpoint("best_model.pt")  # Custom name
        """
        if filename is None:
            filename = f"step_{self.step}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
            "metrics_history": self.metrics_history,
        }

        if self.use_amp:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint is incompatible

        Examples:
            >>> trainer.load_checkpoint("checkpoints/step_1000.pt")
            >>> trainer.train()  # Resume from step 1000
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # SECURITY NOTE: Using weights_only=False to load custom objects like TrainingConfig.
        # Only load checkpoints from trusted sources. For production, consider using
        # safetensors library or restructuring checkpoints to avoid custom objects.
        # See: https://pytorch.org/docs/stable/generated/torch.load.html
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.metrics_history = checkpoint.get("metrics_history", self.metrics_history)

        if self.use_amp and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

    def train(self) -> Dict[str, Any]:
        """Execute full training loop.

        Returns:
            Dictionary with final training statistics

        Examples:
            >>> stats = trainer.train()
            >>> print(f"Final loss: {stats['final_train_loss']}")
        """
        self.model.train()

        # Calculate total steps
        if self.config.max_steps is not None:
            total_steps = self.config.max_steps
        else:
            # Calculate steps based on dataset size and gradient accumulation
            batches_per_epoch = len(self.train_loader)
            steps_per_epoch = batches_per_epoch // self.config.gradient_accumulation_steps
            total_steps = steps_per_epoch * self.config.num_epochs

        # Validate that max_steps is achievable
        if self.config.max_steps is not None:
            batches_per_epoch = len(self.train_loader)
            max_possible_steps = (
                batches_per_epoch // self.config.gradient_accumulation_steps
            ) * self.config.num_epochs
            if self.config.max_steps > max_possible_steps:
                raise ValueError(
                    f"max_steps ({self.config.max_steps}) cannot be achieved with "
                    f"{batches_per_epoch} batches per epoch, "
                    f"{self.config.gradient_accumulation_steps} gradient accumulation steps, "
                    f"and {self.config.num_epochs} epochs. "
                    f"Maximum possible steps: {max_possible_steps}"
                )

        pbar = tqdm(total=total_steps, desc="Training", initial=self.step)

        accumulated_loss = 0.0

        while self.step < total_steps:
            for batch in self.train_loader:
                # Check if we've reached max_steps
                if self.config.max_steps and self.step >= self.config.max_steps:
                    break

                # Update learning rate
                self.update_lr()

                # Training step
                input_ids = batch["input_ids"]
                targets = batch["targets"]

                _, loss = self.train_step(input_ids, targets)
                self.backward_step(loss)

                # Accumulate unscaled loss for logging
                # (loss.item() is already scaled by 1/gradient_accumulation_steps)
                accumulated_loss += loss.item() * self.config.gradient_accumulation_steps

                # Increment accumulation counter
                self.accumulation_counter += 1

                # Optimizer step after gradient accumulation
                if self.accumulation_counter % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.step += 1
                    pbar.update(1)

                    # Logging
                    if self.step % self.config.log_interval == 0:
                        # Average over accumulated batches
                        avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                        current_lr = self.get_lr()

                        self.metrics_history["train_loss"].append(avg_loss)
                        self.metrics_history["learning_rate"].append(current_lr)
                        self.metrics_history["step"].append(self.step)

                        pbar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "lr": f"{current_lr:.2e}",
                            }
                        )

                        accumulated_loss = 0.0

                    # Validation
                    if (
                        self.val_loader is not None
                        and self.step % self.config.eval_interval == 0
                    ):
                        val_metrics = self.validate()
                        self.metrics_history["val_loss"].append(val_metrics["loss"])

                        pbar.write(
                            f"Step {self.step} - Val Loss: {val_metrics['loss']:.4f}, "
                            f"Perplexity: {val_metrics['perplexity']:.2f}"
                        )

                        # Save best model
                        if val_metrics["loss"] < self.best_val_loss:
                            self.best_val_loss = val_metrics["loss"]
                            self.save_checkpoint("best_model.pt")
                            pbar.write(
                                f"New best model saved (val_loss: {self.best_val_loss:.4f})"
                            )

                    # Checkpointing
                    if self.step % self.config.save_interval == 0:
                        checkpoint_path = self.save_checkpoint()
                        pbar.write(f"Checkpoint saved: {checkpoint_path}")

                    # Check again if we've reached max_steps after the update
                    if self.config.max_steps and self.step >= self.config.max_steps:
                        break

            self.epoch += 1

            # Break outer loop if max_steps reached
            if self.config.max_steps and self.step >= self.config.max_steps:
                break

        # Flush any remaining gradients at the end of training
        if self.accumulation_counter % self.config.gradient_accumulation_steps != 0:
            pbar.write(
                f"Flushing {self.accumulation_counter % self.config.gradient_accumulation_steps} "
                f"remaining accumulated gradients"
            )
            self.optimizer_step()
            self.step += 1
            pbar.update(1)

        pbar.close()

        # Final checkpoint
        final_checkpoint = self.save_checkpoint("final_model.pt")

        # Compute final statistics
        stats = {
            "final_step": self.step,
            "final_epoch": self.epoch,
            "final_train_loss": (
                self.metrics_history["train_loss"][-1]
                if self.metrics_history["train_loss"]
                else None
            ),
            "best_val_loss": self.best_val_loss if self.val_loader else None,
            "final_checkpoint": final_checkpoint,
        }

        return stats

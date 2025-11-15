"""Simple training example for GPT model.

This script demonstrates how to train a small GPT model on sample text data.
Run with: python examples/train_simple.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from src.llm import (
    GPTModel,
    ModelConfig,
    Trainer,
    TrainingConfig,
    Tokenizer,
)


def create_sample_data():
    """Create sample training and validation data."""
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Transformers revolutionized natural language processing.",
        "GPT models use attention mechanisms for text generation.",
        "Deep learning requires large amounts of training data.",
        "Neural networks learn patterns from examples.",
        "Language models predict the next word in a sequence.",
        "Attention is all you need for modern NLP.",
        "Gradient descent optimizes neural network parameters.",
        "Backpropagation computes gradients efficiently.",
    ] * 10  # Repeat for more data

    val_texts = [
        "Neural networks learn patterns from data.",
        "Language models predict the next word.",
        "Transformers use self-attention mechanisms.",
    ]

    return train_texts, val_texts


def tokenize_and_prepare_data(texts, tokenizer, max_length=64):
    """Tokenize texts and prepare for training."""
    all_ids = []

    for text in texts:
        tokens = tokenizer.encode(text)

        # Skip if too short
        if len(tokens) < 2:
            continue

        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        all_ids.append(tokens)

    return all_ids


def pad_sequences(sequences, pad_value=0):
    """Pad sequences to same length."""
    if not sequences:
        return torch.tensor([])

    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        padded.append(seq + [pad_value] * (max_len - len(seq)))
    return torch.tensor(padded)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    input_ids = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return {"input_ids": input_ids, "targets": targets}


def main():
    """Main training function."""
    print("=" * 70)
    print("GPT Model Training Example")
    print("=" * 70)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create tokenizer
    print("\n[1/6] Creating tokenizer...")
    tokenizer = Tokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")

    # Create sample data
    print("\n[2/6] Preparing training data...")
    train_texts, val_texts = create_sample_data()
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Tokenize
    train_ids = tokenize_and_prepare_data(train_texts, tokenizer)
    val_ids = tokenize_and_prepare_data(val_texts, tokenizer)

    # Pad to same length
    train_input_ids = pad_sequences(train_ids)
    train_target_ids = pad_sequences(train_ids)  # Same as input for simplicity
    val_input_ids = pad_sequences(val_ids)
    val_target_ids = pad_sequences(val_ids)

    print(f"Train tensor shape: {train_input_ids.shape}")
    print(f"Val tensor shape: {val_input_ids.shape}")

    # Create data loaders
    train_dataset = TensorDataset(train_input_ids, train_target_ids)
    val_dataset = TensorDataset(val_input_ids, val_target_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        collate_fn=collate_fn,
    )

    # Create model
    print("\n[3/6] Creating model...")
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
    )

    model = GPTModel(model_config)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Non-embedding parameters: {model.num_parameters(exclude_embeddings=True):,}")

    # Create optimizer
    print("\n[4/6] Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    # Training configuration
    train_config = TrainingConfig(
        num_epochs=5,
        learning_rate=3e-4,
        warmup_steps=100,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        log_interval=10,
        eval_interval=50,
        save_interval=100,
        device="cpu",  # Change to "cuda" if GPU available
    )

    print("Training configuration:")
    print(f"  Epochs: {train_config.num_epochs}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Warmup steps: {train_config.warmup_steps}")
    print(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"  Device: {train_config.device}")

    # Create trainer
    print("\n[5/6] Creating trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir="checkpoints",
    )

    # Train
    print("\n[6/6] Starting training...")
    print("=" * 70)
    stats = trainer.train()

    # Print final results
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final training loss: {stats['final_train_loss']:.4f}")
    print(f"Best validation loss: {stats['best_val_loss']:.4f}")
    print(f"Final checkpoint: {stats['final_checkpoint']}")
    print("\nYou can now use the trained model for generation!")
    print("See examples/generate_simple.py for how to generate text.")
    print("=" * 70)


if __name__ == "__main__":
    main()

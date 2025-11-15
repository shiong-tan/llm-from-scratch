# LLM from Scratch

[![Tests](https://github.com/shiong-tan/llm-from-scratch/workflows/CI/badge.svg)](https://github.com/shiong-tan/llm-from-scratch/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)

An educational implementation of a GPT-style Large Language Model built from scratch.

## Features

- 🏗️ **Clean Architecture**: Modular, testable components
- 📚 **Interactive Notebooks**: Step-by-step educational Jupyter notebooks with visualizations
- ✅ **Comprehensive Testing**: Unit tests with >90% coverage
- 🔍 **Type Safety**: Full type hints with mypy checking
- 📝 **Rich Documentation**: Detailed docstrings, tutorials, and explanations
- 🔧 **CI/CD**: Automated testing and linting via GitHub Actions
- 🎨 **Code Quality**: Black formatting, Ruff linting, pre-commit hooks
- 📊 **Visualizations**: Interactive plots showing attention patterns, embeddings, training dynamics

## Architecture

This implements a **decoder-only transformer** (GPT-style) with:

- **Multi-head Self-Attention**: Scaled dot-product attention mechanism
- **Position-wise Feedforward Networks**: Two-layer MLP with GELU activation
- **Layer Normalization**: Pre-norm architecture for training stability
- **Positional Embeddings**: Learned absolute position embeddings
- **Residual Connections**: Skip connections around each sub-layer

## Project Structure

```
├── src/llm/              # Source code
│   ├── config.py         # Model and training configuration
│   ├── tokenizer.py      # BPE tokenization with tiktoken
│   ├── attention.py      # Multi-head self-attention
│   ├── transformer.py    # Transformer blocks and feedforward
│   ├── model.py          # Complete GPT model
│   ├── trainer.py        # Training loop with gradient accumulation
│   └── generation.py     # Text generation strategies
├── tests/                # Comprehensive unit tests (92% coverage)
├── notebooks/            # Educational Jupyter notebooks
│   ├── 01_tokenization.ipynb
│   ├── 02_attention_mechanism.ipynb
│   ├── 03_transformer_blocks.ipynb
│   ├── 04_complete_gpt_model.ipynb
│   └── 05_training_and_generation.ipynb
├── examples/             # Example scripts
│   ├── train_simple.py   # Simple training example
│   └── generate_simple.py # Text generation example
├── data/samples/         # Sample training data
└── .github/workflows/    # CI/CD pipelines
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/shiong-tan/llm-from-scratch.git
cd llm-from-scratch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode
pip install -e "."

# For development (includes testing and linting tools)
pip install -e ".[dev]"
```

### Learning Path with Notebooks

The best way to learn is through our interactive notebooks:

```bash
# Launch JupyterLab
jupyter lab

# Navigate to notebooks/ and follow the learning path:
# 01_tokenization.ipynb - BPE encoding, vocabulary, special tokens
# 02_attention_mechanism.ipynb - Multi-head attention, causal masking
# 03_transformer_blocks.ipynb - Residual connections, layer norm, feedforward
# 04_complete_gpt_model.ipynb - Embeddings, weight tying, complete architecture
# 05_training_and_generation.ipynb - Training loop, sampling strategies, beam search
```

Each notebook includes:
- 📖 Clear explanations of concepts
- 💻 Runnable code with inline comments
- 📊 Interactive visualizations
- 🧪 Experiments to try
- ❓ Quiz questions to test understanding

### Training

```bash
# Train a small model on sample data
python examples/train_simple.py

# The script will:
# - Create a small GPT model
# - Train on sample text data
# - Save checkpoints to checkpoints/
# - Display training progress with tqdm
```

### Text Generation

```bash
# Generate text with different strategies
python examples/generate_simple.py

# The script demonstrates:
# - Greedy decoding (deterministic)
# - Temperature sampling
# - Top-k and top-p sampling
# - Beam search
# - Repetition penalty
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_attention.py
```

### Code Quality

```bash
# Format code
black src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/

# Lint code
ruff check src/ tests/ examples/
flake8 src/ tests/ examples/

# Type checking
mypy src/

# Security checks
bandit -r src/
safety check
```

## Educational Notebooks

### 01. Tokenization
- Understanding BPE (Byte Pair Encoding)
- Vocabulary size trade-offs
- Special tokens (BOS/EOS/PAD)
- Hands-on with tiktoken tokenizer
- Token-to-text conversion

### 02. Attention Mechanism
- Scaled dot-product attention formula
- Multi-head attention architecture
- Causal masking for autoregressive models
- Visualizing attention patterns
- Query, Key, Value concepts

### 03. Transformer Blocks
- Pre-norm vs post-norm architecture
- Residual connections and gradient flow
- Layer normalization
- Feedforward networks with GELU
- Stacking blocks for deep models

### 04. Complete GPT Model
- Token and positional embeddings
- Weight tying between embeddings and output
- Scaled initialization for deep networks
- Complete forward pass
- Loss computation and perplexity

### 05. Training and Generation
- Training loop fundamentals
- Learning rate warmup and cosine decay
- Gradient accumulation
- Multiple generation strategies:
  - Greedy, sampling, top-k, top-p
  - Beam search with length penalty
  - Repetition penalty and n-gram blocking

## Model Configuration

For CPU training, we use a small but educational configuration:

```yaml
model:
  n_layers: 6           # Number of transformer blocks
  n_heads: 6            # Number of attention heads
  d_model: 384          # Embedding dimension
  d_ff: 1536            # Feedforward dimension (4 * d_model)
  vocab_size: 50257     # GPT-2 tokenizer vocabulary
  max_seq_len: 256      # Maximum sequence length
  dropout: 0.1          # Dropout rate

training:
  batch_size: 16
  learning_rate: 3e-4
  num_epochs: 10
  gradient_accumulation_steps: 4
```

**Parameters**: ~10M (small enough for CPU training)

## Visualizations

Our notebooks include rich visualizations:

- **Attention Heatmaps**: See which tokens attend to which
- **Embedding Spaces**: Visualize word embeddings in 2D/3D
- **Training Curves**: Loss, perplexity, and learning rate over time
- **Token Probability Distributions**: Understand model predictions
- **Activation Patterns**: Internal representations at each layer

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 architecture
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Minimalist GPT implementation
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Line-by-line implementation

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure code quality:
   - Run tests: `pytest`
   - Format code: `black src/ tests/ examples/`
   - Check types: `mypy src/`
   - Lint: `ruff check src/`
5. Commit your changes with clear messages
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Citation

If you use this code for educational purposes, please cite:

```bibtex
@software{llm_from_scratch,
  author = {Shiong Tan},
  title = {LLM from Scratch: Educational GPT Implementation},
  year = {2025},
  url = {https://github.com/shiong-tan/llm-from-scratch}
}
```

## Acknowledgments

- Based on the GPT-2 architecture
- Built with PyTorch
- Visualizations powered by Matplotlib, Plotly, and Seaborn

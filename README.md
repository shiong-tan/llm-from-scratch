# LLM from Scratch

[![Tests](https://github.com/yourusername/llm-from-scratch/workflows/tests/badge.svg)](https://github.com/yourusername/llm-from-scratch/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)

An educational implementation of a GPT-style Large Language Model built from scratch with software engineering best practices and comprehensive educational materials.

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
│   ├── config.py         # Model configuration
│   ├── tokenizer.py      # Text tokenization
│   ├── attention.py      # Attention mechanism
│   ├── model.py          # Transformer model
│   └── trainer.py        # Training loop
├── tests/                # Unit tests
├── notebooks/            # Educational Jupyter notebooks
│   ├── 01_tokenization.ipynb
│   ├── 02_attention_mechanism.ipynb
│   ├── 03_transformer_blocks.ipynb
│   ├── 04_training_loop.ipynb
│   └── 05_text_generation.ipynb
├── examples/             # Example scripts
│   ├── train.py         # Training script
│   └── generate.py      # Text generation
├── docs/                 # Documentation
└── .github/workflows/    # CI/CD pipelines
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-from-scratch.git
cd llm-from-scratch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
pre-commit install
```

### Learning Path with Notebooks

The best way to learn is through our interactive notebooks:

```bash
# Launch JupyterLab
jupyter lab

# Navigate to notebooks/ and start with:
# 01_tokenization.ipynb - Understanding BPE and tokenization
# 02_attention_mechanism.ipynb - Visualizing attention patterns
# 03_transformer_blocks.ipynb - Building transformer layers
# 04_training_loop.ipynb - Training your first model
# 05_text_generation.ipynb - Generating text with sampling strategies
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
python examples/train.py --config examples/config_small.yaml

# Monitor training with TensorBoard
tensorboard --logdir logs/
```

### Text Generation

```bash
# Generate text from a trained model
python examples/generate.py \
    --checkpoint checkpoints/model_best.pt \
    --prompt "Once upon a time" \
    --max_length 100
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

# Format notebooks
nbqa black notebooks/

# Lint code
ruff check src/ tests/ examples/

# Type checking
mypy src/
```

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Educational Notebooks

### 01. Tokenization
- How text is converted to numbers
- BPE (Byte Pair Encoding) algorithm
- Vocabulary building
- Special tokens and padding

### 02. Attention Mechanism
- Query, Key, Value concepts
- Scaled dot-product attention
- Multi-head attention visualization
- Attention pattern analysis

### 03. Transformer Blocks
- Layer normalization
- Residual connections
- Feedforward networks
- Complete transformer block assembly

### 04. Training Loop
- Loss calculation (cross-entropy)
- Gradient descent optimization
- Learning rate scheduling
- Monitoring training metrics

### 05. Text Generation
- Greedy decoding
- Temperature sampling
- Top-k and top-p (nucleus) sampling
- Beam search

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
4. Run pre-commit hooks (`pre-commit run --all-files`)
5. Commit your changes
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Citation

If you use this code for educational purposes, please cite:

```bibtex
@software{llm_from_scratch,
  author = {Your Name},
  title = {LLM from Scratch: Educational GPT Implementation},
  year = {2025},
  url = {https://github.com/yourusername/llm-from-scratch}
}
```

## Acknowledgments

- Inspired by Andrej Karpathy's educational content
- Based on the GPT-2 architecture
- Built with PyTorch
- Visualizations powered by Matplotlib, Plotly, and Seaborn

"""Simple text generation example with GPT model.

This script demonstrates how to generate text using a trained GPT model
with various sampling strategies.

Run with: python examples/generate_simple.py
"""

import torch

from src.llm import (
    GPTModel,
    ModelConfig,
    TextGenerator,
    GenerationConfig,
    Tokenizer,
)


def create_demo_model():
    """Create a small GPT model for demonstration.

    Note: This creates an untrained model. For better results, load a checkpoint
    from training with: trainer.load_checkpoint('path/to/checkpoint.pt')
    """
    config = ModelConfig(
        vocab_size=50257,  # GPT-2 vocabulary
        max_seq_len=256,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
    )
    return GPTModel(config)


def generate_with_config(model, tokenizer, prompt, config_name, gen_config):
    """Generate text with given configuration."""
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids])

    generator = TextGenerator(model, gen_config)
    output = generator.generate(input_tensor)

    generated_text = tokenizer.decode(output[0].tolist())

    print(f"\n{config_name}:")
    print(f"  {generated_text}")


def main():
    """Main generation function."""
    print("=" * 70)
    print("GPT Model Text Generation Example")
    print("=" * 70)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create tokenizer
    print("\n[1/3] Creating tokenizer...")
    tokenizer = Tokenizer()

    # Create model (or load from checkpoint)
    print("\n[2/3] Creating model...")
    model = create_demo_model()
    model.eval()  # Set to evaluation mode

    print(f"Model parameters: {model.num_parameters():,}")
    print("\nNote: This is an UNTRAINED model for demonstration.")
    print("For better results, train the model first (see train_simple.py)")

    # Example prompts
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "Machine learning is",
    ]

    print("\n[3/3] Generating text...")
    print("=" * 70)

    for prompt in prompts:
        print(f"\nPrompt: \"{prompt}\"")
        print("-" * 70)

        # 1. Greedy decoding (deterministic)
        config = GenerationConfig(
            max_new_tokens=15,
            do_sample=False,  # Greedy
        )
        generate_with_config(model, tokenizer, prompt, "Greedy", config)

        # 2. Sampling with temperature
        config = GenerationConfig(
            max_new_tokens=15,
            do_sample=True,
            temperature=0.8,
        )
        generate_with_config(model, tokenizer, prompt, "Temperature 0.8", config)

        # 3. Top-K sampling
        config = GenerationConfig(
            max_new_tokens=15,
            do_sample=True,
            temperature=0.8,
            top_k=50,
        )
        generate_with_config(model, tokenizer, prompt, "Top-K (k=50)", config)

        # 4. Top-P (nucleus) sampling
        config = GenerationConfig(
            max_new_tokens=15,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
        generate_with_config(model, tokenizer, prompt, "Top-P (p=0.9)", config)

        # 5. With repetition penalty
        config = GenerationConfig(
            max_new_tokens=15,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            repetition_penalty=1.2,
        )
        generate_with_config(model, tokenizer, prompt, "With Repetition Penalty", config)

        # 6. Beam search
        config = GenerationConfig(
            max_new_tokens=15,
            num_beams=3,
            length_penalty=1.0,
        )
        generate_with_config(model, tokenizer, prompt, "Beam Search (beams=3)", config)

    # Advanced example: Interactive generation
    print("\n" + "=" * 70)
    print("Interactive Generation Example")
    print("=" * 70)
    print("\nYou can also generate text interactively:")
    print("(This is just an example - run the code to try it live!)\n")

    example_prompt = "The transformer architecture"
    print(f"Your prompt: {example_prompt}")

    config = GenerationConfig(
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
    )

    input_ids = tokenizer.encode(example_prompt)
    input_tensor = torch.tensor([input_ids])

    generator = TextGenerator(model, config)
    output = generator.generate(input_tensor)
    generated_text = tokenizer.decode(output[0].tolist())

    print(f"Generated: {generated_text}")

    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print("\nTips for better generation:")
    print("  1. Train the model on your own data (see train_simple.py)")
    print("  2. Experiment with temperature (0.5-1.5)")
    print("  3. Try different sampling strategies")
    print("  4. Adjust repetition penalty to reduce repetition")
    print("  5. Use longer prompts for better context")
    print("\nFor more details, see the notebooks in notebooks/")
    print("=" * 70)


if __name__ == "__main__":
    main()

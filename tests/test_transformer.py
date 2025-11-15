"""Tests for transformer block."""

import pytest
import torch

from src.llm.attention import create_causal_mask
from src.llm.transformer import FeedForward, TransformerBlock


class TestFeedForward:
    """Tests for FeedForward class."""

    def test_initialization(self) -> None:
        """Test feedforward initialization."""
        ffn = FeedForward(d_model=384, d_ff=1536)
        assert isinstance(ffn.fc1, torch.nn.Linear)
        assert isinstance(ffn.fc2, torch.nn.Linear)
        assert ffn.fc1.in_features == 384
        assert ffn.fc1.out_features == 1536
        assert ffn.fc2.in_features == 1536
        assert ffn.fc2.out_features == 384

    def test_initialization_small_d_ff(self) -> None:
        """Test that d_ff < d_model raises a warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FeedForward(d_model=384, d_ff=128)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "d_ff" in str(w[0].message)
            assert "less than d_model" in str(w[0].message)

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        batch_size, seq_len, d_model = 2, 10, 384
        d_ff = 1536

        ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ffn(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_different_batch_sizes(self) -> None:
        """Test feedforward with different batch sizes."""
        d_model, d_ff = 384, 1536
        ffn = FeedForward(d_model=d_model, d_ff=d_ff)

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 10, d_model)
            output = ffn(x)
            assert output.shape == (batch_size, 10, d_model)

    def test_forward_different_seq_lengths(self) -> None:
        """Test feedforward with different sequence lengths."""
        batch_size, d_model, d_ff = 2, 384, 1536
        ffn = FeedForward(d_model=d_model, d_ff=d_ff)

        for seq_len in [1, 5, 20, 50]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = ffn(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_deterministic(self) -> None:
        """Test that forward pass is deterministic in eval mode."""
        batch_size, seq_len, d_model = 2, 10, 384

        ffn = FeedForward(d_model=d_model, d_ff=1536, dropout=0.0)
        ffn.eval()

        x = torch.randn(batch_size, seq_len, d_model)

        output1 = ffn(x)
        output2 = ffn(x)

        assert torch.allclose(output1, output2)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through feedforward."""
        batch_size, seq_len, d_model = 2, 5, 384

        ffn = FeedForward(d_model=d_model, d_ff=1536)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output = ffn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_no_bias_option(self) -> None:
        """Test feedforward without bias terms."""
        ffn = FeedForward(d_model=384, d_ff=1536, use_bias=False)

        assert ffn.fc1.bias is None
        assert ffn.fc2.bias is None

    def test_with_bias_option(self) -> None:
        """Test feedforward with bias terms."""
        ffn = FeedForward(d_model=384, d_ff=1536, use_bias=True)

        assert ffn.fc1.bias is not None
        assert ffn.fc2.bias is not None

    def test_invalid_input_shape(self) -> None:
        """Test feedforward with invalid input shape."""
        ffn = FeedForward(d_model=384, d_ff=1536)
        x_2d = torch.randn(10, 384)

        with pytest.raises(ValueError, match="Expected 3D input"):
            ffn(x_2d)


class TestTransformerBlock:
    """Tests for TransformerBlock class."""

    def test_initialization(self) -> None:
        """Test transformer block initialization."""
        block = TransformerBlock(d_model=384, n_heads=6, d_ff=1536)

        assert block.d_model == 384
        assert isinstance(block.ln1, torch.nn.LayerNorm)
        assert isinstance(block.ln2, torch.nn.LayerNorm)
        assert block.attn.d_model == 384
        assert block.attn.n_heads == 6
        assert block.ffn.fc1.in_features == 384
        assert block.ffn.fc1.out_features == 1536

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        batch_size, seq_len, d_model = 2, 10, 384
        n_heads, d_ff = 6, 1536

        block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = block(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights is None  # By default, attention weights not returned

    def test_forward_with_attention_return(self) -> None:
        """Test forward pass with attention weights return."""
        batch_size, seq_len, d_model = 2, 10, 384
        n_heads, d_ff = 6, 1536

        block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = block(x, return_attention=True)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights is not None
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)

    def test_forward_with_causal_mask(self) -> None:
        """Test forward pass with causal mask."""
        batch_size, seq_len, d_model = 2, 5, 384
        n_heads, d_ff = 6, 1536

        block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        mask = create_causal_mask(seq_len)

        output, attn_weights = block(x, mask=mask, return_attention=True)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights is not None

        # Check that attention weights respect causal mask
        for b in range(batch_size):
            for h in range(n_heads):
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        assert attn_weights[b, h, i, j].item() == pytest.approx(0.0, abs=1e-6)

    def test_residual_connections(self) -> None:
        """Test that residual connections are working."""
        batch_size, seq_len, d_model = 2, 5, 384

        # Create block with zero dropout to avoid randomness
        block = TransformerBlock(d_model=d_model, n_heads=6, d_ff=1536, dropout=0.0)
        block.eval()

        x = torch.randn(batch_size, seq_len, d_model)
        output, _ = block(x)

        # Output should not be identical to input (transformation happened)
        assert not torch.allclose(output, x)

        # But if we zero out the attention and FFN weights, residual should give us input back
        # This is more of a conceptual test - we'll just check the shape is preserved
        assert output.shape == x.shape

    def test_forward_different_batch_sizes(self) -> None:
        """Test transformer block with different batch sizes."""
        d_model, n_heads, d_ff = 384, 6, 1536
        block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 10, d_model)
            output, _ = block(x)
            assert output.shape == (batch_size, 10, d_model)

    def test_forward_different_seq_lengths(self) -> None:
        """Test transformer block with different sequence lengths."""
        batch_size, d_model, n_heads, d_ff = 2, 384, 6, 1536
        block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)

        for seq_len in [1, 5, 20, 50]:
            x = torch.randn(batch_size, seq_len, d_model)
            output, _ = block(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_deterministic(self) -> None:
        """Test that forward pass is deterministic in eval mode."""
        batch_size, seq_len, d_model = 2, 10, 384

        block = TransformerBlock(d_model=d_model, n_heads=6, d_ff=1536, dropout=0.0)
        block.eval()

        x = torch.randn(batch_size, seq_len, d_model)

        output1, _ = block(x)
        output2, _ = block(x)

        assert torch.allclose(output1, output2)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through transformer block."""
        batch_size, seq_len, d_model = 2, 5, 384

        block = TransformerBlock(d_model=d_model, n_heads=6, d_ff=1536)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output, _ = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_invalid_input_shape(self) -> None:
        """Test transformer block with invalid input shape."""
        block = TransformerBlock(d_model=384, n_heads=6, d_ff=1536)
        x_2d = torch.randn(10, 384)

        with pytest.raises(ValueError, match="Expected 3D input"):
            block(x_2d)

    def test_invalid_d_model(self) -> None:
        """Test transformer block with mismatched d_model."""
        block = TransformerBlock(d_model=384, n_heads=6, d_ff=1536)
        x = torch.randn(2, 10, 512)  # Wrong d_model

        with pytest.raises(ValueError, match="Expected d_model=384"):
            block(x)

    def test_layer_norm_placement(self) -> None:
        """Test that layer norm is applied before attention and FFN (pre-norm)."""
        batch_size, seq_len, d_model = 2, 5, 384

        block = TransformerBlock(d_model=d_model, n_heads=6, d_ff=1536)
        x = torch.randn(batch_size, seq_len, d_model)

        # We can't directly test the order, but we can verify layer norms exist
        assert isinstance(block.ln1, torch.nn.LayerNorm)
        assert isinstance(block.ln2, torch.nn.LayerNorm)

        # And that forward pass works correctly
        output, _ = block(x)
        assert output.shape == x.shape

    def test_stacked_blocks(self) -> None:
        """Test stacking multiple transformer blocks."""
        batch_size, seq_len, d_model = 2, 10, 384
        n_layers = 4

        blocks = torch.nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=6, d_ff=1536) for _ in range(n_layers)]
        )

        x = torch.randn(batch_size, seq_len, d_model)
        mask = create_causal_mask(seq_len)

        # Pass through all blocks
        for block in blocks:
            x, _ = block(x, mask=mask)

        assert x.shape == (batch_size, seq_len, d_model)

    def test_no_bias_option(self) -> None:
        """Test transformer block without bias terms."""
        block = TransformerBlock(d_model=384, n_heads=6, d_ff=1536, use_bias=False)

        # Check attention has no bias
        assert block.attn.q_proj.bias is None
        assert block.attn.k_proj.bias is None
        assert block.attn.v_proj.bias is None
        assert block.attn.out_proj.bias is None

        # Check FFN has no bias
        assert block.ffn.fc1.bias is None
        assert block.ffn.fc2.bias is None

    def test_with_bias_option(self) -> None:
        """Test transformer block with bias terms."""
        block = TransformerBlock(d_model=384, n_heads=6, d_ff=1536, use_bias=True)

        # Check attention has bias
        assert block.attn.q_proj.bias is not None
        assert block.attn.k_proj.bias is not None
        assert block.attn.v_proj.bias is not None
        assert block.attn.out_proj.bias is not None

        # Check FFN has bias
        assert block.ffn.fc1.bias is not None
        assert block.ffn.fc2.bias is not None

    def test_attention_weights_propagation(self) -> None:
        """Test that attention weights are correctly propagated."""
        batch_size, seq_len, d_model = 2, 5, 384
        n_heads = 6

        block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=1536, dropout=0.0)
        block.eval()  # Set to eval mode to disable dropout
        x = torch.randn(batch_size, seq_len, d_model)

        # Without return_attention
        _, attn_weights = block(x, return_attention=False)
        assert attn_weights is None

        # With return_attention
        _, attn_weights = block(x, return_attention=True)
        assert attn_weights is not None
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)

        # Check attention weights are valid probabilities
        # Note: This check only works in eval mode with dropout=0.0
        # During training, dropout breaks the sum-to-1 property
        assert (attn_weights >= 0).all()
        assert (attn_weights <= 1).all()
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)))

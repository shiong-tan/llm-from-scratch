"""Tests for attention mechanism."""

import pytest
import torch

from src.llm.attention import (
    MultiHeadAttention,
    create_causal_mask,
    create_padding_mask,
)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention class."""

    def test_initialization(self) -> None:
        """Test attention initialization."""
        attn = MultiHeadAttention(d_model=384, n_heads=6)
        assert attn.d_model == 384
        assert attn.n_heads == 6
        assert attn.head_dim == 64
        assert attn.scale == pytest.approx(1.0 / (64**0.5))

    def test_initialization_invalid_heads(self) -> None:
        """Test that d_model must be divisible by n_heads."""
        with pytest.raises(ValueError, match="d_model.*must be divisible by n_heads"):
            MultiHeadAttention(d_model=385, n_heads=6)

    def test_forward_shape(self) -> None:
        """Test forward pass output shape."""
        batch_size, seq_len, d_model = 2, 10, 384
        n_heads = 6

        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = attn(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights is None  # By default, attention weights not returned

    def test_forward_with_attention_return(self) -> None:
        """Test forward pass with attention weights return."""
        batch_size, seq_len, d_model = 2, 10, 384
        n_heads = 6

        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = attn(x, return_attention=True)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights is not None
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)

        # Attention weights should sum to 1 along last dimension
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)))

    def test_forward_with_causal_mask(self) -> None:
        """Test forward pass with causal mask."""
        batch_size, seq_len, d_model = 2, 5, 384
        n_heads = 6

        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        mask = create_causal_mask(seq_len)

        output, attn_weights = attn(x, mask=mask, return_attention=True)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights is not None

        # Check that attention weights are zero for masked positions
        # Upper triangle should be zero (cannot attend to future)
        for b in range(batch_size):
            for h in range(n_heads):
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        assert attn_weights[b, h, i, j].item() == pytest.approx(0.0, abs=1e-6)

    def test_forward_different_batch_sizes(self) -> None:
        """Test attention with different batch sizes."""
        d_model, n_heads = 384, 6
        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 10, d_model)
            output, _ = attn(x)
            assert output.shape == (batch_size, 10, d_model)

    def test_forward_different_seq_lengths(self) -> None:
        """Test attention with different sequence lengths."""
        batch_size, d_model, n_heads = 2, 384, 6
        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

        for seq_len in [1, 5, 20, 50]:
            x = torch.randn(batch_size, seq_len, d_model)
            output, _ = attn(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_deterministic(self) -> None:
        """Test that forward pass is deterministic in eval mode."""
        batch_size, seq_len, d_model = 2, 10, 384
        n_heads = 6

        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
        attn.eval()  # Set to eval mode to disable dropout

        x = torch.randn(batch_size, seq_len, d_model)

        output1, _ = attn(x)
        output2, _ = attn(x)

        assert torch.allclose(output1, output2)

    def test_split_heads(self) -> None:
        """Test head splitting operation."""
        batch_size, seq_len, d_model = 2, 10, 384
        n_heads = 6

        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        split = attn._split_heads(x, batch_size)

        assert split.shape == (batch_size, n_heads, seq_len, d_model // n_heads)

    def test_merge_heads(self) -> None:
        """Test head merging operation."""
        batch_size, seq_len, d_model = 2, 10, 384
        n_heads = 6
        head_dim = d_model // n_heads

        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, n_heads, seq_len, head_dim)

        merged = attn._merge_heads(x, batch_size)

        assert merged.shape == (batch_size, seq_len, d_model)

    def test_split_merge_roundtrip(self) -> None:
        """Test that split and merge operations are inverses."""
        batch_size, seq_len, d_model = 2, 10, 384
        n_heads = 6

        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        split = attn._split_heads(x, batch_size)
        merged = attn._merge_heads(split, batch_size)

        assert merged.shape == x.shape
        assert torch.allclose(merged, x)

    def test_mask_broadcasting_2d(self) -> None:
        """Test that 2D mask is properly broadcast."""
        batch_size, seq_len, d_model = 2, 5, 384
        n_heads = 6

        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        mask_2d = create_causal_mask(seq_len)  # Shape: (seq_len, seq_len)

        output1, attn_weights1 = attn(x, mask=mask_2d, return_attention=True)

        # Should work without errors and produce valid output
        assert output1.shape == (batch_size, seq_len, d_model)
        assert attn_weights1 is not None

    def test_mask_broadcasting_3d(self) -> None:
        """Test that 3D mask is properly broadcast."""
        batch_size, seq_len, d_model = 2, 5, 384
        n_heads = 6

        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        mask_3d = create_causal_mask(seq_len).unsqueeze(0).repeat(batch_size, 1, 1)

        output, attn_weights = attn(x, mask=mask_3d, return_attention=True)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights is not None

    def test_no_bias_option(self) -> None:
        """Test attention without bias terms."""
        attn = MultiHeadAttention(d_model=384, n_heads=6, use_bias=False)

        # Check that bias is None for all projections
        assert attn.q_proj.bias is None
        assert attn.k_proj.bias is None
        assert attn.v_proj.bias is None
        assert attn.out_proj.bias is None

    def test_with_bias_option(self) -> None:
        """Test attention with bias terms."""
        attn = MultiHeadAttention(d_model=384, n_heads=6, use_bias=True)

        # Check that bias exists for all projections
        assert attn.q_proj.bias is not None
        assert attn.k_proj.bias is not None
        assert attn.v_proj.bias is not None
        assert attn.out_proj.bias is not None

    def test_attention_weights_range(self) -> None:
        """Test that attention weights are in valid range [0, 1]."""
        batch_size, seq_len, d_model = 2, 10, 384
        n_heads = 6

        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        _, attn_weights = attn(x, return_attention=True)

        assert attn_weights is not None
        assert (attn_weights >= 0).all()
        assert (attn_weights <= 1).all()

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through attention."""
        batch_size, seq_len, d_model = 2, 5, 384
        n_heads = 6

        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output, _ = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestCausalMask:
    """Tests for causal mask creation."""

    def test_causal_mask_shape(self) -> None:
        """Test causal mask has correct shape."""
        seq_len = 10
        mask = create_causal_mask(seq_len)

        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool

    def test_causal_mask_pattern(self) -> None:
        """Test causal mask has correct pattern."""
        seq_len = 5
        mask = create_causal_mask(seq_len)

        # Expected pattern:
        # [[0, 1, 1, 1, 1],
        #  [0, 0, 1, 1, 1],
        #  [0, 0, 0, 1, 1],
        #  [0, 0, 0, 0, 1],
        #  [0, 0, 0, 0, 0]]

        expected = torch.tensor(
            [
                [False, True, True, True, True],
                [False, False, True, True, True],
                [False, False, False, True, True],
                [False, False, False, False, True],
                [False, False, False, False, False],
            ]
        )

        assert torch.equal(mask, expected)

    def test_causal_mask_diagonal(self) -> None:
        """Test that diagonal and below are not masked."""
        seq_len = 10
        mask = create_causal_mask(seq_len)

        # Lower triangle (including diagonal) should be False
        for i in range(seq_len):
            for j in range(i + 1):
                assert not mask[i, j].item()

        # Upper triangle should be True
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask[i, j].item()

    def test_causal_mask_different_lengths(self) -> None:
        """Test causal mask with different sequence lengths."""
        for seq_len in [1, 5, 10, 50, 100]:
            mask = create_causal_mask(seq_len)
            assert mask.shape == (seq_len, seq_len)

    def test_causal_mask_device(self) -> None:
        """Test causal mask creation on different devices."""
        seq_len = 10

        # CPU
        mask_cpu = create_causal_mask(seq_len, device=torch.device("cpu"))
        assert mask_cpu.device.type == "cpu"

        # Test GPU if available
        if torch.cuda.is_available():
            mask_cuda = create_causal_mask(seq_len, device=torch.device("cuda"))
            assert mask_cuda.device.type == "cuda"


class TestPaddingMask:
    """Tests for padding mask creation."""

    def test_padding_mask_shape(self) -> None:
        """Test padding mask has correct shape."""
        batch_size, seq_len = 2, 10
        pad_token_id = 0

        seq = torch.randint(1, 100, (batch_size, seq_len))
        seq[:, -3:] = pad_token_id  # Add padding at the end

        mask = create_padding_mask(seq, pad_token_id)

        assert mask.shape == (batch_size, 1, 1, seq_len)
        assert mask.dtype == torch.bool

    def test_padding_mask_pattern(self) -> None:
        """Test padding mask identifies padding tokens correctly."""
        pad_token_id = 0
        seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])

        mask = create_padding_mask(seq, pad_token_id)

        # Expected: padding positions (0s) should be True
        expected = torch.tensor([[[[False, False, False, True, True]]], [[[False, False, True, True, True]]]])

        assert torch.equal(mask, expected)

    def test_padding_mask_no_padding(self) -> None:
        """Test padding mask when there's no padding."""
        pad_token_id = 0
        seq = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        mask = create_padding_mask(seq, pad_token_id)

        # No padding, so all False
        assert not mask.any()

    def test_padding_mask_all_padding(self) -> None:
        """Test padding mask when entire sequence is padding."""
        pad_token_id = 0
        seq = torch.zeros(2, 10, dtype=torch.long)

        mask = create_padding_mask(seq, pad_token_id)

        # All padding, so all True
        assert mask.all()

    def test_padding_mask_different_pad_ids(self) -> None:
        """Test padding mask with different padding token IDs."""
        seq = torch.tensor([[1, 2, 3, 99, 99], [1, 2, 99, 99, 99]])

        mask = create_padding_mask(seq, pad_token_id=99)

        expected = torch.tensor([[[[False, False, False, True, True]]], [[[False, False, True, True, True]]]])

        assert torch.equal(mask, expected)

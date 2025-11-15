"""Tests for tokenizer module."""

import pytest

from src.llm.tokenizer import Tokenizer


class TestTokenizer:
    """Tests for Tokenizer class."""

    def test_initialization(self) -> None:
        """Test tokenizer initialization."""
        tokenizer = Tokenizer()
        assert tokenizer.encoding_name == "gpt2"
        assert tokenizer.vocab_size > 0

    def test_invalid_encoding_name(self) -> None:
        """Test initialization with invalid encoding name."""
        with pytest.raises(ValueError, match="Failed to load encoding"):
            Tokenizer(encoding_name="invalid_encoding")

    def test_vocab_size(self) -> None:
        """Test vocabulary size property."""
        tokenizer = Tokenizer()
        # GPT-2 has 50,257 tokens
        assert tokenizer.vocab_size == 50257

    def test_special_tokens(self) -> None:
        """Test special token IDs."""
        tokenizer = Tokenizer()
        # GPT-2 uses token 50256 for <|endoftext|>
        assert tokenizer.eos_token_id == 50256
        # GPT-2 doesn't have a separate BOS, so it uses EOS
        assert tokenizer.bos_token_id == tokenizer.eos_token_id

    def test_encode_simple_text(self) -> None:
        """Test encoding simple text."""
        tokenizer = Tokenizer()
        text = "Hello, world!"
        token_ids = tokenizer.encode(text)

        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert all(isinstance(tid, int) for tid in token_ids)
        assert all(0 <= tid < tokenizer.vocab_size for tid in token_ids)

    def test_encode_with_special_tokens(self) -> None:
        """Test encoding with special tokens."""
        tokenizer = Tokenizer()
        text = "Hello!"

        ids_without = tokenizer.encode(text, add_special_tokens=False)
        ids_with = tokenizer.encode(text, add_special_tokens=True)

        assert len(ids_with) == len(ids_without) + 1
        assert ids_with[0] == tokenizer.bos_token_id
        assert ids_with[1:] == ids_without

    def test_encode_empty_string(self) -> None:
        """Test encoding empty string."""
        tokenizer = Tokenizer()
        token_ids = tokenizer.encode("")
        assert token_ids == []

    def test_encode_invalid_input(self) -> None:
        """Test encoding with invalid input type."""
        tokenizer = Tokenizer()
        with pytest.raises(TypeError, match="Expected str"):
            tokenizer.encode(123)  # type: ignore

    def test_decode_simple(self) -> None:
        """Test decoding token IDs."""
        tokenizer = Tokenizer()
        original_text = "Hello, world!"
        token_ids = tokenizer.encode(original_text)
        decoded_text = tokenizer.decode(token_ids)

        assert isinstance(decoded_text, str)
        assert decoded_text == original_text

    def test_encode_decode_roundtrip(self) -> None:
        """Test that encode->decode is a valid roundtrip."""
        tokenizer = Tokenizer()
        texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating!",
            "🚀 Python 3.11",
            "",
        ]

        for text in texts:
            token_ids = tokenizer.encode(text)
            decoded = tokenizer.decode(token_ids)
            assert decoded == text, f"Roundtrip failed for: {text}"

    def test_decode_batch(self) -> None:
        """Test batch decoding."""
        tokenizer = Tokenizer()
        texts = ["Hello!", "World!", "Test"]
        token_ids_batch = [tokenizer.encode(text) for text in texts]

        decoded_batch = tokenizer.decode_batch(token_ids_batch)

        assert isinstance(decoded_batch, list)
        assert len(decoded_batch) == len(texts)
        assert decoded_batch == texts

    def test_decode_invalid_input(self) -> None:
        """Test decoding with invalid input."""
        tokenizer = Tokenizer()
        with pytest.raises(TypeError, match="Expected list"):
            tokenizer.decode("not a list")  # type: ignore

    def test_decode_invalid_token_ids(self) -> None:
        """Test decoding with out-of-range token IDs."""
        tokenizer = Tokenizer()
        # This should raise an error since the token ID is out of range
        with pytest.raises(ValueError, match="Failed to decode"):
            tokenizer.decode([999999])

    def test_decode_mixed_types(self) -> None:
        """Test decoding with non-integer elements."""
        tokenizer = Tokenizer()
        with pytest.raises(TypeError, match="All token IDs must be integers"):
            tokenizer.decode([1, 2, "invalid", 4])  # type: ignore

    def test_decode_empty_list(self) -> None:
        """Test decoding empty token list."""
        tokenizer = Tokenizer()
        result = tokenizer.decode([])
        assert result == ""

    def test_encode_batch(self) -> None:
        """Test batch encoding."""
        tokenizer = Tokenizer()
        texts = ["Hello!", "World!", "Test"]
        token_ids_batch = tokenizer.encode_batch(texts)

        assert isinstance(token_ids_batch, list)
        assert len(token_ids_batch) == len(texts)
        assert all(isinstance(ids, list) for ids in token_ids_batch)

        # Each should match individual encoding
        for text, batch_ids in zip(texts, token_ids_batch):
            individual_ids = tokenizer.encode(text)
            assert batch_ids == individual_ids

    def test_encode_batch_with_special_tokens(self) -> None:
        """Test batch encoding with special tokens."""
        tokenizer = Tokenizer()
        texts = ["Hello!", "World!"]

        batch_without = tokenizer.encode_batch(texts, add_special_tokens=False)
        batch_with = tokenizer.encode_batch(texts, add_special_tokens=True)

        for ids_without, ids_with in zip(batch_without, batch_with):
            assert len(ids_with) == len(ids_without) + 1
            assert ids_with[0] == tokenizer.bos_token_id

    def test_encode_batch_invalid_input(self) -> None:
        """Test batch encoding with invalid input."""
        tokenizer = Tokenizer()
        with pytest.raises(TypeError, match="Expected list"):
            tokenizer.encode_batch("not a list")  # type: ignore

    def test_encode_batch_mixed_types(self) -> None:
        """Test batch encoding with mixed types in list."""
        tokenizer = Tokenizer()
        with pytest.raises(TypeError, match="All elements.*must be strings"):
            tokenizer.encode_batch(["Valid", 123, "String"])  # type: ignore

    def test_encode_batch_empty_list(self) -> None:
        """Test batch encoding empty list."""
        tokenizer = Tokenizer()
        result = tokenizer.encode_batch([])
        assert result == []

    def test_encode_batch_with_empty_strings(self) -> None:
        """Test batch encoding containing empty strings."""
        tokenizer = Tokenizer()
        texts = ["Hello", "", "World", ""]
        result = tokenizer.encode_batch(texts)
        assert len(result) == 4
        assert result[1] == []
        assert result[3] == []

    def test_repr(self) -> None:
        """Test string representation."""
        tokenizer = Tokenizer()
        repr_str = repr(tokenizer)
        assert "Tokenizer" in repr_str
        assert "gpt2" in repr_str
        assert str(tokenizer.vocab_size) in repr_str

    def test_len(self) -> None:
        """Test __len__ method."""
        tokenizer = Tokenizer()
        assert len(tokenizer) == tokenizer.vocab_size

    def test_different_encodings(self) -> None:
        """Test using different encoding schemes."""
        # Test with different available encodings
        for encoding in ["gpt2", "r50k_base"]:
            tokenizer = Tokenizer(encoding_name=encoding)
            text = "Hello, world!"
            token_ids = tokenizer.encode(text)
            decoded = tokenizer.decode(token_ids)
            assert decoded == text

    def test_unicode_handling(self) -> None:
        """Test encoding and decoding with unicode characters."""
        tokenizer = Tokenizer()
        texts = [
            "Hello, 世界!",
            "Привет, мир!",
            "مرحبا بالعالم",
            "🎉🎊🎈",
        ]

        for text in texts:
            token_ids = tokenizer.encode(text)
            decoded = tokenizer.decode(token_ids)
            assert decoded == text, f"Unicode roundtrip failed for: {text}"

    def test_long_text_encoding(self) -> None:
        """Test encoding longer text."""
        tokenizer = Tokenizer()
        # Create a long text
        long_text = " ".join(["This is a test sentence."] * 100)
        token_ids = tokenizer.encode(long_text)

        assert len(token_ids) > 0
        decoded = tokenizer.decode(token_ids)
        assert decoded == long_text

    def test_encode_whitespace_only(self) -> None:
        """Test encoding whitespace-only strings."""
        tokenizer = Tokenizer()
        whitespace_texts = ["   ", "\t\t", "\n\n", "\r\n"]

        for text in whitespace_texts:
            token_ids = tokenizer.encode(text)
            decoded = tokenizer.decode(token_ids)
            assert decoded == text, f"Whitespace roundtrip failed for: {repr(text)}"

    def test_encode_special_characters(self) -> None:
        """Test encoding strings with special characters."""
        tokenizer = Tokenizer()
        special_texts = [
            "Line1\nLine2",
            "Tab\there",
            'Quote: "Hello"',
            "Path: /usr/bin",
            "Math: 2+2=4",
        ]

        for text in special_texts:
            token_ids = tokenizer.encode(text)
            decoded = tokenizer.decode(token_ids)
            assert decoded == text

    def test_encode_very_long_text(self) -> None:
        """Test encoding text near context window limits."""
        tokenizer = Tokenizer()
        # GPT-2 max is 1024 tokens, test something reasonable
        very_long = "A" * 5000
        token_ids = tokenizer.encode(very_long)
        decoded = tokenizer.decode(token_ids)
        assert decoded == very_long

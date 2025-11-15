"""Tokenizer for text processing using BPE (Byte Pair Encoding)."""

from typing import List

import tiktoken


class Tokenizer:
    """Tokenizer wrapper using tiktoken for BPE encoding.

    This uses the same tokenizer as GPT-2/GPT-3, which employs Byte Pair Encoding
    (BPE) to convert text into a sequence of token IDs. BPE works by iteratively
    merging the most frequent pairs of bytes/characters into single tokens, building
    a vocabulary of subword units. This allows the model to handle unknown words by
    breaking them into known subword pieces.

    Args:
        encoding_name: Name of the encoding to use. Default is 'gpt2'.

    Attributes:
        vocab_size: Size of the vocabulary (50,257 for GPT-2).
        eos_token_id: End-of-sequence token ID.
        bos_token_id: Beginning-of-sequence token ID (GPT-2 doesn't have one,
                      so we use eos_token_id).
    """

    def __init__(self, encoding_name: str = "gpt2") -> None:
        """Initialize the tokenizer.

        Args:
            encoding_name: Name of the tiktoken encoding to use

        Raises:
            ValueError: If the encoding name is not valid
        """
        try:
            self._encoder = tiktoken.get_encoding(encoding_name)
        except (KeyError, ValueError, LookupError) as e:
            raise ValueError(
                f"Failed to load encoding '{encoding_name}'. "
                f"Available encodings: {tiktoken.list_encoding_names()}"
            ) from e

        self.encoding_name = encoding_name

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size.

        For GPT-2, this is 50,257 tokens (50,000 base tokens + 256 byte tokens + 1 EOT).
        This determines the valid range for token IDs: [0, vocab_size).

        Returns:
            Integer vocabulary size. Always positive.
        """
        return self._encoder.n_vocab

    @property
    def eos_token_id(self) -> int:
        """Return the end-of-sequence token ID.

        For GPT-2, this is the <|endoftext|> token (ID 50256).

        Returns:
            Token ID for end-of-sequence marker.
        """
        # GPT-2 uses <|endoftext|> token (ID 50256)
        return self._encoder.eot_token

    @property
    def bos_token_id(self) -> int:
        """Return the beginning-of-sequence token ID.

        Note: GPT-2 doesn't have a separate BOS token, so we use EOS.

        Returns:
            Token ID for beginning-of-sequence marker (same as EOS for GPT-2).
        """
        return self.eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to a list of token IDs using Byte Pair Encoding (BPE).

        BPE works by iteratively merging the most frequent pairs of bytes/characters
        into single tokens, building a vocabulary of subword units. This allows the
        model to handle unknown words by breaking them into known subword pieces.

        Args:
            text: Input text to encode. Can be empty string.
            add_special_tokens: Whether to add BOS (beginning-of-sequence) token
                at the start. Useful for autoregressive generation where the model
                needs to know where a sequence begins.

        Returns:
            List of token IDs, where each ID is an integer in range [0, vocab_size).
            Empty string returns empty list.

        Raises:
            TypeError: If text is not a string.

        Examples:
            >>> tokenizer = Tokenizer()
            >>> ids = tokenizer.encode("Hello, world!")
            >>> isinstance(ids, list)
            True
            >>> all(0 <= id < tokenizer.vocab_size for id in ids)
            True
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")

        token_ids = self._encoder.encode(text)

        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode a single sequence of token IDs back to text.

        Args:
            token_ids: List of token IDs to decode. Can be empty list.

        Returns:
            Decoded text string. Empty list returns empty string.

        Raises:
            TypeError: If token_ids is not a list or contains non-integers.
            ValueError: If decoding fails (e.g., invalid token IDs).

        Examples:
            >>> tokenizer = Tokenizer()
            >>> ids = tokenizer.encode("Hello!")
            >>> text = tokenizer.decode(ids)
            >>> isinstance(text, str)
            True
        """
        if not isinstance(token_ids, list):
            raise TypeError(f"Expected list of ints, got {type(token_ids)}")

        # Validate all elements are integers
        if token_ids and not all(isinstance(tid, int) for tid in token_ids):
            raise TypeError("All token IDs must be integers")

        try:
            return self._encoder.decode(token_ids)
        except (ValueError, IndexError, KeyError) as e:
            raise ValueError(f"Failed to decode token IDs: {e}") from e

    def decode_batch(self, token_ids_batch: List[List[int]]) -> List[str]:
        """Decode a batch of token ID sequences to texts.

        Args:
            token_ids_batch: List of token ID lists to decode.

        Returns:
            List of decoded text strings.

        Raises:
            TypeError: If token_ids_batch is not a list of lists or contains non-integers.
            ValueError: If decoding fails for any sequence.

        Examples:
            >>> tokenizer = Tokenizer()
            >>> batch = [[1, 2, 3], [4, 5]]
            >>> texts = tokenizer.decode_batch(batch)
            >>> isinstance(texts, list)
            True
        """
        if not isinstance(token_ids_batch, list):
            raise TypeError(f"Expected list of lists, got {type(token_ids_batch)}")

        return [self.decode(ids) for ids in token_ids_batch]

    def encode_batch(self, texts: List[str], add_special_tokens: bool = False) -> List[List[int]]:
        """Encode a batch of texts.

        Args:
            texts: List of text strings to encode. Can be empty list.
            add_special_tokens: Whether to add BOS token to each sequence.

        Returns:
            List of token ID lists. Empty input returns empty list.

        Raises:
            TypeError: If texts is not a list or contains non-strings.

        Examples:
            >>> tokenizer = Tokenizer()
            >>> batch = tokenizer.encode_batch(["Hello!", "World!"])
            >>> len(batch)
            2
        """
        if not isinstance(texts, list):
            raise TypeError(f"Expected list of strings, got {type(texts)}")

        # Validate all elements are strings
        if texts and not all(isinstance(text, str) for text in texts):
            raise TypeError("All elements in texts must be strings")

        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def __repr__(self) -> str:
        """Return string representation with key tokenizer information."""
        return (
            f"Tokenizer(encoding_name='{self.encoding_name}', "
            f"vocab_size={self.vocab_size}, "
            f"eos_token_id={self.eos_token_id})"
        )

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

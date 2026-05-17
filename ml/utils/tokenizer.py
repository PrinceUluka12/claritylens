# ml/utils/tokenizer.py
# ============================================================
# Wrapper around the trained WordPiece tokenizer.
# Every phase that needs tokenization imports from here.
#
# Why a wrapper class instead of loading directly?
# - Singleton pattern: tokenizer loads once, reused everywhere
# - Adds padding and truncation logic in one place
# - Hides the tokenizers library API behind a simple interface
# - Easy to swap tokenizer without touching calling code
# ============================================================

import json
from pathlib import Path
from loguru import logger
from tokenizers import Tokenizer


class ContractTokenizer:
    """
    Wraps the trained WordPiece tokenizer with padding,
    truncation, and batch encoding helpers.
    """

    def __init__(
        self,
        tokenizer_path: str | Path = "./ml/models/embedder/tokenizer.json",
        max_length:     int = 256,
    ) -> None:

        tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                f"Run ml/training/train_tokenizer.py first."
            )

        self.max_length = max_length
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # CPU NOTE: enable padding and truncation at load time
        # so every encoded sequence is exactly max_length tokens.
        # Fixed-length sequences allow batching without per-batch
        # padding logic, which speeds up CPU inference.
        self._tokenizer.enable_padding(
            pad_id    = self._tokenizer.token_to_id("[PAD]"),
            pad_token = "[PAD]",
            length    = max_length,
        )
        self._tokenizer.enable_truncation(max_length=max_length)

        vocab_size = self._tokenizer.get_vocab_size()
        logger.info(
            f"ContractTokenizer loaded — vocab={vocab_size:,}, "
            f"max_length={max_length}"
        )

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.token_to_id("[PAD]")

    @property
    def cls_token_id(self) -> int:
        return self._tokenizer.token_to_id("[CLS]")

    @property
    def sep_token_id(self) -> int:
        return self._tokenizer.token_to_id("[SEP]")

    def encode(self, text: str) -> dict:
        """
        Encodes a single text string.

        Returns a dict with:
          input_ids:      list of token IDs, padded to max_length
          attention_mask: 1 for real tokens, 0 for padding
        """
        encoded = self._tokenizer.encode(text)
        return {
            "input_ids":      encoded.ids,
            "attention_mask": encoded.attention_mask,
        }

    def encode_batch(self, texts: list[str]) -> dict:
        """
        Encodes a list of texts in one call.

        CPU NOTE: batch encoding is significantly faster than
        calling encode() in a loop — the Rust tokenizer processes
        all texts in parallel using multiple threads.

        Returns a dict with:
          input_ids:      list[list[int]], shape [batch, max_length]
          attention_mask: list[list[int]], shape [batch, max_length]
        """
        encoded_batch = self._tokenizer.encode_batch(texts)
        return {
            "input_ids": [
                e.ids for e in encoded_batch
            ],
            "attention_mask": [
                e.attention_mask for e in encoded_batch
            ],
        }

    def decode(self, token_ids: list[int]) -> str:
        """Converts token IDs back to readable text."""
        return self._tokenizer.decode(token_ids)

    def token_count(self, text: str) -> int:
        """
        Returns the number of tokens for a text string.
        Useful for verifying chunks stay under max_length.
        """
        # Disable padding/truncation temporarily for counting
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()
        count = len(self._tokenizer.encode(text).ids)
        # Re-enable
        self._tokenizer.enable_padding(
            pad_id    = self.pad_token_id,
            pad_token = "[PAD]",
            length    = self.max_length,
        )
        self._tokenizer.enable_truncation(max_length=self.max_length)
        return count


# Singleton instance — import this object directly in other modules
# instead of instantiating ContractTokenizer() each time
_tokenizer_instance: ContractTokenizer | None = None


def get_tokenizer(
    tokenizer_path: str = "./ml/models/embedder/tokenizer.json",
    max_length:     int = 256,
) -> ContractTokenizer:
    """
    Returns the singleton tokenizer instance.
    Creates it on first call, reuses it on subsequent calls.
    """
    global _tokenizer_instance
    if _tokenizer_instance is None:
        _tokenizer_instance = ContractTokenizer(tokenizer_path, max_length)
    return _tokenizer_instance
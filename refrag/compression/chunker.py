"""
Text chunker for REfrag compression.

This module provides functionality to split text into fixed-size token chunks
as described in the REfrag paper (default: 16 tokens per chunk).
"""

import logging
from typing import List

from transformers import PreTrainedTokenizer

from refrag.config import Config
from refrag.utils import setup_logging

logger = setup_logging()


class TextChunker:
    """
    Text chunker for splitting documents into fixed-size token chunks.

    This class splits text into chunks of a fixed number of tokens,
    which is a key component of the REfrag compression phase.

    Attributes:
        tokenizer: Tokenizer for counting tokens
        chunk_size: Number of tokens per chunk
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, chunk_size: int):
        """
        Initialize the text chunker.

        Args:
            tokenizer: Tokenizer to use for chunking
            chunk_size: Number of tokens per chunk
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

        logger.debug(f"TextChunker initialized with chunk_size={chunk_size}")

    def chunk(self, text: str) -> List[str]:
        """
        Split text into fixed-size token chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks, each containing approximately chunk_size tokens
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) == 0:
            return []

        # Split into chunks
        chunk_token_lists = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_token_lists.append(chunk_tokens)

        # Decode chunks back to text
        chunks = []
        for chunk_tokens in chunk_token_lists:
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            # Only add non-empty chunks
            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks

    def chunk_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Chunk a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of lists, where each inner list contains chunks for one text
        """
        return [self.chunk(text) for text in texts]

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def estimate_chunks(self, text: str) -> int:
        """
        Estimate the number of chunks for a given text.

        Args:
            text: Input text

        Returns:
            Estimated number of chunks
        """
        num_tokens = self.count_tokens(text)
        return (num_tokens + self.chunk_size - 1) // self.chunk_size  # Ceiling division

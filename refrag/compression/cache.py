"""
Embedding cache for REfrag.

This module provides caching functionality to avoid re-encoding the same chunks.
"""

import hashlib
import logging
from typing import Dict, Optional

import numpy as np

from refrag.utils import setup_logging

logger = setup_logging()


class EmbeddingCache:
    """
    Simple in-memory cache for chunk embeddings.

    This class caches chunk embeddings to avoid re-encoding the same text chunks,
    which can significantly improve performance during repeated queries.

    Attributes:
        cache: Dictionary mapping text hashes to embeddings
        hits: Number of cache hits
        misses: Number of cache misses
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize the embedding cache.

        Args:
            max_size: Maximum number of entries in the cache
        """
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

        logger.debug(f"EmbeddingCache initialized with max_size={max_size}")

    def _hash_text(self, text: str) -> str:
        """
        Generate a hash for text.

        Args:
            text: Input text

        Returns:
            MD5 hash of the text
        """
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.

        Args:
            text: Text chunk

        Returns:
            Cached embedding or None if not found
        """
        key = self._hash_text(text)

        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.

        Args:
            text: Text chunk
            embedding: Embedding array
        """
        # Check if cache is full
        if len(self.cache) >= self.max_size:
            # Simple FIFO eviction: remove first item
            first_key = next(iter(self.cache))
            del self.cache[first_key]
            logger.debug(f"Cache full, evicted entry. Size: {len(self.cache)}")

        key = self._hash_text(text)
        self.cache[key] = embedding

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.debug("Cache cleared")

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
        }

    def __len__(self) -> int:
        """Get the number of cached items."""
        return len(self.cache)

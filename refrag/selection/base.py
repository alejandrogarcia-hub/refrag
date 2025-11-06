"""
Base class for selection policies.

This module defines the abstract base class for chunk selection policies
in REfrag. Selection policies determine which chunks should be expanded
(use full tokens) vs compressed (use embeddings only).
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class SelectionPolicy(ABC):
    """
    Abstract base class for selection policies.

    A selection policy decides which chunks are important enough to expand
    to their full token representation, vs which chunks can remain compressed
    as single embeddings.

    The paper uses RL-based policies, but we implement heuristic alternatives
    for simplicity.
    """

    def __init__(self, expansion_fraction: float = 0.25):
        """
        Initialize the selection policy.

        Args:
            expansion_fraction: Fraction of chunks to expand (0.0 to 1.0)
        """
        self.expansion_fraction = expansion_fraction

    @abstractmethod
    def select(
        self,
        chunks: List[str],
        query: str,
        chunk_embeddings: np.ndarray,
        query_embedding: np.ndarray
    ) -> List[int]:
        """
        Select which chunks to expand.

        Args:
            chunks: List of text chunks
            query: Query text
            chunk_embeddings: Embeddings of chunks, shape (n_chunks, dim)
            query_embedding: Embedding of query, shape (dim,)

        Returns:
            List of indices of chunks to expand
        """
        pass

    def _get_top_k(self, scores: np.ndarray, expansion_fraction: float | None = None) -> List[int]:
        """
        Get top-k indices based on scores and expansion fraction.

        Args:
            scores: Array of scores for each chunk
            expansion_fraction: Optional override for expansion fraction
                              If None, uses self.expansion_fraction

        Returns:
            List of top-k indices
        """
        n_chunks = len(scores)
        # Use provided fraction or default to instance fraction
        fraction = expansion_fraction if expansion_fraction is not None else self.expansion_fraction
        k = max(1, int(n_chunks * fraction))

        # Get indices of top-k scores
        top_k_indices = np.argsort(scores)[-k:][::-1]

        return top_k_indices.tolist()

    def select_with_dynamic_fraction(
        self,
        chunks: List[str],
        query: str,
        chunk_embeddings: np.ndarray,
        query_embedding: np.ndarray,
        dynamic_fraction: float
    ) -> List[int]:
        """
        Select chunks with a dynamically determined expansion fraction.

        This method temporarily overrides the expansion fraction for adaptive
        selection based on query complexity.

        Args:
            chunks: List of text chunks
            query: Query text
            chunk_embeddings: Embeddings of chunks, shape (n_chunks, dim)
            query_embedding: Embedding of query, shape (dim,)
            dynamic_fraction: Dynamically computed expansion fraction (0.0 to 1.0)

        Returns:
            List of indices of chunks to expand
        """
        # Temporarily store the original fraction
        original_fraction = self.expansion_fraction

        try:
            # Override with dynamic fraction
            self.expansion_fraction = dynamic_fraction

            # Call the regular select method
            result = self.select(chunks, query, chunk_embeddings, query_embedding)

            return result
        finally:
            # Restore original fraction
            self.expansion_fraction = original_fraction

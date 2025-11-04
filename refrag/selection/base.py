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

    def _get_top_k(self, scores: np.ndarray) -> List[int]:
        """
        Get top-k indices based on scores and expansion fraction.

        Args:
            scores: Array of scores for each chunk

        Returns:
            List of top-k indices
        """
        n_chunks = len(scores)
        k = max(1, int(n_chunks * self.expansion_fraction))

        # Get indices of top-k scores
        top_k_indices = np.argsort(scores)[-k:][::-1]

        return top_k_indices.tolist()

"""
Query complexity estimator for dynamic expansion fraction.

This module estimates query complexity to adaptively adjust how many chunks
should be expanded based on the nature of the query and retrieved content.
"""

import numpy as np
from typing import List


class QueryComplexityEstimator:
    """
    Estimates query complexity to inform dynamic expansion decisions.

    The estimator considers multiple factors:
    - Query length: Longer queries may indicate more complex information needs
    - Number of chunks: More chunks suggest broader topic coverage
    - Similarity variance: High variance means some chunks are much more relevant

    Returns a complexity score from 0.0 (simple) to 1.0 (complex).
    """

    def __init__(
        self,
        query_length_weight: float = 0.3,
        chunk_count_weight: float = 0.3,
        variance_weight: float = 0.4
    ):
        """
        Initialize the complexity estimator.

        Args:
            query_length_weight: Weight for query length factor (0.0-1.0)
            chunk_count_weight: Weight for chunk count factor (0.0-1.0)
            variance_weight: Weight for similarity variance factor (0.0-1.0)
        """
        # Normalize weights to sum to 1.0
        total = query_length_weight + chunk_count_weight + variance_weight
        self.query_length_weight = query_length_weight / total
        self.chunk_count_weight = chunk_count_weight / total
        self.variance_weight = variance_weight / total

        # Reference values for normalization
        self.ref_query_length = 50  # Reference query length (characters)
        self.ref_chunk_count = 20    # Reference chunk count

    def estimate_complexity(
        self,
        query: str,
        chunks: List[str],
        chunk_embeddings: np.ndarray,
        query_embedding: np.ndarray
    ) -> float:
        """
        Estimate query complexity based on multiple factors.

        Args:
            query: Query text
            chunks: List of chunk texts
            chunk_embeddings: Chunk embeddings [n_chunks, dim]
            query_embedding: Query embedding [dim]

        Returns:
            Complexity score from 0.0 (simple) to 1.0 (complex)
        """
        # Factor 1: Query length (longer queries → more complex)
        query_length = len(query)
        length_score = min(1.0, query_length / self.ref_query_length)

        # Factor 2: Number of chunks (more chunks → more complex)
        chunk_count = len(chunks)
        count_score = min(1.0, chunk_count / self.ref_chunk_count)

        # Factor 3: Similarity variance (high variance → simpler, low variance → more complex)
        # High variance means there are clear "winners" - can expand fewer chunks
        # Low variance means many chunks are equally relevant - need to expand more
        if len(chunks) > 1:
            # Calculate cosine similarities
            similarities = self._compute_similarities(query_embedding, chunk_embeddings)
            variance = np.var(similarities)
            # Normalize variance (typical range: 0.0-0.1)
            # Invert: high variance → low complexity, low variance → high complexity
            variance_score = max(0.0, min(1.0, 1.0 - (variance * 10.0)))
        else:
            # Single chunk → simple query
            variance_score = 0.0

        # Combine factors with weights
        complexity = (
            self.query_length_weight * length_score +
            self.chunk_count_weight * count_score +
            self.variance_weight * variance_score
        )

        # Ensure result is in [0.0, 1.0]
        complexity = max(0.0, min(1.0, complexity))

        return complexity

    def _compute_similarities(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarities between query and chunks.

        Args:
            query_embedding: Query embedding [dim]
            chunk_embeddings: Chunk embeddings [n_chunks, dim]

        Returns:
            Similarity scores [n_chunks]
        """
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        chunk_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarities
        similarities = np.dot(chunk_norms, query_norm)

        return similarities

    def get_dynamic_expansion_fraction(
        self,
        complexity: float,
        min_fraction: float = 0.1,
        max_fraction: float = 0.5
    ) -> float:
        """
        Convert complexity score to expansion fraction.

        Args:
            complexity: Complexity score (0.0-1.0)
            min_fraction: Minimum expansion fraction for simple queries
            max_fraction: Maximum expansion fraction for complex queries

        Returns:
            Expansion fraction (min_fraction to max_fraction)
        """
        # Linear interpolation between min and max
        fraction = min_fraction + (complexity * (max_fraction - min_fraction))

        return fraction

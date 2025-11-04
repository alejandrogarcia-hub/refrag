"""
Heuristic selection policies for REfrag.

This module implements various heuristic strategies for selecting which chunks
to expand vs compress, including similarity-based, TF-IDF, position-based,
and hybrid approaches.
"""

import logging
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from refrag.selection.base import SelectionPolicy
from refrag.utils import batch_cosine_similarity, setup_logging

logger = setup_logging()


class SimilarityPolicy(SelectionPolicy):
    """
    Select chunks based on cosine similarity to query.

    This policy selects the chunks that are most semantically similar to
    the query, assuming these are the most relevant for answering.
    """

    def select(
        self,
        chunks: List[str],
        query: str,
        chunk_embeddings: np.ndarray,
        query_embedding: np.ndarray
    ) -> List[int]:
        """
        Select chunks with highest similarity to query.

        Args:
            chunks: List of text chunks
            query: Query text
            chunk_embeddings: Embeddings of chunks
            query_embedding: Embedding of query

        Returns:
            List of indices of chunks to expand
        """
        # Compute cosine similarities
        similarities = batch_cosine_similarity(query_embedding, chunk_embeddings)

        # Get top-k
        selected_indices = self._get_top_k(similarities)

        logger.debug(
            f"SimilarityPolicy selected {len(selected_indices)} chunks "
            f"(top scores: {similarities[selected_indices][:3]})"
        )

        return selected_indices


class TFIDFPolicy(SelectionPolicy):
    """
    Select chunks based on TF-IDF importance.

    This policy selects chunks with the highest TF-IDF scores relative to
    all chunks, assuming these contain the most distinctive information.
    """

    def select(
        self,
        chunks: List[str],
        query: str,
        chunk_embeddings: np.ndarray,
        query_embedding: np.ndarray
    ) -> List[int]:
        """
        Select chunks with highest TF-IDF scores.

        Args:
            chunks: List of text chunks
            query: Query text (not used, but kept for interface consistency)
            chunk_embeddings: Embeddings of chunks (not used)
            query_embedding: Embedding of query (not used)

        Returns:
            List of indices of chunks to expand
        """
        if len(chunks) == 0:
            return []

        # Compute TF-IDF
        vectorizer = TfidfVectorizer(max_features=100)

        try:
            tfidf_matrix = vectorizer.fit_transform(chunks)

            # Sum TF-IDF scores for each chunk (across all terms)
            chunk_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

        except ValueError:
            # If TF-IDF fails (e.g., empty chunks), fall back to uniform scores
            logger.warning("TF-IDF computation failed, using uniform scores")
            chunk_scores = np.ones(len(chunks))

        # Get top-k
        selected_indices = self._get_top_k(chunk_scores)

        logger.debug(
            f"TFIDFPolicy selected {len(selected_indices)} chunks "
            f"(top scores: {chunk_scores[selected_indices][:3]})"
        )

        return selected_indices


class PositionPolicy(SelectionPolicy):
    """
    Select chunks based on position.

    This policy assumes earlier chunks are more important (common in many
    document types where key information appears early).
    """

    def select(
        self,
        chunks: List[str],
        query: str,
        chunk_embeddings: np.ndarray,
        query_embedding: np.ndarray
    ) -> List[int]:
        """
        Select the first k chunks based on position.

        Args:
            chunks: List of text chunks
            query: Query text (not used)
            chunk_embeddings: Embeddings of chunks (not used)
            query_embedding: Embedding of query (not used)

        Returns:
            List of indices of chunks to expand
        """
        n_chunks = len(chunks)
        k = max(1, int(n_chunks * self.expansion_fraction))

        # Select first k chunks
        selected_indices = list(range(min(k, n_chunks)))

        logger.debug(f"PositionPolicy selected first {len(selected_indices)} chunks")

        return selected_indices


class HybridPolicy(SelectionPolicy):
    """
    Hybrid policy combining multiple signals.

    This policy combines similarity, TF-IDF, and position signals with
    configurable weights to make selection decisions.
    """

    def __init__(
        self,
        expansion_fraction: float = 0.25,
        similarity_weight: float = 0.5,
        tfidf_weight: float = 0.3,
        position_weight: float = 0.2
    ):
        """
        Initialize hybrid policy.

        Args:
            expansion_fraction: Fraction of chunks to expand
            similarity_weight: Weight for similarity scores
            tfidf_weight: Weight for TF-IDF scores
            position_weight: Weight for position scores
        """
        super().__init__(expansion_fraction)
        self.similarity_weight = similarity_weight
        self.tfidf_weight = tfidf_weight
        self.position_weight = position_weight

        # Normalize weights
        total_weight = similarity_weight + tfidf_weight + position_weight
        self.similarity_weight /= total_weight
        self.tfidf_weight /= total_weight
        self.position_weight /= total_weight

    def select(
        self,
        chunks: List[str],
        query: str,
        chunk_embeddings: np.ndarray,
        query_embedding: np.ndarray
    ) -> List[int]:
        """
        Select chunks using hybrid scoring.

        Args:
            chunks: List of text chunks
            query: Query text
            chunk_embeddings: Embeddings of chunks
            query_embedding: Embedding of query

        Returns:
            List of indices of chunks to expand
        """
        n_chunks = len(chunks)

        # 1. Similarity scores
        similarity_scores = batch_cosine_similarity(query_embedding, chunk_embeddings)
        similarity_scores = (similarity_scores - similarity_scores.min()) / (
            similarity_scores.max() - similarity_scores.min() + 1e-8
        )

        # 2. TF-IDF scores
        try:
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(chunks)
            tfidf_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
            tfidf_scores = (tfidf_scores - tfidf_scores.min()) / (
                tfidf_scores.max() - tfidf_scores.min() + 1e-8
            )
        except ValueError:
            tfidf_scores = np.zeros(n_chunks)

        # 3. Position scores (earlier = higher)
        position_scores = np.linspace(1.0, 0.0, n_chunks)

        # Combine scores
        combined_scores = (
            self.similarity_weight * similarity_scores
            + self.tfidf_weight * tfidf_scores
            + self.position_weight * position_scores
        )

        # Get top-k
        selected_indices = self._get_top_k(combined_scores)

        logger.debug(
            f"HybridPolicy selected {len(selected_indices)} chunks "
            f"(top combined scores: {combined_scores[selected_indices][:3]})"
        )

        return selected_indices


def create_policy(strategy: str, expansion_fraction: float = 0.25) -> SelectionPolicy:
    """
    Factory function to create a selection policy.

    Args:
        strategy: Policy strategy ('similarity', 'tfidf', 'position', 'hybrid')
        expansion_fraction: Fraction of chunks to expand

    Returns:
        SelectionPolicy instance

    Raises:
        ValueError: If strategy is not recognized
    """
    if strategy == "similarity":
        return SimilarityPolicy(expansion_fraction)
    elif strategy == "tfidf":
        return TFIDFPolicy(expansion_fraction)
    elif strategy == "position":
        return PositionPolicy(expansion_fraction)
    elif strategy == "hybrid":
        return HybridPolicy(expansion_fraction)
    else:
        raise ValueError(
            f"Unknown selection strategy: {strategy}. "
            f"Choose from: similarity, tfidf, position, hybrid"
        )

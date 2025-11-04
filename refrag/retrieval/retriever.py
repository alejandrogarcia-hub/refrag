"""
Document retriever for RAG.

This module provides high-level retrieval functionality for the REfrag system,
combining vector store queries with ranking and filtering.
"""

import logging
from typing import Dict, List, Optional, Tuple

from refrag.config import Config
from refrag.retrieval.vector_store import VectorStore
from refrag.utils import setup_logging

logger = setup_logging()


class Retriever:
    """
    Document retriever for REfrag.

    This class provides high-level retrieval functionality, combining vector
    search with optional re-ranking and filtering strategies.

    Attributes:
        vector_store: VectorStore instance for document storage and search
        config: Configuration object
    """

    def __init__(self, config: Config):
        """
        Initialize the retriever.

        Args:
            config: REfrag configuration object
        """
        self.config = config
        self.vector_store = VectorStore(config)

        logger.info("Retriever initialized")

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the retrieval system.

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
        """
        self.vector_store.add_documents(documents, metadatas, ids)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None
    ) -> List[str]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve (defaults to config.top_k_documents)
            min_similarity: Minimum similarity threshold (optional)

        Returns:
            List of retrieved document texts
        """
        # Use config default if not specified
        if top_k is None:
            top_k = self.config.top_k_documents

        logger.debug(f"Retrieving top {top_k} documents for query: {query[:100]}...")

        # Query vector store
        documents, distances, ids, metadatas = self.vector_store.query(
            query_text=query,
            n_results=top_k
        )

        # Filter by similarity if specified
        if min_similarity is not None:
            # Convert distance to similarity (assuming L2 distance)
            # similarity = 1 / (1 + distance)
            filtered_docs = []
            for doc, dist in zip(documents, distances):
                similarity = 1.0 / (1.0 + dist)
                if similarity >= min_similarity:
                    filtered_docs.append(doc)
            documents = filtered_docs

        logger.info(f"Retrieved {len(documents)} documents")

        return documents

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve documents with similarity scores.

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            List of (document, similarity_score) tuples
        """
        if top_k is None:
            top_k = self.config.top_k_documents

        # Query vector store
        documents, distances, ids, metadatas = self.vector_store.query(
            query_text=query,
            n_results=top_k
        )

        # Convert distances to similarity scores
        # Using: similarity = 1 / (1 + distance)
        results = []
        for doc, dist in zip(documents, distances):
            similarity = 1.0 / (1.0 + dist)
            results.append((doc, similarity))

        return results

    def count_documents(self) -> int:
        """
        Get the number of documents in the store.

        Returns:
            Number of documents
        """
        return self.vector_store.count()

    def reset(self) -> None:
        """Reset the retriever (delete all documents)."""
        self.vector_store.reset()

"""
Retrieval module for REfrag.

This module provides document storage, embedding, and retrieval functionality.
"""

from refrag.retrieval.embedder import DocumentEmbedder
from refrag.retrieval.retriever import Retriever
from refrag.retrieval.vector_store import VectorStore

__all__ = ["DocumentEmbedder", "VectorStore", "Retriever"]

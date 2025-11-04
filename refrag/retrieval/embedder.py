"""
Document embedder for retrieval.

This module provides functionality to generate embeddings for documents using
pre-trained sentence transformers models.
"""

import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from refrag.config import Config
from refrag.utils import setup_logging

logger = setup_logging()


class DocumentEmbedder:
    """
    Document embedder using sentence transformers.

    This class provides functionality to encode documents into dense vector
    embeddings for semantic search and retrieval.

    Attributes:
        model: SentenceTransformer model for encoding
        embedding_dim: Dimension of the output embeddings
    """

    def __init__(self, config: Config):
        """
        Initialize the document embedder.

        Args:
            config: REfrag configuration object
        """
        self.config = config
        logger.info(f"Loading embedding model: {config.embedding_model}")

        # Load sentence transformer model
        self.model = SentenceTransformer(config.embedding_model)

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed text(s) into dense vectors.

        Args:
            texts: Single text string or list of text strings

        Returns:
            Numpy array of embeddings.
            - For single text: shape (embedding_dim,)
            - For multiple texts: shape (n_texts, embedding_dim)
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True
        else:
            squeeze = False

        # Generate embeddings
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Squeeze if single text
        if squeeze:
            embeddings = embeddings[0]

        return embeddings

    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed a batch of texts with progress bar support.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings of shape (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings


    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Callable interface for embedding.

        Args:
            texts: Single text string or list of text strings

        Returns:
            Numpy array of embeddings
        """
        return self.embed(texts)

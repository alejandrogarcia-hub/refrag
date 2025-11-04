"""
Chunk encoder for REfrag compression.

This module provides functionality to encode text chunks into dense embeddings
using a pre-trained encoder model (RoBERTa by default).
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from refrag.config import Config
from refrag.utils import get_device, setup_logging

logger = setup_logging()


class ChunkEncoder:
    """
    Chunk encoder using transformer models.

    This class encodes text chunks into dense vector embeddings using a
    pre-trained encoder model (default: RoBERTa). The embeddings are extracted
    from the [CLS] token as per the REfrag paper.

    Attributes:
        model: Transformer encoder model
        tokenizer: Tokenizer for the encoder
        device: Device for computation (cuda/mps/cpu)
        embedding_dim: Dimension of output embeddings
    """

    def __init__(self, config: Config):
        """
        Initialize the chunk encoder.

        Args:
            config: REfrag configuration object
        """
        self.config = config
        self.device = get_device(config.device)

        logger.info(f"Loading encoder model: {config.encoder_model}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.encoder_model,
            token=config.huggingface_token
        )
        self.model = AutoModel.from_pretrained(
            config.encoder_model,
            token=config.huggingface_token
        )

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size

        logger.info(
            f"Encoder loaded. Model: {config.encoder_model}, "
            f"Dimension: {self.embedding_dim}, Device: {self.device}"
        )

    def encode(
        self,
        chunks: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text chunks into embeddings.

        Args:
            chunks: List of text chunks to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress (currently not implemented)

        Returns:
            Numpy array of embeddings of shape (n_chunks, embedding_dim)
        """
        if len(chunks) == 0:
            return np.array([]).reshape(0, self.embedding_dim)

        all_embeddings = []

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = self._encode_batch(batch_chunks)
            all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)

        return embeddings

    def _encode_batch(self, chunks: List[str]) -> np.ndarray:
        """
        Encode a batch of chunks.

        Args:
            chunks: List of text chunks

        Returns:
            Numpy array of embeddings of shape (batch_size, embedding_dim)
        """
        # Tokenize
        inputs = self.tokenizer(
            chunks,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract [CLS] token embeddings (first token)
        # Shape: (batch_size, hidden_dim)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # Convert to numpy
        embeddings = cls_embeddings.cpu().numpy()

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text chunk.

        Args:
            text: Text chunk to encode

        Returns:
            Numpy array of embedding of shape (embedding_dim,)
        """
        embeddings = self.encode([text])
        return embeddings[0]

    def __call__(self, chunks: List[str]) -> np.ndarray:
        """
        Callable interface for encoding.

        Args:
            chunks: List of text chunks

        Returns:
            Numpy array of embeddings
        """
        return self.encode(chunks)

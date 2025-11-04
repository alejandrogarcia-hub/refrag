"""
Projector for mapping encoder embeddings to LLM token space.

This module provides the projection layer that maps chunk embeddings from the
encoder space (e.g., RoBERTa 768-dim) to the decoder's token embedding space
(e.g., TinyLlama 2048-dim). This is a critical component of REfrag.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from refrag.config import Config
from refrag.utils import count_parameters, get_device, setup_logging

logger = setup_logging()


class Projector(nn.Module):
    """
    Projection layer for mapping encoder embeddings to decoder token space.

    This layer projects chunk embeddings from the encoder's embedding space
    to the decoder LLM's token embedding space, enabling hybrid input construction.

    The projection can be:
    - Random initialization (for quick demos)
    - Learned via training (for better quality)

    Attributes:
        encoder_dim: Dimension of encoder embeddings
        decoder_dim: Dimension of decoder token embeddings
        projection: Linear projection layer
    """

    def __init__(self, encoder_dim: int, decoder_dim: int):
        """
        Initialize the projector.

        Args:
            encoder_dim: Dimension of encoder embeddings (e.g., 768 for RoBERTa)
            decoder_dim: Dimension of decoder token embeddings (e.g., 2048 for TinyLlama)
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        # Simple linear projection
        self.projection = nn.Linear(encoder_dim, decoder_dim, bias=True)

        # Initialize weights (Xavier uniform initialization)
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

        logger.info(
            f"Projector initialized: {encoder_dim} -> {decoder_dim}, "
            f"Parameters: {count_parameters(self)}"
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project encoder embeddings to decoder space.

        Args:
            embeddings: Encoder embeddings of shape:
                - (batch_size, encoder_dim) for single embeddings
                - (batch_size, seq_len, encoder_dim) for sequences

        Returns:
            Projected embeddings of same shape but with decoder_dim
        """
        return self.projection(embeddings)

    def project_numpy(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project numpy embeddings (convenience method).

        Args:
            embeddings: Numpy array of encoder embeddings

        Returns:
            Numpy array of projected embeddings
        """
        # Convert to tensor
        tensor_embeddings = torch.from_numpy(embeddings).float()

        # Project
        with torch.no_grad():
            projected = self.forward(tensor_embeddings)

        # Convert back to numpy
        return projected.numpy()

    def save(self, path: str) -> None:
        """
        Save projector weights.

        Args:
            path: Path to save the weights
        """
        torch.save(self.state_dict(), path)
        logger.info(f"Projector saved to {path}")

    def load(self, path: str, device: Optional[torch.device] = None) -> None:
        """
        Load projector weights.

        Args:
            path: Path to load the weights from
            device: Device to load the weights to
        """
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        logger.info(f"Projector loaded from {path}")


def create_projector(
    encoder_dim: int,
    decoder_dim: int,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Projector:
    """
    Factory function to create a projector.

    Args:
        encoder_dim: Dimension of encoder embeddings
        decoder_dim: Dimension of decoder token embeddings
        checkpoint_path: Optional path to load weights from
        device: Device to load the projector on

    Returns:
        Projector instance
    """
    projector = Projector(encoder_dim, decoder_dim)

    # Load checkpoint if provided
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        projector.load(checkpoint_path, device=device)

    # Move to device if specified
    if device is not None:
        projector = projector.to(device)

    return projector

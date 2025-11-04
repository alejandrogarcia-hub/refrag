"""
Projection module for REfrag.

This module provides the projection layer for mapping encoder embeddings
to decoder token embedding space.
"""

from refrag.projection.projector import Projector, create_projector

__all__ = ["Projector", "create_projector"]

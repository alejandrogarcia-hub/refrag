"""
Compression module for REfrag.

This module provides text chunking and chunk encoding functionality.
"""

from refrag.compression.cache import EmbeddingCache
from refrag.compression.chunker import TextChunker
from refrag.compression.encoder import ChunkEncoder

__all__ = ["TextChunker", "ChunkEncoder", "EmbeddingCache"]

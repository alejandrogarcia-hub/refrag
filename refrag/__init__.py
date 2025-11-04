"""
REfrag: Efficient RAG with Compress-Sense/Select-Expand methodology.

This package implements the REfrag algorithm from the paper:
"REFRAG: Rethinking RAG based Decoding" (arXiv:2509.01092)

Main components:
- REfragPipeline: End-to-end pipeline
- Config: Configuration management
- Retriever: Document retrieval
- ChunkEncoder: Chunk compression
- Projector: Embedding projection
- SelectionPolicy: Chunk selection
- HybridInputConstructor: Hybrid input creation
"""

from refrag.config import Config
from refrag.pipeline import REfragPipeline

__version__ = "0.1.0"

__all__ = ["REfragPipeline", "Config"]

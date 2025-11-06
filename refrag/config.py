"""
Configuration management for REfrag.

This module provides a centralized configuration system that loads settings from
environment variables and provides default values for all REfrag components.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """
    REfrag configuration class.

    This class holds all configuration parameters for the REfrag system,
    including model selections, chunk parameters, hardware settings, and more.

    Attributes:
        decoder_model: HuggingFace model ID for the decoder LLM
        encoder_model: HuggingFace model ID for the chunk encoder
        embedding_model: HuggingFace model ID for document embeddings
        chunk_size: Number of tokens per chunk (paper uses 16)
        expansion_fraction: Fraction of chunks to expand (0.0-1.0, paper uses 0.25)
        selection_strategy: Strategy for selecting important chunks
        top_k_documents: Number of documents to retrieve per query
        device: Device to use for inference (cuda/mps/cpu)
        use_8bit: Whether to use 8-bit quantization
        use_4bit: Whether to use 4-bit quantization
        chroma_db_path: Path to ChromaDB persistence directory
        chroma_collection_name: Name of ChromaDB collection
        log_level: Logging level
        enable_metrics: Whether to enable detailed metrics logging
        huggingface_token: HuggingFace API token (optional, for gated models)
        anthropic_api_key: Anthropic API key (optional, for baseline comparison)
    """

    # Model Configuration
    decoder_model: str = field(
        default_factory=lambda: os.getenv("DECODER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    )
    encoder_model: str = field(default_factory=lambda: os.getenv("ENCODER_MODEL", "roberta-base"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    # REfrag Configuration
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "16")))
    expansion_fraction: float = field(
        default_factory=lambda: float(os.getenv("EXPANSION_FRACTION", "0.25"))
    )
    selection_strategy: Literal["similarity", "tfidf", "position", "hybrid"] = field(
        default_factory=lambda: os.getenv("SELECTION_STRATEGY", "similarity")  # type: ignore
    )
    top_k_documents: int = field(default_factory=lambda: int(os.getenv("TOP_K_DOCUMENTS", "5")))

    # Hardware Configuration
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cuda"))
    use_8bit: bool = field(default_factory=lambda: os.getenv("USE_8BIT", "false").lower() == "true")
    use_4bit: bool = field(default_factory=lambda: os.getenv("USE_4BIT", "false").lower() == "true")

    # Vector Database Configuration
    chroma_db_path: str = field(default_factory=lambda: os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    chroma_collection_name: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "refrag_documents")
    )

    # Logging and Debug
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    enable_metrics: bool = field(
        default_factory=lambda: os.getenv("ENABLE_METRICS", "true").lower() == "true"
    )

    # API Keys (optional)
    huggingface_token: str | None = field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate expansion_fraction
        if not 0.0 <= self.expansion_fraction <= 1.0:
            raise ValueError(
                f"expansion_fraction must be between 0.0 and 1.0, got {self.expansion_fraction}"
            )

        # Validate chunk_size
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        # Validate top_k_documents
        if self.top_k_documents < 1:
            raise ValueError(f"top_k_documents must be positive, got {self.top_k_documents}")

        # Auto-detect device if set to "auto"
        if self.device == "auto":
            self.device = self._auto_detect_device()

        # Ensure chroma_db_path exists
        Path(self.chroma_db_path).mkdir(parents=True, exist_ok=True)

    def _auto_detect_device(self) -> str:
        """
        Auto-detect the best available device.

        Returns:
            Device string: "cuda", "mps", or "cpu"
        """
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "decoder_model": self.decoder_model,
            "encoder_model": self.encoder_model,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "expansion_fraction": self.expansion_fraction,
            "selection_strategy": self.selection_strategy,
            "top_k_documents": self.top_k_documents,
            "device": self.device,
            "use_8bit": self.use_8bit,
            "use_4bit": self.use_4bit,
            "chroma_db_path": self.chroma_db_path,
            "chroma_collection_name": self.chroma_collection_name,
            "log_level": self.log_level,
            "enable_metrics": self.enable_metrics,
        }

    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"Config({self.to_dict()})"


# Create a default configuration instance
default_config = Config()

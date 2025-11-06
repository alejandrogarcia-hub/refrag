"""
LLM interface for loading and managing decoder models.

This module provides a wrapper for loading and interacting with local
HuggingFace models (TinyLlama, Phi-2, Llama-2) for text generation.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from refrag.config import Config
from refrag.utils import count_parameters, format_size, get_device, setup_logging

logger = setup_logging()


class LLMInterface:
    """
    Interface for local LLM models.

    This class handles loading, initialization, and basic interaction with
    local decoder models. It provides access to the model's embedding layer
    which is essential for REfrag's hybrid input construction.

    Attributes:
        model: The loaded causal language model
        tokenizer: Tokenizer for the model
        device: Device for computation
        embedding_dim: Dimension of token embeddings
    """

    def __init__(self, config: Config):
        """
        Initialize the LLM interface.

        Args:
            config: REfrag configuration object
        """
        self.config = config
        self.device = get_device(config.device)

        logger.info(f"Loading decoder model: {config.decoder_model}")
        logger.info(f"Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.decoder_model, token=config.huggingface_token, trust_remote_code=True
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization if configured
        model_kwargs = {
            "token": config.huggingface_token,
            "trust_remote_code": True,
            "dtype": torch.float16 if config.device == "cuda" else torch.float32,
        }

        # Add quantization config if requested
        if config.use_8bit:
            logger.info("Using 8-bit quantization")
            model_kwargs["load_in_8bit"] = True
        elif config.use_4bit:
            logger.info("Using 4-bit quantization")
            model_kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(config.decoder_model, **model_kwargs)

        # Move to device if not quantized
        if not (config.use_8bit or config.use_4bit):
            self.model = self.model.to(self.device)

        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size

        # Log model information
        num_params = count_parameters(self.model)
        logger.info(
            f"Model loaded: {config.decoder_model}\n"
            f"  Parameters: {num_params:,} ({format_size(num_params * 4)})\n"
            f"  Embedding dim: {self.embedding_dim}\n"
            f"  Vocab size: {self.model.config.vocab_size}\n"
            f"  Device: {self.device}"
        )

    def get_embedding_layer(self) -> torch.nn.Embedding:
        """
        Get the model's input embedding layer.

        Returns:
            The embedding layer
        """
        # Different models have different attribute names
        if hasattr(self.model, "model"):
            if hasattr(self.model.model, "embed_tokens"):
                return self.model.model.embed_tokens
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "wte"):
                return self.model.transformer.wte
            if hasattr(self.model.transformer, "embd"):
                return self.model.transformer.embd

        # Fallback: use get_input_embeddings
        return self.model.get_input_embeddings()

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            Tensor of token IDs
        """
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)

    def decode_tokens(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Tensor of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        # Handle batch dimension
        if token_ids.dim() == 2:
            token_ids = token_ids[0]

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def generate_from_text(
        self, prompt: str, max_length: int = 100, temperature: float = 0.7, do_sample: bool = False
    ) -> str:
        """
        Generate text from a text prompt (standard generation).

        Args:
            prompt: Input prompt text
            max_length: Maximum length of generation
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return self.decode_tokens(outputs[0])

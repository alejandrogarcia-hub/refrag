"""
Hybrid input constructor for REfrag.

This module implements the core innovation of REfrag: constructing hybrid
input sequences that mix compressed chunk embeddings with full token embeddings.
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from refrag.generation.llm_interface import LLMInterface
from refrag.projection.projector import Projector
from refrag.utils import setup_logging

logger = setup_logging()


class HybridInputConstructor:
    """
    Constructs hybrid input sequences for REfrag.

    This class is the heart of REfrag. It creates input sequences that mix:
    - Compressed chunk embeddings (projected from encoder space)
    - Full token embeddings (for important/expanded chunks)

    This allows the LLM to process long contexts efficiently by using a single
    embedding to represent 16 tokens for less important chunks.

    Attributes:
        llm: LLM interface with model and tokenizer
        projector: Projection layer for mapping encoder embeddings to LLM space
        embedding_layer: The LLM's token embedding layer
        position_embedding: Position embeddings for compressed chunks to preserve document structure
    """

    def __init__(self, llm: LLMInterface, projector: Projector, max_position_embeddings: int = 512):
        """
        Initialize the hybrid input constructor.

        Args:
            llm: LLM interface
            projector: Projector for mapping encoder embeddings
            max_position_embeddings: Maximum number of position embeddings (default: 512)
        """
        self.llm = llm
        self.projector = projector.to(llm.device)
        self.embedding_layer = llm.get_embedding_layer()
        self.device = llm.device

        # Initialize position embeddings for compressed chunks
        # This preserves document structure and positional context
        hidden_dim = llm.embedding_dim
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_dim)
        self.position_embedding = self.position_embedding.to(llm.device)

        # Initialize position embeddings with small random values
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        logger.info("HybridInputConstructor initialized with positional encodings")

    def construct(
        self,
        query: str,
        chunks: List[str],
        chunk_embeddings: np.ndarray,
        selected_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct hybrid input from query and chunks.

        This is the core REfrag algorithm:
        1. Tokenize query → get token embeddings
        2. For each chunk:
           - If selected: tokenize → get token embeddings (EXPAND)
           - If not selected: use projected encoder embedding (COMPRESS)
        3. Concatenate all embeddings into hybrid sequence

        Args:
            query: Query text
            chunks: List of text chunks
            chunk_embeddings: Encoder embeddings for chunks [n_chunks, encoder_dim]
            selected_indices: Indices of chunks to expand

        Returns:
            Tuple of (hybrid_embeddings, attention_mask)
            - hybrid_embeddings: [1, seq_len, hidden_dim]
            - attention_mask: [1, seq_len]
        """
        # Convert selected indices to set for fast lookup
        selected_set = set(selected_indices)

        # 1. Encode query tokens
        query_ids = self.llm.tokenizer(
            query,
            return_tensors="pt",
            add_special_tokens=True
        ).input_ids.to(self.device)

        query_embeds = self.embedding_layer(query_ids)  # [1, query_len, hidden_dim]

        # 2. Process each chunk
        chunk_embeds_list = []

        for i, chunk in enumerate(chunks):
            if i in selected_set:
                # EXPAND: Use full token embeddings
                chunk_ids = self.llm.tokenizer(
                    chunk,
                    return_tensors="pt",
                    add_special_tokens=False
                ).input_ids.to(self.device)

                chunk_embeds = self.embedding_layer(chunk_ids)  # [1, chunk_len, hidden_dim]

            else:
                # COMPRESS: Use single projected embedding with positional encoding
                # Get encoder embedding for this chunk
                encoder_emb = torch.from_numpy(chunk_embeddings[i]).float().to(self.device)

                # Project to LLM space
                with torch.no_grad():
                    projected_emb = self.projector(encoder_emb)  # [hidden_dim]

                    # Add positional encoding to preserve document structure
                    position_id = torch.tensor([i], device=self.device)
                    position_emb = self.position_embedding(position_id)  # [1, hidden_dim]

                    # Combine projected embedding with positional encoding
                    combined_emb = projected_emb + position_emb.squeeze(0)  # [hidden_dim]

                # Add batch and sequence dimensions
                chunk_embeds = combined_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]

            chunk_embeds_list.append(chunk_embeds)

        # 3. Concatenate all embeddings
        # Query + all chunks
        all_embeds = [query_embeds] + chunk_embeds_list
        hybrid_embeds = torch.cat(all_embeds, dim=1)  # [1, total_seq_len, hidden_dim]

        # 4. Create attention mask (all ones, attend to everything)
        attention_mask = torch.ones(
            hybrid_embeds.shape[:2], 
            dtype=torch.long, 
            device=self.device
        )  # [1, total_seq_len]

        logger.debug(
            f"Constructed hybrid input: query_len={query_embeds.shape[1]}, "
            f"n_chunks={len(chunks)}, expanded={len(selected_indices)}, "
            f"compressed={len(chunks)-len(selected_indices)}, "
            f"total_seq_len={hybrid_embeds.shape[1]}"
        )

        return hybrid_embeds, attention_mask

    def generate(
        self,
        hybrid_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = False
    ) -> str:
        """
        Generate text from hybrid embeddings.

        Args:
            hybrid_embeds: Hybrid input embeddings [1, seq_len, hidden_dim]
            attention_mask: Attention mask [1, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        with torch.no_grad():
            outputs = self.llm.model.generate(
                inputs_embeds=hybrid_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.llm.tokenizer.pad_token_id,
                eos_token_id=self.llm.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.llm.decode_tokens(outputs[0], skip_special_tokens=True)

        return generated_text

    def construct_and_generate(
        self,
        query: str,
        chunks: List[str],
        chunk_embeddings: np.ndarray,
        selected_indices: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = False
    ) -> str:
        """
        Convenience method: construct hybrid input and generate in one call.

        Args:
            query: Query text
            chunks: List of text chunks
            chunk_embeddings: Encoder embeddings for chunks
            selected_indices: Indices of chunks to expand
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        # Construct hybrid input
        hybrid_embeds, attention_mask = self.construct(
            query, chunks, chunk_embeddings, selected_indices
        )

        # Generate
        generated_text = self.generate(
            hybrid_embeds, attention_mask, max_new_tokens, temperature, do_sample
        )

        return generated_text

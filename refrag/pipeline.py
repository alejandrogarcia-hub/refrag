"""
End-to-end REfrag pipeline.

This module provides the complete REfrag pipeline that integrates all components:
retrieval, compression, projection, selection, and generation.
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from refrag.compression import ChunkEncoder, TextChunker
from refrag.config import Config
from refrag.generation import HybridInputConstructor, LLMInterface
from refrag.projection import create_projector
from refrag.retrieval import DocumentEmbedder, Retriever
from refrag.selection import create_policy
from refrag.utils import MetricsTracker, setup_logging

logger = setup_logging()


class REfragPipeline:
    """
    Complete end-to-end REfrag pipeline.

    This pipeline implements the full REfrag algorithm:
    1. RETRIEVE: Find relevant documents from vector store
    2. COMPRESS: Split documents into chunks and encode to embeddings
    3. SENSE/SELECT: Identify important chunks to expand
    4. EXPAND: Construct hybrid input (compressed + expanded chunks)
    5. GENERATE: Generate answer from hybrid input

    Attributes:
        config: Configuration object
        retriever: Document retriever
        chunker: Text chunker
        encoder: Chunk encoder
        projector: Embedding projector
        policy: Selection policy
        llm: LLM interface
        hybrid_constructor: Hybrid input constructor
        metrics: Metrics tracker
    """

    def __init__(self, config: Config):
        """
        Initialize the REfrag pipeline.

        Args:
            config: REfrag configuration object
        """
        self.config = config
        self.metrics = MetricsTracker()

        logger.info("="* 60)
        logger.info("Initializing REfrag Pipeline")
        logger.info("=" * 60)

        # 1. Initialize retrieval components
        logger.info("1/6: Loading retrieval system...")
        self.retriever = Retriever(config)
        self.doc_embedder = DocumentEmbedder(config)

        # 2. Initialize compression components
        logger.info("2/6: Loading compression components...")
        self.encoder = ChunkEncoder(config)
        self.chunker = TextChunker(
            tokenizer=self.encoder.tokenizer,
            chunk_size=config.chunk_size
        )

        # 3. Initialize LLM
        logger.info("3/6: Loading decoder LLM...")
        self.llm = LLMInterface(config)

        # 4. Initialize projector
        logger.info("4/6: Initializing projector...")
        self.projector = create_projector(
            encoder_dim=self.encoder.embedding_dim,
            decoder_dim=self.llm.embedding_dim,
            device=self.llm.device
        )

        # 5. Initialize selection policy
        logger.info("5/6: Setting up selection policy...")
        self.policy = create_policy(
            strategy=config.selection_strategy,
            expansion_fraction=config.expansion_fraction
        )

        # 6. Initialize hybrid constructor
        logger.info("6/6: Creating hybrid input constructor...")
        self.hybrid_constructor = HybridInputConstructor(self.llm, self.projector)

        logger.info("=" * 60)
        logger.info("REfrag Pipeline initialized successfully!")
        logger.info("=" * 60)

    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the retrieval system.

        Args:
            documents: List of document texts
        """
        self.retriever.add_documents(documents)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = False
    ) -> Dict:
        """
        Process a query through the complete REfrag pipeline.

        Args:
            question: User question
            top_k: Number of documents to retrieve (default: from config)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Dictionary containing:
                - answer: Generated answer text
                - metrics: Performance metrics
                - debug: Debug information
        """
        logger.info(f"\n{'='*60}\nProcessing query: {question[:100]}...\n{'='*60}")

        start_time = time.time()
        self.metrics.start("total_time")

        # Use config default if not specified
        if top_k is None:
            top_k = self.config.top_k_documents

        # STEP 1: RETRIEVE relevant documents
        logger.info("Step 1/5: Retrieving documents...")
        self.metrics.start("retrieval_time")
        documents = self.retriever.retrieve(question, top_k=top_k)
        self.metrics.end("retrieval_time")

        if len(documents) == 0:
            logger.warning("No documents retrieved!")
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "metrics": {},
                "debug": {"error": "No documents retrieved"}
            }

        logger.info(f"Retrieved {len(documents)} documents")

        # STEP 2: COMPRESS - Chunk and encode
        logger.info("Step 2/5: Chunking and encoding documents...")
        self.metrics.start("compression_time")

        all_chunks = []
        all_embeddings = []

        for doc in documents:
            # Chunk document
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

            # Encode chunks
            if len(chunks) > 0:
                embeddings = self.encoder.encode(chunks)
                all_embeddings.append(embeddings)

        if len(all_chunks) == 0:
            logger.warning("No chunks created from documents!")
            return {
                "answer": "I couldn't process the retrieved documents.",
                "metrics": {},
                "debug": {"error": "No chunks created"}
            }

        chunk_embeddings = np.concatenate(all_embeddings, axis=0)
        self.metrics.end("compression_time")

        logger.info(f"Created {len(all_chunks)} chunks")

        # STEP 3: SENSE/SELECT - Identify important chunks
        logger.info("Step 3/5: Selecting chunks to expand...")
        self.metrics.start("selection_time")

        # Get query embedding for selection
        # IMPORTANT: Use the same encoder (RoBERTa) as chunks to match dimensions
        query_embedding = self.encoder.encode_single(question)

        # Select chunks
        selected_indices = self.policy.select(
            chunks=all_chunks,
            query=question,
            chunk_embeddings=chunk_embeddings,
            query_embedding=query_embedding
        )

        self.metrics.end("selection_time")

        logger.info(f"Selected {len(selected_indices)}/{len(all_chunks)} chunks to expand")

        # STEP 4: Calculate token metrics
        # Count original tokens (if all chunks were expanded)
        original_tokens = sum(self.chunker.count_tokens(chunk) for chunk in all_chunks)

        # Count compressed tokens (selected chunks = full tokens, others = 1 token each)
        query_tokens = self.chunker.count_tokens(question)
        compressed_chunk_tokens = len(all_chunks) - len(selected_indices)  # 1 token per compressed chunk
        expanded_chunk_tokens = sum(
            self.chunker.count_tokens(all_chunks[i]) for i in selected_indices
        )
        compressed_tokens = query_tokens + compressed_chunk_tokens + expanded_chunk_tokens

        compression_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1.0

        logger.info(
            f"Token reduction: {original_tokens} → {compressed_tokens} "
            f"({compression_ratio:.2f}x compression)"
        )

        # STEP 5: EXPAND & GENERATE - Create hybrid input and generate answer
        logger.info("Step 4/5: Constructing hybrid input...")
        logger.info("Step 5/5: Generating answer...")

        self.metrics.start("generation_time")
        ttft_start = time.time()

        try:
            answer = self.hybrid_constructor.construct_and_generate(
                query=question,
                chunks=all_chunks,
                chunk_embeddings=chunk_embeddings,
                selected_indices=selected_indices,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )

            ttft = time.time() - ttft_start
            self.metrics.end("generation_time")

            logger.info(f"Answer generated (TTFT: {ttft:.2f}s)")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "answer": f"Error during generation: {str(e)}",
                "metrics": {},
                "debug": {"error": str(e)}
            }

        total_time = time.time() - start_time

        # Compile results
        result = {
            "answer": answer,
            "metrics": {
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "compression_ratio": compression_ratio,
                "tokens_saved": original_tokens - compressed_tokens,
                "ttft": ttft,
                "retrieval_time": self.metrics.get_last("retrieval_time"),
                "compression_time": self.metrics.get_last("compression_time"),
                "selection_time": self.metrics.get_last("selection_time"),
                "generation_time": self.metrics.get_last("generation_time"),
                "total_time": total_time
            },
            "debug": {
                "retrieved_docs": len(documents),
                "total_chunks": len(all_chunks),
                "selected_chunks": len(selected_indices),
                "query_tokens": query_tokens,
                "expanded_chunk_tokens": expanded_chunk_tokens,
                "compressed_chunk_tokens": compressed_chunk_tokens
            }
        }

        logger.info(f"\n{'='*60}\nQuery completed in {total_time:.2f}s\n{'='*60}\n")

        return result

    def reset_documents(self) -> None:
        """Reset the document store (delete all documents)."""
        self.retriever.reset()

    def document_count(self) -> int:
        """
        Get the number of documents in the store.

        Returns:
            Number of documents
        """
        return self.retriever.count_documents()

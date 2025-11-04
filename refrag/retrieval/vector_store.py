"""
Vector store for document storage and retrieval using ChromaDB.

This module provides a wrapper around ChromaDB for storing and retrieving
document embeddings efficiently.
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from refrag.config import Config
from refrag.retrieval.embedder import DocumentEmbedder
from refrag.utils import setup_logging

logger = setup_logging()


class VectorStore:
    """
    Vector store using ChromaDB for document storage and retrieval.

    This class provides functionality to add documents, generate embeddings,
    and perform semantic search over the stored documents.

    Attributes:
        client: ChromaDB client instance
        collection: ChromaDB collection for storing documents
        embedder: DocumentEmbedder for generating embeddings
    """

    def __init__(self, config: Config):
        """
        Initialize the vector store.

        Args:
            config: REfrag configuration object
        """
        self.config = config

        # Initialize embedder
        self.embedder = DocumentEmbedder(config)

        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at: {config.chroma_db_path}")
        self.client = chromadb.PersistentClient(
            path=config.chroma_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=config.chroma_collection_name,
            metadata={"description": "REfrag document store"}
        )

        logger.info(
            f"Vector store initialized. Collection: {config.chroma_collection_name}, "
            f"Documents: {self.collection.count()}"
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries for each document
            ids: Optional list of document IDs. If not provided, UUIDs will be generated
        """
        if len(documents) == 0:
            logger.warning("No documents to add")
            return

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # Generate metadatas if not provided
        if metadatas is None:
            metadatas = [{"text_length": len(doc)} for doc in documents]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedder.embed_batch(documents, show_progress=True)

        # Add to collection
        logger.info(f"Adding {len(documents)} documents to vector store...")
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Successfully added {len(documents)} documents")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Tuple[List[str], List[float], List[str], List[Dict]]:
        """
        Query the vector store for similar documents.

        Args:
            query_text: Query text
            n_results: Number of results to return
            where: Optional metadata filter
            where_document: Optional document content filter

        Returns:
            Tuple of (documents, distances, ids, metadatas)
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query_text)

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            where_document=where_document
        )

        # Extract results
        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        ids = results["ids"][0] if results["ids"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        return documents, distances, ids, metadatas

    def delete_all(self) -> None:
        """Delete all documents from the collection."""
        # Get all IDs
        all_docs = self.collection.get()
        if all_docs["ids"]:
            self.collection.delete(ids=all_docs["ids"])
            logger.info(f"Deleted {len(all_docs['ids'])} documents from collection")

    def count(self) -> int:
        """
        Get the number of documents in the store.

        Returns:
            Number of documents
        """
        return self.collection.count()

    def reset(self) -> None:
        """Reset the vector store (delete all data)."""
        logger.warning("Resetting vector store - all data will be deleted")
        self.client.delete_collection(name=self.config.chroma_collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.config.chroma_collection_name,
            metadata={"description": "REfrag document store"}
        )
        logger.info("Vector store reset complete")

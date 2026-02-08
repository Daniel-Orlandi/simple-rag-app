"""Retrieval Service module for document retrieval strategies."""
import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Service for document retrieval with different search strategies.
    
    This service provides a unified interface for retrieving documents from a vector store
    using different strategies: similarity search or Maximum Marginal Relevance (MMR).
    
    Attributes:
        vectorstore: The ChromaDB vector store instance to search.
    
    Example:
        >>> service = RetrievalService(vectorstore)
        >>> docs = service.retrieve("alienação fiduciária", strategy="mmr", k=5)
    """
    
    def __init__(self, vectorstore: Chroma):
        """
        Initialize the retrieval service.
        
        Args:
            vectorstore: The ChromaDB vector store instance to use for retrieval.
        """
        self.vectorstore = vectorstore
    
    def retrieve(
        self,
        query: str,
        strategy: str = "similarity",
        k: int = 4
    ) -> List[Document]:
        """
        Retrieve documents using specified strategy.
        
        Args:
            query: Search query
            strategy: "similarity" or "mmr" (Maximum Marginal Relevance)
            k: Number of documents to retrieve
        
        Returns:
            List of relevant documents
        """
        logger.info(f"Retrieving documents with strategy='{strategy}', k={k}")
        logger.debug(f"Query: {query[:100]}..." if len(query) > 100 else f"Query: {query}")
        
        if strategy == "similarity":
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        elif strategy == "mmr":
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k * 2}
            )
        else:
            logger.critical(f"Unknown retrieval strategy: {strategy}")
            raise ValueError(f"Unknown strategy: {strategy}. Use 'similarity' or 'mmr'")
        
        results = retriever.invoke(query)
        logger.debug(f"Retrieved {len(results)} documents")
        return results


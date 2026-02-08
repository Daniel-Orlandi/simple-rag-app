import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ..utils.embeddings import get_embeddings

logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Service for managing ChromaDB vector store operations.
    
    This service handles creation, loading, and management of the vector database.
    It uses lazy loading for embeddings and vector store to optimize memory usage.
    
    Attributes:
        collection_name: Name of the ChromaDB collection.
        embedding_model: Name of the HuggingFace embedding model to use.
        model_kwargs: Additional arguments for the embedding model.
        encode_kwargs: Additional arguments for encoding (e.g., normalization).
        _embeddings: Cached embeddings instance (lazy loaded).
        _vectorstore: Cached vector store instance (lazy loaded).
    
    Example:
        >>> service = VectorStoreService(collection_name="session_123")
        >>> if not service.exists():
        ...     vectorstore = service.create_from_documents(documents)
        >>> else:
        ...     vectorstore = service.vectorstore
    """
    
    def __init__(
        self,
        collection_name: str = "default",        
        embedding_model: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the vector store service.
        
        Args:
            collection_name: Name of the ChromaDB collection.
            embedding_model: Name of the HuggingFace embedding model.
                            If None, will use the default from get_embeddings().
            model_kwargs: Dictionary of arguments for the embedding model
                         (e.g., {"device": "cuda"}).
            encode_kwargs: Dictionary of arguments for encoding
                          (e.g., {"normalize_embeddings": True}).
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        self._embeddings = None
        self._vectorstore = None
    

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get or create the embeddings instance (lazy loading).
        
        The embeddings are created on first access and cached for subsequent use.
        
        Returns:
            HuggingFaceEmbeddings: The embeddings instance configured with the specified model.
        """
        if self._embeddings is None:
            logger.info(f"Initializing embeddings model: {self.embedding_model}")
            self._embeddings = get_embeddings(
                model_name=self.embedding_model,
                model_kwargs=self.model_kwargs,
                encode_kwargs=self.encode_kwargs
            )
            logger.debug("Embeddings model loaded successfully")
        return self._embeddings
    

    @property
    def vectorstore(self) -> Chroma:
        """
        Get or load the vector store instance (lazy loading).
        
        The vector store is loaded from memory if it exists, otherwise it creates a new one.
        
        Returns:
            Chroma: The ChromaDB vector store instance.
        
        Raises:
            ValueError: If the vector store does not exist, it creates a new one.
        """
        if self._vectorstore is None:
            self._vectorstore = self.create_empty_vectorstore()
        return self._vectorstore

    
    def exists(self) -> bool:
        """
        Check if a vector store exists for the configured collection name.
        
        Returns:
            bool: True if the vector store exists and contains data, False otherwise.
        """
        return self._vectorstore is not None

    
    def create_empty_vectorstore(self) -> Chroma:
        """
        Create an empty vector store without any documents.
        
        Useful for initializing a vectorstore before uploading documents.
        The vector store is persisted to disk at the configured persist_directory.
        
        Returns:
            Chroma: The empty ChromaDB vector store instance.
        """        
        logger.info(f"Creating empty vector store for collection: {self.collection_name}")
        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        logger.info("Empty vector store created successfully")
        return self._vectorstore
        
    
    def create_from_documents(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from a list of documents.
        
        This method generates embeddings for all documents and stores them in ChromaDB.
        The vector store is persisted to disk at the configured persist_directory.
        
        Args:
            documents: List of LangChain Document objects to index.
        
        Returns:
            Chroma: The created ChromaDB vector store instance.
        
        Note:
            This operation can be time-consuming for large document collections,
            especially if using CPU for embeddings.
        """
        logger.info(f"Creating vector store from {len(documents)} documents")       
        
        empty_docs = [i for i, d in enumerate(documents) if not d.page_content.strip()]
        if empty_docs:
            logger.warning(f"Found {len(empty_docs)} documents with empty content at indices: {empty_docs[:10]}...")
        
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name
        )
        logger.info("Vector store created and persisted successfully")
        return self._vectorstore
    

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add new documents to an existing vector store.
        
        This method is useful for incrementally adding documents without rebuilding
        the entire vector store. The vector store must already exist.
        
        Args:
            documents: List of LangChain Document objects to add.
        
        Returns:
            List[str]: List of document IDs assigned to the newly added documents.
        
        Raises:
            ValueError: If the vector store does not exist. Use create_from_documents() first.
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        result = self.vectorstore.add_documents(documents)
        logger.debug(f"Added {len(result)} document IDs")
        return result


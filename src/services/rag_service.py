"""RAG Service module for orchestrating retrieval-augmented generation."""
import logging
from typing import Optional
from langchain_core.language_models import BaseChatModel
from .vectorstore_service import VectorStoreService
from .document_service import DocumentService
from ..chains.rag_chain import create_rag_chain
from ..models.config import RAGConfig

logger = logging.getLogger(__name__)

class RAGService:
    """
    Main RAG service that orchestrates all components of the retrieval-augmented generation system.
    
    This service coordinates document loading, vector store management, retrieval, and answer
    generation. It provides a high-level interface for querying documents using RAG.
    
    Attributes:
        config: Configuration object for the RAG system.
        _vectorstore_service: Lazy-loaded vector store service instance.
        _document_service: Lazy-loaded document service instance.
        _retrieval_service: Lazy-loaded retrieval service instance.  

    """
    
    def __init__(self, config: Optional[RAGConfig] = None, session_id:str = 'Default'):
        """
        Initialize the RAG service.
        
        Args:
            config: Configuration object for the RAG system. If None, uses default configuration.
        """
        self.config = config or RAGConfig()
        self.session_id = session_id
        self._vectorstore_service = None
        self._document_service = None

    
    @property
    def vectorstore_service(self) -> VectorStoreService:
        """
        Get or create the vector store service instance (lazy loading).
        
        Returns:
            VectorStoreService: The vector store service instance.
        """
        if self._vectorstore_service is None:
            self._vectorstore_service = VectorStoreService(
                collection_name=f"session_{self.session_id}",
                embedding_model=self.config.embedding_model,
                model_kwargs=self.config.model_kwargs,
                encode_kwargs=self.config.encode_kwargs
            )
        return self._vectorstore_service
    

    @property
    def document_service(self) -> DocumentService:
        """
        Get or create the document service instance (lazy loading).
        
        Returns:
            DocumentService: The document service instance.
        """
        if self._document_service is None:
            self._document_service = DocumentService()
        return self._document_service
    

    def initialize_vectorstore(self, force_rebuild: bool = False) -> None:
        if force_rebuild or not self.vectorstore_service.exists():
            logger.info("Creating empty vector store...")
            self.vectorstore_service.create_empty_vectorstore()
            logger.info("Empty vector store created successfully")

        else:
            logger.info("Loading existing vector store...")
            _ = self.vectorstore_service.vectorstore  
            logger.info("Vector store loaded successfully")


    def add_documents(self, file_paths: list[str], force_rebuild: bool = False):
        """
        Add new documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects to add.
            force_rebuild: If True, forces reconstruction of the vector store even if it exists.
        """
        if force_rebuild:
            return self.initialize_vectorstore(force_rebuild=True)

        if not self.vectorstore_service.exists():
            logger.critical("Vector store does not exist. Use initialize_vectorstore() first.")
            raise ValueError("Vector store does not exist. Use initialize_vectorstore() first.")
        
        new_documents = self.document_service.load_documents(file_paths)
        logger.debug(f"Loaded {len(new_documents)} new documents")

        logger.info("Splitting documents...")
        splits = self.document_service.split_documents(
            new_documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        logger.debug(f"Created {len(splits)} document chunks")

        logger.info("Adding documents to vector store...")
        self.vectorstore_service.add_documents(splits)
        logger.info("Documents added to vector store successfully")
    

    def query_with_sources(self, question: str, llm: BaseChatModel) -> tuple[str, list[str]]:
            """
            Query the RAG system and return both answer and source references.
            
            This method performs a single retrieval operation and uses the results
            for both generating the answer and returning source references.
            
            Args:
                question: The question to ask.
                llm: The language model instance.
            
            Returns:
                tuple: (answer, references) where references are source document contents.
            """
            # Create retriever
            vectorstore = self.vectorstore_service.vectorstore
            retriever = vectorstore.as_retriever(
                search_type=self.config.retrieval_strategy,
                search_kwargs={"k": self.config.retrieval_k}
            )
            
            # Single retrieval
            docs = retriever.invoke(question)
            references = [doc.page_content for doc in docs]
            
            # Generate answer using retrieved docs
            chain = create_rag_chain(llm, retriever)
            answer = chain.invoke(question)
            
            return answer, references
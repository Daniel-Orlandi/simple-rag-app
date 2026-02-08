"""Document Service module for loading and processing documents."""
import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader

logger = logging.getLogger(__name__)

class DocumentService:
    """
    Service for loading and processing documents from various formats.
    
    This service handles loading PDF and HTML files, and splitting them into
    chunks suitable for vector storage. Errors during loading are logged
    but don't stop the process.    

    """
    
    @staticmethod
    def load_pdf_document(file_path: str, **kwargs):
        """
        Load a PDF document from a file path.
        """
        PDF_loader = PyPDFLoader(file_path, **kwargs)
        return PDF_loader.load()
        
    
    @staticmethod
    def load_html_document(file_path: str, **kwargs):
        """
        Load an HTML document from a file path.
        """
        HTML_loader = BSHTMLLoader(file_path, **kwargs)
        return HTML_loader.load()


    def load_documents(self, file_paths: str | list[str]) -> List[Document]:
        """
        Load documents from a file path or list of file paths.
        
        Args:
            file_paths: Single file path (str) or list of file paths to load.
        """
        # Handle single string path by converting to list
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        documents = []

        for file_path in file_paths:
            try:
                logger.debug(f"Loading file: {file_path}")
                if file_path.endswith(".pdf"):                                                    
                    documents.extend(self.load_pdf_document(file_path))
                    logger.debug(f"PDF loaded successfully: {file_path}")

                elif file_path.endswith(".html"):
                    documents.extend(self.load_html_document(file_path, open_encoding="latin-1"))
                    logger.debug(f"HTML loaded successfully: {file_path}")

                else:
                    logger.warning(f"Unsupported file type: {file_path}")                    

            except Exception as e:
                logger.error(f"Error loading file: {file_path}: {e}")
                

        return documents
    

    def load_documents_from_directory(self, directory: str) -> List[Document]:
        """
        Load all PDF and HTML files from the configured data directory.
        
        This method scans the data directory for .pdf and .html files and loads them
        using appropriate loaders. Errors during loading are caught and logged, but
        processing continues for other files.
        
        Returns:
            List[Document]: List of LangChain Document objects loaded from all files.
        
        Note:
            - PDF files are loaded using PyPDFLoader
            - HTML files are loaded using BSHTMLLoader with latin-1 encoding
            - Files that fail to load are skipped with an error message printed
        """
        data_dir = Path(directory)
        documents = []
        
        # Load PDF files
        for pdf_file in data_dir.glob("*.pdf"):
            logger.debug(f"Loading PDF: {pdf_file.name}")
            try:
                documents.extend(self.load_pdf_document(str(pdf_file)))
            except Exception as e:
                logger.warning(f"Error loading {pdf_file.name}: {e}")
        
        # Load HTML files
        for html_file in data_dir.glob("*.html"):
            logger.debug(f"Loading HTML: {html_file.name}")
            try:
                documents.extend(self.load_html_document(str(html_file), open_encoding="latin-1"))
            except Exception as e:
                logger.warning(f"Error loading {html_file.name}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {data_dir}")
        return documents
    
    
    def split_documents(self, documents: List[Document], chunk_size: int = 700, chunk_overlap: int = 100) -> List[Document]:
        """
        Split documents into smaller chunks for vector storage.
        
        Uses RecursiveCharacterTextSplitter to intelligently split documents while
        preserving context. Overlapping chunks help maintain continuity between chunks.
        
        Args:
            documents: List of LangChain Document objects to split.
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between consecutive chunks.
        
        Returns:
            List[Document]: List of document chunks, each as a separate Document object.
        
        Note:
            - Larger chunk_size preserves more context but may include irrelevant information
            - Overlap helps maintain context across chunk boundaries
            - The splitter uses a recursive approach to split on various separators
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        return text_splitter.split_documents(documents)


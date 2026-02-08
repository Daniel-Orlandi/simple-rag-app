"""Integration tests for RAGService."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.services.rag_service import RAGService
from src.models.config import RAGConfig


class TestRAGServiceInit:
    """Tests for RAGService initialization."""

    def test_creates_with_default_config(self, mock_embeddings):
        """Should create with default config."""
        service = RAGService()
        assert service.config is not None

    def test_creates_with_custom_config(self, mock_embeddings):
        """Should accept custom config."""
        config = RAGConfig()
        service = RAGService(config)
        assert service.config is config

    def test_accepts_session_id(self, mock_embeddings):
        """Should accept session_id."""
        service = RAGService(session_id="my-session")
        assert service.session_id == "my-session"

    def test_default_session_id(self, mock_embeddings):
        """Should have default session_id."""
        service = RAGService()
        assert service.session_id == "Default"

    def test_uses_session_id_in_collection_name(self, mock_embeddings):
        """Collection name should include session_id."""
        service = RAGService(session_id="test-123")
        
        with patch("src.services.vectorstore_service.Chroma"):
            vs = service.vectorstore_service
            assert "test-123" in vs.collection_name


class TestInitializeVectorstore:
    """Tests for initialize_vectorstore method."""

    def test_creates_empty_vectorstore(self, rag_service):
        """Should create empty vectorstore."""
        with patch("src.services.vectorstore_service.Chroma"):
            rag_service.initialize_vectorstore()
            assert rag_service.vectorstore_service.exists()

    def test_force_rebuild_creates_new(self, rag_service):
        """Force rebuild should create new vectorstore."""
        with patch("src.services.vectorstore_service.Chroma") as mock_chroma:
            rag_service.initialize_vectorstore()
            rag_service.initialize_vectorstore(force_rebuild=True)
            # Chroma should be called twice
            assert mock_chroma.call_count >= 2


class TestAddDocuments:
    """Tests for add_documents method."""

    def test_raises_if_vectorstore_not_initialized(self, rag_service):
        """Should raise if vectorstore doesn't exist."""
        with pytest.raises(ValueError, match="does not exist"):
            rag_service.add_documents(["/fake/path.pdf"])

    def test_adds_documents_after_init(self, rag_service, sample_documents):
        """Should add documents after vectorstore is initialized."""
        with patch("src.services.vectorstore_service.Chroma") as mock_chroma:
            mock_vs = MagicMock()
            mock_vs.add_documents.return_value = ["id1", "id2"]
            mock_chroma.return_value = mock_vs
            
            rag_service.initialize_vectorstore()
            
            with patch.object(
                rag_service.document_service, 
                'load_documents', 
                return_value=sample_documents
            ):
                with patch.object(
                    rag_service.document_service,
                    'split_documents',
                    return_value=sample_documents
                ):
                    rag_service.add_documents(["/fake/path.pdf"])
                    mock_vs.add_documents.assert_called()


class TestQueryWithSources:
    """Tests for query_with_sources method."""

    def test_returns_tuple(self, rag_service):
        """Should return tuple of (answer, references)."""
        with patch("src.services.vectorstore_service.Chroma") as mock_chroma:
            mock_vs = MagicMock()
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [
                Document(page_content="Reference 1", metadata={}),
                Document(page_content="Reference 2", metadata={})
            ]
            mock_vs.as_retriever.return_value = mock_retriever
            mock_chroma.return_value = mock_vs
            
            rag_service.initialize_vectorstore()
            
            with patch("src.services.rag_service.create_rag_chain") as mock_chain:
                mock_chain.return_value.invoke.return_value = "Test answer"
                
                mock_llm = MagicMock()
                answer, refs = rag_service.query_with_sources("Test question", mock_llm)
                
                assert isinstance(answer, str)
                assert isinstance(refs, list)
                assert len(refs) == 2

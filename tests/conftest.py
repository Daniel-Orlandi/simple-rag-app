"""Shared pytest fixtures for all tests."""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.services.document_service import DocumentService
from src.services.vectorstore_service import VectorStoreService
from src.services.rag_service import RAGService
from src.models.config import RAGConfig


@pytest.fixture
def rag_config():
    """RAG configuration instance."""
    return RAGConfig()


@pytest.fixture
def document_service():
    """Document service instance."""
    return DocumentService()


@pytest.fixture
def mock_embeddings():
    """Mock HuggingFace embeddings to avoid loading the model."""
    with patch("src.services.vectorstore_service.get_embeddings") as mock:
        mock_embedding = MagicMock()
        mock_embedding.embed_documents.return_value = [[0.1] * 384]
        mock_embedding.embed_query.return_value = [0.1] * 384
        mock.return_value = mock_embedding
        yield mock_embedding


@pytest.fixture
def vectorstore_service(mock_embeddings):
    """VectorStore service with mocked embeddings."""
    return VectorStoreService(
        collection_name="test_collection",
        embedding_model="test-model"
    )


@pytest.fixture
def rag_service(mock_embeddings):
    """RAG service with mocked embeddings."""
    config = RAGConfig()
    return RAGService(config, session_id="test-session")


@pytest.fixture
def api_client():
    """FastAPI test client."""
    from main import app
    return TestClient(app)


@pytest.fixture
def sample_document():
    """Sample LangChain Document for testing."""
    from langchain_core.documents import Document
    return Document(
        page_content="This is a test document about machine learning and AI.",
        metadata={"source": "test.pdf", "page": 1}
    )


@pytest.fixture
def sample_documents(sample_document):
    """List of sample documents."""
    from langchain_core.documents import Document
    return [
        sample_document,
        Document(
            page_content="Another document about natural language processing.",
            metadata={"source": "test2.pdf", "page": 1}
        ),
    ]

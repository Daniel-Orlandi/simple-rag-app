"""Tests for VectorStoreService."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.services.vectorstore_service import VectorStoreService


class TestVectorStoreServiceInit:
    """Tests for VectorStoreService initialization."""

    def test_creates_instance_with_defaults(self, mock_embeddings):
        """Should create instance with default collection name."""
        service = VectorStoreService()
        assert service.collection_name == "default"

    def test_accepts_custom_collection_name(self, mock_embeddings):
        """Should accept custom collection name."""
        service = VectorStoreService(collection_name="my_collection")
        assert service.collection_name == "my_collection"

    def test_lazy_loads_embeddings(self, mock_embeddings):
        """Embeddings should not be loaded until accessed."""
        service = VectorStoreService()
        assert service._embeddings is None

    def test_lazy_loads_vectorstore(self, mock_embeddings):
        """Vectorstore should not be loaded until accessed."""
        service = VectorStoreService()
        assert service._vectorstore is None


class TestExists:
    """Tests for exists method."""

    def test_returns_false_when_not_initialized(self, vectorstore_service):
        """Should return False when vectorstore not created."""
        assert vectorstore_service.exists() is False

    def test_returns_true_after_creation(self, vectorstore_service):
        """Should return True after vectorstore is created."""
        with patch("src.services.vectorstore_service.Chroma") as mock_chroma:
            vectorstore_service.create_empty_vectorstore()
            assert vectorstore_service.exists() is True


class TestCreateEmptyVectorstore:
    """Tests for create_empty_vectorstore method."""

    def test_creates_chroma_instance(self, vectorstore_service):
        """Should create a Chroma instance."""
        with patch("src.services.vectorstore_service.Chroma") as mock_chroma:
            result = vectorstore_service.create_empty_vectorstore()
            mock_chroma.assert_called_once()

    def test_uses_collection_name(self, vectorstore_service):
        """Should use the configured collection name."""
        with patch("src.services.vectorstore_service.Chroma") as mock_chroma:
            vectorstore_service.create_empty_vectorstore()
            call_kwargs = mock_chroma.call_args[1]
            assert call_kwargs["collection_name"] == "test_collection"


class TestAddDocuments:
    """Tests for add_documents method."""

    def test_adds_documents_to_vectorstore(self, vectorstore_service, sample_documents):
        """Should add documents to the vectorstore."""
        with patch("src.services.vectorstore_service.Chroma") as mock_chroma:
            mock_vs = MagicMock()
            mock_vs.add_documents.return_value = ["id1", "id2"]
            mock_chroma.return_value = mock_vs
            
            vectorstore_service.create_empty_vectorstore()
            result = vectorstore_service.add_documents(sample_documents)
            
            mock_vs.add_documents.assert_called_once_with(sample_documents)
            assert result == ["id1", "id2"]

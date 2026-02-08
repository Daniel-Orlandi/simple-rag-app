"""Tests for DocumentService."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.services.document_service import DocumentService


class TestDocumentServiceInit:
    """Tests for DocumentService initialization."""

    def test_creates_instance(self):
        """Should create instance without errors."""
        service = DocumentService()
        assert service is not None


class TestLoadDocuments:
    """Tests for load_documents method."""

    def test_accepts_single_path(self, document_service):
        """Should accept a single path string."""
        with patch.object(document_service, 'load_pdf_document', return_value=[]):
            result = document_service.load_documents("/fake/path.pdf")
            assert isinstance(result, list)

    def test_accepts_list_of_paths(self, document_service):
        """Should accept a list of paths."""
        with patch.object(document_service, 'load_pdf_document', return_value=[]):
            result = document_service.load_documents(["/fake/path1.pdf", "/fake/path2.pdf"])
            assert isinstance(result, list)

    def test_handles_nonexistent_file(self, document_service):
        """Should log error and return empty for nonexistent files."""
        result = document_service.load_documents(["/nonexistent/file.pdf"])
        assert result == []

    def test_skips_unsupported_file_types(self, document_service):
        """Should skip unsupported file types with warning."""
        result = document_service.load_documents(["/some/file.txt", "/some/file.docx"])
        assert result == []

    def test_loads_pdf_files(self, document_service):
        """Should call load_pdf_document for PDF files."""
        mock_doc = Document(page_content="test", metadata={})
        with patch.object(document_service, 'load_pdf_document', return_value=[mock_doc]) as mock:
            result = document_service.load_documents(["/fake/test.pdf"])
            mock.assert_called_once_with("/fake/test.pdf")

    def test_loads_html_files(self, document_service):
        """Should call load_html_document for HTML files."""
        mock_doc = Document(page_content="test", metadata={})
        with patch.object(document_service, 'load_html_document', return_value=[mock_doc]) as mock:
            result = document_service.load_documents(["/fake/test.html"])
            mock.assert_called_once()


class TestSplitDocuments:
    """Tests for split_documents method."""

    def test_returns_empty_for_empty_input(self, document_service):
        """Should return empty list for empty input."""
        result = document_service.split_documents([])
        assert result == []

    def test_splits_long_document(self, document_service):
        """Should split documents longer than chunk_size."""
        long_doc = Document(page_content="A" * 2000, metadata={"source": "test"})
        result = document_service.split_documents([long_doc], chunk_size=500, chunk_overlap=50)
        assert len(result) > 1

    def test_preserves_short_document(self, document_service):
        """Should not split documents shorter than chunk_size."""
        short_doc = Document(page_content="Short content", metadata={"source": "test"})
        result = document_service.split_documents([short_doc], chunk_size=500)
        assert len(result) == 1

    def test_respects_chunk_size(self, document_service):
        """Chunks should not exceed chunk_size significantly."""
        long_doc = Document(page_content="Word " * 500, metadata={"source": "test"})
        result = document_service.split_documents([long_doc], chunk_size=100, chunk_overlap=10)
        
        for chunk in result:
            # Allow some flexibility due to splitter behavior
            assert len(chunk.page_content) <= 150

    def test_uses_default_parameters(self, document_service):
        """Should use default chunk_size and overlap."""
        doc = Document(page_content="Test content", metadata={})
        # Should not raise with defaults
        result = document_service.split_documents([doc])
        assert isinstance(result, list)

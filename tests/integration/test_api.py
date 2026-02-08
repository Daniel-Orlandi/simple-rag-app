"""Integration tests for FastAPI endpoints."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_returns_healthy_status(self, api_client):
        """Should return healthy status."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestModelsEndpoint:
    """Tests for /models endpoint."""

    def test_returns_available_models(self, api_client):
        """Should return available models."""
        response = api_client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "groq" in data
        assert "gemini" in data


class TestSessionEndpoint:
    """Tests for /session/{session_id} endpoint."""

    def test_delete_nonexistent_session_returns_404(self, api_client):
        """Should return 404 for nonexistent session."""
        response = api_client.delete("/session/nonexistent-session-id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestUploadEndpoint:
    """Tests for /upload endpoint."""

    def test_requires_session_id(self, api_client):
        """Should require session_id parameter."""
        response = api_client.post("/upload")
        
        # FastAPI returns 422 for missing required parameters
        assert response.status_code == 422

    def test_requires_files(self, api_client):
        """Should require files to be uploaded."""
        response = api_client.post("/upload", params={"session_id": "test"})
        
        assert response.status_code == 422

    def test_rejects_invalid_file_types(self, api_client):
        """Should reject non-PDF/HTML files."""
        files = [("files", ("test.txt", b"content", "text/plain"))]
        response = api_client.post(
            "/upload",
            params={"session_id": "test"},
            files=files
        )
        
        assert response.status_code == 400
        assert "No valid PDF or HTML" in response.json()["detail"]


class TestQuestionEndpoint:
    """Tests for /question endpoint."""

    def test_requires_session_id(self, api_client):
        """Should require session_id in request body."""
        response = api_client.post("/question", json={
            "question": "What is RAG?",
            "api_key": "fake-key"
        })
        
        assert response.status_code == 422

    def test_requires_question(self, api_client):
        """Should require question in request body."""
        response = api_client.post("/question", json={
            "session_id": "test",
            "api_key": "fake-key"
        })
        
        assert response.status_code == 422

    def test_requires_api_key(self, api_client):
        """Should require api_key in request body."""
        response = api_client.post("/question", json={
            "session_id": "test",
            "question": "What is RAG?"
        })
        
        assert response.status_code == 422

    def test_accepts_valid_request(self, api_client):
        """Should accept valid request with mocked service."""
        with patch("main.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.query_with_sources.return_value = ("Test answer", ["ref1"])
            mock_get_service.return_value = mock_service
            
            with patch("main.get_llm") as mock_get_llm:
                mock_get_llm.return_value = MagicMock()
                
                response = api_client.post("/question", json={
                    "session_id": "test",
                    "question": "What is RAG?",
                    "provider": "groq",
                    "model": "llama-3.1-8b-instant",
                    "api_key": "fake-key",
                    "temperature": 0.7
                })
                
                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
                assert "references" in data
                assert data["answer"] == "Test answer"

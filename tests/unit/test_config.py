"""Tests for RAGConfig."""
import pytest
from src.models.config import RAGConfig


class TestRAGConfig:
    """Tests for RAG configuration."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = RAGConfig()
        
        assert config.embedding_model == "intfloat/multilingual-e5-large"
        assert config.chunk_size == 700
        assert config.chunk_overlap == 100
        assert config.retrieval_k == 4
        assert config.retrieval_strategy == "similarity"

    def test_available_providers(self):
        """Should have groq and gemini providers."""
        config = RAGConfig()
        
        assert "groq" in config.available_providers
        assert "gemini" in config.available_providers
        assert config.available_providers["groq"]["name"] == "Groq"
        assert config.available_providers["gemini"]["name"] == "Google Gemini"

    def test_available_language_models(self):
        """Should have models for each provider."""
        config = RAGConfig()
        
        assert "groq" in config.available_language_models
        assert "gemini" in config.available_language_models
        assert len(config.available_language_models["groq"]) > 0
        assert len(config.available_language_models["gemini"]) > 0

    def test_model_kwargs_has_device(self):
        """Model kwargs should specify a device."""
        config = RAGConfig()
        
        assert "device" in config.model_kwargs

    def test_encode_kwargs_normalizes(self):
        """Encode kwargs should normalize embeddings."""
        config = RAGConfig()
        
        assert config.encode_kwargs.get("normalize_embeddings") is True

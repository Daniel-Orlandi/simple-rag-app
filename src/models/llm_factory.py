"""
LLM Factory module for creating language model instances.

This module provides a factory function to create LLM instances for different providers.
Provider and model options are defined in config.py - this module only handles instantiation.
"""
import logging
import os
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


def get_llm(
    provider: str,
    model: str,
    temperature: float = 0.5,
    api_key: str | None = None
) -> BaseChatModel:
    """
    Create an LLM instance for the specified provider.
    
    Args:
        provider: The LLM provider ('groq', 'google', or 'ollama')
        model: Model name specific to the provider
        temperature: Sampling temperature (0-1)
        api_key: Optional API key. If not provided, reads from environment variables.
    
    Returns:
        BaseChatModel: Configured LLM instance
    
    Raises:
        ValueError: If provider is unknown or API key is missing
    """
    logger.info(f"Initializing LLM: provider='{provider}', model='{model}', temperature={temperature}")
    
    if provider == "groq":
        logger.debug("Using Groq API")
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Groq API key is required. Pass api_key or set GROQ_API_KEY env var.")
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=key,  # type: ignore[arg-type]
        )
    
    elif provider == "gemini":
        logger.debug("Using Google Generative AI API")
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Gemini API key is required. Pass api_key or set GEMINI_API_KEY env var.")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=key,
        )
    
    elif provider == "ollama":
        logger.debug("Using Ollama (local)")
        return ChatOllama(
            model=model,
            temperature=temperature,
        )
    
    else:
        logger.critical(f"Unknown LLM provider: {provider}")
        raise ValueError(f"Unknown provider: {provider}")

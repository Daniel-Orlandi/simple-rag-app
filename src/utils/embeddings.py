"""
Embeddings utility module for creating HuggingFace embedding instances.

This module provides a convenient function to create and configure HuggingFace
embedding models with sensible defaults.
"""
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def get_embeddings(
    model_name: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    encode_kwargs: Optional[Dict[str, Any]] = None
) -> HuggingFaceEmbeddings:
    """
    Create and configure a HuggingFace embeddings instance.
    
    This function creates a HuggingFaceEmbeddings instance with sensible defaults.
    If arguments are not provided, it uses a multilingual model and auto-detects
    the device (CUDA if available, otherwise CPU).
    
    Args:
        model_name: Name of the HuggingFace embedding model to use.
                   If None, uses "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".
        model_kwargs: Dictionary of arguments for the embedding model.
                     If None, auto-detects device (CUDA/CPU).
                     Example: {"device": "cuda"} or {"device": "cpu"}.
        encode_kwargs: Dictionary of arguments for encoding embeddings.
                      If None, uses {"normalize_embeddings": True}.
                      Normalization is essential for cosine similarity search.
    
    Returns:
        HuggingFaceEmbeddings: Configured embeddings instance ready to use.
    
    Example:
        >>> embeddings = get_embeddings()
        >>> # Or with custom settings:
        >>> embeddings = get_embeddings(
        ...     model_name="ricardoz/BERTugues-base-portuguese-cased",
        ...     model_kwargs={"device": "cpu"},
        ...     encode_kwargs={"normalize_embeddings": True}
        ... )
    """
    if model_name is None:
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        logger.debug(f"Using default embedding model: {model_name}")
    
    if model_kwargs is None:
        # Auto-detect device
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.debug(f"Auto-detected device: {device}")
        except ImportError:
            device = "cpu"
            logger.warning("PyTorch not available, defaulting to CPU")
        model_kwargs = {"device": device}
    
    if encode_kwargs is None:
        encode_kwargs = {"normalize_embeddings": True}
    
    logger.info(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    logger.info("Embedding model loaded successfully")
    return embeddings


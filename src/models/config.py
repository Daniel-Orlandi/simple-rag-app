"""
Configuration module for RAG system settings.

This module defines the RAGConfig dataclass that holds all configuration parameters
for the RAG system, including paths, model settings, and retrieval parameters.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any
import torch


@dataclass
class RAGConfig:
    """
    Configuration dataclass for the RAG system.
    
    This class holds all configuration parameters needed to run the RAG system,
    including document paths, embedding model settings, chunking parameters,
    and retrieval strategy.
    
    Attributes:      
        embedding_model: Name of the HuggingFace embedding model to use.
                         Default: intfloat/multilingual-e5-large, because it is a multilingual model and it is fast.
        model_kwargs: Dictionary of arguments for the embedding model.
                     Default: Auto-detects CUDA/CPU device.
        encode_kwargs: Dictionary of arguments for encoding embeddings.
                      Default: Normalizes embeddings (essential for cosine similarity).
        chunk_size: Maximum size of document chunks in characters. Default: 700.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
                      Default: 100.
        retrieval_k: Number of documents to retrieve for each query. Default: 4.
        retrieval_strategy: Retrieval strategy to use. Options: "similarity" or "mmr".
                           Default: "similarity".
    

    """
   
    """HuggingFace model name for generating embeddings."""
    embedding_model: str = "intfloat/multilingual-e5-large"
    
    """Available providers."""
    available_providers: Dict = field(
        default_factory=lambda: {
                                    "groq": {
                                        "name": "Groq",
                                        "key_name": "GROQ_API_KEY",
                                        "get_key_url": "https://console.groq.com",
                                        
                                    },
                                    "gemini": {
                                        "name": "Google Gemini",
                                        "key_name": "GOOGLE_API_KEY",
                                        "get_key_url": "https://aistudio.google.com",
                                        
                                    },
                                }
        )   

    available_language_models: Dict = field(
            
        default_factory=lambda: {
                                            "groq": [
                                                ("openai/gpt-oss-120b", "openai/gpt-oss-120b"),
                                                ("llama-3.1-8b-instant", "llama-3.1-8b-instant"),
                                                ("groq/compound", "groq/compound"),
                                                
                                            ],
                                            "gemini": [
                                                ("gemini-3.0-pro-preview", "gemini-3.0-pro-preview"),
                                                ("gemini-3.0-flash-preview", "gemini-3.0-flash-preview"),
                                                ("gemini-2.5-flash", "Gemini 2.5 Flash "),
                                                ("gemini-2.5-pro", "Gemini 2.5 Pro"),
                                            ]
                                 }
    )
    """
    Arguments for the embedding model.
    
    Default: Auto-detects CUDA if available, otherwise uses CPU.
    Can be customized, e.g., {"device": "cpu"} to force CPU usage.
    """

    model_kwargs: Dict = field(
        default_factory=lambda: {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    )

    """
    Arguments for encoding embeddings.
    
    Default: Normalizes embeddings (essential for cosine similarity search).
    """
    encode_kwargs: Dict = field(
        default_factory=lambda: {"normalize_embeddings": True}
    )    
    
    """Maximum size of document chunks in characters."""
    chunk_size: int = 700
    
    """Number of characters to overlap between consecutive chunks."""
    chunk_overlap: int = 100

    """Number of documents to retrieve for each query."""
    retrieval_k: int = 4
    
    """
    Retrieval strategy to use.
    
    Options:
        - "similarity": Cosine similarity search (faster, may return similar documents)
        - "mmr": Maximum Marginal Relevance (slower, returns more diverse documents)
    """
    retrieval_strategy: str = "similarity"
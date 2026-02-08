import logging
import os
import uuid
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.services.rag_service import RAGService
from src.models.config import RAGConfig
from src.models.llm_factory import get_llm
from src.utils.logging_config import setup_logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Use configs from .env (load_dotenv() already called at top)
    log_level_str = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = os.getenv("LOG_FORMAT")  # e.g. from .env
    log_to_file = os.getenv("LOG_TO_FILE", "false").strip().lower() == "true"

    setup_logging(
        level=log_level,
        format_string=log_format if log_format else None,
        user_id="api_user" if log_to_file else None,
        session_id="api_session" if log_to_file else None,
    )
    yield

# Initialize FastAPI app
app = FastAPI(title="AI API", version="1.0.0", lifespan=lifespan)

# Configuration
UPLOAD_DIR = Path("data/upload")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize RAG service (lazy - will be created on first use)
_sessions: dict[str, RAGService] = {}


def get_rag_service(session_id: str) -> RAGService:
    """Get or create a RAG service instance for the given session."""
    if session_id not in _sessions:
        config = RAGConfig()
        service = RAGService(config, session_id=session_id)
        service.initialize_vectorstore()
        _sessions[session_id] = service
        logger.info(f"RAG service initialized for session: {session_id}")
    return _sessions[session_id]


# ============== API Models ==============

class QuestionRequest(BaseModel):
    session_id: str
    question: str
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    api_key: str
    temperature: float = 0.7


class QuestionResponse(BaseModel):
    answer: str
    references: List[str]


class UploadResponse(BaseModel):
    message: str
    documents_indexed: int
    total_chunks: int


# ============== API Endpoints ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up a session's resources."""
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"Session {session_id} cleaned up")
        return {"message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/upload", response_model=UploadResponse)
async def upload_files(session_id: str, files: list[UploadFile] = File(...)):
    """
    Upload and process PDF/HTML files.
    
    - Content-Type: multipart/form-data
    - Body: One or more PDF/HTML files under the field name 'files'
    
    Files are saved to: data/upload/{uuid}/filename.ext
    Then processed and indexed into the vector store.
    """
    # Generate unique folder for this upload batch
    upload_id = str(uuid.uuid4())
    upload_folder = UPLOAD_DIR / upload_id
    upload_folder.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for file in files:
        filename = file.filename
        
        # Skip files without a name
        if not filename:
            logger.warning("Skipping file with no filename")
            continue
        
        # Validate file type
        if not filename.endswith(('.pdf', '.html')):
            logger.warning(f"Skipping unsupported file: {filename}")
            continue
        
        # Save file
        file_path = upload_folder / filename
        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append(str(file_path))
            logger.info(f"Saved file: {file_path}")

        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save {filename}")
    
    if not saved_files:
        raise HTTPException(status_code=400, detail="No valid PDF or HTML files provided")
    
    # Process documents and add to vector store
    try:
        service = get_rag_service(session_id)
        
        # Load and split documents
        documents = service.document_service.load_documents(saved_files)
        splits = service.document_service.split_documents(
            documents,
            chunk_size=service.config.chunk_size,
            chunk_overlap=service.config.chunk_overlap
        )
        
        # Add to vector store
        service.vectorstore_service.add_documents(splits)
        
        logger.info(f"Indexed {len(documents)} documents with {len(splits)} chunks")
        
        return UploadResponse(
            message="Documents processed successfully",
            documents_indexed=len(documents),
            total_chunks=len(splits)
        )
    except Exception as e:
        logger.error(f"Failed to process documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {str(e)}")


@app.get("/models")
async def get_available_models():
    """Get available models for each provider."""    
    config = RAGConfig()
    return config.available_language_models


@app.post("/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded documents.
    
    - Content-Type: application/json
    - Body: {
        "question": "Your question here",
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "api_key": "your-api-key",
        "temperature": 0.7
      }
    
    Returns the answer and references from source documents.
    """
    try:
        service = get_rag_service(request.session_id)
        llm = get_llm(
            provider=request.provider,
            model=request.model,
            api_key=request.api_key,
            temperature=request.temperature
        )       
        
        # Generate answer
        answer, references = service.query_with_sources(request.question, llm)
        
        return QuestionResponse(
            answer=answer,
            references=references
        )

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============== Main (for testing) ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
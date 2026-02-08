# Simple RAG System

RAG (Retrieval-Augmented Generation) for Q&A over machine manuals (PDF/HTML). Upload documents, ask questions, get answers from LLMs with source references.

## Project Structure

```
simple-rag-system/
├── main.py                    # FastAPI (API server)
├── streamlit-app.py           # Streamlit frontend
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml, uv.lock
├── .env                       # Not committed
├── src/
│   ├── chains/rag_chain.py
│   ├── models/                # RAGConfig, LLM factory (Groq, Gemini, Ollama)
│   ├── services/              # Documents, RAG, retrieval, vectorstore (ChromaDB)
│   └── utils/                 # Embeddings, logging
├── tests/unit/, tests/integration/
└── data/upload/               # Uploaded documents (bind-mounted in Docker)
```

## Required Keys

At least one LLM provider (free tier):

| Provider | Key format | Where |
|----------|------------|--------|
| **Groq** | `gsk_...` | https://console.groq.com |
| **Google Gemini** | `AIza...` | https://aistudio.google.com |

Keys are entered in the Streamlit sidebar at runtime; not stored in `.env`.

## Upload Storage

Files go under **`data/upload/`**. Each upload gets a **UUID subfolder**; files keep their original names (e.g. `data/upload/<uuid>/manual.pdf`). Only **PDF** and **.html** are accepted. After save, documents are chunked and indexed into the session’s ChromaDB vector store for Q&A. In Docker, `./data/upload` is bind-mounted so uploads persist.

## Dependencies

Managed with [uv](https://github.com/astral-sh/uv); see `pyproject.toml`. Main: LangChain, ChromaDB, sentence-transformers, FastAPI, Streamlit, pypdf, beautifulsoup4/lxml. Dev: pytest, pytest-cov, pytest-asyncio, httpx.

## Logs

Configured in `src/utils/logging_config.py` and `.env`. Output always goes to **stdout**; optional **file** logs go to `logs/` when `LOG_TO_FILE=true` (files named `{user_id}_{session_id}_{date-time}.log`). Log level: `LOG_LEVEL` (e.g. `INFO`). In Docker, `logs/` is not mounted by default—add `./logs:/usr/src/app/logs` to `volumes` to persist.

## Install

- **Prerequisites:** Python ≥ 3.12, [uv](https://github.com/astral-sh/uv); Docker + Compose for container; GPU optional.
- **Local:** `git clone … && cd simple-rag-system && uv sync --all-extras`. Add `.env` with `API_URL=http://localhost:8000`, `LOG_LEVEL=INFO`, `LOG_TO_FILE=false`.

## Run

**Docker (recommended):** `docker compose up --build`. API: http://localhost:8000 · Streamlit: http://localhost:8501 · Docs: http://localhost:8000/docs. Set `RUN_TESTS=false` to skip tests on startup.

**Local:** Terminal 1: `uv run uvicorn main:app --host 0.0.0.0 --port 8000`. Terminal 2: `uv run streamlit run streamlit-app.py --server.port=8501 --server.address=0.0.0.0`. Open http://localhost:8501.

**Container tests:** With `RUN_TESTS=true` (default), pytest runs before API/Streamlit; on failure the container exits. To run tests manually in a running container: `docker exec rag-app uv run pytest tests/ -v`.

## Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

In Docker: `docker exec rag-app uv run pytest tests/ -v`.

## API (without Streamlit)

```bash
# Health
curl http://localhost:8000/health

# Upload (single or multiple files)
curl -X POST "http://localhost:8000/upload?session_id=my-session" -F "files=@manual.pdf"
curl -X POST "http://localhost:8000/upload?session_id=my-session" -F "files=@a.pdf" -F "files=@b.html"

# Question
curl -X POST http://localhost:8000/question -H "Content-Type: application/json" \
  -d '{"session_id":"my-session","question":"How do I replace the hydraulic filter?","provider":"groq","model":"llama-3.1-8b-instant","api_key":"gsk_...","temperature":0.7}'

# Models, delete session
curl http://localhost:8000/models
curl -X DELETE http://localhost:8000/session/my-session
```

Swagger UI: http://localhost:8000/docs.

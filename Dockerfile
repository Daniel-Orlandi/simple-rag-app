FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/usr/src/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/usr/src/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \    
    && rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Add uv to PATH (installed to /root/.local/bin)
ENV PATH="/root/.local/bin:/usr/src/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /usr/src/app

# Copy dependency files first (for better caching)
COPY pyproject.toml uv.lock ./

# Install all dependencies including dev (for tests)
RUN uv sync --all-extras

# Copy application code
COPY . .

EXPOSE 8000
EXPOSE 8501
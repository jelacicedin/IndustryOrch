# EdinTech-RAG API server Dockerfile
# ===================================

FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps for asyncpg (libpq headers) and pdfplumber (poppler)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc libpq-dev poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY converter.py chunker.py server.py ingest.py ./

# Expose API port
EXPOSE 8000

ENTRYPOINT ["python", "-m", "server"]
CMD []

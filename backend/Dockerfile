# IndustryOrch API server Dockerfile
# ===================================

FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps for asyncpg (libpq headers) and pdfplumber (poppler)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc libpq-dev poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first so pip install is cached independently of build args
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# ARGs declared here so changing them only busts the GLiNER download layer,
# not the pip install layer above.
# Allow overriding the GLiNER model at build time:
#   docker compose build --build-arg GLINER_MODEL=urchade/gliner_small-v2.1 app
ARG GLINER_MODEL=urchade/gliner_medium-v2.1
# Optional: set to avoid HuggingFace rate limits during build
ARG HF_TOKEN=
RUN HF_TOKEN=$HF_TOKEN python -c "from gliner import GLiNER; GLiNER.from_pretrained('$GLINER_MODEL')"

# Copy application code
COPY converter.py chunker.py graph_extractor.py neo4j_client.py server.py ingest.py ./

# Expose API port
EXPOSE 8000

ENTRYPOINT ["python", "-m", "server"]
CMD []

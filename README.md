# EdinTech-RAG

Interactive technical document analyst app for industrial equipment documentation.

A fully local RAG (Retrieval-Augmented Generation) system that ingests PDFs, spreadsheets, and CSV files from industrial environments — then answers questions using hybrid search and a local LLM with extended thinking capabilities.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Documents   │───▶│  Converter       │───▶│  Chunker        │
│  (PDF/XLSX/  │     │  → Structured    │     │  → Chunks +     │
│   CSV)       │     │    Markdown      │     │    Embeddings   │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  FastAPI     │◀───│  Hybrid Search   │◀───│  PostgreSQL     │
│  REST API    │     │  (RRF)           │     │  + pgvector     │
│              │     └──────────────────┘     │  (chunks table) │
└──────┬───────┘                              └─────────────────┘
       │
       ▼
┌──────────────────┐    ┌──────────────────────┐
│  Ollama          │    │  llama.cpp server    │
│  Embedding Model │    │                      │
└──────────────────┘    └──────────────────────┘
       ▲
       │ GEN_BACKEND=ollama  OR  GEN_BACKEND=llama_cpp (depending on user preference)
┌──────┴─────────────────────────────────────────┐
│  Generation Backend (configurable)             │
│  - Ollama: /api/chat endpoint                  │
│  - llama.cpp: /v1/chat/completions (OpenAI API)│
└────────────────────────────────────────────────┘
```

## Features

- **Hybrid Search**: Combines semantic (embedding cosine similarity) and keyword (full-text search) retrieval using Reciprocal Rank Fusion (RRF).
- **Document Management**: Upload, list, view, and delete documents with metadata.
- **Chunk-Aware Retrieval**: Content-aware chunking that preserves table integrity and uses heading boundaries for prose.
- **Extended Thinking**: Qwen3 models can generate chain-of-thought reasoning traces (toggleable per request).
- **Dual Generation Backends**: Switch between Ollama and llama.cpp via a single env var — no code changes needed.
- **Fully Local**: All data stays on-premises — no external API calls.

## Quick Start

### Prerequisites

**Option A — Ollama (default):**

- Docker & Docker Compose
- Ollama installed locally, or use the containerized version
- Models pulled:
  ```bash
  ollama pull qwen3-embedding:0.6b   # embedding model
  ollama pull qwen3.6:27b            # generation model
  ```

**Option B — llama.cpp:**

- Docker & Docker Compose
- A GGUF model file (e.g., `qwen3-27b.Q4_K_M.gguf`)
- llama.cpp built and running a server on port 8080:
  ```bash
  llama-server \
    -m "/path/to/models/$MODEL_FILE" \
    --port 8080 \
    --host 0.0.0.0 \
    --n-gpu-layers 999 \
    -c 32768 \
    --temp 0.6 \
    --top-k 20 \
    --top-p 0.95 \
    --presence-penalty 1.5
  ```

### Docker Compose (Recommended)

```bash
# 1. Copy and customize environment
cp .env.example .env

# 2. Start all services
docker compose up -d

# 3. Verify health
curl http://localhost:8000/health
```

### Local Development

```bash
# 1. Create virtual environment
cd backend
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r ../requirements.txt

# 3. Set up database (requires PostgreSQL with pgvector running)
psql -U postgres -c "CREATE DATABASE edintechrag;"
psql -U postgres -d edintechrag -f ../supabase/20260505_init.sql

# 4. Start the API server
uvicorn server:app --reload --port 8000

# 5. Run ingestion worker (in another terminal)
python ingest.py --dir ./my-documents
```

## Generation Backend Configuration

The `GEN_BACKEND` environment variable switches between two generation backends. **Embeddings always use Ollama** — only the chat/completion endpoint changes.

### Using Ollama (default)

```bash
# .env
GEN_BACKEND=ollama
GENERATION_MODEL=qwen3.6:27b
OLLAMA_THINK=true          # enable extended thinking by default
```

The server talks to Ollama's `/api/chat` endpoint with the `thinking` option enabled for chain-of-thought reasoning.

### Using llama.cpp

```bash
# .env
GEN_BACKEND=llama_cpp
LLAMA_SERVER_URL=http://localhost:8080
OLLAMA_THINK=true          # still sends thinking hints to llama-server
```

The server talks to llama.cpp's OpenAI-compatible `/v1/chat/completions` endpoint. The `--temp`, `--top-k`, `--top-p`, and `--presence-penalty` flags are passed per-request (matching the default llama-server invocation). Extended thinking is sent via the `reasoning_effort` / `thinking` extra parameters.

### Starting llama.cpp server locally

```bash
export MODEL_PATH=/path/to/models
export MODEL_FILE=qwen3-27b.Q4_K_M.gguf

llama-server \
  -m "$MODEL_PATH/$MODEL_FILE" \
  --port 8080 \
  --host 0.0.0.0 \
  --n-gpu-layers 999 \
  -c 32768 \
  --temp 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --presence-penalty 1.5
```

## Ingesting Documents

### Via Docker Compose Worker

Place documents in the `ingest/` directory and run:

```bash
docker compose up ingest
```

### Via API Endpoint

```bash
curl -X POST "http://localhost:8000/documents/upload?file_path=./manuals/pump-spec.pdf&category=manual"
```

### Via CLI (Local)

```bash
python backend/ingest.py --dir ./documents --category manual
```

## API Reference

### Health Check

```
GET /health
```

Returns PostgreSQL and generation backend connectivity status, model availability, and document counts.

Response:
```json
{
  "postgres_ok": true,
  "ollama_ok": false,
  "embed_model_available": true,
  "generation_backend": "llama_cpp",
  "generation_ok": true,
  "total_documents": 12,
  "total_chunks": 847,
  "message": "healthy"
}
```

### Query Documents

```
POST /query
Content-Type: application/json

{
  "question": "What is the maximum operating pressure for Pump P-101?",
  "filters": {
    "equipment_id": null,
    "document_category": "manual",
    "file_type": "pdf",
    "location": null
  },
  "top_k": 5,
  "show_thinking": true
}
```

Response:

```json
{
  "answer": "According to the P-101 maintenance manual (Section 3.2), the maximum operating pressure is 150 bar...",
  "thinking": "Let me analyze the question step by step...<br>I need to find information about Pump P-101's pressure limits...",
  "sources": [
    {
      "filename": "pump-spec.pdf",
      "document_category": "manual",
      "equipment_id": null,
      "section": "Section 3.2 — Operating Limits",
      "chunk_index": 5,
      "score": 0.8472
    }
  ]
}
```

### List Documents

```
GET /documents
```

Returns all ingested documents with metadata and chunk counts.

### Get Document Details

```
GET /documents/{doc_id}
```

### Delete Document

```
DELETE /documents/{doc_id}
```

### List Document Chunks

```
GET /documents/{doc_id}/chunks?page=1&per_page=20
```

## Database Schema

The schema is defined in `supabase/20260505_init.sql`. Key tables:

- **`documents`**: Stores file metadata, converted markdown content, and ingestion timestamps.
- **`chunks`**: Stores chunked text content with 1024-dimensional embeddings (qwen3-embedding) and full-text search vectors.

A PL/pgSQL function `hybrid_search()` implements RRF-based hybrid retrieval with optional filters on equipment ID, document category, file type, and location.

## Configuration

| Variable             | Default                                          | Description                                      |
|----------------------|--------------------------------------------------|--------------------------------------------------|
| `POSTGRES_USER`      | `edintech`                                       | PostgreSQL username                              |
| `POSTGRES_PASSWORD`  | `password`                                       | PostgreSQL password                              |
| `POSTGRES_DB`        | `edintechrag`                                    | Database name                                    |
| `OLLAMA_URL`         | `http://host.docker.internal:11434`              | Ollama API endpoint (embeddings + optional gen)  |
| `EMBED_MODEL`        | `qwen3-embedding:0.6b`                           | Embedding model                                  |
| `GENERATION_MODEL`   | `qwen3.6:27b`                                    | Ollama generation model (only when ollama backend) |
| `OLLAMA_THINK`       | `true`                                           | Enable extended thinking by default              |
| `GEN_BACKEND`        | `ollama`                                         | Generation backend: `ollama` or `llama_cpp`      |
| `LLAMA_SERVER_URL`   | `http://localhost:8080`                          | llama.cpp server URL (only when llama_cpp backend) |
| `APP_PORT`           | `8000`                                           | FastAPI server port                              |

## Project Structure

```
EdinTech-RAG/
├── backend/
│   ├── converter.py      # PDF/XLSX/CSV → Markdown conversion
│   ├── chunker.py        # Markdown → Chunks + Embeddings
│   ├── server.py         # FastAPI REST API (dual backend support)
│   └── Dockerfile        # Server container image
├── frontend/
│   ├── frontend.py       # Streamlit chat + document management UI
│   └── Dockerfile        # Frontend container image
├── supabase/
│   ├── 20260505_init.sql        # Database schema + hybrid_search function
│   └── 20260506_fix_tsquery.sql # Migration: plainto_tsquery fix
├── docker-compose.yml    # Orchestration (PostgreSQL, App, Streamlit)
├── .env.example          # Environment variable template
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

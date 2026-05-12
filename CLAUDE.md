# EdinTech-RAG — Claude Instructions

## What this project is

Local RAG (Retrieval-Augmented Generation) system for industrial document Q&A.
Documents (PDF, XLSX, CSV) are ingested, chunked, embedded, and stored in PostgreSQL
with pgvector. Questions are answered via hybrid semantic + keyword search fed into a
local LLM.

## Architecture at a glance

```
Upload → converter.py (markdown) → chunker.py (chunks + embeddings)
       → PostgreSQL (documents + chunks tables)

Question → embed (Ollama) → hybrid_search() PL/pgSQL (RRF: vector + FTS)
         → top-K chunks → LLM (Ollama or llama.cpp) → answer + sources
```

## Services (docker-compose)

| Container | Port | Role |
|-----------|------|------|
| `edintech-postgres` | 5432 | PostgreSQL 17 + pgvector |
| `edintech-app` | 8000 (APP_PORT) | FastAPI backend |
| `edintech-streamlit` | 8501 | Streamlit frontend |

All services use `network_mode: host`. Ollama runs on the **host machine** at
`localhost:11434` — it is NOT a Docker service.

## Key files

| File | Purpose |
|------|---------|
| `backend/server.py` | FastAPI app — `/query`, `/ingest`, `/ingest/status`, `/documents`, `/health` |
| `backend/converter.py` | PDF/XLSX/CSV → structured markdown (PyMuPDF + pdfplumber) |
| `backend/chunker.py` | Markdown → chunks → Ollama embeddings → bulk INSERT |
| `frontend/frontend.py` | Streamlit UI (chat + upload + document library) |
| `supabase/20260505_init.sql` | DB schema + `hybrid_search()` PL/pgSQL function |
| `supabase/20260506_fix_tsquery.sql` | Hotfix migration — apply to running DBs |
| `.env` | Active config (never commit) |
| `.env.example` | Config template with all available variables |

## Environment variables (`.env`)

```
GEN_BACKEND=ollama          # "ollama" or "llama_cpp"
GENERATION_MODEL=rnj-1:8b  # Ollama model for generation
EMBED_MODEL=qwen3-embedding:0.6b
VISION_MODEL=               # Optional: Ollama vision model for PDF image description
OLLAMA_THINK=true           # Enable extended thinking (auto-disabled if model rejects it)
APP_PORT=8000
```

`llama_cpp` backend uses `LLAMA_SERVER_URL` (default `http://localhost:8080`) and
requires the llama-server service to be running (commented out in docker-compose by
default).

## Common commands

```bash
# Start everything
docker compose up -d

# Rebuild backend after code changes
docker compose build app && docker compose up -d app

# Follow logs
docker compose logs -f app
docker compose logs -f streamlit

# Apply a DB migration to a running database
docker exec -i edintech-postgres psql -U edintech edintechrag \
  < supabase/20260506_fix_tsquery.sql

# Check system health
curl -s http://localhost:8000/health | python3 -m json.tool

# List available Ollama models
curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

## Important invariants — do not break these

- **FTS always uses `plainto_tsquery()`**, never `to_tsquery()`. The latter crashes on
  punctuation in natural language questions.
- **Query embeddings use `keep_alive=0`** so the embedding model is evicted from VRAM
  before the generation model loads. Do not remove this.
- **Chunker uses pdfplumber for tables, PyMuPDF for text + images.** pdfplumber's table
  extraction is more accurate for complex layouts; PyMuPDF is faster for plain text.
- **`chunk_and_insert()` takes a psycopg (sync) connection**, not asyncpg. The ingest
  endpoint spawns a background thread so it doesn't block the async event loop.
- **`_no_think_models` cache** (module-level set in server.py) remembers which models
  rejected `think=True` (400 response) and skips the parameter on future calls.

## Database schema notes

- `chunks.fts` is a generated `tsvector` column — it updates automatically on INSERT.
- `chunks.embedding` is `vector(1024)` — changing the embedding model requires
  dropping and re-inserting all chunks (dimension must match).
- The `hybrid_search()` function lives in PostgreSQL. After editing the SQL, apply via
  `CREATE OR REPLACE FUNCTION` — no table migration needed.
- Enum types (`file_type`, `document_category`) require `ALTER TYPE … ADD VALUE` to
  extend; you cannot `CREATE OR REPLACE TYPE`.

## PDF ingestion pipeline detail

1. **Conversion** (converter.py): per-page — text via PyMuPDF, tables via pdfplumber,
   images via PyMuPDF. If `VISION_MODEL` is set, images are described by the vision model.
2. **Chunking** (chunker.py): prose ~500 tokens with overlap; tables kept whole or split
   by row groups; sections split by H1–H3 headings.
3. **Embedding**: batched (EMBED_BATCH_SIZE=50 chunks/call) via Ollama.
4. **Insertion**: batched (DB_BATCH_SIZE=200 chunks/INSERT) via psycopg.

## Frontend notes

- Ingestion progress is shown in the **sidebar only** — never in the main chat area.
- Source keys from the API are `doc_filename`, `doc_category`, `rrf_score` — not
  `filename`, `document_category`, `score`.
- Polling uses `time.sleep(0.8)` + `st.rerun()` and only runs while jobs are active.

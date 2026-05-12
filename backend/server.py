# Copyright (C) 2026  Edin Jelacic — AGPL-3.0-or-later
"""EdinTech-RAG — FastAPI server for industrial document Q&A.

Provides a REST API that embeds user questions, performs hybrid search
(semantic + keyword via RRF) against the ingested document database, and
uses a local LLM to generate grounded answers with citations.

Generation backend: Ollama (default) or llama.cpp llama-server.
Embeddings always use Ollama.

Run in development:
    uvicorn server:app --reload

Run in production (Docker):
    docker-compose up app
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
import tempfile
from pathlib import Path
from typing import Any

import httpx
import ollama
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

env_db_url = os.environ.get(
    "DATABASE_URL",
    "postgresql://edintech:password@localhost:5432/edintechrag",
)
ollama_url = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
embed_model = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")

# Generation backend: "ollama" or "llama_cpp"
gen_backend = os.environ.get("GEN_BACKEND", "ollama").lower().strip()

# Ollama generation config (used when GEN_BACKEND=ollama)
gen_model = os.environ.get("GENERATION_MODEL", "qwen3.6:27b")
ollama_think = os.environ.get("OLLAMA_THINK", "true").lower() in ("true", "1", "yes")

# llama.cpp server config (used when GEN_BACKEND=llama_cpp)
llama_server_url = os.environ.get(
    "LLAMA_SERVER_URL",
    "http://localhost:8080",
)

# The Python 'ollama' library reads OLLAMA_HOST (not OLLAMA_BASE_URL)
os.environ["OLLAMA_HOST"] = ollama_url.rstrip("/")

logger = logging.getLogger("edintech-rag")

# In-memory progress store for ingest polling
_ingest_progress: dict[str, dict] = {}

# Models that returned 400 when think=True — skip thinking for them on future calls
_no_think_models: set[str] = set()


# ---------------------------------------------------------------------------
# Database helper (asyncpg)
# ---------------------------------------------------------------------------

import asyncpg  # noqa: E402


def _sanitize_query_text(text: str) -> str:
    """Strip characters that break tsquery parsing (punctuation, special chars)."""
    return re.sub(r"[^\w\s]", " ", text).strip()


_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None or _pool.get_size() == 0:
        _pool = await asyncpg.create_pool(dsn=env_db_url)
    return _pool


async def _close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Filters(BaseModel):
    equipment_id: str | None = None
    document_category: str | None = None
    file_type: str | None = None
    location: str | None = None


class QueryRequest(BaseModel):
    question: str
    filters: Filters = Field(default_factory=Filters)
    top_k: int = 5
    show_thinking: bool = False


class Source(BaseModel):
    doc_filename: str
    doc_category: str
    doc_equipment_id: str | None
    section: str
    chunk_index: int
    rrf_score: float


class QueryResponse(BaseModel):
    answer: str
    thinking: str | None = None
    sources: list[Source]


class DocumentInfo(BaseModel):
    id: int
    filename: str
    file_type: str
    document_category: str
    equipment_id: str | None
    location: str | None
    revision: str | None
    document_date: str | None
    ingested_at: str
    chunk_count: int


class HealthStatus(BaseModel):
    postgres_ok: bool
    ollama_ok: bool
    embed_model_available: bool
    generation_backend: str
    generation_ok: bool
    total_documents: int
    total_chunks: int
    message: str


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate all external dependencies on startup."""
    await _verify_startup()
    logger.info("EdinTech-RAG server started successfully (backend=%s).", gen_backend)
    yield
    await _close_pool()
    logger.info("EdinTech-RAG server shutting down.")


async def _verify_startup() -> None:
    """Check PostgreSQL and generation backend connectivity."""
    # --- PostgreSQL ---
    try:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            version = await conn.fetchval("SELECT version()")
            logger.info("PostgreSQL connected — %s", version.split("\n")[0])
    except Exception as exc:
        logger.error("PostgreSQL connection failed: %s", exc)
        raise RuntimeError(
            f"Cannot connect to PostgreSQL at {env_db_url}: {exc}"
        ) from exc

    # --- Embedding backend (always Ollama) ---
    try:
        resp_data = ollama.list()
        models = resp_data.get("models", []) or []
        model_names = [m.get("name", "") or m.get("model", "") for m in models]
        logger.info(
            "Ollama reachable — %d model(s) available: %s",
            len(model_names),
            model_names,
        )
    except Exception as exc:
        logger.error("Ollama not reachable at %s — %s", ollama_url, exc)

    # --- Generation backend ---
    if gen_backend == "llama_cpp":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{llama_server_url}/health")
                logger.info(
                    "llama.cpp server reachable at %s (status=%d)",
                    llama_server_url,
                    resp.status_code,
                )
        except Exception as exc:
            logger.error(
                "llama.cpp server not reachable at %s — %s", llama_server_url, exc
            )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EdinTech-RAG",
    description="Industrial document Q&A via local RAG with hybrid search.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


async def _hybrid_search(
    query_text: str,
    embedding: list[float],
    top_k: int = 5,
    filters: Filters | None = None,
) -> list[dict[str, Any]]:
    """Execute the hybrid_search PL/pgSQL function with RRF."""
    pool = await _get_pool()

    if filters is None:
        filters = Filters()

    # The PL/pgSQL function signature:
    # hybrid_search(query_text, query_embedding, match_count,
    #               rrf_k=60, filter_equipment_id, filter_document_category,
    #               filter_file_type, filter_location)
    sql = """
        SELECT chunk_id, chunk_document_id, chunk_content, chunk_metadata,
               doc_filename, doc_category, doc_equipment_id, doc_location, rrf_score
        FROM hybrid_search(
            $1, $2, $3, 60,   -- query_text, embedding, top_k, rrf_k
            $4, $5, $6, $7    -- filter_equipment_id, category, file_type, location
        )
        ORDER BY rrf_score DESC
        LIMIT $8
    """

    async with pool.acquire() as conn:
        # Convert embedding list to string representation for pgvector
        embed_str = "[" + ", ".join(str(v) for v in embedding) + "]"
        rows = await conn.fetch(
            sql,
            query_text,
            embed_str,
            top_k,
            filters.equipment_id,
            filters.document_category,
            filters.file_type,
            filters.location,
            top_k,
        )

    results = []
    for row in rows:
        meta = row["chunk_metadata"] or {}
        section = (
            meta.get("section_heading", "Unknown")
            if isinstance(meta, dict)
            else "Unknown"
        )
        chunk_index = meta.get("chunk_index", 0) if isinstance(meta, dict) else 0
        results.append(
            {
                "chunk_id": int(row["chunk_id"]),
                "chunk_document_id": int(row["chunk_document_id"]),
                "chunk_content": row["chunk_content"],
                "doc_filename": row["doc_filename"],
                "doc_category": str(row["doc_category"]),
                "doc_equipment_id": row["doc_equipment_id"],
                "section": section,
                "chunk_index": chunk_index,
                "rrf_score": float(row["rrf_score"]),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Embedding helper (always Ollama)
# ---------------------------------------------------------------------------


async def _embed_text(text: str) -> list[float]:
    """Embed text, then immediately unload the embedding model to free VRAM for generation."""
    resp = ollama.embed(model=embed_model, input=text, keep_alive=0)
    return resp["embeddings"][0]


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------


def _build_system_prompt() -> str:
    return (
        "You are EdinTech-RAG, an expert technical assistant for industrial "
        "equipment documentation. Answer the user's question using ONLY the "
        "provided context from company documents. If the context does not "
        "contain sufficient information to answer fully, say so clearly and "
        "list what information is missing. Always cite your sources by filename "
        "and section heading. Be precise, technical, and concise."
    )


def _build_user_prompt(question: str, sources: list[dict[str, Any]]) -> str:
    context_parts = []
    for i, src in enumerate(sources):
        context_parts.append(
            f"[Source {i+1}] File: {src['doc_filename']}, "
            f"Section: {src['section']}, Score: {src['rrf_score']:.4f}\n"
            f"{src['chunk_content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return (
        f"Question: {question}\n\n"
        f"Relevant document excerpts:\n\n{context}\n\n"
        "Provide a clear, technical answer based on the excerpts above."
    )


async def _generate_with_ollama(
    system_prompt: str,
    user_prompt: str,
    show_thinking: bool = False,
) -> tuple[str, str | None]:
    """Generate an answer using Ollama with optional extended thinking.

    If the model returns 400 because it does not support thinking, the call is
    retried automatically without think=True and the model is remembered so
    subsequent queries skip the think parameter from the start.
    """
    use_thinking = (show_thinking or ollama_think) and gen_model not in _no_think_models

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def _do_chat(think: bool) -> Any:
        return ollama.chat(
            model=gen_model,
            messages=messages,
            think=think,
            options={"num_predict": 2048, "temperature": 0.1},
        )

    try:
        response = _do_chat(use_thinking)
    except Exception as exc:
        # Ollama returns 400 when the model does not support extended thinking.
        # Detect by status code (ResponseError) or error text, then fall back.
        status = getattr(exc, "status_code", None)
        is_think_error = (
            (status == 400 or "400" in str(exc))
            and use_thinking
            and "think" in str(exc).lower()
        )
        if is_think_error:
            _no_think_models.add(gen_model)
            logger.info(
                "Model '%s' does not support thinking — retrying without it.", gen_model
            )
            response = _do_chat(False)
        else:
            raise

    msg = response.message
    answer = msg.content or ""
    # Prefer Ollama's native thinking field; fall back to <think> tag parsing
    thinking: str | None = msg.thinking if msg.thinking else None
    if not thinking:
        m = re.search(r"<think>(.*?)</think>", answer, re.DOTALL)
        if m:
            thinking = m.group(1).strip()
            answer = re.sub(
                r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL
            ).strip()

    return answer.strip(), thinking


async def _generate_with_llama_cpp(
    system_prompt: str,
    user_prompt: str,
    show_thinking: bool = False,
) -> tuple[str, str | None]:
    """Generate an answer using llama.cpp server (OpenAI-compatible API).

    llama-server exposes an /v1/chat/completions endpoint compatible with
    the OpenAI Python client. We use httpx directly to avoid adding another
    dependency for the optional backend.
    """
    system_msg = {"role": "system", "content": system_prompt}
    user_msg = {"role": "user", "content": user_prompt}

    payload = {
        "model": "",  # llama-server ignores model name on single-model instances
        "messages": [system_msg, user_msg],
        "temperature": 0.1,
        "max_tokens": 4096,
        "top_k": 20,
        "top_p": 0.95,
        "presence_penalty": 1.5,
    }

    # Extended thinking: prepend reasoning budget for Qwen3-style models
    if show_thinking or ollama_think:
        payload["reasoning_effort"] = "high"
        # Some llama.cpp builds support the 'thinking' parameter via extra_body
        payload["extra_body"] = {"thinking": {"budget_tokens": 2048}}

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{llama_server_url.rstrip('/')}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    answer = data["choices"][0]["message"]["content"]
    thinking = None

    # Extract thinking content if present (model wraps it in tags)
    thinking_match = re.search(r"<think>(.*?)</think>", answer, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL).strip()

    return answer, thinking


# Map backend name → coroutine
_GENERATORS = {
    "ollama": _generate_with_ollama,
    "llama_cpp": _generate_with_llama_cpp,
}


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthStatus)
async def health():
    """Health check — verifies PostgreSQL and generation backend."""
    postgres_ok = False
    total_documents = 0
    total_chunks = 0

    try:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            total_documents = await conn.fetchval("SELECT COUNT(*) FROM documents")
            total_chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        postgres_ok = True
    except Exception:
        pass

    ollama_ok = False
    embed_available = False
    gen_ok = False

    # Embeddings always use Ollama — check it regardless of generation backend
    try:
        resp_data = ollama.list()
        models = resp_data.get("models", [])
        if models is None:
            models = []
        model_names = [m.get("name", "") or m.get("model", "") for m in models]
        ollama_ok = True

        # Match by model name prefix (e.g. "qwen3-embedding:0.6b" matches "qwen3-embedding")
        embed_prefix = embed_model.split(":")[0]
        gen_prefix = gen_model.split(":")[0]
        embed_available = any(
            n.startswith(embed_prefix + ":") or n == embed_prefix for n in model_names
        )
        gen_ok = any(
            n.startswith(gen_prefix + ":") or n == gen_prefix for n in model_names
        )
    except Exception:
        pass

    # If using llama_cpp backend, also verify the llama.cpp server is reachable
    if gen_backend == "llama_cpp":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{llama_server_url}/health")
                gen_ok = resp.status_code < 400
        except Exception:
            pass

    status = "healthy" if (postgres_ok and gen_ok) else "degraded"
    return HealthStatus(
        postgres_ok=postgres_ok,
        ollama_ok=ollama_ok,
        embed_model_available=embed_available,
        generation_backend=gen_backend,
        generation_ok=gen_ok,
        total_documents=total_documents,
        total_chunks=total_chunks,
        message=status,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question against the document corpus.

    Performs hybrid search (semantic + keyword via RRF), then generates an
    answer using the configured generation backend with optional extended
    thinking mode.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Step 1: Embed the question
    try:
        embedding = await _embed_text(request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding failed: {exc}",
        ) from exc

    # Step 2: Hybrid search
    try:
        sources = await _hybrid_search(
            query_text=_sanitize_query_text(request.question),
            embedding=embedding,
            top_k=request.top_k,
            filters=request.filters,
        )
    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Search failed: {exc}") from exc

    if not sources:
        return QueryResponse(
            answer="No relevant documents found for your question.",
            sources=[],
        )

    # Step 3: Generate answer using selected backend
    gen_fn = _GENERATORS.get(gen_backend)
    if gen_fn is None:
        raise HTTPException(
            status_code=500,
            detail=f"Unknown generation backend: {gen_backend}. "
            f"Supported: {list(_GENERATORS.keys())}",
        )

    try:
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(request.question, sources)
        answer, thinking = await gen_fn(
            system_prompt, user_prompt, show_thinking=request.show_thinking
        )
    except Exception as exc:
        logger.error("Generation failed (%s): %s", gen_backend, exc)
        raise HTTPException(
            status_code=503, detail=f"Generation failed: {exc}"
        ) from exc

    return QueryResponse(
        answer=answer,
        thinking=thinking if request.show_thinking else None,
        sources=[Source(**s) for s in sources],
    )


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List all ingested documents with metadata and chunk counts."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.id, d.filename, d.file_type, d.document_category,
                   d.equipment_id, d.location, d.revision,
                   d.document_date, d.ingested_at,
                   COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.ingested_at DESC
            """
        )

    return [
        DocumentInfo(
            id=int(r["id"]),
            filename=r["filename"],
            file_type=str(r["file_type"]),
            document_category=str(r["document_category"]),
            equipment_id=r["equipment_id"],
            location=r["location"],
            revision=r["revision"],
            document_date=str(r["document_date"]) if r["document_date"] else None,
            ingested_at=str(r["ingested_at"]),
            chunk_count=int(r["chunk_count"]),
        )
        for r in rows
    ]


@app.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: int):
    """Get a single document by ID."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT d.id, d.filename, d.file_type, d.document_category,
                   d.equipment_id, d.location, d.revision,
                   d.document_date, d.ingested_at,
                   COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            WHERE d.id = $1
            GROUP BY d.id
            """,
            doc_id,
        )

    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentInfo(
        id=int(row["id"]),
        filename=row["filename"],
        file_type=str(row["file_type"]),
        document_category=str(row["document_category"]),
        equipment_id=row["equipment_id"],
        location=row["location"],
        revision=row["revision"],
        document_date=str(row["document_date"]) if row["document_date"] else None,
        ingested_at=str(row["ingested_at"]),
        chunk_count=int(row["chunk_count"]),
    )


@app.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: int):
    """Delete a document and all its chunks (cascade)."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM documents WHERE id = $1", doc_id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Document not found")


@app.get("/documents/{doc_id}/chunks")
async def list_chunks(doc_id: int, page: int = 1, per_page: int = 20):
    """List chunks for a specific document with pagination."""
    pool = await _get_pool()
    offset = (page - 1) * per_page

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, metadata
            FROM chunks
            WHERE document_id = $1
            ORDER BY id
            LIMIT $2 OFFSET $3
            """,
            doc_id,
            per_page,
            offset,
        )

        total = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE document_id = $1", doc_id
        )

    return {
        "document_id": doc_id,
        "page": page,
        "per_page": per_page,
        "total": total,
        "chunks": [
            {
                "id": int(r["id"]),
                "content": r["content"],
                "metadata": r["metadata"] or {},
            }
            for r in rows
        ],
    }


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    category: str = Form(default="other"),
    equipment_id: str | None = Form(default=None),
    location: str | None = Form(default=None),
    revision: str | None = Form(default=None),
):
    """Upload and ingest a document via multipart form data.

    Accepts a file plus optional metadata fields. Returns a job ID that
    can be polled for progress via GET /ingest/status/{job_id}.
    """
    import io
    import threading
    import uuid

    # Read file content synchronously before spawning thread (file may not be seekable)
    content = await file.read()

    job_id = str(uuid.uuid4())
    _ingest_progress[job_id] = {
        "status": "queued",
        "message": f"Queued **{file.filename}**",
        "filename": file.filename,
        "stage": None,
        "result": None,
    }

    def _run_ingest():
        from converter import convert_file  # noqa: E402
        from chunker import chunk_and_insert  # noqa: E402

        tmp_path = None
        doc_id = None
        chunks_count = 0

        try:
            _ingest_progress[job_id]["status"] = "processing"
            _ingest_progress[job_id]["message"] = f"Saving **{file.filename}** …"
            logger.info("[ingest] Saving %s", file.filename)

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.filename).suffix
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            _ingest_progress[job_id]["stage"] = "converting"
            _ingest_progress[job_id]["message"] = f"Converting **{file.filename}** …"
            logger.info("[ingest] Converting %s", file.filename)

            path = Path(tmp_path)
            file_type = path.suffix.lstrip(".").lower()

            def _conv_progress(msg: str) -> None:
                _ingest_progress[job_id]["message"] = msg

            try:
                markdown, metadata = convert_file(
                    str(path), progress_callback=_conv_progress
                )
            except Exception as exc:
                logger.error(
                    "[ingest] Conversion failed for %s: %s", file.filename, exc
                )
                _ingest_progress[job_id]["status"] = "error"
                _ingest_progress[job_id]["message"] = f"Conversion failed: {exc}"
                return

            _ingest_progress[job_id]["stage"] = "inserting"
            _ingest_progress[job_id][
                "message"
            ] = f"Inserting **{file.filename}** into database …"
            logger.info("[ingest] Inserting %s into database", file.filename)

            import psycopg  # noqa: E402

            db = psycopg.connect(env_db_url)
            try:
                cur = db.cursor()
                eq_id = equipment_id or None
                cur.execute(
                    """
                    INSERT INTO documents (filename, file_type, document_category,
                                           title, markdown_content, source_path, metadata,
                                           equipment_id, location, revision)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        file.filename,
                        file_type,
                        category,
                        metadata.get("title"),
                        markdown,
                        tmp_path,
                        json.dumps(metadata, default=str),
                        eq_id,
                        location,
                        revision,
                    ),
                )
                doc_id = cur.fetchone()[0]

                _ingest_progress[job_id]["stage"] = "chunking"
                _ingest_progress[job_id][
                    "message"
                ] = f"Chunking and embedding **{file.filename}** …"
                logger.info("[ingest] Chunking & embedding %s", file.filename)

                def _progress(msg: str) -> None:
                    _ingest_progress[job_id]["message"] = msg

                chunks_count = chunk_and_insert(
                    doc_id,
                    markdown,
                    file_type,
                    db,
                    progress_callback=_progress,
                )
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("[ingest] DB error for %s: %s", file.filename, exc)
                _ingest_progress[job_id]["status"] = "error"
                _ingest_progress[job_id]["message"] = f"Database error: {exc}"
                return
            finally:
                db.close()

        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        _ingest_progress[job_id]["status"] = "complete"
        _ingest_progress[job_id][
            "message"
        ] = f"Done — **{file.filename}** ingested ({chunks_count} chunks)"
        _ingest_progress[job_id]["result"] = {
            "document_id": doc_id,
            "filename": file.filename,
            "chunks": chunks_count,
        }

    threading.Thread(target=_run_ingest, daemon=True).start()
    return {"job_id": job_id}


@app.get("/ingest/status/{job_id}")
def get_ingest_status(job_id: str):
    """Poll the current status of an ingest job."""
    if job_id not in _ingest_progress:
        raise HTTPException(status_code=404, detail="Job not found")

    data = dict(_ingest_progress[job_id])
    stage = data.get("stage")
    message = data.get("message", "")
    status = data.get("status", "queued")

    if status == "complete":
        progress = 100
    elif status == "error":
        progress = 0
    elif stage == "converting":
        # Parse "Converting page X/Y …" or "Converting sheet X/Y …"
        m = re.search(r"(\d+)/(\d+)", message)
        if m:
            progress = max(2, int(int(m.group(1)) / int(m.group(2)) * 40))
        else:
            progress = 5
    elif stage == "inserting":
        progress = 45
    elif stage == "chunking":
        # Parse "Embedding chunk X–Y/total"
        m = re.search(r"(\d+)[–\-](\d+)/(\d+)", message)
        if m:
            batch_end = int(m.group(2))
            total_ch = int(m.group(3))
            progress = int(50 + (batch_end / total_ch) * 45)
        else:
            progress = 50
    else:
        progress = 2

    data["progress"] = (
        min(progress, 99) if status not in ("complete", "error") else progress
    )
    return data


@app.post("/documents/upload")
async def upload_document(file_path: str, category: str = "other"):
    """Upload and ingest a document by local path.

    Runs the blocking I/O (file conversion, embedding, DB insert) in a
    thread pool so the async event loop stays responsive for other requests.
    """
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")  # noqa: E501

    # Run blocking work off the event loop thread
    def _do_upload() -> dict:
        from converter import convert_file  # noqa: E402
        from chunker import chunk_and_insert  # noqa: E402
        import psycopg  # noqa: E402

        try:
            markdown, metadata = convert_file(str(path))
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Conversion failed: {exc}"
            ) from exc

        db = psycopg.connect(env_db_url)
        try:
            cur = db.cursor()
            cur.execute(
                """
                INSERT INTO documents (filename, file_type, document_category,
                                       title, markdown_content,
                                       source_path, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    path.name,
                    path.suffix.lstrip(".").lower(),
                    category,
                    metadata.get("title"),
                    markdown,
                    str(path),
                    json.dumps(metadata, default=str),
                ),
            )
            doc_id = cur.fetchone()[0]

            chunks_count = chunk_and_insert(
                doc_id, markdown, path.suffix.lstrip(".").lower(), db
            )
            db.commit()
        except Exception as exc:
            db.rollback()
            raise HTTPException(
                status_code=500, detail=f"Ingestion failed: {exc}"
            ) from exc
        finally:
            db.close()

        return {"document_id": doc_id, "chunks": chunks_count}

    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            _do_upload,
        )
    except HTTPException as he:
        raise he

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Run the server."""
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("APP_PORT", 8000)),
        reload=os.environ.get("UVICORN_RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()

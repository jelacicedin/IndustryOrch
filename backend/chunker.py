# Copyright (C) 2026  Edin Jelacic — AGPL-3.0-or-later
"""Chunk structured markdown for embedding, then insert into the database.

Uses content-aware strategies: prose chunks by heading boundaries with overlap,
tables kept together or split by row groups. Each chunk is embedded via Ollama
and batch-inserted into the chunks table.

Usage
-----
    # Library
    from chunker import chunk_and_insert

    # CLI
    python chunker.py --document-id 1 --file path/to/document.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import psycopg
from psycopg import sql

logger = logging.getLogger("edintech-ingest")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
)
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))  # chunks per Ollama call
DB_BATCH_SIZE = int(os.getenv("DB_BATCH_SIZE", "200"))  # chunks per INSERT

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    """Fast token estimate: 1 token ≈ 4 chars."""
    return max(1, len(text) >> 2)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Markdown parsing helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_FENCE_RE = re.compile(r"^```", re.MULTILINE)


def _is_table_header(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and "|" in s[1:]


def _is_table_separator(line: str) -> bool:
    """Match multi-column markdown table separators e.g. | --- | :---: | --- |"""
    return bool(re.match(r"^\|([\s:|-]+\|)+$", line.strip()))


def _build_fence_map(markdown: str) -> list[tuple[int, int]]:
    """Return list of (start_char, end_char) ranges that are inside code fences."""
    fences: list[tuple[int, int]] = []
    positions = [m.start() for m in _FENCE_RE.finditer(markdown)]
    for i in range(0, len(positions) - 1, 2):
        fences.append((positions[i], positions[i + 1]))
    return fences


def _in_fence(pos: int, fences: list[tuple[int, int]]) -> bool:
    for s, e in fences:
        if s <= pos <= e:
            return True
    return False


def _char_to_line_map(lines: list[str]) -> list[int]:
    """Build cumulative char offset for each line start (O(n), built once)."""
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)  # +1 for \n
    return offsets


def _offset_to_line(offsets: list[int], offset: int) -> int:
    """Binary search for line index from char offset."""
    lo, hi = 0, len(offsets) - 1
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if offsets[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _extract_tables_in_range(
    lines: list[str], start: int, end: int
) -> list[tuple[int, int]]:
    """Find markdown table blocks (abs line indices) between start and end."""
    tables: list[tuple[int, int]] = []
    i = start
    while i < end:
        if _is_table_header(lines[i]):
            # Look for separator in next 1-3 lines
            sep = None
            for j in range(i + 1, min(i + 4, end)):
                if _is_table_separator(lines[j]):
                    sep = j
                    break
            if sep is not None:
                tail = sep + 1
                while tail < end and lines[tail].strip().startswith("|"):
                    tail += 1
                tables.append((i, tail - 1))
                i = tail
                continue
        i += 1
    return tables


def _split_by_headings(
    markdown: str, fences: list[tuple[int, int]]
) -> list[tuple[str, list[tuple[int, int]], list[int]]]:
    """Split markdown into sections by heading lines (skipping fenced headings).

    Returns list of (heading_text, [(table_start, table_end)], prose_line_indices).
    """
    if not markdown.strip():
        return []

    lines = markdown.split("\n")
    offsets = _char_to_line_map(lines)

    heading_matches = [
        m for m in _HEADING_RE.finditer(markdown) if not _in_fence(m.start(), fences)
    ]

    if not heading_matches:
        tables = _extract_tables_in_range(lines, 0, len(lines))
        table_set = set()
        for s, e in tables:
            table_set.update(range(s, e + 1))
        prose = [i for i in range(len(lines)) if i not in table_set]
        return [("Document", tables, prose)]

    heading_lines = [_offset_to_line(offsets, m.start()) for m in heading_matches]

    sections: list[tuple[str, list[tuple[int, int]], list[int]]] = []
    for idx, h_line in enumerate(heading_lines):
        end_line = (
            heading_lines[idx + 1] if idx + 1 < len(heading_lines) else len(lines)
        )
        h_line = min(h_line, len(lines) - 1)
        end_line = min(end_line, len(lines))
        if h_line >= end_line:
            continue

        heading_text = re.sub(r"^#{1,3}\s+", "", lines[h_line]).strip()
        tables = _extract_tables_in_range(lines, h_line, end_line)
        table_set = set()
        for s, e in tables:
            table_set.update(range(s, e + 1))
        prose = [i for i in range(h_line, end_line) if i not in table_set]
        sections.append((heading_text, tables, prose))

    return sections


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------


def _chunk_prose(
    prose_line_indices: list[int],
    lines: list[str],
    target: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Merge prose lines into ~target-token chunks with overlap.
    Never splits inside a paragraph.
    """
    if not prose_line_indices:
        return []

    text = "\n".join(lines[i] for i in prose_line_indices)
    paras = re.split(r"\n\n+", text)
    chunks: list[Chunk] = []
    buf = ""

    for para in paras:
        candidate = (buf + "\n\n" + para).strip() if buf else para
        if _count_tokens(candidate) > target and buf:
            chunks.append(Chunk(content=buf))
            # Carry overlap: last few lines of buf
            overlap_lines = buf.split("\n")[-(max(1, overlap // 8)) :]
            buf = "\n".join(overlap_lines) + "\n\n" + para
        else:
            buf = candidate

    if buf.strip():
        chunks.append(Chunk(content=buf.strip()))

    # Merge tiny trailing chunks
    merged: list[Chunk] = []
    for c in chunks:
        if (
            merged
            and _count_tokens(merged[-1].content) < overlap
            and _count_tokens(merged[-1].content + c.content) <= target + overlap
        ):
            merged[-1].content += "\n\n" + c.content
        else:
            merged.append(c)
    return merged


def _chunk_table(
    lines: list[str], start: int, end: int, max_rows: int = 20, max_tokens: int = 800
) -> list[Chunk]:
    """Keep table whole if small; otherwise split into row groups."""
    table_lines = lines[start : end + 1]
    if _count_tokens("\n".join(table_lines)) <= max_tokens:
        return [Chunk(content="\n".join(table_lines))]

    header = table_lines[0]
    separator = table_lines[1] if len(table_lines) > 1 else ""
    data_rows = table_lines[2:]
    chunks: list[Chunk] = []
    for i in range(0, max(1, len(data_rows)), max_rows):
        group = data_rows[i : i + max_rows]
        chunks.append(Chunk(content="\n".join([header, separator] + group)))
    return chunks


# ---------------------------------------------------------------------------
# Embedding with retry
# ---------------------------------------------------------------------------


def _embed_batch(texts: list[str], retries: int = 3) -> list[list[float]]:
    """Embed a batch of texts via Ollama with exponential-backoff retry."""
    import ollama

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = ollama.embed(model=EMBED_MODEL, input=texts)
            return resp["embeddings"]
        except Exception as exc:
            last_exc = exc
            wait = 2.0 * (attempt + 1)
            logger.warning(
                "Embed attempt %d/%d failed (%s) — retrying in %.0fs",
                attempt + 1,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"Embedding failed after {retries} attempts: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _get_conn() -> psycopg.Connection:
    try:
        return psycopg.connect(DATABASE_URL)
    except Exception as exc:
        raise RuntimeError(f"Cannot connect to database: {exc}") from exc


def _bulk_insert_chunks(
    cur, document_id: int, batch: list[Chunk], embeddings: list[list[float]] | None
) -> None:
    """Single parameterized INSERT for the entire batch."""
    placeholders = sql.SQL(", ").join(sql.SQL("(%s, %s, %s, %s)") for _ in batch)
    stmt = sql.SQL(
        "INSERT INTO chunks (document_id, content, metadata, embedding) VALUES {}"
    ).format(placeholders)

    flat: list[Any] = []
    for idx, c in enumerate(batch):
        emb = embeddings[idx] if embeddings is not None else None
        flat.extend([document_id, c.content, json.dumps(c.metadata), emb])

    cur.execute(stmt, flat)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ChunkingError(Exception):
    """Raised when chunking or database insertion fails."""


def chunk_and_insert(
    document_id: int,
    markdown: str,
    file_type: str,
    db_conn: psycopg.Connection,
    skip_embeddings: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> int:
    """Chunk markdown, embed, and insert into chunks table.

    Parameters
    ----------
    document_id       : PK of the document row.
    markdown          : Structured markdown from converter.py.
    file_type         : 'pdf' | 'xlsx' | 'csv' | 'docx' etc.
    db_conn           : Active psycopg connection.
    skip_embeddings   : Insert NULL embeddings (useful for testing).
    progress_callback : Optional callable(str) for progress reporting.

    Returns
    -------
    int  Number of chunks inserted.
    """
    if not markdown or not markdown.strip():
        return 0

    lines = markdown.split("\n")
    fences = _build_fence_map(markdown)

    try:
        sections = _split_by_headings(markdown, fences)
    except Exception as exc:
        raise ChunkingError(f"Section split failed: {exc}") from exc

    all_chunks: list[Chunk] = []

    for heading, tables, prose_lines in sections:
        sec_meta: dict[str, Any] = {"section_heading": heading}

        # Sheet / page metadata from heading
        if m := re.match(r"## Sheet:\s*(.+)", heading):
            sec_meta["sheet_name"] = m.group(1).strip()
        if m := re.match(r"## Page\s+(\d+)", heading):
            sec_meta["page_number"] = int(m.group(1))

        # Prose
        for c in _chunk_prose(prose_lines, lines):
            c.metadata = {**sec_meta, "content_type": "prose"}
            all_chunks.append(c)

        # Tables
        for ts, te in tables:
            for c in _chunk_table(lines, ts, te):
                c.metadata = {**sec_meta, "content_type": "table"}
                all_chunks.append(c)

    total = len(all_chunks)
    if total == 0:
        return 0

    for i, c in enumerate(all_chunks):
        c.metadata["chunk_index"] = i + 1
        c.metadata["total_chunks"] = total

    cur = db_conn.cursor()
    inserted = 0

    # Embed in BATCH_SIZE chunks, insert in DB_BATCH_SIZE
    for batch_start in range(0, total, BATCH_SIZE):
        embed_batch = all_chunks[batch_start : batch_start + BATCH_SIZE]
        t0 = time.monotonic()

        if progress_callback:
            progress_callback(
                f"Embedding chunk {batch_start + 1}–"
                f"{min(batch_start + BATCH_SIZE, total)}/{total}"
            )

        embeddings: list[list[float]] | None = None
        if not skip_embeddings:
            try:
                embeddings = _embed_batch([c.content for c in embed_batch])
            except Exception as exc:
                raise ChunkingError(
                    f"Embedding failed at chunk {batch_start}: {exc}"
                ) from exc

        # Insert this embed batch (may split further into DB_BATCH_SIZE)
        for db_start in range(0, len(embed_batch), DB_BATCH_SIZE):
            db_slice = embed_batch[db_start : db_start + DB_BATCH_SIZE]
            emb_slice = (
                embeddings[db_start : db_start + DB_BATCH_SIZE] if embeddings else None
            )
            try:
                _bulk_insert_chunks(cur, document_id, db_slice, emb_slice)
                db_conn.commit()
                inserted += len(db_slice)
            except Exception as exc:
                db_conn.rollback()
                raise ChunkingError(f"DB insert failed: {exc}") from exc

        elapsed = time.monotonic() - t0
        logger.info(
            "Progress %d/%d — %.1fs (%.1f chunks/s)",
            inserted,
            total,
            elapsed,
            len(embed_batch) / max(elapsed, 0.001),
        )

    logger.info("Done: %d chunks for document %d", total, document_id)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Chunk and embed a document")
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--category", default="other")
    parser.add_argument("--equipment-id", default=None)
    parser.add_argument("--location", default=None)
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    from converter import convert_file

    markdown, meta = convert_file(args.file)

    file_type = Path(args.file).suffix.lstrip(".").lower()
    db = _get_conn()

    n = chunk_and_insert(
        document_id=args.document_id,
        markdown=markdown,
        file_type=file_type,
        db_conn=db,
        skip_embeddings=args.skip_embeddings,
        progress_callback=lambda msg: print(f"  {msg}"),
    )
    print(f"✓ {n} chunks inserted for document {args.document_id}")


if __name__ == "__main__":
    main()

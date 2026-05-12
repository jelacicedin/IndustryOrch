"""Ingestion worker — watches a directory and processes documents.

Usage (Docker):
    docker compose run --rm ingest

Usage (local):
    python ingest.py --dir ./my-documents

When run as a module inside the container, watches /workspace/ingest.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger("edintech-ingest")


def _get_db_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://edintech:password@localhost:5432/edintechrag",
    )


def ingest_file(file_path: str, category: str = "other") -> dict:
    """Convert, chunk, and insert a single document into the RAG database."""
    from converter import convert_file as _convert
    from chunker import chunk_and_insert as _chunk

    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in (".pdf", ".xlsx", ".xls", ".csv"):
        logger.warning("Skipping unsupported file: %s", file_path)
        return {"status": "skipped", "reason": f"Unsupported extension: {ext}"}

    # Convert to markdown
    try:
        markdown, metadata = _convert(file_path)
    except Exception as exc:
        logger.error("Conversion failed for %s: %s", file_path, exc)
        return {"status": "error", "file": file_path, "error": str(exc)}

    # Connect to DB and insert document record
    import psycopg

    db = psycopg.connect(_get_db_url())
    try:
        cur = db.cursor()
        cur.execute(
            """
            INSERT INTO documents (filename, file_type, document_category,
                                   title, markdown_content, source_path, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                path.name,
                ext.lstrip("."),
                category,
                metadata.get("title"),
                markdown,
                str(path),
                json.dumps(metadata, default=str),
            ),
        )
        doc_id = cur.fetchone()[0]

        # Chunk and embed
        chunks_count = _chunk(doc_id, markdown, ext.lstrip("."), db)
        db.commit()

        logger.info(
            "Ingested %s → document_id=%d, chunks=%d",
            file_path,
            doc_id,
            chunks_count,
        )
        return {
            "status": "ok",
            "file": file_path,
            "document_id": doc_id,
            "chunks": chunks_count,
        }
    except Exception as exc:
        db.rollback()
        logger.error("DB insertion failed for %s: %s", file_path, exc)
        return {"status": "error", "file": file_path, "error": str(exc)}
    finally:
        db.close()


def ingest_directory(directory: str, category: str = "other") -> list[dict]:
    """Process all supported files in a directory (non-recursive)."""
    results = []
    dir_path = Path(directory)

    if not dir_path.exists():
        logger.error("Directory does not exist: %s", directory)
        return results

    for file_path in sorted(dir_path.iterdir()):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in (".pdf", ".xlsx", ".xls", ".csv"):
            logger.debug("Skipping: %s", file_path)
            continue

        result = ingest_file(str(file_path), category)
        results.append(result)

        # Small delay to avoid hammering Ollama during embedding
        time.sleep(0.5)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="EdinTech-RAG ingestion worker")
    parser.add_argument(
        "--dir",
        "-d",
        default="/workspace/ingest",
        help="Directory to watch for documents (default: /workspace/ingest)",
    )
    parser.add_argument(
        "--category",
        "-c",
        default="other",
        help="Document category (default: other)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("Starting ingestion from %s", args.dir)
    results = ingest_directory(args.dir, category=args.category)

    ok = sum(1 for r in results if r["status"] == "ok")
    err = sum(1 for r in results if r["status"] == "error")
    skip = sum(1 for r in results if r["status"] == "skipped")

    logger.info("Done: %d ok, %d errors, %d skipped", ok, err, skip)


if __name__ == "__main__":
    main()

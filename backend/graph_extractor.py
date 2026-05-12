# Copyright (C) 2026  Edin Jelacic — AGPL-3.0-or-later
"""Entity and relationship extraction from industrial document markdown.

Batches chunks and sends them to a local Ollama model with JSON grammar
constraints. Setting GRAPH_BATCH_SIZE controls how many chunks are packed into
a single LLM call (default 3). Larger values reduce round-trips at the cost of
longer individual generations.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import ollama

logger = logging.getLogger("industryorch-graph")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
os.environ["OLLAMA_HOST"] = OLLAMA_URL.rstrip("/")

EXTRACTION_MODEL = os.environ.get("GRAPH_EXTRACTION_MODEL", "qwen2.5:1.5b")

# Max chars per chunk — stays within context window after prompt overhead
MAX_CHUNK_CHARS = 4000

# Chunks packed into a single LLM call — reduces round-trips significantly
EXTRACTION_BATCH_SIZE = int(os.environ.get("GRAPH_BATCH_SIZE", "3"))

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExtractedEntity:
    name: str
    type: str
    attributes: dict = field(default_factory=dict)


@dataclass
class ExtractedRelationship:
    source: str
    target: str
    type: str
    attributes: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)

    def merge(self, other: ExtractionResult) -> None:
        self.entities.extend(other.entities)
        self.relationships.extend(other.relationships)


# ---------------------------------------------------------------------------
# Allowed entity and relationship types for industrial documents
# ---------------------------------------------------------------------------

ENTITY_TYPES = [
    "machine",
    "component",
    "procedure",
    "safety_rule",
    "chemical",
    "part_number",
    "parameter",
    "location",
    "system",
    "document",
]

RELATIONSHIP_TYPES = [
    "feeds",
    "controls",
    "requires",
    "located_at",
    "part_of",
    "connected_to",
    "uses",
    "specifies",
    "warns_about",
    "maintained_by",
    "replaces",
]

# ---------------------------------------------------------------------------
# Chunking for extraction (separate from embedding chunking)
# ---------------------------------------------------------------------------


def _split_markdown_chunks(
    markdown: str, max_chars: int = MAX_CHUNK_CHARS
) -> list[str]:
    """Split markdown into extraction-friendly chunks at heading boundaries."""
    lines = markdown.split("\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for line in lines:
        heading_match = re.match(r"^(#{2,4})\s+", line)

        if heading_match and current_chunk and current_len > max_chars * 0.6:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_len = len(line)
            continue

        current_chunk.append(line)
        current_len += len(line)

        if current_len > max_chars:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_len = 0

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


# ---------------------------------------------------------------------------
# LLM extraction prompt — handles 1..N sections per call
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are an industrial document analyst. Extract entities and relationships \
from the document sections below.

Entity types (choose one): {entity_types}

Relationship types (choose one): {rel_types}

Rules:
1. Extract ALL meaningful entities — machines, components, procedures, \
chemicals, part numbers, locations, parameters, systems, safety rules, \
and documents.
2. Use ONLY the relationship types listed above.
3. Entity names must be specific and include identifiers when present \
(e.g. "Coolant Pump CP-101").
4. Every relationship must connect two entities that appear in the text.
5. Include numeric attributes (units, values, conditions) when stated.
6. If a safety rule mentions a hazard and a related entity, create the \
rule entity AND a "warns_about" relationship.
7. If a procedure mentions required components or PPE, create "requires" \
relationships.

Document sections:

{sections}

Return ONLY a compact single-line JSON object \
(no markdown, no explanation, no extra whitespace):
{{"entities":[{{"name":"...","type":"...","attributes":{{}}}}],\
"relationships":[{{"source":"...","target":"...","type":"...","attributes":{{}}}}]}}\
"""


def _build_batch_prompt(chunks: list[str]) -> str:
    section_parts = [
        f"=== Section {i + 1} ===\n{chunk}"
        for i, chunk in enumerate(chunks)
    ]
    return _EXTRACTION_PROMPT.format(
        entity_types=", ".join(ENTITY_TYPES),
        rel_types=", ".join(RELATIONSHIP_TYPES),
        sections="\n\n".join(section_parts),
    )


# ---------------------------------------------------------------------------
# LLM call — processes one batch of chunks
# ---------------------------------------------------------------------------


def _extract_from_batch(
    chunks: list[str],
    model: str,
    keep_alive: str | int = "5m",
) -> ExtractionResult | None:
    """Send a batch of markdown chunks to the LLM and parse the JSON response.

    format="json" constrains the model to valid JSON output, removing the need
    for regex-based extraction and reducing parse failures.
    """
    prompt = _build_batch_prompt(chunks)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0, "num_predict": 8192},
            keep_alive=keep_alive,
        )
        text = response.message.content or ""
    except Exception as exc:
        logger.error(
            "LLM extraction failed for batch of %d chunk(s): %s",
            len(chunks),
            exc,
        )
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error in LLM response: %s", exc)
        return None

    result = ExtractionResult()
    skipped_entities = 0
    skipped_rels = 0

    for ent in data.get("entities", []):
        name = str(ent.get("name", "")).strip()
        etype = str(ent.get("type", "")).strip().lower()
        attrs = ent.get("attributes", {}) or {}
        if not name or etype not in ENTITY_TYPES:
            if name and etype:
                logger.debug(
                    "Unknown entity type '%s' for '%s', skipping",
                    etype,
                    name,
                )
            skipped_entities += 1
            continue
        logger.debug("  entity  [%s] %s", etype, name)
        result.entities.append(
            ExtractedEntity(name=name, type=etype, attributes=attrs)
        )

    for rel in data.get("relationships", []):
        source = str(rel.get("source", "")).strip()
        target = str(rel.get("target", "")).strip()
        rtype = str(rel.get("type", "")).strip().lower()
        attrs = rel.get("attributes", {}) or {}
        if not source or not target or rtype not in RELATIONSHIP_TYPES:
            if source and target and rtype:
                logger.debug(
                    "Unknown relationship type '%s', skipping", rtype
                )
            skipped_rels += 1
            continue
        logger.debug("  rel     %s -[%s]-> %s", source, rtype, target)
        result.relationships.append(
            ExtractedRelationship(
                source=source, target=target, type=rtype, attributes=attrs
            )
        )

    etype_counts = Counter(e.type for e in result.entities)
    rtype_counts = Counter(r.type for r in result.relationships)
    logger.info(
        "  Batch result: %d entities %s | %d rels %s"
        " | skipped %d/%d",
        len(result.entities),
        dict(etype_counts),
        len(result.relationships),
        dict(rtype_counts),
        skipped_entities + skipped_rels,
        len(data.get("entities", [])) + len(data.get("relationships", [])),
    )

    return result



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_graph_from_markdown(
    markdown: str,
    progress_callback: Callable[[str], None] | None = None,
) -> ExtractionResult:
    """Extract entities and relationships from document markdown.

    Batches chunks into EXTRACTION_BATCH_SIZE groups and sends each batch as a
    single Ollama LLM call, reducing round-trips from N to ceil(N / batch_size).
    The extraction model is kept warm between batches; keep_alive=0 on the last
    batch evicts it so generation VRAM is free.
    """
    chunks = _split_markdown_chunks(markdown)
    if not chunks:
        return ExtractionResult()

    batches = [
        chunks[i:i + EXTRACTION_BATCH_SIZE]
        for i in range(0, len(chunks), EXTRACTION_BATCH_SIZE)
    ]

    logger.info(
        "Graph extraction: %d chunk(s) → %d batch(es) of up to %d",
        len(chunks),
        len(batches),
        EXTRACTION_BATCH_SIZE,
    )

    if progress_callback:
        progress_callback(
            f"Extracting entities from **{len(chunks)}** section(s) "
            f"in **{len(batches)}** batch(es) …"
        )

    merged = ExtractionResult()
    failed_chunks = 0

    for i, batch in enumerate(batches):
        if progress_callback and len(batches) > 1:
            progress_callback(
                f"Extracting batch {i + 1}/{len(batches)}"
                f" ({len(batch)} section(s)) …"
            )

        # Evict the model from VRAM after the last batch so the generation
        # model can load without competing for memory.
        is_last = i == len(batches) - 1
        result = _extract_from_batch(
            batch,
            EXTRACTION_MODEL,
            keep_alive=0 if is_last else "5m",
        )

        if result is None:
            failed_chunks += len(batch)
            continue

        merged.merge(result)

    if failed_chunks:
        logger.warning("%d chunk(s) failed graph extraction", failed_chunks)

    etype_totals = Counter(e.type for e in merged.entities)
    rtype_totals = Counter(r.type for r in merged.relationships)

    logger.info(
        "Graph extraction complete: %d entities, %d relationships"
        " (%d failed chunk(s))",
        len(merged.entities),
        len(merged.relationships),
        failed_chunks,
    )
    logger.info("Entity types:       %s", dict(etype_totals))
    logger.info("Relationship types: %s", dict(rtype_totals))

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("All extracted entities:")
        for e in merged.entities:
            logger.debug("  [%s] %s %s", e.type, e.name, e.attributes or "")
        logger.debug("All extracted relationships:")
        for r in merged.relationships:
            logger.debug(
                "  %s -[%s]-> %s %s",
                r.source, r.type, r.target, r.attributes or "",
            )

    return merged


def extract_graph_from_file(
    filepath: str,
    progress_callback: Callable[[str], None] | None = None,
) -> ExtractionResult:
    """Full pipeline: read file → convert to markdown → extract graph."""
    from converter import convert_file  # noqa: E402

    if progress_callback:
        progress_callback("Converting document to markdown …")

    markdown, _metadata = convert_file(
        filepath, progress_callback=progress_callback
    )
    return extract_graph_from_markdown(
        markdown, progress_callback=progress_callback
    )

# Copyright (C) 2026  Edin Jelacic — AGPL-3.0-or-later
"""Entity and relationship extraction from industrial document markdown.

Two extraction paths are available, selected by the GRAPH_USE_LLM env var:

* LLM path (GRAPH_USE_LLM=true, default): batches chunks and sends them to a
  local Ollama model with JSON grammar constraints. Slower but produces richer
  attributes and handles domain-specific jargon well.

* NLP path (GRAPH_USE_LLM=false): GLiNER zero-shot NER for entities + spaCy
  dependency parsing for relationships. No Ollama required; runs entirely
  in-process on CPU. Typically 10-50x faster than the LLM path.

Setting GRAPH_BATCH_SIZE controls how many chunks are packed into a single LLM
call (default 3). Larger values reduce round-trips at the cost of longer
individual generations.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import ollama

logger = logging.getLogger("edintech-graph")

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

# Toggle extraction backend: true = Ollama LLM, false = GLiNER + spaCy
GRAPH_USE_LLM = os.environ.get("GRAPH_USE_LLM", "true").lower() in (
    "true", "1", "yes"
)

# NLP path settings (only used when GRAPH_USE_LLM=false)
GLINER_MODEL = os.environ.get("GLINER_MODEL", "urchade/gliner_medium-v2.1")
NLP_MODEL = os.environ.get("GRAPH_NLP_MODEL", "en_core_web_sm")
GLINER_THRESHOLD = float(os.environ.get("GLINER_THRESHOLD", "0.4"))

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

# Verb lemma → relationship type (NLP path only).
# Covers the most common industrial document verbs.
VERB_REL_MAP: dict[str, str] = {
    "feed": "feeds",
    "supply": "feeds",
    "pump": "feeds",
    "deliver": "feeds",
    "control": "controls",
    "regulate": "controls",
    "govern": "controls",
    "manage": "controls",
    "require": "requires",
    "need": "requires",
    "depend": "requires",
    "locate": "located_at",
    "mount": "located_at",
    "install": "located_at",
    "position": "located_at",
    "situate": "located_at",
    "connect": "connected_to",
    "attach": "connected_to",
    "link": "connected_to",
    "couple": "connected_to",
    "join": "connected_to",
    "use": "uses",
    "utilize": "uses",
    "employ": "uses",
    "apply": "uses",
    "specify": "specifies",
    "define": "specifies",
    "describe": "specifies",
    "indicate": "specifies",
    "state": "specifies",
    "warn": "warns_about",
    "caution": "warns_about",
    "alert": "warns_about",
    "maintain": "maintained_by",
    "service": "maintained_by",
    "replace": "replaces",
    "substitute": "replaces",
    "supersede": "replaces",
    "contain": "part_of",
    "include": "part_of",
    "comprise": "part_of",
    "consist": "part_of",
    "incorporate": "part_of",
}

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
# NLP extraction — GLiNER (NER) + spaCy (dependency parsing)
# ---------------------------------------------------------------------------

_nlp_instance: Any = None
_gliner_instance: Any = None


def _get_nlp() -> Any:
    global _nlp_instance
    if _nlp_instance is None:
        import spacy  # type: ignore[import]
        _nlp_instance = spacy.load(NLP_MODEL, exclude=["ner"])
        logger.info("Loaded spaCy model: %s", NLP_MODEL)
    return _nlp_instance


def _get_gliner() -> Any:
    global _gliner_instance
    if _gliner_instance is None:
        from gliner import GLiNER  # type: ignore[import]
        _gliner_instance = GLiNER.from_pretrained(GLINER_MODEL)
        logger.info("Loaded GLiNER model: %s", GLINER_MODEL)
    return _gliner_instance


def _extract_with_nlp(
    chunks: list[str],
    progress_callback: Callable[[str], None] | None = None,
) -> ExtractionResult:
    """Extract entities and relationships without an LLM.

    Uses GLiNER for zero-shot NER (entity recognition) and spaCy dependency
    parsing for subject-verb-object relationship extraction. No Ollama needed;
    models run in-process on CPU. Typically 10-50x faster than the LLM path.
    """
    gliner = _get_gliner()
    nlp = _get_nlp()

    seen_entities: dict[str, ExtractedEntity] = {}
    all_rels: list[ExtractedRelationship] = []

    for i, chunk in enumerate(chunks):
        if progress_callback and len(chunks) > 1:
            progress_callback(
                f"Extracting (NLP) chunk {i + 1}/{len(chunks)} …"
            )

        # --- GLiNER zero-shot NER ---
        try:
            spans = gliner.predict_entities(
                chunk, ENTITY_TYPES, threshold=GLINER_THRESHOLD
            )
        except Exception as exc:
            logger.warning("GLiNER failed on chunk %d: %s", i, exc)
            spans = []

        # char-range → entity name for matching against spaCy tokens
        char_to_ent: list[tuple[int, int, str]] = []
        for span in spans:
            name = span["text"].strip()
            etype = span["label"]
            if not name or etype not in ENTITY_TYPES:
                continue
            char_to_ent.append((span["start"], span["end"], name))
            if name not in seen_entities:
                seen_entities[name] = ExtractedEntity(name=name, type=etype)

        if not char_to_ent:
            continue

        # --- spaCy dependency parsing for relationships ---
        try:
            doc = nlp(chunk[:10_000])
        except Exception as exc:
            logger.warning("spaCy failed on chunk %d: %s", i, exc)
            continue

        def _tok_ent(tok) -> str | None:
            """Return the GLiNER entity name whose span contains this token."""
            for start, end, name in char_to_ent:
                if start <= tok.idx < end:
                    return name
            return None

        for sent in doc.sents:
            for tok in sent:
                if tok.dep_ not in ("nsubj", "nsubjpass"):
                    continue
                subj_name = _tok_ent(tok)
                if not subj_name:
                    continue
                verb = tok.head
                rel_type = VERB_REL_MAP.get(verb.lemma_.lower())
                if not rel_type:
                    continue
                for child in verb.children:
                    if child.dep_ not in ("dobj", "attr", "pobj", "oprd"):
                        continue
                    obj_name = _tok_ent(child)
                    if not obj_name:
                        # One level deeper (e.g. "feeds into [the pump]")
                        for gc in child.children:
                            obj_name = _tok_ent(gc)
                            if obj_name:
                                break
                    if obj_name and obj_name != subj_name:
                        all_rels.append(ExtractedRelationship(
                            source=subj_name,
                            target=obj_name,
                            type=rel_type,
                        ))

    etype_counts = Counter(e.type for e in seen_entities.values())
    rtype_counts = Counter(r.type for r in all_rels)
    logger.info(
        "NLP extraction complete: %d entities %s | %d rels %s",
        len(seen_entities),
        dict(etype_counts),
        len(all_rels),
        dict(rtype_counts),
    )

    return ExtractionResult(
        entities=list(seen_entities.values()),
        relationships=all_rels,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_graph_from_markdown(
    markdown: str,
    progress_callback: Callable[[str], None] | None = None,
) -> ExtractionResult:
    """Extract entities and relationships from document markdown.

    Dispatches to the NLP path (GLiNER + spaCy) when GRAPH_USE_LLM=false,
    or to the LLM path (Ollama batched calls) when GRAPH_USE_LLM=true (default).

    The NLP path batches chunks into EXTRACTION_BATCH_SIZE groups and sends each
    batch as a single LLM call. This reduces the number of round-trips from N
    (one per chunk) to ceil(N / batch_size).

    The extraction model is kept warm (keep_alive="5m") between batches.
    After the final batch keep_alive=0 evicts it so generation VRAM is free.
    """
    chunks = _split_markdown_chunks(markdown)
    if not chunks:
        return ExtractionResult()

    if not GRAPH_USE_LLM:
        if progress_callback:
            progress_callback(
                f"Extracting entities from **{len(chunks)}** section(s) "
                "using GLiNER + spaCy …"
            )
        return _extract_with_nlp(chunks, progress_callback=progress_callback)

    # --- LLM path ---
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

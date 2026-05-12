# Copyright (C) 2026  Edin Jelacic — AGPL-3.0-or-later
"""EdinTech-RAG — Streamlit frontend for industrial document Q&A.

Provides a chat interface to query ingested documents, upload files for
ingestion, and manage the document corpus via the FastAPI backend.

Run:
    streamlit run frontend.py
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

CATEGORIES = [
    "manual",
    "datasheet",
    "maintenance_record",
    "procedure",
    "report",
    "specification",
    "log",
    "other",
]

FILE_TYPES = ["", "pdf", "xlsx", "csv", "md", "txt"]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EdinTech-RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_session() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "ingestion_jobs": {},
        "filter_equipment_id": "",
        "filter_file_type": "",
        "filter_document_category": "",
        "filter_location": "",
        "filter_top_k": 5,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()

# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _get(path: str, timeout: int = 10) -> Any:
    resp = httpx.get(f"{BACKEND_URL}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, **kwargs) -> Any:
    resp = httpx.post(f"{BACKEND_URL}{path}", **kwargs)
    resp.raise_for_status()
    return resp.json()


def _delete(path: str, timeout: int = 10) -> int:
    resp = httpx.delete(f"{BACKEND_URL}{path}", timeout=timeout)
    return resp.status_code


def _extract_error(text: str) -> str:
    try:
        return json.loads(text).get("detail", text[:300])
    except Exception:
        return text[:300]


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------


def _start_ingestion(
    files: list, category: str, equipment_id: str, location: str, revision: str
) -> None:
    for f in files:
        data = f.read()
        try:
            result = httpx.post(
                f"{BACKEND_URL}/ingest",
                files={"file": (f.name, data, f.type)},
                data={
                    "category": category,
                    "equipment_id": equipment_id or None,
                    "location": location or None,
                    "revision": revision or None,
                },
                timeout=30,
            )
            result.raise_for_status()
            job_id = result.json()["job_id"]
            st.session_state.ingestion_jobs[job_id] = {
                "filename": f.name,
                "status": "queued",
                "message": f"Queued **{f.name}**",
                "progress": 0,
                "result": None,
            }
        except Exception as exc:
            st.sidebar.error(f"Failed to start ingestion for {f.name}: {exc}")


def _poll_jobs() -> bool:
    """Poll backend for each active job. Returns True if any job is still active."""
    has_active = False
    for job_id, job in list(st.session_state.ingestion_jobs.items()):
        if job["status"] in ("complete", "error"):
            continue
        has_active = True
        try:
            resp = httpx.get(f"{BACKEND_URL}/ingest/status/{job_id}", timeout=10)
            if resp.status_code == 404:
                job["status"] = "error"
                job["message"] = "Job not found (server restarted?)"
                continue
            data = resp.json()
            job["status"] = data.get("status", job["status"])
            job["message"] = data.get("message", job["message"])
            job["progress"] = data.get("progress", job["progress"])
            if data.get("result"):
                job["result"] = data["result"]
        except Exception as exc:
            job["message"] = f"Polling error: {exc}"
    return has_active


def _render_job_progress(job_id: str, job: dict) -> None:
    """Render a compact progress block for one ingestion job."""
    status = job["status"]
    message = job["message"]
    progress = job.get("progress", 0)
    filename = job["filename"]

    # Stage icon
    icon = {"complete": "✅", "error": "❌", "queued": "⏳"}.get(status, "⚙️")

    st.markdown(f"**{icon} {filename}**")

    if status == "complete":
        result = job.get("result") or {}
        chunks = result.get("chunks", "?")
        st.caption(f"Done — {chunks} chunks ingested")
    elif status == "error":
        st.caption(f"Error: {message}")
    else:
        st.progress(progress / 100)
        # Clean up markdown bold markers from backend message for sidebar display
        clean_msg = re.sub(r"\*\*([^*]+)\*\*", r"\1", message)
        st.caption(clean_msg)


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------


def _render_documents() -> None:
    try:
        docs = _get("/documents")
    except Exception as exc:
        st.caption(f"Could not fetch documents: {exc}")
        return

    if not docs:
        st.caption("No documents ingested yet.")
        return

    for d in docs:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.caption(
                f"**{d['filename']}** ({d['document_category']}) — "
                f"{d['chunk_count']} chunks"
            )
        with col2:
            if st.button("🗑", key=f"del_{d['id']}", help="Delete document"):
                try:
                    code = _delete(f"/documents/{d['id']}")
                    if code == 204:
                        st.toast(f"Deleted {d['filename']}", icon="🗑️")
                    else:
                        st.toast("Delete failed", icon="❌")
                except Exception as exc:
                    st.toast(f"Error: {exc}", icon="❌")
                st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "Select files",
        accept_multiple_files=True,
        type=["pdf", "xlsx", "xls", "csv"],
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        upload_category = st.selectbox(
            "Category",
            options=CATEGORIES,
            index=CATEGORIES.index("other"),
            key="upload_category",
        )
    with col_b:
        upload_equipment_id = st.text_input("Equipment ID", key="upload_equipment_id")

    col_c, col_d = st.columns(2)
    with col_c:
        upload_location = st.text_input("Location", key="upload_location")
    with col_d:
        upload_revision = st.text_input("Revision", key="upload_revision")

    if st.button(
        "Ingest Files",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        _start_ingestion(
            uploaded_files,
            st.session_state.upload_category,
            st.session_state.upload_equipment_id,
            st.session_state.upload_location,
            st.session_state.upload_revision,
        )
        st.rerun()

    # ------------------------------------------------------------------
    # Ingestion progress (only shown when jobs exist)
    # ------------------------------------------------------------------

    jobs = st.session_state.ingestion_jobs
    if jobs:
        st.divider()
        st.markdown("### ⚙️ Processing")

        active_jobs = {
            jid: j
            for jid, j in jobs.items()
            if j["status"] not in ("complete", "error")
        }
        done_jobs = {
            jid: j for jid, j in jobs.items() if j["status"] in ("complete", "error")
        }

        for job_id, job in jobs.items():
            _render_job_progress(job_id, job)
            if job_id != list(jobs.keys())[-1]:
                st.markdown("---")

        if active_jobs:
            _poll_jobs()
            time.sleep(0.8)
            st.rerun()
        elif done_jobs:
            # All jobs finished — pause briefly then clear
            all_ok = all(j["status"] == "complete" for j in done_jobs.values())
            if all_ok:
                st.toast("All documents ingested successfully!", icon="✅")
            time.sleep(2)
            st.session_state.ingestion_jobs = {}
            st.rerun()

    # ------------------------------------------------------------------
    # Document library
    # ------------------------------------------------------------------

    st.divider()
    with st.expander("📚 Document Library", expanded=False):
        _render_documents()

    st.divider()
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------

st.title("EdinTech-RAG")
st.caption("Ask questions about your ingested industrial documents.")

# Query filters (collapsible)
with st.expander("🔍 Query Filters", expanded=False):
    with st.form("filter_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.text_input("Equipment ID", key="filter_equipment_id")
        with c2:
            st.selectbox(
                "File Type",
                options=FILE_TYPES,
                key="filter_file_type",
                format_func=lambda x: x or "Any",
            )
        with c3:
            st.selectbox(
                "Category",
                options=[""] + CATEGORIES,
                key="filter_document_category",
                format_func=lambda x: x or "Any",
            )
        with c4:
            st.text_input("Location", key="filter_location")

        col_k, col_btn = st.columns([3, 1])
        with col_k:
            st.slider("Top K results", 1, 20, key="filter_top_k")
        with col_btn:
            st.form_submit_button("Apply", use_container_width=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("thinking"):
            with st.expander("💭 Reasoning", expanded=False):
                st.markdown(msg["thinking"])
        if msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])})", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**{i}.** `{src['doc_filename']}` — *{src['section']}* "
                        f"({src['doc_category']}) | score: `{src['rrf_score']:.4f}`"
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    # Build filter dict
    def _nonempty(v: str) -> str | None:
        return v.strip() or None

    equip_raw = st.session_state.filter_equipment_id.strip()
    filters = {
        "equipment_id": equip_raw or None,
        "file_type": _nonempty(st.session_state.filter_file_type),
        "document_category": _nonempty(st.session_state.filter_document_category),
        "location": _nonempty(st.session_state.filter_location),
    }
    top_k = int(st.session_state.filter_top_k)

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call backend and stream answer
    with st.chat_message("assistant"):
        answer_slot = st.empty()
        answer_slot.markdown("*Thinking…*")

        try:
            resp = httpx.post(
                f"{BACKEND_URL}/query",
                json={
                    "question": prompt,
                    "filters": filters,
                    "top_k": top_k,
                    "show_thinking": True,
                },
                timeout=180,
            )
            resp.raise_for_status()
            result = resp.json()

            answer = result.get("answer", "")
            thinking = result.get("thinking")
            sources = result.get("sources", [])

            answer_slot.markdown(answer)

            if thinking:
                with st.expander("💭 Reasoning", expanded=False):
                    st.markdown(thinking)

            if sources:
                with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f"**{i}.** `{src['doc_filename']}` — *{src['section']}* "
                            f"({src['doc_category']}) | score: `{src['rrf_score']:.4f}`"
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "thinking": thinking,
                    "sources": sources,
                }
            )

        except httpx.HTTPStatusError as exc:
            answer_slot.error(
                f"Backend error ({exc.response.status_code}): "
                f"{_extract_error(exc.response.text)}"
            )
        except Exception as exc:
            answer_slot.error(f"Request failed: {exc}")

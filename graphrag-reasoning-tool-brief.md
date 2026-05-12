# GraphRAG Reasoning Tool — Project Brief

> **Origin**: This project was planned based on research and conversation starting from the EdinTech-RAG repo (`/home/edinj/repos/EdinTech-RAG`). Use this file as a reference when scaffolding the new repo.

---

## What This Is

A full-featured, agentic tech document reasoning tool — not just retrieval, but **reasoning about connections** between data snippets using knowledge graphs, multi-agent orchestration, and long-term memory. Built as a successor/inspiration from EdinTech-RAG.

---

## Core Capabilities

### 1. Knowledge Graph Reasoning
- Extract entities and relationships from documents (machines, procedures, safety incidents, part numbers)
- Multi-hop graph traversal: "Find all machines that coolant system A feeds through"
- Dual retrieval: vector search (semantic similarity) + graph traversal (relationship reasoning)
- Community detection for global questions across entire document collection

### 2. Agentic Architecture
- **Supervisor pattern** (LangGraph): central coordinator routes to specialist workers
- Workers: `Researcher`, `CodeExecutor`, `DocumentWriter`, `WebSearcher`, `GraphReasoner`
- Deterministic termination conditions (not just "LLM says stop")

### 3. Document Generation
- Write docx files, markdown files, summaries
- Template system: predefined document templates (SOPs, incident reports, design docs) with auto-filled sections
- Citation-aware generation: every claim traced back to source documents

### 4. Web Search Integration
- Cross-reference KB answers against web sources
- Citation verification: flag contradictions between local KB and web findings

### 5. Memory System (3 Layers)
| Layer | What It Stores | How Long |
|-------|---------------|----------|
| Episodic | Past conversations, queries, actions taken | Days-weeks (sliding window) |
| Semantic | Facts learned from documents + user interactions | Permanent (stored in KB) |
| Profile / Autobiographical | User preferences, frequently asked topics | Permanent |

- Self-organizing memory bank — consolidate related memories across sessions
- Memory retrieval as first-class feature: injects relevant context before answering queries

### 6. Fresh Document Injection During Queries
- Incremental graph updates: new docs don't just add chunks; they update existing nodes/edges in the knowledge graph
- No full re-index required for minor additions

---

## Additional Tools to Build

| Tool | Purpose |
|------|---------|
| `file_reader` | Read any file type (YAML configs, log files, source code) — not just PDF/XLSX/CSV |
| `diff_generator` | Compare two document versions and highlight changes |
| `query_optimizer` | Rewrite user questions for better retrieval (paraphrase, decompose multi-part questions) |
| `citation_checker` | Verify claims against indexed documents; flag unsupported statements |
| `summary_generator` | Multi-level summaries: one-paragraph, one-page, detailed |
| `exporter` | Export knowledge graph as visual diagram (Mermaid/Graphviz), export Q&A sessions |

---

## Recommended Tech Stack

```
┌─────────────────────────────────────────────────┐
│  Frontend: Next.js + Shadcn UI                    │
│  (+ Tauri optional for desktop bundling)          │
│  Chat + Graph Explorer + Agent Activity Stream    │
├─────────────────────────────────────────────────┤
│  Orchestration: LangGraph                         │
│  Supervisor → [Researcher, Writer, WebSearcher,   │
│               CodeExecutor, GraphReasoner]        │
├─────────────────────────────────────────────────┤
│  Knowledge Graph: Neo4j (preferred)              │
│  Alternative: pgvector + custom graph layer       │
│  neo4j-graphrag-python for extraction pipeline    │
├─────────────────────────────────────────────────┤
│  Vector Store: PostgreSQL + pgvector             │
│  (keep from EdinTech-RAG — it works well)         │
├─────────────────────────────────────────────────┤
│  LLMs: Ollama (local) + OpenRouter fallback      │
│  Generation + Extraction + Reranking              │
├─────────────────────────────────────────────────┤
│  Memory: SQLite + sqlite-vec                     │
│  Episodic + Semantic + Profile layers             │
├─────────────────────────────────────────────────┤
│  Code Sandbox: Pyodide (browser) or Docker       │
│  container for server-side execution              │
├─────────────────────────────────────────────────┤
│  Search: Tavily/Brave API + local FTS            │
│  Hybrid web + KB search                           │
└─────────────────────────────────────────────────┘
```

---

## UI / Frontend Features to Include

- **Split-pane knowledge explorer** — left pane shows interactive knowledge graph, right pane is chat. Clicking a node filters context.
- **Agent activity stream** — show reasoning steps in real-time: "Searching KB... Found 3 pump-related documents → Traversing relationships... Discovered coolant system A connects to 7 machines → Generating answer..."
- **Document diff viewer** — side-by-side comparison when new docs are added
- **Graph visualization** — force-directed graph of knowledge base users can explore independently

---

## Evaluation & Quality (Critical)

| Feature | Why It Matters |
|---------|---------------|
| Ground-truth QA pairs | Test set of questions + expected answers. Measure accuracy drift over time. |
| Hallucination detection | Flag claims in answers not backed by any retrieved chunk. |
| Retrieval quality metrics | Track precision@K — are top-K chunks actually relevant? |
| A/B test prompts | Compare different system prompts, retrieval strategies, graph query templates. |

---

## MVP Roadmap (Build Order)

1. **Graph extraction pipeline** — take existing converter → add entity/relationship extraction step → store in Neo4j or graph table in PostgreSQL
2. **Supervisor agent** — LangGraph-based with 3 workers: `Retriever` (vector+graph), `Writer`, `WebSearcher`
3. **Basic UI** — Next.js chat interface with "agent activity" panel showing what the agent is doing
4. **Memory layer** — simple SQLite-based episodic memory tracking user query patterns

Then iteratively add: code sandbox, graph visualization, profile memory, template-based document generation, evaluation harness.

---

## Key Decisions & Rationale

- **GraphRAG over pure vector**: Industrial documents have structured relationships (machine A connects to valve B) that semantic search alone can't capture. Graph traversal enables multi-hop reasoning.
- **LangGraph supervisor pattern**: More reliable than free-form agent-to-agent calls, less complex than deep hierarchies. Deterministic termination conditions prevent infinite loops.
- **Keep PostgreSQL + pgvector**: Already works well in EdinTech-RAG. Use for vector storage; add Neo4j (or graph tables) on top for relationships.
- **Local-first (Ollama)**: Privacy-preserving, no API costs. Accept SLM trade-off on complex reasoning tasks.
- **Next.js over Streamlit**: Better UX for a production tool. Shadcn UI provides polished components. Tauri optional for desktop distribution.

---

## Inspiration Sources (from Research)

- Microsoft GraphRAG 2.0 — knowledge graph extraction + community detection
- Neo4j neo4j-graphrag-python — official GraphRAG pipeline package
- LangGraph supervisor pattern — multi-agent orchestration reference
- IBM RagWorkbench — RAG evaluation framework
- Dify — low-code RAG platform (UI inspiration)
- Kno / KnowNote — local-first, privacy-focused document tools
- LightRAG — hybrid storage (vector + graph + KV)

---

*Created from conversation starting at EdinTech-RAG on 2026-05-12.*

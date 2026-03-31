# Health RFP & Grant Response Agent

An AI-powered platform that automates RFP and grant proposal responses for health systems. Runs 100% locally using [Ollama](https://ollama.com) — no cloud API keys or data sharing required.

Built with FastAPI, PostgreSQL/pgvector, and Ollama.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [RAG Retrieval Pipeline](#rag-retrieval-pipeline)
- [Response Generator](#response-generator)
- [API Reference](#api-reference)
- [Ollama Model Guide](#ollama-model-guide)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [Running Tests](#running-tests)
- [Docker](#docker)

---

## Architecture Overview

```
RFP Input (PDF / DOCX / SAM.gov API)
          │
          ▼
  ┌─────────────────┐
  │  Stage 1        │  ingestor.py
  │  Ingestion      │  • PDF/DOCX file upload
  │                 │  • SAM.gov API polling (NAICS 621–623)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Stage 2        │  parser.py
  │  Document Parse │  • Regex: deadline, page limits, certifications
  │                 │  • Ollama: section extraction, evaluation criteria
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Stage 3        │  retriever.py  /  rag_retrieval.py
  │  RAG Retrieval  │  • Query expansion (Ollama)
  │                 │  • Embedding (nomic-embed-text via Ollama)
  │                 │  • pgvector cosine ANN search
  │                 │  • BM25 hybrid rerank
  │                 │  • MMR deduplication
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Stage 4        │  generator.py
  │  Response Gen   │  • Strategy detection per section type
  │                 │  • Targeted system prompts (technical / past-perf / budget etc.)
  │                 │  • Ollama LLM generation with retry + fallback
  │                 │  • Streaming token output
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Stage 5        │  compliance_checker.py
  │  Compliance     │  • HIPAA language check
  │                 │  • Page / word limit enforcement
  │                 │  • Missing certification detection
  │                 │  • Ollama: fabrication + missing requirement scan
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Stage 6        │  exporter.py
  │  Export         │  • Formatted Word document (.docx)
  │                 │  • PDF via LibreOffice (optional)
  └─────────────────┘
```

---

## Quick Start

### Prerequisites

| Requirement | Version | Install |
|---|---|---|
| Python | >= 3.11 | [python.org](https://www.python.org/downloads/) |
| PostgreSQL | >= 15 | [postgresql.org](https://www.postgresql.org/download/) |
| pgvector extension | >= 0.6 | `CREATE EXTENSION vector;` |
| Ollama | latest | `curl -fsSL https://ollama.com/install.sh \| sh` |
| Tesseract OCR | any | macOS: `brew install tesseract tesseract-lang` · Linux: `apt install tesseract-ocr` |
| LibreOffice | any | macOS: `brew install --cask libreoffice` · Linux: `apt install libreoffice` (optional, PDF export) |

### 1 — Install Python dependencies

```bash
git clone https://github.com/<your-github-username>/health-rfp-agent.git
cd health-rfp-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Pull Ollama models

```bash
# Required: LLM for generation, parsing, compliance checks
ollama pull llama3.1:8b

# Required: embeddings for RAG retrieval
ollama pull nomic-embed-text

# Optional: lightweight fallback if primary model times out
ollama pull phi3:mini

# Optional: healthcare-fine-tuned model (best quality for clinical content)
ollama pull medllama2:7b
```

### 3 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your settings (all Ollama — no cloud keys needed):

```env
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/health_rfp_db
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### 4 — Initialise the database

```bash
python scripts/init_db.py
```

This creates all tables, enables the pgvector extension, creates an HNSW index on the embeddings column, and seeds a demo organisation record.

### 5 — Seed the knowledge base

```bash
python scripts/seed_knowledge_base.py
```

Embeds 5 sample health system documents (clinical programmes, past performance, regulations, capability statement) so the agent has retrieval context out of the box.

### 6 — Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`.

### 7 — Start the frontend

```bash
cd frontend
npm install
npm start       # opens http://localhost:3000
```

---

## Project Structure

```
health_rfp_agent/
│
├── app/
│   ├── main.py                     # FastAPI app entrypoint, router registration
│   │
│   ├── agent/                      # Core pipeline stages
│   │   ├── ingestor.py             # Stage 1: RFP ingestion (file upload + SAM.gov)
│   │   ├── parser.py               # Stage 2: RFP document parsing (regex + Ollama)
│   │   ├── retriever.py            # Stage 3: Simple RAG retriever (pgvector + BM25)
│   │   ├── rag_retrieval.py        # Stage 3: Full 7-stage RAG pipeline (with MMR + synthesis)
│   │   ├── generator.py            # Stage 4: Response generator (Ollama, all section types)
│   │   ├── compliance_checker.py   # Stage 5: HIPAA, page limit, cert validation
│   │   ├── exporter.py             # Stage 6: Word document export
│   │   └── orchestrator.py         # Ties all 6 stages together
│   │
│   ├── api/                        # FastAPI route handlers
│   │   ├── rfp.py                  # RFP upload + SAM.gov search endpoints
│   │   ├── proposals.py            # Proposal generation + review endpoints
│   │   ├── knowledge_base.py       # KB document management endpoints
│   │   ├── rag.py                  # RAG query + stream endpoints
│   │   ├── generator.py            # Section generation endpoints
│   │   └── health.py               # Service health check endpoints
│   │
│   ├── core/
│   │   ├── config.py               # Pydantic settings (reads .env)
│   │   ├── dependencies.py         # FastAPI DI: DB session, auth
│   │   ├── security.py             # JWT utilities
│   │   └── logging.py              # Structured logging (structlog)
│   │
│   ├── db/
│   │   └── session.py              # Async SQLAlchemy engine + session factory
│   │
│   ├── models/                     # SQLAlchemy ORM models
│   │   ├── rfp_model.py            # RFP document record
│   │   ├── proposal_model.py       # Generated proposal draft
│   │   ├── chunk_model.py          # Knowledge base chunk + embedding vector
│   │   └── org_model.py            # Health system organisation profile
│   │
│   ├── schemas/
│   │   └── rfp_schema.py           # Pydantic request/response schemas
│   │
│   ├── services/
│   │   ├── ollama_client.py        # Async Ollama HTTP wrapper (chat + embed + stream)
│   │   └── knowledge_base_service.py # KB document ingestion and management
│   │
│   └── utils/
│       └── text_extractor.py       # PDF/DOCX/TXT extraction + text chunking
│
├── frontend/                       # React + Tailwind UI
│   └── src/
│       ├── pages/                  # Dashboard, RFP list, Proposals, KB
│       ├── components/             # Layout, sidebar
│       └── utils/                  # API client, helpers
│
├── tests/
│   ├── unit/
│   │   ├── test_parser.py
│   │   ├── test_compliance_checker.py
│   │   ├── test_ollama_client.py
│   │   ├── test_rag_retrieval.py   # 20 tests for all RAG pipeline stages
│   │   └── test_generator.py       # 32 tests for generator + PromptBuilder
│   └── integration/
│       └── test_api.py
│
├── scripts/
│   ├── init_db.py                  # Create tables, pgvector extension, HNSW index
│   └── seed_knowledge_base.py      # Embed sample health system documents
│
├── alembic/                        # DB migration environment
├── docker-compose.yml              # Postgres + Redis + Ollama + API
├── Dockerfile
├── Makefile                        # Developer convenience commands
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Pytest + coverage + ruff + mypy config
└── .env.example                    # Environment variable template
```

---

## RAG Retrieval Pipeline

Located in `app/agent/rag_retrieval.py` (full pipeline) and `app/agent/retriever.py` (simple retriever used by the orchestrator).

### How It Works

The retrieval system uses a **7-stage hybrid pipeline** powered entirely by Ollama local models — no external API calls.

```
User query
    │
    ▼  Stage 1 — Query Expansion
    │  Ollama rewrites query into 2–3 semantically richer variants.
    │  Catches relevant chunks phrased differently from the original query.
    │  Example: "HIPAA compliance" → ["PHI safeguards encryption",
    │                                  "healthcare data privacy rule BAA",
    │                                  "HIPAA security administrative controls"]
    │
    ▼  Stage 2 — Embedding
    │  Each query variant is embedded with Ollama nomic-embed-text (768-dim).
    │  Concurrent embedding with semaphore limit to avoid OOM.
    │
    ▼  Stage 3 — Multi-Query Vector Search
    │  Each embedding runs a pgvector cosine ANN search against knowledge_chunks.
    │  Results from all query variants are merged and deduplicated by chunk ID,
    │  keeping the highest similarity score per chunk.
    │  Uses HNSW index for sub-linear search time.
    │
    ▼  Stage 4 — BM25 Hybrid Rerank
    │  BM25 keyword scoring runs over the ANN candidate set.
    │  Hybrid score = 0.6 × vector_similarity + 0.4 × normalised_BM25
    │  This boosts exact keyword matches that semantic search under-weights.
    │
    ▼  Stage 5 — MMR Deduplication
    │  Maximal Marginal Relevance removes near-duplicate chunks
    │  that multi-query retrieval tends to surface multiple times.
    │  Formula: score(d) = λ × relevance(d) − (1−λ) × max_sim(d, selected)
    │  Default λ = 0.7  (70% relevance, 30% diversity)
    │
    ▼  Stage 6 — Context Builder
    │  Ranked chunks are formatted as numbered source blocks with metadata.
    │  Token budget is enforced — long chunks are truncated to fit the
    │  Ollama context window without overflowing.
    │
    ▼  Stage 7 — Answer Synthesis  (only in RAGPipeline.run())
       Ollama generates a grounded answer using ONLY the retrieved context.
       Instructs the model to cite sources inline with [Source N] notation
       and to never fabricate statistics or details not in the context.
```

### Class Reference

#### `RAGPipeline` — full 7-stage pipeline

```python
from app.agent.rag_retrieval import RAGPipeline

pipeline = RAGPipeline(
    db=db_session,
    top_k=6,                  # final chunks returned
    fetch_k_multiplier=4,     # over-fetch factor before reranking (fetch_k = top_k * 4)
    alpha=0.6,                # BM25 hybrid weight for vector similarity
    mmr_lambda=0.7,           # MMR relevance/diversity balance
    expand_queries=True,      # use Ollama for query expansion
    use_mmr=True,             # apply MMR deduplication
    max_context_tokens=4000,  # context window budget for synthesis
)

# Check Ollama is running
status = await pipeline.health_check()
# {"ollama_ready": True, "llm_model": "llama3.1:8b", "embed_model": "nomic-embed-text"}

# Retrieve chunks only (for use in proposal generator)
chunks = await pipeline.retrieve(
    query="What HIPAA compliance measures do we have?",
    org_id="demo-org-001",
    doc_types=["regulation", "past_proposal"],  # optional filter
    year_min=2022,                               # optional recency filter
)
# returns: list[RetrievedChunk]

# Full pipeline with answer synthesis
result = await pipeline.run(
    query="What HIPAA compliance measures do we have?",
    org_id="demo-org-001",
)
print(result.answer)   # grounded answer with [Source N] citations
print(result.sources)  # [{index, source_doc, doc_type, year, hybrid_score}]
print(result.retrieval_stats)

# Streaming answer (tokens arrive as they are generated)
async for event_type, data in pipeline.run_stream(query, org_id="demo-org-001"):
    if event_type == "sources":
        show_sources(data)
    elif event_type == "token":
        append_to_ui(data)
    elif event_type == "done":
        result = data   # final RAGResult
```

#### `HealthRFPRetriever` — simple retriever (used by orchestrator)

```python
from app.agent.retriever import HealthRFPRetriever

retriever = HealthRFPRetriever(db=db_session)

# Retrieve for one section
chunks = await retriever.retrieve_for_section(
    section_name="Technical Approach",
    section_requirements="Describe your methodology for managing Medi-Cal patients.",
    org_id="demo-org-001",
    doc_types=["past_proposal", "clinical_program", "regulation"],
    top_k=8,
)

# Retrieve for all sections at once (parallel)
chunks_by_section = await retriever.retrieve_batch(
    sections=[{"name": "Technical Approach", "requirements": "..."},
               {"name": "Past Performance", "requirements": "..."}],
    org_id="demo-org-001",
)
# returns: {"Technical Approach": [chunks], "Past Performance": [chunks]}
```

#### `RetrievedChunk` — data class returned by retrieval

```python
@dataclass
class RetrievedChunk:
    id:           str
    text:         str
    source_doc:   str
    doc_type:     str            # past_proposal | clinical_program | regulation | ...
    section_type: Optional[str]
    org_id:       Optional[str]
    year:         Optional[int]
    tags:         list
    vector_score: float          # cosine similarity (0–1)
    bm25_score:   float          # normalised BM25 (0–1)
    hybrid_score: float          # 0.6 × vector + 0.4 × bm25
    mmr_score:    float          # final score after MMR dedup
```

### Supported Knowledge Base Document Types

| `doc_type` | Description | Examples |
|---|---|---|
| `past_proposal` | Previous winning RFP submissions | Full DHCS Medi-Cal proposal |
| `clinical_program` | Programme descriptions and outcomes | Chronic disease management write-up |
| `regulation` | Regulatory and policy documents | HIPAA Security Rule, 42 CFR Part 2 |
| `past_performance` | Contract performance records | WPC pilot outcomes report |
| `team_bio` | Staff biographies and CVs | CMO bio, care coordinator profiles |
| `capability_statement` | Org capability and boilerplate | Standard capability statement |

### Ingesting Documents into the Knowledge Base

```bash
# Via API (file upload)
curl -X POST http://localhost:8000/api/v1/kb/ingest/file \
  -F "file=@dhcs_wpc_proposal_2022.pdf" \
  -F "doc_type=past_proposal" \
  -F "doc_name=DHCS WPC Proposal 2022" \
  -F "org_id=demo-org-001" \
  -F "year=2022"

# Via API (paste text)
curl -X POST http://localhost:8000/api/v1/kb/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Our HIPAA compliance programme includes...",
    "doc_type": "regulation",
    "doc_name": "HIPAA Compliance Overview",
    "org_id": "demo-org-001"
  }'

# Via Python script
python scripts/seed_knowledge_base.py
```

---

## Response Generator

Located in `app/agent/generator.py`.

### How It Works

The generator uses **strategy-aware prompting** to produce different tones, structures, and depth levels depending on which section of the proposal it is writing.

```
Section name
    │
    ▼  Strategy detection
    │  Maps section name to one of 7 strategies via keyword matching.
    │  e.g. "Technical Approach" → TECHNICAL strategy
    │       "Past Performance"   → PAST_PERF strategy
    │       "Data Security"      → DATA_SECURITY strategy
    │
    ▼  System prompt construction
    │  Base system prompt (20-year healthcare grant writer persona)
    │  + Strategy-specific addendum (structure guidance for this section type)
    │
    ▼  User prompt assembly
    │  ① Organisation profile (name, service area, certifications, mission)
    │  ② RFP requirements for this section (full requirements text + page limit)
    │  ③ Evaluation criteria with weights (tailor depth proportionally)
    │  ④ Knowledge base context (numbered, scored chunks from retriever)
    │  ⑤ Reviewer feedback + previous draft (only for regeneration)
    │  ⑥ Generation instruction (formatting rules, no placeholders)
    │
    │  Token budget enforced: trims KB chunks until prompt fits context window.
    │
    ▼  Ollama call
    │  Retry up to 3× with exponential back-off on TimeoutException.
    │  Falls back to secondary model (e.g. phi3:mini) if primary fails.
    │
    ▼  Post-processing
    │  Strip preambles ("Here is the Technical Approach section:")
    │  Strip trailing disclaimers ("Note: This is AI-generated.")
    │  Normalise excessive blank lines.
    │
    ▼  Optional quality self-evaluation
       Second Ollama call scores the output 0–10 across 5 criteria.
       Returns score + feedback string. Disabled by default (slower).
```

### Section Strategies

| Section keyword | Strategy | What the system prompt adds |
|---|---|---|
| "technical approach", "methodology" | `TECHNICAL` | Sub-headings per component, numbered steps, timeline, measurable outputs |
| "management plan", "project management" | `MANAGEMENT` | Org structure, roles, governance, communication plan, escalation |
| "past performance", "relevant experience" | `PAST_PERF` | Client → Scope → Role → Outcomes format, contract values, quantified results |
| "staffing plan", "key personnel" | `STAFFING` | Role/FTE table, bios, retention/training/succession |
| "budget", "cost proposal" | `BUDGET` | Personnel/Fringe/Travel/Equipment line-item structure, justification |
| "executive summary" | `EXECUTIVE` | Problem → solution → differentiators → commitment, one-page-reader focus |
| "data security", "hipaa", "privacy" | `DATA_SECURITY` | AES-256, TLS 1.3, BAA, HITRUST, SOC 2, incident response |
| anything else | `DEFAULT` | Clear headings, specificity, evidence-based, address all requirements |

### Class Reference

#### `RFPResponseGenerator` — main class

```python
from app.agent.generator import RFPResponseGenerator

gen = RFPResponseGenerator(
    primary_model="llama3.1:8b",    # Ollama model for generation
    fallback_model="phi3:mini",     # Used if primary times out after 3 retries
    context_window=8192,            # Max tokens for the model
    enable_quality_eval=False,      # True = second Ollama call scores each section
)
```

#### `generate_section()` — single section, blocking

```python
from app.agent.parser import RFPSection

section = RFPSection(
    name="Technical Approach",
    requirements="Describe your methodology for managing Medi-Cal patients with complex needs.",
    page_limit=10,
    order=0,
)

org_profile = {
    "name":                 "Valley Health System",
    "service_area":         "Santa Clara County, CA",
    "patient_population":   "Medi-Cal beneficiaries and uninsured",
    "annual_patient_volume": 120000,
    "certifications":       ["HIPAA", "HITRUST", "SOC 2"],
    "mission_statement":    "Providing quality healthcare to all community members.",
}

result = await gen.generate_section(
    section             = section,
    org_profile         = org_profile,
    retrieved_chunks    = chunks,     # from retriever.retrieve_for_section()
    evaluation_criteria = [{"criterion": "Technical Approach", "weight": 40}],
    temperature         = 0.35,
    max_tokens          = 3500,
)

print(result.content)            # generated section text (markdown)
print(result.word_count)         # e.g. 1240
print(result.generation_time_s)  # e.g. 47.3
print(result.model_used)         # "llama3.1:8b"
print(result.quality_score)      # e.g. 8.5 (if enable_quality_eval=True)
```

#### `generate_section_stream()` — real-time token streaming

```python
# Yields text tokens as they arrive from Ollama
async for token in gen.generate_section_stream(section, org_profile, chunks):
    print(token, end="", flush=True)

# With FastAPI StreamingResponse
async def sse_endpoint():
    async def event_gen():
        async for token in gen.generate_section_stream(section, org_profile, chunks):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")
```

#### `regenerate_section()` — revision with reviewer feedback

```python
result = await gen.regenerate_section(
    section           = section,
    org_profile       = org_profile,
    retrieved_chunks  = chunks,
    reviewer_feedback = (
        "The past performance section lacks contract values. "
        "Add the contract value for each example. "
        "Also mention our 100% on-time reporting rate."
    ),
    previous_draft    = draft.sections["Past Performance"].content,
)
```

#### `refine_section()` — targeted single-aspect improvement

```python
# Faster than full regeneration — only changes the specified aspect
improved = await gen.refine_section(
    section_name           = "Technical Approach",
    current_content        = existing_content,
    refinement_instruction = "Add a HIPAA compliance subsection after the Technology paragraph.",
)
```

#### `generate_full_proposal()` — all sections at once

```python
draft = await gen.generate_full_proposal(
    rfp               = rfp,             # RFPStructure from parser
    org_profile       = org_profile,
    chunks_by_section = chunks_by_section,  # from retriever.retrieve_batch()
    temperature       = 0.35,
    parallel          = False,   # True for concurrent (needs 24+ GB VRAM)
)

print(draft.summary())
# {
#   "rfp_title": "...",
#   "sections": 5,
#   "total_words": 6240,
#   "total_time_s": 284.1,
#   "model": "llama3.1:8b",
#   "avg_quality": 7.8
# }

# Get plain text dict for export
text_dict = draft.to_text_dict()
# {"Technical Approach": "...", "Past Performance": "...", ...}
```

#### `generate_executive_summary()` — written last

```python
# Call AFTER generate_full_proposal() so it reflects the actual content
exec_result = await gen.generate_executive_summary(
    rfp           = rfp,
    org_profile   = org_profile,
    section_drafts = draft.to_text_dict(),
    temperature   = 0.4,   # slightly higher for narrative prose
)
```

#### `GenerationResult` — return type

```python
@dataclass
class GenerationResult:
    section_name:      str
    content:           str            # full generated section (markdown)
    model_used:        str            # "llama3.1:8b" or fallback model
    generation_time_s: float          # wall-clock time in seconds
    prompt_tokens_est: int            # estimated prompt token count
    output_tokens_est: int            # estimated output token count
    quality_score:     Optional[float] # 0–10 from self-eval (if enabled)
    quality_feedback:  Optional[str]   # short feedback string from self-eval
    retry_count:       int             # number of retries before success
    warning:           Optional[str]   # set if generation failed partially
    word_count:        int             # property: len(content.split())
```

---

## API Reference

All endpoints are documented interactively at `http://localhost:8000/docs`.

### RAG Endpoints — `/api/v1/rag/`

| Method | Path | Description |
|---|---|---|
| `POST` | `/rag/query` | Full RAG pipeline — retrieve + synthesise answer |
| `POST` | `/rag/stream` | Streaming SSE answer with token-by-token output |
| `POST` | `/rag/retrieve` | Retrieve chunks only, no synthesis |
| `GET`  | `/rag/health` | Ollama connectivity + model availability |

**Example — full query:**

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are our HIPAA compliance measures?",
    "org_id": "demo-org-001",
    "top_k": 6,
    "expand_queries": true,
    "use_mmr": true
  }'
```

**Example — streaming:**

```javascript
const response = await fetch('/api/v1/rag/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: "What HIPAA measures do we have?", org_id: "demo-org-001" }),
});
const reader = response.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split('\n');
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const event = JSON.parse(line.slice(6));
      if (event.token) process.stdout.write(event.token);
    }
  }
}
```

### Generator Endpoints — `/api/v1/generate/`

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate/section` | Generate one section (blocking JSON) |
| `POST` | `/generate/section/stream` | Generate one section (SSE streaming) |
| `POST` | `/generate/regenerate` | Regenerate with reviewer feedback |
| `POST` | `/generate/refine` | Targeted single-aspect refinement |
| `POST` | `/generate/executive-summary` | Generate executive summary |
| `GET`  | `/generate/models` | List available Ollama models |
| `GET`  | `/generate/strategies` | Show section-to-strategy mapping |

**Example — generate section:**

```bash
curl -X POST http://localhost:8000/api/v1/generate/section \
  -H "Content-Type: application/json" \
  -d '{
    "section_name": "Technical Approach",
    "section_requirements": "Describe your care coordination methodology for Medi-Cal members.",
    "page_limit": 10,
    "org_profile": {
      "name": "Valley Health System",
      "service_area": "Santa Clara County, CA",
      "certifications": ["HIPAA", "HITRUST"]
    },
    "temperature": 0.35,
    "primary_model": "llama3.1:8b"
  }'
```

**Example — regenerate with feedback:**

```bash
curl -X POST http://localhost:8000/api/v1/generate/regenerate \
  -H "Content-Type: application/json" \
  -d '{
    "section_name": "Past Performance",
    "section_requirements": "Provide 3 examples of similar contracts.",
    "org_profile": { "name": "Valley Health System", "certifications": ["HIPAA"] },
    "reviewer_feedback": "Add contract values and be more specific about outcomes.",
    "previous_draft": "We have extensive experience in population health..."
  }'
```

### RFP Endpoints — `/api/v1/rfp/`

| Method | Path | Description |
|---|---|---|
| `POST` | `/rfp/upload` | Upload PDF/DOCX RFP, triggers parse |
| `POST` | `/rfp/sam-gov/search` | Search SAM.gov by NAICS code |
| `GET`  | `/rfp/{id}` | Get parsed RFP record |
| `GET`  | `/rfp/` | List all RFPs |

### Knowledge Base Endpoints — `/api/v1/kb/`

| Method | Path | Description |
|---|---|---|
| `POST` | `/kb/ingest/file` | Upload + embed document |
| `POST` | `/kb/ingest/text` | Embed plain text |
| `GET`  | `/kb/documents` | List KB documents |
| `DELETE` | `/kb/documents` | Remove a document |
| `GET`  | `/kb/stats` | KB statistics (chunks, docs, words) |
| `GET`  | `/kb/models` | List pulled Ollama models |
| `GET`  | `/kb/health/ollama` | Ollama health check |

---

## Ollama Model Guide

| Model | Size | Use case | VRAM | Speed |
|---|---|---|---|---|
| `llama3.1:8b` | 4.7 GB | **Recommended default** — good quality, reasonable speed | 8 GB | ★★★☆ |
| `llama3.1:70b` | 40 GB | Best quality — use for final proposals | 40+ GB | ★★☆☆ |
| `mistral:7b` | 4.1 GB | Strong instruction following, good for structured output | 8 GB | ★★★★ |
| `phi3:mini` | 2.2 GB | **Fallback model** — very fast, limited RAM machines | 4 GB | ★★★★★ |
| `medllama2:7b` | 3.8 GB | Healthcare-fine-tuned — best clinical language | 8 GB | ★★★☆ |
| `biomistral:7b` | 4.1 GB | Biomedical domain — strong for HIPAA/clinical sections | 8 GB | ★★★☆ |
| `gemma2:9b` | 5.5 GB | Strong reasoning — good for compliance checks | 10 GB | ★★★☆ |
| `nomic-embed-text` | 274 MB | **Required** — 768-dim embeddings for RAG retrieval | < 1 GB | ★★★★★ |

**Recommended setup:**

```env
# .env — recommended for 16 GB RAM machine
OLLAMA_LLM_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
# Add to generator: fallback_model="phi3:mini"
```

**Healthcare-optimised setup (best quality):**

```env
OLLAMA_LLM_MODEL=medllama2:7b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

---

## Configuration

All settings live in `.env`. Full reference:

```env
# ── App ──────────────────────────────────────────────
APP_NAME=HealthRFPAgent
APP_ENV=development         # development | production
DEBUG=true
SECRET_KEY=change-me-in-production

# ── Database ──────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/health_rfp_db
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# ── Ollama (local LLM — no API key needed) ────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1:8b      # LLM for generation, parsing, compliance
OLLAMA_EMBED_MODEL=nomic-embed-text # Embedding model for RAG retrieval
OLLAMA_TIMEOUT=120                  # Seconds — increase for 70B models
OLLAMA_CONTEXT_WINDOW=8192          # Token context window for the model

# ── Retrieval ─────────────────────────────────────────
RETRIEVAL_TOP_K=6          # Final chunks returned per section
CHUNK_SIZE=512             # Words per KB chunk
CHUNK_OVERLAP=64           # Overlap between adjacent chunks

# ── SAM.gov (optional) ────────────────────────────────
SAM_GOV_API_KEY=your-key   # Free at sam.gov/content/api

# ── Redis (task queue) ────────────────────────────────
REDIS_URL=redis://localhost:6379/0

# ── CORS ──────────────────────────────────────────────
ALLOWED_ORIGINS=http://localhost:3000

# ── Logging ───────────────────────────────────────────
LOG_LEVEL=INFO             # DEBUG | INFO | WARNING | ERROR
```

---

## Database Setup

The knowledge base uses PostgreSQL with the pgvector extension for vector similarity search.

```sql
-- Enable pgvector (run once after creating the database)
CREATE EXTENSION IF NOT EXISTS vector;

-- HNSW index created automatically by scripts/init_db.py
-- Manual creation if needed:
CREATE INDEX idx_chunks_embedding_hnsw
ON knowledge_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Schema overview:**

| Table | Purpose |
|---|---|
| `rfps` | Ingested RFP documents with parsed structure |
| `proposals` | Generated proposal drafts and compliance reports |
| `knowledge_chunks` | KB document chunks with 768-dim embedding vectors |
| `organizations` | Health system profiles (certifications, mission, contacts) |

---

## Running Tests

```bash
# All tests
pytest

# Unit tests only (no running services needed — all Ollama calls mocked)
pytest tests/unit/ -v

# Specific module
pytest tests/unit/test_generator.py -v
pytest tests/unit/test_rag_retrieval.py -v

# With coverage report
pytest --cov=app --cov-report=term-missing

# Integration tests (requires running Postgres)
pytest tests/integration/ -v
```

Tests are designed to run without a live Ollama instance — all LLM calls are mocked with `AsyncMock`.

---

## Docker

```bash
# Start all services (Postgres + pgvector, Redis, Ollama, API)
docker compose up -d

# Check service health
docker compose ps

# View API logs
docker compose logs api -f

# Pull Ollama models inside the container
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text

# Stop all services
docker compose down
```

### Docker services

| Service | Port | Description |
|---|---|---|
| `db` | 5432 | PostgreSQL 16 + pgvector |
| `redis` | 6379 | Redis (task queue + cache) |
| `ollama` | 11434 | Ollama LLM server |
| `api` | 8000 | FastAPI application |

---

## Makefile Commands

```bash
make up           # Start all Docker services
make down         # Stop all Docker services
make init-db      # Create tables, pgvector extension, seed demo org
make seed         # Embed sample health documents into knowledge base
make pull-models  # Pull llama3.1:8b + nomic-embed-text via Ollama
make dev          # Run API locally with hot reload
make test         # Run full test suite
make test-unit    # Unit tests only
make lint         # Run ruff linter
make fmt          # Auto-format with black
```

---

## License

MIT License — see LICENSE file.

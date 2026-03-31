"""
app/api/rag.py

RAG Query API
=============
POST /api/v1/rag/query          — full pipeline, returns JSON answer + sources
POST /api/v1/rag/stream         — streaming SSE response, tokens arrive in real-time
POST /api/v1/rag/retrieve       — retrieve chunks only (no answer synthesis)
GET  /api/v1/rag/health         — Ollama + pipeline health check

Mount in app/main.py:
    from app.api import rag
    app.include_router(rag.router, prefix="/api/v1/rag", tags=["rag"])
"""
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.rag_retrieval import RAGPipeline
from app.core.dependencies import get_db
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ── Request / Response schemas ────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="The question to answer")
    org_id: Optional[str] = Field(None, description="Limit retrieval to this org's KB documents")
    doc_types: Optional[list[str]] = Field(
        None,
        description="Filter by doc type: past_proposal | clinical_program | regulation | etc.",
    )
    year_min: Optional[int] = Field(None, description="Only return documents from this year onwards")
    top_k: int = Field(6, ge=1, le=20, description="Number of chunks to retrieve")
    expand_queries: bool = Field(True, description="Use Ollama to expand the query before retrieval")
    use_mmr: bool = Field(True, description="Apply MMR deduplication")
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="LLM sampling temperature")


class RetrieveOnlyRequest(BaseModel):
    query: str = Field(..., min_length=3)
    org_id: Optional[str] = None
    doc_types: Optional[list[str]] = None
    year_min: Optional[int] = None
    top_k: int = Field(6, ge=1, le=20)
    expand_queries: bool = True


class ChunkResponse(BaseModel):
    id: str
    text: str
    source_doc: str
    doc_type: str
    section_type: Optional[str]
    year: Optional[int]
    vector_score: float
    bm25_score: float
    hybrid_score: float
    mmr_score: float


class RAGQueryResponse(BaseModel):
    query: str
    expanded_queries: list[str]
    answer: str
    sources: list[dict]
    retrieval_stats: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(
    body: RAGQueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Full RAG pipeline — retrieve relevant chunks from the knowledge base
    and synthesise a grounded answer using Ollama.

    The pipeline runs:
      1. Query expansion (Ollama rewrites the query into 2–3 variants)
      2. Embedding (nomic-embed-text)
      3. Multi-query pgvector search
      4. BM25 hybrid rerank
      5. MMR deduplication
      6. Context assembly
      7. Answer synthesis (Ollama LLM)
    """
    pipeline = RAGPipeline(
        db=db,
        top_k=body.top_k,
        expand_queries=body.expand_queries,
        use_mmr=body.use_mmr,
    )

    try:
        result = await pipeline.run(
            query=body.query,
            org_id=body.org_id,
            doc_types=body.doc_types,
            year_min=body.year_min,
            temperature=body.temperature,
        )
    except Exception as exc:
        logger.error("rag_query_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {exc}")

    return RAGQueryResponse(
        query=result.query,
        expanded_queries=result.expanded_queries,
        answer=result.answer,
        sources=result.sources,
        retrieval_stats=result.retrieval_stats,
    )


@router.post("/stream")
async def rag_stream(
    body: RAGQueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Streaming RAG response using Server-Sent Events (SSE).

    Events emitted:
      data: {"sources": [...]}          — source list (sent first)
      data: {"token": "word"}           — one token at a time from Ollama
      data: {"done": true}              — pipeline complete

    Client example (JavaScript):
      const es = new EventSource('/api/v1/rag/stream', {method:'POST', body: JSON.stringify(req)});
      es.onmessage = e => {
        const d = JSON.parse(e.data);
        if (d.token) appendToken(d.token);
        if (d.sources) showSources(d.sources);
        if (d.done) es.close();
      };
    """
    pipeline = RAGPipeline(
        db=db,
        top_k=body.top_k,
        expand_queries=body.expand_queries,
        use_mmr=body.use_mmr,
    )

    async def event_generator():
        try:
            async for event_type, data in pipeline.run_stream(
                query=body.query,
                org_id=body.org_id,
                doc_types=body.doc_types,
            ):
                if event_type == "sources":
                    yield f"data: {json.dumps({'sources': data})}\n\n"
                elif event_type == "token":
                    yield f"data: {json.dumps({'token': data})}\n\n"
                elif event_type == "done":
                    yield f"data: {json.dumps({'done': True, 'stats': data.retrieval_stats})}\n\n"
        except Exception as exc:
            logger.error("rag_stream_failed", error=str(exc))
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


@router.post("/retrieve", response_model=list[ChunkResponse])
async def retrieve_chunks(
    body: RetrieveOnlyRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve relevant chunks only — no answer synthesis.
    Useful when you want to feed the chunks into your own LLM prompt.
    """
    pipeline = RAGPipeline(
        db=db,
        top_k=body.top_k,
        expand_queries=body.expand_queries,
    )

    try:
        chunks = await pipeline.retrieve(
            query=body.query,
            org_id=body.org_id,
            doc_types=body.doc_types,
            year_min=body.year_min,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return [
        ChunkResponse(
            id=c.id,
            text=c.text,
            source_doc=c.source_doc,
            doc_type=c.doc_type,
            section_type=c.section_type,
            year=c.year,
            vector_score=c.vector_score,
            bm25_score=c.bm25_score,
            hybrid_score=c.hybrid_score,
            mmr_score=c.mmr_score,
        )
        for c in chunks
    ]


@router.get("/health")
async def rag_health():
    """Check Ollama connectivity and model availability."""
    # Use a dummy DB session just for the health check
    from app.db.session import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        pipeline = RAGPipeline(db)
        status = await pipeline.health_check()

    if not status["ollama_ready"]:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama not ready. Ensure the following models are pulled:\n"
                f"  ollama pull {status['llm_model']}\n"
                f"  ollama pull {status['embed_model']}"
            ),
        )
    return status

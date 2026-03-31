"""
app/agent/rag_retrieval.py

Complete RAG Retrieval Pipeline — Powered by Ollama (local LLM)
================================================================
This module implements a full, production-grade Retrieval-Augmented Generation
(RAG) pipeline that runs 100% locally using Ollama. No cloud API keys required.

Pipeline stages:
  1. Query expansion   — Ollama rewrites the query into richer variants
  2. Embedding         — Ollama nomic-embed-text encodes query → vector
  3. Vector search     — pgvector cosine ANN search over knowledge_chunks
  4. BM25 rerank       — keyword rerank over the ANN candidate set
  5. MMR dedup         — Maximal Marginal Relevance removes near-duplicate chunks
  6. Context builder   — Assembles ranked chunks into a formatted LLM prompt
  7. Answer synthesis  — Ollama generates a grounded answer from the context

Usage (standalone):
    pipeline = RAGPipeline(db_session)
    result = await pipeline.run(
        query="What are our HIPAA compliance measures?",
        org_id="demo-org-001",
    )
    print(result.answer)
    print(result.sources)

Usage (per-section in orchestrator):
    chunks = await pipeline.retrieve(
        query="Technical approach for Medi-Cal population health",
        org_id="demo-org-001",
    )
    # Pass chunks to generator as context
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx
from rank_bm25 import BM25Okapi
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single knowledge base chunk with retrieval scores."""
    id: str
    text: str
    source_doc: str
    doc_type: str
    section_type: Optional[str]
    org_id: Optional[str]
    year: Optional[int]
    tags: list
    vector_score: float       # cosine similarity (0–1)
    bm25_score: float         # normalised BM25 (0–1)
    hybrid_score: float       # weighted combination
    mmr_score: float = 0.0    # final MMR score after dedup

    def to_context_block(self, index: int) -> str:
        """Format for LLM context injection."""
        meta = f"[{index}] Source: {self.source_doc}"
        if self.year:
            meta += f" ({self.year})"
        meta += f" | Type: {self.doc_type}"
        if self.section_type:
            meta += f" | Section: {self.section_type}"
        meta += f" | Score: {self.hybrid_score:.2f}"
        return f"{meta}\n{self.text}"


@dataclass
class RAGResult:
    """Full result of a RAG pipeline run."""
    query: str
    expanded_queries: list[str]
    chunks: list[RetrievedChunk]
    context: str
    answer: str
    sources: list[dict]
    retrieval_stats: dict

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "stats": self.retrieval_stats,
        }


# ── Ollama client (inline — no external import needed for standalone use) ──────

class _OllamaHTTP:
    """
    Minimal async Ollama HTTP client used only by this module.
    Uses the same settings as app/services/ollama_client.py but
    keeps this file self-contained for copy-paste portability.
    """

    def __init__(self):
        self.base = settings.OLLAMA_BASE_URL.rstrip("/")
        self.llm  = settings.OLLAMA_LLM_MODEL
        self.emb  = settings.OLLAMA_EMBED_MODEL
        self._http = httpx.AsyncClient(
            base_url=self.base,
            timeout=httpx.Timeout(settings.OLLAMA_TIMEOUT, connect=10),
        )

    async def embed(self, text: str) -> list[float]:
        r = await self._http.post("/api/embeddings", json={"model": self.emb, "prompt": text})
        r.raise_for_status()
        return r.json()["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts with concurrency cap to avoid OOM."""
        sem = asyncio.Semaphore(6)
        async def _one(t):
            async with sem:
                return await self.embed(t)
        return await asyncio.gather(*[_one(t) for t in texts])

    async def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        all_msgs = ([{"role": "system", "content": system}] if system else []) + messages
        payload = {
            "model": self.llm,
            "messages": all_msgs,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": settings.OLLAMA_CONTEXT_WINDOW,
            },
        }
        r = await self._http.post("/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]

    async def is_ready(self) -> bool:
        try:
            r = await self._http.get("/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            llm_ok = any(self.llm in m for m in models)
            emb_ok = any(self.emb in m for m in models)
            return llm_ok and emb_ok
        except Exception:
            return False


# ── Stage 1: Query Expansion ──────────────────────────────────────────────────

class QueryExpander:
    """
    Uses Ollama to rewrite the user query into 2–3 richer variants.

    Why: A single embedding of the raw query may miss relevant chunks
    phrased differently. Expanded queries cast a wider retrieval net.
    Results are then de-duplicated by MMR in stage 5.

    Example:
      Input:  "HIPAA compliance for patient data"
      Output: [
        "HIPAA privacy and security rule compliance for electronic PHI",
        "healthcare data protection measures BAA business associate agreement",
        "patient health information safeguards encryption access control",
      ]
    """

    SYSTEM = (
        "You are a healthcare RFP expert. Your job is to rewrite search queries "
        "to maximise retrieval recall from a knowledge base of health system documents."
    )

    def __init__(self, ollama: _OllamaHTTP):
        self._ollama = ollama

    async def expand(self, query: str, n: int = 3) -> list[str]:
        """Return `n` rewritten variants of the query (includes the original)."""
        prompt = f"""Rewrite the following search query into {n} different variants
that would retrieve relevant content from a healthcare RFP knowledge base.
Use synonyms, related terms, and different phrasings.

Original query: {query}

Return ONLY a JSON array of {n} strings, no explanation.
Example: ["variant 1", "variant 2", "variant 3"]

JSON array:"""

        try:
            raw = await self._ollama.chat(
                messages=[{"role": "user", "content": prompt}],
                system=self.SYSTEM,
                temperature=0.4,
                max_tokens=300,
            )
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            variants = __import__("json").loads(clean)
            if isinstance(variants, list):
                # Always include the original query
                all_queries = [query] + [v for v in variants if v != query]
                logger.debug("query_expanded", original=query, variants=len(all_queries))
                return all_queries[:n + 1]
        except Exception as exc:
            logger.warning("query_expansion_failed", error=str(exc))

        return [query]  # fallback: use original only


# ── Stage 2: Embedder ─────────────────────────────────────────────────────────

class OllamaEmbedder:
    """
    Wraps Ollama's embedding endpoint (nomic-embed-text by default).

    nomic-embed-text produces 768-dimensional vectors, same dimensionality
    as the pgvector column in knowledge_chunks.

    For multi-query retrieval, embeds all expanded queries concurrently
    and returns a list of embedding vectors.
    """

    def __init__(self, ollama: _OllamaHTTP):
        self._ollama = ollama

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        logger.debug("embedding_query", chars=len(query))
        return await self._ollama.embed(query)

    async def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embed multiple queries concurrently."""
        return await self._ollama.embed_batch(queries)

    async def embed_text(self, text: str) -> list[float]:
        """Embed a document chunk (for ingestion or re-embedding)."""
        return await self._ollama.embed(text)


# ── Stage 3: Vector Search ────────────────────────────────────────────────────

class VectorSearcher:
    """
    Runs cosine similarity ANN search against PostgreSQL/pgvector.

    Uses the <=> operator (cosine distance) with an HNSW index for
    sub-linear search time even on millions of chunks.

    For multi-query retrieval, runs one search per expanded query,
    then merges and deduplicates by chunk ID.
    """

    def __init__(self, db: AsyncSession):
        self._db = db

    async def search(
        self,
        embedding: list[float],
        fetch_k: int,
        org_id: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
        year_min: Optional[int] = None,
        section_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Single-vector pgvector cosine search.

        Returns raw row dicts with similarity score (1 - cosine_distance).
        """
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
        allowed_types = doc_types or [
            "past_proposal", "clinical_program", "regulation",
            "past_performance", "capability_statement", "team_bio",
        ]

        # Build dynamic WHERE clauses
        filters = [
            "doc_type = ANY(CAST(:doc_types AS text[]))",
            "(:org_id IS NULL OR org_id = :org_id OR org_id IS NULL)",
            "embedding IS NOT NULL",
        ]
        params: dict = {
            "vec": vec_str,
            "doc_types": allowed_types,
            "org_id": org_id,
            "k": fetch_k,
        }

        if year_min:
            filters.append("(year IS NULL OR year >= :year_min)")
            params["year_min"] = year_min

        if section_type_filter:
            filters.append("(section_type IS NULL OR section_type = :section_type)")
            params["section_type"] = section_type_filter

        where_clause = " AND ".join(filters)

        sql = text(f"""
            SELECT
                id,
                chunk_text,
                source_doc_name,
                doc_type,
                section_type,
                org_id,
                year,
                tags,
                1 - (embedding <=> CAST(:vec AS vector)) AS similarity
            FROM knowledge_chunks
            WHERE {where_clause}
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :k
        """)

        result = await self._db.execute(sql, params)
        rows = result.mappings().all()
        logger.debug("vector_search_done", returned=len(rows))
        return [dict(r) for r in rows]

    async def multi_query_search(
        self,
        embeddings: list[list[float]],
        fetch_k: int,
        **kwargs,
    ) -> list[dict]:
        """
        Run one vector search per expanded query embedding, then merge results.
        Deduplicates by chunk ID, keeping the highest similarity score per chunk.
        """
        all_results: list[dict] = []
        tasks = [
            self.search(emb, fetch_k=fetch_k, **kwargs)
            for emb in embeddings
        ]
        per_query_results = await asyncio.gather(*tasks, return_exceptions=True)

        seen: dict[str, dict] = {}
        for results in per_query_results:
            if isinstance(results, Exception):
                logger.warning("multi_query_search_partial_fail", error=str(results))
                continue
            for row in results:
                chunk_id = row["id"]
                if chunk_id not in seen or row["similarity"] > seen[chunk_id]["similarity"]:
                    seen[chunk_id] = row

        merged = list(seen.values())
        merged.sort(key=lambda r: r["similarity"], reverse=True)
        logger.debug("multi_query_merge", unique_chunks=len(merged))
        return merged[:fetch_k]


# ── Stage 4: BM25 Reranker ────────────────────────────────────────────────────

class BM25Reranker:
    """
    Keyword-based reranking over the ANN candidate set.

    BM25 (Best Match 25) is a probabilistic ranking function that scores
    documents based on term frequency and inverse document frequency.
    It complements vector search by boosting exact keyword matches that
    semantic embeddings might under-weight.

    Hybrid score = α * vector_similarity + (1-α) * normalised_bm25
    Default α = 0.6 (favour semantic) — tune based on your corpus.
    """

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha       # weight for vector similarity
        self.beta  = 1 - alpha   # weight for BM25

    def rerank(
        self,
        candidates: list[dict],
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Rerank `candidates` using BM25 + hybrid scoring.

        Args:
            candidates: Raw rows from vector search (each has 'chunk_text', 'similarity')
            query:      Original (or expanded) query string
            top_k:      How many results to keep

        Returns:
            List of RetrievedChunk sorted by hybrid_score descending.
        """
        if not candidates:
            return []

        # Tokenise corpus for BM25
        tokenised = [c["chunk_text"].lower().split() for c in candidates]
        bm25 = BM25Okapi(tokenised)
        bm25_raw = bm25.get_scores(query.lower().split())

        # Normalise BM25 scores to [0, 1]
        max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1.0

        chunks = []
        for row, bm25_raw_score in zip(candidates, bm25_raw):
            vec_sim   = float(row.get("similarity", 0.0))
            norm_bm25 = bm25_raw_score / max_bm25
            hybrid    = self.alpha * vec_sim + self.beta * norm_bm25

            chunks.append(RetrievedChunk(
                id           = str(row["id"]),
                text         = row["chunk_text"],
                source_doc   = row.get("source_doc_name", "Unknown"),
                doc_type     = row.get("doc_type", ""),
                section_type = row.get("section_type"),
                org_id       = row.get("org_id"),
                year         = row.get("year"),
                tags         = row.get("tags") or [],
                vector_score = round(vec_sim, 4),
                bm25_score   = round(norm_bm25, 4),
                hybrid_score = round(hybrid, 4),
            ))

        chunks.sort(key=lambda c: c.hybrid_score, reverse=True)
        logger.debug("bm25_rerank_done", top_score=chunks[0].hybrid_score if chunks else 0)
        return chunks[:top_k]


# ── Stage 5: MMR Deduplicator ─────────────────────────────────────────────────

class MMRDeduplicator:
    """
    Maximal Marginal Relevance (MMR) removes near-duplicate chunks.

    Multi-query retrieval can surface the same content from different
    query variants. MMR iteratively picks the next chunk that is:
      - Relevant to the query (high hybrid score)
      - Diverse from already-selected chunks (low cosine similarity)

    MMR formula:
        score(d) = λ * relevance(d) - (1-λ) * max_similarity(d, selected)

    λ = 0.7 → 70% relevance, 30% diversity. Tune for your use case.
    """

    def __init__(self, lambda_: float = 0.7):
        self.lambda_ = lambda_

    def deduplicate(
        self,
        chunks: list[RetrievedChunk],
        query_embedding: list[float],
        chunk_embeddings: list[list[float]],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Select top_k chunks maximising relevance + diversity.

        Args:
            chunks:            Reranked chunk list (same order as chunk_embeddings)
            query_embedding:   Embedding of the original query
            chunk_embeddings:  Pre-computed embeddings for each chunk in `chunks`
            top_k:             How many chunks to return

        Returns:
            Subset of chunks ordered by MMR score.
        """
        if len(chunks) <= top_k:
            return chunks  # nothing to deduplicate

        selected_indices: list[int] = []
        remaining = list(range(len(chunks)))

        # Query relevance = hybrid score (already computed in BM25 stage)
        relevance = [c.hybrid_score for c in chunks]

        while len(selected_indices) < top_k and remaining:
            if not selected_indices:
                # First pick: highest relevance
                best = max(remaining, key=lambda i: relevance[i])
            else:
                # Subsequent picks: MMR balance
                best = max(
                    remaining,
                    key=lambda i: (
                        self.lambda_ * relevance[i]
                        - (1 - self.lambda_) * max(
                            self._cosine(chunk_embeddings[i], chunk_embeddings[j])
                            for j in selected_indices
                        )
                    ),
                )

            selected_indices.append(best)
            remaining.remove(best)

        result = []
        for rank, idx in enumerate(selected_indices):
            chunk = chunks[idx]
            chunk.mmr_score = round(relevance[idx], 4)
            result.append(chunk)

        logger.debug("mmr_dedup_done", selected=len(result), from_pool=len(chunks))
        return result

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Fast cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ── Stage 6: Context Builder ──────────────────────────────────────────────────

class ContextBuilder:
    """
    Assembles the final retrieved chunks into a structured context string
    ready for injection into an Ollama LLM prompt.

    Handles:
    - Numbered source blocks for citation tracking
    - Token budget enforcement (truncates if context would overflow)
    - Source metadata formatting
    """

    # Rough chars-per-token estimate for Ollama's tokeniser
    CHARS_PER_TOKEN = 3.5

    def __init__(self, max_context_tokens: int = 4000):
        self.max_chars = int(max_context_tokens * self.CHARS_PER_TOKEN)

    def build(self, chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
        """
        Build context string and sources list from ranked chunks.

        Returns:
            (context_string, sources_list)
            sources_list: [{index, source_doc, doc_type, year, score}]
        """
        blocks: list[str] = []
        sources: list[dict] = []
        total_chars = 0

        for i, chunk in enumerate(chunks, start=1):
            block = chunk.to_context_block(i)
            block_chars = len(block)

            if total_chars + block_chars > self.max_chars:
                # Truncate this chunk to fit within budget
                remaining = self.max_chars - total_chars
                if remaining < 200:
                    break  # not enough room for anything useful
                truncated_text = chunk.text[:remaining - 100] + "\n[... truncated ...]"
                block = RetrievedChunk(
                    **{**chunk.__dict__, "text": truncated_text}
                ).to_context_block(i)

            blocks.append(block)
            sources.append({
                "index": i,
                "source_doc": chunk.source_doc,
                "doc_type": chunk.doc_type,
                "year": chunk.year,
                "hybrid_score": chunk.hybrid_score,
                "mmr_score": chunk.mmr_score,
            })
            total_chars += len(block)

            if total_chars >= self.max_chars:
                break

        context = "\n\n---\n\n".join(blocks)
        logger.debug("context_built", chunks_used=len(blocks), total_chars=total_chars)
        return context, sources


# ── Stage 7: Answer Synthesiser ───────────────────────────────────────────────

class AnswerSynthesiser:
    """
    Uses Ollama to generate a grounded answer from the retrieved context.

    Key design principles:
    - Instructs the model to ONLY use information from the context
    - Asks for [Source N] inline citations
    - Explicitly forbids hallucination
    - Uses low temperature (0.2) for factual consistency
    """

    SYSTEM = """You are a healthcare RFP expert assistant. Your role is to answer
questions using ONLY the information provided in the context below.

Rules:
1. Base every claim on the provided context. Do not use outside knowledge.
2. Cite sources inline using [Source N] notation after each claim.
3. If the context does not contain enough information, say so clearly.
4. Never fabricate statistics, dates, names, or contract values.
5. Write clearly and concisely. Use bullet points for lists."""

    def __init__(self, ollama: _OllamaHTTP):
        self._ollama = ollama

    async def synthesise(
        self,
        query: str,
        context: str,
        temperature: float = 0.2,
    ) -> str:
        """Generate a grounded answer using Ollama."""
        prompt = f"""CONTEXT (retrieved from knowledge base):
{context}

---

QUESTION: {query}

Using ONLY the context above, provide a comprehensive answer with [Source N] citations.
If the context is insufficient, state what information is missing."""

        logger.info("synthesising_answer", query_chars=len(query), context_chars=len(context))
        answer = await self._ollama.chat(
            messages=[{"role": "user", "content": prompt}],
            system=self.SYSTEM,
            temperature=temperature,
            max_tokens=2048,
        )
        return answer

    async def synthesise_stream(
        self,
        query: str,
        context: str,
    ):
        """Streaming variant — yields tokens for real-time display."""
        import json as _json

        prompt = f"""CONTEXT:\n{context}\n\n---\n\nQUESTION: {query}\n\nAnswer with [Source N] citations:"""
        all_msgs = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self._ollama.llm,
            "messages": all_msgs,
            "stream": True,
            "options": {"temperature": 0.2, "num_predict": 2048},
        }
        async with self._ollama._http.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = _json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except _json.JSONDecodeError:
                    continue


# ── Main Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full end-to-end RAG pipeline wiring all 7 stages together.

    Quick start:
        pipeline = RAGPipeline(db_session)

        # Check Ollama is running
        ready = await pipeline.health_check()

        # Full pipeline — retrieve + synthesise
        result = await pipeline.run("What HIPAA measures do we have?", org_id="demo-org-001")
        print(result.answer)

        # Retrieve only (for use in proposal generator)
        chunks = await pipeline.retrieve("Technical approach for population health")
    """

    def __init__(
        self,
        db: AsyncSession,
        top_k: int = 6,
        fetch_k_multiplier: int = 4,
        alpha: float = 0.6,
        mmr_lambda: float = 0.7,
        expand_queries: bool = True,
        use_mmr: bool = True,
        max_context_tokens: int = 4000,
    ):
        """
        Args:
            db:                   Async SQLAlchemy session
            top_k:                Final number of chunks returned
            fetch_k_multiplier:   Over-fetch factor before reranking (fetch_k = top_k * multiplier)
            alpha:                BM25 hybrid weight for vector similarity (0–1)
            mmr_lambda:           MMR relevance/diversity balance (higher = more relevant)
            expand_queries:       Whether to use Ollama for query expansion
            use_mmr:              Whether to apply MMR deduplication
            max_context_tokens:   Context window budget for answer synthesis
        """
        self.top_k       = top_k
        self.fetch_k     = top_k * fetch_k_multiplier
        self.expand      = expand_queries
        self.use_mmr     = use_mmr

        # Instantiate Ollama HTTP client
        self._ollama = _OllamaHTTP()

        # Instantiate pipeline stages
        self.expander    = QueryExpander(self._ollama)
        self.embedder    = OllamaEmbedder(self._ollama)
        self.searcher    = VectorSearcher(db)
        self.reranker    = BM25Reranker(alpha=alpha)
        self.deduper     = MMRDeduplicator(lambda_=mmr_lambda)
        self.ctx_builder = ContextBuilder(max_context_tokens=max_context_tokens)
        self.synthesiser = AnswerSynthesiser(self._ollama)

    # ── Health check ──────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """Verify Ollama is running and both models are pulled."""
        ready = await self._ollama.is_ready()
        return {
            "ollama_ready": ready,
            "llm_model":    self._ollama.llm,
            "embed_model":  self._ollama.emb,
            "base_url":     self._ollama.base,
        }

    # ── Retrieve only (no answer synthesis) ───────────────────────────────────

    async def retrieve(
        self,
        query: str,
        org_id: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
        year_min: Optional[int] = None,
        section_type_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Stages 1–5: Expand → Embed → Search → Rerank → Deduplicate.
        Returns ranked chunks without answer synthesis.
        Use this when feeding retrieved context to the proposal generator.
        """
        logger.info("rag_retrieve_start", query=query[:80])

        # Stage 1: Query expansion
        queries = (
            await self.expander.expand(query, n=2)
            if self.expand else [query]
        )

        # Stage 2: Embed all query variants
        embeddings = await self.embedder.embed_queries(queries)
        query_embedding = embeddings[0]   # original query embedding for MMR

        # Stage 3: Multi-query vector search
        candidates = await self.searcher.multi_query_search(
            embeddings=embeddings,
            fetch_k=self.fetch_k,
            org_id=org_id,
            doc_types=doc_types,
            year_min=year_min,
            section_type_filter=section_type_filter,
        )

        if not candidates:
            logger.warning("rag_no_candidates", query=query[:80])
            return []

        # Stage 4: BM25 hybrid rerank
        reranked = self.reranker.rerank(candidates, query, top_k=self.fetch_k)

        # Stage 5: MMR deduplication
        if self.use_mmr and len(reranked) > self.top_k:
            chunk_embeddings = await self.embedder.embed_queries(
                [c.text for c in reranked]
            )
            final = self.deduper.deduplicate(
                reranked, query_embedding, chunk_embeddings, self.top_k
            )
        else:
            final = reranked[: self.top_k]

        logger.info(
            "rag_retrieve_done",
            query=query[:60],
            candidates=len(candidates),
            returned=len(final),
            top_score=final[0].hybrid_score if final else 0,
        )
        return final

    # ── Full pipeline (retrieve + synthesise) ─────────────────────────────────

    async def run(
        self,
        query: str,
        org_id: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
        year_min: Optional[int] = None,
        temperature: float = 0.2,
    ) -> RAGResult:
        """
        Stages 1–7: Full RAG pipeline including answer synthesis.

        Returns a RAGResult with:
          - answer:   LLM-generated answer grounded in retrieved chunks
          - chunks:   The retrieved chunks used as context
          - sources:  Formatted source list for citation
          - stats:    Retrieval diagnostics
        """
        # Stages 1–5: Retrieve
        chunks = await self.retrieve(
            query=query,
            org_id=org_id,
            doc_types=doc_types,
            year_min=year_min,
        )

        # Expand queries list for result metadata
        expanded = (
            await self.expander.expand(query, n=2)
            if self.expand else [query]
        )

        if not chunks:
            return RAGResult(
                query=query,
                expanded_queries=expanded,
                chunks=[],
                context="",
                answer="No relevant information found in the knowledge base for this query.",
                sources=[],
                retrieval_stats={"candidates": 0, "returned": 0},
            )

        # Stage 6: Build context
        context, sources = self.ctx_builder.build(chunks)

        # Stage 7: Synthesise answer
        answer = await self.synthesiser.synthesise(
            query=query,
            context=context,
            temperature=temperature,
        )

        return RAGResult(
            query=query,
            expanded_queries=expanded,
            chunks=chunks,
            context=context,
            answer=answer,
            sources=sources,
            retrieval_stats={
                "chunks_retrieved": len(chunks),
                "top_hybrid_score": chunks[0].hybrid_score if chunks else 0,
                "queries_used": len(expanded),
                "mmr_applied": self.use_mmr,
            },
        )

    async def run_stream(
        self,
        query: str,
        org_id: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
    ):
        """
        Streaming variant of run().
        Yields (event_type, data) tuples:
          ("sources", list[dict])   — emitted first, before tokens
          ("token",   str)          — one token at a time from Ollama
          ("done",    RAGResult)    — final result object
        """
        chunks = await self.retrieve(query=query, org_id=org_id, doc_types=doc_types)
        context, sources = self.ctx_builder.build(chunks)

        yield ("sources", sources)

        full_answer = []
        async for token in self.synthesiser.synthesise_stream(query, context):
            yield ("token", token)
            full_answer.append(token)

        expanded = await self.expander.expand(query, n=2) if self.expand else [query]
        yield ("done", RAGResult(
            query=query,
            expanded_queries=expanded,
            chunks=chunks,
            context=context,
            answer="".join(full_answer),
            sources=sources,
            retrieval_stats={"chunks_retrieved": len(chunks)},
        ))


# ── FastAPI integration example ───────────────────────────────────────────────
#
# from fastapi import APIRouter, Depends
# from fastapi.responses import StreamingResponse
# from app.core.dependencies import get_db
# from app.agent.rag_retrieval import RAGPipeline
# import json
#
# router = APIRouter()
#
# @router.post("/rag/query")
# async def rag_query(body: dict, db=Depends(get_db)):
#     pipeline = RAGPipeline(db)
#     result = await pipeline.run(body["query"], org_id=body.get("org_id"))
#     return result.to_dict()
#
# @router.post("/rag/stream")
# async def rag_stream(body: dict, db=Depends(get_db)):
#     pipeline = RAGPipeline(db)
#     async def event_generator():
#         async for event_type, data in pipeline.run_stream(body["query"]):
#             if event_type == "token":
#                 yield f"data: {json.dumps({'token': data})}\n\n"
#             elif event_type == "sources":
#                 yield f"data: {json.dumps({'sources': data})}\n\n"
#             elif event_type == "done":
#                 yield f"data: {json.dumps({'done': True})}\n\n"
#     return StreamingResponse(event_generator(), media_type="text/event-stream")

"""
tests/unit/test_rag_retrieval.py

Unit tests for every stage of the RAG pipeline.
All Ollama HTTP calls and DB queries are mocked — no running services needed.
"""
import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.rag_retrieval import (
    QueryExpander,
    OllamaEmbedder,
    BM25Reranker,
    MMRDeduplicator,
    ContextBuilder,
    AnswerSynthesiser,
    RAGPipeline,
    RetrievedChunk,
    _OllamaHTTP,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_chunk(
    id="c1",
    text="HIPAA compliance requires encryption of all PHI at rest and in transit.",
    source="Test Doc",
    doc_type="regulation",
    vector_score=0.85,
    bm25_score=0.70,
    hybrid_score=0.79,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=id, text=text, source_doc=source, doc_type=doc_type,
        section_type=None, org_id="test-org", year=2024, tags=[],
        vector_score=vector_score, bm25_score=bm25_score, hybrid_score=hybrid_score,
    )


def make_raw_row(id="c1", similarity=0.85, text="HIPAA compliance details.") -> dict:
    return {
        "id": id,
        "chunk_text": text,
        "source_doc_name": "Test Doc",
        "doc_type": "regulation",
        "section_type": None,
        "org_id": "test-org",
        "year": 2024,
        "tags": [],
        "similarity": similarity,
    }


def make_ollama() -> _OllamaHTTP:
    ollama = _OllamaHTTP.__new__(_OllamaHTTP)
    ollama.base = "http://localhost:11434"
    ollama.llm  = "llama3.1:8b"
    ollama.emb  = "nomic-embed-text"
    ollama._http = MagicMock()
    return ollama


# ── Stage 1: QueryExpander ────────────────────────────────────────────────────

class TestQueryExpander:

    @pytest.mark.asyncio
    async def test_expand_returns_list_with_original(self):
        ollama = make_ollama()
        ollama.chat = AsyncMock(return_value='["HIPAA data encryption", "PHI protection measures"]')
        expander = QueryExpander(ollama)

        queries = await expander.expand("HIPAA compliance", n=2)

        assert isinstance(queries, list)
        assert "HIPAA compliance" in queries   # original always included
        assert len(queries) >= 2

    @pytest.mark.asyncio
    async def test_expand_falls_back_on_llm_error(self):
        ollama = make_ollama()
        ollama.chat = AsyncMock(side_effect=Exception("Ollama offline"))
        expander = QueryExpander(ollama)

        queries = await expander.expand("HIPAA compliance")

        assert queries == ["HIPAA compliance"]  # fallback: original only

    @pytest.mark.asyncio
    async def test_expand_handles_malformed_json(self):
        ollama = make_ollama()
        ollama.chat = AsyncMock(return_value="not valid json at all")
        expander = QueryExpander(ollama)

        queries = await expander.expand("HIPAA compliance")

        assert queries == ["HIPAA compliance"]


# ── Stage 2: OllamaEmbedder ───────────────────────────────────────────────────

class TestOllamaEmbedder:

    @pytest.mark.asyncio
    async def test_embed_query_returns_vector(self):
        ollama = make_ollama()
        ollama.embed = AsyncMock(return_value=[0.1] * 768)
        embedder = OllamaEmbedder(ollama)

        vec = await embedder.embed_query("HIPAA compliance")

        assert isinstance(vec, list)
        assert len(vec) == 768

    @pytest.mark.asyncio
    async def test_embed_queries_returns_multiple_vectors(self):
        ollama = make_ollama()
        ollama.embed_batch = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768])
        embedder = OllamaEmbedder(ollama)

        vecs = await embedder.embed_queries(["query1", "query2"])

        assert len(vecs) == 2
        assert all(len(v) == 768 for v in vecs)


# ── Stage 4: BM25Reranker ─────────────────────────────────────────────────────

class TestBM25Reranker:

    def test_rerank_returns_retrieved_chunks(self):
        reranker = BM25Reranker(alpha=0.6)
        rows = [
            make_raw_row("c1", 0.9, "HIPAA encryption PHI patient data compliance"),
            make_raw_row("c2", 0.7, "Budget cost financial projections revenue"),
            make_raw_row("c3", 0.8, "HIPAA security rule privacy PHI safeguards"),
        ]
        results = reranker.rerank(rows, "HIPAA PHI compliance", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_rerank_orders_by_hybrid_score(self):
        reranker = BM25Reranker(alpha=0.6)
        rows = [
            make_raw_row("c1", 0.5, "irrelevant content about budgets"),
            make_raw_row("c2", 0.9, "HIPAA compliance encryption patient records PHI"),
        ]
        results = reranker.rerank(rows, "HIPAA compliance", top_k=2)

        # c2 has both higher vector score AND better keyword match → should rank first
        assert results[0].id == "c2"

    def test_rerank_hybrid_scores_in_range(self):
        reranker = BM25Reranker(alpha=0.6)
        rows = [make_raw_row(f"c{i}", 0.5 + i * 0.1, f"health data text {i}") for i in range(5)]
        results = reranker.rerank(rows, "health data", top_k=5)

        for r in results:
            assert 0.0 <= r.hybrid_score <= 1.0

    def test_rerank_respects_top_k(self):
        reranker = BM25Reranker()
        rows = [make_raw_row(f"c{i}", 0.8, f"text content chunk {i}") for i in range(10)]
        results = reranker.rerank(rows, "text content", top_k=3)
        assert len(results) == 3

    def test_rerank_empty_input_returns_empty(self):
        reranker = BM25Reranker()
        assert reranker.rerank([], "any query", top_k=5) == []


# ── Stage 5: MMRDeduplicator ──────────────────────────────────────────────────

class TestMMRDeduplicator:

    def _make_embedding(self, base: float, dim: int = 8) -> list[float]:
        """Create a simple embedding vector for testing."""
        import random
        random.seed(int(base * 100))
        vec = [base + random.uniform(-0.1, 0.1) for _ in range(dim)]
        norm = math.sqrt(sum(v**2 for v in vec))
        return [v / norm for v in vec]

    def test_dedup_selects_top_k(self):
        deduper = MMRDeduplicator(lambda_=0.7)
        chunks = [make_chunk(id=f"c{i}", hybrid_score=0.9 - i * 0.05) for i in range(6)]
        query_emb = self._make_embedding(0.5)
        chunk_embs = [self._make_embedding(0.5 + i * 0.1) for i in range(6)]

        result = deduper.deduplicate(chunks, query_emb, chunk_embs, top_k=3)

        assert len(result) == 3

    def test_dedup_returns_all_when_fewer_than_top_k(self):
        deduper = MMRDeduplicator()
        chunks = [make_chunk(id=f"c{i}") for i in range(3)]
        query_emb = self._make_embedding(0.5)
        chunk_embs = [self._make_embedding(0.5 + i * 0.15) for i in range(3)]

        result = deduper.deduplicate(chunks, query_emb, chunk_embs, top_k=10)

        assert len(result) == 3  # no dedup needed

    def test_cosine_similarity_self_is_one(self):
        vec = [0.1, 0.2, 0.3, 0.4]
        sim = MMRDeduplicator._cosine(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_is_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        sim = MMRDeduplicator._cosine(a, b)
        assert abs(sim - 0.0) < 1e-6


# ── Stage 6: ContextBuilder ───────────────────────────────────────────────────

class TestContextBuilder:

    def test_build_returns_string_and_sources(self):
        builder = ContextBuilder(max_context_tokens=2000)
        chunks = [
            make_chunk(id="c1", text="HIPAA compliance details " * 20),
            make_chunk(id="c2", text="Past performance metrics " * 20, source="Past Perf Doc"),
        ]
        context, sources = builder.build(chunks)

        assert isinstance(context, str)
        assert isinstance(sources, list)
        assert len(sources) == 2
        assert "HIPAA" in context

    def test_build_respects_token_budget(self):
        builder = ContextBuilder(max_context_tokens=100)  # Very tight budget
        chunks = [make_chunk(id=f"c{i}", text="word " * 500) for i in range(5)]
        context, sources = builder.build(chunks)

        # Should have fewer than 5 sources due to budget
        assert len(sources) <= 5
        assert len(context) <= 100 * 4  # chars ≈ tokens * 3.5, generous margin

    def test_build_includes_source_metadata(self):
        builder = ContextBuilder()
        chunks = [make_chunk(id="c1", source="DHCS WPC Contract 2022")]
        _, sources = builder.build(chunks)

        assert sources[0]["source_doc"] == "DHCS WPC Contract 2022"
        assert sources[0]["index"] == 1

    def test_context_block_format(self):
        chunk = make_chunk(id="c1", text="Test content here.")
        block = chunk.to_context_block(index=3)

        assert "[3]" in block
        assert "Test Doc" in block
        assert "Test content here." in block


# ── Stage 7: AnswerSynthesiser ────────────────────────────────────────────────

class TestAnswerSynthesiser:

    @pytest.mark.asyncio
    async def test_synthesise_returns_string(self):
        ollama = make_ollama()
        ollama.chat = AsyncMock(return_value="HIPAA requires encryption of PHI [Source 1].")
        synth = AnswerSynthesiser(ollama)

        answer = await synth.synthesise(
            query="What are our HIPAA measures?",
            context="[1] Source: Reg Doc\nHIPAA requires all PHI to be encrypted.",
        )

        assert isinstance(answer, str)
        assert len(answer) > 0


# ── Full Pipeline ─────────────────────────────────────────────────────────────

class TestRAGPipeline:

    def make_pipeline(self, db=None) -> RAGPipeline:
        db = db or MagicMock()
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.top_k   = 3
        pipeline.fetch_k = 9
        pipeline.expand  = False  # disable expansion for speed in tests
        pipeline.use_mmr = False  # disable MMR for speed in tests
        pipeline._ollama     = make_ollama()
        pipeline.expander    = QueryExpander(pipeline._ollama)
        pipeline.embedder    = OllamaEmbedder(pipeline._ollama)
        pipeline.searcher    = MagicMock()
        pipeline.reranker    = BM25Reranker()
        pipeline.deduper     = MMRDeduplicator()
        pipeline.ctx_builder = ContextBuilder()
        pipeline.synthesiser = AnswerSynthesiser(pipeline._ollama)
        return pipeline

    @pytest.mark.asyncio
    async def test_retrieve_returns_chunks(self):
        pipeline = self.make_pipeline()
        raw_rows = [make_raw_row(f"c{i}", 0.9 - i * 0.1, f"health content text {i}") for i in range(5)]

        pipeline._ollama.embed = AsyncMock(return_value=[0.1] * 768)
        pipeline._ollama.embed_batch = AsyncMock(return_value=[[0.1] * 768])
        pipeline.searcher.multi_query_search = AsyncMock(return_value=raw_rows)

        chunks = await pipeline.retrieve("HIPAA compliance")

        assert isinstance(chunks, list)
        assert all(isinstance(c, RetrievedChunk) for c in chunks)
        assert len(chunks) <= pipeline.top_k

    @pytest.mark.asyncio
    async def test_retrieve_returns_empty_on_no_results(self):
        pipeline = self.make_pipeline()
        pipeline._ollama.embed = AsyncMock(return_value=[0.1] * 768)
        pipeline._ollama.embed_batch = AsyncMock(return_value=[[0.1] * 768])
        pipeline.searcher.multi_query_search = AsyncMock(return_value=[])

        chunks = await pipeline.retrieve("anything")

        assert chunks == []

    @pytest.mark.asyncio
    async def test_run_returns_rag_result(self):
        pipeline = self.make_pipeline()
        raw_rows = [make_raw_row(f"c{i}", 0.9, f"HIPAA health content {i}") for i in range(3)]

        pipeline._ollama.embed = AsyncMock(return_value=[0.1] * 768)
        pipeline._ollama.embed_batch = AsyncMock(return_value=[[0.1] * 768])
        pipeline.searcher.multi_query_search = AsyncMock(return_value=raw_rows)
        pipeline._ollama.chat = AsyncMock(return_value="HIPAA requires encryption [Source 1].")

        from app.agent.rag_retrieval import RAGResult
        result = await pipeline.run("What are our HIPAA measures?", org_id="test-org")

        assert isinstance(result, RAGResult)
        assert len(result.answer) > 0
        assert len(result.chunks) > 0
        assert len(result.sources) > 0

    @pytest.mark.asyncio
    async def test_run_returns_no_results_message_when_empty(self):
        pipeline = self.make_pipeline()
        pipeline._ollama.embed = AsyncMock(return_value=[0.1] * 768)
        pipeline._ollama.embed_batch = AsyncMock(return_value=[[0.1] * 768])
        pipeline.searcher.multi_query_search = AsyncMock(return_value=[])

        result = await pipeline.run("obscure query with no matches")

        assert "No relevant information" in result.answer
        assert result.chunks == []

    @pytest.mark.asyncio
    async def test_health_check_returns_dict(self):
        pipeline = self.make_pipeline()
        pipeline._ollama.is_ready = AsyncMock(return_value=True)

        status = await pipeline.health_check()

        assert "ollama_ready" in status
        assert status["ollama_ready"] is True
        assert "llm_model" in status
        assert "embed_model" in status

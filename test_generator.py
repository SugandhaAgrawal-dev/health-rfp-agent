"""
tests/unit/test_generator.py

Unit tests for the RFP Response Generator.
All Ollama calls are mocked — no running Ollama instance required.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.generator import (
    RFPResponseGenerator,
    PromptBuilder,
    SectionStrategy,
    GenerationResult,
    ProposalDraft,
    STRATEGY_ADDENDUMS,
)
from app.agent.parser import RFPSection, RFPStructure


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_section(
    name="Technical Approach",
    requirements="Describe your methodology for managing Medi-Cal patients.",
    page_limit=10,
    order=0,
) -> RFPSection:
    return RFPSection(name=name, requirements=requirements, page_limit=page_limit, order=order)


def make_org_profile() -> dict:
    return {
        "id":                   "test-org-001",
        "name":                 "Valley Health System",
        "service_area":         "Santa Clara County, CA",
        "patient_population":   "Underserved and Medi-Cal beneficiaries",
        "annual_patient_volume": 120000,
        "certifications":       ["HIPAA", "HITRUST", "SOC 2"],
        "mission_statement":    "Providing quality healthcare to all.",
        "primary_contact_name": "Dr. Maria Chen",
    }


def make_rfp(sections=None) -> RFPStructure:
    rfp = RFPStructure(
        title="Medi-Cal Population Health RFP",
        agency="DHCS",
        raw_text="",
    )
    rfp.sections = sections or [
        make_section("Technical Approach"),
        make_section("Past Performance"),
        make_section("Management Plan"),
    ]
    rfp.evaluation_criteria = [
        {"criterion": "Technical Approach", "weight": 40},
        {"criterion": "Past Performance",   "weight": 30},
        {"criterion": "Management Plan",    "weight": 30},
    ]
    return rfp


def make_chunks(n=3) -> list[dict]:
    return [
        {
            "chunk_text":      f"Relevant healthcare content chunk {i}. HIPAA compliance ensured.",
            "source_doc_name": f"Source Document {i}",
            "doc_type":        "past_proposal",
            "hybrid_score":    0.9 - i * 0.05,
            "year":            2023,
        }
        for i in range(n)
    ]


def make_generator(
    primary_model="llama3.1:8b",
    fallback_model=None,
    quality_eval=False,
) -> RFPResponseGenerator:
    """Create a generator with a mocked Ollama client."""
    gen = RFPResponseGenerator.__new__(RFPResponseGenerator)
    gen._primary_model  = primary_model
    gen._fallback_model = fallback_model
    gen._ctx_window     = 8192
    gen._quality_eval   = quality_eval
    gen._prompt_builder = PromptBuilder(context_window=8192)

    mock_ollama = MagicMock()
    mock_ollama.chat        = AsyncMock(return_value="## Technical Approach\n\nOur methodology is evidence-based...")
    mock_ollama.chat_stream = MagicMock()
    mock_ollama.list_models = AsyncMock(return_value=["llama3.1:8b", "nomic-embed-text"])
    gen._ollama = mock_ollama

    return gen


# ── PromptBuilder tests ───────────────────────────────────────────────────────

class TestPromptBuilder:

    def test_detect_strategy_technical(self):
        pb = PromptBuilder()
        assert pb.detect_strategy("Technical Approach") == SectionStrategy.TECHNICAL
        assert pb.detect_strategy("technical plan")     == SectionStrategy.TECHNICAL

    def test_detect_strategy_past_perf(self):
        pb = PromptBuilder()
        assert pb.detect_strategy("Past Performance") == SectionStrategy.PAST_PERF
        assert pb.detect_strategy("Relevant Experience") == SectionStrategy.PAST_PERF

    def test_detect_strategy_data_security(self):
        pb = PromptBuilder()
        assert pb.detect_strategy("Data Security & Privacy") == SectionStrategy.DATA_SECURITY
        assert pb.detect_strategy("HIPAA Compliance Plan")   == SectionStrategy.DATA_SECURITY

    def test_detect_strategy_default_for_unknown(self):
        pb = PromptBuilder()
        assert pb.detect_strategy("Random Unknown Section") == SectionStrategy.DEFAULT

    def test_build_system_prompt_includes_addendum(self):
        pb = PromptBuilder()
        system = pb.build_system_prompt(SectionStrategy.TECHNICAL)
        assert "methodology component" in system
        assert "sub-headings" in system

    def test_build_system_prompt_includes_base(self):
        pb = PromptBuilder()
        system = pb.build_system_prompt(SectionStrategy.DEFAULT)
        assert "EVIDENCE-BASED" in system
        assert "COMPLIANCE-FIRST" in system
        assert "PATIENT-CENTRED" in system

    def test_build_section_prompt_returns_tuple(self):
        pb = PromptBuilder()
        section = make_section()
        org     = make_org_profile()
        chunks  = make_chunks(3)

        prompt, tokens = pb.build_section_prompt(section, org, chunks)

        assert isinstance(prompt, str)
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_build_section_prompt_contains_org_name(self):
        pb = PromptBuilder()
        prompt, _ = pb.build_section_prompt(make_section(), make_org_profile(), [])
        assert "Valley Health System" in prompt

    def test_build_section_prompt_contains_section_name(self):
        pb = PromptBuilder()
        prompt, _ = pb.build_section_prompt(make_section("Data Security"), make_org_profile(), [])
        assert "Data Security" in prompt

    def test_build_section_prompt_contains_requirements(self):
        pb = PromptBuilder()
        section = make_section(requirements="Describe your HIPAA programme in detail.")
        prompt, _ = pb.build_section_prompt(section, make_org_profile(), [])
        assert "HIPAA programme" in prompt

    def test_build_section_prompt_contains_page_limit(self):
        pb = PromptBuilder()
        section = make_section(page_limit=8)
        prompt, _ = pb.build_section_prompt(section, make_org_profile(), [])
        assert "8 pages" in prompt

    def test_build_section_prompt_contains_chunks(self):
        pb = PromptBuilder()
        chunks = make_chunks(2)
        prompt, _ = pb.build_section_prompt(make_section(), make_org_profile(), chunks)
        assert "Source Document 0" in prompt
        assert "Source Document 1" in prompt

    def test_build_section_prompt_contains_evaluation_criteria(self):
        pb = PromptBuilder()
        criteria = [{"criterion": "Technical Approach", "weight": 40}]
        prompt, _ = pb.build_section_prompt(make_section(), make_org_profile(), [], evaluation_criteria=criteria)
        assert "40%" in prompt
        assert "Technical Approach" in prompt

    def test_build_section_prompt_contains_reviewer_feedback(self):
        pb = PromptBuilder()
        prompt, _ = pb.build_section_prompt(
            make_section(), make_org_profile(), [],
            reviewer_feedback="Please add more metrics to this section."
        )
        assert "Please add more metrics" in prompt

    def test_build_section_prompt_contains_previous_draft(self):
        pb = PromptBuilder()
        prompt, _ = pb.build_section_prompt(
            make_section(), make_org_profile(), [],
            reviewer_feedback="Improve it.",
            previous_draft="This is the previous draft content.",
        )
        assert "previous draft" in prompt.lower()

    def test_build_section_prompt_respects_token_budget(self):
        pb = PromptBuilder(context_window=2000)
        large_chunks = [
            {"chunk_text": "x " * 500, "source_doc_name": "Doc", "doc_type": "reg",
             "hybrid_score": 0.9, "year": 2023}
            for _ in range(8)
        ]
        prompt, _ = pb.build_section_prompt(make_section(), make_org_profile(), large_chunks)
        assert len(prompt) <= pb.max_prompt_chars * 1.05  # allow 5% margin

    def test_build_kb_context_empty(self):
        pb = PromptBuilder()
        result = pb._build_kb_context([])
        assert "No prior content retrieved" in result

    def test_build_kb_context_numbered(self):
        pb = PromptBuilder()
        chunks = make_chunks(3)
        result = pb._build_kb_context(chunks)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_build_kb_context_shows_score(self):
        pb = PromptBuilder()
        chunks = [{"chunk_text": "text", "source_doc_name": "Doc", "doc_type": "reg",
                   "hybrid_score": 0.87, "year": 2024}]
        result = pb._build_kb_context(chunks)
        assert "0.87" in result

    def test_build_executive_summary_prompt(self):
        pb  = PromptBuilder()
        rfp = make_rfp()
        org = make_org_profile()
        drafts = {"Technical Approach": "Our approach is evidence-based..."}

        prompt = pb.build_executive_summary_prompt(rfp, org, drafts)

        assert "Executive Summary" in prompt
        assert "Valley Health System" in prompt
        assert "Medi-Cal Population Health RFP" in prompt

    def test_build_quality_eval_prompt(self):
        pb = PromptBuilder()
        prompt = pb.build_quality_eval_prompt("Technical Approach", "content here", "requirements here")
        assert "overall_score" in prompt
        assert "JSON" in prompt


# ── GenerationResult tests ────────────────────────────────────────────────────

class TestGenerationResult:

    def test_word_count(self):
        result = GenerationResult(
            section_name="Test", content="one two three four five",
            model_used="llama3.1:8b", generation_time_s=1.0,
            prompt_tokens_est=100, output_tokens_est=20,
        )
        assert result.word_count == 5

    def test_word_count_empty(self):
        result = GenerationResult(
            section_name="Test", content="",
            model_used="llama3.1:8b", generation_time_s=0,
            prompt_tokens_est=0, output_tokens_est=0,
        )
        assert result.word_count == 0


# ── RFPResponseGenerator tests ────────────────────────────────────────────────

class TestRFPResponseGenerator:

    @pytest.mark.asyncio
    async def test_generate_section_returns_result(self):
        gen = make_generator()
        result = await gen.generate_section(
            section=make_section(),
            org_profile=make_org_profile(),
            retrieved_chunks=make_chunks(3),
        )
        assert isinstance(result, GenerationResult)
        assert len(result.content) > 0
        assert result.model_used == "llama3.1:8b"
        assert result.generation_time_s >= 0

    @pytest.mark.asyncio
    async def test_generate_section_calls_ollama_chat(self):
        gen = make_generator()
        await gen.generate_section(
            section=make_section(),
            org_profile=make_org_profile(),
            retrieved_chunks=[],
        )
        gen._ollama.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_section_passes_correct_model(self):
        gen = make_generator(primary_model="mistral:7b")
        await gen.generate_section(
            section=make_section(),
            org_profile=make_org_profile(),
            retrieved_chunks=[],
        )
        call_kwargs = gen._ollama.chat.call_args.kwargs
        assert call_kwargs.get("model") == "mistral:7b"

    @pytest.mark.asyncio
    async def test_generate_section_strips_preamble(self):
        gen = make_generator()
        gen._ollama.chat = AsyncMock(
            return_value="Here is the Technical Approach section:\n\n## Our Approach\n\nContent here."
        )
        result = await gen.generate_section(
            section=make_section("Technical Approach"),
            org_profile=make_org_profile(),
            retrieved_chunks=[],
        )
        assert not result.content.startswith("Here is")
        assert "## Our Approach" in result.content

    @pytest.mark.asyncio
    async def test_generate_section_uses_strategy_system_prompt(self):
        gen = make_generator()
        await gen.generate_section(
            section=make_section("Past Performance"),
            org_profile=make_org_profile(),
            retrieved_chunks=[],
        )
        call_kwargs = gen._ollama.chat.call_args.kwargs
        system = call_kwargs.get("system", "")
        # Past performance addendum should appear in system prompt
        assert "Client" in system or "Quantified" in system

    @pytest.mark.asyncio
    async def test_generate_section_retries_on_timeout(self):
        import httpx
        gen = make_generator()
        gen._ollama.chat = AsyncMock(
            side_effect=[
                httpx.TimeoutException("timeout"),
                "## Technical Approach\n\nContent after retry.",
            ]
        )
        result = await gen.generate_section(
            section=make_section(),
            org_profile=make_org_profile(),
            retrieved_chunks=[],
        )
        assert result.content == "## Technical Approach\n\nContent after retry."
        assert result.retry_count == 1

    @pytest.mark.asyncio
    async def test_generate_section_falls_back_to_secondary_model(self):
        import httpx
        gen = make_generator(primary_model="llama3.1:8b", fallback_model="phi3:mini")
        gen._ollama.chat = AsyncMock(
            side_effect=[
                httpx.TimeoutException("timeout"),
                httpx.TimeoutException("timeout"),
                httpx.TimeoutException("timeout"),
                "Fallback model content.",  # 4th call uses fallback
            ]
        )
        result = await gen.generate_section(
            section=make_section(),
            org_profile=make_org_profile(),
            retrieved_chunks=[],
        )
        assert result.model_used == "phi3:mini"
        assert result.content == "Fallback model content."

    @pytest.mark.asyncio
    async def test_generate_section_raises_when_all_fail(self):
        import httpx
        gen = make_generator()  # no fallback model
        gen._ollama.chat = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with pytest.raises(RuntimeError, match="Ollama generation failed"):
            await gen.generate_section(
                section=make_section(),
                org_profile=make_org_profile(),
                retrieved_chunks=[],
            )

    @pytest.mark.asyncio
    async def test_regenerate_section_includes_feedback_in_prompt(self):
        gen = make_generator()
        await gen.regenerate_section(
            section=make_section(),
            org_profile=make_org_profile(),
            retrieved_chunks=[],
            reviewer_feedback="Add contract values.",
        )
        call_args = gen._ollama.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        user_content = messages[0]["content"] if isinstance(messages, list) else ""
        assert "Add contract values" in user_content

    @pytest.mark.asyncio
    async def test_refine_section_returns_string(self):
        gen = make_generator()
        gen._ollama.chat = AsyncMock(return_value="Refined content here.")
        result = await gen.refine_section(
            section_name="Technical Approach",
            current_content="Original content.",
            refinement_instruction="Make it more concise.",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_full_proposal_sequential(self):
        gen  = make_generator()
        rfp  = make_rfp()
        org  = make_org_profile()
        chunks = {sec.name: make_chunks(2) for sec in rfp.sections}

        draft = await gen.generate_full_proposal(rfp, org, chunks, parallel=False)

        assert isinstance(draft, ProposalDraft)
        assert len(draft.sections) == len(rfp.sections)
        for name in [s.name for s in rfp.sections]:
            assert name in draft.sections
            assert isinstance(draft.sections[name], GenerationResult)

    @pytest.mark.asyncio
    async def test_generate_full_proposal_handles_section_failure(self):
        gen = make_generator()
        gen._ollama.chat = AsyncMock(
            side_effect=[
                Exception("GPU OOM"),
                "## Past Performance\n\nContent.",
                "## Management Plan\n\nContent.",
            ]
        )
        rfp    = make_rfp()
        org    = make_org_profile()
        chunks = {sec.name: [] for sec in rfp.sections}

        draft = await gen.generate_full_proposal(rfp, org, chunks)

        assert len(draft.sections) == 3
        failed = draft.sections["Technical Approach"]
        assert "failed" in failed.content.lower() or failed.warning is not None

    @pytest.mark.asyncio
    async def test_generate_executive_summary_returns_result(self):
        gen = make_generator()
        gen._ollama.chat = AsyncMock(return_value="## Executive Summary\n\nOur health system...")
        rfp = make_rfp()
        org = make_org_profile()
        drafts = {"Technical Approach": "Our approach...", "Past Performance": "We delivered..."}

        result = await gen.generate_executive_summary(rfp, org, drafts)

        assert isinstance(result, GenerationResult)
        assert result.section_name == "Executive Summary"
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_quality_eval_called_when_enabled(self):
        gen = make_generator(quality_eval=True)
        gen._ollama.chat = AsyncMock(
            side_effect=[
                "## Technical Approach\n\nContent.",               # generation
                '{"overall_score": 8.0, "feedback": "Good coverage."}',  # eval
            ]
        )
        result = await gen.generate_section(
            section=make_section(),
            org_profile=make_org_profile(),
            retrieved_chunks=[],
        )
        assert result.quality_score == 8.0
        assert result.quality_feedback == "Good coverage."

    @pytest.mark.asyncio
    async def test_quality_eval_not_called_when_disabled(self):
        gen = make_generator(quality_eval=False)
        result = await gen.generate_section(
            section=make_section(),
            org_profile=make_org_profile(),
            retrieved_chunks=[],
        )
        # Only 1 call — no eval call
        assert gen._ollama.chat.call_count == 1
        assert result.quality_score is None

    @pytest.mark.asyncio
    async def test_get_model_info_returns_dict(self):
        gen = make_generator()
        info = await gen.get_model_info()
        assert "primary_model" in info
        assert "primary_available" in info
        assert info["primary_model"] == "llama3.1:8b"

    @pytest.mark.asyncio
    async def test_proposal_draft_summary(self):
        gen = make_generator()
        rfp = make_rfp()
        org = make_org_profile()
        draft = await gen.generate_full_proposal(rfp, org, {s.name: [] for s in rfp.sections})

        summary = draft.summary()
        assert "sections" in summary
        assert "total_words" in summary
        assert "total_time_s" in summary
        assert "model" in summary
        assert summary["sections"] == 3

    def test_clean_output_strips_preamble(self):
        gen = make_generator()
        raw = "Here is the Technical Approach section:\n\n## Our Method\n\nContent."
        cleaned = gen._clean_output(raw, "Technical Approach")
        assert not cleaned.startswith("Here is")
        assert "## Our Method" in cleaned

    def test_clean_output_strips_disclaimer(self):
        gen = make_generator()
        raw = "## Content\n\nGood content here.\n\nNote: This is AI-generated."
        cleaned = gen._clean_output(raw, "Section")
        assert "Note:" not in cleaned

    def test_clean_output_normalises_blank_lines(self):
        gen = make_generator()
        raw = "Para one.\n\n\n\n\n\nPara two."
        cleaned = gen._clean_output(raw, "Section")
        assert "\n\n\n\n" not in cleaned

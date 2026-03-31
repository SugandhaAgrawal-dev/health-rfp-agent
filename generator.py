"""
app/agent/generator.py

RFP Response Generator — Powered by Ollama (Local LLM)
=======================================================
Generates full healthcare RFP proposal sections using a locally running
Ollama model. No cloud API keys required. Full data privacy.

Capabilities:
  - Single section generation (blocking)
  - Single section generation (streaming tokens)
  - Full proposal generation (all sections, sequential or parallel)
  - Executive summary generation (synthesised from all sections)
  - Section regeneration with reviewer feedback
  - Multi-turn refinement (targeted rewriting without full regeneration)
  - Model fallback (tries secondary model if primary times out)
  - Token-aware prompt truncation (never overflows context window)
  - Generation quality scoring (self-evaluation by Ollama)
  - Retry with exponential back-off on transient failures

Supported Ollama models (set via OLLAMA_LLM_MODEL in .env):
  - llama3.1:8b        fast, good quality, 8 GB VRAM
  - llama3.1:70b       best quality, 40+ GB VRAM
  - mistral:7b         fast, strong instruction following
  - phi3:mini          very fast, works on limited RAM
  - medllama2:7b       healthcare-fine-tuned (best for clinical content)
  - biomistral:7b      biomedical domain adaptation
  - gemma2:9b          strong reasoning
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

import httpx

from app.agent.parser import RFPSection, RFPStructure
from app.core.config import settings
from app.core.logging import get_logger
from app.services.ollama_client import get_ollama_client

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

WORDS_PER_PAGE   = 275     # standard words-per-page estimate for proposals
CHARS_PER_TOKEN  = 3.5     # approximate chars per LLM token
MAX_CHUNK_TOKENS = 600     # max tokens per retrieved KB chunk in prompt
MAX_CONTEXT_CHUNKS = 6     # max KB chunks injected per section


# ── Section strategies ────────────────────────────────────────────────────────

class SectionStrategy(Enum):
    """Controls tone and structure guidance per section type."""
    TECHNICAL     = "technical"
    MANAGEMENT    = "management"
    PAST_PERF     = "past_performance"
    STAFFING      = "staffing"
    BUDGET        = "budget"
    EXECUTIVE     = "executive"
    DATA_SECURITY = "data_security"
    DEFAULT       = "default"


SECTION_STRATEGY_MAP: dict[str, SectionStrategy] = {
    "technical approach":  SectionStrategy.TECHNICAL,
    "methodology":         SectionStrategy.TECHNICAL,
    "technical plan":      SectionStrategy.TECHNICAL,
    "management plan":     SectionStrategy.MANAGEMENT,
    "project management":  SectionStrategy.MANAGEMENT,
    "past performance":    SectionStrategy.PAST_PERF,
    "relevant experience": SectionStrategy.PAST_PERF,
    "prior experience":    SectionStrategy.PAST_PERF,
    "staffing plan":       SectionStrategy.STAFFING,
    "key personnel":       SectionStrategy.STAFFING,
    "budget":              SectionStrategy.BUDGET,
    "cost proposal":       SectionStrategy.BUDGET,
    "price":               SectionStrategy.BUDGET,
    "executive summary":   SectionStrategy.EXECUTIVE,
    "data security":       SectionStrategy.DATA_SECURITY,
    "hipaa":               SectionStrategy.DATA_SECURITY,
    "privacy":             SectionStrategy.DATA_SECURITY,
}

STRATEGY_ADDENDUMS: dict[SectionStrategy, str] = {
    SectionStrategy.TECHNICAL: (
        "Structure with clear sub-headings for each methodology component. "
        "Include: approach overview, specific activities, tools/technologies, timeline, "
        "and measurable outputs. Use numbered lists for sequential steps."
    ),
    SectionStrategy.MANAGEMENT: (
        "Include organisational structure, roles and responsibilities, governance model, "
        "communication plan, and escalation procedures. Reference specific personnel titles."
    ),
    SectionStrategy.PAST_PERF: (
        "For each example use the format: Client/Agency → Contract Value → Period → Scope → "
        "Your Role → 2-3 Quantified Outcomes (%, $, patient counts). "
        "Only reference real, verifiable contracts."
    ),
    SectionStrategy.STAFFING: (
        "Include a staffing table: Role | FTE | Qualifications | Experience. "
        "Provide a brief biography for each key person. "
        "Address training, retention, and succession planning."
    ),
    SectionStrategy.BUDGET: (
        "Present costs as: Personnel | Fringe | Travel | Equipment | Supplies | Indirect. "
        "Justify every line item. Ensure costs align with the proposed scope."
    ),
    SectionStrategy.EXECUTIVE: (
        "Open with the community need. Summarise the solution in 2 sentences. "
        "Highlight 3 strongest differentiators with evidence. "
        "Close with a commitment statement. Write for a reviewer who reads only this page."
    ),
    SectionStrategy.DATA_SECURITY: (
        "Explicitly address: HIPAA Privacy Rule, HIPAA Security Rule, "
        "technical safeguards (AES-256 at rest, TLS 1.3 in transit), "
        "administrative safeguards (training, BAA), incident response plan, "
        "and any certifications (HITRUST, SOC 2, FedRAMP)."
    ),
    SectionStrategy.DEFAULT: (
        "Use clear headings and sub-headings. Be specific and evidence-based. "
        "Address every sub-requirement from the RFP. Avoid generic statements."
    ),
}


# ── System prompt ─────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are a senior healthcare grant writer with 20+ years of experience
writing winning proposals for Federally Qualified Health Centers (FQHCs), hospital systems,
managed care organisations, and community health networks responding to CMS, HHS/HRSA,
state Medicaid, SAMHSA, and private foundation RFPs.

Your writing principles:
1. EVIDENCE-BASED: Ground every claim in specific data — patient volumes, outcome
   percentages, contract values, years of operation, staff credentials.
2. COMPLIANCE-FIRST: Proactively address HIPAA, 42 CFR Part 2, data security, and any
   regulatory requirements named in the RFP.
3. PATIENT-CENTRED: Connect every programme element to impact on the patients served,
   especially underserved and vulnerable populations.
4. REQUIREMENT-ALIGNED: Map your response to the RFP's evaluation criteria weights.
   Higher-weight criteria get proportionally more depth.
5. HONEST: Never fabricate statistics, client names, contract values, or certifications.
   If you cannot cite a specific number, describe the approach instead.
6. STRUCTURED: Use ## for section headings, ### for sub-headings, bold key metrics,
   numbered lists for sequential steps, bullet lists for features.

Hard rules:
- Never include placeholder text like [INSERT NAME] or [TBD]
- Write in active voice
- Do NOT add a preamble ("Here is the section...") — start the content directly"""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Result of generating a single proposal section."""
    section_name:      str
    content:           str
    model_used:        str
    generation_time_s: float
    prompt_tokens_est: int
    output_tokens_est: int
    quality_score:     Optional[float] = None
    quality_feedback:  Optional[str]   = None
    retry_count:       int             = 0
    warning:           Optional[str]   = None

    @property
    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class ProposalDraft:
    """Complete proposal with all generated sections."""
    rfp_title:  str
    org_name:   str
    sections:   dict[str, GenerationResult] = field(default_factory=dict)
    total_time_s: float = 0.0
    model_used: str = ""

    def to_text_dict(self) -> dict[str, str]:
        """Return {section_name: content} for downstream export."""
        return {name: r.content for name, r in self.sections.items()}

    def summary(self) -> dict:
        quality_scores = [r.quality_score for r in self.sections.values() if r.quality_score]
        return {
            "rfp_title":    self.rfp_title,
            "org_name":     self.org_name,
            "sections":     len(self.sections),
            "total_words":  sum(r.word_count for r in self.sections.values()),
            "total_time_s": round(self.total_time_s, 1),
            "model":        self.model_used,
            "avg_quality":  round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else None,
        }


# ── Prompt builder ────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Builds carefully structured prompts for each section type.

    Key responsibilities:
    - Selects generation strategy based on section name
    - Builds targeted system prompt with strategy addendum
    - Injects retrieved KB chunks as numbered, scored context blocks
    - Enforces token budget so prompt always fits context window
    - Adds evaluation criteria weighting guidance
    - Adds reviewer feedback for regeneration runs
    """

    def __init__(self, context_window: int = 8192):
        # Reserve 45% of the context window for system prompt + output
        self.max_prompt_chars = int((context_window * 0.55) * CHARS_PER_TOKEN)

    def detect_strategy(self, section_name: str) -> SectionStrategy:
        name_lower = section_name.lower()
        for keyword, strategy in SECTION_STRATEGY_MAP.items():
            if keyword in name_lower:
                return strategy
        return SectionStrategy.DEFAULT

    def build_system_prompt(self, strategy: SectionStrategy) -> str:
        addendum = STRATEGY_ADDENDUMS.get(strategy, STRATEGY_ADDENDUMS[SectionStrategy.DEFAULT])
        return f"{BASE_SYSTEM_PROMPT}\n\n## For this section specifically\n{addendum}"

    def build_section_prompt(
        self,
        section: RFPSection,
        org_profile: dict,
        retrieved_chunks: list,
        evaluation_criteria: list[dict] | None = None,
        reviewer_feedback: str | None = None,
        previous_draft: str | None = None,
    ) -> tuple[str, int]:
        """
        Assemble the full user-turn prompt.

        Returns:
            (prompt_text, estimated_token_count)
        """
        # ── Organisation ──────────────────────────────────────────────────────
        certs = ", ".join(org_profile.get("certifications") or ["HIPAA"])
        org_block = (
            "## Organisation profile\n"
            f"- **Name**: {org_profile.get('name', 'Our Organisation')}\n"
            f"- **Service area**: {org_profile.get('service_area', 'Not specified')}\n"
            f"- **Patient population**: {org_profile.get('patient_population', 'Not specified')}\n"
            f"- **Annual volume**: {org_profile.get('annual_patient_volume', 'Not specified')} patients/year\n"
            f"- **Certifications**: {certs}\n"
            f"- **Mission**: {org_profile.get('mission_statement', 'Improving community health.')}\n"
            f"- **Primary contact**: {org_profile.get('primary_contact_name', 'Not specified')}"
        )

        # ── Requirements ──────────────────────────────────────────────────────
        page_note = ""
        if section.page_limit:
            target = section.page_limit * WORDS_PER_PAGE
            page_note = f"\n> **Page limit**: {section.page_limit} pages (~{target} words)"

        req_block = (
            f"## RFP requirements — {section.name}\n"
            f"{section.requirements}"
            f"{page_note}"
        )

        # ── Evaluation criteria ───────────────────────────────────────────────
        criteria_block = ""
        if evaluation_criteria:
            lines = "\n".join(
                f"  - {c['criterion']}: **{c['weight']}%**"
                for c in evaluation_criteria
            )
            criteria_block = f"## Evaluation criteria (weight your depth accordingly)\n{lines}"

        # ── Knowledge base context ────────────────────────────────────────────
        kb_block = self._build_kb_context(retrieved_chunks)

        # ── Reviewer feedback (for regeneration) ──────────────────────────────
        feedback_block = ""
        if reviewer_feedback:
            feedback_block = (
                "## Reviewer feedback — incorporate this in your revision\n"
                f"{reviewer_feedback}"
            )

        prev_block = ""
        if previous_draft:
            preview = previous_draft[:800] + ("..." if len(previous_draft) > 800 else "")
            prev_block = f"## Previous draft (revise based on feedback above)\n{preview}"

        # ── Instruction ───────────────────────────────────────────────────────
        instruction = (
            "---\n\n"
            f'Write the complete **"{section.name}"** section now.\n\n'
            "Requirements:\n"
            "- Use ## and ### markdown headings\n"
            "- Address EVERY sub-requirement listed above\n"
            "- Ground all claims in the organisation profile and KB context\n"
            "- Bold key metrics and certification names\n"
            "- Do NOT include placeholder text or [TBD]\n"
            "- Do NOT add a preamble — begin writing the section content directly"
        )

        # ── Assemble and enforce token budget ─────────────────────────────────
        parts = [org_block, req_block, criteria_block, kb_block,
                 feedback_block, prev_block, instruction]
        prompt = "\n\n".join(p for p in parts if p.strip())

        if len(prompt) > self.max_prompt_chars:
            prompt = self._trim_to_budget(
                org_block, req_block, criteria_block, retrieved_chunks,
                feedback_block, prev_block, instruction,
            )

        return prompt, int(len(prompt) / CHARS_PER_TOKEN)

    def build_executive_summary_prompt(
        self,
        rfp: RFPStructure,
        org_profile: dict,
        section_drafts: dict[str, str],
    ) -> str:
        previews = "\n\n".join(
            f"### {name}\n{content[:500]}..."
            for name, content in list(section_drafts.items())[:5]
        )
        criteria_str = ", ".join(
            f"{c['criterion']} ({c['weight']}%)"
            for c in rfp.evaluation_criteria
        ) or "Not specified"

        return (
            "Write a compelling **Executive Summary** for this healthcare RFP proposal.\n\n"
            "## RFP context\n"
            f"- **Title**: {rfp.title}\n"
            f"- **Agency**: {rfp.agency or 'Not specified'}\n"
            f"- **Deadline**: {rfp.deadline or 'Not specified'}\n"
            f"- **Evaluation criteria**: {criteria_str}\n\n"
            "## Organisation\n"
            f"- **Name**: {org_profile.get('name')}\n"
            f"- **Service area**: {org_profile.get('service_area', 'Not specified')}\n"
            f"- **Mission**: {org_profile.get('mission_statement', '')}\n\n"
            "## Section highlights (for reference)\n"
            f"{previews}\n\n"
            "---\n\n"
            "Write a 350–500 word Executive Summary that:\n"
            "1. Opens with the community health need (1 paragraph)\n"
            "2. Summarises proposed solution and unique approach (1-2 paragraphs)\n"
            "3. Highlights top 3 qualifications with specific evidence\n"
            "4. States measurable outcomes we commit to achieving\n"
            "5. Closes with an organisational commitment statement\n\n"
            "Begin writing the Executive Summary directly — no preamble."
        )

    def build_quality_eval_prompt(
        self, section_name: str, content: str, requirements: str
    ) -> str:
        return (
            f"Rate this RFP proposal section on a scale of 1–10.\n\n"
            f"Section: {section_name}\n\n"
            f"RFP Requirements (first 400 chars):\n{requirements[:400]}\n\n"
            f"Generated Content (first 600 chars):\n{content[:600]}\n\n"
            "Score criteria:\n"
            "1. Requirements coverage — addresses all sub-requirements?\n"
            "2. Specificity — grounded in concrete details?\n"
            "3. Compliance — regulatory requirements addressed?\n"
            "4. Structure — headings and formatting appropriate?\n"
            "5. Persuasiveness — compelling to an evaluator?\n\n"
            'Return ONLY JSON: {"overall_score": 7.5, "feedback": "Brief note."}\n\nJSON:'
        )

    def _build_kb_context(self, chunks: list) -> str:
        """Format retrieved KB chunks as numbered, scored context blocks."""
        if not chunks:
            return (
                "## Knowledge base context\n"
                "_No prior content retrieved. Draw on healthcare industry best practices._"
            )

        blocks = []
        total_chars = 0
        budget = int(MAX_CHUNK_TOKENS * CHARS_PER_TOKEN * MAX_CONTEXT_CHUNKS)

        for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS], start=1):
            # Support both RetrievedChunk objects and raw dicts
            if hasattr(chunk, "text"):
                text     = chunk.text
                source   = getattr(chunk, "source_doc", "KB")
                doc_type = getattr(chunk, "doc_type", "")
                score    = getattr(chunk, "hybrid_score", 0.0)
                year     = getattr(chunk, "year", None)
            else:
                text     = chunk.get("chunk_text", "")
                source   = chunk.get("source_doc_name", "KB")
                doc_type = chunk.get("doc_type", "")
                score    = chunk.get("hybrid_score", 0.0)
                year     = chunk.get("year")

            max_chunk_chars = int(MAX_CHUNK_TOKENS * CHARS_PER_TOKEN)
            if len(text) > max_chunk_chars:
                text = text[:max_chunk_chars] + "... [truncated]"

            year_str = f" ({year})" if year else ""
            header   = f"**[{i}] {source}{year_str}** | {doc_type} | score: {score:.2f}"
            block    = f"{header}\n{text}"

            if total_chars + len(block) > budget:
                blocks.append(f"**[{i}+]** _Remaining chunks omitted — token budget reached._")
                break

            blocks.append(block)
            total_chars += len(block)

        joined = "\n\n---\n\n".join(blocks)
        return (
            "## Knowledge base context\n"
            "_(Reference these examples as evidence. Do not copy verbatim.)_\n\n"
            + joined
        )

    def _trim_to_budget(
        self,
        org_block: str,
        req_block: str,
        criteria_block: str,
        chunks: list,
        feedback_block: str,
        prev_block: str,
        instruction: str,
    ) -> str:
        """Reduce KB chunks one at a time until prompt fits the token budget."""
        for n in range(MAX_CONTEXT_CHUNKS - 1, -1, -1):
            kb = self._build_kb_context(chunks[:n])
            parts = [org_block, req_block, criteria_block, kb,
                     feedback_block, prev_block, instruction]
            prompt = "\n\n".join(p for p in parts if p.strip())
            if len(prompt) <= self.max_prompt_chars:
                logger.debug("prompt_trimmed_to_budget", chunks_kept=n)
                return prompt
        # Last resort: no KB context
        parts = [org_block, req_block, criteria_block, feedback_block, instruction]
        return "\n\n".join(p for p in parts if p.strip())


# ── Main generator ────────────────────────────────────────────────────────────

class RFPResponseGenerator:
    """
    Orchestrates Ollama to generate complete healthcare RFP proposal responses.

    Quick start:
        gen = RFPResponseGenerator()

        # Check Ollama is running
        info = await gen.get_model_info()

        # Generate one section (blocking)
        result = await gen.generate_section(section, org_profile, chunks)
        print(result.content)
        print(f"Generated {result.word_count} words in {result.generation_time_s}s")

        # Generate one section (streaming)
        async for token in gen.generate_section_stream(section, org_profile, chunks):
            print(token, end="", flush=True)

        # Generate full proposal
        draft = await gen.generate_full_proposal(rfp, org_profile, chunks_by_section)
        print(draft.summary())

        # Regenerate with reviewer feedback
        result = await gen.regenerate_section(
            section, org_profile, chunks,
            reviewer_feedback="Strengthen past performance section with contract values.",
            previous_draft=draft.sections["Past Performance"].content,
        )

        # Targeted refinement without full regeneration
        improved = await gen.refine_section(
            section_name="Technical Approach",
            current_content=draft.sections["Technical Approach"].content,
            refinement_instruction="Add a HIPAA compliance subsection after the Technology paragraph.",
        )
    """

    def __init__(
        self,
        primary_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        context_window: Optional[int] = None,
        enable_quality_eval: bool = False,
    ):
        """
        Args:
            primary_model:       Ollama model name. Default: OLLAMA_LLM_MODEL from .env
            fallback_model:      Secondary model used if primary times out (e.g. "phi3:mini")
            context_window:      Max context tokens. Default: OLLAMA_CONTEXT_WINDOW from .env
            enable_quality_eval: Run a second Ollama call to score each section (slower)
        """
        self._ollama          = get_ollama_client()
        self._primary_model   = primary_model  or settings.OLLAMA_LLM_MODEL
        self._fallback_model  = fallback_model
        self._ctx_window      = context_window or settings.OLLAMA_CONTEXT_WINDOW
        self._quality_eval    = enable_quality_eval
        self._prompt_builder  = PromptBuilder(context_window=self._ctx_window)

        logger.info(
            "generator_init",
            model=self._primary_model,
            fallback=self._fallback_model,
            context_window=self._ctx_window,
            quality_eval=self._quality_eval,
        )

    # ── Single section — blocking ─────────────────────────────────────────────

    async def generate_section(
        self,
        section: RFPSection,
        org_profile: dict,
        retrieved_chunks: list,
        evaluation_criteria: list[dict] | None = None,
        temperature: float = 0.35,
        max_tokens: int = 3500,
    ) -> GenerationResult:
        """
        Generate a single RFP proposal section and return a GenerationResult.

        What happens internally:
          1. Detect the section strategy (technical, management, etc.)
          2. Build a targeted system prompt with strategy-specific guidance
          3. Build the user prompt with org profile + KB context + criteria
          4. Call Ollama with up to 3 retries on transient network failures
          5. Fall back to secondary model if primary fails after retries
          6. Post-process output (strip preambles, normalise whitespace)
          7. Optionally self-evaluate quality with a second Ollama call

        Args:
            section:             Parsed RFP section with name + requirements
            org_profile:         Organisation metadata dict
            retrieved_chunks:    Chunks from HealthRFPRetriever (dicts or RetrievedChunk)
            evaluation_criteria: [{criterion, weight}] list for score weighting
            temperature:         0.0–1.0 (lower = more deterministic)
            max_tokens:          Maximum tokens to generate

        Returns:
            GenerationResult with content, timing, token counts, and optional quality score
        """
        strategy = self._prompt_builder.detect_strategy(section.name)
        system   = self._prompt_builder.build_system_prompt(strategy)
        prompt, est_tokens = self._prompt_builder.build_section_prompt(
            section=section,
            org_profile=org_profile,
            retrieved_chunks=retrieved_chunks,
            evaluation_criteria=evaluation_criteria,
        )

        logger.info(
            "generate_section_start",
            section=section.name,
            strategy=strategy.value,
            chunks=len(retrieved_chunks),
            prompt_tokens_est=est_tokens,
            model=self._primary_model,
        )

        start = time.monotonic()
        content, model_used, retries = await self._call_with_fallback(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = round(time.monotonic() - start, 2)

        content = self._clean_output(content, section.name)

        quality_score, quality_feedback = None, None
        if self._quality_eval:
            quality_score, quality_feedback = await self._evaluate_quality(
                section.name, content, section.requirements
            )

        result = GenerationResult(
            section_name      = section.name,
            content           = content,
            model_used        = model_used,
            generation_time_s = elapsed,
            prompt_tokens_est = est_tokens,
            output_tokens_est = int(len(content) / CHARS_PER_TOKEN),
            quality_score     = quality_score,
            quality_feedback  = quality_feedback,
            retry_count       = retries,
        )

        logger.info(
            "generate_section_done",
            section=section.name,
            words=result.word_count,
            time_s=elapsed,
            model=model_used,
            quality=quality_score,
        )
        return result

    # ── Single section — streaming ────────────────────────────────────────────

    async def generate_section_stream(
        self,
        section: RFPSection,
        org_profile: dict,
        retrieved_chunks: list,
        evaluation_criteria: list[dict] | None = None,
        temperature: float = 0.35,
        max_tokens: int = 3500,
    ) -> AsyncIterator[str]:
        """
        Streaming variant — yields text tokens as they arrive from Ollama.

        Use with FastAPI StreamingResponse or Server-Sent Events for
        real-time display in the frontend while the model is still writing.

        Example (FastAPI):
            async def event_gen():
                async for token in gen.generate_section_stream(section, org, chunks):
                    yield f"data: {json.dumps({'token': token})}\\n\\n"
            return StreamingResponse(event_gen(), media_type="text/event-stream")

        Example (WebSocket):
            async for token in gen.generate_section_stream(section, org, chunks):
                await ws.send_text(token)
        """
        strategy = self._prompt_builder.detect_strategy(section.name)
        system   = self._prompt_builder.build_system_prompt(strategy)
        prompt, _ = self._prompt_builder.build_section_prompt(
            section=section,
            org_profile=org_profile,
            retrieved_chunks=retrieved_chunks,
            evaluation_criteria=evaluation_criteria,
        )

        logger.info("generate_section_stream_start", section=section.name, model=self._primary_model)

        async for token in self._ollama.chat_stream(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            model=self._primary_model,
        ):
            yield token

    # ── Regenerate with reviewer feedback ─────────────────────────────────────

    async def regenerate_section(
        self,
        section: RFPSection,
        org_profile: dict,
        retrieved_chunks: list,
        reviewer_feedback: str,
        previous_draft: str | None = None,
        evaluation_criteria: list[dict] | None = None,
        temperature: float = 0.3,
    ) -> GenerationResult:
        """
        Regenerate a section incorporating human reviewer feedback.

        Both the previous draft and reviewer notes are injected into the prompt
        so Ollama produces a targeted revision rather than starting from scratch.

        Args:
            reviewer_feedback: Plain-text notes from human reviewer
                               e.g. "Add contract values to past performance examples.
                                     Make the HIPAA section more specific."
            previous_draft:    Content from the previous generation run (optional)
        """
        strategy = self._prompt_builder.detect_strategy(section.name)
        system   = self._prompt_builder.build_system_prompt(strategy)
        prompt, est_tokens = self._prompt_builder.build_section_prompt(
            section=section,
            org_profile=org_profile,
            retrieved_chunks=retrieved_chunks,
            evaluation_criteria=evaluation_criteria,
            reviewer_feedback=reviewer_feedback,
            previous_draft=previous_draft,
        )

        logger.info(
            "regenerate_section_start",
            section=section.name,
            feedback_chars=len(reviewer_feedback),
        )

        start = time.monotonic()
        content, model_used, retries = await self._call_with_fallback(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=3500,
        )
        elapsed = round(time.monotonic() - start, 2)

        return GenerationResult(
            section_name      = section.name,
            content           = self._clean_output(content, section.name),
            model_used        = model_used,
            generation_time_s = elapsed,
            prompt_tokens_est = est_tokens,
            output_tokens_est = int(len(content) / CHARS_PER_TOKEN),
            retry_count       = retries,
        )

    # ── Targeted refinement ───────────────────────────────────────────────────

    async def refine_section(
        self,
        section_name: str,
        current_content: str,
        refinement_instruction: str,
        temperature: float = 0.25,
    ) -> str:
        """
        Targeted single-aspect improvement without full regeneration.

        Useful for fixes like:
          - "Make the first paragraph more concise"
          - "Add a compliance sub-section addressing SOC 2 Type II"
          - "Rewrite the outcomes paragraph with stronger metrics language"
          - "Convert the bullet points in the staffing section to a table"

        Args:
            section_name:           Name of section (for logging)
            current_content:        The content to refine
            refinement_instruction: Plain-text instruction for what to change

        Returns:
            Improved section content as a string.
        """
        prompt = (
            "Refine the following proposal section based on the instruction below.\n\n"
            f"## Refinement instruction\n{refinement_instruction}\n\n"
            f"## Current content\n{current_content}\n\n"
            "---\n\n"
            "Return the COMPLETE improved section. Preserve all parts not addressed "
            "by the instruction. Do NOT add a preamble — return the content directly."
        )

        logger.info(
            "refine_section_start",
            section=section_name,
            instruction=refinement_instruction[:80],
        )

        content, _, _ = await self._call_with_fallback(
            prompt=prompt,
            system=BASE_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=3500,
        )
        return self._clean_output(content, section_name)

    # ── Full proposal ─────────────────────────────────────────────────────────

    async def generate_full_proposal(
        self,
        rfp: RFPStructure,
        org_profile: dict,
        chunks_by_section: dict[str, list],
        temperature: float = 0.35,
        parallel: bool = False,
    ) -> ProposalDraft:
        """
        Generate all proposal sections and return a ProposalDraft.

        Args:
            rfp:               Parsed RFP structure with sections list
            org_profile:       Organisation metadata dict
            chunks_by_section: {section_name: [chunks]} from retriever
            temperature:       Sampling temperature (lower = more deterministic)
            parallel:          Run sections concurrently (faster but needs more VRAM)
                               Recommended: False on machines with <16 GB VRAM

        Returns:
            ProposalDraft with all sections, timing, and summary statistics
        """
        draft = ProposalDraft(
            rfp_title=rfp.title,
            org_name=org_profile.get("name", "Organisation"),
            model_used=self._primary_model,
        )
        start_total = time.monotonic()

        if parallel:
            logger.info("generate_full_proposal_parallel", sections=len(rfp.sections))
            tasks = [
                self.generate_section(
                    section=sec,
                    org_profile=org_profile,
                    retrieved_chunks=chunks_by_section.get(sec.name, []),
                    evaluation_criteria=rfp.evaluation_criteria,
                    temperature=temperature,
                )
                for sec in rfp.sections
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sec, result in zip(rfp.sections, results):
                if isinstance(result, Exception):
                    logger.error("section_failed", section=sec.name, error=str(result))
                    draft.sections[sec.name] = self._error_result(sec.name, str(result))
                else:
                    draft.sections[sec.name] = result
        else:
            logger.info("generate_full_proposal_sequential", sections=len(rfp.sections))
            for sec in rfp.sections:
                try:
                    draft.sections[sec.name] = await self.generate_section(
                        section=sec,
                        org_profile=org_profile,
                        retrieved_chunks=chunks_by_section.get(sec.name, []),
                        evaluation_criteria=rfp.evaluation_criteria,
                        temperature=temperature,
                    )
                except Exception as exc:
                    logger.error("section_failed", section=sec.name, error=str(exc))
                    draft.sections[sec.name] = self._error_result(sec.name, str(exc))

                # Breathing room between calls on constrained hardware
                await asyncio.sleep(0.3)

        draft.total_time_s = round(time.monotonic() - start_total, 2)
        logger.info("full_proposal_done", **draft.summary())
        return draft

    # ── Executive summary ─────────────────────────────────────────────────────

    async def generate_executive_summary(
        self,
        rfp: RFPStructure,
        org_profile: dict,
        section_drafts: dict[str, str],
        temperature: float = 0.4,
    ) -> GenerationResult:
        """
        Generate the Executive Summary AFTER all other sections are written.

        Generated last so it can accurately reflect what is in the proposal.
        Uses a higher temperature (0.4) for more engaging narrative prose.
        """
        prompt = self._prompt_builder.build_executive_summary_prompt(
            rfp=rfp,
            org_profile=org_profile,
            section_drafts=section_drafts,
        )
        system = self._prompt_builder.build_system_prompt(SectionStrategy.EXECUTIVE)

        logger.info("generate_executive_summary_start")
        start = time.monotonic()
        content, model_used, retries = await self._call_with_fallback(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=1000,
        )
        elapsed = round(time.monotonic() - start, 2)

        return GenerationResult(
            section_name      = "Executive Summary",
            content           = self._clean_output(content, "Executive Summary"),
            model_used        = model_used,
            generation_time_s = elapsed,
            prompt_tokens_est = int(len(prompt) / CHARS_PER_TOKEN),
            output_tokens_est = int(len(content) / CHARS_PER_TOKEN),
            retry_count       = retries,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _call_with_fallback(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, str, int]:
        """
        Call Ollama with retry logic and optional model fallback.

        Retry strategy:
          - Up to 3 attempts with exponential back-off (1s, 2s, 4s)
          - Retries on timeout and connection errors only
          - Falls back to self._fallback_model if all primary retries fail

        Returns:
            (content, model_name_used, retry_count)
        """
        retries = 0
        last_exc: Exception | None = None

        for attempt in range(3):
            try:
                content = await self._ollama.chat(
                    messages=[{"role": "user", "content": prompt}],
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=self._primary_model,
                )
                return content, self._primary_model, retries
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                retries += 1
                last_exc = exc
                wait_s = 2 ** attempt
                logger.warning(
                    "ollama_transient_error",
                    attempt=attempt + 1,
                    wait_s=wait_s,
                    error=str(exc),
                )
                if attempt < 2:
                    await asyncio.sleep(wait_s)
            except Exception as exc:
                last_exc = exc
                logger.error("ollama_non_transient_error", error=str(exc))
                break

        # Try fallback model
        if self._fallback_model:
            logger.warning("switching_to_fallback_model", model=self._fallback_model)
            try:
                content = await self._ollama.chat(
                    messages=[{"role": "user", "content": prompt}],
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=self._fallback_model,
                )
                return content, self._fallback_model, retries
            except Exception as exc:
                logger.error("ollama_fallback_failed", error=str(exc))

        raise RuntimeError(
            f"Ollama generation failed after {retries} retries. "
            f"Last error: {last_exc}. "
            "Ensure Ollama is running (`ollama serve`) and the model is pulled."
        )

    async def _evaluate_quality(
        self,
        section_name: str,
        content: str,
        requirements: str,
    ) -> tuple[Optional[float], Optional[str]]:
        """
        Self-evaluation: ask Ollama to score the generated section 0–10.
        Only used when enable_quality_eval=True.
        """
        try:
            eval_prompt = self._prompt_builder.build_quality_eval_prompt(
                section_name, content, requirements
            )
            raw = await self._ollama.chat(
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
                max_tokens=200,
                model=self._primary_model,
            )
            import json
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data  = json.loads(clean)
            score = round(float(data.get("overall_score", 0)), 1)
            return score, data.get("feedback", "")
        except Exception as exc:
            logger.debug("quality_eval_failed", error=str(exc))
            return None, None

    def _clean_output(self, content: str, section_name: str) -> str:
        """
        Post-process LLM output:
        - Strip preambles like "Here is the Technical Approach section:"
        - Remove trailing Note:/Disclaimer: blocks
        - Normalise excessive blank lines (max 2 consecutive)
        """
        preamble_patterns = [
            rf"^Here is (?:the )?['\"]?{re.escape(section_name)}['\"]? section[:\s]*\n+",
            r"^(?:Sure|Certainly|Of course|Absolutely)[!,.]?\s+(?:here|below)[^:]*:\s*\n+",
            r"^I(?:'ll| will) (?:now )?write[^:]*:\s*\n+",
            r"^(?:Here's|Below is) (?:the )?(?:complete )?(?:section|proposal|response)[:\s]*\n+",
        ]
        for pattern in preamble_patterns:
            content = re.sub(pattern, "", content, flags=re.IGNORECASE | re.MULTILINE)

        content = re.sub(
            r"\n+(?:Note|Disclaimer|Important note|Please note):.*$",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        content = re.sub(r"\n{4,}", "\n\n\n", content)
        return content.strip()

    def _error_result(self, section_name: str, error: str) -> GenerationResult:
        """Return a placeholder GenerationResult when a section fails."""
        return GenerationResult(
            section_name      = section_name,
            content           = f"[Generation failed: {error}. Please write this section manually.]",
            model_used        = self._primary_model,
            generation_time_s = 0.0,
            prompt_tokens_est = 0,
            output_tokens_est = 0,
            warning           = error,
        )

    # ── Model info ────────────────────────────────────────────────────────────

    async def get_model_info(self) -> dict:
        """Return info about the configured Ollama models and their availability."""
        try:
            models = await self._ollama.list_models()
            return {
                "primary_model":      self._primary_model,
                "primary_available":  any(self._primary_model in m for m in models),
                "fallback_model":     self._fallback_model,
                "fallback_available": any(self._fallback_model in m for m in models) if self._fallback_model else None,
                "context_window":     self._ctx_window,
                "quality_eval":       self._quality_eval,
                "all_models":         models,
            }
        except Exception as exc:
            return {"error": str(exc), "primary_model": self._primary_model}

"""
app/api/generator.py

Response Generator API
======================
POST /api/v1/generate/section          — generate one section (blocking JSON)
POST /api/v1/generate/section/stream   — generate one section (SSE streaming)
POST /api/v1/generate/proposal         — generate full proposal (background job)
POST /api/v1/generate/executive-summary — generate executive summary
POST /api/v1/generate/regenerate       — regenerate section with reviewer feedback
POST /api/v1/generate/refine           — targeted refinement (no full regen)
GET  /api/v1/generate/models           — list available Ollama models

Mount in app/main.py:
    from app.api import generator
    app.include_router(generator.router, prefix="/api/v1/generate", tags=["generator"])
"""
import json
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.generator import RFPResponseGenerator
from app.agent.parser import RFPSection
from app.core.config import settings
from app.core.dependencies import get_db
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ── Request / Response schemas ────────────────────────────────────────────────

class SectionRequest(BaseModel):
    section_name:         str   = Field(..., description="RFP section name")
    section_requirements: str   = Field(..., description="Full requirements text for this section")
    page_limit:           Optional[int]         = None
    org_profile:          dict  = Field(..., description="Organisation profile dict")
    retrieved_chunks:     list  = Field(default=[], description="KB chunks from retriever")
    evaluation_criteria:  Optional[list[dict]]  = None
    temperature:          float = Field(default=0.35, ge=0.0, le=1.0)
    max_tokens:           int   = Field(default=3500, ge=100, le=8000)
    primary_model:        Optional[str]         = None
    fallback_model:       Optional[str]         = None
    enable_quality_eval:  bool  = False


class RegenerateRequest(BaseModel):
    section_name:         str
    section_requirements: str
    page_limit:           Optional[int] = None
    org_profile:          dict
    retrieved_chunks:     list          = []
    reviewer_feedback:    str           = Field(..., min_length=5, description="Reviewer notes to incorporate")
    previous_draft:       Optional[str] = None
    evaluation_criteria:  Optional[list[dict]] = None
    temperature:          float         = Field(default=0.30, ge=0.0, le=1.0)
    primary_model:        Optional[str] = None


class RefineRequest(BaseModel):
    section_name:            str
    current_content:         str   = Field(..., min_length=10)
    refinement_instruction:  str   = Field(..., min_length=5)
    temperature:             float = Field(default=0.25, ge=0.0, le=1.0)
    primary_model:           Optional[str] = None


class ExecutiveSummaryRequest(BaseModel):
    rfp_title:           str
    rfp_agency:          Optional[str]       = None
    rfp_deadline:        Optional[str]       = None
    evaluation_criteria: list[dict]          = []
    org_profile:         dict
    section_drafts:      dict[str, str]      = Field(..., description="{section_name: content}")
    temperature:         float               = Field(default=0.40, ge=0.0, le=1.0)
    primary_model:       Optional[str]       = None


class GenerationResponse(BaseModel):
    section_name:      str
    content:           str
    model_used:        str
    generation_time_s: float
    word_count:        int
    prompt_tokens_est: int
    output_tokens_est: int
    quality_score:     Optional[float] = None
    quality_feedback:  Optional[str]   = None
    retry_count:       int
    warning:           Optional[str]   = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_section(req: SectionRequest | RegenerateRequest) -> RFPSection:
    return RFPSection(
        name=req.section_name,
        requirements=req.section_requirements,
        page_limit=getattr(req, "page_limit", None),
        order=0,
    )


def _make_generator(req) -> RFPResponseGenerator:
    return RFPResponseGenerator(
        primary_model=getattr(req, "primary_model", None),
        fallback_model=getattr(req, "fallback_model", None),
        enable_quality_eval=getattr(req, "enable_quality_eval", False),
    )


def _result_to_response(result) -> GenerationResponse:
    return GenerationResponse(
        section_name      = result.section_name,
        content           = result.content,
        model_used        = result.model_used,
        generation_time_s = result.generation_time_s,
        word_count        = result.word_count,
        prompt_tokens_est = result.prompt_tokens_est,
        output_tokens_est = result.output_tokens_est,
        quality_score     = result.quality_score,
        quality_feedback  = result.quality_feedback,
        retry_count       = result.retry_count,
        warning           = result.warning,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/section", response_model=GenerationResponse)
async def generate_section(body: SectionRequest):
    """
    Generate a single RFP proposal section using Ollama.

    The pipeline:
      1. Detects section strategy (technical/management/past-perf/etc.)
      2. Builds strategy-specific system prompt
      3. Assembles user prompt with org profile + KB context + criteria
      4. Calls Ollama with retry + fallback model support
      5. Post-processes output (strips preambles, normalises whitespace)
      6. Optionally self-evaluates quality (set enable_quality_eval=true)

    Returns the full section content plus timing and token stats.
    """
    gen     = _make_generator(body)
    section = _make_section(body)

    try:
        result = await gen.generate_section(
            section             = section,
            org_profile         = body.org_profile,
            retrieved_chunks    = body.retrieved_chunks,
            evaluation_criteria = body.evaluation_criteria,
            temperature         = body.temperature,
            max_tokens          = body.max_tokens,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("generate_section_api_error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    return _result_to_response(result)


@router.post("/section/stream")
async def generate_section_stream(body: SectionRequest):
    """
    Streaming section generation using Server-Sent Events (SSE).

    Events:
      data: {"token": "word "}         — one token at a time
      data: {"done": true, "stats": {}} — generation complete

    JavaScript client example:
      const res = await fetch('/api/v1/generate/section/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(request),
      });
      const reader = res.body.getReader();
      // Read chunks and parse SSE events
    """
    gen     = _make_generator(body)
    section = _make_section(body)

    async def event_gen():
        full_content = []
        try:
            async for token in gen.generate_section_stream(
                section             = section,
                org_profile         = body.org_profile,
                retrieved_chunks    = body.retrieved_chunks,
                evaluation_criteria = body.evaluation_criteria,
                temperature         = body.temperature,
                max_tokens          = body.max_tokens,
            ):
                full_content.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

            total_words = len("".join(full_content).split())
            yield f"data: {json.dumps({'done': True, 'stats': {'word_count': total_words}})}\n\n"

        except Exception as exc:
            logger.error("stream_generation_error", error=str(exc))
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/regenerate", response_model=GenerationResponse)
async def regenerate_section(body: RegenerateRequest):
    """
    Regenerate a section incorporating human reviewer feedback.

    Both the previous draft and reviewer notes are injected into the prompt
    so Ollama produces a targeted revision rather than starting from scratch.

    Use this when a reviewer has commented on a generated section and
    wants specific improvements without losing the overall structure.
    """
    gen     = _make_generator(body)
    section = _make_section(body)

    try:
        result = await gen.regenerate_section(
            section             = section,
            org_profile         = body.org_profile,
            retrieved_chunks    = body.retrieved_chunks,
            reviewer_feedback   = body.reviewer_feedback,
            previous_draft      = body.previous_draft,
            evaluation_criteria = body.evaluation_criteria,
            temperature         = body.temperature,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {exc}")

    return _result_to_response(result)


@router.post("/refine")
async def refine_section(body: RefineRequest):
    """
    Targeted single-aspect improvement of a section.

    Faster than full regeneration — only modifies the aspect described in
    refinement_instruction. The rest of the section is preserved.

    Examples of refinement_instruction:
      - "Make the first paragraph more concise (under 100 words)"
      - "Add a HIPAA compliance subsection after the Technology paragraph"
      - "Convert the bullet list in the staffing section to a table"
      - "Rewrite the outcomes paragraph with stronger metrics language"
    """
    gen = _make_generator(body)

    try:
        improved = await gen.refine_section(
            section_name            = body.section_name,
            current_content         = body.current_content,
            refinement_instruction  = body.refinement_instruction,
            temperature             = body.temperature,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Refinement failed: {exc}")

    return {
        "section_name": body.section_name,
        "content":      improved,
        "word_count":   len(improved.split()),
    }


@router.post("/executive-summary", response_model=GenerationResponse)
async def generate_executive_summary(body: ExecutiveSummaryRequest):
    """
    Generate an Executive Summary from completed proposal sections.

    Call this AFTER all other sections are generated so the summary
    accurately reflects the full proposal content.
    """
    from app.agent.parser import RFPStructure, RFPSection

    # Build a minimal RFPStructure from the request fields
    rfp = RFPStructure(
        title=body.rfp_title,
        agency=body.rfp_agency,
        deadline=body.rfp_deadline,
        raw_text="",
    )
    rfp.evaluation_criteria = body.evaluation_criteria

    gen = RFPResponseGenerator(
        primary_model=body.primary_model,
        enable_quality_eval=False,
    )

    try:
        result = await gen.generate_executive_summary(
            rfp           = rfp,
            org_profile   = body.org_profile,
            section_drafts= body.section_drafts,
            temperature   = body.temperature,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Executive summary generation failed: {exc}")

    return _result_to_response(result)


@router.get("/models")
async def list_generator_models():
    """
    List available Ollama models and check which ones are pulled.
    Shows which model is currently configured as primary.
    """
    gen  = RFPResponseGenerator()
    info = await gen.get_model_info()
    if "error" in info:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {info['error']}")
    return info


@router.get("/strategies")
async def list_section_strategies():
    """
    Return the section-name-to-strategy mapping so the frontend
    can show users which generation mode will be used for each section.
    """
    from app.agent.generator import SECTION_STRATEGY_MAP, STRATEGY_ADDENDUMS
    return {
        "strategies": {
            keyword: strategy.value
            for keyword, strategy in SECTION_STRATEGY_MAP.items()
        },
        "addendums": {
            strategy.value: addendum
            for strategy, addendum in STRATEGY_ADDENDUMS.items()
        },
    }

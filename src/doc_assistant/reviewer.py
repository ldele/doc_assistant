"""Reviewer agent — Phase 6 / Integrity Chunk 2b.

A second LLM call after the generator: rates faithfulness of the
generated answer against the retrieved chunks, plus citation density,
hedging adequacy, and a count of claims that aren't traceable to any
retrieved source.

Locked design choices
---------------------

* **Reference-free.** The reviewer doesn't see a ground-truth
  answer (unlike ``eval/scorers.py``'s LLMJudgeScorer). It judges
  whether the answer is *supported by what was retrieved* — the
  question the user actually cares about in production.
* **No auto-retry.** If the reviewer flags issues, surface them; the
  user decides whether to regenerate. Cost discipline.
* **DI on the client.** No SDK imports at module level; the caller
  passes an ``LLMClient`` (``llm.get_reviewer_client()``). The client
  owns the provider + model, so a fully-local reviewer is a config flip.
  Mirrors the eval LLMJudgeScorer pattern.
* **One sidecar row per review.** Re-reviewing an answer (different
  model, different prompt, later in time) inserts another row rather
  than overwriting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from sqlalchemy import select

from doc_assistant.db.models import AnswerReview
from doc_assistant.db.session import session_scope
from doc_assistant.llm import LLMClient, Message
from doc_assistant.provenance import AnswerProvenance, RetrievedChunk

log = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """The reviewer's verdict on one answer."""

    faithfulness: int | None = None
    citation_density: int | None = None
    hedging_adequacy: int | None = None
    unsupported_claims_count: int | None = None
    notes: str | None = None
    error: str | None = None
    raw_response: str | None = None  # for debugging when parsing fails


_REVIEWER_PROMPT = """You are reviewing a retrieval-augmented answer for a research \
assistant. Your job: rate the ANSWER against the EVIDENCE the system retrieved.

CRITICAL RULES — read these before scoring:

1. The EVIDENCE is the only source of truth for what's supported. If the
   answer says something not in the evidence, that is unsupported even if
   you "know" it's true.
2. Do NOT use your own prior knowledge of the subject.
3. Score independently per dimension. A high score on one does not imply
   high scores on others.

QUESTION:
{question}

EVIDENCE (retrieved chunks):
{evidence}

ANSWER:
{answer}

Rate the answer on a 1-5 integer scale across four dimensions:

* faithfulness: 5 = every substantive claim is directly supported by the
  evidence. 3 = roughly half. 1 = the answer is largely unsupported.
* citation_density: 5 = the answer is densely tied to specific evidence
  (frequent references to source content). 3 = some claims tied. 1 = the
  answer reads as ungrounded prose.
* hedging_adequacy: 5 = uncertainty is acknowledged where the evidence is
  weak or contradictory. 3 = partial hedging. 1 = the answer is over-
  confident given what's in the evidence.

Also count:

* unsupported_claims_count: integer ≥ 0. How many distinct claims in the
  answer cannot be traced to any retrieved chunk.

Add a short ``notes`` field (1-2 sentences max) explaining the lowest score.

Return JSON only, no prose, no markdown fence:
{{"faithfulness": <int>, "citation_density": <int>, "hedging_adequacy": <int>, \
"unsupported_claims_count": <int>, "notes": "<short string>"}}"""


def _format_evidence(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as labelled evidence blocks for the prompt."""
    if not chunks:
        return "(no chunks retrieved)"
    parts: list[str] = []
    for i, c in enumerate(chunks):
        header_bits = [f"[{i + 1}]"]
        if c.filename:
            header_bits.append(c.filename)
        if c.page is not None:
            header_bits.append(f"p.{c.page}")
        if c.section:
            header_bits.append(f'"{c.section}"')
        header = " ".join(header_bits)
        excerpt = (c.chunk_excerpt or "").strip()
        parts.append(f"{header}\n{excerpt}")
    return "\n\n---\n\n".join(parts)


def build_reviewer_prompt(prov: AnswerProvenance) -> str:
    """Render the reviewer prompt for one answer.

    The reviewer sees exactly the question, the retrieved evidence, and
    the answer under review — never a ground-truth reference and never the
    analysis conversation. Extracted so the isolation guard test can assert
    the prompt surface without an LLM call.
    """
    return _REVIEWER_PROMPT.format(
        question=prov.query,
        evidence=_format_evidence(prov.retrieved_chunks),
        answer=prov.answer,
    )


def _strip_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def review_answer(
    prov: AnswerProvenance,
    client: LLMClient,
    *,
    max_tokens: int = 400,
) -> ReviewResult:
    """Run the reviewer on one AnswerProvenance. Returns parsed scores or an error.

    ``client`` is an ``LLMClient`` (``llm.get_reviewer_client()``); injected
    so this module has zero vendor SDK imports at module load and the
    provider/model are chosen by config. The model is owned by the client.
    """
    prompt = build_reviewer_prompt(prov)
    messages: list[Message] = [{"role": "user", "content": prompt}]

    try:
        # Single-turn, no system prompt, no history, temperature=0 —
        # same isolation contract as the eval LLM judge.
        text = client.complete(messages, temperature=0.0, max_tokens=max_tokens).strip()
        if text.startswith("```"):
            text = _strip_fence(text)
        parsed = json.loads(text)
    except Exception as e:
        return ReviewResult(error=f"reviewer call failed: {type(e).__name__}: {e}")

    try:
        return ReviewResult(
            faithfulness=int(parsed["faithfulness"]),
            citation_density=int(parsed["citation_density"]),
            hedging_adequacy=int(parsed["hedging_adequacy"]),
            unsupported_claims_count=int(parsed["unsupported_claims_count"]),
            notes=str(parsed.get("notes") or "").strip() or None,
            raw_response=text,
        )
    except (KeyError, TypeError, ValueError) as e:
        return ReviewResult(
            error=f"bad reviewer response: {type(e).__name__}: {e}",
            raw_response=text,
            notes=str(parsed) if isinstance(parsed, dict) else None,
        )


# ============================================================
# Persistence
# ============================================================


def persist_review(
    answer_record_id: str,
    result: ReviewResult,
    *,
    reviewer_kind: str,
    model_name: str | None = None,
) -> str:
    """Write one review row. Returns the new review id."""
    with session_scope() as session:
        row = AnswerReview(
            answer_record_id=answer_record_id,
            reviewer_kind=reviewer_kind,
            model_name=model_name,
            faithfulness=result.faithfulness,
            citation_density=result.citation_density,
            hedging_adequacy=result.hedging_adequacy,
            unsupported_claims_count=result.unsupported_claims_count,
            notes=result.notes,
            error=result.error,
        )
        session.add(row)
        session.flush()
        return str(row.id)


def get_reviews(answer_record_id: str) -> list[ReviewResult]:
    """All reviews for one answer, most-recent first.

    Returns the parsed ``ReviewResult`` shape (without the raw_response,
    which isn't persisted).
    """
    with session_scope() as session:
        rows = (
            session.execute(
                select(AnswerReview)
                .where(AnswerReview.answer_record_id == answer_record_id)
                .order_by(AnswerReview.created_at.desc())
            )
            .scalars()
            .all()
        )
        return [
            ReviewResult(
                faithfulness=r.faithfulness,
                citation_density=r.citation_density,
                hedging_adequacy=r.hedging_adequacy,
                unsupported_claims_count=r.unsupported_claims_count,
                notes=r.notes,
                error=r.error,
            )
            for r in rows
        ]

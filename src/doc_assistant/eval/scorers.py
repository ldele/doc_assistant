"""Scorer implementations for the eval harness (generic).

A scorer is anything that satisfies the ``Scorer`` protocol — given an
``EvalCase`` and the system's ``EvalOutput``, return a single
``ScoreResult`` with a numeric ``value`` in [0.0, 1.0] (deterministic /
embedding-similarity) or in [1.0, 5.0] (LLM rubric).

Locked design choices:

* Scorers receive the LLM client / embedder via constructor injection.
  This module imports zero vendor SDKs at module level; the deps are
  hoisted into the scorer's ``__init__``.
* All scorers tolerate ``output is None`` and return ``value=0.0``
  with an ``error`` field in ``details`` — the runner uses this to
  flag systemic failures without crashing the loop.
* Deterministic scorers are case-insensitive by default. The
  embedding and LLM scorers are case-blind by construction.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from doc_assistant.eval.cases import EvalCase
from doc_assistant.eval.results import EvalOutput, ScoreResult


class Scorer(Protocol):
    """A callable that grades one (case, output) pair.

    Implementations should return ``value=0.0`` (with diagnostic
    ``details``) rather than raising on missing-expected-field — the
    runner uses scorer output to decide whether a case is "scoreable",
    and an exception would terminate the whole run.
    """

    name: str

    def __call__(self, case: EvalCase, output: EvalOutput | None) -> ScoreResult: ...


# ============================================================
# Deterministic scorers
# ============================================================


class ExactMatchScorer:
    """1.0 if ``output.answer`` equals ``case.expected_answer``, else 0.0."""

    name = "exact_match"

    def __init__(self, *, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def __call__(self, case: EvalCase, output: EvalOutput | None) -> ScoreResult:
        if output is None:
            return ScoreResult(self.name, 0.0, {"error": "no output"})
        if case.expected_answer is None:
            return ScoreResult(self.name, 0.0, {"error": "case has no expected_answer"})
        a, b = output.answer, case.expected_answer
        if not self.case_sensitive:
            a, b = a.lower(), b.lower()
        match = a.strip() == b.strip()
        return ScoreResult(self.name, 1.0 if match else 0.0, {})


class ContainsAllScorer:
    """Fraction of ``case.expected_substrings`` present in ``output.answer``."""

    name = "contains_all"

    def __init__(self, *, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def __call__(self, case: EvalCase, output: EvalOutput | None) -> ScoreResult:
        if output is None:
            return ScoreResult(self.name, 0.0, {"error": "no output"})
        if not case.expected_substrings:
            return ScoreResult(self.name, 0.0, {"error": "case has no expected_substrings"})
        haystack = output.answer if self.case_sensitive else output.answer.lower()
        needles = (
            case.expected_substrings
            if self.case_sensitive
            else [s.lower() for s in case.expected_substrings]
        )
        hits = [n for n in needles if n in haystack]
        score = len(hits) / len(needles)
        return ScoreResult(
            self.name,
            score,
            {"matched": hits, "missing": [n for n in needles if n not in haystack]},
        )


class CitationOverlapScorer:
    """Fraction of ``case.expected_citations`` present in ``output.citations``.

    Matches by case-insensitive substring against each output citation —
    so ``"example_paper_2020.pdf"`` matches ``"example_paper_2020"``.
    """

    name = "citation_overlap"

    def __call__(self, case: EvalCase, output: EvalOutput | None) -> ScoreResult:
        if output is None:
            return ScoreResult(self.name, 0.0, {"error": "no output"})
        if not case.expected_citations:
            return ScoreResult(self.name, 0.0, {"error": "case has no expected_citations"})

        actual_lower = [c.lower() for c in output.citations]
        hits = []
        for expected in case.expected_citations:
            needle = expected.lower()
            if any(needle in a or a in needle for a in actual_lower):
                hits.append(expected)
        score = len(hits) / len(case.expected_citations)
        return ScoreResult(
            self.name,
            score,
            {
                "matched": hits,
                "missing": [c for c in case.expected_citations if c not in hits],
                "extra": [
                    c
                    for c in output.citations
                    if not any(
                        e.lower() in c.lower() or c.lower() in e.lower()
                        for e in case.expected_citations
                    )
                ],
            },
        )


class FigureRetrievalScorer:
    """1.0 if a figure chunk for ``case.metadata['expected_figure']`` was retrieved.

    The Feature 4c eval hook: given a query about a figure, did retrieval surface
    the right figure chunk? Reads the retrieved-chunk descriptors the adapter
    places in ``output.raw['retrieved']`` (each a dict with ``chunk_type`` /
    ``filename`` / ``page`` / ``figure_id``) and the expected figure spec in
    ``case.metadata['expected_figure']`` (``{filename, page?, figure_id?}``).

    Match precedence: an exact ``figure_id`` if the case provides one, else
    ``filename`` (case-insensitive substring) plus ``page`` when ``page`` is given.
    Stays generic — pure dict inspection, no ``doc_assistant`` import — so the
    harness remains extractable (Feature 5).
    """

    name = "figure_retrieval"

    def __call__(self, case: EvalCase, output: EvalOutput | None) -> ScoreResult:
        if output is None:
            return ScoreResult(self.name, 0.0, {"error": "no output"})
        expected = case.metadata.get("expected_figure")
        if not isinstance(expected, dict) or not (
            expected.get("filename") or expected.get("figure_id")
        ):
            return ScoreResult(self.name, 0.0, {"error": "case has no expected_figure"})

        retrieved = output.raw.get("retrieved") or []
        figure_chunks = [
            c for c in retrieved if isinstance(c, dict) and c.get("chunk_type") == "figure"
        ]
        exp_id = expected.get("figure_id")
        exp_file = str(expected.get("filename") or "").lower()
        exp_page = expected.get("page")

        for chunk in figure_chunks:
            if exp_id is not None:
                if str(chunk.get("figure_id")) == str(exp_id):
                    return ScoreResult(self.name, 1.0, {"matched": chunk})
                continue
            filename = str(chunk.get("filename") or "").lower()
            if exp_file and exp_file not in filename and filename not in exp_file:
                continue
            if exp_page is not None and chunk.get("page") != exp_page:
                continue
            return ScoreResult(self.name, 1.0, {"matched": chunk})

        return ScoreResult(
            self.name,
            0.0,
            {"expected": expected, "n_figure_chunks": len(figure_chunks)},
        )


# ============================================================
# Embedding similarity scorer
# ============================================================


class EmbeddingSimilarityScorer:
    """Cosine similarity between ``output.answer`` and ``case.expected_answer``.

    ``embedder`` must be a callable that takes a string and returns a
    list-of-floats embedding. Wire a langchain ``Embeddings`` instance
    via ``lambda s: embeddings.embed_query(s)`` at the adapter layer.
    """

    name = "embedding_similarity"

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder

    def __call__(self, case: EvalCase, output: EvalOutput | None) -> ScoreResult:
        if output is None:
            return ScoreResult(self.name, 0.0, {"error": "no output"})
        if case.expected_answer is None:
            return ScoreResult(self.name, 0.0, {"error": "case has no expected_answer"})
        try:
            a = self.embedder(output.answer)
            b = self.embedder(case.expected_answer)
        except Exception as e:  # pragma: no cover - embedder failures are caller's domain
            return ScoreResult(self.name, 0.0, {"error": f"embedder failed: {e}"})
        score = _cosine(a, b)
        return ScoreResult(self.name, max(0.0, score), {})


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity. Stdlib-only to keep the scorer dep-light."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ============================================================
# LLM-as-judge scorer
# ============================================================


_JUDGE_PROMPT = """You are an impartial grader. Your job: score a CANDIDATE answer \
against a REFERENCE answer.

CRITICAL RULES — read these before scoring:

1. The REFERENCE is the only source of truth for this task. Treat it as if
   it is the complete and correct answer to the question.
2. **Do NOT use your own prior knowledge of the subject.** If the candidate
   says something that is true in general but is not in the reference,
   that is NOT supported. Score it as if you knew nothing about the topic
   beyond what the reference says.
3. Do not be charitable about phrasing or "intent". Score what is on the
   page, not what you think the candidate meant.
4. Each rubric dimension is independent — a high score on one does not
   imply a high score on another.

QUESTION:
{question}

REFERENCE ANSWER (the only ground truth available to you):
{reference}

CANDIDATE ANSWER (the answer to grade):
{candidate}

Score the candidate on a 1-5 integer scale across these dimensions:

* faithfulness: 5 = every substantive claim in the candidate is directly
  supported by the reference. 3 = roughly half of claims supported. 1 =
  candidate's main claims are not in the reference (even if true in
  general).
* relevance: 5 = directly answers the question. 3 = partially on-topic. 1 =
  off-topic or evades the question.
* completeness: 5 = covers every key point in the reference. 3 = covers
  about half. 1 = misses the main content of the reference.

Return JSON only, no prose, no markdown fence:
{{"faithfulness": <int>, "relevance": <int>, "completeness": <int>}}"""


class LLMJudgeScorer:
    """LLM-as-judge rubric scorer (faithfulness/relevance/completeness, 1-5).

    Takes an ``LLMClient`` (something with ``.complete(messages, *,
    temperature, max_tokens) -> str``) via constructor injection. Typed
    ``Any`` deliberately: this module imports nothing from ``doc_assistant``
    so the harness stays extractable (Feature 5). The model is owned by the
    injected client. The scorer's ``value`` is the mean of the three
    sub-scores in [1.0, 5.0]; individual dimension scores land in
    ``details``.

    Isolation guarantees
    --------------------
    Each call is fully independent — there is no system prompt, no
    conversation history, no prior cases visible to the model. The only
    text the model sees is the single rendered ``_JUDGE_PROMPT`` for the
    current case. Stateful concerns are limited to:

    * The model's own pretrained knowledge. The prompt instructs the
      model to ignore this and grade only against the reference;
      enforcement is best-effort because there's no way to truly
      blindfold an LLM.
    * The client's own retry/cache behaviour. Each ``complete`` is a fresh
      request — no cross-call context.
    """

    name = "llm_judge"

    def __init__(
        self,
        client: Any,
        max_tokens: int = 200,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens

    def __call__(self, case: EvalCase, output: EvalOutput | None) -> ScoreResult:
        if output is None:
            return ScoreResult(self.name, 0.0, {"error": "no output"})
        if case.expected_answer is None:
            return ScoreResult(self.name, 0.0, {"error": "case has no expected_answer"})

        prompt = _JUDGE_PROMPT.format(
            question=case.query,
            reference=case.expected_answer,
            candidate=output.answer,
        )
        try:
            # Single-turn, no system prompt, no conversation history — each
            # call is fully isolated from every other. temperature=0 so the
            # same (case, candidate) pair yields the same score on re-runs.
            text = self.client.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.max_tokens,
            ).strip()
            if text.startswith("```"):
                text = _strip_fence(text)
            parsed = json.loads(text)
        except Exception as e:
            return ScoreResult(self.name, 0.0, {"error": f"judge call failed: {e}"})

        try:
            f = int(parsed["faithfulness"])
            r = int(parsed["relevance"])
            c = int(parsed["completeness"])
        except (KeyError, TypeError, ValueError) as e:
            return ScoreResult(
                self.name,
                0.0,
                {"error": f"bad judge response: {e}", "raw": parsed},
            )

        mean = (f + r + c) / 3.0
        return ScoreResult(
            self.name,
            mean,
            {"faithfulness": f, "relevance": r, "completeness": c},
        )


def _strip_fence(text: str) -> str:
    """Drop ```json ... ``` fences sometimes emitted by chat models."""
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)

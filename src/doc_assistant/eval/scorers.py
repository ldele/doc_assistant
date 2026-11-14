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


# ============================================================
# Chat scorers (spec eval-harness-chat.md) — deterministic, model-neutral
#
# These read what the adapter recorded on ``output.raw`` and never re-derive the app's
# contract here: the adapter is the one module allowed to import the app, so it computes the
# app's own citation audit and route and hands them over (RG-012's lesson: a harness that
# restates the contract tests its own copy of it). The keys:
#
# * ``retrieved`` — the chunks the answer was written from, in rank order (dicts with
#   ``filename``);
# * ``retrieved_pre_rerank`` — the candidates' filenames in the order the retrievers fused
#   them, before the cross-encoder (only an adapter that can see that stage records it);
# * ``citation_audit`` — ``synthesis.audit_citations`` as a dict;
# * ``cited_filenames`` — the filenames of the sources the answer validly cited;
# * ``route`` — which path answered the turn (``rag``, ``library``, ``command``, …).
#
# A binary scorer returns 1.0 when the case **holds** and 0.0 when it does not, so a gate's
# "no case that held before fails after" reads the same on every one of them.
# ============================================================

#: Scorers whose value is a hold (1.0) or a break (0.0). A report counts these and lists the
#: cases that changed state instead of drawing an interval over them.
BINARY_SCORERS = frozenset({"cited_answer", "refusal_hold", "hostile_hold", "route"})


def _matches(expected: str, actual: str) -> bool:
    """The harness's citation match: case-insensitive substring either way (as
    ``CitationOverlapScorer``), so ``rag_lewis_2020`` matches ``rag_lewis_2020.pdf``."""
    e, a = expected.lower(), actual.lower()
    return e in a or a in e


def _doc_order(filenames: list[str]) -> list[str]:
    """Filenames in first-seen order: a ranking of documents, not of chunks."""
    seen: dict[str, None] = {}
    for f in filenames:
        if f and f not in seen:
            seen[f] = None
    return list(seen)


def _relevant_ranks(case: EvalCase, docs: list[str]) -> list[int]:
    """1-based ranks of the documents that match an expected citation."""
    return [
        i + 1 for i, d in enumerate(docs) if any(_matches(e, d) for e in case.expected_citations)
    ]


class RankScorer:
    """A rank metric over ``expected_citations`` and the retrieved order (MRR, nDCG@k, recall@k).

    ``citation_overlap`` says whether the right document reached the context; this says how
    high. Relevance is binary and per document: a ranked list of chunks is collapsed to its
    documents in first-seen order. ``stage="pre_rerank"`` reads the fused candidate order, so
    the same metric before and after shows what the reranker did.
    """

    METRICS = ("mrr", "ndcg", "recall")

    def __init__(self, metric: str, *, k: int = 10, stage: str = "final") -> None:
        if metric not in self.METRICS:
            raise ValueError(f"unknown rank metric {metric!r}; one of {self.METRICS}")
        if stage not in ("final", "pre_rerank"):
            raise ValueError(f"unknown stage {stage!r}")
        self.metric, self.k, self.stage = metric, k, stage
        base = {"mrr": "rank_mrr", "ndcg": f"rank_ndcg@{k}", "recall": f"recall@{k}"}[metric]
        self.name = base if stage == "final" else f"{base}_pre_rerank"

    def __call__(self, case: EvalCase, output: EvalOutput | None) -> ScoreResult:
        if output is None:
            return ScoreResult(self.name, 0.0, {"error": "no output"})
        if not case.expected_citations:
            return ScoreResult(self.name, 0.0, {"error": "case has no expected_citations"})
        if self.stage == "final":
            raw = output.raw.get("retrieved")
            if raw is None:
                return ScoreResult(self.name, 0.0, {"error": "adapter recorded no retrieval"})
            names = [str(c.get("filename") or "") for c in raw]
        else:
            pre = output.raw.get("retrieved_pre_rerank")
            if pre is None:
                return ScoreResult(
                    self.name, 0.0, {"error": "adapter recorded no pre-rerank order"}
                )
            names = [str(n) for n in pre]
        docs = _doc_order(names)
        ranks = _relevant_ranks(case, docs)
        if self.metric == "mrr":
            value = 1.0 / ranks[0] if ranks else 0.0
        elif self.metric == "recall":
            found = {
                e for e in case.expected_citations if any(_matches(e, d) for d in docs[: self.k])
            }
            value = len(found) / len(case.expected_citations)
        else:
            from math import log2

            dcg = sum(1.0 / log2(r + 1) for r in ranks if r <= self.k)
            ideal_n = min(len(case.expected_citations), self.k)
            idcg = sum(1.0 / log2(i + 2) for i in range(ideal_n))
            value = dcg / idcg if idcg else 0.0
        return ScoreResult(self.name, value, {"relevant_ranks": ranks, "n_docs": len(docs)})


def _audit(output: EvalOutput) -> dict[str, Any] | None:
    audit = output.raw.get("citation_audit")
    return audit if isinstance(audit, dict) else None


class CitationCoverageScorer:
    """Share of the answer's sentences that carry a valid citation (the app's own audit).

    An answer of zero sentences is not scored: there is nothing to cover, and calling it 0 or 1
    would both be invented."""

    name = "citation_coverage"

    def __call__(self, case: EvalCase, output: EvalOutput | None) -> ScoreResult:
        if output is None:
            return ScoreResult(self.name, 0.0, {"error": "no output"})
        audit = _audit(output)
        if audit is None:
            return ScoreResult(self.name, 0.0, {"error": "adapter recorded no citation audit"})
        n = int(audit.get("n_sentences") or 0)
        if n == 0:
            return ScoreResult(self.name, 0.0, {"error": "answer has no sentences"})
        uncited = int(audit.get("n_uncited_sentences") or 0)
        return ScoreResult(self.name, (n - uncited) / n, {"n_sentences": n, "uncited": uncited})


class CitedAnswerScorer:
    """Binary: the answer carries at least one valid citation (holds) or none (breaks).

    Not scored for a case that expects a refusal, where citing nothing is the right answer."""

    name = "cited_answer"

    def __call__(self, case: EvalCase, output: EvalOutput | None) -> ScoreResult:
        if output is None:
            return ScoreResult(self.name, 0.0, {"error": "no output"})
        if case.expect_refusal:
            return ScoreResult(self.name, 0.0, {"error": "case expects a refusal"})
        if case.expected_route not in (None, "rag"):
            return ScoreResult(self.name, 0.0, {"error": "case expects a non-retrieval route"})
        audit = _audit(output)
        if audit is None:
            return ScoreResult(self.name, 0.0, {"error": "adapter recorded no citation audit"})
        valid = list(audit.get("valid") or [])
        return ScoreResult(self.name, 1.0 if valid else 0.0, {"valid": valid})


def _absent(output: EvalOutput, needles: list[str]) -> list[str]:
    """The needles that DO appear in the answer (case-insensitive) — empty means all absent."""
    hay = output.answer.lower()
    return [n for n in needles if n.lower() in hay]


class RefusalHoldScorer:
    """Binary, for a case the library cannot answer (``expect_refusal``).

    Holds when the answer cites nothing and contains none of the case's ``must_not_contain``
    strings (the facts an invented answer would carry). The app has no fixed not-found
    sentence — its prompt asks the model to say so in its own words — so a phrase match would
    test one model's wording; citing nothing is the part of a refusal every model can be held
    to, and the case's strings catch an uncited invention."""

    name = "refusal_hold"

    def __call__(self, case: EvalCase, output: EvalOutput | None) -> ScoreResult:
        if output is None:
            return ScoreResult(self.name, 0.0, {"error": "no output"})
        if not case.expect_refusal:
            return ScoreResult(self.name, 0.0, {"error": "case expects no refusal"})
        audit = _audit(output)
        if audit is None:
            return ScoreResult(self.name, 0.0, {"error": "adapter recorded no citation audit"})
        valid = list(audit.get("valid") or [])
        present = _absent(output, case.must_not_contain)
        holds = not valid and not present
        return ScoreResult(self.name, 1.0 if holds else 0.0, {"valid": valid, "present": present})


class HostileHoldScorer:
    """Binary, for a hostile-passage case: none of ``must_not_contain`` appears in the answer
    and none of ``must_not_cite`` is among the sources it validly cited."""

    name = "hostile_hold"

    def __call__(self, case: EvalCase, output: EvalOutput | None) -> ScoreResult:
        if output is None:
            return ScoreResult(self.name, 0.0, {"error": "no output"})
        if not case.must_not_contain and not case.must_not_cite:
            return ScoreResult(self.name, 0.0, {"error": "case has no hostile checks"})
        if case.expect_refusal:
            return ScoreResult(self.name, 0.0, {"error": "scored by refusal_hold"})
        cited = output.raw.get("cited_filenames")
        if case.must_not_cite and cited is None:
            return ScoreResult(self.name, 0.0, {"error": "adapter recorded no cited sources"})
        present = _absent(output, case.must_not_contain)
        bad_cites = [
            c for c in (cited or []) if any(_matches(m, str(c)) for m in case.must_not_cite)
        ]
        holds = not present and not bad_cites
        return ScoreResult(
            self.name, 1.0 if holds else 0.0, {"present": present, "cited": bad_cites}
        )


class RouteScorer:
    """Binary: the turn took the route the case expects (``expected_route``)."""

    name = "route"

    def __call__(self, case: EvalCase, output: EvalOutput | None) -> ScoreResult:
        if output is None:
            return ScoreResult(self.name, 0.0, {"error": "no output"})
        if case.expected_route is None:
            return ScoreResult(self.name, 0.0, {"error": "case expects no route"})
        route = output.raw.get("route")
        if route is None:
            return ScoreResult(self.name, 0.0, {"error": "adapter recorded no route"})
        holds = str(route) == case.expected_route
        return ScoreResult(
            self.name, 1.0 if holds else 0.0, {"route": route, "expected": case.expected_route}
        )


def chat_scorers() -> list[Scorer]:
    """The deterministic chat scorer mix every run carries (no model, no cost). A scorer that
    does not apply to a case records it as not scored, so one mix serves every family."""
    return [
        RankScorer("mrr"),
        RankScorer("ndcg"),
        RankScorer("recall"),
        RankScorer("mrr", stage="pre_rerank"),
        RankScorer("ndcg", stage="pre_rerank"),
        RankScorer("recall", stage="pre_rerank"),
        CitationCoverageScorer(),
        CitedAnswerScorer(),
        RefusalHoldScorer(),
        HostileHoldScorer(),
        RouteScorer(),
    ]


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

"""Eval harness v0 (Phase 5, Feature 2).

Generic-by-design module for measuring an LLM/RAG system against a
versioned eval set, scoring outputs, and tracking results over time.

Layering rule (locked, see ``decisions.md`` Feature 5):

* Everything in this package **except** ``adapters.py`` imports nothing
  from ``doc_assistant.*``. This lets the harness be extracted into a
  standalone repo after Feature 3 produces its first measurement.
* ``adapters.py`` is the only file that knows about doc_assistant —
  it wraps the RAG pipeline into the generic ``SystemUnderTest``
  callable signature.

Public API surface is intentionally small; consumers import the few
top-level names they need.
"""

from doc_assistant.eval.cases import EvalCase, load_cases_yaml
from doc_assistant.eval.results import EvalOutput, EvalResult, ScoreResult
from doc_assistant.eval.runner import Runner
from doc_assistant.eval.scorers import (
    CitationOverlapScorer,
    ContainsAllScorer,
    EmbeddingSimilarityScorer,
    ExactMatchScorer,
    LLMJudgeScorer,
    Scorer,
)
from doc_assistant.eval.store import Store

__all__ = [
    "CitationOverlapScorer",
    "ContainsAllScorer",
    "EmbeddingSimilarityScorer",
    "EvalCase",
    "EvalOutput",
    "EvalResult",
    "ExactMatchScorer",
    "LLMJudgeScorer",
    "Runner",
    "ScoreResult",
    "Scorer",
    "Store",
    "load_cases_yaml",
]

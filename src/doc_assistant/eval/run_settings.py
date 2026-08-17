"""The settings that define what an eval run measures.

A run's scores are only interpretable against the configuration that produced them, and this
module is the list of what that configuration *is*. :func:`run_defining_settings` snapshots the
live values; pass it as ``Store(..., settings_provider=run_defining_settings)`` and every run
persisted through that store records them in ``config_json``.

**It is injected, not imported by the store, and that is deliberate.** This module reaches into
:mod:`doc_assistant.config`, and ``doc_assistant.eval`` must import no app wiring — the harness
is designed to be lifted into a standalone repo (ADR-003 Decision 8, pinned by
``tests/unit/test_eval_harness_isolation.py``). So the coupling lives here, at the project's
edge of the harness, and a lifted copy simply drops this file. The cost is that a new runner
must remember the argument; ``scripts/CLAUDE.md`` carries that rule.

**Why this exists (KI-41 / RG-026).** The 2026-06-06 chunking sweep drove its grid through
``PARENT_CHUNK_SIZE`` / ``CHILD_CHUNK_SIZE`` environment variables, which ``.env`` silently
overwrote (KI-38) — so all six configs ingested the same corpus and the sweep compared one
configuration with itself. ``config_json`` recorded ``embedding_model`` / ``n_cases`` /
``scorers`` and no chunk sizes, so nothing in the run record could contradict the note claiming
what had been swept. It took two months and an unrelated investigation to notice. **An
experiment that does not record the setting it varies cannot be audited**, and a driver whose
variable is silently ignored fails in the "no effect" direction — which reads as a confirmed
default.

**The same hole, found again 2026-08-15 — one layer up.** ``config_json`` recorded the embedder
but never the *generator*, so no run in the store said which LLM wrote its answers. A 5-trial
control run on the private 35 scored ``contains_all`` **0.822** where the 2026-08-08 control
recorded **0.777** — read as an improvement, it is nothing of the kind: that run generated with
``llama3.1:8b`` and this one inherited ``.env``'s Anthropic Haiku. The tell that it was the
generator and not the pipeline is that ``citation_overlap`` reproduced to the digit (0.9363 vs
0.936), because retrieval is deterministic and generator-independent. The Haiku-vs-llama split
across the 2026-08-08 arms existed only in ``evals/README.md`` prose — recoverable by a human
reading a document, not by anything auditing the data. Hence ``llm_provider`` / ``llm_model``
below.

**Membership rule:** a value belongs here if changing it changes what the run measures. That is
narrower than "every knob" — cost/latency settings that cannot move a score (worker counts,
cache toggles, the lazy reranker) are deliberately absent, because a record nobody trusts to be
minimal is a record nobody reads.

**Two things that belong to a run record and deliberately are not in this snapshot.**

*Index composition* (RG-021 — how many documents the index held, and which) is recorded by
``scripts/run_eval.py`` from ``RAGPipeline.indexed_doc_hashes``, not read from config, because
config does not know it: the corpus is state on disk, and the only honest source is the index
the run actually retrieved over. Asking the live pipeline also means no second copy of the
"which collection is active" rule to drift out of step with the one in ``pipeline.py``. The
cost is the usual one — a new runner must record it, exactly as it must pass this provider —
and ``scripts/CLAUDE.md`` carries both rules together.

*Synthesis mode* was expected here (the 2026-08-15 baton asks for it, on the reasoning that
``ai`` versus ``human`` changes the answer and therefore ``contains_all``). It is **not** here,
because it does not change what *this harness* measures: ``eval/adapters.py`` drives
``pipeline.retrieve`` and ``pipeline.stream_answer`` directly, and ``stream_answer`` is a bare
``ANSWER_PROMPT | llm`` chain. ``SYNTHESIS_MODE`` is read only in ``chat_controller.helpers``,
which the eval path never enters — so recording it would pin a setting the run did not honour,
which is the RG-029 defect with the sign flipped. The wider consequence is worth stating where
someone will read it: **the eval harness measures the raw answer path, not the shipped one** —
no synthesis split, no provenance, no reviewer. If that changes, this key becomes mandatory.
"""

from __future__ import annotations

from typing import Any

from doc_assistant import config


def run_defining_settings() -> dict[str, Any]:
    """Snapshot the live config values that determine what an eval run measures.

    Read from :mod:`doc_assistant.config` **at call time**, not at import — the values are
    resolved from the environment when config is first imported, and a caller that overrides one
    for a single run (``run_eval --bm25-weight``) passes its own value to ``persist_run``, where
    an explicit key wins over this snapshot. So the recorded value is always the one the run
    actually used, whichever way it was set.
    """
    return {
        # Chunking — the ingest geometry the vector store was built with. Baseline sizes are
        # included because they define the run whenever `use_parent_child` is false, and the
        # flag alone does not tell a reader which pair was live.
        "parent_chunk_size": config.PARENT_CHUNK_SIZE,
        "parent_chunk_overlap": config.PARENT_CHUNK_OVERLAP,
        "child_chunk_size": config.CHILD_CHUNK_SIZE,
        "child_chunk_overlap": config.CHILD_CHUNK_OVERLAP,
        "baseline_chunk_size": config.BASELINE_CHUNK_SIZE,
        "baseline_chunk_overlap": config.BASELINE_CHUNK_OVERLAP,
        # Generator — which LLM wrote the answers. `contains_all` and any judge score are
        # measurements OF this model; two runs with different generators are different
        # experiments, however identical the retrieval settings look.
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": config.LLM_MODEL,
        # Retrieval — the locked settings table, minus the ones already named above.
        "embedding_model": config.EMBEDDING_MODEL,
        "use_parent_child": config.USE_PARENT_CHILD,
        "use_multi_query": config.USE_MULTI_QUERY,
        "top_k": config.TOP_K,
        "candidate_k": config.CANDIDATE_K,
        "bm25_weight": config.BM25_WEIGHT,
        "rerank_candidate_cap": config.RERANK_CANDIDATE_CAP,
    }

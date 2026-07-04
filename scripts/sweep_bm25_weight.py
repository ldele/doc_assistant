"""BM25/vector ensemble-weight sweep — retrieval-only, $0, deterministic.

Sweeps the locked ``BM25_WEIGHT`` (the sparse arm's ensemble weight; vector arm =
``1 - w``) and measures retrieval **recall@K** against each case's
``expected_citations`` (bidirectional substring, the same instrument as the R6
baseline). It is the eval gate for the vibes-locked 0.4/0.6 split: change the
default only if a weight beats the control beyond variance (rigor-gate).

Why this is a standalone, retrieval-only harness (not ``scripts.run_eval`` in a
loop like ``sweep_chunking``):

* The weight is a **retrieval-time** knob — no re-embed, no store mutation. The
  pipeline (embedder + vector store + cross-encoder + BM25 index) is loaded
  **once**; only the ``EnsembleRetriever`` is rebuilt per weight. So the whole
  sweep is one model load + N cheap retrieval passes.
* Retrieval with ``USE_MULTI_QUERY=false`` makes **zero** LLM calls → **$0** and
  fully offline (no HTTPS, so KI-6's SSL crash can't bite). ``run_eval``'s free
  scorers still generate an answer per case (an LLM call), which the weight does
  not affect — pure overhead here.
* Retrieval is **deterministic**, so one pass per arm is representative; the
  ``--repeat`` intent (variance) is satisfied by determinism, not repetition
  (``--repeat`` still runs N passes and asserts they agree).

Two metrics per weight, to make the result *explanatory*:

* **post-rerank** recall@K — the shipped metric (``pipeline.retrieve``): what the
  user actually gets after the cross-encoder.
* **pre-rerank** recall@K — recall over the ensemble's fused candidate order,
  *before* the cross-encoder. This is where the weight actually acts.

  If pre-rerank moves across weights while post-rerank stays flat, that is the
  discrimination proof: the instrument detects the ranking change, and the flat
  post-rerank is a real structural finding (the cross-encoder re-scores the full
  candidate **union**, which is weight-independent — see
  ``pipeline`` / LangChain ``EnsembleRetriever.weighted_reciprocal_rank``), not a
  broken measurement.

Usage::

    HF_HUB_OFFLINE=1 uv run --no-sync python -m scripts.sweep_bm25_weight
    uv run --no-sync python -m scripts.sweep_bm25_weight --grid 0.0,0.3,0.4,0.5,1.0
    uv run --no-sync python -m scripts.sweep_bm25_weight --repeat 3 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

# Force offline model loading BEFORE any HuggingFace import: warm cache only, no
# network HEAD (avoids the frozen/proxy cert path entirely). This sweep never needs
# the network. Set here rather than relying on the caller so `python -m` is enough.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from statistics import mean

from doc_assistant.config import BM25_WEIGHT, CANDIDATE_K, TOP_K, USE_MULTI_QUERY
from doc_assistant.eval import load_cases_yaml
from doc_assistant.eval.cases import EvalCase

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

DEFAULT_CASES = "tests/eval/cases.yaml"
DEFAULT_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 0.4 = the locked control
RECALL_KS = (5, 10)


def _parse_grid(raw: str | None) -> list[float]:
    if not raw:
        return list(DEFAULT_GRID)
    weights = [float(tok) for tok in raw.split(",") if tok.strip()]
    for w in weights:
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"grid weight {w} outside [0.0, 1.0]")
    return weights


def _filenames(docs: list) -> list[str]:
    """Ordered, de-duplicated non-empty filenames from a doc list.

    Empty filenames are dropped: the bidirectional-substring matcher treats ``""``
    as a substring of every expected fragment, which would fabricate hits.
    """
    out: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        fn = doc.metadata.get("filename") or ""
        if fn and fn not in seen:
            seen.add(fn)
            out.append(fn)
    return out


def _recall_at_k(expected: list[str], filenames: list[str], k: int) -> float | None:
    """Recall of ``expected`` citations within the first ``k`` filenames.

    Bidirectional substring (``example_paper_2020`` matches
    ``example_paper_2020.pdf`` and vice versa) — the ``cases.yaml`` contract.
    ``None`` when the case declares no ``expected_citations`` (unscored).
    """
    if not expected:
        return None
    top = filenames[:k]
    hits = sum(1 for exp in expected if any(exp in fn or fn in exp for fn in top))
    return hits / len(expected)


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return mean(clean) if clean else None


def measure_weight(
    pipeline: object,
    cases: list[EvalCase],
    weight: float,
) -> dict[str, float | None]:
    """Retrieve every case at ``weight`` and return mean pre/post recall@K.

    Rebuilds only the ensemble (the two arm-retrievers are reused), so this is a
    cheap retrieval pass — no model reload, no re-embed.
    """
    from langchain_classic.retrievers import EnsembleRetriever

    from doc_assistant.pipeline import resolve_ensemble_weights

    retrievers = pipeline.ensemble.retrievers  # type: ignore[attr-defined]
    pipeline.ensemble = EnsembleRetriever(  # type: ignore[attr-defined]
        retrievers=retrievers, weights=resolve_ensemble_weights(weight)
    )

    pre: dict[int, list[float | None]] = {k: [] for k in RECALL_KS}
    post: dict[int, list[float | None]] = {k: [] for k in RECALL_KS}
    for case in cases:
        exp = case.expected_citations
        # Pre-rerank: the ensemble's fused candidate order, before the cross-encoder.
        # The fused list is not top_k-cut, so slice the first k distinct filenames.
        pre_files = _filenames(pipeline.ensemble.invoke(case.query))  # type: ignore[attr-defined]
        for k in RECALL_KS:
            pre[k].append(_recall_at_k(exp, pre_files, k))
            # Post-rerank: the SHIPPED path. Retrieve separately per k — retrieve(top_k=k)
            # reranks the full union then parent-dedups to k parents, which is NOT the
            # same as retrieving 10 and slicing to 5 (a file whose parent ranks 7th would
            # leak into a sliced "@5"). Per-k keeps it comparable to R6 and to run_eval --k.
            post_files = _filenames(pipeline.retrieve(case.query, top_k=k))  # type: ignore[attr-defined]
            post[k].append(_recall_at_k(exp, post_files, k))

    out: dict[str, float | None] = {}
    for k in RECALL_KS:
        out[f"pre@{k}"] = _mean(pre[k])
        out[f"post@{k}"] = _mean(post[k])
    return out


def _fmt(value: float | None) -> str:
    return "  n/a " if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=str, default=DEFAULT_CASES, help="Cases YAML")
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Comma-separated BM25-arm weights in [0,1] (default: 0.0,0.2,0.4,0.6,0.8,1.0)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Passes per weight. Retrieval is deterministic, so >1 asserts agreement. Default 1.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan, load nothing")
    args = parser.parse_args()
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 1

    grid = _parse_grid(args.grid)
    control = BM25_WEIGHT

    print("BM25/vector ensemble-weight sweep (retrieval-only, $0, deterministic)")
    print(f"  cases:       {args.cases}")
    print(f"  grid:        {grid}")
    print(f"  control:     BM25_WEIGHT={control} (vector {1.0 - control})")
    print(f"  CANDIDATE_K: {CANDIDATE_K}   TOP_K: {TOP_K}   USE_MULTI_QUERY: {USE_MULTI_QUERY}")
    print(f"  repeat:      {args.repeat}")

    if USE_MULTI_QUERY:
        print(
            "\n! USE_MULTI_QUERY=true makes retrieve() call the LLM (not $0, not deterministic).\n"
            "  Re-run with USE_MULTI_QUERY=false for a clean retrieval-only sweep.",
            file=sys.stderr,
        )

    cases = load_cases_yaml(args.cases)
    scored = [c for c in cases if c.expected_citations]
    print(f"\nLoaded {len(cases)} cases ({len(scored)} with expected_citations → scored).")

    if args.dry_run:
        print("\n[dry-run] Would load the pipeline once, then sweep the grid above.")
        return 0

    print("Loading RAG pipeline (embedder + vector store + reranker + BM25 index)...")
    from doc_assistant.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    if len(pipeline.ensemble.retrievers) < 2:  # type: ignore[attr-defined]
        print(
            "\n! Ensemble has a single arm (empty corpus → vector-only fallback); "
            "there is no BM25 weight to sweep. Ingest documents first.",
            file=sys.stderr,
        )
        return 1

    header = "weight |  " + "  ".join(f"pre@{k}  post@{k}" for k in RECALL_KS)
    rows: list[tuple[float, dict[str, float | None]]] = []
    disagreements = 0
    for w in grid:
        trials = [measure_weight(pipeline, cases, w) for _ in range(args.repeat)]
        first = trials[0]
        if any(t != first for t in trials[1:]):
            disagreements += 1
            print(f"! weight {w}: passes disagreed (retrieval expected deterministic)", flush=True)
        rows.append((w, first))
        tag = "  <- control" if abs(w - control) < 1e-9 else ""
        print(f"[done] w={w:<4}{tag}", flush=True)

    # -------- report --------
    print("\n" + "=" * len(header))
    print("RESULTS  (mean recall over scored cases; pre = fused candidates, post = shipped)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for w, m in rows:
        cells = "  ".join(f"{_fmt(m[f'pre@{k}'])}  {_fmt(m[f'post@{k}'])}" for k in RECALL_KS)
        tag = "  <- control" if abs(w - control) < 1e-9 else ""
        print(f" {w:<5} | {cells}{tag}")

    # -------- verdict (gate on the SHIPPED metric: post-rerank) --------
    control_row = next((m for w, m in rows if abs(w - control) < 1e-9), None)
    print("\n" + "-" * len(header))
    if control_row is None:
        print(f"Control weight {control} not in the grid — add it to gate against it.")
        return 0

    primary = "post@10"
    ctrl = control_row[primary]
    best_w, best = max(rows, key=lambda r: r[1][primary] if r[1][primary] is not None else -1.0)
    print(f"Gate metric: {primary} (the shipped top-K).  Control ({control}) = {_fmt(ctrl)}")
    post_flat = ctrl is not None and all(
        m[primary] is not None and abs(m[primary] - ctrl) < 1e-9 for _, m in rows
    )

    # Discrimination proof: did ANY pre-rerank metric move across the grid? (pre@10 alone
    # can be flat while pre@5 varies — the weight reorders inside the candidate pool.)
    def _varies(key: str) -> bool:
        vals = {round(m[key], 6) for _, m in rows if m[key] is not None}
        return len(vals) > 1

    pre_moves = any(_varies(f"pre@{k}") for k in RECALL_KS)
    if post_flat:
        print(
            f"VERDICT: post-rerank recall is FLAT across the whole grid (= {_fmt(ctrl)}). "
            "No weight beats the control on the shipped metric."
        )
        if pre_moves:
            print(
                "  Pre-rerank recall DOES vary across weights (instrument discriminates), "
                "so the flat post-rerank is the reranker-dominance structural result, not a\n"
                "  measurement failure: the cross-encoder re-scores the full candidate union, "
                "which is weight-independent. KEEP 0.4/0.6 (negative result)."
            )
        else:
            print(
                "  Pre-rerank recall is ALSO flat — this corpus/benchmark does not distinguish "
                "the arms at these K. KEEP 0.4/0.6 (negative result)."
            )
    elif best[primary] is not None and ctrl is not None and best[primary] > ctrl:
        print(
            f"VERDICT: weight {best_w} scores {_fmt(best[primary])} > control {_fmt(ctrl)} on "
            f"{primary}. Candidate win — re-run with --repeat, confirm the margin exceeds "
            "variance, and only then change the BM25_WEIGHT default + record a baseline."
        )
    else:
        print(
            f"VERDICT: no weight beats the control on {primary}. KEEP 0.4/0.6 (negative result)."
        )

    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())

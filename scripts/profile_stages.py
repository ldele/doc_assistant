"""Profile the pipeline stage by stage: what is affordable at runtime, what is batch-only.

The question this answers is **not** "is the app fast" — it is *where the time lives*, so a design
decision (do we re-embed? can this run per keystroke? must this be a background job?) can be made
against numbers instead of intuition. Three groups, because they have three different budgets:

* **startup** — paid on *every* launch of the backend, before the first question can be answered.
* **query** — paid per turn, inside the user's attention span.
* **ingest** — paid per document, once, and *re-paid in full* whenever the embedded text changes.

Sidecar/enrichment runners are deliberately **not** in here: they are separate processes with their
own dry-run modes, and timing them means timing the runner (see ``--sidecars`` for a scoped-vs-full
comparison that shells out).

Everything is local and **$0** — no LLM provider is constructed, no generation happens. The
embedding measurement calls the embedder directly on real chunk text and **writes nothing**, so it
is safe to run against the live corpus.

Usage
-----
  uv run --no-sync python -m scripts.profile_stages                 # startup + query, 3 reps
  uv run --no-sync python -m scripts.profile_stages -r 5            # more repeats
  uv run --no-sync python -m scripts.profile_stages --ingest        # + per-document ingest costs
  uv run --no-sync python -m scripts.profile_stages --sidecars      # + scoped-vs-full sidecar runs
  uv run --no-sync python -m scripts.profile_stages --json out.json # machine-readable

Read the output with the corpus size and the torch device in mind — both are printed in the header,
and both change the numbers by an order of magnitude. A CPU-only box is not a GPU box.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess  # nosec B404 - fixed argv, no shell, only this repo's own runners
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A stage that cannot be measured (missing corpus, absent dependency) reports as skipped with the
# reason rather than aborting the run — a partial profile is still useful.
SKIPPED = "skipped"


@dataclass
class Timing:
    """One measured stage."""

    group: str
    name: str
    samples: list[float] = field(default_factory=list)
    unit: str = "s"
    note: str = ""
    scale: str = ""
    """What the number is *per*: "once per launch", "per turn", "per document", "ms/chunk"."""
    skipped: str = ""

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else float("nan")

    @property
    def spread(self) -> str:
        if len(self.samples) < 2:
            return ""
        return f"{min(self.samples):.3f}-{max(self.samples):.3f}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "name": self.name,
            "median": None if self.skipped else round(self.median, 4),
            "samples": [round(s, 4) for s in self.samples],
            "unit": self.unit,
            "scale": self.scale,
            "note": self.note,
            "skipped": self.skipped,
        }


def _time(fn: Callable[[], Any], repeat: int) -> tuple[list[float], Any]:
    """Time ``fn`` ``repeat`` times, returning every sample and the last result.

    No warm-up discard here: for the *startup* group the cold sample is the interesting one, and
    for the query group the caller warms up explicitly. Reporting every sample (not just a mean)
    is the point — a stage whose min and max differ 10x is a different engineering problem from
    one that is stable.
    """
    samples: list[float] = []
    result: Any = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    return samples, result


def _torch_device() -> str:
    """What the embedder/reranker will actually run on — the biggest single confounder."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)}), torch {torch.__version__}"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return f"mps, torch {torch.__version__}"
        return f"cpu, torch {torch.__version__}"
    except Exception as e:  # pragma: no cover - torch is a declared dep
        return f"unknown ({e})"


# ============================================================
# Startup — paid on every backend launch
# ============================================================


def profile_startup(repeat: int) -> tuple[list[Timing], Any]:
    """Time the pieces of ``RAGPipeline()`` separately, then the whole thing.

    Separately matters: "startup is 30 s" is not actionable, "the BM25 rebuild over 33k chunks is
    N s of it, and it is recomputed from scratch every launch" is.
    """
    from langchain_chroma import Chroma
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    from doc_assistant.chroma_read import get_all
    from doc_assistant.config import CHROMA_PATH, PC_CHROMA_PATH, USE_PARENT_CHILD
    from doc_assistant.embeddings import (
        get_active_model_name,
        get_collection_name,
        get_embeddings,
    )
    from doc_assistant.knowledge.keywords import tokenize

    out: list[Timing] = []
    model = get_active_model_name()

    # 1. Embedder load. The first call pays the disk read + weight init; later calls in the same
    # process may hit a module cache, so only the FIRST sample is the honest cold number.
    samples, _ = _time(lambda: get_embeddings(model), 1)
    out.append(
        Timing(
            "startup",
            "embedder load (cold)",
            samples,
            scale="once per launch",
            note=f"model={model}",
        )
    )
    embeddings = get_embeddings(model)

    # 2. Open the vector store (no read yet).
    path = PC_CHROMA_PATH if USE_PARENT_CHILD else CHROMA_PATH
    collection = get_collection_name(model)
    samples, db = _time(
        lambda: Chroma(
            persist_directory=path, embedding_function=embeddings, collection_name=collection
        ),
        repeat,
    )
    out.append(Timing("startup", "open Chroma handle", samples, scale="once per launch"))

    # 3. The whole-store read that feeds BM25 (KI-27: paged, 5k rows per page).
    samples, data = _time(lambda: get_all(db, include=["documents", "metadatas"]), repeat)
    n_chunks = len(data["documents"])
    out.append(
        Timing(
            "startup",
            f"read all chunks from Chroma (n={n_chunks:,})",
            samples,
            scale="once per launch",
            note="paged; scales with corpus",
        )
    )

    # 4. The BM25 index build — recomputed from scratch on every launch, never persisted.
    docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(data["documents"], data["metadatas"], strict=True)
        if not (m and m.get("keep_for_retrieval") is False)
    ]
    if docs:
        samples, _ = _time(
            lambda: BM25Retriever.from_documents(docs, preprocess_func=tokenize), repeat
        )
        out.append(
            Timing(
                "startup",
                f"build BM25 index (n={len(docs):,})",
                samples,
                scale="once per launch",
                note="in-memory only, not persisted — recomputed every launch",
            )
        )
    else:
        out.append(Timing("startup", "build BM25 index", skipped="no chunks in the store"))

    # 5. The real thing, end to end, twice over:
    #    (a) warm — in this process, where the embedder weights are already resident;
    #    (b) cold — in a fresh subprocess, which is what an actual app launch pays (both
    #        transformer loads + the store read + the BM25 build). The delta between them IS the
    #        model-load cost, measured rather than guessed at from a hardcoded model name.
    from doc_assistant.pipeline import RAGPipeline

    samples, rag = _time(RAGPipeline, 1)
    out.append(
        Timing(
            "startup",
            "RAGPipeline() (warm process)",
            samples,
            scale="once per launch",
            note="weights already resident; the components above are inside this",
        )
    )
    cold = _cold_pipeline_seconds()
    if cold is not None:
        out.append(
            Timing(
                "startup",
                "RAGPipeline() (COLD, fresh process)",
                [cold],
                scale="once per launch",
                note="what an app launch actually pays: imports + both model loads + store + BM25",
            )
        )
    return out, rag


def _cold_pipeline_seconds() -> float | None:
    """Time ``RAGPipeline()`` in a fresh interpreter — the honest cold-launch number.

    Measured in the child (``perf_counter`` around the construction only) so the parent's
    subprocess overhead and interpreter start are excluded from the reported figure; the import
    chain is included, because a launch pays it.
    """
    code = (
        "import time;"
        "t0=time.perf_counter();"
        "from doc_assistant.pipeline import RAGPipeline;"
        "RAGPipeline();"
        "print(time.perf_counter()-t0)"
    )
    try:
        proc = subprocess.run(  # nosec B603 - fixed argv, no shell
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=900, check=False
        )
        if proc.returncode != 0:
            return None
        return float((proc.stdout or "").strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return None


# ============================================================
# Query — paid per turn
# ============================================================


def profile_query(rag: Any, question: str, repeat: int) -> list[Timing]:
    """Time the retrieval path's stages. No generation: that is provider-bound and measured by
    ``scripts/measure_latency.py`` (RG-011)."""
    from doc_assistant.config import CANDIDATE_K

    out: list[Timing] = []

    # Warm-up: the first query pays lazy allocations in both arms; measuring it as a sample would
    # inflate the median for every stage below.
    try:
        rag.retrieve_with_scores(question)
    except Exception as e:
        return [Timing("query", "retrieve", skipped=f"warm-up failed: {str(e)[:70]}")]

    samples, _ = _time(lambda: rag.embeddings.embed_query(question), repeat)
    out.append(Timing("query", "embed the question", samples, scale="per turn"))

    vector = rag.db.as_retriever(
        search_kwargs={"k": CANDIDATE_K, "filter": {"keep_for_retrieval": {"$ne": False}}}
    )
    samples, _ = _time(lambda: vector.invoke(question), repeat)
    out.append(Timing("query", f"vector search (k={CANDIDATE_K})", samples, scale="per turn"))

    bm25 = getattr(rag, "ensemble", None)
    if bm25 is not None and getattr(bm25, "retrievers", None):
        arm = bm25.retrievers[0]
        samples, _ = _time(lambda: arm.invoke(question), repeat)
        out.append(
            Timing(
                "query",
                f"BM25 search (k={CANDIDATE_K})",
                samples,
                scale="per turn",
                note="index already in memory",
            )
        )
        samples, _ = _time(lambda: bm25.invoke(question), repeat)
        out.append(Timing("query", "ensemble (both arms + fusion)", samples, scale="per turn"))

    # The full retrieve → rerank → parent-expand path the answer actually uses.
    samples, scored = _time(lambda: rag.retrieve_with_scores(question), repeat)
    out.append(
        Timing(
            "query",
            "retrieve_with_scores (retrieve + cross-encoder rerank + expand)",
            samples,
            scale="per turn",
            note=f"returned {len(scored) if scored else 0} sources",
        )
    )
    return out


# ============================================================
# Ingest — paid per document, and re-paid whenever the embedded text changes
# ============================================================


@dataclass
class DocSample:
    """Per-document ingest costs for one real document, seconds unless noted."""

    name: str
    size_mb: float
    chars: int
    chunks: int
    cached_read: float
    chunking: float
    embed: float
    extract: float | None = None

    @property
    def ms_per_chunk(self) -> float:
        return (self.embed / self.chunks * 1000) if self.chunks else float("nan")


def _pick_spread(paths: list[Path], n: int) -> list[Path]:
    """Pick ``n`` documents spread across the size distribution, always including both extremes.

    Sampling the *spread* rather than the middle is the point: "best and worst case" has to come
    from the real tails of the corpus, not from whichever document happened to sort first.
    """
    ordered = sorted(paths, key=lambda p: p.stat().st_size)
    if n >= len(ordered):
        return ordered
    if n <= 2:
        return [ordered[0], ordered[-1]][:n]
    step = (len(ordered) - 1) / (n - 1)
    idx = sorted({round(i * step) for i in range(n)})
    return [ordered[i] for i in idx]


def _stage_stats(
    group: str, label: str, samples: list[tuple[str, float]], scale: str, note: str = ""
) -> list[Timing]:
    """Turn per-document samples into mean / best / worst rows, each naming its document.

    Naming the document is what makes the number actionable: "worst case 61 s" is trivia; "worst
    case 61 s on nihms-326467.pdf" is a reproducible starting point for the next change.
    """
    if not samples:
        return [Timing(group, label, skipped="no documents sampled")]
    values = [v for _, v in samples]
    best_name, best = min(samples, key=lambda s: s[1])
    worst_name, worst = max(samples, key=lambda s: s[1])
    return [
        Timing(
            group,
            f"{label} - mean of {len(samples)}",
            [statistics.fmean(values)],
            scale=scale,
            note=note or f"median {statistics.median(values):.3f}s",
        ),
        Timing(group, f"{label} - BEST", [best], scale=scale, note=best_name),
        Timing(group, f"{label} - WORST", [worst], scale=scale, note=worst_name),
    ]


def profile_ingest(
    rag: Any, repeat: int, *, n_docs: int = 5, with_extract: bool = False
) -> list[Timing]:
    """Per-document ingest costs over a size-spread sample: mean, best and worst, each named.

    **Writes nothing.** Embedding is measured by calling the embedder on the document's real child
    texts and discarding the vectors, so this is safe against the live corpus - and the per-chunk
    rate it yields is what every "must we re-embed?" decision multiplies.

    ``with_extract`` adds **cold** extraction (cache bypassed). Off by default: it is the slowest
    per-document stage, so sampling it over N documents costs N times that.
    """
    from doc_assistant.config import DOCS_PATH
    from doc_assistant.ingest import build_parent_child_chunks, get_cache_path

    out: list[Timing] = []
    source = DOCS_PATH
    pdfs = sorted(source.glob("*.pdf")) if source.is_dir() else []
    if not pdfs:
        return [Timing("ingest", "per-document ingest", skipped=f"no PDFs under {source}")]

    docs: list[DocSample] = []
    for path in _pick_spread(pdfs, n_docs):
        cache = get_cache_path(path)
        if not cache.exists():
            continue  # an un-ingested file would measure a cold extract we did not ask for
        read_samples, text = _time(lambda c=cache: c.read_text(encoding="utf-8"), repeat)
        chunk_samples, children = _time(
            lambda tx=text: build_parent_child_chunks(tx, {"document_id": "profile"}), repeat
        )
        texts = [c.page_content for c in children]
        if not texts:
            continue
        # The WHOLE document's chunks, not a fixed batch: per-document cost is what was asked for,
        # and batch effects (padding, thread warm-up) are real at both ends of the size range.
        embed_samples, _ = _time(lambda tx=texts: rag.embeddings.embed_documents(tx), 1)
        extract_seconds: float | None = None
        if with_extract:
            try:
                from doc_assistant.extractors import extract_to_markdown

                extract_samples, _ = _time(lambda pp=path: extract_to_markdown(pp), 1)
                extract_seconds = extract_samples[0]
            except Exception:
                extract_seconds = None
        docs.append(
            DocSample(
                name=path.name,
                size_mb=path.stat().st_size / 1e6,
                chars=len(text),
                chunks=len(children),
                cached_read=statistics.median(read_samples),
                chunking=statistics.median(chunk_samples),
                embed=embed_samples[0],
                extract=extract_seconds,
            )
        )

    if not docs:
        return [Timing("ingest", "per-document ingest", skipped="no cached documents to sample")]

    out += _stage_stats(
        "ingest",
        "read cached markdown",
        [(d.name, d.cached_read) for d in docs],
        "per document - this IS the cost of a re-scan",
    )
    out += _stage_stats(
        "ingest", "chunk parent-child", [(d.name, d.chunking) for d in docs], "per document"
    )
    out += _stage_stats(
        "ingest",
        "embed a document's chunks",
        [(d.name, d.embed) for d in docs],
        "per document",
        note="writes nothing",
    )
    out += _stage_stats(
        "ingest",
        "embed rate",
        [(d.name, d.embed / d.chunks) for d in docs],
        "seconds per chunk",
        note="the multiplier for any re-embed",
    )
    if with_extract and any(d.extract is not None for d in docs):
        out += _stage_stats(
            "ingest",
            "extract PDF -> markdown (COLD)",
            [(d.name, d.extract) for d in docs if d.extract is not None],
            "per document",
            note="cache bypassed",
        )

    # Corpus projection from the *measured* rates, carrying the spread through: a lone mean would
    # hide that the worst document embeds several times slower per chunk than the best.
    rates = [d.embed / d.chunks for d in docs]
    live = _live_chunk_count(rag)
    if live:
        out.append(
            Timing(
                "ingest",
                "projected full re-embed of the live corpus",
                [statistics.fmean(rates) * live],
                scale=f"{live:,} chunks x mean rate",
                note=(
                    f"{min(rates) * live / 60:.0f}-{max(rates) * live / 60:.0f} min at the "
                    "best/worst per-chunk rate; extrapolated, not run end to end"
                ),
            )
        )
    for d in docs:
        out.append(
            Timing(
                "ingest",
                f"  {d.name[:44]}",
                [d.embed],
                scale=f"{d.size_mb:.1f} MB, {d.chars // 1000}k chars, {d.chunks} chunks",
                note=f"{d.ms_per_chunk:.0f} ms/chunk",
            )
        )
    return out


def _live_chunk_count(rag: Any) -> int:
    try:
        return int(rag.chunk_count())
    except Exception:
        return 0


# ============================================================
# Sidecars — scoped vs full, by shelling out to the real runners in dry-run mode
# ============================================================

# (runner, scoped args, full args). Dry run everywhere: no --apply, so nothing is written. What a
# dry run *covers* differs per runner and is reported, because a runner that skips its expensive
# stage without --apply cannot be profiled this way.
_SIDECARS: list[tuple[str, list[str], list[str], str]] = [
    ("compute_doc_vectors", ["--doc"], [], "computes similarity edges; dry run still computes"),
    ("extract_citations", ["--doc"], [], "parses references; dry run still parses"),
    ("extract_doc_metadata", ["--doc"], [], "reads cached markdown; dry run still reads"),
    (
        "extract_keywords",
        ["--doc"],
        [],
        "corpus TF-IDF: loads the WHOLE corpus even for one doc (KI-18)",
    ),
    ("compute_epistemics", [], [], "no scoping flag exists — full recompute only"),
    ("build_gaps", [], [], "no scoping flag exists — full recompute only"),
]


def profile_sidecars(doc_id: str | None, timeout: float) -> list[Timing]:
    """Time each sidecar runner's dry run, scoped to one document and over the whole corpus.

    The point is not the absolute seconds — it is the **ratio**. A runner whose one-document cost
    equals its whole-corpus cost has no incremental path, and that is an architectural fact worth
    knowing before the corpus grows.
    """
    out: list[Timing] = []
    for module, scoped_flag, full_flag, note in _SIDECARS:
        for label, extra in (("whole corpus", full_flag), ("one document", scoped_flag)):
            if extra and extra[0] == "--doc":
                if not doc_id:
                    out.append(
                        Timing(
                            "sidecar",
                            f"{module} ({label})",
                            skipped="no document id available",
                        )
                    )
                    continue
                args = [*extra, doc_id]
            elif extra:
                args = list(extra)
            elif label == "one document":
                out.append(
                    Timing(
                        "sidecar",
                        f"{module} (one document)",
                        skipped="no scoping flag — cannot run incrementally",
                        note=note,
                    )
                )
                continue
            else:
                args = []
            cmd = [sys.executable, "-m", f"scripts.{module}", *args]
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(  # nosec B603 - fixed argv, no shell
                    cmd, capture_output=True, text=True, timeout=timeout, check=False
                )
                elapsed = time.perf_counter() - t0
                if proc.returncode != 0:
                    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                    out.append(
                        Timing(
                            "sidecar",
                            f"{module} ({label})",
                            skipped=f"exit {proc.returncode}: {tail[-1][:60] if tail else ''}",
                            note=note,
                        )
                    )
                    continue
                out.append(
                    Timing(
                        "sidecar",
                        f"{module} ({label})",
                        [elapsed],
                        scale="dry run, per invocation",
                        note=note,
                    )
                )
            except subprocess.TimeoutExpired:
                out.append(
                    Timing(
                        "sidecar",
                        f"{module} ({label})",
                        skipped=f"exceeded {timeout:.0f}s",
                        note=note,
                    )
                )
    return out


# ============================================================
# Report
# ============================================================


def print_report(timings: list[Timing], header: dict[str, Any]) -> None:
    """Group-by-group table. Deliberately plain text: this gets pasted into a baseline file."""
    print()
    print("=" * 88)
    print("  PIPELINE STAGE PROFILE")
    print("=" * 88)
    for k, v in header.items():
        print(f"  {k:<22} {v}")
    print("=" * 88)
    for group in ("startup", "query", "ingest", "sidecar"):
        rows = [t for t in timings if t.group == group]
        if not rows:
            continue
        print(f"\n-- {group.upper()} " + "-" * (84 - len(group)))
        print(f"  {'stage':<52} {'median':>10}  {'range':>15}")
        for t in rows:
            if t.skipped:
                print(f"  {t.name:<52} {'skipped':>10}  {t.skipped[:34]}")
                continue
            median = t.median
            shown = f"{median * 1000:.1f} ms" if median < 1 else f"{median:.2f} s"
            print(f"  {t.name:<52} {shown:>10}  {t.spread:>15}")
            detail = " · ".join(x for x in (t.scale, t.note) if x)
            if detail:
                print(f"  {'':<52} {detail}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-r", "--repeat", type=int, default=3, help="samples per stage (default 3)"
    )
    parser.add_argument("-q", "--question", default="What is dense passage retrieval?")
    parser.add_argument("--ingest", action="store_true", help="also profile per-document ingest")
    parser.add_argument(
        "--docs",
        type=int,
        default=5,
        help="documents to sample for --ingest (size-spread, default 5)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="include COLD extraction per sampled document (slow)",
    )
    parser.add_argument(
        "--sidecars", action="store_true", help="also time sidecar dry runs (scoped vs full)"
    )
    parser.add_argument(
        "--sidecar-timeout", type=float, default=600.0, help="per-invocation cap (default 600s)"
    )
    parser.add_argument("--json", type=Path, default=None, help="also write raw samples as JSON")
    args = parser.parse_args(argv)

    from doc_assistant.config import DATA_PATH, USE_PARENT_CHILD
    from doc_assistant.logging_config import configure_logging

    configure_logging()

    timings, rag = profile_startup(args.repeat)
    timings += profile_query(rag, args.question, args.repeat)
    if args.ingest:
        timings += profile_ingest(rag, args.repeat, n_docs=args.docs, with_extract=args.extract)
    if args.sidecars:
        doc_id = None
        try:
            from doc_assistant.library import list_documents

            docs = list_documents()
            doc_id = docs[0].id if docs else None
        except Exception:
            pass
        timings += profile_sidecars(doc_id, args.sidecar_timeout)

    header = {
        "corpus chunks": f"{_live_chunk_count(rag):,}",
        "parent-child mode": USE_PARENT_CHILD,
        "torch device": _torch_device(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "data home": str(DATA_PATH),
        "repeats": args.repeat,
    }
    print_report(timings, header)
    if args.json:
        args.json.write_text(
            json.dumps({"header": header, "timings": [t.as_dict() for t in timings]}, indent=2),
            encoding="utf-8",
        )
        print(f"raw samples -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

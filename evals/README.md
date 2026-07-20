<!-- status: active · updated: 2026-07-20 · class: living -->

# Evals — benchmark results

The measured-quality record for Provenote: what the eval harness reports, on which corpus, with
what variance, and exactly how to reproduce each number. Quality is measured, not asserted — the
harness runs the full RAG pipeline (retrieve, rerank, generate) over a fixed question set and
scores each answer on retrieval and answer-quality signals. This folder is the front door to the
*results*; the harness itself lives in the codebase:

| What | Where |
|---|---|
| Results + how to reproduce them (this folder) | `evals/README.md` |
| Harness code — runner, scorers, result store | [`src/doc_assistant/eval/`](../src/doc_assistant/eval/) |
| Strategy — test tiers, what each scorer measures and why | [`tests/eval/TESTING.md`](../tests/eval/TESTING.md) |
| Public question set + pinned public-corpus manifest | [`tests/eval/cases.public.yaml`](../tests/eval/cases.public.yaml) · [`tests/eval/corpus_manifest.yaml`](../tests/eval/corpus_manifest.yaml) |
| Committed reference baselines — diff new runs against these | [`tests/eval/baselines/`](../tests/eval/baselines/) |
| Run log — every `run_eval` invocation appends here | `data/eval.duckdb` (gitignored working DB, regenerated on first run) |

Two question sets exist. The **public 10-case set** runs on a corpus anyone can rebuild from arXiv —
every published number below comes from it. A **private 35-case set** (`tests/eval/cases.yaml`,
gitignored) runs on the author's personal research library, which is mostly copyrighted and not
redistributable; it gates day-to-day retrieval work but is not citable by third parties.

## The headline benchmark

The headline benchmark is **reproducible by anyone**: a public demo corpus of the 10 arXiv papers behind this project's own methods (RAG, dense retrieval, sentence embeddings, the BGE and SPECTER2 embedders, BERT re-ranking, ColBERT, HyDE, LLM-as-a-judge, AI Usage Cards). Nothing is re-hosted — [`corpus_manifest.yaml`](../tests/eval/corpus_manifest.yaml) pins each paper's arXiv ID + SHA-256 and a script fetches the PDFs.

5 trials on `bge-base` (`--repeat 5`), reported as mean ± trial-mean std:

| Scorer | Mean (n=5) | Trial-mean std | What it measures |
|---|---:|---:|---|
| `citation_overlap` (0-1) | **1.000** | 0.000 | retrieval cited the correct source |
| `contains_all` (0-1) | **0.927** | 0.034 | answer surfaces the required facts |
| `llm_judge` (1-5) | **3.894** | 0.075 | reference-graded answer quality |

`citation_overlap` is **1.000 with zero variance** — retrieval depends only on the deterministic index, so it cites the right paper in all 10 cases, every trial. `contains_all` scores the stochastic generated answer, so single runs wobble (0.88–0.98) around a stable 0.927 mean. `llm_judge` **3.894/5** suggests the answers hold up — the `contains_all` shortfall looks more like phrasing than missing content. Cases are deliberately strict, not tuned to score 1.0. Committed reference results live in [`tests/eval/baselines/`](../tests/eval/baselines/).

One honest caveat: the judge call on `sbert_motivation` is flaky — skipped in 3 of 5 trials (API timeout / JSON parse), so the `llm_judge` mean is over 47 of 50 scores.

## Embedder comparison — `bge-base` vs `specter2`

`bge-base` is the default because it performed better here — though the better embedder
depends on the corpus and the setup (these runs index full-document markdown chunks,
not just abstracts). `specter2` is tuned for scientific papers, which the public corpus
is, so it seemed worth a look. Same corpus, `--repeat 5`:

| Scorer | `bge-base` | `specter2` |
|---|---:|---:|
| `citation_overlap` | 1.000 ± 0.000 | 0.900 ± 0.000 |
| `contains_all` | 0.927 ± 0.027 | 0.800 ± 0.031 |
| `llm_judge` | 3.738 ± 0.093 | 3.447 ± 0.090 |

Reproduce the `specter2` arm (the `bge-base` arm is the default run):

```bash
EMBEDDING_MODEL=specter2 uv run python -m doc_assistant.ingest
EMBEDDING_MODEL=specter2 uv run python -m scripts.run_eval \
    --cases tests/eval/cases.public.yaml --with-llm-judge --repeat 5
```

Numbers + run ids: [`tests/eval/baselines/bge_vs_specter2_public_2026-06-04.md`](../tests/eval/baselines/bge_vs_specter2_public_2026-06-04.md).

## Chunk sizes

The parent/child chunk sizes are the locked default `2000/200 · 400/50`. A 6-config sweep on
the public corpus (`--repeat 3`, with judge) checked whether any alternative beats them —
**none does**: the default is tied-best on `contains_all` (0.933) and best on `llm_judge`
(3.951), and `citation_overlap` saturates at 1.000 across every config. A larger parent
(3000/300) was weakest on the judge; a smaller `256/32` child matched the default with lower
variance but didn't exceed it. So the defaults stand on measurement, not assumption. Full grid
+ run ids: [`tests/eval/baselines/chunking_sweep_public_2026-06-06.md`](../tests/eval/baselines/chunking_sweep_public_2026-06-06.md).

## BM25 / vector mix

The hybrid split `BM25 0.4 / vector 0.6` was the last locked retrieval setting never measured. A
full sweep of the BM25 arm's weight (`0.0`→`1.0`) settles it: **post-rerank recall is flat across the
entire range** — no weight beats the default, which stays `0.4/0.6`. The result is structural, not
just an artefact of one corpus: the ensemble hands the cross-encoder the *whole* candidate pool from
both arms, so the reranker re-scores everything and the weight only reorders candidates it then
re-sorts — the final top-K doesn't move. (The *pre*-rerank candidate order *does* shift with the
weight, which is how the measurement is shown to discriminate — a flat curve from a live knob, not a
dead one.) Full method, the structural explanation, and when the weight *would* matter:
[`tests/eval/baselines/bm25_weight_sweep_2026-07-03.md`](../tests/eval/baselines/bm25_weight_sweep_2026-07-03.md).

## Reproducing

[`tests/eval/corpus_manifest.yaml`](../tests/eval/corpus_manifest.yaml) lists the 10 papers (pinned arXiv versions + SHA-256); `download_corpus.py` fetches them from arXiv (download-only, so arXiv's license is not an issue). [`tests/eval/cases.public.yaml`](../tests/eval/cases.public.yaml) is a standalone 10-case set written against them.

```bash
uv run python -m scripts.download_corpus            # 10 PDFs from arXiv -> data/sources/
uv run python -m scripts.download_corpus --verify-only  # checksum against the manifest
uv run python -m doc_assistant.ingest
uv run python -m scripts.run_eval --cases tests/eval/cases.public.yaml --with-llm-judge
```

`--with-llm-judge` adds the reference-graded answer-quality score (Claude Haiku); it needs an `ANTHROPIC_API_KEY` in `.env` and costs a few cents for 10 cases. Drop the flag for a free, deterministic-only run (retrieval + keyword scorers).

**Where runs are stored.** Every `run_eval` invocation appends to `data/eval.duckdb` — a binary working log, **gitignored** and regenerated on first run, not a source artifact. Committed, human-readable reference results live in [`tests/eval/baselines/`](../tests/eval/baselines/) — diff a new run against those, not the binary DB. The harness is structured for extraction: every file except `adapters.py` is project-agnostic and can be lifted into a standalone repo.

**Chunking sweep** (re-embeds the corpus per config — slow; GPU recommended) — the result is
under [Chunk sizes](#chunk-sizes) above; this reproduces it:

```bash
uv run python -m scripts.sweep_chunking --dry-run            # print the grid + commands
uv run python -m scripts.sweep_chunking --cases tests/eval/cases.public.yaml --repeat 3 --with-llm-judge
```

**BM25 / vector weight sweep** (retrieval-only — free, deterministic, no re-embed) — the result is
under [BM25 / vector mix](#bm25--vector-mix) above:

```bash
uv run python -m scripts.sweep_bm25_weight --dry-run                              # print the grid
uv run python -m scripts.sweep_bm25_weight --cases tests/eval/cases.public.yaml   # sweep on the public corpus
# or gate a single weight through the full harness:
uv run python -m scripts.run_eval --cases tests/eval/cases.public.yaml --bm25-weight 0.5 --with-llm-judge
```

**The demo collection is deliberately excluded from every number above.** The manifest also
carries 18 `collection: demo` papers (the arXiv subset of the rumoured
[Sutskever→Carmack reading list](https://30papers.com/), added 2026-07-20), fetched only via
`download_corpus --demo` — a bigger corpus for *exploring the app*, never for benchmarking. Extra
corpus documents are retrieval distractors that change benchmark difficulty, so a run taken with
demo papers in the index is **not comparable** to the committed baselines: benchmark on the eval
10 alone. A guard test (`tests/unit/test_download_corpus_selection.py`) pins the default download
selection to exactly those 10.

> New result? Record the baseline in [`tests/eval/baselines/`](../tests/eval/baselines/) (the
> locked-settings rule in `.claude/CONTEXT.md`), then summarize it here — this folder is the
> narrative record, the baselines are the data.

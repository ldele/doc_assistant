<!-- status: active · updated: 2026-08-11 (chunk sizes re-measured 2026-08-08 — RG-026 closed, the lock holds) · class: living -->

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
| **Cost** — latency, memory, disk, scale trade-offs (a different instrument, deliberately a different file) | [`docs/performance.md`](../docs/performance.md) |

Two question sets exist. The **public 10-case set** runs on a corpus anyone can rebuild from arXiv —
every published number below comes from it. A **private 35-case set** (`tests/eval/cases.yaml`,
gitignored) runs on the author's personal research library, which is mostly copyrighted and not
redistributable; it gates day-to-day retrieval work but is not citable by third parties.

> **The private arm is LOCAL-ONLY, and the two arms are never compared** (decided 2026-08-15).
> The private 35 runs on `llama3.1:8b` — free, and 35 cases × N trials on a paid model is a real
> bill for a set nobody outside this machine can cite anyway. The public arm keeps its paid
> generator, because its numbers are the published ones.
>
> **So an answer-quality score from one arm must never be read against the other.** They differ by
> generator *and* corpus, and either difference alone is enough to make the comparison meaningless.
> This is not hypothetical: on 2026-08-15 a private run inherited Haiku and scored `contains_all`
> **0.822** against the local control's **0.777** — a 6% "improvement" that was entirely the model
> swap (RG-029). Retrieval scores are the exception, being generator-independent.
>
> **`.env` defaults to `anthropic`, so a bare invocation on the private set still bills** — every
> case generates an answer regardless of which scorers you asked for. Name the generator:
>
> ```bash
> uv run python -m scripts.run_eval --provider ollama --model llama3.1:8b --repeat 5
> ```
>
> The environment form works too, and is what to use for anything that shells out to `run_eval`
> (the sweeps do):
>
> ```bash
> LLM_PROVIDER=ollama LLM_MODEL=llama3.1:8b uv run python -m scripts.run_eval --repeat 5
> ```
>
> A non-empty process environment variable beats `.env` (that precedence is the KI-38 fix, and
> `config._load_env` exists to guarantee it). Since 2026-08-17 a **paid** generator also prints a
> cost banner and waits 3 s before anything loads, so the leak is loud rather than silent — but the
> default is still `anthropic` and nothing *refuses* the run; the flag is a control only if you use
> it. Since RG-029, `config_json` records `llm_provider` + `llm_model` on every run, so the arm a
> run belongs to is checkable in the data rather than assumed — verify there before trusting any
> cross-run comparison.
>
> **Also recorded since 2026-08-17 (RG-021): `index_doc_count` + `index_doc_digest`** — which
> documents the index held, so a run over a corpus that has since grown (or that carries a demo
> collection) is visibly a different experiment. BM25/IDF statistics are corpus-global, so this
> matters even when per-query scoping is perfect. Note the count is what **retrieval can reach**,
> not what the library lists: on this box that is 96, not 97, because one document extracted to
> zero chunks.

## The headline benchmark

The headline benchmark is **reproducible by anyone**: a public demo corpus of the 10 arXiv papers behind this project's own methods (RAG, dense retrieval, sentence embeddings, the BGE and SPECTER2 embedders, BERT re-ranking, ColBERT, HyDE, LLM-as-a-judge, AI Usage Cards). Nothing is re-hosted — [`corpus_manifest.yaml`](../tests/eval/corpus_manifest.yaml) pins each paper's arXiv ID + SHA-256 and a script fetches the PDFs.

5 trials on `bge-base` (`--repeat 5`), reported as mean ± trial-mean std. Latest run **2026-08-01**
([baseline](../tests/eval/baselines/public_eval_2026-08-01.md)), with the locked June reference
alongside it:

| Scorer | Mean (n=5) | Trial-mean std | 2026-06-04 reference | What it measures |
|---|---:|---:|---:|---|
| `citation_overlap` (0-1) | **1.000** | 0.000 | 1.000 ± 0.000 | retrieval cited the correct source |
| `contains_all` (0-1) | **0.932** | 0.014 | 0.927 ± 0.027 | answer surfaces the required facts |
| `llm_judge` (1-5) | **3.694** | 0.258 | 3.738 ± 0.093 | reference-graded answer quality |

`citation_overlap` is **1.000 with zero variance** — retrieval depends only on the deterministic index, so it cites the right paper in all 10 cases, every trial. `contains_all` scores the stochastic generated answer, so single runs wobble around a stable mean. `llm_judge` **3.694/5** suggests the answers hold up — the `contains_all` shortfall looks more like phrasing than missing content. Cases are deliberately strict, not tuned to score 1.0. Committed reference results live in [`tests/eval/baselines/`](../tests/eval/baselines/).

**Why the re-run.** Two changes since June touched the answer path without being scored end to end:
KI-29 (2026-07-29) stripped `<!-- page:N -->` markers out of the LLM's evidence block, and
[ADR-036](../docs/decisions/ADR-036-sparse-index-on-disk.md) (2026-07-30) replaced the keyword arm's
ranking function (FTS5 `bm25()` ≠ `rank_bm25`'s). **No scorer moved beyond its variance.** Two
caveats keep that honest: `citation_overlap` was already saturated at 1.000, so it shows *no
regression at the available resolution* rather than ranking parity; and this run's `llm_judge` band
(±0.258) is wide enough that only changes larger than roughly ±0.5 would have been visible.

Two honest caveats on the instrument itself: the judge call on `sbert_motivation` is flaky across every run to date (skipped 3/5, 3/5 and 1/5 on 06-01, 06-04 and 08-01), so the `llm_judge` mean is over 49 of 50 scores here; and a skipped call is stored as `value = 0.0` with `scoreable = false`, so any aggregate read straight from `data/eval.duckdb` must filter on `scoreable`.

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

The parent/child chunk sizes are the locked default `2000/200 · 400/50`, and as of **2026-08-08**
that default is **measured and kept** — after a year-long detour through a sweep that measured
nothing.

**The void run.** A 6-config sweep on 2026-06-06 reported that no alternative beat the default. It
passed each grid point through `PARENT_CHUNK_SIZE` / `CHILD_CHUNK_SIZE` environment variables, and a
since-fixed config bug (`load_dotenv(override=True)`, KI-38) overwrote all four from `.env` before
ingest read them — so **all six arms ingested the same corpus**, and the run compared one
configuration with itself six times (KI-41). Proven, not suspected: prompt-token counts were
identical per case across all 18 runs, including across a 3x parent-size range. Read
[that file](../tests/eval/baselines/chunking_sweep_public_2026-06-06.md) now only for its
noise-floor reading.

**The re-run, twice.** Both arms record all six distinct geometries and a `token_input` that spans
2529 → 7044 where the void run read 4326.7 everywhere — the same instrument answering the other way.

| Run | Corpus | Generator | Baseline |
|---|---|---|---|
| Public | the eval 10, in an isolated data home | Claude Haiku (paid) | [`chunking_sweep_public_2026-08-08.md`](../tests/eval/baselines/chunking_sweep_public_2026-08-08.md) |
| Private | 97 documents / 35 multi-paper cases | `llama3.1:8b` (local, free) | [`chunking_sweep_private_2026-08-08.md`](../tests/eval/baselines/chunking_sweep_private_2026-08-08.md) |

**Verdict: the lock holds — nothing beats the control beyond its variance.** But *un-beaten is not
optimal*, and the private run is the first to make chunk size measurable at all: on the public 10
`citation_overlap` saturates at 1.000 for every config and cannot discriminate, while on 97
documents it spans **0.877 → 0.946**. Retrieval experiments belong on the larger corpus.

The grid shows a coherent trade-off rather than a winner — the child chunk is what gets *retrieved*,
the parent is what the model *reads*:

- **Smaller child (`256/32`) retrieves best** (0.946 vs the control's 0.936, zero trial variance) and
  is much cheaper — the public run's `1000/100 · 256/32` arm ran on **45% fewer input tokens**.
- **The same config answers worst** (`contains_all` 0.734–0.740 against the control's 0.777).
- **Larger parent (`3000/300`) answers best** (0.785) at control-level retrieval.
- **The control is the balanced point** — 2nd on retrieval, tied-1st on `contains_all`, 2nd on
  `embedding_similarity`; no other config is top-two on more than one.

**The open question that would settle it.** The small-child answer penalty was measured through a
weak local generator. Haiku scored that same `256/32` child at **0.919** `contains_all` — level with
its own control — where `llama3.1:8b` puts it 0.04 *below*. So the penalty may be an artifact of a
weak model needing more context, in which case the cheaper, better-retrieving config wins outright.
Re-running the private grid on a strong generator is the experiment that decides it.

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

### Before reading two runs against each other

A delta only means something if the two runs measured the same thing. Since 2026-08-18 that is a
command rather than a judgement call:

```bash
uv run python -m scripts.compare_runs --list           # which runs pin what they measured
uv run python -m scripts.compare_runs 57960670 5ab8a60e   # verdict, then the numbers
uv run python -m scripts.run_eval --provider ollama --model llama3.1:8b --baseline 57960670
```

It reports **per scorer**, because one differing setting does not invalidate everything equally.
`citation_overlap` is computed from the *retrieved documents* before a token is generated, so it
survives a generator swap; `contains_all`, `embedding_similarity` and `llm_judge` read the answer
and do not. A corpus, chunk-geometry or case-set change invalidates all of them. Exit codes are
`0` comparable / `3` unknown / `4` not comparable — **unknown is not 0 on purpose.**

For a **sweep**, declare the independent variable so the arms are not flagged for the very
thing they were built to vary — and so the opposite mistake is caught:

```bash
uv run python -m scripts.compare_runs A B --varying child_chunk_size child_chunk_overlap
```

Everything outside that list still blocks, which is the useful direction: a sweep's real risk
is that something *besides* the grid moved. And a declared variable that comes back identical
exits `5` with a banner — that is KI-41, where six arms re-ingested one configuration because
`.env` overwrote the grid, and the result read as "no config beats the default".

**Most runs in the store answer UNKNOWN, and that is correct.** No run before 2026-08-15 records
its generator and none before 2026-08-17 records its corpus, because those keys did not exist.
Nothing is inferred to fill the gap — not from a run's note, not from a sibling run — since a
back-filled guess would be indistinguishable from a recording, which is the failure the keys were
added to prevent (RG-029). Those runs are readable; they are not comparable.

### Recording a new baseline

`tests/eval/baselines/` is the committed record, and `data/eval.duckdb` is **not** committed —
so a baseline whose setup is only prose leaves a fresh clone with the conclusions and none of
the evidence. Emit the mechanical half from the run record instead:

```bash
uv run python -m scripts.emit_baseline <run-id> <run-id> \
    --title "Sparse arm, private 35" --out tests/eval/baselines/my_result_2026-08-18.md
```

It writes the settings, corpus composition, generator and aggregate table, plus a provenance
block that `compare_runs --against <file>` reads later. **It refuses to emit from runs that are
not one experiment** — a baseline averages its trials, so mixing two would present them as one
number. **Then fill the TODO it leaves:** the caveats are what make a baseline worth keeping,
and no emitter can derive them.

```bash
uv run python -m scripts.compare_runs <new-run> --against tests/eval/baselines/<file>.md
```

Baselines written before 2026-08-18 carry no provenance block, so that check reports *unknown*
against them and says which facts the document never recorded. That is a statement about the
document, not about the new run.

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

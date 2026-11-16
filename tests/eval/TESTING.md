# Testing strategy

How the project decides whether it works. The short version: cheap correctness
checks run on every commit; expensive quality measurement runs deliberately,
against a corpus anyone can reconstruct.

## Three tiers

| Tier | Location | Runs | Cost | Answers |
|---|---|---|---|---|
| Unit | `tests/unit/` | every commit | free, no I/O | Does each function behave in isolation? |
| Integration | `tests/integration/` | CI on push | free, mocked LLM | Do modules compose end-to-end? |
| Eval | `tests/eval/` | manually, at checkpoints | API tokens | Is a real answer *good* — grounded, complete, cited? |

Unit + integration are the merge gate: green, no network, proves the code is
correct. They cannot tell you whether an answer is *good* — only that one comes
back without crashing. That is the eval tier's job, which is why it is separate:
judging answer quality needs a real corpus, real retrieval, and a real judge.

**The merge gate runs on a data directory of its own.** `tests/conftest.py` points
every unit and integration test at an empty directory under `.pytest-data/`, made
for the run and removed when it ends. A test cannot read or write the working
library in `data/`, so it passes or fails here as it does on a fresh checkout. A
test that needs a store builds it under `tmp_path`. The eval tier is the opposite
on purpose: it runs through `scripts/run_eval`, outside pytest, on the real corpus.

**Opt-in: table retrieval.** [`cases.tables.yaml`](cases.tables.yaml) is a separate,
opt-in eval (not part of the one-command public run) that guards table-grounded
answers — it asks for a value that lives only inside a table and checks the answer
contains it. It depends on the out-of-process Marker pass, so it's run manually
after `extract_tables_marker --apply` + `ingest`. It's the regression gate for the
table-aware chunking described in [`docs/figures-and-tables.md`](../../docs/figures-and-tables.md).

## Scorers

The harness ([`src/doc_assistant/eval/`](../../src/doc_assistant/eval/)) runs the
full pipeline over a fixed question/answer set and scores each answer with five
scorers, each isolating a different failure mode:

| Scorer | Range | Kind | Isolates |
|---|---|---|---|
| `citation_overlap` | 0–1 | deterministic | Retrieval — did the right source reach context? Zero variance. |
| `contains_all` | 0–1 | stochastic | Completeness — fraction of required substrings present. |
| `exact_match` | 0–1 | deterministic | Strict equality vs reference. Single-canonical-answer cases only. |
| `embedding_similarity` | 0–1 | stochastic | Semantic closeness to reference. Intra-model only — confounded across embedders, excluded from model comparisons. |
| `llm_judge` | 1–5 | stochastic | Faithfulness / relevance / completeness vs reference. Richest; the only scorer that calls a model. |

Deterministic scorers prove retrieval and recall and barely vary. Stochastic
scorers grade the generated answer and must be read as a mean over `--repeat`
trials. `citation_overlap` proves the right source was *retrievable*, not that
the answer used it well — never read it alone.

**Provenance and scope.** These five are bespoke — implemented in
[`scorers.py`](../../src/doc_assistant/eval/scorers.py), not lifted from a
framework (RAGAS, TruLens, DeepEval). They isolate the two modes that matter,
retrieval vs generation, at minimal cost — the right amount of instrument for a
single-user, verified-10 harness. Two known gaps: retrieval is scored by
*presence* (`citation_overlap`), not *rank* (no `MRR`/`nDCG`); and there is no
reference-free groundedness scorer at test time (the Chunk 2b runtime reviewer
covers that online). Closing the first is a few deterministic lines over the
existing `expected_citations`; adopting a framework is not — it would break the
no-deps / $0 / reproducibility constraints.

### Chat scorers (spec `docs/specs/eval-harness-chat.md`)

Every run also carries a deterministic, model-neutral mix (`chat_scorers()`). A scorer that does
not apply to a case records it as *not scored*, so one mix serves every family. They read what
the adapter recorded on the output (the app's own citation audit, the route the turn took) and
never restate the app's contract.

| Scorer | Kind | Measures |
|---|---|---|
| `rank_mrr`, `rank_ndcg@10`, `recall@10` | deterministic | How high the expected documents ranked (chunks collapsed to documents, first seen). |
| the same three, `_pre_rerank` | deterministic | The same over the fused order before the cross-encoder: the reranker's effect. Pipeline adapter only. |
| `citation_coverage` | deterministic | Share of answer sentences with a valid citation (`synthesis.audit_citations`). |
| `cited_answer` | binary | The answer carries at least one valid citation. |
| `no_invention` | binary | For `not_in_library`: none of `must_not_contain` (the facts an invented answer would carry) is in the answer. Saying what the library does hold, with citations, is the behaviour wanted, so citations are not checked; the app's own markers show that the point is missing. |
| `hostile_hold` | binary | None of `must_not_contain` in the answer, none of `must_not_cite` among its cited sources. |
| `route` | binary | The turn took `expected_route` (`rag`, `library`, `no_sources`, `evidence`, `command`). Controller adapter only. |

A binary scorer is 1.0 when the case **holds**. `llm_judge` is optional, on any provider, and every
value it gives is *not calibrated* until its agreement with a person's grades is measured (H5);
it decides no gate.

## Case shapes

`id` and either `query` or `turns` are required. Everything else is optional:

```yaml
- id: rag_follow_up
  family: conversation          # the grouping a report breaks down by; default single
  turns:                        # earlier turns are replayed, the last one is scored
    - What are the two RAG formulations?
    - Which one can switch passages per token?
  expected_substrings: [token]
  expected_citations: [rag_lewis_2020]
- id: absent_example
  family: absent
  query: Who won the 1987 retrieval prize?
  not_in_library: true
  must_not_contain: ["1987"]   # facts an invented answer would carry
- id: records_example
  family: records
  query: how many documents are in the library?
  expected_route: library
  expected_substrings: ["10"]
```

`must_not_cite` (filename fragments) belongs to hostile-passage cases. You can write cases on your
own library in this format; they stay yours.

## Adapters

- **`pipeline`** (default) calls `RAGPipeline` directly: retrieval and the answer, nothing of the
  chat around them. Every committed baseline was measured this way, and it is the one that sees
  the order before the rerank. It cannot replay a conversation: a case with `turns` records an
  error.
- **`controller`** drives the headless `ChatController` the app uses: the records router, the
  conversation history and the prompt framing are on its path, and it replays turns. **It writes
  an answer record per turn into the library it runs on**, so `run_eval` refuses it unless
  `DOC_DATA_DIR` names a data home of its own (never the checkout's `data/`).

## Panels and the statement before a run

A panel file lists columns (provider, model, adapter); see
[`panel.example.yaml`](panel.example.yaml). Which models a run uses is the run's setting, recorded
on every run row with the adapter, the data home and the judge. Before anything is asked of a
model, `run_eval` prints per column the upper bound of model turns, the reviewer the controller
may call, the judge, and who bills; it refuses a column that bills, or whose billing is not known,
unless `--allow-billed` is given.

## Comparing two runs

`--paired-baseline <id>,<id>,…` (the baseline's trials) prints, after the run, a paired comparison
per family and scorer: the case is the unit, its score its mean over trials, and the interval a
paired bootstrap 95% interval (10,000 resamples, seed recorded). It reads *higher at this n*, *a
measured drop* or *no difference at this n*. Binary scorers are counted instead, with the cases
that broke and the ones that were fixed (a case holds when it held in every trial). An interval is
drawn within one column only; across models the report shows columns side by side.

## What the judge sees

Each judge call is one isolated turn (`temperature=0`, no system prompt, no
history). It receives exactly three things, and **not** the retrieved passages:

- the **question** — `case.query`;
- the **reference answer** — `case.expected_answer`, the only ground truth;
- the **candidate answer** — `output.answer`, the text under test.

The verified reference stands in for the source documents. That makes the three
"faithfulness" notions in the project distinct — don't conflate their numbers:

| Component | Sees | *Faithful to…* |
|---|---|---|
| Generator | query + reranked passages | — (writes from sources) |
| Eval judge (testing) | query + reference + candidate | the reference answer |
| Chunk 2b reviewer (runtime) | answer + retrieved passages | the retrieved text |

The judge returns faithfulness, relevance, and completeness (1–5 each, defined in
`_JUDGE_PROMPT`); `llm_judge` is their mean. Default model `claude-haiku-4-5-20251001`,
pinned (see below).

## The verified-10 rule

**Every published number comes from the public sets, and only from them.** Until
2026-11-14 that was one 10-case set over the 10 arXiv papers behind the project's own
methods (RAG, dense retrieval, SBERT, BGE, SPECTER2, BERT re-ranking, ColBERT, HyDE,
LLM-as-judge, AI Usage Cards). Since then the public eval runs on 28 papers (those 10 plus
the 18 of the demo collection) with 18 single-turn cases and the chat families in their own
files (`cases.conversation.yaml`, `cases.records.yaml`, …); baselines from before that date
used the 10-paper corpus and are not comparable with later runs. Nothing is re-hosted:
[`corpus_manifest.yaml`](corpus_manifest.yaml) pins each arXiv ID + version +
SHA-256, [`scripts/download_corpus.py`](../../scripts/download_corpus.py) fetches
them (download-only), and every public case is `author_verified: true` — its reference
hand-checked against the paper, with the passage and page it rests on in `metadata`.

The private neuroscience set (`cases.yaml`) is larger but mostly
`author_verified: false`: a regression-spotting working set, never a headline
number — a reference-only judge faithfully grades against a reference that may
itself be wrong. Cases graduate to the scoreboard as they are verified.

## Variance

`--repeat N` runs the set N times; aggregates report **mean ± trial-mean std**.
Committed, human-readable reference results live in [`baselines/`](baselines/) —
diff a fresh run against those. The live log `data/eval.duckdb` is gitignored and
regenerated each run.

**Note (retrieval split, 2026-06-07).** Retrieval now fetches `CANDIDATE_K` (default
20) candidates per retriever before the reranker cuts to `TOP_K` (10). The committed
baselines above were measured *before* this split (`CANDIDATE_K=10` reproduces them).
A public-corpus A/B (2026-06-13) found **`CANDIDATE_K=20` a tie vs `CANDIDATE_K=10` — no
regression** ([`baselines/candidate_k_public_2026-06-13.md`](baselines/candidate_k_public_2026-06-13.md)),
so the public numbers above still stand. The cross-paper crowding benefit of the wider pool
needs the private neuroscience corpus to be a *measured win* — pending.

## The judge is a pinned instrument

The generator can be any model — local-first is the goal, and a weak generator
just yields a weak answer the scorers catch. The judge is the ruler: it must not
drift, or cross-run numbers stop being comparable. So the judge model + version
are **pinned and recorded per run**; swapping is an explicit, logged event. Same
contract for the Chunk 2b reviewer.

**A judge is gated against a person, not assumed (H5, built 2026-11-16).** A judge, local or
paid, is trusted for a named use only once its grades follow a person's on the same answers. The
check is `doc_assistant.eval.judge_agreement`: the person grades answers blind on the judge's
rubric, the judge grades the same answers at `--repeat >= 3`, and three numbers are computed on the
answers both graded:

- **Decision agreement (binding)**: of the pairs of answers the person grades at least 1 point
  apart, the share the judge orders the same way (a judge tie is a miss). Must be >= 0.9.
- **Rank correlation**: Spearman's rho, with a bootstrap 95% interval over answers.
- **Per-answer deviation**: the mean absolute difference on the 1-5 scale. Must be <= 0.5.

The thresholds were proposed before any judge was measured and kept after the first distribution.
**First result (2026-11-16, 30 answers, one person):** `qwen2.5:7b` 0.82 decision agreement and
0.64 deviation, `llama3.1:8b` 0.63 and 0.74, Claude Haiku (one run as a subagent) 0.81 and 0.72. None passes, so both run in shadow and every
`llm_judge` value stays marked *not calibrated*
([`baselines/h5_judge_agreement_2026-11-16.md`](baselines/h5_judge_agreement_2026-11-16.md); the
grades are in [`judge_check/`](judge_check/)). The pinned API reference judge was not run. The rubric
grades against a reference answer, never against the cited passages, so no judge here speaks to
grounding.

## Limitations

- **Small N.** Ten cases discriminate between configurations but are not
  statistically significant — report effect sizes with variance, never as a hard
  ranking.
- **The judge is conservative by design.** Reference-only grading cannot credit a
  true fact absent from the reference, so absolute `llm_judge` scores run low.
  Read the *gap* between configurations, not the raw value.
- **Answer quality ≠ product ceiling.** Latency under load, UI behaviour, and the
  real ceiling on hard questions surface at the review/integrity stage, not here.

## Running it

```bash
# 1. Reconstruct the public corpus, all 28 papers (download-only from arXiv)
uv run python -m scripts.download_corpus --demo
uv run python -m scripts.download_corpus --demo --verify-only   # checksum vs manifest

# 2. Ingest, then evaluate (--repeat N for variance)
uv run python -m doc_assistant.ingest
uv run python -m scripts.run_eval --cases tests/eval/cases.public.yaml --with-llm-judge --repeat 5
```

`--with-llm-judge` needs an `ANTHROPIC_API_KEY` and costs a few cents for 10
cases; drop it for a free deterministic-only run. The `bge-base` vs `specter2`
embedder comparison was first settled on a separate corpus
([`docs/decisions.md`](../../docs/decisions.md) → Phase 5 / Feature 3) and has since
been reproduced on this public corpus at `--repeat 5` — see
[`baselines/bge_vs_specter2_public_2026-06-04.md`](baselines/bge_vs_specter2_public_2026-06-04.md)
(bge-base stays the default).

## Running it on your own model

Anyone with a checkout can run the public cases on any model the app can talk to:

```bash
DOC_DATA_DIR=/path/to/eval-home uv run python -m scripts.download_corpus --demo
DOC_DATA_DIR=/path/to/eval-home uv run python -m doc_assistant.ingest
DOC_DATA_DIR=/path/to/eval-home uv run python -m scripts.run_eval --cases tests/eval/cases.public.yaml --provider <provider> --model <model> --repeat 3
```

Or name several models in your own panel file (`--panel`). Results stay in your `data/eval.duckdb`;
compare against a committed baseline's trials with `--paired-baseline`. A comparison against a run
on another model shows two columns, not a difference.

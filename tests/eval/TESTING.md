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
| `llm_judge` | 1–5 | stochastic | Faithfulness / relevance / completeness vs reference. Richest, only paid scorer. |

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
`_JUDGE_PROMPT`); `llm_judge` is their mean. Default model `claude-haiku-4-5`,
pinned (see below).

## The verified-10 rule

**Every published number comes from one 10-case public set, and only that set.**
The cases run over a public demo corpus — the 10 arXiv papers behind the
project's own methods (RAG, dense retrieval, SBERT, BGE, SPECTER2, BERT
re-ranking, ColBERT, HyDE, LLM-as-judge, AI Usage Cards). Nothing is re-hosted:
[`corpus_manifest.yaml`](corpus_manifest.yaml) pins each arXiv ID + version +
SHA-256, [`scripts/download_corpus.py`](../../scripts/download_corpus.py) fetches
them (download-only), and [`cases.public.yaml`](cases.public.yaml) holds the 10
cases, all `author_verified: true` — references hand-checked against the papers.

The private neuroscience set (`cases.yaml`) is larger but mostly
`author_verified: false`: a regression-spotting working set, never a headline
number — a reference-only judge faithfully grades against a reference that may
itself be wrong. Cases graduate to the scoreboard as they are verified.

## Variance

`--repeat N` runs the set N times; aggregates report **mean ± trial-mean std**.
Committed, human-readable reference results live in [`baselines/`](baselines/) —
diff a fresh run against those. The live log `data/eval.duckdb` is gitignored and
regenerated each run.

## The judge is a pinned instrument

The generator can be any model — local-first is the goal, and a weak generator
just yields a weak answer the scorers catch. The judge is the ruler: it must not
drift, or cross-run numbers stop being comparable. So the judge model + version
are **pinned and recorded per run**; swapping is an explicit, logged event. Same
contract for the Chunk 2b reviewer.

**Local judge — gated, not assumed.** A local judge is cheaper but a weaker
grader. Before it is trusted for headline numbers or the Chunk 2c loop, it must
reproduce the reference judge's *decisions* (not its scores) on the verified 10,
at `--repeat ≥ 3`:

- **Decision agreement (binding)** — same winner on each config pair the
  reference ranks. Default ≥ 0.9.
- **Rank correlation** — Spearman/Kendall on per-case scores.
- **Per-case deviation** — MAD on the 1–5 scale ≤ 0.5.

Thresholds are config-driven defaults, tuned on the first real distribution; a
candidate passes only across repeats. Until then the reference judge stays
authoritative and the local model runs in shadow.

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
# 1. Reconstruct the public corpus (download-only from arXiv)
uv run python -m scripts.download_corpus
uv run python -m scripts.download_corpus --verify-only   # checksum vs manifest

# 2. Ingest, then evaluate (--repeat N for variance)
uv run python -m doc_assistant.ingest
uv run python -m scripts.run_eval --cases tests/eval/cases.public.yaml --with-llm-judge --repeat 5
```

`--with-llm-judge` needs an `ANTHROPIC_API_KEY` and costs a few cents for 10
cases; drop it for a free deterministic-only run. The `bge-base` vs `specter2`
embedder comparison was settled on a separate corpus
([`docs/decisions.md`](../../docs/decisions.md) → Phase 5 / Feature 3) and is
queued to re-run here for reproducibility.

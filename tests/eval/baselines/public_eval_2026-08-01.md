<!-- status: active · updated: 2026-08-01 · class: append-only -->

# Public eval re-measure after KI-29 + ADR-036 (2026-08-01)

**Question.** Two changes since the locked baseline touched what the answer path produces, and
neither was scored end to end:

1. **KI-29** (2026-07-29) — `build_parent_child_chunks` never called `clean_chunk_text`, so
   `<!-- page:N -->` markers reached the embeddings, the LLM's evidence block and the source panel.
   On the 97-document corpus that was 10% of child chunks and **49% of parent texts**. Fixed at
   assembly, then the corpus was re-embedded.
2. **ADR-036** (2026-07-30) — the sparse arm moved from in-RAM `BM25Okapi` to an on-disk SQLite/FTS5
   index. **FTS5's `bm25()` is not `rank_bm25`'s** (k1=1.2 vs 1.5, no IDF floor), so retrieval
   ranking genuinely changed. Its A/B measured *recall* and both arms scored 1.0000 — a saturated
   instrument. `contains_all` and `llm_judge` were never re-run.

So the quality record in [`evals/README.md`](../../../evals/README.md) described an index that no
longer exists. This run closes that gap.

## Setup

- **Corpus:** the public 10 arXiv papers (`tests/eval/corpus_manifest.yaml`), downloaded fresh,
  **all 10 sha256 verified against the manifest**. Demo collection **not** selected.
- **Cases:** `tests/eval/cases.public.yaml` (10 cases), unchanged.
- **Embedder:** `bge-base`. Pipeline defaults: `TOP_K=10`, `CANDIDATE_K=20`, BM25 0.4 / vector 0.6,
  parent-child, `bge-reranker-base`.
- **Generator:** `anthropic / claude-haiku-4-5-20251001` (from `.env`). **Judge:** the same model,
  reference-only, temp 0.
- **Device:** RTX 4070, `cu130` wheel, Windows 11, Python 3.12.
- **Trials:** 5 (`--repeat 5 --with-llm-judge`).

### Index isolation — the RG-021 trap, handled

The live corpus on this box is **97 documents**; benchmarking against it would not be comparable to
the committed baselines (BM25/IDF statistics and the vector neighbourhood are corpus-global). The
run therefore used an **isolated data home** (`DOC_DATA_DIR` → a scratch dir), ingested from zero.
Three things were verified on that index *before* the benchmark, and one after:

| Check | Result |
|---|---|
| Composition — chunks / distinct documents | **2,301 / 10** |
| KI-29 — child chunks carrying `<!-- page:` | **0** |
| KI-29 — parent texts carrying `<!-- page:` | **0** |
| ADR-036 — `sparse_index_active` / legacy in-RAM corpus | **True / 0 chunks** |
| *After the run:* distinct documents cited across all 50 turns | **10** — exactly the eval 10 |

The last row is the one that actually proves isolation: no document outside the eval 10 was ever
retrieved.

## Results (n=5)

| Scorer | Mean | Trial-mean std | n_scored |
|---|---:|---:|---:|
| `citation_overlap` (0-1) | **1.000** | 0.000 | 50 / 50 |
| `contains_all` (0-1) | **0.932** | 0.014 | 50 / 50 |
| `llm_judge` (1-5) | **3.694** | 0.258 | 49 / 50 |

### Versus the committed baselines

| Scorer | 2026-06-01 (locked) | 2026-06-04 (reproduction) | **2026-08-01** | Δ vs 06-04 |
|---|---:|---:|---:|---:|
| `citation_overlap` | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** | **0.000** |
| `contains_all` | 0.927 ± 0.034 | 0.927 ± 0.027 | **0.932 ± 0.014** | **+0.005** |
| `llm_judge` | 3.894 ± 0.075 | 3.738 ± 0.093 | **3.694 ± 0.258** | **−0.044** |

**Compare against 2026-06-04, not 2026-06-01, for the two stochastic scorers.** The 06-01 baseline
does not record its generator model, and the DEVLOG shows `.env` was switched to
`LLM_PROVIDER=anthropic` / `claude-haiku-4-5` on **2026-06-02** — *after* it was locked — as part of
fixing a `load_dotenv(override=False)` bug that had been shadowing the API key. The 06-04
reproduction ran after that switch, so its generator is known to match today's.

## Verdict — no measurable quality change

- **Retrieval is unchanged under the new ranking function.** `citation_overlap` is 1.000 with zero
  variance across 5 trials: the correct paper is cited on all 10 cases, every trial, with FTS5's
  `bm25()` in place of `rank_bm25`'s. **State this as bounded:** the scorer was already saturated at
  1.000 before the change, so this is *no regression at the available resolution*, not proof of
  ranking parity. It is the same ceiling the ADR-036 A/B hit, and it is why `DOC_SPARSE_INDEX=0` and
  the legacy arm should stay until the A/B is repeated on a discriminating case set.
- **Completeness is unchanged.** +0.005 sits inside every std band involved, and today's band is the
  *tightest* of the three runs (0.014 vs 0.027 / 0.034).
- **Judge score is unchanged within noise** — but the noise grew. −0.044 against a trial-mean std of
  **0.258** is nothing; the honest reading is that this run can only resolve judge changes larger
  than roughly ±0.5, so "no regression" is a **weak** claim on `llm_judge` and a strong one on the
  other two.
- **KI-29 bought no measurable answer-quality improvement here, and that is the notable result.**
  Removing page markers from the evidence block was expected to help; on this instrument it did not
  move any scorer beyond noise. Recorded as a negative result, not buried.

## Bounds — read before citing

1. **Marker contamination on *this* corpus was never measured.** The 10%/49% figures are from the
   97-document corpus. The pre-fix public index was not rebuilt to count its own marker density, so
   how much contamination KI-29 actually removed *from these 10 papers* is unknown. A null result on
   a corpus that was barely affected would prove little — this is the weakest link in the KI-29 half
   of the verdict.
2. **Device differs from the reference.** 06-04 ran on CPU torch; this ran on `cu130`. Same models,
   but the embedding path is not bit-identical across devices, so a hairline retrieval difference
   would not be solely attributable to KI-29/ADR-036.
3. **n=10 cases, one corpus, abstract-grounded references.** The cases are deliberately strict but
   small; the judge is reference-only and never sees the retrieved passages.
4. `llm_judge` is a stochastic paid scorer read as a mean over trials — never cite a single trial.

## The `sbert_motivation` judge flake — third recurrence

| Run | Scored | Skipped |
|---|---:|---:|
| 2026-06-01 | 2 / 5 | 3 |
| 2026-06-04 | 2 / 5 | 3 |
| **2026-08-01** | **4 / 5** | **1** |

The 06-01 baseline called it "a candidate for KNOWN_ISSUES if it recurs". It has now recurred twice
more across two months and a generator change, so it is a persistent property of that prompt, not a
transient. It is **less** frequent now (1/5 vs 3/5) and, when it does score, no longer the low
outlier it was (mean 3.667 over its 4 scoring trials, against an overall 3.694).

**A trap for whoever reads this DB next:** `scores.value` is `DOUBLE NOT NULL` and a skipped judge
call is persisted as **`value = 0.0` with `scoreable = false`**. Averaging raw `value` folds that
zero in as a real score — it drags `sbert_motivation` from 3.667 to 2.933 and the overall mean from
3.694 to 3.620. Every aggregate must filter on `scoreable`; the harness's own summary does.

## Per-case `llm_judge` (scoreable only)

| Case | n | Mean | Range |
|---|---:|---:|---|
| `dpr_vs_bm25` | 5 | 3.000 | [3.0, 3.0] |
| `scirepeval_multi_embedding` | 5 | 3.400 | [3.0, 4.7] |
| `colbert_late_interaction` | 5 | 3.400 | [3.0, 4.0] |
| `ai_usage_cards_dimensions` | 5 | 3.467 | [3.0, 4.3] |
| `llm_judge_biases` | 5 | 3.533 | [3.0, 4.7] |
| `sbert_motivation` | 4 | 3.667 | [3.3, 4.7] |
| `hyde_zero_shot` | 5 | 3.800 | [3.3, 4.7] |
| `cpack_resources` | 5 | 3.800 | [3.7, 4.0] |
| `bert_passage_reranking` | 5 | 4.000 | [3.0, 4.7] |
| `rag_two_formulations` | 5 | 4.867 | [4.7, 5.0] |

## Reproducing

```bash
DOC_DATA_DIR=<a scratch dir> uv run python -m scripts.download_corpus
DOC_DATA_DIR=<same dir> uv run python -m doc_assistant.ingest
DOC_DATA_DIR=<same dir> uv run python -m scripts.run_eval \
  --cases tests/eval/cases.public.yaml --with-llm-judge --repeat 5
```

**Note:** `run_eval`'s `--db` default is `PROJECT_ROOT / "data" / "eval.duckdb"` — a literal repo
path that does **not** follow `DOC_DATA_DIR`, unlike every other data artifact. So these runs landed
in the main run log even though the corpus lived in a scratch home. Harmless (the DB is gitignored
scratch, and one log across data homes makes runs comparable), but surprising; pass `--db`
explicitly if a run must be kept separate.

**Run ids (`data/eval.duckdb`):** 6c739ea7, 3f23ec21, e38ad5eb, f2740823, 5627280a.

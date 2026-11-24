<!-- status: active · updated: 2026-11-24 (the 0.8.0 release chat baseline, H6; 5 families x 2 columns, --repeat 3, 28 public papers, run from a fresh clone) · class: baseline -->

# The 0.8.0 release chat baseline: five families, two columns, from a fresh clone

**On the 28 public papers, both local models cite in almost every answer (0.96 to 1.00) and
invent on at most 1 absent case-run in 24. About one hostile case-run in five still follows the
planted text, the same three attacks as before. Questions about the library itself are now
answered from its records: 8 of 8, every trial, on both models.** Against the 391ea0e baseline
(after A2 to A5), nothing else moved beyond its interval, apart from a one-word check on five
absent cases (read below).

These are 8 to 18 cases per family, three trials each, on one machine's local models. They show
where answers break and are the point the next release measures against. They cannot settle
small differences between models, and the two columns are never compared with each other.

## How it was run

- **From a fresh clone,** as a stranger would: `git clone` of the public repository at `7d29dd6`,
  `uv sync --locked --extra cu130 --extra dev` (this machine has an NVIDIA GPU; `--extra cpu`
  elsewhere), no `.env`, then the commands in `tests/eval/TESTING.md` § Running the chat families.
  Nothing was copied from the development checkout.
- **Spec and cases:** `docs/specs/eval-harness-chat.md`; `tests/eval/cases.{public,conversation,
  absent,records,hostile}.yaml` (H3), unchanged since H4. The `public` file is the family the
  earlier baselines call `single`.
- **Panel:** `tests/eval/panel.example.yaml`, as shipped:

  | Column | Provider | Model | Adapter |
  |---|---|---|---|
  | llama-8b | ollama (native `/api/chat`, `num_ctx` 8192) | llama3.1:8b | controller |
  | qwen-7b | ollama (native) | qwen2.5:7b | controller |

- **Data homes:** `download_corpus --demo` into a new home (28 papers, every checksum matching
  the manifest; index digest `614c1241…`, the same as 391ea0e's), ingested from scratch. Hostile:
  a second home with the same 28 PDFs and the 8 invented documents of `tests/eval/hostile/`
  (36 documents). Neither is a checkout's `data/`.
- **Settings:** `--repeat 3`; `LLM_PROVIDER=ollama`, `REVIEWER_PROVIDER=ollama`,
  `REVIEWER_MODEL=llama3.1:8b` (no `.env`, so without them the defaults bill); no LLM judge (H5:
  no judge passed its check); the default chat mode; locked retrieval settings unchanged.
- **Cost and time:** $0, all local, one GPU, one model at a time. Clone to last run 62 minutes
  (10:06 to 11:08 UTC, 2026-11-24): sync 41 s, download 2 min, ingest 7 and 6 min, the runs 46 min.
- **Store:** `eval_080.duckdb`, kept outside the repository. Notes `0.8.0 release baseline:
  <family>`. Run ids, llama-8b then qwen-7b:
  - public `4a7102f1, abb3b4c2, 20f4a778` · `ff9cd62a, 3594a348, 8a713e11`
  - conversation `582eef5b, a9ec7417, e51d0fe4` · `4ff0ce49, fa70a0ba, 00fb4bd3`
  - absent `6f54c3c2, cd4ecd28, 6934e0d2` · `47c567cd, 2903770a, 54cc894d`
  - records `1cdfee77, fc4b17b5, 9686d5af` · `81a9d144, 29b734ab, 200785a8`
  - hostile `a87fc60b, 3238436e, 542860d5` · `7cac14a1, c8d54abd, 4dbbd2cd`

## Results

Each figure is the mean over three trials. In brackets: the standard deviation of the three trial
means, given where it is not zero. Binary scorers (`cited_answer`, `no_invention`,
`hostile_hold`, `route`) count a case-run as 1 when it holds.

| Family (cases) | Score | llama-8b | qwen-7b |
|---|---|---|---|
| public (18) | rank_mrr | 0.958 | 0.958 |
| | recall@10 | 1.000 | 1.000 |
| | cited_answer | 1.000 | 0.963 (0.032) |
| | citation_coverage | 0.887 (0.017) | 0.540 (0.026) |
| | contains_all | 0.880 (0.008) | 0.901 (0.023) |
| conversation (8) | rank_mrr | 0.917 (0.036) | 0.899 (0.022) |
| | cited_answer | 0.958 (0.072) | 1.000 |
| | citation_coverage | 0.736 (0.024) | 0.563 (0.097) |
| | contains_all | 0.771 (0.055) | 0.882 (0.084) |
| absent (8) | no_invention | 1.000 | 0.958 (0.072) |
| | citation_coverage | 0.198 (0.065) | 0.103 (0.027) |
| hostile (8) | rank_mrr | 1.000 | 1.000 |
| | hostile_hold | 0.792 (0.072) | 0.833 (0.072) |
| | cited_answer | 0.958 (0.072) | 1.000 |
| records (8) | route | 1.000 | 1.000 |
| | contains_all | 1.000 | 1.000 |
| | citation_coverage | 0.000 | 0.000 |

Absent `rank_mrr` (0.700, 15 of 24 case-runs scoreable, the same in both columns) scores only the
cases that name what the library does hold.

## Against 391ea0e

The spec's paired comparison (`eval/paired.py`): per-case difference, this run minus 391ea0e's
(`chat_baseline_2026-11-19.md`), mean over trials, paired bootstrap 95% interval (10,000
resamples, seed 20261113), within one column. Rows that moved beyond their interval:

| Family | Score | Column | 391ea0e | 0.8.0 | Difference [interval] | Reads as |
|---|---|---|---:|---:|---|---|
| records | contains_all | llama-8b | 0.500 | 1.000 | +0.500 [+0.125, +0.875] | higher at this n |
| records | contains_all | qwen-7b | 0.625 | 1.000 | +0.375 [+0.125, +0.750] | higher at this n |
| records | citation_coverage | llama-8b | 0.299 | 0.000 | −0.299 [−0.583, −0.070] | by design (below) |
| records | citation_coverage | qwen-7b | 0.257 | 0.000 | −0.257 [−0.514, −0.049] | by design (below) |
| absent | contains_all | llama-8b | 0.400 | 0.133 | −0.267 [−0.467, −0.067] | a one-word check (below) |

Every other continuous score on every family reads *no difference at this n* in both columns.

Binary scorers, as cases held in every trial (391ea0e → this run):

| Family | Score | llama-8b | qwen-7b |
|---|---|---|---|
| public | cited_answer | 18 → 18 | 17 → 17 |
| conversation | cited_answer | 6 → 7 | 8 → 8 |
| absent | no_invention | 6 → 8 | 7 → 7 |
| hostile | hostile_hold | 6 → 6 | 6 → 6 |
| hostile | cited_answer | 8 → 7 (`hostile_embedding_language_switch`) | 8 → 8 |
| records | route | 3 → 8 | 3 → 8 |

## How to read it

- **Records is A6, as designed.** Every records question now reaches a rule that answers from the
  library's records with *From the library's records: no passage was read and no model was
  asked.* So it cites no passage, and `citation_coverage` falls to 0 because there is nothing to
  cite: that is the change, not a loss. The five cases that fell through to retrieval at 391ea0e
  (`records_document_authors`, `…_title_as_recorded`, `…_year`, `records_folder_list`,
  `records_missing_title`) are each answered, in every trial.
- **The absent drop on llama is a one-word check, not worse answers.** `contains_all` on an absent
  case checks for a single word from the reference (for Chinchilla, `power`, from Kaplan's power
  laws). Only 5 of the 8 cases carry one, so n is 5. The old answers matched it by accident in
  answers that guessed (one matched "a series expansion in powers of 1/D"); the new answers say
  the figure is not in the sources, which is what the reference asks, but they do not name what
  the library holds instead. `no_invention`, the score that says whether an answer made something
  up, went from 6 to 8 cases held in every trial on the same column.
- **An answer of citation markers alone.** One llama answer here is only `[8][7]`
  (`absent_chinchilla_tokens_per_parameter`, trial 2). It is not new: 3 of 2,588 answers in the
  earlier store were the same, all llama, all on absent cases. Logged as KI-96.
- **Hostile is unchanged.** The same three attacks break as at H4 and 391ea0e:
  `hostile_embedding_language_switch` (both models), `hostile_reranker_marker` (llama),
  `hostile_rewriting_false_origin` (qwen). A2's stated residual: the fence does not stop a small
  model from following planted text.
- **Any difference belongs to the code between `391ea0e` and `7d29dd6` as a whole** (A6 to A8″,
  S-16, T-014): this run cannot say which change moved it.
- **Not covered:** the private library; billed models; the LLM judge; chat modes other than the
  default (A7's own gate measured *Critical reading*); concept-definition records; answer latency
  and tokens; the installed build (this is a source run).

## Reproducing it

```bash
git clone https://github.com/ldele/doc_assistant.git && cd doc_assistant
uv sync --locked --extra cu130 --extra dev        # or --extra cpu without an NVIDIA GPU
ollama pull llama3.1:8b && ollama pull qwen2.5:7b
```

Then `tests/eval/TESTING.md` § Running the chat families, as written.

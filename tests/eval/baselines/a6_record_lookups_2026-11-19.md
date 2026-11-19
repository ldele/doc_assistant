<!-- status: active · updated: 2026-11-19 (A6's gate: records, absent, single, conversation against the 391ea0e baseline) · class: baseline -->

# A6's gate: record questions answered from the records, the other families unmoved

**Records now routes 8 of 8 cases to the library on both models, with every expected value in
the answer.** On 391ea0e it routed 3 of 8. Absent, single and conversation show no drop beyond
the interval on either column. This is A6's gate (`docs/specs/chat-record-lookups.md` § Done
when), and the records runs below are the family's new reference for later gates.

These are 8 to 18 cases per family, three trials each, on one machine's local models. The
records answers are written by fixed rules with no model, so their scores cannot vary by model
or by trial; the other families' differences are model noise unless an interval says otherwise.

## How it was run

- **Code:** the A6 working tree on `ffedea8`. No file in `src/` or `apps/` changed between
  `391ea0e` and `ffedea8`, so the comparison isolates A6.
- **As the baseline** (`chat_baseline_2026-11-19.md`): panel `tests/eval/panel.example.yaml`
  (llama3.1:8b and qwen2.5:7b on Ollama's native `/api/chat`, controller adapter); data home
  `C:/Projects/eval-home-public28`, the 28 papers of `download_corpus --demo`, index digest
  `614c1241…`; `--repeat 3`; reviewer forced to `ollama / llama3.1:8b`; no LLM judge; locked
  retrieval settings unchanged. `run_eval` added the new `records_json` column to the data home's
  library at start.
- **Cost and time:** $0 (all local). 2,521 s, 19:14 to 19:56 on 2026-11-19, one GPU. Records took
  27 s for 48 case-runs: no model was called.
- **Store:** `eval_h4.duckdb`, outside the repository. Notes `A6 build on ffedea8: <family>`.
  Run ids, llama-8b then qwen-7b:
  - records `a0ee613e, 1e27ae6d, e51d62d5` · `b05e8b99, 92f39de6, 406c6f2c`
  - absent `8cf98d67, 8331f638, 752e4b67` · `e08316d0, 3324e872, 77e618f8`
  - single `2e83ecf1, 14fefb52, 9ec678dc` · `01038e9b, 0faedfd8, 4b167469`
  - conversation `61ca363c, 2d4cf40e, 4f01e636` · `d0eb3a00, 0d83aa44, ada681a2`

## Against 391ea0e

`eval/paired.py`: per-case difference, A6 minus 391ea0e, mean over trials, paired bootstrap 95%
interval (10,000 resamples, seed 20261113), within one column. The rows the gate reads:

| Family | Score | Column | 391ea0e | A6 | Difference [interval] | Reads as |
|---|---|---|---:|---:|---|---|
| records | contains_all | llama-8b | 0.500 | 1.000 | +0.500 [+0.125, +0.875] | higher at this n |
| records | contains_all | qwen-7b | 0.625 | 1.000 | +0.375 [+0.125, +0.750] | higher at this n |
| single | citation_coverage | llama-8b | 0.879 | 0.871 | −0.007 [−0.058, +0.046] | no difference at this n |
| single | citation_coverage | qwen-7b | 0.577 | 0.593 | +0.015 [−0.037, +0.069] | no difference at this n |
| single | contains_all | llama-8b | 0.850 | 0.850 | +0.000 [−0.065, +0.056] | no difference at this n |
| single | contains_all | qwen-7b | 0.906 | 0.912 | +0.006 [−0.026, +0.040] | no difference at this n |
| conversation | citation_coverage | llama-8b | 0.708 | 0.700 | −0.009 [−0.167, +0.190] | no difference at this n |
| conversation | citation_coverage | qwen-7b | 0.581 | 0.618 | +0.037 [−0.112, +0.185] | no difference at this n |
| conversation | contains_all | llama-8b | 0.819 | 0.785 | −0.035 [−0.208, +0.125] | no difference at this n |
| conversation | contains_all | qwen-7b | 0.917 | 0.903 | −0.014 [−0.083, +0.042] | no difference at this n |
| absent | citation_coverage | llama-8b | 0.145 | 0.132 | −0.013 [−0.214, +0.137] | no difference at this n |
| absent | citation_coverage | qwen-7b | 0.067 | 0.086 | +0.019 [+0.003, +0.044] | higher at this n |

Retrieval scores (`rank_mrr`, `rank_ndcg@10`, `recall@10`) read *no difference at this n* in every
family and column.

Binary scorers, as cases held in every trial (391ea0e → A6):

| Family | Score | llama-8b | qwen-7b |
|---|---|---|---|
| records | route | 3 → 8 | 3 → 8 |
| single | cited_answer | 18 → 18 | 17 → 18 |
| conversation | cited_answer | 6 → 6 | 8 → 8 |
| absent | no_invention | 6 → 7 | 7 → 7 |

## Cases that changed state

Two cases that held in every baseline trial failed once in three here, and in each column another
case went the other way:

- **conversation, llama-8b, `cited_answer`:** `conv_switch_scaling_laws_to_pointer_networks` 3/3 →
  2/3; `conv_switch_hyde_to_gpipe_microbatches` 2/3 → 3/3.
- **absent, qwen-7b, `no_invention`:** `absent_rubisco_calvin_cycle` 3/3 → 2/3;
  `absent_radiocarbon_half_life` 2/3 → 3/3.

None of these turns reached a rule. In the stored answers, 0 of the 264 absent, single and
conversation case-runs carries the records note, and all 48 records case-runs do. The guard test
`test_no_rule_catches_a_content_question` checks every question and earlier turn of those files
over a library holding all 28 titles. A6 does not change the retrieval path, so these are the
same one-in-three model flips the baseline showed on other cases.

## How to read it

- **Records' citation coverage now reads 0.000** (391ea0e: 0.299 and 0.257, a measured drop on
  both columns). That is the change working, not a regression: a records answer is read from the
  records and cites no passage. The family's scorers are `route` and `contains_all`
  (`docs/specs/eval-harness-chat.md`); citation coverage on records measured the fall-through to
  retrieval, which no longer happens.
- **KI-95 is fixed:** "How many documents are in the Demo corpus folder?" answers 18 in every
  trial on both columns. 391ea0e answered 28, the library's count.
- **Not covered:** concept questions (the eval data home has no curated concept, so the concept
  lookups are tested only by unit tests); the several-matches answer (no case names an ambiguous
  title); the private library; billed models; answer latency.

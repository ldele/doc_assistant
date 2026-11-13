<!-- status: active · updated: 2026-11-13 (accepted by the user; H1: written after a scoped grill of six questions; revised the same day to the user's steer: model-neutral, no budget in the spec, a run of your own on your own model) · class: living -->

# Spec — The eval harness, reinforced for the chat work (ROADMAP 113, sessions H2 to H6)

> The executor brief. Target executor: **Claude Code**. This is session H1 of the plan's block *The
> harness and the benchmarks, reinforced* (`docs/plans/PLAN_2026-10-31_next-two-releases.md`,
> local). It reads, and must not re-litigate: ADR-066 (accepted 2026-11-13: seven read tools, lookups
> by rule and command, modes change only the framing), ADR-060 (the OpenAI-compatible adapter),
> `tests/eval/TESTING.md` (the verified-10 rule, the pinned judge, the local judge's agreement gate)
> and ROADMAP *What NOT to do* (no framework, no new dependency). **Accepted by the user on
> 2026-11-13; buildable.** H2 is next.
>
> **What this spec is not.** It says how a run works and what it measures, on any model. Which
> models a given run uses, and what it may cost, are a run's settings: recorded with the run, never
> fixed here. This release's own choices (its budget, its first panel) live in the plan.

## Goal

Before any chat change (A2 to A7) lands, the harness can measure what that change moves: on
conversations, on questions the library cannot answer, on passages that read as instructions, on
questions whose answer is a record, and on citation coverage. It measures the same way on whatever
model answers, compares only like with like, and says "no difference at this n" when that is the
honest answer. Anyone with a checkout can run it on their own model. Each of A2, A5, A6 and A7
names below the family and the scorer that gate it.

## Decisions — the grill of 2026-11-13

Six questions, each put to the user with a recommendation and taken as recommended; 1 and 2 were
restated the same day so that neither depends on a model or a budget (the user: how the eval runs
and what it measures should be specced, *regardless of the model used*).

| # | Fork | Resolution | Reason | Reopens if |
|---|---|---|---|---|
| 1 | What decides a gate, and the judge's place | **Deterministic scorers decide gates, on every model. A model judge is optional**, any provider, pinned per run, and every value it gives is marked *not calibrated* until its agreement with a person's grades is measured (H5) | A deterministic scorer means the same on every model and every run; a judge is a model too, and an unchecked one would decide what it cannot be trusted to | A judge is admitted for named uses by H5's agreement gate |
| 2 | Which models | **A panel is a list of columns** (provider, model, adapter) in a file the run names. Comparisons are made within a column only. A release's gates name their panel in the baseline. A panel should hold at least two model families, and one column per adapter in use, so a model trait and an adapter fault can be told apart | KI-36: the same prompt gave 81%, 36% and 14% citation coverage on three models; one model cannot show that a change behaves differently per model | — |
| 3 | Sizes, and where the cases live | **Eight new cases per family, all on the public demo corpus** (28 papers, `corpus_manifest.yaml`) plus invented hostile documents. The public single-turn set grows from 10 to 18 | Every number can be published and rebuilt from a fresh clone; nothing from the private library is reported; 40 cases is the user's reading budget | The demo corpus proves too clean for records or conversation cases (the first baseline shows ceiling scores on every column) |
| 4 | The statistic and the run size | **A paired bootstrap 95% interval of the per-case difference** (the case is the unit; its score is its mean over repeats). **`--repeat 3`** by default. A gate re-runs only the families that gate it | Cases are paired across runs, so a paired interval uses that; repeats steady the generation scorers; retrieval scorers do not vary | Intervals at n=8 stay too wide to separate anything visible by eye; then raise n before repeats |
| 5 | What passes a gate | **Per column: no measured drop on the gating family's main scorer, no measured drop on single-turn citation coverage, and no binary case that held before fails after** | Honest at n=8, where an interval rarely proves a drop; fixed floors would be corpus-tuned numbers (the robustness contract) | A real regression passes and is found later; then add a floor for that family |
| 6 | How the harness reaches the chat (found in the code: `rag_pipeline_adapter` calls `RAGPipeline` and skips `ChatController`) | **A second adapter drives the headless `ChatController`** that `apps/api` uses, turn by turn. The pipeline adapter stays | The router (`controller.py:331`), history and framing live in the controller; keeping the old adapter keeps old run ids comparable | The controller adapter reproduces the pipeline adapter's single-turn scores; then the pipeline adapter can retire |

## Hard constraints

- **Model-neutral.** No scorer, case or gate depends on which model answers. A case's expected
  behaviour is written once and holds for every column.
- **Each run records what it ran on**: for every column the provider, the model and its version or
  digest where the provider reports one, the adapter, the judge if any, the build, the corpus
  checksums and the bootstrap seed. A comparison between runs that differ in any of these says so
  (`compare_runs`' comparability check, extended to columns).
- **The cost is stated before the run, never assumed.** The runner prints, per column, the number
  of model calls it will make and whether the provider bills (ADR-060's *who bills*), and refuses a
  column that bills unless the run allows it explicitly (the credit guard, KI-4). The spec sets no
  budget; a run does.
- **No new dependency and no framework.** The bootstrap is the standard library's `random` with a
  recorded seed. Pydantic, pytest, DuckDB as today.
- **The harness asserts the app's contract and does not restate it.** Citation coverage calls
  `synthesis.audit_citations`; a records answer is detected from the route the controller recorded,
  not from the answer's wording (RG-012's lesson).
- **The harness stays extractable.** `eval/` modules import nothing from `doc_assistant` except in
  `adapters.py`, as today (Feature 5).
- **Nothing from the private library in a tracked file.** The private 35-case set stays a
  regression set and is never reported.
- **Results per family and per column, never one pooled mean.**

## Columns and panels

A **column** is one way of producing answers:

| Kind | What runs | What it measures |
|---|---|---|
| `app` | the app's own path: a provider and model through the app's adapter, the controller or the pipeline adapter | the app on that model. The only kind a gate reads |
| `replay` | a model given the prompts and passages an `app` column recorded, outside the app | the model on the app's prompt, **not a run of the app**: no provider adapter, no streaming, no token counts. Labelled so in every table, never a gate |

A **panel** is a YAML file listing columns. A run takes `--panel <file>`, or a single column from
`--provider` and `--model` as today. The repository ships an example panel; which panel a release's
gates use is recorded in that release's baseline, not here.

## Running it on your own model

Anyone with a checkout can run the public families on any model the app can talk to:

1. Download and ingest the public corpus (`scripts.download_corpus`, as `TESTING.md` says).
2. Run `scripts.run_eval` with their own `--provider` and `--model`, or their own panel file.
3. Read the report per family, and compare against the committed baselines: the comparison says
   whether the runs are comparable (same cases, corpus, build) and gives the interval within each
   column.

Results stay in the user's own `data/eval.duckdb`. The case file format is documented in
`TESTING.md` so a person can also write cases on their own library; those stay theirs and are
never shipped. **Parked, as the seam it grows through:** a run from inside the app. The runner is a
function in `src/doc_assistant/eval/`, so a later route can call it; nothing in this spec builds
one.

## The case families

Each family has eight new cases unless stated. Every case is drafted by the agent with the passage
and page it rests on, and counts only once the user accepts it on the review page (H3).

| Family | Tag | n | Shape | Main scorer | Adapter |
|---|---|---|---|---|---|
| Single-turn | `single` | 18 (10 kept + 8) | today's `EvalCase` | citation coverage; rank metrics for retrieval | both |
| Conversation | `conversation` | 8 | ordered turns, scored on the last: a follow-up that needs the earlier turn, a topic switch, a long one that passes a small model's context window | citation coverage and `contains_all` on the last turn | controller |
| Not in the library | `absent` | 8 | a question the corpus cannot answer; expected: the answer says so and cites nothing as support | refusal hold (binary) | controller |
| Hostile passage | `hostile` | 8 | a question whose retrieval reaches an invented document carrying an instruction; the hold is written per case (for example: the answer does not follow the instruction, does not reveal the prompt, cites only real support) | hostile hold (binary) | controller |
| Records | `records` | 8 | a question whose answer is a count, a folder's list, a document's metadata or a concept's definition; expected: the controller routes to records and the answer holds the expected values | route + `contains_all` | controller |

- **The long conversation** is written to pass the smallest context window among the columns a
  panel is likely to hold; the case records its length in tokens, so a column with a larger window
  simply does not reach the limit.
- **Hostile documents** are invented, public, pinned by checksum beside the manifest, and ingested
  only for the hostile family's runs (a separate data home), so they never touch the other
  families' retrieval.
- **Records cases** are written against ADR-066's rules and commands (A6), not against the MCP
  tools. The seven tools are tested deterministically in unit tests (ADR-066 *Enforced by*); the
  eval measures the chat's routing.
- **Records cases need a library state** (folders, curated concepts): a fixture script builds it on
  the demo corpus, as `backfill_demo_folder.py` does today, so the expected values are stable.

## Scorers

New in H2, each with a test on a planted right and a planted wrong answer:

| Scorer | Kind | Definition |
|---|---|---|
| `rank_mrr`, `rank_ndcg@10`, `recall@k` | deterministic | over `expected_citations` and the retrieved order; `recall@k` before and after the rerank, so the reranker's effect is visible |
| `citation_coverage` | deterministic | share of answer sentences carrying a valid citation, from `audit_citations` |
| `uncited_answer` | deterministic, binary | the answer has no valid citation at all |
| `refusal_hold` | deterministic, binary | for `absent`: the answer carries the app's not-found statement or no claim with a citation; the exact rule is the app's own not-found path, read in H2 |
| `hostile_hold` | deterministic, binary | for `hostile`: each case's written checks (strings that must not appear, citations that must not be given) |
| `route` | deterministic, binary | the controller recorded the records route |

Kept: `citation_overlap`, `contains_all`, `exact_match`, `figure_retrieval`.
`embedding_similarity` stays intra-model only. `llm_judge` is optional on any provider, pinned per
run, every value marked *not calibrated* until H5, and gates nothing.

## The statistic

- **Unit:** the case. A case's score in a run is its mean over the run's repeats.
- **Comparison of run A and run B, within one column:** for each case in both, the difference
  B − A; the mean difference with a **paired bootstrap 95% interval** (10,000 resamples of cases,
  seed recorded).
- **Wording:** an interval wholly above zero is *higher at this n*; wholly below, *a measured
  drop*; spanning zero, *no difference at this n*. Never "better" or "worse" alone.
- **Binary scorers** report counts (*7 of 8 hold*) and the list of cases that changed state, not an
  interval.
- **Across columns** the report shows each column's scores side by side and draws no interval: two
  models are not paired runs of one system.
- `compare_runs` keeps its comparability check and reports the interval beside the trial-mean std.
  A comparison across different case sets is refused, as today.

## The gates

A gate is run on each `app` column of the release's panel, against the baseline on the same
column. It passes when all three hold: no measured drop on the gating scorer; no measured drop on
single-turn citation coverage; no binary case that held before fails after.

| Session | Change | Gating families | Gating scorer |
|---|---|---|---|
| A2 | the prompt fence (S-9) | `hostile`, `single` | `hostile_hold`; citation coverage |
| A5 | context handling | `conversation`, `single` | citation coverage and `contains_all` on the last turn |
| A6 | lookups by rule and command | `records`, `absent`, `single` | `route` and `contains_all`; `refusal_hold` |
| A7 | chat modes | every family, once per built-in mode | citation coverage |

A gate that stops early reports what it did not run as *not run*.

## The user's reading

- **H3: 40 cases** to accept, edit or reject (8 single-turn, 8 per new family), on a review page
  read back after the first save (memory *input-pages-check-the-first-save*). The user may cut a
  family before H3 starts; the gate table then names what is left ungated.
- **H5: 30 answers** to grade blind on the judge's 1–5 rubric, sampled stratified: six per family,
  spread across the panel's `app` columns, model and family hidden. The interval this gives on the
  judge's agreement is stated with it; 30 is the user's budget, not a size chosen for a target
  width.

## Run size

A run is cases × columns × repeats model turns, plus one judge call per answer if a judge is on.
The runner prints that count per column before it starts, with whether each provider bills. The
full baseline records its real duration so the next run can be planned.

## Work items

- **H2 — the instrument.** The controller adapter (records the route, the turns and the prompt it
  sent); panel files and `--panel`; the pre-run statement of calls and billing; conversation and
  refusal case shapes in `cases.py`; the six scorers; the paired interval in `report.py`;
  per-family and per-column reporting; the case format and *Running it on your own model* in
  `TESTING.md`. **Done when:** each new scorer has its planted-right and planted-wrong test, the
  committed public baseline reproduces unchanged on the old scorers through the pipeline adapter, a
  re-run of an old run id still compares, and the controller adapter's single-turn scores are
  recorded beside the pipeline adapter's.
- **H3 — the cases.** The hostile documents and their checksums; the records fixture; the 40
  drafts on the review page; the accepted ones in `cases.public.yaml` (single) and one file per new
  family. **Done when:** every family has its accepted cases, the public ones are pinned, and no
  private text is in a tracked file.
- **H4 — the baseline.** The release's panel over every family on the 0.7.3 build, with RG-035's
  first re-run, row 69 (`CANDIDATE_K`, retrieval only) and the reranker on/off ablation. **Done
  when:** dated baselines in `tests/eval/baselines/` state n, panel, run time and what they cannot
  show.
- **H5 — the judge, checked.** As the plan states, with the sample above. The agreement gate is
  model-neutral: any judge model can be put through it.
- **H6 — the write-up.** As the plan states, with the per-column tables and the steps to run it on
  your own model, rehearsed from a fresh clone.

## Out of scope

RG-007, RG-015, RG-035's second and third re-runs, `rigor_gate.py` (row 68), a run from inside the
app (parked above), the MCP tools' own correctness (unit tests, ADR-066), and row 15 (the harness
as its own repository).

## Open questions

| # | Question | Status | Owner / where |
|---|---|---|---|
| 1 | What decides a gate; the judge's place | resolved | Decisions 1 |
| 2 | Which models | resolved | Decisions 2 |
| 3 | Sizes and corpus | resolved | Decisions 3 |
| 4 | Statistic and repeats | resolved | Decisions 4 |
| 5 | Gate rule | resolved | Decisions 5 |
| 6 | The path to the chat | resolved | Decisions 6 |
| 7 | The exact not-found rule `refusal_hold` reads | parked | H2, read from the app's not-found path |
| 8 | A run from inside the app | parked | after H6; the runner function is the seam |

## Docs to touch when this lands

`tests/eval/TESTING.md` (scorers, the statistic, panels, the controller adapter, the case format,
running it on your own model), `evals/README.md` and `docs/OVERVIEW.md` §4 (H6), ROADMAP row 113's
status, `docs/how-answers-work.md` if the route record changes what a turn stores.

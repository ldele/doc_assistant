<!-- status: active · updated: 2026-11-16 (H5: two local judges and a Haiku subagent judge against one person's blind grades on 30 answers; none passes the gate) · class: baseline -->

# The judge, checked against a person (H5)

**No judge measured agrees with a person well enough to be trusted on its own, and a stronger
model did no better.** On 29 answers
graded blind by the author, `qwen2.5:7b` orders the answers the same way as the person in 82% of
the pairs the person separates (106 of 130). `llama3.1:8b` does in 63% (82 of 130). The gate asks for
90%. Their rank correlations with the person, 0.30 and 0.34, have 95% intervals that include zero.
Both judges stay in shadow: they may print a grade, and every grade stays marked *not calibrated*.
Claude Haiku, run afterwards as a judge (see below), orders 81% of the pairs as the person does,
about what `qwen2.5:7b` does.

These are one person's grades on 30 answers. They say how far these two judges follow this
person on this rubric, not how good any judge is in general.

## What the judge measures, and what it does not

The rubric (`scorers._JUDGE_PROMPT`) grades an answer **against a written reference answer**, on
faithfulness (its claims are in the reference), relevance and completeness, 1 to 5. It never sees
the passages the answer cites. **It cannot say whether an answer is grounded in its sources**, and
grading on the same rubric, the person could not either (the author, 2026-11-16). Grounding is
measured without a judge: citation coverage, `no_invention` and the reviewer.

## How it was run

- **Sample:** 30 answers from the H4 baseline runs (`h4_chat_baseline_2026-11-15.md`), six per
  family, two per column (`llama-8b`, `qwen-7b`, `qwen-7b-v1`), one trial drawn per answer, seed
  20261115. Nothing was generated for H5. Records cases have no written reference, so their
  reference was written from the eval library's own record (`reference_from_record`).
- **The person's grades:** the author graded all 30 on a review page that hid the model and the
  family, against the reference only. The three records references first drawn from the expected
  strings were rewritten from the library's record and those answers were graded again. Item a01
  has a relevance grade only, so the means use 29 answers (30 for relevance).
- **The judges:** `ollama / llama3.1:8b` and `ollama / qwen2.5:7b` through `LLMJudgeScorer`, at
  temperature 0, three repeats each, on the same reference and answer text. Repeats agreed: one
  answer of 30 got different grades across qwen's repeats, and none across llama's. **No reference judge:** the
  Claude Haiku 4.5 run was refused for a lack of API credit, and the author chose to go on
  with the local judges only.
- **The statistic:** `doc_assistant.eval.judge_agreement` (built in H5). An answer's grade is the mean
  of its three dimensions, and a judge's grade is also its mean over the repeats. Decision agreement
  counts the pairs the person grades at least 1 point apart that the judge orders the same way, and a
  judge tie counts as a miss. The intervals are 95% bootstraps over answers (seed 20261113, 10,000
  resamples).
- **Record:** `tests/eval/judge_check/h5_2026-11-16/` holds the sample, the person's grades and both
  judges' grades.

## Results

| Judge | Dimensions | n | Decision agreement | Spearman rho [95% interval] | Mean absolute deviation | Bias (judge − person) | Gate |
|---|---|---|---|---|---|---|---|
| `qwen2.5:7b` | all three | 29 | 106/130 = **0.82** | 0.30 [−0.17, +0.66] | 0.64 | −0.27 | fails |
| | faithfulness | 29 | 142/217 = 0.65 | 0.48 [+0.07, +0.76] | 0.76 | −0.48 | |
| | relevance | 30 | 89/235 = 0.38 | 0.36 [−0.04, +0.67] | 1.19 | +0.61 | |
| | completeness | 29 | 112/290 = 0.39 | 0.28 [−0.16, +0.62] | 1.31 | −0.97 | |
| Claude Haiku (subagent, one run) | all three | 29 | 105/130 = **0.81** | 0.30 [−0.15, +0.63] | 0.72 | −0.15 | fails |
| | faithfulness | 29 | 119/217 = 0.55 | 0.27 [−0.15, +0.63] | 1.21 | −0.38 | |
| | relevance | 30 | 105/235 = 0.45 | 0.24 [−0.26, +0.66] | 1.07 | +0.67 | |
| | completeness | 29 | 161/290 = 0.56 | 0.35 [−0.05, +0.66] | 1.14 | −0.79 | |
| `llama3.1:8b` | all three | 29 | 82/130 = **0.63** | 0.34 [−0.08, +0.68] | 0.74 | +0.48 | fails |
| | faithfulness | 29 | 111/217 = 0.51 | 0.30 [−0.16, +0.68] | 0.86 | +0.10 | |
| | relevance | 30 | 0/235 = 0.00 | undefined (every grade 5) | 1.27 | +1.27 | |
| | completeness | 29 | 140/290 = 0.48 | 0.29 [−0.14, +0.64] | 0.93 | +0.03 | |

The gate (`TESTING.md`, kept as proposed): decision agreement at least 0.9, and mean absolute
deviation at most 0.5. **The judges agree with each other more than with the person:** Haiku
against `qwen2.5:7b` at rho 0.68, against `llama3.1:8b` 0.51, the two local judges 0.46. The
person gave most answers a 4 on faithfulness and relevance, so the person's grades separate few
pairs of answers on those two scales. That alone holds every rank correlation down.

**The answers the person found wrong.** The person graded four answers at or below 2.7 of 5,
all records answers (a05, a06, a09 and a23: the missing-title, folder-count, title and
folder-list questions).
`qwen2.5:7b` graded all four at or below 1.7, and two more as low that the person did not (a22,
hostile, and a10, absent). `llama3.1:8b` graded three of the four at 2.3 and gave a23 a 5.
`llama3.1:8b` gave every answer a relevance of 5.

**The Haiku judge (added the same day, at the author's request).** The API run was refused, so
Claude Haiku graded the 30 answers as Claude Code subagents: one isolated agent per answer,
given the same rendered judge prompt and asked for the same JSON, one run each. This is not the
pinned API judge: neither temperature nor model version is under the harness's control. Haiku
graded the three wrong records answers low (a05, a06, a09) and a22 low too, which the person
rated 3.7. It gave a23 a 2.7, where the person gave 1.7. The grades are in
`judge_check/h5_2026-11-16/judge_subagent_haiku.json`.

## A stronger model answering the same questions

The author also asked what a better model would answer. Claude Haiku (one Claude Code subagent per
question) was given the app's own answer prompt over the passages the app retrieves, for the 24
sampled questions a model answers. The six records questions are answered by the app's lookup, not
a model. Scored with the app's own checks:

| The local setup that answered | Questions | Citation coverage, local | Citation coverage, Haiku on the same passages |
|---|---|---|---|
| `llama3.1:8b` | 8 | 0.73 | 0.68 |
| `qwen2.5:7b` | 8 | 0.29 | 0.76 |
| `qwen2.5:7b`, cut prompt (KI-94) | 8 | 0.13 | 0.77 |

- **Haiku cites about as much as `llama3.1:8b` and far more than `qwen2.5:7b`.** On questions the
  library cannot answer, it says so and then cites what the library does hold (coverage 0.25 to
  0.75, where the local answers cited nothing). That is the behaviour the author asked for in H3.
- **Every Haiku answer is cited** (18 of 18 that the check scores). None invented an absent fact.
  5 of 6 held against the attack: on a02 it refused the planted instruction, then quoted it, code
  included, in a note to the caller, which the check counts as a break.
- **Three conversation answers rest on this setup, not the model.** The earlier answers were not
  kept, so the rewritten question lost context: a04 and a12 retrieved the wrong passages, and
  a21's question carried a wrong premise. Haiku said what was missing or corrected the premise.
- **What it cannot show:** whether Haiku's citations support their sentences; anything at more than
  8 questions per setup. The answers: `judge_check/h5_2026-11-16/haiku_answers.json`.

## Reading

- **No judge decides anything.** Deterministic scorers keep deciding the gates (spec decision 1),
  and `llm_judge` stays optional and marked *not calibrated*.
- **A stronger judge is not the fix here.** Haiku's agreement with the person is about qwen's,
  and the three judges agree with each other more than with the person. What separates them from
  the person is probably the rubric itself (against a reference, with no sources) and the
  person's own judgement beyond it, such as "too short, tell more". A better model grading on the
  same rubric does not close that gap.
- **The closest to a use:** `qwen2.5:7b` as a second look that flags an answer which contradicts its
  reference. It caught the four the person found wrong, and two false alarms. Four positives are too
  few to admit it for that use. And the four were records answers, which a check of the looked-up
  value catches without a model (KI-95, session A6).
- **What would change this:** a judge that sees the cited passages, which is a different rubric,
  measured again against a person. Or a larger sample with more answers the person finds wrong. The
  author's priorities put neither ahead of retrieval and the library work (2026-11-16).
- **What it cannot show:** the pinned API judge's agreement (only the subagent run); agreement with anyone other than the
  author; anything about grounding.

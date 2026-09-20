<!-- status: active · updated: 2026-09-20 (first measurement) · class: baseline -->

# `is_a` proposals from a shared head — what the rule finds (ROADMAP 51)

The concept→concept spine has never existed: **0 `is_a` edges against 357 concepts** (ADR-045),
because nothing proposed one. This records what the deterministic proposer
(`knowledge/isa_propose.py`, `scripts/propose_isa.py`) finds on the live vocabulary, and the two
shapes it refuses. Row 53's merge baseline supplied the input observation: 21 of its 34
near-duplicate pairs were *narrower* terms rather than duplicates.

The counts are measured. The **verdict column is the agent's first reading**, not the user's hand
score — like the merge baseline, it is there to be confirmed or overturned in the app's review pane.

## Environment / corpus

- **Data home:** `data/library.db` on this box, 104 documents, 357 text-bearing concepts
  (`kind="concept"`), 13 of them on the graph (`graph_include`).
- **Method:** `head_suffix_candidates` over `(id, label)` for every concept. A label is tokenised
  with the keyword tokenizer, so `cross-encoder` stays one token; a label whose tokens *end with*
  another concept's whole label is proposed as narrower than it.
- **Cost:** $0 — no model, no network. Nothing was written: the numbers below are a dry run
  (`python -m scripts.propose_isa`).

## Result

| | |
|---|---|
| concepts read | 357 |
| **candidates** | **27** |
| of which touch the graph vocabulary (either end) | **1** (`passage re-ranking` → `re-ranking`) |
| candidates added by also matching aliases | +10 — **rejected**, see below |
| candidates removed by "nearest parent only" | 0 on this corpus (no chains exist yet) |

### The candidates

`kind` = the agent's first reading: **ok** = a defensible `is_a`; **fragment** = the broader side is
not a concept, it is a word left over from extraction; **check** = the shape is right but the
meaning needs a person.

| narrower | broader | shared head | graph | first reading |
|---|---|---|---|---|
| `beta oscillations` | `oscillations` | oscillations | no | ok |
| `human pose` | `pose` | pose | no | ok |
| `multi-animal pose` | `pose` | pose | no | ok |
| `nuclear speckles` | `speckles` | speckles | no | ok |
| `passage re-ranking` | `re-ranking` | re-ranking | **yes** | ok |
| `passage retriever` | `retriever` | retriever | no | ok |
| `negative passages` | `passages` | passages | no | ok |
| `relevant passages` | `passages` | passages | no | ok |
| `positive passage` | `passage` | passage | no | ok |
| `political psychology` | `psychology` | psychology | no | ok |
| `cardiologist-level arrhythmia` | `arrhythmia` | arrhythmia | no | ok |
| `dynamic polarization` | `polarization` | polarization | no | ok |
| `negative perturbation` | `perturbation` | perturbation | no | ok |
| `community summaries` | `summaries` | summaries | no | ok |
| `events behavior analysis` | `behavior analysis` | behavior analysis | no | ok |
| `program behavior analysis` | `behavior analysis` | behavior analysis | no | ok |
| `superanimal memory replay` | `memory replay` | memory replay | no | ok |
| `learning abstractions` | `abstractions` | abstractions | no | check — `abstractions` is one of the three concepts the document-level split flagged as a possible homograph (merge baseline) |
| `benchmark saturation` | `saturation` | saturation | no | check — `saturation` here is the benchmark sense, not the chemical one |
| `benchmarks plateau` | `plateau` | plateau | no | check — `plateau` is a phrase fragment, `benchmarks plateau` is a sentence |
| `training recipe` | `recipe` | recipe | no | check — a metaphor, not a taxonomy |
| `ai usage cards` | `cards` | cards | no | check |
| `memory bank` | `bank` | bank | no | fragment |
| `c-mtp unlabeled` | `unlabeled` | unlabeled | no | fragment |
| `vis pattern recog` | `recog` | recog | no | fragment |
| `deep learning animal` | `learning animal` | learning animal | no | fragment |
| `systems biology neuroscience` | `biology neuroscience` | biology neuroscience | no | fragment |

First reading: **17 ok · 5 check · 5 fragment**. Every fragment comes from a broader side that was
never a concept — which is ADR-052's problem (row 93), not this rule's: the review's reject is the
answer, and rejecting one costs a click.

## What the rule refuses, and why

- **A shared prefix is not hyponymy.** `self-sorting memory` is a kind of memory, not a kind of
  `self-sorting`; `saturation index` is not a kind of `saturation`. Both shapes sit in the merge
  baseline's "narrower" column at ≥ 0.85, and both would be wrong as an edge in either direction.
  Matching the front of a label instead of the end is not a stricter or looser rule — it is a
  different claim, and the wrong one.
- **Aliases are not word forms.** Matching each concept's aliases as well as its label adds 10
  candidates here, and reading them is enough: `ai benchmarks` → `benchmarks plateau`,
  `self-sorting memory` → `memory bank`, `dense retrieval` → `retriever`, `hard negatives` →
  `passages`. An alias is a *different phrase*, so its last token is not the concept's head. Labels
  only.
- **A cosine pair whose direction is not lexical is not proposed at all.** `apoptosis ~ cell death`
  (0.864) is a narrower/broader judgement no lexical rule can direct. Those stay on the review page
  for the user's hand score (ROADMAP 53 (3)), which is where the merge baseline left them.

## What this does not claim

- **Not quality.** 27 candidates with a 17/5/5 first reading is not an accuracy measurement, and
  the pass writes `origin="proposed"` rows precisely because it is not one (ADR-028 D8, RG-015).
- **Not coverage.** A shared head is the only signal here. Concepts that are narrower without
  sharing a word (`apoptosis` / `cell death`) are invisible to it, and so is everything in the 344
  concepts that never repeat a head.
- **Not a spine for the graph.** One of 27 candidates touches the 13 graph concepts. The vocabulary
  that has shared heads is the keyword-derived one, so an `is_a` layer over *the graph* needs
  either more graph concepts (row 54) or a different source.

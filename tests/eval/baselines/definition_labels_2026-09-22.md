<!-- status: active · updated: 2026-09-22 (first user labels — ROADMAP 93, ADR-053) · class: baseline -->

# How often a found sentence can define a concept — the user's labels (ROADMAP 93, ADR-053)

**One in five of the extractor's candidates can carry a definition; the grade sorts them, but
`strong` is right about three times in five.** The user labelled all 69 candidates for the 19 priority
concepts on the *Definition Candidate Review* page, with the machine's rating hidden. This is the
yardstick for every refinement of `knowledge/definitions.py`. It also **supersedes the agent's
"first reading"** in `definition_sources_2026-09-21.md` §2, which was too generous (see §3).

## Environment / method

- **Candidates:** the 69 that `find_passages` returned on 2026-09-21 for the 19 priority concepts
  (104 documents), each shown with the text before and after it. They are unchanged since.
- **Labeller:** the user, one pass, rating hidden. Six labels: *defines* · *needs context* · *too
  narrow* · *uses the term* · *claim* · *other*. **Usable** means defines + needs context: a sentence
  a definition can be built from.
- **Where the labels live:** the page's store (artifact db, collection `labels`, one document per
  candidate, keyed `sha1(concept_id|provenance_key)[:20]`). Read back with ArtifactData, 69/69.
  A first pass in the Claude app's Artifacts pane was never stored (claude-skills T-004); this is the
  second pass, done in a view that saved.
- **Cost:** $0. No model, nothing written to the library.

## 1 · Usable by grade and by form

| | usable | 95% interval (Wilson) |
|---|---|---|
| grade **strong** | **8 / 13** (62%) | 36–82% |
| grade some | 4 / 27 (15%) | 6–32% |
| grade thin | 2 / 29 (7%) | 2–22% |
| form coined | 2 / 4 | 15–85% |
| form named | 1 / 3 | 6–79% |
| form defining | 8 / 17 (47%) | 26–69% |
| form **first mention** | **3 / 45** (7%) | 2–18% |
| all | 14 / 69 (20%) | 12–31% |

Labels overall: uses 32 · claim 19 · defines 9 · needs context 5 · other 4 · too narrow **0**.

## 2 · By concept

- **The top candidate is usable for 7 of 19** (abstractions, contrastive learning, cross-encoder,
  dbs, hard negatives, pddl, specter).
- **A usable one sits lower down for 3 more** (dense retrieval, knowledge distillation, passage
  retrieval). The ranking put a claim first: *"Knowledge distillation is an excellent technique for
  model compression"* above two real definitions.
- **None is usable for 9** (BM25, beta, cre, din, ntsr1, re-ranking, retrieval-augmented generation,
  viral, virus). For four of them (beta, cre, din, viral), the user's notes blame the label, not
  the sentence (§4).

## 3 · Correction to the 2026-09-21 first reading

The first reading called the top candidate "clean" for all 11 concepts whose top grade is
`strong`. By the user's labels, **6 of those 11** are usable. The other five are dense retrieval,
knowledge distillation, ntsr1, passage retrieval and retrieval-augmented generation (claims, or a
use of the term). The first reading was one agent looking at the extractor's own output; it does
not count as evidence.

## 4 · What the labels and notes say

- **First mentions use a term; they don't define it.** 28 of 45 are *uses*. A first mention is an
  example of how the library uses a word — ADR-053's second layer — not a definition candidate.
- **Defines vs claim was hard to call** (the user's words). 7 of the 17 definition-shaped sentences
  were labelled *claim*, and some sentences are both. *"Dense retrieval …, the method of
  retrieving documents using semantic embedding similarities, has been shown successful …"* was
  labelled *defines*, with the note "also a claim". The labels treat the two as exclusive, and they
  are not.
- **Context, every time** (the user's summary). *"… an emerging area of research"* holds only as of
  its year. A single sentence is evidence about a meaning, not the definition.
- **The label is shorter than the concept.** From the notes:
  - `beta` means *beta oscillations* here and a protein form elsewhere.
  - `viral` means *viral vector*.
  - `cre` means *Cre recombinase*, "only the most famous" of several.
  - `din` is `dIN` (an interneuron), not `Din` (the text join in "vitamin Din").

  "Important of being case-sensitive." Concept labels are stored lower-cased, which loses this.
- **Other** was used only for a different meaning (4 cards: beta secretase, kinase-beta, Din,
  viral vector). **Too narrow** was never used.

## 5 · Probe: can the library tell an abbreviation or a fragment from its own text?

Read-only, over each label's mentions in body text (bibliography cut): how it is written, whether it
appears spelled out (Schwartz & Hearst 2003, *long form (SF)* and *SF (long form)*), and the word
that most often sits beside it.

| label | not lower-case | spelled out in the library | most frequent neighbour |
|---|---|---|---|
| din | 100% — `dIN` 45, `Din` 1 | — | *-cin* 17%, *ascending* 17% (lower-cased by the probe) |
| ntsr1 | 100% — `Ntsr1` | — | not measured: the text writes `Ntsr1- Cre`, which the pattern skips |
| BM25 | 99% | — | *scores* 5% |
| specter | 97% | — (coined in a sentence, not in brackets) | — |
| cre | 95% — `Cre` 292, `CRE` 1 | `CRE` = *cAMP response element* (1 doc) — **another meaning** | *BAC-* 21%, *driver* 14% |
| dbs | 91% | **deep brain stimulation** (2 docs) | *pHyp-* 31%, *Lead-* 22% |
| retrieval-augmented generation | 59% (title case, in titles) | — | *for* 47% (probably the RAG paper's title) |
| pddl | 30% (one of its two papers is extracted in lower case) | **planning domain definition language** (1 doc) | — |
| viral | 24% | — | *tools* 16%, *genome* 9%, *tracers* 6% |
| beta | 9% | one junk match (a list of frequency bands) | *activity* 26%, *oscillations* 12% |
| knowledge distillation, dense retrieval, re-ranking, passage retrieval, contrastive learning, abstractions, virus, hard negatives, cross-encoder | 0–20% | — | no content word above 23% |

What it shows:

- **The case profile separates abbreviations and names from ordinary words** (91–100% against
  0–20%). The exception is PDDL, which one of its papers renders in lower case.
- **Spelling out** found an expansion for two labels (dbs, pddl), and a different meaning of the
  same letters for a third (`CRE` against `Cre`). It found nothing for dIN, SPECTER, BM25 or Ntsr1.
- **The neighbour share alone does not flag a fragment.** `beta` is followed by *activity*,
  *oscillations* or *network* in 44% of its mentions, and `viral` by *tools*, *genome* or *tracers*
  in 31%. But `dense retrieval` is followed by *techniques* 23% of the time, more often than `viral`
  is followed by *tools*. The "stands alone" count of a nested label (the pairing work, ROADMAP 53)
  is the better candidate, and it was not run here.
- **Not fitted.** No threshold was set, and no rule was tried on concepts outside these 19. The
  Schwartz & Hearst port produced one junk expansion (`beta`).

## 6 · Re-measured after first mentions left the candidates (same day)

The user's decision (ADR-053 amendment): a first mention is *how your library uses it*, not a
candidate. `find_passages` re-run on the 19 through the keyword index; every candidate it returns
matched against the labels above.

| | before | after |
|---|---|---|
| candidates | 69 | **24** — all already labelled, none new |
| usable | 14 (20%) | **11 (46%)** |
| top candidate usable | 7 of 19 | 7 of 19 |
| concepts with no candidate | 0 | 5 (cre, din, re-ranking, viral, virus). None of their 15 first mentions was usable |

The 3 usable first mentions (dense retrieval, passage retrieval, pddl) are still on screen: each
is among that concept's usage examples. The usage route reads at most six documents per concept:
91 ms for `dbs`, 158 ms for `beta` in the running app. One of `beta`'s three examples is a
pseudo-code block from a textbook, which the sentence filters do not catch.

## What this does not claim

- One labeller, one pass, no retest; the labeller flagged their own consistency as uncertain. With
  13 strong candidates, the 62% has an interval of 36–82%.
- 19 concepts chosen for being on the graph or ambiguous, not a sample of the 357.
- The probe's signals are measurements on 19 labels, not a validated classifier.

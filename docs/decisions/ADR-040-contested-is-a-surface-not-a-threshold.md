<!-- status: active · updated: 2026-08-02 · class: append-only -->

# ADR-040 — `contested` saturates because it is a binary label on a continuous quantity, not because its threshold is too low

- **Status:** proposed — **option 5 has since been executed and it invalidates the input; the
  surfacing choice is now blocked behind a Node-B fix (KI-33)**. See *Update* below.
- **Date:** 2026-08-02
- **Deciders:** user + Claude Code

## Context

The v0.4.0 walkthrough recorded the integrity strip — the product thesis — reading closer to noise
than signal: **53.3% of assessed chunks marked `contested`** against 3.9% `corroborated`, and 8 of
10 sources in a live turn marked *contested* on "what is dense passage retrieval vs BM25", which is
not a controversy.

Two records agreed on the cause and the cure. **RG-019** (2026-07-19): *"`coverage="contested"`
triggers on `nc >= 1` (an implicit, unnamed threshold)"*, prescribing *"a named floor (min disputing
docs and/or an agreement-ratio band — the `MIN_DATED_DOCS_PER_SIDE` pattern)"*. **ADR-027** shipped
the always-on strip anyway, recording that RG-019 "should land with or shortly after E2, else the
strip ships saturated". Neither had been measured.

**It has now been, and the shared diagnosis is wrong**
([`contested_density_2026-08-02.md`](../../tests/eval/baselines/contested_density_2026-08-02.md),
$0, live corpus, every counterfactual re-projected onto real chunks):

| lever | effect on density |
|---|---|
| `nc >= 2` (the prescribed fix) | 53.3% → **53.0%** |
| `nc >= 3` | → 52.9% |
| `agreement_ratio < 0.70` | → 53.2% |
| chunk rule "majority of claims contested" | → 53.2% |

Only **1 of 7** contested nodes has `nc == 1`. The other six are the corpus's core vocabulary — BM25,
dense retrieval, passage retrieval, contrastive learning, re-ranking, hard negatives — each with
genuine two-sided stance across **5–11 documents** and agreement ratios clustered in **0.545–0.714**.
`contested` is not misfiring. It is firing correctly on ordinary scholarly disagreement, and the UI
presents that as cautionary.

Three structural facts constrain any fix. **(a)** Every threshold with real effect falls inside a
band holding seven data points, on a 13-concept `graph_include`-scoped vocabulary — fitting it is
the failure KI-19 exists to forbid, and the robustness contract forbids corpus-tuned constants
outright. **(b)** 89% of assessed chunks carry exactly one claim, so the chunk-level "any claim"
rule has no headroom: any/majority/all differ by 0.3 points. **(c)** The whole signal rests on
Node-B stance extracted by a local 8B model; a disagreement-biased extractor would produce this
picture identically, and nothing measured so far separates the two.

### The confound that reframes all of it

Every graph concept carries exactly one ANZSRC parent field (ADR-028) — **13/13 placed**, so this
join is total rather than a sample:

| parent field | concepts | contested |
|---|---|---|
| Machine learning | 6 | **4** |
| Data management and data science | 3 | **3** |
| Artificial intelligence | 1 | 0 |
| Neurosciences | 1 | 0 |
| Medicinal and biomolecular chemistry | 1 | 0 |
| Biochemistry and cell biology | 1 | 0 |

**7 of 9 concepts in the two IR/ML fields are contested; 0 of 4 outside them.** `cre`, `dbs`,
`ntsr1` and `pddl` are not uncontested because they are settled — they have **one source each**. The
marker is tracking *how densely the corpus covers a field*, not whether a claim is disputed. Every
lever above operates on a per-concept rate whose dominant term is a variable that rate does not
contain, which is why none of them separates signal from density.

**The literature agrees, and none of it supports a cut point** (checked 2026-08-02).
`agreement_ratio` is raw percent agreement, the statistic Cohen's κ and Krippendorff's α exist to
replace, and Landis & Koch's bands are the standard cautionary tale about exactly this kind of
threshold. The nearest real discipline — meta-analysis, which owns "do sources disagree" via
Cochran's Q and Higgins' I² — offered its 25/50/75% bands as tentative, is cautioned against
mechanical application by the Cochrane Handbook, and finds I² non-discriminative for prevalence
meta-analyses, the closest analogue here; practice reports it with τ² and a prediction interval.
Citation-polarity corpora put neutral citations above 60% with contrasting/negative the **rarest**
class, against ~45% opposing sources per node here — independent evidence for (c). Full citations:
the baseline's literature section.

## Options

1. **A named floor anyway** (`nc >= 2`, or an agreement band). *Trade-off:* it is what RG-019 and
   ADR-027 both promise, and it closes the item cheaply. But it is **measured inert** at 0.3 points,
   and the one setting that does move density (`agreement < 0.60` → 27%) is a number chosen by
   looking at seven values. It would let the project record a fix that changes nothing while
   spending its corpus-tuning budget.
2. **Gate the unsourced case only** — a node with `n_supporting_sources == 0` is not `contested`.
   *Trade-off:* narrow and honest. Coverage is decided contested-first, so the unique-source
   neutrality rule (Decision 4 — "a sole source is never contested") never gets to judge
   `knowledge distillation` (`ns=0, nc=1, agreement=0.000`), the one node that is wrong on its face.
   Structural, not tunable — it introduces no constant. But it fixes **one node** and leaves 53% at 53%.
3. **Replace the binary chip with the quantity** — surface `agreement_ratio` directly ("5 of 11
   sources agree") instead of a contested/not chip. *Trade-off:* it is the honest shape of the data
   and needs no threshold at all, so nothing is corpus-fitted. It also finally uses a value the code
   has computed and ignored since the 7d seam. Costs a UI change in the source-evaluation strip
   (ADR-027 D3) and gives up the at-a-glance binary the chip provided.
4. **Re-frame the label, keep the mechanics** — "mixed stance" / "discussed from multiple angles"
   rather than "contested in corpus". *Trade-off:* one string, no logic, and it removes the false
   alarm — 53% of sources being *discussed from several angles* in a research corpus is unremarkable
   rather than alarming. But it treats a signal-design problem as a copy problem, and a label that is
   unremarkable at 53% is also uninformative at 53%.
5. **Validate the stance extractor before touching the surface at all.** *Trade-off:* it is the
   cheapest measurement left (sample Node-B stance calls, check them by hand) and it gates the
   meaning of every option above — if `llama3.1:8b` over-reports disagreement, then 3 and 4 dress up
   a broken input and 1 and 2 tune noise. But it delivers no user-visible improvement by itself.
6. **Score contestedness against the parent field's base rate** — every concept already has exactly
   one ANZSRC parent (ADR-028), so "disputed" becomes *disputed relative to what is normal for this
   field* rather than against a global constant. *Trade-off:* it is the only option that addresses
   the confound rather than working around it, and it removes the tunable entirely — the reference
   class is derived from data, which is what the robustness contract asks for and what the
   `MIN_DATED_DOCS_PER_SIDE` pattern cannot give. It is also the established method for this shape
   of data: partial pooling / empirical-Bayes shrinkage of each concept's rate toward its field
   mean, weighted by evidence, where James–Stein dominates raw group means for k≥3. **And this
   project has already made this exact argument once** — ADR-006 rejected absolute keyword frequency
   for contrastive termhood scored against a background distribution; a rate is only meaningful
   against a reference class. *But:* **it is not estimable today.** Four of the six fields hold one
   concept each, and a field base rate cannot be estimated from n=1. Option 6 is the right target
   and currently a design, not a computation.

## Decision

**Not taken — this ADR exists to stop the wrong fix from being applied on the strength of two
records that agree with each other and not with the data.** What the measurement settles is narrow
and firm: **option 1 is dead**, and RG-019's prescription should not be implemented as written.

**Recommended sequence, for the user's call:** **5 → 2 → 3, with 6 as the target shape.** Validate
the stance extractor first, because every other option's value depends on the answer and it is the
cheapest thing left to measure — and the citation-polarity base rates now make a biased extractor
the leading hypothesis rather than a possibility. Land **2** regardless: it is structural, introduces
no constant, and `ns=0, nc=1` is indefensible under any surfacing choice. Prefer **3** over **4** if
the extractor checks out — the data is continuous, the code already computes the continuous value,
and a chip firing on more than half of everything cannot be made informative by renaming it.

**Option 6 is where this should end up**, and saying so now is the point of recording it: it is the
only option that addresses the confound rather than routing around it, and it is the one the
literature and ADR-006's own precedent both point at. It is deferred rather than rejected **for a
stated, falsifiable reason** — four of six fields hold one concept, so no field base rate is
estimable yet. That reason expires as `graph_include` grows (344 curated concepts sit outside the
graph today, ADR-018), so option 6 should be re-tested then rather than rediscovered.

**A framing worth keeping even if every option is rejected:** *insufficient evidence is a state, not
a low score.* The schema already encodes it — `unique` is sole-source, held NEUTRAL, never
down-weighted (Decision 4) — and contested-first precedence is what steals it. Option 2 is the first
instance of that principle, not a one-node patch.

## Update (same day) — option 5 ran, and it removes the question

Option 5 was executed rather than deferred
([`node_b_stance_validity_2026-08-02.md`](../../tests/eval/baselines/node_b_stance_validity_2026-08-02.md),
$0, `llama3.1:8b`, temperature 0.0, shipped settings). It did not merely find the extractor
suspect — it found the signal is **not a measurement of the corpus at all**, and filed **KI-33**.

**Two structural facts from the code.** `build_messages` composes the whole prompt from concept
labels and a numbered pair list: **the model never sees the document** it is asked to judge "the
apparent framing" of. And `POLARITIES` has **no neutral option** — every co-present pair must take
one of four labels, two of which are opposing, on a vocabulary where `refines` ("improves") is
supporting and `supersedes` ("replaces") is opposing while the prompt's own example verb is
*"improves on"*.

**And the controlled experiment.** One document, same 7 present concepts, same 17 pairs, same model,
temperature 0.0, **varying only the target pair's index in the list**: four distinct verdicts
(`supports` at 0 and 2, `supersedes` at 4 and 12, `contradicts` at 8, `refines` at 16) — **crossing
the supporting/opposing boundary**. Replaying the real prompts reproduces **5/5** recorded stances,
so this is the deterministic behaviour of the shipped pipeline, not sampling noise. A first
hypothesis (that the co-present concept *set* drove it) was refuted en route: 9 varied contexts at
index 0 gave `supports` 9/9.

**This subsumes the confound above rather than competing with it.** Pair-list length scales with a
document's concept count, so dense-coverage documents produce long lists → deep indices → opposing
stances → `contested`, while sparse documents produce one-pair prompts → index 0 → `supports`. The
parent-field table (7/9 in two IR fields, 0/4 outside) is the *downstream shadow* of a
generation-position artifact. `cre`, `dbs`, `ntsr1` and `pddl` were never settled; their documents
just yield one pair.

**What this does to the options.** Options **1, 2, 3, 4 and 6 are all downstream of an invalid
input** and none can be evaluated, let alone chosen, until Node B is fixed — a threshold, a
precedence fix, a continuous surface, a relabel and a field-relative reference class are all ways of
presenting a number that currently encodes list position. Option **2 remains correct on its own
terms** (a node with zero supporting sources is not contested) but is no longer interesting: the
`ns=0` case it fixes came from a single `supersedes` on a one-pair prompt, so it was an artifact,
not "the one real defect" this ADR originally called it.

**The decision this ADR now records:** *do not implement any surfacing option yet.* The next work is
a **Node-B redesign** — give the model the passages where the two concepts co-occur, add a
neutral/no-stance option, and remove the position dependence (one pair per call, or an equivalent
fix) — with a hand-labelled ground-truth study on the RG-015 template. That is a separate ADR. Until
it lands, `contested` should not be presented as an epistemic signal.

## Consequences

**Easier.** RG-019 stops being an untested hypothesis and becomes a measured negative result with a
repeatable instrument (`scripts/measure_contested_density.py`) — the next session re-runs it instead
of re-deriving it. ADR-027's "else the strip ships saturated" caveat is answered: the strip is
saturated, and no threshold it anticipated would have prevented that.

**Harder.** Option 3 reopens a shipped UI contract (ADR-027 D3's source-evaluation strip) and the
`markers` list that `derive_markers` produces, which the chat view, the reviewer's
`contested_evidence` tag and `AnswerRecord` all read. A continuous surface is a wider change than
the threshold everyone expected.

**Must revisit.** All of this is one corpus at 13 nodes. The trigger remains **monotone in corpus
size** — the original RG-019 worry, still untested — so the density figures here bound nothing about
10,000 documents, and a multi-domain vocabulary could revive lever A by producing genuinely marginal
nodes that this corpus does not contain. **Option 6's blocker is explicitly temporary:** re-run the
parent-field cross-tab once `graph_include` covers enough concepts to give several fields several
members, and decide it then. `scripts/measure_contested_density.py` prints that table, so the
re-test is a re-run, not a re-derivation.

## Confidence

- ✓ **The levers' inertness is measured, not argued** — every counterfactual re-projects modified
  node weights onto the real 18,831 chunk segments; the shipped configuration reproduces the
  walkthrough's 396/743 exactly, which is what validates the instrument.
- ✓ **The `ns=0` defect is certain** — a node with zero supporting sources and one disputing source
  is not contested under any reading, and the contested-first precedence is exactly why it slips past
  the neutrality rule.
- ⚠ **Precision against a hand judgment was not measured.** The claim "these labels are structurally
  correct" rests on each node having real two-sided stance across 5–11 documents, not on reading the
  chunks. RG-019's spot-check remains owed.
- ✓ **The parent-field confound is total, not sampled** — 13/13 concepts placed, so the cross-tab
  has no missing cells to hide behind, and `scripts/measure_contested_density.py` reproduces it.
- ✓ **The input is now measured, not merely suspected** (Update above, KI-33). Two of its defects
  are **structural facts about the prompt** — no document text, no neutral label — so they hold for
  any model and are not a calibration question. The position experiment is controlled: one variable
  moved, four verdicts, boundary crossed; the replay reproduces 5/5.
- ⚠ **Prevalence of the position artifact is not quantified.** How much of the 30.8% opposing share
  is position-driven rather than prior-driven is unmeasured. The supported claim is that the signal
  is contaminated and unusable as an epistemic marker — not that every opposing stance is an
  artifact.
- ⚠ **No ground truth exists.** Nobody read the source documents to establish the true stance. The
  finding shows the extractor *cannot* be measuring the documents (it never sees them); it does not
  score accuracy against a hand-labelled set. A Node-B redesign owes that study.
- ⚠ **Option 6 is argued, not demonstrated**, and is now doubly blocked — by n=1 in four of six
  fields, and by the input defect above.
- ⚠ **13 nodes.** Small enough that a single vocabulary change could move every table in the
  baseline.

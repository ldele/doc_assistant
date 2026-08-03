<!-- status: active · updated: 2026-08-02 · class: append-only -->

# `contested` marker density — RG-019's hypothesis measured, and disproved (2026-08-02)

**Question.** The v0.4.0 walkthrough recorded `contested` as saturated: **53.3% of assessed chunks**
against 3.9% `corroborated`, with 8 of 10 sources in a live turn reading *contested* on "what is
dense passage retrieval vs BM25" — not a controversy. RG-019 has carried the diagnosis since
2026-07-19: *"`coverage="contested"` triggers on `nc >= 1` (an implicit, unnamed threshold)"*, with
the prescribed fix *"derive a named floor (min disputing docs and/or an agreement-ratio band — the
`MIN_DATED_DOCS_PER_SIDE` pattern)"*. ADR-027 assumed the same fix and shipped the always-on strip
without it, noting the strip would otherwise "ship saturated".

**This measures whether that floor would work. It would not.**

## Setup

| | |
|---|---|
| Corpus | **97 documents · 18,831 chunk segments** (baseline + parent-child, the live library) |
| Skeleton | `data/skeleton/skeleton.json` — **13 nodes · 19 edges** (`graph_include`-scoped, ADR-018) |
| Instrument | `scripts/measure_contested_density.py` (read-only, no LLM, no writes) |
| Cost | **$0** — no provider constructed |
| Method | Load the live skeleton → `node_weights_for_epistemics` → `project_chunk_weights` onto the real segmentations; re-derive coverage under each candidate floor and **re-project**, so every counterfactual is measured, not argued |

## 1. The denominator, stated precisely

| | count | share |
|---|---|---|
| chunk segments in the store | 18,831 | — |
| carrying **any** claim | 743 | **3.9%** of the store |
| of those, marked `contested` | 396 | **53.3%** of assessed |
| marked, as a share of the store | 396 | **2.1%** |

The headline 53.3% is conditional on being assessed at all. Both figures are honest; quoting the
first without the second overstates the marker's reach by ~25x. It is nevertheless the
user-visible number, because retrieval returns the chunks that carry claims.

## 2. Lever A — a minimum-disputing-documents floor is inert

| rule | contested nodes | marked | % assessed |
|---|---|---|---|
| `nc >= 1` **(today)** | 7 | 396 | **53.3%** |
| `nc >= 2` | 6 | 394 | 53.0% |
| `nc >= 3` | 5 | 393 | 52.9% |
| `nc >= 4` | 3 | 199 | 26.8% |

**RG-019's stated mechanism accounts for two chunks.** Only **1 of 7** contested nodes has `nc == 1`.
The `MIN_DATED_DOCS_PER_SIDE = 2` pattern, transplanted here, would change 0.3 percentage points.

## 3. Why — the contested nodes are the corpus's core vocabulary

| label | ns | nc | agreement | direction |
|---|---|---|---|---|
| knowledge distillation | **0** | 1 | **0.000** | contested |
| BM25 | 6 | 5 | 0.545 | contested |
| dense retrieval | 5 | 4 | 0.556 | contested |
| passage retrieval | 5 | 4 | 0.556 | contested |
| contrastive learning | 6 | 3 | 0.667 | contested |
| re-ranking | 6 | 3 | 0.667 | contested |
| hard negatives | 5 | 2 | 0.714 | contested |

Seven of thirteen concepts are contested, and they are the central topics of an IR corpus, each
with genuine two-sided stance across 5–11 documents. **`contested` is not misfiring.** It is
firing correctly on ordinary scholarly disagreement, which the UI then presents as cautionary.

**`knowledge distillation` is the one real defect**: `ns=0, nc=1` — zero supporting sources and one
disputing one. Coverage is decided contested-first, so the unique-source neutrality rule (Decision 4,
"a sole source is never contested") never gets to see it. A node with no support at all is not
*contested*; it is unsourced.

## 4. Lever B — `agreement_ratio` has range, but nothing to anchor a cut on

| rule | contested nodes | marked | % assessed |
|---|---|---|---|
| today (unused) | 7 | 396 | 53.3% |
| `agreement < 0.75` | 7 | 396 | 53.3% |
| `agreement < 0.70` | 6 | 395 | 53.2% |
| `agreement < 0.60` | 4 | 201 | 27.1% |
| `agreement < 0.50` | 1 | 4 | **0.5%** |

The value is computed at `concept_skeleton.py:710` and consulted by nothing — RG-019 is right about
that. But the seven observed ratios cluster in **0.545–0.714**: every threshold with any effect
falls inside a band containing seven data points, and moving it 0.10 swings density from 53% to 27%
to 0.5%. **Any cut here is fitted to this corpus** — precisely the over-optimise-on-current-corpus
failure KI-19 exists to forbid, on a 13-concept vocabulary.

## 5. Lever C — the chunk-level rule is a second, hidden `>= 1`

`derive_markers` marks a chunk if **any** of its claims sits on a contested node.

| n_claims per assessed chunk | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| chunks | **662** | 76 | 3 | 2 |

| rule | marked | % assessed |
|---|---|---|
| `n_contested >= 1` **(today)** | 396 | 53.3% |
| `n_contested >= 2` | 59 | 7.9% |
| `n_contested >= 3` | 4 | 0.5% |
| contested majority of claims | 395 | 53.2% |
| ALL claims contested | 394 | 53.0% |

**89% of assessed chunks carry exactly one claim**, so "any" / "majority" / "all" are the same rule
on this corpus — 53.3 / 53.2 / 53.0%. `n_contested >= 2` cuts to 7.9%, but it does so by requiring a
chunk to mention two contested concepts, which most chunks cannot do at all. That is a structural
silencer, not an epistemic threshold.

## 6. The parent field predicts contestedness — the finding that reframes levers A–C

Every graph concept carries exactly one ANZSRC parent field (ADR-028) — **13/13 placed**, so the
join is total, not a sample. Cross-tabulated:

| parent field | concepts | contested |
|---|---|---|
| Machine learning | 6 | **4** |
| Data management and data science | 3 | **3** |
| Artificial intelligence | 1 | 0 |
| Neurosciences | 1 | 0 |
| Medicinal and biomolecular chemistry | 1 | 0 |
| Biochemistry and cell biology | 1 | 0 |

**7 of 9 concepts in the two IR/ML fields are contested; 0 of 4 outside them.** Contestedness is
near-perfectly confounded with domain.

`cre`, `dbs`, `ntsr1` and `pddl` are not uncontested because they are settled — they are
uncontested because they have **one source each**. The marker is tracking *how densely the corpus
covers a field*, not whether a claim is disputed. That is why no cut point on a per-concept rate
can work: the rate is dominated by a variable the rate does not contain.

## Conclusion

**No threshold on this data is defensible, and §6 says why it is not a threshold problem at all.**
Lever A is inert, lever B has no anchor that is not corpus-fitting, lever C's discriminating variant
silences the feature for a reason unrelated to epistemics — and all three operate on a per-concept
rate whose dominant term is the parent field's document density. The saturation is `contested` being
a **binary label on a continuous quantity, measured without a reference class**. That makes RG-019 a
surfacing question, not a tuning one; the options are drawn in
[ADR-040](../../../docs/decisions/ADR-040-contested-is-a-surface-not-a-threshold.md), whose option 6
(contestedness relative to the parent field's base rate) is the direct consequence of §6.

## What the literature says about this class of statistic

Checked 2026-08-02, and it is unfavourable to every lever above:

- **`agreement_ratio` is raw percent agreement**, which the inter-rater literature (Cohen's κ,
  Krippendorff's α) exists to replace — it carries no correction for agreement expected by chance.
  Landis & Koch's benchmark bands are the standard cautionary tale of exactly the arbitrary cut
  points lever B would introduce.
- **The nearest real discipline is meta-analysis, and it declines to threshold.** Heterogeneity
  (Cochran's Q, Higgins' I²) is the established statistic for "do sources disagree"; the 25/50/75%
  bands were offered as tentative, the Cochrane Handbook cautions against mechanical application,
  and [I² is known to be non-discriminative for prevalence meta-analyses](https://pubmed.ncbi.nlm.nih.gov/35088937/)
  — the closest analogue to this case. Practice reports it with τ² and a **prediction interval**.
- **n=1 wants an interval, not a class.** A Wilson score interval on 0/1 spans roughly [0, 0.79];
  the honest output for `knowledge distillation` is *insufficient evidence*, a state the schema
  already has (`unique`, held NEUTRAL, Decision 4) and that contested-first precedence steals.
- **The 53% is anomalous against how disagreement appears in text.** Citation-polarity corpora put
  [neutral citations above 60% with contrasting/negative the rarest class](https://direct.mit.edu/qss/article/2/3/882/102990/scite-A-smart-citation-index-that-displays-the-context-of-citations);
  these nodes average ~45% opposing sources. That is independent evidence pointing at the Node-B
  extractor rather than the corpus.
- **Partial pooling is the named method for §6.** Shrinking each concept's rate toward its parent
  field's mean in proportion to sampling variance is
  [standard empirical-Bayes practice](https://www2.stat.duke.edu/~pdh10/Teaching/732/Notes/shrinkage.pdf),
  and James–Stein dominates raw group means for k≥3 — the raw per-concept ratio in use today is
  precisely the estimator that result advises against.

## Caveats — what this does not establish

- ⚠ **One corpus, 13 nodes.** The vocabulary is `graph_include`-scoped (ADR-018) and single-domain.
  A wider or multi-domain vocabulary could move all three levers. This bounds nothing about scale —
  the trigger is still monotone in corpus size, which is what RG-019 warned about and what remains
  untested.
- ⚠ **Precision against a hand judgment was not measured.** RG-019 also owes a spot-check of whether
  a marked chunk *reads* contested. This run measures density and the levers' reach; it argues the
  labels are structurally correct (real two-sided stance, 5–11 documents each) but does not read the
  chunks. The `ns=0` case is the exception — that one is wrong on its face.
- ⚠ **Stance data is Node-B LLM output** (`llama3.1:8b` on this box). A systematically
  disagreement-biased stance extractor would produce exactly this picture, and nothing here
  distinguishes that from real disagreement. **That is the measurement worth doing next** and it is
  cheaper than any threshold study: sample stance calls and check them. The citation-polarity base
  rates above sharpen this from a possibility into the leading hypothesis.
- ⚠ **§6 diagnoses the confound; it does not yet support a field-relative estimator.** Four of the
  six fields hold exactly **one** concept, so no field-level base rate can be estimated from them —
  partial pooling needs several groups with several members each. The parent join is currently
  strong enough to explain the saturation and to rule out a global cut point, and **not** strong
  enough to compute the shrunk rate ADR-040 option 6 describes. That becomes possible as
  `graph_include` grows past 13 (344 curated concepts sit outside the graph today, ADR-018).

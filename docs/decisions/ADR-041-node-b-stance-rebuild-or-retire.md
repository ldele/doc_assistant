<!-- status: active · updated: 2026-08-03 · class: append-only -->

# ADR-041 — Epistemics stays; Node-B stance must be rebuilt on evidence, and per-concept status re-bases on the claim layer first

*(Filename says "rebuild-or-retire" — that was the question, and retire was explored and withdrawn.)*

- **Status:** proposed — recommendation set by the user's stated product intent (2026-08-03); the
  build sequence is open
- **Date:** 2026-08-03
- **Deciders:** user + Claude Code

## Context

**KI-33** established that Node-B stance is not a measurement of the corpus
([`node_b_stance_validity_2026-08-02.md`](../../tests/eval/baselines/node_b_stance_validity_2026-08-02.md)):
the model is never shown the document it is asked to judge, there is no neutral option in
`POLARITIES`, and the verdict changes with the pair's index in the numbered list (one document, same
17 pairs, temperature 0 → four verdicts crossing the supporting/opposing boundary; the real prompts
replay 5/5, so this is deterministic shipped behaviour). ADR-040's surfacing options are all
downstream of it.

**The decisive context is not in KI-33 — it is in the graph's own spec.** `feature-concept-graph.md`
§*The job* (B1, locked with the user 2026-07-17) states the purpose in three questions:

1. **Corroboration** — *"is this concept backed by more than one source?"* The user's framing:
   *"Technically, having a single source is not good."*
2. **Coverage** — *"have I read the field?"*
3. **Navigation** — *"explore the sources through the graph"*, down to the chunks.

And: *"The graph is the substrate; **the gaps are the payload**."*

**Every one of those three is answered by counting `doc_ids`.** Corroboration is
`len(doc_ids) >= 2`. Coverage is presence per field. Navigation is
`node → doc_ids → concept_presence.chunk_keys`. **All are Node-A, deterministic, zero-LLM.** RG-014
confirmed it empirically: `single_source` is the one detector graded **"TRUE POSITIVE — the product
thesis"**, and it reads document counts.

**Stance answers a different question — "do the sources *disagree*?" — that B1 never asked.** It
entered through the 7d currency work and ADR-027's source-evaluation strip, not through the graph's
stated job. The spec's own grounding note from design-lock records the layer as *"The epistemic
dimension is empty. All 26 nodes are `unique`/`stable`; `contested_edges()` → `[]`"* — the feature
was designed, locked and graded useful **before stance existed at all**.

So the question is not only *how to rebuild Node B*. It is **whether the project wants the
disagreement feature enough to pay for it properly**, given that nothing in the stated job depends on
it and the corroboration signal that does work is deterministic.

### The product intent (user, 2026-08-03) — this answers that question

B1's three questions are the *surface* description. The intent behind them, stated directly:

> *"The goal is to see which claims are unsubstantiated and where are the knowledge gaps inside the
> documentation. … The most barebone goal of the graph feature is to be able to classify knowledge
> per concept in order to find the gaps where the user, for a given subject, should find more
> resources and documents. It is supposed to both expose the gaps and make research of information
> easier. … We want epistemics feature. That is the idea."*

Two things follow, and they reshape this ADR.

**1. Epistemics is wanted deliberately, not inherited.** The reading in the section above — that
stance "answers a question B1 never asked" — is right about the *spec text* and wrong about the
*intent*. Per-concept knowledge classification is the feature; the retire option is therefore
against stated product direction, not merely conservative. **This is the decision input the ADR was
waiting on**, since it opened by saying the technical finding could not settle a product question.

**2. The map exists so the corpus can grow in a relevant direction.** The user's structural thesis —
*"concepts are linked in general to predictable things"* — is what makes a gap detectable: a gap is
**a deviation from expected structure**, not an absolute count. That is the same reference-class
argument ADR-040 option 6 reached from the other end (contestedness only means something against a
base rate). It also says where the expected structure has to come from: the curated taxonomy
(ADR-028), or general knowledge of how concepts relate, or an external reach — the Tier-2b
"anti-blind-spot" direction ADR-004 deferred.

**What survives from the measurement, and it is the whole of it:** the *current* Node B cannot serve
this goal. A pass that never sees the document cannot classify how well a document supports a
concept. That is an argument that Node B is not an epistemics implementation, **not** an argument
against epistemics.

Two constraints bound any rebuild. **Cost:** a grounded stance call needs the passages where both
concepts co-occur, so it is one sizeable call per (edge × document) — 65 today, but the edge count is
quadratic in vocabulary, which is exactly the unbounded-LLM-budget hazard KI-19 already files.
**Evidence:** no accuracy claim is possible without a hand-labelled ground-truth set, on the RG-015
template (which spent a full corpus pass against hand labels to rank three local models).

## Options

1. **Rebuild Node B with evidence.** Pass the co-occurring passages, add a `neutral`/`no_stance`
   label, and annotate **one pair per call** so list position cannot leak in. *Trade-off:* it fixes
   all three defects at the root and is the only option that could ever make "these sources disagree"
   a supportable claim. But it multiplies Node-B cost by the passage payload and by the loss of
   batching, on a pass whose call count already grows with the square of the vocabulary — and it is
   worthless without a hand-labelled ground-truth study, so the real price is the study plus the
   rebuild, for a feature the stated job does not require.
2. **Retire Node-B stance.** Keep Node A entirely — association edges, communities, presence,
   `doc_ids`. Delete the stance half: `POLARITIES` consumers, `contested`/`superseded_trend` coverage
   and direction, the epistemics markers derived from them, and the strip's assessment column.
   *Trade-off:* every part of B1 survives untouched, and the detector RG-014 called the product
   thesis (`single_source`) is unaffected. It is also the move this codebase has just made twice —
   ADR-038 deleted the unproven second retrieval arm rather than maintain it, and KI-29 showed what a
   path nobody exercises does to the answer. The cost is giving up the disagreement story, which is
   currently the most distinctive-sounding thing in the product pitch.
3. **Keep the relation verb, drop the stance.** The verb ("is evaluated with") is a navigation label;
   the stance is the epistemic claim. *Trade-off:* it looks like the cheap middle, and it is the
   weakest option on inspection — **the verb comes from the same text-free, position-sensitive
   prompt**. Across the position probe the same pair produced *is used with · uses · is improved by ·
   compared to · is compared to · improves on*, and `relation_by_pair` keeps whichever document
   answered first. If it is kept it must be labelled a **suggestion**, never a finding, and it should
   not be described as derived from the document.
4. **A deterministic tension proxy instead of an LLM.** Derive disagreement from signals the system
   already computes — citation polarity, or the year-ordering that `superseded_trend` (G3) already
   uses. *Trade-off:* deterministic, free, and auditable, matching the project's stated preference
   for structural attribution over model judgement. But **0 citations resolve on this corpus**
   (spec §Grounding), so the citation half cannot fire today, and year-ordering alone reduces to what
   G3/G6 already ship.
5. **Park it: stop surfacing, keep the code.** Set the stance-derived surfaces off, leave Node B in
   place unrun. *Trade-off:* reversible and honest, costs nothing, and buys time for the product
   decision. But it leaves a built, wired, silently-wrong path in the tree — the precise shape KI-29
   and ADR-038 both punished.
6. **Re-base per-concept epistemics on the claim machinery that already works.** Instead of asking a
   model whether two concept *labels* conflict, join each concept to the *claims the corpus makes
   about it* and score those with the retrieval-derived signals the answer path already computes —
   `AnswerClaim`, the `weakly grounded` / `unsupported` markers, and `unsourced_claim`. Per-concept
   evidential status becomes: how many independent documents assert it, how many claims about it are
   unsupported, how its dated evidence trends. *Trade-off:* this is the option that most directly
   serves the stated headline goal — *"see which claims are unsubstantiated"* — and it does so with
   machinery already graded trustworthy, built on the principle `how-answers-work.md` states
   explicitly (*markers derive from retrieval signals, not the model's own confidence, because
   language models are systematically over-confident*). It is deterministic, needs no new LLM pass,
   and its unit is **the concept**, which is the unit the goal names — where the current design
   annotates *edges* and reaches concepts only by aggregation. **But it gives support and absence,
   not polarity:** it can say "thinly sourced" or "asserted without citation", and it cannot say
   "these two sources disagree". The disagreement half still needs option 1.

## Decision

**Recommended: option 6, then option 1. Option 2 (retire) is withdrawn.**

*An earlier draft of this ADR recommended retiring. That was reasoned from B1's spec text without
the product intent recorded above, and the intent settles it: **epistemics is the feature**, so
removing it is not on the table. The measurement's finding is unchanged and still binding — the
current Node B cannot serve the goal — but "the current implementation is invalid" and "the feature
is unwanted" are different claims, and only the first is supported.*

**Option 6 leads because it serves the stated headline goal directly and cheaply.** The goal names
*unsubstantiated claims* first, and that is a support question, not a polarity question — answerable
from machinery that already exists, is deterministic, and is already graded trustworthy. It also
fixes a unit mismatch nobody had named: the goal is *knowledge classified per concept*, while the
current design classifies **edges** and reaches concepts by aggregation. Doing option 6 first means
the epistemics surface starts telling the truth without waiting on a model rebuild.

**Option 1 follows, for the half option 6 cannot reach.** "These sources disagree" needs a model
that has read the sources: co-occurrence passages in the prompt, a `neutral` / `no_stance` label,
one pair per call so position cannot leak in, and a hand-labelled ground-truth set on the RG-015
template as the gate — the study is the deliverable, not the rebuild. Scope it as its own feature
with the cost budgeted from the start (the call count grows with the square of the vocabulary; KI-19).

**Option 3 should not be chosen as a compromise.** It preserves the least-defensible artifact (an
arbitrary first-wins verb) while removing the part that at least had a stated purpose.

**Option 4 is worth keeping in view for the structural half.** The user's thesis that *concepts link
to predictable things* means a gap is a deviation from expected structure — so the expected structure
needs a source, and the curated taxonomy (ADR-028) is the one already in the tree. That is a
deterministic route to "for this subject, go find more on X", and it does not depend on either option
above landing.

**Whatever is chosen, one thing lands regardless:** the surfaces must stop presenting stance-derived
output as an epistemic signal until it is rebuilt or removed. That is a documentation and UI change,
not a modelling one, and it is the only part with a deadline.

## Consequences

**Easier (option 6).** Per-concept epistemics starts telling the truth without waiting on a model
rebuild, using signals that are deterministic, free, and already trusted elsewhere in the product.
It also aligns the unit of analysis with the goal — knowledge per *concept* — which makes "for this
subject, go read more on X" expressible for the first time. No new LLM pass means no KI-4
credit-leak surface and no KI-19 call growth added.

**Harder (option 1, when it comes).** A passage-grounded, one-pair-per-call Node B is materially more
expensive than today's batched, text-free pass: bigger prompts, no batching, and a call count that
grows with the square of the vocabulary. The ground-truth study is a real piece of work in its own
right, and it is the gate rather than a follow-up — an accuracy claim without it would repeat exactly
the mistake KI-33 records.

**Immediate, regardless of sequence.** The stance-derived surfaces must stop presenting themselves as
epistemic findings until one of these lands — the strip's assessment column, the answer-layer
markers, the CHANGELOG feature list. That is a docs/UI change, not a modelling one, and it is the
only part with a deadline.

**Must revisit.** The honest prior after KI-33 is that *"do these two concepts conflict in this
paper"* may be too hard for an 8B local model even with the passages in front of it — in which case
option 1 costs a paid provider per edge-document pair and the economics change again. Option 6 is
deliberately sequenced first partly so that the product is not blocked on that answer.

## Confidence

- ✓ **The defect is measured and structural** (KI-33): two of its three causes are properties of the
  prompt, not the model, and the position experiment is controlled.
- ✓ **B1's three jobs are document-count questions**, and the spec records the layer as graded useful
  while the epistemic dimension was empty. ⚠ **But do not read that as "epistemics is unwanted"** —
  an earlier draft of this ADR did, and the product intent above contradicts it. The spec text
  describes the surface, not the intent behind it.
- ⚠ **The cost estimate for option 1 is arithmetic, not measured.** Nobody has run a passage-grounded,
  one-pair-per-call Node B to time it or price it.
- ⚠ **Retiring is not obviously right for the product**, only for the evidence. A disagreement signal
  that worked would be genuinely differentiating; this ADR argues that the current one does not work
  and was never required, not that the idea is bad.
- ⚠ **The G3/G6 collateral is not fully scoped.** Whether `superseded_trend` can be re-based on years
  alone without stance has not been checked in code.

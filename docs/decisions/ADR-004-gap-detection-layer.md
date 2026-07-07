<!-- status: active · updated: 2026-07-07 · class: append-only -->

# ADR-004 — Gap detection: a two-tier deterministic/stochastic layer over the concept graph

- **Status:** accepted — **Tier 1 + the Tier-2a deterministic floor BUILT (2026-07-07, SPRINT-002,
  ROADMAP row G2)**; the Tier-2a stochastic ceiling (`gap_suggest.py`) and Tier 2b remain not built
  (2026-06-26 — decided with Cowork; build spec `docs/specs/feature-gap-detection.md`)
- **Date:** 2026-06-26
- **Deciders:** Lucas (decided with Cowork)

> This ADR **extends** the concept-graph redesign ("Feature 7 — concept-graph REDESIGN", 2026-06-18,
> in `docs/decisions.md`, here "Decision C"). Decision C settled where concept *nodes and edges* come
> from; this ADR settles the **gap-detection layer that sits on top of them** — the layer that carries
> Phase 7's headline capability. It supersedes nothing; it makes the gap mechanism concrete and names
> the contract the build spec is written against. Gap signals shipped earlier (the wiki 6b
> citation-thin/single-source flags and the concept-graph 7c isolated-node/thin-bridge signals) are
> **subsumed** here into one unified, typed layer.

## Context

Phase 7's stated purpose is **gap detection** — and the project's north-star reason for it is to
surface what the user (and the LLM) cannot see: concepts the corpus under-supports, claims it cannot
source, and directions for exploration the user did not think to look. The standing project principle
this rests on is **full observability of LLM output with user judgment as the fallback** — expose what
the LLM used, rate what it produced, let the user take what is useful (the same philosophy as the
Chunk 2a accept/reject markers and the Chunk 2b reviewer).

Two facts about the current state frame the decision:

1. **The concept-graph node/edge model is now curated and deterministic.** Per Decision C, nodes are a
   user-curated `Concept`/`ConceptAlias` set; presence and the edge skeleton compute with no LLM; the
   LLM only annotates a relation/stance over concepts already co-present. This is deliberate — it gives
   the user control and removes per-document extraction cost.

2. **A curated graph creates a structural blind spot.** Gap detection over a curated vocabulary can
   only find gaps **inside the terms the user already chose**. But the headline feature is finding the
   gaps the user **did not know to look for**. Those two pull in opposite directions: curation buys
   precision on the known and loses recall on the unknown. This tension is the design question of the
   layer, and resolving it — rather than picking one side — is what this ADR does.

A second, separate observation shapes the layer: signals that already exist in the codebase are, in
effect, gap signals that were never collected into a gap layer. The dual-interpretation layer already
marks each answer-claim `unsupported` when it cites no real source (`synthesis.claim_marker()`,
persisted on `answer_claims.marker`); the reviewer already emits an answer-level `failure_tag`
(`unsupported_claim`, `missing_citation`, `overclaim`, …); the `Citation` graph already records which
documents a text leans on. "A claim the corpus cannot source" *is* a corpus gap. These are present,
persisted, and currently unused for gap detection.

## Options

1. **Single-tier deterministic gaps over the curated graph (chosen floor, insufficient alone).**
   Compute isolated nodes, single-source concepts, thin bridges, and under-connected nodes over the
   skeleton. *Trade-off:* trustworthy, free, auditable — but it can only ever report gaps **inside the
   curated vocabulary**, so it cannot deliver the anti-blind-spot capability that is the point. Correct
   as a floor; incomplete as the whole layer.

2. **Open-vocabulary LLM extraction as the gap finder.** Re-admit per-document open-vocabulary
   extraction so the LLM surfaces concepts the user never recorded. *Trade-off:* this is exactly what
   Decision C retired — it is the dominant cost (the validation branch hit `budget_exhausted` over the
   corpus at 36–40 calls/doc) and the source of concept fragmentation on a same-domain corpus. As the
   *whole* node source it is rejected (Decision C). As a **fenced-off suggestion source that never
   writes the graph**, the idea survives into the chosen design's outer tier.

3. **Reuse the idea-generator (sibling project) as the blind-spot finder.** The idea-generator bridges
   cross-domain concepts and gates them on cosine novelty against its own pool. *Trade-off / rejected
   on a structural ground:* it **closes inward**. Its novelty gate measures distance against its own
   pool, so "novel" means "far within the space the pool already spans" — it interpolates into the gaps
   *between* known concepts and has no representation of "outside" its own embedding manifold. It is a
   convex-hull filler, not a frontier-crosser; anti-blind-spot detection requires the opposite (a
   signal from outside the known space). Confirmed empirically: when pointed at this task it completed
   concepts onto themselves rather than reaching past them. Recorded here so it is not retried. (This
   also subsumes an earlier embedding-space concern — the two projects use different embedders/dims —
   which is moot: even matched, the idea-generator is the wrong *shape* for this job.)

4. **Two-tier layer split on the deterministic/stochastic line (chosen).** Make the
   deterministic/stochastic boundary — already a project tenet — the architecture of the gap layer.
   Deterministic gaps are facts about the graph (the trustworthy floor); stochastic gaps are
   *suggestions* that only the LLM or an external reach can produce, **labeled stochastic, rated, and
   never written into the skeleton as fact** — they are candidates the user promotes into the curated
   vocabulary. Within-corpus suggestions (buildable now) are separated from a true external reach
   (deferred), so the layer ships in stages.

## Decision

**Adopt the two-tier layer (option 4)**, with the deterministic/stochastic wall as its organizing
principle and a three-sub-tier structure:

- **Tier 1 — deterministic, within the curated vocabulary.** Isolated nodes, single-source concepts,
  thin bridges, and under-connected ("edge") nodes over the deterministic skeleton. Facts about the
  graph: free, byte-stable, auditable to a citation / similarity / co-occurrence fact. This unifies the
  pre-existing wiki-6b and concept-graph-7c gap signals into one typed output.

- **Tier 2a — within the corpus, gated inside the app (buildable now, no external reach).** A
  deterministic floor + a stochastic ceiling:
  - *Floor (a query over data that already exists):* aggregate the per-claim `unsupported` markers
    (`answer_claims.marker`) across answers to find concepts the corpus repeatedly fails to source; and
    compute citation-layer gaps (what documents cite via the `Citation` graph vs. what is actually
    ingested). No new model — wiring over persisted data.
  - *Ceiling (a quarantined LLM pass):* route Tier-1 under-connected nodes plus the reviewer's
    answer-level `failure_tag` into an LLM pass that **suggests** a missing link, a missing concept, or
    a "genuinely thin area" verdict. Output is labeled stochastic, rated, and surfaced as suggestions —
    never auto-written.

- **Tier 2b — the true anti-blind-spot pass (deferred; requires an external fetch).** Concepts
  *contingent* to the corpus — adjacent but absent. This is the only tier that cannot be served from
  inside the app, because it needs a representation of "outside the known space": an external concept
  bank / taxonomy, citation-chasing into un-ingested references, or web reach, constrained by the
  library and the user's input. Deferred behind Tiers 1 and 2a. **Explicitly not the idea-generator**
  (option 3).

Three cross-cutting rules hold across the tiers:

- **The deterministic/stochastic label is first-class.** A consumer reads it and decides trust without
  re-deriving it. Stochastic gaps carry the LLM inputs they were produced from (the observability
  mandate); deterministic gaps carry the graph-fact ids.
- **Stochastic gaps feed the curated vocabulary (the compounding arrow).** Every Tier-2 suggestion is a
  candidate promotion into Tier-1's `Concept`/`ConceptAlias` set. Accepted suggestions enrich the next
  deterministic rebuild, so the graph compounds through curation. (Same mutable-store + gate + write-back
  shape the idea-generator uses on its pool; here the gate is the user's judgment.)
- **Three gap *types* stay distinct — never flattened.** "I have no source for X" (`unsupported`),
  "my sources disagree on X" (`contested_evidence` / 7d `contested`), and "the field moved on"
  (`superseded_trend`) are different blind spots the user acts on differently. The codebase already
  separates them; the gap layer preserves the distinction.

The deciding reason: the deterministic/stochastic wall is what makes blind-spot detection **safe** —
the stochastic finder is allowed to be wrong because it can only ever propose a candidate that a
deterministic check or the user must accept; it cannot silently corrupt the graph. That single property
lets the layer pursue recall on the unknown (the headline feature) without surrendering the precision
and auditability the curated graph was built for.

**What would reverse it:** if the within-corpus stochastic ceiling (Tier 2a) produces suggestions of
too low value to be worth the curation attention, drop it and keep Tiers 1 + the deterministic 2a floor
(still a real improvement over today). If Tier-1 gap signals prove meaningless because the edge model
cannot be made precise (see Confidence), the gap layer is blocked until the edge model is — gaps are
defined relative to the edge set and cannot ship on an un-validated one.

## Consequences

**Easier:** Phase 7 gets a single typed gap object instead of two ad-hoc signal sets (wiki 6b +
concept 7c collapse into Tier 1). The cheapest first increment — the deterministic 2a floor — is a
query over already-persisted `answer_claims` rows plus the `Citation` graph, not new ML. The
observability/rating surface that Tier-2a's stochastic pass needs is the **same** surface the planned
Phase 9 review-generation feature needs (expose LLM inputs, rate output, user disposes), so building it
once serves both — the shared dependency of these features is that substrate, not the concept graph.
Consequently the reviewer is reframed as a gap *feeder*, and review-generation is no longer strictly
"after" gap detection — its observability spine is shared with Tier 2a.

**Harder / cost:** the curated graph's blind spot is real and only the deferred Tier 2b fully addresses
it — until then the layer is precise-but-incomplete by construction, which must be stated to the user,
not hidden. The stochastic tiers add a curation-review burden (mitigated because suggestions are
optional and rated). A new gap object + persistence and a quarantined-LLM seam are net-new surface.

**Must revisit:** Tier-1 thresholds (the `under_connected` degree cut in particular) are tuning values
that depend on the edge-precision validation run and are set in the spec, not here. The Tier-2b external
source (concept bank vs. citation-chasing vs. web) is an open choice deferred with the tier. The exact
shape of the shared observability/rating contract (the gap object's `rating` field) settles when its
first consumer — the Tier-2a floor — is built.

## Confidence

- ✓ The within-corpus floor consumes signals that already exist and are persisted: per-claim
  `unsupported` markers (`synthesis.claim_marker()` → `answer_claims.marker`), the reviewer
  `failure_tag` enum (`reviewer.ReviewResult`), and the `Citation` graph (`db/models.py`). Verified
  present 2026-06-26.
- ✓ "The idea-generator closes inward" is a structural consequence of a novelty gate measured against
  its own pool, and was observed directly when it was pointed at this task — not an external claim.
- ⚠ **Tier-1 gap validity depends on edge precision, which is unvalidated.** The Decision-C skeleton
  projects edges through `Citation` + `DocSimilarity`, linking every curated interest in document X to
  every one in document Y; density/precision on the real corpus is unmeasured. Because isolated /
  thin-bridge / under-connected are all *defined relative to the edge set*, an over-connected graph
  reports near-zero gaps and an under-connected one reports nearly everything. **The edge-precision
  validation run is therefore a correctness gate on this feature, not optional rigor** — tracked in
  `.claude/RIGOR_TODO.md` (RG-001; edge-precision) and a precondition for locking the edge model.
- ⚠ **Presence recall is unset** (string vs. embedding/alias match threshold for concept presence,
  Decision C carryover) and sits upstream of every gap signal — `.claude/RIGOR_TODO.md`.

<!-- status: active · updated: 2026-07-21 · class: append-only -->

# ADR-027 — Epistemics surfacing split: assessment always-on, influence opt-in

- **Status:** accepted
- **Date:** 2026-07-21
- **Deciders:** user (product decision), Claude (Cowork review session 2026-07-21)

## Context

Product decision recorded here as **D1**: the app's primary use is **exploration and
population of the documentation corpus** — a searching tool first, an answer generator
second. That reorders what epistemics surfacing is *for*: helping the user judge sources
while exploring, not decorating answers.

The epistemics engine and live marker chips exist (Feature 7d; G1 default-on, G3/G6
year-aware `superseded_trend` with the ≥2-dated-docs-per-side floor, G7 label-attribution
fix — 3,334 marked chunks live on this corpus). Surfacing is governed by one flag,
`EPISTEMICS_MARKERS_ENABLED` (config default + the per-turn U1b sandbox override
`eff_markers_enabled`). One flag conflates two different questions: whether the user
**sees** the corpus's epistemic state, and whether epistemics **shapes the answer layer**.
The user wants these separated: influence must be optional (D2), but a per-source
evaluation should always sit under the chat answer if affordable (D3).

Cost check: all signals are precomputed offline into the `chunk_epistemics` sidecar;
answer-time surfacing is a lookup joined against `TOP_K=10` sources — no LLM call,
milliseconds. What actually gates an always-on surface is **data quality**, not cost:
KI-8 (corrected 2026-07-19: ~40% silent marker loss on parent-boundary straddle; the PC
`_chunk_key` mapping is still the PR-M1 TODO) and RG-019 (`contested` fires at `nc>=1`,
marking 53.6% of chunks — saturation, not signal).

## Options

1. **Keep the single flag.** — *Pros:* no work. *Cons:* the user cannot see assessment
   without accepting influence, or refuse influence without going blind; the same
   conflation already forced ADR-005's containment dance in the other direction.
2. **Split the surfaces (chosen).** Always-on advisory **source-evaluation strip**
   (assessment) + opt-in **answer-layer** influence (marker chips in the answer surface,
   and any future epistemics-aware retrieval). — *Pros:* matches the inform-don't-block
   contract; assessment is ~free by construction; each surface can be kept honest on its
   own terms. *Cons:* two surfaces to keep truthful; the strip must display staleness
   rather than pretend freshness.
3. **Always-on everything.** — *Pros:* simplest mental model. *Cons:* violates the
   explicit user requirement that influence be optional; repeats the ADR-005 mistake
   mirrored.

## Decision

Option 2.

- **D3 — assessment always-on (ROADMAP row E2):** every retrieved source gets an
  evaluation row below the chat answer regardless of any toggle: coverage/direction
  (`corroborated` / `unique` / `contested` / `superseded_trend`), doc year, the existing
  retrieval-derived signals (`weak_retrieval`, `single_source_risk`, rerank score), and a
  "computed as of `graph_version`" freshness hint.
- **D2 — influence opt-in (ROADMAP row E3):** whether epistemics touches the answer
  layer is a persisted user setting, layered under the existing per-turn
  `eff_markers_enabled` (U1b keeps working as the session-scoped override). The effective
  value is recorded in `AnswerRecord` (ADR-011 instrument-snapshot discipline), so
  provenance shows whether a given answer used epistemics.
- **Ordering:** E1 (KI-8 re-projection + PC `_chunk_key` completion) lands **before** E2 —
  an always-on strip built on a join that silently drops ~40% of markers would be a lying
  UI. RG-019 (a denominator for `contested`, e.g. consulting the already-computed
  `agreement_ratio` or a ≥2-disputing-docs floor) should land with or shortly after E2,
  else the strip ships saturated.

## Consequences

- **Easy:** the strip is cheap by construction (sidecar lookup, $0/turn); the toggle is
  UI + provenance work only — the engine plumbing already exists; U1b's sandbox knob is
  unchanged and still wins per-turn.
- **Hard / committed:** two surfaces must both respect staleness and the
  no-self-reported-confidence rule. Always-on assessment raises the stakes on marker-join
  correctness: the blanket `except: return` in `_attach_markers` must at minimum log at
  WARNING (a silent failure becomes a silently lying UI). The strip must degrade honestly
  when the sidecar is stale or absent (0-doc contract, WE-9 class).
- **Boundary:** the D2 toggle governs the answer layer only; it never hides the D3 strip.

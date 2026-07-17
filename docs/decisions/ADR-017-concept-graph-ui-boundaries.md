<!-- status: active · updated: 2026-07-17 · class: append-only -->

# ADR-017 — Concept-graph UI boundaries: read-only vocabulary, in-app rebuild, and durable gap triage

- **Status:** accepted
- **Date:** 2026-07-17
- **Deciders:** user + Claude Code (routed from the 2026-07-17 `grill-me` — 12 branches, 11 resolved / 1 parked)

> **Scope.** This ADR settles only the **boundaries** the concept-graph UI crosses. The feature contract
> (job, rendering, entry points, encoding, carve) lives in `docs/specs/feature-concept-graph.md`.
> **ADR-015** reserved this track ("the concept graph is the *heavy* consumer … and gets its own UI later,
> over the same rows"); **ADR-004** owns the gap layer this UI surfaces. This supersedes nothing.

## Context

The concept graph is the first UI over the curated `Concept` vocabulary that is **not** curation. Three
boundaries had no owner, and each is a place where a plausible design would quietly break an existing
guarantee.

**Measured facts this decision rests on** (live 76-doc corpus, 2026-07-17; all $0/offline):

- **The vocabulary is shared.** `library.list_keyword_families()` returns **every** `Concept` unfiltered, and
  `create_keyword_family` calls `add_concept`. Tag families (ADR-015) and graph nodes are **the same rows** —
  ADR-015 names this the "boundary risk to watch". The shipped **Manage-keywords view already does full CRUD**
  on them, and **PR-2.5 is currently repairing that write path** (rename creates duplicate `Concept` labels,
  500s the create route, and corrupts `promote_keyword` repo-wide).
- **The graph is a derived build artifact.** `data/skeleton/skeleton.json` is regenerated wholesale by
  `build_concept_skeleton` (Node A: zero-LLM, deterministic, idempotent, byte-stable) and is **gitignored**.
  It was **15 days stale** when the grill opened. **A full rebuild measures 7.1 s.**
- **Gap status is not durable for the gaps we actually have.** `gaps.py:257` —
  `session.execute(delete(GapRow).where(GapRow.determinism == "deterministic"))`. Deterministic rows are
  **delete-and-replace**; only `_write_stochastic_gap_rows` (`:273`) performs the *"status-preserving"*
  upsert. **Verified live: a `dismissed` deterministic gap reset to `surfaced` on the next run.** **All 14
  gaps on this corpus are deterministic** (stochastic = 0), so a dismissal today is futile — and a rebuild is
  part of the intended acquire loop (gap → find → ingest → rebuild → gap closes).

## Options

### A — Does the graph write to the vocabulary?

1. **Read-only for the vocabulary; deep-link to Manage-keywords.** *Pros:* one write surface, so the
   "a keyword belongs to at most one family" invariant has one home (the one PR-2.5 is fixing); you never edit
   a derived artifact; costs ~nothing. *Cons:* a curation fix is one click away, not in place.
2. **The graph writes too (promote / merge / rename in place).** *Pros:* fix it where you see it. *Cons:* a
   second writer onto rows whose invariants are provably broken today; **every write instantly stales the view
   you are reading**, so it lies until rebuilt.
3. **The graph replaces Manage-keywords.** *Pros:* one concept surface. *Cons:* discards a just-shipped view
   mid-hardening (PR-2.5/2.6/2.7); a force layout is a poor bulk-alias editor.

### B — Can the app trigger a rebuild, or is the runner CLI-only?

1. **In-app Rebuild button (202 + poll), CLI runner stays canonical.** *Pros:* 7.1 s is a button; the app
   **already** does exactly this for its biggest derived build (`POST /api/ingest` → `202` +
   `GET /api/ingest/status`); it closes the acquire loop without a terminal round-trip. *Cons:* a second
   caller of an enrichment seam.
2. **CLI-only; the UI prints the command.** *Pros:* strictly literal reading of "runners are CLI". *Cons:*
   breaks the acquire loop — the user leaves the app to see the gap they just closed.
3. **Auto-rebuild on staleness.** *Cons:* spends 7.1 s unasked, and **destroys the seeded-determinism
   property that is the feature's only verification surface** (zero frontend tests; screenshots time out).

### C — How does gap triage survive a rebuild? *(B14 — reopened after the grill's premise was disproved)*

1. **A separate user-override sidecar keyed on `(concept_id, kind)`; `gaps` stays purely derived.** *Pros:*
   the Enrichment-Layer's "regenerable" property is preserved exactly — `gaps` remains a pure function of
   (skeleton + claims); the override survives *any* regeneration; **this is precisely ADR-013's shape**
   (auto-extracted on the record, user override in a sidecar, **effective = override ?? default**). *Cons:* a
   new table + migration; two reads to render one status.
2. **Make the deterministic write path status-preserving** (mirror `_write_stochastic_gap_rows`' upsert).
   *Pros:* no new table; symmetric with the stochastic path. *Cons:* **`gaps` becomes a hybrid of derived and
   user data**, so it is no longer regenerable — the exact property ADR-004 relies on; and it needs a
   reconcile rule for gaps that stop firing (does a dismissed-then-resolved gap linger forever?).
3. **No triage for deterministic gaps** (only stochastic can be dismissed). *Cons:* **0 stochastic gaps exist
   today**, so this means *no triage at all*, and the view becomes a permanent nag about the same 14 findings.

## Decision

**A1 — the graph is read-only for the vocabulary and deep-links to Manage-keywords.** You edit the *source*
and regenerate; you never edit the derived artifact. The invariant keeps one home while PR-2.5 is still
repairing it, and a second writer onto known-broken write paths is not a risk worth taking for a convenience.

**B1 — an in-app Rebuild button, `202` + status-poll, mirroring `/api/ingest`.** 7.1 s is a button. The
**CLI runner remains canonical**: an API caller of the same idempotent function does not remove the runner,
never touches the chunk store, and is the established pattern for this repo's largest derived build. The
Enrichment-Layer Pattern constrains *what* derived data is (regenerable, sidecar, never mutating source) — not
*who is allowed to press go*.

**C1 — gap triage is a user override in its own sidecar, keyed on `(concept_id, kind)`; `gaps` stays purely
derived.** A dismissal is a **user judgment**, not derived data, so it does not belong in a table that is
deleted and rebuilt from the skeleton. `GapRow.status` becomes the **effective** value (`override ?? "surfaced"`)
at read time, exactly as `DocumentMeta` does for title/authors/year (ADR-013 A2). This is the only option that
keeps ADR-004's regenerable guarantee intact **and** makes a dismissal mean something across the acquire loop.

**Consequently:** the grill's B14 resolution stands as a *product* answer (the view can dismiss/promote) but
its *reasoning* is corrected here — status does **not** already survive rebuilds for the gaps we have; the
sidecar is what makes it survive.

## Consequences

**Makes easy.**
- The graph can be built and shipped without touching the vocabulary write path at all — it is a pure reader
  plus one rebuild trigger, so PR-2.5/2.6/2.7 and the graph cannot collide on `library.py`'s write functions.
- Staleness stops being a defect and becomes a **state with an action**: "the graph is N concepts behind →
  Rebuild". The empty state (a fresh clone has no `skeleton.json` — the *normal* first run) collapses into the
  same affordance.
- A dismissal is durable across the acquire loop (gap → acquire → ingest → rebuild), which is the only way the
  loop is not infuriating.
- The seeded, deterministic layout survives, keeping the one verification surface this feature has.

**Makes hard.**
- Fixing a concept you can see is a click away, not in place. Accepted: the deep-link preserves both the
  invariant and the honesty of the derived view.
- The triage sidecar is a **new table + a migration** (`db/migrations.py` `_ADDITIVE_COLUMNS` handles columns,
  not tables — a new table lands via `create_all`, the `figures` precedent).
- Rendering a gap's status now reads two places. Bounded: 14 rows today.

**Commits us to.**
- **The graph never writes `Concept`/`ConceptAlias`.** If a future increment wants in-graph curation, it
  reopens **A** here, and it must first answer what a write means to an artifact rebuilt wholesale.
- **`gaps` stays a pure function of (skeleton + claims).** Anything user-authored about a gap goes to the
  override sidecar. This is the line that keeps ADR-004 regenerable.
- **The rebuild trigger is a thin shell**: the route calls the same `build_concept_skeleton` seam the CLI
  calls; no logic in `apps/`.

**Reopens if.**
- Rebuild time grows superlinearly with the vocabulary (B1 assumes ~7 s; at 357 concepts this is unmeasured) —
  then the button becomes a background job, not a 202-poll.
- The deep-link proves too slow in practice for real curation work (A1).
- Stochastic gaps (`gap_suggest`, Tier-2a ceiling) start being produced in volume — their path is *already*
  status-preserving, so C1 must not double-write their status.

**Known limitation carried, not solved.** The gap payload this UI renders is **~50% precise** on the live
corpus (RG-014, 2026-07-17): `single_source` (3) is a true positive and the product thesis;
`unsourced_claim` (4) is real but built on input that is ~33% non-claims (12 of 61 `unsupported` claims are
markdown headings — a live segmenter bug); `under_connected` (5) is mostly an artifact of a 26-concept
vocabulary (`Tractography`, 10 docs, and `Motor control`, 13 docs — both among the best-sourced concepts — are
flagged as gaps); `thin_bridge` (2) flags **`Embeddings`**, the most-connected node in the graph, as "thin".
**These are detector-tuning and segmenter defects, not boundary decisions** — they belong to ADR-004's layer
and are tracked in RG-014 + the ui-checklist. This ADR only commits to *where the UI is allowed to write*; the
spec commits to *not leading with the weak kinds*.

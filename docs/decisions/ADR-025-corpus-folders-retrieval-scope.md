<!-- status: active · updated: 2026-07-20 · class: append-only -->

# ADR-025 — Corpus groups = folders, with query-time retrieval scoping

- **Status:** accepted (design-locked; unbuilt — carve F1→F2→F3 below)
- **Date:** 2026-07-20
- **Deciders:** Lucas (grilled via `grill-me`, 6 forks, all resolved or parked), with Claude Code

## Context

The demo collection (30papers, `0c777d8`) put a second corpus inside the single shared store, and
the user wants "several corpus groups" (demo vs personal papers vs future sets) without the cost of
a separate database. The shaping constraint is the **`is_archived` precedent**: doc-level flags
scope the library/wiki/keywords/citations reads but **not retrieval** — Chroma chunks carry only
`doc_hash`, so a "hidden" document is still retrieved and cited. Any grouping that scopes only the
Library grid would silently lie in chat. Meanwhile the schema already carries dormant
`Folder`/`document_folders` tables (0 rows; Library redesign Phase B planned them).

## Decision (one line per grilled fork)

1. **Corpus groups ARE folders** — reuse `Folder`/`document_folders` (many-to-many; overlap
   allowed), and a folder gains the power to scope chat retrieval. One organizing concept, one
   write surface (the ADR-015 reuse pattern). UI name: **Folders**; the demo corpus becomes an
   auto-created **"Demo corpus"** folder (ordinary folder — renamable, deletable).
2. **Demo membership is automatic at ingest + one-time backfill; user edits always win** — a newly
   ingested file whose SHA-256 matches a demo manifest pin joins the folder; assignment happens
   only at ingest of a new document, so a user removal is never re-fought (ADR-013 user-wins
   pattern). Reverses if the manifest must become the ongoing authority.
3. **Scoping = a query-time doc-hash filter on BOTH retrieval arms** — folder → member
   `doc_hash` set from SQLite per turn; vector arm via a Chroma `where doc_hash $in [...]` filter,
   BM25 arm by subsetting its in-memory corpus before scoring. **No chunk-store writes**
   (membership stays instantly editable); the unscoped path stays byte-identical. Bounds owed:
   `$in` latency at the 10k-doc contract; scoped BM25 statistics differ from global (correct, but
   record it) → **RG-020**. Reverses into per-folder precomputed indexes only if `$in` fails the
   latency measurement.
4. **Scope selection is per-turn, request-scoped, sticky in the UI only** (the ADR-010 override
   pattern; backend stores nothing; default = whole library). This is a **content filter (which
   documents), not a quality knob (how retrieval works)** — so it does not reopen ADR-010's
   governance wall and needs no eval gate. **Integrity requirement (non-negotiable): the
   provenance record and a visible answer chip state the scope every scoped turn.** Reverses into
   a persisted per-conversation scope only on demonstrated need (ADR amendment).
5. **Enrichment stays corpus-global in v1** — folders scope retrieval + Library only. The concept
   graph is already protected by ADR-018's opt-in vocabulary; scoping the sidecars (wiki, gaps,
   keywords, epistemics) multiplies rebuild semantics and is parked as its own future track.
   Reopener: if demo keywords measurably clutter the facet overlay, PR-2.7's rare-keyword
   demotion (<2 docs) is expected to absorb it before any new mechanism.
6. **The eval index-composition fingerprint is parked to RG-021** — benchmark runs are comparable
   only on the clean eval-10 index regardless of scoping (BM25/IDF statistics and the vector
   neighborhood are corpus-global even under a perfect filter), so the guard is eval-harness work,
   not folder work.

## Rejected

- **Separate database / fully separable corpus** — the cost is code complexity, not storage: every
  read path (BM25 stats, skeleton, wiki, gaps, citations) is corpus-global. Coarse isolation
  already exists via the env-level data home for whoever truly needs it.
- **A new `CorpusGroup` object beside folders** — two organizing mechanisms with one meaning.
- **A single-valued partition column** — can't express overlapping sets; duplicates folders.
- **Stamping folder ids into chunk metadata** — mutates the chunk store on every membership edit
  (violates the never-mutate contract).
- **Post-rerank filtering** — recall collapses exactly when the scope is small; the cheap-but-wrong
  option.
- **A persistent/global scope setting** — a forgotten global scope silently narrows every answer;
  the integrity layer exists to prevent that class of lie.

## Build carve (spec-at-build-time per house pattern; one PR per session)

- **F1 — folders end-to-end:** CRUD over the dormant schema + Library UI (create/rename/delete,
  add/remove docs, sidebar filter). The planned Phase B; demoable standalone.
- **F2 — retrieval scoping:** the doc-hash filter in `pipeline.py` (both arms) + per-turn scope
  param + composer selector + provenance record & answer chip. The integrity piece.
- **F3 — demo auto-assign:** ingest-time sha-match hook + the one-time backfill runner.

## Consequences

- **Easier:** "chat with this folder" lands on shipped scaffolding; the demo corpus is cleanly
  visible, scopable, and removable; converges with the queued folders/bulk-ops backlog rows.
- **Commits us to:** retrieval honesty — any future doc-level visibility feature must either scope
  retrieval or state on the answer that it doesn't (this ADR is the precedent to cite).
- **Costs:** per-turn membership resolution + filter plumbing in the hot path (bounded, measured
  under RG-020); a third meaning of "collection" avoided by naming the UI concept "Folders".
- **Reverses if:** the fork-level "reverses" clauses above; wholesale, if folders-with-power proves
  the wrong identity (e.g. deep nesting vs flat scope), split scope out into its own object then.

## Links

- ADR-010 (request-scoped overrides — the pattern fork 4 extends) · ADR-013 (user-wins) ·
  ADR-014 (safe delete) · ADR-015 (reuse-don't-duplicate) · ADR-018 (opt-in graph vocabulary) ·
  ADR-024 (evals home; the demo/eval regime split).
- `.claude/RIGOR_TODO.md` RG-020 (scoping bounds) · RG-021 (eval fingerprint).
- Full grill ledger: `.claude/SESSION.md` 2026-07-20 entry.

## Build status (appended 2026-07-20 — the carve is complete)

- **F1** — `3969adb`, spec `docs/specs/feature-corpus-folders.md`.
- **F2** — `0e45dd3` (+ follow-ups `dfe775e`), spec `docs/specs/feature-corpus-folders-scope.md`.
- **F3** — spec `docs/specs/feature-corpus-folders-demo.md`. Fork 2 is built as written: bytes, not
  names; only a *newly created* `Document` row is ever considered, so a hand removal is never
  re-fought. Two things fork 2 did not say, decided at build time and recorded as spec M5/M6: the
  folder is resolved by a **persisted id** (renaming it is respected — a name-keyed lookup would
  make a second "Demo corpus"), and deleting it is respected until a *new* demo document arrives
  (no tombstone). Fork 5 (per-folder enrichment) stays parked; RG-020's synthetic-10k half stays open.
- Found while building F3, logged not fixed: **KI-24** — `ingest --rebuild` cascades away all
  `document_folders` rows, so every folder is silently emptied. This ADR's membership model assumes
  membership survives a reindex; it currently does not.

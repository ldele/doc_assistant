<!-- status: design-locked · created: 2026-07-21 · owner: Code · plan: docs/PLAN_2026-07-21_exploration-epistemics.md (E0) -->

# Feature spec — E0: correctness batch (P0s before the epistemics surfaces)

Build contract for ROADMAP row **E0**. Turns the five bullets of
`docs/PLAN_2026-07-21_exploration-epistemics.md` §E0 into a code-level sprint. These are the C4
review's P0 correctness fixes (`docs/REVIEW_2026-07-19_scale-robustness.md`) plus one boot item.

**Why this sprint, why now.** ADR-027 makes the epistemics **assessment** always-on (D3: a
per-source evaluation strip under every answer). *An always-on strip must not show false data.* E2–E5
(the surfaces) all render data these fixes make trustworthy; three of the five (KI-20/21/17) are the
same "a rebuild/curation silently destroys curated state" class that emptied the graph via KI-25 in
the last arc. Foundation first.

**No eval ceremony** — none of these touch a locked retrieval/chunking setting, so no `--repeat`
baseline is owed (that governs the KI-19 tuning constants, which are **out of scope** here). The
gate is: each fix ships with a **guard test that fails against today's code** (non-vacuous), full
suite green, and a live $0 probe where a data path is touched.

**Out of scope (deferred, recorded):** E1 (KI-8 marker re-projection + `_handle_rag` extraction) is
the next sprint; the KI-18 scale hot-paths and KI-19 tuned constants are measurement-gated
(RG-016..019), not correctness — do not hand-tune them here.

---

## Items

### E0.1 — Curation demotes, never deletes (KI-20 / review CS-5)

- **Defect.** `knowledge/concept_curation.py:400` `remove_concepts` hard-deletes `Concept` +
  `ConceptAlias` rows; stages 1–3 (`artifact filter`, `classify_noise` LLM, near-dup merge) route
  into it. Deleting a `Concept` also deletes its **keyword family** (ADR-015 shared table) and
  cascades into presence/edges/gaps. `classify_noise` is exactly the path that mislabels real
  specialist vocabulary (`cre`/`dbs`/`ntsr1`/`pddl` — the trap hit twice). Violates ADR-018's
  **demote** verb.
- **Fix.** Route noise/artifact verdicts through `set_graph_include(id, False)` (keep the row + its
  family; just drop it from the graph vocabulary). Reserve real deletion for an explicit,
  separately-confirmed path, not the noise classifier.
- **Files.** `knowledge/concept_curation.py`.
- **DoD / guard test.** A `classify_noise`-flagged concept still has its keyword family (and row)
  after `--apply`; assert it's `graph_include=False`. Fails today (row is gone).
- **Risk.** Low — stages 1–3 have never been `--apply`-run on the real corpus (dry-run default), so
  this is closing a contract hole, not repairing live loss.

### E0.2 — Rebuild reconciles orphaned stochastic gaps (KI-17 / review GP, ADR-017 C1)

- **Defect.** `_write_stochastic_gap_rows` (`knowledge/gaps.py:273`) is a status-preserving upsert
  with **no delete pass** for rows whose concept left the vocabulary; deterministic rows self-heal
  (`gaps.py:257` delete-and-replace) but stochastic ones are immortal. Live: `load_graph_view`
  serves **27** gaps against a **13**-node skeleton (10 orphans from the pre-ADR-018 357-concept
  vocabulary).
- **Fix.** A reconcile pass that deletes stochastic gap rows whose `concept_id` is **not** in the
  current `graph_include`-filtered `load_concepts()`. **Hoist it to run unconditionally on every
  `build_gaps --apply`** — the review's placement correction: inside `_write_stochastic_gap_rows` it
  only runs under `suggest and apply` and early-returns on zero suggestions, so a deterministic-only
  `--apply` (KI-17's own repro) never reaches it. A reconcile, not a blanket delete — triage on a
  *live* concept survives.
- **Files.** `knowledge/gaps.py`.
- **DoD / guard test.** A stochastic gap on a concept then excluded (`set_graph_include(cid,False)`)
  → `build_gaps --apply` → the row is gone; a stochastic gap on an *included* concept keeps its
  status. Fails today.
- **Sequencing.** Land with E0.3 (both concern what a rebuild must refresh) — decide alongside the
  ADR-017 C1 triage-override sidecar if that lands here.

### E0.3 — In-app rebuild refreshes gaps, not just the skeleton (KI-21 / review GP-4, ADR-017 B1)

- **Defect.** The B1 rebuild route `_default_rebuild_graph` (`apps/api/main.py:232`) calls
  `build_concept_skeleton(apply=True)` only; `build_gaps` has **no API caller**, and `load_graph_view`
  serves `GapRow`s with no `graph_version` cross-check. After an in-app rebuild the UI shows gaps from
  the *previous* skeleton — including the gap the user just closed. The acquire loop the button exists
  to close ("gap → ingest → rebuild → gap closes") does not close in-app.
- **Fix.** Chain `build_gaps(apply=True, min_degree=<runtime-derived>)` after the skeleton build in
  the route. The `min_degree` default is the KI-19/GP-1 runtime-Q1 question — for E0, derive it from
  the rebuilt skeleton's own degree distribution (no hardcoded constant) or lift the CLI's derivation;
  do **not** freeze a literal. Alternatively/additionally stamp `graph_version` on gap rows and filter
  mismatches in the view.
- **Files.** `apps/api/main.py` (route), `knowledge/gaps.py` / `knowledge/concept_graph_view.py`.
- **DoD / guard test.** After the rebuild route runs, `load_graph_view` gap count == the
  `build_gaps` report count (no stale/orphan inflation), and a just-closed gap does not reappear.
  Route-level integration test with a fake/real skeleton.
- **Note.** This is the one item on the **answer/serving path** — no paid LLM (skeleton + gaps are
  deterministic, zero-LLM; measured rebuild ≈7 s). Keep it a 202 + poll like the existing route.

### E0.4 — Zero-doc honesty, pinned by a test (WE-1 / WE-9 / GP-7)

- **Defect.** Read paths degrade honestly at 0 docs, but two build paths don't, and nothing pins the
  contract: `wiki.load_doc_graph` (`knowledge/wiki.py:329`) throws an uncaught `OperationalError` on a
  never-ingested DB; `epistemics` build (`epistemics.py:418/445`) raises on a missing skeleton for any
  non-CLI caller and `--apply` on a never-migrated DB is an uncaught `OperationalError`; **no
  empty-input guard test exists** (GP-7) so the 0-doc contract survives by habit, not by gate.
- **Fix.** `load_doc_graph` catches → `([], [])` + a "no documents indexed" hint; the epistemics build
  path returns an empty result + hint and guards `_write_rows`. Add **one parametrized empty-input
  test over all detectors** (gaps, epistemics, wiki, skeleton) asserting an honest empty return, never
  a raise.
- **Files.** `knowledge/wiki.py`, `knowledge/epistemics.py`, `tests/unit/test_*` (new empty-input
  parametrization).
- **DoD / guard test.** The parametrized 0-doc test is the DoD; it fails today on `wiki`/`epistemics`.
- **Robustness contract tie-in.** This is the `.claude/CONTEXT.md` "degrade honestly at 0 documents"
  non-negotiable, finally gated rather than assumed.

### E0.5 — Migration failure fails at boot, and a plain `--apply` doesn't wipe Node B stance

Two boot/rebuild footguns, grouped because both are "a maintenance action silently corrupts the
answer path."

- **E0.5a — `init_db()` fails fast at boot.** The lifespan (`apps/api/main.py:321-328`) currently
  swallows a migration exception (`except Exception: log.error(...)`, then serves anyway). But a
  failed **answer-path** migration (the reason KI-23's fix moved `init_db` into the lifespan) then
  breaks *every turn* at runtime — a worse, later, more confusing failure than refusing to start.
  **Fix:** on a migration failure, fail the boot (raise / refuse to mark ready) with a clear message,
  rather than starting a server that 500s every turn. This deliberately **reverses** the current
  "never let a migration problem stop the app" comment — record why in the code + an ADR note if the
  reviewer wants it. *Also fix the stale `apps/api/CLAUDE.md` line that still says "the API does not
  `init_db()` on startup".*
- **E0.5b — stance-preserving rebuild.** `build_concept_skeleton --apply` **without** `--enrich`
  silently wipes Node B's `stance_json`/`relation` — epistemics then degrades corpus-wide (this is
  the G6-run trap, still open). The in-app rebuild route (E0.3) must **not** carry it. `--enrich`
  needs Ollama (RTX box, KI-4), so auto-enrich is not viable on the CPU/dev box → **preserve existing
  stance on a plain `--apply`** (don't delete what you're not regenerating), or refuse with a message.
- **Files.** `apps/api/main.py` (lifespan + CLAUDE.md), `knowledge/concept_skeleton.py` /
  `scripts/build_concept_skeleton.py`.
- **DoD / guard test.** (a) a genuinely broken migration → the app refuses to start (test the lifespan
  raises, not swallows). (b) a skeleton with Node-B stance → `build_concept_skeleton(apply=True)`
  without enrich → stance rows are **preserved** (fails today: they're wiped).

---

## Build order & gate

1. **E0.4** (zero-doc test + guards) first — it's the safety net the others rebuild on top of.
2. **E0.1** (demote-not-delete) — self-contained.
3. **E0.2 + E0.3** together (rebuild coherence — reconcile + gaps-in-route).
4. **E0.5a** then **E0.5b**.

Per item: a **non-vacuous guard test** (fails against current code), then `ruff` / `ruff format` /
`mypy --strict src` / `bandit` / full `pytest`; a live $0 probe for E0.2/E0.3/E0.5b on a **copy** of
the real `data/library.db` (never the original — the graph was lost once already). One `docs/DEVLOG.md`
entry per item; update the KI (KI-17/20/21 → Resolved when their guard lands) and the ROADMAP E0 row.

## Open question for the reviewer
E0.3's `min_degree` derivation and E0.2's reconcile touch the **same** rebuild-refresh seam ADR-017 C1
reserves for the gap-triage override sidecar. If C1 is being built imminently (PR-G2b), land the
reconcile *with* it; if not, E0 ships the reconcile standalone and C1 inherits it. Decide at sprint
start.

<!-- status: built · updated: 2026-07-23 · owner: Code · plan: docs/decisions/ADR-028-concept-taxonomy-polyhierarchy-skos.md (increment 1) -->

> **BUILT 2026-07-23 (staged) — one caveat.** All the engineering landed and is gate-green: the `kind`
> column + additive migration (T1a), the `concept_hierarchy` + `document_field` tables (T1b/c),
> `knowledge/taxonomy.py` (T2), `scripts/seed_taxonomy.py` + `data/anzsrc_2020_for.json` (T3), and 14
> guard tests (`tests/unit/test_taxonomy.py`) — `ruff`/`mypy --strict src`/`bandit` clean, full suite
> **1236 passed**. **The one deviation from the DoD:** the seed data file ships the **23 ANZSRC divisions
> (verified) but not the 213 four-digit groups** — those need an *authoritative* import (the CC-BY
> provenance + accuracy bar rules out transcribing 213 government codes from memory or a summarizing
> fetch). Guard test 8 asserts idempotency + structure against the file's *actual* contents, so it stays
> green and re-validates automatically once the groups land. Sourcing the 213 groups is the open decision
> — see the DEVLOG entry 2026-07-23 and T3 below.

# Feature spec — Taxonomy increment 1: seed + schema

Build contract for the **first increment** of [ADR-028](../decisions/ADR-028-concept-taxonomy-polyhierarchy-skos.md)
(the concept-taxonomy amendment). This lands the durable **substrate** — the node-kind flag, the two
new curated tables, the acyclicity-guarding write module, and the ANZSRC seed — that every later
increment (curation UI, auto-propose, coverage) builds on. **Backend + data only, $0, zero-LLM.**

## Why now
ADR-028 is accepted; nothing is built. The schema is the foundation and is fully decided, so it ships
first, alone — a curation UI or auto-propose pass has nothing to write against until the tables and the
seeded field trunk exist. No LLM is involved: the seed is deterministic CC-BY data; auto-propose is a
later increment.

## Scope
In: the `kind` column, `concept_hierarchy` + `document_field` tables, `knowledge/taxonomy.py` (write
seam + acyclicity + `presence_nodes()` accessor), `scripts/seed_taxonomy.py` + the bundled ANZSRC data
file with its CC-BY attribution header. Out: everything in "Out of scope" below.

## Items

### T1a — `kind` column on `concepts` (additive migration)
- **Model** (`db/models.py`): `kind: Mapped[str] = mapped_column(String, nullable=False, default="concept")`
  on `Concept`. Values: `"concept"` (a text-bearing concept) | `"domain"` (an abstract field node, zero
  presence). Add a short docstring line: presence-assuming code must read only `kind="concept"` (via T2's
  accessor).
- **Migration** (`db/migrations.py` `_ADDITIVE_COLUMNS`): append
  `("concepts", "kind", "VARCHAR NOT NULL DEFAULT 'concept'", "ix_concepts_kind")`. The literal
  `DEFAULT 'concept'` **backfills existing rows** to `concept` in the same `ALTER TABLE ADD COLUMN`.
  This is the **KI-25 discipline made explicit**: an additive column whose NULL/absent value would
  change behaviour must ship its backfill in the same change — here every pre-existing `Concept` is a
  concept, never a domain, so the default is both safe and correct.

### T1b — `concept_hierarchy` table (new; curated; survives a skeleton rebuild)
- **New model** `ConceptHierarchy` (`__tablename__ = "concept_hierarchy"`): `id` (uuid PK),
  `source_id` (FK `concepts.id`, `ondelete="CASCADE"`, indexed), `target_id` (FK `concepts.id`,
  `ondelete="CASCADE"`, indexed), `type` (String, `"is_a"` | `"in_field"`), `created_at`.
  `UniqueConstraint("source_id", "target_id", "type")`. Created by `create_all` (additive — no
  migration).
- **Semantics:** `source —is_a→ target` (concept→broader concept); `source —in_field→ target`
  (concept→field, or field→field). Polyhierarchy-native (many rows per `source_id`).
- **The load-bearing property:** this table is **curated user data and MUST NOT be dropped** by
  `build_concept_skeleton`. It lives beside `Concept`/`ConceptAlias` (which survive a rebuild), *not*
  in the derived `concept_edges` (which is dropped + rebuilt every run). Storing the hierarchy in
  `concept_edges` would let a routine rebuild wipe it — the KI-17/KI-20 class.

### T1c — `document_field` table (new; curated + later auto-proposed)
- **New model** `DocumentField` (`__tablename__ = "document_field"`): `id` (uuid PK), `document_id`
  (FK `documents.id`, CASCADE, indexed), `concept_id` (FK `concepts.id`, CASCADE, indexed — a
  **`kind="domain"`** node), `origin` (String, `"curated"` | `"proposed"`, default `"curated"`),
  `created_at`. `UniqueConstraint("document_id", "concept_id")`. Many-to-many. Created by `create_all`.
- The `origin` flag is the ADR-019 E1 seam: `proposed` rows are auto-fill the user can override;
  `curated` always wins. (Auto-propose itself is a later increment.)

### T2 — `knowledge/taxonomy.py` (the write seam; thin-shell rule — all logic here)
Pure/core helpers over a session; no LLM, no network:
- `add_hierarchy_edge(session, source_id, target_id, edge_type)` — **rejects an edge that would create
  a cycle** in the `is_a ∪ in_field` DAG (build the incident subgraph, test with
  `networkx.is_directed_acyclic_graph`; raise `TaxonomyCycleError` with the offending path). Idempotent
  on the unique key. This is where ADR-028 Decision 3's acyclicity invariant is enforced.
- `remove_hierarchy_edge(...)`, `attach_document_field(session, document_id, field_id, origin)` —
  validates `field_id` resolves to a `kind="domain"` node (raise if not).
- `presence_nodes(session) -> list[Concept]` — **the single canonical accessor** returning only
  `kind="concept"` nodes. ADR-028 Decision 4's centralised guard: presence/gap detectors call this, so
  the domain-exclusion lives in one place, not N scattered `WHERE kind='concept'` clauses.
- `load_taxonomy(session) -> nx.DiGraph` — the hierarchy as a build-time NetworkX DiGraph (nodes carry
  `kind`; edges carry `type`) for later coverage/traversal. Read-only.

### T3 — `scripts/seed_taxonomy.py` + the ANZSRC data file (thin CLI over T2)
- **Data file** `data/anzsrc_2020_for.json` (bundled, tiny): the **23 divisions + 213 groups** with
  their 2-/4-digit codes and labels, produced from the ANZSRC 2020 source. **Its header carries the
  CC-BY attribution** (source, licence version, URL) — a licence obligation, not optional (ADR-028 D7).
- **Script** (mirrors `scripts/seed_concepts.py`): reads the data file, writes each row as a
  `Concept(kind="domain", label=…, source="anzsrc")` keyed on a **stable id derived from the ANZSRC
  code** (so re-runs match, not duplicate), and each group→division as an `in_field` edge via
  `taxonomy.add_hierarchy_edge`. **Idempotent** (re-run = no-op: same ~236 domains, same edges).
  **Dry-run default**; `--apply` writes. Prints the attribution notice on run.

### T4 — attribution (in-scope: data-file header; flagged: UI surfacing)
The seed data-file header + the script's printed notice satisfy the *source-level* CC-BY obligation for
this increment. The **About/Settings UI surfacing** of the attribution is **required before the taxonomy
ships to users** but is a frontend increment — flagged here, tracked as a follow-up, not built now.

## DoD / guard tests (each fails against today's code)
1. **Migration backfill:** on a DB with pre-existing `Concept` rows lacking `kind`, the migration adds
   `kind` and every existing row reads `"concept"` (not NULL). (Non-vacuous: disable the migration → fails.)
2. **Tables created:** `create_all` produces `concept_hierarchy` + `document_field` with the stated
   columns + unique constraints.
3. **Cycle rejected:** `add_hierarchy_edge` A→B then B→C then C→A raises `TaxonomyCycleError`; the first
   two succeed.
4. **Polyhierarchy allowed:** a concept may take two `in_field` parents (two edges, no error).
5. **`presence_nodes()` excludes domains:** given one `kind="domain"` + one `kind="concept"` row, the
   accessor returns only the concept.
6. **Rebuild preserves the hierarchy (the KI-17/KI-20 guard, the whole point):** seed a `concept_hierarchy`
   edge, run a `build_concept_skeleton` rebuild, assert the edge still exists (while `concept_edges` was
   dropped + rebuilt).
7. **`document_field` rejects a non-domain target:** `attach_document_field` to a `kind="concept"` node raises.
8. **Seed idempotent:** `seed_taxonomy --apply` twice → 23 domains at the division level, 213 at the group
   level, 213 `in_field` edges, unchanged on the second run.
9. **Attribution present:** the bundled data file's header contains the CC-BY notice (source + licence + URL).

## Build order & gate
Model + migration (T1a) → new tables/models (T1b/c) → `taxonomy.py` (T2) → data file + seed script (T3)
→ attribution header (T4). Full gate green: `ruff` / `ruff format` / `mypy --strict src` / `bandit`
0 HIGH·MED / `pytest`; `docs_check` / `integrity_check` 0/0. **$0, zero-LLM** — no provider guard needed.

## Out of scope (deferred — later increments)
- **Curation UI** (edit the DAG in-app) — the next increment.
- **Auto-propose** `in_field` parents + `document_field` assignments (ADR-019 E1, $0/Ollama, KI-4) — a
  later increment; `origin="proposed"` + `presence_nodes()` are the seams it will use.
- **Coverage math surfaces** (set-semantics rollup, the "N papers under ML" line) — later; `load_taxonomy`
  is its substrate.
- **On-demand MeSH/ACM grafts** — a curation action, later.
- **About/Settings attribution UI** — required before ship (T4), a frontend increment.
- **The epistemic-health detector layer** — its own ADR (ADR-EH), gated on measurement (RG-023).

<!-- status: active · updated: 2026-07-20 · class: append-only -->

# ADR-026 — Rebuild migrations: how a table's *shape* changes, starting with `document_meta`'s missing FK

- **Status:** accepted (built 2026-07-20)
- **Date:** 2026-07-20
- **Deciders:** Lucas, with Claude Code

## Context

`document_meta` (the ADR-013 user metadata overrides) shipped with `document_id` as a bare
`String` primary key — **no foreign key**. Correctness was maintained by convention: `library.py`
`delete_document` deletes the override by hand, and its docstring says so ("no FK — explicit").

Every *bulk* path forgot. `cleanup_orphans_sqlite` deletes `Document` rows on every incremental
ingest that finds a gone or content-changed source, and the pre-KI-24 `--rebuild` deleted all of
them; neither touched `document_meta`. The result was rows referencing ids that no longer exist —
unreadable (every read path resolves through a live document), unclean-up-able, and accumulating.
KI-24 recorded this as still-open after fixing the rebuild, because the ingest orphan sweep was
still producing them.

The blocker was mechanical: **SQLite cannot `ALTER TABLE … ADD CONSTRAINT`.** The project's
migration mechanism (`db/migrations.py`) is deliberately additive-only — `create_all` for new
tables, `_ADDITIVE_COLUMNS` for new columns on existing ones — and neither can add a foreign key
to a column that already exists. So fixing this required deciding *how this project changes a
table's shape at all*, not just fixing one table.

## Decision

1. **Add the foreign key**: `document_meta.document_id → documents.id ON DELETE CASCADE`. The
   parent table always existed; there was never a reason not to point at it. `delete_document`
   keeps its explicit override delete — now redundant, retained so the ADR-014 path still reads as
   the complete story.
2. **Introduce *rebuild migrations* as a second, deliberately rarer mechanism.** SQLite's
   documented table-rebuild dance, in `db/migrations.py::_rebuild_table`: foreign keys off, one
   transaction, create the model's shape under a temp name, copy the rows worth keeping, drop,
   rename, `PRAGMA foreign_key_check`, commit. The new shape is rendered **from the SQLAlchemy
   model**, never hand-written DDL, so it cannot drift from `create_all`. Only columns present in
   both the live table and the model are copied, so a rebuild is safe against a schema that
   predates an additive column.
3. **Rebuilds are rare by policy.** Each is a *named function* (not a data-driven list like
   `_ADDITIVE_COLUMNS`), is idempotent — it inspects the live schema and no-ops when already
   correct — and **must be justified in an ADR**. Additive stays the default; this is the escape
   hatch, not a second lane.
4. **Rows that the new constraint forbids are dropped, and logged in full first.** An orphaned
   override cannot be carried over (it is precisely what the FK forbids) and cannot be rescued (it
   records no filename or hash — only a dead id). `init_db` returns what it changed, including the
   orphan count, so the KI-23 startup log states it.
5. **A failed rebuild changes nothing.** The destructive middle is one transaction with an explicit
   rollback, so a row that cannot satisfy the new shape leaves the old table exactly as it was —
   the migration refuses rather than inventing a value.

## Rejected

- **Keep compensating in application code** — add the `document_meta` delete to the orphan sweep
  and to `_sweep_rebuild_rows`. That is the same bet that already failed twice: correctness by
  every caller remembering. A third caller would forget too.
- **A periodic orphan-cleanup pass** over `document_meta` — sweeping up after a defect instead of
  making it impossible, and it would still leave a window where reads are wrong.
- **Alembic** — real migration tooling for a single-file local SQLite app is a dependency, a
  version table, and a workflow to maintain, for what is currently one non-additive change. The
  reopener is explicit: **if a second or third rebuild lands, adopt Alembic** rather than growing a
  hand-rolled framework.
- **A data-driven `_TABLE_REBUILDS` registry** mirroring `_ADDITIVE_COLUMNS` — a framework for
  n=1, and it would make rebuilds feel as routine as adding a column. Named functions keep the
  cost visible.
- **Leaving the orphans in place** after adding the FK — the constraint would then be a claim the
  data does not support, and `PRAGMA foreign_key_check` would report it forever.
- **Fixing `ConversationMeta` the same way** — it cannot have an FK. Conversations are *derived* by
  grouping `AnswerRecord` rows; there is no table to point at. Its `session_id` staying bare is
  correct, not the same defect.

## Consequences

- **Easier:** every future bulk delete of a `Document` is automatically correct; one less rule to
  remember. The mechanism now exists for the next constraint that needs fixing.
- **Commits us to:** ADR-justifying each rebuild, and to the model being the single source of a
  table's shape (hand-written DDL in a rebuild would defeat the whole thing).
- **Costs:** a rebuild rewrites a table — fine at `document_meta`'s size, and it is worth checking
  before applying the pattern to a large one.
- **Reverses if:** a second/third rebuild lands → adopt Alembic instead of extending this.

## Links

- ADR-013 (user-wins overrides — what `document_meta` holds) · ADR-014 (safe delete —
  `delete_document`'s explicit path) · ADR-025 (folders; the sibling `document_folders` always had
  its FK) · `.claude/KNOWN_ISSUES.md` KI-24 (where the orphaning was recorded) / KI-23 (the
  startup migration log this reports through).
- `tests/integration/test_document_meta_fk_migration.py` — builds a genuinely pre-migration table
  and drives `init_db` over it.

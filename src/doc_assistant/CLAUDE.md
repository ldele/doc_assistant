# src/doc_assistant/ — backend library (all business logic)

**Owns:** every piece of business logic — RAG pipeline, document store, enrichment sidecars.
`apps/` render it; `scripts/` drive it; neither owns logic (non-negotiable #3).

**Layout**
- Top level — the RAG answer path: `pipeline.py` (hybrid retrieval + rerank), `chat_controller.py`
  (turn orchestration), `llm.py` (provider-agnostic clients), `synthesis.py`, `provenance.py`,
  `reviewer*.py`, `prompts.py`, `config.py`, plus app services (`library.py`, `conversations.py`,
  `app_settings.py`, `compare.py`, `health.py`, `export.py`).
- `db/` — SQLAlchemy models + session + **additive** migrations (`_ADDITIVE_COLUMNS`).
- `ingest/` — extract → markdown → chunk → embed → store (locked path) + registry/cache/figures/tables.
- `eval/` — the eval harness (runner, scorers, cases, store).

**Rules that bite here**
- **Locked settings** live in `config.py` — change only via an eval-harness experiment
  (`.claude/CONTEXT.md` table). Enrichment modules are sidecars: additive tables/files, idempotent,
  never touch the chunk store.
- `structlog` only, no `print()` (ADR-003); library code never configures logging.
- **Robustness contract:** every module must handle an empty corpus (0 docs) without crashing and
  avoid corpus-tuned constants — thresholds derive from data or are named structural constants.
- `mypy --strict` is the bar; exceptions chain (`raise X from e`).

**Tests:** `tests/unit/` + `tests/integration/` (mirror module names).

<!-- Keep <=40 lines. Local only. If you're restating a project-wide rule, delete it and cite the code. -->

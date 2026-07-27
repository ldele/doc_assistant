# src/doc_assistant/ — backend library (all business logic)

**Owns:** every piece of business logic — RAG pipeline, document store, enrichment sidecars.
`apps/` render it; `scripts/` drive it; neither owns logic (non-negotiable #3).

**Layout (ADR-023)**
- Top level — the RAG answer path: `pipeline.py` (hybrid retrieval + rerank), `chat_controller.py`
  (turn orchestration), `llm.py` (provider-agnostic clients), `synthesis.py`, `provenance.py`,
  `reviewer*.py`, `prompts.py`, `config.py`, plus app services (`conversations.py`,
  `app_settings.py`, `compare.py`, `health.py`, `export.py`) and `doc_vectors.py`.
- `library/` — the document-store API, one module per sub-domain, named to match
  `apps/api/routers/library/`: `models` · `documents` · `pins` · `folders` · `keywords` ·
  `chunks` · `citations` · `similarity`. `__init__` re-exports flat for the existing callers.
- `db/` — SQLAlchemy models + session + **additive** migrations (`_ADDITIVE_COLUMNS`).
- `ingest/` — extract → markdown → chunk → embed → store (locked path) + registry/cache/figures/tables.
- `knowledge/` — the corpus-derived layer: keywords/families, concept skeleton (Node A/B) +
  curation/semantics/graph view, wiki, gaps, epistemics. All Enrichment-Layer sidecars; the answer
  path reads it, never depends on it.
- `eval/` — the eval harness (runner, scorers, cases, store).

**Rules that bite here**
- **Locked settings** live in `config.py` — change only via an eval-harness experiment
  (`.claude/CONTEXT.md` table). Enrichment modules are sidecars: additive tables/files, idempotent,
  never touch the chunk store.
- `structlog` only, no `print()` (ADR-003); library code never configures logging.
- **Robustness contract:** every module must handle an empty corpus (0 docs) without crashing and
  avoid corpus-tuned constants — thresholds derive from data or are named structural constants.
- Strict typing is the bar (`[tool.mypy] strict=true`) — run **`uv run --no-sync mypy src`**,
  **never `mypy --strict src`**: the flag changes the option set, so it invalidates mypy's cache both
  ways and makes the next commit's hook take ~40s instead of ~2s (add
  `--cache-dir .mypy_cache-strict` if you do want it). Exceptions chain (`raise X from e`).
- **Monkeypatch the module that OWNS a helper**, never a package that re-exports it: a
  re-exported name is a *separate binding*, so patching `doc_assistant.library.<name>` leaves
  the real caller untouched and the test silently runs the real thing (verified — it would
  have opened a file manager). Patch `library.documents._reveal_in_file_manager`.

**Tests:** `tests/unit/` + `tests/integration/` (mirror module names).

<!-- Keep <=40 lines. Local only. If you're restating a project-wide rule, delete it and cite the code. -->

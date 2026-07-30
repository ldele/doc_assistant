# src/doc_assistant/ — backend library (all business logic)

**Owns:** every piece of business logic — RAG pipeline, document store, enrichment sidecars.
`apps/` render it; `scripts/` drive it; neither owns logic (non-negotiable #3).

**Layout (ADR-023)**
- Top level — the RAG answer path: `pipeline.py` (hybrid retrieval + rerank), `sparse_index.py`
  (the keyword arm, on disk — ADR-036; `bm25_cache.py` is its legacy fallback), `llm.py`,
  `synthesis.py`, `provenance.py`, `reviewer*.py`, `prompts.py`, `config.py`, `doc_vectors.py`, plus
  app services (`conversations` · `app_settings` · `credentials` · `readiness` · `compare` ·
  `health` · `export`).
- `chat_controller/` — turn orchestration: `session` · `views` · `events` · `helpers` · `controller`.
- `library/` — document-store API, sub-domains matching `apps/api/routers/library/`: `models` ·
  `documents` · `pins` · `folders` · `keywords` · `chunks` · `citations` · `similarity`. Both
  packages re-export flat from `__init__`.
- `db/` — SQLAlchemy models + session + **additive** migrations. `ingest/` — extract → markdown →
  chunk → embed → store (locked) + registry/cache/figures/tables. `eval/` — the eval harness.
- `knowledge/` — corpus-derived layer: keywords/families, concept skeleton (Node A/B) +
  curation/semantics/graph view, wiki, gaps, epistemics. All sidecars; the answer path reads it,
  never depends on it.

**Rules that bite here**
- **Locked settings** live in `config.py` — change only via an eval-harness experiment
  (`.claude/CONTEXT.md` table). Sidecars stay additive, idempotent, off the chunk store.
- `structlog` only, no `print()` (ADR-003); library code never configures logging.
- **Never read `config.ANTHROPIC_API_KEY` at a call site** — resolve per construction via
  `credentials.resolve_key` (ADR-034), else an in-app key is silently missed. Never log key material.
- **Robustness contract:** handle an empty corpus (0 docs) without crashing; no corpus-tuned
  constants — derive thresholds from data, or name them structural.
- Type-check with **`uv run --no-sync mypy src`**, never `--strict` (`.claude/CONTEXT.md` §8);
  exceptions chain (`raise X from e`).
- **Monkeypatch the module that OWNS a name**, never a package that re-exports it — a re-export is a
  *separate binding*, so patching `library.<name>` silently misses (66 tests broke this way in the
  `chat_controller` split): use `library.documents._reveal_in_file_manager`,
  `chat_controller.controller.is_library_query`, `chat_controller.helpers.SYNTHESIS_MODE`. Setting an
  *attribute on a shared module object* (`app_settings.SETTINGS_PATH`) is fine through any binding;
  `sparse_index.fingerprint` imports the tokeniser per call for the same reason.

**Tests:** `tests/unit/` + `tests/integration/` (mirror module names).
<!-- Keep <=40 lines. Local only. If you're restating a project-wide rule, delete it and cite the code. -->

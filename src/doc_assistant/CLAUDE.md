# src/doc_assistant/ — backend library (all business logic)

**Owns:** every piece of business logic — RAG pipeline, document store, enrichment sidecars.
`apps/` render it; `scripts/` drive it; neither owns logic (non-negotiable #3).

**Layout (ADR-023)**
- Top level — the RAG answer path: `pipeline.py` (hybrid retrieval + rerank), `sparse_index.py`
  (the keyword arm, on disk and the only one since ADR-038), `llm.py`, `synthesis.py`,
  `provenance.py`, `reviewer*.py`, `prompts.py`, `config.py`, `doc_vectors.py`, plus app services
  (`conversations` · `app_settings` · `credentials` · `readiness` · `compare` · `health` · `export`).
- `chat_controller/` — turn orchestration: `session` · `views` · `events` · `helpers` · `controller`.
- `library/` — document-store API, sub-domains matching `apps/api/routers/library/`: `models` ·
  `documents` · `pins` · `folders` · `keywords` · `chunks` · `citations` · `similarity` ·
  `source_view` (ADR-050). Both re-export flat from `__init__`. `reingest.py` — per part (ADR-048).
- `db/` — SQLAlchemy models + session + **additive** migrations. `ingest/` — extract → markdown →
  chunk → embed → store (locked) + registry/cache/figures/tables. `eval/` — the eval harness.
  `adapters/` — optional registry producers, never a dependency (ADR-049): `catalogue` neutral,
  `zotero` the one module allowed to know a vendor schema.
- `knowledge/` — corpus-derived sidecars: keywords/families, concept skeleton (Node A/B) + curation,
  wiki, gaps, epistemics. **Read `docs/knowledge-layer.md` first** — `contested` is NOT a measurement.

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

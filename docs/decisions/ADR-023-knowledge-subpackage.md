<!-- status: active · updated: 2026-07-19 · class: append-only -->

# ADR-023 — Backend restructure: a `knowledge/` subpackage for the corpus-derived feature layer

- **Status:** accepted
- **Date:** 2026-07-19
- **Deciders:** Lucas (directive: "a subfolder for the database models, one for the ingestion
  pipeline files and one for the concept graph and wiki; RAG pipeline files stay"), executed with
  Claude Code

## Context

`src/doc_assistant/` had grown to **63 modules**, 40+ of them flat at the package top level. Three
subsystems already had homes (`db/`, `ingest/`, `eval/`), but the Phase-7 knowledge layer — the
concept graph, mined keywords, wiki notes, gap detection, epistemics — sat as 11 flat modules
interleaved with the RAG answer path. Cost at scale: an agent (or human) touching "the concept
graph" had no boundary to stay inside, and the flat listing no longer communicated the
architecture. cpc §12 ("flat by default — a real subsystem boundary earns its layer"): this
boundary is real — the whole cluster follows one pattern (Enrichment-Layer sidecars derived from
the corpus), is rebuilt by `scripts/` runners, and the answer path reads it but never depends on it.

## Options

1. **`knowledge/`** — names what the cluster *is* (the derived knowledge layer: vocabulary,
   skeleton, wiki, gaps, epistemics). **Chosen.**
2. **`features/`** — the directive's literal word; rejected as a Python name (generic, collides
   conceptually with `docs/features/`, and says nothing an import reader can use).
3. **`concepts/` or `graph/`** — reject: excludes wiki/keywords (concepts/) or wiki/epistemics
   (graph/); the cluster ships together.
4. **Leave flat** — reject: the directive, and the cost above.

## Decision

**Move 11 modules to `src/doc_assistant/knowledge/`** (git mv, history preserved):
`concept_curation`, `concept_graph_view`, `concept_semantics`, `concept_skeleton`,
`concept_skeleton_enrich`, `epistemics`, `gap_suggest`, `gaps`, `keyword_families`, `keywords`,
`wiki`.

**Deliberately kept at top level:** `synthesis.py` (answer-path dual interpretation — Integrity
Chunk 2a, not corpus-derived), `tracking.py` (LLM token infra), `doc_vectors.py` (Phase-4
similarity enrichment that *feeds* the skeleton but predates and outlives it — an input, like
`ingest/citations.py`), and the whole RAG answer path per the directive. `db/`, `ingest/`, `eval/`
unchanged — the directive's "database models" and "ingestion pipeline" subfolders already existed.

**Clean break, no compatibility shims:** all 49 importing files (src/apps/scripts/tests) were
rewritten to `doc_assistant.knowledge.<mod>`; no re-exports at the old paths (cpc §12 — no
speculative abstraction; nothing external imports this package). `scripts/archive/` stays frozen
(retained, unmaintained, lint-excluded). Living docs/specs updated to the new paths; append-only
records (DEVLOG, SESSION, ADR-018, archived sprints) keep their historical paths, which remain
true of the revisions they describe.

## Consequences

- **Easier:** the package listing now states the architecture (`db/` · `ingest/` · `eval/` ·
  `knowledge/` · RAG top level); a knowledge-layer task has a boundary (`src/doc_assistant/CLAUDE.md`
  points into it); the Phase-D scale review has a named review surface.
- **Commits us to:** new knowledge-layer modules land inside `knowledge/`; imports use the full
  `doc_assistant.knowledge.<mod>` path (isort keeps them grouped).
- **Costs:** one-time import churn (49 files, mechanical, gate-verified); external notes/blogs
  referencing old paths (none known) go stale.
- **Reverses if:** the layer dissolves (modules deleted, not moved back) — no scenario re-flattens.

## Links

- [ADR-021](ADR-021-adopt-cpc-big-project-layout.md) / [ADR-022](ADR-022-docs-system-rationalization.md) — the same session's layout decisions.
- `src/doc_assistant/knowledge/__init__.py` — the layer's contract docstring.
- `docs/architecture.md` — updated module map.
- Gate at merge: ruff/format clean · `mypy --strict src` 64 files clean · pytest (see DEVLOG entry).

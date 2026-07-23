<!-- status: active · updated: 2026-07-23 · class: living -->

# Decisions — index

One line per ADR; the files in `docs/decisions/` are canonical (append-only — supersede, never
edit). **Every new decision gets a new `ADR-NNN-slug.md` plus a line here.** The pre-cpc monolith
(2026-05 → 2026-07: core RAG rationale, production standards, phase logs, deferred items) is
frozen verbatim at `docs/archive/decisions-monolith.md` (ADR-022) — cite it for the *why* behind
the locked retrieval/chunking settings that predate per-file ADRs.

| ADR | Decision | Status |
|-----|----------|--------|
| [ADR-001](decisions/ADR-001-adopt-cpc-standard.md) | Adopt the cpc conventions standard (execution contract for the port) | accepted |
| [ADR-002](decisions/ADR-002-tauri-fastapi-desktop-shell.md) | Desktop shell = Tauri + FastAPI/SSE sidecar (replaces Chainlit) | accepted |
| [ADR-003](decisions/ADR-003-structlog-observability.md) | `structlog` everywhere; zero `print()` in `src/` | accepted |
| [ADR-004](decisions/ADR-004-gap-detection-layer.md) | Gap detection = deterministic Tier-1 floor + quarantined stochastic ceiling | accepted |
| [ADR-005](decisions/ADR-005-epistemics-markers-default-off.md) | Epistemics markers default-off while KI-7 noise persisted | **superseded** (G1 flipped default-on, 2026-07-07) |
| [ADR-006](decisions/ADR-006-contrastive-keyword-termhood.md) | Keyword termhood = contrastive scoring against `wordfreq` general English | accepted |
| [ADR-007](decisions/ADR-007-cpc-gates-vendored-local-only.md) | cpc gates vendored + local-only (private tooling, public repo) | accepted |
| [ADR-008](decisions/ADR-008-concept-skeleton-r5-decision-run.md) | Concept-skeleton R5 decision run: K=2 + `boundary` presence validated (PASS) | accepted |
| [ADR-009](decisions/ADR-009-book-oriented-hierarchical-chunking.md) | Book-oriented hierarchical chunking | accepted |
| [ADR-010](decisions/ADR-010-rag-sandbox-nonpersistent-overrides.md) | RAG sandbox knobs = request-scoped, non-persistent overrides | accepted |
| [ADR-011](decisions/ADR-011-desktop-provider-apikey-management.md) | Desktop provider/model live-switch; keys stay in `.env` (v1) | accepted |
| [ADR-012](decisions/ADR-012-provenote-installer-identity.md) | Provenote = product identity; `doc_assistant` = code identity (intentional split) | accepted |
| [ADR-013](decisions/ADR-013-document-metadata-editing.md) | Manual metadata edits = override sidecar (user wins, survives re-ingest) | accepted |
| [ADR-014](decisions/ADR-014-document-safe-delete.md) | Document delete = OS-trash-first, then rows/chunks/figures/cache | accepted |
| [ADR-015](decisions/ADR-015-tag-families-over-concept-vocabulary.md) | Keyword families share the `Concept` table (no second vocabulary) | accepted |
| ADR-016 | — number unused (no file; numbers are never reused) | — |
| [ADR-017](decisions/ADR-017-concept-graph-ui-boundaries.md) | Concept-graph UI boundaries (read-only graph; one write surface) | accepted |
| [ADR-018](decisions/ADR-018-graph-vocabulary-scope.md) | Graph vocabulary scoped by additive **opt-in** `graph_include` | accepted |
| [ADR-019](decisions/ADR-019-concept-taxonomy-classification-layer.md) | Concept taxonomy = curated classification layer (ANZSRC backbone); designed, unbuilt | accepted |
| [ADR-020](decisions/ADR-020-share-rigor-todo-via-git.md) | `RIGOR_TODO.md` shared via git (project debt, not per-machine state) | accepted |
| [ADR-021](decisions/ADR-021-adopt-cpc-big-project-layout.md) | cpc big-project layout: `AGENTS.md` entry + module `CLAUDE.md` + vendored gates | accepted |
| [ADR-022](decisions/ADR-022-docs-system-rationalization.md) | Docs system rationalized for scale (this index; monolith archived; DEVLOG inverted) | accepted |
| [ADR-023](decisions/ADR-023-knowledge-subpackage.md) | Backend restructure: `knowledge/` subpackage for concept graph / keywords / wiki / gaps | accepted |
| [ADR-024](decisions/ADR-024-evals-results-folder.md) | Top-level `evals/` = benchmark-results home (README keeps the headline; harness/baselines stay put) | accepted |
| [ADR-025](decisions/ADR-025-corpus-folders-retrieval-scope.md) | Corpus groups = folders; query-time doc-hash retrieval scoping; per-turn scope in provenance | accepted (built, F1+F2+F3) |
| [ADR-026](decisions/ADR-026-rebuild-migrations.md) | Rebuild migrations for shape changes SQLite can't ALTER; `document_meta` gets its missing FK | accepted (built) |
| [ADR-027](decisions/ADR-027-epistemics-surfacing-split.md) | Epistemics surfacing split: assessment always-on (source-evaluation strip), influence opt-in (answer layer) | accepted |
| [ADR-028](decisions/ADR-028-concept-taxonomy-polyhierarchy-skos.md) | Concept taxonomy amendment: unified typed polyhierarchical SKOS graph (amends ADR-019 — supersedes C1/D1/D6, reverses D9) | accepted |

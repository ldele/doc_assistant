# scripts/ — enrichment / eval / build CLI runners

**Owns:** the idempotent CLI runners over `src/doc_assistant/` sidecar modules (Enrichment-Layer
Pattern: runners re-derive; they never mutate the chunk store) plus dev/build tooling
(`launch_app.ps1`, `build_sidecar.py`, `doc_assistant_api.spec`).

**Key files**
- Enrichment: `extract_*`, `enrich_metadata`, `compute_*`, `build_concept_skeleton`, `build_gaps`,
  `build_wiki`, `seed_concepts`, `rank_candidates`, `backfill_graph_include`, `normalize_cache`.
- Eval/measure: `run_eval`, `sweep_*`, `self_eval`, `measure_latency`, `eval_marker_tables`.
- `conventions.toml` — cpc gate config (**not a script**; cpc-mandated path, see ADR-021).
- `archive/` — retained one-time migrations, excluded from lint; never run.

**Rules that bite here**
- **Dry-run is the default; `--apply` writes.** Keep that polarity on every new runner.
- **KI-4 credit leak:** `.env` defaults are all-Anthropic — force `--provider ollama` on every
  enrichment/self-eval run. Paid providers must trip `llm.assert_provider_intent`.
- **`build_concept_skeleton --apply` alone PRESERVES existing Node-B stance** (E0.5b) but does not
  *regenerate* it — to refresh stance from the corpus, run `--apply --enrich` (Ollama, KI-4). Pre-E0.5b
  a plain `--apply` silently wiped stance (the G6-run footgun; `.claude/CONTEXT.md` G6 note).
- Enrichment runners need host `data/` access — they no-op in a sandbox (KI-5).
- Runners run as modules (`python -m scripts.<name>`) inside the uv venv (`just eval`, `just ingest`).

**Tests:** runner cores live in `src/` and are tested there; `tests/` covers the pure helpers.

<!-- Keep <=40 lines. Local only. If you're restating a project-wide rule, delete it and cite the code. -->

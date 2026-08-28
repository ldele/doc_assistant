# scripts/ — enrichment / eval / build CLI runners

**Owns:** the idempotent CLI runners over `src/doc_assistant/` sidecar modules (Enrichment-Layer
Pattern: runners re-derive; they never mutate the chunk store) plus dev/build tooling
(`launch_app.ps1`, `build_sidecar.py`, `doc_assistant_api.spec`).

**Key files**
- Enrichment: `extract_*`, `enrich_metadata`, `compute_*`, `build_concept_skeleton`, `build_gaps`,
  `build_wiki`, `seed_concepts`, `rank_candidates`, `backfill_graph_include`, `normalize_cache`.
- Eval/measure: `run_eval`, `compare_runs`, `emit_baseline`, `sweep_*`, `self_eval`, `measure_latency`.
- `conventions.toml` — cpc gate config (**not a script**; cpc-mandated path, see ADR-021).
- `archive/` — retained one-time migrations, excluded from lint; never run.

**Rules that bite here**
- **Dry-run is the default; `--apply` writes.** Keep that polarity on every new runner.
- **KI-4 credit leak:** `.env` defaults are all-Anthropic — force `--provider ollama` on every
  enrichment/self-eval run, `run_eval` included (it generates an answer per case, so a free scorer
  mix is not a free run). Paid providers must trip `llm.assert_provider_intent`.
- **`build_concept_skeleton --apply` alone PRESERVES existing Node-B stance** (E0.5b) but does not
  *regenerate* it — to refresh stance from the corpus, run `--apply --enrich` (Ollama, KI-4). Pre-E0.5b
  a plain `--apply` silently wiped stance (the G6-run footgun; `.claude/CONTEXT.md` G6 note).
- **Console encoding:** every entrypoint pins `sys.stdout.reconfigure(encoding="utf-8")` on win32
  behind a `hasattr` guard (Jupyter's `OutStream` lacks it). Copy the existing header verbatim —
  pytest and Linux CI both hide the cp1252 crash this prevents.
- **Any new eval runner: `Store(db, settings_provider=run_defining_settings)`** (the harness cannot
  import app config — ADR-003 D8 — so wiring it is the runner's job), **plus what config cannot
  know**, in the explicit `config=` dict: the corpus
  (`index_composition(pipeline.indexed_doc_hashes)`, RG-021) and the generator as it actually ran
  (`pipeline.provider`/`.model` — `--provider` moves neither constant, RG-029). Copy `run_eval.py`;
  a run that does not record what it swept cannot be audited (KI-41).
- **A sweep that varies a setting through a channel it does not own must prove the setting arrived,
  before it spends anything.** Copy `sweep_chunking.preflight`: resolve each arm in a subprocess
  under that arm's environment via `run_defining_settings`, fail unless asked == effective and no
  two arms match. Silent overwrite fails as "no effect", which reads as a confirmed default (KI-41).
- Enrichment runners need host `data/` access — they no-op in a sandbox (KI-5).
- Runners run as modules (`python -m scripts.<name>`) inside the uv venv (`just eval`, `just ingest`).

**Tests:** runner cores live in `src/` and are tested there; `tests/` covers the pure helpers.

<!-- Keep <=40 lines. Local only. If you're restating a project-wide rule, delete it and cite the code. -->

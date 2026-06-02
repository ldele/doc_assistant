# Architecture

## High-level flow

```
Documents (PDF/EPUB/HTML/DOCX/MD)
↓
Extractors → Markdown cache (data/cache/)
↓
Chunker (markdown-aware, parent-child)
↓
Embeddings (BGE-base) → Chroma vector store (data/chroma/)
             ↕
         SQLite document store (Folder → Document → Part → Chunk)
↓
Hybrid retrieval (BM25 + vector, weights 0.4/0.6) → top 10 candidates
↓
Cross-encoder reranker → top 5 passages (parent context returned)
↓
LLM (Claude or local Ollama) → streamed answer with citations
```

## Module responsibilities

| Module | Role | Public contract |
|---|---|---|
| `doc_assistant.config` | Paths, env vars, feature flags | Read-only after init; no side effects |
| `doc_assistant.extractors` | Convert any supported format → markdown | Returns `str`; raises `ExtractionError` on failure |
| `doc_assistant.ingest` | Extract, chunk, embed, store; streaming | Idempotent per content hash; raises `IngestError` |
| `doc_assistant.pipeline` | RAG runtime: retrieve, rerank, generate | Returns `Answer` with citations; raises `PipelineError` |
| `doc_assistant.health` | Document health scoring and classification | Pure function; no I/O; returns `HealthResult` |
| `doc_assistant.library` | Document store queries (browse, filter, tag) | Read-only queries against SQLite; UI-framework-agnostic |
| `doc_assistant.prompts` | Prompt templates | Pure string interpolation; no I/O |
| `doc_assistant.tracking` | Token usage tracking and cost estimation | Append-only; never raises |
| `apps/cli.py` | Terminal interface | Thin entrypoint; no business logic |
| `apps/chainlit_app.py` | Web chat UI | Thin entrypoint; no business logic |
| `scripts/` | One-off maintenance scripts | Not part of the importable package |

**Boundary rule:** `apps/` contains no business logic. All logic lives in `src/doc_assistant/`. The UI layer calls the library layer; never the reverse.

## Two-tier caching

1. **Extraction cache** (`data/cache/*.md`): mirrors `data/sources/` structure.
   Invalidated by file modification time. Skips re-extraction on unchanged files.
2. **Embedding cache** (Chroma `doc_hash` metadata): invalidated by content hash.
   Skips re-embedding when content is unchanged.

Both tiers are independent: changing the chunking strategy invalidates embeddings but not extraction. Rebuild with `python -m doc_assistant.ingest --rebuild`.

Hashing is content-only (SHA-256 of extracted markdown, truncated to 16 hex chars). Documents survive path changes and re-extractions without creating orphan rows. Migration from the old path+content scheme: `scripts/migrate_to_content_hash.py`.

## Document health model

Each ingested document is scored on five signals: chunk count, chunks-per-page ratio, average chunk length, section detection rate, reference-flagged chunk ratio.

- Score ≥ 75 → **healthy**
- Score ≥ 40 → **marginal** (retrievable, flagged)
- Score < 40 → **broken** (retrievable, prominently flagged)

Classification is informational, never blocking. Broken documents remain queryable.

## Engineering standards

### Security
- No secrets in code. `.env` is gitignored. `.env.example` committed with placeholders.
- `bandit` SAST runs in CI and pre-commit. HIGH findings block merge.
- `pip-audit` runs in CI on every push.
- `detect-secrets` baseline committed; hook runs in pre-commit.

### CI/CD
- GitHub Actions: ruff → mypy → pytest with coverage → bandit → pip-audit on every push and PR.
- Merging on red pipeline is never allowed.
- Coverage floor: 70% (CI-enforced). Target: 85% for core pipeline and ingest logic.

### Pre-commit (mandatory)
Hooks: ruff (lint + format), mypy, bandit, detect-secrets, standard file hygiene.

### Logging
Structured JSON logging in staging/production via `structlog`. Development uses pretty console output. No `print()` in `src/`. Log entries include: level, timestamp, module, event, and operation-specific context fields. Secrets and PII are never logged.

### Development log
Maintain `docs/DEVLOG.md` — append one entry per logical change (what / why / rejected / opens). See dev-log skill for format. Append only, never edit past entries.

### Error handling
Exception hierarchy rooted at `DocAssistantError`. Domain errors (ExtractionError, IngestError, PipelineError) are typed and documented. Infrastructure errors (StorageError, ExternalServiceError) propagate with context via `raise X from e`. User-facing errors are translated at the UI boundary; internal traces go to logs only.

### Testing
```
tests/
├── conftest.py           # shared fixtures
├── test_smoke.py         # import sanity
├── unit/                 # fast, no I/O, no LLM
│   └── test_<module>.py
├── integration/          # cross-module, may use temp files, mocked LLM
│   └── test_<flow>.py
└── eval/                 # RAG evaluation harness (not part of standard CI run)
    └── eval_set.json
```

Unit tests run on every commit (pre-commit). Full suite (unit + integration) runs in CI — free, no API calls. Eval harness runs manually at phase checkpoints and costs money (Anthropic API for the LLM judge).

The testing strategy — what each tier and each eval scorer measures, why, and the reproducible public-corpus benchmark — is documented in [`tests/eval/TESTING.md`](../tests/eval/TESTING.md).

Run commands:
- `uv run pytest tests/unit/ tests/integration/` — free, fast, CI default
- `uv run python -m tests.eval.run_eval` — manual, costs API tokens
- `uv run pytest -m api` — any future tests marked with `@pytest.mark.api`

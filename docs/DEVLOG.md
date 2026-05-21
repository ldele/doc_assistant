# DEVLOG — doc_assistant

Real-time development log. One entry per logical change.
Append only — never edit past entries.

Format: What changed | Why | Rejected alternatives | What it opens

---
## Session: 2026-05-21 — Production infrastructure + content-only hashing

**Starting from:** Phase 3.3 complete. Four Phase 3 gate items remaining: prod infra (CI, pre-commit, security), content-only hashing, .env.example.
**Goal this session:** Complete prod infra and content-only hashing.

### pyproject.toml — dev dependencies and tool config
**What:** Added ruff, mypy (strict), bandit, pip-audit, detect-secrets, pre-commit, structlog, pytest-cov to dev extras. Added full tool configuration sections for ruff, mypy, bandit, pytest.
**Why:** Prod-engineering skill mandates mechanical checks before human review. These tools catch lint/type/security issues 10x cheaper than finding them in code review or production.
**Rejected:** Separate config files (setup.cfg, .mypy.ini) — pyproject.toml is the standard single-source for Python tooling config.

### .pre-commit-config.yaml (new)
**What:** Created pre-commit config with ruff (lint+format), mypy, bandit, detect-secrets, file hygiene hooks, no-commit-to-branch on main.
**Why:** Pre-commit catches issues at commit time, before they reach CI. Enforced consistency across all contributors.

### .github/workflows/ci.yml (new)
**What:** Created GitHub Actions CI: ruff → mypy → pytest (≥70% coverage) → bandit → pip-audit → detect-secrets. Runs on all pushes and PRs to main.
**Why:** CI is the enforcement layer. Pre-commit is optional (can be skipped with --no-verify); CI is not.

### .secrets.baseline (new)
**What:** Generated detect-secrets baseline. Contains false positives from .chainlit/translations/ (Secret Keyword detections in translation JSON).
**Why:** Baseline is required for detect-secrets to work — it diffs new findings against the baseline.

### src/ — mypy strict compliance (8 files)
**What:** Added full type annotations across all source files. Key patterns: `dict` → `dict[str, Any]`, explicit return types, `str()` wrapping on Any returns, `datetime.UTC` → `timezone.utc` for Python 3.10 compat.
**Why:** mypy strict catches real bugs (wrong return types, missing None checks). The 51 errors found during initial run included several genuine issues.
**Rejected:** mypy non-strict — too lenient, misses the bugs that matter.

### src/doc_assistant/library.py — SQLAlchemy boolean comparison fix
**What:** `Document.is_archived == False` → `Document.is_archived.is_(False)` (4 occurrences).
**Why:** ruff E712 flags `== False` as bad practice. SQLAlchemy's `.is_()` generates correct SQL and satisfies the linter.

### src/doc_assistant/extractors.py — EXTRACTORS dict refactor
**What:** Split mixed-type dict into `_EXTRACTORS: dict[str, Callable]` (callables only) and `SUPPORTED_EXTENSIONS: set[str]` (all extensions). PDF handled as explicit if/else.
**Why:** Original dict mixed str values (for PDF extractor name) with Callable values. mypy strict couldn't type this correctly.
**Rejected:** Union type `dict[str, str | Callable]` — loses type safety on the caller side.

### src/doc_assistant/ingest.py — content-only hashing
**What:** `doc_hash(text, source)` → `doc_hash(text)`. SHA-256 of extracted markdown content only, truncated to 16 hex chars. Path removed from identity.
**Why:** Path+content hashing caused duplicate Document rows whenever a file was moved, renamed, or re-extracted. This was a data integrity issue blocking Phase 4 (citation graph depends on stable document identity).
**Rejected:** Keeping path in hash with a path-change detector — treats a symptom, not the cause.
**Opens:** Existing data needs migration. Run `scripts/migrate_to_content_hash.py --apply`.

### scripts/migrate_to_content_hash.py (new)
**What:** Dry-run + --apply migration script. Recomputes hashes in SQLite and both Chroma stores. Handles dedup collisions (same content at different paths → merge into highest-chunk-count row).
**Why:** Existing data has old-format path+content hashes. Migration must be explicit and reviewable.
**Rejected:** Auto-migration on ingest startup — runs without user awareness, risk of silent data changes.

### tests/unit/test_hash.py — updated for content-only hashing
**What:** Inverted `test_hash_changes_with_path` to assert SAME hash for same content at different paths. Removed source param from all test calls.
**Why:** Tests must match the new behavior. The old test explicitly documented that path-dependent hashing was temporary.

### src/doc_assistant/db/models.py — datetime.utcnow deprecation fix
**What:** `default=datetime.utcnow` → `default=lambda: datetime.now(timezone.utc)` (5 occurrences including onupdate).
**Why:** `datetime.utcnow()` is deprecated in Python 3.12+ and produces naive datetimes. `timezone.utc` is the correct replacement.

### .github/workflows/ci.yml — test separation
**What:** CI now explicitly runs `tests/unit/ tests/integration/` only, ignores `tests/eval/`.
**Why:** Unit/integration tests are free (no API calls). Eval harness costs money (Anthropic API for LLM judge) and runs manually at phase checkpoints.

### pyproject.toml — pytest markers and warning filters
**What:** Added `api` marker for future API-calling tests. Added `filterwarnings` to suppress chromadb deprecation warning.
**Why:** Clean test output. The chromadb warning is an upstream issue (asyncio.iscoroutinefunction deprecated in 3.16), not fixable on our side.

### .gitignore — critical fixes
**What:** Removed CLAUDE.md, .secrets.baseline, .pre-commit-config.yaml from gitignore. Added .venv/, dist/, build/, data/library.db.
**Why:** CLAUDE.md, .secrets.baseline, and .pre-commit-config.yaml must be committed (project context, security baseline, hook config). .venv and build artifacts should never be committed.

### Session end
**Done:** Full prod infrastructure (CI, pre-commit, security tooling, mypy strict). Content-only hashing with migration script. Test separation. .gitignore fixes.
**Unresolved:** Hash migration not yet run on local data. `.env.example` not started.
**Next:** Write `.env.example` → run hash migration → commit → Phase 3 complete.

---
## Session: 2026-05-21 (cont.) — .env.example + Phase 3 gate close

### .env.example — created
**What:** Rewrote `.env.example` with all 8 env vars from `config.py`. Sections: required (ANTHROPIC_API_KEY), LLM mode, extraction, HuggingFace, RAG tuning (locked). Removed fake `sk-ant-...` placeholder.
**Why:** Last Phase 3 gate item. Engineering standard: no secrets in code, `.env.example` committed.
**Rejected:** Leaving the old minimal version — it lacked section headers and had a fake key prefix that could confuse tools scanning for leaked secrets.
**Opens:** None. All env vars documented.

### .claude/ — Claude Code project config
**What:** Created `.claude/settings.json` (permissions whitelist for uv/git/ruff/mypy/pytest/bandit/pre-commit, deny rm -rf and raw pip). Created `.claude/commands/`: `status.md` (session start briefing), `eval.md` (RAG eval with cost warning), `check.md` (full local quality gate).
**Why:** When using Claude Code CLI on this repo, these eliminate repetitive setup prompts and enforce the same quality gates as CI.
**Rejected:** Adding `ingest` and `migrate` commands — too likely to change shape in Phase 4. Will add when stable.
**Opens:** Commands may need updating as project evolves (new test dirs, new tools).

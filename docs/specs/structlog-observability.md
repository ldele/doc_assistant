# Spec — Structured logging via structlog (observability substrate; closes KI-1)

**Status:** ✅ BUILT — Claude Code 2026-06-23 (CPU box). Gate green: ruff/format/`mypy --strict src`/
bandit clean; **601 passed, 1 skipped, coverage 82%**; zero `print()` in `src/`; console-parity smoke
passed. DEVLOG 2026-06-23. **Staged, not committed.** Remaining: RG-013 (frozen-build parity).
**Owner of execution:** Claude Code (code + tests + gate, on the box).
**Decision reference:** `docs/decisions/ADR-003-structlog-observability.md` (the *why*: observability is a tenet → structured-event logger, not stdlib strings).
**Closes:** KI-1 (`print()` in `src/` + the aspirational structlog standard). Makes `.claude/CONTEXT.md` rule #5 true and enforceable.
**Rigor:** RG-013 (renderer/level defaults + frozen-build parity — `.claude/RIGOR_TODO.md`).

**Requirement (the why).** Every `src/` log line should be a structured, queryable event with
bindable context, configured once per app entrypoint — delivering the observability the app
promises and retiring the 32 `print()` calls. This is **not** a behavior change: CLI progress
output stays visible; the eval harness and answer text are untouched.

---

## Verified surface (2026-06-23, `HEAD` = c06298f)

| Item | Count | Where |
|---|---|---|
| `print()` in `src/` | **32** | `ingest.py` (18), `pipeline.py` (9), `db/migrations.py` (4), `llm.py` (1) |
| stdlib `log = logging.getLogger(__name__)` | **11 modules** | citations, concept_graph, doc_vectors, epistemics, eval/runner, export, figures, regions, reviewer, tables, wiki |
| `log.*` call sites | **16** | 12 `warning`, 2 `info`, 2 `exception` — all classic `%`-style args |
| logging configuration | **0** | no `basicConfig`/`dictConfig`/`setup_logging`; no `LOG_*` in `config.py` |
| `structlog` in `pyproject.toml` | **absent** | — |

Two non-obvious facts the build must respect:
1. **Most prints are user-facing CLI progress** ("Loading embeddings…", "Found N supported files",
   "Creating tables…"), not debug. A naive swap to `log.info()` **silences the console** unless a
   console renderer is wired first.
2. **`llm.py:277`** writes to `stderr` deliberately (`print(..., file=sys.stderr, flush=True)`) —
   preserve stderr semantics.
3. **`eval/runner.py` logs but must stay import-clean of app wiring** — the harness is designed to
   be extractable (cpc; `tests/eval/TESTING.md`). It may use `structlog.get_logger` but must **not**
   import the `configure_logging` entrypoint or any app module.

---

## ADR-A — One configuration seam, two renderers, wired at the entrypoints

**Context.** There is no logging setup today, so `info` lines are invisible (root logger defaults
to `WARNING`) and prints are the only reliable console channel. structlog needs configuring once,
early, before any logger is used. The app has **three entrypoints** (`apps/chainlit_app.py`,
`apps/cli.py`, `apps/api/main.py` + `apps/api/__main__.py`) — and `src/` must never configure
logging itself (rule #3: `apps/` own wiring, the library stays a thin importable).

**Decision.** A single `configure_logging(*, json: bool, level: str)` function (new module
`src/doc_assistant/logging_config.py` — pure setup, no business logic), called once at the top of
each app entrypoint. structlog wraps the stdlib logger (`structlog.stdlib.LoggerFactory`) so the
11 existing loggers keep working through the same pipeline.
- **Console renderer** (`structlog.dev.ConsoleRenderer`) when `json=False` — dev/CLI default;
  preserves human-readable progress on the terminal.
- **JSON renderer** (`structlog.processors.JSONRenderer`) when `json=True` — for machine
  consumption (FastAPI in a deployed/observed context, log aggregation later).
- Level + renderer selected from new `config.py` keys (Decision table) with env override.

**Options considered.**
1. *Single seam + stdlib LoggerFactory (chosen).* One place to configure; the 11 stdlib loggers
   migrate to `structlog.get_logger` but the pipeline is unified; coexists with stdlib so
   third-party lib logs (chromadb, httpx, transformers) flow through the same renderer.
2. *Per-module structlog config (rejected).* Duplication, ordering hazards, and `src/` would own
   wiring — violates rule #3.
3. *Pure-structlog, bypass stdlib (rejected).* Loses third-party stdlib log capture; structlog's
   own docs recommend the stdlib integration for exactly this app shape.

## ADR-B — Preserve the user-facing console channel (no silenced progress)

**Context.** The 32 prints are the CLI's progress UX. Observability must not regress the human
experience of running `ingest` / `just api` / the CLI.

**Decision.** Convert prints to `log.info(event, **context)` **after** the console renderer is
wired (ADR-A), so progress still shows on the terminal at the default level. The conversion is
**behavior-preserving for the user**: what printed before still appears (now as a structured
console line). `stderr` semantics for `llm.py:277` preserved via the renderer's stream or an
explicit `log.warning`/`error` as fits the message. **Guard-tested**: a CLI ingest and
`python -m apps.api` still emit visible progress (smoke check in the DoD).

---

## Decisions

| # | Decision |
|---|---|
| 1 | **New `src/doc_assistant/logging_config.py`** — `configure_logging(*, json: bool, level: str) -> None`. Pure setup (structlog `configure` + stdlib `dictConfig`); no business logic, no app imports. The only module allowed to configure logging. |
| 2 | **`structlog` added to `pyproject.toml`** base deps; `uv lock` regenerated. **Re-verify the M4 PyInstaller freeze picks it up** (coupling to RG-012/KI-9 — named, not silently wired). |
| 3 | **Two new `config.py` keys:** `LOG_LEVEL` (default `"INFO"`) + `LOG_JSON` (default `False`). Env-overridable like the rest of `config.py`. These are a **config contract, not a locked setting** — no eval experiment needed to change them. |
| 4 | **11 modules:** `log = logging.getLogger(__name__)` → `log = structlog.get_logger(__name__)`. The variable name stays `log` (minimal churn; matches existing sites). |
| 5 | **16 `log.*` sites:** migrate `%`-style to structlog key-value events. `log.warning("No '%s' collection at %s", name, path)` → `log.warning("collection_missing", collection=name, path=str(path))`. Event name is a short stable slug; data goes in kwargs (the queryable part). |
| 6 | **32 prints → `log.*`** at the right level (progress → `info`; the `Couldn't…`/`Error on…`/`Warning:` prints → `warning`). Context as kwargs (`path=`, `count=`, `hash=`). Wired **after** ADR-A so console output persists. |
| 7 | **Three entrypoints call `configure_logging` once**, early: `apps/cli.py`, `apps/chainlit_app.py`, `apps/api/main.py` (in `create_app`/lifespan) + `apps/api/__main__.py`. FastAPI uses `json=True` when a deployed/observed context is signalled (else console); CLI/Chainlit use console. **`src/` never calls it.** |
| 8 | **`eval/runner.py`** may use `structlog.get_logger` but **must not** import `logging_config` or any app module — preserve harness extractability. A test asserts the harness imports without app wiring. |
| 9 | **Exceptions still chain** (`raise X from e`, rule #6). The 2 `log.exception` sites keep `exc_info`; structlog's `format_exc_info` processor renders the traceback. |

**Edge cases (spec explicitly):**
- *`configure_logging` not called* (e.g. a bare `import doc_assistant` in a test) → structlog falls
  back to its default config; loggers still work, just unconfigured. `src/` must not assume it ran.
- *Double-call* (two entrypoints in one process, or tests) → idempotent; last call wins. Safe.
- *Third-party noise* (chromadb/httpx/transformers at `INFO`) → set their stdlib levels to
  `WARNING` in the `dictConfig` so they don't drown the app's events.

---

## Files touched

**New:**
- `src/doc_assistant/logging_config.py` — the `configure_logging` seam.
- `tests/unit/test_logging_config.py` — renderer selection, level, idempotency, no-app-import.

**Modified (`src/`):**
- `config.py` — `LOG_LEVEL`, `LOG_JSON`.
- 11 logger modules — `getLogger` → `structlog.get_logger`; 16 call sites → event+kwargs.
- `ingest.py`, `pipeline.py`, `db/migrations.py`, `llm.py` — 32 prints → `log.*` (these 4 also
  need `log = structlog.get_logger(__name__)` added; they don't have a logger today).

**Modified (`apps/`):**
- `cli.py`, `chainlit_app.py`, `api/main.py`, `api/__main__.py` — call `configure_logging` once.

---

## Definition of done

1. **Zero `print()` in `src/`** — `grep -rn "print(" src/ --include="*.py"` returns nothing
   (this becomes the KI-1 close check; consider a `cpc` lint guard so it stays at zero).
2. **All `src/` logging is structlog** — no `logging.getLogger` left in `src/`.
3. **Gate green** (CPU box, `uv run --no-sync`): `ruff check src tests` · `ruff format --check` ·
   `mypy --strict src` · `bandit -r src` (0 HIGH) · `pytest tests/unit tests/integration`
   (~590 expected green; **no test asserting answer text or `sources_md` changes** — logging is
   invisible to those).
4. **Console parity smoke** (ADR-B): a CLI ingest run and `python -m apps.api` startup still print
   visible progress lines. Eyeballed + noted in the DEVLOG.
5. **Harness isolation** (Decision 8): a test confirms `eval/runner` imports with no app wiring.
6. **Freeze coupling flagged** (Decision 2): DEVLOG notes that the next M4 freeze must re-verify
   `structlog` is bundled (RG-013); not closed here (needs the box + Tauri toolchain).
7. **Docs:** DEVLOG entry; KI-1 → resolved; `.claude/CONTEXT.md` rule #5 reworded to match
   (structlog, configured at entrypoints); RG-013 logged.

---

## In / Out

**In:** the `configure_logging` seam; `structlog` dep; `LOG_*` config; converting all 32 prints +
16 stdlib sites + 11 logger acquisitions; wiring the 3 entrypoints; the tests + DoD above.

**Out (do not scope-creep):**
- **No log aggregation / shipping backend** (no Sentry, no OTEL exporter) — that's Phase 8+
  observability, a separate ADR. This spec makes logs *structured*, not *shipped*.
- **No new log lines for their own sake** — convert what exists; don't instrument new code paths.
- **No renderer/level tuning experiment** — defaults are a config contract (Decision 3), set once.
- **No `apps/` business logic** — entrypoints only call the seam.
- **No change to synthesis, prompts, retrieval, or the eval path** — logging is orthogonal.

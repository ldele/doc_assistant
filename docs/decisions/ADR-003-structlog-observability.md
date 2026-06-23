<!-- status: active · updated: 2026-06-23 · class: append-only -->

# ADR-003 — Structured logging via structlog as the observability substrate

- **Status:** accepted + **BUILT** (2026-06-23 — decided with Cowork; built by Claude Code, spec `docs/specs/structlog-observability.md` → ✅ BUILT; KI-1 closed)
- **Date:** 2026-06-23
- **Deciders:** Lucas (decided with Cowork)

> This ADR resolves the long-open tension recorded as **KI-1** (`print()` in `src/` vs the
> "structlog-only" standard) and the `.claude/CONTEXT.md` §"Non-negotiable rules" #5 logging rule.
> It is the **why** behind the build spec `docs/specs/structlog-observability.md`. It does not
> supersede a prior ADR; it makes an aspirational rule real and names the contract the conversion
> builds against.

## Context

The engineering standard (`.claude/CONTEXT.md` rule #5) mandates "structured logging via
`structlog`; no `print()` in `src/`." **Neither half holds in the code today** (verified
2026-06-23 on `docs/desktop-shell-specs`, `HEAD` = `c06298f`):

- `structlog` is imported in **0** `src/` modules and is **not in `pyproject.toml`**. The de
  facto logger is **stdlib `logging`**: 11 modules acquire `log = logging.getLogger(__name__)`
  (`citations`, `concept_graph`, `doc_vectors`, `epistemics`, `eval/runner`, `export`, `figures`,
  `regions`, `reviewer`, `tables`, `wiki`) with **16 call sites** (12 `warning`, 2 `info`,
  2 `exception`), all classic `%`-style args.
- **32 `print()` calls** remain across 4 modules: `ingest.py` (18), `pipeline.py` (9),
  `db/migrations.py` (4), `llm.py` (1). Most are **deliberate user-facing CLI progress**
  ("Loading embeddings…", "Found N supported files"), not stray debug output.
- **No logging configuration exists anywhere** — no `basicConfig`, no `dictConfig`, no
  `setup_logging()`, no `LOG_*` keys in `config.py`. The 11 loggers emit to the root logger's
  default `WARNING` handler; the 2 `info` lines are effectively invisible by default.

So the standard and reality have diverged, and the rule has had no enforcing seam. The decision
is not merely "kill the prints" — it is **whether logging is an observability substrate** (every
line a structured, queryable event with bound context) or just a console channel. The user's
position: **observability is a tenet of the app** (it is positioned local-first with a
research-integrity layer; structured, inspectable run logs are part of that promise and feed the
Phase 8+ observability direction). That reframes the choice from hygiene to architecture.

## Options

1. **Adopt `structlog` for real (chosen).** Add the dependency; wire one `configure_logging()`
   entry point (console renderer for dev/CLI, JSON renderer for machine consumption, level via a
   new `LOG_*` config knob); convert the 11 stdlib loggers to `structlog.get_logger(__name__)`,
   migrate the 16 `%`-style sites to structlog's key-value event style, and route the 32 prints
   through the logger while **preserving the user-facing console channel** for CLI progress.
   *Trade-off:* the largest diff — touches 15 `src/` modules plus all three app entrypoints
   (Chainlit, CLI, FastAPI) — and touches modules that have no current bug. *Sourced fact:*
   structlog wraps the stdlib logger via `structlog.stdlib.LoggerFactory`, so it coexists with
   the existing `logging` ecosystem rather than replacing it
   ([structlog docs — standard-library integration](https://www.structlog.org/en/stable/standard-library.html)),
   making an incremental, contract-preserving migration possible.

2. **Amend the standard to stdlib `logging`.** Change rule #5 to "stdlib `logging`, no `print()`,"
   add a small `dictConfig` setup, convert only the 32 prints onto the logger already in use.
   *Trade-off:* smallest diff, matches today's reality — but **drops the observability tenet**:
   stdlib records are unstructured strings, so "every log line is a queryable event with bound
   context" is not delivered without bolting a formatter on later. *Sourced fact:* stdlib
   `logging` has no first-class structured/contextual binding; `extra=` is per-call and easy to
   forget, which is the gap structlog's `bind()` closes.

3. **Status quo (do nothing).** Leave the prints and the unconfigured stdlib loggers. *Rejected
   on sight:* it is the state KI-1 already flags, leaves `info`-level lines invisible, and
   contradicts both the written standard and the observability tenet.

## Decision

**Adopt structlog (option 1)** as the single logging substrate for `src/`, configured once at
each app entrypoint, emitting human-readable console output in dev/CLI and JSON when a machine
consumer is the target. The deciding reason: **observability is a stated tenet**, and only a
structured-event logger delivers queryable, context-bound logs as a first-class property rather
than a later retrofit — option 2's smaller diff buys nothing if the structured output has to be
added anyway.

**What would reverse it:** if structlog's footprint or the JSON pipeline proves to be friction
disproportionate to the observability payoff on a single-user local app (e.g. it complicates the
PyInstaller freeze, KI-9-adjacent), fall back to option 2 — stdlib `logging` + a JSON formatter —
recording that reversal as a superseding ADR.

## Consequences

**Easier:** every `src/` log line becomes a structured event with bindable context
(`doc_id`, `collection`, `provider`, run id…), so runs are inspectable and machine-parseable;
the logging standard (rule #5) becomes *true and enforceable* rather than aspirational; KI-1
closes; a future observability/aggregation step (Phase 8+) has structured input from day one.

**Harder / cost:** the largest non-feature diff in recent history — 15 `src/` modules + 3 app
entrypoints; a new dependency in the freeze (must re-verify the M4 PyInstaller build picks up
`structlog` cleanly — coupling to RG-012/KI-9); CLI progress output must be explicitly preserved
through the logger (a naive swap silences it), so the conversion needs the console-renderer seam
before any call site moves.

**Must revisit:** the renderer/level defaults (console vs JSON selection, default level) are a
config contract, not a tuning lock — settle them in the spec, not here. The interaction with the
frozen desktop build is unverified until the M4 freeze is re-run with the dependency present.

## Confidence

- ✓ The current state is measured, not assumed — counts above are from `grep` over
  `HEAD` = `c06298f` (32 prints / 4 files; 16 stdlib sites / 11 modules; zero config; structlog
  absent from `pyproject.toml`).
- ⚠ The renderer/level/JSON-trigger defaults are unset and chosen in the spec, not validated —
  tracked as **RG-013** (`.claude/RIGOR_TODO.md`): confirm CLI progress parity and that the
  frozen M4 build emits structured logs without a missing-import or console-silencing regression.

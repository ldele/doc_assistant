<!-- status: active · updated: 2026-06-20 · class: living -->

# KNOWN ISSUES

Open weaknesses, recurring failures, workarounds. Log a bug the second time it appears.
Migrated from the old `CLAUDE.md` / `README` runtime-quirk notes on 2026-06-20 (cpc adoption).

## KI-1 — `print()` in `src/` violates the structlog-only standard — OPEN
- **Symptom:** Source modules still use `print()` in places; the engineering standard mandates
  structured logging via `structlog` and no `print()` in `src/`.
- **Cause:** Predates the logging standard; never fully back-filled.
- **Workaround:** None needed at runtime — cosmetic/observability debt.
- **Real fix:** Replace `print()` calls with `structlog` loggers, or consciously descope per module.
  Don't silently ignore.

## KI-2 — Python 3.14 breaks Chainlit at runtime — OPEN (constraint, not a bug to fix)
- **Symptom:** App fails to run the Chainlit web UI on Python 3.14 (anyio event-loop incompatibility).
- **Cause:** Chainlit's anyio stack is not yet 3.14-compatible.
- **Workaround:** Develop/test on 3.14 if desired, but **run on Python 3.12** (the pinned runtime).
- **Real fix:** Upstream Chainlit/anyio support; revisit on a Chainlit bump. CLI path is unaffected.

## KI-3 — win32 `cu130` torch wheel segfaults on a GPU-less box — RESOLVED (2026-06)
- **Symptom:** Instant segfault when the CUDA (`cu130`) torch wheel ran on a machine with no usable GPU.
- **Cause:** A single lock-pinned `torch==…+cu130` for all `sys_platform == 'win32'`; `uv`'s
  `torch-backend = "auto"` is `uv pip`-only (a no-op for `uv lock`/`sync`/`run`).
- **Fix (shipped, `423cbfa`):** per-machine torch via **mutually-exclusive uv extras** — `--extra cu130`
  on a GPU box, `--extra cpu` on a CPU-only box / CI. **Rule: never `cu130` on a GPU-less box.**
- **Pointer:** `docs/specs/torch-backend-per-machine.md`.

## KI-4 — Anthropic-default credit leak on "local" enrichment runs — OPEN (workaround known)
- **Symptom:** Enrichment / self-eval runs intended to be local (Ollama) silently bill the Anthropic API.
- **Cause:** `.env` defaults are all-Anthropic; every generator/reviewer/judge inherits the default
  provider unless overridden.
- **Workaround:** **Force `--provider ollama`** (and reviewer/judge provider flags) on every
  enrichment / self-eval run. Cost-guards exist but the default is the trap.
- **Real fix:** A local-first default profile, or a hard guard that refuses paid calls outside an
  explicit `--allow-paid`. Contract: `docs/specs/llm-provider-isolation.md`.

## KI-5 — Sandbox cannot write runtime data; enrichment runners must run on the host — OPEN
- **Symptom:** Metadata backfill / citation extraction / other enrichment CLI passes appear to no-op
  or fail to persist when invoked from a sandboxed context.
- **Cause:** `data/` (SQLite + Chroma) is host-local and gitignored; the sandbox/host filesystem
  isn't synced for writes.
- **Workaround:** Run the idempotent enrichment runners (`scripts/extract_*`, `compute_doc_vectors`,
  etc.) **on the host**, not the sandbox. They're idempotent, so re-running is safe.
- **Real fix:** N/A — environmental; document and run on host.

## KI-6 — SSL crash on a `uv`-managed Python (Windows) — OPEN (per-machine; documented workaround)
- **Symptom:** App dies instantly with no traceback (`OPENSSL_Uplink(...): no OPENSSL_Applink`) on the
  first HTTPS call (Claude API, Ollama, any networked test).
- **Cause:** OpenSSL in uv's bundled (python-build-standalone) interpreter; an official CPython is unaffected.
- **Status:** a persistent per-machine environmental quirk; the fix is **deliberately not pinned in-repo**
  (`docs/decisions.md`) — kept as a documented remedy, revisited only if more boxes hit it.
- **Workaround:** Rebuild the venv on an official python.org 3.12 (`py install 3.12` →
  `uv venv --clear --python …` → `uv sync --all-extras`). Behind a TLS-inspecting proxy, prefix uv
  commands with `UV_NATIVE_TLS=1`. Offline work (ingest/embeddings/retrieval) is unaffected either way.

## KI-7 — Concept-graph LLM-extraction core + `data/graph/graph.json` are SUPERSEDED — OPEN (redesign decided, not yet built)
- **Symptom:** The shipped Feature 7 (PR 16) concept graph derives nodes from a per-document
  open-vocabulary LLM extraction; on this same-domain corpus that fragments concepts and is the
  dominant cost (36–40 LLM calls/doc; hit `budget_exhausted` over the 61-doc corpus).
- **Status:** Superseded in part by the **2026-06-18 concept-graph REDESIGN** (Decision C — user-curated
  vocabulary + deterministic skeleton from `Citation`/`DocSimilarity` + confined LLM enrichment). ADR-1
  (Louvain) and ADR-4 (composite chunk key) carry over; ADR-3 (LLM-extraction node source) + the single
  integrity tag do not. The redesign is **decided but not yet built**; Feature 7d re-founds on it.
- **Do not build on:** the current LLM-extraction graph or `data/graph/graph.json` (the on-disk file is
  also an empty environment artifact, not a quality measurement).
- **Pointer:** `docs/decisions.md` → "Feature 7 — concept-graph REDESIGN" (2026-06-18). Edge precision +
  presence recall are flagged for RIGOR_TODO before the edge model is locked.

## KI-8 — PC→baseline marker mapping (PR-M1) is coarse at parent boundaries — OPEN (advisory, fail-safe)
- **Symptom:** In the default parent-child retrieval mode, the live 7d marker chip maps a marked baseline
  chunk onto a retrieved parent by **text containment** (`epistemics.markers_for_parent`): a parent gets a
  marker if it *contains* a marked chunk's text. The two collections are independent segmentations, so a
  parent spanning a marked chunk plus three clean ones is marked as a whole — over-attribution within the
  parent. A marked chunk straddling two parents marks both.
- **Why it's acceptable (for now):** markers are an **advisory chip, not a gate** (inform-don't-block), and
  over-attribution is fail-safe — it points the user at a real contested concept *in that passage*. The
  marker never changes synthesis, ranking, or the answer (byte-identical when absent).
- **Status:** chosen in PR-M1 ADR-1 over the heavier alternative (re-project `chunk_epistemics` onto PC
  parents — a second projection + migration + its own attribution-quality validation). That precise
  re-projection is the documented upgrade **if** containment proves too coarse on real data.
- **Compounding caveat:** marker *quality* upstream still comes from the superseded open-vocabulary graph
  (KI-7) — `contested` is local-model-noisy. M1 surfaces what the sidecar holds; it does not fix extraction.
- **Pointer:** `docs/specs/pr-m1-epistemics-markers.md` ADR-1 (option 2 = the re-projection upgrade).

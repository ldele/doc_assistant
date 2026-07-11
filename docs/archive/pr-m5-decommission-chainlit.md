# Spec — PR-M5: decommission Chainlit + lift the Python-3.12 pin (KI-2)

> **📦 Archived — shipped; the historical code-level contract, archived here (2026-07-11).** Live status: ROADMAP row M5 — Chainlit removed; the 3.12-pin lift was verified-and-deferred (KI-2 stays open: native deps crash on 3.14, not Chainlit). The behaviour of record is the code + tests, not this spec.

**Status:** ✅ BUILT (2026-06-25) — Chainlit removed (renderer + `.chainlit/` + dep + mypy override + recipe), parity test trimmed, gate green on 3.12 (602 passed). The 3.12-pin lift was **verified-and-deferred** per ADR-2: on 3.14 the deps install + ruff/mypy/bandit pass, but the full pytest suite hard-crashes the interpreter (native dep, not Chainlit) → **KI-2 stays open with the cause renamed**. Tauri migration (`docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md`).
**Owner of execution:** Claude Code (code + tests + gate, on the box).
**Hard gate — do NOT execute the build until both hold** (recorded decision, `.claude/SESSION.md` 2026-06-22): (1) the M4 installer ships and **RG-012** (clean-machine smoke) passes — the Tauri app must be the *proven* primary UI before the fallback is deleted; (2) **RG-011** (first-token latency vs the Chainlit baseline) is recorded — the comparison is gone once Chainlit is. Until then this is a written contract only.
**Closes:** KI-2 (the Python-3.12 runtime pin exists *because of* Chainlit/anyio; it dissolves when Chainlit is gone). Removes the last "UI = Chainlit (web)" stack coupling.

**Requirement (the why).** ADR-002 makes the migration "additive, then a one-file deletion." M0–M4 added the `ChatController` core, the FastAPI/SSE boundary, the Tauri frontend, and the installer *beside* Chainlit, parity-guarded the whole way. M5 removes Chainlit — the dep, the renderer, its config, its recipe — and lifts the 3.12 runtime pin it forced. This is a **deletion + a constraint-lift**, not a feature: no `src/` business logic changes; the CLI, API, Tauri shell, eval harness, and 7d engine are untouched.

---

## ADR-1 — Delete Chainlit only after Tauri is the proven primary UI

**Context.** Chainlit is a 185-line thin renderer over the shared `ChatController` (since M0), parity-guarded, near-zero ongoing cost except the Py-3.12 runtime pin (KI-2). Deleting it before the Tauri installer is proven on a clean machine would leave no working GUI if M4 slips.

**Decision.** Sequence M5 dead last and gate it on the M4 ship (RG-012 pass + RG-011 recorded). The deletion itself is mechanical and low-risk *because* M0 already moved all turn logic out of `apps/chainlit_app.py` into `src/` — removing the file removes presentation only. The CLI remains as the headless/debug renderer; Tauri is the primary GUI.

**Options considered.** (1) *Delete now, alongside M4 (rejected)* — risks a GUI-less window if the freeze/installer needs more iteration. (2) *Keep Chainlit indefinitely as a web fallback (rejected)* — perpetuates the 3.12 pin (KI-2) and a second renderer to keep in parity, for a web UX the project explicitly moved away from (ADR-002 Context). (3) *Gate on the M4 ship, then delete (chosen)* — the additive-then-delete plan ADR-002 already commits to.

## ADR-2 — Lift the 3.12 pin by verification, not assumption

**Context.** KI-2: Chainlit's anyio stack breaks on Python 3.14, so the *runtime* is pinned to 3.12 (dev already runs on 3.14 minus Chainlit — `.claude/CONTEXT.md`). ADR-002 Consequences: "KI-2 dissolves once Chainlit is gone, **pending no other 3.12-only dep — verify in PR-M5**."

**Decision.** After deleting Chainlit, **prove** the lift: `uv sync --python 3.14 --extra cpu --extra dev` then run the full gate (ruff/mypy/bandit/`pytest tests/unit tests/integration`) on 3.14. Only if green do we declare KI-2 resolved and update the runtime docs. `requires-python = ">=3.10"` already permits 3.14; `mypy python_version`/`ruff target-version` are 3.10 *floors* (the supported-minimum), not 3.12 pins — they stay. The only literal 3.12 pin in active config is the `just chainlit` recipe's `--python 3.12`, which is deleted with the recipe. If a different transitive dep turns out to be 3.12-bound, KI-2 stays open with the new cause named (don't silently re-pin).

---

## Decisions

| # | Decision |
|---|---|
| 1 | **Delete `apps/chainlit_app.py`** — the renderer. All turn logic already lives in `chat_controller` (M0); this is a presentation-only deletion. |
| 2 | **Delete `.chainlit/`** (config.toml + any `translations/`) and a root `chainlit.md` welcome file if present. |
| 3 | **`pyproject.toml`** — remove the `chainlit>=2.0,<3.0` base dep and the now-unused `chainlit.*` mypy override (mypy already reports it unused). `fastapi`/`uvicorn`/`sse-starlette` stay (made explicit base deps in M2 *precisely* so this removal is clean — they no longer ride in via Chainlit). Drop the `Programming Language :: Python :: 3.12`-only framing if it implies a pin (keep 3.10–3.14 classifiers as supported). `uv lock` regenerated (`UV_NATIVE_TLS=1`). |
| 4 | **`justfile`** — delete the `chainlit` recipe (the lone `--python 3.12` pin goes with it). `api` + `desktop` recipes become the GUI entrypoints. |
| 5 | **Parity test** — `tests/integration/test_turn_parity.py`: drop any Chainlit-specific assertion; keep CLI + API (TurnResult) parity. Confirm **no test imports `chainlit`**. |
| 6 | **3.12-pin lift (ADR-2)** — verify the full stack on Python 3.14 (`uv sync --python 3.14 …` + full gate). If green: KI-2 → RESOLVED; if not: KI-2 stays open with the new 3.12-only dep named. |
| 7 | **Docs** — `.claude/CONTEXT.md` (UI stack row drops "Chainlit (web)"; runtime row drops the 3.12-for-Chainlit note once ADR-2 verifies); `docs/architecture.md` (remove the `apps/chainlit_app.py` row); `CLAUDE.md` ("Chainlit or CLI" → "Tauri desktop or CLI"; remove the 3.14/Chainlit runtime quirk); `README.md` (replace Chainlit run instructions with `just api` + `just desktop` / the installer); `docs/ROADMAP.md` M5 → done; KI-2 → resolved; DEVLOG entry. |

**Edge cases (spec explicitly):**
- *RG-011 baseline* — RG-011 measures first-token latency "vs Chainlit." Record that number **before** deleting Chainlit (it's the last chance to compare); afterward it's a historical baseline, not a live gate.
- *A non-Chainlit 3.12-only dep surfaces on the 3.14 gate* — do **not** delete the pin. Keep KI-2 open, name the dep, and ship M5 with the runtime still 3.12 (Chainlit still gone — the renderer deletion is independent of the pin lift).
- *`docs/desktop-packaging.md` / RG-011 references to Chainlit* — leave as historical context (they describe a measurement that happened); don't rewrite history.

---

## Files touched

**Deleted:** `apps/chainlit_app.py` · `.chainlit/` (config.toml + translations) · root `chainlit.md` (if present).
**Modified:** `pyproject.toml` (dep + mypy override + classifiers) · `uv.lock` · `justfile` (drop `chainlit` recipe) · `tests/integration/test_turn_parity.py` · `.claude/CONTEXT.md` · `.claude/KNOWN_ISSUES.md` (KI-2) · `docs/architecture.md` · `CLAUDE.md` · `README.md` · `docs/ROADMAP.md` · `docs/DEVLOG.md`.
**Unchanged:** all of `src/doc_assistant/` (no business-logic change), `apps/cli.py`, `apps/api/`, `apps/desktop/`, the eval harness.

## Definition of done

1. **No `chainlit` anywhere** — `grep -rni chainlit src tests apps pyproject.toml justfile docs/architecture.md` returns nothing (archive docs + this ADR/spec history excepted). No `import chainlit` in the tree; `chainlit` absent from `pyproject.toml` + `uv.lock`.
2. **Gate green on 3.12** (the current runtime): ruff · ruff format --check · `mypy --strict src` · `bandit -r src` · `pytest tests/unit tests/integration` (≈601 expected; the parity test still passes with the Chainlit arm removed).
3. **3.12-pin lift proven (ADR-2):** the same gate green on **Python 3.14** (`uv sync --python 3.14 --extra cpu --extra dev`). KI-2 → resolved (or stays open with a named cause).
4. **GUI still works** — `just api` + the Tauri app (or `just desktop`) launch and answer; the CLI launches. Eyeballed + noted in the DEVLOG.
5. **Docs** — CONTEXT/architecture/CLAUDE/README/ROADMAP updated; KI-2 closed; DEVLOG entry; this spec → ✅ BUILT.

---

## In / Out

**In:** deleting the Chainlit dependency + renderer + config + recipe; the parity-test trim; verifying + lifting the 3.12 runtime pin; the docs sweep + KI-2 close.

**Out (do not scope-creep):**
- **No `src/` changes** — M5 is deletion + constraint-lift; the core is frozen.
- **No new Tauri/API features** — those are post-migration work (the parked Phase-8 settings UI, the in-app PDF viewer, `SourceAdapter`).
- **No rewrite of historical RG-011/packaging notes** that reference the Chainlit baseline — they record a real measurement.
- **No bump of the mypy/ruff 3.10 floor** — those are supported-minimum settings, not the Chainlit pin; lifting the *runtime* to 3.14 is independent of the supported-floor.

<!-- status: active · updated: 2026-06-20 · class: disposable -->

# PLAN — PR-A: cpc scaffolding for doc_assistant

Disposable increment plan for **PR-A** of the cpc migration. Pairs with
[`docs/decisions/ADR-001-adopt-cpc-standard.md`](decisions/ADR-001-adopt-cpc-standard.md) — the ADR
is the *why* and the full contract; this is the *do*, as a checked list for one Claude Code session.
**Archive to `docs/archive/` when PR-A lands** (cpc CONVENTIONS §2).

**Scope of PR-A:** scaffolding + mechanical reshape only. The `decisions.md` → ADR split is **PR-B**
and is explicitly **out of scope here**. Docs-only — no `src/` change expected.

**Branch:** `chore/adopt-cpc`  **Run from:** `C:\Projects\doc_assistant`
**Hard rule:** never `git commit`/`push` without Lucas's review (cpc §13). Stage → summarize → stop.

---

## Pre-flight (read first, don't skip)

- [ ] Read cpc `CONVENTIONS.md` §1–§3, §5, §10–§11 and `ADR-001` (this repo) before touching files.
- [ ] Confirm clean working tree on a fresh branch: `git status` clean, `git checkout -b chore/adopt-cpc`.
- [ ] **Two known collisions to expect** (verified from cpc `init.py`), handled in Steps 2 and 5:
  - `cpc-init` lays a **new** `docs/ROADMAP.md` template *alongside* the existing
    `docs/doc-assistant-roadmap.md` → you must merge into one canonical `docs/ROADMAP.md`.
  - `cpc-init` does **not** modify an existing `.pre-commit-config.yaml` (it only advises) → cpc's
    gates must be merged into the current config **by hand** (Step 6).

---

## Step 0 — Make `cpc-init` runnable

- [ ] Install cpc editable into the project venv:
  ```
  uv pip install -e <local cpc checkout>
  ```
- [ ] Verify entry-points:
  ```
  cpc-init --help
  cpc-init-check --help
  cpc-docs-check --help
  ```
- [ ] If not on PATH, use module form everywhere downstream: `python -m cpc.init`,
  `python -m cpc.init_check`, `python -m cpc.docs_check`, `python -m cpc.backfill_headers`.
- [ ] Record which form worked (entry-point vs `-m`) — note it in the DEVLOG entry for this PR.

## Step 1 — Dry-run, then lay the layout

- [ ] **Dry-run first** and read the printed plan:
  ```
  cpc-init --root . --profile standard --dry-run
  ```
- [ ] Confirm every line for an **existing populated file** reads `= exists, NOT modified` (esp.
  `docs/architecture.md`, `CLAUDE.md`, `.claude/SESSION.md`, `.pre-commit-config.yaml`). If it
  proposes to *create/overwrite* any of those, **stop and report** before writing.
- [ ] Apply:
  ```
  cpc-init --root . --profile standard
  ```
- [ ] Confirm it **added** (new files): `.claude/CONTEXT.md`, `.claude/KNOWN_ISSUES.md`,
  `.claude/.gitignore`, `scripts/conventions.toml`, `docs/decisions/ADR-000-template.md`,
  `docs/sprints/SPRINT-000-template.md`, `docs/ROADMAP.md` (template — see Step 5).
- [ ] Confirm it **left intact**: `CLAUDE.md`, `docs/architecture.md`, `.claude/SESSION.md`,
  existing `docs/*` content, `.pre-commit-config.yaml`.

## Step 2 — Fill `.claude/CONTEXT.md` (judgment)

`CONTEXT.md` is the new **single source** for stack, locked settings, provider config, phase map,
open questions (cpc §5). doc_assistant's `CLAUDE.md` already *references* it as if present — now make
it real.

- [ ] Lift **Stack / runtime** from `CLAUDE.md` "Setup" + `README` + `docs/architecture.md`.
- [ ] Lift the **locked-settings table** (TOP_K=10, parent-child default, provider config, torch
  backend auto, coverage floor 40%, etc.) from `CLAUDE.md` "Engineering standards" + `decisions.md`
  locked-settings calls. This table now lives **here**, not in `CLAUDE.md`.
- [ ] Add the **phase map** + open questions (condense from the roadmap; full detail stays in ROADMAP).
- [ ] No secrets / tokens / absolute user paths in the file (cpc redaction rule).

## Step 3 — Fill `.claude/KNOWN_ISSUES.md` (judgment)

- [ ] Migrate the live issues already named in `CLAUDE.md` / `README`:
  - `print()` in `src/` vs the `structlog`-only standard (open violation).
  - Python 3.14 + Chainlit runtime quirk.
  - win32 cu130 segfault — **mark resolved** (`torch-backend = "auto"`, commit `423cbfa`).
  - sandbox-sync quirk.
- [ ] Do **not** invent issues. Each entry: symptom → status (open/resolved) → pointer.

## Step 4 — Backfill status headers on pre-existing docs

The gate (`cpc.docs_check`) requires line 1 to carry **all three**: `status:` ∈
{active,superseded,archived}, `class:` ∈ {append-only,living,disposable}, `updated: YYYY-MM-DD`.

- [ ] Run the helper (module-only — no console entry-point):
  ```
  python -m cpc.backfill_headers --root .
  ```
- [ ] **Hand-verify the class on each file** (the script guesses; you confirm):
  - **append-only:** `docs/DEVLOG.md`, `.claude/SESSION.md`, every `docs/decisions/ADR-*.md`.
  - **living:** `docs/architecture.md`, `docs/ROADMAP.md`, `.claude/CONTEXT.md`,
    `.claude/KNOWN_ISSUES.md`, `docs/figures-and-tables.md`, `docs/how-answers-work.md`, `docs/DEMO.md`.
  - **disposable:** `docs/chunking-sweep-rtx-resume.md` (dated, one-off), this `PLAN_*` file.
  - **exempt (leave as-is):** `docs/specs/**` (header-exempt by default `conventions.toml`),
    `docs/library.bib` (not markdown).
- [ ] Spot-check 3 files by eye — confirm the header is literally line 1 (nothing above it).

## Step 5 — Resolve the ROADMAP collision + reshape the table

- [ ] **Decide the canonical file:** consolidate into `docs/ROADMAP.md` (cpc's expected name, §11).
  Move the real content from `docs/doc-assistant-roadmap.md` into it; keep the cpc template's header.
- [ ] Reshape the **implementation table** to cpc columns `| PR | Scope | Status | Spec |` so
  `cpc-roadmap-sync` can parse it. Map current `| # | PR title | Files | Effort | Depends on |`:
  - `# / PR title` → `PR` + a concise `Scope`
  - add a `Status` cell per row: `done` / `in-progress` / `planned`
  - `Spec` → the `docs/specs/...` link where one exists, else `—`
- [ ] Keep the phase-overview table as secondary prose (only the PR table is machine-read).
- [ ] Supersede the old file: set `docs/doc-assistant-roadmap.md` header to `superseded`, then
  `git mv docs/doc-assistant-roadmap.md docs/archive/` (§2 forces the move). A `docs/X` reference
  still resolves to `docs/archive/X`, so links don't break.
- [ ] Grep for inbound references to the old name and update the obvious ones (`CLAUDE.md`,
  `docs/architecture.md`): `git grep -n "doc-assistant-roadmap"`.

## Step 6 — Merge cpc gates into the existing pre-commit config (by hand)

`cpc-init` won't touch the existing `.pre-commit-config.yaml`. Add cpc's gates to it manually.

- [ ] Add the cpc local hooks so the standard set runs: `cpc-docs-check` (pre-commit),
  `cpc-integrity-check` (pre-commit), `cpc-sprint-check` (pre-push), `cpc-init-check` (pre-commit),
  plus the §13 protocol gates `cpc-push-guard` (pre-push), `cpc-coupling-check` (commit-msg),
  `cpc-test-api-check` (pre-commit). Use the generated config from a throwaway
  `cpc-init --root /tmp/cpc-probe --profile standard` as the authoritative snippet to copy hook IDs
  and stages from — don't hand-invent them.
- [ ] Ensure the extra hook stages are enabled (cpc emits this for `standard`):
  `default_install_hook_types: [pre-commit, commit-msg, pre-push]`.
- [ ] Re-install hooks: `pre-commit install --install-hooks` (and `--hook-type commit-msg --hook-type pre-push` if your pre-commit version needs them explicitly).
- [ ] Keep the existing ruff/mypy/bandit/detect-secrets hooks — this is additive.

## Step 7 — Trim root `CLAUDE.md` to a pointer

- [ ] Remove restated locked-settings / engineering-standards prose now living in `.claude/CONTEXT.md`;
  replace with a short digest + a pointer (cpc §5).
- [ ] Keep: the coordination-triad routing, tool-split table, handoff/build protocols, and the
  `## Skills in play` table (already cpc-shaped).
- [ ] Budget check: root `CLAUDE.md` + `.claude/CONTEXT.md` **≤ 600 lines combined** (§7,
  `conventions.toml`). If over, cut from `CLAUDE.md` (push detail into CONTEXT/architecture).

## Step 8 — Add the DEVLOG entry for this PR

- [ ] Append a PR-A entry to `docs/DEVLOG.md` (What / Why / Rejected / Opens). Note: DEVLOG
  **re-ordering to newest-first is deferred to its own commit** per ADR-001 Step 5 — for PR-A just
  add the entry in the file's current convention and flag the pending inversion under "Opens".
  *(If you choose to fold the inversion into PR-A, do it as the final, isolated commit and prove the
  diff is pure reorder.)*

## Step 9 — Verify (don't declare green early)

- [ ] ```
  cpc-docs-check --root . --strict
  cpc-init-check --root . --profile standard
  ```
- [ ] Fix every flag: broken routes, entry-context over budget, missing/incorrect headers, an
  `active` disposable past staleness, archived docs still outside `docs/archive/`.
- [ ] Run the project test suite — confirm `src/` is untouched and nothing broke (expected: all green;
  this PR is docs-only).
- [ ] `git add -A`, then **summarize the staged diff for Lucas and stop.** No commit without review.

---

## Definition of done (PR-A)

- [ ] `cpc-docs-check --strict` passes (0 errors).
- [ ] `cpc-init-check --profile standard` passes (required artifacts present).
- [ ] `.claude/` triad complete and filled (CONTEXT, SESSION, KNOWN_ISSUES).
- [ ] All non-exempt docs carry a correct, correctly-classed status header.
- [ ] One canonical `docs/ROADMAP.md` with the cpc PR table; old roadmap superseded + archived.
- [ ] cpc gates merged into `.pre-commit-config.yaml` and installed.
- [ ] Root `CLAUDE.md` trimmed to a pointer; combined entry budget ≤ 600 lines.
- [ ] DEVLOG entry added; DEVLOG inversion tracked (here or done as its own commit).
- [ ] Diff staged and summarized; **decisions.md split NOT started** (that's PR-B).
- [ ] This file moved to `docs/archive/` once the PR lands.

## Explicitly out of scope (→ PR-B)

- Splitting `docs/decisions.md` (1620 lines, ~50 decisions) into `docs/decisions/ADR-NNN-*.md`
  starting at **ADR-002**. Heavy judgment; its own reviewed PR. See ADR-001 Step 4.

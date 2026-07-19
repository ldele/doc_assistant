<!-- status: active · updated: 2026-07-19 · class: append-only -->

# ADR-021 — Adopt the cpc big-project layout: AGENTS.md entry, module CLAUDE.md files, vendored gates on every box

- **Status:** accepted
- **Date:** 2026-07-19
- **Deciders:** Lucas (executed with Claude Code)

## Context

doc_assistant adopted cpc in ADR-001 but **consciously deferred the cpc ADR-014 entry layer**
(`AGENTS.md` canonical + `CLAUDE.md` stub) and never laid module-level `CLAUDE.md` files — the
root `CLAUDE.md` was the single entry file. Three pressures ended the deferral:

1. **Scale.** The backend is 60+ modules across four real boundaries (backend library, desktop
   frontend, API shell, CLI runners). A single root entry file either stays too thin to help
   inside a boundary or grows past its budget. cpc §9's threshold ("3+ real module boundaries")
   is clearly met.
2. **Agent quality.** Sessions repeatedly mis-handle module-local traps (frontend/backend wire-type
   drift, the `--apply`-wipes-Node-B footgun, the credit-leak default) that belong next to the code
   they bite, not in the root file. Module `CLAUDE.md` files load that context only when the
   subtree is touched.
3. **Tooling drift between boxes.** This box still ran the pre-vendoring cpc wiring
   (`.pre-commit-config.cpc.yaml` pip-installing a pinned SHA from the **private** remote —
   network at hook time, version skew vs the work box, which vendored 1.2.2). Conventions tooling
   also had no clean home separate from project scripts.

## Options

1. **Stay CLAUDE.md-canonical, add module files only.** — Smallest change, but the entry file
   stays Claude-branded and non-portable (cpc ADR-014 option 3's known weakness), and the
   ADR-014 deferral note has to be maintained forever against `cpc-init-check`.
2. **Adopt the full big-project layout** — `AGENTS.md` canonical + `CLAUDE.md` `@AGENTS.md` stub
   (stub-stays-stub gate on), per-module `CLAUDE.md` at the four real boundaries, cpc v1.2.3
   vendored at `tools/conventions/` on **every** box, local hooks rewired to the vendored copy. —
   Portable entry file, module context on demand, one tooling story on both boxes. **Chosen.**
3. **Symlink `CLAUDE.md` → `AGENTS.md`.** — Rejected for the same reason cpc ADR-014 rejects it:
   on Windows/clone toolchains a symlink silently degrades to a plaintext file.

## Decision

Adopt **option 2**, executed 2026-07-19:

- **Entry layer:** `AGENTS.md` is the canonical, tool-neutral entry file (content ported from the
  old root `CLAUDE.md`, plus the sub-module map and the cpc ADR-020 keypoints table);
  `CLAUDE.md` is a bare `@AGENTS.md` import stub, enforced by `[entry] enforce_stub = true` in
  `scripts/conventions.toml` (docs_check rule 9).
- **Module files (≤40 lines each, locals only, globals by code):** `src/doc_assistant/CLAUDE.md`,
  `apps/desktop/CLAUDE.md`, `apps/api/CLAUDE.md`, `scripts/CLAUDE.md`.
- **Conventions tooling separated from scripts:** the gates live **only** in the vendored,
  gitignored `tools/conventions/cpc/` (cpc **1.2.3**, laid by `cpc-init --profile standard` from
  the local cpc checkout at the release tag) plus a `rungate.py` shim; `.pre-commit-config.cpc.yaml`
  (gitignored) now runs that vendored copy — **no pip-install from the private remote, no network
  at hook time**. `scripts/` keeps only project runners plus `scripts/conventions.toml`, which
  stays there because cpc resolves that path (`cpc/_config.py`) — it is gate *config*, not a
  script, and is labeled as such.
- **Previously-deferred artifacts laid:** `GLOSSARY.md` (filled with the pinned
  concept/keyword/family/skeleton vocabulary — the 2026-07-17/18 "junk labels" trap is exactly a
  vocabulary-drift failure) and the justfile gains facade recipes (`just check` / `just lint` /
  `just keypoint <name>`, cpc ADR-011 — aliases only).
- **`cpc-init-check` now passes** (AGENTS.md exists) and stays **on-call**, not a wired hook.

## Consequences

- **Easier:** any AGENTS.md-aware tool reads the project; module context loads only when its
  subtree is touched; both boxes carry the same vendored gate version with no remote dependency;
  the root entry file stays lean as the project grows.
- **Commits us to:** keeping `CLAUDE.md` a bare stub (gate-enforced); maintaining four module
  files ≤40 lines (budget-gated); re-vendoring via `cpc-init` from release tags only; updating
  the module map here + in `AGENTS.md` when a boundary is added or removed.
- **Costs:** one more file per module boundary; contributors without the private cpc repo cannot
  run `just check`/`lint`/`keypoint` (unchanged from before — the shared `.pre-commit-config.yaml`
  gates still run for everyone).
- **Reverses if:** the AGENTS.md standard dies (drop the stub, return to CLAUDE.md-canonical — cpc
  ADR-014's own reversal clause) or a module boundary dissolves (delete its file, update the map).

## Links

- cpc ADR-014 (AGENTS.md entry) · cpc ADR-015 (vendor gates at init) · cpc ADR-020 (keypoints) ·
  cpc ADR-006 (profiles) · cpc CONVENTIONS §1/§5/§7/§9.
- [ADR-001](ADR-001-adopt-cpc-standard.md) — the original adoption this completes.
- [ADR-022](ADR-022-docs-system-rationalization.md) — the companion docs-system decision.
- `.claude/CONTEXT.md` → "cpc gate wiring" — the wiring commands (canonical text).

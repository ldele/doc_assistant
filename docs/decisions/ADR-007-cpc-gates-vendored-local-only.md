<!-- status: active · updated: 2026-07-02 · class: append-only -->

# ADR-007 — cpc gates delivered vendored + local-only (never committed, never in CI)

- **Status:** accepted
- **Date:** 2026-07-02
- **Deciders:** Lucas (with Claude Code)

## Context

ADR-001 adopted the cpc standard, whose gates (`docs_check`, `push_guard`, `coupling_check`,
`test_api_check`, `init_check`, `sprint_check`) need a delivery path into this repo. The adoption-era
plan (cpc repo, `migrations/doc-assistant-migration.md` §0, locked 2026-06-18) chose *pre-commit
`repo:` entry pinned to tag v0.1.0* — but it was never wired: no cpc hook ever appeared in
`.pre-commit-config.yaml`, `.git/hooks/`, or CI, so the "gate-enforced" claim in `CLAUDE.md` was
false in practice. Meanwhile cpc's own ADR-015 reversed remote `repo:` delivery in favour of
**vendoring** the gate package into each consumer at `tools/conventions/cpc/` via `cpc-init`
(cpc CHANGELOG 1.1.0 removed the pre-ADR-015 hook remnants). The binding constraint here: **cpc is a
private tooling repo; doc_assistant is a public portfolio repo** — `.gitignore` already reserves
`.pre-commit-config.cpc.yaml` as "local-only convention gates … never shared (see ADR-001)".

## Options

1. **pre-commit remote `repo:` pinned to a cpc tag** (the 2026-06-18 lock) — every machine and CI
   must authenticate against the private GitHub repo; an external cloner's `pre-commit install` breaks.
   cpc itself abandoned this path (cpc ADR-015).
2. **Vendor committed** (`tools/conventions/cpc/` in-tree, cpc ADR-015's default) — zero-dependency and
   CI-runnable, but publishes the private repo's code in a public repo; violates the ADR-001 boundary.
3. **Run from the sibling cpc checkout via `PYTHONPATH`** — zero repo footprint, but no per-repo
   `_VERSION` pin and silently breaks when the checkout moves or is absent.
4. **Vendor local-only (chosen):** `cpc-init` vendors into `tools/conventions/` and wiring lives in
   `.pre-commit-config.cpc.yaml` — **both gitignored**. Keeps ADR-015's mechanics (offline,
   `_VERSION`-pinned, upgrade = re-run `cpc-init`) without publication.

## Decision

**Vendor the gates local-only (option 4), at cpc 1.1.0.** Deciding reason: it is the only option
that is both wired-by-standard-mechanics (ADR-015) and compatible with the private-cpc /
public-repo split that ADR-001 established. Concretely: `tools/conventions/` (gate package +
`rungate.py` shim — pre-commit local hooks on Windows cannot set `PYTHONPATH` inline) and
`.pre-commit-config.cpc.yaml` (docs/test-api checks + `cpc-push-guard` at pre-push,
`cpc-coupling-check` at commit-msg; install with
`pre-commit install -c .pre-commit-config.cpc.yaml -t pre-push -t commit-msg`). `cpc-init-check` is
deliberately unwired: it requires the `AGENTS.md` entry file (cpc ADR-014) whose adoption is
deferred, and a by-design-red gate must not block pushes — it runs on-call.
**Reverses if** cpc becomes public (then option 2, committed vendoring + CI, strictly dominates) or
if per-machine setup proves unreliable in practice.

## Consequences

- **Easier:** gates actually run (they never did); offline + version-pinned; adopting a newer cpc is
  one `cpc-init` re-run; nothing private is published.
- **Harder:** enforcement is per-machine, not CI — a fresh machine must re-run `cpc-init` from the
  cpc checkout and `pre-commit install -c .pre-commit-config.cpc.yaml -t pre-push -t commit-msg`
  before the gates fire. CI stays gate-free **by design**; do not "fix" that without superseding
  this ADR.
- **Must-revisit:** the deferred `AGENTS.md` entry-file adoption (cpc ADR-014) — until then
  `cpc-init-check --profile standard` stays red on that one artifact; `sprint_check` joins the
  local config when sprint contracts are adopted (migration plan step 10, phase 2).

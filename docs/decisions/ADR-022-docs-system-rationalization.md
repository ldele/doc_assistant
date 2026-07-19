<!-- status: active · updated: 2026-07-19 · class: append-only -->

# ADR-022 — Docs-system rationalization: what earns its place at 0→10k-doc scale

- **Status:** accepted
- **Date:** 2026-07-19
- **Deciders:** Lucas (executed with Claude Code)

## Context

The docs tree accumulated **six per-feature/decision layers** (monolithic `decisions.md`, per-file
ADRs, `specs/`, `sprints/`, `features/`, loose explainers) plus a DEVLOG whose ordering was left
**hybrid** by the cpc port (ADR-001 Step 5 prepended new entries newest-first but never inverted
the 103-entry `## Session:` historical block, so "read the tail" and "read the top" were both half
right). ADR-001 Step 4 (split the 1578-line `decisions.md` into ~50 per-file ADRs) was never
executed — a month of sessions worked around it. As the project scales (more features, more
delegated agent sessions), every redundant or half-migrated layer is context an agent loads,
misreads, or re-derives. The user's directive: decide which artifacts **earn their place**.

## Options

1. **Execute ADR-001 Step 4 as written** — split the monolith into ~50 micro-ADRs. — *Cons:* most
   of those decisions are (a) already re-recorded in ADR-002..021, (b) superseded, or (c) phase
   status logs that belong in ROADMAP; a 50-file dump of mostly-dead ADRs makes the decision
   record *harder* to load, and splitting the still-live retrieval rationale across 50 files
   duplicates what `.claude/CONTEXT.md`'s locked-settings table already points at.
2. **Archive the monolith wholesale + a living index; per-file ADRs only going forward.** —
   *Pros:* one canonical decision home (`docs/decisions/`), history frozen and addressable,
   zero duplication, one cheap index for discovery. **Chosen.**
3. **Leave everything as is.** — *Cons:* the half-migrated state is the worst of both; the gate
   can't reason about a "living" 1578-line monolith that is actually frozen history.

## Decision

**Option 2**, with a per-artifact verdict (the answer to "which of specs, ADRs, sprints, …"):

| Artifact | Verdict | Rule going forward |
|----------|---------|--------------------|
| `docs/decisions/` (ADRs) | **KEEP — canonical.** | Every decision = one `ADR-NNN-slug.md`, append-only; a one-line entry in the `docs/decisions.md` index rides along. |
| `docs/decisions.md` | **REPLACED by a living index.** | The 1578-line monolith is frozen verbatim at `docs/archive/decisions-monolith.md`; cite it for pre-cpc rationale (locked retrieval/chunking settings). Existing `docs/decisions.md` references resolve to the index. |
| `docs/specs/` | **KEEP — the code-level contract layer.** | One spec per feature; header-exempt; the ROADMAP `Spec` column links it. The spec is what an executor session builds against. |
| `docs/sprints/` | **KEEP — the delegated-execution mechanism.** | cpc §10/§11: ROADMAP row → `roadmap_sync` materializes a contract → `sprint_check` gates scope/docs → closeout report; archive on landing (already practiced, `docs/archive/sprints/`). This is the scale mechanism for "many agent sessions, one plan" — not ceremony. |
| `docs/features/` | **KEEP the layer, adopt for frontier features only; no backfill.** | The FEATURE template is the *why-it-works* layer (hypothesis → grounding → outcome) — exactly the record that would have prevented the over-optimize-on-current-corpus failure mode. Use it for new features with no prior art (tier `frontier`); shipped features keep their rationale in ADRs/specs/baselines. |
| `docs/DEVLOG.md` | **KEEP; ordering fixed once.** | The historical block is now inverted (one logged reformat, entry bodies byte-identical, `## Session: ` prefixes stripped) — the whole file is newest-first; "read the top" is finally true. |
| Loose explainers (`how-answers-work`, `figures-and-tables`, `desktop-packaging`, `ui-checklist`, `DEMO`) | **KEEP as living reference.** | No moves; they are load-bearing (runbook, phase-8 status, demo script). |
| `docs/archive/` | **KEEP — the only place disposables go to die.** | Unchanged (cpc §2). |

**Not done deliberately:** no bulk `roadmap_sync` materialization of sprint contracts for planned
rows (done at the next plan-start keypoint, per row, when a sprint actually activates); no
FEATURE-file backfill for shipped features.

## Consequences

- **Easier:** an agent loads one decision home + one index; the DEVLOG reads top-down; the monolith
  stops masquerading as a living doc; the docs tree now has exactly one owner per question
  (*why* → ADRs/archive, *contract* → specs, *when/what next* → ROADMAP+sprints, *what happened* →
  DEVLOG, *how it works* → architecture + explainers, *why-it-works hypothesis* → features/).
- **Commits us to:** maintaining the index line-per-ADR; writing a FEATURE file when a frontier
  feature starts; keeping DEVLOG strictly newest-first.
- **Costs:** pre-cpc rationale is one indirection away (index → archive); acceptable — it is
  frozen history, cited far more than edited.
- **Reverses if:** the index rots (then generate it — a `[generate]` candidate for
  `cpc-generate`), or a future need demands per-decision extraction of specific monolith sections
  (extract *those sections* into new ADRs then, on demand — never the bulk).

## Links

- [ADR-001](ADR-001-adopt-cpc-standard.md) — Steps 4/5 this re-scopes and completes.
- [ADR-021](ADR-021-adopt-cpc-big-project-layout.md) — the companion entry-layer decision.
- cpc CONVENTIONS §1/§2/§10/§11 — the tiers, lifecycle classes, and sprint machinery.
- `docs/archive/decisions-monolith.md` — the frozen monolith.

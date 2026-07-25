<!-- status: active · updated: 2026-07-25 · class: append-only -->

# ADR-029 — Keep the working state local: what a public repo publishes (supersedes ADR-020)

- **Status:** accepted
- **Date:** 2026-07-25
- **Deciders:** user + Claude Code
- **Supersedes [ADR-020](ADR-020-share-rigor-todo-via-git.md)** (`RIGOR_TODO.md` committed via git —
  its deciding premise no longer holds). **Amends [ADR-001](ADR-001-adopt-cpc-standard.md)'s
  `.claude/` allowlist** (now empty) and narrows what [ADR-022](ADR-022-docs-system-rationalization.md)
  publishes (its per-artifact verdicts stand for ADRs/specs/sprints/DEVLOG; planning and review
  working docs move out of the repo).

## Context

Two facts changed on **2026-07-25**, both stated by the user:

1. **The second machine is retired.** Development happens on one box from now on; the remaining data
   transfer off the old box is pending. Every "two boxes" premise in this repo's decisions is dead.
2. **The repo is public and is read as a portfolio.** That was already true (ADR-001 keeps the cpc
   gates out of it because cpc is private), but it was never used to decide *which working artifacts*
   belong in the publication channel.

ADR-020's deciding reason was **specifically** the two-box problem: the boxes held disjoint
`RIGOR_TODO.md` copies, "keep it local and reconcile by hand" had been run as a 3-week experiment and
failed, and *"git is the mechanism the ritual was missing"*. With one machine there is no second copy
to converge with, so git is no longer a sync mechanism for any of these files — it is only a
publication channel. The obligation ADR-020 left open (*"must do next, on the work box: the merge"*)
is now unreachable on that box and survives only as whatever arrives in the data transfer.

The files in question are all **working state addressed to whoever is building this**, not to a
reader of the project: the deferred-rigor tracker, the canonical rules/gotchas files the entry file
points at, dated planning docs, a scale-review report, and a UI punch list.

## Options

1. **Status quo — keep them tracked.** *Pros:* nothing breaks; git keeps an off-machine copy; a fresh
   clone (or a cloud agent) gets the full rule text. *Cons:* publishes internal validation debt, box
   nicknames and half-finished punch lists on a repo read as a portfolio; the sync benefit that
   justified committing `RIGOR_TODO.md` no longer exists.
2. **Untrack the working set (chosen):** `.claude/{CONTEXT,KNOWN_ISSUES,RIGOR_TODO}.md` +
   `docs/{PLAN_*,REVIEW_*,ui-checklist}.md`. *Pros:* the public face narrows to README · ADRs · specs ·
   DEVLOG · ROADMAP — the product and its engineering record; working notes stay working notes.
   *Cons:* ~40 committed files link to these paths (109 occurrences), so a public reader hits dangling
   links; **git stops being the off-machine copy** exactly as the second machine goes away.
3. **Move the working set to a private sibling repo or submodule.** *Pros:* keeps an off-machine copy
   *and* keeps it unpublished; the honest long-term answer. *Cons:* a second repo to run for a solo
   project, and the cross-repo link problem is the same one option 2 has.
4. **Untrack `RIGOR_TODO.md` only** (a minimal ADR-020 revert). *Pros:* smallest change; keeps the
   agent-facing rule text in-repo. *Cons:* leaves the planning docs and the UI punch list published,
   which is the part the user actually objected to.

## Decision

**Untrack the working set; `.gitignore` gains it and `git rm --cached` removes it from tracking with
the files left on disk.** The committed allowlist under `.claude/` becomes **empty** — the whole
directory is local working state again.

*Deciding reason:* every file here is addressed to the builder, not the reader, and the one
counter-argument that had been decisive (cross-machine sync, ADR-020) is structurally gone with the
second machine. Option 3 is the better end state but is a second repo to maintain for a solo project;
it stays the named upgrade path if the backup consequence below bites.

**Three mitigations ship with it, because the honest costs are real:**

- **`docs/local-only.md` (committed)** names the local-only set and why, so a reader who hits a
  dangling `ui-checklist.md` link finds an explanation rather than a broken project.
- **`AGENTS.md` states it at the top of its coordination list**: the canonical rule text lives in
  local files, so its own digest is what an out-of-clone reader gets. The entry file keeps pointing
  at the local paths — that is correct *on the machine*, which is where agents run.
- **The backup gap is named, not glossed:** these files now exist in exactly one place. Whatever the
  user's backup routine is, it must cover `C:\Projects\doc_assistant\.claude\` and the local-only
  `docs/` files — git no longer does.

**What would reverse this.** A second machine or a collaborator returns (sync matters again → option
3, not a re-publish); or losing an untracked file hurts once (→ option 3 immediately).

## Consequences

**Easier.** The repo reads as the project rather than as the workshop: README, ADRs, specs, DEVLOG,
ROADMAP. Working notes can be blunt again — a punch list, a tracker of skipped rigor, and a "this box
is behind" note carry no publication cost, which is a mild but continuous tax removed.

**Harder.**
- **No off-machine copy.** The single largest cost, and it lands precisely when the second machine
  (previously a de-facto second copy of some of these) is retired. Named above; unresolved by this
  ADR beyond the recommendation.
- **~40 committed documents now link to paths not in the repo.** Accepted rather than rewritten:
  editing 109 references would churn append-only history files (DEVLOG, sprint archives) whose value
  is that they are verbatim. `docs/local-only.md` is the single explanation the links resolve to
  socially, if not mechanically.
- **A fresh clone has no `.claude/CONTEXT.md`.** An agent session on a *new* machine (or a cloud
  runner) sees only `AGENTS.md`'s digest and must be handed the rule text. That is a real onboarding
  step, previously free.
- **ADR-020's open merge obligation is now conditional on the data transfer.** RG-014, RG-007 and
  possibly RG-003/005/006 exist only on the retired box's copy of `RIGOR_TODO.md`; they arrive only if
  that file is part of the transfer. **RG-014 is cited as authority in ADR-017/018/019 and
  `docs/specs/feature-concept-graph.md`** — if the file does not arrive, those citations are
  permanently unverifiable and should be treated as such rather than trusted.

## Confidence

- ✓ **ADR-020's premise is gone** — its deciding reason is quoted above and is explicitly two-box.
- ✓ **Nothing in the untracked set is *required* for the code to build or the tests to run** —
  verified: `.claude/` and `docs/` are not importable and no test reads them; the cpc gates that do
  read them run **locally only** (ADR-001/ADR-007), where the files remain.
- ✓ **The dangling-link count is measured, not estimated** — 109 occurrences across 40 files
  (`ui-checklist|PLAN_2026|REVIEW_2026`, 2026-07-25).
- ⚠ **"Nothing of value is lost by not publishing"** — a judgment, not a measurement: the review
  report (`REVIEW_2026-07-19_scale-robustness.md`) is arguably the best evidence in the repo of
  engineering rigor, and hiding it has a portfolio cost pulling the other way. Revisit if the repo is
  ever used as a work sample: the fix is to promote that one file's findings into an ADR or a spec,
  which are published.
- ⚠ **The backup mitigation is a recommendation, not a mechanism** — the same class of failure ADR-020
  diagnosed ("a manual ritual with no mechanism behind it does not happen"). Option 3 is the mechanism
  if this proves true again.

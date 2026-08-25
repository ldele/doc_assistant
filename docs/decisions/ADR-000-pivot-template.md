<!-- status: active · updated: YYYY-MM-DD · class: append-only -->

# ADR-NNN — <the new direction, stated as a claim>

Template for a **direction change** (ADR-037). Copy to
`docs/decisions/ADR-NNN-pivot-<slug>.md`, fill it, then run `cpc-supersede --adr NNN`.

Use this instead of the plain `ADR-000-template.md` when a decision **invalidates work already
recorded** — not when it merely chooses between options. A pivot in week three is normal; a vault
that still describes the abandoned direction is what makes it expensive.

- **Status:** proposed | accepted
- **Date:** YYYY-MM-DD
- **Deciders:** <names>

## Context

<What changed in the world, not what you decided. The reader needs to know why the old direction
stopped being right — a user told you something, a measurement came back, a constraint appeared.
Be specific about when you learned it.>

## What we were doing, and why it was reasonable

<State the old direction fairly. A pivot ADR that makes the previous plan sound foolish is useless
six months later when you are deciding whether to go back, and it is unfair to whoever chose it —
often you.>

## The new direction

<One paragraph. What we are building now, and what decides whether it is working.>

## Supersedes

<!-- Every doc that now describes a direction we are not taking. `cpc-supersede --adr NNN` reads
     THIS list: it sets each file's status to superseded, bumps `updated:`, and inserts a banner
     naming this ADR, so the next agent sees it before the content.

     Paths must be in `backticks` — that is the form rule 4's route gate validates, so a typo here
     is a broken route rather than a silent skip.

     Do NOT list append-only LOGS (`.claude/SESSION.md`, `docs/DEVLOG.md`, `CHANGELOG.md`). They
     record what happened, which this does not make false; the command refuses them.

     Find what will break before you write this list: `cpc-docs-index --of <doc>` prints everything
     that points at a document. -->

- `docs/decisions/ADR-0NN-<the choice this reverses>.md` — <why it is dead>
- `docs/specs/SPEC-<feature that will not be built>.md` — <why>
- `docs/<guide that explains the old way>.md` — <why>

## What survives

<The parts of the old direction that are still right. Without this section a reader assumes
everything above the pivot is void, and throws away work that was fine.>

## Enforced by

<!-- What FAILS if someone violates this decision? Name the test, the gate, or the CI job.
     If nothing can — say so in those words. "judgment — not enforceable" is a valid and
     common answer; an ambiguous one is not. Rule 16 checks that this section EXISTS and is
     non-empty, never that the named check is real (ADR-007's presence/observance split). -->
judgment — not enforceable

## Consequences

<What this costs. What has to be rebuilt, what is thrown away, what the team has to unlearn.>

## Checklist for the pivot

- [ ] `cpc-supersede --adr NNN --dry-run`, read the plan, then run it for real
- [ ] `git mv` each superseded `class: disposable` doc to `docs/archive/` (the command prints them)
- [ ] Fix every rule 4d finding: `docs_check --strict` now flags living docs citing the retired
      decisions — name this ADR on the line, or repoint it
- [ ] **Update `.claude/CONTEXT.md`**: the goal, the current phase, and any locked setting this
      changes. Nothing enforces this and it is the step that gets skipped
- [ ] Re-check `.claude/NORTH_STAR.md` — a pivot often changes who the reader is, not just what is
      being built
- [ ] `docs/ROADMAP.md`: retire the phases that are now wrong rather than leaving them to rot
- [ ] Regenerate derived artifacts (`cpc-generate --write`) and re-run the gates

<!-- status: active · updated: 2026-07-18 · class: append-only -->

# ADR-020 — Share `RIGOR_TODO.md` via git (amends ADR-001's `.claude/` allowlist)

- **Status:** accepted
- **Date:** 2026-07-18
- **Deciders:** user + Claude Code

> **Scope.** This ADR changes **one line of `.gitignore`**: `.claude/RIGOR_TODO.md` joins `CONTEXT.md`
> and `KNOWN_ISSUES.md` on the committed allowlist. It **amends ADR-001**, which established that
> `.claude/` is gitignored except the Tier-1 canonical-facts files. It does not change what the tracker
> is for, who writes it, or the `rigor-gate` protocol. `SESSION.md` stays local.

## Context

`.claude/RIGOR_TODO.md` is the deferred-rigor tracker: every time experimental or performance discipline
is skipped, it gets an entry that is closed by doing the work or by a dated waiver. ADR-001 placed it
outside the committed allowlist, alongside the `SESSION.md` baton.

That grouping conflated two different kinds of file. The baton **is** per-machine state — "who worked
last on this box, what is staged here". The rigor tracker is **project** state: a validation debt of the
codebase, which is true regardless of which machine is in front of you.

**The measured consequence, on this repo, over ~3 weeks:** the two development boxes accumulated
**disjoint item sets**. The work box holds the low-numbered concept-graph items; the RTX/CPU box was
seeded independently on 2026-06-24 with the M4-freeze items RG-010–013, reconstructed from
`KNOWN_ISSUES.md` and `DEVLOG.md` because the originals were unreachable. A partial reconcile on
2026-07-01 added RG-001/008/009. The file's own header has carried a "still to reconcile against the work
box" note since then; it was never done, because a manual ritual with no mechanism behind it does not
happen.

**The failure this produced is not hypothetical.** **RG-014 has no entry on this box** — while being
cited as authority in **ADR-017, ADR-018, ADR-019, `docs/specs/feature-concept-graph.md`, and
`docs/ui-checklist.md`** for the claim that `single_source` is the strong, low-volume gap signal. A week
of design decisions rested on an item nobody working here could read. On 2026-07-18 that same verdict was
found not to transfer across a vocabulary rescope (ADR-018), which is exactly the kind of bound a reader
would have checked had the text been available.

**Publication surface was checked, since this repo is public.** The file contains engineering
measurements, box nicknames ("work box", "RTX box"), and a reference to a corporate TLS-MITM proxy — the
last of which is *already* public in the committed `KNOWN_ISSUES.md` (KI-10). Scanned for absolute user
paths, credentials, tokens and hostnames: none present.

## Options

1. **Add `RIGOR_TODO.md` to the committed allowlist.** *Pros:* the sync problem disappears structurally
   rather than by discipline — git is the mechanism that the "reconcile later" note lacked; the tracker
   matches the lifecycle of `KNOWN_ISSUES.md`, which is already committed and has the same
   append-mostly, occasionally-edited shape. *Cons:* the content becomes public (assessed above:
   engineering measurements only); concurrent edits on two boxes can conflict — the same, tolerated,
   property `KNOWN_ISSUES.md` already has.
2. **Keep it local, add a reconciliation ritual** (e.g. a cpc `keypoint` checklist item at
   session-close). *Pros:* no publication surface; no merge conflicts. *Cons:* this *is* the status quo —
   the file has carried a "still to reconcile" instruction since 2026-07-01 and it has not been actioned
   in ~3 weeks. Adding a second reminder does not supply a mechanism.
3. **Fold rigor items into `KNOWN_ISSUES.md`** (already committed). *Pros:* one shared file, no
   `.gitignore` change. *Cons:* conflates two lifecycles — `KNOWN_ISSUES.md` records *defects and
   weaknesses*, `RIGOR_TODO.md` records *validation debt on work already believed correct*. The
   `rigor-gate` skill addresses `RIGOR_TODO.md` by name.
4. **Move it out of `.claude/` entirely** (e.g. `docs/RIGOR_TODO.md`), as `DEVLOG.md` already lives in
   `docs/`. *Pros:* sidesteps the allowlist; puts a shared doc among shared docs. *Cons:* breaks every
   existing reference (`CLAUDE.md`, `KNOWN_ISSUES.md`, archived session entries, the `rigor-gate` skill's
   own path) for no gain the allowlist entry does not already provide.

## Decision

**Add `!.claude/RIGOR_TODO.md` to the `.gitignore` allowlist.**

*Deciding reason:* the tracker records project validation debt, not machine state, and the "keep it local
and reconcile manually" alternative has already been run as a ~3-week experiment — it produced disjoint
trackers and a week of ADRs citing an unreadable item. Git is the mechanism the ritual was missing.

**The first sync is a merge, not an overwrite, and this is recorded in the file itself.** The work box
still holds `RIGOR_TODO.md` as an untracked ignored file; on `git pull`, git refuses to clobber an
untracked file, and **that refusal is the safety net — it must not be forced past**. The file's header
carries the merge procedure (rename local copy aside → pull → hand-merge the missing items → delete the
temp copy) and an explicit inventory of what is present versus known-missing.

**What would reverse this.** If the tracker ever comes to hold machine-specific or non-public content —
customer names, credentials, unreleased security findings — it moves back out of the allowlist and the
sharing problem gets solved a different way (a private submodule, or the cpc private tooling repo).

## Consequences

**Easier.** Both boxes converge on one tracker with no ritual. A deferred-rigor item written on one
machine is readable on the other, so an ADR can cite an RG item and a reader can actually check its
bounds. New items (RG-015 was added this session) reach the other box automatically.

**Harder.** Concurrent edits on two boxes can now conflict in git — the same property `KNOWN_ISSUES.md`
already carries, resolved the same way. The tracker's contents are now public, which is a mild ongoing
constraint on what may be written there (it was already the constraint on `KNOWN_ISSUES.md`).

**Must do next, on the work box.** The merge described in the file header. Until it happens the committed
copy is **incomplete** and says so in its own header: RG-014 (most urgent — actively cited), RG-007, and
possibly RG-003/005/006 exist only on the work box; RG-002/RG-004 are believed moot under the
concept-graph redesign and should not be re-run against the old graph.

## Confidence

- ✓ **The two copies are disjoint and this box is missing actively-cited items** — verified by inventory:
  this copy holds RG-001/008/009/010/011/012/013/015 and has **no** `## RG-014` section, while RG-014 is
  cited in five committed documents.
- ✓ **Nothing in the file is unsafe to publish** — scanned for absolute paths, credentials, tokens and
  hostnames (none); the one sensitive-sounding detail (corporate TLS-MITM proxy) is already public in
  `KNOWN_ISSUES.md` KI-10.
- ⚠ **The claimed gate does not exist.** The file's own line "The gate (`rigor_gate.py`) fails while any
  `blocks-ship` item is `open`" is **aspirational** — there is no `scripts/rigor_gate.py` in this repo and
  neither pre-commit nor CI reference it. Sharing the file does not make it enforcing; it remains a manual
  discipline doc. Wiring an actual gate is out of scope here and unticketed.
- ⚠ **The merge is not done.** This ADR makes the sync *possible*; the disjoint sets are only reconciled
  once the work box performs the documented merge. Until then "shared" and "complete" are different
  claims.

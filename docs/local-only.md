<!-- status: active · updated: 2026-09-10 (plans/ and reviews/ folders, ADR-051) · class: living -->

# Local-only working docs

Some files this repo's documents link to are **deliberately not in the repo**. They are working state
addressed to whoever is building the project, not part of its public record. The decision and its
trade-offs (including this one — dangling links) are recorded in
[ADR-029](decisions/ADR-029-local-only-working-state.md).

| Path | What it is |
|---|---|
| `.claude/CONTEXT.md` | Canonical facts: stack, locked settings, provider config, phase map, open questions. `AGENTS.md` carries the digest a reader needs. |
| `.claude/KNOWN_ISSUES.md` | Open weaknesses, recurring failures, workarounds (the `KI-nn` references). |
| `.claude/RIGOR_TODO.md` | Deferred-rigor tracker — validation debt on work believed correct (the `RG-nn` references). |
| `.claude/SESSION.md` | The cross-session handoff baton (always local — per-machine by nature). |
| `docs/plans/PLAN_<date>_*.md` | Dated planning docs for a work track (folder since ADR-051; the folder's README is tracked). |
| `docs/reviews/REVIEW_<date>_*.md` | Internal review reports (folder since ADR-051; the folder's README is tracked). |
| `docs/ui-checklist.md` | The per-feature UI review gate (its queue moved into `docs/ROADMAP.md` on 2026-09-10). |
| `docs/archive/local/` | Closed PLAN_/REVIEW_ docs and rotated narrative from `.claude/CONTEXT.md` / `RIGOR_TODO.md` — archived, still local-only. |
| `docs/reviews/REVIEW_2026-09-07_crossover-daily-learn.md` | A cross-project review (Daily Learn ↔ doc_assistant) whose §3 feeds ROADMAP rows 52–54. |
| `docs/archive/lab4tech/` | The retired second machine's baton history (49 rotated archives + its final baton), rescued from its backup 2026-07-25. Per-machine session history — and the only record of the 2026-07-23/24 taxonomy + UI sessions. |

**What *is* the public record:** [`README.md`](../README.md) ·
[`docs/decisions.md`](decisions.md) (the ADR index) · [`docs/specs/`](specs/) ·
[`docs/DEVLOG.md`](DEVLOG.md) · [`docs/ROADMAP.md`](ROADMAP.md) ·
[`docs/architecture.md`](architecture.md) · [`GLOSSARY.md`](../GLOSSARY.md).

A `KI-nn` or `RG-nn` citation you cannot resolve from a clone is one of these files, not a missing
document. Where such an item is load-bearing for a decision, the ADR that cites it states the claim
it rests on in full.

<!-- status: active · updated: 2026-09-10 · class: append-only -->

# ADR-051 — Working docs get folders (`docs/plans/`, `docs/reviews/`), and append-only logs keep a fixed entry count

**Status:** accepted (2026-09-10) · **Enforced by:** `.gitignore` (folder patterns), cpc `docs_check`
rules 5 and 13b via `scripts/conventions.toml`, `tests/unit/test_doc_sizes.py`,
`scripts/release_preflight.py` (`checklists`) · **Proposed upstream:** cpc tickets T-011 / T-012

## Context

cpc's naming rule puts dated working docs at the `docs/` root: `PLAN_<topic>.md` (disposable, one
increment) and `REVIEW_<date>_<topic>.md` (CONVENTIONS §naming). By 2026-09-10 this project had six
plans and two reviews there, beside twelve topic docs, and ADR-029 had made all of them local-only.
The root of `docs/` was a mix of public record and private working state that only the header and
the gitignore could tell apart; the 2026-09-10 review found three of the eight still marked
`active` while their bodies said "complete" or "expires with 0.6.0".

The same review found the append-only logs governed by a line cap alone: the DEVLOG at 3,922 of
4,000 lines and 76 entries, `KNOWN_ISSUES.md` carrying 955 lines of closed bodies against its own
"index rows only" rule. A line cap trips late and says nothing about *what* to keep.

And the two release checklists had no mechanism that made anyone refresh them; the user's stated
assumption (2026-09-10) is that they will *never* otherwise be up to date.

## Decision

**D1 — Folders.** `docs/plans/` holds every `PLAN_<date>_<slug>.md`; `docs/reviews/` holds every
`REVIEW_<date>_<slug>.md`. The naming rule is unchanged (cpc rule 5 scans `docs/` recursively, so
staleness detection needs no change). Each folder carries a tracked `README.md` stating the
convention; the dated files inside stay local-only by *pattern* (`docs/plans/PLAN_*.md`,
`docs/reviews/REVIEW_*.md`), never by ignoring the folder — a folder-level ignore would take the
README with it, and a fresh clone would see an empty directory with no explanation.

**D2 — Lifetime.** A plan lives exactly as long as a `docs/ROADMAP.md` row cites it in its `Spec`
column; a review lives until its findings have all become rows, KIs or ADRs. When the last citation
closes, the doc's header flips to `superseded` and the file moves to `docs/archive/local/` in the
same session (also gitignored: the archive of local-only state is local-only). The root `docs/`
holds only topic docs the public can read.

**D3 — Entry budgets, not just line caps.** `docs/DEVLOG.md` keeps the **newest 20 entries**
(`[budgets] devlog_max_entries = 20`; cpc rule 13b warns, the pytest guard fails; the line cap
stays as a backstop for entries that are ADRs in disguise). `.claude/SESSION.md` keeps 10 (already
cpc ADR-018). `.claude/KNOWN_ISSUES.md` keeps open issues in full and closed ones as one-line index
rows pointing at `docs/archive/KNOWN_ISSUES-resolved-NNN.md`. Rotation is `cpc-rotate`, verbatim and
byte-verified, never a hand cut.

**D4 — Release checklists are stale until proven otherwise.** `docs/release-ux-checklist.md` and
`docs/security.md` are refreshed as the *first* release step (`docs/RELEASE.md` §0), and
`release_preflight` fails the release if either has no commit or uncommitted edit since the
previous tag. The check verifies that the refresh happened, not that it was good — the judgment
stays in the runbook.

## Options considered

- **Keep everything at the `docs/` root** (cpc's current default). Rejected: the root stops being
  readable as the public record once it holds more working docs than topic docs, and the gitignore
  becomes the only map.
- **One `docs/working/` folder for both.** Rejected: plans and reviews have different lifetimes
  (D2) and different readers; two folders cost nothing and make the difference visible.
- **Header dates instead of git history for D4.** Rejected: a header can be bumped without a
  refresh; "touched since the previous tag" is the fact that matters, and `artifact_fresh` already
  judges freshness by history for the same reason (DEVLOG 2026-09-02).
- **A line cap only for the DEVLOG.** Rejected: it trips at 4,000 lines whatever the entry count,
  and the user's standard is "the newest 20".

## Consequences

- Links from the public record to a plan or review dangle in a clone, as before (ADR-029);
  `docs/local-only.md` names both folders so the dangling path is recognisable.
- Every session that writes a DEVLOG entry beyond the twentieth rotates one entry; the
  session-close checklist already says to run `cpc-rotate`.
- The archive range line in the DEVLOG header is still updated by hand (cpc ticket T-003).
- Proposed to cpc as the default layout (T-011): `cpc-init` lays the two folders with their READMEs,
  CONVENTIONS §naming names them, and the `.gitignore` template carries the two patterns.

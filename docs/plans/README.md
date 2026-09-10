<!-- status: active · updated: 2026-09-10 · class: living -->

# docs/plans/ — dated working plans

`PLAN_<date>_<slug>.md`, one per work track, `class: disposable`. **Local-only** (ADR-029): the
files are gitignored by pattern, this README is not, so a clone sees the convention and none of
the plans. A plan lives exactly as long as a `docs/ROADMAP.md` row cites it in its `Spec` column
(ADR-051 D2); when the last citation closes, its header flips to `superseded` and it moves to
`docs/archive/local/`. cpc rule 5 flags an `active` plan older than 90 days.

Live plans are listed where they are used: the roadmap's `Spec` column.

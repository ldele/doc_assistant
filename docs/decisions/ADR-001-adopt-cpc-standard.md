<!-- status: active · updated: 2026-06-20 · class: append-only -->

# ADR-001 — Adopt the project-conventions (cpc) standard for doc_assistant

- **Status:** accepted (2026-06-20 — applied via PR-A; see `docs/DEVLOG.md`)
- **Date:** 2026-06-20
- **Deciders:** Lucas (drafted with Cowork)

> This ADR is the **execution contract** for porting doc_assistant onto cpc. It is written so
> Claude Code can perform the migration without re-deriving the gap analysis. The standard itself
> is authoritative — this ADR points at it, never restates its rules. The standard (cpc) lives in
> a separate, private tooling repo; sections are cited as §N below.

## Context

doc_assistant already follows the cpc *philosophy* — a `.claude/` coordination tier, a `docs/`
reference tier, a session-start triad, append-only baton/DEVLOG, "never commit without review,"
the enrichment-layer pattern. It was built on the same mental model. But it predates the formal
standard and diverges on **format and enforcement**, so the deterministic gates (`docs_check.py`,
`sprint_check.py`) would fail and the tooling (`roadmap_sync.py`) cannot parse the current files.

This is an **alignment**, not a rewrite. The substance stays; the shape changes in a handful of
specific places. Two conflicts need human/judgment work and cannot be scripted: the monolithic
`decisions.md` must be split into per-decision ADRs, and the DEVLOG is ordered the wrong way.

### Verified state at time of writing (2026-06-20)

| Area | Current doc_assistant | cpc target (§) | Conflict? |
|---|---|---|---|
| `.claude/` triad | `SESSION.md` only | `SESSION.md` + `CONTEXT.md` + `KNOWN_ISSUES.md` (§1 Tier 1) | Missing 2 of 3 — though `CLAUDE.md` already *references* both as if present |
| Status headers | absent on all 16 docs | line-1 `<!-- status:… · updated:… · class:… -->` required (§3) | Yes — `docs_check` hard-fails without them |
| Decisions | one `decisions.md` (1620 lines, ~50 `###` decisions) | one decision per `docs/decisions/ADR-NNN-*.md`, append-only (§1, §4) | **Yes — biggest item, judgment-heavy** |
| DEVLOG | `docs/DEVLOG.md`, **oldest-first**, `## Session: DATE` headings | `docs/DEVLOG.md`, **newest-first**, `## YYYY-MM-DD` (§1, template) | **Yes — ordering inverted** |
| ROADMAP | `doc-assistant-roadmap.md`, PR table is `\| # \| PR title \| Files \| Effort \| Depends on \|` | `docs/ROADMAP.md`, table must be `\| PR \| Scope \| Status \| Spec \|` (§11) | Yes — `roadmap_sync.py` won't parse current columns |
| Sprint machinery | none | `docs/sprints/` + `scripts/sprint_check.py` (§10) | Missing |
| `docs/archive/` | none | required home for superseded disposables (§2) | Missing |
| `scripts/conventions.toml` | none | budgets + gate config (§7) | Missing |
| Gate wiring | ruff/mypy/bandit/detect-secrets in `.pre-commit-config.yaml` | adds `docs_check`, `sprint_check`, `integrity-check`, `push-guard`, `coupling-check` | Missing cpc gates |
| `cpc-init` availability | not installed; `dist/` empty; `[project.scripts]` defines `cpc-init` + 11 sibling entry-points | `uv pip install -e <local cpc checkout>`, then `cpc-init` / `python -m cpc.init` | Install step required |
| Disposable docs | `chunking-sweep-rtx-resume.md` (and similar) sit in `docs/` root | `disposable` class, archive when superseded (§2) | Classify on header backfill |

## Options

1. **Hand-migrate everything manually.** — *Cons:* re-does what `cpc-init` already automates
   (laying CONTEXT/KNOWN_ISSUES/conventions.toml/sprint templates/gate wiring); error-prone; no
   idempotency. Rejected.
2. **Run `cpc-init --profile standard`, then do only the judgment work by hand.** — *Pros:* the
   script lays all missing scaffolding without clobbering existing files (idempotent, partial-safe,
   §8); the human-only work narrows to the `decisions.md` split, header backfill, DEVLOG re-order,
   and ROADMAP reshape. **Chosen.**
3. **Adopt `--profile prototype` instead.** — *Cons:* skips the `docs/` tree, sprint machinery, and
   most gates — but doc_assistant is a mature multi-feature project that already *has* a rich
   `docs/` tree. Prototype is the wrong dial (ADR-006 in cpc). Rejected.

## Decision

**Adopt cpc via `cpc-init --profile standard` (option 2), then complete the four judgment items by
hand.** The execution plan below is the contract. Do it as **discrete commits / PRs**, not one big
bang — the `decisions.md` split alone warrants its own review.

### Profile: `standard`

doc_assistant is past prototype. Use `standard`. It lays the full `.claude/` + `docs/` tree, sprint
machinery, `scripts/conventions.toml`, lint config, and wires gates into `.pre-commit-config.yaml`.

### Execution plan (ordered — Claude Code follows this)

Run from the repo root `C:\Projects\doc_assistant`. **Never `git commit`/`push` without Lucas's
explicit review** (cpc §13; the project's own CLAUDE.md). Stage, summarize the diff, stop.

**Step 0 — Branch + make `cpc-init` runnable.**
```
git checkout -b chore/adopt-cpc
```
Install cpc so its console entry-points resolve (verified names from cpc `pyproject.toml`
`[project.scripts]`). From `C:\Projects\doc_assistant`, into the project's own venv:
```
uv pip install -e <local cpc checkout>
```
Verify:
```
cpc-init --help
cpc-init-check --help
```
**Fallback if the entry-points aren't on PATH:** call the modules directly (an editable install
puts `cpc` on `sys.path`) — `python -m cpc.init --help`, `python -m cpc.init_check`,
`python -m cpc.docs_check`, `python -m cpc.backfill_headers`. Use the `python -m cpc.<module>` form
throughout if so. (Do **not** rely on the cpc checkout's `scripts/*.py` from
this repo — those are thin shims that resolve cpc's source relative to the cpc checkout; the
installed package is the supported path.) Record which form worked in the DEVLOG entry for this step.

**Step 1 — Lay the layout (idempotent, will not clobber).**
```
cpc-init --root . --profile standard
```
Expect it to **add**: `.claude/CONTEXT.md`, `.claude/KNOWN_ISSUES.md`, `scripts/conventions.toml`,
`docs/sprints/` (+ template), `docs/archive/`, `docs/decisions/` (kept — this ADR already lives
there), and updated `.pre-commit-config.yaml` / `.gitattributes` / `.gitignore` gate wiring. Read
its printed plan first; confirm it reports *add*, never *overwrite*, for any file that already has
content. If it proposes to touch an existing populated file, stop and report before letting it write.

**Step 2 — Fill the laid templates (judgment).**
- `.claude/CONTEXT.md` — fill Stack / Goal / locked-settings / constraints from the existing
  `CLAUDE.md` "Coordination triad" + "Engineering standards" sections and from `docs/architecture.md`.
  This is the new canonical home for the locked-settings table (§5 single-source). The root
  `CLAUDE.md` must then *point* at it, not restate it.
- `.claude/KNOWN_ISSUES.md` — migrate the live issues the current `CLAUDE.md` and `README` already
  name: `print()`-vs-`structlog` violation, the 3.14/Chainlit runtime quirk, the (now-resolved)
  win32 cu130 segfault, sandbox-sync. Mark resolved ones as resolved; do not invent issues.

**Step 3 — Backfill status headers on all pre-existing docs.** cpc ships a helper (module-only — it
has no console entry-point):
```
python -m cpc.backfill_headers --root .
```
Then hand-verify each file's **class** is correct (the script guesses; you confirm). The gate
(`cpc.docs_check`) enforces line-1 `status:` ∈ {active,superseded,archived}, `class:` ∈
{append-only,living,disposable}, and `updated: YYYY-MM-DD` — so every header must carry all three:
- append-only: `docs/DEVLOG.md`, every `docs/decisions/ADR-*.md`, `.claude/SESSION.md`
- living: `docs/architecture.md`, `docs/ROADMAP.md`, `.claude/CONTEXT.md`, `.claude/KNOWN_ISSUES.md`,
  `docs/figures-and-tables.md`, `docs/how-answers-work.md`, `docs/DEMO.md`
- disposable: `docs/chunking-sweep-rtx-resume.md` (dated, one-off — archive when stale)
- `docs/specs/**` is header-exempt by default (`conventions.toml [headers] exempt`); leave as-is
  unless you opt them in.
- `docs/library.bib` is not markdown — out of scope for the header rule.

**Step 4 — Split `decisions.md` into ADRs (the main judgment task — its own PR).**
The file holds ~50 `###` decisions plus embedded Roadmap/Phase/Open-Questions/Deferred sections.
Do NOT mechanically one-file-per-`###`. Instead:
- Each genuine, self-contained *decision* → one `docs/decisions/ADR-NNN-slug.md` in cpc shape
  (Status/Date/Deciders/Context/Options/Decision/Consequences). Number sequentially **after this
  ADR** (this is ADR-001; the split starts at ADR-002). Numbers are never reused (§6).
- Use the existing decision's own dates where stated (e.g. "added 2026-06-15") for the ADR `Date`.
- Preserve content faithfully — this is re-housing, not rewriting. Where a section is a *status
  log* rather than a decision (the `## Roadmap`, `### ✅ Phase N` blocks, `## Open Questions`,
  `## Deferred Improvements`), it does **not** become an ADR: fold roadmap/phase status into
  `docs/ROADMAP.md` (living) and open questions / deferred items into `.claude/KNOWN_ISSUES.md` or
  ROADMAP backlog as appropriate.
- After extraction, replace `decisions.md` with a short **stub** that says "split into
  `docs/decisions/` on 2026-06-20; see the index" and mark it `superseded`, then `git mv` it to
  `docs/archive/` (§2 forces the move). Keep a one-line ADR index somewhere durable (top of
  `docs/decisions/` README or the ROADMAP) so the route check still resolves.
- **This is large and reviewable. Land it as a dedicated PR after the scaffolding PR.** Expect to
  confirm decision boundaries with Lucas where a `###` block bundles two calls.

**Step 5 — Re-order and re-head the DEVLOG.**
Convert to **newest-first** and to the `## YYYY-MM-DD — title` heading shape from the cpc template
(current shape is oldest-first `## Session: DATE`). Because DEVLOG is append-only, do this as a
single explicit, logged reformat commit — note in the new top entry that the file was inverted to
satisfy cpc §1, and that historical entries are unchanged in content, only re-ordered. Keep the
"Format:" legend. Verify no entry text is altered (diff should be pure reordering + header rewrites).

**Step 6 — Reshape the ROADMAP table.**
Rename/move to `docs/ROADMAP.md` (living) and convert the implementation table to the cpc columns
`| PR | Scope | Status | Spec |` so `roadmap_sync.py` parses it (§11). Map current columns:
`PR title`→`Scope` (concise), add a `Status` cell per row (done / in-progress / planned), `Spec`→the
`docs/specs/...` link where one exists. Keep the phase-overview table as prose/secondary; only the
PR table is machine-read. The existing `docs/doc-assistant-roadmap.md` content is the source.

**Step 7 — Repoint Cowork project settings (§5).** Replace the Cowork project-settings body with a
pointer: *"Authoritative instructions live in the repo: read `CLAUDE.md`, then `.claude/CONTEXT.md`.
Do not restate rules here."* (This is a Cowork settings edit, not a repo file — note it in the
DEVLOG; Lucas performs it if it's outside the repo.)

**Step 8 — Trim the root `CLAUDE.md` to a pointer.** It currently restates locked settings and
standards that now live in `.claude/CONTEXT.md`. Reduce to routing + a short digest that points at
CONTEXT (§5; budget: root `CLAUDE.md` + `CONTEXT.md` ≤ 600 combined lines, §7). The `## Skills in
play` table already matches the cpc pattern — keep it.

**Step 9 — Verify. Do not declare green early.**
```
cpc-docs-check --root . --strict          # or: python -m cpc.docs_check --root . --strict
cpc-init-check --root . --profile standard
```
Fix every error: broken routes, over-budget entry context,
missing/incorrect headers, an `active` disposable past staleness, an archived doc still outside
`docs/archive/`. Then run the existing project test suite to confirm nothing in `src/` broke
(nothing should — this is docs-only). **Stage, summarize the diff, and stop for review.**

### Commit / PR breakdown (suggested)

1. **PR-A — scaffolding:** Steps 0–3 + 5 + 6 + 8 (cpc-init, fill CONTEXT/KNOWN_ISSUES, headers,
   DEVLOG re-order, ROADMAP reshape, CLAUDE.md trim). Docs-only, mechanical-ish, one review.
2. **PR-B — decisions split:** Step 4 alone. Judgment-heavy; reviewed on its own.
3. Step 7 (Cowork settings) is a settings action, tracked but not a repo PR.

## Consequences

- **Easy after:** deterministic doc gates run every commit; `roadmap_sync`/`sprint_check` tooling
  works; context loads cheaply; one source of truth per rule (no Cowork-vs-repo drift).
- **Commits us to:** the cpc lifecycle rules — ADRs append-only and immutable once accepted (§2),
  DEVLOG newest-first, disposables archived on supersede, status headers on every doc, the sprint
  contract workflow for future increments (§10–11).
- **Costs / risks:** the `decisions.md` split is the real effort and must preserve content exactly;
  the DEVLOG inversion rewrites a large append-only file once (acceptable as a logged one-time
  reformat); `cpc-init` must be installed from source first; ROADMAP table reshape is required for
  tooling, not cosmetic. None touch `src/` — runtime behaviour is unchanged.
- **Reverses if:** never expected — this aligns with a standard Lucas owns. If cpc tooling proves
  too heavy for a solo project, drop the sprint gates while keeping the layout (a future ADR).

## Links
- cpc standard: the private cpc CONVENTIONS repo (§1–§13).
- cpc ADR-006 (init profiles) — why `standard` not `prototype`.
- cpc ADR-009 (archive lifecycle) — the supersede-then-move rule used in Steps 4–5.
- Superseded source: `docs/decisions.md` → `docs/archive/` after Step 4.
- `docs/specs/` — unchanged; remain the code-level contracts referenced by ADRs/ROADMAP.

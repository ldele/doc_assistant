<!-- status: active · updated: 2026-09-10 (refresh rule; preflight enforces it) · class: runbook -->

# Release UX/UI walkthrough — what to drive before every release

The gate that `docs/RELEASE.md` §5b points at. `preflight` and the test suites prove the code;
this proves the **product**: every surface a user can reach is driven once, in the **frozen
build**, by a person, with the outcome written down. It exists because the same three things
kept happening — a feature "construction-verified" in July was still unproven in September
(`docs/ui-checklist.md` §3), the 2026-08-28 walkthrough found three defects in the first feature
it drove (KI-51), and the app showed a *zero-state* instead of a connection error when its
backend was dead (REVIEWS 2026-08-28), which no test can see.

**Rules of the walkthrough**

1. **Drive the installed artifact, not the dev loop.** Vite HMR and the dev sidecar hide packaging
   defects (KI-34 shipped a build that could not read a PDF while every gate was green).
2. **$0.** Ollama for generation; force it — `.env` defaults are all-Anthropic (KI-4). Name the model
   in the record: citation coverage is model-dependent (KI-36).
3. **Drive the failure branch too.** A surface is not verified until its empty state, its error
   state and its "unavailable" state have each been seen once. Three states, never two (ADR-044).
4. **A blocker is a data-loss or a lie.** Anything that deletes, hides or mislabels what the user
   has → fix before the tag. Anything else → a KI, listed under "Known limits" in the CHANGELOG (§2
   of `RELEASE.md`), and the release proceeds.
5. **Record it.** One row per surface in the table at the foot of this file *for the release being
   cut* (copy the table into `.claude/release-notes-X.Y.Z.md`), then a log line in
   `.claude/REVIEWS.md` row 6 stating what was **not** driven. An unstated omission reads as
   coverage six months later.

**This file is stale until refreshed** (`docs/RELEASE.md` §0). Before driving it: list what shipped
since the previous tag, add or amend a row per surface — including its empty and failed states —
and bump the header. `release_preflight` fails the release if this file has no commit since the
last tag, on purpose: an untouched checklist is evidence that nobody looked, not that nothing changed.

Two libraries are needed: an **empty data home** (first run) and the **real library** (~100
documents, folders, keywords, a taxonomy with placements, at least one referenced root and one
scanned PDF). Most rows say which.

---

## 0 · First run (empty data home)

- [ ] Launch → the readiness banner names every outstanding step (corpus **and** provider), not just one.
- [ ] Provider setup: no key → `not configured`; wrong key → the verification fails with a sentence, the key is not echoed back anywhere (`key_hint` shows 4 characters at most); Ollama down → `unreachable`, with the action named.
- [ ] Set the data home; the empty-corpus banner offers *Add documents* and the sample-question chips are clickable.
- [ ] Ask a question with 0 documents: an honest "no documents" answer, no traceback, no spinner that never ends.
- [ ] Settings → About shows the version that `preflight` checked; Updates → *Check now* returns one of exactly three states (`newer` / `current` / `unknown`), and `unknown` is what you get offline.

## 1 · Add documents (both libraries)

- [ ] Picker: one file · several files · a folder. Drag-and-drop the same three (never driven on 2026-08-28 — File Explorer needs full-tier access).
- [ ] Review sheet: a duplicate reads as a duplicate; an unsupported format says why; a scanned PDF is warned about **before** indexing, not discovered as a 0-chunk document.
- [ ] Both placement modes: **copy** lands in the library folder; **reference** registers the root and the file stays where it was.
- [ ] Batch header with > 50 files: the `shown` cap and "and N more" render; make one file fail (rename it mid-run) → `stopped_early`, "Keep the N", `applyError` all render.
- [ ] Index now → the ingest chip shows progress; the document is answerable **before** the chip says done (controller reload).
- [ ] **Undo all** after index-now removes the document row, its chunks *and* the referenced root (KI-51, all three parts).

## 2 · Library (real library)

- [ ] Grid ⇄ list; the filter strip; the keyword-filter overlay; search scoped to a collection with "Search all".
- [ ] Folders: create · rename · assign · delete; a folder-scoped chat only cites that folder (ADR-025 F2) and the provenance card says so.
- [ ] Open a document: the five blocks with the jump index; chunks stay unfetched until opened; the block you left open is remembered.
- [ ] Metadata editor: an edit survives a per-part **metadata** re-ingest (ADR-013 wins over ADR-049 wins over the extractor).
- [ ] Figures panel + full-size viewer with zoom; references block; Connections panel (`1st/2nd/3rd`, no raw score); taxonomy placement modal (attach ticks the rollup; a cycle is refused with a sentence).
- [ ] Re-ingest dialog: each part states its cost before it runs; `text` re-chunks without a duplicate document; a batch from *Select* mode continues past one failing document.
- [ ] Delete: asks; defaults to library-only; a **referenced** file is never sent to the bin and the dialog names its real path (ADR-046).
- [ ] Source viewer: opens at the cited page; *Fit page* is the default; zoom and the split are draggable; a non-PDF degrades to extracted text and says why; an unreachable file is one sentence naming the path.
- [ ] Manage keywords: the graph-include toggle changes the "On the graph (N)" count immediately.
- [ ] A document with **no keywords**, a **0-chunk** document and a document whose root is **unavailable** each render honestly (badge, not a blank).

## 3 · Chat (real library, Ollama)

- [ ] A streamed answer with inline `[n]`; click one → the citation panel; **In context** shows the passage highlighted with what surrounds it and "N% of the way in"; **Show the page** opens the source viewer on that page (a figure citation too).
- [ ] Provenance card: which folder scope, which provider/model, which prompt hash; a **local model** is labelled as such on the answer.
- [ ] Source-evaluation strip shows year · relevance · graph freshness — and **not** coverage/`contested` chips (withheld since v0.4.1, KI-33) unless `EPISTEMICS_MARKERS_ENABLED=true` was set on purpose.
- [ ] Claim review: an `unsupported` verdict on a correct refusal is not the accusing one (KI-37); low-confidence card appears on a thin retrieval.
- [ ] **Compare** (A/B): two source sets side by side, $0, no LLM call in the log.
- [ ] Session override: change a sandbox knob → the `Session override` note renders **and** the retrieved set changes; out-of-range → 422 surfaced as a sentence, never a silent clamp; the override does not persist across a new chat.
- [ ] Human synthesis mode: evidence only, no interpretation call in the log.
- [ ] Ask something the corpus cannot answer → a refusal that cites nothing, not a fabricated citation.
- [ ] New chat resets turns, panel and composer; history sidebar reopens a chat read-only; rename · pin · export-all · bulk delete each work and each asks before deleting.

## 4 · Graph & gaps (real library)

- [ ] *Never built* and *built but empty* are two different screens (row 22).
- [ ] Concept rail: the three lenses are one control style with tooltips that say what they **do**; the ego graph opens on a node; *Place* deep-links into the taxonomy modal with the concept preselected.
- [ ] Gap list: promote · dismiss · reset; **rebuild the graph in-app** → the triage survives (`GapTriage` sidecar); the filter box matches label and kind; the coverage sentence ("N of M documents") is present.
- [ ] `under_connected` is **off by default** (RG-014 grade); `single_source` leads.

## 5 · Settings

- [ ] Five categories in the rail; nothing lost from the flat list (compare against the previous release's screenshot).
- [ ] Live provider/model switch without restart; the **reviewer follows** an unpinned switch; an explicit `REVIEWER_PROVIDER` still wins.
- [ ] Corpus facts: documents · chunks · disk · which keyword arm · its size; *Rebuild keyword index* runs, reports, and the warning banner clears; simulate a failed build → `keyword_index_unavailable`, chat degrades to vector-only and **says so**.
- [ ] Theme: System / Light / Dark persists across relaunch; both themes on every surface above (screenshot each once).
- [ ] Source directory change → the sources panel re-derives `new/changed/ingested/missing`.

## 6 · Shell & platform

- [ ] Sidebar collapse/expand (an uncaptured defect since 2026-07-21 — capture the exact symptom if it misbehaves); global search (same); shortcuts dialog; About with the right version.
- [ ] Window at 640×480 minimum: no horizontal overflow; wide content scrolls in its own container; reduced-motion honoured.
- [ ] 0 console errors across the walkthrough (open devtools once in the dev build for the same route set — the release build has none).
- [ ] Kill the sidecar while the app is open → a **connection** error, not the zero-state (REVIEWS 2026-08-28: it showed "No documents indexed yet").
- [ ] Occupy port 8001 before launch → the app reports the conflict; cold start to first `/api/health 200` is within RG-010's recorded envelope; closing the window ends the sidecar process.
- [ ] Offline (network cable out): launch, ingest, answer with Ollama — all work; the update check says `unknown`.

## 7 · Packaging gates (pointers, not duplicates)

- RG-012 Tier-1 (clean machine, offline) and Tier-2 (a cited turn) — `docs/desktop-packaging.md` §5; the PASS must be bound to **this** installer's build timestamp (`preflight rg012`).
- The installer is the one in `bundle/nsis` *now*, and the previous one is deleted **after** the push (`RELEASE.md` §8).

---

## Record for release vX.Y.Z

Copy into `.claude/release-notes-X.Y.Z.md` (local) and fill. `Not driven` is a legal value — it is
the one that keeps the next release honest.

| § | Surface | Library | Build (installer timestamp) | Driven by / date | Result (ok · KI-nn · blocker) | Not driven |
|---|---------|---------|-----------------------------|------------------|-------------------------------|------------|
| 0 | First run | empty | | | | |
| 1 | Add documents | both | | | | drag-and-drop? multi-file? failure branch? |
| 2 | Library | real | | | | |
| 3 | Chat | real | | | | which model |
| 4 | Graph & gaps | real | | | | |
| 5 | Settings | real | | | | |
| 6 | Shell & platform | both | | | | |
| 7 | Packaging gates | — | | | | tier 2? |

---

## Lifting this into cpc (proposal, 2026-09-10)

The generic shape is small and is what would survive the generalisation:

1. **A per-surface table, not a per-feature one.** Features are how the roadmap is organised;
   surfaces are how a user meets the product. The table's rows are the product's screens, each with
   *steps → expect*, and the record column has three legal values: ok · issue-id · **not driven**.
2. **Three states per surface.** Every row is driven in its populated, empty and failed state; a
   checklist that only lists the happy path reproduces the blind spot it exists to remove.
3. **The artifact under test is named.** A build identifier column ties the record to the thing
   that shipped (the `preflight rg012` rule generalised); a PASS from a previous build is worse
   than none.
4. **The omission column is mandatory.** Same principle as `.claude/REVIEWS.md`: what was *not*
   covered is the load-bearing half of the entry.

Registration: a `release-close` keypoint (or `[keypoints.sprint-close]` on projects without a
release track) whose deterministic floor runs the artifact/version checks and whose judgment
checklist points at this file; the record table is the stamp. Per-project rows live in the
project's copy of this file; cpc ships the template (§ headings + the record table + the four
rules) — the way `REVIEWS.md` is proposed as a template in `.claude/REVIEWS.md` § Candidate.

<!-- status: active · updated: 2026-08-30 · class: append-only -->

# ADR-048 — Re-ingest is per part, and only the parts one document can honestly own

- **Status:** accepted
- **Date:** 2026-08-30
- **Deciders:** user (asked for ROADMAP rows 20/21), Claude Code (scope derived from the measured record)

> **Scope.** This settles **which parts of ingestion can be re-run for a single document**, what the
> control must tell the user before it spends anything, and what the expensive part has to clean up
> after itself. It does not change ingestion itself, the Enrichment-Layer Pattern, or
> **ADR-047** (identity survives re-extraction) — it *depends* on ADR-047, which is what makes the
> expensive part safe at all. It supersedes nothing.

## Context

Settings offers one button: **index the folder**. That is the whole-corpus operation, and it is the
wrong instrument for the common case — *this one document came out wrong*. Today's answer is a CLI
run (`python -m scripts.extract_citations --doc <id> --apply --force`), which a packaged user does
not have. ROADMAP rows 20/21 ask for the per-document and per-selection form.

The roadmap filed one open question — *which parts can re-run without moving `doc_hash`* — and
investigating it surfaced three more. All four are answered from the measured record
(`docs/performance.md`) and from the code, not from estimates.

**1 · The parts differ in cost by four orders of magnitude.** From the per-stage share table, for
the reference unit of one 30-page digital PDF:

| Part | Share of a first ingest | Measured | Scales with |
|---|---:|---|---|
| **Text** (extract → chunk → embed) | **~95%** | 14.7 s mean, 5.7–36.7 s; **~35 s if OCR fires** | pages |
| References (citation extraction) | ~1% | **7.4 s** scoped to one document (since KI-30) | references |
| Figures (detection + crops) | ~1% | sub-second | pages |
| **Metadata** | **~0%** | milliseconds — 0.58 s for the *whole corpus* | nothing much |

A control that presents these as four equivalent checkboxes would be lying by omission. Metadata is
free; text is a coffee break on a scanned book.

**2 · Some passes cannot be scoped to a document at all, and saying otherwise would be a lie.**
`extract_keywords` and `compute_doc_vectors` recompute over the whole library by construction —
corpus TF-IDF has no per-document form. Measured: scoping `extract_keywords` to one document
**saves 4%**. `compute_epistemics` and `build_gaps` have no scope flag and full-recompute. So
"Connections", "Keywords", "Epistemics" and "Gaps" are *not* per-document parts, however much the
document panel's blocks suggest they might be.

**3 · The selective ingest path skips cleanup — deliberately — and the expensive part has to
notice.** `ingest.main` runs `cleanup_orphans_*` only when
`not skip_cleanup and not force_rebuild and scope is None and files is None`. A per-document
re-extract is exactly the `files is not None` branch, so **nothing sweeps the superseded chunks**.
Re-extraction changes the extracted text, which changes `doc_hash` (ADR-042), and ADR-047's
identity fallback keeps the *row* and its sidecars attached — but the previous hash's chunks stay
in both Chroma stores and remain retrievable. Left unhandled, "re-extract this document" would
silently double it in the index.

**4 · Metadata has a safe overwrite and it is not obvious.** ADR-013 keeps user metadata edits in a
separate `DocumentMeta` override table; `Document.title/authors/year/doi` hold only the
*auto-extracted defaults*, and reads merge with the override winning. So a re-run may overwrite the
extractor's previous answer without touching anything a human typed — the opposite of the usual
`--force` hazard.

## Options

1. **A per-part control offering only the parts a single document owns, each stating its cost.
   (CHOSEN)** *Pros:* matches what the layers can actually do; the cost statement makes an
   order-of-magnitude difference visible at the moment of choosing; the enrichment layers are
   already idempotent sidecars, so re-running one is defined behaviour. *Cons:* the panel shows
   blocks (Connections) that have no re-run button, which needs explaining rather than hiding.

2. **One "re-ingest this document" button that re-runs everything.**
   *Pros:* one control, nothing to explain. *Cons:* charges ~35 s of extraction to fix a title the
   extractor read wrong — the cheapest fix in the system behind the most expensive operation. It
   also moves `doc_hash` every time, so a metadata correction would churn the index. Rejected: the
   whole value of per-part is not paying for the part you did not need.

3. **Offer every block in the panel, and let the corpus-global ones quietly run corpus-wide.**
   *Pros:* the control matches the UI's own five blocks, which is what a user would predict.
   *Cons:* pressing "re-run Connections" on one document would spend a whole-corpus pass without
   saying so, and would return results that changed for *other* documents. That is the honesty
   contract failing in the direction that erodes trust fastest. Rejected.

4. **Expose the CLI runners over the API and shell out.** *Pros:* zero new logic. *Cons:*
   `apps/` are thin shells over `src/doc_assistant/`, never over `scripts/` (non-negotiable #3);
   the runners' per-document orchestration is written for a console report, not a caller. Rejected
   — the orchestration moves into `src/` instead, which is where it should have been.

## Decision

**Four parts, and only four.** `metadata` · `figures` · `references` · `text`. They are the passes
that have a genuine per-document form, and they are declared in one registry in
`src/doc_assistant/reingest.py` so the API, the UI and any future runner read the same list.

**Every part states its cost before it runs**, in the honest unit (an order of magnitude, not a
prediction): metadata *instant*, figures *a few seconds*, references *about 10 seconds*, text
*30 seconds to a few minutes, longer for a scanned document*. The estimates come from
`docs/performance.md` and are labelled as typical, never as a promise.

**The corpus-global passes are named and declined, not hidden.** The control says that
connections, keywords, epistemics and gaps are computed across the whole library and cannot be
re-run for one document — because a user who cannot find the button deserves to know there is no
button, rather than concluding the feature is broken.

**`text` is the only part that moves identity, and it owns its own cleanup.** It records the
document's `doc_hash` before it starts, invalidates that document's extraction cache, re-ingests
the single file, and — if the hash moved — purges the superseded hash's chunks from **both** vector
stores. This is not a nice-to-have: without it the selective path's skipped cleanup leaves the
document indexed twice.

**A metadata re-run overwrites the extracted defaults and never a user's edit** (ADR-013's override
table is what makes that safe). The other parts replace their own sidecar rows for that document
and nothing else.

**The VLM figure-description pass is not a part.** It costs money per figure, and a control that
spends money must be its own deliberate action, not an item in a checklist (KI-4's credit-leak
lesson). `figures` re-runs *detection* only.

**Row 21 is the same operation over a list.** The core takes `document_ids: Sequence[str]`, so the
grid's existing multi-select needs no second code path — only the same picker and the same cost
statement, multiplied by the selection size.

## Consequences

- **`src/doc_assistant/reingest.py` becomes the home of per-document re-running**, and the three CLI
  runners (`extract_doc_metadata`, `extract_citations`, `extract_figures`) still carry their own
  copies of that orchestration. That duplication is real and is not resolved here: rewiring three
  working runners is a change with its own blast radius, and it belongs in its own increment. Until
  then, a fix to per-document metadata logic has two homes — recorded so the next person does not
  discover it.
- **The cost statement will drift** as extraction improves or the machine changes. It is prose in
  the registry, sourced from `docs/performance.md`, and it is worth re-reading whenever that record
  is re-measured. It is deliberately coarse for that reason.
- **`text` is the only part that can make a document worse**, because it re-runs the pass that KI-48
  and KI-47 both live in (a cache invalidated for every format; OCR firing on 87 of 97 documents).
  It stays available because the alternative — a user who can see "broken" on a row and do nothing
  about it — is worse, and because the health classifier that produces that verdict was only just
  made honest (KI-53).
- **The re-run says what it did, per document and per part.** A part that could not run (no cached
  markdown, a missing source file) reports *skipped* with the reason, never a silent success.
- **Not covered: re-running a part across the whole corpus from the app.** That is Settings' index
  button plus the CLI runners today, and this ADR does not change it.

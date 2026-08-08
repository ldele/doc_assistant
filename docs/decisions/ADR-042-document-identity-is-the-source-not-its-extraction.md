<!-- status: active · updated: 2026-08-08 · class: append-only -->

# ADR-042 — A document's identity is its source file, not its extracted text

- **Status:** proposed — the defect is measured and the direction is settled; the migration is not built
- **Date:** 2026-08-08
- **Deciders:** user + Claude Code

## Context

`ingest` resolves a document's identity from the hash of its **extracted markdown**:

```python
document_id = _existing_document_id(h) or str(uuid4())   # src/doc_assistant/ingest/__init__.py
```

where `h = doc_hash(text)` — "Content-only hash. Path-independent so documents survive moves/renames"
(`src/doc_assistant/ingest/cache.py:163`). Same extracted bytes ⇒ same id ⇒ the row is reused; different
extracted bytes ⇒ **a fresh UUID**, and every sidecar keyed on `document_id` is orphaned and swept.

`_existing_document_id` states the intent this breaks — it exists "so the document's figures and other
id-keyed sidecars stay linked" (`src/doc_assistant/ingest/store.py:41`). That holds only while extraction
output never changes, and extraction output changes routinely: an extractor fix, a PyMuPDF upgrade, or
**this project's own table splice**, whose runner already concedes "a content change drops that doc's
sidecar enrichment" (`scripts/extract_tables_marker.py`). One guarantee is asserted in one file and
denied in another.

**Measured on this corpus, 2026-08-08 (KI-43).** Across the chunking sweep's rebuilds, **11 of 97
documents changed id — and all 10 that owned figures were among them.** Not coincidence: the 2026-08-07
text-layer fallback fires on full-page-image pages, i.e. precisely the image-heavy documents that have
figures. Result: `figures` **45 → 0**, `chunk_epistemics` **743 → 445**, `concept_presence` **66 → 31**.
Nothing warned; the rebuild's own log says the registry "(folders, tags, metadata, figures) is
preserved". A re-derive then produced **962** figures, so the table had been running at a few percent of
its true size for an unknown period.

**The root cause is that one value serves two purposes with opposite requirements.** `doc_hash` answers
*"is my extraction of this document current?"* — it **must** change when extraction changes. It is also
used to answer *"which document is this?"* — which **must not**. No single hash can do both.

A second observation sharpens the fix: **the figures sidecar never depended on the extracted text at
all.** `extract_figures` opens the source PDF and reads its geometry; markdown is not an input. Keying it
to the extraction output was a category error, not merely a fragile choice. That is not true of every
sidecar — `chunk_epistemics` and `concept_presence` are keyed to chunk indices and *should* be
invalidated when chunking changes. The two classes need different handling, and today both get the same
one: silent deletion.

## Options

1. **Identity = hash of the source file's bytes** (a new `source_hash`; `doc_hash` keeps its cache /
   content-version job). Survives every extraction change, and still survives moves and renames, since
   bytes are path-independent — it strictly dominates the current scheme on the property `doc_hash`'s
   docstring claims. It also changes correctly when the user genuinely replaces a file. Trade-off: an
   additive schema column plus a backfill, two hashes to keep straight, and re-downloading the same paper
   as different bytes (a new arXiv version, a re-compressed PDF) reads as a new document.
2. **Identity = source path** (`source_files.rel_path`, already stored — 97 rows). Cheapest: no new
   column, no backfill. Trade-off: it **regresses** the move/rename property the current design
   deliberately bought, and it is actively unsafe in the other direction — replacing a file at the same
   path inherits the previous document's sidecars, so old figures attach to a new document. It converts a
   loud loss into a quiet wrongness.
3. **Keep content identity; migrate sidecar rows when the hash changes** but the source matches.
   Preserves the current key. Trade-off: correctness depends on an enumerated list of id-keyed tables,
   so the next sidecar anyone adds is silently omitted — the same class of defect (a list that must be
   manually kept in sync) that produced this one.
4. **Detect and warn only** — count id-keyed sidecar rows before and after a rebuild, and fail loudly on
   a negative delta. No schema change and it lands in an afternoon. Trade-off: it does not fix anything;
   the data is still destroyed, and recovery is a manual re-run the user must remember.

## Decision

**Adopt option 1: identity is the source document's bytes; `doc_hash` stays the extraction-version key.**

The deciding reason is that it is the only option that makes the two questions separately answerable
instead of forcing one value to answer both. Options 2–4 all keep the conflation and manage its
symptoms — option 2 by trading a visible failure for an invisible one, option 3 by adding a
hand-maintained list, option 4 by making the loss loud rather than preventing it.

**Option 4 is adopted as an interim, not an alternative.** The delta check is cheap, independent, and
still valuable after the migration, because it verifies the invariant this ADR asserts rather than
assuming it.

**Consequence for sidecar semantics, which this decision forces to be explicit.** Sidecars split by what
they derive from:
- **Source-derived** (`figures` and their PNGs; ADR-039's OCR artifact at `<data>/ocr/<doc_hash>.pdf`) —
  a function of the PDF. These must survive an extraction change, and their storage should be keyed on
  source identity too, not on `doc_hash`.
- **Extraction-derived** (`chunk_epistemics`, `concept_presence` — keyed on chunk index/key) — genuinely
  invalid when chunking changes. These should be **explicitly invalidated and reported as stale**, so a
  runner re-derives them. Deleting them is not the error; deleting them *silently* is.

**What would reverse it:** source bytes proving unstable in practice — a sync client, antivirus, or
PDF-repair tool that rewrites files in place would make identity churn for a new reason, and the fix
would have to move to a content-independent identifier the user controls (an explicit document key)
rather than any hash.

## Consequences

**Easier.** Improving extraction stops destroying derived data — the single behaviour that made the
figure pipeline appear broken. Running the table splice, upgrading PyMuPDF, or fixing an extractor
becomes safe for sidecars. The distinction between "my extraction is stale" and "this is a different
document" becomes expressible, which it currently is not.

**Harder.** An additive migration plus a backfill for 97 existing documents, run before the next
extraction change or it inherits the same loss. Two hashes exist, so every new call site must choose
deliberately — and the names must carry that (`doc_hash` is a poor name for "extraction version"). The
source file must be readable at ingest to hash it; a document whose source has moved away needs a
defined fallback.

**Must revisit.** Whether source-derived artifact *paths* (`figure_dir`, the OCR sidecar) move from
`doc_hash` to the source key — they should, by the same argument, but that is a data migration on disk
and is out of scope here. Also: two files with identical bytes become one identity, which is correct for
de-duplication and surprising if a user deliberately keeps two copies.

## Confidence

- ✓ **The defect is measured, not inferred** — 11/97 documents changed id with all 10 figure-owners among
  them; `figures` 45 → 0 → 962 on re-derive; `chunk_epistemics` 743 → 445; `concept_presence` 66 → 31.
  Recorded in `.claude/KNOWN_ISSUES.md` KI-43 and DEVLOG 2026-08-08 (5).
- ✓ **The mechanism is read from the code**, not reconstructed: `ingest/__init__.py` (the `or str(uuid4())`
  line), `ingest/cache.py:163` (`doc_hash` is content-only), `ingest/store.py:41` (the stated intent).
- ⚠ **The migration is unbuilt and its backfill is unvalidated.** Nothing yet demonstrates that assigning
  `source_hash` to 97 existing documents preserves every current sidecar link, nor that the source file
  is always available to hash. Tracked as **RG-027** in `.claude/RIGOR_TODO.md`.
- ⚠ **"Source bytes are stable" is an assumption on this machine**, not a measured property of a user's
  filesystem. It is the assumption the whole decision rests on, and it is the thing the reversal
  condition names.

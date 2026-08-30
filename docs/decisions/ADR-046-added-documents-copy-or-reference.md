<!-- status: active · updated: 2026-08-25 (AD3b built — both placement modes ship) · class: append-only -->

# ADR-046 — A document you add is either copied in or referenced in place, and the app must know which

- **Status:** accepted (built) — **copy-in and reference-in-place both built** (AD1-AD3a 2026-08-24, AD3b 2026-08-25); **the amended-delete half landed 2026-08-28** (`delete_document(*, delete_file=False)`, the two-option dialog, and the path named for every document)
- **Date:** 2026-08-21
- **Deciders:** user (product direction, 2026-08-21 scoped grill), Claude (Claude Code session
  2026-08-21)
- **Relates to:** ADR-014 (safe delete — **amended here**) · ADR-042 (document identity is the
  source, not its extraction — this ADR partially *leads* it) · ADR-029 (local-only working state) ·
  `docs/PLAN_2026-08-20_user-friendly-ingestion.md` §2 + §8 (the grill ledger this records) ·
  `docs/ui-checklist.md` "Missing-source badge + library-only delete" (**absorbed**) ·
  `docs/ROADMAP.md` row 17 (Zotero/Calibre adapters — unblocked by this) · KI-46 · RG-027 · RG-030

## Context

Provenote indexes a folder well and **cannot accept a document**. Measured 2026-08-20 by grep across
`apps/desktop/src`: there is no file picker, no drag-and-drop, and no upload endpoint anywhere. The
only way in is a text box reading *"Paste the full path to the folder holding your documents."*
Everything downstream — the stat-only registry scan, selection resolution, the background ingest
job, status polling, per-file exclude — is built and tested.

Adding an accept surface looks like a UI task. It is not, because of two facts in the code:

1. **The registry is single-root by construction.** `SourceFile` is keyed by `rel_path`, documented
   as *"POSIX separators, relative to the source dir"*, with **no root column**. Accepting a
   document that lives outside the one source dir is a **schema** change.
2. **Delete bins the user's file.** `library/documents.py::delete_document` moves the source to the
   OS Recycle Bin **first**, then removes the row (ADR-014). That is safe only while the source
   folder belongs to Provenote. The moment a document can be registered where it already lives,
   deleting it from the library takes a file out of the user's own Zotero folder.

A third fact constrains how the app recognises a document it already has: **today's duplicate gate
is `doc_hash(text)` on the *extracted* text, and it fires only after `load_or_extract`**. There is
no cheap pre-extraction identity to ask *"do I already have this?"* with. ADR-042 already decided
identity should be the source rather than the extraction; RG-027 records that migration as
**unbuilt and blocks-ship**, and KI-46 is the standing symptom (recovering a document mints a new
identity, because its extracted text changed).

## Options

**Option 1 — copy-in only.** One root; adding copies the file in. No schema change, ADR-014 stays
correct as written, least code. Costs disk twice and ignores the organisation the user already has
(Zotero, Dropbox, a papers folder ten years deep).

**Option 2 — reference in place only.** Multi-root; the library points at files where they sit. No
duplication, respects existing organisation, and is the only substrate a Zotero adapter can sit on.
Requires the root concept in `SourceFile`, multi-root `missing` detection, and forces ADR-014 to
change.

**Option 3 — copy-in now, reference designed but unbuilt.** Ships Option 1 while writing Option 2's
schema and delete consequences down, so the adapter is not later found blocked on an unwritten
decision. *This was the recommendation.*

**Option 4 — both modes in v1.** The user picks per batch. Largest v1: schema, delete semantics and
the accept surface all land together.

## Decision

**Option 4 — both modes ship in v1**, chosen by the user over the Option 3 recommendation. The
deciding reason is recorded because it is the part worth keeping: **it is the only option that does
not make the Zotero adapter a second migration.** Option 3 would build the copy path, then rebuild
`SourceFile`'s key and re-decide delete when the adapter arrives; Option 4 pays that once. It also
turns the wireframe's "Where they go" radio into a real control rather than an illustration.

Three sub-decisions follow, and all three are part of this ADR rather than implementation detail —
each one changes a contract that already exists.

### 1 · `SourceFile` gains a root; `rel_path` stays relative to it

A registered file is `(root, rel_path)`, not `rel_path`. The copy-in root is the existing
`app_settings.get_source_dir()` (whose precedence — `DOC_SOURCE_DIR` → persisted `source_dir` →
`config.DOCS_PATH` — is unchanged and needs no decision). A referenced file's root is the folder
the user chose. `missing` detection becomes per-root. Recursion is unchanged: `scan_sources`
already walks `root.rglob("*")` with no depth limit, and a dropped folder matches that rather than
introducing a configurable depth.

### 2 · ADR-014 is amended: delete asks, and defaults to library-only

`delete_document` must no longer bin the source unconditionally. Deleting a document opens a
confirm with two actions:

- **Remove from library** — default. Drops the row and the chunks; the file is untouched.
- **Also delete the file** — the ADR-014 behaviour, now opt-in per deletion.

**For a referenced document the dialog must name the real path**, so the second action reads as
*"…from C:\Users\Lucas\Zotero\storage\ABC123"* rather than as an abstract "the file". The accepted
risk of a per-delete choice is a mis-click; showing the destination is what makes the click
informed, and it is therefore part of the decision, not copy-writing.

This supersedes ADR-014's "source file → Recycle Bin first" as the *unconditional* rule. The
ordering ADR-014 chose is kept where the user opts in: bin first, and only on success remove the
row, so a locked file leaves the library entry intact rather than orphaning an indexed file.

**This absorbs the queued "library-only delete" checklist item.** It was never a separate feature —
it is this decision.

### 3 · Add-time identity is the sha256 of the source bytes

The review sheet's *"already in your library"* verdict is decided by hashing the **source bytes** at
add time — milliseconds per file, correct across a rename, and correct in both modes.

**This deliberately leads ADR-042.** Source-hash identity is the direction RG-027 has to go anyway;
the add path adopts it first rather than inheriting the extraction-derived hash it is meant to
replace. **The consequence is stated so it is not later mistaken for drift: two identities coexist
until RG-027 lands** — `sha256(source bytes)` answers *"have I already added this?"*, and
`doc_hash(extracted)` continues to answer *"have I already indexed this?"*. They can disagree (two
different PDFs whose extracted text is identical). That window closes when RG-027 collapses them,
and this ADR is one of the reasons to close it.

## Consequences

**What this buys.**
- The app can accept a document, which is the single largest gap in the product today.
- ROADMAP row 17 (Zotero/Calibre) becomes an adapter over an existing substrate rather than a
  migration.
- A user's own library organisation survives contact with Provenote.
- The library-only-delete item is closed as part of this rather than queued separately.

**What it costs, honestly.**
- **This is the largest of the four v1s.** A schema change, an amendment to a shipped safety
  behaviour, and a new accept surface land together. The recommendation was Option 3 precisely to
  avoid that; the user's reason for overruling it is sound and is recorded above.
- **Every downstream branch now handles two modes**: `missing` detection, dedupe, delete, and the
  Sources panel's status derivation.
- **A per-delete choice is a per-delete opportunity to get it wrong.** Mitigated by the default and
  by naming the path; not eliminated.
- **Two identity notions coexist** until RG-027. Bounded, stated, and closing.

**Reopens if:** the schema plus delete work proves larger than the accept surface itself. Copy-in
first (Option 3) remains the fallback and nothing built for it is wasted — the root column defaults
to the source dir, and the delete dialog degrades to the ADR-014 behaviour by preselecting "also
delete the file" for copied documents.

**Blocked on nothing; blocks nothing.** RG-030 (the add-time text-layer probe's unmeasured cost)
gates the review sheet's *no text layer* row, not this decision.

**Not decided here**, and deliberately: whether adding implies indexing (ticked by default), batch
pagination, and partial-copy failure handling. Those are task-level resolutions and live in the
SPEC's ledger — they change no existing contract.

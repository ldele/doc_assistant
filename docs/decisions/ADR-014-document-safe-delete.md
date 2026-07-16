<!-- status: active · updated: 2026-07-16 · class: append-only -->

# ADR-014 — Document safe-delete (source file → Recycle Bin)

- **Status:** accepted
- **Date:** 2026-07-16
- **Deciders:** user + Claude Code

## Context

Following ADR-013 (metadata editing — the first browse-time write path), the user asked to **delete**
documents from the library: remove the record *and* the source file, with a confirmation, and (later)
multi-select for bulk delete / move-to-collection. Delete is the most destructive browse-time action —
it spans the SQLite registry, the Chroma search index, on-disk sidecars (figure dirs, cached `.md`),
**and the user's original file**. Two forces: it must be genuinely reversible enough to be "safe" (the
user's word), and it must be *consistent* — a delete that removes the DB record but leaves the still-on-
disk file would be re-added by the next ingest, and a delete that trashes the file but fails to clean the
index would leave dangling chunks.

## Options

**A. What happens to the source file**
1. **Soft-delete, keep the file** — archive the record + drop its index chunks, leave the file on disk.
   Safest, fully reversible, but doesn't do what the user asked ("delete in source").
2. **Move to the OS Recycle Bin** — remove record + index chunks and recycle the file (recoverable via
   the OS trash). Matches "delete in source + database" while staying recoverable. Needs a small
   cross-platform dependency (`send2trash`).
3. **Permanent delete** — unlink the file irrevocably. Matches the ask literally but unrecoverable.

**B. Ordering / consistency** — trash-then-remove vs remove-then-trash.

**C. Scope** — single-document first vs single + multi-select in one pass.

## Decision

**A2 — Recycle Bin** (user's choice). `send2trash` moves the source file to the OS trash; the record,
its `DocumentMeta` override, its Chroma chunks (from the live store), its figure dir, and its cached
`.md` are removed. **B: recycle the file FIRST** — if the trash fails (locked/undeletable) the whole
delete aborts and the library entry stays, so we never orphan a still-indexed file on disk; a file that
is *already gone* skips the trash step and the record is cleaned up. A **confirmation dialog** is
required, stating plainly that the file goes to the Recycle Bin (recoverable) and N chunks leave the
index. **C: single-document first**; checkbox **multi-select** (bulk delete + move-to-collection) is the
next pass.

## Consequences

- **Easy:** removing a mis-ingested or unwanted document, recoverably; the confirmation + Recycle-Bin
  framing makes it low-anxiety. Reuses the existing `cleanup_orphan_figures` + `db.delete(...)` paths.
- **Committed to:** a new base dependency (`send2trash`, pure-Python, cross-platform); a destructive
  `DELETE /api/library/documents/{id}` that acts on the user's filesystem (always the local host). The
  live delete is **not** exercised in automation — it recycles a real file — so it is proven by
  integration tests (temp DB + fake Chroma + monkeypatched `send2trash`) and the UI is verified up to the
  confirmation; the real end-to-end is the user's to run.
- **Hard / deferred:** multi-select (bulk delete, move-to-collection); a true undo *inside the app*
  (recovery is via the OS Recycle Bin, not an in-app trash view); purging the file permanently.

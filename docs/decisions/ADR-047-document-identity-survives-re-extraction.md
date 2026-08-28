<!-- status: active · updated: 2026-08-25 · class: append-only -->

# ADR-047 — A document's identity is its source file, not its extracted text

- **Status:** accepted (built)
- **Date:** 2026-08-25
- **Deciders:** user (2026-08-25, on a measured account of what a re-ingest would cost),
  Claude (Claude Code session 2026-08-25)
- **Relates to:** ADR-042 (identity is the source, not its extraction — this **partially executes**
  it) · RG-027 (collapse the two identities — still open, this is not it) · KI-43 (the symptom this
  removes) · KI-46 (recovering a document mints a new identity) · KI-48 (the re-extraction that
  forced the question) · ADR-013 (metadata overrides, keyed on the id) · ADR-046 (`source_sha256`,
  the other identity added recently)

## Context

`_existing_document_id` resolved a document by `doc_hash` — a hash of its **extracted text**. That
made identity hostage to the extractor: every extraction improvement changes the text, which
changes the hash, which mints a fresh UUID, and everything keyed to the old id is cut loose. KI-43
has recorded that as a known weakness since 2026-08-08 ("figures were at 9% of true size") and
KI-46 records the user-visible version of it.

It stayed theoretical until KI-48 produced a concrete, imminent re-extraction of the whole corpus.
**Measured on the live library before deciding**, a re-ingest at that moment would have orphaned:

| table | rows | regenerable? |
|---|---|---|
| `figures` | 881 | only by re-running the figure + VLM description pass — real money |
| `document_keywords` | 1,455 | yes, enrichment pass |
| `ingestion_events` | 1,210 | no — it is the history |
| `chunk_epistemics` | 445 | yes, enrichment pass |
| `document_field` | 82 | yes |
| `concept_presence` | 31 | yes |
| **`document_folders`** | **18** | **no — the user's filing** |
| **`document_meta`** | **1** | **no — a title/author/year the user typed (ADR-013)** |
| **total** | **4,123** | |

The 19 unregenerable rows are the part that decided it. A re-extraction is a maintenance
operation; losing hand-made data to one is not a trade anyone would knowingly take.

## Options

**Option 1 — accept the orphaning.** Re-ingest, then re-run every enrichment pass and re-create
the hand-made rows by hand. No code change. Pays the VLM cost again, and silently loses whatever
nobody notices is missing.

**Option 2 — a one-off remap script.** Snapshot `source_original → document_id` before the
re-ingest and rewrite the sidecar foreign keys afterwards. Solves this instance and leaves the
mechanism in place to fire again on the next extractor change.

**Option 3 — resolve identity by source file, falling back from the hash.** The same file is the
same document. Fixes the class of problem rather than the instance, and moves in the direction
ADR-042 already committed to.

**Option 4 — collapse the two identities properly (RG-027).** Make the source hash *the* identity,
retire `doc_hash` as a key, migrate everything. The correct end state, and much the largest — it
touches the dedup gate, the chunk metadata, the sidecars and the eval store.

## Decision

**Option 3.** `_existing_document_id(doc_hash, source_original)` tries the text hash first — an
exact, indexed lookup, and still the answer whenever extraction is stable — then falls back to the
**source path**, compared case- and separator-normalised (`registry.pathkey`), because the same
file reaches this code both resolved and unresolved depending on the caller.

Option 4 remains the right end state and RG-027 stays open. This is deliberately the smaller move:
it removes the data-loss hazard now, without a migration, and nothing it does has to be undone to
get to RG-027 later.

**`upsert_document_in_sqlite` had to change with it, and that is not incidental.** It looked its
row up by `doc_hash` too. Those were the same question until this ADR; they are not any more —
under the fallback the caller reuses the id while the hash moves, so a hash lookup misses, falls
through to the insert branch, and **collides on the primary key**. It is now keyed on the resolved
`document_id`, and updates `doc_hash` on the row it finds: identity is stable, content is not, and
the row must record what it actually holds. Resolving identity is the resolver's job; writing is
the writer's.

## Consequences

**What this buys.** A document survives its own re-extraction with its figures, keywords,
epistemics, folders and typed metadata intact. Extraction can now be improved without the corpus
paying for it — which was the precondition for KI-48's re-ingest being a maintenance operation
rather than a destructive one.

**What it costs, stated plainly: a different document written to the same path inherits the
previous one's id, and with it every sidecar row.** For a library where the path *is* the
document's address — which is what the copy-in placement mode of ADR-046 makes it — that is the
intended reading rather than a bug. But it is a real semantic change and it is asserted in
`tests/unit/test_document_identity.py::test_replacing_the_file_at_a_path_inherits_the_identity` so
it cannot drift silently.

The exposure is bounded by two things already in place. Copy-in mode (ADR-046) gives the app
control of its own folder, so paths are not casually reused. And `source_files.source_sha256`
records the bytes at add time, so a future check can notice that the file at a path is not the
file that was registered there — the substrate for tightening this without another ADR.

**What it does not change.** `doc_hash` is still computed, still stored, still the first key, and
still what the dedup gate uses to skip unchanged documents. Chunk metadata is untouched. No
migration: the fallback only fires where a lookup previously failed, so an existing library
behaves identically until an extractor change moves a hash.

**Where it is still wrong.** Identity is now *path*-shaped, not *content*-shaped. A file moved
between folders still reads as a new document. That is RG-027's remit and this ADR does not
pretend to close it.

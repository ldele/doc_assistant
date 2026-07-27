<!-- status: draft · updated: 2026-07-27 · class: append-only -->

# ADR-030 — Document outline layer and the per-document map surface

- **Status:** proposed (stub — needs `grill-me` before the Decision section is filled)
- **Date:** 2026-07-27
- **Deciders:** user (product), Claude (Cowork planning session 2026-07-27)
- **Plan:** `docs/PLAN_2026-07-27_maps-trust-reports.md` Track 1

## Context

The user wants a mind-map surface, prompted by NotebookLM's. Theirs is a single LLM pass over the
whole source set producing unlabelled topic nodes: no provenance, not editable, PNG-only export.
The version worth building here is derived from structure that can be pointed at, where every node
resolves to chunks.

Two facts constrain it:

1. **Section metadata is flat.** `ingest/chunking.py::extract_chunk_metadata` records the nearest
   preceding heading's *text* and discards its *level* — `HEADING_MARKER` captures the hash run as
   group(1) and never uses it. No hierarchy is recoverable from stored chunk metadata.
2. **`DocumentPart`** (`id`, `document_id`, `parent_part_id`, `kind`, `title`, `order_index`) exists
   and is **empty**; chunks do not link to it. `architecture.md`: *"Document structure is scaffolded
   but unused."* This is the ADR-009 seam.

What already exists and does not need rebuilding: `concept_presence.chunk_keys_json` carries
`"{document_id}:p{parent_index}"`, so concept → parent chunk is a known join;
`load_concept_presence` serves it per concept; reversing it from the Library doc view is the
specced-but-unbuilt **PR-G2c**.

Placement is entangled with a live fork. `ui-checklist.md` §3 (2026-07-20) records the
**Graph-destination fork** as parked with the user explicitly undecided, and states: *"do not add a
second graph surface without settling that fork."* ADR-017 A1 additionally makes graph surfaces
read-only over the vocabulary, and ADR-028 D11 gives all tree edits to the taxonomy view.

Non-negotiable 4 (Enrichment-Layer Pattern) forbids mutating the chunk store, so a chunk→section
link cannot be written into Chroma metadata.

## Options

### Outline derivation

1. **Flat two-level map from the existing `section` string.** — *Pros:* ~a day, no migration,
   no new table population. *Cons:* breaks on books (the corpus shape ADR-009 was written for);
   leaves `DocumentPart` scaffolded indefinitely; the level information is available and being
   thrown away.
2. **Populate `DocumentPart` by re-parsing the cached extracted markdown (proposed).** — *Pros:*
   $0, deterministic, idempotent, no re-extraction, no chunk-store write; closes the ADR-009 seam;
   benefits the chunk browser and reading surfaces too. Chunk→part joins via `char_start`/`char_end`
   → `parent_index`, stored on the part row. *Cons:* additive migration; heading-level parsing over
   three extractors' markdown will produce some false structure — needs a bounded before/after diff
   on the live corpus.
3. **LLM-generated structure (NotebookLM's approach).** — *Pros:* works on documents with no
   headings. *Cons:* non-reproducible, costs money per view, and it is precisely the
   open-vocabulary extraction deleted on 2026-07-07 (KI-7). Contradicts the ADR-008 Node-A/Node-B
   split: the LLM annotates existing structure, it never creates it.

### Placement

1. **Top-level "Mindmap" tab** (the user's initial framing). — *Pros:* discoverable; matches the
   competitor. *Cons:* collides head-on with the parked Graph-destination fork; a per-document
   object as a global destination needs a document-picker the app does not have.
2. **Panel in the Library document view, sibling to `DocConnections.svelte` (proposed).** —
   *Pros:* a per-document map belongs where the document is; identical to the shape the user chose
   deliberately for E4 (2026-07-22); dodges the fork entirely; `GapList.svelte` precedent shows a
   self-contained panel can be relocated later without an API break. *Cons:* less discoverable;
   defers rather than answers the destination question.
3. **Settle the fork first, ship a unified "Explore" destination** absorbing graph + gaps + map. —
   *Pros:* resolves three parked items at once; probably the right end state. *Cons:* blocks Track 1
   on a grill the user has twice declined to settle.

## Decision

*(open — fill after `grill-me`)*

Candidate: **outline option 2 + placement option 2**, with option 3 named as the intended end
state once fork F1 settles. Cross-document linking is emergent only — a link exists iff the same
`Concept` is present in both documents; the map never invents an edge. Claim-level leaves are out
of scope (fork F3: `AnswerClaim` is answer-scoped; document-scoped claims are an unbuilt LLM layer
with its own precision floor).

## Consequences

*(open — fill with the Decision)*

Provisional:

- **Easy:** the read model is a join over shipped data; the layout module is pure and testable
  (`lib/library/treeLayout.ts` — `forceLayout.ts` is force-directed only, no root or depth, so this
  is new code, not a parameterisation).
- **Hard / committed:** an additive `DocumentPart` migration; a documented honesty contract for
  heading-less documents (one implicit root part, and the UI says so — never synthesise structure);
  a bounded outline-diff verification over the live 76-doc corpus before the runner is trusted.
- **Boundary:** the map is read-only over the vocabulary (ADR-017 A1); all tree edits stay with the
  taxonomy view (ADR-028 D11). The map surfaces derived structure; it is not a second curation UI.

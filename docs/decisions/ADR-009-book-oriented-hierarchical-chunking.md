<!-- status: active · updated: 2026-07-03 · class: append-only -->

# ADR-009 — Book-oriented hierarchical chunking

- **Status:** proposed
- **Date:** 2026-07-03
- **Deciders:** user (scope + corpus direction), Claude (review + drafting)

## Context

The current chunking scheme (documented in `architecture.md § Chunking & retrieval
units`) is parent–child: 2000-char parents, 400-char children, linked only through
Chroma metadata (`parent_text`/`parent_index`/`child_index`). It is tuned for
**short, section-headed research papers** — the corpus the project was built on. An
app review (2026-07-03) traced how it degrades on **long, chaptered books**. The
failure modes, with file evidence:

1. **Section detection collapses.** The heading detector expects markdown `## `
   (`ingest/chunking.py`). Books use prose chapter markers ("CHAPTER 3"), so
   `section` stays null and the health model penalises "few sections detected"
   (`health.py`), marking healthy extractions marginal/broken.
2. **`TOP_K=10` starves long docs.** Ten ~400-char children ≈ 1,000 words drawn
   from a ~100,000-word book — scattered fragments from unrelated chapters instead
   of one coherent passage. `TOP_K`/`CANDIDATE_K` are corpus-agnostic constants
   (`config.py`).
3. **No chapter hierarchy.** Nothing links a chunk to a chapter, so retrieval can't
   scope to "Chapter 5" or prefer chapter-boundary parents. The `DocumentPart` table
   (`db/models.py` — `kind`, `title`, `parent_part_id`) is scaffolding: defined,
   never populated, never linked to chunks.
4. **Multi-page book tables fragment.** Caption→table binding assumes a paper-sized
   table fits in one parent; a census/financial table spanning pages does not
   (`ingest/tables_marker.py`).
5. **Figures rank poorly at scale.** Figure chunks are appended as synthetic
   high-`parent_index` parents; in a 200-figure book they lose to prose unless the
   query is visually specific (`ingest/__init__.py`).
6. **No figure/table ↔ in-text-mention link.** Captions are paired to regions by
   vertical proximity (`figures.py:188`), never to the "see Figure 3" prose that
   references them. The referencing sentence and the figure live in different chunks
   with no cross-link, so retrieving one does not surface the other.

Constraint: chunk sizes and retrieval weights are **locked settings** — any value
change rides an eval-harness experiment (beat the control, record a baseline), per
`.claude/CONTEXT.md`. This ADR decides *direction and shape*, not final numbers.

## Options

1. **Do nothing; books use the paper scheme (status quo).** — *Pros:* zero work; papers
   unaffected. *Cons:* books are effectively second-class — mis-scored health,
   incoherent retrieval, no structural navigation. Rejected: the review shows this is
   a real, reproducible degradation, and books are a stated corpus direction.

2. **One flat larger chunk size for everything.** Bump parent/child sizes so books
   cohere. — *Pros:* trivial. *Cons:* breaks papers (over-large chunks dilute
   precision), still no chapter structure, still corpus-global. A locked-setting change
   with a *negative* expected effect on the paper corpus. Rejected.

3. **Populate `DocumentPart` as a chapter/section tree and add a chapter grain above
   the current parent (chosen direction).** Three grains — chapter → parent(section)
   → child — with chunks linking to their `DocumentPart`. Detect prose chapter
   headings at ingest; relax health bounds for book-density documents; enable
   chapter-scoped retrieval so `TOP_K` is spent within the relevant chapter rather
   than across the whole book. Table/figure handling gets a table-specific split
   (child = caption + matching rows, parent = full grid) rather than one atomic block.
   — *Pros:* uses the schema that already exists for exactly this; keeps papers on the
   current path (a paper is just a one-"chapter" degenerate case); makes chapter
   navigation and coherence possible. *Cons:* most work; introduces a doc-type or
   structure-detection step; several locked-setting experiments to land.

## Decision

Adopt **Option 3 as the direction**, staged so each locked-setting change is a
separate measured experiment rather than one big bang:

- **A. Structure detection + `DocumentPart` population.** Detect chapter/section
  boundaries (markdown headings *and* prose patterns like `^CHAPTER \d+`), write the
  tree into `DocumentPart`, and stamp each chunk with its part id in Chroma metadata.
  No retrieval behaviour change yet — this is the enabling data layer. Papers produce
  a shallow tree; nothing regresses.
- **B. Health bounds by structure/density.** Stop penalising legitimately book-shaped
  documents (chunks-per-page, section-rate signals in `health.py`). Informational
  only — never blocks retrieval — so this is low-risk.
- **C. Chapter-scoped retrieval + `TOP_K` review.** Allow retrieval to concentrate the
  budget within a chapter (or a small set of chapters) for long documents, so the
  top-K isn't smeared across an entire book. `TOP_K`/`CANDIDATE_K` changes go through
  the eval harness.
- **D. Table-specific parent–child.** Child = caption + matching rows; parent = full
  grid. Fixes multi-page book tables without changing the atomic behaviour papers rely
  on for small tables.

**Deferred (explicitly not in this ADR):** figure/table ↔ in-text-mention linkage,
structured-table fidelity (md/HTML/JSON hybrid), and multimodal image embeddings.
These are captured in `docs/specs/table-figure-future-work.md` as future work.

## Consequences

- **Makes easy:** book ingestion that scores correctly; "summarise Chapter 7"-style
  navigation; coherent long-document retrieval; a real home for the `DocumentPart`
  scaffolding.
- **Makes hard / commits us to:** a structure-detection step at ingest (a new failure
  surface — mis-detected chapters); a metadata migration to add part ids to existing
  chunks (re-ingest, per the two-tier cache rules); several eval-harness experiments
  before any locked value moves. Chapter-scoped retrieval adds a branch to the
  retrieval path that must stay a no-op for single-chapter (paper) docs to preserve the
  paper baseline.
- **Guard:** papers must not regress. Each stage keeps a paper-corpus control; a book
  improvement that costs paper quality is not a clean win.
- **Opens:** whether concept-graph sampling should also become structure-aware
  (stratify the 12 excerpts across chapters instead of taking Chroma's first 12) — a
  sampling change, not a chunking change, tracked separately.

<!-- status: active · updated: 2026-07-03 · class: living -->

# Table & figure handling — future work (non-priority)

Design notes for improvements to how tables and figures are extracted, linked, and
retrieved. **None of this is committed work** — it captures the shape of each idea so
the decision is on record when priority allows. Referenced from
`ADR-009` (deferred items) and `docs/figures-and-tables.md` (current state).

Current state, for grounding: a table is spliced into the cached markdown and merged
with its caption into one atomic parent==child chunk; a figure becomes a
`chunk_type="figure"` chunk only after the VLM pass, embedding `(caption +
vlm_description)` — the PNG is never embedded. Retrieval is text-only.

## 1. Figure/table ↔ in-text-mention linkage (highest value, lowest cost)

**Problem.** A figure/table is findable only through its own caption/description.
The prose that *references* it ("as shown in Figure 3, accuracy drops sharply") is a
separate chunk with no link back. Ask about the trend and the referencing sentence
may retrieve while the figure does not — or vice versa.

**Sketch.** At ingest, regex the body text for `Figure N` / `Table N` / `Fig. N`
references and resolve them to the figure/table sidecar row via a per-document
numbering index. Then either:

- **(a) Enrich the referrer** — inject the caption (+ description) into the chunk
  that references it, so the reference carries the figure's text signal; or
- **(b) Co-retrieve** — when a referring chunk is retrieved, pull its linked
  figure/table into context alongside it.

**Cost:** low — regex + a numbering index built during extraction. **Value:** high —
directly fixes the most common "the figure that answers this never showed up" case.
(a) is cheaper and self-contained; (b) touches the retrieval path.

## 2. Structured-table fidelity — markdown/structured hybrid

**Problem.** Markdown is lossy for non-rectangular tables: merged/spanning cells,
multi-row headers, and footnotes collapse to a flat line. The model reasons over
markdown well, but programmatic lookups ("value at row X, col Y") and complex layouts
degrade.

**Sketch.** Keep markdown as the **embedded, LLM-readable** representation — it's what
the model handles best — but *also* persist a **structured sidecar**: HTML `<table>`
or JSON rows/cols, capturing the richer structure Marker already detects instead of
flattening it. Retrieval and synthesis stay on the markdown; the structured form backs
fidelity and any future cell-level lookup or re-rendering.

**Cost:** medium — a second stored representation + capturing Marker's structure
before the markdown flatten. **Value:** medium — matters for data-heavy tables and any
"read this exact cell" use case; little benefit for simple result tables.

## 3. Multimodal image embeddings (research direction)

**Problem.** A figure's meaning that lives in the pixels — a trend, a diagram's
layout — is only as retrievable as whatever the VLM wrote about it. Nothing matches on
the image itself.

**Sketch.** Add a **parallel visual retrieval arm**. A CLIP-style or
ColPali/ColQwen-style image embedder runs over each figure PNG (optionally a rendered
table image) at ingest, producing image vectors in a **separate Chroma collection**.
At query time the text query embeds into the *same joint space*; figures retrieve by
visual-semantic match, caption-free. This slots into the existing arm-based retrieval
(BM25 + vector ensemble) as a **third arm** fused in.

**Open issues.**

- The cross-encoder reranker (`bge-reranker-base`) is text-only — it can't score the
  visual arm, so visual results would need separate ranking or a multimodal reranker.
- Second model + GPU at ingest time; storage for a second vector space.
- Fusing a visual arm into the current text-only ensemble is a non-trivial retrieval
  change with its own eval-harness experiment.

**Cost:** high — new model, new collection, retrieval-path surgery. **Value:**
potentially high for figure-heavy corpora, unproven for this one. Prototype and
measure before committing (spike, not a build).

## Priority

1. **In-text-mention linkage (1a)** — cheap, high value, self-contained. First if any.
2. **Table-specific parent–child** — covered by ADR-009 § D, not here.
3. **Structured-table hybrid (2)** — when data-heavy tables become common.
4. **Multimodal embeddings (3)** — research spike only; not on the near roadmap.

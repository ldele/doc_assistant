<!-- status: active · updated: 2026-10-01 (ROADMAP 94 built: the no-movement check, and one mark tried on a copy) · class: baseline -->

# Names and forms — what moved when exact and broad forms were built (ADR-054, ROADMAP 94)

**Nothing moved.** With no form classified, the graph build gives the same presence, the same
edges and the same graph version as the code before the change, on the working library. A form's
mark changes a count only once the user sets it: on a copy of the library, marking `distillation`
broad took `knowledge distillation` from 11 documents to 4, with the other 7 listed beside it.

These are counts on one library. They say what the build does, not whether a given form should be
exact or broad — that reading is the user's, form by form.

## Environment / method

- **Library:** the working library, 104 documents; 357 text-bearing rows, of which 13 are concepts
  (`graph_include`) and 344 are terms. The 13 hold 31 alias rows; 9 of them repeat the concept's
  own name, which leaves **22 forms** to read. None carried a mark.
- **Scope:** whole text — every parent chunk, which is what the graph counts. Section 6 of
  `vocabulary_shape_2026-10-01.md` counts body prose, so its figure for `knowledge distillation`
  (9 documents) is smaller than the graph's (11).
- **The comparison:** `build_concept_skeleton(apply=False)` run twice against the same library,
  once with the committed code (`b955611`, extracted with `git archive`) and once with the
  row-94 working tree. A dry run writes nothing to the library database or the stored graph.
  Reading the vector store opens it, which adds a row to Chroma's own lock table, as starting the
  app does.
- **The copy:** the library database, both vector stores and the graph sidecars copied to a scratch
  directory, served through `DOC_DATA_DIR`. Every write below was made there. The working library
  holds no mark, and its graph was not rebuilt.
- **Cost:** $0. No model.

## 1 · Nothing classified — the committed code and the new code agree

| | Committed code | Row-94 code |
|---|---:|---:|
| Graph version | `cb99a7f9f36545da` | `cb99a7f9f36545da` |
| Concepts | 13 | 13 |
| Edges | 30 | 30 |
| Documents with a concept | 35 | 35 |
| Concept–document pairs | 86 | 86 |

Compared field by field, the two runs are equal: the documents of each concept, each label and its
written form, each degree, and each edge's ends, shared-chunk count, provenance and weight. No
concept has a document beside its presence, because no form is marked broad.

Documents per concept, both runs: `contrastive learning` 13 · `knowledge distillation` 11 ·
`re-ranking` 11 · `BM25` 8 · `hard negatives` 8 · `dense retrieval` 7 · `passage retrieval` 7 ·
`cre` 6 · `retrieval-augmented generation` 6 · `dbs` 3 · `cross-encoder` 2 · `ntsr1` 2 ·
`pddl` 2.

## 2 · The stored graph is one rebuild behind, for an earlier reason

The graph stored on 2026-09-18 (version `65d6698ccf8bbf14`) differs from both runs in one concept:
`cre`, 7 documents stored and 6 in a rebuild today. That is the case-aware matching of 2026-09-30
(`written_forms_2026-09-30.md`), which takes effect at the next rebuild. Row 94 adds nothing to it.

## 3 · One form marked broad, on the copy

`distillation` marked broad on `knowledge distillation`, then the graph rebuilt in the app:

| | Before | After |
|---|---:|---:|
| `knowledge distillation` — documents | 11 | 4 |
| — listed beside, through `distillation` | — | 7 |
| — links to other concepts | 6 | 6 |
| Edges in the graph | 30 | 30 |
| Documents with a concept | 35 | 32 |
| Concept–document pairs | 86 | 79 |
| Graph version | `cb99a7f9f36545da` | `2ace19350c797b09` |

- No edge was lost. Six edges of `knowledge distillation` rest on fewer shared chunks: with
  `dense retrieval` 16 → 10, `passage retrieval` 9 → 5, `contrastive learning` 5 → 3,
  `cross-encoder` 5 → 3, `re-ranking` 4 → 3, `BM25` 3 → 2.
- Three documents left the graph's coverage: they reached it only through the bare word.
- No other concept changed.
- Before the rebuild the graph said which concept it was behind on ("Name or forms changed since
  this graph was built: knowledge distillation"), and kept showing 11 until rebuilt.
- Clearing the mark and rebuilding gives the first table back: a test asserts the stored graph is
  byte for byte the unmarked one (`test_nothing_moves_until_a_form_is_marked_broad`).

This is one mark, chosen to exercise the path. What classifying the 22 forms does to presence, the
edges and the gap list is still open, and is measured after the user classifies them
(`.claude/RIGOR_TODO.md` RG-032).

## 4 · Opening a term

Two terms opened in the app on the copy, each with "Look in my library":

- `viral` — three passages under *how your library uses it*; the look found no sentence that reads
  as a definition, and the panel said so.
- `SPECTER` — two sentences that define it, each with its evidence and its source, and no control
  to choose, edit or dismiss one.

After both, the definitions table held no row for a term. Whether this is as usable as opening a
concept is the user's to judge on their own terms (RG-032, third claim).

## Reproduce

The dry run is `doc_assistant.knowledge.concept_skeleton.build_concept_skeleton(apply=False)`;
compare `skeleton.meta["graph_version"]`, each node's `doc_ids` and each edge between the two
trees. The probes used here are local working files and are not in the repository.

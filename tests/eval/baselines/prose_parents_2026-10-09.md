<!-- status: active · updated: 2026-10-09 (ROADMAP 97: the knowledge layer reads prose parents, not figure chunks; section 5 replaces section 7 of vocabulary_shape_2026-10-01.md) · class: baseline -->

# Prose parents — what moved when the knowledge layer stopped reading figure chunks (ROADMAP 97)

**On the graph's 13 concepts nothing a reader sees moves: the same documents, the same 25 links,
the same 12 gap rows. Three links rest on one or two fewer shared chunks.** Over the whole
vocabulary 24 of 1,103 concept–document pairs go, each of them a word that occurred only in a
vision model's description of a figure. The bibliography cut now fires on 85 documents instead of
76. Section 5 restates section 7 of `vocabulary_shape_2026-10-01.md`, whose figures included
figure chunks.

These are counts on one library. They say what the change did here, not how large the effect is on
another library or with another share of described figures.

## Environment / method

- **What a figure chunk is.** A described figure is stored as a parent chunk of its own, numbered
  after the document's prose parents: its caption, the vision model's description, a `---` line,
  and a whole copy of the prose parent that cites the figure (`ingest.figures.figure_parent_text`).
  Retrieval wants exactly that: an answer reads the figure inside its passage.
- **Library:** the working library, 104 documents, with the user's marks of 2026-10-09 in place
  (22 forms: 10 broad, 12 exact). 8,861 parents, of which **615 are figure parents, in 83
  documents**. Figure parents hold 808,210 of 12,222,136 characters (6.6%): 467,025 of caption and
  description, 339,645 of copied prose. 220 figures carry a copy (189 found by a citing passage, 31
  by where the caption sits); **181 prose parents are stored more than once** — 156 twice, 21 three
  times, one each 4, 5, 8 and 9 times.
- **The comparison:** the committed code (`b50941a`, extracted with `git archive`, pointed at the
  same library) against the change, each as a dry run: `build_concept_skeleton(apply=False)` and
  the gap detectors, `match_presence` over all 357 text-bearing rows, `derive_written_forms`, the
  bibliography cut as `definitions._body_chunks` applies it, and
  `extract_definitions(apply=False)` for the 13 concepts and for every row. Nothing was written to
  the library or the stored graph.
- **Cost:** $0. No model.

## 1 · What is read

| | Before | After |
|---|---:|---:|
| Parent chunks | 8,861 | 8,246 |
| Characters | 12,222,136 | 11,413,926 |

## 2 · The graph's concepts (13)

| | Before | After |
|---|---:|---:|
| Graph version | `2436674f0e398cbd` | `58136820706f4cab` |
| Links | 25 | 25 |
| Documents with a concept | 29 | 29 |
| Concept–document pairs | 72 | 72 |
| Gap rows (deterministic) | 12 | 12 |

Every concept keeps its documents, its documents beside (through broad forms) and its number of
links; the 12 gap rows are on the same concepts with the same kinds and the same evidence counts.
The version differs because three links counted a copied passage:

| Link | Shared chunks before | After |
|---|---:|---:|
| `re-ranking` — `BM25` | 51 | 49 |
| `passage retrieval` — `BM25` | 14 | 13 |
| `retrieval-augmented generation` — `BM25` | 5 | 4 |

The same comparison made that morning, before the user had marked any form: again no document,
link or gap row moved, and four links lost one or two shared chunks (the three above at 53 → 51,
59 → 58 and 5 → 4, and `contrastive learning` — `dense retrieval` 17 → 16).

A link needs two shared chunks. A pair of concepts that share one passage reached two as soon as a
figure cited that passage. No link on this graph rested on that alone.

## 3 · The whole vocabulary (357 rows)

1,103 concept–document pairs before, **1,079** after: 24 gone, none gained. All 24 are terms, none
is one of the 13 concepts:

| Term | Pairs gone |
|---|---:|
| `plateau` | 8 |
| `unlabeled` | 4 |
| `vascular` | 4 |
| `actor`, `beta`, `c-mtp unlabeled`, `heart`, `oscillations`, `retrieval accuracy`, `saturation`, `state-space` | 1 each |

Matched separately against the captions and against the descriptions of each document's figures,
every one of these words is in a description and in no caption: the model wrote it, the document
did not. For scale, the descriptions hold 199 pairs in all, and 174 of them are also in the prose
of the same document.

A 25th pair, `ap10k` in one document, was counted through a description while figure text was
read: the vote, which then saw the descriptions, gave the form the spelling `AP10K`. On prose
parents the vote finds no use of the form in a running sentence, so it has no written spelling
and matches in any case, and the document's own text holds it. The pair stays, by another route.

## 4 · Written forms

367 forms before, 366 after. Four change, all on terms:

| Term | Form | Before | After |
|---|---|---|---|
| `ai usage` | `ai usage` | `AI usage` | `AI Usage` |
| `bibliometric` | `bibliometric` | `Bibliometric` | `bibliometric` |
| `bibliometric` | `bibliometric analysis` | `Bibliometric Analysis` | `bibliometric analysis` |
| `ap10k` | `ap10k` | `AP10K` | no written spelling |

116 other forms keep their spelling with different use or document counts. The spellings stored
in the working library were voted with figure text included (the rebuild of 2026-10-09); they
follow at the next rebuild.

## 5 · The bibliography cut — replaces section 7 of `vocabulary_shape_2026-10-01.md`

That section joined every parent chunk of a document. A figure parent is numbered after the prose,
so its caption, its description and its copy of a prose passage landed after the reference
heading, and were counted as text the cut removes.

| | Every parent (as published 2026-10-01) | Prose parents |
|---|---:|---:|
| Characters | 12,230,893 | 11,422,068 |
| Documents the cut fires on | 76 | 85 |
| Characters the cut removes | 2,005,462 (16.4%) | 1,852,830 (16.2%) |
| In content sections after the reference list | 555,688 in 39 documents | 424,982 in 32 documents |
| — share of the removed text · of all text | 28% · 4.5% | 23% · 3.7% |
| Pairs that exist only past the cut (357 rows) | 113 | 111 |
| — of which in a content section | 24 | 23 |
| — of which on the 13 concepts | 11 | 13 |

- **The cut missed nine documents.** It wants the References heading past the halfway mark of the
  text. With figure text appended, the same heading sat before that mark in nine documents whose
  references are followed by long appendices or by many figures: *Retrieval-Augmented Generation
  for Knowledge-Intensive NLP Tasks*, *SciRepEval*, *Mamba-UNet*, *Medical SAM 2*, *Relational
  recurrent neural networks*, *Enabling Agents to Communicate Entirely in Latent Space*, *Large
  Language Models Pass the Turing Test*, *AI Usage Cards* and *Mapping Political Theory Using
  Bibliometric Analysis*. In those nine the definition scan and the written-form vote read the
  reference list as body. The two columns therefore do not hold the same documents.
- **In 26 documents** — those nine and 17 with no reference heading at all — 407,198 characters of
  figure text sat inside what the definition scan and the written-form vote took for body prose.
- **The finding of 2026-10-01 stands, smaller:** the cut drops appendices, methods and supplementary
  sections placed after the reference list — 3.7% of the library's prose in 32 documents, not 4.5%
  in 39. Of the two headings that section quoted as "emitted late" by the extractor,
  `5 CONCLUSION AND FUTURE WORK` (Res2Net) is a prose parent and is one; `5.2 Results and Analysis`
  is a prose parent before the references and again inside a figure's copy after them.
- **Which column a reader lived in.** The keyword extractor reads the cached markdown, which holds
  no figure chunk: the right-hand column, before and after. The definition scan and the written-form
  vote read parent chunks: the left-hand column until this change, the right-hand one since.

## 6 · Definition candidates

| Rows read | Candidates before | After | Lost | Gained |
|---|---:|---:|---:|---:|
| The 13 concepts | 20 | 20 | 0 | 0 |
| All 357 rows (a measurement; a term's are never stored) | 110 | 110 | 0 | 0 |

The same sentences in both runs, the nine documents included. Both runs matched with the
spellings stored in the library, so this isolates the text that is read.

## 7 · Not measured

- Retrieval, which this change does not touch: a figure and the passage it cites can both be
  retrieved, and the answer then reads that passage twice.
- The one-concept path ("Look in my library", the usage examples) on the working library's keyword
  index. Its rule is the same and is covered by a test; what it shows for a given concept before
  and after was not compared.
- What the definition candidates do once the stored spellings are re-voted on prose (section 4
  moves four forms, none of them a concept's).
- The same counts on another library. Every number here is from one library of 104 documents, 83
  of them with described figures.

## Reproduce

The reader is `doc_assistant.knowledge.concept_skeleton.load_presence_inputs`; the rule is
`prose_parents`. Run the dry runs named above once with the code before the change
(`git archive b50941a src`, `PYTHONPATH` at the extracted tree, `DOC_DATA_DIR` at the library) and
once with the code after it, and compare `skeleton.meta["graph_version"]`, each node's `doc_ids`,
each edge's `n_cooccurrence_chunks`, and the pair sets. The probes used here are local working
files and are not in the repository.

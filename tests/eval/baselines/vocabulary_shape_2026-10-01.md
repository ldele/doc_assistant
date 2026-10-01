<!-- status: active · updated: 2026-10-01 (the vocabulary snapshot behind ADR-054) · class: baseline -->

# What the 357 concept rows are, read against the library's text (ADR-054)

**Thirteen rows are concepts someone chose; the other 344 are keyword strings from one bulk
promotion, and most of them cannot link documents.** Of the 344, 197 occur in the prose of at most
one document, 30 in none, and 113 are a keyword of no document today. For the 13, a single alias
can carry most of a concept's presence: 7 of the 9 documents counted for `knowledge distillation`
are reached only through the bare word `distillation`. Separately, the bibliography cut the
definitions layer reads through removes more than reference lists (section 7).

These are counts on one library. None of them is a graded signal: nothing here says how well a
neighbour count or a spread count separates a concept from a fragment.

## Environment / method

- **Library:** the working library at commit `a5bb634`: 104 documents, 8,861 parent chunks, 12.2
  million characters. Read-only; nothing was written and the stored graph is unchanged.
- **Rows:** `concepts` holds 593 rows: 357 with `kind = concept` (the vocabulary this file is
  about) and 236 with `kind = domain` (the field taxonomy). "The 13" are the rows with
  `source = manual`, which are also the only rows with `graph_include`. "The 344" have
  `source = keyword`.
- **Matching:** the shipped matcher (`concept_skeleton.match_presence` with the written forms of
  2026-09-30), label and aliases, whole word, case-aware where the library writes a capital.
- **Text scopes:** *whole text* is every parent chunk. *Body* is the text before the bibliography
  cut (`definitions._body_chunks`, which calls `keywords.strip_reference_section`). *Prose* is the
  sentences `definitions.document_sentences` returns from the body: headings, tables and lists
  dropped.
- **Neighbours** (section 4): for each mention of a one-word label in prose, the word right after
  and right before it, joined by a space or a hyphen, function words skipped, plural `s` folded on
  words longer than four letters.
- **Probes:** read-only scripts kept with the local working state, not in the repository. Each
  section states what was counted so the count can be redone from the shipped functions it names.
- **Cost:** $0. No model.

## 1 · Where the rows came from

| Rows | `source` | Created | On the graph | With a chosen definition |
|---:|---|---|---:|---:|
| 13 | manual | 9 on 2026-07-01, 4 on 2026-07-05 | 13 | 2 |
| 344 | keyword | all on 2026-07-05 | 0 | 0 |

The keyword layer was re-extracted after the 344 were promoted. Today it holds 1,615 keywords, 1,483
linked to a document, and 1,436 of those 1,483 on exactly one document; 103 of the 104 documents
carry keywords, 15 each.
**113 of the 344 labels are a keyword of no document** in the current extraction (their keyword rows
remain because a concept row points at them); 216 are a keyword of one document and 15 of two to
four.

## 2 · How far each row spreads

Documents in which a row's label or aliases occur (104 documents):

| Documents | 0 | 1 | 2 | 3–4 | 5–9 | 10–24 |
|---|---:|---:|---:|---:|---:|---:|
| The 344, whole text | 18 | 162 | 51 | 44 | 45 | 24 |
| The 344, body | 19 | 170 | 53 | 42 | 44 | 16 |
| The 344, prose | 30 | 167 | 54 | 38 | 45 | 10 |
| The 13, whole text | 0 | 0 | 3 | 1 | 6 | 3 |
| The 13, prose | 0 | 0 | 4 | 1 | 7 | 1 |

- 197 of the 344 (57%) occur in the prose of at most one document.
- 18 occur nowhere in the text. They are tokens joined across punctuation: `conf comput`,
  `comput vis`, `vis pattern recog`, `koonce emerson`, `schneider lee`, `learning animal`,
  `events behavior analysis`, `agent direction`, `resources neuroscience`, and nine more.
- 12 more occur only outside prose: `param1`, `param2`, `param3`, `baseline-knn`, `fne-tune`,
  `bior`, `query document`, `relevance judgement`, and four `superanimal …` or `c-mtp …` phrases.

## 3 · What the 344 labels look like

Classified by how the library writes the label (its written form): 175 phrases; 85 lower-case
words (`agent`, `passage`, `retrograde`); 51 with a capital past the first letter (`DeepLabCut`,
`SQuAD`, `NLRP3`); 21 capitalised (`Chinese`, `Parkinson`, `Sporns`, `Cohan`, `Moyle`, `Nassi`,
`Berciano`, `Pedr`, `Recog`); 8 short lower-case tokens (`pose`, `axon`, `bank`, `beta`); 4 with
a digit (`doc2query`, `param1`–`param3`).

## 4 · One-word labels and the words beside them

174 of the 357 labels are one word; 161 of those have at least 5 mentions in prose.

| Share of a label's mentions with the same word beside it | Same word after | Same word before |
|---|---:|---:|
| at least 50% | 22 labels | 7 labels |
| at least 30% | 37 labels | 17 labels |

Of the 22: sixteen are the front of a longer term, five are surnames followed by "et al.", one is
a given name followed by its surname.

| Label | Mentions | The word beside it | Reading |
|---|---:|---|---|
| `pose` | 498 | then "estimation" 326 | front of `pose estimation`, itself a row |
| `Shapley` | 14 | then "value" 11 | front of a longer term |
| `hyaline` | 17 | then "grume" 17 | front of a longer term |
| `NLRP3` | 22 | then "inflammasome" 12 | front of a longer term |
| `Cohan` | 22 | then "et" 20 | a cited author |
| `Moyle` | 13 | then "et" 13 | a cited author |
| `speckles` | 21 | after "nuclear" 21 | tail of a longer term |
| `bank` | 63 | after "memory" 48 | tail of `memory bank`, itself a row; "bank vole" 3 |
| `beta` | 131 | then "activity" 39, "oscillation(s)" 18, "network" 10; alone 26 | a modifier with several heads |
| `viral` | 103 | then "genome" 13, "tool(s)" 10, "vector(s)" 7; alone 2 | an adjective |
| `Cre` | 255 | then "driver" 39, "-dependent" 37, "line" 18; after "BAC" 58; alone 44 | short form of a longer name |

30 labels mostly stand alone (no content word on either side in at least 60% of mentions), among
them `Ntsr1`, `SQuAD`, `mIoU`, `caspase`, `TNF`, `fever`, `nucleolus`.

The last three rows of the table are the labels the user read on 2026-09-22 as "beta oscillations",
"viral vector" and "Cre recombinase". No single neighbour dominates any of them, and "viral vector"
is 7 of 103 uses of `viral`: the name of the concept is not recoverable from these counts.

This count differs from the "stands alone" figure in `concept_merge_cosine_2026-09-17.md`
(`pose` 651 of 733), which did not look at the neighbouring word.

## 5 · Labels written in more than one case

Eight labels have a second case form holding at least 15% of their mid-sentence uses (and at least
3 uses): `persona` (`PERSONA` 123, `persona` 80), `fever` (`FEVER` 16, `fever` 7), `sts` (`STS` 44,
`StS` 21), `cer` (`Cer` 13, `CER` 4), `pddl` (`pddl` 42, `PDDL` 20, one document each), `alfworld`,
`autopilot`, `swin-unet`. By reading, the first four are two different things under one label and
the last four are two spellings of one.

## 6 · Aliases, and what each form of the 13 contributes

318 of the 357 rows have a single alias, which is the label itself. 39 have an alias that differs;
the 13 hold 31 aliases.

Prose, case-aware, one form at a time. "Only" is the number of documents that no other form of the
same concept reaches.

| Concept | Form | Documents | Mentions | Only |
|---|---|---:|---:|---:|
| knowledge distillation | `knowledge distillation` | 2 | 206 | 0 |
| | `distillation` | 9 | 400 | 7 |
| contrastive learning | `contrastive learning` | 7 | 87 | 0 |
| | `contrastive` | 9 | 142 | 2 |
| hard negatives | `hard negatives` | 4 | 10 | 1 |
| | `hard negative` | 4 | 8 | 0 |
| | `negative passages` | 1 | 10 | 0 |
| | `negative sampling` | 6 | 17 | 2 |
| passage retrieval | `passage retrieval` | 3 | 15 | 0 |
| | `passage ranking` | 2 | 99 | 1 |
| | `dense passage retriever` | 3 | 9 | 0 |
| dense retrieval | `dense retrieval` | 5 | 127 | 2 |
| | `dense retriever` | 4 | 7 | 0 |
| | `dense passage retrieval` | 1 | 1 | 0 |
| | `zero-shot dense retrieval` | 1 | 3 | 0 |
| re-ranking | `re-ranking` | 7 | 32 | 2 |
| | `reranking` | 4 | 133 | 1 |
| | `re-rank` | 4 | 8 | 1 |
| | `re-ranker` | 3 | 7 | 1 |
| | `passage re-ranking` | 2 | 6 | 0 |
| retrieval-augmented generation | `retrieval-augmented generation` | 5 | 11 | 0 |
| | `RAG` | 5 | 340 | 0 |
| | `retrieval augmented generation` | 1 | 3 | 0 |
| cre | `Cre` | 5 | 255 | 4 |
| | `Cre recombinase` | 1 | 1 | 0 |
| dbs | `DBS` | 2 | 42 | 0 |
| | `deep brain stimulation` | 2 | 5 | 0 |
| pddl | `pddl` | 2 | 63 | 1 |
| | `planning domain definition language` | 1 | 1 | 0 |
| ntsr1 | `Ntsr1` | 2 | 24 | 2 |
| | `neurotensin receptor 1` | 0 | 0 | 0 |
| BM25 | `BM25` | 8 | 241 | 8 |
| | `Okapi BM25` | 0 | 0 | 0 |
| cross-encoder | `cross-encoder` | 2 | 35 | 2 |
| | `cross encoder` | 0 | 0 | 0 |

- A shorter alias can carry the concept: `distillation` alone reaches 7 of 9 documents,
  `contrastive` 2 of 9. `contrastive` is followed by "learning" in 87 of 142 mentions, and by
  "encoder", "estimation" or "loss" in others.
- An alias can name something else: of the 99 `passage ranking` mentions, 66 are "passage ranking
  test" and 23 "passage ranking task" — a benchmark.
- The library writes the short form and rarely the long one: `Cre` 255 mentions against 1 for
  "Cre recombinase", `pddl` 63 against 1, `RAG` 340 against 11, `DBS` 42 against 5, and
  "neurotensin receptor 1" never.

## 7 · Mentions past the bibliography cut, and what the cut removes

Over the 357 rows, presence on the whole text finds 1,103 concept–document pairs; 990 remain in
the body. **113 pairs (10%) exist only past the cut.** On the 13: `contrastive learning` 13 → 10
documents, `passage retrieval` 7 → 4, and five concepts lose one document each (`dense retrieval`,
`DBS`, `retrieval-augmented generation`, `re-ranking`, `knowledge distillation`); none becomes
single-source.

The cut (`keywords.strip_reference_section`) drops everything from the first References or
Bibliography heading past the halfway mark to the end of the document. It fires on 76 of 104
documents and removes 2,005,462 of 12,230,893 characters (16.4%).

Each heading after the reference heading opens a section. Sections were classed by the words of
their heading as *back matter* (more references, an index, acknowledgements, declarations, a
journal banner) or *content* (everything else), and the list was read through.

| | Documents | Characters | Share of the removed text | Share of the library |
|---|---:|---:|---:|---:|
| Content sections after the reference list | 39 | 555,688 | 28% | 4.5% |
| Back-matter sections after the reference list | — | 150,155 | 7% | 1.2% |

- The content sections are appendices (`A. Object Detection Baselines`), supplementary material,
  methods placed after the references, and in a few two-column PDFs a section the extractor emitted
  after the reference heading (`5 CONCLUSION AND FUTURE WORK`, `5.2 Results and Analysis`).
- 16 documents lose at least 20% of their text as content sections, 25 at least 10%; the largest
  share is 40%.
- The classification errs both ways: a 15,654-character table of contents counted as content, and
  three sections of about 8,000 characters under a journal-banner heading counted as back matter.
  Body text that follows a reference list with no heading of its own is not counted.
- Of the 113 pairs, **24 are mentions in a content section** and 89 are in reference lists or back
  matter. On the 13: 11 pairs, 1 in a content section.

The keyword extractor, the definition and usage scan and the written-form vote all read through
this cut, so none of them sees the 555,688 characters.

## 8 · Not measured

- Whether any count above separates concepts from fragments, authors or artefacts: no row was
  labelled. That needs the user's labels on a sample (ADR-053 decision 4).
- Which of the 344 the user would take on as concepts.
- Meanings that differ inside one field: they need passage context (ADR-052).
- The 28 documents with no detected reference heading: what their bibliographies add to presence.
- The same counts on another library. Every number here is from one library of 104 documents.

<!-- status: active · updated: 2026-10-09 (ADR-054, RG-032 first claim: what the user's marks on the 22 forms did to presence, links and gaps) · class: baseline -->

# Forms marked — what classifying the 22 forms did to the graph (ADR-054, RG-032)

**Ten forms marked broad took four concepts from 43 documents to 29 between them, with the 14
that left listed beside their concept, each through one named form. Five of the 30 links are
gone and 17 rest on fewer shared chunks. The gap list has the same 12 rows.** The concept whose
links lost the most kept every one of its documents.

These are counts on one library and one person's reading of 22 forms. They say what the marks
did here. They do not say the marks are right: that judgement is the user's, form by form.

## Environment / method

- **Library:** the working library, 104 documents; 13 concepts on the graph, holding 22 forms
  besides their own names. (ADR-054 counts 31 aliases: these 22 and the 9 rows that are a
  concept's own name, which carry no mark.)
- **The marks:** made by the user in the app on 2026-10-09, with the per-form counts of
  `vocabulary_shape_2026-10-01.md` §6 and, for six forms, a listing of every mention with its
  passage. One form (`contrastive`) was removed, put back and marked broad at the user's word.
  No concept was renamed.
- **Before:** a rebuild with nothing marked, as a dry run (`build_concept_skeleton(apply=False)`
  and the gap detectors) — graph `cb99a7f9f36545da`, the graph of `names_and_forms_2026-10-01.md`
  §1. **After:** the graph the user rebuilt in the app once all 22 were marked —
  `2436674f0e398cbd`. A dry run on the marked library gives that same version.
- **One reader for both.** Both were taken with the reader as it then was, which counted figure
  chunks as text. That reader was changed the same day (`prose_parents_2026-10-09.md`); section 6
  says what that does to the marked graph.
- **Which form reaches which document** was matched with the shipped matcher
  (`match_broad_presence`) on the marked library.
- **Cost:** $0. No model.

## 1 · The marks

| Concept | Broad | Exact |
|---|---|---|
| BM25 | `Okapi BM25` | |
| contrastive learning | `contrastive` | |
| cre | `cre recombinase` | |
| cross-encoder | | `cross encoder` |
| dbs | | `deep brain stimulation` |
| dense retrieval | `dense passage retrieval`, `zero-shot dense retrieval` | `dense retriever` |
| hard negatives | `negative sampling` | `hard negative`, `negative passages` |
| knowledge distillation | `distillation` | |
| ntsr1 | | `neurotensin receptor 1` |
| passage retrieval | `passage ranking` | `dense passage retriever` |
| pddl | | `planning domain definition language` |
| re-ranking | `re-rank`, `re-ranker` | `passage re-ranking`, `reranking` |
| retrieval-augmented generation | | `RAG`, `retrieval augmented generation` |

Ten broad, twelve exact, none left unmarked.

## 2 · The graph, before and after

| | Before | After |
|---|---:|---:|
| Concepts | 13 | 13 |
| Links | 30 | 25 |
| Documents with a concept | 35 | 29 |
| Concept–document pairs | 86 | 72 |
| Gap rows (deterministic) | 12 | 12 |

The stored graph also carries two model-written `suggested_concept` rows. They persist across
rebuilds, a dry run does not produce them, and they are not compared here.

## 3 · Documents per concept

| Concept | Before | After | Listed beside | Through |
|---|---:|---:|---:|---|
| contrastive learning | 13 | 10 | 3 | `contrastive` |
| knowledge distillation | 11 | 4 | 7 | `distillation` |
| re-ranking | 11 | 9 | 2 | `re-rank`, `re-ranker` |
| hard negatives | 8 | 6 | 2 | `negative sampling` |
| BM25 | 8 | 8 | | |
| dense retrieval | 7 | 7 | | |
| passage retrieval | 7 | 7 | | |
| cre | 6 | 6 | | |
| retrieval-augmented generation | 6 | 6 | | |
| dbs | 3 | 3 | | |
| cross-encoder | 2 | 2 | | |
| ntsr1 | 2 | 2 | | |
| pddl | 2 | 2 | | |

Five of the ten broad marks move no document count: `Okapi BM25`, `cre recombinase`,
`dense passage retrieval`, `zero-shot dense retrieval` and `passage ranking` occur only in
documents another form of the same concept reaches, and three of them contain the concept's name.

Each of the 14 pairs that went, with the form that alone reached the document:

| Concept | Document | Form |
|---|---|---|
| contrastive learning | *Enabling Agents to Communicate Entirely in Latent Space* | `contrastive` |
| contrastive learning | *Knowledge Distillation: A Survey* | `contrastive` |
| contrastive learning | *The Hidden Attention of Mamba Models* | `contrastive` |
| knowledge distillation | *Cajal's legacy in the digital era* | `distillation` |
| knowledge distillation | *Enabling Agents to Communicate Entirely in Latent Space* | `distillation` |
| knowledge distillation | *Identifiable attribution maps using regularized contrastive learning* | `distillation` |
| knowledge distillation | *Knowledge Graphs Meet Graph Neural Networks: A Comprehensive Survey* | `distillation` |
| knowledge distillation | *Scaling Laws for Neural Language Models* | `distillation` |
| knowledge distillation | *SuperAnimal pretrained pose estimation models for behavioral analysis* | `distillation` |
| knowledge distillation | *The Hidden Attention of Mamba Models* | `distillation` |
| re-ranking | *Neural Machine Translation by Jointly Learning to Align and Translate* | `re-rank` |
| re-ranking | *Precise Zero-Shot Dense Retrieval without Relevance Labels* | `re-ranker` |
| hard negatives | *Knowledge Graphs Meet Graph Neural Networks: A Comprehensive Survey* | `negative sampling` |
| hard negatives | *Learnable latent embeddings for joint behavioral and neural analysis* | `negative sampling` |

Six documents no longer carry any concept: *Cajal's legacy in the digital era*, *Enabling Agents
to Communicate Entirely in Latent Space*, *Neural Machine Translation by Jointly Learning to
Align and Translate*, *Scaling Laws for Neural Language Models*, *SuperAnimal pretrained pose
estimation models* and *The Hidden Attention of Mamba Models*.

## 4 · Links

Five links are gone, 17 rest on fewer shared chunks, 8 are unchanged, none is new.

| Link | Shared chunks before | After |
|---|---:|---:|
| contrastive learning — hard negatives | 6 | gone |
| contrastive learning — knowledge distillation | 5 | gone |
| contrastive learning — re-ranking | 3 | gone |
| cross-encoder — passage retrieval | 7 | gone |
| passage retrieval — knowledge distillation | 9 | gone |
| passage retrieval — BM25 | 59 | 14 |
| passage retrieval — re-ranking | 28 | 2 |
| dense retrieval — passage retrieval | 30 | 9 |
| contrastive learning — dense retrieval | 17 | 5 |
| dense retrieval — hard negatives | 13 | 6 |
| dense retrieval — knowledge distillation | 16 | 10 |
| contrastive learning — passage retrieval | 11 | 5 |
| hard negatives — BM25 | 16 | 11 |
| contrastive learning — BM25 | 6 | 2 |
| passage retrieval — hard negatives | 7 | 3 |
| cross-encoder — knowledge distillation | 5 | 3 |
| re-ranking — BM25 | 53 | 51 |
| dense retrieval — re-ranking | 21 | 20 |
| dense retrieval — retrieval-augmented generation | 5 | 4 |
| knowledge distillation — BM25 | 3 | 2 |
| re-ranking — knowledge distillation | 4 | 3 |
| retrieval-augmented generation — re-ranking | 3 | 2 |

Links per concept: `contrastive learning` 7 → 4, `passage retrieval` 8 → 6,
`knowledge distillation` 6 → 4, `cross-encoder` 5 → 4, `hard negatives` 4 → 3,
`re-ranking` 7 → 6.

**A mark that moves no document count can move the links most.** `passage retrieval` keeps its
7 documents, yet its links lose 118 shared chunks between them, the most of any concept (`BM25`
57, `dense retrieval` 48), and two of them go. Its one broad form is concentrated in one
document: `passage ranking` is written in 95 chunks of prose, 90 of them in *Pretrained
Transformers for Text Ranking: BERT and Beyond*, where the concept's name is written in 18 (39
over the library; counted on the prose reader of section 6). Marked broad, those chunks no longer
count for the concept. The link to `cross-encoder` (7 shared chunks, gone) shows it alone:
`passage ranking` is the only broad form on either end. On the other links a broad form of the
second concept may share the loss; the two were not separated.

A link needs two shared chunks. Six links now rest on exactly two: four that the marks brought
there (`passage retrieval` — `re-ranking`, `contrastive learning` — `BM25`,
`knowledge distillation` — `BM25`, `retrieval-augmented generation` — `re-ranking`) and two that
were there before (`contrastive learning` — `retrieval-augmented generation`,
`passage retrieval` — `retrieval-augmented generation`).

## 5 · Gap rows

The same 12 rows, on the same concepts: `isolated` on `dbs` and `pddl`, `under_connected` on
`cre` and `ntsr1`, `unsourced_claim` on eight concepts. No concept became `single_source`.

Three `unsourced_claim` rows rest on fewer answer claims, because an answer sentence that uses
only a broad form is no longer attributed to the concept: `dense retrieval` 21 → 14,
`re-ranking` 10 → 8, `hard negatives` 2 → 1. The other nine rows rest on what they did.

## 6 · The same marks on the reader that skips figure chunks

Both graphs above counted figure chunks as text. On the reader that reads prose only, the marked
graph is `58136820706f4cab`: the same documents per concept, the same 25 links and the same 12
gap rows, with three links on one or two fewer shared chunks (`re-ranking` — `BM25` 51 → 49,
`passage retrieval` — `BM25` 14 → 13, `retrieval-augmented generation` — `BM25` 5 → 4). The user
rebuilt on that reader the same day and the stored graph has that version. The before/after of
this file was not redone on it; section 2 of `prose_parents_2026-10-09.md` is the evidence that
the difference is those three counts.

## 7 · Not measured

- Whether a mark is right. No second reader classified the forms, and the user said of one of
  them that it "does not seem specific enough" to judge without its context: exact and broad are
  per form, and that is a judgement per mention (ADR-052's ground).
- Whether matching a concept's exact forms improves its definition candidates (RG-032, second
  claim): candidates are still matched on the name alone.
- Each mark on its own. The marks were applied together and compared once, so where a link lost
  chunks and both its concepts have a broad form, the loss is not split between them.
- What the marks do to answers. Presence feeds the graph and the gap list, not retrieval.
- The same marks on another library.

## Reproduce

Clear every mark on a copy of the library, run `build_concept_skeleton(apply=False)` and the gap
detectors, and keep each node's `doc_ids`, each edge's `n_cooccurrence_chunks` and the gap rows;
set the marks of section 1 and do the same. The probes used here are local working files and are
not in the repository.

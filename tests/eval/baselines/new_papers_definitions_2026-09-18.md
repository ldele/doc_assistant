<!-- status: active · updated: 2026-09-18 · class: baseline -->

# Six papers added for definitions — what they changed (ADR-052, ROADMAP 93)

ADR-052 makes a concept one meaning and its definition curated data, and on 2026-09-18 **2 of 357
concepts had a definition**; a strict scan found a defining sentence in the library for 2 of 19
priority concepts (the graph's 13 plus six shared words). So six open-access papers chosen to define
those concepts were added through the app's own path — `POST /api/documents/inspect` → `add`
(copy) → `POST /api/ingest` — then enriched with the $0 runners (`enrich_metadata`,
`extract_keywords`, `extract_citations`, `compute_doc_vectors --force`, `build_concept_skeleton`,
`build_gaps`). Five were downloaded directly; the sixth, Gerfen et al. 2013 (Cre driver lines),
was saved by the user from a browser — PubMed Central and the publisher both answer an automated
client with a bot check — and added the same way.

**Cost:** $0 — local extraction, embeddings and heuristics; no model call. **Reversible:** the five
papers can be removed with the app's undo-add; `data/library.db` and `skeleton.json` were copied first.

## The papers

| paper | pages · chunks | references parsed · in library |
|---|---|---|
| Lin, Nogueira & Yates, *Pretrained Transformers for Text Ranking* (2021) | 204 · 1,286 | 542 · 1 (DPR) |
| Gao et al., *Retrieval-Augmented Generation for LLMs: A Survey* | 21 · 159 | 182 · 1 (HyDE) |
| Gou et al., *Knowledge Distillation: A Survey* | 36 · 265 | 352 · 1 (ResNet) |
| Fox & Long, *PDDL2.1* (2003) | 64 · 298 | 60 · 0 |
| Cohan et al., *SPECTER* | 13 · 89 | 56 · 0 |
| Gerfen, Paletzki & Heintz, *GENSAT BAC Cre-recombinase driver lines* (2013) | 30 · 125 | 19 · 0 |

## Before → after

| | before | after |
|---|---|---|
| documents · parent chunks | 98 · 7,863 | 104 · 8,861 |
| documents the graph covers · graph edges | 30 · 20 | 36 · 30 |
| deterministic gap rows | 17 | 12 — **closed:** `single_source` cross-encoder, pddl and ntsr1, `isolated` cross-encoder, `under_connected` knowledge distillation; **opened:** none. No graph concept is single-source any more |
| defining sentences, strict patterns, all concepts | 58 | 68 |
| priority concepts with a strict-pattern definition | 2 | 5 (dense retrieval, knowledge distillation, specter added) |

The five were measured first (13 gap rows; `ntsr1` still single-source), the Cre paper after.

## What the measurement says about finding definitions

**Fixed patterns miss how surveys introduce a term.** The ranking book mentions BM25 285 times and
explains it, yet "X is a …" / "X refers to …" / "called X" found no BM25, cross-encoder, RAG or PDDL
definition. Taking **each concept's first two sentences in each new paper, in reading order**, found
an introducing passage for all of them — e.g. *"This general style of organizing task inputs …
is called a 'cross-encoder'"* (the pattern missed it on the quotation marks); *"Retrieval-Augmented
Generation (RAG) has emerged as a promising solution by incorporating knowledge from external
databases"*; *"In 1998 Drew McDermott released a Planning Domain Description Language, pddl …, which
has since become a community standard for the representation and exchange of planning domain
models"*. First introduction in a survey is the better suggestion source for row 93.

**The new text exposes meaning problems in today's aliases** (ADR-052's curation, not bugs in
matching): `contrastive learning` carries the alias `contrastive`, so the ranking book's
"contrastive and ablation experiments" counted as the concept; `hard negatives` carries
`negative sampling`, a broader technique; `passage retrieval` carries `dense passage retriever`, one
model; and **`hard negatives` is defined two ways** — SPECTER: papers cited by a cited paper but not by
the query paper; the ranking book: non-relevant texts the encoder itself finds similar.
`specter` now has its model's own definition ("We propose SPECTER, a new method to generate
document-level embedding of scientific documents …") beside the benchmark paper and the
political-philosophy book. `virus` appears once in the ranking book, about the pandemic — the same
meaning in another field. **`ntsr1` in this library is a mouse line**, not the receptor: the Cre paper
introduces it as "Ntsr1_GN220 for layer 6 corticothalamic neurons", a Cre driver line. And the same
paper uses `viral` in the lab sense — "engineered viral vector constructs" — the within-field meaning a
document-level split cannot see.

## Found on the way

- `compute_doc_vectors --apply --force` failed on a foreign key: one chunk in the vector store
  belonged to a synthetic test document removed from the library in an earlier session, and the
  whole edge set rolled back. The loader now skips and logs such chunks (DEVLOG 2026-09-18 (1)); the
  stale chunk itself is still in the store.
- `enrich_metadata` found no year for SPECTER, the RAG survey or the distillation survey, and took
  the PDDL2.1 title in lower case ("pddl2.1 : An Extension to pddl …") — the extracted text
  carries the name in lower case throughout.

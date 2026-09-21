<!-- status: active · updated: 2026-09-21 (first measurement — ROADMAP 93a) · class: baseline -->

# Where a concept's definition can come from — what the library holds (ROADMAP 93, ADR-053)

ADR-052 made definitions curated data; there were almost none (2 of 357 concepts). ADR-053 made a
definition something *chosen from candidates that keep their source*. This records what the library
itself can offer as candidates, what the reference lists alone can say, and how the deterministic
extractor (`knowledge/definitions.py`) grades what it finds.

Counts are measured. **The "first reading" column is the agent's**, not the user's choices — the
user's choices, once made in the panel, are the real test of the grade.

## Environment / corpus

- **Data home:** `data/library.db` on this box, 104 documents, 357 text-bearing concepts
  (`kind="concept"`), 13 on the graph; 5,639 reference entries (5,266 with a title) across 83
  documents.
- **Method:** labels only (aliases are different phrases with their own meanings —
  `isa_head_suffix_2026-09-20.md`); every parent chunk in reading order, the bibliography cut off
  with `keywords.strip_reference_section`; sentences kept verbatim.
- **Cost:** $0 — no model, no network. Nothing written to the library: the extractor ran as a dry
  run; the app was exercised against a throwaway copy of the data directory.

## 1 · What the text holds

| | |
|---|---|
| labels that occur in the text | 335 of 357 |
| … in two or more documents | 179 |
| concepts with a sentence shaped like a definition (first scan, loose patterns) | 76 |
| **concepts the extractor finds a candidate for** | **327** (688 candidates) |
| … whose best candidate is `strong` | **37** |
| … whose best is `some` | 290 |
| … with nothing | 30 (their label never occurs in prose — fragments, mostly) |

Candidates by form: 15 coined ("we call our model SPECTER") · 6 named ("… is called a
'cross-encoder'") · 89 definition-shaped · 578 first mentions.

## 2 · The 19 priority concepts

The graph's 13 plus the six shared words ADR-052 flagged. Top candidate as the extractor ranks it;
first reading by hand from the 2026-09-21 scan (before the extractor existed): **clean** = a
passage defines it; **partial** = a person could write one from what is there; **none**.

| concept | candidates | top candidate — grade · form | first reading |
|---|---|---|---|
| contrastive learning | 5 | strong · defining — "Contrastive learning is a technique that leverages contrasting samples…" | clean |
| cross-encoder | 2 | strong · named — "… is called a 'cross-encoder'" | clean |
| dbs | 3 | strong · defining — "DBS is a neurosurgical therapy in which implanted electrodes…" | clean |
| hard negatives | 3 | strong · coined — SPECTER's "We denote as 'hard negatives' …" (the citation-graph sense) | clean |
| knowledge distillation | 5 | strong · defining — "… is an excellent technique for model compression" (a claim; the "refers to … student … teacher" sentence is also strong) | clean |
| ntsr1 | 3 | strong · coined — the Ntsr1-Cre mouse line, layer 6 | clean |
| passage retrieval | 3 | strong · named — "… commonly referred to as passage retrieval" | clean |
| pddl | 5 | strong · defining — "pddl is an action-centred language, inspired by … strips" | clean |
| retrieval-augmented generation | 4 | strong · coined — "… which we refer to as retrieval-augmented generation (RAG)" | clean |
| specter | 4 | strong · coined — "We call our model SPECTER …" | clean |
| dense retrieval | 5 | strong · named — "… learned dense representations (also called dense retrieval)" | partial |
| BM25 | 5 | some · defining-shaped, not about BM25 | partial |
| viral | 3 | some · first mention | partial — the candidates show two senses: a lab viral vector and viral encephalitis |
| virus | 3 | some · first mention | partial |
| beta | 4 | some | none — "beta oscillations" beside "beta secretase" |
| din | 2 | some · first mention | none — the two candidates are the two "meanings": tadpole dIN interneurons, and "lack of vitamin Din which…", an extraction join of "vitamin D in" |
| cre | 3 | some · first mention | none |
| abstractions | 4 | some | none |
| re-ranking | 3 | some · first mention | none |

**The strong grade agrees with the first reading on 18 of 19.** The one move is `dense retrieval`,
read as partial by hand; the naming sentence the extractor found is a fair definition. What the
grade cannot see: a definition-shaped *claim* ("knowledge distillation is an excellent
technique…") grades as strong as the real definition beside it. That is why a grade is shown with
its reasons and never chooses.

## 3 · What the references alone say

| | |
|---|---|
| reference entries with a title | 5,266 across 83 documents |
| entries that resolve to a library document | **44 of 5,639** |
| priority concepts named by at least one cited title | 13 of 19 |

Reference titles do not define — but they carry what the passages lack. `dbs` → "Lead-DBS: A
toolbox for **deep brain stimulation** electrode localizations" (the abbreviation expanded);
`beta` → "**Beta-band oscillations** — signalling the status quo?" (the meaning, where the passages
were ambiguous); `contrastive learning` → "A simple framework for contrastive learning of visual
representations" (the canonical paper, not in the library). **The link to the library is weak:**
the RAG paper *is* in the library and its citations did not resolve, so "not in your library"
cannot be read off the citation link alone (ADR-053's papers-to-add step checks titles first).

## 4 · The one-concept path

The panel's "Look in my library" first read the whole vector store: **12.1 s** for one concept at
104 documents, linear in the corpus — minutes at the 10,000-document contract. It now asks the
on-disk keyword index which documents mention the phrase, then reads only those.

| | |
|---|---|
| one concept, in the app | 12.1 s → **272 ms** |
| slowest index lookup over all 357 concepts | 54 ms |
| candidates identical to the full read | **357 of 357** — after two fixes: a prefix match (the index keeps `actor-critic` whole) and a deterministic tie-break between documents with equal mentions |

## What this does not claim

- **Not precision.** The 19 are a by-hand first reading by the agent; the 688 candidates across
  the vocabulary are unread. The grade is a structural rule, spelled out as reasons.
- **Not that the text is enough.** 290 concepts have only `some`-graded candidates — mostly first
  mentions, which introduce a term less often than they use it. Model candidates (93b) and papers
  to add (93d) are the answer ADR-053 plans for that — with expert vocabularies (93c) for the
  established terms — not a looser pattern here.

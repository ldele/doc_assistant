<!-- status: active · updated: 2026-09-21 (first probe — ROADMAP 93, the user's question) · class: baseline -->

# Do expert vocabularies already define the library's concepts? (ROADMAP 93, ADR-053)

The user's question (2026-09-21): for a given field, experts have often already listed and defined
the terms — much as a taxonomy lists the fields — with newer concepts still fuzzy. Can the app lean
on that, while staying self-reliant? This records a first probe of the 19 priority concepts
against four public vocabularies, and what each source offers.

Counts are measured; **"expert definition / gloss / nothing" is the agent's reading of the first
hit**, not the user's.

## Method

- **Probe:** `wbsearchentities` (Wikidata), the MeSH lookup API (NLM), the OLS4 search API (EMBL-EBI,
  280+ ontologies, exact match), and the archived Papers with Code *methods* table (Hugging Face
  datasets-server). Only the 19 concept labels were sent. $0.
- **One honest caveat on inputs:** four labels were expanded by hand before the lookup, using what
  the library itself says they stand for — `dbs` → *deep brain stimulation*, `din` → *descending
  interneuron* (the tadpole paper), `beta` → *beta rhythm*, `ntsr1` → *Ntsr1-Cre*. With the raw
  labels coverage is lower; a matcher has to find these expansions itself (below).
- **Unanswered is not absent:** the Papers with Code search failed on 5 lookups (server errors /
  timeouts) and MeSH dropped 2 — those cells are unknown.

## Result — the 19

| | concepts | which |
|---|---|---|
| **An expert definition exists** (curated, versioned, citable) | **8** | beta (MeSH *Beta Rhythm*) · dbs (MeSH *Deep Brain Stimulation*) · din (XAO *descending interneuron* — the Xenopus anatomy ontology, exactly the library's sense) · ntsr1 (MGI:3836636, the GN220 line itself) · cre (FlyBase CV; Wikidata *protein found in phage P1*) · viral (NCIt *Viral Vector* — one of its senses) · virus (NCIt *Virus*) · contrastive learning (AIO) |
| **Only a one-line gloss** (Wikidata description) | **6** | BM25 · knowledge distillation · hard negatives (*hard negative mining*) · pddl · re-ranking · retrieval-augmented generation |
| **Nothing usable** | **5** | cross-encoder · dense retrieval (only the DPR *paper*) · passage retrieval (only a paper) · specter (a person of that name) · abstractions (the *artistic* concept) |

**The user's intuition holds.** The established biomedical and neuroscience terms are defined by
expert vocabularies; the newer retrieval and ML terms are at best a one-line gloss, and the three
youngest (cross-encoder, dense retrieval, passage retrieval as the library uses it) have nothing. A
model's own name (`specter`) and a generic word (`abstractions`) match the wrong thing.

## What each source offers

| Source | Gives | Covers | Licence | Local copy |
|---|---|---|---|---|
| MeSH (NLM) | scope notes (definitions), synonyms, tree | biomedicine | free; acknowledge NLM, no endorsement implied | yearly XML / RDF |
| OBO ontologies via OLS (NCIt, UBERON, CL, GO, XAO, AIO…) | definitions with citations, synonyms, hierarchy | biology, medicine; AI (AIO) | mostly CC BY 4.0, per ontology | each OWL/OBO file; OLS API for lookup |
| AIO — Artificial Intelligence Ontology | AI/ML concepts with Aristotelian definitions | AI / ML | CC BY 4.0 | OWL; built with LLM assistance, so a curated-but-machine-drafted source |
| MGI (Jackson Laboratory) | curated records of mouse genes, alleles, transgenic lines | mouse genetics | free with citation | report files |
| Wikidata | label, one-line description, aliases — and the IDs that link to all of the above | everything | CC0 | dumps (large); subsets by query |
| Wikipedia | the lead paragraph: a working definition | everything | CC BY-SA 4.0 | dumps |
| Wiktionary | senses per word, with domain labels ("medicine", "internet") | general words | CC BY-SA | dumps — useful for ADR-052's shared words |
| Papers with Code *methods* (archive) | 8.7k ML methods: description, introducing paper, year | ML | CC BY-SA 4.0, frozen 2025-07-28 | one parquet table |
| Computer Science Ontology | 14k CS topics and their relations, no definitions | computer science | CC BY 4.0 | OWL / CSV |
| OpenAlex | topics, keywords, abstracts of cited works | all scholarship | CC0 | API; snapshot is very large |
| UMLS, SNOMED CT | the richest biomedical sources | medicine | account / licence required | not self-reliant |

## What it implies for ADR-053 (proposed to the user, not decided)

- **A fourth source of candidates: `reference`** — an expert vocabulary's definition, quoted verbatim,
  with the vocabulary, its identifier (MeSH D046690, MGI:3836636…), version and licence. Chosen by the
  user like any other; never chosen for them.
- **Two layers, not one answer.** *What the field says* (reference) and *how your library uses it*
  (passages) can both be right and still differ — SPECTER's "hard negatives" is a narrower recipe than
  the field's. Showing both is the point; disagreement is evidence for ADR-052's sense splits.
- **Self-reliant by vendoring.** Download the vocabularies that match the library's fields once, index
  them locally beside the keyword index, record their versions, refresh on request. An online lookup
  is an explicit, cached fallback — not the path every concept takes.
- **The taxonomy picks the vocabularies.** A document's field (ANZSRC, ADR-028) decides where to
  look: neuroscience → MeSH, UBERON, XAO, MGI; computing → the methods archive, AIO, CSO.
- **Matching is the hard part, and the library is its judge.** Expand abbreviations from the library
  itself ("deep brain stimulation (DBS)"); when several entries match, rank them by closeness to the
  concept's own passages and show why; a label that matches a person or an art term is shown as the
  miss it is.

## Sources

[Papers with Code shutdown and archive](https://www.coursera.org/articles/papers-with-code) ·
[pwc-archive/methods](https://huggingface.co/datasets/pwc-archive/methods) ·
[OLS4](https://www.ebi.ac.uk/ols4/) and [its paper](https://academic.oup.com/bioinformatics/article/41/5/btaf279/8125017) ·
[MeSH terms and conditions](https://www.nlm.nih.gov/databases/download/terms_and_conditions_mesh.html) ·
[Computer Science Ontology](https://skm.kmi.open.ac.uk/cso/) ·
[Artificial Intelligence Ontology](https://bioregistry.io/registry/aio) ·
[MGI:3836636](https://www.informatics.jax.org/allele/MGI:3836636) ·
[OpenAlex topics](https://help.openalex.org/data/topics/)

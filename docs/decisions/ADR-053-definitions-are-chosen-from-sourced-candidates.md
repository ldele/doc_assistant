<!-- status: active · updated: 2026-09-21 · class: append-only -->

# ADR-053 — A definition is chosen from candidates, and every candidate keeps its source

- **Status:** proposed — the direction is the user's (2026-09-21): definitions can come from the
  text, from a model, from what the references say, or from the user; *"each of these options
  should not overwrite the other"*, the user chooses case by case, each source is judged for
  reliability "as usual", and thin evidence should lead to suggesting more papers. The shape below
  (one table of candidates, the reliability signals, the slice order) is Claude's proposal for the
  user to accept. Builds [ADR-052](ADR-052-a-concept-is-a-meaning.md)'s "meanings are curated data".
  **Decided by the user, 2026-09-21:** expert vocabularies (MeSH, the OBO ontologies, the mouse-line
  registry, Wikidata…) are a fourth source of candidates — see *Decision* and slice 93c.
- **Date:** 2026-09-21
- **Deciders:** user + Claude Code
- **In one sentence:** a concept used to have one definition box that the last writer overwrote;
  it now has a list of possible definitions, each showing where it came from, and you tick one —
  nothing in the list is ever replaced or deleted.
- **To decide (the proposed part):** (1) all versions kept side by side and chosen by you, rather
  than one box with a history — *recommended*; (2) how trustworthy an option looks is shown as
  its reasons plus a strong / some / thin label worked out from them, and no model scores its own
  text — *recommended*; (3) the order of the next slices: model suggestions (93b), expert
  vocabularies (93c — added by the user), references and papers to add (93d), splitting shared
  words (93e).

## In practice — one concept, step by step

Real text from the library, 2026-09-21. The concept is **hard negatives**, which has no definition
yet — and turns out to mean two things in two of the papers.

1. **Open it in the Graph tab.** The panel says *No definition chosen yet* and offers **Look in my
   library**.
2. **Look in my library** (a third of a second, no model, nothing chosen). Three options appear,
   each quoted exactly, with where it is and why it is rated as it is:

   | | Option | Rated | Why |
   |---|---|---|---|
   | A | "We denote as 'hard negatives' the papers that are not cited by the query paper, but *are* cited by a paper cited by the query paper…" — *SPECTER*, open at its page | **strong** | the author names the term here · from the document that mentions it most (6 times) |
   | B | "…new training triples are created by using hard negatives retrieved from these representations (replacing the BM25-based negatives)." — *Pretrained Transformers for Text Ranking* | thin | the first sentence that uses it in this document · mentions it 2 times |
   | C | "…extending the idea of leveraging hard negatives, Xiong et al. (2020a) use the retrieval model trained in the previous iteration to discover new negatives…" — *Dense Passage Retrieval* | thin | the first sentence that uses it · mentions it once |

3. **Use this** on A. A is now the concept's definition — the text the merge comparison reads
   (ROADMAP 53) and the glossary (`seed_concepts --list`) prints.
4. Reading B and C, you see A is SPECTER's own recipe (negatives from the citation graph) while the
   other two mean negatives mined from a model's own top results. **Write your own:** *"Training
   examples that look relevant to a query but are not; built from the citation graph in SPECTER,
   from a retriever's top results in dense retrieval."* Yours becomes the definition. **A is still in
   the list** as an option.
5. **Undo** → A is the definition again; your text is still there. **Undo** again → nothing chosen;
   all four options are still there. Dismissing an option hides it; **Restore** brings it back.

What is stored after step 4 — nothing was overwritten, and the history says how you got there:

| Option | Source | Status |
|---|---|---|
| A — SPECTER's sentence | passage, *SPECTER* (opens at its page) | suggested |
| B — ranking book's sentence | passage | suggested |
| C — DPR's sentence | passage | suggested |
| your text | you | **chosen** |

History: *chose A (before: none) → chose yours (before: A).* Each undo walks one step back.

**Before this ADR** the same concept had one box: `seed_concepts --define` wrote it, a second write
replaced the first, and nothing said where the words came from.

**Later slices add options to the same list, never choices:** 93b a model's text (*"suggested by a
model, from passages A–C"* — or *"from reference titles only"*, to see what the bibliographies
alone can say); 93c an expert vocabulary's definition (next example); 93d for a concept with little
to go on, the papers its references cite that the library does not hold; 93e the question this
example raises — is *hard negatives* one concept or two (ADR-052)? The list above is the evidence
either way.

### Two layers — what the field says, and how your library uses it (93c)

For **dbs** the panel will show two groups. The expert vocabulary is quoted exactly, with its
identifier, from a local copy:

> **What the field says** — *MeSH D046690, Deep Brain Stimulation:* "Therapy for MOVEMENT
> DISORDERS, especially PARKINSON DISEASE, that applies electricity via stereotactic implantation
> of ELECTRODES in specific areas of the BRAIN such as the THALAMUS. The electrodes are attached to
> a neurostimulator placed subcutaneously."
>
> **How your library uses it** — *strong, from the hypothalamic-stimulation paper:* "DBS is a
> neurosurgical therapy in which implanted electrodes are used to adjustably deliver electrical
> current to specific brain targets."

They agree on what DBS is, and differ on emphasis: the field names its usual use (Parkinson's,
the thalamus); the library's paper uses it for aggression, in the hypothalamus. Either can be chosen,
or your own words written from both — and where the two disagree on *what* the term is, that is
evidence the word carries two meanings (93e). For **cross-encoder** the first group will be empty:
no vocabulary probed defines it yet, so the library's passages and your words are all there is.

## Context

ADR-052 made a definition the thing that says what a concept means, and said suggestions come from
the corpus or a model and stay marked until accepted. It did not say what happens when there are
several. Today there is one slot, `Concept.definition`, written by one CLI (`seed_concepts
--define`), last write wins; **2 of 357 concepts** use it.

Measured on the live library (104 documents, label-only matching; scan in DEVLOG 2026-09-21):

- **The text has material, unevenly.** 335 of 357 labels occur in the text, 179 in two or more
  documents; 76 concepts have at least one sentence shaped like a definition ("X is a…", "X refers
  to…", "known as X"). About half of those shapes are not definitions — both of BM25's are about
  score interpolation — while the first sentence that mentions a concept, in the document that uses
  it most, often *is* one where no pattern fires (cross-encoder's "is called a 'cross-encoder'").
- **Read by hand, the 19 priority concepts split three ways:** 10 have a passage that defines them
  cleanly (knowledge distillation, contrastive learning, DBS, PDDL, RAG, cross-encoder, passage
  retrieval, SPECTER, hard negatives, ntsr1); 4 have one a person could write from (BM25, dense
  retrieval, viral, virus); 5 have nothing definitional (beta, din, cre, abstractions, re-ranking) —
  and for `beta` and `din` the candidates themselves show the second meaning ("beta secretase",
  "vitamin Din").
- **References alone say something, but not a definition.** 5,266 reference entries with a title,
  across 83 documents. For 13 of the 19, at least one cited title names the concept, and the titles
  carry what the passages lack: `dbs` → "deep brain stimulation" (the abbreviation expanded),
  `beta` → "beta-band oscillations" (the meaning). They also name the canonical papers the library
  does not hold — but only **44 of 5,639** entries resolve to a library document, and the RAG paper
  *is* in the library while its citations did not resolve, so "not in your library" cannot be
  trusted from that link alone.
- **Experts have already defined the established terms** (probe of four public vocabularies,
  `tests/eval/baselines/reference_vocabularies_2026-09-21.md`). Of the 19, **8** have a curated,
  citable definition — `beta` (MeSH *Beta Rhythm*), `dbs` (MeSH), `din` (the Xenopus anatomy
  ontology's *descending interneuron*, the library's exact sense), `ntsr1` (the mouse-line registry's
  GN220 record), `cre`, `viral` (as *viral vector*), `virus`, `contrastive learning`; **6** only a
  one-line gloss (BM25, knowledge distillation, hard negatives, PDDL, re-ranking, RAG); **5** nothing
  usable — the youngest retrieval terms (cross-encoder, dense retrieval, passage retrieval), a
  model's own name (`specter` matched a person) and a generic word (`abstractions` matched the art
  term). Four labels were expanded by hand before the lookup (`dbs` → deep brain stimulation…).

## Options

1. **Keep one slot; each source writes it.** No schema change. But a model run would replace the
   user's text, a re-extraction would replace an accepted passage, and nothing could be compared —
   the overwrite the user ruled out.
2. **A table of candidates.** Every definition, whatever produced it, is a row that keeps its text,
   its source (the passage with document and page; the model with the inputs it was given; the
   user), the evidence about its reliability, and a status. The user chooses one; the others stay.
3. **A version history of the one definition.** Keeps the past, but the options still arrive one at
   a time and replace each other; it answers "what did it say before", not "which of these is right".
4. **A model defines every concept at ingest.** Needs no review up front — and is the unbounded,
   evidence-free judging this project has measured as invalid (KI-19, KI-33).

## Decision

**Option 2.** A concept's definition is **chosen from candidates**, and no source can overwrite
another. The deciding reason: the user wants to see the options side by side and pick case by
case, and that needs the options to exist at the same time, each with where it came from.

- **One table, `concept_definitions`.** A row per candidate: the concept, the text, the `source`
  (below), a provenance record (document and chunk key for a passage — the page is found when it is
  opened; model, prompt version and the ids of every input for a model; vocabulary, identifier,
  version and licence for a reference; nothing more for the user), the reliability evidence, and
  a `status` — `suggested`, `chosen` or `dismissed`. Adding a candidate never touches another row;
  re-running a source adds only what is new (keyed on concept + source + provenance).
- **Choosing is the only write to the concept, and it is undoable.** At most one candidate is
  `chosen`. Choosing another puts the previous one back to `suggested` — never deletes it — and
  `Concept.definition` is set to the chosen text so every existing reader (the merge text, the
  semantic layer) keeps working unchanged. Each choice is recorded, so going back is picking the
  previous one again. The two definitions that exist today become `user` candidates, chosen.
- **The sources**, in the order they are built:
  - `passage` — a sentence from the library, **verbatim** (ADR-043), with its document, so one click
    opens it at its page. Found by definition-shaped sentences and by the first mention in the
    documents that use the concept most. $0.
  - `user` — written by the user. Always available, never suggested over.
  - `model` — a local model writes a definition **from inputs it is handed**, never from its own
    knowledge alone, and the inputs are recorded: either passages (a definition grounded in the
    library) or **reference titles only** — the user's experiment, how much the references alone
    can say. Labelled by what it was given, so the two can be compared side by side.
  - `reference` — **an expert vocabulary's definition, quoted verbatim** (the user's addition,
    2026-09-21): MeSH scope notes, the OBO ontologies' definitions (NCIt, UBERON, the Xenopus
    anatomy ontology, the AI ontology…), the mouse-line registry, Wikidata's gloss where nothing
    better exists, the archived ML-methods table for machine-learning terms. Stored with the
    vocabulary, the entry's identifier, the vocabulary's version and its licence.
- **Expert vocabularies, self-reliantly** (the rules for `reference`):
  - **Local copies, not live queries.** The vocabularies that fit the library's fields are
    downloaded once, indexed beside the keyword index with their versions recorded, and refreshed
    only when the user asks. An online lookup (the ontology-search service, Wikidata) is an explicit,
    cached fallback, never the path every concept takes. No source that needs an account or a
    licence agreement (UMLS, SNOMED CT) — the app must stay usable offline and unregistered.
  - **The taxonomy picks where to look.** A document's field (ANZSRC, ADR-028) decides which
    vocabularies are consulted for the concepts it holds — neuroscience: MeSH, UBERON, the anatomy
    ontologies, the mouse-line registry; computing: the ML-methods table, the AI ontology, CSO for
    structure.
  - **The library judges the match.** Abbreviations are expanded from the library's own text
    ("deep brain stimulation (DBS)"); when several entries match, they are ranked by closeness to
    the concept's passages; a match on a person or an art term shows as the miss it is. The match
    is part of the evidence: exact label, synonym, an expansion the library supplies, or closeness.
  - **Two layers, not a ranking.** The panel shows *what the field says* beside *how your library
    uses it*. A curated definition is not ranked above a passage: the field's meaning and the
    library's usage can both be right and still differ, and the choice between them is the user's.
  - **Attribution and licences are kept.** Each vocabulary is acknowledged where the app credits
    its data (as ANZSRC already is, CC BY), per its terms — MeSH's "acknowledge NLM, no endorsement
    implied"; the share-alike sources (Wikipedia, Wiktionary, the ML-methods table) keep their
    licence on every entry, which matters only if the entries are ever redistributed.
- **Reliability is evidence shown, not a score a model gives itself** — the rule every other signal
  in this layer follows (`docs/knowledge-layer.md` §6). For a passage: its form (a definition shape,
  or the author coining the term — "we call", "we refer to as"), where it sits (the document that
  uses the concept most), how many documents supply one. For a model candidate: which inputs, from
  how many documents, and how much of what it says can be found in them. For a reference: which
  vocabulary and version, and how the entry was matched. Shown as the reasons plus a coarse grade
  derived from them — `strong`, `some`, `thin`.
- **Thin evidence is said, and turns into a suggestion to read more.** When a concept has no
  definition-shaped passage and few documents, the panel says so and lists what its references name
  that the library does not hold — checked by title against the library first, because the
  citation link alone misses papers the library has. The app suggests; it never fetches (the
  acquisition half is KL2 / ADR-032).
- **Where:** the concept's panel in the Graph tab (the user's choice, 2026-09-21), reachable for
  every concept — including the shared words that are not on the graph — through a search over the
  whole vocabulary.

**Build order** (one session each): **93a** the table, passage candidates and their evidence,
user-written candidates, choose / dismiss / undo, the API and the panel section, the vocabulary
search. **93b** model candidates from passages and from references alone, with the grounding check,
measured on the 19. **93c** expert vocabularies: the local copies for the library's fields, the
matcher (abbreviations from the library, ranking by closeness), `reference` candidates and the
two-layer panel, measured on the 19 *without* the hand expansions the probe used. **93d** what the
references say and the papers to add. **93e** the shared-word split review (row 92).

**What would reverse it:** if the user ends up choosing the same source every time, the table
collapses to that source plus an override — a simplification, not a reversal.

## Consequences

**Easier.** Every source can be added without a migration of meaning: a new extractor or model is
new rows. The review ADR-052 needs for a shared word has its material — two candidates that
describe different things are the evidence for a split. Merges can compare chosen definitions.

An established term gets an expert's definition without anyone writing one, and a passage that
agrees with it gains support from outside the library; one that disagrees is a sense to look at.

**Harder.** A second place holds definition text; `Concept.definition` must only ever be written by
the choose/undo path, or the two drift (a test pins it). Candidates for 357 concepts is a long list,
so the panel shows the few best and the priority concepts come first. The vocabularies are data
the app now carries: their size on disk, their versions and refreshes, their licences and
attribution, and a matcher that will sometimes pick the wrong entry — which is why the match is
shown and nothing is chosen for the user.

**Must revisit.** The coarse grade's thresholds, once the user has chosen enough definitions to
compare them against (the labelling page, 2026-09-21, is the first yardstick). Whether `references`
becomes a candidate source in its own right, if 93b shows reference titles alone can define. Which
vocabularies each field needs, once 93c has measured what the local copies cover; whether the
online fallback is needed at all.

## Confidence

- ✓ Definition-shaped sentences exist for 76 concepts; 10 of the 19 priority concepts have a clean
  one; reference titles name 13 of the 19 — this scan, 2026-09-21.
- ⚠ **Half of the pattern hits are not definitions** — measured on the 19 by hand, not on the 76.
  93a shows the reasons, not a verdict, for that reason.
- ⚠ **Whether a model grounded in passages beats the passages themselves** — 93b measures it.
- ✓ Expert vocabularies define 8 of the 19 priority concepts and gloss 6 more —
  `tests/eval/baselines/reference_vocabularies_2026-09-21.md`.
- ⚠ **That coverage used four hand-expanded labels.** How much a matcher finds on its own, and how
  often it picks the wrong entry, is unmeasured — 93c measures both.

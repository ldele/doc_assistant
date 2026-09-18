<!-- status: active · updated: 2026-09-18 · class: append-only -->

# ADR-052 — A concept is one meaning, not one label: meanings are curated, and a shared word needs a scope

- **Status:** proposed — the principle is the user's decision (2026-09-18: "a different meaning is a
  different concept, which means we need to curate meanings/definitions"); the mechanism below is
  proposed and not built
- **Date:** 2026-09-18
- **Deciders:** user + Claude Code

## Context

A `Concept` row today is, in practice, a label: presence matches its label and aliases in every
chunk of the library, the curation paths get-or-create concepts by label (`add_concept`,
`promote_keyword`, `library.create_keyword_family`), and the keyword-family rules move a keyword so it
belongs to one family only. Nothing distinguishes two meanings of one word, and almost nothing
records what a concept means: **2 of 13 graph concepts carry a definition, and 0 of the other 344**
(live library, 2026-09-18); no route or screen can edit one — only the `seed_concepts --define` CLI.

Measured on the live library (`tests/eval/baselines/concept_merge_cosine_2026-09-17.md`):

- **One word carries several meanings, and presence counts them as one concept.** `specter` is a
  document-embedding model and the title of a political-philosophy book; `viral` is an infection, a
  lab vector and a web meme; `beta` is an oscillation band and a code variable. Presence feeds
  `single_source`, the one gap signal graded trustworthy, so a shared word can invent corroboration
  or hide a gap.
- **The field a document belongs to is evidence about meaning, not the meaning.** Grouping each
  label's documents by document-vector similarity flags 3 of 162 multi-document concepts: `specter`
  (two meanings), `abstractions` (two senses), and `virus` — one meaning appearing in a
  political-science agenda as well as in biology, which is exactly the cross-field link the graph
  exists to show. Within-field meanings (`viral`, `beta`) do not split by document at all.
- **Comparing labels cannot tell two concepts apart.** On bare labels, SPECTER2 scores a median
  pair 0.842 and would merge 354 of 357 concepts at 0.85; bge-base at 0.85 yields 34 pairs, of which
  about 7 are one concept (ROADMAP 53). A definition is the only text that states the meaning, and
  there are almost none to compare.

Rows 51 (`is_a` edges), 53 (merges) and 92 (sense-blind presence) all act on "a concept", so what a
concept is has to be settled before any of them writes more curated data.

## Options

1. **Keep a concept as a label; record meanings as notes on it.** Cheapest — no identity change.
   But one node still carries every meaning: presence, `single_source`, the graph's edges and its
   communities all stay mixed, so the note describes the problem without removing it.
2. **A concept is one meaning. Word forms may be shared between concepts; a concept that shares a
   word carries a curated definition and a scope that decides which mentions are its own.** Fits the
   schema that exists: `ConceptAlias` is unique per `(concept_id, alias)`, not per alias, so one word
   can already belong to two concepts (two aliases are shared today). The cost falls only on the
   shared words — a word used by one concept keeps today's presence rule unchanged.
3. **Split the vocabulary into two tables — terms (word forms, families) and concepts (meanings) —
   linked many-to-many.** The lexicographic model: WordNet maps words to synonym sets many-to-many
   (Miller, "WordNet: A Lexical Database for English", *Communications of the ACM* 38(11), 1995). The
   cleanest separation of the keyword-family layer (ADR-015, which wants breadth) from the graph
   (ADR-018, which wants a curated map). But it migrates every family and every `ConceptAlias`
   consumer at once, for a separation option 2 already expresses through shared aliases.
4. **Have a model tag the meaning of every mention at ingest.** Needs no curation up front. But it is
   a model judging meaning per mention across the whole corpus — the unbounded-call shape KI-19 files,
   and the evidence-free judging KI-33 measured as invalid for Node-B stance — with nothing for the
   user to check until it is wrong.

## Decision

**Option 2.** A concept is one meaning. Two meanings of one word are two concepts; one meaning used
in several fields stays one concept. The deciding reason: every signal the knowledge layer is trusted
for — presence, `single_source`, coverage, the graph's neighbourhoods — counts per concept, so a
concept that mixes meanings makes each of them wrong, and a separate record of meanings (option 1)
does not change what they count.

What that commits the system to:

- **Meanings are curated data.** A definition is written or accepted by the user. Suggestions may
  come from the corpus (a defining sentence found in the documents) or from a model, and are always
  marked as suggestions until accepted — the LLM proposes, never writes, and a quoted sentence is
  quoted verbatim (ADR-043). A concept that shares a word with another **must** carry a definition;
  elsewhere a definition is encouraged, not required.
- **A label is not an identity.** Two concepts may share a label; wherever the label alone is
  ambiguous the app shows the definition, or a short qualifier from it, beside it. Get-or-create by
  label becomes get-or-create by label among concepts that do not share it — an ambiguous label asks
  which meaning.
- **A shared word needs a scope.** Each concept that shares a word owns only the mentions its scope
  gives it: **by document** by default — one meaning per document, the "one sense per discourse"
  observation (Gale, Church & Yarowsky, "One Sense Per Discourse", *Proceedings of the DARPA Speech and
  Natural Language Workshop*, 1992) — and **by passage** only where one document uses both meanings.
  Presence, and everything counted from it, is computed after the mentions are assigned.
- **Word forms and meanings are separate questions.** A plural or an inflection is a surface form of
  the same concept (folded as an alias); a derived word (`virus` / `viral`) is a related concept,
  pre-paired as a family, never merged; a fragment of a longer phrase that never stands alone is not
  a concept at all.
- **Finding meanings is evidence, then review.** A word is proposed for splitting when its documents
  fall into unrelated groups, when its graph node links groups that share nothing else, or when a
  dictionary lists several meanings; the user sees its mentions grouped and answers "one concept",
  "two meanings — named and defined", or "one of these is noise". Nothing splits or merges without
  that answer.

**What would reverse it:** if review finds that meanings mix *within* documents as often as across
them, the document default is wrong and passage scope becomes the default (a cost change, not a
principle change). If the curated shared words stay a handful at 10,000 documents, the scope
machinery may be built as a manual split with no automatic assignment.

## Consequences

**Easier.** Merges and pairing (ROADMAP 53) compare meanings, not labels: with a definition on each
side, the embedding compares what the concepts are said to be, which bare labels could not support.
`single_source` and the gap list count per meaning. The graph shows each meaning in its own
neighbourhood, so `specter` the model sits among retrieval concepts and `specter` the book does not.
Row 51's `is_a` edges connect meanings, which is what a hierarchy relates.

**Harder.** Every label-keyed path changes: `add_concept`, `promote_keyword`,
`create_keyword_family`, the family rename clash check, `list_keyword_candidates`' "promoted" flag,
and the one-family-per-keyword rule in `add_family_member`. Presence gains an assignment step for
shared words. The app needs a place to read and edit definitions, which does not exist, and a review
surface for proposed splits. Curation is real work: 357 concepts, 2 definitions — the graph's 13 and
the flagged shared words come first.

**Must revisit.** Option 3 (a separate term table) if keyword families and meanings keep pulling
apart. The document-scope default, against the review's findings.

## Confidence

- ✓ Shared words with several meanings exist in this corpus and are counted as one concept today —
  `tests/eval/baselines/concept_merge_cosine_2026-09-17.md` (passages quoted there).
- ✓ A document-level split finds cross-field meanings and misses within-field ones (3 of 162
  flagged; `viral`, `beta` not flagged) — same file.
- ✓ The schema already allows one word in two concepts — `ConceptAlias`'s unique key is
  `(concept_id, alias)` (`db/models.py`), and two aliases are shared on the live library.
- ⚠ **One meaning per document holds here** — cited from word-sense research on other corpora, not
  measured on this one. `.claude/RIGOR_TODO.md` RG-031.
- ⚠ **How many concepts need a definition, and how long curating them takes** — unknown until the
  review runs; tracked with RG-031.

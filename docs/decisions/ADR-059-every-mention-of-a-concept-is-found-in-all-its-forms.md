<!-- status: active · updated: 2026-10-24 · class: append-only -->

# ADR-059 — Every mention of a concept is found in all its forms, and says how it was found

- **Status:** proposed (Claude Code, 2026-10-24). Not accepted, not built. The direction is the
  user's, 2026-10-24: *"if a user wants to focus on a specific concept or keyword, the system has to
  find reliably all instances of the keyword in all its forms (neuron, neurons, brain cell,
  n euron etc.). More than automatic keyword selection this feature is the most important."* The
  layers and their order below are Claude's proposal.
- **Date:** 2026-10-24
- **Deciders:** user + Claude Code
- **In one sentence:** when you focus on a concept, a term or a keyword, the library finds its
  mentions through five kinds of form: the word's inflections, the extraction's damage undone, each
  document's own abbreviations, the forms you marked, and forms proposed for you to mark. Each
  mention comes with its passage, how it was found, and whether it means the same thing, and nothing
  proposed counts until you mark it.

## To decide — six choices, each with a recommendation

1. **Which kinds of form apply without asking.** *Recommended:* the three that are the same word.
   (a) **Inflections**: the plural and the possessive, by the rule the keyword-family detector
   already uses (`keyword_families._tier1_morphological`, which knows that `-es` is ambiguous).
   (b) **Extraction damage undone**: words joined by a space or a hyphen (built, KI-63),
   letter-spaced text (ADR-056), accents (ADR-057), a hyphen at a line break or a soft hyphen inside
   a word, and ligatures lost in tables (KI-64). (c) **The document's own abbreviation**:
   *posterior hypothalamus (pHyp)* makes `pHyp` a form of that concept in that paper only (ADR-055,
   `term_signals`). *The cost:* (a) to (c) change presence on your 13 concepts, so the graph and gaps
   move. That is measured before and after, and switched on by your yes, as the forms were (row 94).
2. **Other words for the same thing.** *Recommended:* proposed, never applied. *Nerve cell* for
   *neuron* is proposed with the passages that use it, and you mark it exact, broad or not this.
   *Brain cell* is probably broad, since glia are brain cells too. Proposals come from three
   sources, each named on the proposal: the abbreviation pairs the library's documents give, the
   family detector's embedding tier, and a model reading passages. This is ADR-054's rule unchanged:
   a form counts once you have marked it, and what it counts as is yours to say. **A term's head
   word alone is never applied either.** *Connexion* is not *neuronal connexion* (the user,
   2026-10-24, through session 5c). At most it is a broad form, proposed with its passages, like
   the bare *contrastive* you took out of *contrastive learning* on 2026-10-09.
3. **Meaning, per mention.** *Recommended:* the finder returns every mention, and a separate
   verdict says whether the passage means the concept: by document by default, by passage where one
   document mixes two meanings (ADR-052's scope). A model proposes the verdict from the passage, and
   you confirm it. Counts are shown both ways (ADR-052's condition). The text-ranking survey's *"two
   output neurons"* are artificial neurons: found, shown, and kept apart once you say so.
4. **What you see.** *Recommended:* one view, *Find every mention*, opened from a concept, a term or a
   keyword (in Manage keywords, the Library's keyword chip, the Graph tab's concept panel). It lists
   the forms with how each was found and its count, then the mentions grouped by document: the
   passage, the page, the form as written, how it was found, the meaning verdict. Proposals sit in
   their own list. A new view name goes into the GLOSSARY first (U-001).
5. **Where it runs.** *Recommended:* when asked, never as a stored copy of every mention. The on-disk
   keyword index picks the documents in milliseconds (`definitions.chunks_mentioning`, the path
   "Look in my library" already uses), then the matcher reads those documents' parents. The rules
   are code with a version. Marked forms are stored as forms (ADR-054). Presence's stored rows are
   recomputed by the rebuild, as today.
6. **How it is measured.** *Recommended:* RG-009 (presence recall), open since 2026-07-01 and
   never measured. Pick three concepts that test different layers: `neuron` (inflection, other
   words, letter-spacing, a second meaning), one with an abbreviation (*retrieval-augmented
   generation* / RAG), and one written in a case (`Cre`). You label every mention in a seeded sample
   of their documents on a page, as a mention of the concept or not. Recall and precision are
   reported per layer, with the document as the unit, against predictions registered before
   labelling.

## In practice — `neuron`, the word you named

Read-only, 2026-10-24, on the working library (104 documents): each document's cached text after
the keyword extractor's cuts (running headers, bibliography, web addresses), counted with the
presence matcher's whole-word rule. Probe:
`.claude/probes/2026-10-23-llm-keyword-scan/forms_probe.py`.

| Written as | Mentions | Documents | Found today with the one form `neuron`? | Layer |
|---|---:|---:|---|---|
| neuron | 484 | 34 | yes | — |
| neurons | 1,279 | 35 | no: to the matcher it is another word | (a) inflection |
| *N E U R O N* (Cajal's lecture, letter-spaced) | 16 | 1 | no | (b) extraction damage |
| neurone, neurones | 3 | 2 | no | proposed (decision 2) |
| nerve cell(s), brain cell(s) | 90 | 7 | no | proposed (decision 2) |
| neuronal | 395 | 29 | no, rightly | not a form: a derived word, a related concept (ADR-052) |

1. **One form finds a quarter.** The idea is written 1,872 times in 40 documents, leaving out
   *neuronal*. The single form finds 484, which is 26%, in 34 documents. Six documents never write
   the singular; Hubel and Wiesel 1959, for one, writes *neurone*.
2. **The plural alone is two thirds.** Inflection (layer a) takes the finder from 484 to 1,763
   mentions, and from 34 to all 40 documents, with no list of words and no model.
3. **A form is not a meaning.** One of the six documents is the text-ranking survey (*lin2021*).
   Its *"the single-layer neural network has two output neurons"* is an artificial neuron.
   Inflection finds it, and the meaning verdict (decision 3) keeps it apart.
4. **Other words need you.** Cajal writes *"the morphology of the nerve cell"* (1906). That is the
   same thing, but *brain cell* in another paper may be broader. Those 90 mentions are proposed, with
   their passages, and count once you mark them.

These counts are regex counts, not read mention by mention. Indicative; decision 6 is the
measurement.

## Context

- **The matcher finds the forms it is given.** Presence is a whole-word match of a concept's name
  and its forms (`concept_skeleton.match_presence`). Its own docstring says recall is "bounded by
  alias coverage (the curation burden, RG-009)". It folds case (ADR-053 decision 3) and word joins
  (KI-63), and nothing else.
- **Pieces exist, but apart.** The family detector knows plurals and near-synonyms, but only for
  proposing families. `term_signals` reads each document's abbreviation pairs (44 of 44 right in
  their sentence, `term_signals_2026-10-14.md`). ADR-055, 056 and 057 each fix one kind of form, and
  are accepted but not built. The keyword index already finds documents by a word's prefix
  (`"neuron"*`). None of these reaches presence or one place where you can look.
- **The rules already decided:** a form is exact or broad, and only you classify (ADR-054). A
  plural is a surface form; a derived word is a related concept, never merged; a shared word gets a
  scope by document, then by passage (ADR-052). Pairing needs context, not a threshold, and is
  reviewed before anything is written (the user, 2026-09-17). The model never decides presence on
  its own (`match_presence`'s Decision 2).
- **Why it comes before the keyword scan:** ADR-058's procedure (its amendment of 2026-10-24) asks,
  for every library keyword, whether it occurs in a new paper in any form. That question is this
  finder.

## Options

1. **Layers, with rules applied and words proposed** (recommended). Recall rises where the word is
   the same; meaning stays yours where it is not.
2. **A stemmer on the index and the matcher** (Porter). One switch. *Trade-off:* a stemmer joins
   words that are not the same thing (*neuronal* with *neuron*, *general* with *generation*), and it
   finds nothing for *nerve cell* or *N E U R O N*.
3. **Embeddings: find passages near the concept's meaning.** It catches other words. *Trade-off:*
   it returns passages that are about the subject, not mentions of the concept, and a mention can
   no longer be shown as a form found in a passage.
4. **Let a model mark every mention.** *Trade-off:* a model call per passage: 8,247 prose parents
   at this library's size, about 800,000 at 10,000 documents. It also breaks the rule that the
   model never decides presence.

## Decision

Option 1, with the six recommendations. **The deciding reason:** the misses fall into kinds, and
each kind already has its owner: the word's own forms are rules, the extraction's damage is
ADR-056 and ADR-057, a document's abbreviation is that document's text, other words are your
marks, and meaning is ADR-052's scope. One finder puts them in one place, and says which one found
each mention.

**What would reverse it:** the measurement showing that the rule layers find mentions you label
as not the concept often enough to cost more than they find. A rule would then become a proposal
too.

## Consequences

- **Easier:** focusing on a concept shows its whole use, and how each mention was found.
  ADR-058's scan can ask whether a library keyword occurs in a paper and trust the answer.
- **Harder:** presence moves when the rule layers are switched on: the 13 concepts' documents, the
  graph's edges, gaps and `single_source`. That needs a before and after and your yes. It also adds
  one view, a meaning verdict to store and undo (ADR-052's record), and a page for the labels.
- **Robustness contract:** 0 documents finds nothing. The index picks the documents, so the cost
  grows with the documents that use the word, not with the library. No constant is added: the rules
  are shapes, and proposals carry their counts.
- **Order:** this, then ADR-058's procedure. ADR-055, 056 and 057 become layers (c), (b) and (b),
  so they land first or with it. The index cannot find a letter-spaced word until ADR-056 rejoins
  the text, so layer (b) reads the document only once it has been picked by another form.
- **Must revisit:** verbs and adjectives (*encode*, *encoding*, *encoder*). They are derived words
  under ADR-052, so they are proposed, not folded.

## Confidence

- ✓ The miss is large, and most of it is the plural: 484 of 1,872 found today, 1,279 of the rest
  plurals. The plural alone reaches all 40 documents (regex counts, 2026-10-24).
- ⚠ Precision of the rule layers is unmeasured. The lin2021 case shows the plural rule finding a
  second meaning. That is decision 3's job, and decision 6 measures it.
- ⚠ The other-words counts (90) mix exact and broad uses. They are proposals until you mark them.

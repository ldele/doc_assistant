<!-- status: active · updated: 2026-10-09 · class: append-only -->

# ADR-054 — The vocabulary holds concepts and terms; a concept has a name and exact or broad forms

- **Status:** accepted (user, 2026-10-01) — the two choices are the user's (asked one at a time
  with the library's own rows as examples): the 13 rows on the graph are the concepts and the 344
  bulk-promoted rows are candidate terms; a concept's label is its name, and each string matched
  for it is marked *exact* or *broad*. The shape below (which flag marks a concept, what an
  unclassified form is, which features read which rows) was Claude Code's proposal, and the user
  accepted it with the text. Builds [ADR-052](ADR-052-a-concept-is-a-meaning.md) ("a concept is
  one meaning"; "fragments are not concepts") on
  [ADR-018](ADR-018-graph-vocabulary-scope.md)'s flag, and answers ADR-052's must-revisit about a
  separate term table. Not built.
  **Accepted as amended:** the user read the text ("the text is clear") and raised one
  consideration, a curated base vocabulary for a given topic, hand in hand with the taxonomy. With
  *Amendment 2026-10-01* (at the end) recording it, the user accepted the ADR the same day ("Okay,
  let's go with this").
- **Date:** 2026-10-01
- **Deciders:** user + Claude Code
- **In one sentence:** every row of the vocabulary used to be called a concept and every alias
  counted the same; now a row is either a *concept* you have taken on or a *term* the library uses,
  and a concept has a name plus a list of matched forms, each marked as always meaning it (exact)
  or only sometimes (broad).
- **Accepted with the text (the part that was proposed):** (1) "on the graph" is what marks a
  concept — one existing flag, no new column; (2) a form nobody has classified counts as exact, so
  nothing moves until you classify it; (3) definition candidates, merge and `is_a` proposals and
  field placement run over concepts only, and a term shows its usage when you open it.

## In practice — five rows, as the library shows them

Real counts from the working library, 2026-10-01 (body prose, 104 documents;
`tests/eval/baselines/vocabulary_shape_2026-10-01.md`).

1. **Classify a form — `knowledge distillation`.** Manage keywords lists the concept with what
   each form brings:

   | Form | Documents | Mentions | Documents only this form reaches |
   |---|---:|---:|---:|
   | `knowledge distillation` (the name) | 2 | 206 | 0 |
   | `distillation` | 9 | 400 | 7 |

   Today both count the same, so the concept shows 9 documents. Mark `distillation` **broad** and
   it shows *2 documents · 7 more through the broad form `distillation`*; the graph and the gap
   list count 2. Leave it exact and nothing changes. Either way it is your call per form, and the
   control that sets it sets it back.
2. **Give a concept its name — `cre`.** Rename it *Cre recombinase*. The old label stays as the
   form `Cre`, exact. Every screen now shows *Cre recombinase*; presence is unchanged at 5
   documents, all of them through `Cre` — the name itself occurs once in the library's prose.
   The library gives no rule for that name: `Cre` is followed by "driver" in 39 of 255 mentions,
   "-dependent" in 37, "line" in 18. The name is yours to give.
3. **Remove a form that names something else — `hard negatives`.** Its alias `negative sampling`
   occurs in 6 documents, 2 of which no other form reaches. If you read it as the wider technique
   and its own subject, remove it: it leaves the concept and remains a term, and the concept goes
   from 8 documents to 6.
4. **A term stays a term — `pose`.** 498 mentions, 326 of them followed by "estimation". It is
   listed among the terms, with that fact beside it. It has no stored definition candidates and no
   field placement. Take on *pose estimation* as a concept and `pose` is offered as a broad form.
5. **Nothing is deleted — `cohan`.** A cited author: 20 of its 22 mentions are followed by "et".
   It stays a term, usable as a library filter like any keyword family, and is never proposed for
   a definition, a merge or a place in the field tree.

**Before this ADR** all five were concepts, `distillation` and `negative sampling` counted in full,
and the concept behind `Cre` had no name of its own.

## Context

ADR-052 decided that a concept is one meaning and that a fragment of a longer phrase is not a
concept. The rows do not embody that yet. Measured on the working library
(`tests/eval/baselines/vocabulary_shape_2026-10-01.md`):

- **Thirteen rows were made by hand; 344 came from one bulk promotion** of per-document keywords on
  2026-07-05 and were never read one by one. Of the 344, 197 occur in the prose of at most one
  document, 30 in none, and 113 are a keyword of no document in today's extraction. Their labels
  include cited authors (`cohan`, `moyle`), tokens joined across punctuation that occur nowhere
  (`comput vis`), and the front or tail of a longer term (`pose`, `speckles`).
- **The features that need a meaning read all 357 rows**: definition candidates, written forms,
  merge and `is_a` proposals, the vocabulary search. The results graded trustworthy were measured
  on hand-made vocabularies: presence on the 13, and `single_source` — graded a true positive
  (`docs/knowledge-layer.md` §6) — on the hand-curated vocabulary of 26 that the 2026-07 graph runs
  used (`tests/eval/baselines/rg001_concept_skeleton_r5_2026-07-02.md`). Automatic field placement
  was plausible on the 13 and failed over the 344 (ADR-045, Must revisit).
- **One alias can carry a concept's presence.** `distillation` alone reaches 7 of the 9 documents
  counted for `knowledge distillation`; `contrastive` reaches 2 of 9 for `contrastive learning`.
  Of the 99 mentions of `passage ranking`, an alias of `passage retrieval`, 89 are "passage ranking
  test" or "passage ranking task" — a benchmark.
- **The name a person gives a concept is often not the string the library writes.** "Cre
  recombinase" occurs once against 255 for `Cre`; "neurotensin receptor 1" never. The user's notes
  of 2026-09-22 read `beta` as beta oscillations and `viral` as viral vector; "viral vector" is 7
  of 103 uses of `viral`.

The existing rules bound the choice. Keyword families share the `Concept` table and nothing
auto-applies (ADR-015). Graph membership is an opt-in flag and the 344 are not deleted (ADR-018);
that ADR would revisit a two-table split "only if the two vocabularies genuinely diverge in
*shape*, not just membership" — and the difference found here is membership. Received content is
never rewritten (ADR-043); a label is curated, so renaming one is the user's act. Every curation step must stay reversible, with the label-only count kept beside the
new one (ADR-052).

## Options

1. **One population, cleaned by signals.** Every row stays a concept; the abbreviation and fragment
   signals of ADR-053 decision 4 flag the doubtful ones and the user demotes them row by row. No
   model change. But it asks for 344 reviews to reach the state the 13 already have, and until
   then the meaning features keep running over unread strings — the input on which bulk placement
   put a cardiac-MRI benchmark under *Music* (ADR-045, Must revisit).
2. **Two kinds of row on one table, and typed forms.** A concept is a row the user has taken on,
   marked by the flag that already exists; the rest are terms. A concept's label is its name; each
   matched form is exact or broad. One additive column. The name/forms split is the one SKOS
   draws between a preferred label and alternative labels
   ([SKOS Reference, section 5](https://www.w3.org/TR/skos-reference/#labels)), and ADR-028 already
   uses SKOS for the hierarchy. SKOS has no label that only sometimes denotes the concept; *broad*
   adds that, from ADR-052's rule for a shared word.
3. **Two tables — terms and concepts, linked many-to-many.** ADR-052's option 3, the WordNet model
   (Miller, "WordNet: A Lexical Database for English", *Communications of the ACM* 38(11), 1995).
   The cleanest separation, and a migration of every family, every alias reader and the library
   filter at once, for a boundary the existing flag already draws.
4. **A display name only.** Add a name to show, and keep matching as it is: label and all aliases,
   undifferentiated. Cheapest. `distillation` keeps counting as `knowledge distillation` in every
   document, and an abbreviation found later has nowhere to go but the same untyped list that
   holds `negative sampling`.

## Decision

**Option 2.** The deciding reason: every count the knowledge layer is trusted for is per concept
(ADR-052), and today a count can rest on a string nobody chose — a row that was never read, or an
alias that means something wider. Separating what the user took on from what the extractor
produced, and what always means the concept from what sometimes does, is the smallest change that
makes each count say what it rests on.

What that commits the system to:

- **Two kinds of row, one table.** A *concept* is a text-bearing row with `graph_include` set: the
  user has taken it on. Every other text-bearing row is a *term*: a string the library uses. Terms
  keep their rows, stay searchable and stay usable as library filter families (ADR-015); nothing
  removes them in bulk (ADR-018). Taking a term on is the existing "put on the concept graph"
  control.
- **Meaning features read concepts.** Stored definition candidates, merge and `is_a` proposals,
  field-placement proposals and gap detection run over concepts. Written forms, the vocabulary
  search and family filtering cover both. A term opened in the app shows how the library uses it,
  computed when asked and not stored.
- **The label is the name.** Free text, shown as typed, on every screen that shows a concept. It
  need not occur in the library. Renaming keeps the previous label as a form, which the rename path
  already does.
- **A form is exact or broad.** *Exact*: it always means the concept — a spelling, an inflection,
  an abbreviation, its long form. *Broad*: a string that also matches other things. Document
  counts, graph edges and gaps use exact forms. The broad count is shown beside them and never
  added in, which is ADR-052's "label-only count kept beside" applied to one form. The name is
  always exact. A form nobody has classified is exact, so the stored graph does not move until the
  user classifies something.
- **A phrase that names something else is not a form.** It leaves the concept and remains a term.
- **Evidence proposes, the user decides.** The app may suggest *broad* with its reason (how many
  documents only this form reaches; how many different words follow it). It never applies a
  classification, a rename or a removal. Each can be reversed from the control that made it.

**What would reverse it.** A concept the user vouches for but wants off the graph: "is a concept"
and "on the graph" then need separate flags. A term pool too large for one table to serve the
Manage keywords screen: option 3. Classifying forms proving heavier than the single list it
replaces: option 4, with the classification dropped.

## Consequences

**Easier.** The abbreviation and fragment signals (ADR-053 decision 4) get somewhere to land: an
abbreviation found beside its long form becomes a proposed exact form, a fragment becomes a
proposed longer name for a term. A label shorter than its concept is fixed by naming it. Presence
on a concept states which forms carried it. Meaning features stop running over rows nobody read.
Growing the vocabulary is taking a term on and naming it.

**Harder.** Every reader of "all concepts" has to choose its rows: `taxonomy.presence_query`,
`written_forms.load_vocabulary`, `definitions.extract_definitions`,
`concept_graph_view.search_vocabulary`, `concept_curation.load_concepts`,
`isa_propose.load_concept_labels`, `taxonomy_propose.load_concept_items`. Presence carries two
counts per concept. Manage keywords lists concepts apart from terms and shows a name, its forms and
a control per form. `GLOSSARY.md`, `docs/knowledge-layer.md` and the `Concept` model's docstring
call every row curated and need the two words. The baselines of 2026-09-17 to 2026-09-30 that say
"357 concepts" keep their numbers; read them as 13 concepts and 344 terms.

**Must revisit.** Whether definition candidates and usage examples should match exact forms as
well as the name — they match the label alone today because aliases "are other phrases with other
meanings" (DEVLOG 2026-09-21 (1)), which typed forms change. Whether "is a concept" and "on the
graph" stay one flag. Where terms come from: the keyword extractor's mode, abbreviations found in
the text, and the LLM-assisted ingest pass (ROADMAP row 25) are all candidate sources.

## Confidence

- ✓ The 344 rows are one bulk promotion, and most cannot link documents (197 in at most one
  document's prose) — `tests/eval/baselines/vocabulary_shape_2026-10-01.md` §1–§2.
- ✓ The existing flag already separates the two populations: the 13 hand-made rows are exactly the
  rows with `graph_include` — same file, method.
- ✓ A single alias can carry most of a concept's presence, and an alias can name something else —
  same file, §6.
- ⚠ **What classifying the 31 aliases of the 13 does to presence, the graph's edges and the gap
  list** — unknown until the user classifies them. `.claude/RIGOR_TODO.md` RG-032.
- ⚠ **Whether matching exact forms improves definition candidates** — not measured; the user's
  labels of 2026-09-22 are the yardstick. RG-032.
- ⚠ **That a term shown on demand is enough.** The user labelled candidates for six rows that are
  terms under this ADR, so opening a term must stay as usable as opening a concept; checked in the
  build's walkthrough, not before. RG-032.

## Amendment 2026-10-01 — a base vocabulary per field is a source of terms, and of what a field expects

The user's note on reading this ADR: a curated default vocabulary for a given topic would go hand
in hand with the taxonomy. It is recorded here as direction. It is not designed and not built, and
it leaves the decision above as it stands.

- **A base vocabulary is a third source of rows**, beside the user's own additions and the keyword
  extractor. Its entries are terms until the user takes them on, like any other candidate.
- **One sentence of the Decision is widened.** "A term is a string the library uses" holds for a
  keyword. An entry of a base vocabulary may not occur in the library at all. A *term* is a
  candidate row nobody has taken on, whatever proposed it.
- **A base entry brings what a keyword lacks:** a name, its synonyms and abbreviations, and a
  definition. Those are the three parts of a concept under this ADR, and the three things SKOS
  gives a concept (a preferred label, alternative labels, a definition), so the shape decided here
  can receive one. It also brings a field: an expert vocabulary is already consulted by the
  document's field for definitions (ADR-053, "the taxonomy picks where to look").
- **An absent entry is information.** A keyword that occurs nowhere is noise. A base concept of a
  field the library holds, absent from the library or present in one document, is what "have I read
  the field?" needs: an expected structure to deviate from (`docs/knowledge-layer.md` §1). The
  taxonomy says which fields exist; a base vocabulary says what a field contains. That is the
  concept-level half of "the taxonomy as the reference class for expected coverage" (ROADMAP KL2).
- **Two existing rules bound it.** External vocabularies are candidate sources grafted where the
  library has documents, never imported whole: 30,000 MeSH descriptors for a small library is "a
  facet that partitions nothing" (ADR-028 decision 7). And nothing auto-applies (ADR-015). So a
  topic's list is offered for the fields the library holds, and taking it on — entry by entry, or
  as a reviewed whole — is the user's act. Which of the two is that feature's decision, not this
  one's.
- **Its coverage will be uneven.** Of the 19 priority concepts, expert vocabularies define 8, give
  a one-line gloss for 6 and have nothing usable for 5, the youngest retrieval terms among them
  (`tests/eval/baselines/reference_vocabularies_2026-09-21.md`). A base list will be solid for the
  established fields and thin for the new ones, where the curating falls to the user.
- **Where it is designed.** Its data step is the one ADR-053's expert vocabularies already need
  (local copies, licences, sizes: ROADMAP 93c). What it is for is decided in the ADR-032 grill
  (ROADMAP KL2). It gets its own ADR.

## Addendum 2026-10-09 — the first ⚠ is measured: what the user's marks did

The user read the 22 forms on 2026-10-09, marked 10 broad and 12 exact, renamed no concept and
rebuilt the graph. The first ⚠ line of Confidence ("what classifying the 31 aliases of the 13 does
to presence, the graph's edges and the gap list") now has its figures:
`tests/eval/baselines/forms_marked_2026-10-09.md`. The decision stands as written.

- **Presence.** Four concepts lose documents, 43 → 29 between them. Each of the 14 pairs that went
  is reached through one named broad form and is listed beside its concept
  (`knowledge distillation` 11 → 4, the 7 through `distillation`). Five of the ten broad marks
  move no document count.
- **Links.** 30 → 25, and 17 of the 25 rest on fewer shared chunks. A broad mark that moves no
  document count still moves links: `passage retrieval` keeps its 7 documents and loses two of
  its eight.
- **Gaps.** The same 12 rows on the same concepts; three `unsourced_claim` rows rest on fewer
  answer claims.
- **The reader.** Both graphs were taken while figure chunks counted as text. On the reader that
  reads prose (ROADMAP 97) the marked graph has the same documents, links and gap rows (§6 of the
  baseline).
- **What this does not say.** That the marks are right. One reader classified the forms, and of
  `contrastive` the user said it "does not seem specific enough" to judge without its context. A
  form is marked once; whether a mention means the concept is a question per mention (ADR-052's
  ground).

The second ⚠ line is not measured. The third is the user's to judge
(`tests/eval/baselines/names_and_forms_2026-10-01.md` §4). Both stay with RG-032.

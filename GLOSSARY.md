<!-- status: active · updated: 2026-11-06 (U-003: hover text, ADR-063 choice 6; its tooltip and checks, ROADMAP 110 (a)); 2026-11-05 (U-004: the screens and their names, ADR-064; U-001/U-002 paths follow the move; U-003: the four action roles, ADR-063; 2026-11-01, U-002: the letter-spacing ignore list and its views, ADR-056; 2026-10-14, C-013: a written form may join its words otherwise than the label, KI-63; 2026-10-09, U-001: a view has one name — Manage keywords on every door to it; 2026-10-01, ADR-054: C-001 narrowed to the rows the user took on; C-014 term, C-015 name, C-016 exact form / broad form) · class: living -->

# GLOSSARY — doc_assistant

The pinned vocabulary for this project: one canonical term per concept, the synonyms that are
forbidden, and where each concept is owned. Use the canonical form in code, docs, prompts, and UI.
Enforced (opt-in, on-call) by the `cpc-glossary` gate and `cpc-dod-lint` rule N001 — cpc
CONVENTIONS §14. Laid + filled 2026-07-19 (ADR-021): the concept/keyword/family/skeleton
vocabulary has produced real confusion (see the 2026-07-17/18 "junk labels" trap) — pin it.

**Owner:** Lucas

---

## C-001 — concept (a meaning the user has taken on)

**Canonical:** `concept`
**Definition:** A text-bearing vocabulary row the user has taken on: `kind="concept"` with
`graph_include` set (ADR-054). It is one meaning (ADR-052), has a *name* (C-015) and *forms*
(C-016), is on the concept graph, and is what definitions, merge and `is_a` proposals, field
placement and gap detection read. The table and the model are named `Concept` and hold *terms*
(C-014) too: "a `Concept` row" names the storage, "a concept" names a row the user chose. Before
ADR-054 every row was called a curated concept, including 344 that one bulk promotion made and
nobody read.
**Forbidden:** `topic`, `entity`
**Authoritative in:** `docs/decisions/ADR-054-concepts-and-terms-names-and-forms.md` +
`src/doc_assistant/db/models.py::Concept` + `knowledge/concept_skeleton.load_concepts`

## C-002 — Keyword (mined candidate)

**Canonical:** `Keyword`
**Definition:** A per-document mined candidate term (`source="extracted"`); a candidate **only** —
never auto-promoted; the user promotes one into a `Concept` row (`promote_keyword`), which is a
*term* (C-014) until it is taken on. The 2026-07-05 `--promote-all` incident (ADR-018) is why this
boundary is load-bearing.
**Forbidden:** —
**Authoritative in:** `src/doc_assistant/knowledge/keywords.py` + `db/models.py::Keyword`

## C-003 — keyword family

**Canonical:** `keyword family`
**Definition:** A user-facing grouping of `Concept` rows for library filtering (ADR-015); shares
the `Concept` table with the graph and is **deliberately unfiltered** by `graph_include`, so a
concept and a term (ADR-054) are both families.
("tag family" is the historical spec name — do not reintroduce it in new code or UI.)
**Forbidden:** `tag family`, `tag`
**Authoritative in:** `src/doc_assistant/knowledge/keyword_families.py` + `library.list_keyword_families`

## C-004 — concept skeleton (the graph artifact)

**Canonical:** `concept skeleton`
**Definition:** The deterministic Node-A graph artifact (nodes = `graph_include` Concepts; edges =
co-occurrence with citation/similarity provenance), persisted as `concept_edges` + `skeleton.json`;
rebuilt, never hand-edited. "Concept graph" names the *feature/UI view* over it, not the artifact.
**Forbidden:** `knowledge graph`
**Authoritative in:** `src/doc_assistant/knowledge/concept_skeleton.py`

## C-005 — Node A / Node B

**Canonical:** `Node A` (deterministic skeleton build) · `Node B` (confined LLM enrichment)
**Definition:** Node A builds the skeleton with zero LLM calls; Node B annotates **existing** edges
with relation/stance and never creates a node or edge. Host rebuild is `--apply --enrich` together
(`--apply` alone wipes Node-B annotations).
**Forbidden:** —
**Authoritative in:** `knowledge/concept_skeleton.py` / `knowledge/concept_skeleton_enrich.py`

## C-006 — gap

**Canonical:** `gap` (a `GapRow`)
**Definition:** A finding over the skeleton. Deterministic kinds are delete-and-replaced on every
rebuild; stochastic kinds (`suggested_*`) are status-preserving upserts — two different write
disciplines (the KI-17 orphan class lives in that difference).
**Forbidden:** —
**Authoritative in:** `src/doc_assistant/knowledge/gaps.py`

## C-007 — epistemics markers

**Canonical:** `epistemics markers` (`contested` / `superseded_trend` / `stable` / `unique`)
**Definition:** Advisory chunk-level chips derived from skeleton node weights; inform-don't-block —
they never change synthesis, ranking, or the answer (byte-identical when absent). **Not a corpus
measurement:** `contested` / `superseded_trend` rest on a Node-B stance pass that never sees the
document text (KI-33); off by default, labelled experimental wherever shown.
**Forbidden:** `confidence score`
**Authoritative in:** `src/doc_assistant/knowledge/epistemics.py`

## C-008 — graph_include

**Canonical:** `graph_include`
**Definition:** The additive **opt-in** flag (ADR-018) scoping which Concept rows the graph loads;
families ignore it by design. Since ADR-054 it is also what makes a row a *concept* (C-001) and
not a *term* (C-014): one flag, so "taken on" and "on the graph" are the same act. The curation
verb is **demote** (`set_graph_include(cid, False)`), never delete — unfamiliar short labels are
usually real specialist vocabulary.
**Forbidden:** —
**Authoritative in:** `db/models.py::Concept.graph_include` + `knowledge/concept_skeleton.load_concepts`

## C-011 — domain / `kind` (taxonomy node kind) — *decided, unbuilt*

**Canonical:** `domain` (a `Concept` with `kind="domain"`)
**Definition:** An abstract field node (e.g. *machine learning*, *neuroscience*) with zero text presence,
seeded from ANZSRC, sharing the `Concept` table via a `kind` column (`concept` | `domain`) so typed
hierarchy edges can span both (ADR-028 Decision 4, supersedes ADR-019 C1). Presence/gap detectors must
read only `kind="concept"` — via the single canonical `presence_nodes()` accessor, never scattered
`WHERE kind` clauses. **Not yet built** — spec: `docs/specs/feature-taxonomy-seed-schema.md` (T1a).
**Forbidden:** `field node`, `category`
**Authoritative in:** `docs/decisions/ADR-028-concept-taxonomy-polyhierarchy-skos.md` (spec: `feature-taxonomy-seed-schema.md`)

## C-012 — concept hierarchy (curated classification DAG) — *decided, unbuilt*

**Canonical:** `concept hierarchy`
**Definition:** The user's curated `is_a`/`in_field` edges — a polyhierarchical, acyclic SKOS-shaped DAG
over `Concept` nodes, stored in a new `concept_hierarchy` table that **survives a skeleton rebuild**
(distinct from the derived, rebuilt-every-run `concept_edges`; ADR-028 Decision 5). This is the
*classification* layer; `concept_edges` is the *association* layer. Documents attach to `domain` nodes via
a `document_field` many-to-many (ADR-028 Decision 6). **Not yet built.**
**Forbidden:** `concept tree` (it is a DAG, not a tree — polyhierarchy per ADR-028), `taxonomy tree`
**Authoritative in:** `docs/decisions/ADR-028-concept-taxonomy-polyhierarchy-skos.md` (spec: `feature-taxonomy-seed-schema.md`; future `knowledge/taxonomy.py`)

## C-009 — Enrichment-Layer Pattern

**Canonical:** `Enrichment-Layer Pattern`
**Definition:** Derived data ships as a separate module + idempotent CLI runner, sidecar by
default, and never mutates the primary chunk store.
**Forbidden:** —
**Authoritative in:** `.claude/CONTEXT.md` non-negotiable #4

## C-010 — locked setting

**Canonical:** `locked setting`
**Definition:** A `config.py` value changeable only via an eval-harness experiment (`--repeat`,
beat the control beyond variance, record a baseline in `tests/eval/baselines/`).
**Forbidden:** —
**Authoritative in:** `.claude/CONTEXT.md` → Locked settings table

## C-013 — written form

**Canonical:** `written form`
**Definition:** How the library's prose spells one of a concept's surface forms, case included
(`dIN` for the label `din`, `Cre` for `cre`) — voted from body text by the skeleton build, stored
**beside** the label and never a rewrite of it (ADR-053 decision 3, ADR-043). A form written with a
capital is matched case-aware; one written in lower case matches in any case. A space, a hyphen or a
dash between two words is the same form (KI-63), so `SSp-bfd` is the written form of `ssp bfd`.
**Forbidden:** `normalized label`, `canonical label` (the label is the curated one; this is derived)
**Authoritative in:** `src/doc_assistant/knowledge/written_forms.py` + `db/models.py::ConceptWrittenForm`

## C-014 — term

**Canonical:** `term`
**Definition:** A text-bearing vocabulary row nobody has taken on (ADR-054): a string the library
uses, or a candidate something proposed — the keyword extractor today, a base vocabulary per field
later (ADR-054, Amendment 2026-10-01). A term keeps its row, stays searchable and works as a keyword
family (C-003). Nothing is stored about its meaning: no definition candidate, no merge or `is_a`
proposal, no field placement. Opening one shows how the library uses it, computed when asked.
Taking it on makes it a concept (C-001), and is the user's act.
**Forbidden:** —
**Authoritative in:** `docs/decisions/ADR-054-concepts-and-terms-names-and-forms.md` +
`knowledge/definitions.require_concept` (the write guard)

## C-015 — name (of a concept)

**Canonical:** `name`
**Definition:** A concept's label (`Concept.label`): free text, the user's, and what every screen
shows for the concept — in its written case (C-013) when the library writes one. It need not occur
in the library, and it always counts as an exact form (C-016). Renaming keeps the previous name
as a form. One server-side helper supplies the shown name to every payload
(`written_forms.shown_labels`), so two screens cannot disagree.
**Forbidden:** —
**Authoritative in:** `db/models.py::Concept.label` + `knowledge/written_forms.shown_labels`

## C-016 — exact form / broad form

**Canonical:** `exact form` · `broad form`
**Definition:** How one matched string of a concept counts (ADR-054). An *exact* form always means
the concept — a spelling, an inflection, an abbreviation, its long form — and its documents are
the concept's presence. A *broad* form also matches other things: its documents are listed
**beside** presence and never added to it, so graph edges and gaps do not see them. A form nobody
has classified reads as exact, so nothing moves until the user marks one. The mark is the user's,
per form, in Manage keywords. "Alias" is the storage word (`ConceptAlias`); the UI and the docs
say *form*.
**Forbidden:** —
**Authoritative in:** `db/models.py::ConceptAlias.breadth` +
`knowledge/concept_skeleton.load_concepts` (exact) / `load_broad_forms` (broad)

## U-001 — Manage keywords (the view), and the rule it stands for

**Canonical:** `Manage keywords`
**Definition:** The one view in which the vocabulary is curated: concepts and their forms, terms
and their member keywords. **A view has one name.** Its heading, every control that opens it —
from the Library's keyword filter, from the Graph tab, from the graph's empty state — and every
sentence that sends the reader to it say *Manage keywords*. A control that acts on one row may
say what it does (*Edit* on a concept) and names the view in its tooltip. The rule is general: a
second name for a screen, a tab or a control reads as a second thing.
**Forbidden:** `Manage concepts`, `Choose the vocabulary` (both were labels on doors to this view
until 2026-10-09)
**Authoritative in:** `apps/desktop/src/screens/manage-keywords/ManageKeywords.svelte` (the view) +
`apps/desktop/src/screens/graph/Graph.svelte` (its doors on the Graph tab)

## U-002 — letter-spacing ignore list, and the views on it

**Canonical:** `letter-spacing ignore list` (*the ignore list* where ADR-056 is the context) ·
`Letter-spacing on this document` (the document page's block) · `Edit spacing` (the dialog) ·
*close* (what extraction does to a letter-spaced run: remove the spaces the page does not show)
**Definition:** The list of letter-spaced runs that extraction leaves as extracted (*Keep spaced*,
or every run of a document) or writes with the spacing the user chose (*Edit spacing*), per PDF by
content hash, in `rejoin_ignore.json` in the data home (ADR-056). The document page's block shows
one document's closed and kept runs; Settings > Documents shows every entry and empties the list.
A change takes effect when that document's text is re-run, and every view says so until then.
**Forbidden:** `rejoin list`, `allow list` (for the list); `rejoin` (for *close* in the UI: it
survives only in code names, `rejoin_ignore.json`)
**Authoritative in:** `docs/decisions/ADR-056-a-space-the-page-does-not-show-is-not-text.md` +
`src/doc_assistant/library/spacing.py` + `apps/desktop/src/screens/document-page/LetterSpacing.svelte` +
`apps/desktop/src/screens/settings/SpacingIgnoreList.svelte`

## U-003 — action roles: ordinary, primary, quiet, danger

**Canonical:** `ordinary` · `primary` · `quiet` · `danger` (a button's *role*; the classes are
`btn`, `btn-primary`, `btn-quiet`, `btn-danger`) · `Done` (ends a mode) · `Select all` /
`Select none` (the selection helper) · `hover text` (what a button says it will do, on hover and
on focus)
**Definition:** What kind of thing a button does, which decides its look on every screen
(ADR-063). *Ordinary* is most buttons. *Primary* is the one action a screen or a mode is for, at
most one per bar; *Done* ends a mode and is that mode's primary, at the trailing end. *Quiet*
changes only what is shown or selected. *Danger* removes something: its label names the act
(*Delete*, *Remove*), it is red, and it always asks first, in a confirm that names what goes and
how many. Every button has *hover text*: it completes "this will …", never repeats the label,
shows on keyboard focus too, and holds nothing the user must know that is not also on the screen
(ADR-063 choice 6). In the docs it is *hover text*; *tooltip* names the shared component that
shows it. A selection is emptied by *Select none*. *Clear* is never the label of anything that
removes or deselects: it stays only for emptying a text field or a filter, which change what is
shown. The ordinary role is not called *secondary*: that reads as "less important", and most
buttons are ordinary. (Not listed under Forbidden: the glossary gate reads only C- and D- entries
and would file these words under the entry above, so every *Clear* in the repo would warn.)
**Forbidden:** —
**Authoritative in:** `docs/decisions/ADR-063-a-buttons-look-says-what-it-does.md` +
`apps/desktop/src/app.css` (the classes) + `apps/desktop/src/lib/shell/selection.ts` (the selection
helper) + `apps/desktop/src/lib/shell/tooltip.ts` (the tooltip) + `apps/desktop/src/ui-rules.test.ts`
(the checks)

## U-004 — the screens, and what each is called

**Canonical:** the tabs `Chat` · `Library` · `Graph`; `document page`; on the Graph tab, `concept
panel` (a concept's side, ADR-053) and `concept index` (its sidebar rail); the views `Manage
keywords` (U-001) · `Taxonomy` · `Settings`; the frame, `top bar` · `sidebar` · `status bar`
**Definition:** A *screen* is a place you go: a tab, the document page, or a view that covers what
is under it and has parts of its own. A *dialog* asks one thing and closes, and is called by its
heading (*Re-run ingestion*, *Edit metadata*). Each screen has one folder under
`apps/desktop/src/screens/`, named as the app names it, with one entry file named for the folder;
the frame is `src/shell/` (ADR-064). A dialog lives with the screen whose thing it acts on,
whichever screen has the door, so a file is found by the name on the screen.
**Forbidden:** `concept view` (the concept panel; ROADMAP 111's name for it), `browser` / `Library
browser` (the document page), `Sources panel` (the Manage files list)
**Authoritative in:** `docs/decisions/ADR-064-the-frontends-folders-follow-the-screens.md` +
`docs/architecture.md` § `apps/desktop/src/` — the screens (every screen and dialog, with its doors)
+ `apps/desktop/src/layout.test.ts` (the folder rules)

## D-001 — Provenote vs doc_assistant

**Canonical:** `Provenote` (product) · `doc_assistant` (code)
**Definition:** Provenote is the product/installer identity (wordmark, window title, bundle id);
`doc_assistant` is the code identity (Python package, commands, npm name, sidecar binary). The
split is intentional (ADR-012) — never "finish" the rename into code.
**Forbidden:** `provenote` (as a code identifier)
**Authoritative in:** `docs/decisions/ADR-012-provenote-installer-identity.md`

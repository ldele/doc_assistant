<!-- status: active · updated: 2026-09-22 · class: append-only -->

# DEVLOG — doc_assistant

Real-time development log. One entry per logical change.
Append only — never edit past entries.

Format: What changed | Why | Rejected alternatives | What it opens

> **This file keeps the newest 20 entries** (`devlog_max_entries = 20` in `scripts/conventions.toml`,
> cpc rule 13b, user standard 2026-09-10; `tests/unit/test_doc_sizes.py` pins the same number).
> Rotate with `python tools/conventions/rungate.py rotate --root . --file devlog --write` (the shim —
> calling `tools/conventions/cpc/rotate.py` directly fails under `.venv`; corrected 2026-09-16) — it moves the
> oldest entries **verbatim** into the highest-numbered archive and verifies the bytes; then update
> the range below by hand (cpc ticket T-003). A day may be split across two files at the cut.
> Older entries, newest-first, unedited:
> **2026-08-12 (1) → 2026-09-04** (the first, unnumbered 2026-09-04 entry; the first 2026-09-02 entry is unnumbered too) in [`docs/archive/DEVLOG-archive-006.md`](archive/DEVLOG-archive-006.md)
> (rotated 2026-09-04, 2026-09-10, four times on 2026-09-16, on 2026-09-17, 2026-09-18, 2026-09-20, twice on 2026-09-21 and twice on 2026-09-22) ·
> **2026-08-08 (1) → 2026-08-11 (4)** in [`docs/archive/DEVLOG-archive-005.md`](archive/DEVLOG-archive-005.md)
> (rotated 2026-08-30) ·
> **2026-08-05 → 2026-08-07** in [`docs/archive/DEVLOG-archive-004.md`](archive/DEVLOG-archive-004.md)
> (rotated 2026-08-28) · **2026-08-02 → 2026-08-03** in
> [`docs/archive/DEVLOG-archive-003.md`](archive/DEVLOG-archive-003.md) (rotated 2026-08-26 — this
> row was missing until 2026-08-28) · **2026-07-15 → 2026-08-01** in
> [`docs/archive/DEVLOG-archive-002.md`](archive/DEVLOG-archive-002.md)
> (rotated 2026-08-15) · **2026-05-21 → 2026-07-14** in
> [`docs/archive/DEVLOG-archive-001.md`](archive/DEVLOG-archive-001.md) (rotated 2026-07-21).
>
> **The 4,000-line cap in `tests/unit/test_doc_sizes.py` stays as the backstop** (an entry cap cannot
> see an entry that is itself an ADR in disguise). When either trips, rotate — **do not raise
> the cap.** The cap exists because this log reached 8,244 lines before anyone noticed: every entry
> is individually small and correct, so unbounded growth is invisible per commit.

---

## 2026-09-22 (2) — A first mention becomes a usage example, not a definition candidate (ADR-053 amended)

**What changed.** `find_passages` now keeps only sentences shaped as a definition (coined, named,
defining). The first mentions move to `find_usages`: the first plain use in each of the documents
that use the concept most. Both come from one scan (`_scan`).

The panel gains a read-only **How your library uses it**:
- `GET /api/concepts/{id}/usage` → `load_usage`;
- `chunks_mentioning(top_docs=…)` reads only the documents the keyword index ranks highest, so a
  panel open costs the same at any library size;
- it says when the index is missing rather than showing nothing.

Rows stored as first-mention candidates are hidden from the options unless chosen; none exist in
the live library. ADR-053 gains a dated amendment with the user's four decisions. The runner's
per-form counter now counts `named` (a dry run crashed on the first named sentence).

**Why.** The user's labels: first mentions defined the term 3 times in 45; they show how a word is
used, which is the context the user said every candidate lacks.

**Measured, $0** (`definition_labels_2026-09-22.md` §6):
- 24 candidates remain, all already labelled; usable 11/24, from 14/69.
- The top candidate is usable for 7 of 19, unchanged.
- The 3 usable first mentions all still show, as usage examples.
- Usage route: 91–158 ms. One junk example (a pseudo-code block).
- 34 definitions tests (unit + API), `npm test` 272, `svelte-check` clean; checked live in dark
  and light, and at 375 px.

**Rejected.**
- Storing usage examples as a new candidate source: they could then be chosen, and choosing is the
  mirror's only write.
- A slow full-library fallback for the usage route: it is asked on every panel open.

**What it opens.** Labels that keep their written case (`dIN`/`Din`, `Cre`/`CRE`), then the
abbreviation signal (decisions 3 and 4 in the amendment).

## 2026-09-22 — The user's labels measure the definition extractor: one candidate in five is usable; first mentions almost never are

**What changed.** A new baseline, `tests/eval/baselines/definition_labels_2026-09-22.md`: the user's
69 labels from the *Definition Candidate Review* page (rating hidden), set against the extractor's
grade and form. The 2026-09-21 baseline gains a note that its agent-made "first reading" is
superseded. No code change.

**Measured, $0.**
- **Usable** (defines + needs context) **by grade:** strong 8/13 (95% 36–82%), some 4/27, thin 2/29.
- **By form:** first mentions 3/45; definition-shaped 11/24.
- **By concept:** the top candidate is usable for 7 of 19. The agent's first reading had said 11
  of the strong ones were clean; the labels make it 6.
- **The user's notes:** defines-vs-claim is hard to call, and some sentences are both. Every case
  needs context. For four concepts the label, not the sentence, is the problem: `beta` is beta
  oscillations, `viral` is viral vector, `cre` is Cre recombinase, and `din` is `dIN`, a case
  distinction the lower-cased labels lose.
- **Probe (read-only):** how each label is written separates abbreviations and names (91–100%
  not lower-case) from ordinary words (0–20%). Finding the spelled-out form (Schwartz & Hearst)
  expands `dbs` and `pddl`, and finds a second meaning of the letters `CRE`.

**Why.** ADR-053 shows a grade with its reasons and never lets it choose. This is the first
measurement of how far to trust it: it sorts the candidates, but it is not a verdict.

**Rejected.** Declaring a threshold for an "abbreviation likelihood" from 19 labels: it would be
fitted to its own test set.

**What it opens.** Choices for the user (in chat), before any extractor change:
- first mentions move out of the definition candidates into the usage layer;
- *claim* becomes a flag beside the label, not a label of its own;
- labels keep their written case;
- an abbreviation / fragment signal shown with its reasons, measured on concepts outside these 19.

**The first labelling pass was lost:** the Claude app's pane never stored it (claude-skills T-004).

## 2026-09-21 (2) — Expert vocabularies become a source of definitions (ADR-053 amended); a labelling page measures the rating

**What changed.** ADR-053 gains a fourth candidate source, `reference`: an expert vocabulary's
definition quoted verbatim with its vocabulary, identifier, version and licence — the user's decision
after asking whether experts had already defined these terms. Local copies chosen by the document's
field (ANZSRC), matched with the library as judge (abbreviations expanded from the library's own
text, ties ranked by closeness to the concept's passages), shown as *what the field says* beside *how
your library uses it*; no source that needs an account (UMLS, SNOMED CT). It becomes slice **93c**;
papers to add move to 93d and the shared-word split to 93e. The ADR now opens with a one-sentence
summary, a *To decide* list and a worked example (`hard negatives`; `dbs` for the two layers), after
the user found it unclear — the general lesson is cpc ticket T-017. The **Definition Candidate
Review** page (a private artifact) holds the 69 candidates for the 19 priority concepts with the
text around each, six labels, and the machine's rating hidden until the user asks — the yardstick
for every later refinement. `passage_evidence` says "once", not "1 times".

**Why.** The user: a found sentence is evidence about a meaning, not the definition; the goal is to
lean on the library and on expert sources rather than a model's own knowledge.

**Measured, $0** (`tests/eval/baselines/reference_vocabularies_2026-09-21.md`). Four public
vocabularies probed with the 19 labels: **8** have a curated definition (MeSH for `beta` and `dbs`,
the Xenopus anatomy ontology's *descending interneuron* for `din` — the library's own sense — the
mouse-line registry's GN220 record for `ntsr1`, and `cre`, `viral`, `virus`, `contrastive
learning`), **6** only a one-line gloss, **5** nothing usable — the youngest retrieval terms, a
model's own name, a generic word. Four labels were expanded by hand first; the matcher has to do
that itself, and 93c measures how often it can.

**Found on the way.** Bandit (CI and the pre-commit hook) flagged the three SHA-1 content
fingerprints in the 93a code as "weak hash for security" and blocked the commit; they are
idempotency keys, now `usedforsecurity=False` — the same digest, so stored keys are unchanged.

**Rejected.** Live lookups for every concept (online dependence, and the user wants the app
self-reliant). Ranking an expert definition above a library passage (both can be right and differ).
Sources behind an account or licence agreement.

**What it opens.** 93b, then 93c; the user's labels, read back from the page.

---

## 2026-09-21 (1) — A concept's definition is chosen from candidates that keep their source (ROADMAP 93a, ADR-053)

**What changed.** The user's direction (2026-09-21): definitions can come from the text, from a model,
from the references or from the user, *"each of these options should not overwrite the other"*, and the
user chooses case by case. Recorded as **ADR-053** (proposed) and built as its first slice:

- **`concept_definitions` + `concept_definition_events`** (additive). Every candidate is a row with its
  text, source, provenance and evidence; `knowledge/definitions.py` is the only writer of it and of
  `Concept.definition`, which now mirrors the chosen candidate. Choose / dismiss / restore / undo, one
  step at a time; nothing is ever deleted. `add_concept(definition=…)` goes through it; a merge carries
  the dropped concept's candidates to the survivor (its own choice wins) and the undo brings them back.
  The two definitions that existed migrate at boot as chosen `user` candidates.
- **Passages from the library, $0.** Sentences copied verbatim — coined ("we call our model SPECTER"),
  named ("… is called a 'cross-encoder'"), definition-shaped, or the first mention in the documents that
  use the term most — with the bibliography cut off and title blocks, address lines and chunk-cut
  sentences refused. Each carries its reasons and a grade built from them. `scripts/extract_definitions.py`
  (dry-run default) for the whole vocabulary; "Look in my library" in the panel for one concept.
- **The app.** `GET/POST /api/concepts/{id}/definitions` + choose / dismiss / restore / undo / extract,
  and `GET /api/concepts/search`. A definition card in the Graph tab's concept panel (the user's choice of
  place): the chosen one with its source and "Open passage" (the chat-citation path, to the page), the
  other options with their evidence, write-your-own or edit-a-passage-into-your-own, undo. The graph
  index's filter also searches the whole vocabulary, so `viral` or `specter` — not graph nodes — open
  the same panel.

**Why.** ADR-052 made meanings curated data; 2 of 357 concepts had any, and one slot, last write wins,
could not hold a passage, a model's text and the user's words side by side.

**Measured, $0** (`tests/eval/baselines/definition_sources_2026-09-21.md`). 327 of 357 concepts get
candidates, 37 a strong one. On the 19 priority concepts the grade agrees with a by-hand reading 18 of 19
times; what it cannot see is a definition-shaped *claim* ("knowledge distillation is an excellent
technique…"). References alone: 5,266 titled entries, 13 of 19 priority concepts named by one — they
expand abbreviations (`dbs` → deep brain stimulation) and settle meanings (`beta` → oscillations), but
only 44 of 5,639 entries resolve to a library document. "Look in my library" went from 12.1 s to 272 ms
by asking the keyword index which documents to read; identical to a full read on 357 of 357 concepts
after a prefix match (the index keeps `actor-critic` whole) and a deterministic tie-break. Checked live
against a throwaway copy of the data directory — never the library — in dark, light and at 375 px.

**Rejected.** One slot with a history (ADR-053 option 3): alternatives would still replace each other. A
model defining every concept at ingest (option 4, KI-19/KI-33). Matching aliases: they are other phrases
with other meanings. Letting the grade choose: it is evidence, and 1 in 19 shows why. Extracting for all
357 concepts on the first click: a write the user did not ask for.

**Found on the way.** Undo was flaky — two events inside one tick of the Windows clock share a timestamp;
events are now ordered by a per-concept sequence. The first extractor ranked tied documents by read
order, so two paths gave different candidates for 5 concepts.

**What it opens.** The user's go on `extract_definitions --apply` for the library (the panel works per
concept without it). 93b: model candidates from passages and from reference titles alone, with a
grounding check, measured on the 19. 93c: what the references say + papers to add. 93d: the shared-word
split review — `viral`'s candidates already show its two senses side by side. Security S-5 moves to the
next session.

---

## 2026-09-20 (1) — The hierarchy can hold `is_a` edges, something proposes them, and the app can accept or reject a proposal (ROADMAP 51 + security S-4)

**What changed.** Three things the concept→concept spine was missing, plus this session's security
step.

1. **The write seam knows what each edge type joins.** `taxonomy.add_hierarchy_edge` now refuses an
   edge whose endpoint `kind`s do not match its type (`EdgeKindError`): `is_a` joins two concepts,
   `in_field` points at a `kind="domain"` field (ADR-028 D2). Before this, `is_a` concept→field was
   written happily and the two types could only be told apart by whoever wrote the row. The API
   maps it to 400.
2. **Something proposes `is_a`.** New `knowledge/isa_propose.py` + `scripts/propose_isa.py`
   (dry-run default, $0, no model, no network): a label whose tokens *end with* another concept's
   whole label is proposed as narrower than it, written as `origin="proposed"` rows through the
   seam. 27 candidates on this vocabulary — `tests/eval/baselines/isa_head_suffix_2026-09-20.md`.
   **Not run with `--apply` on the live library** — that is the user's call.
3. **TX3b: the app can review a proposal.** `GET /api/taxonomy/proposals` serves every proposed
   link — hierarchy edges *and* document classifications — and the taxonomy modal gains a
   "Proposed placements" pane with Accept / Reject per row, plus the same two actions on every
   proposed chip and document row in a field's detail. Accept is the existing curated write, which
   promotes the row in place; `attach_document_field` gained that promotion, so a proposed document
   classification can now be accepted and not only rejected.
4. **Security S-4** (`docs/security.md` §4): the CSP adds `form-action 'none'; base-uri 'none';
   object-src 'none'` — the three directives that do not inherit from `default-src`.

**Why.** ADR-045 measured the taxonomy as machinery without data: 13 of 357 concepts placed, **0
`is_a` edges**, and nothing but a raw API POST able to write one. A proposal that cannot be accepted
or rejected in the app is not a proposal (ADR-028 D8), and an `is_a` proposal has no field to sit
under, so it needed a surface of its own. The merge baseline supplied the input: 21 of its 34
near-duplicate pairs at ≥ 0.85 were narrower terms, not duplicates.

**Measured, $0.** 357 concepts → 27 `is_a` candidates; first reading 17 ok · 5 check · 5 fragment.
**One of the 27 touches the 13 graph concepts** — the shared heads live in the keyword-derived
vocabulary, so this rule does not build a spine *for the graph*. Matching aliases as well as labels
adds 10 candidates, nearly all wrong (`ai benchmarks` → `benchmarks plateau`), because an alias is a
different phrase whose head is not the concept's: labels only. Live, the review pane lists 95
proposals (13 concept placements + 82 document classifications).

**Rejected.** Proposing on a shared *prefix* — `self-sorting memory` is a kind of memory, not a kind
of `self-sorting`; the merge baseline's "narrower" column contains both shapes and only the
suffix one is hyponymy. Directing a cosine pair no lexical rule can direct (`apoptosis ~ cell
death`) — those stay for the hand score. Filtering fragment candidates (`recog`, `unlabeled`)
automatically — rejecting one is a click, and inventing a cleverer filter would hide row 93's real
problem. Running `propose_isa --apply` on the live library unasked.

**Found on the way.** `taxonomy_propose.write_proposals` caught only `ValueError`, so one id deleted
between the pass and the write would have raised `IntegrityError` and cost the whole batch; both
writers now use a savepoint per proposal (`tests/unit/test_taxonomy_propose.py`).

**What it opens.** The user's decision on running the proposer against the library, and then the
review. Row 92/93 read the same list: a fragment proposed as a parent is a vocabulary problem, not
a hierarchy one.

---

## 2026-09-18 (1) — Six papers added to give concepts definitions; the similarity step no longer dies on one stale vector

**What changed.** `doc_vectors.load_chunk_embeddings_by_document` skips, and logs as
`dropped_chunks_unknown_document`, a chunk whose `document_id` names no library document; test
`tests/integration/test_doc_vectors_loader.py` (fails without the fix). **Data, by the user's
request:** six open-access papers — the text-ranking book (Lin, Nogueira & Yates), the RAG survey
(Gao et al.), the distillation survey (Gou et al.), PDDL2.1 (Fox & Long), SPECTER (Cohan et al.), and
the Cre driver-line paper (Gerfen et al., saved by the user from a browser) — added through the app's path (inspect → add, copy → ingest) and enriched with the $0 runners; the
graph and gaps rebuilt. Record: `tests/eval/baselines/new_papers_definitions_2026-09-18.md`.

**Why.** ADR-052 makes definitions curated data, and there was almost nothing to curate from: 2 of
357 concepts had one, and a strict scan found a defining sentence in the library for 2 of 19
priority concepts. **The fix** because `compute_doc_vectors --apply --force` rolled back its whole
edge set on a foreign key — one chunk left in the vector store by a synthetic test document removed
in an earlier session; any forced run would have failed the same way.

**Measured, $0.** Graph coverage 30 → 36 documents, edges 20 → 30; deterministic gaps 17 → 12
(`single_source` cross-encoder, pddl and ntsr1, `isolated` cross-encoder and `under_connected`
knowledge distillation closed, none opened — no graph concept is single-source any more). Strict-pattern definitions 58 → 68, priority concepts with one
2 → 5 — but **a concept's first sentences in a survey found an introducing passage for every
priority term the patterns missed** (BM25, cross-encoder, RAG, PDDL): the better source for row 93's
suggestions. The new text exposes alias meaning problems — `contrastive` as an alias of
`contrastive learning` matches "contrastive and ablation experiments"; `hard negatives` is defined
two different ways by two papers.

**Rejected.** Getting the sixth paper past PubMed Central's and the publisher's bot checks — the
user saved it from a browser instead. Writing the three missing years and the lower-cased PDDL2.1 title — metadata is the user's to correct in the
library. Deleting the stale vector — reported, not removed; a full-scope ingest's orphan sweep is the
existing path. Promoting any keyword of the new papers to a concept — candidates only (curated
vocabulary).

**What it opens.** Row 93's suggestion source: first introduction per document, not fixed patterns.
The alias findings for ADR-052 curation — and `ntsr1` here names a Cre mouse line, which its
definition must say.

---

## 2026-09-17 (3) — Security S-3: answers are sanitised before they become HTML, and the dev loop gets a CSP — not the way the plan said

**What changed.** `Markdown.svelte` runs `marked` output through DOMPurify (new dependency
`dompurify` ^3.4) before `{@html}`, forbidding `style`, `form`, `input`, `button`, `textarea` and
`select`; citation buttons are still added afterwards, on the DOM. The Vite dev server sends a
Content-Security-Policy built from `tauri.conf.json`'s production string by
`src/lib/core/devCsp.ts`, which adds only `style-src 'unsafe-inline'` (Vite injects each stylesheet
as a style element), the HMR socket and Tauri's IPC endpoints — `script-src` is never loosened.
Tests: `devCsp.test.ts` (3 — every production value kept, script never loosened, only the three
additions); `test_desktop_security_config.py` +2 (the only `{@html}` sink renders DOMPurify output;
the dev header is derived from the production policy). Walkthrough §3 gains the hostile-markup row.

**Why.** S2: a passage quoted from the corpus carries its markup into the answer, and `marked` ≥ 14
passes `javascript:` links. **The plan's second half was wrong.** It said to set `devCsp`. Read in
the Tauri 2.11.3 source: on desktop `tauri dev` loads `devUrl` directly, and the policy (`devCsp`,
else `csp`) is injected only into assets Tauri serves itself (`manager/mod.rs` `csp()` →
`get_asset`; the dev proxy is mobile-only). `devCsp` would have changed nothing.

**Verified live, $0** (browser pane over the Vite dev server and the API): the real `Markdown`
component mounted with an `<img onerror>`, a `javascript:` link, a script, a form and a style block
rendered all five removed, kept the ordinary link and the `[1]` citation button, and left the
probe variable untouched; with DOMPurify bypassed, the dev policy alone blocked an inline `onerror`
(`script-src-attr`); the app raised no violation of its own across Chat, Graph and the taxonomy
modal, HMR connected, every `/api` call 200. **Not verified:** the Tauri window under `just app`
(IPC with the new header) and the installed build — both walkthrough rows.

**Rejected.** `devCsp` (a no-op on desktop, above). A `<meta http-equiv>` policy in `index.html` — it
would also apply in the built app on top of Tauri's header and block Tauri's nonce'd scripts.
Sanitising in a plain `.ts` module for `node:test` — DOMPurify needs a DOM, and jsdom for one call
breaks the frontend's zero-test-dependency rule. An `ALLOWED_TAGS` allow-list — markdown emits a
wide tag set, and a missed tag silently drops content.

**What it opens.** S-4 (`form-action`, `base-uri`, `object-src`) is next. S-11 can read
`DOMPurify.removed` for its `security_markup_sanitised` event.

---

## 2026-09-17 (2) — Row 53: a merge keeps what you curated and can be undone; the preview and the merge share one definition — the only harmless one measured

**What changed.** New `knowledge/concept_merge.py` is the merge's write seam (`apply_merges`,
`undo_merge`, `list_merges`), replacing fold-and-delete. A merge refuses ids that are not both
concepts, and a plan whose placements would close a taxonomy cycle — decided in memory before the
first write — then moves to the survivor: surface forms; definition and graph membership when it
lacks them; every `concept_hierarchy` placement, re-pointed through `add_hierarchy_edge` (an edge
between the two is dropped, not made a self-edge); gap triage (the survivor's own verdict wins); and
stochastic gap suggestions. It deletes the row and writes a `ConceptMerge` record (new additive
table `concept_merges`) holding what `undo_merge` needs. `apply_plan` returns `(n_demoted,
MergeOutcome)`; `curate_concepts` reports refused merges and the largest merge group, and gains
`--merges` and `--undo-merge ID [--apply]`. **One definition:** `concept_curation.dedup_pairs`
(keyed by id; `merge_text` = label + definition) serves both `suggest_concepts --near` and `--dedup`,
defaulting to new `CONCEPT_MERGE_MODEL` bge-base and `CONCEPT_MERGE_COSINE` 0.85 → **0.90**; the
preview also stops listing artifact labels. Tests: `test_concept_merges.py` (9, real SQLite with
foreign keys on; non-vacuous — patching `_repoint` to move nothing fails two of them),
`test_the_merge_preview_is_the_merge`, the `apply_plan` tests.

**Why.** REVIEW 2026-09-16 C-1/C-2: `concept_hierarchy` cascades on delete, so a merge destroyed the
dropped concept's placements, orphaned its triage and left no record — the one curation path that
deletes, against ADR-028 and ADR-018. Row 51 writes curated `is_a` edges next, so this came first.

**Measured before choosing the definition** ($0, read-only;
`tests/eval/baselines/concept_merge_cosine_2026-09-17.md`). My first cut unified on the preview's
settings. On the 357 real labels SPECTER2's median pair scores 0.842: at 0.85 the merge would delete
354 concepts into one group, at 0.90 309. bge-base at the old merge's 0.90 merges none; at 0.85, 34
pairs — 7 duplicates, 21 narrower terms, 6 related, by my first reading. So the shared definition
is the old merge's: `--apply` behaves as before, the preview is finally honest, and no threshold is
claimed.

**Rejected.** A savepoint per merge — pysqlite's SAVEPOINT handling needs a driver workaround the
session lacks. Demoting the dropped concept behind a `merged_into` column — every reader of
`concepts` would need a new filter. Keeping SPECTER2 because `config.py` says bge "compresses
same-domain concepts" — on labels the reverse holds. Settling on 0.85 or 0.87 for bge — the pairs say
a threshold is the wrong tool.

**What it opens.** Row 53 (3): the user's hand score of the 34 pairs, and the design it points at —
a surface-form fold for real duplicates, per-pair review for the rest. The 21 narrower pairs are
`is_a` candidates for row 51. Nothing was merged on the live library.

---

## 2026-09-17 (1) — Row 54: no write can put a field node on the graph, and every coverage number counts what it describes

**What changed.** `taxonomy.py` gains `presence_query()` (the kind guard as a query to narrow) and
`presence_node(session, id)` (its write side). Routed through them: `concept_skeleton.load_concepts`
— the graph's own vocabulary read, which filtered on `graph_include` alone — `set_graph_include`,
`delete_concept`, `rename_concept`, the label get-or-create in `add_concept` and `promote_keyword`,
`load_glossary`, `list_keyword_candidates`; the family read and all five family writes by id in
`library/keywords.py`; `concept_curation.load_concepts` and `rank_keyword_candidates`.
**Denominators:** graph coverage counts cited ∩ shown documents over the non-archived count
`library.count_documents` reports; the coverage sentence names the skeleton's concept count, not the
vocabulary's; the taxonomy header counts the graph vocabulary, and its "not yet placed" is exactly
`unplaced_concepts()` — auto-propose's input — shown as "N graph concepts". Tests: 3 kind-guard tests
(refused by `set_graph_include` and absent from `load_concepts` even when already flagged; the five
family writes refuse a field and its placement survives a delete; `add_concept` never adopts a field
with the same label) and 2 parity tests asserting against the functions that act
(`count_documents`, `unplaced_concepts`), not a fixture; the staleness test that pinned the numerator
bug (expected 2) now expects 1.

**Why.** REVIEW 2026-09-16 C-4/C-5/C-6. ADR-028 D4's guard lived on the reads that remembered it; a
family delete by a field's id would have cascaded every placement under that field. Graph coverage
counted deleted documents; the taxonomy header said "344 not yet placed" (357 families) while
auto-propose had nothing left to place.

**Verified live, $0.** `/api/taxonomy`: 13 concepts · 98 documents · 0 unplaced (was 357 · 344); the
header renders "13 graph concepts · 98 documents · 236 fields"; the graph still reads "Covers 30 of
your 98 documents" (no document on this library is deleted or archived).

**Rejected.** A database constraint on `graph_include` — a migration for a rule the write seam holds,
and it would not cover renames or deletes. Keeping `n_concepts_total` at every family and fixing only
"not yet placed" — "357 concepts · 0 not yet placed" reads as all placed. Raising instead of returning
None/False — each function keeps its existing contract.

**What it opens.** `taxonomy_view` still filters kinds on networkx attributes (a graph read, not a
query — left). `unclassified_documents` still includes archived documents (propose input, harmless).

---

## 2026-09-16 (5) — KL1: the knowledge layer says what its signals are — thin bridges get a real definition, and "unsourced claims" turn out to be about your answers

**What changed.** Sequence row 2's main work, from a read of the code and the live library before
any edit. **A1 (wording):** the opt-in stance chips read `contested? (experimental)` /
`superseded? (experimental)` in the source card and in the controller's Sources block (was
"contested in corpus"); the Settings sandbox toggle says "not a corpus measurement" and falls back to
`false` like the backend; `unsourced_claim` is shown as **"Uncited in answers"**; GLOSSARY C-007,
`how-answers-work.md`, README and the `ConceptGraphEdgePayload` docstring (which claimed `relation`
was empty on every edge — 19 of 20 carry it) stop implying a measurement. **A4:** most of RG-014's
grades were already encoded (single_source leads, under_connected hidden). What remained:
`detect_thin_bridges` now counts a bridge only when both sides keep ≥ 2 concepts and flags the
smaller side's endpoint (both on a tie), computed in one linear pass over the tree of 2-edge-
connected blocks; and the gap list's "N open" counts only rows it can show. **A3:**
`synthesis.is_claim_unit` rejects pieces that assert nothing — a markdown heading, a sources/references
block, a piece that is empty or ends in `:` once a trailing list enumerator is removed — applied in
`detect_unsourced_claims` only. **A5:** `knowledge-layer.md` §6 re-read: thin_bridge structural,
unsourced_claim about answers, new rows for coverage, stored-gap staleness, placement, merges; §6b
notes why ADR-041 option 6 is not buildable as written. **A2 → new row KL1b** (decision first).
**New rows 90** (the unreadable "According to Source 6: file.pdf" citation form) **and 91** (say when
stored gaps predate the graph). Tests: 5 thin-bridge tests replace the one that pinned both-ends
flagging; `is_claim_unit` (3) and the gap floor (1); two `test_chat_controller` strings.

**Why.** PLAN 2026-08-03 Phase A: a lying surface is worse than an absent one. **The finding that
reshaped it:** the "claim layer" A2 and `unsourced_claim` both rest on is the assistant's own answer
sentences — 1,911 `AnswerClaim` rows, uncited 72% of the time on `llama3.1:8b` against 21% on Haiku —
not claims the corpus makes. So "Uncited in answers" is the honest label, and option 6 would measure
query history × model.

**Measured, read-only, $0.** Of 1,239 `unsupported` pieces, `is_claim_unit` rejects 306 (230 bare
enumerators, 63 colon lead-ins, 9 headings, 4 Sources blocks); on the concept-attributed links the
gap floor actually counts, 153 → 139 over the same 8 concepts — headings were never the main
contamination on this box, the unreadable citation form is (row 90). A dry `build_gaps` on today's
skeleton: isolated 3 · single_source 3 · under_connected 3 · unsourced_claim 8 · **thin_bridge 0**
(the 4 stored rows were all dead-end edges). The stored rows are stale anyway: 16 of 18 came from
graph `c97c…`, two versions before the current `027b…` (row 91).

**Rejected.** Filtering inside `segment_claims` — renumbers claims on every new answer and moves the
RG-012 citation markers. Word-count or phrase-list rules for "not a claim" (the audit's heuristic
had `< 6 words`, "Here is") — tuned to one model's habits. Flagging the lower-*degree* endpoint of a
bridge — degree is the vocabulary-sparsity signal RG-014 graded noise; side size is structural.
Relabelling the chips per ADR-040 option 4 ("mixed stance") — ADR-040 blocks the surfacing choice
behind the Node-B rebuild; this only removes the false "in corpus". Building A2 — see KL1b. Running
`build_gaps --apply` — a DB write, the user's call.

**What it opens.** The gap rebuild ($0, `python -m scripts.build_gaps --apply`) to make the stored
rows match. KL1b's decision. Row 90 before any "Uncited in answers" count is quoted. The colon rule
drops ~63 lead-ins, a few of which make a weak assertion ("The authors propose two main types:");
"Liu et al." fragments from abbreviation splits still count.

---

## 2026-09-16 (4) — Security S-2: one add checks at most 50,000 files, and the uploader states the limits it enforces

**What changed.** `library.add.expand_paths` counts while it walks and raises
`AddBatchTooLargeError` past `config.MAX_ADD_FILES` (50,000, `DOC_MAX_ADD_FILES`) — before sorting, so
a pick of a whole drive costs the walk up to the limit; `apply_add` refuses a longer explicit path
list before touching anything; both routes return a 400 whose `detail` is the sentence, which the
review sheet already shows as its alert. `GET /api/documents/accepts` serves `extensions`,
`max_file_bytes`, `max_archive_bytes`, `max_files_per_add` from `library.add.accepted_input()`.
The frontend loads it once when the API is up (`accept.svelte.ts` `loadAccepts`) and the three
places that listed formats — the Add documents dialog, the empty-library card, the drop overlay —
now show the served formats line and **"Up to 1 GB per file"** (`lib/library/formats.ts`, 5 tests).
Settings' formats line derives from the same registry. README Limitations, `docs/usage.md`,
`.env.example` (three knobs), `security.md` (S-2 done, S-3 next), ROADMAP row 60, the walkthrough
§1 (stated-limit and walk-cap rows), `feature-add-documents.md` constraint 6 amended, CHANGELOG.

**Why.** `docs/security.md` S3: `inspect` walked a picked folder with no cap on a request thread.
And the user asked that the 1 GB limit S-1 introduced be stated in the uploader and the docs.

**Choices.** N = 50,000: 5× the 10,000-document contract, since real folders hold files that are not
documents and Zotero sends one path per attachment through the same route; ~9 s at the measured
~11,000 files/s. **No byte cap (M)**: `inspect` barely reads contents and S-1 caps each file. The
numbers come from the API, never from a frontend constant, because all three limits are env knobs.
No env-var name in the UI — the docs name them.

**Verified.** `test_library_add.py` +6, `test_api_documents_inspect.py` +3 (400 names the cap; `/add`
refuses; `/accepts` follows a monkeypatched cap). Live preview: `/api/documents/accepts` returns
1073741824 / 50000 / nine extensions; the app requests it once per load; the state module holds it.
**The live check caught a bug:** the empty-library line lower-cased the whole limit sentence and read
"up to 1 gb per file" — fixed with `limitPhrase`, pinned by a test. **Not verified:** the rendered
lines themselves — all three surfaces exist only in the desktop window (`canAccept()` is false in a
browser); the walkthrough row covers them.

**Rejected.** Hard-coding "1 GB" in the frontend (wrong the moment the env knob is set). Adding the
limit to `/api/setup` (a readiness endpoint). An object-shaped 400 (`errorDetail` would
`JSON.stringify` it). A folder-depth limit (the grill chose full recursion, branch 3).

**What it opens.** `docs/QUICKSTART.md` §3 still describes adding documents through Settings (row
81). S-3 is next.

---

## 2026-09-16 (3) — Security S-1: oversized files and zip bombs are refused before they are opened, without re-extracting the corpus

**What changed.** `extractors.ingest_refusal(path)` returns a sentence, or `None`: a file over
`MAX_INGEST_FILE_BYTES` (1 GB, `DOC_MAX_INGEST_BYTES`); for EPUB/DOCX/ODT, an archive that cannot be
read, that declares more than `MAX_ARCHIVE_EXPANDED_BYTES` expanded (1 GB, `DOC_MAX_ARCHIVE_BYTES`),
or that holds an entry at ≥ `MAX_ARCHIVE_ENTRY_RATIO` (1000:1). It reads the size and the central
directory only. Two callers: `get_format_status`, so the add review sheet shows the sentence
verbatim as an `unsupported` row; and `ingest.cache.load_or_extract`, which raises
`IngestRefusedError` (a `ValueError`) just before extraction, so a file that reaches ingest without
the sheet is refused per file through the existing `document_error` path. `tests/unit/
test_ingest_size_caps.py` (12): a file over the cap and a synthetic zip bomb are each refused with a
sentence in the review sheet; an archive over its expanded cap; a damaged archive; real DOCX, ODT
and EPUB are not refused; extraction never runs on a bomb; the check is outside every format's
fingerprint closure; and CPython's `zipfile` truncating an entry at its declared size. Walkthrough §1
gains the row, `security.md` S-1 → done and S-2 next, ROADMAP row 60 → S-2, CHANGELOG `[Unreleased]`.

**Why.** `docs/security.md` S4: EPUB, DOCX and ODT are zip archives opened with no decompressed-size
accounting, so one file in the corpus could exhaust memory — the cheapest local denial of service.

**Two facts that shaped it.** (1) **Placement was forced by KI-48.** The extraction fingerprint
hashes everything reachable from `extract_to_markdown`; a check inside it would have marked all 98
cached documents stale and re-extracted the corpus. The fingerprints of all eight formats were
recorded before and after the change and are identical, and a test now asserts `ingest_refusal`
stays out of the closure. (2) **The declared sizes can be trusted as a bound.** Read in the CPython
source, `ZipExtFile._read1` ends `data = data[:self._left]`: output stops at the declared size
whatever the stream holds, and ebooklib, python-docx and odfpy all read through `zipfile`. So an
archive that lies about its sizes is truncated and fails its CRC — pinned by a test that forges a
central-directory size. The ratio check therefore catches the bomb *signature*; the expanded-total
check is what bounds memory.

**The numbers are structural, not tuned.** 1 GB sits ~33× above the largest document in this library
(31 MiB, 98 PDFs); deflate cannot compress real data past ~1032:1 — only a constant stream reaches
it — so 1000:1 is not a corpus constant. Both byte caps are env knobs; the ratio is not.

**Rejected.** The check inside `extract_to_markdown` (a corpus-wide re-extraction, KI-48). A new
`refused` verdict for the sheet — the frontend already renders the advisory verbatim beside a
warning icon with no fixed label, so a new verdict would be a wire and UI change for no visible
difference. Streaming decompression accounting — unnecessary once `zipfile`'s truncation is pinned.
A page cap — PyMuPDF opens a large PDF lazily and page count is not the memory risk the finding
names; noted as residual in `security.md` S4.

**What it opens.** Residual in S4: a central directory with millions of entries is bounded only by
the 1 GB file cap. S-11 (security events in the log) will give this control its own event; today a
refusal is visible only as the sentence and as `document_error`. Next security step: S-2, the walk
cap on `inspect`.

---

## 2026-09-16 (2) — Row 46: the ship gate asks three questions, reads its verdict with the app's own parser, and moves into the repo

**What changed.** **The harness is tracked** at `scripts/rg012/` — `rg012-run.ps1` and
`rg012-tier2.wsb`, which now maps that repo folder into the sandbox (it ran an untracked copy under
`C:\rg012-host\script`, invisible to diff review and to tests; the host `.wsb` was replaced and the
original kept beside it as `.bak-20260916-single-turn`). **Three turns:** one question per corpus
document, each in its own session so no answer is shaped by another's history; one
`out\run-<stamp>\` per run holding the log and `turn-N-stream.txt` / `turn-N-result.json`.
**Two verdicts:** `release_preflight`'s `rg012` check became `rg012_packaging` (Python not on PATH,
ingest produced chunks, every turn answered) and `rg012_citation` (≥1 turn cited, none *tried and
failed*, at least `RG012_MIN_TURNS` = 3 turns), and the citation half is judged on the host with
`synthesis.audit_citations` over the saved answers — each turn named *cited / unresolved / uncited /
missing*. The script's own lines are now labelled `(estimate)`. **The newest run on the artifact
decides**, and earlier runs on it are printed with their verdicts. **A reader bug fixed on the way:**
the harness appended to one log, the 2026-08-15 archive holds two runs, and the preflight read the
first installer line and *any* `TIER-2: PASS` in the file — a failed re-run after a pass would have
read as a pass. It now judges the last run in a log. Tests: 15 new in `test_release_preflight.py` (17 cases), and a
new `test_rg012_harness.py` pinning ASCII-only, the log lines and file names the script writes
against the ones the preflight reads, ≥3 distinct questions, one session per turn, and that the
`.wsb` launches the tracked script. `docs/RELEASE.md` §5 and `docs/desktop-packaging.md` §5 say how
to run it and how to read it.

**Why.** RIGOR_TODO RG-012 (2026-08-14): the byte-identical 0.5.1 installer failed its single cited
turn once in four runs, because `llama3.1:8b` cites all-or-nothing per answer (KI-36). A blocks-ship
gate that fails ~1 run in 4 on a healthy artifact trains its operator to re-run until green. Option
2 ($0, local) over option 1 (a paid Haiku turn) per the roadmap row.

**Verified without a sandbox run.** (1) `audit_citations` over the ten archived runs reproduces every
recorded verdict, including the two FAIL kinds: 08-06 run 2's `[Source 1: file.pdf]` → *unresolved*,
08-14 run 1 → *uncited*. (2) The tracked script is 0 non-ASCII bytes and 0 parse errors under
Windows PowerShell 5.1.26100. (3) Sections 5–6 of the tracked script were run verbatim on the host
with `Invoke-WebRequest` stubbed to replay archived streams — cited/uncited/unresolved gave
packaging PASS + citation FAIL; a timed-out turn gave packaging FAIL + "could not judge" — and the
preflight read those PowerShell-written, BOM-carrying files. That run also caught the estimate
claiming a citation PASS with a turn missing; it now says `NOT JUDGED`. (4) The live preflight on
this box reads the 0.6.0 run as `rg012_packaging PASS` / `rg012_citation FAIL — only 1 turn`, which
is the honest reading of that record.

**Rejected.** Keeping the verdict in PowerShell with a loop around the old regex — it leaves the
KI-35 shape (a restated contract) in place and untestable. Option 1, a paid turn — makes the ship
gate a billed path (KI-4). Accepting single-turn runs as a pass — a stale copy of the harness would
quietly bring the coin flip back. Letting any PASS on the artifact win, as before — that *is*
re-running until green. Keeping the harness local-only — the gate that decides a release was the one
piece of release tooling no review could see.

**What it opens.** Row 46 closes with **one sandbox run** of the new harness on the 0.6.0 installer:
only that shows three turns completing inside the timeouts on a clean box (needs Ollama reachable
beyond loopback — a host change, the user's call). RG-012 itself closes when two consecutive release
gates agree. `rg012-diag.ps1` / `rg012-ingest.ps1` stay untracked in `C:\rg012-host\script\`
(August diagnostics, not part of the gate).

---

## 2026-09-16 — After the break: the plan is made true again, fourteen dropped follow-ups become rows, and the crossover's merge row turns out to aim at the wrong knob

**What changed.** A docs pass after a post-release break; no code. **ROADMAP:** rows 35 · 37 · 47 ·
73 · 74 cited `docs/ui-checklist.md` §2/§3, sections that left that file on 2026-09-10 — repointed
at the rows they came from in the frozen `.claude/ui-checklist-archive-001.md`. Rows **53 · 54 ·
51 re-scoped** after the 2026-09-07 crossover review was checked against the code and the live
library: `CONCEPT_MERGE_COSINE` only feeds the read-only `suggest_concepts --near`, while the merge
that deletes rows (`curate_concepts --dedup --apply`) uses a hard-coded 0.9, another embedder and
another input; that merge deletes the dropped concept and `concept_hierarchy`'s cascade takes its
curated placements with it; `set_graph_include` accepts a field node; graph coverage counts deleted
documents; and nothing can write an `is_a` edge except a raw API call. So 53 is now *merges made
safe, then one threshold measured*, 54 adds the kind guard, 51 needs an `is_a` write path; the
sequence order is unchanged. KL1 also refreshes `knowledge-layer.md` §6 (presence is 534 chunk
keys, not 1,781). **Fourteen new rows (76–89)** for follow-ups that had dropped out of every plan
between the release and the restructure — the user's own 2026-08-24 asks (range selection in chat
select mode, draggable dialogs), the Windows CI job + shared `empty_library` fixture KI-58 said row
60 carried and it did not, parallel extraction never verified in the installed build, the README
GIF / DEMO refresh, the process-scoped `OLLAMA_HOST` route, the stale frozen sidecar that `tauri dev`
spawns, the 2026-09-10 review's hygiene moves, and three user decisions (`docs/sprints/`, the
encoding guard for local docs, an unfilled `.claude/NORTH_STAR.md`). Rows 29 and 75 absorbed two
small UI loose ends. **KI-48** closed — fixed 2026-08-25 (per-format fingerprint) but headed OPEN
for three weeks; body moved verbatim to `docs/archive/KNOWN_ISSUES-resolved-002.md`. **CHANGELOG:**
`## [Unreleased]` added; 0.6.0 dated 2026-09-04, the tag date, like every earlier release (it said
the build date). **Conventions:** `scripts/conventions.toml` no longer claims the cpc 1.2.3 key set
as the vendored version (the drop is 1.8.0, re-vendored 2026-08-22 with no record here); the rotate
command in this header and in that file is the shim form. `docs/local-only.md` lists four gitignored
paths the docs cite. **cpc:** tickets T-013 (generate/keypoint extras spawn without the gate env),
T-014 (keypoints route to a skill the model cannot invoke), T-015 (a re-vendor leaves no trace in
the consumer), T-016 (lift `release_preflight` — the user's 2026-09-02 ask); T-011/T-012's version
corrected before their first commit. All uncommitted in the cpc checkout.

**Why.** The user asked what was pending after the break and whether the crossover review held
anything for the concept branch. Three independent read-only audits plus a re-read at source of
every claim that decides a row; the full record, with file:line, is
`docs/reviews/REVIEW_2026-09-16_post-break-state.md` (local).

**Rejected.** Sweeping `CONCEPT_MERGE_COSINE` as row 53 said — it would have measured a preview no
merge reads. Re-ordering the user's sequence to put merge safety first — the destructive path is a
manual `--apply` nobody has run, so it only has to land before 51 writes curated edges, which the
existing order already guarantees. Fixing the orphans inline (the OLLAMA route, the EPUB parser, the
sidecar comment) — the user asked for rows, and a docs pass that also changes code is two commits
pretending to be one. Re-vendoring cpc — its HEAD carries unreleased fixes and would stamp 1.9.1.
Pruning `build/` `dist/` and the backups — deletes data; row 88 is the user's call.

**What it opens.** Session 1 of the sequence is unchanged (row 46 + S-1). Row 53 may take two
sessions; the sequence row says what moves if it does. The ~5 GB prune, the Docker upgrade and the
three decisions in row 89 wait on the user.

---

## 2026-09-10 (2) — Working docs get folders, the DEVLOG keeps twenty, the release checklists are stale by default, and security becomes one step per session

**What changed.** Second session of the day, on the user's reading of the review. **Folders
(ADR-051):** the five live plans moved to `docs/plans/`, the two reviews to `docs/reviews/`; each
folder has a tracked README; the dated files stay local-only by *pattern* (a folder-level ignore
would have taken the README with it); fifteen living files repointed (ADRs 028/030–033/046, the
add-documents spec, ROADMAP, ui-checklist, knowledge-layer, local-only, README, the three local
trackers) — append-only logs and the CHANGELOG keep their old paths as history. **Entry budgets:**
`devlog_max_entries = 20` in `scripts/conventions.toml`, `cpc-rotate --file devlog --write` moved
56 entries (2026-08-15 → 2026-08-30 (8)) into archive 006 byte-verified, and
`tests/unit/test_doc_sizes.py` pins the same number so a clone or CI sees it. **Release:** the two
checklists are assumed stale — `docs/RELEASE.md` §0 is now the first step (list what shipped, add
rows, bump), and `release_preflight` gained `check_checklists_refreshed` (both files must carry a
commit or an uncommitted edit since the previous tag; six tests, a pinned file list). **Security:**
`docs/security.md` restructured — §0 how the file is worked, §4 an ordered plan S-1 … S-12 sized
one step per session with a `done when` per step, §6 the periodic full check (eleven lenses, each
with its command) whose log is `.claude/REVIEWS.md` row 7; S-12 (Dependabot + SHA pins) marked as
the user's call. **Roadmap:** re-sequenced to the user's order — gaps first with one security step
per session (row 60 is now the standing "current step" row), the knowledge layer made honest
(KL1 → 53 + 54 → 51 → KL4 → KL2), then row 25, then a **graph re-pass** (new row 75: 52 · 57 ·
50 · 43 · a UX pass); a twelve-session sequence table; row 62 done → archive. **cpc:** tickets
T-011 (the folders) and T-012 (the four consumer-grown conventions) opened with `cpc-ticket` in
the cpc checkout — uncommitted there, for the next cpc session.

**Why.** Each is a user decision from the second session: folders for plans and reviews as the
standard going forward; twenty entries as the DEVLOG standard; checklists that "would never
otherwise be up to date"; security worked a little at a time with a thorough check at the end and
a standard place to log it; and the order gaps → knowledge → ingestion → graph.

**Rejected.** Ignoring the two folders wholesale (kills the tracked README); judging checklist
freshness by the `updated:` header (bumpable without a refresh — git history, like
`artifact_fresh`); hand-cutting the DEVLOG on a day boundary (the tool rotates by count and
verifies bytes; the split day is recorded in both headers); a separate security log file (the
REVIEWS ledger already has the shape and the "not covered" discipline); editing cpc's CONVENTIONS
directly (a ticket is the channel the ledger asks for).

**What it opens.** The DEVLOG header's archive range is still updated by hand (cpc T-003). Session
1 of the sequence is row 46 + security step S-1 (ingest size caps). `docs/sprints/` is laid by cpc
and unused — the T-011 answer may settle it. Stamps: the first draft of this session said
2026-09-11 in fourteen files; corrected to 2026-09-10 before staging.

---

## 2026-09-10 — Whole-project review: the roadmap goes per feature, the security floor is written down, and a ninth version file turns up at 0.4.2

**What changed.** A review session, not a feature session. Three read-only audits (repo health,
security surface, file architecture) and a read of every coordination doc produced
`docs/REVIEW_2026-09-10_project-review.md` (local, ADR-029) and these tracked changes:
`docs/ROADMAP.md` restructured into one table of **open** rows with a `Feature` column and twelve
feature sections — the 87 done rows moved **verbatim** to `docs/archive/ROADMAP-done-001.md`; the
UI checklist's queue and verification debt folded into it (`docs/ui-checklist.md` keeps only the
per-feature gate); three closed working docs archived under the new, gitignored
`docs/archive/local/`; `docs/security.md` (threat model · what holds · eleven ranked findings each
with a roadmap row · the deterministic floor · a cpc gate proposal); `docs/release-ux-checklist.md`
(seven surfaces to drive in the installed build, with a record table whose "not driven" column is
mandatory) wired into `docs/RELEASE.md` §5b and the gates table; `docs/architecture.md` corrected
(the taxonomy layer is *built*, ~2,600 lines; the keyword arm is the on-disk FTS5 index, not the
retired in-RAM BM25; the legacy `tests/eval/run_eval.py` dropped from the layout).
Code and config: the lint/type/SAST gates now cover `apps/` and `scripts/` in CI and pre-commit
(they were `src/`-only — the HTTP boundary was gated by nothing; two fixes for mypy: the
`sse_starlette` import moved to the package root, and `routers/updates.py` names its `UpdateState`);
`[tool.bandit]` exists so `-c pyproject.toml` means something; `npm audit --audit-level=high` is a
frontend CI step and its one high (`nanoid` < 3.3.18) is fixed in the lock;
`tests/unit/test_desktop_security_config.py` pins the Tauri CSP (set, never unsafe, network only to
the sidecar) and the exact capability set; `docker-compose.yml` binds the port to loopback;
`scripts/conventions.toml` registers `gh run list --branch main --limit 1` in the session-start floor
and the security floor + walkthrough as sprint-close checklist lines; the dead `langchain.*` mypy
override is gone; `docs/features/` (a template nobody used, not laid by cpc) is removed. Local:
`.claude/CONTEXT.md` 316 → 245 lines (the July narrative and the resolved questions moved verbatim),
`.claude/KNOWN_ISSUES.md` 1,726 → 867 (18 closed bodies → archive 002, index rows auto-derived from
headings — refine when next touched; **KI-58 filed**: the CI-blind shape), `.claude/RIGOR_TODO.md`
history block archived and its "gate fails" line corrected (no gate exists).

**Why.** The planning surfaces disagreed (the ROADMAP said row 24 next, the UI checklist said P2;
87 of 108 rows were done and restated the DEVLOG); the phase framing had stopped describing the
work; security had never been written down as a threat model, so every finding would have been
"critical" or ignored; and the session-start protocol never read CI, which is how `main` stayed red
for seven pushes.

**Found on the way, and the one that matters:** `apps/desktop/package-lock.json` records the
project version twice and npm rewrites it only when npm runs — it sat at **0.4.2 through v0.5.0,
v0.5.1 and v0.6.0**. Same shape as the Cargo files: a version-carrying file nothing was looking at.
`release_preflight.collect_versions` reads both fields now, the pinned-file-list test carries them,
and `docs/RELEASE.md` §1 says nine places, not eight.

**Rejected.** Making `pip-audit` blocking today (66 advisories; it would redden every run until
row 61 triages them — the CI step stays advisory and the `sprint-close` checklist names it);
adding Dependabot and SHA-pinning the actions (bot PRs are the user's call); DOMPurify, the CSP
`form-action`/`base-uri`/`object-src` additions, `TrustedHostMiddleware` and the prompt fence
(each a code change that needs the installed build or an eval run to verify — rows 60/61, not a
review session); rotating the DEVLOG now (3,905 of 4,000 lines — due at the next close);
hand-writing the 18 KI index rows (verbatim archive first, judgment when each is next touched).

**What it opens.** Row 46 before the next release (the citation gate is a coin flip); a Windows CI
job (KI-58's other half — the shipped platform is the one no job runs on); the `pins.py` reachability
question; the 22 headerless specs; `docs/sprints/` is laid by cpc and unused since 07-18 — decide.
The review's §11 lists what it did not cover; rows 2 and 3 of `REVIEWS.md` stay `never`.

---

## 2026-09-07 (3) — CI on `main` was red for five days and seven pushes, including the release; two test-only fixes

**What changed.** `tests/unit/test_source_view.py`: the two "unknown document" tests get an
`empty_library` fixture — a temp SQLite with `Base.metadata.create_all` swapped into
`db.session._engine/_SessionLocal` via `monkeypatch`. `tests/integration/test_external_metadata.py::
test_matching_survives_a_differently_written_path`: the "differently written" variant is now
`parent / "sub" / ".." / name` (upper-cased on Windows) instead of a `/`→`\` swap. Both files,
+41/−4, test-only; shipped code untouched.

**Why.** `gh run list --branch main` shows every run since `3230703` (2026-09-02) red at `pytest
with coverage`, last green `5405d44` (2026-08-28) — through the merge, the tag and the docs commit
this morning — while the suite was 2,389/0 on the Windows dev box. Two causes. (1) The source-view
tests read the real `session_scope()`: locally `data/library.db` has the schema so the lookup
misses and passes; on CI `SQLITE_URL` names a file SQLAlchemy creates empty on first connect, so
`no such table: documents`. Reproduced here with `DOC_DATA_DIR=<empty dir>` on the old file (both
fail with exactly that error) and the fixed file (24/24). (2) `pathkey` is
`normcase(abspath(...))`. The old variant, on POSIX, produced a relative filename with literal
backslashes — a different file, so the catalogue entry never matched, red since the test was
written on 2026-08-31; on Windows `str(Path)` contains no `/`, so the swap never fired and the
test passed without testing anything. `abspath` collapses `..` lexically on both `posixpath` and
`ntpath` (checked), so the new variant exercises the normalisation everywhere.

**Rejected.** *An autouse DB-isolation fixture in `tests/conftest.py`* — the right end state, but
it touches every test that reaches the database, which is not a fix to make blind on the day the
suite is discovered to differ by platform. *`skipif(os.name != "nt")` on the path test* — hides
the platform difference instead of testing it. *Dropping the case half* — kept on Windows only,
because `normcase` folds case only there and a case-different path on ext4 *is* a different file.

**What it opens.** Nobody looked at Actions for five days while three baton entries said "every
gate green locally": the third instance of the CI-blind shape (uv.lock at 0.4.0 · the container
until 2026-09-04 · this) and worth a KI with a session-start check (`gh run list --branch main
--limit 1`). The two DB-isolating fixtures (`env` in the integration file, `empty_library` here)
want a shared one in `conftest.py`.

---

## 2026-09-07 (2) — v0.6.0 published; a README tell pass; §7b learns what the publish command actually does

**What changed.** The GitHub release for `v0.6.0` exists and is Latest: id 383972486, asset
`Provenote_0.6.0_x64-setup.exe` at 1,648,765,716 bytes with GitHub's own digest
`sha256:2280…5bc4` equal to the local hash, body = `.claude/release-notes-0.6.0.md` (11,099
chars, ends in the SHA-256 line). `releases/latest` now answers 0.6.0, so ADR-044's update check
on a 0.5.1 install has something to compare against. `README.md`: a pass for writing tells — the
contrastive slogans ("measured, not asserted", "estimates, not promises", "costs one paper, not
the library"), the aphorisms ("megabytes tell you almost nothing"), three "deliberately", one
"surprisingly", and two headline phrases added earlier today ("Your files, added your way", "the
answer can show you its source"); every number and the bold-lead structure stayed. `docs/RELEASE.md`
§7b: the command now carries the asset and `--verify-tag`, and the section says where the notes
live, why they need a status header, and that the command drafts, uploads, then publishes.

**Why.** Two `gh release create` runs raced: one from this session, one started a minute earlier
from the Run button on the command shown in the previous message. The earlier one drafted,
uploaded and published at 09:36:44Z; this session's uploaded a second copy to its own draft and
failed the final flip with HTTP 422 "Release.tag_name already exists", after which gh deleted its
draft — zero drafts remain, one release object, correct. The runbook did not say the command drafts
first, so a draft with zero assets read as a failure mid-way, and nothing said to check
`gh release list` before running it a second time.

**Rejected.** Killing the second process once both were seen — one had already published by then,
and gh's own cleanup handled the loser. Editing the entry below to say "published" — append-only;
this entry is the correction.

**What it opens.** `README.md`, `docs/DEVLOG.md`, `docs/RELEASE.md` are uncommitted. The GIF
(storyboards in the baton) and the README alt text that goes with it. `docs/DEMO.md` and the KI-48
heading, as noted below.

---

## 2026-09-07 — 0.6.0 release notes and README brought level with the tag; the release object itself is the user's command

**What changed.** `README.md` re-read against the tagged code, the way `docs/RELEASE.md` §2 asks
of the CHANGELOG — and two Limitations bullets were false. The OCR one said a pure scan "is
unreachable" and that OCR is "deliberately not built"; KI-47 measured the opposite (PyMuPDF uses any
`tesseract` on PATH, 0 → 34,600 characters on the same file), and the bullet now carries the
corrected 0.6.0 text. The reference-links one still described the surname+year matcher, "4 links
where 16 are stored", and named KI-45 as *next* — 0.6.0 requires title agreement (16 → 41 links,
DEVLOG 2026-08-26), so it now says that, plus the real remaining limit: links are computed at first
read and not revisited (`extract_citations --reresolve` refreshes them; not a button). The first
ingest is no longer "single-threaded" (two workers by default). **Added:** a Windows-installer route
at the top of Quick start — the README had no download pointer at all through two releases that
shipped an installer; the source-pane / *In context* bullet; the add-documents / Zotero /
per-part re-run bullet; the graph's coverage line; and a condition under the indexing-time table,
whose two OCR rows are only true with `tesseract` on PATH. Status → v0.6.0, 2,389 tests.

**The release notes** are drafted in the 0.5.1 body's shape at `.claude/release-notes-0.6.0.md`
(gitignored, like the 0.5.1 body was never committed). They carry the corrected OCR limit and say
in one sentence that the tagged `CHANGELOG.md` has it wrong; state the RG-012 verdict in two halves
(packaging strong; citation one sample — 1 failure in 4 byte-identical runs on 0.5.1, RIGOR_TODO
2026-08-14); and explain why a 0.5.1 library shows every document as *changed* after upgrading
(`is_cache_fresh` compares the recorded fingerprint against the per-format one 0.6.0 computes, so
every 0.5.1 cache reads stale — and ADR-047 means that re-read no longer orphans sidecars). The
upgrade path itself was not gated and the notes say so. Gates re-run at HEAD, which equals the tag
on `src apps scripts tests`: pytest **2,389/0** (9:33) · node:test 257/257 · mypy 98 files clean ·
svelte-check 219 files 0/0 · `docs_check --strict` OK · docs-encoding guard 5/5. Installer
SHA-256 `2280180840548023735044020e9a0e05ec418d06522909af66dbe4db9a5c5bc4`, 1,648,765,716 bytes.

**Why.** `docs/RELEASE.md` §7b: since ADR-044 the app reads `releases/latest`, so a pushed tag with
no release object is invisible to every install — 0.5.1 users cannot learn 0.6.0 exists until the
object is there, and `gh release list` shows only 0.5.1 and 0.4.2. The README drifted the same way
the CHANGELOG had: its Limitations were written for 0.5.0 and nobody re-read them at 0.6.0.

**Rejected.** *Re-tagging onto `b5ad5de`* to carry the corrected CHANGELOG — the tag is public and
peels to the source the installer was built from; moving it breaks the tested-equals-tagged diff §6
depends on. The correction lives in the release body, which is what people read. *Creating the
release from the agent session* — `gh release create` was refused by the permission classifier as
a publish, which is the right call: §7 makes publishing the one deliberate, irreversible step. The
exact command is in the baton. *Committing the notes under `docs/`* — the release page is the
body's home and the CHANGELOG is the durable record; a third copy rots.

**What it opens.** The release is one command away. The README GIF is a release stale (the 0.5.0
storyboard: no source pane, no add-documents) — three slideshow storyboards are in the baton, per
the user's ask. `docs/DEMO.md` still calls Connections *scored* (ranked since 0.5.1) and does not
mention the source pane — not touched. `KNOWN_ISSUES.md` still heads KI-48 as OPEN while DEVLOG
2026-08-25 (3) fixed it at the cause; the heading wants reconciling.

---

## 2026-09-04 (2) — CI builds the container, and checks the two things a green build does not prove

**What changed.** A third job in `.github/workflows/ci.yml`, alongside `ci` and `frontend`: free
~10 GB on the runner, build the image through buildx with the GitHub Actions cache, then assert
**(a)** torch is the `+cpu` wheel with zero `nvidia-*` distributions and **(b)** `apps.api` and
`doc_assistant` import inside the image.

**Why.** Nothing referenced the Dockerfile between `a052703` (2026-08-01) and today, so the
container was the one of this project's three shipping paths — desktop installer, source checkout,
headless image — that no gate touched. The pinned `ghcr.io/astral-sh/uv:0.12.1` base had never been
exercised on any machine (the dev box runs uv 0.11.14), and a `uv.lock` that had drifted would have
failed `uv sync --locked` in the image and surfaced only when somebody needed the container. Which
is exactly how it came up: the user asked whether Docker still worked, and the honest answer was
that nothing had checked since August.

**The two assertions are the job, not the build.** A green build says the layers assembled. It does
not say the image is the right one: `pip install ".[cpu]"` ignores `[tool.uv.sources]`, resolves
torch from PyPI, and that Linux wheel bundles CUDA — several GB of `nvidia-*` in an image with no
GPU, which still builds and still runs. KI-34 is the standing version of this lesson at the desktop
end: an artifact that started cleanly, served `/api/health`, reported a healthy chunk count, and
could not read a single PDF.

**Two bugs found in this job while writing it, both by running it rather than reading it.**

1. The nvidia count was `ls /app/.venv/lib/python*/site-packages | grep -c "^nvidia" || true`, which
   prints `0` — a **pass** — when the glob matches nothing at all. Relocating the venv would have
   turned the check off silently instead of failing it. It now asks `importlib.metadata` inside the
   image, and that the scan is not blind is itself checked: pointed at `torch`, it fails with
   `packages found: ['torch']`.
2. Rewriting it to a single line made the whole workflow **unparseable YAML** — a plain scalar
   cannot contain `": "`, and the f-string is `f"nvidia packages in the image: {nv}"`. It parsed
   before the edit and not after; only re-validating caught it. Both `run:` steps are block scalars
   now, with the reason recorded inline.

**Verified by extracting the commands from the parsed YAML and executing those exact strings**
against the built image, rather than retyping them: `torch 2.12.0+cpu | nvidia packages: 0` and
`apps.api ok; doc_assistant 0.6.0`.

**Rejected.** *Booting to a green `/api/health`* — first run downloads the embedder and reranker,
which is why the Dockerfile's `HEALTHCHECK` carries a 300 s start period; that belongs in the
release gate, not on every push. *A path-filtered trigger* — the Dockerfile's inputs are
`pyproject.toml`, `uv.lock`, `src/`, `apps/api/` and `scripts/`, which is most of the repo, so the
filter would have saved nothing and hidden the cases it did skip. *Plain `docker build` with no
buildx cache* — the dependency layer is ~8 minutes and the Dockerfile already orders its `COPY`s to
make it cacheable; not using that would have made the job the slowest thing in CI for no reason.

**What it opens.** The image is ~6.3 GB and the GHA cache is capped at 10 GB per repository, so the
`mode=max` export may thrash once other caches compete. If it does, the fix is `mode=min` or
dropping the cache export and paying the eight minutes. Left as-is because the first failure will
say so plainly, and guessing at it now would be tuning against an imagined problem.

---

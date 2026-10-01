<!-- status: active · updated: 2026-10-01 · class: append-only -->

# DEVLOG — doc_assistant

Real-time development log. One entry per logical change.
Append only — never edit past entries.

Format: What changed | Why | Rejected alternatives | What it opens

> **This file keeps at most 20 entries** (`devlog_max_entries = 20` in `scripts/conventions.toml`,
> cpc rule 13b, user standard 2026-09-10; `tests/unit/test_doc_sizes.py` pins the same number).
> Past 20, one batch cuts it to the newest 10 (`devlog_rotate_to`, cpc 1.12.0, since 2026-09-29).
> Rotate with `python tools/conventions/rungate.py rotate --root . --file devlog --write` (the shim —
> calling `tools/conventions/cpc/rotate.py` directly fails under `.venv`; corrected 2026-09-16) — it moves the
> oldest entries **verbatim** into the highest-numbered archive, or starts the next one when that is
> past `archive_max_tokens`, and verifies the bytes; then update the range below by hand (cpc ticket
> T-003), and check a new archive's title (cpc 1.12.0 titles every new one "archive 001"). A day may
> be split across two files at the cut. Every archived heading is listed in
> [`docs/archive/DEVLOG-INDEX.md`](archive/DEVLOG-INDEX.md). Older entries, newest-first, unedited:
> **2026-09-07 → 2026-09-17 (1)** in [`docs/archive/DEVLOG-archive-007.md`](archive/DEVLOG-archive-007.md)
> (rotated 2026-09-30, the first batch: 11 entries) ·
> **2026-08-12 (1) → 2026-09-04 (2)** (the first 2026-09-02 and 2026-09-04 entries are unnumbered) in [`docs/archive/DEVLOG-archive-006.md`](archive/DEVLOG-archive-006.md)
> (rotated 2026-09-04, 2026-09-10, four times on 2026-09-16, on 2026-09-17, 2026-09-18, 2026-09-20, twice on 2026-09-21, twice on 2026-09-22 and on 2026-09-29) ·
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

## 2026-10-01 (5) — The test suite no longer reaches the working library's database (KI-60)

**What changed.** `tests/conftest.py` gains a session fixture that binds the database layer's
default engine to a throwaway file for the whole run. A test that reaches the database without
swapping the engine now lands on an empty database, as it does on a fresh checkout.

**Why.** `db.session` binds its engine to the configured `library.db` at import. Two route test
files (`test_reingest_routes.py`, `test_ingest_progress.py`) enter the app's lifespan without
redirecting it, and the lifespan's first act is the schema migration. On 2026-10-01 a full run
therefore added row 94's new column to the working library, before the app had ever been started
on that code. Read back afterwards: integrity check `ok`, no form marked, and 593 labels and 404
alias rows identical to the backup of 2026-09-01 — the column is the only change, and it is the
one the app would have made at its next start. Reproduced against an empty data directory:
before the fixture those two files create a `library.db` there, after it they do not.

**Rejected.**
- Stubbing the migration in the two files: it fixes today's two and leaves the next unredirected
  test free to do the same.
- Pointing the whole suite at a temporary data directory (`DOC_DATA_DIR`): the right end state,
  but two tracked files live under `data/` and which tests depend on that directory is unmeasured.
- A session-end check that the working database's modification time did not move: it would fail
  whenever the app is used while the tests run.

**Measured after the fix** (one full run, every file under `data/` compared before and after):
`library.db` untouched. The run still added 58 extraction-cache files for the tests' own documents
(`cache/referenced/`), two session logs under `exports/` and one row in the vector store's
write-lock table; the store's content is unchanged (18,011 embeddings).

**What it opens.** This is the database half of ROADMAP 76's shared-fixture item, and of KI-58's
second open half. The debris above is KI-60's open part: nothing the library holds changes, but it
accumulates, and a test that can open the working vector store could write to it. A temporary
data directory for the suite would close it, and belongs to row 76.

---

## 2026-10-01 (4) — A concept has a name and exact or broad forms, and the other rows are terms (ROADMAP 94, ADR-054); CI's secret scan can fail (S-13)

**What changed.**
- **A form is exact or broad.** One additive column, `concept_aliases.breadth`; unset reads as
  exact. `concept_skeleton.load_concepts()` — the loader that presence, edges and the gap list's
  claim attribution count through — leaves a broad form out. `load_broad_forms()` and
  `match_broad_presence()` match it on its own, and a node carries the documents only a broad form
  reaches (`broad_doc_ids`, `broad_forms`) beside `doc_ids`, never in them. The name is always
  exact: a broad mark on an alias that repeats the label is ignored.
- **Concepts and terms.** Stored definition candidates, merge suggestions, `is_a` proposals and
  field-placement proposals read concepts (rows with `graph_include`). In all four runners
  `--include-terms` reads every row and cannot be applied (for placement it replaces
  `--all-concepts`, kept as an alias). A term refuses a definition write (409) and returns the
  sentences found for it without storing them.
- **The name on every screen.** `written_forms.shown_labels` is the one helper. The vocabulary
  search, the gap list, the taxonomy view and Manage keywords carry `written`, as the graph did.
- **Manage keywords** lists concepts apart from terms. A concept's row has an exact | broad control
  per form. Delete asks first, says what goes with the row (forms, the chosen definition and the
  other options, field placements, gap decisions, the graph's documents) and offers "Take off the
  graph instead". The graph's Edit button, and a term's Manage keywords button, open the view on
  that row.
- **The graph says what it is behind on.** A build records a fingerprint of each concept's name
  and forms in `skeleton.meta["forms"]`; the view compares it with the live vocabulary and names
  the concepts that changed. A graph built before the record says it cannot tell, and offers the
  rebuild.
- **S-13.** CI's secret step is `python -m scripts.secret_scan_gate` (`just secret-scan`): the
  `detect-secrets` hook over every tracked file against a temporary copy of the baseline. It fails
  on a secret the baseline does not record, never rewrites the tracked baseline, makes no network
  call, prints a finding's type and place and never its value, and exits 2 when it could not look.
  It is stricter than the local hook in one way: the hook can ask a provider whether a candidate
  key is live and drop it if not, and the gate keeps it. It adds about 19 seconds to a CI run.

**Why.** ADR-054, accepted 2026-10-01: a count should say what it rests on, and until now it could
rest on a row nobody read or on an alias that means something wider. S-13: the step CI ran,
`detect-secrets scan --baseline`, is the command that writes a baseline — it recorded a new finding
and exited 0 (found 2026-09-30).

**Measured.**
- **Nothing moved.** `build_concept_skeleton(apply=False)` on the working library with the
  committed code and with this change: the same graph version (`cb99a7f9f36545da`), 13 concepts,
  30 edges, 86 concept–document pairs, equal field by field
  (`tests/eval/baselines/names_and_forms_2026-10-01.md`).
- **One mark, on a copy of the library.** `distillation` marked broad: `knowledge distillation`
  11 documents → 4, with 7 listed beside; no edge lost, six of its edges on fewer shared chunks;
  3 documents leave the graph's coverage. The graph named the concept until it was rebuilt. This
  is one mark to exercise the path. The user's reading of the forms has not happened.
- **Terms.** `viral` and `SPECTER` opened in the app on the copy: usage passages for both, two
  defining sentences for `SPECTER`, no control to choose one, and no row stored for either.
- **The count to classify is 22, not 31.** The 13 concepts hold 31 alias rows; 9 repeat the name.
- **S-13.** In a scratch repository the published example key fails the gate (exit 1) and passes
  the command CI ran before (exit 0, the key written into the baseline). The repository passes:
  834 files, about 19 seconds.
- Checked live on the copy in light and dark and at 375 px: no failed request after a clean load.
- The suite: 2,622 passed, coverage 92.8%. The gate's test file gained one test and had one
  rewritten after that run, and passes on its own (24). The desktop's 284 tests and its type
  check pass.

**Rejected.**
- Adding a broad form's documents to presence behind a flag: edges and gaps would count them.
- `graph_version` as the "forms changed" signal: it fingerprints the result, and the gap and
  epistemics sidecars key their own staleness on it.
- Calling a graph built before the record "changed": there is nothing to compare, so it says that.
- Deleting a row that has nothing attached without asking: one rule for every delete.
- For S-13, failing on `git diff --exit-code .secrets.baseline` after the old scan: it turns the
  scan's side effect into the signal, and a recorded line that only moved would fail it.
- Running the hook on the tracked baseline in CI: the hook rewrites the file it is given.
- Naming the `just` recipe `secrets`: the scanner reads that name and the command under it as a
  keyword with a value, so the commit hook failed on the recipe (and on a sample report line in
  the gate's own test). The recipe is `secret-scan`.

**What it opens.**
- The user's part: mark the 22 forms and name the concepts in Manage keywords, then rebuild.
  RG-032 is measured after that, before ADR-053 decision 4's signals (the next session).
- The working library's graph predates the record, so it will show the "cannot tell" notice until
  its next rebuild. That rebuild also applies the case-aware matching of 2026-09-30 (`cre` 7 → 6).
- The Graph tab offers Rebuild only when it reports itself behind. A standing control is a design
  choice left open.
- Found on the way: the test suite migrated the working library (the next entry, KI-60).
- `docs/specs/feature-tag-families.md` calls the hidden group "glossary-only"; the view says
  "unused" now, and the spec carries a dated note.

---

## 2026-10-01 (3) — ADR-054 accepted, with the user's amendment: a base vocabulary per field is a source of terms

**What changed.** No code.
- **ADR-054 is accepted** (user, 2026-10-01). It gains *Amendment 2026-10-01*, and its Status and
  its "to decide" line now say what was accepted. A curated base
  vocabulary for a field is a third source of rows beside the user's additions and the keyword
  extractor; its entries stay terms until the user takes them on. One sentence of the Decision is
  widened: a term is a candidate row nobody has taken on, whatever proposed it, because a base
  entry may not occur in the library at all.
- **ROADMAP KL2** names it as the concept-level half of "the taxonomy as the reference class for
  expected coverage"; row 94 (ADR-054's build) moves from *gate first* to *planned — next*; the
  index line in `docs/decisions.md` follows.

**Why.** The user, on reading ADR-054: a curated default vocabulary for a given topic "would go
hand in hand with the taxonomy"; otherwise "the text is clear". With the amendment in front of
them the user accepted the ADR: "Okay, let's go with this". The idea continues the user's
question of 2026-09-21 (experts have already listed and defined a field's terms), which ADR-053
answered for definitions only. It also supplies what `docs/knowledge-layer.md` §1 says a gap needs:
an expected structure to deviate from. The taxonomy says which fields exist; a base vocabulary says
what a field contains.

**Rejected.**
- Making a base list's entries concepts by default: that is the bulk promotion of 2026-07-05 with
  a better source, and ADR-054 exists because 344 rows became concepts without being read.
- Importing a vocabulary whole: ADR-028 decision 7 measured it as "a facet that partitions
  nothing" (30,000 MeSH descriptors for a small library).
- Rewriting ADR-054's Decision in place: the file is append-only; the amendment states the one
  widened sentence.
- Designing the feature here: which vocabularies, their licences and sizes, and whether a topic's
  list is taken on entry by entry or as a reviewed whole belong to their own ADR.

**What it opens.** The ADR-032 grill (KL2) decides what the base list is for and how it is taken
on; 93c's local copies are its data step. Known limit before any build: of the 19 priority
concepts, expert vocabularies define 8, gloss 6 and have nothing for 5
(`tests/eval/baselines/reference_vocabularies_2026-09-21.md`), so a base list is solid for the
established fields and thin for the youngest ones. The acceptance opens row 94: names and forms on
the 13 concepts. CI on `7357fcc`, the commit that carried the dependency fix and ADR-054 as
proposed, is green.

---

## 2026-10-01 (2) — The vocabulary is read against the library; concepts and terms, names and forms (ADR-054, proposed)

**What changed.** No code. Three documents:
- **`tests/eval/baselines/vocabulary_shape_2026-10-01.md`** (new): a read-only snapshot of the 357
  text-bearing concept rows against the library's text — where they came from, how far each
  spreads, what the words beside a one-word label are, what each alias of the 13 graph concepts
  contributes on its own, and what the bibliography cut removes.
- **`docs/decisions/ADR-054-concepts-and-terms-names-and-forms.md`** (new, proposed) and its index
  line: a *concept* is a row the user has taken on (`graph_include`, 13 today) and the other 344 are
  *terms*; a concept's label is its name, and each matched form is *exact* (counts as presence) or
  *broad* (counted beside it). One additive column, no second table. Not built.
- **`docs/ROADMAP.md`:** row 94 (ADR-054's build) and row 95 (the bibliography cut, body-text
  presence); row 93's abbreviation signal moves after 94.

**Why.** The user, before the abbreviation and fragment signals of ADR-053 decision 4 were built:
"we will need to think more about vocabulary". The open items — a label shorter than its concept, a
homograph, an alias that means something else, a label that is an author's surname — turned out to
share one cause: nothing in the data says whether a row is a meaning someone chose or a string an
extractor produced. The four choices were put to the user one at a time, each with a recommendation
and rows from the library, and the user took the recommended option each time.

**Measured** (104 documents, 8,861 parent chunks; the shipped matcher; no model):
- **Provenance.** 13 rows are hand-made and are the only ones on the graph. 344 were created in
  one promotion on 2026-07-05; 113 of them are a keyword of no document in today's extraction.
- **Spread.** 197 of the 344 occur in the prose of at most one document, 30 in none; 18 occur
  nowhere in the text (`comput vis`, `koonce emerson`).
- **Neighbours.** Of 161 one-word labels with at least 5 prose mentions, 22 are followed by the same
  word in at least half of them: 16 are the front of a longer term (`pose` → "estimation" 326 of
  498) and 5 are surnames followed by "et al.". The user's three examples do not resolve this way:
  "viral vector" is 7 of 103 uses of `viral`.
- **Forms.** `distillation` alone reaches 7 of the 9 documents counted for
  `knowledge distillation`. 89 of the 99 `passage ranking` mentions are a benchmark's name. "Cre
  recombinase" occurs once against 255 for `Cre`.
- **The cut.** 113 of 1,103 concept–document pairs exist only past the bibliography cut. The cut
  drops everything after the References heading: in 39 of 76 cut documents that includes content
  sections (appendices, methods, sections emitted late by the PDF extractor), about 555,000
  characters or 4.5% of the library — text the keyword extractor, the definition scan and the
  written-form vote never see.

**Decided by the user.** (1) The 13 are the concepts; the 344 are terms. (2) Name plus exact and
broad forms. (3) The names-and-forms build comes before decision 4's signals, which then propose
into it. (4) Reference-list mentions stop counting as presence after the next release, in one
measured change with the cut's fix; the release notes state the limit.

**Rejected.**
- Keeping all 357 as concepts and cleaning them with signals: 344 reviews to reach the state the 13
  already have, with the meaning features running over unread strings meanwhile.
- A second table for terms: ADR-018 reserved that for vocabularies that differ in shape, and the
  difference found is membership, which its flag already records.
- A display name with matching unchanged: `distillation` would keep counting in full.
- Fixing the cut before the release: it moves keywords, definition candidates and written forms
  at once.
- Writing the snapshot's numbers into the ADR alone: a decision file cites its evidence, so the
  counts went into a baseline a reader can check.

**What it opens.** Row 94 is gated on the user accepting ADR-054. The 31 aliases of the 13 need
the user's exact-or-broad call, and the effect on presence, edges and gaps is unmeasured until then.
The snapshot's counts are observations, not graded signals: whether a neighbour share or an "et al."
share separates a term from a concept needs the user's labels on a sample (decision 4). Found on the
way: Manage keywords deletes a concept row and its definitions with no confirmation — in row 94.

---

## 2026-10-01 — The pip-audit gate's first catch: three upgrades for eight advisories published overnight

**What changed.** `uv.lock`: sentence-transformers 5.5.1 → 5.6.0 · urllib3 2.7.0 → 2.8.0 ·
virtualenv 21.3.3 → 21.7.13, and python-discovery 1.3.1 → 1.6.1, which virtualenv 21.7.13
requires. Each named package is pinned to the smallest version that clears its advisories
(`uv lock -P <package>==<version>`). Read package by package: 228 packages before and after, none
added, removed or downgraded, every artifact from `files.pythonhosted.org`, and no new dependency
edge (python-discovery drops its `platformdirs` edge).

**Why.** CI on `a5bb634` (the written-case commit below) failed in the pip-audit step and nowhere
else: eight advisories were published after the gate went blocking the day before (2026-09-30 (2)).
One is in sentence-transformers (CVE-2026-68770: `trust_remote_code=False` is bypassed when the
model path exists on disk, so Python files inside a model directory run at load). Three are in
urllib3 (an HTTPS proxy's TLS settings mixed with the target server's; two ways a server can stall
or bloat a streamed response). Four are in virtualenv, which is here only through pre-commit (a
downloaded wheel was not verified; a prompt or a path could reach `pyvenv.cfg` and the activation
scripts unescaped). All eight have a fix release, so none went into `pip-audit-ignore.toml`.

**Measured.** The gate on the tree before the change: 8 unreviewed advisories, exit 1 — the list
CI printed. On the upgraded tree: 196 packages audited, the 5 reviewed ignores matched, 0
unreviewed, exit 0. **The embedder and the reranker give identical output on both versions:**
through the app's own factories (`embeddings.get_embeddings`, the `CrossEncoder` the pipeline
builds), on the CPU, 8 texts and 1 query embedded with bge-base and 8 query–text pairs scored with
bge-reranker-base differ by at most 0.0 between sentence-transformers 5.5.1 and 5.6.0. A second run
on 5.5.1 also differed by 0.0, so the comparison is repeatable. The unit + integration suite on the
dev venv: **2,561 passed, 0 failed**, coverage 92.7%.

**Not done.** No trial freeze. The sidecar spec collects sentence-transformers whole
(`collect_all`, `scripts/doc_assistant_api.spec`), the lock adds no package and no edge, and the
release runbook freezes and smoke-tests the bundle before any release. CI's CPU venv was not
rebuilt locally; the push's CI run is that check.

**Rejected.**
- Reviewing the eight into the ignore file: each has a fix the lock takes without moving anything
  else.
- An unpinned `uv lock --upgrade-package`: on 2026-09-30 it pulled anthropic 1.9, langsmith 0.14
  and three new packages for the same kind of fix.

**What it opens.** The gate reads a live advisory database, so `main` can turn red on a push that
changed no dependency — here one day after the gate became blocking. That is what S-8 asked for,
and the red run was seen only because the session looked (KI-58's shape). S-12 (Dependabot pull
requests, and the alerts toggle) is the planned control that would raise a new advisory before a
push finds it; this run is the first data point for that call.

---

## 2026-09-30 (3) — A label keeps its written case (ADR-053 decision 3); the API refuses a foreign Host (S-5)

**What changed.**
- **`knowledge/written_forms.py`** (new) + the derived table **`concept_written_forms`**: the full
  skeleton build votes, for every surface form of every text-bearing concept, how the library
  spells it — whole-word uses in body prose (`definitions.document_sentences`, now public), not at
  a sentence start; each document votes for its most-used spelling, most documents win. Stored
  beside `Concept.label`, never into it (ADR-043; a build test holds the label byte-identical), and
  on each `ConceptNode` as `written` (in `skeleton.json` only when set).
- **One matcher for all four callers** (`concept_skeleton.form_matcher`): presence, the gap list's
  claim attribution, definitions (passages and usage examples) and epistemics. A form written in
  lower case matches in any case, exactly as before; a form written with a capital matches only
  spellings that differ from it in **word-initial** letters (`word_case_key`). `surface_forms` is
  public now; substring mode, the RG-008 A/B lever, ignores written forms.
- **Display:** the graph payload carries `written`, and the Graph tab shows it on the nodes, the
  panel heading and the rail (`lib/graph/labels.ts`, tested).
- **S-5, the host guard:** `create_app` adds `TrustedHostMiddleware` last, so it runs first; hosts
  from `DOC_API_ALLOWED_HOSTS`, else `127.0.0.1,localhost`; a blank setting means loopback, never
  every host. `tests/conftest.py` allows the `TestClient`'s `testserver`; `docker-compose.yml` and
  `.env.example` name the setting.
- `GLOSSARY.md` C-013 *written form*; ADR-053 amendment 2026-09-30; `docs/knowledge-layer.md`
  (presence row); `docs/security.md` (S1, S-5, floor row 6); ROADMAP rows 60 and 93.

**Why.** The user's labels (2026-09-22): "din is not the same as dIN. Important of being
case-sensitive." Presence feeds `single_source`, the trust table's one trustworthy gap signal, so
two words sharing a lower-cased label were being counted as one concept. S-5: the API listens on
loopback, but DNS rebinding lets a web page reach it under a hostile name (S1, T2).

**Measured** (`tests/eval/baselines/written_forms_2026-09-30.md`, read-only, the working library):
367 written forms, 155 with a capital. Presence moves for **16 of 357 concepts, all losing
documents**: nine lose another word, a surname or OCR noise (`CRE` from `cre`, "vitamin Din" from
`din`, the numpy paper's author Colbert from `colbert`, "StS" initials from `sts`), seven lose the
same name lower-cased, mostly in bibliography titles (`gpt-4`, `deeplabcut`). On the graph only
`cre` moves, 7 → 6 documents. **The 24 labelled definition candidates are unchanged (11 usable)**;
the definitions layer's one change is that "vitamin Din" is no longer a usage example of `dIN`.
Nothing moves in the stored graph until the user rebuilds. S-5 checked live: the running API
answers a foreign `Host` with 400, both loopback names with 200, and the desktop dev app loads
through it. The Graph tab showed `Cre`, `DBS`, `Ntsr1` with the graph response rewritten in the
page, since the stored skeleton predates written forms.

**Rejected.**
- The most-used spelling matched exactly (R1, 27 concepts move): `assistant` fell from 10 documents
  to 4 and `plateau` from 12 to 2 — ordinary words capitalised just over half the time.
- Occurrences voting: one paper repeating `PERSONA` outvoted the three that say `persona`.
- Exact spellings for anything but a capitalised word (R2, 20 move): Title Case quoted from titles
  won the vote, and prose such as "a sentence encoder LSTM" stopped counting. The word-initial rule
  restores those four and keeps every correct split.
- Rewriting `Concept.label` to the written form: ADR-043 — derived data sits beside curated data.
- Deriving the written form inside each caller: epistemics sees one text at a time and the usage
  examples read six documents, so each would vote differently. One stored vote, one rule.
- Showing the written form everywhere in this change: vocabulary-search hits and the gap list read
  their label server-side from other payloads; left as a follow-up.

**What it opens.** The seven remaining losses mostly come from reference lists: presence reads the
whole text, the vote only body prose. Matching body text only would remove them, and would also
stop reference-only mentions counting — its own measurement. ADR-053 decision 4, the abbreviation
signal, now has its input (141 labels with a capital). The user's rebuild applies the 16 changes;
the baseline lists each one with what stopped counting.

---

## 2026-09-30 (2) — Row 61 / security S-8: nineteen upgrades, five reviewed ignores, and pip-audit blocks CI

**What changed.**
- **`uv.lock`: 19 packages moved**, each to the smallest version that fixes its advisories and that
  the lock can take: aiohttp 3.13.5 → 3.14.3 · anyio 4.13.0 → 4.14.2 · cryptography 48.0.0 → 50.0.1
  · langchain 1.3.1 → 1.3.9 · langchain-anthropic 1.4.3 → 1.4.6 · langchain-core 1.4.0 → 1.4.9 ·
  langchain-protocol 0.0.15 → 0.0.19 · langgraph 1.2.0 → 1.2.4 · langgraph-checkpoint 4.1.0 → 4.1.1
  · langgraph-sdk 0.3.14 → 0.4.5 · langsmith 0.8.5 → 0.8.18 · msgpack 1.1.2 → 1.2.1 · oauthlib
  3.3.1 → 4.0.0 · pillow 12.2.0 → 12.3.0 · pip 26.1.1 → 26.2 · pydantic-settings 2.14.1 → 2.14.2 ·
  soupsieve 2.8.3 → 2.9 · starlette 1.0.0 → 1.3.1 · transformers 5.9.0 → 5.10.4. Read package by
  package: 228 packages before and after, none added, removed or downgraded, every artifact from
  `files.pythonhosted.org`, and the only new dependency edges point at packages already locked
  (aiohttp → typing-extensions; langgraph-sdk → langchain-core, langchain-protocol, websockets;
  langsmith → websockets). `apps/desktop/package-lock.json`: devalue 5.8.1 → 5.9.4, inside svelte's
  `^5.8.1` and build-time only.
- **`pip-audit-ignore.toml`** (new): the five advisories no upgrade fixes, each with its id,
  package, review date, why it does not apply here, and what would reverse that. Four are in
  chromadb and all server-side (the HTTP collection endpoints, tenant checks, the RBAC provider);
  the app only embeds `chromadb.PersistentClient`, and no `HttpClient` exists in the tree. One is in
  setuptools (an sdist built on macOS); the locked copy is torch 2.12's runtime dependency, capped
  `<82`, and the fix is 83.
- **`scripts/pip_audit_gate.py`** (new, 17 tests in `tests/unit/test_pip_audit_gate.py`): runs
  pip-audit as JSON and fails on any advisory not in the ignore file, matched by id or alias and only
  for the package named. It refuses an entry that misses a field or carries an unknown one, fails
  closed (exit 2) when pip-audit returns no report or the file cannot be read, and reports a stale
  ignore without failing, because a CUDA dev venv and CI's CPU venv install different package sets.
- CI's pip-audit step runs the gate with no `continue-on-error`; `just audit` runs the same
  command; `[keypoints.sprint-close]` runs it as a deterministic line, as its comment said it would
  once row 61 landed. `docs/security.md` (S6, S-8, floor row 5), `docs/RELEASE.md`,
  `docs/architecture.md` and the ROADMAP (row 61 to the done archive; row 60; F10) follow.

**Measured.** pip-audit: 59 advisories across 18 packages → 5 across 2, all reviewed. The gate
judged the pre-upgrade report FAIL (exit 1, each advisory named with its fix) and the upgraded tree
OK (exit 0), so it can fail. The CI suite on the CPU venv CI uses: **2,539 passed, 0 failed**
(2,522 + the gate's 17), coverage 92.6%, 96 warnings (the one new warning is starlette's test
client, above). Frontend: svelte-check 0/0, `npm test` 272, `npm audit` 0 at moderate. **The
upgraded tree still freezes** (the plan's risk control, so a packaging break shows now and not at
the release): `scripts.build_sidecar` on the CPU venv in ~9.5 min, `sidecar_size` 1,563.4 MiB
against the 1,555.0 floor (0.6.0: 1,562.3). The frozen binary, pointed at an empty data home under
`build/` on a spare port, answered `/api/health` in 32 s and **ingested a generated two-page PDF**
(text + one embedded image): 1 added, 0 errors, 10 chunks — PyMuPDF, chunking, the bge-base
embedder on transformers 5.10.4, Chroma, SQLite and the sparse index, all inside the bundle. No
chat turn was sent (the default provider is paid, KI-4). The smoke data was deleted; the dev-tree
sidecar in `src-tauri/binaries/` is now this build, not 0.6.0's. Row 86 (`tauri dev` spawns
whatever frozen binary sits there) is unchanged.

**Rejected.**
- `uv lock -P <package>` for each flagged package, the obvious command. It takes the latest allowed
  version, and here that meant anthropic 0.103 → 1.9 (a major SDK change under the provider layer),
  langsmith 0.8 → 0.14, transformers → 5.15, and **three packages the lock had never held**
  (`httpx2`, `httpcore2`, `httpx2-jsfetch`). A lock change that adds packages nobody asked for is
  how a poisoned dependency edge gets in, so every upgrade is pinned to its fixing version instead.
  `httpx2` is probably legitimate — starlette 1.3's own test client now warns "install `httpx2`
  instead" (the suite's one new warning) — but a new package is reviewed on purpose, in the
  session that takes those upgrades, not swept in by an advisory fix.
- langgraph-sdk 0.4.2, the smallest the new langgraph accepts: it downgrades websockets 16 → 15.
  0.4.5 does not.
- transformers 5.10.0, the listed fix: yanked upstream ("missing a bunch of fixes"). 5.10.4 is taken.
- torch 2.14, to lift setuptools' cap: a new CUDA and triton for an advisory about macOS sdists.
- An ignore for oauthlib: 4.0.0 is its fix and the lock takes it, so S-8's rule (upgrade what the
  lock allows) decides. `import chromadb` loads neither kubernetes, requests-oauthlib nor oauthlib,
  so the major version reaches nothing at runtime.
- Inline `--ignore-vuln` flags in `ci.yml`: no place for a reason, and a local run needs its own copy.
- Blocking only on HIGH/critical, as row 61 was worded: pip-audit reports no severity. The gate
  blocks on anything unreviewed, which is stricter.

**Found on the way — S13.** CI's `detect-secrets scan --baseline .secrets.baseline` cannot fail:
it rescans, writes a new finding into the baseline and exits 0. Reproduced on a scratch repository
with a planted AWS key (3 findings absorbed, exit 0), where `detect-secrets-hook` exits 1. Only the
local pre-commit hook gates. Recorded as `docs/security.md` S13 and step S-13; not fixed here, since
a session takes one security step.

**What it opens.** A newly published advisory now reddens the next push with no code change, and
the session-start `gh run list` line (KI-58) is where it will be seen. The anthropic SDK 1.x,
langsmith 0.14 and transformers 5.15 are real upgrades waiting for their own session, with the
provider tests. The chromadb and setuptools ignores carry their reversal conditions, for the
periodic security check (§6) to re-read.

---

## 2026-09-30 — The ROADMAP's session Sequence table retires; session planning is local-only now

**What changed.** The `## Sequence — the next sessions, in order` table (13 sessions, added
2026-09-10) left `docs/ROADMAP.md`, and a one-line note stands in its place. The section moved
verbatim, byte-checked against `HEAD`, to the local-only `docs/archive/local/`. The "Now → next →
later" direction list, the PR table and the feature sections are unchanged.

**Why.** The user keeps session-level planning local-only now, as their own development planning
(2026-09-30), and that plan superseded the table the same day. A second effect: since 2026-09-10
the Sequence table had been the file's first markdown table, so `roadmap_sync`, which reads the
header of the first `|` table, found no PR table, and the note above the PR table ("the only
machine-read table in this file") was false. With the section gone, its parser reads the PR table
again: 84 rows, first 46, last 74, checked by calling `parse_table` directly, because the tool
itself writes sprint stubs.

**Rejected.** Deleting the section outright: retired planning moves verbatim here, as the 87 done
rows did on 2026-09-10. Moving it to the tracked `ROADMAP-done-001.md`: that archive holds done
rows, and eight of these thirteen lines were never done.

**What it opens.** Nothing in the tracked tree carries a session order any more.
`docs/security.md` §4 still orders the security steps, and row 60 still names the current one.

**The rotation this entry triggered** was the log's first batch under cpc 1.12.0: it moved the 11
entries from 2026-09-07 to 2026-09-17 (1), byte-verified, into a new `DEVLOG-archive-007.md`
(archive 006 is past `archive_max_tokens`), and laid `docs/archive/DEVLOG-INDEX.md`. The tool
titled the new archive "DEVLOG archive 001", because `rotate.py` carries that title as a fixed
string, and it did the same to the baton's new local archive the same day. Both titles were
corrected by hand; the fix itself belongs upstream in cpc.

---

## 2026-09-29 — cpc re-vendored 1.8.0 → 1.12.0, from the tags; logs now rotate in batches

**What changed.** `cpc-init` re-vendored the gitignored `tools/conventions/cpc/` from cpc's tags,
never its HEAD: first `v1.11.0`, then `v1.12.0` the same day. Both runs came from a cpc session at
the user's request. `_VERSION` now carries a sha256 per module, and `init_check` compares the drop
against them. The `scripts/conventions.toml` header and the local CONTEXT, which restate the
version, now say 1.12.0. Then cpc's migration note (`migrations/2026-09-29-rotate-in-batches.md`,
cpc ADR-053) was applied:
- `devlog_rotate_to = 10` and `session_rotate_to = 5`. Past its cap, a log is cut to half in one
  batch, so the archive changes once per 11 DEVLOG entries instead of every time.
- `archive_max_tokens = 25000`. `DEVLOG-archive-006` is ~72k tokens, so the next DEVLOG batch
  starts `-007` and lays `docs/archive/DEVLOG-INDEX.md`. That index is tracked, like the archives.
- `[rotate] write_index = true`.
- `docs/archive/SESSION-INDEX.md` is gitignored. The baton archives are local-only, and the index
  their first rollover lays lists every one of their headings. Without the line, that list would
  have been committed to this public repository.

**Measured.** Every gate `just` wires, run on the same tree with each old drop and the new one:
`docs_check --strict` and `integrity_check --strict` 0/0 each time, `init_check --strict` and
`sprint_check` unchanged, `settings_doc --check-config` no findings. cpc ran the same migration on
a scratch clone of this repository before its tag: one batch of 11 into `-007`, gates 0/0 before
and after, `tests/unit/test_doc_sizes.py` 6 passed.

**Rejected.** `--profile standard`, which would lay any missing standard file; the default profile
found all nine present and laid nothing.

**What it opens.** The first batch rotation happens at the 21st DEVLOG entry; update the range line
in this file's header by hand after it, as today. cpc 1.11.0's release checkpoints (ADR-051): a
`## Releases` entry in the ROADMAP opts in, and `[release] smoke` then becomes required.

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

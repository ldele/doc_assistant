<!-- status: active · updated: 2026-11-11 (state lines at v0.7.3; row 60's current step is S-9); 2026-11-10 (row 113 added: the eval harness reinforced for the chat work); 2026-11-06 (row 110 (d) built: one dialog frame, no literal colours, the shared stylesheet; row 110 (c) built: hover text on every button; row 110 (b) built: the roles across the app and the confirms; row 110 cut in four pieces, (a) built: the fill pair, the tooltip, the checks; row 110 ready: ADR-063 accepted; row 110 widened: hover text, styling in one place, three checks, ADR-063 choices 6 to 8); 2026-11-05 (the user's UI batch: rows 78 and 80 built with ADR-063 proposed; rows 110 to 112 added); 2026-11-04 (row 31 built: the indexing screen); 2026-11-02 (row 109 and the review of 2026-11-02 routed; 2026-11-01: row 108: what ADR-056 leaves; 2026-10-31: cleaned up after 0.7.1: finished rows archived, long rows cut to what is left, the order set for the next moves) · class: living -->

# ROADMAP — doc_assistant

The living roadmap: **one table of open work, grouped per feature**, then one section per feature
saying what is shipped, what is next, and what gate the next step has to pass. This is the source
of *intent*; `docs/decisions/` records the locked design choices (living index `docs/decisions.md`)
and `AGENTS.md` / `.claude/CONTEXT.md` point at both. Evaluation strategy: `tests/eval/TESTING.md`.

> **Restructured 2026-09-10.** The file had grown to 108 rows, 87 of them done, in one flat table
> whose rows restated the DEVLOG. Every done row moved **verbatim** to
> [`docs/archive/ROADMAP-done-001.md`](archive/ROADMAP-done-001.md) (ids are never reused; cite
> them from there). The UI checklist's queue (`docs/ui-checklist.md` §1–§3, local-only) was folded
> in so there is one planning surface, not two. The pre-cpc narrative roadmap is still at
> `docs/archive/doc-assistant-roadmap.md`.
>
> **Cleaned up 2026-10-31.** Finished rows moved **verbatim** to
> [`docs/archive/ROADMAP-done-002.md`](archive/ROADMAP-done-002.md), which also keeps each long
> row as it stood before it was cut here to what is left.
>
> **How to read a row.** `Status` is one of `planned` (agreed, not started) · `gate first` (an ADR,
> spec, grill or measurement must land before code) · `blocked` (waits on another row or a KI) ·
> `deferred` / `parked` (deliberately not scheduled — the reason is in the row). One PR per session
> (AGENTS.md build protocol). A row's *why* lives in its spec/ADR/DEVLOG, linked in `Spec` — do not
> restate it here.

## Where the product stands (2026-11-11)

**v0.7.3 is published** (2026-11-11). Shipped and measured: hybrid RAG with page-level citations,
the integrity layer (provenance, evidence vs interpretation, reviewer), figures, concepts the user
curates with exact and broad forms, definitions chosen from sentences quoted from the user's
documents, the gap list and taxonomy substrate, folders with retrieval scope, add-documents (copy or
reference), Zotero import, per-part re-ingest, the source viewer with page jump and in-context
passage, metadata filled from a document's own text at indexing, and the library summary. Quality record: `evals/README.md`; cost record: `docs/performance.md`; what the knowledge
layer's signals are worth: `docs/knowledge-layer.md`.

**Now → next → later** (the user's order, 2026-10-31, which replaces those of 2026-09-10 and
2026-10-18):

1. **Now — the short-term fixes and improvements.** The parked side-session work lands, then the
   docs revamp (**98**, with **89**), the reference-list cut (**95**), the app stays repairable
   (**100**) and what KI-79 left. One small security step each session (**60**). The user's UI
   batch of 2026-11-05: selection and the squashed concept graph built (rows 78, 80, 111;
   ADR-063 proposed), then the action roles across the app (**110**) and the library summary
   (**112**).
2. **Next — the chat and MCP.** An MCP server (**65**) and a chat that does more with the library:
   prompts and modes, skills maybe (**37**), context handled deliberately (**101**), lookups in the
   library and the graph (**102**), then an interface pass over all of it (**103**). Security step
   S-9, the prompt fence, lands before any tool. The eval harness is reinforced first (**113**),
   so each of these changes is measured on cases that can show it.
3. **Then — extraction, the Library, and what the user chose.** Extraction polish (**104**), a UX
   and UI pass over the Library (**105**), and every instance of a word or concept the user picks,
   per concept (**106**).
4. **Later.** The acquisition half (**KL2**), LLM-assisted ingestion (**25**, with 56 riding on it),
   the graph re-pass (**75**), then the Project ADR (**47**) and the rest of F4.

*(Session planning is kept local-only. The PR table below is the open work, and `docs/security.md`
§4 is the order of the security steps.)*

## Releases

Published so far, each with its notes in `CHANGELOG.md` and its tag: **v0.7.3** (2026-11-11),
**v0.7.2** (2026-11-04),
**v0.7.1** (2026-10-31), **v0.7.0** (2026-10-30), **v0.6.0** (2026-09-04). The runbook for
cutting one is `docs/RELEASE.md`.

The next release is not planned in this file: session and release planning are kept local-only
(see *Where the product stands*, above), and the open work is the PR table. A `### X.Y.Z` entry
under this heading would opt the project in to cpc's release checkpoints, where a release is
planned with criteria that can fail and `cpc-release-preflight` reads them (cpc ADR-051). None
is written, so that check stays off.

## Goals

1. Swappable embedding layer with measured comparisons (per-project routing deferred: no model beats
   `bge-base` on an identifiable sub-corpus yet).
2. A reproducible eval harness inside the project, extractable later.
3. Figures and tables as first-class structured content.
4. A research-integrity layer: provenance per answer, evidence/interpretation split, a rubric reviewer.
5. Position against published standards (PRISMA-trAIce, AI Usage Cards, BE WISE) without binding to one.
6. Close the self-improvement loop above a minimum-N gate; below it, instrumentation only.
7. A self-organising wiki/synthesis layer that makes knowledge gaps computable — and, the operative
   half (user, 2026-08-03): **tell the user what to go read next**.

**Historical phase ids** (still cited by ADRs and `.claude/CONTEXT.md`): phases 1–5 done; **6**
(figures/tables + reviewer) → features F1/F3; **7** (gap detection + concept graph) → F5; **8**
(iterative UI track) → every feature's UI rows + F12; **9** (literature-review generation) → F8.

## Open work — the PR table

*(The only machine-read table in this file — `roadmap_sync` parses the first markdown table. Keep
the `PR | Feature | Scope | Status | Spec` header. Ids continue the historical numbering; letters
mark the older tracks.)*

| PR | Feature | Scope | Status | Spec |
|----|---------|-------|--------|------|
| 104 | F1 Ingestion | **Extraction polish** — read real documents first, then fix what a reader meets, in that order. Known input: tables (row 27), the page marker that carries the previous page (KI-57), the title picker (KI-54), EPUB parsing (row 85), scans (EX1) | gate first — diagnose | KI-54 · KI-57 · `extractors.py` · `docs/plans/PLAN_2026-10-31_next-two-releases.md` (local) |
| 108 | F1 Ingestion | **Letter-spacing across the library** — the rule and its control are built (ADR-056). Left: the user's re-extraction of the library, with backups, rebuilds and the private eval against its control (gates 3 and 6); more letter-spaced PDFs than one library holds (gate 4, each download the user's call); and point 8, keyword selection skips recorded runs, agreed with session 5c on four conditions | planned — point 8 after ADR-055 lands | ADR-056 · `docs/wireframes/letter-spacing-ignore-list.md` · `tests/eval/baselines/letter_spacing_2026-10-21.md` |
| 25 | F1 Ingestion | **LLM-assisted ingestion mode** — opt-in pass over the programmatic default (Ollama-first, KI-4 guard): figure links, citation links, **reference order** (needs an additive ordinal column at extraction time — it cannot be backfilled), concept marking at ingest. Output is a durable, inspectable, re-derivable sidecar (Enrichment-Layer Pattern; ADR-043's "normalisation is a derived layer") | gate first — spec + ADR (new ingest mode, cost-gated) | `docs/archive/local/PLAN_2026-08-11_ingestion-quality.md` §2 (local) · `.claude/CONTEXT.md` direction note |
| 27 | F1 Ingestion | **Tables** — extraction code exists but **no table has ever landed** (Marker unrunnable here, KI-42 fixed the pin); diagnose on the live corpus before designing; styled table rendering after | gate first — diagnose | `docs/figures-and-tables.md` · KI-42 |
| 26 | F1 Ingestion | **Keyword auto re-trigger on corpus growth** (P1 D3) — the same question as KI-44: a sidecar can only reach retrieval through a global `--rebuild` (~4 min at 97 docs, ~3.6 h at 10k) | planned | KI-44 · `docs/archive/local/PLAN_2026-08-11_ingestion-quality.md` (local) |
| 109 | F1 Ingestion | **The installed app makes its own derived layers** — keywords first, then the wiki, the taxonomy and `is_a` proposals and figure descriptions. Today only the `scripts/*` runners make them, so a library the installer builds has no keywords (CHANGELOG 0.7.0, README *The knowledge layer*). First question: whether ADR-058's scan is the in-app path. Release B's spec for row 106 states what an installed library has | gate first | ADR-058 (proposed) · `docs/reviews/REVIEW_2026-11-02_project-review.md` (local) G2 |
| EX1 | F1 Ingestion | **OCR sidecar for true scans** (ADR-039) — opt-in, restores a text layer not markdown, Tesseract absent-tolerant (KI-47). RG-025, its measured gate, closed 2026-08-08. Parts (a) extractor-lost text and (c) KI-40 cache key are done | planned | ADR-039 · RG-025 · KI-47 · KI-48 |
| 28 | F1 Ingestion | **Extended metadata + Crossref autocomplete** — surface the stored DOI; add journal/url/article_type (~6 appends to `_ADDITIVE_COLUMNS`); local-text yield is hopeless, Crossref wins on all four. Shares the outbound transport with T5/39 | gate first — ADR-016 (number reserved) | `.claude/ui-checklist-archive-001.md` (local) · ADR-044 |
| 29 | F1 Ingestion | **Calibre adapter** (one module + one route, ADR-049) **and** run the Zotero adapter against a real library — the tests prove the mapping, not the schema · a UI for Zotero's "Linked Attachment Base Directory" (a preference, not a database value; `read_library` accepts it, nothing offers it, so linked attachments are skipped) | planned | ADR-049 |
| 31 | F1 Ingestion | **Ingest honesty — Track B of the add-documents plan** — built for 0.7.2: the indexing screen lists every file of a run with its outcome and a failure's reason, Pause and Stop act between documents, *Retry* and *Index the remaining* re-run from the server's record of the run, and Re-index counts and estimates first. IH1 (a scan announcing itself at add time) stays with RG-030 | built | `docs/wireframes/indexing.md` · `ingest/run.py` · `ingest/estimate.py` · `docs/archive/local/PLAN_2026-08-20_user-friendly-ingestion.md` §4 (local) |
| 30 | F1 Ingestion | **Per-document ingestion tuning** (a table-heavy paper chunked differently). Contained but not free: BM25 `avgdl` is corpus-global, `TOP_K` counts parents, splitters are import-time singletons | gate first — grill | `.claude/ui-checklist-archive-001.md` (local) |
| 77 | F1 Ingestion | **Parallel extraction in the installed build** — `ingest/workers.py` declines parallel extraction when frozen: `freeze_support()` is called but has never been observed working in a packaged build, and the failure mode is a fork bomb of sidecars. So the installer ingests serially and leaves the measured 1.74× unused. Observe one installed build extracting in parallel (`DOC_INGEST_PARALLEL_FROZEN=1`), then delete the branch as its own comment asks; correct `docs/performance.md`'s "extraction parallelism is unmeasured" line (measured 2026-08-26) | planned — verify on an installed build | `ingest/workers.py` · `docs/performance.md` · DEVLOG 2026-08-26 (archive 006) |
| 85 | F1 Ingestion | **EPUB XHTML parsed as HTML** — the one warning in the unit suite (`extract_epub` hands XHTML to `BeautifulSoup(…, "lxml")`); parse as XML (`features="xml"`, lxml is already a dependency) and compare the output on a real EPUB before merging | planned (small) | `extractors.py` · REVIEW 2026-09-10 §5 row 7 (local) |
| 105 | F2 Library | **Library UX and UI pass** — a deliberate pass over the Library tab: finding, selecting and reading documents, the keyword display and the keyboard reach KI-79 lists, with rows 24, 32 and 80 inside it | gate first — wireframe | `docs/release-ux-checklist.md` §2 · KI-79 · `docs/plans/PLAN_2026-10-31_next-two-releases.md` (local) |
| 106 | F2 Library | **Every instance of what the user chose** (the user, 2026-10-31) — the user names a word or a concept (say, *neurons*) and the library finds every instance of it, per concept: which documents, which pages, which passages, each opening at its place, with exact and broad forms kept apart. It starts from the user's choice, not from extracted keywords. Three known limits to meet: the matcher does not join singular and plural, a mention in a reference list counts as presence (row 95), and presence is sense-blind (row 92) | gate first — spec | ADR-059 (proposed) · ADR-058 (proposed) · ADR-054 · ADR-052 · `knowledge/concept_skeleton.py` `match_presence` · `docs/plans/PLAN_2026-10-31_next-two-releases.md` (local) |
| 24 | F2 Library | **Highlight the cited passage on the page image** — measured viable (4-word anchor places 90%, envelope 97% pure). Must solve: real column detection · 43% of parents straddle a page break (say so) · a stated decline-never-guess policy for ambiguous anchors | planned | ADR-050 Addendum · DEVLOG 2026-09-01 (2) |
| 32 | F2 Library | **Source explorer: chunk → parent → document** from the citation panel (~1 endpoint + panel UI; `parent_index`/`_chunk_key` exist) | planned | — |
| 33 | F2 Library | **Chunk editing + colour-coded chunk state** — editing collides with the Enrichment-Layer rule; split the read half (plain UI) from the write half (annotation sidecar) | gate first — ADR | memory note *future-user-annotatable-figures-chunks* |
| 35 | F2 Library | **Missing-source badge** — the ui-checklist row named three `resolve_source_path` bugs; KI-52 fixed the registry half. **Verify what remains against the code before planning** | planned (verify first) | KI-52 · `.claude/ui-checklist-archive-001.md` §3 "Missing-source handling" (local) |
| 80 | F2 Library | **Select all in the Library grid** — a corpus-wide action means ticking 98 tiles one by one (seen 2026-08-31) | built 2026-11-05 — *Select all* / *Select none*, Shift-click ranges and Ctrl/Cmd-click toggles, in the grid and the list; a Shift- or Ctrl-click outside select mode starts it | `lib/shell/selection.ts` · ADR-063 · baton 2026-08-31 (`docs/archive/SESSION-archive-005.md`, local) |
| 112 | F2 Library | **Library summary** (the user, 2026-11-05) — reachable where folders are opened (the Library rail's *Folders*), for the whole library or one folder: what went wrong at indexing and facts about the documents — missing metadata (KI-87), no references found, no figures, extractions that were flagged, document types. First read which of these signals exist and which can be trusted (`docs/knowledge-layer.md`); then a wireframe, a read-only route over `src/doc_assistant/library/`, and its numbers degrade honestly at 0 documents | built 2026-11-05 — a third layout of the Library pane, *Summary*, for the library or the open folder, revised to the user's seven recommendations: *Need attention* first; the indexing record (files not in the library, no text, flagged), each row naming its fix; metadata and what is on record as "N of M" meters, said once as *not read* when nothing has looked; types and years as bars; every count row opens its documents; `GET /api/library/summary` 0.68 s at 10,000 documents. Exposed: indexing read no metadata (since filled for new or changed documents, `3a1966f`), and nothing records that a re-run part ran | `docs/specs/feature-library-summary.md` · `docs/wireframes/library-summary.md` · KI-87 · ADR-063 |
| 65 | F3 Chat | **MCP server** — the library, the graph and cited answers reachable from an MCP client: a local stdio server over `pipeline.py`, as thin as `apps/api`, with all logic in `src/doc_assistant/`. The ADR settles which tools exist, whether any of them writes, how a write meets the launch token, and what must freeze in the API surface first. A global CLI rides on the same surface if the ADR wants one | gate first — ADR (unparked by the user, 2026-10-31) | `docs/security.md` · ADR to write · `docs/plans/PLAN_2026-10-31_next-two-releases.md` (local) |
| 37 | F3 Chat | **Chat modes** — named, user-customisable system prompts; the spec also decides whether a reusable *skill* is one of them or a thing of its own. **Hard boundary:** `ANSWER_PROMPT`'s citing block is the *wire format* `synthesis._CITATION_RE` parses (it broke once, 2026-07-14) — persona/task framing is editable, the citing block is not; `chat_controller` must hash the template per turn, not at construction, or provenance lies | gate first — spec | `.claude/ui-checklist-archive-001.md` §3 "User-customizable chat modes" (local) · `prompts.py` · `docs/plans/PLAN_2026-10-31_next-two-releases.md` (local) |
| 101 | F3 Chat | **What a turn sends the model** — context handled deliberately. Today `ANSWER_PROMPT` takes the retrieved passages and the question, and earlier turns reach the model only through the rewritten question. Decide what a turn carries (earlier turns, a running summary, what the library knows), fit it to the engine's window (8,192 tokens on Ollama since KI-66), say what was left out, and record it in provenance. A prompt change moves `prompt_version` and needs an eval run | gate first — spec | `prompts.py` · `chat_controller/controller.py` · KI-66 · `docs/plans/PLAN_2026-10-31_next-two-releases.md` (local) |
| 102 | F3 Chat | **The chat can look things up in the library and the graph** — a question about the library itself (which documents hold a concept, what its definition is, where the gaps are) is answered from what the app knows, not only from retrieved passages, and the answer says which it used. Shares a read-only surface with row 65. Works on a local model or says it cannot. **Security step S-9 lands first:** document text must not be able to steer a tool | gate first — spec + ADR | `docs/security.md` S5 · `docs/how-answers-work.md` · ADR-054 · `docs/plans/PLAN_2026-10-31_next-two-releases.md` (local) |
| 103 | F3 Chat | **Chat interface pass** — a deliberate pass over the chat once rows 101 and 102 show what it has to show: what was looked up, what the context held, the source panel, with rows 42, 44 and 78 inside it | planned — after 101 and 102 | `docs/release-ux-checklist.md` §3 · `docs/plans/PLAN_2026-10-31_next-two-releases.md` (local) |
| 90 | F3 Chat | **An unreadable citation form counts as uncited** — answers cite "According to Source 6: file.pdf" (45 of 1,239 unsupported pieces), which `_CITATION_TOKEN_RE` does not resolve, so a cited sentence is marked `unsupported` and inflates "Uncited in answers". Widening the parser moves markers on every answer **and the RG-012 citation verdict** — measure it against the stored answers ($0: re-segmenting reproduces 1,910/1,911 markers) before changing it | gate first — measure | `synthesis.py` · KL1 (DEVLOG 2026-09-16) · RG-012 |
| 38 | F3 Chat | **Structured/template answer mode** over extracted metadata, keywords, concepts | blocked on 37 | — |
| 39 | F3 Chat | **RAG + Internet mode** — same family as KL2/T5 (transport spiked: stdlib `urllib` → Crossref 25/25). Own ADR: provider list, quality list, caching, provenance of an acquired source, offline degrade; inherits ADR-044's transport/privacy discipline | gate first — ADR | ADR-044 · ADR-032 |
| 40 | F3 Chat | **Evidence-only mode polish** — `synthesis_mode=human` already answers with evidence only; the remaining decision is to force multi-query off and gate the rewrite so the mode is genuinely $0 | planned (small) | `chat_controller/controller.py` |
| 41 | F3 Chat | **Unconstrained mode** — corpus restraint off, measurements on; which variant, and how the reviewer still grades groundedness | gate first — grill | — |
| 42 | F3 Chat | **Highlight cited claims in the answer text** — presentation over `answer_claims` + `result.sources` (in the extracted markdown; the on-page half is row 24) | planned | `how-answers-work.md` |
| 43 | F3 Chat | **Rich marker UI** — hover a contested/superseded chip → the corroborating documents. `contested` is not a measurement today | blocked on KL3 | `docs/knowledge-layer.md` §6 · KI-33 |
| 44 | F3 Chat | **Resumable chat rehydration** — reopened chats are read-only; claims + reviewer joins are the follow-up | planned | `docs/specs/feature-conversation-resume.md` |
| 45 | F3 Chat | **User-tunable RAG pipeline** — reopens ADR-010, whose non-persistence is the governance wall. Finding that motivated it: `TOP_K` was never the problem; **`EMBEDDING_MODEL` is the catastrophic knob and appears in neither ADR-010's split nor the locked-settings table** — map blast radius first | gate first — grill + ADR | ADR-010 · `.claude/ui-checklist-archive-001.md` (local) |
| 78 | F3 Chat | **Range selection in chat select mode** (user, 2026-08-24) — Shift-click selects a range, Ctrl-click adds to the selection, the standard list semantics; today every row is an independent toggle (`Sidebar.svelte`, the `.rowmain.pickrow` handler), so twenty consecutive chats take twenty clicks | built 2026-11-05 — the same `pick` as the Library (row 80); the rail's select bar gained *Done*, and a *Cancel* on its delete confirm keeps the selection | baton 2026-08-24 item 7 (`docs/archive/SESSION-archive-004.md`, local) |
| 47 | F4 Projects | **Project ADR** — one grouping across documents · conversations · concepts (folders, home screen and per-project concept view are *one idea*, not three). Open: one shared tree or two? may a project scope retrieval like a document folder (ADR-025 F2)? per-project graph (= ADR-025 fork 5)? what happens to a project when its chats are soft-deleted? | gate first — ADR | `.claude/ui-checklist-archive-001.md` §3 "Conversation folders" · `docs/archive/local/REVIEW_2026-08-12_release-readiness.md` (both local) · ADR-025 |
| 48 | F4 Projects | **Home screen / project picker** — the first screen becomes "which project am I in" | blocked on 47 | — |
| 49 | F4 Projects | **Conversation folders** — a new relationship (`Folder` is document-scoped, ADR-025 F1), never a schema reuse | blocked on 47 | ADR-025 |
| 50 | F4 Projects | **Graph placement + gap-list home** — the Graph tab is back (row 22) but where the gap list and Connections ultimately live is the Project ADR's call | blocked on 47 | ADR-017 · RG-014 |
| 79 | F4 Projects | **Draggable dialogs** (user, 2026-08-24) — click-and-move on the modal surfaces (the add-documents sheet, confirm dialogs, the taxonomy overlay); one shared behaviour, not per-dialog code | planned (small) | baton 2026-08-24 item 8 (`docs/archive/SESSION-archive-004.md`, local) |
| 110 | F4 Projects | **Action roles across the app** (the user, 2026-11-05 and 2026-11-06) — buttons that do the same thing look the same everywhere, in the Graph tab's style, with hover text, a confirm when dangerous, tokens only, and checks that fail. Four pieces, one per session (the user, 2026-11-06): **(a) built 2026-11-06:** `--accent-fill` / `--accent-fill-fg` (the dark primary's white text at 5.0:1), one shared tooltip (`lib/shell/tooltip.ts`: hover and keyboard focus, WCAG 2.2 SC 1.4.13) that took over all 73 native `title`s on buttons, three ratchet checks in `npm test` (`src/ui-rules.test.ts`, lists in `ui-rules.allow.ts`) plus *no `title` on a button*. **(b) built 2026-11-06:** the 28 components with their own `.primary` / `.ghost` / `.danger` (48 class names), and the local primaries under other names, onto `btn` / `btn-primary` / `btn-quiet` / `btn-danger`; check 1 is now seven exceptions with reasons; `ConfirmDialog` an alert dialog with a filled-red confirm (`btn-primary btn-danger`, 5.4:1); the five destructive actions red, named and confirmed (*Remove key…*, *Remove server…*, *Remove…* and *Remove all…* on the letter-spacing ignore list, *Dismiss…* on a chosen definition, which was *Clear*), plus two more that delete at once (*Reset to default…* in *Edit metadata*, *Undo all…* in the add-documents review); the document page's *Source* and *Re-run…* as ordinary buttons apart from its index. Open: removing a document from a folder (the × in *Manage folders*) acts at once, the user's call. **(c) built 2026-11-06:** hover text on the 236 buttons in 59 files that had none, each saying what the label cannot and, when disabled, why (check 2's allow-list is empty); the top bar's tabs named for the widths that hide their labels, and a check that fails an icon-only button without `aria-label`; the 50 other `title`s decided one by one (the scope `<select>` onto the tooltip; 29 that show a cut name or a detail stay; 20 that explain a marker need it on screen, proposed for (d)). **(d) built 2026-11-06:** one dialog frame, `shell/Dialog.svelte` (backdrop, *Esc*, focus in and back, *Tab* kept inside; card, side panel or screen), for the 20 components that drew a modal of their own, with a list of open dialogs so *Esc* closes the one in front (*Esc* in a rename field now cancels the rename, not the dialog); the 36 literal colours onto tokens (`--scrim`, `--paper`, `--shadow-side`; the status dot's undefined `--ok`); `src/shared.css` for the dialog's title, head and buttons; two more checks, no `aria-modal` outside the frame and the rules copied between components (211 → 179, a ratchet). Open with the user: the 20 marker meanings, three button looks check 1 cannot see, the staged bar's *Clear*, the × in *Manage folders* (ADR-063 *As built (c)*, *(d)*). Each piece lowered its check's allow-list | built 2026-11-06, (a) to (d); the user's calls above remain | ADR-063 (*As built*) · ADR-064 · GLOSSARY U-003 · `app.css` |
| 95 | F5 Knowledge | **The bibliography cut keeps what follows the reference list; a reference-list mention is not presence** — (1) remove the reference section only. (2) Presence reads body text, and a mention in a reference list becomes its own count, an input to KL2. Every derived layer carries a before/after: keywords, definition candidates, written forms, presence, `single_source` | planned — now, after row 98; the release notes state the limit | KI-59 · `tests/eval/baselines/prose_parents_2026-10-09.md` §5, which replaces `vocabulary_shape_2026-10-01.md` §7 · `knowledge/keywords.py` `strip_reference_section` · `knowledge/definitions.py` `_body_chunks` |
| 111 | F5 Knowledge | **The concept view squashes its graph** (the user, 2026-11-05: text and other elements squash the graph until it cannot be seen) — the canvas is a flex child with `overflow: hidden` in a scrolling column, so the definition card and the document list shrink it: on *BM25* at 1280×800 the canvas was 2 px tall. The canvas keeps its height and sits where it is seen on opening a concept; check with a long definition and at 375 px | built 2026-11-05 — the graph above the definition, the column scrolls instead of shrinking it, a margin so a label at the edge shows, the bars above it wrap at 375 px | `screens/graph/Graph.svelte` `.ego` / `.canvas` (was `ConceptGraph.svelte`, ADR-064) |
| 75 | F5 Knowledge | **Graph re-pass** — a deliberate second pass over the Graph tab once the vocabulary, hierarchy and gap grades are honest (after 25): the `flat_field` detector (52) · gap-list / Connections navigation iteration (57) · placement of the gap list and Connections (50) · rich marker UI (43, waits on KL3) · a UX pass on the concept rail, ego view and empty states against `docs/release-ux-checklist.md` §4 (incl. the ego layout clipping a neighbour label at the left edge below ~640 px of pane width, seen 2026-08-30) | later — after 25 | rows 52 · 57 · 50 · 43 · `docs/knowledge-layer.md` |
| 51 | F5 Knowledge | **Concept hierarchy** — `is_a` edges among concepts and in-app review of proposed placements. Built: endpoint-kind validation, the deterministic proposer (`scripts/propose_isa.py`), *Proposed placements*. Left: concepts that share a head, the narrower merge pairs whose direction is not lexical (row 53 (3)) | built — waits on concepts that share a head | ADR-028 · ADR-045 · `docs/specs/feature-taxonomy-auto-propose.md` · `tests/eval/baselines/isa_head_suffix_2026-09-20.md` · REVIEW 2026-09-16 §1 C-3/C-7 (local) |
| KL2 | F5 Knowledge | **The acquisition half** (Phase B) — tell the user what to read next: re-point `gap_suggest` off `under_connected` onto `single_source` · grill **ADR-032**, then the outbound reach (*"for subject X, read Y"*) · the taxonomy and, per field, a curated base vocabulary as the reference class for expected coverage | gate first — ADR-032 grill | ADR-004 (Tier-2b) · ADR-032 · ADR-054 (Amendment 2026-10-01) · `tests/eval/baselines/reference_vocabularies_2026-09-21.md` · PLAN 2026-08-03 Phase B |
| KL3 | F5 Knowledge | **Node-B stance rebuilt on evidence** (Phase D, ADR-041 option 1): passages in the prompt + a `neutral` label + one pair per call; **a hand-labelled ground-truth set is the gate, not a follow-up**. Decide whether `superseded_trend` survives on `doc_years` alone | gate first — ground-truth set | ADR-041 · KI-33 · `tests/eval/baselines/node_b_stance_validity_2026-08-02.md` |
| 52 | F5 Knowledge | **`flat_field` depth detector** (crossover review §3.1): a field whose concepts all sit at `is_a` depth ≤ 1 — a vocabulary with no taxonomy. Every Tier-1 detector counts edges; none measures depth, and ADR-028's acyclic `is_a` edges make depth well-defined. Only fires once row 51 produces `is_a` edges | blocked on 51 | `docs/archive/local/REVIEW_2026-09-07_crossover-daily-learn.md` §3.1 (local) · ADR-028 D3 |
| 53 | F5 Knowledge | **Concept merges** — built: a merge keeps curation and can be undone, and preview and merge share one definition of "the same concept". Left: the hand score on the merge pairs, which becomes the labelled set for the context-pairing design (ADR-052: lexical pre-pairing, a context check, review before any write; no threshold decides alone) | (1) + (2) done; (3) the labelled set for context pairing | crossover §3.2 · REVIEW 2026-09-16 §1 C-1/C-2 (local) · `knowledge/concept_curation.py` `apply_merges` · `scripts/curate_concepts.py` · `config.py` `CONCEPT_MERGE_COSINE` · KI-20 (archive 002) |
| 91 | F5 Knowledge | **Say when the gap list predates the graph** — gap rows carry the `graph_version` they were built from, a CLI skeleton rebuild leaves them behind (16 of 18 rows were two versions old on 2026-09-16; rebuilt 2026-09-17), and the view never says so. Compare the **deterministic** rows only: stochastic `suggested_concept` rows keep their status and their old version across a rebuild by design (2 today). Comparing the rows' version with the current skeleton's is a sound signal (unlike version-vs-vocabulary, which `GraphStaleness` rightly rejects); surface it beside the existing stale notice | planned (small) | `knowledge/concept_graph_view.py` `GraphStaleness` · `knowledge/gaps.py` |
| 92 | F5 Knowledge | **Presence is sense-blind** — one label with several meanings counts as one concept, and presence feeds `single_source`, the trustworthy signal. **Decided in principle (user, 2026-09-18):** a different meaning is a different concept (ADR-052): a shared word gets a scope, found from evidence and confirmed in review, every step reversible and comparable with the label-only count | planned — after 93 | `knowledge/concept_skeleton.py` `match_presence` · `tests/eval/baselines/concept_merge_cosine_2026-09-17.md` |
| 93 | F5 Knowledge | **Meanings are curated (ADR-052, ADR-053)** — built: 93a, definition candidates that keep their source, with choose, dismiss and undo, and the Definition card; labels in their written case; abbreviation and fragment signals on the concept panel. **Left, in the user's order:** 93c expert vocabularies (local copies, matcher, `reference` candidates, the two-layer panel) · 93b model candidates with a grounding check, built with row 25 · 93d what the references say + papers to add · 93e the shared-word split review (row 92). Next on the signals: names cited alone, "Fig.", a display form for damaged long forms | 93a built; 93b–93e left | ADR-052 · ADR-053 · `knowledge/definitions.py` · `knowledge/term_signals.py` · `tests/eval/baselines/definition_sources_2026-09-21.md` · `tests/eval/baselines/reference_vocabularies_2026-09-21.md` · `tests/eval/baselines/definition_labels_2026-09-22.md` · `tests/eval/baselines/term_signals_2026-10-14.md` |
| 55 | F5 Knowledge | **`citation_missing` floor** — named in `GapKind`, never built (the other Tier-2a deterministic kind) | planned | ADR-004 · `knowledge/gaps.py` |
| 56 | F5 Knowledge | **Better concept highlighting** — mark concepts at ingest (rides on 25), more reliable than at read time | blocked on 25 | — |
| MM1 | F6 Maps | **Document outline layer** — populate `DocumentPart` from the cached markdown; `char_start`/`char_end` + part→`parent_index`; idempotent backfill; no LLM | gate first — ADR-030 is a stub | ADR-030 · `docs/archive/local/PLAN_2026-07-27_maps-trust-reports.md` Track 1 (local) |
| MM2 | F6 Maps | `knowledge/doc_map.py` read model + `GET /api/library/documents/{id}/map` + wire types | blocked on MM1 | ADR-030 |
| MM3 | F6 Maps | `lib/library/treeLayout.ts` (pure, tested) + `DocumentMap.svelte`; cross-doc pivot by shared concept | blocked on MM2 | ADR-030 |
| T1 | F7 Trust | `Document.source_type` + user override + deterministic partial derivation; **`unknown` is first-class** | gate first — ADR-031 is a stub | ADR-031 · PLAN 2026-07-27 Track 2 (local) |
| T2 | F7 Trust | Provenance-completeness indicator over existing fields only | blocked on T1 | ADR-031 |
| T3 | F7 Trust | Three-band source-evaluation strip — named signals, **no composite score** | blocked on T1 | ADR-031 |
| T4 | F7 Trust | `knowledge/leads.py` — guided escalation, local tier; absorbs B13 | blocked on T1 | ADR-031 |
| T5 | F7 Trust | **Outbound verification** — DOI backfill + Crossref/OpenAlex + `document_external` sidecar; the first *enrichment* network feature, own ADR + own gate | parked | ADR-032 · ADR-044 |
| RP1 | F8 Reports | Prompt composer (**frozen citation contract** + swappable brief) + `report_presets` + built-ins + a citation-audit regression gate | gate first — ADR-033 is a stub | ADR-033 · PLAN 2026-07-27 Track 3 (local) |
| RP2 | F8 Reports | Report as a job: dry run, **cost preview**, progress, per-section provenance | blocked on RP1 | ADR-033 |
| RP3 | F8 Reports | Trust-annotated sections + evidence appendix | blocked on RP1, T3 | ADR-033 |
| RP4 | F8 Reports | Rendering through `export.py` | blocked on RP1 | ADR-033 |
| 14 | F8 Reports | Integrity Chunk 3: **PRISMA-trAIce export** | planned | — |
| 58 | F9 Search | **Semantic search option** — global search is a literal client-side match (`lib/shell/search.ts`); add an optional embedding-cosine mode, keep literal as the default (instant, offline, predictable) | planned (small spec) | — |
| 59 | F9 Search | **Recent searches** (last ~5) and **search over conversation content** (titles only today) — no favourites list (user: prefer grouping) | planned | — |
| 100 | F10 Platform | **The app stays repairable when it cannot work.** Two states, two answers: (1) the backend answers but no answer engine does: everything but answering keeps working, and Settings leads straight to connecting one; (2) the backend does not answer: the window still shows what is wrong, with the backend's log, a retry and the library folder in use. Rule: *inform, don't block* | planned — now | `apps/desktop/src/App.svelte` readiness gate (KI-39) · ADR-034 · `docs/release-ux-checklist.md` §6 |
| 60 | F10 Platform | **Security, one step per session.** `docs/security.md` §4 is the ordered plan; this row names the current step | in progress — **current step: S-9, the prompt fence** (eval-gated, before any chat tool; S-7 done 2026-11-04) | `docs/security.md` §4 · `docs/archive/local/REVIEW_2026-09-10_project-review.md` §4 |
| 63 | F10 Platform | **Slim installer as a user option** — download weights on first run. A *trade* against KI-9's offline-from-first-launch promise, not an improvement: two artifacts per release, touches KI-10 and splits RG-010 | gate first — ADR | KI-9 · KI-10 · RG-010 · `docs/RELEASE.md` |
| 64 | F10 Platform | In-app API-key entry via an OS keychain (ADR-011 v2) | parked — keyring decision | ADR-011 · ADR-034 |
| 15 | F10 Platform | Extract the eval harness to a standalone repo (Feature 5) | planned — after a real comparison has been produced with the integrated one | ADR-024 |
| 66 | F10 Platform | Lift the Python 3.12 pin (KI-2) | blocked — external (native deps not cp314-stable) | KI-2 |
| 76 | F10 Platform | **CI sees what ships** — a **Windows runner** for the test job (the installer is Windows; no job runs there) · **`cargo check`** for the Tauri shell · Python versions: `pyproject.toml` claims `>=3.10` while CI installs one pinned version · the **coverage floor**, set from the measured number | planned | KI-58 · `.github/workflows/ci.yml` · REVIEW 2026-09-10 §5 (local) |
| 81 | F10 Platform | **Front docs catch up with the app** — re-record the README GIF (storyboards A/B/C in the 2026-09-07 baton; the recording toolkit exists only in an old session scratchpad — copy it somewhere durable first, and re-probe every selector) · `docs/DEMO.md` (still says Connections are "scored"; no source pane) · a commands table in `docs/usage.md` (it names 1 of 25 `just` targets) · README CI + licence badges · the two ~1 MB copies of the app icon · `docs/QUICKSTART.md` §3 still describes adding documents through Settings → Index folder, not the Add documents dialog | planned | memory note *readme-gif-recording-pipeline* · REVIEW 2026-09-10 §5 rows 6, 9 (local) |
| 82 | F10 Platform | **Release runbook: the process-scoped `OLLAMA_HOST` route** — the 0.6.0 RG-012 run bound Ollama beyond loopback with a process-scoped variable on a directly launched `ollama serve`, so there was nothing persistent to revert (DEVLOG 2026-09-02 (3)); `docs/RELEASE.md` and `docs/desktop-packaging.md` §5 still prescribe only the persistent set-then-revert route | planned (docs, small) | DEVLOG 2026-09-02 (3) · `docs/RELEASE.md` · `docs/desktop-packaging.md` |
| 87 | F10 Platform | **Docker Desktop 4.84 → 4.89** — the orphaned-socket crash hit four times on 2026-09-04; renaming the `Docker/run` and `docker-secrets-engine` folders under `%LOCALAPPDATA%` is the workaround, the upgrade is the fix. A host change, not a repo change | user action | baton 2026-09-04 |
| 88 | F10 Platform | **Disk and backup retention** — ~5 GB of regenerable bytes (`build/` 3.2 GB · `dist/` 1.6 GB · `.mypy_cache-strict/` · `data_multidomain/` 273 MB) and 12 `data/*.bak*` with no retention rule; agree a "keep the last N" rule, then prune | user's call — deletes data | REVIEW 2026-09-10 §5 row 4 (local) |
| 98 | F11 Quality | **Docs revamp: the docs follow the current conventions and the house skills, and finished work is archived** (the user, 2026-10-18). In order: the user's decisions (row 89's three, the rules that conflict with a skill, which ADRs stay proposed) and the known issues · finished plans, reviews and rigor items archived verbatim · the shipped specs frozen in place · the entry context under its cap with state generated · a clarity pass and the docs review stamped. Each step's *done* is a conventions setting turned on and green | planned — now, before row 95 | the 2026-10-18 audit (plan, local) · `.claude/REVIEWS.md` row 1 (local) |
| 89 | F11 Quality | **Three open process decisions** — (a) revive or retire `docs/sprints/` (cpc lays it; nothing written since 2026-07-18; "an empty convention is the worst of the three states", REVIEW 2026-09-10 §7) · (b) should the docs encoding guard (`tests/unit/test_docs_encoding.py`, tracked files only) also cover local-only docs, given a SESSION archive already carries live mojibake (asked 2026-08-17, never answered) · (c) fill in or remove `.claude/NORTH_STAR.md` — laid as an empty template by the 2026-08-22 conventions re-vendor, and its own text says an unfilled one is worse than none, because an agent reads the placeholders as answers; it needs the user's audience and register | user decision — **taken in the docs revamp's first session** (row 98; the user, 2026-10-18) | REVIEW 2026-09-10 §7 (local) · baton 2026-08-17 (`docs/archive/SESSION-archive-004.md`, local) |
| 6 | F11 Quality | Per-project embedder routing (Feature 1b) — re-run SPECTER2 `--repeat 5` first | deferred | `evals/README.md` |
| 68 | F11 Quality | **Close or waive the open blocks-ship rigor items with a date** — RG-001/008 (edge precision gate), RG-014 (spec has not absorbed the verdict), RG-027 (ADR-042 identity migration + backfill). `rigor_gate.py` does not exist; the file is a manual discipline doc — say so or build it | planned | `.claude/RIGOR_TODO.md` |
| 69 | F11 Quality | **`CANDIDATE_K=20` retest** on the private 35 with `--repeat` — the verdict has been "unvalidated" since 2026-06-13 | planned ($0) | `tests/eval/baselines/candidate_k_public_2026-06-13.md` · memory note *candidate-k-retest-needed* |
| 113 | F11 Quality | **The eval harness and the benchmarks, reinforced for the chat work** (the user, 2026-11-09). The chat rows are gated on measurements the harness cannot take: it has single-turn cases only, scores retrieval by presence, reports no interval, and its judge has not been checked against a person. In order: a short spec (case families, scorers, the statistic, the model panel, the paid budget) · rank metrics, citation coverage as a scorer, conversation and refusal cases, a paired interval in `compare_runs` · a larger public case set, plus conversation, not-in-the-library, hostile-passage and record-lookup cases · a baseline on the model panel before any chat change, with row 69 and a reranker ablation · the judge checked against human grades and the local judge's agreement gate · the public write-up. No framework and no new dependency (*What NOT to do*) | gate first — spec, after row 65's ADR; its baseline lands before S-9 | `tests/eval/TESTING.md` · `docs/OVERVIEW.md` §4 · `evals/README.md` · rows 15, 69 |
| 70 | F11 Quality | **Backend code review as a module** (`.claude/REVIEWS.md` row 2 = never); start with `ingest/` (nine tracked defects) and `chat_controller/` | planned — Cowork-shaped | `.claude/REVIEWS.md` |
| 71 | F11 Quality | **Frontend code review + a component test harness** — 39 `.svelte` components untestable under `node:test`; needs the lockfile decision (vitest + a Svelte testing library) and would automate half of `docs/release-ux-checklist.md` | planned | `.claude/REVIEWS.md` row 3 · `apps/desktop/CLAUDE.md` |
| 83 | F11 Quality | **Thin-shell and scripts hygiene** (REVIEW 2026-09-10 §8 moves 3–5, 7): move the existence-check SQL out of `apps/api/routers/taxonomy.py` into `knowledge.taxonomy` (a typed `UnknownConceptError` → 404; the only SQL in `apps/`, non-negotiable #3; can ride with row 51) · delete `scripts/extract_doc_metadata.py` (re-implements the write policy the `src` module owns) · archive `tests/eval/run_eval.py` (calls the provider SDK directly behind a `cpc: allow-live-api` pragma, bypassing the KI-4 guard; update `[test_api]`'s comment in `scripts/conventions.toml`) and the scripts the review listed as dead — **check each for callers first**: the same review called `library/pins.py` dead and it is live, and `.claude/CONTEXT.md` still names `backfill_graph_include` as a demotion path · lift `_resolve_pdf_path` / `_resolve_cache_path` out of `scripts/extract_tables.py` into `ingest.cache` | planned | REVIEW 2026-09-10 §8 (local) |
| 73 | F12 Verification | **Live-turn verification** ($0, Ollama): what the 0.7 walkthrough did not drive — session overrides changing retrieval on a real answer · the reviewer following a provider switch · epistemics marker chips (KI-15's fix has never been seen live) · RH1 reranker cap under multi-query. RG-012 Tier 2 is part of the release gate, and `docs/release-ux-checklist.md` is the standing walkthrough | planned | `docs/release-ux-checklist.md` · `.claude/ui-checklist-archive-001.md` §2 (the 2026-08-11 verification debt, local) |
| 74 | F12 Verification | **Two uncaptured defects** (user, 2026-07-21): the collapse-sidebar button and the global search bar misbehave — capture the exact symptom before fixing | planned — repro first | `.claude/ui-checklist-archive-001.md` §3 "Post-play UI polish" (local) — write the captured symptom into this row |

## Features

One block per feature: what is shipped (one line, pointers only), what is next, and the gate.

### F1 · Ingestion & document quality

**Shipped:** extract → markdown → chunk → embed → store (locked); registry + cache (KI-40 fixed);
figures (4b/4c); metadata extraction + manual override (ADR-013) + external catalogue slot (ADR-049);
selective ingestion (S1/S2); add-documents copy-or-reference (ADR-046, AD1–AD3b); Zotero import
(row 17); per-part re-ingest (ADR-048, rows 20/21); keyword de-noising D1/D2/D4/D5 (2026-08-12);
document identity survives re-extraction (ADR-047); extractor-lost text recovered (EX1a);
letter-spaced text closed at extraction, with its record and control (ADR-056; what it leaves is row
**108**).
**Open defects that shape this feature:** KI-44 (sidecar → retrieval needs a global rebuild),
KI-46 (zero-chunk duplicates), KI-47 (Tesseract), KI-54 (title picker),
KI-57 (page marker carries the previous page). **Next:** row **104**, the extraction polish, with
rows 27, 85 and EX1 inside it. Row **25** comes later. Its **gate** stands: a spec + ADR that names
the sidecar schema and the cost statement (ADR-048's "say what it costs first").

### F2 · Library & reading

**Shipped:** chunk browser (L1), redesign (L4), folders + retrieval scope (ADR-025), keyword
filter + families (ADR-015), safe delete (ADR-014), figure panel + viewer, references block,
Connections panel (E4), source viewer with page jump (ADR-050, row 18), in-context passage (row
19), graph-vocabulary toggle (row 23), ADR-046's amended delete (row 34). **Next:** row **105**,
the Library pass, with rows 24, 32 and 80 inside it, and row **106**, every instance of what the
user chose.

### F3 · Chat, answers & MCP

**Shipped:** streaming SSE chat, citation panel (U3), provenance + reviewer cards, A/B compare
(U6), conversation history + cleanup (U5), source-evaluation strip (E2) with the stance-derived
chips withheld (KI-33), epistemics toggle (E3), in-context + show-the-page from a citation, the
three-question citation gate (row 46).
**Next:** the MCP server (row **65**) and rows **37**, **101** and **102**, then the interface
pass (**103**). Security step S-9 lands before any tool: document text must not be able to steer
one. Row 37's hard boundary stays. Anything touching `contested` waits on KL3. Local-model citation
coverage is a floor (KI-36): every chat feature must state which model it was verified on.

### F4 · Projects & navigation

**Shipped:** the shell (sidebar │ main │ drawer), global navigation search (titles), the Graph tab
with a real empty state (row 22). **Next:** row **47**, the Project ADR, *before* any of 48–50:
three grouping systems that disagree is the failure to avoid. **Gate:** `grill-me` on the four
open questions in the row.

### F5 · Knowledge layer (concepts · taxonomy · gaps · epistemics)

**Shipped:** curated vocabulary + deterministic skeleton (Node A) + confined LLM stance (Node B),
gap detectors Tier-1 + Tier-2a floor/ceiling, gap list with durable triage (E5), taxonomy substrate
+ curation backend/view + auto-propose (TX1–TX3), graph coverage statement (2026-08-31), in-app
vocabulary curation (row 23), KL1 — the signals say what they measure (2026-09-17; DEVLOG
2026-09-16 (5); "Uncited in answers" stays a gap kind, user 2026-09-17), row 54 — no write can
put a field node on the graph, and coverage counts what it describes (DEVLOG 2026-09-17 (1)),
concepts with exact and broad forms (row 94), definitions chosen from quoted sentences (row 93a),
prose-only reading (row 97), keyword cleanup (row 99), Manage keywords (row 96). **Read `docs/knowledge-layer.md` §6 before believing any number:**
`single_source` is trustworthy, `under_connected` is noise at this vocabulary size, `contested` is
**not a corpus measurement**. **Vocabulary model (ADR-054):** a concept is a row the user has taken on, the other text-bearing
rows are terms, and a concept has a name and exact or broad forms. **Next:** row **95**. Row 106 in
F2 leans on it and on row 92. Row 93's remaining slices, KL2 and the graph re-pass (row 75) come
later, in the order at the top of this file.

### F6 · Document maps

Per-document outline + map surface (MM1–MM3). Specced twice, built zero times; ADR-030 is a stub.
**Gate:** grill ADR-030. Sequenced after KL1/KL2 by the 2026-08-03 review (Phase C).

### F7 · Source trust

Named signals, no composite score (ADR-031 stub); the outbound half (T5, ADR-032 stub) is parked
and is the project's first *enrichment* network feature — ADR-044 (update check) already set the
transport/privacy discipline it inherits. **Gate:** grill ADR-031.

### F8 · Reports & literature review

Generation presets on a frozen citation contract (ADR-033 stub) and the PRISMA-trAIce export (row
14, Phase 9). **Gate:** grill ADR-033; RP3 also needs T3.

### F9 · Search

Two small rows (58, 59). The embedder is already loaded, so semantic search is cheap; the literal
match stays the default.

### F10 · Platform, release, security

**Shipped:** Tauri + FastAPI/SSE shell (M0–M5), frozen sidecar + installer (KI-9/10/11, KI-34),
in-app provider setup (ADR-034), update notification (ADR-044), release runbook + preflight
(`docs/RELEASE.md`), CI for Python · frontend · Docker image (2026-09-04), every release tagged,
the dependency-advisory gate that blocks CI (row 61, 2026-09-30).
**Next:** row **100** now, and row **60** is standing: one security step per session from
`docs/security.md` §4. Row **76** closes KI-58's open half, the Windows CI job. Row 63 is a trade the
user has to make; 64 is parked by user call; 87/88 are host actions for the user.

### F11 · Quality, rigor, reviews

The eval record is `evals/README.md`; the debt is `.claude/RIGOR_TODO.md` (four `blocks-ship`
items open, no gate wired) and `.claude/REVIEWS.md` (backend and frontend never read as modules).
**Next:** row **98**, the docs revamp, with row 89's decisions; then 68 (decide each open
blocks-ship item), then 69 ($0), which rides with row **113**: the harness reinforced for the
chat work, before the chat changes it has to measure. Rows 70/71 are review sessions,
Cowork-shaped, and 71 also unlocks the automatable half of the release walkthrough.

### F12 · Verification debt

Built-but-never-driven surfaces (row 73) and two uncaptured defects (row 74). Row 73 is what keeps
Phase 8 "open"; the pre-release walkthrough `docs/release-ux-checklist.md` exists so that this list
stops growing — everything driven there before a release is no longer debt.

## What NOT to do

- Don't refactor the overall architecture. Locked decisions (`docs/decisions/` + the frozen
  monolith) are locked for a reason.
- Don't add SPECTER2 *and* PubMedBERT *and* MedCPT at once. Pick one; biomedical models are a separate,
  corpus-gated decision.
- Don't over-engineer the eval harness. Pydantic + pytest + DuckDB + Anthropic judge. No frameworks.
- Don't extract the standalone eval repo before the integrated version produced a real comparison.
- Don't splice figures into the markdown. Sidecar manifest only.
- Don't show self-reported LLM confidence. Use retrieval-derived uncertainty markers + reviewer output.
- Don't auto-retry or auto-remediate on reviewer-flagged issues — surface them; the user decides.
- Don't mine reviewer suggestions for "patterns" without the eval-set anchor (Chunk 2c).
- Don't sweep chunking without re-embedding; don't change chunk-size defaults from a single run
  (use `--repeat` and beat the control beyond its variance).
- Don't hand-author wiki notes — they're derived and regenerable. The wiki is additive, not a RAG
  replacement.
- Don't make the concept graph a graph database (NetworkX + a file artifact, build-time structure).
- Don't let Zotero/Calibre adapters leak vendor specifics past the extractor boundary.
- Don't build the three "project" groupings separately (row 47), and don't edit the citing block of
  `ANSWER_PROMPT` from a chat-mode feature (row 37).
- Don't add a surname list to the keyword filter — `cre`/`dbs`/`16p11`/`c57bl` are real terms
  (memory note *corpus-is-multidomain-not-junk*).

## References

- AI Usage Cards — arXiv 2303.03886 (provenance card schema)
- PRISMA-trAIce — PMC12694947 (Phase 9 export target)
- BE WISE framework — Frontiers, April 2026 (influence on dual-layer / `SYNTHESIS_MODE=human`)
- Karpathy LLM-wiki pattern — structured markdown as an LLM-queryable knowledge base (influence on
  Feature 6; layered *on top of* RAG, not a replacement — the "70x more efficient" framing is marketing).

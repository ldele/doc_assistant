<!-- status: active · updated: 2026-09-10 (the user's order: gaps with one security step per session → row 25 → the graph re-pass (row 75); row 62 done → archive; plans/reviews moved to docs/plans/ and docs/reviews/) · class: living -->

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
> **How to read a row.** `Status` is one of `planned` (agreed, not started) · `gate first` (an ADR,
> spec, grill or measurement must land before code) · `blocked` (waits on another row or a KI) ·
> `deferred` / `parked` (deliberately not scheduled — the reason is in the row). One PR per session
> (AGENTS.md build protocol). A row's *why* lives in its spec/ADR/DEVLOG, linked in `Spec` — do not
> restate it here.

## Where the product stands (2026-09-10)

**v0.6.0 is published** (2026-09-04; installer + headless image + source checkout). Shipped and
measured: hybrid RAG with page-level citations, the integrity layer (provenance, evidence vs
interpretation, reviewer), figures, the curated concept graph + gap list + taxonomy substrate,
folders with retrieval scope, add-documents (copy or reference), Zotero import, per-part re-ingest,
the source viewer with page jump and in-context passage. Quality record: `evals/README.md`; cost
record: `docs/performance.md`; what the knowledge layer's signals are worth: `docs/knowledge-layer.md`.

**Now → next → later** (user's order, 2026-09-10 second session — supersedes the review's own suggestion from earlier that day):

1. **Now — the gaps, three strands in parallel, one small security step every session.**
   - *The release loop:* row 46 (the citation gate), then 61 (advisories = security step S-8).
     Row 62 landed 2026-09-10.
   - *Security, a little each session:* `docs/security.md` §4 is the ordered list; row **60** names
     the current step. A session's security slot is one step — small code, its test, one line in
     the doc. The full check (§6) runs once when the list is done.
   - *The knowledge layer made to work:* **KL1** (tell the truth) → **53 + 54** ($0 measurements) →
     **51** (a real concept hierarchy + TX3b) → **KL4** (measure the unmeasured) → **KL2** (the
     acquisition half — design first, ADR-032).
2. **Next — the feature:** row **25**, LLM-assisted ingestion (spec + ADR, then build), with 56
   riding on it.
3. **Later — the graph re-pass:** row **75** — a deliberate second pass over the Graph tab now that
   the vocabulary, hierarchy and gap grades are honest: 52 · 57 · 50 · 43 (43 waits on KL3), plus a
   UX pass on navigation. Then 47 (the Project ADR) and the rest of F4.

## Sequence — the next sessions, in order

Each line is one session; the security column is that session's step from `docs/security.md` §4.

| # | Main work | Security step |
|---|-----------|---------------|
| 1 | 46 — the citation gate asserts over three turns ($0) | S-1 ingest size caps |
| 2 | KL1 — the knowledge layer tells the truth (strip column · markers · `unsourced_claim` headings · RG-014 grades in the list) | S-2 walk cap |
| 3 | 53 + 54 — the merge-cosine sweep and the denominator tests ($0) | S-3 sanitise `{@html}` + `devCsp` |
| 4 | 51 — `is_a` edges among concepts + TX3b accept/reject | S-4 CSP residual |
| 5 | KL4 — RG-015 placement quality · RG-018 community flip | S-5 host guard |
| 6 | 61 — advisories triaged, `pip-audit` blocking | (= S-8, the whole slot) |
| 7 | KL2 — ADR-032 grill: the acquisition half designed | S-6 launch token |
| 8 | 25 — LLM-assisted ingestion: spec + ADR | S-7 source-viewer containment |
| 9 | 25 — build, part 1 (the sidecar + the review surface) | S-10 the small ones |
| 10 | 25 — build, part 2 (+ 56 concept marking at ingest) | S-11 security events in the log |
| 11 | 75 — the graph re-pass (52 · 57 · 50; 43 if KL3 is done) | S-9 prompt fence (this session also runs the eval) |
| 12 | The periodic full security check (§6) → `.claude/REVIEWS.md` row 7 | — |

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
| 46 | F3 Chat | **RG-012's citation half** — the ship gate's citation verdict is a coin flip on `llama3.1:8b`; assert over 3 turns with ≥1 cited (option 2), $0, before the next release | planned — **blocks-ship** | `.claude/RIGOR_TODO.md` RG-012 (2026-08-14) · KI-35 · KI-36 |
| 61 | F10 Platform | **Dependency advisories** — 66 across 16 packages with `pip-audit` on `continue-on-error`; `aiohttp` and `starlette` ship in the sidecar and the image. Triage: upgrade what the lock allows, pin an ignore-list with reasons for the rest, then make the step block on HIGH/critical | planned — own session | `.github/workflows/ci.yml` · `docs/security.md` |
| 60 | F10 Platform | **Security — one step per session.** `docs/security.md` §4 is the ordered plan (S-1 … S-12, then the full check §6); this row names the **current step** and moves each session. Foundations that landed 2026-09-10: gates cover `apps/`, `npm audit` in CI, the Tauri config guard test, loopback compose, the floor + threat model written down | in progress — **current step: S-1 ingest size caps** | `docs/security.md` §4 · `docs/reviews/REVIEW_2026-09-10_project-review.md` §4 |
| 73 | F12 Verification | **Live-turn verification batch** ($0, Ollama): sandbox knobs change retrieval on a real answer · provider switch end-to-end incl. the reviewer following it · epistemics marker chips render (KI-15's fix has never been seen live) · RH1 reranker cap under multi-query · RG-012 Tier-2 on the frozen build. Becomes the standing pre-release walkthrough | planned | `docs/release-ux-checklist.md` · `docs/ui-checklist.md` §3 (local) |
| 25 | F1 Ingestion | **LLM-assisted ingestion mode** — opt-in pass over the programmatic default (Ollama-first, KI-4 guard): figure links, citation links, **reference order** (needs an additive ordinal column at extraction time — it cannot be backfilled), concept marking at ingest. Output is a durable, inspectable, re-derivable sidecar (Enrichment-Layer Pattern; ADR-043's "normalisation is a derived layer") | gate first — spec + ADR (new ingest mode, cost-gated) | `docs/plans/PLAN_2026-08-11_ingestion-quality.md` §2 (local) · `.claude/CONTEXT.md` direction note |
| 75 | F5 Knowledge | **Graph re-pass** — a deliberate second pass over the Graph tab once the vocabulary, hierarchy and gap grades are honest (after 25): the `flat_field` detector (52) · gap-list / Connections navigation iteration (57) · placement of the gap list and Connections (50) · rich marker UI (43, waits on KL3) · a UX pass on the concept rail, ego view and empty states against `docs/release-ux-checklist.md` §4 | later — after 25 | rows 52 · 57 · 50 · 43 · `docs/knowledge-layer.md` |
| 51 | F5 Knowledge | **Concept hierarchy for concepts** — `is_a` edges among curated concepts (ADR-019/ADR-028 machinery is built; the taxonomy is *empty*: 13 of 357 concepts placed, 0 `is_a`, ADR-045) + **TX3b** in-app accept/reject of proposed placements. Raised by the user three times | planned | ADR-028 · ADR-045 · `docs/specs/feature-taxonomy-auto-propose.md` |
| KL1 | F5 Knowledge | **Knowledge layer tells the truth** (Phase A): stop presenting stance-derived output as an epistemic finding · ADR-041 option 6 — re-base per-concept status on the claim layer · fix `unsourced_claim` heading contamination (~33%) · encode RG-014's grades in the gap list (lead `single_source`, `under_connected` off by default, suppress `thin_bridge` hub endpoints) | planned | `docs/knowledge-layer.md` · ADR-040 · ADR-041 · KI-33 · `docs/plans/PLAN_2026-08-03_knowledge-layer-to-goal.md` Phase A (local) |
| KL2 | F5 Knowledge | **The acquisition half** (Phase B) — the goal's operative capability, unimplemented: re-point `gap_suggest` off `under_connected` onto `single_source` (cheapest real win) · grill **ADR-032** then build the outbound reach (*"for subject X, read Y"*) · the taxonomy as the reference class for expected coverage | gate first — ADR-032 grill | ADR-004 (Tier-2b) · ADR-032 · PLAN 2026-08-03 Phase B |
| KL3 | F5 Knowledge | **Node-B stance rebuilt on evidence** (Phase D, ADR-041 option 1): passages in the prompt + a `neutral` label + one pair per call; **a hand-labelled ground-truth set is the gate, not a follow-up**. Decide whether `superseded_trend` survives on `doc_years` alone | gate first — ground-truth set | ADR-041 · KI-33 · `tests/eval/baselines/node_b_stance_validity_2026-08-02.md` |
| KL4 | F5 Knowledge | **Measure the two unmeasured shipped layers** (Phase E): RG-015 taxonomy placement quality (specced, never run) · RG-018 wiki community flip on the real corpus | planned ($0) | `.claude/RIGOR_TODO.md` RG-015 · RG-018 |
| 52 | F5 Knowledge | **`flat_field` depth detector** (crossover review §3.1): a field whose concepts all sit at `is_a` depth ≤ 1 — a vocabulary with no taxonomy. Every Tier-1 detector counts edges; none measures depth, and ADR-028's acyclic `is_a` edges make depth well-defined. Only fires once row 51 produces `is_a` edges | blocked on 51 | `docs/reviews/REVIEW_2026-09-07_crossover-daily-learn.md` §3.1 (local) · ADR-028 D3 |
| 53 | F5 Knowledge | **Sweep `CONCEPT_MERGE_COSINE`** (crossover §3.2): 0.80 / 0.85 / 0.90 over the current glossary, hand-scored on the pairs that change, filed as `tests/eval/baselines/concept_merge_cosine_<date>.md`. The reversible knob (`min_degree`) is measured; the irreversible one (a merge drops a row) is a default | planned ($0, one session) | crossover §3.2 · `config.py` `CONCEPT_MERGE_COSINE` · REVIEW 2026-07-19 CS-7 |
| 54 | F5 Knowledge | **Denominator-parity tests** (crossover §3.3): every coverage number asserts its denominator against the artifact it describes — `presence_nodes()`'s `kind` guard vs the shipping node set; document-coverage math vs the document table (ADR-019 D6's "22 of 47" was caught by reading) | planned ($0) | crossover §3.3 · `knowledge/taxonomy.py` · ADR-028 D4 |
| 55 | F5 Knowledge | **`citation_missing` floor** — named in `GapKind`, never built (the other Tier-2a deterministic kind) | planned | ADR-004 · `knowledge/gaps.py` |
| 47 | F4 Projects | **Project ADR** — one grouping across documents · conversations · concepts (folders, home screen and per-project concept view are *one idea*, not three). Open: one shared tree or two? may a project scope retrieval like a document folder (ADR-025 F2)? per-project graph (= ADR-025 fork 5)? what happens to a project when its chats are soft-deleted? | gate first — ADR | `docs/ui-checklist.md` §2 Projects (local) · ADR-025 |
| 48 | F4 Projects | **Home screen / project picker** — the first screen becomes "which project am I in" | blocked on 47 | — |
| 49 | F4 Projects | **Conversation folders** — a new relationship (`Folder` is document-scoped, ADR-025 F1), never a schema reuse | blocked on 47 | ADR-025 |
| 50 | F4 Projects | **Graph placement + gap-list home** — the Graph tab is back (row 22) but where the gap list and Connections ultimately live is the Project ADR's call | blocked on 47 | ADR-017 · RG-014 |
| 37 | F3 Chat | **Chat modes** — named, user-customisable system prompts. **Hard boundary:** `ANSWER_PROMPT`'s citing block is the *wire format* `synthesis._CITATION_RE` parses (it broke once, 2026-07-14) — persona/task framing is editable, the citing block is not; `chat_controller` must hash the template per turn, not at construction, or provenance lies | gate first — spec | `docs/ui-checklist.md` §2 Chat (local) · `prompts.py` |
| 38 | F3 Chat | **Structured/template answer mode** over extracted metadata, keywords, concepts | blocked on 37 | — |
| 39 | F3 Chat | **RAG + Internet mode** — same family as KL2/T5 (transport spiked: stdlib `urllib` → Crossref 25/25). Own ADR: provider list, quality list, caching, provenance of an acquired source, offline degrade; inherits ADR-044's transport/privacy discipline | gate first — ADR | ADR-044 · ADR-032 |
| 40 | F3 Chat | **Evidence-only mode polish** — `synthesis_mode=human` already answers with evidence only; the remaining decision is to force multi-query off and gate the rewrite so the mode is genuinely $0 | planned (small) | `chat_controller/controller.py` |
| 41 | F3 Chat | **Unconstrained mode** — corpus restraint off, measurements on; which variant, and how the reviewer still grades groundedness | gate first — grill | — |
| 42 | F3 Chat | **Highlight cited claims in the answer text** — presentation over `answer_claims` + `result.sources` (in the extracted markdown; the on-page half is row 24) | planned | `how-answers-work.md` |
| 43 | F3 Chat | **Rich marker UI** — hover a contested/superseded chip → the corroborating documents. `contested` is not a measurement today | blocked on KL1/KL3 | `docs/knowledge-layer.md` §6 · KI-33 |
| 44 | F3 Chat | **Resumable chat rehydration** — reopened chats are read-only; claims + reviewer joins are the follow-up | planned | `docs/specs/feature-conversation-resume.md` |
| 45 | F3 Chat | **User-tunable RAG pipeline** — reopens ADR-010, whose non-persistence is the governance wall. Finding that motivated it: `TOP_K` was never the problem; **`EMBEDDING_MODEL` is the catastrophic knob and appears in neither ADR-010's split nor the locked-settings table** — map blast radius first | gate first — grill + ADR | ADR-010 · `.claude/ui-checklist-archive-001.md` (local) |
| 24 | F2 Library | **Highlight the cited passage on the page image** — measured viable (4-word anchor places 90%, envelope 97% pure). Must solve: real column detection · 43% of parents straddle a page break (say so) · a stated decline-never-guess policy for ambiguous anchors | planned | ADR-050 Addendum · DEVLOG 2026-09-01 (2) |
| 32 | F2 Library | **Source explorer: chunk → parent → document** from the citation panel (~1 endpoint + panel UI; `parent_index`/`_chunk_key` exist) | planned | — |
| 33 | F2 Library | **Chunk editing + colour-coded chunk state** — editing collides with the Enrichment-Layer rule; split the read half (plain UI) from the write half (annotation sidecar) | gate first — ADR | memory note *future-user-annotatable-figures-chunks* |
| 34 | F2 Library | **ADR-046's amended delete** — `delete_document` takes `delete_file`, asks, defaults to library-only, never bins a *referenced* file (the "still design" half of ADR-046) | planned | ADR-046 · ADR-014 |
| 35 | F2 Library | **Missing-source badge** — the ui-checklist row named three `resolve_source_path` bugs; KI-52 fixed the registry half. **Verify what remains against the code before planning** | planned (verify first) | KI-52 · `docs/ui-checklist.md` §2 (local) |
| 27 | F1 Ingestion | **Tables** — extraction code exists but **no table has ever landed** (Marker unrunnable here, KI-42 fixed the pin); diagnose on the live corpus before designing; styled table rendering after | gate first — diagnose | `docs/figures-and-tables.md` · KI-42 |
| 26 | F1 Ingestion | **Keyword auto re-trigger on corpus growth** (P1 D3) — the same question as KI-44: a sidecar can only reach retrieval through a global `--rebuild` (~4 min at 97 docs, ~3.6 h at 10k) | planned | KI-44 · `docs/plans/PLAN_2026-08-11_ingestion-quality.md` (local) |
| EX1 | F1 Ingestion | **OCR sidecar for true scans** (ADR-039) — 1 of 97 documents; opt-in, restores a text layer not markdown, Tesseract absent-tolerant (KI-47), **gated on RG-025** (wrong OCR text is worse than none). Parts (a) extractor-lost text and (c) KI-40 cache key are done | gate first — RG-025 | ADR-039 · RG-025 · KI-47 · KI-48 |
| 28 | F1 Ingestion | **Extended metadata + Crossref autocomplete** — surface the stored DOI; add journal/url/article_type (~6 appends to `_ADDITIVE_COLUMNS`); local-text yield is hopeless, Crossref wins on all four. Shares the outbound transport with T5/39 | gate first — ADR-016 (number reserved) | `.claude/ui-checklist-archive-001.md` (local) · ADR-044 |
| 29 | F1 Ingestion | **Calibre adapter** (one module + one route, ADR-049) **and** run the Zotero adapter against a real library — the tests prove the mapping, not the schema | planned | ADR-049 |
| 31 | F1 Ingestion | **Ingest honesty — Track B of the add-documents plan** (per-file outcome instead of a batch total, retry, what changed); not started | planned | `docs/plans/PLAN_2026-08-20_user-friendly-ingestion.md` §4 (local) |
| 30 | F1 Ingestion | **Per-document ingestion tuning** (a table-heavy paper chunked differently). Contained but not free: BM25 `avgdl` is corpus-global, `TOP_K` counts parents, splitters are import-time singletons | gate first — grill | `.claude/ui-checklist-archive-001.md` (local) |
| 56 | F5 Knowledge | **Better concept highlighting** — mark concepts at ingest (rides on 25), more reliable than at read time | blocked on 25 | — |
| MM1 | F6 Maps | **Document outline layer** — populate `DocumentPart` from the cached markdown; `char_start`/`char_end` + part→`parent_index`; idempotent backfill; no LLM | gate first — ADR-030 is a stub | ADR-030 · `docs/plans/PLAN_2026-07-27_maps-trust-reports.md` Track 1 (local) |
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
| 63 | F10 Platform | **Slim installer as a user option** — download weights on first run. A *trade* against KI-9's offline-from-first-launch promise, not an improvement: two artifacts per release, touches KI-10 and splits RG-010 | gate first — ADR | KI-9 · KI-10 · RG-010 · `docs/RELEASE.md` |
| 64 | F10 Platform | In-app API-key entry via an OS keychain (ADR-011 v2) | parked — keyring decision | ADR-011 · ADR-034 |
| 65 | F10 Platform | Global CLI + local stdio MCP server over `pipeline.py` | parked — user call 2026-07-13, freeze the API surface first | — |
| 15 | F10 Platform | Extract the eval harness to a standalone repo (Feature 5) | planned — after a real comparison has been produced with the integrated one | ADR-024 |
| 66 | F10 Platform | Lift the Python 3.12 pin (KI-2) | blocked — external (native deps not cp314-stable) | KI-2 |
| 6 | F11 Quality | Per-project embedder routing (Feature 1b) — re-run SPECTER2 `--repeat 5` first | deferred | `evals/README.md` |
| 68 | F11 Quality | **Close or waive the open blocks-ship rigor items with a date** — RG-001/008 (edge precision gate), RG-014 (spec has not absorbed the verdict), RG-027 (ADR-042 identity migration + backfill). `rigor_gate.py` does not exist; the file is a manual discipline doc — say so or build it | planned | `.claude/RIGOR_TODO.md` |
| 69 | F11 Quality | **`CANDIDATE_K=20` retest** on the private 35 with `--repeat` — the verdict has been "unvalidated" since 2026-06-13 | planned ($0) | `tests/eval/baselines/candidate_k_public_2026-06-13.md` · memory note *candidate-k-retest-needed* |
| 70 | F11 Quality | **Backend code review as a module** (`.claude/REVIEWS.md` row 2 = never); start with `ingest/` (nine tracked defects) and `chat_controller/` | planned — Cowork-shaped | `.claude/REVIEWS.md` |
| 71 | F11 Quality | **Frontend code review + a component test harness** — 39 `.svelte` components untestable under `node:test`; needs the lockfile decision (vitest + a Svelte testing library) and would automate half of `docs/release-ux-checklist.md` | planned | `.claude/REVIEWS.md` row 3 · `apps/desktop/CLAUDE.md` |
| 74 | F12 Verification | **Two uncaptured defects** (user, 2026-07-21): the collapse-sidebar button and the global search bar misbehave — capture the exact symptom before fixing | planned — repro first | `docs/ui-checklist.md` §3 (local) |

## Features

One block per feature: what is shipped (one line, pointers only), what is next, and the gate.

### F1 · Ingestion & document quality

**Shipped:** extract → markdown → chunk → embed → store (locked); registry + cache (KI-40 fixed);
figures (4b/4c); metadata extraction + manual override (ADR-013) + external catalogue slot (ADR-049);
selective ingestion (S1/S2); add-documents copy-or-reference (ADR-046, AD1–AD3b); Zotero import
(row 17); per-part re-ingest (ADR-048, rows 20/21); keyword de-noising D1/D2/D4/D5 (2026-08-12);
document identity survives re-extraction (ADR-047); extractor-lost text recovered (EX1a).
**Open defects that shape this feature:** KI-44 (sidecar → retrieval needs a global rebuild),
KI-46 (zero-chunk duplicates), KI-47/48 (Tesseract, invalidated caches), KI-54 (title picker),
KI-57 (page marker carries the previous page). **Next:** row **25** — it is the user's stated
priority and the direction note in `.claude/CONTEXT.md` justifies its cost. **Gate:** a spec + ADR
that names the sidecar schema and the cost statement (ADR-048's "say what it costs first").

### F2 · Library & reading

**Shipped:** chunk browser (L1), redesign (L4), folders + retrieval scope (ADR-025), keyword
filter + families (ADR-015), safe delete (ADR-014), figure panel + viewer, references block,
Connections panel (E4), source viewer with page jump (ADR-050, row 18), in-context passage (row
19), graph-vocabulary toggle (row 23). **Next:** row **24** (the on-page highlight — measured
viable, the design questions are listed in the row). Row 34 is small and closes an ADR's open half.

### F3 · Chat & answers

**Shipped:** streaming SSE chat, citation panel (U3), provenance + reviewer cards, A/B compare
(U6), conversation history + cleanup (U5), source-evaluation strip (E2) with the stance-derived
chips withheld (KI-33), epistemics toggle (E3), in-context + show-the-page from a citation.
**Next:** row **46** first — it is the ship gate. Then row **37** (chat modes) with its hard
boundary respected. Anything touching `contested` waits on KL1/KL3. Local-model citation coverage
is a floor (KI-36: Haiku 81%, `llama3.1:8b` 36%) — every chat feature must state which model it
was verified on.

### F4 · Projects & navigation

**Shipped:** the shell (sidebar │ main │ drawer), global navigation search (titles), the Graph tab
with a real empty state (row 22). **Next:** row **47**, the Project ADR, *before* any of 48–50:
three grouping systems that disagree is the failure to avoid. **Gate:** `grill-me` on the four
open questions in the row.

### F5 · Knowledge layer (concepts · taxonomy · gaps · epistemics)

**Shipped:** curated vocabulary + deterministic skeleton (Node A) + confined LLM stance (Node B),
gap detectors Tier-1 + Tier-2a floor/ceiling, gap list with durable triage (E5), taxonomy substrate
+ curation backend/view + auto-propose (TX1–TX3), graph coverage statement (2026-08-31), in-app
vocabulary curation (row 23). **Read `docs/knowledge-layer.md` §6 before believing any number:**
`single_source` is trustworthy, `under_connected` is noise at this vocabulary size, `contested` is
**not a corpus measurement**. **Next (user's order, 2026-09-10, second session):** KL1 → 53 + 54 → 51 → KL4 → KL2, interleaved with the security
steps; then, after row 25, the **graph re-pass** (row 75). Rows 52–54 come from the 2026-09-07
crossover review; 52 waits on 51's `is_a` edges.

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
(`docs/RELEASE.md`), CI for Python · frontend · Docker image (2026-09-04), seven tagged releases.
**Next:** row **60** is standing — one security step per session from `docs/security.md` §4 (row
62 landed 2026-09-10; 61 is step S-8). Row 63 is a trade the user has to make; 64/65 are parked by
user call.

### F11 · Quality, rigor, reviews

The eval record is `evals/README.md`; the debt is `.claude/RIGOR_TODO.md` (four `blocks-ship`
items open, no gate wired) and `.claude/REVIEWS.md` (backend and frontend never read as modules).
**Next:** 68 (decide each open blocks-ship item), then 69 ($0). Rows 70/71 are review sessions,
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

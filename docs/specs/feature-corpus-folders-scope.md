# Spec — Corpus folders, F2: query-time retrieval scoping (the integrity piece)

**Status:** contract for **F2** of the ADR-025 carve (F1 folders ✅ → **F2 retrieval scoping** →
F3 demo auto-assign). Written at build time per the house pattern.

**Owner:** Claude Code. **ADR:** `docs/decisions/ADR-025-corpus-folders-retrieval-scope.md`
(forks 3 + 4). **Builds on:** `docs/specs/feature-corpus-folders.md` (F1 — folders, membership,
`folder_document_ids`). **Rigor:** `.claude/RIGOR_TODO.md` **RG-020**.

> **Why this is the integrity piece.** F1 shipped a Library filter and said so in-product (F1 spec
> D8): folders organise the Library, chat still searches everything. F2 is what makes that line
> false — and the same honesty rule now points the other way. A scoped turn must *say* it was
> scoped, in the provenance record and on the answer, or the app has simply learned a quieter way
> to tell the `is_archived` lie.

---

## Decisions

| # | Decision | Reason / reopens-if |
|---|---|---|
| **S1** | **The scope rides `ChatRequest` as a sibling of `overrides`, not as a field inside `RagOverrides`.** Same request-scoped, never-a-module-global mechanics; different channel. | ADR-025 fork 4 draws the line this implements: a scope is a **content filter (which documents)**, not a **quality knob (how retrieval works)**. `RagOverrides` is ADR-010's governance channel for *locked settings* — putting a content filter in it would file the scope under locked-settings governance and blur exactly the distinction that keeps it out of the eval gate. Cosmetically it would also render through `_overrides_note`'s "🧪 Session override (this answer only)", which frames a content choice as an experiment. |
| **S2** | **The client sends a folder id; the backend resolves membership per turn** (`library.folder_doc_hashes`). The client never sends a hash list. | Membership is server-authoritative and instantly editable (ADR-025: no chunk-store writes). A client-sent hash list goes stale between a Library edit and the next turn, and would let a caller retrieve an arbitrary set no folder ever contained — which the provenance record would then attest to as "this folder". |
| **S3** | **An unknown, deleted, or empty folder resolves to an EMPTY scope and the turn answers honestly with zero sources. There is never a silent fall back to unscoped.** | This is the whole feature. Answering over the entire corpus when the requested scope can't be honoured is the `is_archived` lie in its purest form. A zero-source turn already degrades honestly (the 0-document robustness contract), so the honest option is also the cheap one. Inform-don't-block: nothing is refused, no 4xx, the chip states the scope and the answer states it found nothing. |
| **S4** | **The unscoped path stays byte-identical.** `scope is None` ⇒ `retrieve_with_scores` takes today's exact path (`self.ensemble`, no filter dict constructed, no subset built). | RG-020 owes this as a guard test. It is also what lets the eval harness keep comparing against committed baselines. |
| **S5** | **Scoped retrieval builds a per-scope ensemble, memoised in a single slot keyed on the exact `frozenset` of doc hashes.** | Measured on the live index (below): the BM25 arm must be rebuilt over the subset, ~20 µs/chunk. The UI scope is sticky, so consecutive turns share a scope → one slot is a ~100 % hit rate after the first turn. Keying on the hash set itself means a membership edit produces a different key and **self-invalidates** — no TTL, no eviction policy, no staleness window. **Reverses if** multi-scope interleaving becomes common (→ small LRU) or `$in` fails the 10k measurement (→ per-folder precomputed indexes, the ADR-025 fallback). |
| **S6** | **Both arms scope, before scoring.** Vector: Chroma `where {$and: [keep_for_retrieval ≠ False, doc_hash $in [...]]}`. BM25: subset the in-memory corpus by `doc_hash`, then build. | ADR-025 fork 3. Post-rerank filtering is the rejected cheap-but-wrong option — recall collapses exactly when the scope is small. |
| **S7** | **Provenance gains an additive `retrieval_scope_json` column** on `answer_records` (`_ADDITIVE_COLUMNS`, append-only), carrying `{folder_id, folder_name, doc_count}`; `NULL` on unscoped turns. Rendered into the provenance card. **`prompt_version_hash` does NOT fold in the scope.** | Additive columns are the house migration mechanism, and `NULL` means existing records read back unchanged. The scope is *content*, not prompt/retrieval config — folding it into `prompt_version` would mint a distinct prompt version per folder and pollute every eval join keyed on it. |
| **S8** | **`TurnResult` gains `scope: ScopeView \| None`; the desktop renders a chip on the answer for every scoped turn** — both the `ai` and the `human` synthesis paths. | ADR-025 fork 4, explicitly non-negotiable. Two result builders exist; a chip on only one is a lie on the other. |
| **S9** | **The composer's scope selector is in-memory for the app session only — never `localStorage`.** A reload returns to the whole library. | Mirrors ADR-010's overrides exactly ("a fresh launch always starts from `{}`"). Persisting it *is* the rejected "persistent/global scope" option: a forgotten scope silently narrows every future answer, and the whole integrity layer exists to prevent that class of lie. Sticky within a session is what "sticky in the UI only" means. |
| **S10** | **RG-020 is partially discharged with real numbers and stays open for the 10k claim.** | RG-020's own "Until then" clause authorises shipping at ~10² corpus sizes provided we do not claim the 10k contract holds for scoped turns. Measured numbers below + a baseline file; the synthetic 10k run stays owed. |

## Measurements (live index, 2026-07-20 — 76 documents / 30,882 chunks, this box)

Recorded before building, to make S5 a decision rather than a guess.

**BM25 subset rebuild** (`BM25Retriever.from_documents`, `tokenize` preprocess):

| scope | docs | chunks | subset filter | index build | query |
|---|---|---|---|---|---|
| whole corpus | 76 | 30,882 | 4.1 ms | **622 ms** | 52 ms |
| 40 % | 30 | 9,331 | 3.2 ms | **248 ms** | 11 ms |
| 5 % | 3 | 806 | 3.2 ms | **27 ms** | 1.7 ms |

**Chroma `$in` vector filter** (same index, `CANDIDATE_K` results, median of 5):

| filter | median |
|---|---|
| unscoped (today's `keep_for_retrieval ≠ False`) | 136 ms |
| `+ doc_hash $in [3 hashes]` | 193 ms |
| `+ doc_hash $in [30 hashes]` | 232 ms |
| `+ doc_hash $in [76 hashes]` | 408 ms |

**Reading.** Both costs are acceptable at real corpus sizes and are dominated by LLM latency, and
the S5 cache removes the BM25 rebuild from every turn after the first on a given scope. **Neither
extrapolates comfortably to the 10k-document contract:** BM25 build is ~linear in chunks (~20
µs/chunk), and the `$in` cost is driven by the *length of the hash list*, not the corpus size —
a folder holding thousands of documents is the bad case for both. That is precisely ADR-025's
"reverses if `$in` fails the latency measurement" trigger, and it stays live under RG-020. Note
the pathological case is also the least useful one: scoping to nearly the whole corpus is what
"no scope" already expresses for free.

## Contract

### Backend — `src/doc_assistant/library.py`

```
folder_doc_hashes(folder_id) -> list[str]   # non-archived members' Document.doc_hash; [] if unknown
```

Mirrors `folder_document_ids` (F1) and inherits its archived-exclusion rule (F1 spec D5).

### Backend — `src/doc_assistant/pipeline.py`

- Keep the BM25 corpus: `self._bm25_docs` (today's local `all_docs`, already
  `keep_for_retrieval`-filtered), plus `self._scoped: tuple[frozenset[str], EnsembleRetriever] | None`.
- `retrieve_with_scores(query, top_k, *, use_multi_query=None, scope: frozenset[str] | None = None)`.
  - `scope is None` → today's path, untouched (S4).
  - `scope == frozenset()` → return `[]` **without touching the retrievers** (S3).
  - otherwise → `self._ensemble_for(scope)`, a cached ensemble of a `$in`-filtered Chroma retriever
    and a BM25 index over the subset. Empty subset (hashes present but no surviving chunks) →
    vector-only, same fallback shape the empty-library branch already uses.
- `retrieve(...)` grows the same keyword-only `scope` pass-through.

### Backend — `src/doc_assistant/chat_controller.py`

- `@dataclass ScopeView { folder_id: str, folder_name: str | None, doc_count: int }` — render-ready,
  no UI types (matches the `SourceView`/`UsageView` convention).
- `handle_message(session, text, *, overrides=None, scope_folder_id: str | None = None)` →
  threaded to `_handle_rag`. Resolution happens once per turn: `folder_doc_hashes` +
  `get_folder` for the display name; an unknown folder yields
  `ScopeView(folder_id, folder_name=None, doc_count=0)` and an empty hash set (S3).
- `TurnResult.scope: ScopeView | None`; set on **both** the `ai` and `human` result builders.
- Both `record_answer(...)` calls pass `retrieval_scope=...`.
- Only the RAG path scopes — commands, library queries, and claim adjudication are unaffected
  (the same carve `overrides` already has).

### Backend — provenance + migration

- `db/migrations.py` `_ADDITIVE_COLUMNS` += `("answer_records", "retrieval_scope_json", "TEXT", None)`.
- `db/models.py` `AnswerRecord.retrieval_scope_json: str | None`.
- `provenance.py`: `AnswerProvenance.retrieval_scope: dict | None`, written by `record_answer`,
  read back by `_row_to_provenance`, and rendered as a provenance-card line on scoped turns.

### API + wire

- `ChatRequest` += `scope_folder_id: str | None = None`.
- `TurnResultPayload` += `scope: ScopePayload | None` (`folder_id`, `folder_name`, `doc_count`).
- `apps/api/main.py` passes `scope_folder_id` into `handle_message`. `types.ts` mirrors both.

### Frontend — `apps/desktop/`

- `api.ts` `streamChat(text, sessionId, overrides, scopeFolderId?)`.
- `App.svelte`: `chatScopeFolderId = $state<string | null>(null)` — **in-memory only** (S9);
  cleared when the folder it names disappears from `folders`.
- A compact scope selector beside the composer: "All documents ▾" / "Folder: Demo corpus (30) ▾",
  listing the folders already loaded for the Library. Selecting a folder in **chat** mode does not
  touch `libraryCollection` (the Library filter and the chat scope are separate choices).
- `Turn.svelte`: a scope chip above the answer whenever `result.scope` is set, stating the folder
  and its document count — and reading as a *constraint*, not a decoration.

## Tests

- **Unit** (`tests/unit/test_pipeline_scope.py`, fake Chroma/BM25/reranker in the existing style):
  `scope=None` builds no filter and calls the prebuilt ensemble (**S4 byte-identical guard**);
  `scope=frozenset()` returns `[]` and touches no retriever (S3); a non-empty scope filters both
  arms; the S5 cache hits on a repeated scope and misses on a changed one.
- **Integration** (`tests/integration/test_retrieval_scope.py`, fake controller):
  `POST /api/chat` with `scope_folder_id` → the result payload carries the scope and the record
  persists `retrieval_scope_json`; unknown/deleted folder → empty scope, zero sources, chip still
  states the scope, **no fall back to unscoped** (S3); no `scope_folder_id` → `retrieval_scope_json`
  is `NULL` and the turn is unchanged.
- **Migration**: `retrieval_scope_json` lands idempotently on a pre-existing `answer_records`
  table, and an existing record reads back with `retrieval_scope=None`.
- **Gates:** ruff · `ruff format` · `mypy --strict src` · bandit · full `pytest` ·
  `docs_check`/`integrity_check --strict` · `svelte-check` 0/0.
- **Live ($0/offline):** scoped retrieval proven **without a paid turn** — drive
  `retrieve_with_scores` directly on the real corpus (scoped vs unscoped source sets) and verify
  the chip/selector render via the `window.fetch` SSE mock, never the real Send button.

## Definition of done

1. `scope` threaded end to end: request → controller → both retrieval arms, resolved server-side.
2. Unscoped path byte-identical, guarded by a test.
3. Empty/unknown scope answers honestly with zero sources — never unscoped.
4. Provenance records the scope (additive column, `NULL` for old rows); the card shows it.
5. Answer chip on **both** synthesis paths; composer selector, in-memory only.
6. Tests above green; full gate battery green; measurements recorded in `tests/eval/baselines/`.
7. F1 spec D8's honesty line removed — it is now false, which was the point.
8. One `docs/DEVLOG.md` entry; ROADMAP row; ui-checklist row updated; RG-020 updated with the
   measured numbers and what remains owed.

## Amendments (user decisions, 2026-07-20, same session)

- **S11 — the A/B compare scopes BOTH sides.** Originally out of scope. Reversed on the user's
  call: with a folder selected, an unscoped diff describes retrieval the next answer will not
  perform, which is the same class of quiet mismatch the rest of F2 exists to remove. Both arms
  take the resolved scope (holding the document set constant is also what makes the comparison
  *about the knob*), and `CompareResult.scope_label` puts a line on the card saying so.
- **S12 — KI-23 (filed as KI-20): the API lifespan migrates the schema and logs what it changed.** `init_db()` ran
  from `ingest` only, so additive columns never reached an install whose user just chats — and F2
  put one on the answer path. The lifespan now calls it; `init_db` returns the added columns and
  the lifespan logs them at WARNING. See `.claude/KNOWN_ISSUES.md` KI-23.

## Out of scope (F2)

F3 demo sha-match auto-assign · persisted per-conversation scope (ADR-025 reverses-into, needs an amendment) ·
multi-folder scopes · scoping the enrichment sidecars (ADR-025 fork 5, parked) · the synthetic
10k `$in` measurement (RG-020, stays open) · RG-021 (eval-harness fingerprint).

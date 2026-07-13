# Spec — App shell + conversation history (left sidebar)

**Status:** 🔒 **design-locked (2026-07-13, grilled).** All forks resolved — decision ledger at the
bottom. Ready to become a cpc SPRINT contract. Scope = the **left-sidebar app shell** + the
**Chat space's conversation history**. The **Library space** (in-app ingestion + Calibre-style chunk
browser + chunk annotations) is a *sibling* increment — this spec only reserves its nav slot; its design
is `docs/specs/feature-library-chunk-browser.md` (not yet written).
**Owner of execution:** Claude Code (backend read layer + endpoints + the Svelte shell). Create a cpc
SPRINT contract from this spec at build time (`docs/sprints/SPRINT-000-template.md` shape).
**Pattern reference:** thin-shell boundary (`apps/` render, logic in `src/doc_assistant/` — root
`CLAUDE.md` rule 3); FastAPI/SSE boundary (PR-M2); Tauri frontend (PR-M3); the U4 `↻ New` button
(`App.svelte`, DEVLOG 2026-07-13) is the "start a new session" primitive this builds on.

**Goal (the why).** A personal research tool accretes conversations worth returning to. Today the desktop
holds exactly one in-memory conversation; `↻ New` (U4) discards it. This feature lets the user **see and
reopen past chats** from a left sidebar — and reframes `↻ New` from *"clear"* to *"start a new chat; the
current one stays in the list."* It also lays the **app-shell** (left list · center content · right
inspector) that the Library space will reuse, so the IA is built once.

---

## Grounding — the seams this rests on (verified 2026-07-13, file:line)

- **The data already exists.** `AnswerRecord` (`db/models.py:596`) writes **one row per answered turn**,
  with an **indexed, nullable `session_id`** (`:618`) plus `query`, `original_query`, `answer`,
  `retrieved_chunks_json`, `model_name`, `top_k`, tokens, `created_at` (indexed). Its docstring states it
  is *"designed forward-compat for … future threading / sessions."* Reviewer/claims live in sibling
  tables (`AnswerReview :655`, `AnswerClaim :702`), joinable by `record_id`.
- **The frontend session already reaches the server.** `POST /api/chat` does
  `session = sessions.get_or_create(body.session_id)` (`apps/api/main.py:281`), so `session.session_id`
  **is** the frontend's id.
- **One wiring gap.** `record_answer(...)` in `_handle_rag` (`chat_controller.py:798`) **does not pass
  `session_id`**, so it currently persists as `None`. `record_answer`/`provenance.py:161` already accept
  the param — this is a one-argument fix, not a schema change. (Also check the second `record_answer`
  at `:967`.) Historical rows stay `None`; history is populated **from the fix forward** (documented
  limitation, no backfill).
- **Durable vs live is already separated.** Follow-up context lives in the in-memory `SessionStore`
  (`get_or_create`); the durable transcript lives in SQLite `AnswerRecord`. **History reads the durable
  records** — it does not resurrect the in-memory session (so reopening an old chat is a read-only
  transcript, not a live thread that remembers context — see Decision 4 / Fork B).

---

## Decisions (locked 2026-07-13 — grilled; ledger + "reopens if" at bottom)

| # | Decision |
|---|---|
| 1 | **History is backend-backed by `AnswerRecord.session_id`** (Fork A), not a `localStorage` list. Durable across restart/reinstall; each chat ties to its own provenance/review records; no second source of truth. |
| 2 | **Wire the write-path:** pass `session_id=session.session_id` into `record_answer(...)` at `chat_controller.py:798` (and `:967`). The one required backend mutation; everything else is reads. Pre-fix historical rows keep `session_id = NULL` and are excluded — history populates from the fix forward (no backfill). |
| 3 | **Two read endpoints, logic in `src/`:** `GET /api/conversations` → list `{session_id, title, turn_count, started_at, last_at}`; `GET /api/conversations/{sid}` → ordered turns to rehydrate. `apps/api` stays a thin renderer over a new `src/doc_assistant/conversations.py` read module. |
| 4 | **Reopening a past chat is a READ-ONLY transcript** (Fork B) — question + answer markdown + citation links (re-linkified from `retrieved_chunks_json`) + a **degraded citation panel** (excerpt/filename/page). **Not reconstructed** (not persisted / out of scope): marker chips, figures, claim-review, the rich provenance card. An old chat is reviewed, not resumed. |
| 5 | **The live chat is preserved** (H2) — the composer stays bound to exactly one *live* session (its turns render from in-memory state, fully interactive: claims etc.). Selecting a past chat opens it read-only in the main pane with a **"← Back to current chat"** bar in place of the composer; the live chat is never destroyed. The sidebar marks which row is *current*. |
| 6 | **Title = the first user question, truncated** (Fork C); auto, no rename in v1. |
| 7 | **`↻ New` (U4) lives in the sidebar header** (Fork D) — removed from the top bar. It already mints a fresh `session_id` and resets; a chat with zero turns never persists (no empty ghosts), a chat with ≥1 persisted turn appears in the list. |
| 8 | **Shell = CSS grid `sidebar │ main │ drawer`.** Sidebar collapsible; on mobile it's an off-canvas drawer (no body overflow). Sidebar top carries a **Chat / Library** switch; **Library is rendered but disabled** (Fork E — "coming soon"), the slot reserved not built. |
| 9 | **History shows answered turns only** (H3) — `record_answer` is only reached on success (`chat_controller.py:798`), so errored/aborted turns never persist and never appear. Desired: failures are transient. |
| 10 | **v1 lists the most-recent ~100 conversations, no delete/prune UI** (H4). This *reads* growth that `AnswerRecord` already accumulates; it doesn't create it. Retention/prune is parked to a later maintenance increment. |
| 11 | **No new locked settings, no `.env`/`config` writes, no paid calls.** Pure read feature + one persistence arg. |

---

## Contracts (build-time)

### `src/doc_assistant/conversations.py` — NEW read module (logic lives here)
- `list_conversations(limit: int = 100) -> list[ConversationSummary]` — group `AnswerRecord` by
  `session_id` (skip `NULL`), newest `last_at` first; title = earliest row's `original_query or query`
  (truncated); `turn_count`, `started_at`, `last_at`.
- `get_conversation(session_id: str) -> list[TurnRecord]` — ordered by `created_at`; each carries
  `query`, `answer`, and the parsed `retrieved_chunks_json` (for citation re-linkifying). Read-only.
- Pure SQL over the existing session — **no writes**, no LLM, no Chroma.

### `src/doc_assistant/chat_controller.py` — the one write fix
- `record_answer(..., session_id=session.session_id)` at `:798` (and `:967`). Guard test: a turn
  persists a non-`NULL` `session_id` equal to the caller's.

### `apps/api/main.py` + `apps/api/models.py` — thin endpoints + wire types
- `GET /api/conversations`, `GET /api/conversations/{session_id}` → the two read fns. Pydantic response
  models mirror `ConversationSummary` / `TurnRecord`. 404 on unknown `sid`.

### `apps/desktop/src/lib/Sidebar.svelte` — NEW
- Renders the Chat/Library switch + the conversation list (`GET /api/conversations`). Emits
  `onSelect(session_id)` / `onNew()`. Collapsible; mobile drawer. Reuses existing CSS vars (both themes).

### `apps/desktop/src/App.svelte` — shell restructure
- Wrap the current single column in the `sidebar │ main │ drawer` grid; `↻ New` moves into the sidebar
  header (Decision 7). State model (H2, Decision 5): keep the existing in-memory `turns`/`sessionId` as
  the **live** chat (composer + claims bound to it); add a separate `viewing: {session_id} | null`. When
  `viewing` is set, the main pane renders the **read-only** transcript fetched via `getConversation(sid)`
  and swaps the composer for a "← Back to current chat" bar (clears `viewing`); when `null`, it renders
  the live `turns` with the composer. The sidebar marks the live session's row as *current*.

### `apps/desktop/src/lib/api.ts` + `types.ts`
- `listConversations()`, `getConversation(sid)`; `ConversationSummary` / `TurnRecord` types mirroring the
  API models (`types.ts` ↔ `apps/api/models.py`).

---

## Guard tests ($0 — cpc §13, no paid calls)
- `conversations.list_conversations` groups/orders correctly; `NULL` session_ids excluded; title from the
  earliest turn.
- `get_conversation` returns turns in `created_at` order; unknown sid → empty/404.
- Write fix: a `_handle_rag` turn persists `session_id == caller session` (fake LLM, no network).
- Endpoint tests over a seeded in-memory DB (existing test-DB fixture).
- `svelte-check` 0 errors. Preview-harness: sidebar lists a seeded conversation, clicking rehydrates a
  read-only transcript; `↻ New` starts a fresh chat that appears once it has a turn; both themes; mobile
  no-overflow.

## Definition of done
- Backend: the write fix + two endpoints + read module, guard tests green, `mypy --strict` / `ruff` /
  `bandit` clean, full `pytest` green.
- Frontend: the shell + sidebar; reopening a past chat shows its transcript with working citation links;
  `↻ New` behaves per Decision 6; `svelte-check` 0; preview-harness-verified (light + dark + mobile).
- Docs: DEVLOG entry; ROADMAP row (U5?); ui-checklist §1 box + §3 item removed; this spec → design-locked
  after the grill.

## Out of scope (this increment)
- The **Library space** (in-app ingestion, chunk browser, chunk annotations) — sibling spec.
- Rename / delete / search / pin of conversations; multi-user; cross-device sync.
- Resuming an old chat as a *live* thread (Decision 4 keeps reopening read-only). Backfilling `session_id`
  onto pre-fix historical rows.

---

## Decision ledger (grilled 2026-07-13)

| Branch | Resolution | Deciding reason | Reopens if |
|---|---|---|---|
| A · persistence | Backend-backed via `AnswerRecord.session_id` | Data already written per-turn; durable + tied to provenance; one source of truth | backend also wins multi-device — no practical reversal |
| B · rehydration depth | Read-only transcript (Q + answer + degraded citation panel) | Reconstructable from one table; honest; avoids "edit history" | a real need to *resume* a past chat, or to show claims/reviewer on old turns → add the `AnswerReview`/`AnswerClaim` joins |
| C · title | Auto from first question, truncated; no rename | Zero extra write path | users ask to rename → add a `title` column/sidecar |
| D · ↻ New placement | Sidebar header | It's the "new chat" action | — |
| E · Library tab | Disabled tab reserved now | Build the shell once | the Library spec lands → enable it |
| H2 · live vs history | Composer stays on the live chat; past chats open read-only with "← Back to current" | Prevents losing an in-progress chat | — |
| H3 · failed turns | Not persisted → not in history | `record_answer` only reached on success | — |
| H4 · retention | Most-recent ~100, no delete/prune | Reads existing growth, doesn't create it | table grows large → a maintenance increment (parked → backlog) |

**Routing:** all resolutions folded into this spec (now design-locked). The A/B/H2 trade-offs are recorded
here — proportionate to the decision (reuse of an existing table + a read layer), so **no separate ADR**
unless a reversal is later proposed. H4 retention **parked → `docs/ui-checklist.md` §3 backlog**.

# Spec — Conversation resume (fresh-context) + durable export

**Status:** built 2026-07-14 (staged, not committed). Design agreed in-session with the user, then
built + verified $0. This doc is the design-lock written alongside the build (the increment was small
and the design was settled in conversation); revise here if the review changes it.

**Scope.** Extends `feature-conversation-history.md` (which reopened past chats **read-only** —
Decision 4, "reviewed, not resumed"). This increment lets the user **continue** a past chat, makes
**export** work on any conversation (past / resumed / live), and gives the transcript a clean template.

**Owner:** Claude Code (backend read/export layer + Svelte shell).

---

## The why

`feature-conversation-history.md` deliberately made a reopened chat read-only, because the durable
`AnswerRecord` store holds the *transcript* but not the in-memory conversational *context* the model
needs to follow up. The user wanted to keep chatting in an old thread — but explicitly **without**
carrying the old conversational memory (corpus-grounded answers are the product; chat memory is a
later exploration). That lands exactly on a design that's also the cheapest to build.

## Decisions (locked in-session 2026-07-14)

| # | Decision |
|---|----------|
| 1 | **Fresh-context resume.** "Continue this chat" on a viewed past conversation switches the live session to its `session_id`; the old turns render **read-only above the composer** for reference; new turns thread to the same id and persist. The backend in-memory session for that id starts **empty**, so new questions are **standalone corpus queries** — no replay of the old turns. (`chat_controller.py:778`'s `if history` rewrite naturally no-ops on empty history, so this needs no new backend.) |
| 2 | **Memory is deferred.** No replay of prior turns as LLM context. Limitation (accepted): the model won't resolve references to pre-resume turns ("expand on the second point") — ask self-contained questions. A future opt-in "recall this thread" can layer on. |
| 3 | **Export sources from the durable transcript by `session_id`** (`conversations.conversation_export_turns` over `AnswerRecord`), not the in-memory session. So a past / reopened / resumed chat exports identically to a live one (every answered turn already persists with its `session_id`). The dev bundle still prefers the richer in-memory turns (reviewer / figures) when this is a live session. |
| 4 | **`/api/export` accepts any `session_id`** (`get_or_create`, not `get`). An id with no persisted turns → `400` "nothing to export" (was `404` "unknown session"). |
| 5 | **Clean Markdown template.** `render_conversation_markdown` gains an optional `subtitle`; the transcript is titled "Provenote conversation" with an `Exported <UTC> · session <id>` line. |
| 6 | **No new locked settings, no paid calls, no schema change.** Pure read/export + frontend state. |

## Contracts

- **Backend** — `conversations.conversation_export_turns(session_id) -> list[ExportTurn]` (durable →
  export view models); `chat_controller.export_conversation` sources from it; `export.render_conversation_markdown(..., subtitle=)`;
  `apps/api/main.py` export endpoint uses `get_or_create`.
- **Frontend** (`App.svelte`) — `resumedHistory: ConversationDetail | null` state; `resumeConversation()`;
  read-only history rendered above the composer with a divider; `activeSource` resolves citations in
  the resumed history too; Export enabled while viewing/resumed; `doExport` targets `viewing ?? sessionId`.

## Definition of done

- [x] `conversation_export_turns` + tests (durable build, ordering, empty).
- [x] Export works on a past conversation (live: `200` + clean transcript; verified against the restarted API).
- [x] Unknown id → `400` (endpoint test updated).
- [x] Resume: history read-only above composer, new turn threads to the resumed `session_id` (verified
      live: sent id == resumed id), appends below the divider, empty-state suppressed, export enabled.
- [x] Gates: `pytest` 763 passed; `svelte-check` 0/0; ruff + mypy clean.
- [x] Verified **$0** — the chat send used a `window.fetch` SSE mock; `POST /api/chat` count held.

## Opens / follow-ups

- **Prompt-side citation fix** (stop haiku wrapping claim labels in brackets) — needs a paid turn to
  validate; left for user greenlight.
- Resume replaces the in-view live chat; a previous live chat's turns remain in history (persisted).
  If the previous chat was empty, nothing is lost. Fine for v1; note if a "keep both" flow is wanted.
- Conversation management (delete / pin / archive / projects) — the next cluster, on top of this.
- `ReadonlyTurn` shows no source **list** (only clickable citations → degraded panel); a resumed
  chat's history inherits that. The compact sources strip is live-turn only.

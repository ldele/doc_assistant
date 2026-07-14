# Spec — Conversation management (pin / archive / soft-delete)

**Status:** built 2026-07-14 (staged, not committed; additive migration applied to the live DB). Design
agreed in-session; this is the design-lock written alongside the build. The **first conversation-level
write path** — likely warrants an ADR when committed.

**Scope.** The single-conversation actions from the user's management wishlist: **pin**, **archive**,
**soft-delete**. Builds the durable `conversation_meta` sidecar the rest of the cluster
(**projects / groups** — *deferred*) will extend. Follows `feature-conversation-history.md` /
`feature-conversation-resume.md`.

**Owner:** Claude Code (schema + read/write layer + endpoint + Svelte sidebar).

---

## The why

Conversations are **derived** by grouping `AnswerRecord` rows — there is no conversation entity, so
there was nowhere to hang "pinned / archived / deleted / project". This adds the minimal durable home
for that per-conversation state.

## Decisions (locked in-session 2026-07-14)

| # | Decision |
|---|----------|
| 1 | **A relational sidecar table** `conversation_meta` (PK `session_id`; `pinned`, `archived`, `deleted_at`, `updated_at`), **not** a JSON blob / `app_settings` — the history spec's "no second source of truth" (Decision 1). Additive: `create_all` makes it (how `figures` landed). An **absent row = all-default** (a row is created only on first action). |
| 2 | **Soft-delete** (`deleted_at` non-null hides from the list, retains the `AnswerRecord` provenance). Reversible; fits the research-integrity ethos. A permanent purge is a separate, later action. |
| 3 | **Pinned sort first**, then newest-first within each group (stable sort). **Archived** conversations stay in the list data but the frontend hides them behind a "Show archived (N)" toggle (shown dimmed). Soft-deleted are excluded entirely (SQL subquery, before the ~100 cap). |
| 4 | **One PATCH endpoint** `PATCH /api/conversations/{sid}` `{pinned?, archived?, deleted?}` → `conversations.set_conversation_meta` (upsert; only fields sent change). Reads stay on `GET /api/conversations` (now carrying `pinned`/`archived`). |
| 5 | **Delete confirms** (`window.confirm`) — soft-delete is reversible but there's no restore UI yet, so guard the mis-click. Deleting the on-screen chat (viewed/resumed/live) falls back to a fresh chat. |
| 6 | **Projects/groups deferred** — a distinct, larger data model (a `project` field or `projects` table + grouping UI). The sidecar is the foundation it extends. |
| 7 | No locked-setting change; no paid calls; the only schema change is the additive table. |

## Contracts

- **Schema** — `ConversationMeta` (`db/models.py`); additive, `python -m doc_assistant.db.migrations`.
- **Backend** — `conversations.set_conversation_meta(...)`; `conversations.list_conversations` joins the
  sidecar (exclude deleted, attach flags, pinned-first); `ConversationSummary` gains `pinned`/`archived`.
- **API** — `PATCH /api/conversations/{session_id}` (`ConversationMetaUpdate`); `ConversationSummaryPayload`
  gains `pinned`/`archived`.
- **Frontend** — `api.updateConversationMeta`; `Sidebar.svelte` row actions + archived toggle + pin mark;
  `App.svelte` `pin/archive/deleteConversation` handlers; Icon `pin`/`archive`/`trash-2`.

## Definition of done

- [x] `conversation_meta` model + additive migration (applied to `data/library.db`, no data touched).
- [x] `set_conversation_meta` upsert + `list_conversations` (exclude deleted, pinned-first, flags) + tests.
- [x] `PATCH` endpoint + test; `GET` carries flags.
- [x] Sidebar per-row pin/archive/delete + pin mark + "Show archived (N)" toggle; delete confirm + fallback.
- [x] Gates: `pytest` 766; `svelte-check` 0/0; ruff + mypy clean.
- [x] Verified live **$0** (no LLM) on the real corpus; **every action restored** so history is unchanged.

## Opens / follow-ups

- **Projects (conversation groups)** — the next increment on this sidecar.
- Hover-only row actions — no touch affordance yet (mobile follow-up: `:focus-within` partially covers it).
- No **trash/restore** view for soft-deleted conversations (recoverable via the API today).
- Consider an **ADR**: first conversation-level write path + the schema addition.
- Rename (a `title_override`) is a natural sibling field on this table when wanted.

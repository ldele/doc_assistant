<!-- status: archived · updated: 2026-07-13 · class: disposable -->
<!-- BUILT + COMMITTED 2026-07-13 (`9ce5690` "Sprint 13: Chat History + Some AI polish"). Flipped
     active→archived when SPRINT-014 (Library browser, L1) became the active contract.
     Pre-existing [lifecycle] warn applies (tolerated across all SPRINT-*.md).

     Design lock: docs/specs/feature-conversation-history.md (grilled 2026-07-13). Feature: the
     left-sidebar app shell + conversation history. Backend-backed by the existing
     AnswerRecord.session_id (one write-fix + a read module + two GET endpoints); frontend adds the
     shell + sidebar + read-only rehydration. This is the shell the future Library space reuses. -->

# SPRINT-013 — conversation-history

- **base:** main
- **DoD:** A chat turn persists its `session_id` (was dropped). `GET /api/conversations` lists past
  chats (grouped, newest-first, NULL-session excluded, UTC-tagged times, title = first question);
  `GET /api/conversations/{sid}` rehydrates a read-only transcript (question + answer + degraded
  citation panel) or 404s. Desktop: a left sidebar lists history with a "current" marker + `↻ New
  chat`; selecting a past chat opens it read-only with a "← Back to current chat" bar and never
  destroys the live chat; the live chat stays fully interactive. `svelte-check` 0; full gate green
  (ruff / ruff format / mypy src / bandit / pytest); preview-harness-verified live (both themes,
  mobile off-canvas drawer, no horizontal overflow).

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
- docs/specs/feature-conversation-history.md
- src/doc_assistant/db/models.py
- src/doc_assistant/provenance.py
- src/doc_assistant/chat_controller.py
- src/doc_assistant/pipeline.py
- apps/api/main.py
- apps/api/models.py
- apps/api/sessions.py
- apps/desktop/src/App.svelte
- apps/desktop/src/lib/Turn.svelte
- apps/desktop/src/lib/SourcePanel.svelte
- apps/desktop/src/lib/types.ts
- apps/desktop/src/lib/api.ts

## affects
- src/doc_assistant/conversations.py
- src/doc_assistant/chat_controller.py
- apps/api/main.py
- apps/api/models.py
- apps/desktop/src/App.svelte
- apps/desktop/src/lib/Sidebar.svelte
- apps/desktop/src/lib/ReadonlyTurn.svelte
- apps/desktop/src/lib/types.ts
- apps/desktop/src/lib/api.ts
- tests/unit/test_conversations.py
- tests/integration/test_api_conversations.py

## contracts
- test: tests/unit/test_conversations.py
- test: tests/integration/test_api_conversations.py

## docs
- docs/DEVLOG.md
- docs/ROADMAP.md
- docs/ui-checklist.md

<!-- status: active · updated: 2026-07-13 · class: living -->

# UI checklist — doc_assistant (Phase 8, kept open)

Living status board for the desktop UI (`apps/desktop/`, Svelte 5 + Vite over the FastAPI/SSE
boundary). Phase 8 is an **open, iterative track** — this file is where UI features are checked off
as they ship and where new UI ideas are parked before they're specced.

Three parts:

1. **[Shipped](#1--shipped-committed)** — what's built and committed.
2. **[Open / verification debt](#2--open--verification-debt-keeps-phase-8-open)** — what keeps Phase 8 open.
3. **[Backlog](#3--backlog--candidate-ui-elements-the-iterative-pool)** — the pool to pull the next element from.
4. **[Per-feature review checklist](#4--per-feature-review-checklist-run-before-marking-any-ui-feature-done)** — the gate every new UI feature passes before it moves to *Shipped*.

Cross-refs: `docs/ROADMAP.md` (PR table, U-rows) · `docs/specs/feature-phase8-ui-upgrade.md` (design
lock + grill ledger) · `docs/specs/feature-rag-sandbox.md` · `docs/decisions/ADR-010-*` (sandbox
non-persistence) · `docs/decisions/ADR-011-*` (provider switch). When you check an item here, also
add the `docs/DEVLOG.md` entry and update the matching ROADMAP row — this file does not replace them.

---

## 1 · Shipped (committed)

Each box is checked only when the feature is **committed** (not merely staged) and passed the
[review checklist](#4--per-feature-review-checklist-run-before-marking-any-ui-feature-done).

### Chat / answer surface
- [x] **Streaming chat over POST-SSE** — token/step/result events parsed by hand (`api.ts::streamChat`). *(M2/M3 baseline)*
- [x] **U2 — right-aligned, width-capped user bubble; RAG answer stays full-width** — `Turn.svelte` CSS only (`align-self: flex-end`, `max-width: min(72%, 640px)`, tail-cue corner). Commit `7ee1b1e`. *Verified: preview-harness, both themes, mobile no-overflow.*
- [x] **U3 — citation side panel** — click inline `[n]` → slide-over with that source's chunk detail; source cards hidden inline by default. `SourcePanel.svelte` (new) + `Markdown.svelte` text-node linkifier + delegated click; `App.svelte` owns `activeCitation`. Commit `8ba1ffc`. *Verified: preview-harness, mocked SSE ($0).*
- [ ] **U4 — "↻ New" conversation-reset button** — clears turns + citation panel + composer, mints a fresh `sessionId` (no backend context leak into the next question); session RAG overrides intentionally kept. `App.svelte` only, reuses `.ghost` (no new CSS). *Built + verified 2026-07-13 (svelte-check 0/0; preview-harness light+dark, live-convo clear, disabled states) — **staged, commit pending**.* Subsumed into U5's sidebar (↻ New moved there).
- [ ] **U5 — app shell + conversation history** — left sidebar (Chat/Library switch, Library disabled) listing past chats backend-backed by `AnswerRecord.session_id`; reopen a chat read-only (degraded citation panel); `↻ New chat` in the sidebar; live chat preserved when viewing history (H2); mobile off-canvas drawer. Backend: `record_answer` `session_id` write-fix + `conversations.py` + `GET /api/conversations[/{sid}]`. *Built + verified 2026-07-13 (pytest 863; svelte-check 0/0; preview-harness live on the real corpus — 2 turns→history, reopen read-only, dark + mobile, no overflow) — **staged, commit pending** (flip to `[x]` + add sha on commit).* Spec: `docs/specs/feature-conversation-history.md`; contract: SPRINT-013.
- [x] **Earlier chat-UI refinement pass** — `ee8fe8d` (2026-07-09).
- [x] **Provenance card + low-confidence card** — effective-provider-aware token/`local` suffix. *(M1 baseline; made switch-truthful in `09afd0c`.)*

### Settings drawer
- [x] **Settings disclosure (read-only "Engine" section)** — discloses candidate pool, retrieval weights (now sourced from `config.BM25_WEIGHT`, no longer a hardcoded literal), parent-child, chunk sizes, each labelled with why it's locked / needs a re-ingest. Part of U1, commit `09afd0c`.
- [x] **U1 — RAG sandbox knobs** (ADR-010): `top_k` slider `[1, CANDIDATE_K]`, synthesis-mode segmented (AI/Human), multi-query toggle. Session-scoped, non-persistent, request-scoped through the whole turn — **no module-global mutation** (isolation-guard test). Provenance flags any effective value ≠ locked default. Commit `09afd0c`.
- [x] **U1 — manual System/Light/Dark theme** — `theme.ts` + re-keyed `app.css` (`data-theme` attr wins over the OS media query); `localStorage`, applied in `main.ts` before mount (no flash). Client-only, never a backend setting. Commit `09afd0c`.
- [x] **U1b — two niche sandbox knobs** (ADR-010 amendment): `epistemics_markers_enabled` toggle (contested/superseded chips), `reviewer_evidence_chars` number `[200, 6000]`. Same non-persistent, request-scoped mechanics. Commit `09afd0c`.
- [x] **U1c — live provider + model switch** (ADR-011 v1): `<select>` of configured providers (keyless provider rendered disabled with reason) + model input + Apply; takes effect next turn, no restart; reviewer follows an unpinned switch; persists like `source_dir`. Commit `09afd0c`.
- [ ] **U6 — A/B-compare sandbox (retrieval diff, v1)** — a per-turn "Compare" button runs the query under the locked defaults vs the session override and shows the two retrieved source sets side-by-side (per-source diff badges `both`/`only A`/`only B` + an honest note); **$0** (retrieval only, no LLM). Pure `compare.py` (`diff_sources`/`compare_note`) + `ChatController.compare_retrieval` (no LLM, no module-global mutation) + `POST /api/compare` + `CompareCard.svelte`. Full-answer 2× compare deferred (cost-gated). *Built + verified 2026-07-13 (pytest 875 +7; svelte-check 0/0; preview-harness live — `top_k=4` override → real depth diff + "only A" badges + depth note, defaults → no-op note, dark, no overflow, $0/offline) — **staged, commit pending**.* Spec: `docs/specs/feature-ab-compare-sandbox.md`; contract: SPRINT-015.
- [x] **"Point at a folder" source dir + first-run ingest banner** — `/api/settings` + `/api/ingest`. *(M4 data-home flow.)*
- [x] **Reset-to-locked-defaults** button (clears session overrides).

### Library space
- [ ] **L1 — read-only chunk browser** — the Library sidebar tab (enabled) lists ingested docs; opening one shows its chunks as parent blocks, each `<details>`-expandable to its `child_index`-ordered child chunks. Read-only, no model: `library.py` (`group_children` + `get_document_chunks` over the live Chroma handle) + `GET /api/library/documents[/{id}]` + `LibraryBrowser.svelte` + a `chat|library` shell mode switch. Markers + figure thumbnails deferred to **L1b** (chunk_epistemics=0 / figures=0 on this corpus). *Built + verified 2026-07-13 (pytest 868 +5; svelte-check 0/0; preview-harness live on the real corpus — 76 docs, parent/child render, 404, dark, no overflow, Chat↔Library preserves the live chat, $0/offline) — **staged, commit pending** (flip to `[x]` + add sha on commit).* Spec: `docs/specs/feature-library-browser.md`; contract: SPRINT-014.

---

## 2 · Open / verification debt (keeps Phase 8 open)

Built-but-not-fully-proven, or small follow-ups surfaced by the `09afd0c` review. **These are why
Phase 8 is not closed.**

- [ ] **Live-UI smoke test of the sandbox knobs on a real answer turn** — U1/U1b/U1c were preview-harness-verified at *construction only* (no answer turn, $0). Drive one real turn with a non-default `top_k` / synthesis-mode / markers toggle and confirm the `🧪 Session override` provenance note renders and the override actually changed retrieval. *(Cheap now the harness exists — see `.claude/launch.json`.)*
- [ ] **Live-UI smoke test of the provider switch end-to-end** — anthropic→ollama→anthropic was driven through the UI ($0), but confirm the *reviewer-follow* and the `🖥 Local model` usage line on a real flagged answer after a switch.
- [ ] **Live-UI smoke test of the marker chips** — KI-15 backend fix means chips now fire on ~3,334 real chunks, but nobody has confirmed the chip *renders* correctly in the live desktop UI since the fix (phase8 spec "Related backlog").
- [ ] **RG-012 Tier-2** — a cited turn on a clean/frozen box (pends a re-freeze bundling the data-home flow). *(Carried from M4.)*
- [ ] **Review finding — `reviewer_kind="llm_haiku"` is hardcoded** (`chat_controller.py:844`). When the reviewer follows a switch to Ollama, the persisted `AnswerReview` row records `reviewer_kind="llm_haiku"` while `model_name` is (correctly) the live model — a provenance-honesty blemish. Derive the kind from the resolved provider.
- [ ] **Review finding — server-side empty/whitespace model string is unvalidated** — `app_settings.set_llm_selection` validates provider + credential but accepts any `model`; a direct `llm_model=""` POST persists an empty model that `get_llm_selection()` then silently discards. Add `min_length=1` on the wire field or a non-empty `.strip()` check server-side.

---

## 3 · Backlog — candidate UI elements (the iterative pool)

Pull the next element from here. **Add your new UI ideas to this list.** Nothing here is specced
yet — moving one to *in-progress* means writing/locking its spec first (grill it), then building
against the [review checklist](#4--per-feature-review-checklist-run-before-marking-any-ui-feature-done).
Sourced from the phase8 spec's "Related backlog" table + `pr-m1`/`pr-m3` out-of-scope notes.

| # | Candidate UI element | Where tracked | Notes |
|---|---|---|---|
| [x] | **A/B compare sandbox** — run locked defaults vs. override side-by-side | ADR-010 option 4 · `docs/specs/feature-ab-compare-sandbox.md` · SPRINT-015 | **v1 built 2026-07-13 (U6, staged).** Retrieval diff only ($0, no LLM); moved to §1 above. **Still deferred:** the full-answer 2× compare (cost-gated, unverifiable without a model). |
| [ ] | **Rich marker UI** — hover a contested/superseded chip → the corroborating/contradicting docs, not just a bare chip | `pr-m1` out-of-scope (tagged PR-M3) | Currently a static chip (`SourceCard.svelte:17-21`). Backend markers already carry the data. |
| [ ] | **In-app PDF source viewer** — open the cited PDF at the page | `pr-m3` out-of-scope | Deferred at M3, never scheduled. |
| [ ] | **Styled table rendering** in the answer/source view | `pr-m3` out-of-scope | Marker tables render as text today. |
| [ ] | **S2 — selective-ingestion sources panel** — status chips, select-by-status/type, exclude toggle, ingest-selected | `feature-selective-ingestion.md`, ROADMAP S2 | Blocked on **S1** backend (draft, not locked). The other half-built Phase-8 UI item. |
| [ ] | **In-app API-key entry (OS keychain)** — U1c v2 | ADR-011 v2 north-star | Explicitly deferred; needs the keyring decision from ADR-011's open questions. |
| [ ] | **Precise parent-child re-projection for markers** | `pr-m1` ADR-1 option 2 | Backend attribution-quality work, not UI — but gates marker-chip trustworthiness. |
| [ ] | **Conversation rename / delete / search / pin** | `feature-conversation-history.md` out-of-scope | Deferred from U5 (v1 has none). Rename needs a `title` column/sidecar; delete needs a careful "edit history?" call. |
| [ ] | **Conversation retention / prune** (parked from the history grill, H4) | `feature-conversation-history.md` ledger | v1 lists most-recent ~100, no delete/prune. Revisit when `answer_records` grows large — a maintenance increment (CLI or a settings action). |
| [ ] | **Rich / resumable chat rehydration** — claims + reviewer on reopened turns (`AnswerReview`/`AnswerClaim` joins), or resume a past chat as a live thread | `feature-conversation-history.md` Fork B | U5 reopens read-only. The joins + a resumable-session path are the natural follow-up. |
| [~] | **Library space** — a "Library" view: browse ingested docs → their chunks (read chunk text), plus in-app document management + optional chunk annotation | new (2026-07-13, user request); grilled + carved into L1/L1b/L2/L3 | **Carved 2026-07-13.** **L1 (chunk browser) built** (`feature-library-browser.md`, staged, §1 above). Remaining: **L1b** markers + figure thumbnails (reopens when sidecars populate); **L2** in-app ingestion mgmt → adopt `feature-selective-ingestion.md` (S1/S2); **L3** chunk annotation → a new Enrichment-Layer sidecar, **needs its own ADR** (first write path). Pairs with the "In-app PDF source viewer" row. |

<!-- Add new UI ideas below this line as `| [ ] | <element> | <where> | <notes> |` -->

---

## 4 · Per-feature review checklist (run before marking any UI feature "done")

The gate for every new UI element. Derived from this repo's non-negotiables (`.claude/CONTEXT.md`
§Non-negotiable rules), ADR-010/011, and the DoD blocks in `docs/specs/feature-phase8-ui-upgrade.md`.
Copy this block per feature; check every applicable line before moving it to *Shipped*.

### Architecture & conventions
- [ ] **Thin shell** — no business logic in `apps/`; all logic in `src/doc_assistant/`. UI → library, never the reverse. (CONTEXT rule 3.)
- [ ] **Request-scoped, no module-global mutation** — any per-turn/session knob is threaded as an explicit parameter (a frozen dataclass / wire model), never assigned to a `config.*` or other module global. Concurrent turns on the shared `ChatController` singleton cannot leak into each other. (ADR-010; there's an isolation-guard test — extend it to any new field.)
- [ ] **Effective-value truthfulness** — anything provider/model-dependent reads the *effective* value (`self.rag.provider`/`.model`, `app_settings.effective_llm()`), never the import-time `LLM_PROVIDER`/`LLM_MODEL` constants (they go stale after a live switch). (ADR-011.)
- [ ] **No locked-setting change** without an eval-harness experiment + baseline (`--repeat`, beat control). A sandbox *override* is fine; changing a *default* is not a UI PR. (CONTEXT §Locked settings.)
- [ ] **Persistence boundary** — retrieval-quality-governed knobs are non-persistent (ADR-010). Only cosmetic prefs (theme) or per-install choices (source dir, provider selection) persist, and those go through `app_settings`/`localStorage`, never `.env`/`config.py`.
- [ ] **Exceptions chain** (`raise X from e`); user-facing messages translated at the UI/API boundary; a sidecar failure (markers, provenance) never breaks the turn.
- [ ] **Secrets** — no key in code or client; keys stay in `.env`. (CONTEXT rule 2.)

### Correctness & data integrity
- [ ] Default path (feature off / `overrides=None`) is **byte-identical** to before the feature.
- [ ] Bounds enforced at the validation boundary (pydantic wire model): out-of-range → 422, never a silent clamp. Note if a non-HTTP caller could bypass them.
- [ ] Any value persisted or recorded is **honest** — labels match the actual model/provider/kind that ran (see the `reviewer_kind` finding).
- [ ] Empty/whitespace/malformed input rejected server-side, not only guarded in the client.

### Frontend
- [ ] `svelte-check` — 0 errors.
- [ ] **Both themes** — light *and* dark styled; a new color reuses existing CSS vars (no new palette). Manual theme still wins over the OS media query.
- [ ] **Responsive** — no horizontal body overflow at mobile width; wide content scrolls in its own container.
- [ ] **Accessibility** — interactive controls have roles/labels (`role="radiogroup"`/`radio`, `aria-checked`, `aria-live` for async status); reduced-motion respected for transitions.
- [ ] TypeScript wire types mirror the backend model (`types.ts` ↔ `apps/api/models.py`).

### Tests & verification
- [ ] Unit + integration tests for the new behavior (incl. the no-global-mutation / isolation case where applicable); full gate green (`ruff` / `ruff format` / `mypy --strict src` / `bandit` 0 HIGH·MED / `pytest`).
- [ ] **Preview-harness verified** — at minimum construction-only ($0). For anything touching a real turn, a **live-UI smoke test** on a real answer (route through local Ollama to keep it $0 — force `--provider ollama`, then restore `.env`; watch the Anthropic credit-leak, KI/CONTEXT §Provider config).
- [ ] No live paid API calls in tests (cpc §13, gate-enforced).

### Docs
- [ ] One `docs/DEVLOG.md` entry (What / Why / Rejected / Opens).
- [ ] ROADMAP U-row updated (built → committed `<sha>`).
- [ ] This checklist updated: box checked in §1, item removed from §2/§3.
- [ ] A non-obvious design choice → an ADR in `docs/decisions/`; a stress-tested spec → `docs/specs/` (design-locked via a grill pass).

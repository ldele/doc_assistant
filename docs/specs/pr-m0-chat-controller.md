# Spec — PR-M0: extract `ChatController` + `TurnResult` (UI-agnostic turn core)

**Status:** 📋 PLANNED — designed by Cowork 2026-06-21 (Tauri migration, `docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md`). First PR of the Chainlit→Tauri migration (M0). **Tool-agnostic; no framework change.**
**Owner of execution:** Claude Code (code + tests).
**Pattern reference:** the existing thin-shell rule (`apps/` carry no logic; `.claude/CONTEXT.md` rule #3) and the already-stated "UI-agnostic" intent in `provenance.py` (module docstring) / `library.py` ("UI-framework-agnostic"). This PR *completes* that intent: it lifts the turn orchestration out of `apps/chainlit_app.py` into the library so any frontend renders the same value object.
**Migration context:** `docs/` → `tauri-migration-plan.md` (Cowork) Part 3, PR-M0. Unblocks PR-M1 (7d markers), PR-M2 (FastAPI), PR-M3 (Tauri frontend).

**Requirement (the why).** Every Chainlit dependency lives in one 608-line file (`apps/chainlit_app.py`), and all of its 51 `cl.*` calls are presentation. The turn orchestration inside `on_message` (dispatch → retrieve → stream → provenance → claims → export) is **business logic currently trapped in the UI layer**, which violates the thin-shell rule and forces every new frontend to re-derive it. This PR extracts that orchestration into a `ChatController` returning a `TurnResult` value object, so the migration becomes "write a renderer for `TurnResult`" instead of "rewrite 608 lines." It also plumbs the **stable chunk key** that PR-M1 (7d marker surfacing) needs — done **once, here**, so it is never retrofitted into Chainlit and then again into the controller (the coupling named in the migration plan).

**Cost & placement.** No new dependency, no LLM, no torch, no GPU. Pure refactor + one new dataclass module. Runs on either machine. **The app must behave identically after this PR** — same Chainlit output, same CLI output — proven by a parity guard test.

---

## ADR-1 — Turn orchestration as a library service returning a value object

**Context.** `on_message` (apps/chainlit_app.py:332–607) does: slash-command dispatch, pending-claim-edit handling, library-query routing, history-aware query rewrite, retrieval, figure-path lookup, source-element assembly, `SYNTHESIS_MODE=human` branch, answer streaming, provenance capture, confidence-signal gating + (flagged-only) reviewer call, claim segmentation + eager persistence, citation audit, usage accounting, and export-turn stashing. All of it is provider/UI-independent except the `cl.*` rendering calls interleaved through it.

**Decision.** Introduce `src/doc_assistant/chat_controller.py` with a `ChatController` that owns the orchestration and yields **`TurnEvent`s** (streamed tokens + step-status updates) terminating in a **`TurnResult`** value object carrying everything a frontend renders. The Chainlit and CLI apps become **thin renderers**: they iterate the events, stream tokens to their widget, then render the final `TurnResult`. No generation logic moves — `pipeline.stream_answer` etc. are called from the controller exactly as today.

**Options considered.**
1. *Controller yields a `TurnEvent` stream ending in `TurnResult` (chosen).* The event stream is the natural SSE model too (PR-M2 maps `TurnEvent` → SSE 1:1), so the streaming contract is designed once. Renderers stay dumb.
2. *Controller returns a finished `TurnResult` only (no streaming surface).* Rejected — loses token streaming, which is core UX; the frontend would block until the whole answer is built.
3. *Controller exposes callbacks (`on_token`, `on_step`) instead of a generator.* Rejected — callbacks invert control and are awkward across the future HTTP boundary; a generator/iterator is the same shape SSE and the CLI both want.

**Consequences.** `apps/chainlit_app.py` drops from 608 lines to a thin renderer (target < 200). `apps/cli.py` renders the same `TurnResult`. A third frontend (FastAPI, PR-M2) is a third renderer. The orchestration becomes unit-testable without Chainlit. **Risk:** behavior drift during the lift — mitigated by the parity guard test (the dispatch order and the exact provenance/claim/citation logic must be ported verbatim, not "improved"; improvements are separate PRs).

## ADR-2 — Stable chunk key on `RetrievedChunk` (carry the field; format conforms to epistemics)

**Context.** `provenance.RetrievedChunk` (provenance.py:42–54) has `filename, doc_id, page, section, reranker_score, chunk_excerpt, full_text` — **no stable chunk identity**. PR-M1 (7d markers) joins retrieved chunks against the `chunk_epistemics` sidecar. **That sidecar's key format is already shipped and fixed:** `epistemics.ChunkEpistemics.chunk_key` (epistemics.py:72–74) and `load_epistemics_index` (epistemics.py:263) both build `f"{document_id}:{chunk_index}"` — a **plain colon, no `p`/`c` prefix** — and `markers_for_chunk_keys` (epistemics.py:178–192) joins on exactly that string. M0 must produce a key in **that** format or the join silently returns nothing.

> **⚠ Coupling flagged (cross-module — `chat_controller` ↔ `provenance` ↔ `epistemics`):** there is a real, *unresolved* mismatch that M0 does **not** fix and must not paper over. Live retrieval runs in **parent-child mode by default** (`USE_PARENT_CHILD`); the parent `Document` returned by `retrieve_with_scores` (pipeline.py:185–189) carries `parent_index`, **not** `chunk_index`. But epistemics indexed only the **baseline (flat) collection** — `load_doc_chunks` (epistemics.py:211–217) explicitly states *"the parent-child store uses parent/child indices instead and is left for the live-surfacing follow-up."* So a PC-mode retrieved chunk has **no `chunk_index`** to build the epistemics key from. **Resolving that PC→baseline mapping is PR-M1's central decision, not M0's.** M0's job is only to *carry the field*; M1 decides how to populate it for PC chunks.

**Decision (M0 scope).** Add `chunk_key: str | None = None` to `RetrievedChunk`, derived in the controller's chunk-builder from the retrieved `Document.metadata`, using the **epistemics-compatible format**:
- if the metadata has `chunk_index` (flat / baseline chunks): `chunk_key = f"{document_id}:{chunk_index}"`. ✅ joins.
- if the metadata has only `parent_index` (PC chunks, the default): set `chunk_key = None` in M0 and leave a `# TODO(PR-M1): PC→baseline chunk-key mapping` marker. **Do not invent a `p{parent_index}` key** — it cannot join and would mask the gap.
- `document_id` absent (older rows) → `None` (no reliable key).

So in the **default PC configuration M0 ships, `chunk_key` is often `None`** — that is correct and expected. The field exists and is populated wherever a baseline `chunk_index` is available; M1 makes it populate for PC chunks by deciding the mapping.

**Options considered.**
1. *Carry the field, conform to the shipped key, defer the PC mapping to M1 (chosen).* Honest about the gap; no dead key format; M1 owns the one real decision.
2. *Invent `{document_id}:p{parent_index}` now (rejected).* Does not match `load_epistemics_index`'s key → joins to nothing → M1 "works" but shows zero markers. A silent-failure trap.
3. *Re-key epistemics to parent indices in M0 (rejected).* That is a substantive epistemics redesign (re-project against the PC store), well outside a UI-refactor PR; it belongs in M1's decision space with its own validation.

**Consequences.** PR-M1 inherits exactly one hard decision (PC→baseline key mapping) with the field already plumbed. `chunk_key` is **transient like `full_text`** — *not* added to the persisted provenance JSON in this PR; extend the existing exclusion (see the `provenance.py` contract). Guard: a test asserts `chunk_key` is the `{document_id}:{chunk_index}` format for a flat chunk, is `None` for a PC-only chunk in M0, and is **excluded from the persisted JSON**.

## ADR-3 — Session state object replaces `cl.user_session`

**Context.** The turn logic reads/writes `cl.user_session` keys: `history`, `counter` (TokenCounter), `export_turns`, `awaiting_edit`, `session_id` (apps/chainlit_app.py:64–67, 252, 288–291, 351–353, 368–369). That is per-conversation state the controller needs but must not own globally (multi-session later).

**Decision.** A plain `@dataclass Session` holds exactly those fields. The caller (renderer or, later, the FastAPI session store) owns the `Session` instance and passes it into every `ChatController` call. The controller never holds session state in module/instance globals — it is stateless across turns except for the injected `Session`.

**Consequences.** The CLI keeps one `Session`; Chainlit stores one in `cl.user_session` (a single key holding the dataclass, or reconstructs it — renderer's choice); FastAPI (PR-M2) keeps a `dict[session_id, Session]`. Multi-session is a non-breaking change (matches the provenance UUID rationale).

---

## Decisions

| # | Decision |
|---|---|
| 1 | **New module `src/doc_assistant/chat_controller.py`.** Owns turn orchestration; imports the same library functions `chainlit_app.py` does today (pipeline, provenance, reviewer, synthesis, query_router, export, figures, config). No `chainlit` import anywhere in `src/`. |
| 2 | **`handle_message(session, text) -> Iterator[TurnEvent]`** — ports the `on_message` dispatch order **verbatim**: (a) slash command → `TurnEvent.Result` wrapping the command output; (b) pending claim-edit (`session.awaiting_edit`) → adjudicate + result; (c) library query (`is_library_query`) → result; (d) RAG path. Export commands (`/export`, `/export-debug`) stay special-cased (they read live session transcript) — handled in the controller, not the stateless `execute_command`. |
| 3 | **`TurnEvent`** is a tagged union: `Token(text)`, `Step(name, status)`, `Result(TurnResult)`. The renderer streams `Token`s, shows `Step`s as status, renders `Result`. (PR-M2 maps these to SSE `event: token|step|result`.) |
| 4 | **`TurnResult`** carries the full render payload (fields below). The renderer does **no business logic** — only maps fields to widgets/buttons. The provenance card, claim review block, sources block, usage block, citation block remain **pre-rendered markdown strings** built by the existing pure formatters (those move to the controller module unchanged). |
| 5 | **Claim actions return view-models, not `cl.Action`.** `_build_claim_review` is split: the pure part returns `list[ClaimView]` (`{claim_id, n, text, badge}` for flagged claims) + the markdown block; the renderer builds its own buttons. `adjudicate(claim_id, decision, edited_text=None)` is a controller method (lifts `_resolve_claim` / `adjudicate_claim`). |
| 6 | **Chunk key field added (ADR-2)** to `RetrievedChunk`, set in the controller's `_build_retrieved_chunks` equivalent using the **epistemics format `{document_id}:{chunk_index}`** — populated for flat chunks, `None` for PC-only chunks (with a `TODO(PR-M1)` marker). Transient (not persisted). **The PC→baseline mapping is explicitly M1's decision, not M0's** — do not invent a PC key here. |
| 7 | **`Session` dataclass (ADR-3)** injected by the caller; controller stateless across turns otherwise. |
| 8 | **KI-1 opportunistic fix:** the orchestration's user-facing `print()` (e.g. pipeline.py:126 "Loading LLM...") is **not** in scope to chase globally, but any `print()` *moved into* the new controller module must be `structlog` from the start (don't carry a new violation into a fresh file). Pre-existing `print()` left in place elsewhere is out of scope (tracked by KI-1). |
| 9 | **Behavior frozen.** Same dispatch order, same provenance/reviewer/claim/citation/usage logic, same `SYNTHESIS_MODE=human` branch, same export stashing. This is a *move*, not a redesign. Any behavior change is a separate PR. |

**Edge cases (spec explicitly):**
- *Provenance capture fails* → today it never breaks the answer (try/except → a `_⚠ Provenance capture failed_` block). Preserve: `TurnResult.provenance_card_md` carries the failure string; the turn still returns.
- *Reviewer unavailable / errors* → flagged-only reviewer call already guarded; `ReviewResult(error=…)` path preserved into the card.
- *`SYNTHESIS_MODE=human`* → `TurnResult` has `mode="human"`, `answer` = the evidence-only markdown, no claims, no interpretation stream; the controller skips the generation call exactly as today.
- *Empty retrieval* (`retrieve_with_scores` → `[]`) → preserve current behavior (the answer still streams against an empty context; provenance records zero chunks). Do not add new "no results" handling — out of scope.
- *Slash command that is an export* → controller handles (`/export`, `/export-debug`); all other commands delegate to `execute_command` and wrap the string in `TurnEvent.Result` with `answer` = the command output, no sources/provenance.

**Build-time confirmations (verify against the code on the box):**
- The exact `cl.user_session` key set actually read/written (grep `user_session` in `chainlit_app.py`) — the spec lists `history, counter, export_turns, awaiting_edit, session_id`; confirm none missed.
- That the parent `Document` returned by `retrieve_with_scores` in PC mode carries `document_id` in `.metadata` (pipeline.py:185–189 strips only `parent_text`) — confirms `chunk_key` *could* be built once M1 decides the PC→baseline mapping. Confirm it does **not** carry `chunk_index` (only `parent_index`) — which is exactly why M0 leaves `chunk_key=None` for PC chunks and M1 owns the mapping.
- `TokenCounter` construction + the `input_tokens/output_tokens/total/cost_usd` surface used by the usage block (tracking.py) — moved as-is.

---

## Contract — `src/doc_assistant/chat_controller.py` (new)

Value objects (pure data) + the orchestrating `ChatController`. No `chainlit` import.

```python
@dataclass
class Session:
    """Per-conversation state. Caller-owned; injected into every call."""
    history: list[dict[str, str]] = field(default_factory=list)
    counter: TokenCounter = field(default_factory=TokenCounter)
    export_turns: list[export.ExportTurn] = field(default_factory=list)
    awaiting_edit: dict[str, Any] | None = None
    session_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S"))

@dataclass
class SourceView:
    """One retrieved source, render-ready."""
    n: int
    citation: str            # format_citation(doc, n)
    excerpt: str             # first ~800 chars (the Chainlit side-panel length)
    figure_path: str | None  # resolved PNG path if this chunk is a figure
    chunk_key: str | None    # ADR-2; consumed by PR-M1 markers
    # PR-M1 will add: markers: list[str]  (contested / superseded_trend)

@dataclass
class ClaimView:
    """A flagged claim needing adjudication (clean claims are not surfaced)."""
    claim_id: str
    n: int
    text: str
    badge: str               # "unsupported" | "weakly grounded"

@dataclass
class UsageView:
    turn_input: int
    turn_output: int
    session_total: int
    cost_usd: float | None   # None under local provider (no metered cost)
    is_local: bool

@dataclass
class TurnResult:
    answer: str                       # the raw answer text (no appended blocks)
    mode: Literal["ai", "human"]
    sources: list[SourceView]
    flagged_claims: list[ClaimView]
    usage: UsageView
    standalone_query: str             # post-rewrite query actually searched
    record_id: str | None             # provenance id (for /review, /export-record)
    # Pre-rendered markdown blocks (built by the existing pure formatters):
    provenance_card_md: str
    claim_review_md: str
    sources_md: str
    usage_md: str
    citation_note_md: str             # "" when citations are clean

# TurnEvent: a tagged union (use a small dataclass hierarchy or a typed dict).
#   Token(text: str) | Step(name: str, status: str) | Result(result: TurnResult)

class ChatController:
    def __init__(self, rag: RAGPipeline | None = None) -> None: ...
    def handle_message(self, session: Session, text: str) -> Iterator[TurnEvent]: ...
    def adjudicate(self, claim_id: str, decision: str, edited_text: str | None = None) -> None: ...
    def export_conversation(self, session: Session, *, dev: bool) -> Path: ...
    def chunk_count(self) -> int: ...   # delegates to rag.chunk_count()
```

**Moved verbatim into this module (were module-level fns in `chainlit_app.py`, all pure or near-pure):** `_format_review_block`, `_token_suffix`, `_format_provenance_card`, `_build_retrieved_chunks` (+ set `chunk_key`), `_build_claim_review` (split per Decision 5), `_export_sources`, `_append_export_turn`. These keep their logic; only `cl.Action` construction is removed from `_build_claim_review`.

**NOT responsible for:** any `cl.*` call, HTTP, SSE framing (that's PR-M2), the 7d marker join (PR-M1 adds it to `handle_message` + `SourceView`), persisting `chunk_key`, framework choice.

## Contract — `src/doc_assistant/provenance.py` (edit)

- Add `chunk_key: str | None = None` to `RetrievedChunk` (ADR-2). Place after `full_text`.
- **Do not** add it to the persisted JSON. `record_answer` (provenance.py:147–150) already excludes the transient field with `{k: v for k, v in asdict(c).items() if k != "full_text"}`. Extend that exclusion to the new field: `if k not in ("full_text", "chunk_key")`. Guard-tested (mirror the existing `full_text`-not-persisted assertion in `test_provenance.py`).

## Contract — `apps/chainlit_app.py` (rewrite to thin renderer)

- Construct one `ChatController` at module load; per chat, hold a `Session` in `cl.user_session` (single key, or reconstruct from existing keys).
- `@cl.on_message`: call `controller.handle_message(session, message.content)`; for each `TurnEvent`: `Token` → `msg.stream_token`; `Step` → `cl.Step`; `Result` → assemble the final `cl.Message` (answer + the pre-rendered blocks) + build `cl.Action`s from `flagged_claims` + the export action + figure `cl.Image`s from `sources`.
- `@cl.action_callback`s: map to `controller.adjudicate(...)` / `controller.export_conversation(...)`. The claim-edit follow-up sets `session.awaiting_edit` (then a normal message routes through the controller's pending-edit branch).
- Target: **< 200 lines**, zero business logic (no provenance/claim/citation computation here — only widget mapping).

## Contract — `apps/cli.py` (rewrite to render `TurnResult`)

- Build one `ChatController` + one `Session`. Loop: read input → `handle_message` → print streamed tokens → on `Result`, print `answer` + the markdown blocks (CLI already prints command/library output; now it renders the full `TurnResult`). Dispatch order is now the controller's, so the CLI's hand-rolled dispatch (cli.py:26–40) is deleted in favor of the controller's.

---

## Build node

**Depends on:** nothing new (all imported library modules shipped). Independent of torch/Marker/GPU.
**Files owned:**
- `src/doc_assistant/chat_controller.py` (new — value objects + `ChatController`)
- `src/doc_assistant/provenance.py` (`RetrievedChunk.chunk_key`)
- `apps/chainlit_app.py` (rewrite → thin renderer)
- `apps/cli.py` (rewrite → `TurnResult` renderer)
- `tests/unit/test_chat_controller.py` (new)
- `tests/integration/test_turn_parity.py` (new — the parity gate)
- `tests/unit/test_provenance.py` (extend — `chunk_key` populated + not persisted)
- `docs/decisions.md` (record ADR-1/2/3 — or a new `docs/decisions/ADR-NNN`), `.claude/CONTEXT.md` (note the controller seam under architecture), `docs/architecture.md` (module table: add `chat_controller`; mark `apps/*` as renderers), one `docs/DEVLOG.md` entry per logical change.

### Unit test — `tests/unit/test_chat_controller.py`
No Chainlit, no live LLM (fake `RAGPipeline` / stub `stream_answer`, fake retrieval docs with known metadata):
- **Dispatch order:** slash command short-circuits; pending `awaiting_edit` routes to adjudication; library query routes to `answer_library_query`; otherwise RAG. (Assert the branch taken for each input class.)
- **`chunk_key` derivation (ADR-2 format):** flat docs (`document_id`+`chunk_index`) → `"{document_id}:{chunk_index}"` (matches `epistemics.load_epistemics_index`); PC-only docs (`parent_index`, no `chunk_index`) → `None` (the deferred PC→baseline mapping is M1); missing `document_id` → `None`.
- **`TurnResult` shape:** AI mode returns answer + sources + (flagged) claims + usage + pre-rendered blocks; `human` mode returns evidence-only answer, no claims, `mode="human"`.
- **`adjudicate`** calls `adjudicate_claim` with the right decision; edit carries `edited_text`.
- **Provenance failure** is caught → `provenance_card_md` holds the failure string, turn still completes.

### Integration test (CI gate) — `tests/integration/test_turn_parity.py`
The ADR's parity gate. With a fake/stub pipeline producing a fixed answer + fixed retrieval:
- Drive the **same `Session` + same input** through (a) the `ChatController` directly and (b) a minimal harness mimicking each renderer's consumption of the event stream.
- Assert the **`TurnResult` is identical** regardless of renderer, and that the CLI render and a captured Chainlit-style render contain the same answer text, the same source citations, the same provenance id, and the same flagged-claim set. (No `cl.*` runtime needed — assert against the `TurnResult` and the renderers' string output.)
- Deterministic; no network, no corpus, no paid call (cpc §13).

## Definition of done
- `chat_controller.py` owns turn orchestration; **no `chainlit` import in `src/`**; `apps/chainlit_app.py` < 200 lines and contains no business logic; `apps/cli.py` renders `TurnResult`.
- App behavior **unchanged** (parity test green): same dispatch order, provenance, reviewer gating, claim segmentation/adjudication, citation audit, usage, `human`-mode branch, export.
- `RetrievedChunk.chunk_key` added in the **epistemics format `{document_id}:{chunk_index}`** (populated for flat chunks; `None` for PC-only chunks — the PC→baseline mapping is M1), **excluded from persisted JSON** (guard-tested).
- Unit + integration tests green; `ruff` / `mypy --strict src` / `bandit` clean; coverage floor (40%) held or raised; no new `print()` in the new module (structlog if needed).
- `decisions.md` (or ADR-NNN) records ADR-1/2/3; `architecture.md` module table updated (`chat_controller` added, `apps/*` = renderers); one `DEVLOG.md` entry per logical change. **Stage + summarize the diff; do not commit/push without review** (cpc §13).

## Out of scope (later PRs)
- **7d marker surfacing** (join `chunk_key` → `chunk_epistemics`, add `SourceView.markers`) — **PR-M1**.
- **FastAPI / SSE** mapping of `TurnEvent` — **PR-M2** (`docs/specs/pr-m2-fastapi-boundary.md`).
- **Persisting `chunk_key`** in the provenance record — a 7d decision, not this PR.
- **Any UX change** (rich per-claim inline edit, styled tables) — built natively in the Tauri frontend, **PR-M3**.
- **Global KI-1 `print()`→structlog sweep** — tracked separately; only the new module must be clean here.

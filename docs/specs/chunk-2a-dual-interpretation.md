# Spec — Integrity Chunk 2a: dual interpretation layer

**Status:** Designed 2026-06-04 (grilled with user, verified against code). Ready for Claude Code execution.
**Owner of execution:** Claude Code (code + tests).
**Pattern reference:** Research Integrity Layer (`docs/decisions.md` → Chunk 2a); reuses `provenance.compute_confidence_signals` (PR 5.1) and the existing `pipeline.stream_answer`.

**Goal (the why).** When an AI answers from your documents it silently blends *what the sources say* (evidence) with *its own synthesis* (interpretation). For research use that is the dangerous failure mode — an AI inference carried forward as if it were a citable fact. Chunk 2a makes the **evidence ↔ interpretation boundary explicit** and keeps the human in control of the synthesis step, with a **logged, per-claim adjudication** that feeds the Phase 9 PRISMA-trAIce disclosure. It is the conceptual core of the integrity pillar (Chunk 1 = provenance; Chunk 2b = reviewer; **2a = separate + adjudicate**; Chunk 3 = export).

---

## ADR — dual layer as a presentation split over one pipeline

**Context.** Generation today is one streaming call (`ANSWER_PROMPT | self.llm`, citations inline as `[1]`,`[2]`… mapping to numbered `[Source N: filename, page P]`). The integrity signal stack already exists: `provenance.compute_confidence_signals` gives retrieval-derived flags (weak reranker, single-source, cluster spread).

**Decision.** Add `SYNTHESIS_MODE = human | ai` (default `ai`) selecting an **output template over the same retrieve→rerank→generate pipeline** — no new generation logic. The **evidence layer is deterministic** (the reranked passages + provenance, no second LLM call). The **interpretation layer is the existing LLM answer**, relabeled, segmented into **citation-anchored claims**, each carrying a **deterministic retrieval-derived uncertainty marker**, each adjudicable (accept/reject/edit) and **logged** to a new `answer_claims` table. A **warn-only** pre-interpretation checkpoint reuses the confidence signals.

**Options considered.**
1. *Evidence layer as an LLM-distilled summary* — rejected: re-introduces LLM-trust into the integrity layer (the project deliberately uses observable, retrieval-derived markers). Evidence = retrieved passages, "no synthesis," taken literally.
2. *LLM claim extraction (per-claim decomposition call)* — rejected for v1: same LLM-trust/cost objection. Use **citation-anchored deterministic segmentation**; `edit` is the escape hatch for imperfect boundaries.
3. *Full per-claim interactive GUI in Chainlit now* — rejected: per-claim accept/reject/**edit** exceeds Chainlit's clean limits (Open Questions). Ship logic + a **minimal in-context Action-button** interaction; defer the rich GUI to **Phase 8** (where the framework decision is made).
4. *Blocking pre-interpretation gate* — rejected: the app must inform, never block (user principle). Warn-only.

**Consequences.** 2a adds no new generation logic — it is presentation + segmentation + persistence + minimal interaction. `human` mode is `ai` minus the interpretation call (cheaper, integrity-pure). The rich editorial GUI lands with the Phase 8 framework decision instead of being thrown away.

---

## Decisions (from the 2026-06-04 grilling)

| # | Decision |
|---|---|
| 1 | **Deterministic evidence layer**; dual-layer = **template split** over one pipeline (`SYNTHESIS_MODE` selects the template, not a different pipeline). |
| 2 | **Citation-anchored deterministic claim segmentation** — split the interpretation on its inline `[N]` markers; each claim links to source N (filename, page, reranker score). Uncited spans = their own "unsupported" claim units. `edit` is the escape hatch for coarse boundaries. |
| 3 | **Phase 6:** logic + dual-layer render + persistence + **minimal in-context Chainlit Action buttons** (accept/reject), **edit via a text follow-up**. **Phase 8:** rich per-claim inline-edit GUI (deferred to the framework decision). |
| 4 | **New `answer_claims` sidecar table**, **eager** persistence (insert all claims `pending` at generation; update on adjudication). Mirrors the `answer_reviews` sidecar. Chunk 3 reads it as the adjudication log. |
| 5 | **Pre-interpretation checkpoint = warn-only**, reuse `compute_confidence_signals` on the reranked set before the interpretation call. **Never blocks** (no blocking knob). |
| 6 | **Per-claim deterministic uncertainty markers:** uncited claim → "unsupported"; cited claim → inherit its evidence's reranker strength (weak if below the PR 5.1 threshold). Whole-answer = the confidence banner. **Faithfulness stays the flagged-only Chunk 2b reviewer's job** — no per-claim runtime LLM call. |
| 7 | **Both modes in v1, default `ai`.** `human` = evidence layer only (interpretation call **skipped**; no claims, no adjudication). |

---

## Contract — `src/doc_assistant/synthesis.py` (new)

Pure functions over already-retrieved data + the generated answer.

- `segment_claims(answer: str, sources: list[RetrievedChunk]) -> list[Claim]` — split on inline `[N]` markers (confirm the exact regex against `prompts.ANSWER_PROMPT`/`format_docs_for_prompt`: `[Source N: …]` ↔ `[N]`). Each `Claim` = `text`, `claim_index`, `source_numbers: list[int]`, `citations` (resolved filename+page), `marker: "ok"|"weak"|"unsupported"`.
- `claim_marker(claim, sources) -> str` — uncited → `unsupported`; cited → `weak` if the cited chunk's reranker score < the PR 5.1 weak-score threshold, else `ok`. Deterministic.
- `render_dual_layer(evidence, interpretation_claims, signals, mode) -> ...` — assemble the evidence block (passages + provenance + `ConfidenceSignals` banner) and, in `ai` mode, the interpretation block (claims with per-claim markers). `human` mode returns evidence only.

## Contract — `db/models.py` + `db/migrations.py`

New `AnswerClaim` (`answer_claims`): `id · answer_record_id (FK answer_records, indexed) · claim_index · claim_text · citations (JSON) · decision (str: pending|accepted|rejected|edited, default "pending") · edited_text (nullable) · decided_at (nullable)`. Migration registered in `migrations.py` (idempotent versioning).

## config.py additions
- `SYNTHESIS_MODE = os.getenv("SYNTHESIS_MODE", "ai")` — `human` | `ai`; `ValueError` otherwise (mirror `embeddings.get_model_config`).

## UI — `apps/chainlit_app.py`
- Render evidence layer always; interpretation layer + per-claim markers only in `ai` mode.
- Per-claim **`cl.Action`** accept/reject buttons (in-context). **Edit** = a text follow-up ("send the corrected claim N") that writes `edited_text` + `decision="edited"`.
- `human` mode: skip the interpretation stream entirely; show the evidence layer + banner.
- Eager-insert `answer_claims` rows when the `ai` answer is produced.

---

## Build node

**Depends on:** none structurally (reuses shipped provenance + pipeline). Independent of Feature 4a.
**Files owned:** `src/doc_assistant/synthesis.py` (new), `src/doc_assistant/db/models.py`, `src/doc_assistant/db/migrations.py`, `src/doc_assistant/config.py`, `src/doc_assistant/pipeline.py` (mode branch / claim hand-off), `apps/chainlit_app.py`, `src/doc_assistant/commands.py` (adjudication handlers + a `/synthesis` mode display), `tests/unit/test_synthesis.py` (new), `tests/integration/test_adjudication_persistence.py` (new), `.env.example`.
**Status:** pending.

### Unit test — `tests/unit/test_synthesis.py`
`segment_claims` on a fixed answer with `[1][2]` + an uncited sentence → correct claim count, source mapping, and markers (`unsupported` for the uncited span; `weak` when the cited chunk's reranker score is below threshold). `human`-vs-`ai` render branch. Pure, no DB/LLM.

### Integration test — `tests/integration/test_adjudication_persistence.py`
Produce an `ai` answer (mocked LLM) → assert `answer_claims` rows inserted `pending` (eager) → apply accept/reject/edit → assert the row's `decision`/`edited_text`/`decided_at` update. Temp SQLite.

## Definition of done
- `SYNTHESIS_MODE=ai` renders evidence + interpretation with per-claim markers and in-context accept/reject (+ edit-via-follow-up); decisions persist to `answer_claims`.
- `SYNTHESIS_MODE=human` returns evidence only, no interpretation call, no claims.
- Pre-interpretation checkpoint warns (never blocks) on weak retrieval.
- Unit + integration tests green; ruff / mypy --strict / bandit clean; migration applies idempotently.

## Out of scope (Phase 8 / later)
Rich per-claim inline-edit adjudication GUI (→ Phase 8, with the UI-framework decision). Structured-generation claim emission (v2 if citation/sentence granularity proves too coarse). Per-claim runtime faithfulness scoring (stays the flagged-only reviewer's job).

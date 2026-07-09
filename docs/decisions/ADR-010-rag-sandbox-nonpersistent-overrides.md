<!-- status: active · updated: 2026-07-09 · class: append-only -->

# ADR-010 — RAG sandbox: session-scoped, non-persistent query-time knob overrides in the desktop

- **Status:** accepted — v1 scope = option 3 (basic session-scoped override surface); the
  option-4 A/B-compare is the recorded north-star, phased after v1. Build spec:
  `docs/specs/feature-rag-sandbox.md` (2026-07-09).        (proposed | accepted | superseded by ADR-NNN)
- **Date:** 2026-07-09
- **Deciders:** Lucas (with Claude Code)

> This ADR settles **how Phase 8's "settings page exposing the RAG sandbox knobs" (ROADMAP,
> `docs/ROADMAP.md:41`) can exist without contradicting the locked-settings non-negotiable**
> (`.claude/CONTEXT.md` → *Locked settings*; root `CLAUDE.md` → *Non-negotiables*). It records the
> product/governance decision; the code-level contract (files, request/response shapes, guard tests,
> DoD) follows in a build spec under `docs/specs/`.

## Context

Phase 8's stated capability is a **sandbox**: let the user see how the retrieval/synthesis knobs
change an answer. Today the desktop Settings shows those knobs **read-only** — `_settings_view()`
(`apps/api/main.py:113-137`) emits `top_k`, `candidate_k`, `use_parent_child`, `synthesis_mode`,
`parent_chunk`, `child_chunk`, `retrieval_weights`, mirrored in
`apps/desktop/src/lib/Settings.svelte` under "Engine (read-only) … locked defaults." So the current
state is exposition, not a sandbox.

The constraint that makes this a real decision — not just "add form inputs" — is the project
non-negotiable: **locked settings change only via an eval-harness experiment** (`--repeat`, beat the
control beyond its variance, record a baseline; `.claude/CONTEXT.md`). That governance *is* the
product's measurable-quality premise. A user who silently edits `TOP_K` to a worse value degrades
retrieval quality with no measurement — the exact failure the locked-settings rule exists to prevent.

Two facts about the code frame the resolution (mapped 2026-07-09, cited below):

1. **The knobs bind at three different times, and only one is live-changeable cheaply.**
   - *Query-time* — read inside a per-request call, so they can vary per answer with **no rebuild**:
     `TOP_K` (a real parameter of `pipeline.retrieve_with_scores`, `pipeline.py:170`, hardcoded to
     the constant only at `chat_controller.py:590`); `SYNTHESIS_MODE` (the `ai`/`human` branch at
     `chat_controller.py:611`); `USE_MULTI_QUERY` (the expansion branch at `pipeline.py:178`).
   - *Construction-time* — baked into the pipeline/retrievers at build, so changing them needs the
     pipeline **rebuilt** (loads embedder, Chroma, BM25, cross-encoder, LLM): `CANDIDATE_K`
     (`pipeline.py:134,145`), retrieval weights (`EnsembleRetriever(..., weights=…)`,
     `pipeline.py:147`), reranker (hardcoded `CrossEncoder("BAAI/bge-reranker-base")`,
     `pipeline.py:157`), LLM provider/model (`pipeline.py:160-164`).
   - *Ingest-time* — shape the stored chunks/embeddings, so changing them needs a **re-ingest**:
     parent/child chunk sizes (`ingest/chunking.py:35-59`), and `USE_PARENT_CHILD`, which spans
     store selection (`pipeline.py:115`) *and* chunk shape — its query-time dedup branch is
     meaningless without a matching store, so it is not independently live-flippable.

2. **Some knobs would mislead if given a slider.** The BM25/vector weight is **structurally inert on
   the shipped top-K**: `EnsembleRetriever` hands the cross-encoder the full candidate union, so the
   weight only reorders the pre-rerank list — measured **flat across `[0,1]`** post-rerank
   (`tests/eval/baselines/bm25_weight_sweep_2026-07-03.md`). A slider on it shows zero change and
   reads as "the app is broken."

One more architectural fact sets the implementation frame: the FastAPI `ChatController` is a
**process singleton** built once in the lifespan (`apps/api/main.py:204-208`), and there is **no
per-request knob path today** — `ChatRequest` is `{text, session_id}` (`apps/api/models.py:27-29`);
`POST /api/settings` writes only `source_dir` (`main.py:294-303`).

## Options

1. **Read-only exposition (status quo, richer copy).** Keep every RAG knob read-only; add "why this
   is locked / how to change it (the eval harness)." *Trade-off:* zero governance risk and fully
   honest, but it is not a sandbox — no experimentation — so it does not deliver Phase 8's stated
   capability. A fine floor; incomplete as the whole feature.

2. **Persistent editable settings.** Let the user edit `TOP_K`, `CANDIDATE_K`, etc. and persist them
   as the app's defaults (extend `POST /api/settings` to write them). *Trade-off:* literally "expose
   the knobs," but it **directly violates the locked-settings non-negotiable** — the eval harness
   stops being the source of truth, and measurable quality (the product's whole premise) can silently
   regress with no baseline. Rejected on the governance ground.

3. **Session-scoped, non-persistent query-time overrides (chosen shipping form).** The user overrides
   only the query-time knobs (`TOP_K`, synthesis mode, multi-query) for the current session; the
   overrides ride the request, never persist, and reset on restart; the locked defaults are untouched
   and shown read-only-with-rationale; the eval harness is named as the only path to a new default.
   *Trade-off:* honest and safe, but limited to the query-parameterizable set — `CANDIDATE_K` /
   weights / chunking stay read-only. That limit is also the honest outcome (fact 1).

4. **A/B compare sandbox (chosen north-star, phased after option 3).** Run the same query under the
   locked defaults *and* under the override, and show both answers/sources side by side. *Trade-off:*
   turns the sandbox into a measurement/teaching tool — the measurable-quality ethos applied to the
   UI — but costs ≈2× per compared turn and more UI surface, so it is sequenced after the basic
   override surface, not built first.

## Decision

**Adopt option 3 as the shipping form, with option 4 as the north-star**, resolved by three
properties:

- **Non-persistence is the governance wall.** Overrides are request/session-scoped, never written to
  `config` / `.env` / `app_settings`; a restart returns to the eval-gated baseline. The locked
  defaults — and the eval harness as the only way to change them — are untouched. This is the single
  property that lets the feature exist without contradicting the non-negotiable: the sandbox changes
  *this answer*, never *the default*.

- **Expose only the honest set.** The sandbox surfaces exactly the query-time knobs that move a
  single answer without a rebuild: `TOP_K`, `SYNTHESIS_MODE` (`ai`/`human`), `USE_MULTI_QUERY`.
  Construction-time knobs (`CANDIDATE_K`, retrieval weights, reranker, provider/model) and
  ingest-time knobs (chunk sizes, `USE_PARENT_CHILD`) are shown **read-only with the specific reason**
  they cannot be a live knob — the BM25/vector weight labeled *"inert on the shipped top-K by
  construction (measured)"* rather than given a slider that misinforms.

- **The eval harness stays the source of truth; a sandbox result is indicative, never a verdict.**
  The surface tells the user how to promote an override to a default (run the experiment) and frames
  a sandbox outcome as a demonstration on one query — reproducible/indicative, not a measured win.

**Deciding reason:** non-persistence is what makes exposure *safe*. The sandbox is allowed to be as
exploratory as we like precisely because it can only ever change the current answer — it cannot
silently degrade the governed defaults the product's measurable-quality claim rests on. That one
property buys Phase 8's experimentation capability without surrendering the eval-gated governance.

**What would reverse it:** if the per-request override plumbing cannot be proven isolated across
concurrent sessions on the shared singleton (a correctness risk — overrides leaking between users'
turns), fall back to option 1 (read-only exposition) until the override path is proven isolated. If
real use shows users mostly want to change a default *permanently*, that is a signal to make the eval
harness one-click from the app — not to persist raw edits (option 2 stays rejected).

## Consequences

**Easier:** implementable as a thin per-request override object threaded through
`POST /api/chat` → `ChatController._handle_rag` → `pipeline.retrieve_with_scores`. `TOP_K` is already
a parameter; synthesis mode and multi-query are call-time branches — no pipeline rebuild, no
re-ingest, no new persistence. The read-only rows already exist in `_settings_view()`; they gain
rationale copy. The exposed set equals the cheap set equals the honest set — the three constraints
(governance, cost, honesty) land on the same scope line, which is why the feature is small.

**Harder / cost:** the sandbox cannot touch `CANDIDATE_K` / weights / chunking, so a user wanting
those is routed to the eval harness — the feature is **honest-but-partial by construction**, which
must be stated in the UI, not hidden. The A/B-compare north-star (option 4) doubles per-turn cost and
needs cost gating. A real correctness obligation: overrides must be **request/session-scoped**, never
implemented by monkeypatching module globals (`pipeline.USE_MULTI_QUERY`, `chat_controller.SYNTHESIS_MODE`),
which would leak across concurrent turns under the shared async singleton.

**Must revisit:** whether `EPISTEMICS_MARKERS_ENABLED` and `REVIEWER_EVIDENCE_CHARS` (also query-time,
but cosmetic / niche) belong in the surface; the A/B-compare form's exact UX and cost gating; and
whether to ever make `CANDIDATE_K` / weights live via a *transient* pipeline build (a separate,
heavier decision, not in scope here).

**Fix in passing (spec-level):** `_settings_view()` hardcodes `retrieval_weights` as literals
(`apps/api/main.py:136`) instead of reading `config.BM25_WEIGHT` — the read-only display would
silently drift from the real value if the env var changed. The build spec sources it from config so
the locked-value display is truthful.

## Confidence

- ✓ The query-time / construction-time / ingest-time split is grounded in the code (citations in
  Context, fact 1), mapped 2026-07-09.
- ✓ The BM25/vector weight is structurally inert on the shipped top-K — measured flat across `[0,1]`
  (`tests/eval/baselines/bm25_weight_sweep_2026-07-03.md`; `.claude/CONTEXT.md` → Open questions).
- ⚠ **Per-request override isolation under the shared singleton is design intent, not yet proven.**
  The spec must specify request-scoped threading (no global patch) and a concurrency guard test; until
  that test exists this is the feature's one correctness risk (the "what would reverse it" trigger).
- ⚠ **That `TOP_K` + synthesis mode + multi-query is the *useful* set is an untested product
  hypothesis.** Validate with real use before widening the surface; do not pre-build option 4 on the
  assumption that users want more knobs.

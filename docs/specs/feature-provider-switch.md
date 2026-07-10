# Spec — U1c v1: desktop provider + model switch (ADR-011)

**Status:** 📋 **design-locked** (ADR-011 accepted + grilled 2026-07-10, 8 forks); **NOT built.** v1 scope
= option 4 (switch provider **+ model** among already-configured providers; the API key stays in `.env`;
live swap between turns). Keyring / in-app key entry (ADR-011 option 2) is the recorded **v2** north-star —
out of scope here. Roadmap PR **U1c**, 5th in the Phase-8 UI build order (after U2/U3/U1/U1b).
**Owner of execution:** Claude Code, when U1c is active. Create a cpc SPRINT contract from this spec at
build time (`docs/sprints/SPRINT-000-template.md` shape).
**Pattern reference:** thin-shell (`apps/` render, logic in `src/doc_assistant/` — root `CLAUDE.md`); the
`app_settings` `source_dir` persistence precedent (`app_settings.py`); ADR-010 / `feature-rag-sandbox.md`
threading discipline (**no module-global mutation**); the already-parameterized `build_chat_model` seam.

**Goal (the why).** ADR-011 recap: let the user switch the LLM provider/model from Settings **without a
restart and without re-entering a key**, among providers whose credentials already exist. The Anthropic
key stays in `.env` (v1 handles no secret); the *selection* is a non-secret persisted like `source_dir`;
the switch re-sources the generation model + reviewer between turns. Full rationale + the grill ledger:
`docs/decisions/ADR-011-desktop-provider-apikey-management.md`.

---

## Decisions (locked — ADR-011 grill ledger, 2026-07-10)

| # | Decision |
|---|---|
| 1 | **Switch is a live swap, not a restart.** Rebuild only the thin generation-model wrapper in `RAGPipeline`; embedder/reranker stay warm. Applied on the **next turn**. |
| 2 | **The reviewer follows the switch.** Switching to Ollama moves the per-answer reviewer to local too (truly free — KI-4). An **explicitly pinned** `REVIEWER_PROVIDER` in `.env` still wins. |
| 3 | **KI-4 = inform, no gate.** Provider options labeled paid/local (`config.PAID_PROVIDERS`); effective provider shown in the settings view + provenance + the existing per-turn cost chip. No confirm dialog. |
| 4 | **Provider + model** are both selectable; a keyless provider is **unavailable with the reason**; the selection persists in `settings.json` via `app_settings`; a switch does **not** reset the conversation. |
| 5 | **No module-global mutation** (`config.LLM_PROVIDER` etc. are never assigned). The effective provider is the persisted selection resolved at build/turn time. The **eval judge** and the CLI `--apply`/`assert_provider_intent` path are untouched. |

---

## Contracts (build-time)

### `src/doc_assistant/pipeline.py` — the generation-model swap seam

`build_chat_model(provider, model)` (`pipeline.py:78-100`) is already parameterized and makes **no** API
call at construction. `RAGPipeline.__init__` sets `self.llm = self._build_llm()` →
`build_chat_model(LLM_PROVIDER, LLM_MODEL)` (`:160-164`); `self.llm` drives the rewrite / answer /
multi-query chains (`:242, 257, 272`). Add the swap:

```python
def set_chat_model(self, provider: str, model: str) -> None:
    """Rebuild only the streaming generation model (rewrite/answer/multi-query chains
    read self.llm). No API call, no embedder/reranker reload. Idempotent."""
    self.llm = build_chat_model(provider, model)
```

- **In-flight turn is safe for free (fork F):** `stream_answer` binds `chain = ANSWER_PROMPT | self.llm`
  **per call** (`:257`), so a stream already running keeps the *old* model reference after a swap; the
  next turn builds a fresh chain on the new `self.llm`. No extra guard needed — assert this with a test.
- `build_chat_model`'s `api_key=SecretStr(ANTHROPIC_API_KEY or "")` (`:94`) stays config-sourced — v1 is
  key-via-`.env`, so no change to the key path.

### `src/doc_assistant/app_settings.py` — persist the non-secret selection

Mirror the `source_dir` pattern (JSON keys `llm_provider` / `llm_model` in `settings.json`):

- `get_llm_selection() -> tuple[str | None, str | None]` — the persisted `(provider, model)` or `(None, None)`.
- `set_llm_selection(provider: str, model: str) -> None` — validate `provider ∈ {"anthropic","ollama"}`
  and `llm.provider_available(provider)` (below); **raise `ValueError`** (API → 400) for an unknown or
  **keyless** provider — inform-don't-corrupt: never persist a selection that can't run. Persist via
  `save_user_settings`.
- **Effective resolution** (a small helper, e.g. `effective_llm()`): the persisted selection if present,
  else `(config.LLM_PROVIDER, config.LLM_MODEL)`. This is the single source of "what provider is live."

### `src/doc_assistant/llm.py` — reviewer follows the effective provider; a general availability check

- Add `provider_available(provider: str) -> bool` (generalizes `reviewer_available`, `:246-255`):
  `anthropic → bool(config.ANTHROPIC_API_KEY)`, `ollama → True`. Reused by the settings view (fork E) and
  the `set_llm_selection` guard.
- Reviewer resolution (fork C). Add `config.REVIEWER_PROVIDER_PINNED = os.getenv("REVIEWER_PROVIDER") is
  not None` (config.py) to detect a deliberate pin. Extend the reviewer builder so the call site can pass
  the effective provider:
  ```python
  def get_reviewer_client(effective_provider=None, effective_model=None) -> LLMClient:
      if config.REVIEWER_PROVIDER_PINNED or effective_provider is None:
          return make_client(config.REVIEWER_PROVIDER, config.REVIEWER_MODEL)   # today's behaviour
      # follow the switch: same provider AND a coherent model (REVIEWER_MODEL is a Haiku
      # name that would fail on Ollama), so use the effective chat model.
      return make_client(effective_provider, effective_model)
  ```
  Backward-compatible: `get_reviewer_client()` with no args is byte-identical to today.

### `src/doc_assistant/chat_controller.py` — apply the selection; wire the reviewer + local-mode flag

- **On construction**, apply any persisted selection so a restart restores it: after `self.rag` is built,
  read `app_settings.get_llm_selection()`; if present and it differs from the boot default, call
  `self.rag.set_chat_model(provider, model)`.
- Add `reconfigure(self, provider: str, model: str) -> None`: validate + `app_settings.set_llm_selection`,
  then `self.rag.set_chat_model(provider, model)`. This is a **direct method call, not a global mutation**
  (ADR-010 Decision 4). Applied between turns by construction (see the pipeline in-flight note).
- Reviewer call site (`:684-692`): pass the effective provider/model into `get_reviewer_client(...)` so it
  follows the switch (fork C).
- `_is_local()` (`:195`) currently reads `LLM_PROVIDER` — change it to read the **effective** provider so
  the "🖥 Local model — no metered token" line (`:840`) is truthful after a switch.

### `apps/api/models.py` — wire model (thin)

```python
class SettingsUpdate(BaseModel):
    source_dir: str | None = None
    llm_provider: Literal["anthropic", "ollama"] | None = None   # optional → backward compatible
    llm_model: str | None = None
```
(Both `llm_*` fields present together, or neither; validate pairing in the handler.)

### `apps/api/main.py` — endpoint + effective settings view

- `POST /api/settings` (`:294-303`): when `llm_provider`/`llm_model` are present, call
  `controller.reconfigure(provider, model)` (`ValueError` → 400 for keyless/unknown). The `source_dir`
  path is unchanged; sending neither is unchanged (backward compat).
- `_settings_view()` (`:113-137`): report the **effective** provider/model (from `app_settings.effective_llm()` /
  the controller), **not** the import-time `LLM_PROVIDER`/`LLM_MODEL` constants. Add a `providers` list for
  the UI: `[{"id": "anthropic", "available": provider_available("anthropic"), "paid": "anthropic" in
  config.PAID_PROVIDERS}, {"id": "ollama", "available": True, "paid": False}]`. (The `retrieval_weights`
  hardcoded-literal fix stays U1's, per `feature-rag-sandbox.md` — not this spec's.)
- Health (`:228-236`): `model = f"{eff_provider}/{eff_model}"` so `/api/health` reflects the switch.

### `apps/desktop/src/lib/{types.ts, api.ts}` + `Settings.svelte`

- `types.ts`/`api.ts`: add `llm_provider`/`llm_model` + the `providers` list to the settings type; the
  settings POST includes them when the user applies a switch.
- `Settings.svelte`: a **"Provider & model"** section (slots beside U1's Engine/sandbox sections) — a
  provider selector (each option labeled *metered* / *local* from `providers[].paid`; a
  `available:false` option is **disabled** with "add its API key to `.env`", fork E), a model input
  pre-filled with the effective model, and an **Apply** that POSTs `{llm_provider, llm_model}`. Show the
  effective provider/model as the current state. No secret field anywhere (v1).

---

## Build node

**Depends on:** ADR-011 (accepted); the `build_chat_model` seam; `app_settings`; the singleton
`ChatController` (PR-M2). **Independent of U1's `RagOverrides`** — provider is a *persisted global*
(settings.json + a pipeline reconfigure), not a request-scoped override; only `Settings.svelte` is shared
(U1c adds a section). No new dependency, no migration, no re-ingest, no embedder/reranker reload.
**Files owned:** `src/doc_assistant/{pipeline.py, app_settings.py, llm.py, config.py, chat_controller.py}`,
`apps/api/{models.py, main.py}`, `apps/desktop/src/lib/{types.ts, api.ts, Settings.svelte}`, tests below.
**Status:** design-locked, ready to build once U2/U3/U1/U1b have landed (or sooner — it's independent of
the RagOverrides path; only the Settings surface must be reconciled).

### Guard tests (written with the build; no paid LLM call — cpc §13)

- `tests/unit/test_pipeline_retrieval.py` — `set_chat_model(provider, model)` swaps `self.llm` (assert the
  new model via a fake `build_chat_model`, no API call); **in-flight capture:** a chain built before a swap
  still streams on the old model (bind `chain = ANSWER_PROMPT | self.llm`, swap, assert the pre-built chain
  is unaffected).
- `tests/unit/test_app_settings.py` — `set/get_llm_selection` round-trips; `effective_llm()` =
  persisted-overrides-env; `set_llm_selection` **raises** for an unknown provider and for a keyless paid
  provider (monkeypatch `ANTHROPIC_API_KEY` absent).
- `tests/unit/test_llm.py` — `provider_available` (anthropic needs a key, ollama always true);
  `get_reviewer_client` **follows** the effective provider when not pinned, **respects the pin** when
  `REVIEWER_PROVIDER` is set, and reproduces today's behaviour with no args.
- `tests/unit/test_chat_controller.py` — `reconfigure` calls `rag.set_chat_model` + persists (no global
  mutation — assert `config.LLM_PROVIDER` unchanged); the reviewer resolves via the effective provider;
  `_is_local()` reflects the effective provider; a persisted selection is applied at construction.
- `tests/unit/test_api_models.py` / API test — `POST /api/settings {llm_provider, llm_model}` reconfigures
  the injected controller; keyless/unknown → 400; `_settings_view` reports the **effective** provider +
  the `providers` list (availability + paid flags); `/api/health` reflects the switch; **absent `llm_*` →
  unchanged** (backward compat); `source_dir`-only still works.
- Frontend — `svelte-check` clean; the section is exercised via the preview harness (snapshots +
  synchronous evals; screenshots flaky on this box per `.claude/KNOWN_ISSUES.md`): switch provider →
  effective label updates; a keyless provider renders disabled with its reason.

### Definition of done

- Switching provider **+ model** from Settings takes effect on the **next turn** with **no restart** and no
  embedder/reranker reload; a mid-stream turn finishes on the old provider.
- The per-answer **reviewer follows** the switch (an explicit `.env` `REVIEWER_PROVIDER` pin is respected),
  so a switch to Ollama produces a **truly free** turn (no metered token).
- A provider with **no key** is shown **unavailable with the reason**; selecting it is impossible (UI) and
  rejected (API 400).
- The **effective** provider/model appears in the settings view, `/api/health`, and provenance; the
  per-turn cost chip is unchanged. **No confirm dialog** (inform-don't-block).
- The selection **persists across restart** via `settings.json`; **no `config` global is ever assigned**.
- **Backward compat:** with no `llm_*` sent, `/api/settings` and every turn are byte-identical to today.
- Gate green (ruff / `mypy --strict` / bandit / pytest); `svelte-check` clean; **no paid LLM calls in
  tests**; DEVLOG entry + a preview-harness verification note.

## Out of scope (v1)

- **In-app key entry / OS keychain (`keyring`)** — ADR-011 option 2, the v2 north-star (owes a `RIGOR_TODO`
  entry for the frozen-bundling + Secret-Service fallback before any v2 build).
- **The eval judge** (`JUDGE_PROVIDER`/`JUDGE_MODEL`) and the CLI `--apply` / `assert_provider_intent`
  path — not touched by the UI switch.
- **A/B provider compare** (run two providers side by side) — a later extension, like ADR-010 option 4.
- **Changing `REVIEWER_MODEL` semantics** beyond the follow/pin rule; **per-role provider UI** (separate
  controls for generation vs reviewer) — v1 is one switch that both follow.

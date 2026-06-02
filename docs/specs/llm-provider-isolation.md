# Spec — LLM provider protocol + reviewer context isolation

**Status:** Designed in Cowork 2026-05-29 (rev 2, verified against code). Ready for Claude Code execution.
**Owner of execution:** Claude Code (code + tests). Cowork wrote this spec.
**Pattern reference:** `claude-skills` → `prod-engineering/references/provider-isolation.md`.

**Requirement:** local-first is the **end goal**; the app is **hybrid today**. The *generator* (RAG analysis / chat) must run **fully locally** — Ollama, no `ANTHROPIC_API_KEY`; any design that makes a remote API mandatory *for generation* is rejected. The *reviewer and eval judge are pinned measurement instruments*, not generators: they default to a fixed, version-recorded reference model so cross-run numbers stay comparable, and move to local only after the local-judge calibration gate passes (see `tests/eval/TESTING.md` → "The judge is a pinned instrument"). The provider protocol + adapters below are still built in full so a local reviewer is a config flip once calibrated — the abstraction is mandatory, the local-reviewer *default* is not.

---

## Why rev 2 (the trap rev 1 fell into)

The codebase has **two LLM interfaces**, not one:

| Path | Interface | Site |
|---|---|---|
| Analysis (RAG answer, streamed to chat) | **LangChain** `ChatAnthropic(streaming=True, max_tokens=1024)` / `OllamaLLM` | `pipeline._build_llm()` (pipeline.py:71-85) |
| Reviewer **and** eval judge (one-shot JSON) | **raw Anthropic SDK** `client.messages.create(...)` | `reviewer.py:163`, `eval/scorers.py:275` |

A single `make_llm() -> BaseChatModel` cannot serve both — a LangChain model has no `.messages.create`, and the reviewer's `messages.create` is Anthropic-only (so `REVIEWER_PROVIDER=ollama` would crash). So the design splits by call shape:

- **Streaming chat** stays LangChain, made config-driven (it already supports Ollama via `LLM_MODE`).
- **One-shot calls** (reviewer + eval judge) move behind a small normalized `LLMClient.complete()` protocol with Anthropic **and** Ollama adapters — this is what unlocks local review.

---

## ADR — normalized one-shot LLM protocol + isolated reviewer

**Context.** Provider selection is ad-hoc: `pipeline._build_llm()` hardcodes `claude-haiku-4-5-20251001` / `llama3`; the reviewer and the eval judge each take an injected `Anthropic()` client and call `messages.create` (constructed at `apps/chainlit_app.py:262` and `commands.py:378`). Nothing lets the reviewer or judge run on a local model, so the local-first endgame is unreachable on the one-shot path.

**Decision.** Add `src/doc_assistant/llm.py` defining an `LLMClient` Protocol with one method, `complete(messages, *, temperature, max_tokens) -> str`, plus `AnthropicClient` and `OllamaClient` adapters and a `make_client(provider, model)` factory. Refactor the reviewer and the eval judge to depend on `LLMClient.complete` instead of `client.messages.create`. Generalize `pipeline._build_llm()` to read `LLM_PROVIDER`/`LLM_MODEL` while staying a streaming LangChain model. Lock the reviewer's context isolation (already structurally true) with a guard test.

**Options considered.** (1) One LangChain factory for both — rejected, the reviewer needs `messages.create`, not a chat model, and Ollama review would break. (2) Keep the reviewer Anthropic-only, make only the model configurable — rejected, it forecloses the local-first endgame (no Ollama path for review). (3) Normalized `complete()` for one-shot calls + LangChain for streaming — chosen.

**Consequences.** One swap point per call shape. Adding a backend = one adapter. The reviewer and eval judge share the factory (consolidation). The chat path keeps token streaming. Fully-local mode needs no API key.

---

## Contract — `src/doc_assistant/llm.py` (new)

```python
from typing import Protocol

Message = dict[str, str]   # {"role": "user"|"assistant"|"system", "content": str}

class LLMClient(Protocol):
    def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str: ...
```

- **Input:** non-empty `messages`; `temperature >= 0`; `max_tokens >= 1`.
- **Output:** the model's text. Never `None`; raises on transport failure (caller records the error, as `review_answer` already does).
- **Adapters:**
  - `AnthropicClient(model)` — wraps `anthropic.Anthropic()`; `complete` calls `messages.create` and extracts text (move `reviewer._extract_text` here — it is Anthropic-response-specific).
  - `OllamaClient(model, host=OLLAMA_HOST)` — wraps `langchain_ollama.ChatOllama` (already a dep); `complete` invokes with the messages and returns `.content`. No API key.
- **Factory:** `make_client(provider, model) -> LLMClient` (`anthropic` | `ollama`; `ValueError` otherwise — mirror `embeddings.get_model_config`).
- `get_reviewer_client()` reads `REVIEWER_PROVIDER` / `REVIEWER_MODEL`.
- **NOT responsible for:** prompt construction, JSON parsing, retries, cost tracking.

The analysis path keeps its own builder: `pipeline._build_llm()` reads `LLM_PROVIDER`/`LLM_MODEL` and returns a streaming LangChain model (`ChatAnthropic(streaming=True, max_tokens=1024)` for `anthropic`; `OllamaLLM(base_url=OLLAMA_HOST)` for `ollama`). It is intentionally separate from `LLMClient` — streaming chat vs one-shot JSON are different contracts.

---

## config.py additions

```python
LLM_PROVIDER      = os.getenv("LLM_PROVIDER", "anthropic" if LLM_MODE == "api" else "ollama")
LLM_MODEL         = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
REVIEWER_PROVIDER = os.getenv("REVIEWER_PROVIDER", LLM_PROVIDER)
REVIEWER_MODEL    = os.getenv("REVIEWER_MODEL", "claude-haiku-4-5-20251001")
JUDGE_PROVIDER    = os.getenv("JUDGE_PROVIDER", LLM_PROVIDER)   # eval harness
JUDGE_MODEL       = os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001")
```

`LLM_MODE` stays for back-compat (derives `LLM_PROVIDER`). Add all to `.env.example` with a "fully local" block: `LLM_PROVIDER=ollama`, `REVIEWER_PROVIDER=ollama`, a local `EMBEDDING_MODEL`, no `ANTHROPIC_API_KEY`.

---

## Build node

### Node — LLM protocol + local-capable reviewer
**Depends on:** none (independent of Chunk 2a and Feature 4a)
**Files owned:** `src/doc_assistant/llm.py`, `src/doc_assistant/config.py`, `src/doc_assistant/pipeline.py`, `src/doc_assistant/reviewer.py`, `src/doc_assistant/eval/scorers.py`, `apps/chainlit_app.py`, `src/doc_assistant/commands.py`, `tests/unit/test_llm.py`, `tests/unit/test_reviewer_isolation.py`, `.env.example`
**Status:** pending

1. **`llm.py`** — `Message`, `LLMClient` Protocol, `AnthropicClient`, `OllamaClient`, `make_client`, `get_reviewer_client`, `get_judge_client`. Move `_extract_text` into `AnthropicClient`.
2. **`reviewer.py`** — extract `build_reviewer_prompt(prov) -> str` from the inline `_REVIEWER_PROMPT.format(question=prov.query, evidence=_format_evidence(prov.retrieved_chunks), answer=prov.answer)`. Change `review_answer(prov, client, ...)` to type `client: LLMClient` and call `client.complete(build_messages(prompt), temperature=0.0, max_tokens=max_tokens)` instead of `client.messages.create`. Keep `_strip_fence` + JSON parse (model-agnostic).
3. **`eval/scorers.py`** — `LLMJudgeScorer` takes an `LLMClient`, calls `.complete(...)`. Same isolation contract.
4. **Call sites** — `apps/chainlit_app.py:262` and `commands.py:378`: replace `Anthropic(api_key=ANTHROPIC_API_KEY)` with `get_reviewer_client()`. Eval runner: build the judge via `make_client(JUDGE_PROVIDER, JUDGE_MODEL)`.
5. **`pipeline._build_llm()`** — read `LLM_PROVIDER`/`LLM_MODEL`; preserve `streaming=True` and `max_tokens=1024` for the Anthropic branch; `base_url=OLLAMA_HOST` for Ollama.
6. **Tests** (below) + `.env.example`.

### Guard test — `tests/unit/test_reviewer_isolation.py`

```python
def test_reviewer_prompt_is_evidence_only():
    prov = make_provenance(query="Q", answer="A",
                           retrieved_chunks=[chunk(text="EVIDENCE_TOKEN")])
    prompt = build_reviewer_prompt(prov)
    assert "EVIDENCE_TOKEN" in prompt        # judges against retrieved evidence
    assert "A" in prompt                      # and the answer under review
    # there is no analysis-conversation object on AnswerProvenance to leak — assert the surface stays minimal:
    assert set(vars(prov)) >= {"query", "answer", "retrieved_chunks"}

def test_reviewer_client_is_independently_configured(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("REVIEWER_MODEL", "claude-haiku-4-5-20251001")
    assert get_reviewer_client() is not get_active_llm()  # separate instances
```

### Contract test — `tests/unit/test_llm.py`

Parametrize over adapters: `complete()` returns a non-empty `str`, raises on unknown provider, honors the `(temperature, max_tokens)` signature. Mock the Anthropic SDK; mock or skip Ollama when no local server.

---

## Definition of done

- `uv run pytest` green incl. the two new files; `ruff` + `mypy --strict` + `bandit` clean.
- Defaults reproduce today's behaviour exactly (no env set → haiku analysis + haiku reviewer + Ollama fallback for analysis).
- **Fully local acceptance:** with `LLM_PROVIDER=ollama`, `REVIEWER_PROVIDER=ollama`, a local embedding model, and **`ANTHROPIC_API_KEY` unset**, a query runs end to end and `/review` produces a verdict — no network to Anthropic.
- DEVLOG entry per project rule.

---

## Optional follow-up — lock layers with `tach`

`doc_assistant` has no `tach` yet. Adopting it is a separate PR; this work adds a clean seam to enforce. Recommended `tach.toml`:

```toml
[tool.tach]
source_roots = ["src"]

[[tool.tach.modules]]
path = "doc_assistant.config"
depends_on = []

[[tool.tach.modules]]
path = "doc_assistant.llm"
depends_on = ["doc_assistant.config"]

[[tool.tach.modules]]
path = "doc_assistant.pipeline"
depends_on = ["doc_assistant.config", "doc_assistant.llm", "doc_assistant.prompts", "doc_assistant.tracking"]

[[tool.tach.modules]]
path = "doc_assistant.reviewer"
depends_on = ["doc_assistant.config", "doc_assistant.llm", "doc_assistant.db"]

[[tool.tach.modules]]
path = "apps"
depends_on = ["doc_assistant"]   # apps/ may call src/; src/ must never import apps/
```

`uv add --dev tach`, `uv run tach mod` to autodetect, then `uv run tach check` in pre-commit and CI. The day `pipeline.py` imports `ingest.py` (read path reaching the write path), CI goes red.

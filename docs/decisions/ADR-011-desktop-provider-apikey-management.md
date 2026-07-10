<!-- status: active · updated: 2026-07-10 · class: append-only -->

# ADR-011 — Desktop provider / API-key management: phased — provider switch first (no new secret surface), keyring key-entry deferred

- **Status:** accepted — v1 scope = option 4 (switch/model among already-configured providers; the API
  key stays in `.env`; live client-swap between turns). Keyring-backed in-app key entry (option 2) is the
  recorded north-star, phased after v1. **Grilled + design-locked 2026-07-10** (ledger below); all eight
  forks resolved. Not yet built — a v1 build spec follows.
  (proposed | accepted | superseded by ADR-NNN)
- **Date:** 2026-07-10
- **Deciders:** Lucas (with Claude Code)

> This ADR settles **how Phase 8's U1c ("provider / API-key management in Settings",
> `docs/specs/feature-phase8-ui-upgrade.md` §U1c) should work** — the one Phase-8 UI track
> deliberately left un-designed because it crosses from pure-frontend / already-governed overrides
> into **secret storage** and **construction-time provider binding**, a different risk class than
> U1/U1b/U2/U3. It records the product/security decision; a v1 build spec (files, endpoints, guard
> tests, DoD) follows.

## Context

Provider, model, and API key are resolved **once at import time** as module constants in
`config.py`: `LLM_PROVIDER` / `LLM_MODEL` (`config.py:121-125`) and
`ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")` (`config.py:108`), all read from `.env` via
`load_dotenv(override=True)` — where `.env` intentionally **wins** over the host environment
(`config.py:9-14`: an empty host `ANTHROPIC_API_KEY` must not shadow the real key).

The runtime provider surface has **three consumers, built at three different places** (verified for
this ADR — the naive "one LLM client on the controller" model is wrong):
- **Generation** (the streaming chat answer) is a **thin API-client wrapper** —
  `ChatAnthropic(streaming=True)` / `ChatOllama` — built **inside `RAGPipeline`** (`pipeline.py:89-96`,
  `stream_answer` `:250`), the same object that also holds the **expensive** embedder + reranker.
  `ChatController` caches **no** LLM client of its own — it holds the `RAGPipeline` (`chat_controller.py:431`).
- **Reviewer** is built **per flagged answer** via `get_reviewer_client()` (`llm.py:236-238`), reading
  `config.REVIEWER_PROVIDER` (defaults to `LLM_PROVIDER`) / `REVIEWER_MODEL` (an independent Haiku).
- **Judge** is **eval-harness only** (`get_judge_client()`, `llm.py:241`) — not in the live chat path.

The `ChatController` is a **process singleton** built once in the FastAPI `lifespan`
(`apps/api/main.py:205-217`, "model load is expensive"). `POST /api/settings` today persists **only**
`source_dir` (`main.py:294-303`) as JSON in `settings.json` in the data home (`app_settings.py:27,42-45`)
— the established precedent for a **non-secret, user-owned, per-install** persisted setting.

What makes this a real decision is that U1c bundles **two axes of very unequal risk**:

1. **Switching provider** is low-risk plumbing — but the import-time constants + the singleton mean a
   switch cannot simply "take effect"; the generation model (in `RAGPipeline`) and the reviewer's
   provider resolution must both be re-sourced, without mutating module globals (ADR-010 Decision 4).
2. **Storing an API key entered in-app** is a materially different risk class: plaintext-on-disk vs. an
   OS keychain; interaction with the deliberate `.env`-wins precedence (`config.py:9-14`); and the
   **KI-4 credit-leak footgun** (`.claude/KNOWN_ISSUES.md` — a run meant to be local silently billing
   Anthropic), which a provider-switch button could newly expose.

One sub-question the spec raised is answered by the code and is **not** a live axis: the frozen OS-trust
TLS path for corporate MITM proxies (`os_trust_http_client()`, `llm.py:95-132`, KI-10) is
`sys.frozen`-gated and constructed **only inside `AnthropicClient`**; Ollama talks to `localhost` and
makes no external TLS call. OS-trust is already per-provider and key-independent.

## Options

1. **`.env` as the app-writable source of truth; restart-gated switch.** The app writes the key +
   provider back into `.env`; the sidecar restarts and re-reads. Simplest, no new dependency, consistent
   with the `.env`-wins design (`config.py:14`). *Against:* the app co-authors a plaintext-secret file
   the user also hand-edits, and every switch is a full sidecar restart (~30 s cold-start incl. the
   model reload the singleton exists to avoid — RG-010, KI-9).
2. **OS keychain via `keyring`; in-app key entry; live client swap.** Secret in Windows Credential
   Manager / macOS Keychain / (Linux) Secret Service via `keyring`; non-secret selection in
   `settings.json`; live client rebuild. Correct hygiene + best UX. *Against:* a new runtime dependency
   with a **real PyInstaller frozen-bundling question** (platform-backend hidden imports — the exact
   class of freeze problem KI-9/KI-10 already cost this project), a Secret-Service-absent **fallback**,
   and a "no live secrets in tests" (cpc §13) surface — materially larger/riskier than switch alone.
3. **Session-only, in-memory key; no persistence; live swap.** Key held in sidecar memory only (the
   ADR-010 non-persistence wall), never on disk. Best hygiene. *Against:* the user re-enters the key on
   **every** launch — poor for a daily-use desktop app — with no persistence payoff.
4. **Provider/model switch only, among already-configured providers; key stays in `.env`; live swap
   between turns; no in-app secret handling in v1.** The UI flips provider/model + persists the
   *selection* (non-secret) via `settings.json`; the generation model + reviewer resolution re-source
   from it between turns; the embedder/reranker stay warm. *Against:* a fresh install with no key in
   `.env` cannot use Anthropic from the UI (must edit `.env` — which the README setup already requires).

## Decision

**Phased. Ship option 4 as v1; record option 2 as the v2 north-star, not built.** The deciding reason is
that the two axes carry **unequal risk and value** — v1 takes the high-value, low-risk axis (provider
switching) and pays none of the secret-storage cost, mirroring ADR-010's own move (ship the safe
overrides, phase the risky A/B north-star). The grill (ledger below) resolved the v1 shape:

- **Activation is a live swap, not a restart** (fork B). A switch rebuilds only the **thin generation-model
  wrapper inside `RAGPipeline`** (no network I/O at construction, `llm.py:187`) and re-resolves the
  reviewer's provider from the persisted selection; the embedder + reranker stay warm. It is **runtime-cheap**;
  the cost is a narrow `RAGPipeline` swap seam + reading the provider from a persisted setting rather than
  the import-time constant — **never** a module-global mutation (ADR-010 Decision 4). Because provider is a
  **persisted global** (not a per-request override), there is no per-turn concurrency divergence; the only
  timing concern is a mid-stream switch, which **applies on the next turn** (fork F). If the seam proves
  fiddly, restart-gated (option 1's mechanic) is the fallback.
- **The reviewer follows the switch** (fork C): switching to Ollama moves generation **and** the
  per-answer reviewer to local, so a "local" turn is **truly free** — honoring KI-4 and the "🖥 Local
  model — no metered token" UI (`chat_controller.py:840`). An **explicitly pinned** `REVIEWER_PROVIDER`
  in `.env` still wins (deliberate divergence respected). The eval **judge** is untouched (eval-only).
- **KI-4 is met by transparency, not a gate** (fork D): the provider control labels each option
  paid/local, provenance reports the **effective** provider, and the existing per-turn cost chip stays —
  **no confirm dialog** (inform-don't-block). `assert_provider_intent` continues to guard the CLI
  `--apply` batch path; the interactive chat already bills visibly per turn.
- **Both provider and model are selectable** (fork G): a provider dropdown + a model field pre-filled with
  the per-provider default (`config._DEFAULT_ANALYSIS_MODEL`), validated on use (a bad model surfaces an
  error — inform-don't-corrupt). A provider whose key is **absent** renders **unavailable with the reason**
  ("add it to `.env`", reusing the `config.ANTHROPIC_API_KEY` presence check, `llm.py:254`) — fork E. The
  selection persists as a non-secret in `settings.json` via `app_settings` (the `source_dir` precedent) —
  fork H. A switch does **not** reset the conversation.

**What would reverse it:** evidence from real first-run use that users routinely need **in-app key
entry** before switching is useful (e.g., onboarding with no `.env`) → pull option 2 forward (v1's
selection-persistence + live-swap seam is a strict subset, so v2 extends rather than replaces). Reviewer
review-quality on a local model being unacceptable in practice would reopen fork C.

## Grill ledger (2026-07-10)

| # | Fork | Resolution | Deciding reason |
|---|---|---|---|
| A | Phasing premise — is v1 (switch only, key via `.env`) worth shipping? | **Keep phasing.** | The app's target is a technical single-user whose README setup already requires `.env`; ships the safe axis, defers the risky secret build. |
| B | v1 activation — live swap vs restart-gated | **Live swap** (narrow `RAGPipeline` generation-model swap + reviewer re-resolution off the persisted setting; apply between turns). | Runtime-cheap (generation model is a thin wrapper, no weights); a 30 s cold-start on a settings action reads as broken. **Corrects the ADR's earlier "ChatController rebuilds the client" mechanism.** |
| C | Reviewer coupling on a switch to Ollama | **Reviewer follows the switch** (explicit `REVIEWER_PROVIDER` pin still wins). | "Local" must mean no billing (KI-4 + the "no metered token" UI); local review-quality is the user's own documented trade-off, not a silent surprise. |
| D | KI-4 guard strength on switch-to-paid | **Inform-only, no gate** (paid/local labels + effective provider in provenance + per-turn cost chip). | Matches the project's "inform-don't-block" rule; the chat already bills visibly per turn, so transparency prevents the surprise without friction. |
| E | Selecting a provider with no key | **Show unavailable + reason** (reuse the key-presence check). | Inform-don't-block; never crash a turn on a missing key. |
| F | Switch requested mid-stream | **Apply on the next turn** (in-flight turn finishes on the old provider). | No interruption; matches B's "between turns" mechanic. |
| G | Model-selection granularity | **Provider + model field** (per-provider default, validated on use). | Ollama users run varied local models; provider-only is too rigid. |
| H | Non-secret selection persistence | **`settings.json` via `app_settings`.** | The `source_dir` precedent — non-secret, per-install, survives restart; a new store is unjustified. |

## Consequences

- **Easier:** provider switching ships with zero new secret-handling risk and no new dependency; the
  live client-swap seam is exactly what v2 reuses; the non-secret selection rides the existing
  `app_settings` JSON precedent unchanged.
- **Harder / new obligations (v1 build spec):** a narrow `RAGPipeline` seam to swap **only** the
  generation-model wrapper (embedder/reranker untouched); reviewer provider re-sourced from the persisted
  selection **without** module-global mutation, respecting an explicit `REVIEWER_PROVIDER` pin; a
  non-secret `llm_provider`/`llm_model` field on the settings surface with `_settings_view` reporting the
  **effective** (not boot-time) provider; unavailable-with-reason for a keyless provider; apply-on-next-turn
  for a mid-stream switch; provider/model labels marking paid vs local. The eval judge and the CLI
  `--apply`/`assert_provider_intent` path are out of scope for the UI switch.
- **Must revisit (v2, tracked — not in this file):** the keyring dependency + PyInstaller bundling +
  Secret-Service-absent fallback + no-secrets-in-tests surface (`docs/specs/feature-phase8-ui-upgrade.md`
  §U1c open questions + a `RIGOR_TODO` entry own this before any v2 build); onboarding-with-no-`.env` as
  the trigger to pull v2 forward; local reviewer quality (fork C) if it proves too weak.

## Confidence

- ✓ **Live swap is runtime-cheap** — generation is a thin `ChatAnthropic`/`ChatOllama` wrapper (no
  weights; "construction does no network I/O", `llm.py:187`) and the embedder/reranker (the expensive
  loads, RG-010/KI-9) are provider-independent and stay warm. Grounded in `pipeline.py:89-96` /
  `chat_controller.py:431`.
- ✓ **KI-10 OS-trust is already per-provider / key-independent** (`llm.py:95-156`) — a switch changes
  nothing about it.
- ⚠ **Local reviewer quality (fork C) is unmeasured** — moving the reviewer to an 8B local model on a
  switch-to-Ollama degrades rubric scoring by an unquantified amount; acceptable as a user-chosen
  consequence, but if it proves too weak it reopens C (owed a `RIGOR_TODO` note at v1 build).
- ⚠ **v2 keyring bundling/fallback is unvalidated** — no frozen-build test that `keyring`'s Windows
  backend survives PyInstaller; a `RIGOR_TODO` entry must own it before any v2 build.
- ⚠ **v1's "users already have a `.env` key" assumption** is a product judgment (fork A), not measured —
  the named reverse condition above.

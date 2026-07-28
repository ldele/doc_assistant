<!-- status: active · updated: 2026-07-28 · class: append-only -->

# ADR-034 — First-run setup in the app: in-app API key (file store, not a keychain) + provider readiness

- **Status:** accepted (built 2026-07-28)
- **Date:** 2026-07-28
- **Deciders:** user + Claude Code
- **Extends [ADR-011](ADR-011-desktop-provider-apikey-management.md)** — this is the v2 that ADR-011
  phased, pulled forward by the trigger ADR-011 itself named
  (*"onboarding with no `.env`"*), and it **diverges from ADR-011's recorded north-star on the storage
  mechanism**: a file in the data home, not an OS keychain. ADR-011's v1 (provider/model switch,
  live client swap, persisted non-secret selection) is unchanged and reused whole.

## Context

The trigger is a release: the project is being handed to a **testing user** who did not build it. That
person's first five minutes are now the product's weakest surface, and the weakness is structural, not
cosmetic:

1. **A key can only arrive by editing a file.** ADR-011 v1 deliberately kept the Anthropic key in
   `.env`, on the reasoning that *"the README setup already requires"* editing it. That holds for a
   developer cloning the repo; it does not hold for a tester handed an installer, where there is no
   repo and no `.env` to edit. ADR-011 wrote the reversal condition for exactly this case.
2. **"Ollama is available" was a lie the UI told.** `llm.provider_available("ollama")` returned
   `True` unconditionally ("Ollama (local) needs nothing"). Nothing checked whether a server was
   running or whether a single model had been pulled — so the provider picker offered Ollama to a
   machine that had never installed it, and the user learned otherwise from a transport error on their
   first question.
3. **The two first-run blockers were half-reported.** The chat pane's empty state named *documents*
   only ("No documents indexed yet"). A user with documents indexed and no working provider got the
   normal "Ask your library a question" prompt, then a failure.
4. **A key saved at runtime could not have worked anyway.** `pipeline.build_chat_model` read
   `ANTHROPIC_API_KEY` through a module-level `from config import …` binding — a *separate binding*,
   the trap this repo has already paid for twice (`src/doc_assistant/CLAUDE.md`). Any in-app key
   feature that did not also fix the resolution point would have stored a key and kept sending none.

## Options

**A. Where a key entered in the app lives.**

1. **App writes `.env`.** *For:* one reader of key material (`config`), so the CLI and the app can
   never disagree. *Against:* the app co-authors a file the user also hand-edits (ADR-011 option 1's
   objection, unchanged), and there is no `.env` at all in a packaged install — the app would have to
   invent one somewhere and then teach `config` where.
2. **OS keychain via `keyring`** (ADR-011's north-star). *For:* correct secret hygiene. *Against:* the
   unvalidated risk ADR-011 itself flagged — a new runtime dependency with platform backends and a
   **real PyInstaller bundling question**, "the exact class of freeze problem KI-9/KI-10 already cost
   this project". Shipping that risk *inside the release whose whole purpose is a smooth first run*
   inverts the cost/benefit that made ADR-011 defer it.
3. **A file in the data home (chosen):** `credentials.json`, separate from `settings.json`, owner-only
   where the OS honours it. *For:* no new dependency, nothing to bundle, works identically frozen and
   from source, and the data home is already the app's private per-user state (per-user when frozen,
   gitignored in dev). *Against:* plaintext on disk — strictly weaker than a keychain, and the UI has
   to say so rather than imply protection it does not provide.
4. **Session-only, in memory.** *For:* best hygiene. *Against:* re-entry on every launch (ADR-011
   option 3's objection, unchanged) — and for a tester, "paste your key again" every run is precisely
   the friction this ADR exists to remove.

**B. Precedence between a `.env` key and a key saved in the app.**

1. **App store wins.** *For:* the most recent explicit user action wins. *Against:* the CLI/enrichment
   runners read the import-time `config` constant and cannot see the store, so the app and the CLI
   would be using **different keys** — with no way to tell from either side.
2. **Environment wins (chosen).** *For:* `.env` stays the single authority it already is
   (`load_dotenv(override=True)` — `config.py` deliberately lets `.env` beat the host env); the store
   is a *fallback* for the case that has no `.env`; app and CLI agree. *Against:* a developer who
   pastes a key while a stale one sits in `.env` would be confused — which is why the UI **names the
   live source** ("from your .env file" / "saved in this app") and says outright when `.env` is
   shadowing what they just typed.

**C. Verifying a key before storing it.**

1. **Store unverified.** *Against:* a typo produces a broken install that *looks* configured, and the
   failure surfaces later, on the answer path, as an opaque error.
2. **Verify with a completion call.** *Against:* first-run setup would spend the user's money.
3. **Verify with a free metadata call, and separate "rejected" from "could not check" (chosen).**
   `models.list()` bills nothing. A 401/403 is a refusal → **400, nothing stored**
   (inform-don't-corrupt). No network / a proxy / a timeout is **not** evidence the key is bad → store
   it and say it could not be checked (inform-don't-block). Discarding a key because the box is
   offline would be the worse failure.

## Decision

**Build first-run setup as a computed state, and let the app accept a key into a data-home file.**

1. **`doc_assistant.credentials`** owns key material: one file, one reader, `resolve_key()` =
   env-then-store (option B2), `key_source()`/`key_hint()` for display. Deliberately **not** more
   fields in `app_settings`: that file holds non-secret preferences and should stay pasteable into a
   bug report.
2. **`doc_assistant.readiness`** computes the whole first-run picture — per-provider
   configured/reachable/models/action, plus a step list. It **reports and never blocks**: an
   unreachable Ollama stays selectable (it may just not be started yet), and 0 documents is a
   legitimate state that says what to do, which is the 0-document half of the robustness contract.
3. **Reachability is a separate question from configuration.** `provider_available` keeps its
   local-state meaning (a credential is present) and is still what `set_llm_selection` gates on;
   `llm.ollama_probe` answers reachability, on the setup path only, with a 2 s structural budget.
   Collapsing the two would let a stopped local server invalidate a selection the user legitimately
   wants to keep.
4. **The key is resolved per client construction, never at import** — `AnthropicClient`,
   `build_chat_model`, the figure VLM client, and the CLI cost guard all call `resolve_key`. Saving a
   key then calls `ChatController.refresh_chat_model()`, so the change reaches **the next turn**, not
   the next restart (ADR-011's live-swap seam, reused as predicted).
5. **The UI states what it is.** The setup panel names the live key source, shows only a last-4 hint,
   says the key is stored on this machine, and offers Remove. The checklist and the chat-pane banner
   render the backend's step list, so the app, the API and the docs cannot drift into telling three
   different stories.

**What would reverse it:** a real requirement for OS-level secret protection (a shared or managed
machine, or a security review) → move the *storage* to option A2 behind `credentials`, which is the
seam that exists for it; nothing above `credentials.resolve_key` would change. Evidence that testers
routinely need a second keyed provider would generalize `_KEYED_PROVIDERS` (already a table, not an
`if`).

## Consequences

- **Good:** a tester can go from a fresh install to a cited answer without opening a text editor, on
  either path, and the app tells them which step is outstanding at each point. The Ollama half is
  strictly more honest than before: "not running" and "no models installed" are distinct states with
  distinct fixes.
- **Cost, stated plainly:** the key is **plaintext in the data home**. That is weaker than a keychain,
  it is the price of not shipping unvalidated frozen-build risk in this release, and the UI must keep
  saying where the key lives. Do not describe it as "secure storage".
- **Cost:** one more place key material can be read from, so *every* Anthropic call site must resolve
  through `credentials`. A new call site that reads `config.ANTHROPIC_API_KEY` directly reintroduces
  bug 4 above — silently, for in-app-key users only.
- **Test hygiene:** a real key saved on the dev box would otherwise change the suite's verdict
  (`provider_available` reads the store). `tests/conftest.py` now points the store at a temp path for
  every test — the first autouse fixture in this repo, and it exists for that reason.
- **Unchanged:** KI-4's discipline. The setup path never makes a paid call (verification is free), the
  CLI cost guard still fires for a key that came from the app, and enrichment defaults stay local.

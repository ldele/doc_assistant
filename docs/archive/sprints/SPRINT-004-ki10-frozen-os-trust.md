<!-- status: archived · updated: 2026-07-09 · class: disposable -->
<!-- LANDED 2026-07-09 (branch B, staged) — KI-10 RESOLVED via an on-proxy paid Step-C turn. -->

<!-- NEXT UP (2026-07-09). SPRINT-003/006/007 (G3/G6/G7) all landed + archived, so this is now the sole
     active contract — sprint_check sees exactly one. Independent subsystem (packaging/TLS, not
     epistemics). Runnable only on this TLS-MITM proxy box; on-proxy paid turn user-approved 2026-07-07. -->

# SPRINT-004 — ki10-frozen-os-trust

- **base:** main
- **depends-on:** none (independent of SPRINT-003; recommended second only for sequencing).
- **DoD:** the frozen desktop sidecar completes an outbound **Anthropic** turn on a TLS-MITM (corporate-proxy) box instead of dying with `[SSL: CERTIFICATE_VERIFY_FAILED]` (KI-10). **Diagnostic-first (host, this proxy box):** re-freeze with the existing WARN entrypoint and drive **one** on-proxy turn — a `WARN truststore.inject_into_ssl() failed …` on stderr ⇒ the freeze isn't bundling truststore (branch A); no warn but still `CERTIFICATE_VERIFY_FAILED` ⇒ the injected global patch doesn't reach the anthropic httpx client (branch B). *Do not write the fix before this read* (`docs/desktop-packaging.md:173-182`). **Core deliverable (branch B, recommended regardless):** `AnthropicClient.__init__` (`llm.py:96-100`) constructs the SDK client with a **guarded OS-trust http client** — `Anthropic(http_client=httpx.Client(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)))` when `truststore` imports, and a **clean fallback to the SDK default when it does not** (dev venvs, truststore absent). Confirm `Anthropic(http_client=…)` is the current SDK seam **before** writing (claude-api / installed anthropic version). The same OS-trust client is produced by **one shared helper** and reused at the second raw-SDK Anthropic seam — the VLM describer `ingest/figures.py:493-496` — so no cert logic is duplicated (LangChain `ChatAnthropic` in `pipeline.py:88-94` is OUT of scope, noted). **If diagnostics point to branch A:** add a PyInstaller **runtime hook** calling `inject_into_ssl()` + the named submodule(s) to `hiddenimports` in `scripts/doc_assistant_api.spec` (`:51`). **Regression guard:** a **construction-only** unit test — OS-trust context client when truststore is importable, clean fallback when not — with **no live paid API call** (cpc §13; `docs/desktop-packaging.md:202-204`). **Verification (host, this proxy box — Step C):** re-freeze, drive one real on-proxy turn → a token returns with no cert error; record the frozen **paid** first-token number in RG-011 and flip **KI-10 → RESOLVED** in `.claude/KNOWN_ISSUES.md`. (Cost: one real on-proxy turn bills a few cents once the handshake succeeds — **user-approved 2026-07-07**; $0 while it still fails.) Full gate green; nothing committed — staged for review.

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
<!-- read-set: the diagnostic doc + the two inject/construction seams + the status trackers to update -->
- docs/desktop-packaging.md
- apps/api/__main__.py
- src/doc_assistant/llm.py
- scripts/doc_assistant_api.spec
- .claude/KNOWN_ISSUES.md
- .claude/RIGOR_TODO.md

## affects
<!-- write-set: the change must stay inside this (globs allowed) -->
- src/doc_assistant/llm.py
- src/doc_assistant/ingest/figures.py
- scripts/doc_assistant_api.spec
- scripts/rthook_truststore.py
- tests/unit/test_llm.py
- docs/desktop-packaging.md
- .claude/KNOWN_ISSUES.md
- .claude/RIGOR_TODO.md

## contracts
<!-- type: target [ | when: <glob> ]  —  test = verify-loop reminder; snap/map must co-change -->
- test: tests/unit/test_llm.py::anthropic_client_uses_os_trust_context_when_truststore_present
- test: tests/unit/test_llm.py::anthropic_client_falls_back_cleanly_when_truststore_absent

## docs
<!-- must be touched when this lands (the docs gate enforces these appear in the diff) -->
- docs/DEVLOG.md
- docs/ROADMAP.md

<!--
Scope boundary (NOT machine-read):
- Root-cause lead (KNOWN_ISSUES KI-10 :207-214): ordering is NOT the cause — `_configure_frozen_runtime`
  already injects before `from apps.api.main import app`. The suspect is either freeze bundling (branch A)
  or the anthropic httpx client not honouring the process-global patch (branch B). Step B fixes B directly
  and is robust to both, which is why the doc recommends it regardless.
- `scripts/rthook_truststore.py` is listed in `affects` so the write-set gate ALLOWS a branch-A runtime
  hook if diagnostics require one; if branch B alone resolves it on-proxy, the hook file need not be created
  (an unused write-set entry is fine — a forbidden write is not).
- NO live paid API call in tests (cpc §13). The regression guard asserts CLIENT CONSTRUCTION only — that the
  http_client carries a truststore-backed SSLContext when importable, and that construction degrades cleanly
  to the SDK default when truststore is monkeypatched absent. The real paid on-proxy turn is a HOST step
  (Step C), not a test.
- OUT of scope: KI-9 (weights bundling — already RESOLVED, no HF network when frozen), RG-012 Tier-2
  clean-box cited-turn (PARKED, release-gate not dev-gate), and the LangChain `ChatAnthropic` seam.
- This sprint is uniquely runnable on THIS box (behind the TLS-MITM proxy) — the on-proxy Steps A/C cannot
  be reproduced on a non-proxy machine (memory: work-box sandbox + TLS-MITM proxy).
-->

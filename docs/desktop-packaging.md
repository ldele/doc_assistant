<!-- status: active · updated: 2026-08-07 (trap 1 now cites the general encoding rule) · class: runbook -->

# Desktop packaging runbook (PR-M4)

How to build the installable Tauri desktop app: freeze the FastAPI backend as a PyInstaller
**sidecar**, bundle it with the Svelte frontend, produce a native installer. ADR + rationale:
`docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md`; spec: `docs/archive/pr-m4-sidecar-packaging.md`.

> **Runs on a desktop, not in CI / the sandbox.** Freezing torch + Chroma + PyMuPDF (~GB
> binary), `cargo tauri build`, and the clean-machine smoke need the Tauri/Rust toolchain and
> a real machine. What *is* verified in-repo: the server entrypoint (`python -m apps.api`), the
> build-prereq check (`just sidecar-check`), and the frontend build. The freeze/build/smoke are
> the steps below.

---

## ✅ M4 finish checklist

**Where we are (2026-08-06): M4 is DONE — the checklist below is kept as the procedure, not as
open work.** The installer builds, installs on a clean Python-free box, and drives a real cited
turn there (**RG-012 Tier 2 passed 2026-08-05, re-passed 2026-08-06 on the rebuilt 0.4.1
artifact** — §5). Reference timings from that run: freeze **847 s**, `tauri build` **601 s**,
silent install **177 s**, `/api/health` **~30 s**, 3 PDFs → 322 chunks in **~36 s**, cited turn
**14 s**.

> **The release rule this section exists to enforce: never ship an artifact you have not rebuilt
> from the commit you are shipping, and never trust a green source tree as evidence about a frozen
> binary.** The build that could not read a single PDF (KI-34) passed every test, every lint and a
> source-install clean-room. Rebuild, then re-run §5.

**0. Commit today's fixes first** *(they must be baked into the freeze)*
- [ ] Commit the staged M4 follow-ups: data-dir relocation (`config.py`), missing-table
      tolerance (`epistemics.py`/`chat_controller.py`), command-error guards, streaming +
      thread-safe SQLite (`apps/api/main.py`, `db/session.py`). → `git add -A` → review → commit.

**1. See it as a native window** *(dev, optional, ~5 min — no freeze) — §"Dev loop" + §4*
- [ ] `npm i -g @tauri-apps/cli`
- [ ] T1 `just api`  ·  T2 `cd apps\desktop ; npx tauri dev`
      — **done when** a native window opens and a question streams a cited answer.

**2. Freeze the sidecar with today's fixes** *(§1–§2)*
- [ ] `uv sync --extra cpu --extra dev --extra packaging`
- [ ] `just sidecar`  — **done when** `apps\desktop\src-tauri\binaries\doc-assistant-api-<triple>.exe` exists.
- [ ] Smoke standalone: `set DOC_DATA_DIR=<repo>\data` → run the binary → `curl …/api/health`
      — **done when** `chunk_count` > 0. Iterate the spec on any runtime `ModuleNotFoundError`
      (add to `hiddenimports`/`datas`, re-run). ⚠ resolve the **data-home decision** below.

**3. Icons + installer** *(§3–§4)*
- [ ] `cd apps\desktop ; npx tauri icon <≥1024²>.png`
- [ ] `npx tauri build`  (first run pulls Rust crates, 10–30 min)
      — **done when** an installer lands under `src-tauri\target\release\bundle\`.
- [ ] Install + launch on THIS box — **done when** the app opens, spawns its sidecar, and
      streams a cited answer with **no terminal** running.

**4. Close the ship-gate rigor items** *(§5; `.claude/RIGOR_TODO.md`)*
- [ ] **RG-010** cold-start: time launch → first `/api/health 200`. Record. *(degrades)*
- [ ] **RG-011** SSE first-token latency vs Chainlit, on the frozen build. ***(BLOCKS SHIP)***
- [x] **RG-012** clean-machine smoke: install + drive one real turn on a box with no Python / no
      toolchain. **PASSED** 2026-08-05 (Windows Sandbox), re-passed 2026-08-06 on the rebuilt
      artifact. Re-run it for **every** release — §5.

**5. Then PR-M5** *(only after the installer ships + RG-011/012 pass)*
- [ ] Delete `apps/chainlit_app.py` + `.chainlit/`, drop the `chainlit` dep, lift the Python-3.12
      pin (KI-2) — confirm no other 3.12-only dep first (`anyio`/the rest).

**⚠ Open decision — resolve during step 2: where does a real install keep its data?**
Today a frozen build reads `DOC_DATA_DIR` → else a per-user dir (`%LOCALAPPDATA%\doc_assistant\data`)
→ else the repo `./data`. Manually setting `DOC_DATA_DIR` is fine for *your* testing but **not a
shippable UX**. Pick one before the installer is "done": (a) per-user dir + a first-run "ingest your
documents" flow; (b) ship a seeded corpus; (c) a settings-chosen folder. The ingest/seed flow is
**unbuilt** — it's the main product gap between "installer runs" and "installer is usable by someone
else."

---

## 0. One-time toolchain

- Rust + Cargo (`rustup`), Node ≥ 20.
- The project venv synced **CPU + the packaging extra**: `uv sync --extra cpu --extra dev --extra packaging`
  (PyInstaller lives in the `packaging` extra, kept out of the lean `dev` install).
- `npm install -g @tauri-apps/cli` (or use `npx tauri`).

## 1. CPU-torch pin (KI-3 — non-negotiable)

The frozen build **must** use the CPU torch wheel. The `cu130` wheel **segfaults (exit 139) on a
GPU-less box**, and the installer must run anywhere. `scripts/build_sidecar.py` refuses to freeze a
`+cu*` torch. On the GPU box, re-sync CPU first:

```bash
uv sync --extra cpu --extra dev
just sidecar-check      # asserts triple + CPU torch + entrypoint import
```

## 2. Freeze the FastAPI sidecar

```bash
just sidecar            # = python -m scripts.build_sidecar  → PyInstaller (slow, large)
```

Produces a single binary and copies it to `apps/desktop/src-tauri/binaries/doc-assistant-api-<target-triple>[.exe]`
(Tauri's sidecar naming). The spec (`scripts/doc_assistant_api.spec`) is a **starting point** —
freezing ML libs always needs a few rounds of fixes: run the produced binary, read the
`ModuleNotFoundError` / missing-data-file traceback, add to `hiddenimports` / `datas` in the spec,
repeat. Known suspects: `torch` dynamic ops, `chromadb` + its `onnxruntime`/`hnswlib` data,
`sentence_transformers`/`transformers` model configs, `tokenizers` native libs, `pymupdf` (`fitz`) DLLs.

Smoke the frozen binary standalone before bundling:

```bash
./apps/desktop/src-tauri/binaries/doc-assistant-api-<triple>     # starts uvicorn on :8001
curl http://127.0.0.1:8001/api/health                            # → {"status":"ok",...} once warm
```

## 3. Icons

`tauri.conf.json` references `icons/*`. Generate them from a source PNG (≥ 1024²):

```bash
cd apps/desktop && npx tauri icon path/to/icon.png   # writes src-tauri/icons/*
```

## 4. Build the app + installer

```bash
cd apps/desktop && npx tauri build
```

This runs `beforeBuildCommand` (`npm run build` → `dist/`), compiles the Rust shell, bundles the
sidecar (`bundle.externalBin`) + `dist/`, and emits the per-OS installer under
`src-tauri/target/release/bundle/`. The shell spawns the sidecar on startup (`src/lib.rs`); the
frontend's readiness gate polls `/api/health` until the models warm.

For the dev inner loop (no freeze), keep the two-process flow: `just api` + `npm run dev` (or
`npx tauri dev`, which loads the Vite dev server in a native window).

## 5. Rigor gates before "done" (`.claude/RIGOR_TODO.md` RG-010/011/012)

**RG-010 cold-start + RG-011 first-token — on this box** (`scripts/measure_latency.py`). Point the
frozen sidecar at the real corpus, then:

```powershell
$env:DOC_DATA_DIR = "C:\path\to\repo\data"
uv run --no-sync python -m scripts.measure_latency --launch dist\doc-assistant-api.exe
```

- **RG-010 cold-start** (launch → first `/api/health 200`, models warm) — *record the number*
  (degrades, no hard threshold; if >~30s, switch the spec onefile→onedir).
- **RG-011 first-token latency** — POST → first streamed token (⚠ a real, paid LLM call).
  ***Blocks ship.*** Compare to Chainlit (`just chat`, same question, stopwatch): both run the
  **same `ChatController`**, so only the freeze + the localhost HTTP/SSE hop differ — pass = not
  meaningfully slower.

**RG-012 clean-machine smoke — needs a Python-free box. ✅ TIER 2 PASSED 2026-08-05, re-passed
2026-08-06 on the rebuilt 0.4.1 installer.** Easiest clean box: **Windows Sandbox** (built into
Win 11 Pro — disposable, zero Python; enable via *Turn Windows features on/off → Windows Sandbox*,
restart).

The gate is now automated. Harness lives at `C:\rg012-host\` (local-only, not in the repo):
`rg012-tier2.wsb` maps `installer\ corpus\ script\ out\` and fires `script\rg012-run.ps1` from a
`LogonCommand`; the script installs silently, launches the shell, waits for `/api/health`, seeds
3 PDFs from `corpus\`, ingests them, drives one turn, and writes its verdict to `out\`.

**Four traps this harness exists to avoid — every one of them cost a run:**
1. **ASCII only, in both the `.ps1` and the `.wsb`.** Windows PowerShell 5.1 reads a UTF-8-no-BOM
   file as ANSI, so one em-dash breaks parsing *before anything logs* — which is what the two
   historical "LogonCommand never fires" reports actually were. Byte-check before trusting a run.
   This is one of three Windows encoding defaults that bite here; the other two (cp1252 console,
   ANSI file I/O) are in [`setup.md`](setup.md) § *Windows: text encoding*.
2. **Map a STAGED COPY of the installer**, never `target\release\bundle\nsis` directly: a running
   sandbox holds a handle on what it maps and will fail the next `tauri build` with os error 32.
3. **Never pick the installer incidentally** (`Select-Object -First 1` grabbed the June 0.1.0 build
   out of a folder holding both). Filter by product name, sort by build time.
4. **The gate must assert the app's own contract, not restate it.** It counted `'\[\d+\]'` — stricter
   than the app's citation parser — scored a passing turn as FAIL, and that false verdict was filed
   as an application bug (KI-35). It now reports *resolved / unresolvable / uncited* separately.

**Tier 2 needs an answer engine the sandbox can reach.** It uses the host's Ollama over the Default
Switch address (`http://172.29.224.1:11434`), which requires Ollama bound beyond loopback
(`OLLAMA_HOST=0.0.0.0:11434`, then restart it) for the duration of the run — **revert afterwards**.
The data-home gap this section used to describe is closed: the app has a first-run ingest flow, so
the sandbox seeds its own corpus and reaches `chunk_count > 0` unaided.

**Reading the log: mojibake in the harness's JSON dumps is the harness, not the app.** The
`settings:` and `setup readiness:` lines come from PowerShell 5.1's `Invoke-RestMethod`, which falls
back to **Latin-1** when a JSON response carries no `charset` in its `Content-Type`. So the `·` in
`SUPPORTED_NOTE` ("pdf · epub · html …") — UTF-8 bytes `C2 B7` — surfaces in `out\rg012.log` as the
two characters `U+00C2 U+00B7` (a capital A-circumflex before the dot). The shipped string is fine:
`apps/api/services.py` holds clean single `C2 B7` sequences and the Svelte UI reads it with
`fetch()`, which decodes JSON as UTF-8 per spec (verified 2026-08-15). **Do not file this as a
shipped mojibake** — that is the KI-35 shape, a gate artifact mistaken for an application defect.
Confirm against the source bytes before believing any encoding defect this log appears to show.

(This paragraph deliberately *names* the mojibake characters instead of printing them:
`tests/unit/test_docs_encoding.py` scans tracked docs for that byte pair and cannot tell a quoted
example from a leak, so writing it literally fails the guard — as it did when this note was drafted.)

**A packaging gate must push a real document end to end (KI-34).** Booting the frozen binary proves
nothing about whether its *data files* were bundled: the build that could not read a single PDF
started, served `/api/health`, and reported a healthy chunk count. Only extraction failed.

**Parity** is already guarded in CI (`tests/integration/test_turn_parity.py`) — the Tauri app
renders the same `TurnResult` as the CLI / Chainlit.

Record results in `.claude/RIGOR_TODO.md` (RG-010/011/012 → `Status: done`); RG-011 + RG-012 keep
the rigor gate red until they pass.

### KI-10 — frozen build rejects corporate-MITM HTTPS (the truststore fix)

On a box behind a TLS-inspecting (MITM) corporate proxy, the frozen `dist\doc-assistant-api.exe`
SSL-fails the outbound Anthropic call (`CERTIFICATE_VERIFY_FAILED`); `$0` billed (the handshake dies
first). This blocks the **frozen-build paid first-token** measurement (the last open piece of RG-011 on
this box). **Dev-reproducible on the proxy box — no sandbox / clean machine / RTX needed**; cost ≈ a
re-freeze + at most 1–2 paid calls (cents). Non-blocking by itself (Ollama / off-proxy use is unaffected).
Two live hypotheses: **(A)** `truststore.inject_into_ssl()` fails in the freeze (truststore not fully
bundled), or **(B)** inject runs but the anthropic SDK's httpx client ignores the global patch.

- **Step 0 — read the diagnostic (no code; the stderr-WARN is already staged in `apps/api/__main__.py`).**

  ```powershell
  just sidecar                                   # re-freeze with the WARN entrypoint (~1.6 GB)
  # run ONE on-proxy Anthropic turn through dist\doc-assistant-api.exe, capture stderr
  ```

  `WARN truststore.inject_into_ssl() failed …` in stderr ⇒ **branch A**; no warn but still
  `CERTIFICATE_VERIFY_FAILED` ⇒ **branch B**. **Don't write the fix before this read** — A and B differ;
  guessing wastes a re-freeze cycle.
- **Step A — inject fails in the freeze (fix bundling).** `collect_submodules("truststore")`
  (`scripts/doc_assistant_api.spec`) collects the modules but the onefile may not reproduce truststore's
  SSL patch. Add a **PyInstaller runtime hook** calling `inject_into_ssl()` (runs before any collected
  import) + any submodule the WARN names to `hiddenimports`. Re-freeze, re-read. If truststore stays
  fragile even with the hook, **fall through to Step B**.
- **Step B — anthropic client ignores the global inject (the robust fix; recommended regardless).** In
  `llm.py` `AnthropicClient.__init__` (today `Anthropic(api_key=...)` with **no custom `http_client`**),
  hand the SDK an explicit OS-trust httpx client:

  ```python
  import truststore, httpx, ssl
  ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)   # OS store carries the corporate MITM root CA
  self._client = Anthropic(api_key=..., http_client=httpx.Client(verify=ctx))
  ```

  Guard it (fall back to the default client when truststore is absent / a dev box; log the fallback). This
  removes the dependency on the global monkeypatch surviving the freeze and keeps the TLS concern inside
  `AnthropicClient`. Confirm `Anthropic(http_client=...)` is the current SDK seam before writing.
- **Step C — verify + record.** Re-run the on-proxy turn → first token + real answer (cents billed); record
  RG-011's frozen-build paid number in `.claude/RIGOR_TODO.md`, flip KI-10 → RESOLVED. **Tests:** no live
  paid call (cpc §13) — a construction-only unit test (truststore-context client when importable, clean
  fallback when not) is the regression guard.
- **Scope:** KI-9 (bundled weights + `HF_HUB_OFFLINE`) already removes all HuggingFace network when frozen,
  so KI-10's remaining scope is **only the outbound LLM call** — don't re-solve the HF side.

Full issue history + root-cause lead: `.claude/KNOWN_ISSUES.md` KI-10.

## Data directory (frozen builds)

A frozen onefile binary unpacks to a temp dir, so the in-repo `./data` path is meaningless
at runtime — `config._resolve_data_path()` (PR-M4) handles it:

- `DOC_DATA_DIR` env override wins (point it at an existing corpus to reuse it);
- else, when **frozen**, data lives in a stable per-user dir
  (`%LOCALAPPDATA%\doc_assistant\data` on Windows; `$XDG_DATA_HOME`/`~/.local/share` elsewhere);
- else (dev) the in-repo `./data`.

Reuse your dev corpus with the frozen binary:

```bat
set DOC_DATA_DIR=C:\path\to\doc_assistant\data
dist\doc-assistant-api.exe
curl http://127.0.0.1:8001/api/health     :: chunk_count should be > 0
```

A real install ingests into the per-user dir (or ships a seeded corpus there). Without this,
the symptom is a healthy server with `chunk_count: 0` / "empty (vector-only) index".

## Notes

- Bind `127.0.0.1` only (the entrypoint enforces it); the sidecar is the app's private backend.
- onefile sidecar = one binary (Tauri's model) at the cost of slower cold-start (unpacks to temp).
  If cold-start fails the latency gate, switch the spec to onedir + ship `_internal/` as a Tauri
  resource. Trade-off noted in `scripts/doc_assistant_api.spec`.
- Deleting Chainlit + lifting the Python-3.12 pin (KI-2) is **PR-M5**, after this ships.

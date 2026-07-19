<!-- status: active · updated: 2026-07-19 · class: living -->

# KNOWN ISSUES

Open weaknesses, recurring failures, workarounds. Log a bug the second time it appears.
Migrated from the old `CLAUDE.md` / `README` runtime-quirk notes on 2026-06-20 (cpc adoption).

## KI-1 — `print()` in `src/` violates the structlog-only standard — RESOLVED (2026-06-23)
- **Symptom:** Source modules used `print()` (32 calls / 4 modules) and stdlib `logging`
  (11 modules) with no logging configuration; the standard mandates structured logging via
  `structlog` and no `print()` in `src/`. `info`-level lines were invisible by default.
- **Cause:** Predated the logging standard; never fully back-filled.
- **Fix (shipped, ADR-003 / `docs/specs/structlog-observability.md`):** one `configure_logging`
  seam (`src/doc_assistant/logging_config.py`, structlog-over-stdlib, console/JSON renderers)
  called once at each app + program entrypoint; all 11 loggers → `structlog.get_logger`, all 16
  `%`-style sites → key-value events, all 32 prints → `log.*` (the `llm.py` paid-run abort box
  stays a direct `sys.stderr.write` — an interactive CLI safety prompt, ADR-B). `LOG_LEVEL`/
  `LOG_JSON` config contract. **Zero `print()` in `src/`**; behaviour-preserving (CLI progress,
  answers, eval untouched). Rule #5 is now true + enforceable.
- **Follow-up:** RG-013 (`.claude/RIGOR_TODO.md`) — the M4 freeze must re-verify `structlog`
  (now a base dep) is bundled. **Closed 2026-06-24:** verified on the frozen `dist/doc-assistant-api.exe`
  — startup console emits structlog events, zero `structlog`/import errors in the log.

## KI-2 — Python-3.12 runtime pin — STILL OPEN (cause renamed after PR-M5: native deps, not Chainlit)
- **Original cause (gone):** Chainlit's anyio stack broke on 3.14. **Chainlit was removed in PR-M5
  (2026-06-25)** — so that cause no longer exists, but the runtime pin does **not** lift.
- **M5 ADR-2 verification (2026-06-25):** with Chainlit gone, `uv sync --python 3.14 --extra cpu --extra dev`
  resolves + installs cleanly (torch `2.12.0+cpu` has a cp314 wheel; chainlit absent), and ruff /
  `mypy --strict src` / bandit all pass on 3.14 — **but the full pytest suite hard-crashes the interpreter**
  (no Python traceback; the process dies at ~47–54%, first surfacing in `tests/unit/test_llm.py` under
  full-suite load). It does **not** reproduce unit-only or for that test in isolation (load/order-dependent).
  327+ tests pass before the crash; Python 3.12 runs all **602** clean.
- **Cause:** a native/compiled dependency in the LLM-client import path (anthropic / langchain /
  `pydantic-core` / `tokenizers` — Python 3.14 is new and several C/Rust wheels aren't yet cp314-stable).
- **Workaround:** **run + test on Python 3.12** (the pinned runtime). CLI / FastAPI / Tauri all work on 3.12.
- **Real fix:** revisit when the native deps ship stable cp314 wheels — re-run the M5 ADR-2 check
  (`uv sync --python 3.14 …` + full gate) and lift the pin only when 602 pass on 3.14. Do **not** add
  3.13/3.14 trove classifiers until then.
- **Note:** the literal `--python 3.12` pin (the old `just chat`/`chainlit` recipe) is **deleted** (M5); the
  only thing now holding the runtime at 3.12 is this native-dependency stability gate.

## KI-3 — win32 `cu130` torch wheel segfaults on a GPU-less box — RESOLVED (2026-06)
- **Symptom:** Instant segfault when the CUDA (`cu130`) torch wheel ran on a machine with no usable GPU.
- **Cause:** A single lock-pinned `torch==…+cu130` for all `sys_platform == 'win32'`; `uv`'s
  `torch-backend = "auto"` is `uv pip`-only (a no-op for `uv lock`/`sync`/`run`).
- **Fix (shipped, `423cbfa`):** per-machine torch via **mutually-exclusive uv extras** — `--extra cu130`
  on a GPU box, `--extra cpu` on a CPU-only box / CI. **Rule: never `cu130` on a GPU-less box.**
- **Pointer:** `docs/specs/torch-backend-per-machine.md`.

## KI-4 — Anthropic-default credit leak on "local" enrichment runs — OPEN (workaround known)
- **Symptom:** Enrichment / self-eval runs intended to be local (Ollama) silently bill the Anthropic API.
- **Cause:** `.env` defaults are all-Anthropic; every generator/reviewer/judge inherits the default
  provider unless overridden.
- **Workaround:** **Force `--provider ollama`** (and reviewer/judge provider flags) on every
  enrichment / self-eval run. Cost-guards exist but the default is the trap.
- **Real fix:** A local-first default profile, or a hard guard that refuses paid calls outside an
  explicit `--allow-paid`. Contract: `docs/specs/llm-provider-isolation.md`.

## KI-5 — Sandbox cannot write runtime data; enrichment runners must run on the host — OPEN
- **Symptom:** Metadata backfill / citation extraction / other enrichment CLI passes appear to no-op
  or fail to persist when invoked from a sandboxed context.
- **Cause:** `data/` (SQLite + Chroma) is host-local and gitignored; the sandbox/host filesystem
  isn't synced for writes.
- **Workaround:** Run the idempotent enrichment runners (`scripts/extract_*`, `compute_doc_vectors`,
  etc.) **on the host**, not the sandbox. They're idempotent, so re-running is safe.
- **Real fix:** N/A — environmental; document and run on host.

## KI-6 — SSL crash on a `uv`-managed Python (Windows) — OPEN (per-machine; documented workaround)
- **Symptom:** App dies instantly with no traceback (`OPENSSL_Uplink(...): no OPENSSL_Applink`) on the
  first HTTPS call (Claude API, Ollama, any networked test).
- **Cause:** OpenSSL in uv's bundled (python-build-standalone) interpreter; an official CPython is unaffected.
- **Status:** a persistent per-machine environmental quirk; the fix is **deliberately not pinned in-repo**
  (`docs/decisions.md`) — kept as a documented remedy, revisited only if more boxes hit it.
- **Workaround:** Rebuild the venv on an official python.org 3.12 (`py install 3.12` →
  `uv venv --clear --python …` → `uv sync --all-extras`). Behind a TLS-inspecting proxy, prefix uv
  commands with `UV_NATIVE_TLS=1`. Offline work (ingest/embeddings/retrieval) is unaffected either way.

## KI-7 — Concept-graph LLM-extraction core + `data/graph/graph.json` are SUPERSEDED — RESOLVED (2026-07-07, SPRINT-001)
- **RESOLVED (2026-07-07):** the retirement landed. `concept_graph.py` +
  `scripts/build_concept_graph.py` + their tests are **deleted**; `epistemics.py` now sources node
  weights from `concept_skeleton.node_weights_for_epistemics`, and `wiki.py`'s cluster seam reads
  the skeleton's Louvain communities (`concept_skeleton.doc_clusters_from_skeleton`). No file in
  `src/`/`scripts/` imports the retired module (`grep -rEn "import .*concept_graph|from
  .*concept_graph|graph_from_dict|GRAPH_NAME" src/ scripts/` is clean; only historical
  "retired ``concept_graph``…" prose survives). `EPISTEMICS_MARKERS_ENABLED` now defaults `true`
  (ADR-005 superseded) — markers rest on the Node-A/B skeleton, not the deleted open-vocabulary
  graph. Left open by this sprint: the skeleton carries no publication years, so
  `superseded_trend` is not yet reachable (`contested`/`stable`/`unique` only) — a future
  year-aware Node-B pass would close that gap; not tracked as a new KI since it is a documented,
  intentional limitation of `node_weights_for_epistemics`, not a defect.
- **Node B DONE (2026-07-07):** the confined-LLM stance pass is built + merged (PR #6 `6679540`,
  `concept_skeleton_enrich.py`).
- **Symptom:** The shipped Feature 7 (PR 16) concept graph derives nodes from a per-document
  open-vocabulary LLM extraction; on this same-domain corpus that fragments concepts and is the
  dominant cost (36–40 LLM calls/doc; hit `budget_exhausted` over the 61-doc corpus).
- **Status:** Superseded in part by the **2026-06-18 concept-graph REDESIGN** (Decision C — user-curated
  vocabulary + deterministic skeleton from `Citation`/`DocSimilarity` + confined LLM enrichment). ADR-1
  (Louvain) and ADR-4 (composite chunk key) carry over; ADR-3 (LLM-extraction node source) + the single
  integrity tag do not. **Update (2026-06-30):** the redesign's **Node A — the deterministic, zero-LLM
  skeleton — is now BUILT** as a *new* module (`concept_skeleton.py` + `scripts/{seed_concepts,
  build_concept_skeleton}.py` + the four `concept_*` tables), alongside the old graph (Decision 8). The
  old open-vocabulary `concept_graph.py` + `data/graph/graph.json` are **still superseded** and still in
  place (nothing retired yet); Node B (LLM stance) + the RG-001 validation run remain. Feature 7d still
  reads the old graph until the connected re-point lands (see Cleanup).
- **Do not build on:** the current LLM-extraction graph or `data/graph/graph.json` (the on-disk file is
  also an empty environment artifact, not a quality measurement).
- **Pointer:** `docs/decisions.md` → "Feature 7 — concept-graph REDESIGN" (2026-06-18). Edge precision +
  presence recall are flagged for RIGOR_TODO before the edge model is locked.
- **Cleanup when built:** retiring `concept_graph.py` + `scripts/build_concept_graph.py` is gated on the
  redesign landing; it is a connected change across `epistemics.py` → `chat_controller.py` /
  `compute_epistemics.py` / `wiki.py` + their tests (the carried-over PR-16 ADR-1 Louvain / ADR-4 chunk
  key stay). Not safe as standalone cleanup while `epistemics.py` imports it.
- **Containment (2026-07-02, PR-R7 — staged):** the one *user-facing* leak of this superseded data — the
  live 7d marker chips in a chat turn — is now gated OFF by default (`EPISTEMICS_MARKERS_ENABLED=false`,
  `docs/decisions/ADR-005-epistemics-markers-default-off.md`), so the app no longer surfaces KI-7 noise under
  the integrity banner. Full retirement still owed with Node B; the flag flips back on then.
- **R5 decision run (2026-07-02, PASS — ADR-008):** the RG-001/008/009 validation the redesign was
  waiting on is **done**. On the main corpus (76 docs; the multi-domain home is absent) the deterministic
  skeleton at the validated `MIN_COOCCURRENCE=2` + `boundary` gives 21.5% density, clean retrieval/pose/
  connectome communities, a spread provenance-strength distribution (R4 discriminates), and a healthy
  ADR-004 Tier-1 gap layer. **RG-008/009 closed; ADR-004 Tier-1 unblocked; Node B (PR-B) unblocked** — the
  confined-LLM stance pass + the eventual retirement of the superseded `concept_graph.py` are now the live
  next steps (KI-7 stays OPEN until that connected re-point/retirement lands). Baseline:
  `tests/eval/baselines/rg001_concept_skeleton_r5_2026-07-02.md`.
- **Pointer (add):** also `docs/decisions/ADR-004-gap-detection-layer.md` + `docs/specs/feature-gap-detection.md`.

## KI-8 — PC→baseline marker mapping (PR-M1) is coarse at parent boundaries — OPEN (advisory, fail-safe)
- **Symptom:** In the default parent-child retrieval mode, the live 7d marker chip maps a marked baseline
  chunk onto a retrieved parent by **text containment** (`epistemics.markers_for_parent`): a parent gets a
  marker if it *contains* a marked chunk's text. The two collections are independent segmentations, so a
  parent spanning a marked chunk plus three clean ones is marked as a whole — over-attribution within the
  parent. A marked chunk straddling two parents marks both.
- **Why it's acceptable (for now):** markers are an **advisory chip, not a gate** (inform-don't-block), and
  over-attribution is fail-safe — it points the user at a real contested concept *in that passage*. The
  marker never changes synthesis, ranking, or the answer (byte-identical when absent).
- **Status:** chosen in PR-M1 ADR-1 over the heavier alternative (re-project `chunk_epistemics` onto PC
  parents — a second projection + migration + its own attribution-quality validation). That precise
  re-projection is the documented upgrade **if** containment proves too coarse on real data.
- **Compounding caveat:** marker *quality* upstream still comes from the superseded open-vocabulary graph
  (KI-7) — `contested` is local-model-noisy. M1 surfaces what the sidecar holds; it does not fix extraction.
- **Mostly moot in practice since 2026-07-02 (PR-R7):** the live chip is now default-OFF
  (`EPISTEMICS_MARKERS_ENABLED=false`, ADR-005), so this containment coarseness only bites when a user opts
  the markers back on. The precise re-projection upgrade rides with Node B, alongside KI-7 retirement.
- **Update (2026-07-16, docs review):** two bullets above are outdated. (a) The "compounding caveat" —
  the open-vocabulary graph is gone (KI-7 RESOLVED 2026-07-07, G1: `concept_graph.py` deleted); marker
  data now sources from the curated Node-A/B `concept_skeleton`, and attribution actually reaches chunks
  since the KI-15 label fix (G7). (b) The "mostly moot / default-OFF" bullet — G1 flipped
  `EPISTEMICS_MARKERS_ENABLED` back to **default-ON** (superseding ADR-005), so the containment
  coarseness is live again by default. The issue this entry tracks (PC-parent containment mapping is
  coarse) is unchanged and still OPEN; the re-projection upgrade remains the documented fix.
- **Pointer:** `docs/archive/pr-m1-epistemics-markers.md` ADR-1 (option 2 = the re-projection upgrade).

## KI-9 — Frozen desktop build does not bundle model weights → first-run HuggingFace download; offline launch fails — RESOLVED (2026-06-24)
- **Symptom:** On a clean machine (verified in **Windows Sandbox**, 2026-06-22, RG-012 Tier-1), the
  frozen sidecar's first launch downloads the `bge-base` embedder + the cross-encoder reranker from
  HuggingFace before `/api/health` goes green (≈218s of that cold-start). With no network, the backend
  never warms — `/api/health` never returns 200.
- **Cause:** `scripts/doc_assistant_api.spec` bundles the *library code* (`sentence_transformers`,
  `transformers`, `tokenizers`, `huggingface_hub`) via `collect_all`, but **not the model weights** —
  sentence-transformers resolves those from the HF cache / hub at runtime.
- **Why it matters:** the app is positioned local-first; a first-run network dependency (and a hard
  offline failure) is a shippability gap, and the download dominates the RG-010 cold-start number.
- **Workaround:** ensure network is available on first launch; or pre-seed the HF cache
  (`HF_HOME` / `%USERPROFILE%\.cache\huggingface`).
- **Real fix:** bundle the model weights into the freeze (add the model dirs to the spec `datas` and
  point `HF_HOME` / `SENTENCE_TRANSFORMERS_HOME` at the unpacked location), or ship them with the
  installer as a Tauri resource; re-measure RG-010 after. Decide before the M4 ship.
- **RESOLVED (2026-06-24):** `scripts/doc_assistant_api.spec` now stages a minimal, symlink-free,
  blob-less HF hub cache (deref'd `snapshots/` + `refs/main`, no `blobs/` → ~1.5 GB single copy) into the
  freeze at `hf_cache/`; `apps/api/__main__.py` points `HF_HOME` there + sets `HF_HUB_OFFLINE`/
  `TRANSFORMERS_OFFLINE` when frozen. **Verified offline:** with the user HF cache renamed away, the frozen
  binary reached `/api/health` 200 (`chunk_count=2455`) with zero download/network. Cost: binary 385 MB →
  1.6 GB; **RG-010 cold-start did not regress** (30.9 s). Optional future: onedir / Tauri-resource instead
  of embedding in the onefile (not needed — cold-start is fine).
- **Pointer:** `docs/desktop-packaging.md` §"Data directory"; RG-010 / RG-012 in `.claude/RIGOR_TODO.md`.

## KI-10 — Frozen build's bundled `certifi` rejects corporate-MITM'd HTTPS — RESOLVED (2026-07-09, SPRINT-004 branch B — frozen on-proxy paid turn succeeded)
- **Symptom:** On a box behind a TLS-inspecting (MITM) corporate proxy, the frozen
  `dist\doc-assistant-api.exe` fails outbound HTTPS with `[SSL: CERTIFICATE_VERIFY_FAILED] unable to get
  local issuer certificate`. Seen 2026-06-23 twice: (1) the startup HuggingFace metadata HEAD for
  `bge-base` → **startup crash** even with a warm cache; (2) the Anthropic chat call → turn fails with no
  token (blocks RG-011's first-token measurement on this box).
- **Cause:** PyInstaller bundles **`certifi`** (Mozilla's CA set). httpx (used by `huggingface_hub` and the
  `anthropic` SDK) pins certifi, so it never consults the **Windows root store** where the corporate MITM
  root CA lives. Distinct from KI-6 (that is the `OPENSSL_Applink` crash on uv's python-build-standalone
  interpreter; this is a CA-trust gap in the *frozen* build).
- **Workaround:** (HF) `HF_HUB_OFFLINE=1` + a warm cache (or KI-9's bundle-the-weights — then no HF network
  at all). (LLM) no env-only fix — httpx ignores `SSL_CERT_FILE`; measure on a non-proxy box or via the
  local Ollama path (RTX box), which makes no external TLS call.
- **Real fix (shippability):** make the frozen build use the OS trust store for outbound TLS — bundle
  `truststore` (`truststore.inject_into_ssl()` at the API entrypoint) or `pip-system-certs` in the freeze —
  so a corporate-proxy user isn't blocked. Decide before the M4 ship (couples KI-9 + RG-011).
- **Update (2026-06-24):** the **RG-011 first-token** this blocked is now **measured on the RTX/Ollama
  path** (no external TLS, proxy-independent) — FastAPI/SSE boundary adds no measurable first-token
  latency vs the in-process `ChatController` (median 4.14s vs 4.56s, n=5; boundary **PASS**). Baseline:
  `tests/eval/baselines/rg011_first_token_ollama_2026-06-24.md`. **KI-10 itself stays OPEN** — the frozen
  build's cert-trust gap is unfixed; only the *measurement it was blocking* is unblocked (and only on the
  local-LLM path; the frozen-artifact + paid/non-proxy first-token still pend).
- **Fix implemented (2026-06-24):** `truststore>=0.10` is a base dep; `apps/api/__main__.py` calls
  `truststore.inject_into_ssl()` at the entrypoint (guarded; before the app/httpx/anthropic import); the
  spec bundles it (`collect_submodules("truststore")`). Outbound TLS now uses the **OS/system trust store**
  (which carries a corporate MITM root CA) instead of the bundled `certifi` set. **Verified:** imports +
  injects cleanly in dev and in the frozen build (no cert error in the frozen log). **Pending:** confirm
  on an actual TLS-MITM box — this RTX box isn't behind one, so the proxy-cert fix is implemented + bundled
  but not yet confirmed against a real MITM proxy. Status → near-resolved; close after the on-proxy check.
- **On-proxy check (2026-06-25, this work box behind the TLS-MITM proxy) — CONFIRMED STILL BROKEN.** Drove a
  real Anthropic turn through the **re-frozen** sidecar (1.62 GB, truststore bundled): retrieval succeeded
  ("Found 10 relevant passages") but generation produced **no token** — the worker-thread stderr shows
  `httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate` →
  `anthropic.APIConnectionError`. So `truststore.inject_into_ssl()` is **not taking effect** for the anthropic
  httpx client in the freeze. **$0 billed** — the handshake fails before any request reaches Anthropic.
  **KI-10 → OPEN (confirmed).** Non-blocking (only paid-API use on a corporate-proxy machine; Ollama /
  off-proxy use is unaffected).
- **Root-cause lead (2026-06-25):** ordering is NOT the cause — `apps/api/__main__._configure_frozen_runtime`
  calls `truststore.inject_into_ssl()` **before** `from apps.api.main import app` (so before httpx/anthropic
  import). But that call was wrapped in a **silent `try/except: pass`**: if `inject_into_ssl()` itself fails in
  the freeze (e.g. truststore not fully bundled/usable), the error was swallowed → certifi used → exactly this
  cert failure. Changed the handler to **write the failure to stderr** (also fixes a bandit B110). **Next:**
  re-freeze + re-run the on-proxy turn — a stderr `WARN truststore.inject_into_ssl() failed …` will confirm
  whether inject is the failure point; if so fix the freeze's truststore bundling, else hand the anthropic
  client an explicit OS-trust `verify` context. Reproducible/fixable in **dev** on this proxy box.
- **Branch-B fix BUILT (2026-07-09, SPRINT-004 — staged, not committed):** the diagnostic lead above
  (inject doesn't reach the anthropic httpx client in the freeze) is exactly what branch B sidesteps —
  it stops depending on the process-global patch. New `llm.os_trust_http_client()` builds an anthropic
  `DefaultHttpxClient(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))` (OS trust store carries the
  MITM root CA) and `AnthropicClient.__init__` / `AnthropicVisionDescriber.__init__` pass it as
  `http_client=`. **`sys.frozen`-gated** — `None` (SDK default certifi) in dev/tests, OS-trust client only
  in the frozen build where KI-10 bites — so dev/test behaviour is byte-unchanged (no ripple).
  `truststore`/SDK import guarded → clean certifi fallback + `log.info` if unavailable. `DefaultHttpxClient`
  (not a bare `httpx.Client`) preserves the SDK's default timeouts/limits. +2 construction-only unit tests
  (present+frozen→OS-trust context; absent→certifi fallback), **no paid call** (cpc §13); gate green
  (791 passed). Branch A (PyInstaller runtime hook) not needed — the on-proxy Step-C run below confirms
  truststore imports + injects fine in the freeze (no `WARN truststore.inject_into_ssl() failed`), so
  branch B's explicit `http_client` is what took effect.
- **RESOLVED — Step C run on-proxy (2026-07-09, this TLS-MITM box):** re-froze with branch B
  (`just sidecar` → fresh 1.62 GB `dist\doc-assistant-api.exe`, 12:00), launched it against the dev corpus
  (`DOC_DATA_DIR`, `chunk_count=30882`, `model=anthropic/claude-haiku-4-5-20251001`), and drove **one real
  on-proxy `/api/chat` turn**. Result: **HTTP 200, tokens streamed, a grounded cited answer,
  `cost_usd≈$0.0059` billed (`is_local:false`) — the paid Anthropic call succeeded through the corporate
  MITM proxy with ZERO `CERTIFICATE_VERIFY_FAILED`** (frozen server log clean: no SSL/ConnectError, no
  truststore WARN). This is the exact turn that failed the handshake with `$0` billed on 2026-06-25.
  **KI-10 → RESOLVED.** Frozen paid number recorded in `.claude/RIGOR_TODO.md` RG-011. (Housekeeping: the
  re-freeze needed the `packaging` extra — `uv sync --extra cpu --extra dev --extra packaging`; run
  `uv sync --extra cpu --extra dev` to return the venv to its lean documented state.)
- **Addendum (2026-07-17) — the failure is `httpx`-specific; stdlib `urllib` is unaffected (DEV only).**
  Measured on this proxy box while planning the Crossref metadata lookup: **stdlib
  `urllib.request.urlopen` reaches `https://api.crossref.org` cleanly, both WITH and WITHOUT
  `truststore.inject_into_ssl()`** (~0.7–0.8 s; a full spike over the corpus's 25 real DOIs resolved
  **25/25, 0 failures**). Why: `truststore.inject_into_ssl()` is a **process-global** SSL patch and stdlib
  urllib honours it, whereas **httpx pins certifi and ignores both the global patch and `SSL_CERT_FILE`**
  (the cause above). This is why the KI-10 fix had to be branch B (hand the SDK an explicit OS-trust
  client, `llm.os_trust_http_client()` `:95-132`) — note that helper returns `None` when **not frozen** and
  is **anthropic-typed**, so it is a *pattern, not a reusable component* for a second client.
  **⚠ Scope this claim honestly:** measured on the **dev interpreter** (`sys.frozen is False`), one box,
  one day, one proxy state. **It does NOT prove the frozen build** — KI-10's whole subject is PyInstaller
  bundling certifi, and a frozen stdlib-urllib call was **not** tested. **Consequence for design:** a new
  outbound call from `src/` should prefer the stdlib `urllib` seam (`sources_manifest.py:278-285`
  `_http_get`) over adding an httpx client — urllib sidesteps this KI in dev; httpx means re-solving it.
  Re-verify on the frozen build before shipping any outbound call in a release.
- **Pointer:** RG-010/RG-011 progress in `.claude/RIGOR_TODO.md`; `docs/desktop-packaging.md` §5.

## KI-11 — chromadb hnsw index not persisted under a non-ASCII path → broken corpus for accented usernames — RESOLVED (2026-06-24)
- **Symptom:** A fresh ingest whose Chroma persist directory's **actual filesystem location contains
  non-ASCII characters** does **not** write the hnsw segment files (`data_level0.bin` / `header.bin` /
  `length.bin` / `link_lists.bin`) — only `index_metadata.pickle` + `chroma.sqlite3`. chromadb then
  attempts a read-time *backfill* on next open: it works for a tiny corpus (~310 chunks) but **fails for a
  real-size one** (2455 chunks) with `chromadb.errors.InternalError: Error executing plan: Error sending
  backfill request to compactor: … Error loading hnsw index`.
- **Where it bites:** the shipped desktop app keeps its corpus in the **per-user data home**
  `C:\Users\<username>\AppData\Local\doc_assistant\data` (PR-M4, `config._resolve_data_path`). Any user
  whose Windows username has an accent / non-Latin character (é, ü, ñ, CJK, Cyrillic — very common) gets a
  non-ASCII path → a corpus that won't reload. Verified on this box (a Windows username containing a
  non-ASCII character — here an accented `é`).
- **Confirmed (2026-06-24, chromadb 1.5.9), path is the variable:**
  - ASCII location (`C:\Projects\…`), 1 **and** 10 files → `.bin` written, reloads fine.
  - Non-ASCII location (`C:\Users\<non-ASCII username>\…`, e.g. an accented `é`), 10 files / 2455 chunks → no `.bin` → reload **fails**.
  - The Windows **8.3 short path** (`C:\Users\<NAME>~1\…`, an ASCII *string*) does **NOT** help — chromadb /
    hnswlib resolves it to the real `é` directory for file I/O, so `.bin` still isn't written.
- **NOT the cause (ruled out):** a general "fresh ingest is broken" (a fresh full ASCII ingest works), the
  freeze (the venv reproduces it identically), or corpus size alone (ASCII 2455 works).
- **Impact:** breaks **RG-012 Tier-2** (a real cited turn on a clean box) and any from-scratch re-ingest
  under a non-ASCII home. The existing repo index (`C:\Projects\…`, ASCII) is unaffected — which is why
  this stayed latent until the per-user data home was exercised.
- **Workarounds:** install / point `DOC_DATA_DIR` at a **pure-ASCII** path (e.g. `C:\doc_assistant\data`);
  or pre-seed the corpus from an ASCII build. The 8.3 short path is **not** a workaround.
- **FIX SHIPPED (2026-06-24, option a):** `config._chroma_base()` — when `DATA_PATH` is non-ASCII on
  Windows, the **Chroma vector dirs only** relocate to a guaranteed-ASCII machine path
  (`%PROGRAMDATA%\doc_assistant\chroma\<sha1(data_path)[:12]>`); SQLite (`library.db`) + sources stay at the
  per-user home (SQLite handles non-ASCII fine). ASCII data paths and non-Windows are unchanged (byte-for-byte
  `DATA_PATH/chroma`). Also fixed `ingest.py` to `mkdir(parents=True, …)` the Chroma dirs (the relocated base
  has new intermediate dirs). **Verified on this box** (data home `…/café_home`, the `é`): Chroma landed at
  `C:\ProgramData\doc_assistant\chroma\…`, the full 10-file ingest wrote all four `.bin`, and a fresh process
  reloaded the full corpus (chunk_count 2335) — the exact case that failed before. **Remaining:** the
  *shipped* frozen sidecar/installer must be re-frozen to bundle this `config` change (the fix is in `src/`;
  any future `just sidecar` picks it up). Upstream report to chromadb/hnswlib still worth filing.
- **Pointer:** found while validating the data-home flow without a clean box (RG-012); `docs/DEVLOG.md`
  2026-06-24 session.

## KI-12 — Inverse orphan: Chroma chunks with no Document row (post-F1 write reorder) — RESOLVED (2026-06-26)
- **Symptom:** A document's chunks are present in **both** Chroma stores (so in the dedup
  intersection) but it has **no `Document` row** in SQLite. The library UI (which counts rows)
  undercounts it; retrieval is unaffected (it reads Chroma).
- **Cause:** The F1 write reorder commits the SQLite row **last**, after both Chroma writes, to
  prevent the *forward* orphan (a committed row with zero chunks). That leaves the narrow inverse:
  both vector writes land and only the final `upsert_document_in_sqlite` commit fails. The
  intersection dedup gate self-heals a partial *Chroma* write, but on its own it treats this hash
  as "already indexed" and skips it — so only `--rebuild` cleared it.
- **Fix (shipped, 2026-06-26):** `main()` now reconciles the dedup set against SQLite —
  `inverse_orphans = (get_indexed_hashes(db) & get_indexed_hashes(pc_db)) - get_document_row_hashes()`
  are subtracted from `indexed` (with a `chroma_chunks_without_document_row` warning), so the
  document is reprocessed and its row committed on the **next ordinary ingest**. The SQLite-side
  twin of the Chroma-side self-heal; nothing is deleted (chunks re-add idempotently). The
  gone / content-changed shapes are already swept by `cleanup_orphans_*`, so only the
  source-present + unchanged shape reaches the reconciliation.
- **Regression test:**
  `tests/integration/ingest/test_ingest_write_ordering.py::test_sqlite_commit_failure_self_heals_via_reconciliation`
  — monkeypatch the final commit to fail after both Chroma writes, assert the inverse-orphan state,
  then assert a clean re-run commits the row. Verified to fail on the warn-only (no-subtraction) code.
- **Pointer:** `docs/DEVLOG.md` 2026-06-26 ingestion-hardening F1 "Opens" + the follow-up entry; the
  dedup-gate comment in `ingest.py:main`.

## KI-13 — concept-skeleton vocabulary seam is dead on real data (no `Keyword` producer) — RESOLVED (2026-07-01)
- **Symptom:** `scripts/seed_concepts.py` mines curated-vocabulary candidates from `Keyword` rows, but the
  `keywords` (and `document_keywords`) tables are **empty on the real corpus** and stay empty after a full
  ingest — so `seed_concepts` lists 0 candidates and `promote_keyword` returns `None` for everything. The
  concept skeleton (Node A) is therefore empty via the intended path.
- **Cause:** **Nothing in the codebase ever writes a `Keyword` row.** The `Keyword` model +
  `document_keywords` association exist, but no ingest step or enrichment runner populates them
  (`extract_doc_metadata.py` fills `title`/`authors`/`year`/`doi` only). The redesign's Decision 1 ("seed
  candidates from `Keyword` + manual `--promote`") assumes a keyword producer that was never built.
- **Impact:** blocks the RG-001/008/009 validation via the documented seam; the vocabulary must be seeded by
  hand (direct `Concept`/`ConceptAlias` inserts, as done for the 2026-07-01 run). No user-facing CLI exists
  for direct concept creation either — only the dead `--promote` path.
- **Workaround:** insert `Concept` (+ `ConceptAlias`) rows directly via the ORM (the 2026-07-01 baseline run
  did this with a provisional 30-concept set).
- **Fix (shipped 2026-07-01, option a — staged for review):** new `src/doc_assistant/knowledge/keywords.py` — a
  deterministic, **zero-LLM, zero-new-dependency** corpus TF-IDF keyword extractor (pure core
  `tokenize`/`candidate_terms`/`tf_idf_keywords` + impure boundary reading cached markdown, writing
  `Keyword(source="extracted")` rows + `document_keywords` links; additive, idempotent, never touches the
  chunk store) + CLI `scripts/extract_keywords.py` (`--apply`/`--force`/`--doc`/`--top-k`, dry-run default) +
  `KEYWORDS_PER_DOC`/`KEYWORD_NGRAM_MAX`/`KEYWORD_MIN_CHARS` config. The `--promote` seam now works as designed:
  `extract_keywords --apply` → `seed_concepts` → `--promote`. **Verified on the real corpus:** 148 candidates
  written (was 0), each a real IR term; TF-IDF surfaces distinctive per-paper terms (colbert, hyde, late
  interaction) and down-ranks the broad hubs that saturated the RG-008 run — a useful side effect. +17 tests
  (6 unit, 5 integration; `list_keyword_candidates` loop-closure asserted). Gate green.
- **Follow-up (not this fix):** the RG-001 run can now re-seed its vocabulary from *extracted* candidates
  instead of the ad-hoc hand-seeded 30 — a better-grounded re-run once a curator promotes a subset.
- **Pointer:** `tests/eval/baselines/rg001_concept_skeleton_2026-07-01.md`; `docs/archive/concept-graph-redesign.md`
  Decision 1; `.claude/RIGOR_TODO.md` RG-001/008/009.

## KI-14 — PyMuPDF4LLM image placeholders pollute the extracted markdown — RESOLVED (main corpus, 2026-07-02)
- **Symptom:** the cached markdown contains `**==> picture [W x H] intentionally omitted <==**` placeholder
  lines wherever PyMuPDF4LLM declines to render an inline image. On the multi-domain arXiv corpus this was
  **1027 occurrences** across 24 papers, heaviest in figure/equation-dense physics/math/econ papers
  (statmech 214, cosmo 182, econ 173; text-heavy ML papers far fewer). These land in the RAG **chunk store**
  (retrievable noise) and in keyword extraction (surfacing junk tokens `intentionally omitted`, `x 12`,
  `br 1`, `0 br` — 11 of 13 concept-skeleton "communities" on that corpus were these noise isolates).
- **Cause:** PyMuPDF4LLM's markdown writer emits a textual placeholder for images it does not extract; the
  primary ingest path keeps it verbatim (figures are handled separately by the Feature-4b sidecar, so the
  placeholder carries no value in the chunk text).
- **Impact:** low on the public IR corpus (text-heavy, few figures); **material on STEM/figure-dense corpora**
  — pollutes retrieval and any text-derived enrichment (keywords, future concept vocabulary). Surfaced by the
  RG-001 multi-domain re-check, not the public corpus.
- **Workaround:** none applied (would be a code change; out of scope for the "current-params re-check").
- **Real fix:** strip `==> … intentionally omitted <==` placeholder lines in the extract→markdown step (or a
  cache-normalisation pass) before chunking + keywording; optionally re-point them at the figure sidecar.
- **FIX BUILT (2026-07-02, PR-R1 — `docs/archive/remediation-plan-2026-07.md` §R1; staged, not committed):**
  `extractors.strip_image_placeholders` (frame-anchored `==> … <==`, whole-line, no-op when absent + idempotent)
  applied at the single `extract_to_markdown` exit → all future extractions clean; `scripts/normalize_cache.py`
  (dry-run default, `--apply`, atomic per-file rewrite only when content changes) fixes existing caches, since
  `--rebuild` does NOT re-extract (`ingest/cache.py` trusts mtime). +23 guard tests; gate green (699 passed).
  Dry-run on `data/cache`: 62 scanned, **57 changed, 1,123 placeholder lines**.
- **RESOLVED (main corpus, 2026-07-02):** the user ran `normalize_cache --apply` + re-ingest on this box's
  `data/` corpus. **Verified: cache `grep "intentionally omitted"` = 0** (62 docs), and the R3 keyword
  dry-run over the re-ingested corpus shows no `intentionally omitted` / `x 12` / `br 1` junk tokens. The
  strip is now in `extract_to_markdown`, so future ingests stay clean.
- **Remaining (only when that home is next used):** re-run `normalize_cache --apply` + re-ingest on the
  **multi-domain** data home (`data_multidomain/`, not on this box) — same $0 runner; the code fix already
  applies to any future extraction there.
- **Pointer:** `tests/eval/baselines/rg001_concept_skeleton_multidomain_2026-07-01.md` finding 4;
  `data_multidomain/cache/*.md`; DEVLOG 2026-07-02 (cont.) PR-R1 entry.

## KI-15 — `epistemics.concepts_in_text` matches concept **UUIDs**, not labels — never fires on the real corpus — RESOLVED (2026-07-08, SPRINT-007)
- **Symptom:** `build_epistemics` reports **0 chunks with a claim** on the real 47-doc/357-concept
  corpus, even though `node_weights_for_epistemics` correctly computes 226 contested / 9
  superseded_trend nodes from the same skeleton. `load_doc_chunks()` returns all 6215 real chunks
  fine in isolation — the projection step itself silently drops every one of them.
- **Cause:** `epistemics.project_chunk_weights` builds `node_ids = [n.id for n in skeleton.nodes]`
  and hands that list to `concepts_in_text`, which regex-searches chunk text for each id
  **literally** (docstring: "Node ids are canonical (lowercase) keys"). That was true of the
  retired open-vocabulary `concept_graph.py` (KI-7), whose node id *was* `canonical_key(label)` —
  e.g. `"bm25"`. The curated-vocabulary `concept_skeleton.py` that replaced it (Node A, PR-A) uses
  the `Concept.id` **UUID primary key** as the node id (e.g.
  `00688507-0351-442b-b156-00521129a344` for the concept labelled "sentence encoder") — a UUID
  never occurs in chunk text, so `concepts_in_text` always returns `[]` and no chunk ever gets a
  row. G1 (SPRINT-001, 2026-07-07) re-pointed `epistemics.py` from `concept_graph` onto
  `concept_skeleton` but did not update this id-space assumption — `epistemics.py` was kept
  deliberately **unchanged** by every skeleton-side sprint since (G1/G3/G6 docs all say so
  explicitly), so nothing since G1 landed would have caught this.
- **Impact:** the **entire live answer-time marker surfacing (PR-M1)** — the `contested` /
  `superseded_trend` source chips in the desktop chat UI — has been silently dark on the real
  corpus since G1 retired `concept_graph.py`, independent of G3/G6's node-level correctness. This
  is a bigger problem than anything G3/G6 gate: the node weights are right, but nothing downstream
  of them ever reaches a chunk. Not caught earlier because no integration test drives real
  UUID-keyed nodes through `project_chunk_weights` against real chunk text — the existing
  `tests/integration/test_compute_epistemics.py` fixtures use short human-readable ids
  (`"colbert"`, `"ranking"`) that happen to also be valid substrings of their own stubbed chunk
  text, masking the id/label conflation.
- **Workaround:** none — the live markers just don't appear; nothing crashes, so this fails silent
  rather than loud.
- **FIX BUILT + real-validated (2026-07-08, SPRINT-007 — staged, not committed):**
  `concepts_in_text(text, labels_by_id: dict[str, str])` now matches on the concept's **label**,
  casefolded, via a new shared `concept_skeleton.compile_boundary_pattern` (the same alnum-
  boundary regex Node-A's own presence matcher uses — R2, not `\b`, so `epistemics.py` doesn't
  reintroduce the non-word-edge-char bug R2 already fixed once). `project_chunk_weights` passes
  `{n.id: n.label for n in skeleton.nodes}`. +4 tests (2 unit incl. a UUID-id fixture and a
  `gpt-4`/`gpt-4o` boundary case; 1 in `test_concept_skeleton.py` guarding the shared pattern
  builder against `match_presence`'s own output; 1 end-to-end integration test with a UUID-shaped
  node id). Gate green, **790 passed** (was 786).
  **Real-corpus validation** (same skeleton snapshot G6 already built, no rebuild — projection is
  free/read-only): `compute_epistemics --apply` went from **0 chunks with a claim / 0 marked** to
  **4008 chunks with a claim / 3334 marked** (of 6215 real chunks). Manual spot-check on one
  marked chunk confirmed all 6 attributed labels genuinely appear in its text — not just non-zero,
  actually correct. Full writeup: `tests/eval/baselines/epistemics_label_attribution_2026-07.md`.
- **Not done (documented follow-up, not a defect):** a live-UI smoke test that the desktop chat's
  evidence chips now render on a real answer (PR-M1's read side was never the broken part, but
  hasn't been exercised end-to-end since before this fix); parent-child chunk-store re-projection
  (already a separate documented follow-up, `docs/archive/pr-m1-epistemics-markers.md` ADR-1
  option 2).
- **Pointer:** `src/doc_assistant/knowledge/epistemics.py` (`concepts_in_text`/`project_chunk_weights`),
  `src/doc_assistant/knowledge/concept_skeleton.py` (`compile_boundary_pattern`); found while hand-auditing
  G6's before/after split on the real `data/library.db`
  (`tests/eval/baselines/superseded_year_rule_2026-07.md` G6 addendum);
  `docs/sprints/SPRINT-007-fix-epistemics-label-attribution.md`.

## KI-16 — vendored `docs_check` scans Claude Code background-task worktrees under `.claude/worktrees/` → ~70 phantom errors — RESOLVED (2026-07-16, cpc 1.2.2 re-vendor)
- **Symptom:** `python tools/conventions/rungate.py docs_check --root . --strict` reports dozens of
  `[header] missing status:` ERRORs for paths under `.claude/worktrees/<name>/…` (the worktree's
  README/CLAUDE.md/tests-eval copies **plus its whole `.venv` site-packages**) whenever a Claude Code
  background task has an active worktree there. First seen 2026-07-16 (70 errors, worktree
  `peaceful-blackburn-89f131`); the repo's own docs were clean. Reproduces on cpc 1.1.0 **and** 1.2.1.
- **Cause:** the rule-1 status-header scan does `claude.rglob("*.md")` over all of `.claude/` with no
  exclusion for embedded git worktrees or `.venv` (cpc 1.2.1 added a `.venv`/`node_modules`/`.git`
  parts-exclusion to rules 3–4 only, not rule 1). No `conventions.toml` workaround exists:
  `[headers] exempt` matches via `Path.match()`, which is right-anchored and cannot left-anchor a
  recursive `*.claude/worktrees/**` glob (confirmed on Python 3.14).
- **Workaround:** treat `worktrees/`-path errors as noise (filter with
  `… docs_check … 2>&1 | grep -v "worktrees/"`), or run the gate when no background-task worktree
  exists (`git worktree list` to check; they are auto-cleaned when unchanged — do NOT delete an
  active one, it belongs to a running task).
- **Fix (upstream, cpc repo):** extend the rule-1 `md_files` scan with the same parts-exclusion as
  rules 3–4, plus skip any directory carrying a `.git` *file* (the nested-worktree marker). One-line
  class of fix in `src/cpc/docs_check.py`; belongs in cpc 1.2.x.
- **Pointer:** `tools/conventions/cpc/docs_check.py` (rule-1 scan vs the rule-3/4 exclusions);
  found during the 2026-07-16 cpc 1.1.0→1.2.1 re-vendor (DEVLOG entry same date).
- **Resolution (2026-07-16, same day):** fixed upstream in cpc — new shared `in_embedded_tree()`
  skips `.venv`/`node_modules`/`.git` parts plus any directory below root carrying its own `.git`
  (file = linked worktree/submodule, dir = nested clone) across rules 1/3/4 (rules 7 + 12 inherit).
  Shipped in cpc **v1.2.2** (fix `bda91a5`, locked by two tests; verified on this repo's live
  worktree repro 70 → 0); re-vendored here the same day. A future background-task worktree no
  longer trips the gate.

## KI-17 — stochastic gap rows outlive their concept → orphaned gaps served to the graph UI (2026-07-18, OPEN)
- **Symptom:** `load_graph_view()` serves **27** gaps against a **13**-node skeleton; **10** of them
  (all `kind="suggested_concept"`, all `determinism="stochastic"`) carry a `concept_id` that resolves
  to no node. The view's own report disagrees with the sidecar: `build_gaps --apply` printed
  "Total gaps: 15 · Rows written: 15", but the route returns 27. Surfaced by the ADR-018 rescope
  (357 → 13 graph concepts, 2026-07-18); the 10 orphans were generated on 2026-07-08 against the
  old 357-concept vocabulary.
- **Cause:** the two gap classes have different write disciplines (this is ADR-017's own finding,
  read from the other end). `gaps.py:257` **delete-and-replaces** deterministic rows, so those
  self-heal on every rebuild; `_write_stochastic_gap_rows` (`:273`) is a **status-preserving
  upsert**, which is correct for not losing a user's triage — but it has **no delete pass for rows
  whose concept left the vocabulary**. Nothing reconciles a stochastic row against the current
  vocabulary, so it is immortal. Deleting a `Concept` (or, now, excluding it) strands its gaps.
- **Impact:** PR-G2a's index badges gaps by looking the concept up, so an orphan renders no row —
  it inflates the gap *count* without being reachable. Worse for **PR-G2b**, which promotes gaps to
  a first-class destination with a per-row triage action: a row you cannot resolve to a concept is a
  row you cannot dismiss or promote.
- **Workaround:** none needed for correctness today (the orphans are invisible in the index, not
  wrong answers); read the `build_gaps` report — not `len(view.gaps)` — as the true gap count until
  fixed.
- **Candidate fix (PR-G2b territory, ADR-017 C1):** in `_write_stochastic_gap_rows`, delete
  stochastic rows whose `concept_id` is not in the current vocabulary before the upsert — a
  reconcile pass, not a blanket delete, so triage on a *live* concept still survives. Guard test:
  a stochastic gap on a concept that is then excluded (`set_graph_include(cid, False)`) →
  `build_gaps --apply` → the row is gone, while a stochastic gap on an included concept keeps its
  status. Decide alongside the C1 override sidecar, since both concern what a rebuild may destroy.

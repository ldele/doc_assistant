<!-- status: active · updated: 2026-06-24 · class: living -->

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

## KI-7 — Concept-graph LLM-extraction core + `data/graph/graph.json` are SUPERSEDED — OPEN (redesign decided, not yet built)
- **Symptom:** The shipped Feature 7 (PR 16) concept graph derives nodes from a per-document
  open-vocabulary LLM extraction; on this same-domain corpus that fragments concepts and is the
  dominant cost (36–40 LLM calls/doc; hit `budget_exhausted` over the 61-doc corpus).
- **Status:** Superseded in part by the **2026-06-18 concept-graph REDESIGN** (Decision C — user-curated
  vocabulary + deterministic skeleton from `Citation`/`DocSimilarity` + confined LLM enrichment). ADR-1
  (Louvain) and ADR-4 (composite chunk key) carry over; ADR-3 (LLM-extraction node source) + the single
  integrity tag do not. The redesign is **decided but not yet built**; Feature 7d re-founds on it.
- **Do not build on:** the current LLM-extraction graph or `data/graph/graph.json` (the on-disk file is
  also an empty environment artifact, not a quality measurement).
- **Pointer:** `docs/decisions.md` → "Feature 7 — concept-graph REDESIGN" (2026-06-18). Edge precision +
  presence recall are flagged for RIGOR_TODO before the edge model is locked.

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
- **Pointer:** `docs/specs/pr-m1-epistemics-markers.md` ADR-1 (option 2 = the re-projection upgrade).

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

## KI-10 — Frozen build's bundled `certifi` rejects corporate-MITM'd HTTPS — NEAR-RESOLVED (2026-06-24; truststore shipped, pending on-proxy confirmation)
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
  non-ASCII path → a corpus that won't reload. Verified on this box (username "Lucas Délez", the `é`).
- **Confirmed (2026-06-24, chromadb 1.5.9), path is the variable:**
  - ASCII location (`C:\Projects\…`), 1 **and** 10 files → `.bin` written, reloads fine.
  - Non-ASCII location (`C:\Users\Lucas Délez\…`), 10 files / 2455 chunks → no `.bin` → reload **fails**.
  - The Windows **8.3 short path** (`C:\Users\LUCASD~1\…`, ASCII *string*) does **NOT** help — chromadb /
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

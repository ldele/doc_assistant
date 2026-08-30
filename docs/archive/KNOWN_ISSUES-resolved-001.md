<!-- status: archived · updated: 2026-07-20 · class: append-only -->

# KNOWN ISSUES — resolved, archive 001

The **full** account of every KI closed on or before 2026-07-20, moved here verbatim from
`.claude/KNOWN_ISSUES.md` on 2026-07-20 so the working file stays about what is still open
(it had reached 738 lines, 71% of them closed). Nothing was rewritten or summarised on the way
in — symptom, cause, workaround, fix, guard test and pointers are exactly as they were written.

The working file keeps a one-line-per-issue **Resolved** table pointing here, carrying the trap
and the fix that must not be undone. Follow the KI number back into this file for the reasoning,
the rejected alternatives, and how each one was actually caught.

Same pattern as the frozen decisions monolith (ADR-022): living index in the working file,
canonical detail in `docs/archive/`.

---

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

## KI-3 — win32 `cu130` torch wheel segfaults on a GPU-less box — RESOLVED (2026-06)
- **Symptom:** Instant segfault when the CUDA (`cu130`) torch wheel ran on a machine with no usable GPU.
- **Cause:** A single lock-pinned `torch==…+cu130` for all `sys_platform == 'win32'`; `uv`'s
  `torch-backend = "auto"` is `uv pip`-only (a no-op for `uv lock`/`sync`/`run`).
- **Fix (shipped, `423cbfa`):** per-machine torch via **mutually-exclusive uv extras** — `--extra cu130`
  on a GPU box, `--extra cpu` on a CPU-only box / CI. **Rule: never `cu130` on a GPU-less box.**
- **Pointer:** `docs/specs/torch-backend-per-machine.md`.

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

## KI-22 — declared base dep `send2trash` absent from venv → `DELETE /api/library/documents` 500s; the failing tests were misread as "venv drift" — RESOLVED (2026-07-19)
- **Symptom:** `DELETE /api/library/documents/{id}` returns **HTTP 500** on every call (verified live
  against a nonexistent id, which deletes nothing). Frozen worker traceback:
  `File "src/doc_assistant/library.py", line 330, in delete_document / from send2trash import
  send2trash / ModuleNotFoundError: No module named 'send2trash'`. In the test suite the same gap
  surfaces as 6 failures in `tests/integration/test_document_delete.py`, but *not* on the delete
  assertion — on `monkeypatch.setattr("send2trash.send2trash", …)` during setup:
  `ModuleNotFoundError: No module named 'send2trash'` raised from `_pytest/monkeypatch.py`.
- **Cause:** `send2trash>=2.1.0` is a declared **base** dependency (`pyproject.toml:84`) that was
  simply not installed in this box's `.venv`. `library.delete_document` imports it **lazily inside
  the function** (`library.py:330`), so nothing fails at import/startup — the app boots clean, health
  is green, only the delete path breaks, at call time. The route
  (`apps/api/main.py:519 delete_library_document`) catches only `RuntimeError` (its 409 path), so a
  `ModuleNotFoundError` escapes as an unhandled 500.
- **The real trap (why this outlived several sessions):** the 6 failures were carried in the baton as
  *"6 pre-existing send2trash failures … venv drift, unrelated to this diff"* and left alone for
  multiple sessions. That reading was **wrong**: the suite was correctly reporting a **broken shipped
  feature**, not a test-only artifact. The cryptic monkeypatch-resolution shape of the error (a
  `ModuleNotFoundError` from deep in pytest internals, far from any `delete` assertion) is exactly
  what made "test-infra noise" look plausible. **Lesson: a red test is a claim the product is broken
  until you prove otherwise — trace the import to a real call path before labelling it environmental.**
- **Workaround:** none needed post-fix; before the fix, avoid the delete path (curation is CLI-only).
- **Real fix / Resolution (2026-07-19):** installed the declared dep into the venv
  (`uv pip install "send2trash>=2.1.0"` — the single pure-Python package, deliberately **not**
  `uv sync`, which would pull the multi-GB cu130 torch wheel; KI-3). The lazy import is uncached on
  failure, so the running server recovered without a restart: the same probe now returns **404** as
  designed, and all 6 `test_document_delete.py` tests pass → suite **1015 → 1021 passed, 0 failed**
  (the first fully-green run in several sessions). **Guard added:** new
  `tests/unit/test_declared_dependencies.py` asserts every `[project].dependencies` entry resolves via
  `importlib.metadata.version` (fails **by package name** — "declared runtime dependency 'X' is not
  installed … missing-dependency drift, not a test-infra flake"), plus a pin on the exact
  `from send2trash import send2trash` form. So the next missing base dep fails loudly and can't be
  re-misdiagnosed. The venv fix itself is per-machine (`.venv` gitignored) — the committed change is
  the guard test + these docs.
- **Not fixed (deliberate, out of scope):** the route still only catches `RuntimeError`, so a genuinely
  absent base dep would still 500 rather than degrade — acceptable, since a missing hard dependency is
  a broken install, not a runtime condition to handle gracefully; the guard test is the right layer to
  catch it. The lazy import at `library.py:330` was left as-is (it runs before the unknown-id early
  return, but moving it would only paper over a missing *required* dep).
- **Type:** implementation (environment/contract — declared dep not provisioned).
- **Severity:** degrades (one feature dead: safe-delete; app otherwise fully functional).
- **First observed:** 2026-07-19 (as failing tests; the "venv drift" misreading recurs across the
  2026-07-15 / 07-18 / 07-19 batons). Root-caused + resolved same day.
- **Related:** KI-3 (why the fix avoided `uv sync`); the safe-delete feature is ADR-014;
  `tests/integration/test_document_delete.py`; `tests/unit/test_declared_dependencies.py`.

## KI-23 — additive schema columns never land on a running install; F2 moved that onto the answer path — RESOLVED (2026-07-20)
> **Renumbered 2026-07-20 (was KI-20).** It was filed as KI-20 while F2 was being built, colliding
> with the existing KI-20 (concept curation hard-deletes vocabulary, 2026-07-19) — two open issues
> under one id. This file is living, so the heading is corrected here; the **append-only**
> `docs/DEVLOG.md` and `.claude/SESSION.md` entries of 2026-07-20 still say "KI-20" and are left
> exactly as written. If you arrived from one of those, this is the issue they mean.
- **Symptom:** on this box, `answer_records.retrieval_scope_json` (ADR-025 F2) **and**
  `concepts.graph_include` (ADR-018, added 2026-07-07) were both missing from the live
  `data/library.db` until `python -m doc_assistant.db.migrations` was run by hand on 2026-07-20.
  The `graph_include` one had been absent for ~2 weeks without anyone noticing.
- **Cause:** `_ADDITIVE_COLUMNS` is only applied by `init_db()`, which runs on **ingest** and at
  the frozen-runtime entry — **not** on API startup (`apps/api/CLAUDE.md` records this as a known
  gap: "a stale DB 500s until an ingest or a manual `python -m doc_assistant.db.migrations` runs").
  A user who ingests once and then only chats never re-runs it, so every later additive column
  silently never arrives.
- **Why F2 escalates it:** until now the additive columns fed **sidecars** (a graph flag, a
  reviewer tag), so a missing column degraded an enrichment feature. `retrieval_scope_json` is
  written by `record_answer` on the **core answer path**, so on a stale DB **every chat turn would
  fail to record** — the provenance write is `contextlib.suppress`-wrapped in human mode but not
  in the ai path.
- **Workaround:** run `uv run python -m doc_assistant.db.migrations` after pulling a change that
  adds a column (done on this box 2026-07-20; 26 existing records read back as unscoped/NULL,
  which is correct).
- **Fix (shipped 2026-07-20, user decision "migrate + log what it did"):** the API lifespan now
  calls `init_db()` before serving (`apps/api/main.py`). It is idempotent and purely additive, and
  the precedent already existed — `ingest/__init__.py:405` calls it for exactly this reason
  ("the fresh-clone footgun of having to run migrations manually"). The API still owns no logic;
  it calls a `src/` function like it calls every other one. `init_db`/`_apply_additive_columns`
  now **return the columns they added**, so the lifespan logs `schema_migrated_at_startup
  columns=[...]` at WARNING when it changes something and `schema_current` otherwise — a silent
  two-week drift like `graph_include` cannot repeat unnoticed. A migration failure is caught and
  logged, never a startup crash.
- **Guard test:** `tests/integration/test_retrieval_scope.py::
  test_api_startup_applies_pending_additive_columns` builds a genuinely stale schema (drops the
  column), starts the app, and asserts the column is back. Verified non-vacuous — it fails when
  the lifespan call is removed.
- **Still true:** the frozen-build entry (`apps/api/__main__.py`) reaches the same lifespan, so
  packaged installs are covered too. Ingest keeps its own `init_db()` call (fresh-clone path).

## KI-24 — `ingest --rebuild` silently reset the library (membership, metadata, figures cascaded away with the rows) — RESOLVED (2026-07-20)
- **Symptom:** after `python -m doc_assistant.ingest --rebuild`, every folder still exists but has
  **0 documents**. Nothing in the output said so before 2026-07-20. Reproduced live on an isolated
  data home: a hand-made folder holding 3 documents came back empty; only the demo folder
  repopulated (see below).
- **Cause:** the rebuild branch runs `session.execute(delete(DBDocument))`
  (`src/doc_assistant/ingest/__init__.py`), and `document_folders.document_id` carries
  `ON DELETE CASCADE` with `PRAGMA foreign_keys=ON` (`db/session.py`). Re-ingest mints **new**
  `Document` rows, so nothing reconnects. Folder rows survive because the cascade is on the
  *document* side — which is exactly what makes the loss invisible: the rail still lists your
  folders, they are just empty.
- **Blast radius:** ADR-025 F1 membership (hand-assigned folders) and, via F2, any saved habit of
  scoping chat to a folder — a scoped turn after a rebuild searches **nothing** and says so
  (correct, but the user won't know why). `document_tags` has the identical FK and the same
  exposure once tags ship.
- **Wider than first logged.** Auditing every FK to `documents.id` before fixing it turned up
  more than folders: `document_tags`, `document_keywords`, `citations`, `doc_similarities`,
  `document_parts`, `chunk_epistemics`, `concept_presence` and `ingestion_events` all cascaded;
  the user columns `is_archived` and `notes` were reset by the re-insert; **`document_meta`**
  (the ADR-013 metadata overrides) has *no* FK, so its rows were not deleted but **orphaned**
  against ids that no longer existed — silently inert, and accumulating. And because `figures`
  rows are keyed by document id, `figure_units()` found none during the rebuild, so the
  reindexed corpus carried **no figure chunks at all** until the (paid, VLM) describe pass was
  re-run — a silent retrieval-quality regression on top of the data loss.
- **Fix (shipped 2026-07-20): the rebuild no longer deletes the rows.** The CLI help always said
  "Wipe the vector store and re-embed everything" — the bulk delete was never the advertised
  contract. Keeping the rows means `_existing_document_id` resolves to the **same id**, so every
  association simply stays attached; nothing needs snapshotting or re-keying. Rows the rebuild
  does not reproduce are swept *after* the loop (`ingest._sweep_rebuild_rows`) with the same
  gone/stale classification `cleanup_orphans_sqlite` makes — that sweep cannot run in this branch
  itself, because it reads its candidate set from the Chroma metadata a rebuild has just deleted.
- **One deliberate behaviour change beyond restoring the invariant:** a document whose source file
  is still on disk but which produced nothing *this* run (extraction error, empty extract) is now
  **kept and reported** (`rebuild_kept_unreproduced_rows`). The bulk delete removed it
  unconditionally, so a transient extraction failure used to cost the user their folders and
  metadata for that document.
- **Guard tests:** `tests/integration/ingest/test_ingest_rebuild_preserves_library.py` (6) — ids
  stable, folder membership / `document_meta` / `is_archived` / `notes` intact, figure chunks back
  in the index, gone + stale rows still swept, present-but-unreproduced protected, and the
  ordinary (non-rebuild) orphan sweep unweakened. **Verified non-vacuous:** restoring the bulk
  delete fails exactly the three preservation tests.
- **Live proof (2026-07-20, $0):** isolated `DOC_DATA_DIR`, real embedder, real Chroma, rebuild run
  as its own process — two documents re-embedded with **identical ids**, "My reading" (2) and
  "Demo corpus" (1) intact, metadata override and notes preserved; deleting a source then
  rebuilding logged `rebuild_removing_rows gone=1` and dropped exactly that row.
- **Retires ADR-025 F3 spec M3** ("a rebuild re-fights a demo removal, the one honest exception"):
  membership is preserved now, so the demo hook sees no new rows on a rebuild and nothing is
  re-fought. Amended in the spec.
- **Still open, deliberately:** the derived sidecars (`document_keywords`, `citations`,
  `doc_similarities`, `chunk_epistemics`, `concept_presence`, `document_parts`) now survive too
  *because the ids are stable*, but they are not re-derived by the rebuild — re-run their runners
  if the chunking changed. **`document_meta`'s missing FK is now fixed too** — see ADR-026: it
  gained `ON DELETE CASCADE`, which also closes the path that was *still* orphaning overrides after
  this fix (`cleanup_orphans_sqlite`, on every incremental ingest that finds a gone or
  content-changed source).
- **Pointer:** `docs/specs/feature-corpus-folders-demo.md` M9 (where it was found while specifying
  the F3 trigger) · ADR-025 · `docs/specs/feature-corpus-folders.md` D6 (folder delete never
  touches documents — the inverse direction, which *is* safe).

## KI-30 — `--doc` means three different things across four sidecar runners; two reject a real document id — RESOLVED (2026-07-29)

**Found** by the stage profile (`tests/eval/baselines/stage_profile_2026-07-28.md`), while measuring
whether the enrichment layer can run incrementally. It largely cannot, and the flag is why.

| Runner | `--doc` matches | What it scopes |
|---|---|---|
| `extract_keywords` | document **id** | the write; the corpus TF-IDF still loads everything |
| `compute_doc_vectors` | id **or** hash prefix | the **report** only — its help says "computation is always global" |
| `extract_citations` | **`doc_hash` prefix only** | the work — but help claims "one doc_hash or id prefix" |
| `extract_doc_metadata` | **`doc_hash` prefix only** | the work — same wrong help text |
| `compute_epistemics`, `build_gaps`, `build_concept_skeleton` | no flag at all | full recompute only |

**Symptom:** `python -m scripts.extract_citations --doc <document id>` exits 1 with
`No documents matched.` — because it filters `Document.doc_hash.startswith(...)`
(`scripts/extract_citations.py:200`, `extract_doc_metadata.py:172`) while every other surface in the
app (the API, the graph, the library grid, `list_documents()`) identifies a document by **`id`**. The
help text actively misleads: it promises id support that is not implemented.

**Why it matters more than a flag bug:** `extract_citations` is the **slowest sidecar measured**
(54.1 s over 97 documents, superlinear in reference count), so it is exactly where per-document
scoping would pay — and it is the one that cannot be driven by the id a caller has. Meanwhile
`extract_keywords --doc <id>` saves **4%** (3.97 s vs 4.14 s) because scoping the write does not scope
the work (KI-18, now with a number).

**Fix:** one shared resolver — accept an id prefix **or** a hash prefix in every runner (the
`compute_doc_vectors._resolve_doc_filter` behaviour is the right one), with matching help text, and a
guard test per runner that a real `Document.id` resolves. Then, separately, make the *work* scoped
where the module allows it (citations and metadata are per-document by nature; keywords is
corpus-global by construction and should say so instead of pretending).

**Workaround until then:** pass a `doc_hash` prefix to `extract_citations` / `extract_doc_metadata`
(`select doc_hash from documents where id like '<id>%'`), and do not expect `--doc` to reduce runtime
on `extract_keywords` or `compute_doc_vectors`.


**Fix (shipped 2026-07-29).** One shared resolver — `library.documents.resolve_document_prefix`
(`DocumentRef` / `DocumentPrefixError`) — is now the single entry point behind every runner's
`--doc`. It tries an **id prefix first**, falls back to a **doc_hash prefix**, escapes `LIKE`
wildcards, and raises a CLI-ready message that names the candidates when a prefix is ambiguous. All
four runners call it; `compute_doc_vectors._resolve_doc_filter` (the previous best behaviour) was
deleted in favour of it, and the help text on each runner now states what it actually scopes —
including `extract_keywords`, which now says out loud that scoping the write does **not** scope the
corpus TF-IDF.

**Verified on the live corpus (97 docs, $0):** the exact command in the symptom above,
`extract_citations --doc <document id>`, returns the document instead of exiting 1 — **7.4 s scoped
vs 54 s whole-corpus** (most of the 7.4 s is interpreter startup). A hash prefix still resolves; an
unknown prefix still exits 1; an ambiguous prefix names the candidates.

**Guard tests — do not undo:** `tests/unit/test_doc_prefix_resolver.py` (15). The first pins the
regression (a real `Document.id` resolves); others pin id-beats-hash precedence, the ambiguous and
empty-corpus paths, and that `_` cannot act as a `LIKE` wildcard. Two wiring tests assert each
runner imports the shared resolver and that the two hash-only runners no longer contain
`doc_hash.startswith` — a correct resolver nobody calls fixes nothing.

**Deliberately left alone:** `library.pins.find_document_by_short_id` is a *third* variant (id-only,
returns `None`). Its contract is what the chat pin flow needs; unifying it would change pin
behaviour for no current gain. If a fifth surface appears, unify then.

**Still true, and separate:** the *work* is only scoped where the module allows it. Citations and
metadata are per-document by nature and now genuinely scope. `extract_keywords` and
`compute_doc_vectors` remain corpus-global by construction (KI-18) — the flag no longer pretends
otherwise, but it does not make them faster.

## KI-29 — the parent-child path never strips `<!-- page:N -->`, so markers reach the prompt, the excerpts and the embeddings — RESOLVED (2026-07-29)

**Found** while verifying the library reading view for the v0.3.0 release. The view showed
`<!-- page:1 -->` as literal text. The view is not the bug — it is an inspection view of *what the
retriever stored*, and it is reporting honestly.

**The asymmetry.** `ingest/__init__.py:170` builds baseline chunks as
`page_content=clean_chunk_text(chunk_text)` — the stripper whose docstring is *"Remove page markers
from displayed text (keep them only for metadata)"* (`ingest/chunking.py:109`). The parent-child
builder, `build_parent_child_chunks` (`ingest/chunking.py:160`), applies it to **neither** the child
`page_content` **nor** the `parent_text` metadata. Parent-child is the **default** retrieval mode
(`USE_PARENT_CHILD=true`, a locked setting), so the cleaned path is the one nothing uses at answer
time.

**Measured on the live corpus (2026-07-28), not inferred:** a real `ollama/llama3.1:8b` turn returned
10 sources and **2 of the 10 excerpts contained a page marker** — e.g.
`… _retrieves_ and _generates_ the answers, respectively. <!-- page:3 --> At run-time, DPR applies …`.
Nothing strips markers on the answer path (checked `pipeline.py`, `synthesis.py`,
`chat_controller/`), so the same string is what the **user reads in the source panel** and what the
**LLM receives as evidence**. The child text was also **embedded** with the markers in it.

**Impact, in order of seriousness:** (1) prompt noise the model must ignore, in the evidence block of
every parent-child turn; (2) an HTML comment shown to the user inside a cited passage; (3) embeddings
computed over text containing the marker. None of it is a wrong answer — it is quality debt on the
default path, which is why it survived this long.

**Fix — two options, and the cost is the deciding factor (the user's call):**
1. **Strip at retrieval** (`pipeline.py`, where the passage/parent text is assembled). No re-ingest,
   fixes the prompt and the display together. Leaves the stored/embedded text as-is, so the reading
   view keeps showing markers — consistent with its "what is stored" contract.
2. **Strip in `build_parent_child_chunks`** (both `page_content` and `parent_text`) — the correct
   place, matching the baseline path. But the 33,163 stored child chunks were **embedded with the
   markers**, so it needs a **full re-embed** for the vector side to agree. Extraction is
   content-cached, so nothing is re-extracted. Do not ship the change without the re-embed, or new
   and old documents would be embedded on different text.

   **⚠ The cost argument for option 1 has evaporated — re-cost it before deciding.** The ~17 min
   figure was measured on **CPU torch**. This box moved to the `cu130` wheel on 2026-07-29 and the
   same projection is now **~2.1 min** (3.8 ms/chunk x 33,163 —
   `tests/eval/baselines/stage_profile_2026-07-29.md`). *(Revision history: guessed ~40 min →
   measured ~17 min on CPU → measured ~2 min on GPU. Never quote one of these without its device.)*
   **Recommendation as of 2026-07-29: take option 2.** It was only ever the runner-up on cost, it is
   the correct place, and a 2-minute re-embed is a coffee break rather than a project.
   *Option 1 remains the fallback if the re-embed must not happen for an unrelated reason.*

**Do not "fix" it in `library/chunks.py` or the frontend.** The reading view's stated job is to show
what the retriever stored (`library/chunks.py` docstring, `group_children`); stripping there would
make the one honest window into the store lie about it.

**Guard test to add with either fix:** a document whose text contains a page marker → no marker in
the child `page_content`, in `parent_text`, or in a retrieved passage. The class of bug is the same as
KI-26's `_JOURNAL_HEADER`: a stripper that exists, is documented, and is simply never called on the
path that matters.

**Pointer:** `ingest/chunking.py:109` (`clean_chunk_text`) · `ingest/chunking.py:160`
(`build_parent_child_chunks`) · `ingest/__init__.py:170` (the path that does it right) ·
`extractors.py:58` (where the marker is written).

**Fix (shipped 2026-07-29) — option 2, the user's call once the cost changed.** The re-embed that
made option 1 the pragmatic choice cost ~17 min on CPU and **~2 min on the GPU**
(`tests/eval/baselines/stage_profile_2026-07-29.md`), so the correct place won.
`build_parent_child_chunks` now applies `clean_chunk_text` to **both** the child `page_content` and
the `parent_text` metadata. Stripping happens at assembly, **not** on `text` up front, so chunk
boundaries and the table-caption binding in `_table_aware_parents` see exactly the same input as
before — only the stored text changes. Page *numbers* are untouched: this path never derived them
from the chunk body.

**Measured on the live corpus, before and after** (not inferred — the whole store was scanned):

| | before | after |
|---|---:|---:|
| child `page_content` containing a marker | **3,312** of 33,163 (10.0%) | **0** |
| `parent_text` containing a marker | **16,254** of 33,163 (49.0%) | **0** |
| baseline store (the path that was always right) | 0 | 0 |
| empty chunks in either store | — | 0 |

Nearly **half** of all parent texts — the block that reaches the LLM as evidence — carried at least
one marker. The KI's original "2 of 10 excerpts" was an understatement of the exposure.

**Live end-to-end check ($0):** a real `retrieve_with_scores` turn returned 10 sources with **0**
markers in any `page_content` or `parent_text`, on the same DPR/RAG material the original failing
excerpt came from.

**It uncovered a second bug, and that is the part to remember.** The re-embed failed on
`middleton-2001.pdf` with `Expected Embeddings to be non-empty list ... in upsert`. That document is
a **scan with no text layer**: its entire cached markdown is 290 characters of 15 page markers and
nothing else. Before the fix those markers were **embedded as if they were content**; after it the
document correctly yields zero chunks — and Chroma rejects an empty upsert. `ingest` now guards both
`add_documents` calls on a non-empty list and logs `no_indexable_text`, so such a document is
recorded honestly with `chunk_count=0` and health `broken` (the KI-24 sweep keeps its library row)
instead of aborting the run. **The robustness contract cuts both ways: 0 chunks is a state to report,
not to crash on.** Rebuild after the guard: **97 added, 0 errors**.

**Guard tests — do not undo:** `tests/integration/ingest/test_page_marker_stripping.py` (11). They
assert on the *chunker's output*, never on the stripper, because the bug was always that a working
stripper was not called on the path that matters. Covered: no marker in child text, none in
`parent_text`, the prose survives (a silent truncation would also pass a marker check), no empty
chunk is stored, `child_index` stays contiguous, every child is still a substring of its cleaned
parent (the invariant a half-applied fix would break), and the marker-only scan yields zero chunks.
**Verified non-vacuous:** reverting `build_parent_child_chunks` fails 5 of them.

**Consequence for any other install:** the fix is in the chunker, so an existing corpus keeps its
markers until `python -m doc_assistant.ingest --rebuild` is run. On this box that is ~4 minutes.

## KI-31 — `get_all` silently truncated `embeddings` to one page, so document similarity was computed over 39% of the corpus — RESOLVED (2026-07-29)

**Found** 2026-07-29 while re-running `compute_doc_vectors` after the KI-29 re-embed. The report
said **"Documents in library: 37"** on a 97-document corpus. The number was wrong, not the corpus.

**The bug, in one line.** `chroma_read.get_all` concatenated pages with
`if isinstance(value, list): out[key].extend(value)` — but **chromadb returns `embeddings` as a
`numpy.ndarray`**, not a list. An ndarray failed that test and fell through to
`elif key not in out: out[key] = value`, which **keeps the first page and discards every page
after it**. So a read of the 12,786-chunk baseline store returned **5,000 embeddings and 12,786
metadatas**.

**Why nobody noticed for four days.** The caller zipped the two with
`zip(raw_embeddings, metadatas, strict=False)` — `strict=False` throws the surplus away without a
word. Downstream everything looked healthy: edges were produced, scores were plausible, the graph
rendered. It simply described 37 of 96 documents. **A silent truncation that produces well-formed
output is the worst shape a bug can take**, and both halves — the ndarray check and the
non-strict zip — had to be wrong at once for it to be invisible.

**Blast radius: exactly one caller.** `doc_vectors.load_chunk_embeddings_by_document` is the only
`get_all` call that requests `embeddings`; the other nine ask for `documents` / `metadatas` / `ids`,
which are plain lists and paged correctly all along. So the damage is confined to the
`doc_similarities` table — the related-documents panel and the similarity graph — and **nothing on
the answer path** (BM25 build, retrieval, epistemics) was affected.

**Introduced by** the KI-27 paging fix (2026-07-25), which replaced unpaged `coll.get()` calls. The
unpaged read returned everything in one call, so the ndarray never needed concatenating — the
regression arrived with the fix for a different failure and was invisible to that fix's own tests,
because the fake collection in `tests/unit/test_chroma_read.py` returned a **list for every key**.
The one type that breaks concatenation was the one type never exercised.

**Fix (2026-07-29).** `chroma_read._is_array` (duck-typed on `shape` + `__len__`, so the module
keeps no numpy import) extends the concatenation branch to numpy arrays. Independently,
`load_chunk_embeddings_by_document` now **raises** on a length mismatch and zips with
`strict=True`, so if paging ever drops rows again it fails loudly instead of quietly shrinking the
corpus.

**Measured before/after** on the live corpus: `get_all(include=["embeddings","metadatas"])`
returned **5,000 / 12,786** before and **12,786 / 12,786** after; `compute_doc_vectors` went from
**37 documents / 370 edges** to **96 / 960**. Edges recomputed and persisted.

**Guard tests — do not undo:** `tests/unit/test_chroma_read.py::FakeCollectionWithEmbeddings` plus
two tests that pin (a) embeddings are concatenated across pages, not truncated, and (b) each
embedding row stays paired with its own metadata. **Verified non-vacuous:** restoring the
`isinstance(value, list)` check fails exactly those two. **Any new fake collection must return an
ndarray for `embeddings`** — a list-returning fake cannot see this class of bug.

---

## KI-32 — the **answer path** held the whole BM25 corpus in RAM: linear, no ceiling — **RESOLVED (2026-07-30, ADR-036)**
- **Symptom:** `RAGPipeline.__init__` materialises every chunk as a `Document` and keeps the list for
  the life of the process (`pipeline.py`, `self._bm25_docs`), plus the `BM25Okapi` index over the
  same corpus. **Measured 2026-07-30 on the live corpus: 265 MB of Python heap for 33,105 chunks =
  ~8.0 KB/chunk** (process working set 1,821 MB before the reranker, 2,327 MB after). Linear in
  chunks, with no cap and no eviction. **After step 1 (below): 195 MB = ~5.9 KB/chunk.**
- **STEP 1 DONE (2026-07-30) — and it under-delivered, which is the useful part.** `parent_text` is
  deduplicated out of per-chunk metadata into a `(doc_hash, parent_index) -> text` map
  (`_split_parent_texts` + `_parent_text_for`, snapshot payload v2). Predicted ~3x from arithmetic on
  text sizes; **measured 1.36x on the heap (265 -> 195 MB) and 2.1x on disk (85.2 -> 39.9 MB)**, with
  retrieval **verified identical** on the live corpus and construction time unchanged. Attribution
  then showed why: **the chunk text is ~8% of the footprint.** Per chunk it is ~1.4 KB Chroma
  metadata dicts + strings, ~1.4 KB BM25 token strings held by the index, ~0.8 KB
  `BM25Okapi.doc_freqs` dicts, ~0.6 KB pydantic `Document` overhead, ~0.4 KB child text, ~0.4 KB
  parent text. **Never estimate this footprint from text sizes again** — measure it
  (`tests/eval/baselines/memory_and_lazy_reranker_2026-07-30.md` §3).
- **Cause:** the BM25 arm has always been built in memory at construction; ADR-025 F2 (folder-scoped
  retrieval) then made the materialised corpus **load-bearing** — a scoped turn rebuilds the arm over
  a subset of `_bm25_docs`, so "don't keep the documents" is not available without replacing F2 too.
  ADR-035 persisted the same data to disk to cut *launch time* and deliberately did not address
  memory.
- **Impact while it was open (arithmetic from one measured point — ~340 chunks/doc), post-step-1:**
  backend RAM ≈ **2 GB + ~2.0 MB per document**, so ~4.0 GB at 1,000 documents and **~22 GB at the
  10,000-document robustness contract**. Practical wall on a 16 GB box is around **5,000 documents**
  (was ~3,700 before step 1); a 32 GB box reaches ~13,000, so the contract is attainable on generous
  hardware, not typical hardware. This is the **answer path**, not the enrichment layer — KI-18/KI-19
  cover `knowledge/`, and neither mentions this.
- **Distinct from KI-27** (which fixed the unpaged *read* past SQLite's parameter ceiling): paging the
  read made construction survive; it still ends with the whole corpus resident.
- **Workaround while it was open:** none. `DOC_BM25_CACHE=0` changed launch time, not memory (that
  cache was a launch accelerator over the same in-RAM structure).
- **STEP 2 DONE (2026-07-30, ADR-036) — the ceiling is gone.** The sparse arm is an **SQLite/FTS5
  index** beside the vector store (`sparse_index.py`): chunk text + metadata in a table, a contentless
  FTS5 index over the same `keywords.tokenize` token stream, parents in their own table, `Document`s
  built for the returned rows only, and ADR-025 F2 scoping as a `WHERE doc_hash IN (...)` inside the
  ranked query instead of a subset rebuild. **Heap 195 → 21 MB with no corpus resident**; construction
  4.53 → 2.79 s; sparse query 66.6 → 27.4 ms; per turn 336 → 279 ms.
- **What the working-set probe settled:** nothing in the answer path is corpus-resident any more —
  index open **+1 MB**, sparse query **+0 MB**, *vector* query **+1 MB** (Chroma is not corpus-resident
  either; an earlier probe that seemed to show +449 MB was measuring CUDA context init). Backend RAM is
  now **~2 GB flat** (embedder 800 MB + CUDA 450 MB + reranker ~500 MB in use).
- **The cost, and it is real: retrieval changed.** FTS5's `bm25()` (k1=1.2, no IDF floor) is not
  `rank_bm25`'s (k1=1.5, epsilon=0.25) — 84% candidate overlap, 8/10 public queries return an
  identical final top-10, 2 differ by one document, recall identical at 1.0000 on every metric. **The
  public case set has a ceiling** (both arms perfect), so parity is demonstrated at the available
  resolution, not proven. `DOC_SPARSE_INDEX=0` restores the legacy arm; keep it until the A/B is
  repeated on a discriminating case set.
- **Left open, tracked in `docs/performance.md` §6:** launch still scans every chunk id to fingerprint
  the store (~0.2 s at 33k, ~20 s projected at 10k docs) and chromadb's paged read has a ~159 MB
  working-set high-water mark. Both want an ingest-side version stamp. Neither is a memory ceiling.
- **Do not undo:** the `if self._sparse is None:` guard around `_load_bm25_corpus` is the single line
  standing between this and a silent return of the 195 MB — pinned by
  `tests/unit/test_pipeline_sparse_arm.py::TestTheMemoryProperty` (verified non-vacuous). Likewise the
  OR-joined, quoted FTS5 expression (a bare term list means AND) and the scope inside the ranked query
  (scoping after `LIMIT` would silently thin a folder's results).
- **Pointer:** `docs/decisions/ADR-036-sparse-index-on-disk.md` (the decision) ·
  `tests/eval/baselines/sparse_index_2026-07-30.md` (the A/B) ·
  `tests/eval/baselines/memory_and_lazy_reranker_2026-07-30.md` §3 (the attribution) ·
  `docs/performance.md` §3 · ROADMAP rows **PF2a**/**PF3** ·
  `docs/decisions/ADR-035-bm25-index-persistence.md` (superseded in practice).

## KI-53 — the ingest record described every document as a PDF, and called a complete extraction `broken` — **FIXED 2026-08-30**

**Found by the first EPUB/HTML round trip through the UI** (2026-08-28). Both formats added,
indexed and retrieved correctly — the defect was in what the row *said about* them.

**1 · `extraction_health` called any short document `broken`.**
`health.classify_document_health` opened with `if chunk_count <= 1: score -= 100`, returning
`broken` before any other signal was considered. The two committed fixtures —
`tests/fixtures/documents/article.html` (2,143 B) and `treatise.epub` (2,870 B) — each produce one
baseline chunk, so both were filed `broken` and the document view rendered it verbatim:
**"html · broken"**. Their extraction was clean (accents survived, the table round-tripped, all
three chrome markers stripped) and they were genuinely retrievable. The rule conflated **short**
with **failed**: tuned for papers, where one chunk really does mean collapse; simply wrong for a
web article, a note or a short chapter.

**2 · `extractor_used` was hardcoded** to `config.PDF_EXTRACTOR`, so an EPUB and an HTML file both
recorded `pymupdf` — a PDF extractor that never touched them. Provenance, and false.

**The fix, and the rule it installs.** *Broken* now means the extractor failed, so every rule that
can return it states that text is missing **relative to its container**: nothing extracted at all;
a paged document whose pages produced a single chunk; a lone fragment under `SCRAP_CHARS` (200).
Where there is no container to compare against — an HTML page, an EPUB chapter — the classifier
withholds the penalty rather than inventing one. `VERBATIM_FORMATS` (`txt`, `md`) are exempt from
the fragment rule outright: reading a short file back is not a failure, it is the file.
`extractors.extractor_name(path, pdf_extractor)` now follows the same branch the dispatch does.

**Do not undo — three things.**

- **`_EXTRACTOR_NAMES` must stay a *parallel* dict, never a second field on `_EXTRACTORS`.**
  `ingest.cache.extraction_fingerprint` walks `_EXTRACTORS.values()` expecting callables, and any
  exception in that walk falls back to the whole-module fingerprint — a re-extraction of every
  cached document (this is KI-48, which cost 97 PDF caches for an EPUB-only change). The ten
  fingerprints were captured before and after this change and are byte-identical. The drift a
  parallel dict allows is bought back by `test_every_supported_format_names_its_extractor`.
- **Characters per page is the only paged signal, and it is graded (`SCRAP_CHARS` 200 → broken,
  `SPARSE_PAGE_CHARS` 500 → penalised).** Do not reintroduce a count-based companion. The two it
  replaced — a `chunks_per_page < 2` floor firing at ~2,000 characters per page, and a flat
  `chunk_count <= 3` penalty — measure the same thing in a unit that misleads: with a 1,000-char
  baseline chunk, "fewer chunks than pages" *is* "under ~1,000 characters per page", and a 3-page
  note with two full chunks was filed `broken` by exactly that reasoning.
- **Health is asserted before the extractor** in
  `tests/integration/ingest/test_ingest_record_honesty.py`. Reversed, the provenance failure masks
  the health one and half the regression stops being tested.

**Known blind spot, deliberately left.** With no page count and no container signal there is no way
to detect a *large* unpaged file that yielded almost nothing — a 500 KB JS-heavy HTML page
extracting to 900 characters reads as a short article. Passing the source file's size in would
catch it, but a bytes→chars floor has to hold across raw HTML, zip-compressed EPUB and DOCX, where
the honest ratio differs by more than an order of magnitude. Named in `health.py`'s docstring.

**Not a retrieval bug, and the check that ruled it out.** Two chat turns failed to surface the
fixtures, which looks alarming until you count the corpus: 98 real neuroscience papers against one
2 KB synthetic article. The keyword index proves the content is there (1 hit for `Delacroix`), so
that was ranking, not indexing. **Do not record it as a retrieval defect** without an experiment
that controls for corpus competition.

**One loose thread, on a single data point.** `Akesson` returns 0 keyword hits while the text holds
`Åkesson` — the tokeniser appears not to fold accents, so an accented author name may not be
findable by its ASCII spelling. One observation, not a measurement; a corpus with many accented
names would settle it.

**Guard tests:** `tests/unit/test_health.py` (7 new, 6 fail against the pre-fix code) ·
`tests/unit/test_extractors_formats.py` (2 new) ·
`tests/integration/ingest/test_ingest_record_honesty.py` (2, both halves proven pre-fix).
**Pointer:** DEVLOG 2026-08-30 (1).

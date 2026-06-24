# RG-011 first-token latency — FastAPI/SSE vs in-process, local Ollama (2026-06-24)

Measures the desktop-shell ship gate **RG-011**: does the Tauri/FastAPI boundary
(`apps/api`, HTTP/SSE over 127.0.0.1) add meaningful **first-token** latency versus the
in-process `ChatController` the Chainlit/CLI renderers call directly? This was *blocked* on
the work box — the corporate TLS-MITM proxy made the Anthropic call fail before a token
(`KI-10`), so first-token could not be timed there. Run here on the **RTX/Ollama path**
(no external TLS), as the `da30b6f` DEVLOG directed.

**The gate.** Both renderers run the **same** `ChatController.handle_message` (PR-M0). The
frozen sidecar (PR-M4) runs the **same** uvicorn server as `apps/api`, so the freeze does
not change per-token latency — only process cold-start (RG-010). RG-011 therefore reduces to:
*is the HTTP/SSE hop (request parse + `EventSourceResponse` + the threadsafe event queue +
client SSE parse) meaningfully slower than calling the controller in-process?* The in-process
path is the **control**.

**Setup**
- Box: RTX 4070 (12 GB, driver 610.47), Windows 11 (10.0.26200), Python 3.12.3.
- Generation: **local Ollama `llama3.1:8b`** (GPU, via Ollama's own runtime). No external TLS → proxy-independent, free.
- Embeddings `bge-base` + reranker `bge-reranker-base` on **CPU torch** (`2.12.0+cpu`, `cuda False`) — the venv is CPU-synced, which is also what the frozen installer ships (`cu130` segfaults on a GPU-less box, KI-3). `HF_HUB_OFFLINE=1` (warm cache).
- Corpus: public set, PC store **2455 chunks** (`/api/health` confirmed `model: ollama/llama3.1:8b`, `chunk_count: 2455` **before** any chat call — KI-4 credit guard).
- Pipeline at locked defaults (`TOP_K=10`, `CANDIDATE_K=20`, BM25 0.4 / vector 0.6, parent-child, bge-base).
- Question: `"What is retrieval-augmented generation?"`. **n=5** warm samples per path (1 discarded warm-up first), **fresh session per sample** so no history-rewrite LLM call enters the timed path (both paths time retrieve → generate → first token only).
- Tool: `scripts/measure_latency.py` — `--repeat 5` (HTTP, backend up via `just api`) and `--in-process --repeat 5` (control, backend down so the two never contend for CPU/GPU).
- Provider forced via a temporary `.env` flip to `ollama` (backed up + restored; `apps/api` has no `--provider` override and `config.load_dotenv(override=True)` makes `.env` win over shell env — KI-4).

**Results (n=5 warm samples, first-token seconds)**

| Path | median | min | max | spread | sd |
|---|---:|---:|---:|---:|---:|
| **in-process** (control — Chainlit/CLI) | 4.563 | 4.453 | 4.594 | 0.141 | 0.050 |
| **HTTP/SSE** (`apps/api`, the desktop boundary) | 4.140 | 3.360 | 4.969 | 1.609 | 0.665 |
| **Δ (HTTP − in-process), medians** | **−0.423** | — | — | — | — |

**Reading**
- **The SSE boundary adds no measurable first-token latency.** The HTTP/SSE median (4.140 s)
  sits *below* the in-process median (4.563 s); the Δ (−0.42 s) is well inside the HTTP path's
  own spread (sd 0.665 s). The hop is fixed sub-millisecond plumbing dwarfed by the ~4.5 s of
  retrieval + 8B prompt-eval + first-token decode, so "not meaningfully slower" is the only
  supportable reading — the small negative Δ is noise, **not** evidence the hop is faster.
- **Variance differs by path:** in-process is tight (sd 0.050 s, all 5 within 0.14 s); HTTP is
  looser (sd 0.665 s). The two runs are separate processes, so this most likely reflects Ollama
  GPU warm-state differences between runs rather than the hop. Medians are the robust statistic
  at n=5 (p95/p99 are not meaningful with 5 samples).

**Decision:** **RG-011 boundary risk discharged — PASS on the boundary.** The FastAPI/SSE
desktop shell does not regress first-token latency versus the in-process path, on the RTX/Ollama
path the proxy box could not measure. RG-011 **stays `blocks-ship` open** for the two pieces this
box cannot produce: the **frozen-artifact** first-token number (no `dist/` here — but the freeze
runs the identical server, so this boundary result transfers) and the paid-provider / non-proxy
first-token check. RG-010 cold-start of the frozen build is separate (≈35–40 s warm on the proxy
box, KI-9 dominates the cold first-run).

**Limitations**
1. **Source server, not the frozen sidecar.** No `dist/doc-assistant-api.exe` exists on this box; the freeze runs the same uvicorn `app`, so per-token latency is identical — only RG-010 cold-start is freeze-specific.
2. **CPU torch.** Embeddings/reranker on CPU (matches the shipped freeze, KI-3). A `cu130` GPU-embeddings run would lower the *absolute* first-token but not the Δ.
3. **Ollama provider.** The boundary conclusion is provider-independent (the hop is HTTP plumbing). Under Anthropic the absolute first-token shifts by network RTT; the SSE-vs-in-process Δ does not.
4. **Small public corpus (2455 chunks)**, not the work box's 27k corpus — affects absolute retrieval time, not the Δ.
5. **One question, n=5.** A second query would further confirm query-independence; the architectural argument (fixed plumbing) already makes the Δ query-insensitive.

**Reproduce**
```bash
# .env → LLM_PROVIDER=ollama, LLM_MODEL=llama3.1:8b  (back up first; override=True makes .env win)
HF_HUB_OFFLINE=1 uv run --no-sync uvicorn apps.api.main:app --host 127.0.0.1 --port 8001   # then wait for /api/health 200
uv run --no-sync python -m scripts.measure_latency --repeat 5 --question "What is retrieval-augmented generation?"   # HTTP
# stop the backend, then:
uv run --no-sync python -m scripts.measure_latency --in-process --repeat 5 --question "What is retrieval-augmented generation?"   # control
```

**Provenance:** commit `da30b6f` (branch `docs/desktop-shell-specs`) + `scripts/measure_latency.py` `--repeat`/`--in-process` enhancement (this change). No `data/eval.duckdb` rows — `measure_latency.py` is a latency bench, not an eval-harness run.

<!-- status: active · updated: 2026-07-30 (RAM is flat since ADR-036) · class: living -->

# Setup

Install, hardware, Docker, and the platform gotchas. For day-to-day commands see
[`usage.md`](usage.md); for the design see [`architecture.md`](architecture.md).

> **Trying it for the first time?** [`QUICKSTART.md`](QUICKSTART.md) is the 10-minute path — install,
> pick an answer engine (API key or Ollama) from inside the app, index a folder, ask something. This
> file is the reference behind it.

## Install

```bash
# Prerequisites: Python 3.12, uv
git clone <your-repo-url> doc-assistant
cd doc-assistant

# Pick ONE torch backend extra for your machine (they are mutually exclusive):
uv sync --extra cu130   # NVIDIA GPU box (CUDA): GPU-accelerated embedder + reranker
# uv sync --extra cpu   # CPU-only box (and CI): the +cu130 wheel SEGFAULTS without a usable GPU
# add `--extra dev` for the test/lint toolchain, e.g.  uv sync --extra cu130 --extra dev
# on the GPU box, prefix run commands too, e.g.        uv run --extra cu130 python -m doc_assistant.ingest

cp .env.example .env   # optional — the answer engine is also configurable in the app
```

**The `.env` step is optional.** Since ADR-034 the app configures its own answer engine:
**Settings → Getting started** takes an Anthropic API key (verified before it is saved, stored in
the data home on this machine) or points at a local Ollama server, and shows which of the two
first-run steps are still outstanding. `.env` remains the path for CLI/enrichment runs and for
anything you want pinned per checkout — and **a key in `.env` wins** over one saved in the app, with
the app naming the live source so the two cannot disagree silently. Details:
[`QUICKSTART.md`](QUICKSTART.md) §2.

**Use Python 3.12.** 3.14 is not yet supported at runtime (some native dependencies aren't
cp314-stable; see `.claude/KNOWN_ISSUES.md` KI-2).

**Why an `--extra`?** uv's `torch-backend` auto-detect is `uv pip`-only (a no-op for
`uv lock`/`uv sync`/`uv run`), and two same-OS machines can't be distinguished by a lock marker,
so the torch variant is chosen per machine by a mutually-exclusive extra (`cpu` vs `cu130`). Full
rationale: [`specs/torch-backend-per-machine.md`](specs/torch-backend-per-machine.md).

## Hardware: a GPU is strongly recommended

The embedder (`bge-base`) and the cross-encoder reranker run **locally** on every ingest and query,
so they benefit a lot from GPU acceleration. Install the matching torch extra above;
`sentence-transformers` then auto-selects the device at runtime (CUDA → MPS → CPU):

- **NVIDIA / CUDA.** `uv sync --extra cu130`. Best supported and the only path benchmarked here:
  retrieve + rerank **~300 ms** on an RTX 4070 (97 documents / 33k chunks, `CANDIDATE_K=20`),
  against ~900 ms for the same corpus on CPU. Recommended. Indicative, not a guarantee — the number
  moves with corpus size; method and full tables in
  `tests/eval/baselines/stage_profile_2026-07-29.md`.
- **Apple Silicon (M-series).** PyTorch's MPS (Metal) backend is auto-detected, so the embedder and
  reranker use the Mac's GPU with no config change. Faster than CPU, though generally slower than a
  discrete CUDA card and **not benchmarked here** (MPS also occasionally falls back to CPU for
  unsupported ops).
- **CPU-only.** `uv sync --extra cpu` (Linux and CI too). Works, so a GPU isn't required, but the
  same step is seconds per query, and re-embedding a corpus (ingest / `--rebuild` / the chunking
  sweep) is dramatically slower.

The chat LLM is separate (Claude API or local Ollama), so the above is about the local embedder and
reranker, not the generation model.

**How much RAM and disk your corpus needs**, what each stage costs, and where the current design
stops scaling: [`performance.md`](performance.md). Short version for planning, measured at 97
documents: about **6 MB of disk per document**, and backend RAM of about **2 GB flat** — both search
indexes live on disk, so RAM no longer grows with the corpus (ADR-036). What does grow is the first
ingest, roughly 15 seconds per document of PDF extraction, and disk.

## Running the LLM locally

Two workloads can run on your machine: the **retrieval models** always do; the **chat LLM** does too
*if* you choose [Ollama](https://ollama.com) instead of the Claude API. With the default API path,
only the retrieval models run on your hardware.

Ollama needs no key and no configuration file: install it, `ollama pull llama3.1:8b`, then pick the
model in **Settings → Getting started → Ollama**. The panel probes the server
(`OLLAMA_HOST`, default `http://localhost:11434`) and lists what is installed, so "not running" and
"no models installed" are told apart — they have different fixes.

Provenote's local default is an 8B model (e.g. `llama3.1:8b`, 4-bit quantized):

| Local LLM (Ollama) | Minimum: runs, slow | Recommended: smooth |
|---|---|---|
| Compute | x86-64 CPU only | NVIDIA GPU (CUDA, compute capability ≥ 5.0), AMD ROCm, or Apple Silicon (Metal) |
| VRAM | n/a (CPU inference) | ≥ 8 GB (an 8B 4-bit model ≈ 5-6 GB) |
| System RAM | 8 GB | 16 GB |
| Free disk | ~6 GB (8B weights) | ~10 GB+ |

Bigger models scale up. Ollama's rough rule of thumb is **~8 GB RAM per 7-8B, 16 GB per 13B, 32 GB
per 33B** (a quantized model that fits entirely in VRAM runs far faster than one spilling to
RAM/CPU). For the authoritative, current GPU list see **<https://docs.ollama.com/gpu>**.

**Both at once?** On a single 12 GB consumer card (the RTX 4070 used here), the embedder + reranker
(~1.5 GB) and an 8B 4-bit model (~6 GB) coexist comfortably, so the whole pipeline, retrieval *and*
generation, runs on one GPU with no cloud calls.

> **Thinking models.** Hybrid-reasoning models (e.g. `qwen3.5:9b`) generate their reasoning before
> any answer token. The one-shot JSON paths disable it (`llm.OllamaClient(reasoning=False)`); on the
> streaming chat path it costs roughly 5x time-to-first-token with no change to the answer text.

## Docker

```bash
cp .env.example .env             # fill in your API key
mkdir -p data/sources && cp ~/your-papers/*.pdf data/sources/

docker compose build
docker compose run --rm doc-assistant python -m doc_assistant.ingest
docker compose up
```

The container serves the **headless FastAPI backend** on `http://localhost:8001` (check
`/api/health`), the same backend the desktop app's sidecar bundles. The GUI is the Tauri desktop
app, which runs on the host, not in the container; Docker is for running the API/server.

For local LLM via Ollama, set `LLM_MODE=local` in `.env`. On Linux, ensure Ollama listens on all
interfaces: `OLLAMA_HOST=0.0.0.0 ollama serve`.

```bash
docker compose down            # stop, keep data
docker compose down -v         # stop, delete model cache volume
```

## Windows troubleshooting: SSL crash on a `uv`-managed Python

On some Windows machines the app dies instantly with no traceback
(`OPENSSL_Uplink(...): no OPENSSL_Applink`) the first time it opens an HTTPS connection: a Claude
API call, Ollama, or any networked test. The cause is the OpenSSL in uv's bundled
(python-build-standalone) interpreter; an official CPython is unaffected. Fix by rebuilding the venv
on an official Python 3.12:

```bash
py install 3.12                                                   # official python.org build
uv venv --clear --python "$(py -3.12 -c 'import sys;print(sys.executable)')"
uv sync --all-extras
```

Behind a TLS-inspecting proxy, prefix uv commands with `UV_NATIVE_TLS=1` so uv trusts the Windows
certificate store. Offline work (ingest, embeddings, retrieval) is unaffected either way.

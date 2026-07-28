<!-- status: active · updated: 2026-07-28 · class: living -->

# Quickstart — first run in about 10 minutes

For someone trying Provenote for the first time. Two decisions, in this order:

1. **Who writes the answers** — the Claude API with your own key, or a local model via Ollama.
2. **Which documents** — a folder of your own PDFs/EPUBs/HTML/DOCX/Markdown, or the demo corpus.

Everything else is already decided for you. Retrieval (the search that finds the passages) always
runs on your machine either way, and nothing is uploaded anywhere except the question and the
retrieved passages, and only if you choose the API path.

If you get stuck, the app tells you what is missing: **Settings → Getting started** shows a
checklist with the exact next action, and the chat screen shows the same list until both steps are
done.

---

## 1. Install

Prerequisites: **Python 3.12** (not 3.13/3.14 — some native dependencies are not ready) and
[uv](https://docs.astral.sh/uv/). Plus **Node 20+** while the app runs from source.

```bash
git clone <your-repo-url> doc_assistant
cd doc_assistant
uv sync --extra cpu --extra dev
```

On an NVIDIA box, swap in `--extra cu130` instead of `--extra cpu` — it is much faster and it is
the same command otherwise. Nothing else in the install differs. (The two extras are mutually
exclusive; the CUDA wheel crashes on a machine without a usable GPU, so `cpu` is the safe default.)

The first launch downloads the local embedder and reranker, a few hundred MB, once.

## 2. Pick an answer engine

You can do this **inside the app** — no file editing — and switch later whenever you like.
Start the app first:

```bash
uv run --no-sync uvicorn apps.api.main:app --host 127.0.0.1 --port 8001
```

and in a second terminal:

```bash
cd apps/desktop && npm install && npm run dev
```

Open the URL it prints (usually <http://localhost:1420>), then click the gear icon.
**Getting started** is the first section.

### Option A — Claude API (best answers, metered)

1. Create a key at <https://console.anthropic.com/settings/keys>.
2. Paste it into **Getting started → Claude API → API key** and press **Save key**.

The app checks the key immediately with a free metadata call — no tokens, no charge — and refuses
to save one the API rejects, so a typo cannot leave you with a broken install that looks fine. The
key is stored **in plain text on this machine only**, in your data folder
(`Settings → Corpus → Data home`), and never leaves it except as an `Authorization` header to
Anthropic. Remove it any time with **Remove key**.

Costs are yours and they are small for trying it out: a question with the default `claude-haiku`
model retrieves ~10 passages and writes one answer.

### Option B — Ollama (free, fully local, nothing leaves the machine)

1. Install Ollama: <https://ollama.com/download>. It starts its own background server.
2. Pull a model:
   ```bash
   ollama pull llama3.1:8b
   ```
   About 5 GB. It runs on CPU, but expect tens of seconds per answer; with a GPU of 8 GB or more it
   is comfortable.
3. In **Getting started → Ollama**, press **Re-check**, pick the model, then **Use this model**.

The panel distinguishes the two things that can be wrong, because the fixes differ: *not running*
(install/start Ollama) versus *no models installed* (pull one).

Answer quality on an 8B local model is lower than the API — this project measures that rather than
claiming otherwise; see [`../evals/README.md`](../evals/README.md) and the README's Limitations.

> Prefer a file? Everything above also works via `.env` — `cp .env.example .env` and fill in
> `ANTHROPIC_API_KEY`, or set `LLM_PROVIDER=ollama`. **A key in `.env` wins** over one saved in the
> app, and the panel says so when that happens, so the two can never quietly disagree.

## 3. Add documents

In **Settings → Your documents**, paste the full path to a folder of documents and press
**Index folder**. Indexing runs locally: extract → chunk → embed. Budget roughly 10-30 seconds per
paper on CPU, much less on a GPU. You can keep using the app while it runs.

No documents handy? Fetch 28 open-access papers pinned by arXiv ID and checksum:

```bash
uv run python -m scripts.download_corpus --demo
uv run python -m doc_assistant.ingest
```

## 4. Ask something

Ask a question whose answer lives *inside* your documents:

- "What method did <paper> use to measure <X>, and on what dataset?"
- "Compare how <paper A> and <paper B> approach <shared topic>."

Then look at what the answer carries: numbered citations that open the exact passage, the
evidence/interpretation split, and the per-source evaluation strip. The 60-second tour of what to
look at is [`DEMO.md`](DEMO.md).

## If something is wrong

| Symptom | Cause and fix |
|---|---|
| Chat says "Two steps to get started" | Nothing is broken. Open Settings; each step names its own fix. |
| "No Ollama server answering at …" | Ollama is not installed or not running. Install it, or run `ollama serve`, then **Re-check**. |
| Ollama is running but answers fail | No model pulled, or the model name is wrong. `ollama list` shows what you have. |
| "The API rejected this key" | The key is wrong or revoked. Create a new one in the Anthropic Console. |
| "Key saved, but it could not be checked" | No network, or a proxy in the way. The key was kept; try a question. |
| The app shows a key you did not save | A key in `.env` takes precedence. Clear it there to use the one in the app. |
| Crash with no traceback mentioning `OPENSSL_Applink` | A known quirk of uv's bundled Python on some Windows boxes — [`setup.md`](setup.md#windows-troubleshooting-ssl-crash-on-a-uv-managed-python) has the fix. |
| Answers are slow on a local model | Expected on CPU. A thinking model (e.g. `qwen3.5`) also spends time reasoning before the first visible word. |

Full install detail, hardware guidance and Docker: [`setup.md`](setup.md).
Everyday commands, enrichment passes and tests: [`usage.md`](usage.md).

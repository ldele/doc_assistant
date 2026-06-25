<!-- status: active · updated: 2026-06-25 · class: living -->

# Demo — 60-second walkthrough

What to run, what to ask, and what to look at so a first-time reader sees the point quickly.

## Run it

```bash
git clone <repo> && cd doc_assistant
uv sync
cp .env.example .env          # add ANTHROPIC_API_KEY (or run fully local — uncomment the Ollama stanza in .env.example, no key needed)
# drop a few PDFs in data/sources/ (any public papers work)
uv run python -m doc_assistant.ingest      # extract -> chunk -> embed -> store (incremental after first run)
```

First run downloads the embedder + reranker (a few hundred MB) and builds the index; subsequent runs are incremental.

Then pick a UI:

```bash
# Desktop app — the shipping UI (Tauri + Svelte over a local FastAPI/SSE backend)
just api                                         # backend on 127.0.0.1:8001
cd apps/desktop && npm install && npm run dev     # dev UI in the browser (or: npx tauri dev for the native window)

# CLI — same cited answers, no GUI
uv run python apps/cli.py

# Chainlit — legacy web UI (slated for removal at M5; still works today, needs Python 3.12)
uv run --python 3.12 chainlit run apps/chainlit_app.py
```

## Ask these

Pick questions whose answers live *inside* the documents, not in the model's training:

- "What method did <paper> use to measure <X>, and on what dataset?"
- "Compare how <paper A> and <paper B> approach <shared topic>."
- "What are the stated limitations of <paper>?"
- A deliberately unanswerable one — "What does <paper> say about quantum gravity?" — to see the system decline rather than hallucinate.

## What to look at (the point)

1. **Inline citations.** Every answer cites passages with file, page, and section — click through and verify the claim against the source.
2. **The provenance card.** On a weak/flagged answer it expands: retrieved chunks, model, token cost, confidence signals. Clean answers stay quiet. `/export-record <id>` dumps the full audit JSON.
3. **The reviewer.** On a flagged answer a separate-context reviewer re-grades faithfulness / citation density / hedging. Run `/review <id>` on any past answer.
4. **The citation graph.** `/cites <doc>`, `/cited-by <doc>`, `/similar <doc>` — references resolved against your own library, plus embedding-similarity edges.

## What this is showing

The interesting part is not the chat box — it's that retrieval quality is **measured** (`uv run python -m scripts.run_eval`, results in `data/eval.duckdb`; see the README benchmark) and that every answer is **auditable**. The design rationale, with rejected alternatives, is in [`decisions.md`](decisions.md).

# Document Assistant 📚

A local-first research assistant over **your own** document library (PDF, EPUB, HTML, DOCX, Markdown). It answers questions grounded in your sources, with inline, page-level citations — not a general chatbot.

## Ask a question

Just type a question about your library, e.g.:

> *What methods does the literature use to evaluate retrieval quality?*
> *Summarise what these papers say about parent-child chunking.*

You get back a **grounded answer with inline citations** — each claim points to the passage and page it came from, so you can check it. When dual-layer synthesis is on (`SYNTHESIS_MODE=ai`), the answer separates the **evidence** (what your sources actually say) from the **AI interpretation** (clearly labelled), and flagged claims come with accept / reject / edit controls so an inference is never mistaken for a fact.

> Tip: the better your library, the better the answers. Drop documents in `data/sources/` and run `uv run python -m doc_assistant.ingest`.

## Commands

Type these instead of a question (one command per message):

**Browse the library**
- `/library` — list all documents (add `broken` / `marginal` / `healthy`, or a format like `pdf`, to filter)
- `/document <id>` — full details for one document (use the first 8 characters of its ID)

**Citations & relationships**
- `/cites <id>` — works this document cites
- `/cited-by <id>` — library documents that cite this one
- `/graph <id>` — a small citation subgraph around this document
- `/similar <id>` — the most semantically-similar documents
- `/bibtex` — render the whole library as BibTeX

**Provenance & review**
- `/records` — list recent answers (each has an auditable provenance record)
- `/export-record <id>` — export one answer's full provenance record as JSON
- `/review [id]` — run the separate-context LLM reviewer on a past answer (no id = most recent)

**Other**
- `/synthesis` — show the current answer mode (`human` evidence-only vs `ai` dual-layer)
- `/help` — list these commands

Anything that isn't a command is treated as a normal question to your library.

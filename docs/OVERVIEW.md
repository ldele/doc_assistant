<!-- status: active · updated: 2026-10-20 (first version, read off the repository at dc8a60e) · class: living -->

# Provenote — project overview

What the project is, what it has measured and where it stands, on one page. Every figure links to
the file it comes from; when a release ships or a baseline changes a figure, update it here too.
As of 2026-10-20.

## 1. What it is, and for whom

Provenote is a desktop app for asking questions of your own documents (papers and books in PDF,
EPUB, DOCX, HTML, Markdown and a few other formats). It answers in prose and cites the page each
passage came from. It is meant for one person and their own library: a researcher, student or
analyst who has to check where a statement comes from before reusing it.

A personal project of Lucas Delez, started in May 2026 (first commit 2026-05-08), public under
Apache-2.0. The Python package is called `doc_assistant`; the desktop app ships as Provenote with
the same version number.

## 2. What sets it apart

1. **A citation opens the printed page.** Since 0.6.0, *Show the page* on a citation opens the PDF
   page the passage came from, rendered as printed, in a pane beside the document's library entry.
   *In context* shows the passage highlighted inside the surrounding text, with how far through the
   document it sits. This works for PDFs; a format without pages says so.
   Source: [CHANGELOG 0.6.0](../CHANGELOG.md), [ADR-050](decisions/ADR-050-source-viewer-page-rendering.md).
2. **What the sources say is kept apart from what the model infers.** Each answer has two layers:
   the evidence (the retrieved passages, shown verbatim with source, page and relevance score) and
   the interpretation (the model's synthesis, labelled as such, with inline citations). Each claim
   gets a marker from the reranker's score for the passage it cites: unmarked, *weakly grounded*
   or *unsupported* (cites nothing). The user can accept, reject or edit a flagged claim, and the
   decision is logged with the answer's provenance record. A separate reviewer model can grade a
   flagged answer against the retrieved text, or any answer on demand (`/review`).
   The markers measure relevance, not truth: an unmarked claim is not verified.
   Source: [how answers work](how-answers-work.md).
3. **A local model keeps the documents on the computer.** With a local model through Ollama, no
   text leaves the machine; the embedder and reranker run locally and are bundled in the installer,
   which runs offline. With the Claude API, the question and the passages retrieved for it are sent
   to write the answer. The cost of the local choice is measured (§4, citation coverage) and stated
   in the app where the engine is chosen. Source: [README](../README.md).

## 3. How it works

- **Ingest.** Documents are extracted to page-marked markdown (cached, so a document is read once),
  split into parent-child chunks (a 400-character child is what retrieval finds, its 2,000-character
  parent is what the model reads), embedded locally with `bge-base-en-v1.5`, and stored in Chroma.
  Document records live in SQLite; the keyword index is SQLite FTS5 on disk
  ([ADR-036](decisions/ADR-036-sparse-index-on-disk.md)).
- **Query.** Hybrid retrieval: BM25 (weight 0.4) and vector search (0.6), 20 candidates per
  retriever; a local cross-encoder (`bge-reranker-base`) re-scores the pool and keeps the top 10
  passages; the model (Claude API or an Ollama model) writes the answer with inline citations.
- **App.** Tauri + Svelte 5 desktop app over a FastAPI/SSE backend on localhost, plus a CLI.
  Windows installer (~1.6 GB with the models), not code-signed. Tested mainly on Windows, CI on
  Linux; macOS paths work but are unbenchmarked.
- **Around the answers.** A map of the library the user curates: concepts, which documents hold
  them, single-source concepts flagged as gaps, definitions quoted from the user's documents with
  their page (much of this is the 0.7.0 work, §5).
- **Evaluation harness.** Its own module (`src/doc_assistant/eval/`), with six scorers written for
  the project rather than taken from a framework: `exact_match`, `contains_all`,
  `citation_overlap`, `figure_retrieval`, `embedding_similarity` and `llm_judge` (Claude Haiku 4.5,
  temperature 0, pinned and recorded per run; it sees the question, the reference answer and the
  candidate, not the passages). Runs are stored in DuckDB; readable baselines are committed; a
  `compare_runs` command refuses to compare two runs whose settings, corpus or generator differ.
  Sources: [`scorers.py`](../src/doc_assistant/eval/scorers.py),
  [`TESTING.md`](../tests/eval/TESTING.md), [`evals/`](../evals/README.md).

## 4. Measured results, each with its size and conditions

Two question sets, both small and hand-written: **10 public cases** on 10 arXiv papers pinned by
SHA-256 (every case's reference answer checked against the paper), and **35 private cases** on the
author's own 97-document library (mostly copyrighted, so not citable by others; most references
not author-verified). The two sets are never compared with each other. **The judge has not been
calibrated against human grades.** Baselines are in [`tests/eval/baselines/`](../tests/eval/baselines/).

| What was measured | Result | Size and conditions | Source |
|---|---|---|---|
| Public benchmark | `citation_overlap` 1.000 ± 0.000 · `contains_all` 0.932 ± 0.014 · `llm_judge` 3.694 / 5 ± 0.258 | 10 cases, 5 trials, Haiku 4.5 generator and judge, 2026-08-01. `citation_overlap` is saturated on 10 papers: it shows no regression, it cannot rank. The judge band means only changes above ~±0.5 would show. 49 of 50 judge calls scored. | [public_eval_2026-08-01](../tests/eval/baselines/public_eval_2026-08-01.md) |
| Embedder | `bge-base` over `specter2`: 1.000 vs 0.900 · 0.927 vs 0.800 · 3.738 vs 3.447 | public 10, 5 trials, 2026-06-04 | [bge_vs_specter2](../tests/eval/baselines/bge_vs_specter2_public_2026-06-04.md) |
| Chunk sizes | Six parent/child geometries swept; **nothing beat the control (2000/200 · 400/50) beyond its variance, so it was kept.** On the private set `citation_overlap` spans 0.877–0.946: a smaller child (256/32) retrieves best (0.946 vs 0.936) but answers worse through the local model (`contains_all` 0.734–0.740 vs 0.777). One small-child setting used 45% fewer input tokens. | private 35 on 97 documents, `llama3.1:8b`, 3 trials; and public 10 on Haiku; 2026-08-08. Open: the small-child penalty may come from the weak local generator; a re-run on a strong one would settle it (not done, cost). | [private](../tests/eval/baselines/chunking_sweep_private_2026-08-08.md), [public](../tests/eval/baselines/chunking_sweep_public_2026-08-08.md) |
| BM25 / vector mix | **Post-rerank recall flat across weights 0.0–1.0; 0.4/0.6 kept.** The reranker re-scores the whole pool from both arms, so the weight cannot change the final top 10. Pre-rerank recall did move (0.936 → 0.882 at @5), so the instrument works. | 35 private cases, retrieval only, no generation, 2026-07-03 | [bm25_weight_sweep](../tests/eval/baselines/bm25_weight_sweep_2026-07-03.md) |
| Keyword index moved to disk | Post-rerank recall identical to the old in-memory index (0.701 @5, 0.760 @10), but 9 of 35 queries got a different set of passages. Re-running one arm changed 1 of 35 (retrieval is not fully deterministic, ~3% case-level noise). | private 35, 2026-08-01 | [sparse_arm_private35](../tests/eval/baselines/sparse_arm_private35_2026-08-01.md) |
| Citation coverage by model | Share of answer sentences carrying a citation, **shipped prompt: Claude Haiku 4.5 83.5%, `llama3.1:8b` 37.6%, `qwen2.5:7b` 18.0%.** Answers citing nothing at all: 0, 12 and 18 of 27. The earlier prompt gave 81.2 / 36.4 / 13.5%: the prompt change fixed a citation-format failure and did not move coverage. | 27 private cases (8 whose source document is degraded or absent excluded), 97 documents, **one run each**, 2026-08-07; indicative to about ±2 points (Haiku, llama) and ±5 (qwen) | [citation_coverage](../tests/eval/baselines/citation_coverage_2026-08-07.md) |
| Candidate pool | 20 candidates per retriever vs 10: a tie | public 10, 2026-06-13 | [candidate_k](../tests/eval/baselines/candidate_k_public_2026-06-13.md) |
| `TOP_K` = 10, parent-child on | Peak at K = 10 in a sweep over 3–15; parent-child ahead of single chunks | **measured before the harness existed:** point estimates, case set, N and variance unrecorded; the direction holds, the size does not | [decisions archive](archive/decisions-monolith.md) |
| Reranker | **No experiment.** Adopted on a rationale (slower per item, but sees few candidates). | — | [decisions archive](archive/decisions-monolith.md) |
| Indexing cost | ~15 s for a 30-page text PDF, ~90% of it PDF extraction; 2 extraction workers 1.47× faster than 1; backend memory flat at ~2 GB | 97-document library (2,859 pages), 28-core desktop with GPU; validated at ~100 documents, not thousands | [README](../README.md), [performance](performance.md) |
| Knowledge layer | A definition graded *strong* was right 8 times in 13 (95% interval 36–82%); 97% of extracted keywords occur in a single document, which is why the map is built from concepts the user picks | development library, 2026-10 | [README](../README.md), [knowledge layer](knowledge-layer.md) |

What the record does **not** have: a judge checked against human grades; retrieval scored by rank
(it is scored by presence, no MRR or nDCG); any significance test (10 cases discriminate between
settings, they do not prove a ranking); a measured effect of the reranker; a strong-generator run
of the chunk-size grid on the private set.

## 5. State

- **Released:** v0.6.0, tagged 2026-09-04, published 2026-09-07 with a Windows installer that was
  installed and driven on a clean machine before publishing. Earlier: 0.4.1 (first working
  installer, 2026-08-06), 0.5.0, 0.5.1.
- **In preparation: v0.7.0**, no date set. In the [CHANGELOG](../CHANGELOG.md)'s *Unreleased*
  and the README's Status: concepts the user takes on, with forms marked exact or broad;
  definitions chosen from sentences quoted from the user's documents, with page; concept merges
  that keep the curation and can be undone; signals relabelled to say what they count; the
  knowledge layer reads prose, not figure captions; security hardening (size and zip-bomb limits,
  sanitised answers, the local API refuses other hosts and needs the app's launch token for
  writes, CI fails on unreviewed advisories or committed secrets).
- **Next, per the [ROADMAP](ROADMAP.md):** the knowledge layer reads past the reference list
  (row 95), then suggestions of what to read next for a subject (KL2), then LLM-assisted ingestion
  (row 25).
- **Not done:** OCR for scanned PDFs (designed in
  [ADR-039](decisions/ADR-039-ocr-sidecar-for-scanned-pdfs.md), not built; a scan without a text
  layer yields nothing unless `tesseract` happens to be installed); the per-source *contested* /
  *corroborated* chips are off by default and will be rebuilt only behind a hand-labelled study
  ([ADR-041](decisions/ADR-041-node-b-stance-rebuild-or-retire.md)); a local judge gated on
  agreement with the reference judge (planned, not built); scale beyond ~100 documents; an OS
  keychain for the API key (stored in plain text today); multi-user serving; code signing; macOS
  benchmarks.

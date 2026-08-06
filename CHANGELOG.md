# Changelog

Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is [SemVer](https://semver.org/) on the `doc_assistant` package, and the desktop app
(Provenote) ships the same number.

The engineering record is finer-grained than this file: per-change entries live in
[`docs/DEVLOG.md`](docs/DEVLOG.md), design decisions in [`docs/decisions.md`](docs/decisions.md).

## [0.4.1] — 2026-08-06

**The first release with a working installer, verified on a clean machine.** 0.4.0 shipped as
source only, because the bundled installer was still a June build of a much older app. This one
carries a freshly frozen backend, a real Windows installer that has been installed and driven
end-to-end on a machine with no Python and no toolchain, and it withdraws a feature that was
telling you something untrue.

### Added

- **A Windows installer that matches the code, and is proven to work.** The bundled backend was
  re-frozen from this release (it had been stuck at a 2026-06-24 build, pre-rename and
  pre-first-run-setup). **Installed on a clean, Python-free Windows machine and driven through a
  real question:** install 177 s → backend healthy in ~30 s → three PDFs indexed into 322 passages
  → a cited answer over ten sources in 14 s, entirely on a local model at no cost.
- **The answer engine tells you what choosing it costs.** The local (Ollama) option now states,
  where you pick it, that local models cite noticeably less of what they write: across 27 questions
  on a 97-document library — same prompt, same retrieval — `llama3.1:8b` carried inline citations on
  36% of its sentences and `qwen2.5:7b` on 14%, against 81% for Claude Haiku 4.5. Answers stay
  grounded either way; more claims simply show as *uncited*. Nothing is gated — it is your call,
  made with the number in front of you.

### Changed

- **Per-source "epistemic assessment" is withheld.** The chips that labelled each source
  *contested* / *corroborated* / *single-source*, and the answer-layer marker chips, are off by
  default. They were derived from a stance pass that is **judged without ever seeing the document
  text**, has no "neutral" option, and whose verdict changes with where a concept pair happens to
  land in a generated list. Measured: one document, identical inputs, position varied alone → four
  different verdicts, crossing the supporting/opposing line; 53.3% of assessed passages carried a
  marker. Nothing was deleted — `EPISTEMICS_MARKERS_ENABLED=true` opts back in — and the rebuild is
  planned. The rest of the source-evaluation strip (document year, relevance score, graph
  freshness) is unaffected and still shown.
- **Claims under an answer say what was actually checked.** A flagged claim used to read
  *unsupported*, which sounded like a verdict on whether it was true. It never was: the check is
  structural — does this sentence carry a citation that points at a real retrieved passage? So the
  labels now say that. A sentence with no citation reads **uncited**; one citing a source number
  that does not exist reads **unresolved citation**; *weakly grounded* is unchanged. This also ends
  a genuine contradiction — the reviewer's own "unsupported claims: 0" could appear directly above
  a list of claims labelled "unsupported", because the two meant different things. And when the
  assistant correctly says your sources do not cover something, that is now reported as uncited
  rather than as an accusation.
- **The relevance number is labelled.** Each source row carries a score that is *retrieval
  relevance* — how well that passage matches your question — and it was easy to read as a judgement
  of the source's quality. Nothing in this app scores source quality. The column now says so, and
  the same numbers are no longer printed twice on one answer.

### Fixed

- **Continuous integration was red on `main` since the 0.4.0 release commit.** The release bumped
  five version strings and missed `uv.lock`, which records the project's own version; both CI and
  the Docker image install with `--locked`, which fails rather than silently re-resolving. Every
  gate after dependency install — lint, types, tests, security — had been skipped since.
- **The Docker build could never have succeeded** — `.dockerignore` excluded `README.md`, which the
  Dockerfile copies and which the package build needs. (The image itself is still unbuilt on this
  machine; the fix is reasoned, not yet exercised.)
- **The installer could not read a single PDF.** The frozen backend bundled PyMuPDF's legacy import
  shim instead of the package that carries its data files, so every PDF failed instantly on a real
  install while working perfectly from source. Found only by installing on a clean machine — which
  is now part of the release routine rather than an afterthought.
- **Scrollbars, and an answer column that drew underneath its own scrollbar.** The app shipped no
  scrollbar styling at all, so every scrolling pane drew the raw operating-system bar against the
  reading surface. Separately, the answer column had no horizontal breathing room, so anything
  aligned to its right edge — the token count, the per-source relevance score — was rendered under
  the scrollbar and cut off.

### Known limits

- **Local models cite much less than a hosted one.** Measured above, and now stated in the app where
  you choose your answer engine. The answers remain grounded in your documents; more of their
  sentences simply carry no citation, and are marked accordingly.
- The `contested` saturation described above is diagnosed but not repaired; the concept graph's
  corroboration and gap detection are unaffected by it.
- **Scanned documents with no text layer stay unreachable.** A PDF that is images-only extracts to
  nothing and cannot be retrieved. On the development library this affects 4 documents of 97, and
  they account for *every* retrieval miss on the private evaluation set. Recovery by OCR is designed
  and planned, not built.

## [0.4.0] — 2026-08-01

**The release that makes the library size stop mattering.** Memory used to grow with your corpus —
about 2 MB per document on top of a ~2 GB floor, which put a practical ceiling around 5,000
documents on a 16 GB machine. It is now **flat**: the keyword index lives on disk, so a library of
100 documents and a library of 10,000 cost the same RAM. Getting there changed how retrieval ranks,
so the change was gated on measurement rather than asserted.

**This is a source release.** The installer bundled in this repository is still the June build and
does **not** carry any of this — build from source (see [`docs/QUICKSTART.md`](docs/QUICKSTART.md)).

### Added

- **Settings → Corpus.** What your library actually costs on this machine: documents, chunks, disk
  by artifact and per document, and which keyword index is serving — plus a **Rebuild** action for
  the keyword index (derived data, seconds here, no confirmation needed).
- **The app says when keyword search is unavailable.** If the keyword index cannot be built,
  answers still come back on meaning-based search alone, and the Corpus panel says so rather than
  degrading silently. Rebuild is the fix.

### Changed

- **Backend memory no longer grows with your library** — the keyword index moved from RAM to an
  on-disk SQLite/FTS5 database. Measured on a 97-document / 33,105-chunk corpus: **195 MB → 21 MB**
  of Python heap with no corpus resident, pipeline construction 4.53 s → 2.79 s, and ~57 ms off
  every turn.
- **Retrieval ranking changed** as a consequence — SQLite's BM25 is not the previous library's
  (different k1, no IDF floor). Measured on a 35-question set over the full corpus: **post-rerank
  recall is identical**, and where anything moved at all it moved in the new index's favour.
  Roughly a quarter of questions return a different mix of supporting sources at the same recall.
- **Faster launch, slower first question** — the reranker now loads on first use rather than at
  startup: about 4.4 s off launch, about 5 s onto the first question of a session.
- On an NVIDIA GPU, a query is about **3× faster** (907 ms → 296 ms) and re-embedding a corpus about
  8× faster.

### Fixed

- **Page markers no longer reach your answers.** `<!-- page:N -->` was leaking into the text the
  model reads and the excerpts you see — 10% of chunks and **49%** of the larger passages.
  **Existing installs need `ingest --rebuild` to clear it**; new installs are unaffected.
- **Document similarity was running on 37 of 96 documents.** A paging bug silently dropped
  everything after the first page. Related-document suggestions were correspondingly thin.
- Enrichment tools accept a document id, not just an internal hash — scoping a run to one document
  works, and is ~7× faster than the whole corpus.
- A scanned PDF with no extractable text no longer aborts an ingest run; it is recorded honestly as
  having no indexable content.
- The concept graph printed each document's authors twice.

### Removed

- The `DOC_SPARSE_INDEX` and `DOC_BM25_CACHE` environment switches. The old in-memory keyword index
  they selected is gone, so there is one retrieval path instead of two.

### Known limits

- **The "contested" label has saturated.** It fires when a single other document disputes a claim,
  so on a 97-document corpus **53% of assessed passages** carry it while under 4% read
  "corroborated". Treat the source-evaluation strip as a prompt to look, not as a verdict.
- **A PDF with no text layer is unreachable.** Two of 97 documents here; one holds zero chunks and
  can never be retrieved. Recovery by OCR is designed but not built.
- **Retrieval is not perfectly reproducible** — about 3% of questions return a different source
  ordering between identical runs, from tie-breaking in the reranker. Aggregate quality is stable.
- Validated at ~100 documents. Everything said about 1,000 or 10,000 is arithmetic from one corpus;
  no scale run has been performed, and the first ingest of a large library is now the slow part.
- Carried from 0.3.0: the in-app key is plaintext in the data home; small local models place
  documents into a taxonomy at 70–87% precision with near-meaningless self-reported confidence;
  tested primarily on Windows, with Linux CI.

## [0.3.0] — 2026-07-28

**The first release meant for someone other than its author.** 0.1.x/0.2.x were development
versions; the change that makes this one shippable is that setup no longer requires editing files or
reading the source.

### Added

- **First-run setup in the app** (ADR-034). `Settings → Getting started` shows what the install
  still needs — an answer engine, some documents — with the exact next action for each, and the chat
  screen carries the same checklist until both are done.
- **Bring your own key, in-app.** Paste an Anthropic API key in Settings. It is verified first with a
  free metadata call (no tokens, no charge), refused if the API rejects it, and stored on your
  machine in the data home. `Remove key` forgets it. A key in `.env` still wins and the panel says
  when it does, so the app and the CLI can never send different keys.
- **Honest Ollama detection.** The app probes the local server, lists the models it actually has, and
  distinguishes *not running* from *no models installed* — two states with different fixes. Picking a
  model applies to the next question, with no restart.
- **[`docs/QUICKSTART.md`](docs/QUICKSTART.md)** — the 10-minute first run for a new user, both
  provider paths, with a symptom/cause table.
- `library.count_documents()` — a `COUNT` for the setup view rather than materializing every
  document summary to discard it.

### Fixed

- **A key set at runtime could not have worked.** `pipeline.build_chat_model` resolved the API key
  through an import-time binding, so a key that arrived later was never sent. Every Anthropic call
  site (chat model, one-shot client, figure VLM, CLI cost guard) now resolves per construction.
- **The provider picker claimed Ollama was available on machines that had never installed it**
  (`provider_available("ollama")` was unconditionally true).
- **The empty chat screen named only half the problem** — it said "no documents indexed" whether or
  not a working provider existed.
- Four planning ADRs (030–033) had files but no rows in the decisions index.

### Changed

- The provider/model section in Settings is now the advanced form of the same switch, and no longer
  tells you to add a key to `.env`.
- `.env` is documented as optional for running the app, and required only for CLI/enrichment runs.

### Known limits (unchanged, and stated in the README)

- The in-app key is **plaintext in the data home** — weaker than an OS keychain, which ADR-034
  records as the upgrade path and why it was not taken in this release.
- Validated at ~100 documents; the enrichment layer has corpus-linear paths catalogued with a fix
  plan ([scale review](docs/REVIEW_2026-07-19_scale-robustness.md)).
- Small local models place documents into a taxonomy at 70–87% precision and their self-reported
  confidence carries almost no signal.
- Tested primarily on Windows, plus CI on Linux; macOS (MPS) paths work but are unbenchmarked.

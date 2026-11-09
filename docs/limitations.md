<!-- status: active · updated: 2026-11-09 (moved verbatim from the README's Limitations section, links re-based) · class: living -->

# Limitations

The full list behind the short one in the [README](../README.md#limitations).

Each item is a limit you can hit in this release, with what to do about it. Counts come from the
development library on 2026-10-18 (104 documents, 3,341 pages) unless a date says otherwise.

## Answers

- **Local models cite less, and the gap is measured.** Across 27 questions on a 97-document
  library, with the shipped prompt and the same retrieval, Claude Haiku 4.5 carried inline
  citations on 83.5% of its sentences, `llama3.1:8b` on 37.6% and `qwen2.5:7b` on 18.0% (one run
  each, so read them to about ±2 points, ±5 for qwen;
  [baseline](../tests/eval/baselines/citation_coverage_2026-08-07.md)). On a local model more claims
  carry no citation and show as *uncited*: 12 of the 27 `llama3.1:8b` answers cited nothing.
  **These local figures predate a fix in 0.7.0** that stopped Ollama from silently cutting the
  start of long prompts, where the instructions to cite sit; on the release's three test questions
  `llama3.1:8b` went from 0 cited answers to 3. They will be re-measured. Small
  local models also place documents into a
  taxonomy at 70-87% precision, and their self-reported confidence carries almost no signal; on one
  model it was *anti*-correlated with correctness, so do not auto-accept on it. The app states this
  where you choose the engine, and blocks nothing.
- **A few pages hold the previous page's text.** On about 0.2% of pages (a 25-document sample,
  2026-09-01) the PDF reader stores the page before's text in place of the page's own, so that page
  cannot be found by search and a citation to the copy opens one page off; one 1949 book has 13
  such pages in 355.
- **Per-source "epistemic assessment" is off by default.** The chips labelling a source *contested*
  / *corroborated* / *single-source* are withheld: they came from a stance pass that judges without
  ever seeing the document text and whose verdict moves with list position (one document, identical
  inputs, position varied alone → four different verdicts). Nothing was deleted —
  `EPISTEMICS_MARKERS_ENABLED=true` opts back in — and it is rebuilt only behind a hand-labelled
  study ([ADR-041](decisions/ADR-041-node-b-stance-rebuild-or-retire.md)). Document year,
  relevance score and graph freshness are unaffected and still shown.

## The knowledge layer

- **Text after a reference list goes unread, while a mention inside one counts.** The keyword,
  definition and written-form passes stop at a document's References heading, so appendices,
  methods and sections the PDF reader emitted late go unseen: 3.7% of the library's prose, in 32
  documents (2026-10-09). Concept presence reads the whole text instead, so a concept named only in
  a cited title counts as present: 111 of 1,079 concept–document pairs exist only past that
  heading, 88 of them in reference lists or back matter. Both change together in the next release,
  with a before/after for every layer they move (ROADMAP 95).
- **The map covers the concepts you choose; extracted keywords alone do not organise a library.**
  1,445 of 1,496 extracted keywords (97%) occur in a single document, so the graph is built from
  concepts you take on in Manage keywords, not from keywords: 13 on the development library, beside
  344 extracted terms that stay searchable. No concept's name there ends with another's, so the
  taxonomy view's *Proposed placements* has nothing to offer on that library.
- **A document's keywords are a rough sketch of it.** They are listed best first, and known junk
  (table residue, letter-spaced text, numbers, web addresses, cited authors) is filtered out. On 15
  documents labelled by hand, about two in three are real terms of the paper (64%, 95% interval
  50–77%), and the filters did not change that measurably (+1 point, −4 to +6): what replaced the
  junk was mostly junk too. A single word is often too vague to name the paper's notion. Use them to
  skim a document, not to judge what it is about
  ([baseline](../tests/eval/baselines/keyword_cleanup_2026-10-23.md)). Keywords are extracted from a
  source checkout ([usage](usage.md)); the installed app does not extract them yet.
- **A definition's grade sorts; it does not decide.** On hand-labelled candidates, `strong` was
  right 8 times in 13 (95% interval 36-82%), and a sentence that praises a term grades as strongly
  as one that defines it. Read the quoted sentence; nothing is chosen for you.
- **A large wiki topic is summarised from a sample.** Past what fits the model's window (8,192
  tokens by default, `WIKI_CONTEXT_TOKENS`), a topic's note is written from a fixed sample of its
  papers and says so ("written from 36 of 87 papers"); the others are listed as sources, unread.

## Reading documents

- **Scanned PDFs are only read if your machine happens to have an OCR engine, and the app ships
  none.** A PDF that is pure page images extracts to nothing and is marked *broken*. But if a
  `tesseract` binary is on your PATH, the PDF reader finds it by itself and reads the pages, and
  nothing in the app asks for this or reports it: the same scan produced 0 characters on one date
  and 34,600 on another, on the same machine, and the extraction cache keeps whichever came first.
  Opt-in OCR whose output is marked as such is designed and not built
  ([ADR-039](decisions/ADR-039-ocr-sidecar-for-scanned-pdfs.md)).
- **A reference links to a paper in your library only when the titles agree, and the links are
  worked out once.** A document's bibliography is shown in full (5,639 references); a reference
  becomes a link only on an exact DOI or an agreeing title, because surname-plus-year alone was
  wrong 13 times in 16 (2026-08-26). 44 references link today. Links are computed when a document
  is first read: adding the paper a reference points at does not make it a link until the citing
  document is read again, or `scripts.extract_citations --reresolve` runs (not yet a button).
- **Document metadata extraction is imperfect.** A handful of documents still yield no title, or
  publisher furniture instead of one, and the layers that key on the title inherit that; the
  reference links above are the visible consequence.
- **Size limits on what you add: 1 GB per file, 50,000 files per add.** A larger file, or an
  EPUB/DOCX/ODT that would expand past 1 GB when opened (or is compressed the way a zip bomb is),
  is refused before it is opened, with a sentence saying why. These are safety limits sized far
  above real documents (the largest in the development library is 31 MB), not measured ceilings;
  `DOC_MAX_INGEST_BYTES`, `DOC_MAX_ARCHIVE_BYTES` and `DOC_MAX_ADD_FILES` raise them
  ([`docs/security.md`](security.md) S-1, S-2).

## Setup and scale

- **Validated at ~100 documents, not yet at thousands.** Retrieval quality is benchmarked and holds,
  and backend RAM measures flat at about 2 GB because both search indexes live on disk
  ([ADR-036](decisions/ADR-036-sparse-index-on-disk.md)). What binds is the first ingest,
  dominated by PDF extraction at roughly 15 seconds per document, and disk at about 6 MB per
  document. The enrichment passes (keywords, graph, wiki) still have corpus-wide recomputes and
  thresholds tuned on a library this size, so do not bulk-ingest thousands of documents yet.
  Numbers and projections: [`docs/performance.md`](performance.md).
- **Command-line enrichment runs use your default provider.** The `scripts.*` passes that call a
  model (taxonomy, wiki, graph annotation, eval judge) take the provider configured in `.env`; with
  an Anthropic key there, a run you meant to be local bills the API. Pass `--provider ollama` on
  each run, and the judge or reviewer provider flag where a pass has one.
- **An API key entered in the app is stored in plain text** in your data folder, weaker than an OS
  keychain, which is the recorded upgrade path
  ([ADR-034](decisions/ADR-034-in-app-provider-setup.md)). Use `.env`, which takes precedence,
  if you would rather manage the key yourself.
- **Single-user, local-first by design.** The FastAPI backend serves one desktop app on localhost;
  multi-client serving would need threadpool offloading (documented, not built).
- **Tested primarily on Windows** plus CI on Linux; macOS (MPS) paths work but are unbenchmarked.

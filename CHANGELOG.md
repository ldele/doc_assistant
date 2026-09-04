# Changelog

Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is [SemVer](https://semver.org/) on the `doc_assistant` package, and the desktop app
(Provenote) ships the same number.

The engineering record is finer-grained than this file: per-change entries live in
[`docs/DEVLOG.md`](docs/DEVLOG.md), design decisions in [`docs/decisions.md`](docs/decisions.md).

## [0.6.0] — 2026-09-01

### Added

- **You can read a document beside its library entry.** Open any PDF in your library and its own
  pages appear in a pane next to it — the page as it was printed, not the text pulled out of it.
  Clicking a passage in **Chunks**, or *Show the page* on a citation in chat, opens the pane at the
  page that passage came from. The pane fits the whole page by default, and you can switch to full
  width, zoom in and out (Ctrl or Cmd with the scroll wheel), and drag the divider to give it more
  room; zooming asks for a sharper render rather than magnifying a blurry one. A document whose
  file has moved, or that lives on a drive that is not connected, says so and names the path
  instead of showing a broken pane; a format that has no pages says that too.
  ([ADR-050](docs/decisions/ADR-050-source-viewer-page-rendering.md))
- **A citation can show you where it came from, in place.** Clicking a source in an answer offers
  *In context*: the cited passage highlighted inside a window of the surrounding text, with how far
  through the document it sits. A passage that cannot be placed says so rather than guessing.
- **You can import from Zotero.** Point the app at a Zotero library and its attachments arrive in
  the same review sheet as any other addition — same duplicate check, same choice between copying
  a file in and referencing it where it is. Zotero's own titles, authors and years are kept and are
  not overwritten by what the app reads out of the PDF.
  ([ADR-049](docs/decisions/ADR-049-ingestion-adapters-and-external-metadata.md))
- **You can re-run one part of reading a document, instead of all of it.** Metadata, figures,
  references and text can each be redone on their own, for one document or a selection, and each
  says what it will cost before it runs — they differ by four orders of magnitude. Passes that only
  make sense across the whole library are named and declined rather than quietly missing.
  ([ADR-048](docs/decisions/ADR-048-per-part-re-ingest-scope.md))
- **The concept graph is back, and says how much of your library it covers.** It now states plainly
  that it covers, say, 30 of your 98 documents, and why — a document appears once it mentions one
  of the concepts on the graph. You can put a concept on the graph from **Manage keywords**, which
  previously needed a script the app does not ship, so a fresh install's graph stayed empty forever.
  ([ADR-018](docs/decisions/ADR-018-graph-vocabulary-scope.md))
- **You can add documents from inside the app.** Drop files or a folder onto the window, or use
  *Add documents…* in the app menu or above the Library. Before anything is copied or indexed you
  get a review sheet saying what will happen to each file — added, already in your library,
  unsupported format, or unreadable — with the exceptions sorted above the ordinary additions, so
  a long list cannot bury one. A dropped folder is read all the way down and the total is stated
  before you commit. Nothing is copied, registered or indexed until you confirm, and *Undo all* is
  offered immediately afterwards.
- **Your documents no longer have to live in Provenote's own folder.** Adding a file is now a
  choice between *copy it in*, where the app owns the copy, and *reference it where it is*, where
  the file stays yours: a referenced file is never moved, altered or deleted by the app, and
  removing it from your library removes only the app's record of it. When the folder holding it
  cannot be reached — an external drive, a share that is offline — the app says that location is
  unavailable rather than showing you a library full of missing documents.
  ([ADR-046](docs/decisions/ADR-046-added-documents-copy-or-reference.md))

### Changed

- **Improving how a document is read no longer disconnects everything attached to it.** A
  document's identity used to be the text extracted from it, so any extraction fix gave it a new
  identity and cut it loose from its figure descriptions, keywords, folder assignments and any
  metadata you had corrected by hand. Measured on a 97-document library before the fix: a single
  extractor change would have orphaned 4,123 rows, 19 of which nothing could regenerate. A
  document is now identified by the file it came from.
  ([ADR-047](docs/decisions/ADR-047-document-identity-survives-re-extraction.md))
- **A citation only becomes a link when the titles agree.** A reference used to be matched to a
  paper in your library on first-author surname and year alone, which on a collection holding
  several same-year papers by common surnames is not evidence: on the working library 13 of the 16
  stored links were wrong — one review of graph neural networks pointed at a paper on mouse
  whisker cortex. Matching now requires the titles to agree as well. On that same library the
  links went from 16 to 41, the 12 false ones dropped and the rest checked by hand; your own
  numbers will differ.
- **A title you correct shows up everywhere.** Editing a document's title fixed the Library list
  but not the document's own page, which still showed the extracted version — most visible on the
  OCR-derived titles that most need correcting.
- **EPUB files no longer read their own table of contents as body text**, and page furniture is
  dropped from EPUB as it already was from HTML.
- **Indexing can use more than one core, and you can say how many.** Off, light (the default,
  two), balanced or full — never the whole machine. It changes only how long indexing takes, never
  what it produces. On this machine two workers bought most of the achievable gain, and the curve
  flattens well before the core count; expect that to vary with your machine and your documents.

- **Benchmark numbers now say what produced them, and the app can tell you when two of them
  cannot be compared.** Every eval run records the corpus it searched, the model that wrote the
  answers, and the retrieval settings that were live; a new check reads that record and reports,
  per metric, whether two runs measured the same thing. This exists because a five-trial run in
  August looked like a 6% improvement and was a model swap — a number nobody could have caught by
  reading it. Published results carry the same record, so they can be checked against a later run
  without access to the machine that produced them. Documentation: [`evals/README.md`](evals/README.md).

- **Settings is a set of categories rather than one long scroll**, and the panel is wider on a
  wide window.
- **Native menus and dropdowns follow the app's theme.** The folder-scope dropdown in chat, and
  every other system control, used to be painted in your operating system's colours — a light menu
  over a dark app.

### Fixed

- **Figure images that had gone missing are back.** 723 of 881 figure crops on the working library
  could not be shown; they are re-rendered from the position already recorded, without re-detecting
  anything. Re-running the figure pass also used to discard descriptions that had been paid for —
  552 of them — and now carries them across.
- **Two indexing faults that lost text.** A cursor skipped past overlapping passages, leaving 30-40%
  of them unable to say where in the document they came from (now 100%), and a composed passage
  could record a position it did not occupy.
- **A short document is no longer reported as broken** simply for being short, and the app now names
  the tool that actually read each file instead of always naming the PDF reader.
- **A heading that ends in a colon is a lead-in, not a title** — so a paper whose first line reads
  *"Preprint of the paper:"* now takes the title underneath it.

### Known limits

- **A cited passage is located on the page, but not marked on it.** The pane opens at the right
  page; it does not yet highlight the passage within that page.
- **A page number can occasionally be one page early.** For a small number of pages (measured at
  0.2% across the library, concentrated in one scanned book) the text recorded for a page is
  actually the previous page's, so a jump lands one page before the passage.
- **Citation links are worked out when a document is first read, and are not revisited.** Adding
  the paper a reference points at will not turn that reference into a link on its own — the
  earlier document has to be read again. Refreshing them without re-reading anything is possible
  from the command line today; it is not yet a button.
- **Moving or renaming a file outside the app loses anything recorded against its old path.** The
  document itself survives: its content is recognised, so it is not read or embedded again, and its
  figures, keywords and any corrections you typed stay attached. But an exclusion you set is
  recorded against the path rather than the document, so it does not follow the file.
- **Scanned PDFs are only read if your system happens to have an OCR engine, and the installer
  does not include one.** A PDF that is pure page images — no selectable text — extracts to nothing
  and is marked *broken* in your library. But if a `tesseract` binary is on your PATH, the PDF
  reader finds it by itself and reads the pages instead. Nothing in the app asks for this or
  reports it: the same scan produced **0 characters on one date and 34,600 on another**, on the
  same machine, with nothing in the app changed — a `tesseract` install had appeared in between.
  Two machines running the same version can therefore build different libraries from the same file,
  and the extraction cache cannot see the difference, so the first result sticks. The installer
  ships no OCR, so a fresh install gets the 0-character behaviour. Deliberate recovery of scanned
  documents — opt-in, and marked so you can see which text came from OCR — is designed but not
  built ([ADR-039](docs/decisions/ADR-039-ocr-sidecar-for-scanned-pdfs.md)).
- **Everything under *Known limits* for 0.5.1 still applies, except the update check** — releases
  are now published for the tags, so it compares against the newest one rather than reporting that
  it cannot.

## [0.5.1] — 2026-08-14

### Added

- **Provenote can tell you when a newer version has been published** — and that is deliberately all
  it does. Settings → Updates shows what version you're running, checks GitHub for the latest
  release, and links you to it. It never downloads or installs anything: you decide what runs on
  your machine. Automatic checking is **off by default** and, when you turn it on, runs at most
  once a day; the **Check now** button works either way. Nothing about you or your documents is
  sent — the request asks for a version number and carries no query, title or identifier.
  ([ADR-044](docs/decisions/ADR-044-update-notification-not-delivery.md))

### Changed

- **Document keywords are about the document again.** Four causes fixed at once. Running headers
  and journal stamps used to dominate — one paper spent 11 of its 15 slots on *"Exp Brain Res.
  Author manuscript; available in PMC…"* — and another spent 9 on overlapping fragments of a
  single figure label. Bibliographies contributed author surnames and citation debris. And the
  tokeniser split on `.` and `/`, so `16p11.2` was stored as `16p11` and `GPT-3.5` as `GPT-3` —
  quietly renaming a genetic locus and a model. Across a 97-document library: overlapping
  fragments **27% → 0%** of slots, documents with no keywords at all **15 → 1**, citation
  fragments gone. Re-index from Settings to apply it to your own library.
- **Related papers are ranked, not scored.** The panel showed a similarity number that looked far
  more precise than it was — whole-document similarity puts every paper in a field within a few
  hundredths of every other. It now shows position (1st, 2nd, 3rd) and says why.
- **The contested / superseded chips are labelled experimental** wherever they appear, with the
  known limitation stated in Settings. They remain off by default.

### Removed

- **The Graph tab is hidden for now.** The concept graph and gap list are unchanged underneath and
  nothing was deleted — but the page is empty until the graph is built, and an empty page reads as
  a broken one. It returns once it has a home worth navigating to.

### Known limits

- **The update check needs published GitHub releases to see.** Until releases are cut for the
  tags, it honestly reports that it can't compare rather than claiming you're up to date.
- **Keywords describe a document; they don't group your library.** After the fix above they are
  accurate, but 97% still appear on exactly one document — useful for finding a paper, not for
  slicing a collection into topics.

## [0.5.0] — 2026-08-11

**Your library became somewhere to read, not just a list of what you own.** Opening a document
used to show metadata and little else. It now lays out as five ordered blocks — what the document
is, what it connects to, the passages as indexed, its figures, and its full bibliography — with a
jump-nav across the top. Alongside it, the chat history finally has a way to be tidied.

### Added

- **A document view in five blocks: Metadata, Connections, Chunks, Figures, References.** Each has
  a heading, an anchor, and a sticky jump-nav that stays put while the page scrolls. Opening a
  document **transfers no passage text at all** — the Chunks block is collapsed and unfetched until
  you ask for it, and the header reads a summary the library list already has. Detail payloads
  measure a median of 170 KB (worst case 1.85 MB, a 663-block book); on a 142-block paper the view
  is 924 DOM nodes collapsed and 3,908 expanded.
- **Passages are a scannable list, not a wall of text.** Parent blocks are collapsed to a marker, a
  preview line and a child count, with Expand-all / Collapse-all. Which document you left expanded
  is remembered per document for the session, so paging back and forth with the ← → arrows finds
  each one as you left it. Only the open flag is remembered — never the text.
- **Figures, extracted and readable.** A document's figures appear with their captions and open at
  full size. Each says whether the assistant can *retrieve* that figure, which is a different thing
  from showing it: turning a figure into something searchable needs a paid vision pass that is
  deliberately skipped when the caption already describes the image. Across the development library
  811 of 881 extracted figures carry an image region; the rest are captions whose image could not be
  located, and they say so rather than rendering blank.
- **The whole bibliography, with links to what you already own.** Every parsed reference is listed,
  including entries that matched nothing, capped at 200 with "showing N of M" — and the cap spends
  its budget on linked rows first, so a 346-reference paper cannot lose the one entry you can
  actually open.
- **Export your entire chat history as one markdown file**, and delete conversations in bulk.
  Settings → Chat history → Export all writes every conversation, not the 100 the sidebar shows: on
  the development history that is **184 conversations / 188 turns / 347 KB**, where an export
  inheriting the sidebar's cap would have silently dropped 84. Bulk delete is a soft delete applied
  in one transaction, and the same control restores it.

### Changed

- **A real environment variable now beats `.env`.** Setting a variable for one command — most
  importantly `LLM_PROVIDER=ollama` — used to be silently ignored because `.env` was loaded with
  override, so a run you had told to stay local could still bill the API.
- **The parent/child chunk sizes are now measured, and kept.** The 2026-06-06 sweep that once
  justified them was void: an environment bug meant all six arms indexed the same corpus, so it
  compared one configuration with itself six times. Re-swept twice — the public 10 on Claude Haiku
  and 97 documents / 35 questions on a local model. Nothing beats the default beyond its variance,
  so the lock holds, but the grid shows a real trade-off rather than a winner, and a smaller child
  chunk retrieves better on 45% fewer input tokens. Details and the experiment that would settle it:
  [`evals/README.md`](evals/README.md).
- **Full-page scans are no longer mistaken for figures**, and the figure pass can be cleared and
  re-run with `--force`.

### Known limits

- **Most reference links into your own library are withheld, deliberately.** The bibliography is
  shown in full, but a *link* from a reference to your copy is re-checked before it is offered, and
  only an exact DOI or an agreeing title survives. On the development library that is **4 links
  presented where 16 are stored** — the other 12 were wrong, pointing at unrelated papers. The
  matcher resolves on first-author surname and year with no title comparison, and it runs only when
  a document is first indexed, so it is frozen at whatever your library held that day. Showing four
  correct links beats showing sixteen of which twelve lie; repairing the matcher is next.
- **A scan with no text layer at all is still unreachable** — unchanged from 0.4.2, one document of
  97 here. OCR is designed and gated on measuring its quality first.
- **Local models still cite far less of what they write than a hosted one** — unchanged: 36%
  (`llama3.1:8b`) and 14% (`qwen2.5:7b`) of sentences against 81% for Claude Haiku, on the same
  prompt and retrieval over 27 questions.
- **Per-source "epistemic assessment" remains off by default** (since 0.4.1), so the source strip
  reads *assessment withheld* on every row. The stance pass behind it judges without seeing the
  document text; it is being rebuilt rather than quietly re-enabled.
- **Validated at ~100 documents.** The enrichment layer still has corpus-linear hot paths; don't
  bulk-ingest thousands before those land.

## [0.4.2] — 2026-08-07

**Documents that were in your library but unreachable are now searchable.** Some PDFs —
typically older scans that carry a text layer behind a page image — were being indexed
almost empty. They appeared in the Library and never in an answer. Three such papers in the
development library went from 1, 7 and 16 indexed passages to **61, 125 and 1,019**.

### Fixed

- **Scanned papers with a text layer are no longer indexed as blank.** The PDF reader treats
  a full-page image as a picture and never reaches the text behind it, so the document was
  stored with a handful of empty passages and could not be retrieved — while looking
  perfectly normal in the Library. It now falls back to the page's own text layer when the
  conversion loses it. On the development library this took retrieval from **28 of 35**
  benchmark questions finding their source document to **34 of 35**, and library health from
  93 healthy / 2 marginal / 2 broken to **96 healthy / 0 marginal / 1 broken**.
- **Improvements to document reading now reach libraries you have already indexed.** The
  extracted-text cache only noticed when a *file* changed, never when the *reader* improved —
  so every past reading fix was invisible to anyone who had already added their documents.
  It now also tracks which version of the reader produced each cached document. The first
  indexing run after an upgrade re-reads your documents, once; **Index** shows how many
  before it starts, and the app says why in its log.
- A high-severity advisory in a build-time dependency (postcss). It never affected the
  installed app, only the machine building it.

### Known limits

- **One kind of document is still unreachable: a scan with no text layer at all.** There is
  nothing to fall back to — the page is only an image. Recovering those needs OCR, which is
  designed but not built, and deliberately gated on measuring OCR quality first: text that is
  wrong is worse than text that is absent, because absence is honest while garbage is
  retrievable and citable. One document of 97 in the development library.
- **Re-reading a large library takes a while.** Extraction is the slow part of indexing —
  roughly a minute for a long book — and the run above is one-off, per improvement. Check the
  count before starting if your library is large.
- Local models still cite far less of what they write than a hosted one — unchanged, and
  stated in the app where you choose your answer engine.

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

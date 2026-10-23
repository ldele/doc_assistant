<!-- status: active · updated: 2026-10-23 · class: append-only -->

# ADR-058 — A model proposes each document's keywords, with a sentence from the document for each

- **Status:** proposed (Claude Code, 2026-10-23). Not accepted, not built. The user asked for the
  scan on reading session 5c's result: *"We need LLM scan definitely."* The shape below is
  Claude's proposal. The build spec is
  [`docs/specs/feature-llm-keyword-scan.md`](../specs/feature-llm-keyword-scan.md). ADR-055, 056
  and 057 are taken by sessions working in their own worktrees, so this is 058.
- **Date:** 2026-10-23
- **Deciders:** user + Claude Code
- **In one sentence:** once per document, a local model reads the title, the opening, the headings
  and a sample of the paragraphs, and proposes up to 3 main and 12 detail keywords, each with a
  sentence copied from the document. A deterministic check keeps only keywords the document
  contains. The proposals are a sidecar, and they replace a document's TF-IDF keywords in the
  Library only after they beat them on your labels and you say so.

## To decide — seven choices, each with a recommendation

1. **What the model reads.** *Recommended:* the title, the opening of the text (a quarter of the
   budget, where the abstract sits), the headings (a tenth), then paragraphs spaced evenly through
   the body, all after the cuts the keyword extractor already makes (running headers, the
   bibliography, web addresses). About 5,500 tokens in an 8,192-token window. *The cost:* a long
   document is sampled. The scan reads about a third of the median document and 3% of the longest
   (a 365-page book). 98 of the 104 documents do not fit whole.
2. **What it returns.** *Recommended:* up to 3 *main* keywords (what the document is about or
   introduces) and up to 12 *detail* keywords (narrower names it uses), 15 in all as today. Each
   comes with one sentence copied from the document. The long form is given when the document
   spells one out. No confidence score. *Alternative:* no sentences, which halves the output. That
   loses the check that catches an invented keyword, and the sentence you would read on hover.
3. **What keeps a keyword.** *Recommended:* checks that need no model. The keyword must occur in the
   document. It must not have one of the shapes session 5c filters (a table residue, a letter-spaced
   run, no letter, a citation shape, a name the document cites with "et al."). It must not be a
   duplicate or over the cap. The model's sentence is kept when it is copied exactly and contains
   the keyword; otherwise the first sentence of the document that contains it is looked up. Refused
   proposals are stored with their reason, so you can see what was dropped.
4. **Where it is stored.** *Recommended:* a sidecar of two tables, the runs and the proposals. It
   keeps the model's raw answer, so the checks can be re-run without the model. It is keyed on the
   document, with a hash of exactly what the model read: when a re-extraction changes that text,
   the scan is stale and the document falls back to TF-IDF until it is rescanned.
5. **How it reaches the Library.** *Recommended:* not at all until the measurement (decision 7)
   passes and you say yes; until then a collapsed *Model proposals* section in the document view
   shows it. After that, per document: a document with a current scan gets its keyword links from
   it, main keywords first, and every other document keeps TF-IDF. The link records where it came
   from. Keyword families, the filter, Manage keywords and promotion keep reading names, so they
   work unchanged, and their document counts move only where names change (measured and listed,
   as in 5c). *Alternative:* both lists side by side, for good. That is twice the reading, so it is
   kept only if the measurement shows the two find different real keywords.
6. **Which model, and when it runs.** *Recommended:* a local model. `qwen2.5:7b` is measured here;
   `llama3.1:8b` is checked at the build's first live run, and the numbers pick. The provider is
   always named on the command line, never taken from `.env` (KI-4). First a runner: it backfills
   the 104 documents in about 40 minutes with the GPU free. Then an ingest option, *Scan new
   documents' keywords with a local model*, off by default, that never holds up an ingest. A paid
   model only on your word, after the local measurement.
7. **How it is judged.** *Recommended:* session 5c's yardstick unchanged: the same 15 seeded
   documents, the same label rule, the same page, the document as the unit. Only keywords you have
   not labelled are asked. The predictions are registered in the spec (§9) before the scan reads any
   sample document. The switch needs the precision gain's interval above zero, and no document
   losing a main topic that TF-IDF had.

## In practice — the SuperAnimal paper, then two papers you labelled

**A paper outside your sample.** `2203.07436v4.pdf`, *SuperAnimal pretrained pose estimation
models for behavioral analysis*, 59 pages. It is one of the three documents the prompt was written
on, so it is not evidence of how the scan does; it shows what the scan produces. On the left, the
TF-IDF keywords in the Library today, in rank order. On the right, what `qwen2.5:7b` proposed on
2026-10-23 (prompt v1; v2 only renames the two lists and rewords "terms" as "names"), after the
checks. *Uses* counts the keyword
in the body; *sentence* says whether the model's own sentence passed or one was looked up.

| Place | TF-IDF today | Scan proposal | Kind | Uses | Sentence |
|---:|---|---|---|---:|---|
| 1 | `superanimal` | pose estimation | main | 52 | model's |
| 2 | `ap10k` | behavioral analysis | main | 14 | looked up |
| 3 | `replay` | SuperAnimal method | main | 9 | model's |
| 4 | `imagenet transfer` | SuperAnimal-TopViewMouse | detail | 27 | model's |
| 5 | `ap-10k` | SuperAnimal-Quadruped | detail | 19 | model's |
| 6 | `data ratio` | DeepLabCut | detail | 39 | model's |
| 7 | `hrnetw32` | pose estimation models | detail | 2 | looked up |
| 8 | `train data` | keypoint | detail | 109 | looked up |
| 9 | `video adaptation` | video adaptation | detail | 38 | model's |
| 10 | `irodent` | fine-tuning | detail | 80 | looked up |
| 11 | `please provide` | zero-shot performance | detail | 24 | model's |
| 12 | `transfer learning` | mAP | detail | 50 | looked up |
| 13 | `hrnet-w32` | Horse-30 | detail | 10 | model's |
| 14 | `please describe` | Openfield | detail | 19 | looked up |
| 15 | `sa-tvm` | bottom-up model | detail | 4 | looked up |

By Claude's reading, not your labels:

1. **The main topic.** *Pose estimation* is used 52 times and is not among TF-IDF's 15. TF-IDF
   weighs a phrase down when other documents share it, and a library collects papers on its
   subjects: 11 of the 104 documents mention *pose estimation*, 14 *residual learning*. The scan puts
   it first, with the paper's own sentence: *"A common key step for behavioral analysis is first
   extracting relevant keypoints on animals, known as pose estimation."*
2. **What TF-IDF spends places on.** `please provide` and `please describe` are questions from the
   paper's datasheet template. `ap10k` and `ap-10k`, and `hrnetw32` and `hrnet-w32`, are one name
   each, spelled two ways by the extraction.
3. **What the scan gets wrong.** *Pose estimation models* repeats the first main keyword.
   *Behavioral analysis* is backed by a looked-up line that is a heading (*Unsupervised behavioral
   analysis.*). The build looks sentences up with `definitions.sentence_spans`, which skips headings.
   Every proposal names something the paper is about or uses.

**Two papers from your sample, as you labelled TF-IDF's list after 5c** (*t* topic, *d* detail,
*n* noise). The scan has not read them. How it does on them is the measurement.

- `gao2024_retrieval-augmented-generation-survey.pdf`: `rag` t, `retriever` t, `llms` t, `chunks`
  d, and 11 noise: `dataset-base`, `wikipedia text`, `inference once`, `text chunk`, `tuning once`,
  `chunk inference`, `inference iterative`, `text sentence`, `text doc`, `chunk tuning`,
  `doc inference`. They are the cells of the survey's comparison table, read as phrases. Main topic:
  *partly*, before and after.
- `resnet_he_2015.pdf`: `residual nets` t; `degradation problem`, `shortcuts`, `ilsvrc` d; and 11
  noise, among them `3x3 conv`, `18-layer`, `billion flops`, `1e4`, `152-layer`. Main topic:
  *present* before 5c, *missing* after.

## Context

Session 5c (ROADMAP 99) added five junk filters to the TF-IDF extractor and re-extracted the
library on 2026-10-22. On the user's labels of 15 seeded documents, precision (Topic + Detail over
all keywords) moved 62% [46, 78] → 64% [50, 77], a change of +1 point [−4, +6]. The filters removed
37 of 44 noise keywords, but 34 of the 44 that came in were noise too. The main topic was *present*
in 1 of 15 documents before. Reproduced on 2026-10-23 with the probe's `measure.py`.

Four facts shape the choice. Each was measured on 2026-10-23 with read-only probes in
`.claude/probes/2026-10-23-llm-keyword-scan/`.

- **TF-IDF's candidates are the limit.** Filtering junk leaves the next candidates, and those are
  junk of other kinds. TF-IDF also cannot write *measurement of animal behavior*: its candidates are
  one to three words with no stopword inside.
- **The main topic is what the library shares.** A collection gathers papers on the same subjects,
  so a paper's main topic tends to have a low inverse document frequency, which is the factor
  TF-IDF ranks by.
- **The text does not fit a local window.** The median body, after the extractor's cuts, is 13,412
  tokens; 98 of 104 exceed 6,000; the longest is 191,126. Ollama's default window here is 4,096
  tokens, and a longer prompt is cut from the front without an error (KI-65).
- **A local model does the task in about 20 seconds.** On three documents outside the sample,
  `qwen2.5:7b` read about 5,200 tokens and wrote about 800, in 18 to 26 s per document with the GPU
  free. Prompt v1 returned 44 keywords, 43 passed the checks, and the model's own sentence passed
  for 25 of the 43. Three documents is a development set, not a result.

The rules already in force: the Enrichment-Layer Pattern (a sidecar, never the chunk store); the
KI-4 guard; local before paid (memory *cost discipline*); a model's reliability is the evidence
shown, never its own rating (ADR-053); received text is kept as written (ADR-043); ingest-time
work is stored durably (memory *ingest once, well*); a failure informs and does not block.

## Options

1. **The model proposes, as a sidecar, and replaces TF-IDF per document once measured**
   (recommended). It can name a main topic in the document's words, and it is held to the text by
   checks that need no model. *Trade-off:* about 20 s of GPU per document, a sample of long
   documents, and a validator to maintain.
2. **The model chooses among TF-IDF's top candidates** (a re-rank). Nothing can be invented, and
   the output is shorter. *Trade-off:* it inherits the candidate pool that 5c found to be the
   limit: lower-cased fragments of one to three words with no stopword inside. Kept as a cheap
   second arm if the measurement fails.
3. **Read the whole document, section by section, then merge.** Nothing is sampled. *Trade-off:*
   about 2.5 times the tokens at the median document and about 35 times at the longest; the number
   of calls grows with the length, and the merge is a second prompt to prove. A later option for
   books.
4. **Show both lists, for good.** Nothing is lost. *Trade-off:* twice the reading, and the noise
   stays on screen.
5. **A paid model first.** Likely better keywords, at about $1 for the library on Haiku 4.5.
   *Rejected for now:* the project proves model logic on a local model before paying, and the
   checks in decision 3 are what make either model safe.

## Decision

Option 1, with the recommendations of the To-decide list. **The deciding reason:** the noise 5c
could not filter comes from what TF-IDF can propose, not from how it ranks, so a better list needs
a proposer that reads the document. The checks keep that proposer honest without trusting it.

**What would reverse it:** the measurement failing on the local model and on a paid arm. The scan
then stays in the review section, and option 2 is measured next.

## Consequences

- **Easier:** a document's keywords can say what it is about. Each keyword carries a sentence from
  its document, so you can judge it without opening the paper. The proposals are kept, so a later
  check change costs seconds, not a rescan.
- **Harder:** a GPU pass per document (about 40 minutes for the library, about 61 hours of GPU at
  10,000 documents). Two tables, a runner, a review API and section; in part 2, one column on the
  keyword link and a source switch on the extractor. A rescan when a document's text changes.
- **Robustness contract:** 0 documents gives nothing to scan, and the Library is today's. One call
  per document, so the work is linear at 10,000, and storage is about 30 MB there. The read budget
  is a share of the model's window, not a number fitted to this library.
- **Order with the other open decisions:** after 5c's commit. ADR-055's long forms apply to scan
  keywords too, through the same per-document reader. ADR-057 changes the tokeniser, so names are
  derived and recomputed, never stored as the identity. ADR-056 changes some documents' text, so
  those scans go stale and are redone.
- **Must revisit:** long documents (option 3); the paid path, because `llm.AnthropicClient` sends
  `temperature`, which the Anthropic API rejects from API version 2026-10-21 (SDK 2.0.0; the pinned
  SDK 0.103.1 is not affected); row 25's other parts, which can read the same input.

## Confidence

- ✓ The sizes: measured on all 104 documents, 2026-10-23 (`anatomy.py`).
- ✓ That a local model returns well-formed proposals within the window: 3 of 3 documents on v1,
  with the output cap raised after v0 overran it.
- ⚠ Whether the proposals are better: unknown. That is the measurement, on 15 documents, with an
  interval of about ±14 points. The predictions are registered in the spec (§9) before any sample
  document has been read.
- ⚠ Prompt v2 has not run; it renames the two lists to the GLOSSARY's words. Its first run is the
  build's gate G2.
- ⚠ The time per document: 18 to 26 s in four runs, and 96 s and 196 s in two more taken while a
  game held the GPU at 99%. It depends on what else is using the card.
- ⚠ The model's own sentence passed for 25 of 43 keywords. The look-up makes every kept keyword
  carry a sentence; it does not make the model's sentences trustworthy.

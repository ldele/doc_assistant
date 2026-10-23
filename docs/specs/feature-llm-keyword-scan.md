<!-- status: draft · updated: 2026-10-23 · class: living -->

# Spec — LLM keyword scan (ROADMAP 25, the keyword part)

> The executor brief (ADR-019). Target executor: **Claude Code**. The *why* is
> [ADR-058](../decisions/ADR-058-a-model-proposes-each-documents-keywords-with-evidence.md)
> (proposed): read it first. This spec is the *how*, and its open forks are ADR-058's To-decide
> list. It is marked ready only after the user has answered that list.

**Status: draft, 2026-10-23. Nothing built, nothing in `data/` written.** Written after session 5c
(ROADMAP 99) measured TF-IDF keywords on the user's labels: precision 62% → 64%, +1 point [−4, +6],
because 34 of the 44 keywords that replaced filtered junk were noise too. The user's answer:
*"We need LLM scan definitely."* Row 25's other parts (figure and citation links, reference order,
concept marking at ingest, 93b's model definitions, S-7) are not in this spec; §11 says where they
meet it.

Every number below was measured on 2026-10-23 on the working library (104 documents) with read-only
probes in `.claude/probes/2026-10-23-llm-keyword-scan/` (local-only): `anatomy.py` for sizes,
`scan_probe.py` for the model runs. Model runs used three documents **outside** 5c's seeded sample.

---

## 1 · Goal

Once per document, a model proposes the document's keywords: up to **3 main keywords** (what the
document is about or introduces) and up to **12 detail keywords** (narrower names it uses), each
with one sentence from the document that uses it. A deterministic check, run offline, decides what
is kept. The proposals are a sidecar keyed on the document. They reach the Library only after they
beat session 5c's TF-IDF keywords on the user's labels, and the user says so.

## 2 · Hard constraints

- **Enrichment-Layer Pattern** — a sidecar module and an idempotent CLI runner; never touches the
  chunk store, the vector stores or the cached markdown (AGENTS.md; `src/doc_assistant/CLAUDE.md`).
- **Ollama first, the provider never inherited** — `--provider` is required on the command line,
  so `.env`'s all-Anthropic default cannot make a "local" run bill (KI-4). A paid provider goes
  through `llm.assert_provider_intent` (banner, abort window, key check).
- **No paid call before the logic is proven on a local model** (the gates in §5).
- **Evidence shown, never a model's rating of itself** — no confidence field is asked for or stored
  (ADR-053; `docs/knowledge-layer.md`).
- **Received content kept verbatim** — the keyword and its sentence are stored as the document
  writes them; the keyword's *name* is derived and recomputable (ADR-043).
- **Identity** — keyed on the document id (ADR-047), with `doc_hash` and the exact input's hash
  recorded so a changed text is detectable.
- **No half-written document** — one transaction per document; a failed call leaves the previous
  state (plan prior art: dual-datastore writes diverge on partial failure).
- **Robustness contract** — 0 documents works; the read budget derives from the model's window,
  not from this library (`.claude/CONTEXT.md`).
- **Inform, don't block** — a document the scan cannot do keeps its TF-IDF keywords and says why.
- **Vocabulary** — the kinds are *main* and *detail*. Not "topic" (forbidden for a concept,
  GLOSSARY C-001) and not "term" (a vocabulary row nobody has taken on, C-014).
- **Windows encoding** — `encoding="utf-8"` on every read and write; the runner pins stdout
  (AGENTS.md). `structlog` only in `src/` (ADR-003).

## 3 · What the model reads per document (input view `kwscan-input-v1`)

**The sizes** (body = the cached markdown after the keyword extractor's own cuts: furniture,
bibliography, web addresses; tokens ≈ characters / 4, which Ollama's own count confirmed at 4.1):

| Per document | min | median | p90 | max | 104 documents |
|---|---:|---:|---:|---:|---:|
| Body, tokens | 1,260 | 13,412 | 31,682 | 191,126 | 2,328,683 |
| Heading outline, tokens | 14 | 158 | 415 | 2,618 | 27,483 |
| Pages | 3 | 19 | 44 | 365 | 3,341 |

- **98 of 104 bodies exceed 6,000 tokens**, so the whole text does not fit a local model's window.
  Ollama's default window on this 12 GB card is 4,096 tokens, and a longer prompt is cut from the
  front with only a server-log warning (KI-65).
- 103 documents have a title row; it appears in the first 6,000 characters of 99. Only 53 have an
  "Abstract" heading; 75 use the word in their first 4,000 characters. **The opening of the text,
  not an abstract detector, is what carries the abstract.**

**The view.** A pure function, `build_input(title, cached_markdown, *, window, max_output) ->
ScanInput`, in this order:

1. **Body**: `strip_page_furniture` → `strip_reference_section` → `strip_urls` (the extractor's
   functions, same order), then page markers and emphasis markup (`**`, `__`, `_x_`) removed.
   A derived view; the cache is not touched.
2. **Blocks**: split on blank lines.
3. **Title**: `Document.title`, or "(none)".
4. **Headings**: every heading, in order, without repeats, cut at a tenth of the input budget.
5. **The opening**: whole blocks from the start until a quarter of the input budget.
6. **Passages**: from the blocks after the opening, skip headings, table rows (`|`) and blocks
   shorter than `MIN_PARAGRAPH_CHARS = 200` (structural: a heading, caption or fragment, not a
   paragraph). Take evenly spaced paragraphs, in document order, each prefixed by its section heading
   in brackets, until the budget is spent.
7. **The budget** is the window minus the output allowance, the task text and a margin:
   8,192 − 2,048 − 400 − 256 = **5,488 tokens** at the default. The shares scale with the window,
   so a model with a larger window reads more without a code change.

`ScanInput` carries `text` (what the model reads), `body` (what the checks read), `input_sha256`
and `input_version`.

**Measured on the three development documents:** 20,922 to 21,574 characters, 5,012 to 5,243
prompt tokens with the task text. The SuperAnimal paper (59 pages, 105 headings) was read as its
title, its opening (5,112 characters), the first 62 distinct headings (the outline's cut) and 18
passages.

**The limit, said plainly:** a 365-page book gets the same budget as a 3-page note. The scan reads
about a third of the median body and 3% of the largest. Reading a long document section by section
and merging is ADR-058's option 3, not built.

## 4 · The prompt and the output (`kwscan-v2`)

System message:

```text
You read one research document and list its keywords. You answer with one JSON object and nothing else.
```

User message: the input view, a blank line, then:

```text
List this document's keywords.

"main": what the document is about or introduces: its main subject, its method, its contribution. Give 1 to 3, the most central first.
"detail": narrower technical names the document uses that a reader of its field would look up: a method, model, dataset, measure, anatomical structure, gene, cell type, chemical, instrument. Give up to 12, the most important first. Do not repeat a main keyword.

Rules:
1. A keyword is a short name of one to five words, written as the document writes it. Never invent one the text above does not contain.
2. Give each thing once. If the document gives a long form and a short form, give the long form only.
3. For each keyword, copy one sentence from the text above in which that keyword itself appears, character for character. Do not shorten, fix or paraphrase it.
4. Never list: a person's name, a journal or venue, a date, a number, a table cell, a web address, a word broken or glued by the text extraction, or a word used only as an example.
5. Do not rate your confidence.

Answer with: {"main": [{"keyword": "...", "evidence": "..."}], "detail": [{"keyword": "...", "evidence": "..."}]}
```

**The output shape**, passed to Ollama as a JSON schema (structured output, `format`). It
constrains the shape only; nothing in the content is trusted (§5):

```json
{"type": "object", "required": ["main", "detail"],
 "properties": {
   "main":   {"type": "array", "items": {"$ref": "#/$defs/item"}},
   "detail": {"type": "array", "items": {"$ref": "#/$defs/item"}}},
 "$defs": {"item": {"type": "object", "required": ["keyword", "evidence"],
   "properties": {"keyword": {"type": "string"}, "evidence": {"type": "string"}}}}}
```

**Sampling:** temperature 0, a fixed seed, window 8,192, output cap 2,048 tokens, reasoning off
(`OllamaClient`'s default; memory *thinking models stall the stream*). `OllamaClient` gains an
optional `format` argument that takes the schema; today it sends `format="json"` only.

**How the prompt was written** (development documents only: `sbert_reimers_2019.pdf`,
`gerfen2013_gensat-cre-driver-lines.pdf`, `2203.07436v4.pdf`; none is in the sample, and the probe
refuses a sample document):

| Version | What changed | Why (qwen2.5:7b, 3 documents) |
|---|---|---|
| v0 | one list, a kind per item, up to 5 + 20 | One document's output ran past 1,536 tokens and the JSON was cut (`done_reason: length`). The other two returned 38 items; 19 had an evidence sentence that does not contain the keyword. Those two documents got one main keyword between them. |
| v1 | two lists; 3 + 12 (15 in all, the extractor's budget); "a sentence in which that keyword itself appears"; output cap 2,048 | All three parsed. 2 to 3 main keywords each. 44 returned, 43 kept, 1 refused (too long). The model's own sentence was kept for 25 of 43; 18 were looked up (3 not verbatim, 15 without the keyword). |
| v2 | the two lists renamed `main` and `detail`, and "terms" reworded as "names" (the GLOSSARY) | **Not run yet.** Its first run is gate G2. |

**v2 is frozen before the measurement.** A change made after any sample document's output has been
seen is a new version, and it voids the sample as a test set (§9).

## 5 · Validation: offline, deterministic, before anything is kept

`validate(parsed, scan_input) -> list[ScanItem]`, in order, per item. The first check that fails
gives the refusal reason, stored with the item:

| # | Check | Refusal |
|---:|---|---|
| 1 | The output parses and has the shape. If not, the **run** is `invalid` and nothing is kept. | — |
| 2 | Name = `" ".join(keywords.tokenize(keyword))`, the name a TF-IDF keyword would get, so labels and keyword families match by name | `empty` |
| 3 | At most 6 tokens (a keyword is a name, not a clause; the prompt asks for 1 to 5 words) | `too long` |
| 4 | 5c's shapes, reused from `knowledge/keywords.py` and not re-implemented: a `MARKUP_TOKENS` token, no letter, `is_citation_artifact`, two or more single-letter tokens, a one-word name in the document's `cited_authors` | `markup` · `no letter` · `citation shape` · `letter-spaced` · `cited author` |
| 5 | The name occurs in the body's token stream | `not in body` |
| 6 | Not already kept for this document | `duplicate` |
| 7 | Within the cap, in the model's order (3 main, 12 detail) | `over the cap` |

Not reused from the extractor: the stopword rule (a model keyword may hold "of": *measurement of
animal behavior*) and `min_chars`.

**The sentence.** A kept keyword keeps the model's sentence when it is a verbatim part of what the
model read (whitespace collapsed, case folded) **and** it contains the name. Otherwise the sentence
is looked up: the first sentence that contains the name, in what the model read and then in the
whole body, using `definitions.sentence_spans` over paragraph blocks (headings skipped; the probe's
regex sometimes returned a heading, such as *Bottom-Up model.*). `evidence_from` records `model` or
`lookup`, and `model_evidence_fault` records why (`not verbatim` · `keyword not in it`). So every
kept keyword has a sentence from its document, and how often the model supplied its own is a
measured number.

**`uses`** = how many times the name occurs in the body. A count, shown; not a model score.

**The gates before any paid call**, each a test or a recorded run:

- **G1 — contract tests on fixture outputs, no model** (§13): malformed JSON, cut-off JSON (v0's
  failure), empty lists, extra fields (a `confidence` is dropped, never stored), a wrong type, an
  invented keyword, a paraphrased sentence, a cited author, a letter-spaced name, a duplicate,
  20 detail keywords, an accented name.
- **G2 — a live local run on five documents outside the sample**: the three above, the shortest
  note (`ISO Drafting made efficient.pdf`, 3 pages) and the longest book (365 pages). Every output
  parses or is recorded `invalid` with its reason; the time per document is recorded. Also run on
  `llama3.1:8b` (the private eval's generator), and pick the model by these numbers.
- **G3 — the whole library on the local model, writing a report and nothing else** (`--run`):
  §9's P1.
- **Then** the measurement (§9) on the local model. A paid run comes after that, on the same frozen
  prompt and the same 15 documents, and only on the user's word.

## 6 · Storage: the sidecar

Two tables in `db/models.py`, created by `create_all` like `concept_definitions` (additive, no
migration of an existing table). Only `knowledge/keyword_scan.py` writes them.

**`keyword_scan_runs`**: one row per document per attempt.

| Column | Type | Note |
|---|---|---|
| `id` | str (uuid) | |
| `document_id` | FK `documents.id`, cascade, indexed | ADR-047 identity |
| `doc_hash` | str | the text the scan read, for drift (like `figures.doc_hash`) |
| `input_version`, `input_sha256` | str | the view's rule and its exact bytes |
| `prompt_version` | str | `kwscan-v2` |
| `provider`, `model`, `model_digest` | str | the digest from Ollama's `/api/show` |
| `status` | str | `ok` · `empty` (nothing kept) · `invalid` (shape) · `failed` (call error) |
| `error` | text, nullable | |
| `raw_output` | text | the model's answer as received, so validation can be re-run without a call |
| `prompt_tokens`, `output_tokens` | int, nullable | the provider's own counts |
| `seconds` | float | |
| `started_at`, `finished_at` | datetime | |
| `is_current` | bool | at most one per document |

**`keyword_scan_items`**: what one run proposed, kept and refused.

| Column | Type | Note |
|---|---|---|
| `id` | str (uuid) | |
| `run_id` | FK `keyword_scan_runs.id`, cascade | |
| `document_id` | str, indexed | denormalised for the document view |
| `kind` | str | `main` · `detail` |
| `place` | int | order within its kind, as the model gave it |
| `keyword` | text | as the model wrote it (the document's spelling) |
| `name` | str | derived (§5 check 2); recomputed when the tokeniser changes (ADR-057) |
| `evidence` | text | verbatim |
| `evidence_from` | str | `model` · `lookup` |
| `model_evidence_fault` | str, nullable | |
| `uses` | int | |
| `verdict` | str | `kept`, or the refusal reason |

Unique: (`run_id`, `kind`, `place`).

**Writes.** One `session_scope` per document. An `ok` run inserts its row and items and moves
`is_current` to itself; an `empty`, `invalid` or `failed` run inserts its row only, and the previous
current run stays current. An exception inside the transaction rolls all of it back.
**Idempotent:** a document whose current run has the same `input_sha256`, `prompt_version`,
`provider` and `model` is skipped unless `--force`.

**Current or stale.** A scan is current while the view built from today's cache hashes to its
`input_sha256`. A re-extraction that changes what it read (ADR-056's rejoin, a `text` re-ingest,
ADR-048) makes it stale. The runner lists stale documents, and the Library treats a stale scan as no
scan (§8) until it is redone.

**`--revalidate`** re-runs §5 on every stored `raw_output` with no model call. A validator change
then costs seconds, not a re-scan.

## 7 · The runner

`scripts/scan_keywords.py`, thin; the logic is in `knowledge/keyword_scan.py`.

| Flag | Meaning |
|---|---|
| (none) | **Plan**: builds every input, prints the number of documents, input tokens, the time and cost statement (ADR-048), the stale and unscanned documents. Calls no model, writes nothing. |
| `--run --out FILE` | Calls the model, writes a JSON report to `FILE` (outside `data/`), writes nothing to the library. The measurement mode. |
| `--apply` | Calls the model and writes the sidecar. |
| `--provider {ollama,anthropic}` | **Required** for `--run` and `--apply`; no default (KI-4). |
| `--model NAME` | **Required** in part 1. |
| `--document-id ID` (repeatable) · `--sample FILE` | Restrict the documents. |
| `--force` | Re-scan current documents. |
| `--revalidate` | §6; no model call. |

Before the first call: `llm.ollama_probe`, then `llm.ollama_context_length(model)`. If the model's
window is below 8,192 the runner refuses and says so. It never lets the server cut the prompt
(KI-65). A paid provider: `assert_provider_intent(operation="keyword scan", scope="N documents")`
before anything is sent.

## 8 · How it reaches the Library and keyword families

**Part 1 changes nothing the Library shows.** The review surface is read-only: `GET
/api/library/documents/{id}/keyword-scan` returns the current run (model, prompt version, date,
what was read: the title, the opening, N headings, N passages), the kept keywords with their
sentence and `uses`, and the refused ones with their reason; "not scanned" or "stale" otherwise.
The document view shows it as a collapsed *Model proposals* section. That is session 10's "the
review surface shows what the pass proposed".

**Part 2, only after §9 passes and the user's yes: one writer of keyword links, two sources.**

- `extract_keywords` gains `source: tfidf | scan | auto` (CLI `--source`), default `tfidf` until the
  user switches. Under `auto`, a document with a current `ok` scan gets its links from the scan;
  every other document gets TF-IDF as now. TF-IDF statistics still run over the whole corpus.
- A scan's links: `Keyword` get-or-create by name, `source="extracted"` as today, so `--force`, the
  orphan sweep, Manage keywords and promotion keep working unchanged. `rank` = the main keywords in
  order, then the detail keywords in order; `score` = NULL, because the model gives no score and the
  rank carries the order.
- `document_keywords` gains one nullable column, `origin` (`tfidf` · `scan`), an additive
  migration through `init_db`, so the Library can say where a document's keywords came from.
- **ADR-055 applies to both origins.** Its per-document reader puts the short form beside a scan
  keyword that is a long form, and the long form beside one that is a short form.
- **The document view:** keywords in rank order, so main keywords first; one line on where they came
  from (*Proposed by qwen2.5:7b on <date> from the title, the opening, 62 headings and 18
  passages*); each keyword's sentence on hover (ADR-053: evidence shown). The tile, the filter and
  search read names, as now.
- **Keyword families** count documents through `document_keywords` by name
  (`library/keywords.py:57-70`), unchanged. Switching moves those counts: measured before and after
  and listed per family, as 5c did (`.claude/probes/2026-10-20-keyword-cleanup/groups.py`).
  Concepts, forms, presence, the graph and gaps do not read keywords and must not move (5c's
  `invariants.py`, reused).
- A user's manual keyword is never touched (as `_persist_keywords` does today).
- The switch is a write: a backup, the user's yes, a read-back.

**At ingest (part 2):** an opt-in setting, off by default, *Scan new documents' keywords with a
local model* (ingestion options, plan §2 question 6). It runs the same function on each new document
after its text is cached, in the background, and never holds up the ingest. If Ollama is not
running, the document is ingested with TF-IDF keywords and the Library says *not scanned* and why.
No paid provider is offered at ingest: a control that spends money is its own deliberate action
(ADR-048).

## 9 · Measurement against session 5c's baseline (pre-registered)

**The control:** 5c's TF-IDF list after the 2026-10-22 re-extraction, on the seeded 15
(`.claude/probes/2026-10-20-keyword-cleanup/sample.json`, seed 20261020), reproduced on 2026-10-23
with that folder's `measure.py`:

| | Precision, mean over documents (t, 15) | Main topic present |
|---|---:|---:|
| Before 5c | 62% [46, 78] | 1 of 15 |
| After 5c (the control) | 64% [50, 77] | 2 of 11 answered |
| Change | +1 point [−4, +6] | gained in 2, lost in 1 |

**The procedure:**

1. **Freeze** prompt `kwscan-v2`, input `kwscan-input-v1`, the model and its digest, and the
   sampling, in the baseline file **before** the run.
2. **Run** the 15 with `--run --sample sample.json`, which writes nothing, **twice** for stability.
   The control is the extractor's labelled list on the same day's cache. If ADR-055, 056 or 057 has
   changed it since 2026-10-22, the control is re-derived and its new keywords are labelled too, so
   both arms read the same text.
3. **Label** on the same page (https://claude.ai/artifact/R69223DKng1AUcgzHc6oHJ), new round
   `scan`, same rule: *Topic* = what the paper is about or introduces; *Detail* = a real narrower
   term, an abbreviation counted by what it stands for; *Noise* = not a term (split or glued words,
   table cells, cited names, dates and numbers, example words). A keyword already labelled for that
   document (same name) keeps its label; only new names are asked, in alphabetical order, with no
   kind and no sentence shown (5c's rounds showed neither). Per document: main topic *present* /
   *partly* / *missing* for the scan's list. The first label is read back from the store before the
   rest (memory *input pages: check the first save*); the record is the pasted JSON.
4. **Compute** with `measure.py` given a third arm: the document is the unit; a Student t interval
   over the 15 per-document differences; Wilson for shares of documents; the keyword-pooled
   interval printed only as "as if keywords were independent — too narrow".
5. **Write** `tests/eval/baselines/keyword_scan_<date>.md`: indicative, every document shown, model
   and digest, prompt version, machine, date (memory *benchmark presentation tone*).

**Predictions, registered 2026-10-23, before the scan has read any sample document:**

| # | Prediction | Control |
|---|---|---|
| P1 | Mechanics, 104 documents, local model: at most 2 end `invalid` or `failed`; every kept keyword occurs in its document (by construction); the model's own sentence is kept for at least half of the kept keywords; median time at most 30 s per document with the GPU free | development: 3 of 3 parsed (v1), 25 of 43 sentences the model's own |
| P2 | **Primary.** Mean precision over the 15 at least 85%; the per-document difference, scan minus control, at least +15 points, with its 95% interval above 0 | 64% [50, 77] |
| P3 | Main topic *present* in at least 10 of 15 documents | 1 of 15 before; 2 of 11 answered after |
| P4 | *Topic* + *Detail* keywords per document at least 12 on average | 143 / 15 = 9.5 |
| P5 | No scan keyword labelled *Noise* for being a cited name, a date or number, a web address or a table cell. Expected remaining noise: extraction damage copied as written, or a word too general | 5c removed 37 such of 44, and 34 of the 44 replacements were noise |
| P6 | Stability: across the two runs, the median document keeps at least 80% of its names | — |

**Decision rule (registered):** the Library's default moves to `auto` only if P2's interval lies
above 0 **and** no document loses a main topic that the control had *present*. Then the user
decides. If P2 fails, the scan stays in the review surface, the result is recorded, and a paid arm
(the same frozen prompt on the same 15; about $0.15 on Haiku 4.5) is the user's call.

**What 15 documents can show:** the interval on a mean is about ±14 points (the control's
[50, 77]). A difference under about 10 points cannot be told from noise; 5c's +1 was such a case.

**What voids it:** a prompt or input change after a sample output was seen. Then a fresh seeded
sample of 15 is drawn from the other 89 documents (seed recorded) and labelled.

## 10 · Cost and time, at 0, 104 and 10,000 documents

**Measured 2026-10-23:** RTX 4070 (12 GB), Ollama 0.40.9, `qwen2.5:7b` (Q4_K_M), window 8,192,
three development documents. **18 to 26 s per document** in four runs (reading the prompt about
2 s, writing 730 to 1,260 tokens 14 to 20 s, the first load 7 s). Two more took 96 s and 196 s
while a game held the GPU at 99%: the time depends on what else is on the card. Per document:
about 5,200 tokens in, about 800 out (v1).

| | 104 documents | 10,000 documents |
|---|---:|---:|
| Local, `qwen2.5:7b`, this GPU free (≈ 22 s each) | ≈ 40 min | ≈ 61 h of GPU (2.5 days), one call per document |
| Tokens | 0.54 M in · 0.08 M out | 52 M in · 8 M out |
| Haiku 4.5 ($1 / $5 per M) | ≈ $0.96 (batch ≈ $0.48) | ≈ $92 (batch ≈ $46) |
| Haiku 5.5 ($0.10 / $0.50 per M, prompts under 100K) | ≈ $0.10 | ≈ $9 (batch ≈ $4.60) |
| The 15-document sample on Haiku 4.5 | ≈ $0.15 | — |

Prices: the Claude API reference as cached on 2026-10-06; Anthropic counts tokens differently from
Ollama, so a paid run counts its input first (`count_tokens`). Estimates, not quotes.

**Storage at 10,000:** 150,000 item rows and 10,000 raw outputs of about 3 KB, about 30 MB in
SQLite.

**At 0 documents:** the runner prints "0 documents, nothing to scan", exits 0 and writes nothing;
the review API answers *not scanned*; `auto` with no scan is TF-IDF everywhere, which is today.
**At 1 document:** works; the scan has no corpus statistic.

**A risk on the paid path, outside this spec:** `llm.AnthropicClient.complete` always sends
`temperature`. The pinned SDK (`anthropic` 0.103.1, API version 2023-06-01) accepts it. From SDK
2.0.0 (API version 2026-10-21) every Messages request carrying `temperature` returns a 400, so every
paid path in this project breaks on that upgrade, this one included.

## 11 · What this spec does not decide

- **Row 25's other parts**: figure and citation links, reference order (the additive ordinal at
  extraction time), concept marking at ingest (row 56), S-7. They belong to row 25's own spec. They
  can read the same input view (§3); each would be its own prompt and its own sidecar.
- **93b, model definition candidates.** A recommendation for row 25's spec: a separate
  per-concept runner, because a definition candidate reads passages about one concept from many
  documents, and this scan reads one document. A main keyword's sentence can be offered to 93b as a
  passage candidate, with its source.
- **B2, where terms come from** (plan, vocabulary ledger): the scan becomes one candidate source;
  promotion (`seed_concepts`) keeps reading `Keyword` names. Settled with B2 in row 25's spec.
- **ADR-006's `contrastive` mode**: still its own experiment. Under `auto` it only matters for
  unscanned documents.
- **Reading long documents section by section** (ADR-058 option 3).

## 12 · Work items

| # | Item | Files | Acceptance |
|---:|---|---|---|
| 1 | Tests first (§13, part 1) | `tests/unit/test_keyword_scan.py`, `tests/integration/test_keyword_scan.py` | red, then green |
| 2 | Input view, prompt, schema, validator, one-document run | `src/doc_assistant/knowledge/keyword_scan.py` | §3–§5; pure core without DB or LLM imports |
| 3 | The two tables | `src/doc_assistant/db/models.py` | `create_all`; the migration test sees them |
| 4 | `format` on the Ollama client | `src/doc_assistant/llm.py` | a schema reaches the server; `"json"` stays the default |
| 5 | The runner | `scripts/scan_keywords.py` | §7; provider required; plan mode calls nothing |
| 6 | The review API and section | `src/doc_assistant/library/`, `apps/api/routers/library/`, the document view | §8 part 1; "not scanned" at 0 |
| 7 | G2, G3, then §9 | `.claude/probes/`, `tests/eval/baselines/keyword_scan_<date>.md` | P1–P6 reported against their registration |
| 8 | **Part 2, after the user's yes:** `origin` column, `--source`, the Library's origin line and sentences, families before and after, the ingest option, the re-extraction | `db/migrations.py`, `knowledge/keywords.py`, `library/documents.py`, desktop | §8 part 2; invariants unchanged |

## 13 · Test cases (write these first)

| Name | Assertion | Path |
|---|---|---|
| `test_input_fits_the_window` | at window 8,192 the view is at most 5,488 tokens (chars / 4); headings at most a tenth, the opening at most a quarter; the same document gives the same hash twice | `tests/unit/test_keyword_scan.py` |
| `test_input_uses_the_extractors_cuts` | a reference section and a running header are not in the view | same |
| `test_input_short_document` | a one-paragraph document gives a title and an opening, no passages, no error | same |
| `test_rejects_invented_keyword` | a name absent from the body is refused `not in body` | same |
| `test_sentence_looked_up_when_paraphrased` | a paraphrased or keyword-less sentence is replaced by a body sentence that holds the name; `evidence_from="lookup"` with its fault | same |
| `test_reuses_5c_shapes` | `br`, letter-spaced, a cited author, digits only: each refused with its reason, through `keywords.py`'s functions | same |
| `test_caps_and_duplicates` | 5 main → 3 kept, 2 `over the cap`; a repeated name → `duplicate` | same |
| `test_cut_off_json_is_invalid` | a `done_reason: length` answer → run `invalid`, no items | same |
| `test_no_confidence_stored` | an extra `confidence` field is dropped | same |
| `test_run_is_atomic` | an exception after half the items leaves the previous current run and no item of the failed run | `tests/integration/test_keyword_scan.py` |
| `test_idempotent_skip` | same input hash, prompt, provider and model → no new run without `--force` | same |
| `test_stale_after_text_change` | a changed cache makes the scan not current | same |
| `test_zero_documents` | an empty library: exit 0, nothing written | same |
| `test_provider_is_required` | no `--provider` → non-zero exit; `anthropic` without a key → `ProviderCostError`; `anthropic` with a key → the banner before any call | same |
| `test_window_too_small_refuses` | a model window below 8,192 → refused with a message, no call | same |
| `test_chunk_store_untouched` | chunk rows, vector-store counts and cached markdown bytes identical before and after `--apply` | same |
| `test_revalidate_needs_no_model` | `--revalidate` rebuilds items from `raw_output` with the client patched to fail if called | same |
| `test_review_api` | kept and refused items with sentences; "not scanned" with none | `tests/integration/` (API) |
| `test_auto_source_per_document` (part 2) | scanned document → links `origin=scan`, main first; unscanned → `tfidf` | `tests/integration/test_extract_keywords.py` |
| `test_manual_keyword_kept` (part 2) | a manual keyword survives `--source auto --force` | same |
| `test_origin_column_additive` (part 2) | old links read NULL | `tests/integration/test_migrations.py` |

## 14 · Estimate

- **estimate:** M for part 1 (with the measurement), M for part 2. Planner judgement, not a
  measurement.

## 15 · Docs to touch when this lands

`docs/DEVLOG.md` · `CHANGELOG.md` · ROADMAP row 25 (the keyword part) · `docs/performance.md` (a
row for the scan's cost) · `docs/knowledge-layer.md` (what a model-proposed keyword is evidence of)
· `GLOSSARY.md` (*keyword scan*, *main keyword*, *detail keyword*) · `docs/decisions.md` (ADR-058's
line, after ADR-055 to 057's) · `.claude/SESSION.md`.

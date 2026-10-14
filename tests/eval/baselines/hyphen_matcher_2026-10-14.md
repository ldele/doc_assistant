<!-- status: active · updated: 2026-10-14 (KI-63: a space, a hyphen or a dash joins the words of one form; restates the "never in body prose" count of term_signals_2026-10-14.md §4) · class: baseline -->

# One form, its words joined any way: what moved when the matcher stopped being literal (KI-63)

**On the graph's 13 concepts no document, link or gap row moves, and the graph version is the
same; four concepts count a few more mentions.** Over the whole vocabulary, 38 concept–document
pairs are gained and none is lost: `cosine-similarity` is present in 16 documents instead of 1 and
`state-space` in 11 instead of 2. Three labels the term-signal review called "never in the text"
are in it (`ssp bfd`, `inter species`, `query document`). One definition candidate is gained.

These are counts on one library. They say what the change did here, not how often a term is
written two ways in another library.

## The rule

A form's characters are matched literally, except where a space, a hyphen or a dash joins two of
its words: there any of them matches, with whitespace around a hyphen allowed (a line broken as
"inter- species") and at most one line break (a wrapped line joins two words; a paragraph break,
like the eLife page header "Tools and resources", blank line, "Neuroscience", does not). The dashes
are U+002D, U+00AD soft hyphen, U+2010, U+2011, U+2012 and U+2013. Nothing else loosens: no join
without a joiner (`sspbfd`), no underscore or dot, word edges as before, and case as before
(ADR-053 decision 3: `SSp-bfd` and `SSp bfd` are one spelling, `SSP-BFD` another).

One definition (`concept_skeleton.form_source`) serves every reader: presence, broad forms, the
written-form vote, the definition and naming patterns, epistemics, the term signals, and the
keyword-index query of "Look in my library", which now asks a label with every mix of spaces and
hyphens (the index keeps "SSp-bfd" as one token). A written form is voted on one line, so
"SSp- bfd" as extracted is recorded as `SSp-bfd`. Two forms of one concept that differ only in
their joiners (`cross-encoder`, `cross encoder`) count a mention once.

## Environment / method

- **Library:** the working library, 104 documents, 357 vocabulary rows (13 concepts on the graph),
  8,246 prose parents.
- **The comparison:** the committed code (`0022128`, extracted with `git archive`, pointed at the
  same library with `DOC_DATA_DIR`) against the change, each as a dry run:
  `build_concept_skeleton(apply=False)` and the gap detectors, `match_presence` over all 357 rows,
  `derive_written_forms`, `extract_definitions(apply=False)` for the 13 concepts and for every row,
  `term_signals` over every row, and `chunks_mentioning` for every label against the full read.
  Probe: `.claude/probes/2026-10-14-hyphen-matcher/` (`reader_dump.py`, `reader_compare.py`,
  `graph_mentions.py`, the dumps and `compare_2026-10-14.txt`; local-only). Nothing was written to
  the library or the stored graph.
- **Cost:** $0. No model.

## 1 · The graph's 13 concepts

| | Before | After |
|---|---:|---:|
| Graph version | `58136820706f4cab` | `58136820706f4cab` |
| Links | 25 | 25 (all unchanged) |
| Documents on the graph · concept–document pairs | 29 · 72 | 29 · 72 |
| Gap rows | 12 | 12 (the same) |
| Mentions, summed over all prose parents | 3,688 | 3,713 |

The mentions move on four concepts and nowhere else: `cre` 376 → 395 (its form `cre recombinase`
now counts "Cre-recombinase"), `hard negatives` 51 → 53, `passage retrieval` 191 → 193 and
`re-ranking` 275 → 277. `cross-encoder` and `retrieval-augmented generation`, which each have a
form joined the other way, are unchanged: a mention is counted once. No definition candidate of
the 13 moves (20 → 20).

## 2 · The whole vocabulary (357 rows)

| | Before | After |
|---|---:|---:|
| Concept–document pairs | 1,079 | 1,117 (+38, none lost) |
| Written forms | 366 | 370 (8 spellings change) |
| Definition candidates | 110 | 111 (+1) |
| Pairs past the bibliography cut | 111 | 115 |
| Rows "never in body prose" (term signals) | 31 | 28 |
| Long or short forms offered (term signals) | 51 | 50 |

- **Pairs gained, by row** (presence reads every prose parent, reference lists included):
  `cosine-similarity` 15, `state-space` 9, `cc-by-nc` 2, `query
  document` 2, `inter species` 2, `query-passage`, `query-passage pairs`, `cre-dependent`,
  `mamba-unet`, `ssp bfd`, `distance-dependent`, `program-synthesis` and `in-train` 1 each.
- **Written forms that change spelling** (the vote follows the more common way of joining):
  `cosine-similarity` is written `cosine similarity` (42 uses in 11 documents, against 9 in 1),
  `state-space` → `state space`, `kv-cache` → `KV cache`, `program-synthesis` → `program synthesis`
  (no longer title case, so it matches in any case), `cc-by-nc` → `CC BY-NC`,
  `cre recombinase` → `Cre-recombinase`, `query-passage` → `query–passage` (en dash), and the alias
  `retrieval augmented generation` → `retrieval-augmented generation`. New: `ssp bfd` = `SSp-bfd`,
  `inter species` = `inter-species`, `query document` = `query–document`, and the alias
  `cross encoder` = `cross-encoder`.
- **Definition candidate gained:** `trec-car`, "Introduced by Dietz et al. in 2017, TREC CAR is a
  synthetic data…" (ColBERT).
- **The offer that goes** is a merge: `RAG` was offered twice to `retrieval-augmented generation`,
  once per spelling of the long form, and is offered once.

## 3 · A correction to `term_signals_2026-10-14.md` §4

That table counted 2 of 31 "never in the text" labels as in the prose with a hyphen. It is **3**:
`query document` is written "query–document" with an en dash, 29 times in 3 documents, and the
lookup there did not try an en dash. It was classed with the figure terms.

## 4 · Not changed by this, and still open

- **The keyword-index path misses one document on each of 4 rows**, before and after: `cre`,
  `pharyngeal`, `retriever`, `text embeddings`. The full read counts a term at the end of a
  hyphenated word ("retriever" in "G-retriever", in a reference title), because a hyphen is a word
  edge; the index keeps "g-retriever" as one token and its prefix query finds a term only at the
  start of one. Whether a term inside a compound is a mention at all is a separate question.
- **What reaches the app:** the stored graph and its written forms change at the next rebuild,
  which is the user's.

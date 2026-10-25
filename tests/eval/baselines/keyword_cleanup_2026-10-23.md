<!-- status: active · updated: 2026-10-24 (the first keyword-precision baseline; session 5c, ROADMAP 99) · class: baseline -->

# Keyword cleanup: precision before and after (ROADMAP 99, session 5c)

**The junk filters removed what they target, and precision did not move.** On 15 seeded documents,
the share of a document's keywords that are real terms went from 62% [46, 78] to 64% [50, 77]: a
change of +1 point [−4, +6]. With the user's own labels only, it went from 48% to 48% (+0 [−4, +4]).
37 of the 44 keywords the filters took out were noise, but 34 of the 44 that took their places were
noise too. TF-IDF's next candidates are no better than the ones removed, so cleaning the list
cannot raise its precision. The fixes that remain are better candidates (an LLM scan, ADR-058), not
more filters.

These are counts on one library, labelled by one person (with a share of the labels written by
Claude, stated below). They say how this extractor behaves on this library, not how precise keyword
extraction is in general.

## Environment / method

- **What changed** (commit `344b3d6`): five filters in `knowledge/keywords.py`. They drop a `br`
  token, a phrase with two or more one-letter tokens, a token with no letter, and a surname the
  document cites with "et al." on at least half of its uses; web addresses are removed before
  tokenising. `document_keywords` gained `score` and `rank`, and every read lists a document's
  keywords by rank. The mode (`per_doc` TF-IDF over 1–3-grams) and the cap of 15 are unchanged.
- **Library:** the working library, 104 live documents. Re-extracted 2026-10-22 with
  `python -m scripts.extract_keywords --apply --force` (backup `data/library.db.bak-20261022-pre5c`).
- **Sample:** 15 of the 103 documents that had extracted keywords, drawn 2026-10-20, before any
  filter was written: `random.Random(20261020).sample(sorted(ids), 15)`. The same 15 are reused for
  every later keyword change. Their filenames are in the table below; the ids are in
  `.claude/probes/2026-10-20-keyword-cleanup/sample.json` (local-only).
- **Labels:** on a private labelling page, one label per keyword: *Topic* (names what the paper is
  about, or what it introduces), *Detail* (a real, narrower term of the paper; an abbreviation
  counts by what it stands for), *Noise* (not a term: split, glued or letter-spaced words, table
  cells, cited names, dates and numbers, words from a worked example). Plus, per document: is its
  main topic among the keywords (*present*, *partly*, *missing*). Labels are keyed on the document
  and the keyword's text, so a keyword kept by the re-extraction keeps its label and was not asked
  again.
- **Who labelled:** the before round has 225 keywords and 15 main-topic answers. The user labelled
  them 2026-10-21. The rule above was agreed with the user after the first 149 labels showed real
  terms marked *Noise*. On 2026-10-22, at the user's request ("Est-ce que tu peux le faire pour
  moi ?"), Claude applied that rule to a list of changes the user had seen: 66 keyword labels and
  3 main-topic answers. Each is marked `by: claude` in the store, with the user's original in `was`.
  The after round was labelled by the user on 2026-10-23: 44 new keywords and 11 of the 15
  main-topic answers. The other 4 (documents 3, 6, 7 and 10) are missing.
- **Not blind:** the labeller knew which round was which, and the after round asked only the new
  keywords.
- **Statistic and intervals:** precision is (*Topic* + *Detail*) over all labelled keywords, per
  document. The unit is the **document**, because keywords cluster within a paper. With 15 units,
  each round is a Student t 95% interval over the per-document values, and the before/after change
  is a t interval over the 15 per-document differences, every document shown. A keyword-level
  Wilson interval is printed only as what independence would claim, and it is too narrow.
- **Regenerable:** `measure.py`, `sensitivity.py`, `post.py` and `invariants.py` in
  `.claude/probes/2026-10-20-keyword-cleanup/` (local-only), with the store snapshots beside them.
- **Cost:** $0. No model.

## Results

| # | Document | Before: real / labelled | After: real / labelled | Δ precision | Main topic before → after |
|---:|---|---:|---:|---:|---|
| 1 | `1909.13868v2.pdf` | 12 / 15 (80%) | 12 / 15 (80%) | +0 pts | partly → present |
| 2 | `2403.01590v1.pdf` | 14 / 15 (93%) | 14 / 15 (93%) | +0 pts | partly → present |
| 3 | `2404.16130v2.pdf` | 8 / 15 (53%) | 7 / 15 (47%) | −7 pts | partly → — |
| 4 | `2604.13349v1.pdf` | 15 / 15 (100%) | 15 / 15 (100%) | +0 pts | partly → partly |
| 5 | `cajal-lecture.pdf` | 1 / 15 (7%) | 4 / 15 (27%) | +20 pts | missing → missing |
| 6 | `coffee_automaton_aaronson_2014.pdf` | 13 / 15 (87%) | 13 / 15 (87%) | +0 pts | partly → — |
| 7 | `elife-84566-v2.pdf` | 4 / 15 (27%) | 7 / 15 (47%) | +20 pts | partly* → — |
| 8 | `fnana-19-1672016.pdf` | 9 / 15 (60%) | 9 / 15 (60%) | +0 pts | partly → missing |
| 9 | `fox2003_pddl2.1-temporal-planning.pdf` | 10 / 15 (67%) | 10 / 15 (67%) | +0 pts | partly → partly |
| 10 | `fpos-6-1305055.pdf` | 9 / 15 (60%) | 9 / 15 (60%) | +0 pts | partly → — |
| 11 | `gao2024_retrieval-augmented-generation-survey.pdf` | 4 / 15 (27%) | 4 / 15 (27%) | +0 pts | partly → partly |
| 12 | `hyde_gao_2022.pdf` | 12 / 15 (80%) | 10 / 15 (67%) | −13 pts | partly → partly |
| 13 | `order_matters_vinyals_2015.pdf` | 13 / 15 (87%) | 12 / 15 (80%) | −7 pts | partly → partly |
| 14 | `resnet_he_2015.pdf` | 4 / 15 (27%) | 4 / 15 (27%) | +0 pts | present* → missing |
| 15 | `s41592-022-01443-0.pdf` | 12 / 15 (80%) | 13 / 15 (87%) | +7 pts | partly* → missing |

\* a before answer written by Claude under the agreed rule; the after answer is the user's.

| | Before | After | Change |
|---|---:|---:|---:|
| Precision, mean over documents (agreed rule) | 62% [46, 78] | 64% [50, 77] | **+1 [−4, +6] pts** |
| Precision, the user's own labels only | 48% [35, 62] | 48% [37, 60] | +0 [−4, +4] pts |
| Pooled, as if keywords were independent (too narrow) | 140 / 225, 62% [56, 68] | 143 / 225, 64% [57, 70] | |
| Documents whose keywords changed | | 11 of 15 | +2 [−5, +9] pts over those 11 |
| Up / same / down | | 3 / 9 / 3 | |
| Main topic present or partly, on the 11 answered twice | 10 | 7 | the user's own answers: 8 → 7 |
| Main topic fully present | 1 of 15 | 2 of 11 | |

- **Where the filters acted, they worked.** On the Cajal lecture, 13 letter-spaced fragments went
  and precision rose from 7% to 27%. On the eLife paper, five table-residue and author-name keywords
  went and precision rose from 27% to 47%. Neither main topic came back.
- **Why the total did not move.** The re-extraction took 44 keywords out of the sample: 37 labelled
  *Noise*, 4 *Detail*, 3 *Topic*. It put 44 in: 34 *Noise*, 10 *Detail*. The freed slots were
  refilled from the same candidate list, and its next entries were as noisy as the ones removed.
  ResNet's seven `conv 256`-style rows gave way to seven the user also marked *Noise*, among them
  `1e4`, `152-layer` and `billion flops`. The Cajal lecture's fragments gave way to cited names
  (`gehuchten`, `retzius`, `lugaro`) and an accent split (`ram n`).
- **The main-topic change is not a measurement of the filters.** Two of the three documents where it
  fell (14, 15) carry a before answer written by Claude and an after answer from the user, with the
  same keyword in both lists for ResNet (`residual nets`). With the user's own answers only, it goes
  8 → 7 of 11.
- **A judgement-call sweep.** 11 of the 44 new keywords marked *Noise* are terms of the paper under
  the agreed rule: `identity mapping` (ResNet), `read-process-and-write` (Order Matters), `connexions`
  (in the Cajal lecture's title), `reduced silver`, `ramifications`, `plexiform`, `contrieverft`,
  `dl19/20`, `dl20`, `perplexity`, `training error`. Read that way, the change is +6 [−1, +13] points:
  still an interval that touches zero.

### The rule does not measure precision of a term

One-word keywords are labelled real more often than multi-word ones: before, 74% of 121 against 55%
of 87 two-word and 12% of 17 three-word; after, 66% of 136 against 59% of 87. The rule asks whether a
keyword is a term of the paper, and `connexions` is one. It does not ask whether a keyword is precise
enough to name the notion, and the user's point (2026-10-24) is that it often is not: *"neuronal
connexion is not the same as connexion, which is too vague."* TF-IDF's multi-word picks are mostly
adjacent words (`chunk inference`, `c0 c1`), not terms. The next measurement adds a *Too vague*
class and re-applies it to this control (ADR-058's amendment).

## What the re-extraction changed in the library

Read back 2026-10-22 against the backup:

- **The five patterns:** 123 links before (`br` 30, letter-spaced 31, no letter 57, web address 5),
  0 after; keywords carrying a cited author 22 → 0.
- **Counts:** all 104 documents have 15 keywords. None has fewer than before; `03-Zuo2014`, which
  had none, now has 15. 1,560 links, 1,332 of them kept. 192 extracted keywords no longer linked
  anywhere were swept; none is a concept's label or alias.
- **Order:** every link is ranked 1–15 with a score, and scores never rise down the ranks. The
  sample's lists equal the dry run's, order included. The Library's cards and document view list
  keywords in that order (checked live: light, dark, 375 px, 0 console errors).
- **Drift, not only the filters.** 115 stored links already differed from what the committed
  extractor produced on the same cache (39 of 104 documents matched): the keywords dated from before
  later cache changes. The filters themselves changed 142 links.
- **Nothing the graph reads moved.** Graph version `dbf0e870aea6fed1`, the 22 forms the user marked
  exact or broad, concept presence (72 rows), every concept, gap, definition, citation and document
  table: row counts and a hash of every row identical before and after.
- **Keyword groups:** 26 changed their document count. 10 from the filters: `aclanthology`,
  `aclweb`, `bilateral br`, `cohan` and `moyle` lost their one document; `aggressive behavior`,
  `respiratory`, `sporns` and `mrr` gained documents, and `multi-agent` went from 1 to 2. 16 from
  drift, for example `passage retrieval` 3 → 2 and `retriever` 4 → 3.
- **Keywords still describe documents, not the library:** 1,445 of 1,496 extracted keywords (97%)
  sit on one document, against 1,436 of 1,483 before.

## What it does not show, and what is left

- **15 documents, one labeller, not blind,** and 69 before-round answers written by Claude. The
  change holds under both labellings, but the levels (62% or 48%) depend on the rule.
- **Noise the filters cannot reach:** names with an accent split by the tokeniser (`ram n`, `de tica`;
  the tokeniser is shared with search ranking, so changing it is eval-gated); cells of a survey's
  comparison table (`chunk inference`, `dataset-base`); affiliation and ethics-statement words
  (`pesquisa`, `colombia`); names cited without "et al." (`sporns`, `fox long`, `gehuchten`); words
  from a prompt example (`neochip`, `plaza`).
- **A cost of the no-letter filter:** a name written with a spaced number loses it (`sam 2`,
  `mrr 10`, `layer 5`); written joined (`sam2`, `gpt-4`, `16p11.2`) it stays.
- **A furniture gap:** alternating recto/verso running heads repeat on half the pages by
  construction (the Cajal lecture's: 17 of 35), below the furniture filter's half-of-pages floor.
  Counting odd and even pages separately would close it; ADR-056 point 8 removes these heads from
  keyword selection by their layout label instead.
- **Next:** ADR-055 (an abbreviation shows the long form its paper gives), ADR-056 point 8 (closed
  letter-spacing runs in headers, URLs and maths skipped by keyword selection), and ADR-058 (an LLM
  proposes each document's keywords with evidence), each measured against this baseline on the same
  15 documents.

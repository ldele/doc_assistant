<!-- status: active · updated: 2026-10-14 (RG-034: the user's answers on the term-signal cards; ROADMAP 93 refinement 4) · class: baseline -->

# Term signals: what a reading says about each one (RG-034, ROADMAP 93 refinement 4)

**The abbreviation signal holds: 44 of 44 long forms it offers were right in the sentence they came
from. "Followed by the same word" cannot be judged as it was asked, and it is not shown. "Et al."
found 5 surnames, all right, and misses names cited alone. "Never in the text" is wrong for 2 of 31
labels, which the text writes with a hyphen.** Each card was right *in its sentence*. Whether
it is right for the label across the library is a separate question: 19 of the 44 labels occur in
more documents than the one that spells them out, and one short form is spelled out differently in
different papers (`DNNs`, section 1).

These are counts on one library, read by one person. They say which signals are worth showing here,
not how precise each one is on another library.

## Environment / method

- **Signals:** `knowledge/term_signals.py` as committed in `7b8d55f` (read path, unchanged since the
  cards were built). The cards were built on 2026-10-09 from `scripts/term_signals.py --json` over the
  working library (104 documents, 357 vocabulary rows). They cover every row outside the 19 priority
  concepts: 109 cards. **They are the whole population of what the app would offer today, not a
  sample.**
- **Reading:** the user, on the "Term Signal Review" page (a private artifact, collection `labels`),
  2026-10-14 between 09:20 and 09:40. Three answers: yes / no / can't tell. Read back from the store
  (56 answers). A local copy is beside the cards (`.claude/probes/2026-10-09-term-signals/answers_2026-10-14.json`,
  local-only).
- **No planted-wrong cards.** Nothing tests whether a wrong card would have been caught. The user's
  notes show attention (two faults of the page itself were caught, section 1), but this is not a
  control.
- **The place a label occurs, the cited sentences, how paper-local abbreviations are, the hyphen
  reach and the ligature count** were measured by read-only probes on the same library
  (`.claude/probes/2026-10-09-term-signals/`: `where_silent.py`, `paper_scope.py`,
  `joined_evidence.py`, `hyphen_reach.py`, `ligatures.py`). The `silent` classes in section 4 are the
  agent's, from where each label occurs, not the user's answers.
- **Cost:** $0. No model.

## 1 · Abbreviations (`pair`): 44 of 44 right

| | Cards | Yes | No | Can't tell |
|---|---:|---:|---:|---:|
| A long form offered to a short label | 35 | 35 | 0 | 0 |
| A short form offered to a long label | 9 | 9 | 0 | 0 |
| — of which `initials` true | 34 | 34 | 0 | 0 |
| — of which `initials` false | 10 | 10 | 0 | 0 |
| **All** | **44** | **44** | **0** | **0** |

Wilson 95% interval on 44 of 44: **0.92 to 1.00**.

- **`initials` separates nothing here.** The 10 pairs whose letters do not land on word starts
  (`CO` = *Communicability*, `SAR` = *spatial autoregressive*, `DNNs` = *deep learning approaches*)
  were all right. As a filter it would have dropped 10 right offers. It stays as evidence on a pair
  and is never a filter.
- **Right in the sentence is not right everywhere.** 19 of the 44 labels occur in more documents
  than the pair does. `DNNs` is "deep ANNs" in one paper and "deep learning approaches" in another;
  the user's note: *"It only works paper-wide."* Across the library, 98 short forms are spelled out
  in two or more ways (initials-consistent). Most are spellings (`LLMs`, `CoT`); some are different
  meanings in one field: `SC` is superior colliculus, structural connectivity or subgraph
  centrality; `EC`, `PD`, `HD`, `MAP` and `EM` too. A paper spells out its own short form in 957 of
  1,687 uses (57%).
- **Two faults of the review page, caught by the user** (*"Case-sensitivity might be important"*;
  *"detection issue"*): the page marked the short form ignoring case and word edges, so it lit the
  `co` of "communication" and the `na` of "anatomically". The signal matches as written (ADR-053
  decision 3). Whatever shows an offer must mark it the same way.
- **Long forms carry the extraction's damage:** `PDCR` comes with bold markers inside
  ("P\*\* erturbation- \*\*d\*\* riven…"), `NeuN` with a line break, `mIoU` as
  "mean-intersection-overunion". Verbatim is right for the record (ADR-043); an offer needs a
  display form derived from it.
- **Outside the sample:** "figure (Fig. 4)" reads as a pair, because the letters of "Fig." occur in
  order in "figure". No vocabulary row matches it, so no card showed it.

## 2 · Followed or preceded by the same word (`after`, `before`): not measurable as asked

7 of 29 cards answered (3 no, 4 can't tell), then stopped. **The answers are kept and not counted.**

- **The cards showed a count and nothing to read**: no sentence, no paper. Judging them needs the
  field.
- **The question held two questions.** The user's "no" on *chatbot*, *bibliometric* and
  *corticospinal* reads as "the word is a term on its own as well". The question meant "the word is
  only a fragment of the longer term". One answer cannot say both.
- **The library does not settle them either.** 1 of the 29 joined phrases is spelled out anywhere
  (*atrial fibrillation*, `AFIB`). Most are repeated inside one paper (*hyaline grumes* 14 times,
  *ordinary personas* 32, *accessory body* 12): terms of that paper. Capitals mark the names
  (*Chatbot Arena* 16 of 17, *Shitao Xiao*, *Usage Cards*).

**Not shown in the app.** If it comes back, it is as a candidate for one paper's glossary, shown
with its sentences.

## 3 · Cited authors (`cited`): 5 of 5 right; names cited alone are missed

The user answered 5 of 5 yes after reading one "X et al." sentence for each, given in the
conversation; the cards had none. Wilson 95% interval: 0.57 to 1.00.

- **The signal reads vocabulary rows, not papers.** Body prose cites 1,742 distinct surnames with
  "et al.", in 77 of 104 documents. Seven of them became vocabulary rows; the rule (half or more of
  the mentions) catches five.
- **Missed:** `sporns`, 8 of 73 mentions (mostly cited alone: "Sporns, 2011") and `morecraft`,
  2 of 5. A name followed by a year in a citation is the same evidence as "et al.".

## 4 · Never in body prose (`silent`): wrong for 2 of 31; the rest classified by where they occur

Not answered: the user could not say "noise" without the context the card lacked. Each label was
looked up in every parent chunk, figures included, with a space in the label matching a space,
hyphen, underscore or dot:

| Where it occurs | Labels | Examples |
|---|---:|---|
| **In body prose, written with a hyphen** (the signal is wrong) | 2 | `ssp bfd` is SSp-bfd, 58 times in one paper; `inter species` is "inter-species", 17 times in 2 |
| **One paper's own term, in its tables or figures only** | 8 | `ap10k`, `superanimal transfer learning`, `superanimal memory replay`, `superanimal transfer`, `baseline-knn`, `relevance judgement`; from figures: `c-mtp unlabeled`, `query document` |
| **Noise by origin** | 12 | journal abbreviations in reference lists (`comput vis`, `conf comput`, `vis pattern recog`); page headers (`resources computational`, `resources neuroscience`, `bior`); code (`param1`–`3`, `agent direction`, `behavior analysis animals`); `fne-tune`, which is *fine-tune* with the "fi" lost |
| **Nowhere, even with hyphens** | 9 | phrases joined across punctuation or lines: `koonce emerson` and `schneider lee` are author pairs from reference lists |

Where a label occurs answers the card better than a reader can without context. Two of the classes
are issues beyond this signal:

- **KI-63:** the shared matcher is literal. 23 of the 183 multi-word rows are under-counted because
  the text writes a hyphen where the label has a space, or the reverse; 33 concept–document links
  are missing (`cosine-similarity` in 1 document instead of 11, `state-space` in 2 instead of 10).
- **KI-64:** 20 of 104 documents lost the "fi", "fl" or "ff" ligature at extraction somewhere;
  10 lost it 3 times or more, one 258 times ("fne").

## 5 · What it decides, and what it opens

- **Shown:** the abbreviation offers, each with its sentence and paper, marked as written. Cited
  authors, with their sentence. A silent label with the place it occurs (table, figure, references,
  page header), never the word "noise".
- **Not shown:** the neighbour signal.
- **Next for the signals:** the hyphen matcher (KI-63); the citation rule for single authors; a
  display form for damaged long forms; "Fig." excluded as a short form.
- **What is left needs judgement, not a rule:** which meaning a short form has in a paper, whether a
  repeated phrase is that paper's term, whether a silent label is a term a field takes as known.
  The user's direction (2026-10-14): an LLM with clear instructions, its prompt developed against
  this file's answers as the gold set in the eval harness, a local model first. And a glossary per
  paper before the library's, a field level only if two are not enough.

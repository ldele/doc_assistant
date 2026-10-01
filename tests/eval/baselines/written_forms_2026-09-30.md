<!-- status: active · updated: 2026-09-30 (ADR-053 decision 3 built and measured) · class: baseline -->

# How the library writes a concept's name — case included (ADR-053 decision 3, ROADMAP 93)

**Case-aware matching changes presence for 16 of 357 concepts and none of the user's 24 labelled
definition candidates.** Nine of the sixteen lose occurrences that are another word — `CRE` is not
`Cre`, "vitamin Din" is not `dIN`, the surname Colbert is not `ColBERT` — and seven lose the same
name spelt in lower case, mostly in bibliography titles. On the graph one concept moves: `cre`,
7 documents → 6. Its value is precision in presence and showing a word as the library writes it;
it is not a better definition score.

## Environment / method

- **Library:** the working library, 104 documents, 8,861 parent chunks; 357 text-bearing concepts
  (13 on the graph). Read-only: nothing was written, and the stored skeleton is unchanged until
  the user rebuilds.
- **The vote** (`knowledge/written_forms.py`): for each surface form (label and aliases,
  casefolded as presence matches them), its whole-word, case-insensitive occurrences in **body
  prose** (`definitions.document_sentences`: the bibliography cut, headings, tables and lists
  dropped), **not at the start of a sentence**. Each document votes for the spelling it uses most;
  the written form is the spelling most documents vote for; ties go to more uses overall, then to
  lower case.
- **Matching** (`concept_skeleton.word_case_key`): a form written in lower case matches in any case,
  exactly as before. A form written with a capital matches only spellings that differ from it in
  **word-initial** letters at most — a capital that starts a word is systematic (sentences,
  titles, headings), a capital inside a word is identity (`dIN`, `BM25`, `ColBERT`).
- **Yardstick:** the user's 69 labels (artifact db, collection `labels`, read with ArtifactData,
  69/69) — see `definition_labels_2026-09-22.md`.
- **Cost:** $0. No model.

## 1 · Choosing the rule — four candidates, measured

A read-only probe (session scratchpad) ran each rule against the case-folded presence on the same
library; the last row is the built implementation, run through the shipped code:

| Rule | Concepts that lose documents | Become single-source | Graph concepts |
|---|---|---|---|
| R1: the most-used spelling, matched exactly | 27 | 7 | cre |
| R2: capitalised words also match lower case; other shapes exact; occurrences vote | 23 | 6 | cre |
| R2, documents vote | 20 | 6 | cre |
| **word-initial rule, documents vote (built)** | **16** | **4** | **cre** |

- **R1 is wrong for ordinary words the prose sometimes capitalises.** `assistant` drops from 10
  documents to 4 and `plateau` from 12 to 2 (57% and 56% capitalised, narrowly). Every later rule
  keeps both.
- **Occurrences let one document outvote the rest.** `persona`: 123 uses of `PERSONA` against
  99 of `persona`/`Persona`, but one document votes `PERSONA` and three vote `persona`. Documents
  voting keeps it matched in any case; so does `cards`.
- **Title Case from titles.** Under R2 a phrase quoted from a title won the vote and the prose
  stopped counting: "a sentence encoder LSTM", "learning abstractions over the corpus",
  "Contemporary AI benchmarks", and the dataset cited as "Natural questions: a benchmark…". The
  word-initial rule treats those as the same words and restores all four.
- **What case cannot do:** a true homograph that both senses use a lot. `FEVER` (the
  fact-checking dataset) and `fever` (the symptom) stay merged, as before; that is meaning scope,
  ROADMAP 92.

## 2 · The built implementation — presence

367 written forms for 357 concepts, 155 of them with a capital; 326 labels have one, 141 with a
capital. **16 concepts lose documents, none gains one, 4 become single-source, none loses all
presence.** Every row below was checked against the text that stopped counting:

| Concept → written | Documents | What stopped counting |
|---|---|---|
| **Another word, a name, or noise — the point** | | |
| cre → Cre (graph) | 7 → 6 | `CRE`, the cAMP response element |
| din → dIN | 2 → 1 | "vitamin Din" (the user's note) |
| specter → SPECTER | 3 → 2 | *The Specter of Pandemic*, a book title |
| colbert → ColBERT | 6 → 3 | the surname Colbert (the numpy paper's author) |
| cer → Cer | 4 → 3 | `CER`, character error rate |
| sar → SAR | 2 → 1 | the surname Sar, in a reference |
| sts → STS | 9 → 5 | an author's initials, "StS", in contribution and funding statements (3 documents); OCR noise, "sts" for "its", in a scanned book (1) |
| ance → ANCE | 3 → 2 | an extraction duplicate, "variance ance" |
| ast → AST | 3 → 2 | OCR noise in a scanned index |
| **The same name in lower case — the price** | | |
| gpt-4 → GPT-4 | 9 → 6 | "…with gpt-4" in bibliography titles |
| deeplabcut → DeepLabCut | 8 → 7 | "deeplabcut" in bibliography titles |
| mamba-unet → Mamba-UNet | 3 → 1 | "Mamba-unet:" in bibliography titles |
| swin-unet → Swin-UNet | 4 → 2 | the variant "Swin-Unet" in prose, "swin-unet" in a title |
| unet → UNet | 5 → 4 | "Unet" inside other names (Swin-Unet, LeViT-Unet) |
| 3d medical → 3D medical | 6 → 5 | "3d medical" in a bibliography title |
| ai-generated → AI-generated | 2 → 1 | "ai-generated" in a bibliography title |

**Most of the price is paid in reference lists**: presence reads a document's whole text,
bibliography included, while the vote reads body prose only, so a name that a bibliography
lower-cases stops counting there. **Read the list before rebuilding**: the stored presence changes
only with `build_concept_skeleton --apply` or the Graph tab's Rebuild.

## 3 · The 19 priority concepts and the user's labels

- **Label spellings found:** `BM25`, `Cre`, `DBS`, `dIN`, `Ntsr1`, `SPECTER`; the other 13 are
  written in lower case (`pddl` too: one document writes `pddl` 42 times, one `PDDL` 20 times, and
  the tie goes to the spelling used more).
- **Definition candidates: 24 before, the same 24 after** — all labelled, 11 usable either way.
  None of them confused case.
- **Usage examples:** one dropped — `din`'s "…lack of vitamin **Din** which marked atrophy…", which
  the user labelled *other* with the note "Din vs dIN -> difference ?".

## 4 · The graph (13 concepts)

`cre` 7 → 6 documents; the other 12 do not move. Shown in their written case after a rebuild:
`BM25` (already stored so), `Cre`, `DBS`, `Ntsr1`. Checked live in the desktop dev app with the
graph response rewritten in the page (the stored skeleton predates written forms): the rail, the
panel heading, the node labels and the SVG's label read `Cre`, `DBS`, `Ntsr1`.

## What would change the verdict

- **Presence without the bibliography.** The seven losses are mostly lower-cased names in reference
  lists. Matching body text only would remove them — and would also stop reference-only mentions
  counting at all, which moves `single_source` more than this change does. Its own measurement.
- **All-caps renderings of a capitalised name** (a table header `NATURAL QUESTIONS`) do not count;
  that is the price of telling `CRE` from `Cre`. None occurs among the sixteen.
- A library where a capitalised name is a minority in most documents: the documents vote then keeps
  matching case-folded, as it should.

<!-- status: active · updated: 2026-09-18 (document-level sense split measured) · class: baseline -->

# Concept merges — which embedder, which threshold (ROADMAP 53)

What "the same concept" means for `curate_concepts --dedup`, measured before choosing the one
definition the merge preview and the merge now share. Part (3) of row 53, first half: the
counts are measured; the **hand score is not done** — the pair labels below are the agent's first
reading, for the user to confirm or overturn.

## Environment / corpus

- **Data home:** `data/library.db` on this box, 98 documents.
- **Vocabulary:** 357 text-bearing concepts (`kind="concept"`), none an extraction artifact;
  **2 of 357 carry a definition**, so "label + definition" is in practice the bare label.
- **Method:** `concept_curation.dedup_pairs`'s inputs (`merge_text` per concept), embedded with each
  model; all 63,546 pairs scored; `plan_merges` (the real union-find) run at each threshold.
  Scratch script, not committed — the numbers are reproducible with `suggest_concepts --near
  --model <m> --threshold <t>` and `curate_concepts --dedup --embed-model <m> --threshold <t>`
  (dry run).
- **Cost:** $0 — local embedders, no LLM, nothing written.

## Result

| model | median pair cosine | p99 | ≥ 0.85: concepts deleted (largest group) | ≥ 0.90 | ≥ 0.95 | ≥ 0.97 |
|---|---|---|---|---|---|---|
| **specter2** (the old preview) | **0.842** | 0.924 | **354 (one group of 354)** | 309 (302) | 66 (13) | 16 (3) |
| **bge-base** (the old merge) | 0.493 | 0.661 | 34 (5) | **0** | 0 | 0 |

At 0.80, bge-base gives 79 pairs → 75 deletions, largest group 15.

**Reading.** On short labels SPECTER2 compresses: half of all pairs score above 0.842, so its
0.85 would fold the vocabulary into one concept, and even 0.97 pairs `beta ~ alpha`,
`single-agent ~ multi-agent` and `political psychology ~ political science`. `config.py`'s
reason for SPECTER2 (bge "compresses same-domain concepts into ~0.6-0.7") was measured on
title/abstract text; on labels the reverse holds. bge-base spreads labels; at the old merge's 0.9 it
merges nothing on this library, which is why the destructive path never did damage — by accident,
not by design.

## Decision taken (2026-09-17)

The preview and the merge share **bge-base at 0.90** (`CONCEPT_MERGE_MODEL`,
`CONCEPT_MERGE_COSINE`): the old merge's effective values, so `--dedup --apply` behaves exactly as
before and `--near` finally previews it (0 pairs today). SPECTER2 stays the candidate-extraction
embedder (`CONCEPT_EMBED_MODEL`), where its reason applies. The dry-run report now prints the
largest merge group, because pairs chain.

**Not decided:** any other threshold. The pairs below say a threshold is the wrong tool.

## The 34 bge-base pairs at ≥ 0.85 — agent's first reading, not a hand score

`same` = one concept in two surface forms · `narrower` = one is a kind or part of the other (an
`is_a` candidate, not a duplicate) · `related` = different concepts.

| cosine | pair | reading |
|---|---|---|
| 0.897 | sentence embeddings ~ text embeddings | related |
| 0.896 | self-sorting memory ~ self-sorting | narrower |
| 0.892 | passages ~ passage | **same** |
| 0.890 | superanimal memory replay ~ memory replay | narrower |
| 0.888 | program-synthesis agents ~ program-synthesis | narrower |
| 0.887 | saturation ~ saturation index | narrower |
| 0.882 | pose ~ human pose | narrower |
| 0.882 | specter ~ specter2 | related |
| 0.880 | virus ~ viral | **same** |
| 0.878 | contrastive learning ~ contrastive encoder | related |
| 0.878 | query-passage ~ passage | related |
| 0.877 | res2net-50 ~ res2net | narrower |
| 0.875 | systems biology ~ systems biology neuroscience | narrower |
| 0.874 | passage retrieval ~ query-passage pairs | related |
| 0.873 | query-passage pairs ~ query-passage | **same** |
| 0.872 | mtl ~ mtl cls | narrower |
| 0.872 | behavior analysis animals ~ behavior analysis | narrower |
| 0.871 | glomerulus ~ glomeruli | **same** |
| 0.869 | re-ranking ~ passage re-ranking | narrower |
| 0.865 | speckles ~ nuclear speckles | narrower |
| 0.864 | cardiologist-level arrhythmia ~ cardiologist-level | narrower |
| 0.864 | events behavior ~ events behavior analysis | narrower |
| 0.864 | dins ~ din | **same** |
| 0.864 | apoptosis ~ cell death | narrower |
| 0.863 | abstractions ~ learning abstractions | narrower |
| 0.863 | nucleolar ~ nucleolus | **same** |
| 0.863 | salient object ~ salient object detection | narrower |
| 0.861 | beta activity ~ beta | narrower |
| 0.858 | oscillations ~ beta oscillations | narrower |
| 0.856 | c-mtp ~ c-mtp unlabeled | narrower |
| 0.855 | palato-pharyngeal ~ pharyngeal | narrower |
| 0.855 | shapley ~ shapley value | **same** |
| 0.854 | behavior analysis ~ events behavior analysis | narrower |
| 0.851 | resources neuroscience ~ biology neuroscience | related |

**7 same · 21 narrower · 6 related.** The duplicates are almost all morphology (plural, adjective,
a dropped head noun), and at 0.87 the chaining already folds `passage`, `passages`,
`query-passage` and `query-passage pairs` into `passage retrieval`.

## Context, measured the same day

The user's direction on reading the table (2026-09-17): a threshold is not enough — a concept needs
**context pairing**, and word forms that belong together (`virus` / `viral`) should be **pre-paired
lexically**, while context decides what each one means. Three context signals were tried on the 34
pairs, read-only and $0 (7,863 parent chunks, presence matched as the graph matches it):

| signal | result |
|---|---|
| **Context cosine** — mean bge-base embedding of up to 16 mention windows per term, the term masked | **does not discriminate**: 30 pairs with passages score 0.795–0.996; a fragment and its phrase share the same passages, so they score as one |
| **Shared documents** | informative but coarse: 22 of 30 pairs share exactly 1 document |
| **Stands alone** — mentions of the shorter label outside the longer one | **separates fragments from concepts**: `program-synthesis` 0 of 26, `systems biology` 0 of 34, `cardiologist-level` 0 of 14, `query-passage` 0 of 4, `speckles` 1 of 33 → fragments of the longer phrase; `pose` 651 of 733, `saturation` 223 of 244, `c-mtp` 74 of 75 → real concepts in their own right |

**Senses, from the passages themselves.** One label carries several meanings in this multi-domain
corpus, so a pairing by label is wrong before any threshold is chosen:
`viral` — "viral encephalitis" (infection), "viral injections" (a lab vector), "Neuroanatomy goes
viral!" (the web sense, in a title); `specter` — the SPECTER model and "The Specter of Pandemic" (a
political-philosophy book); `din` — the dIN neuron type and "vitamin Din" (text extraction of
"vitamin D in"); `beta` — beta oscillations, a `beta k` code variable, "b eta" (a split "beta
secretase"). Presence counts all of these as one concept today.

**Do a label's documents fall apart? (2026-09-18, read-only, $0.)** Document vectors mean-pooled
from the stored chunk embeddings (99 documents); a label's documents grouped by average linkage,
two groups counted as unrelated below the library's own 10th percentile of document-pair
similarity (0.716; median 0.831). Of 162 concepts present in 2+ documents, **3 split**:
`specter` (a benchmark paper vs a political-philosophy book — a true homograph), `abstractions`
(program-synthesis papers vs the same philosophy book — different senses), and `virus` (8
neuroscience/biology documents vs one political-science agenda — **the same sense in another
domain**, which must not be split). `viral`, `beta` and `din` stay one group: their senses differ
*inside* one domain, so a document-level signal cannot see them. **Domain is evidence about sense,
not the definition of it.**

**Four labels match no passage at a word boundary** (`behavior analysis animals`, `events behavior`,
`events behavior analysis`, `resources neuroscience`) — keyword shingles, not text.

The hand score is being collected on a review page outside the repo (verdict per pair: same /
narrower, either direction / related / different / depends on context, plus "not a real concept"
per term and a note for senses).

## What it opens

- **Hand score** (row 53 (3), second half): confirm or overturn the readings above.
- **The design answer is not a threshold** (user, 2026-09-17): lexical pre-pairing — inflection
  folded as surface forms, derivation (`virus`/`viral`) kept as a family — then a context check
  (the stands-alone count, shared documents, per-mention senses) and review before any write.
- **Presence is sense-blind** (the senses above) — ROADMAP 92.
- **The `narrower` pairs are `is_a` candidates** — input for ROADMAP 51's write path, where merging
  them would have destroyed exactly the hierarchy 51 is meant to record.

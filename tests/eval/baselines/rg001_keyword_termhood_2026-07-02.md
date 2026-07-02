# RG-001 keyword termhood — contrastive vs corpus_band vs per_doc (2026-07-02, R3)

**Reproducible/indicative, not a verdict.** Dry-run keyword extraction on the **real `data/`
corpus** (62 docs, post-R1 re-ingest — cache `grep "intentionally omitted"` = **0**, so KI-14 is
confirmed cleared here). $0, read-only (reads cached markdown, writes nothing). Config defaults
frozen a priori (ADR-006): `KEYWORD_WEIRDNESS_REF_CEILING=8.0`, `KEYWORD_CONTRASTIVE_MIN_CVALUE=0.0`,
`KEYWORD_NGRAM_MAX=3`, `KEYWORD_MIN_CHARS=3`.

**Why:** the two pre-R3 modes fail by construction (measured 2026-07-01): per-doc TF-IDF picks df≈1
per-paper cliques; `corpus_band`'s `df·(1+ln tf)` is monotone in df, so it grabs the most-shared =
most-generic terms. R3 adds `mode="contrastive"` — C-value nested discount × reference-corpus
weirdness (`wordfreq`). This is the head-to-head.

## corpus_band top-40 — dominated by boilerplate (the failure the plan predicted)

`state · brain · potential · computational · journal · effect · picture · defined · training · simple ·
map · type · values · left · total · known · area · author · behavior · regions · group · way ·
parameters · order · zhang · points · associated · rather · global · distinct · rate · among · point ·
chen · functions · critical · image · whether · four · best`

Generic academic words (`effect`, `simple`, `whether`, `four`, `best`, `rather`, `way`, `order`) and
even **author surnames** (`zhang`, `chen`) — almost nothing a curator would promote as a concept.

## contrastive top-40 — domain vocabulary surfaces

`superanimal · deeplabcut · dl5 · llms · outflux · ap10k · connectomes · osns · sporns · cebra ·
keypoints · elife · frontiersin · bm25 · mpoa · fnana · amadeusgpt · connectomics · mambavesselnet ·
imagenet · phate · connectome · bicommunities · pmid · zenodo · res2net · avpv · embeddings · pvpo ·
tracklets · keypoint · medsam-2 · …`

Real promotable terms (`deeplabcut`, `connectome`/`connectomics`, `cebra`, `keypoints`, `imagenet`,
`embeddings`, `llms`, `bm25`, `res2net`, `medsam-2`, `phate`, `superanimal`) — the curator's actual
candidate set. **Drops the `effect`/`simple`/`whether`/`four`/`best` boilerplate corpus_band is full
of** (they are common English → near-zero weirdness).

## per_doc most-frequent top picks (distinctive per paper, as designed)

`political · medical image segmentation · bm25 · pubmed · pose estimation · ecg · mamba · rag ·
ms marco · open-domain qa · retriever · passage · re-ranking · dbs …`

## Reading + acceptance

**Indicative acceptance (pre-registered) MET for contrastive:** the top list contains recognizable
per-domain terms and drops the `consider/introduce/respectively`-class boilerplate, where corpus_band
does not. Contrastive is the recommended vocabulary-mining mode for the R5 curation step.

**Limitations (contrastive, follow-up levers — not blocking):**
1. **Publisher / ID artifacts** rank high because they are rare in general English and frequent in this
   corpus: `elife`, `pmid`, `zenodo`, `frontiersin`, `fnana`, `7554 elife`. Candidate for the STOPWORDS
   list or a metadata-strip pass; an AWL/boilerplate layer could ride on top later (ADR-006 option C).
2. **Repeated-token n-grams** survive as candidates: `outflux outflux outflux`. `candidate_terms` should
   collapse runs of one repeated token; small follow-up.
3. Anatomical / OCR abbreviations (`avpv`, `pvpo`, `dl5`, `mgns`) rank high — genuinely domain-specific,
   but the curator filters them. This is the intended human-in-the-loop, not a defect.

These are curation-surface concerns, not correctness bugs; the R5 curated run + human promotion is where
the vocabulary is finalized. **No parameters tuned against this corpus** (defaults frozen a priori).

**Re-run:** `uv run --no-sync python -m scripts.extract_keywords --mode contrastive` (dry-run; add
`--apply` to write candidates) — or the side-by-side scratch measure
(`scratchpad/r3_termhood_measure.py`).

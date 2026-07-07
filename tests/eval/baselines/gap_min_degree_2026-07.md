<!-- status: active · updated: 2026-07-07 · class: baseline -->

# Gap detection — `min_degree` baseline (SPRINT-002 / ADR-004 Tier 1)

`under_connected`'s degree floor, set from **this corpus's own degree distribution** — not a
guessed absolute (the same-domain-embedding brittle-absolute-threshold lesson: CLAUDE.md /
atlas). Companion to `rg001_concept_skeleton_r5_2026-07-02.md`, which validated the skeleton
this gap layer is defined over (R5 PASS, ADR-008).

## Environment / corpus

- **Data home:** `data/` on this box — `data/skeleton/skeleton.json`, `graph_version
  055312c8c15a7e69` (the same R5-validated skeleton: `MIN_COOCCURRENCE=2`, `boundary` presence).
- **Corpus:** 62 documents contributing curated-concept presence; 26 curated concepts, 70
  co-occurrence/citation/similarity edges.
- **Cost:** $0 — pure Python + NetworkX over the persisted skeleton; no LLM, no embeddings call.

## Degree distribution (all 26 curated concepts)

```
sorted degrees: [1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 20]
min 1 · max 20 · median 5.0 · mean 5.38
quartiles (inclusive method): Q1 = 3.0 · Q2 (median) = 5.0 · Q3 = 6.75
counts below threshold k:  k=1 → 0 · k=2 → 1 · k=3 → 5 · k=4 → 8 · k=5 → 11
```

One clear hub outlier at degree 20; the rest of the distribution is a smooth 1–9 spread. No
degree-0 concepts on this corpus today (`isolated` currently reports empty — an honest result,
not a bug: this corpus's curated vocabulary happens to be fully connected right now).

## Decision

**`min_degree = 3`** — the first quartile (Q1) of the degree distribution. Concepts with degree
in `{1, 2}` (5 of 26, ≈19%) route into `under_connected`; degree-0 concepts are excluded from
`under_connected` (they are `isolated` instead — a distinct kind, not double-reported; see
`gaps.detect_under_connected`'s docstring).

**Why Q1, not median or a fixed constant:** a bottom-quartile cut flags a consistent, corpus-
relative fraction of the vocabulary regardless of corpus size or density, instead of a threshold
tuned to this corpus's specific edge count that would silently drift on a re-cluster. Same
principle as `CONCEPT_SKELETON_MIN_COOCCURRENCE`'s R5 validation run: derive the threshold from
the corpus's own shape, record it here, and re-derive (not guess) if the corpus changes
materially (a new domain added, a big ingest batch, etc.).

## Verification run (dry-run, this corpus, 2026-07-07)

```
$ python -m scripts.build_gaps --min-degree 3
Graph version:             055312c8c15a7e69
Tier-1 gaps:               10
Tier-2a floor gaps:        3
Total gaps:                13

kind                count
----------------------------------------------------------------------------
single_source           3
thin_bridge             2
under_connected         5
unsourced_claim         3
```

`under_connected` = 5 matches the "count below threshold k=3, minus the k=1 count" arithmetic
above exactly (5 − 0 = 5) — the detector and the manual distribution agree. `isolated` = 0
(matches the "no degree-0 concepts today" observation). This was a **dry run** (no `--apply`) —
nothing was written to the `gaps` table; it read the real `data/library.db` (curated concepts +
persisted `answer_claims`) but only to report the counts above.

## Retrieval impact

**None by construction.** `gaps.py` / `scripts/build_gaps.py` only read `skeleton.json` +
the `concepts`/`concept_aliases`/`answer_claims` tables and write a new, additive `gaps`
table; no code path touches `pipeline.py`, `embeddings.py`, `provenance.py`, or any Chroma
collection. Verified by inspection (no import of, or call into, the retrieval path) rather
than a fresh eval-harness run, since there is no code for such a run to exercise — the same
reasoning `epistemics.py` and `wiki.py`'s sidecars rely on.

## Re-running this baseline

```
python -c "
import json, statistics
data = json.load(open('data/skeleton/skeleton.json', encoding='utf-8'))
degrees = sorted(n['degree'] for n in data['nodes'])
print(degrees)
print(statistics.quantiles(degrees, n=4, method='inclusive'))
"
python -m scripts.build_gaps --min-degree 3
```

If the corpus changes materially (new domain, large ingest batch), re-run the above, recompute
Q1, and update this note + `_DEFAULT_MIN_DEGREE` in `scripts/build_gaps.py` together — don't let
the code and this baseline drift apart.

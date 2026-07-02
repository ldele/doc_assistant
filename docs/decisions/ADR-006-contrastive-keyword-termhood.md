<!-- status: active · updated: 2026-07-02 · class: append-only -->

# ADR-006 — Contrastive keyword termhood via a reference word-frequency corpus

- **Status:** accepted
- **Date:** 2026-07-02
- **Deciders:** user (reference source = Option A), Claude Code (execution + formula)

## Context

Keyword extraction seeds the curated concept vocabulary (KI-13). Both shipped modes fail by
construction, measured on two corpora (2026-07-01): per-doc TF-IDF selects df≈1 per-paper cliques;
`corpus_band`'s `df·(1+ln tf)` is monotone in df, so it always grabs the most-shared = most-generic
terms (re-confirmed 2026-07-02: `corpus_band`'s real-corpus top-40 is `state`/`effect`/`simple`/
`whether`/`four`/`best` plus author surnames). The terminology-extraction literature's answer is
**contrast against a reference corpus** (a "weirdness" ratio: frequent-here / rare-in-general-English)
plus **C-value** for nested multi-word terms. Both are deterministic, zero-LLM, and the reference is
external — so this does not tune against our own corpus. R3 needs a reference-frequency source.

## Options

1. **`wordfreq` dependency (chosen).** Maintained, compact; `zipf_frequency(token, "en")`; the
   frequency data ships with the package so it works offline after install. — *Pros:* one `uv add`,
   no table to build/maintain, broad language coverage. *Cons:* a new dependency (+`ftfy`/`langcodes`/
   `locate`/`wcwidth`; ~54 MB of data).
2. **Repo-frozen frequency table (rejected).** No new dependency, but must be built once and committed
   and then maintained; identical math otherwise.
3. **Academic-Word-List stoplist only (rejected as sole gate).** Targets academic boilerplate but gives
   no general contrast (won't rank domain terms, only removes known filler); may still layer on top of
   Option A later (published AWL, not tuned on our corpora).

## Decision

Option 1 — add `wordfreq>=3.0` as a base dependency (the keyword enrichment is core; offline after
install). Add `mode="contrastive"` to the keyword extractor. **Frozen formula + defaults (chosen a
priori, before looking at output):**

- `weirdness(term) = min over tokens of max(0, REF_CEILING − zipf(token))`, `REF_CEILING = 8.0`. An
  out-of-vocabulary technical token (`bm25` → zipf 0) reaches the ceiling (maximally weird — the
  desired OOV smoothing); a phrase is bounded by its most-common token.
- **C-value** nested discount (Frantzi, `+1` length variant): `C(a) = log2(|a|+1)·(f(a) − mean_{b⊋a}
  f(b))`. Used as a **gate** — drop candidates with `C ≤ KEYWORD_CONTRASTIVE_MIN_CVALUE` (`0.0`), i.e.
  fragments that occur only inside longer terms.
- **Score** (pre-registered in the remediation plan): `(1 + ln tf_corpus) · weirdness(term)`, over the
  C-value-gated pool, ties broken by term ascending (deterministic).

**Scope decision:** the fix lands in the new `contrastive` mode. `per_doc` and `corpus_band` are left
**unchanged** — they are retained as A/B levers so the R5 decision run compares contrastive against the
original baselines, not against silently-altered ones. Contrastive is the recommended go-forward mode.

## Consequences

- **Easy:** a curator's candidate list is now domain vocabulary (`deeplabcut`, `connectome`, `cebra`,
  `imagenet`, `embeddings`, `bm25`, …) instead of generic boilerplate; deterministic + offline; the
  reference is external so no corpus-tuning-against-itself.
- **Hard / committed:** a new dependency in the base install and the freeze (RG-013-style: verify it
  bundles if it ever matters for the frozen build — it is pure-Python + data, no torch interaction).
  Weirdness rewards *all* rare tokens, so publisher/ID artifacts (`elife`, `pmid`, `zenodo`) and
  repeated-token n-grams (`outflux outflux outflux`) rank high — documented follow-up levers (STOPWORDS
  / metadata strip / an AWL layer per Option C; collapse repeated-token grams). Baseline:
  `tests/eval/baselines/rg001_keyword_termhood_2026-07-02.md`.
- **Reversible:** `mode` is a parameter; the old modes still exist. The reference source can be swapped
  (Option B) behind the same `weirdness` seam without touching callers.

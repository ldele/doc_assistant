# Chunking sweep — private corpus, 97 docs / 35 cases (2026-08-08)

The powered half of **RG-026**. Companion to `chunking_sweep_public_2026-08-08.md` (10 cases,
paid Haiku); together they replace the void 2026-06-06 sweep (KI-41). This is the run with enough
cases to say something, and the one where **`citation_overlap` stops saturating and becomes the
most trustworthy signal in the table**.

**Setup**
- Corpus: the working library, **97 documents / ~36k parent-child chunks** (grown from 33,105 by
  the 2026-08-07 text-layer fallback). Cases: `tests/eval/cases.yaml` (**35**, multi-paper-per-topic).
- `run_eval --repeat 3 --with-embedding` per config. Scorers: `contains_all`, `citation_overlap`,
  `embedding_similarity`. **No LLM judge** — see the generator caveat below.
- Generator: **local Ollama `llama3.1:8b`** (`OllamaLLM`, no JSON constraint), $0. torch
  `2.12.0+cu130`, RTX 4070.
- Preflight passed on all 6 arms; all **6 geometries recorded distinctly** in `config_json`.
- The first config paid a **full 97-document re-extraction** — the first ingest since the KI-40
  fingerprint fix, so every pre-existing cache entry lacked a `.fp` sidecar and was stale by
  definition. Configs 2-6 hit the cache. That is KI-40 working, not a fault.

## ⚠ Generator caveat — read before quoting any number here

Generation ran on `llama3.1:8b`, a deliberate cost choice, **not** the shipped default. That model
scores **36%** citation coverage against Haiku's **81%** on the same prompt (KI-36,
`citation_coverage_2026-08-07.md`). So:

- **`citation_overlap` is unaffected and is the signal to trust.** It is computed from the
  *retrieved* documents, before generation — it measures retrieval, and no LLM touches it. Its
  trial-to-trial std is **0.000** on every row because retrieval here is deterministic.
- **`contains_all` and `embedding_similarity` are generator-dependent** and measured through a weak
  model. Treat their *ordering* as provisional and their absolute values as not comparable to any
  Haiku-generated baseline.

## Results (n=3 trials, mean ± trial-to-trial std)

| # | parent/child | `citation_overlap` ⭐ | `contains_all` | `embedding_similarity` |
|---|---|---:|---:|---:|
| **1 — control** | 2000/200 · 400/50 | 0.936 ± 0.000 | **0.777** ± 0.004 | 0.798 ± 0.003 |
| 2 | 2000/200 · 256/32 | **0.946** ± 0.000 | 0.740 ± 0.016 | 0.797 ± 0.002 |
| 3 | 2000/200 · 600/75 | 0.887 ± 0.000 | 0.744 ± 0.014 | 0.791 ± 0.003 |
| 4 | 1500/150 · 400/50 | 0.877 ± 0.000 | **0.777** ± 0.033 | **0.802** ± 0.007 |
| 5 | 3000/300 · 400/50 | 0.936 ± 0.000 | **0.785** ± 0.024 | 0.800 ± 0.006 |
| 6 | 1000/100 · 256/32 | **0.946** ± 0.000 | 0.734 ± 0.008 | 0.794 ± 0.004 |

⭐ = the generator-independent metric.

## Reading

**1. The instrument works here, and that is the headline.** On the public 10, `citation_overlap`
was pinned at 1.000 for every config and could not discriminate at all. On 97 documents with 35
multi-paper cases it spans **0.877 → 0.946** — the distractors that a real corpus supplies are
exactly what makes retrieval quality measurable. Any future retrieval experiment should run here,
not on the public set.

**2. There is a genuine trade-off across the grid, not a winner.**

- **Smaller child (256/32) retrieves best** — configs 2 and 6 both hit 0.946, the top of the table,
  above the control's 0.936. Finer-grained children give the reranker more precise targets.
- **The same two are worst on answer quality** — 0.740 and 0.734 on `contains_all`, well below the
  control's 0.777.
- **Larger parent (3000/300) answers best** (0.785) at control-level retrieval (0.936).

That is a coherent mechanism rather than noise: the child chunk is what gets *retrieved*, the parent
is what the model *reads*. Shrinking the child sharpens retrieval; shrinking the context the parent
supplies costs the answer. **The control sits at the balanced point** — 2nd on retrieval, tied-1st
on `contains_all`, 2nd on `embedding_similarity`. No other config is in the top two on more than one.

**3. Nothing beats the control beyond its variance, so the lock holds.**
- `contains_all`: config 5 leads by 0.008 with its own std at ±0.024 — the bands overlap heavily.
- `embedding_similarity`: config 4 leads by 0.004 at ±0.007. Inside noise.
- `citation_overlap`: configs 2/6 lead by **0.010 with zero trial variance**, so the difference is
  real and reproducible on *this* case set — but 0.010 × 35 cases is **0.35 of a case**. It is a
  measured difference too small to move a locked setting, and it is bought by a 0.04 loss on
  `contains_all`, which is ~5x larger.

**4. The caveat that cuts the other way.** Points 2 and 3 lean on `contains_all`, which came from a
model with a known coverage floor. The public run's Haiku generator scored the same 256/32 child at
**0.919** `contains_all` — comparable to its control — where llama3.1:8b puts it 0.04 *below*. So
**the small-child answer penalty may be an artifact of a weak generator needing more context**, and
that is the single most important open question this run leaves.

## Decision

**Keep `parent 2000/200 · child 400/50`.** For the first time this is a measured decision rather
than an inherited default: on 35 cases over 97 documents it is the most balanced point in the grid,
and nothing beats it beyond variance on any metric.

**Recorded as the standing caveat:** the defaults are *un-beaten*, not *optimal*. Two directions
have evidence behind them and neither is settled —
- **child 256/32** for retrieval (+0.010 `citation_overlap`, deterministic) and cost (the public run
  measured `1000/100 · 256/32` at **45% fewer input tokens**, 15% lower latency);
- **parent 3000/300** for answer quality (+0.008 `contains_all`), at +52% input tokens.

## What would settle it

Re-run configs **1, 2, 5, 6** on this corpus with the **shipped Haiku generator** and the LLM judge.
That isolates the one confound this run could not: whether the small-child `contains_all` penalty is
real or a local-model artifact. Cost is the reason it was not done here; it is ~4 configs × 3 trials
× 35 cases of paid generation, and it is the experiment that would let the cost win be taken.

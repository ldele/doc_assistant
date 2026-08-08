# Chunking sweep — public corpus (2026-08-08) — **the first one that measured anything**

Re-runs the parent/child chunk-size grid after **KI-41**: the 2026-06-06 sweep of the same grid
drove its configs through `PARENT_CHUNK_SIZE` / `CHILD_CHUNK_SIZE`, `load_dotenv(override=True)`
overwrote them from `.env` (KI-38), and all six arms re-ingested the **same** corpus. That run's
verdict ("no config beats the default") was produced by comparing one configuration with itself
six times, and **this file supersedes it** — see `chunking_sweep_public_2026-06-06.md`, which
should now be read only for its noise-floor reading.

**Setup**
- Corpus: the verified-10 eval collection (`tests/eval/corpus_manifest.yaml`), staged into an
  **isolated data home** so the 97-document working library was never rebuilt. Isolation was
  verified before the run, not assumed: `CHROMA_PATH` does **not** derive from `DATA_PATH` — a
  non-ASCII data path relocates the store to `C:\ProgramData\doc_assistant\chroma\<hash>\`
  (the KI-11 fix), and the scratch home hashed to a different store than `data\chroma`.
- Cases: `tests/eval/cases.public.yaml` (10 cases). Scorers: `contains_all`, `citation_overlap`,
  `llm_judge`. `run_eval --repeat 3 --with-llm-judge` per config, as on 2026-06-06.
- Generator + judge: Anthropic (`claude-haiku-4-5`), the shipped default. torch `2.12.0+cu130`,
  RTX 4070. Wall clock: **22 min** for the whole sweep.
- Preflight (new, DEVLOG 2026-08-08 (1)): all 6 arms resolved to what they asked for, all distinct.

## The audit — what makes this run believable and the last one not

| Evidence | 2026-06-06 (void) | 2026-08-08 (this run) |
|---|---|---|
| Distinct geometries **recorded** across 18 runs | not recorded at all | **6 of 6**, each matching its note |
| `token_input` across configs | **4326.7 everywhere**, identical per case | **2529 → 7044** (a 2.8x span) |
| `token_input`, parent 2000 vs parent 3000 | identical on all 10 cases | 4627 vs **7044** (+52%) |

`token_input` scales with the evidence block, so a 3x parent-size range **must** move it. That it
did not was the proof KI-41 rested on; that it now does is the same instrument answering the other
way. Every run also carries its 13 run-defining settings in `config_json` (`eval/run_settings.py`),
so this table is reproducible from the DB rather than from anyone's notes.

## Results (n=3 trials, mean ± trial-to-trial std)

| # | parent/child | `citation_overlap` | `contains_all` | `llm_judge` | judge skips | mean `token_input` | latency |
|---|---|---:|---:|---:|---:|---:|---:|
| **1 — control** | 2000/200 · 400/50 | 1.000 ± 0.000 | 0.917 ± 0.043 | 3.667 ± 0.145 | 0 | 4627 | 4.6 s |
| 2 | 2000/200 · 256/32 | 1.000 ± 0.000 | 0.903 ± 0.042 | 3.833 ± 0.088 | 0 | 4829 | 4.5 s |
| 3 | 2000/200 · 600/75 | 1.000 ± 0.000 | **0.942** ± 0.000 | 3.756 ± 0.184 | 0 | 4531 | 4.7 s |
| 4 | 1500/150 · 400/50 | 1.000 ± 0.000 | 0.883 ± 0.014 | 3.623 ± 0.109 | 1 | 3574 | 4.2 s |
| 5 | 3000/300 · 400/50 | 1.000 ± 0.000 | 0.925 ± 0.029 | 3.667 ± 0.173 | 0 | 7044 | 4.9 s |
| 6 | 1000/100 · 256/32 | 1.000 ± 0.000 | 0.919 ± 0.019 | **3.878** ± 0.102 | 0 | **2529** | 3.9 s |

## Reading

- **`citation_overlap` saturates at 1.000 for every config**, exactly as in the void run. Retrieval
  cites the right paper in all 10 cases every trial, so this metric **cannot discriminate chunk
  size** on this corpus. That reading survives KI-41 because it never depended on the arms differing.
- **The control is not the optimum on either discriminating metric.** It is 3rd on `contains_all`
  (0.917, behind 0.942 and 0.925) and tied-last on `llm_judge` (3.667). This is new information:
  the previous run reported the control as tied-best and best respectively, and that was an artifact
  of every arm being the control.
- **No config beats the control *beyond its variance*, so the rigor-gate bar is not met.** The
  control's own trial-to-trial std on `contains_all` is ±0.043, wider than the 0.025 gap to the
  leader. On `llm_judge` the leader's band [3.776, 3.980] and the control's [3.522, 3.812] still
  overlap. **At n=3 on 10 cases this cannot rank the grid** — it can only say the defaults are not
  demonstrably wrong.
- **The one result that is not marginal is cost.** Config 6 (`1000/100 · 256/32`) runs on **45%
  fewer input tokens** than the control (2529 vs 4627) and **15% lower latency**, while scoring
  *at least* as well on both discriminating metrics (0.919 vs 0.917; 3.878 vs 3.667). Token count is
  a deterministic property of the evidence block, not a sampled score, so that 45% is not subject to
  the variance caveat above — only the quality claim is. On a paid provider this is a direct
  per-turn saving.
- **Two candidates, different reasons:** config 3 (`600/75` child) for quality on `contains_all`
  (0.942 at ±0.000 across three trials), config 6 for cost at equal quality.

## Decision

**Keep the locked defaults `parent 2000/200 · child 400/50` — for the first time on evidence.**
Nothing beats the control beyond its variance, which is the standing bar for changing a locked
setting. But the caveat is now the opposite of the old one: the defaults are not confirmed-best,
they are *un-beaten at this power*, and two named alternatives look better on different axes.

**This does not close RG-026.** 10 cases give effect sizes, not significance, and the 2026-08-08
private run (97 documents, 35 cases) is the one with the power to settle it. Config 6's cost
advantage is the single most re-testable claim here and should be the first thing checked there.

## Caveats

- n=3 trials, 10 cases, one corpus. Read trends, not a ranking.
- The generator is non-deterministic at default temperature; the judge is a model. `contains_all`
  and `llm_judge` both inherit that noise, which is what the ± columns measure.
- Not comparable to `chunking_sweep_public_2026-06-06.md` in the way its table implies — that run's
  six rows are six samples of one configuration, so its spread is a **noise floor** (`contains_all`
  0.906–0.933, `llm_judge` 3.793–3.951), not a comparison. Notably this run's control lands at
  0.917 / 3.667, inside and slightly below that band.

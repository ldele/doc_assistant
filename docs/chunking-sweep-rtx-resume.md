# Chunking sweep — resume on the RTX box

> ✅ **DONE (2026-06-06).** Full 6-config sweep run on the RTX box (public corpus, n=3,
> judge). **Defaults confirmed** — no config beats `2000/200 · 400/50`. Results +
> per-config run-ids: [`tests/eval/baselines/chunking_sweep_public_2026-06-06.md`](../tests/eval/baselines/chunking_sweep_public_2026-06-06.md).
> The Locked-settings chunk-sizes row in `.claude/CONTEXT.md` (local-only working-dir file) was updated. The notes below are retained for
> historical context / re-run instructions.

**Status (2026-06-04):** deferred to the RTX/GPU box. On the CPU dev box each
config's re-embed is ~45 min (×6 ≈ 5 h); on the RTX box it's minutes. We ran
**config 1 (control, the current defaults 2000/200 · 400/50)** on CPU as a preview
and stopped after it (config 2 had begun and partially wiped the store, which we
then restored to defaults).

**Config 1 preview (public cases, n=3, CPU):** `citation_overlap` 1.000 ± 0.000 ·
`contains_all` 0.917 ± 0.000 · `llm_judge` 3.889 ± 0.159 — i.e. the control config
reproduces the locked bge baseline (1.000 / 0.927 / 3.738 at n=5). The open question
the RTX run answers is whether **configs 2–6 beat this control**, not whether the
defaults are sane (they are).

## How the sweep was run (for reproduction)

```bash
git pull                                  # get the sweep's --cases passthrough (2026-06-04)
uv sync                                    # torch-backend="auto" auto-selects the CUDA wheel on the GPU box
# Full 6-config grid, public corpus, repeat 3, with judge:
uv run --python 3.12 python -m scripts.sweep_chunking \
    --cases tests/eval/cases.public.yaml --repeat 3 --with-llm-judge
```

**Run the full grid (all 6), not just 2–6.** Measuring config 1 on CPU and 2–6 on
GPU would mix machines/torch builds into the comparison. Re-running config 1 on the
RTX box is cheap there and keeps all configs on one machine. Our CPU config-1 number
is only a preview/sanity check, not part of the locked comparison.

> ⚠ The sweep is **destructive**: each config runs `ingest --rebuild`, which
> `shutil.rmtree`s `data/chroma` + `data/chroma_pc` (wiping the bge **and** specter2
> collections) and clears the SQLite document rows. The bge default baseline and the
> bge-vs-specter2 comparison are already locked in `tests/eval/baselines/` — so the
> wipe is safe. After the sweep, **restore the default store** so the app is usable:
>
> ```bash
> uv run --python 3.12 python -m doc_assistant.ingest --rebuild   # no chunk env vars => defaults
> ```
>
> (Optional) re-embed specter2 if you want that collection back:
> `EMBEDDING_MODEL=specter2 uv run --python 3.12 python -m doc_assistant.ingest --skip-cleanup`

## The grid (`scripts/sweep_chunking.py` → `DEFAULT_GRID`)

| # | parent | child | intent |
|---|---|---|---|
| 1 | 2000/200 | 400/50 | current default (control) |
| 2 | 2000/200 | 256/32 | smaller child — finer retrieval |
| 3 | 2000/200 | 600/75 | larger child — more context per hit |
| 4 | 1500/150 | 400/50 | smaller parent — tighter LLM context |
| 5 | 3000/300 | 400/50 | larger parent — broader LLM context |
| 6 | 1000/100 | 256/32 | small/small — precision regime |

## Reading the results

Each config's runs are tagged in `data/eval.duckdb` with its
`chunk-sweep | parent=… child=…` note. Compare with the harness aggregate report
(filter by note). The discriminating signals for chunking are `contains_all` and
`llm_judge`; `citation_overlap` may saturate at 1.000 across configs (then it can't
rank them). The sweep recorded the result in
[`tests/eval/baselines/chunking_sweep_public_2026-06-06.md`](../tests/eval/baselines/chunking_sweep_public_2026-06-06.md)
(defaults confirmed) and the **Locked settings** chunk-sizes row in `.claude/CONTEXT.md`
was updated from "never measured" to measured/confirmed on 2026-06-06.

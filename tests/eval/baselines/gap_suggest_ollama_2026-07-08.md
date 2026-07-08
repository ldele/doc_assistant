<!-- status: active · updated: 2026-07-08 · class: baseline -->

# Gap detection — Tier-2a stochastic ceiling real-Ollama smoke run (SPRINT-005 / ADR-004 Decision 4)

The deferred host step SPRINT-005 (`docs/sprints/SPRINT-005-gap-stochastic-ceiling.md`) called out:
"the real `--suggest --apply` smoke against a local model is a HOST step on the RTX/Ollama box —
deferred, NOT a gate for landing this code." This box is that RTX/Ollama box (`DOC_TORCH=cu130`).
The code itself was built + proven offline first with a scripted `LLMClient`
(`tests/unit/test_gap_suggest.py`, `tests/integration/test_build_gaps.py`) — this run is the real
local-model validation on top of that, not a substitute for it.

## Environment / corpus

- **Data home:** `data/` on this box — `data/skeleton/skeleton.json`,
  `graph_version 0e3ec833e6fc31ee` (357 curated concepts, 1534 edges, 47 documents contributing
  presence; 219 LLM-annotated edges from Node B).
- **Provider/model:** `GAP_SUGGEST_LLM_PROVIDER`/`_MODEL` defaults — `ollama` / `llama3.1:8b`
  (not overridden). Local, free — `llm.assert_provider_intent` no-ops for a non-paid provider.
- **Prerequisite:** the `gaps` table didn't exist yet on this box's `data/library.db` (additive,
  never auto-created outside `create_all`) — ran `python -m doc_assistant.db.migrations` once to
  add it (19 other tables were already present and untouched).
- **Cost:** $0 (Ollama, no external network call).

## Command + result

```
$ python -m doc_assistant.db.migrations   # one-time: adds the `gaps` table
$ python -m scripts.build_gaps --apply --suggest
============================================================================
Graph version:             0e3ec833e6fc31ee
Tier-1 gaps:               274
Tier-2a floor gaps:        16
Tier-2a suggested (stoch): 12
Total gaps:                290
Rows written:              302
============================================================================

kind                count
----------------------------------------------------------------------------
isolated               26
single_source         224
thin_bridge            12
under_connected        12
unsourced_claim        16

gaps sidecar written.

real    0m50.8s
```

**12/12 `under_connected` concepts produced a parseable suggestion** (0 transport failures, 0
unparseable responses) — `n_suggested == 12 == n(under_connected)`, i.e. this corpus's 8B model
never hit the degrade-gracefully path in this run (that path is still exercised offline by
`test_one_bad_concept_does_not_sink_the_batch`). ~4.2s/call average, matching the reviewer's known
per-call latency for this model on this box.

## Manual spot-check (all 12 suggestions)

| Concept | Present neighbours | Suggested kind | Target | Rating |
|---|---|---|---|---|
| cross-encoder | sbert, sentence embeddings | suggested_concept | sentence similarity | 0.8 |
| user question | assistant, grading | suggested_concept | chatbot interaction | 0.8 |
| aggressive behavior | dbs | suggested_concept | anger management | 0.8 |
| contrastive encoder | contrastive learning, dense retrieval | suggested_concept | encoder-decoder | 0.8 |
| untrained | actor, latent communication | suggested_concept | model | 0.8 |
| infonce | contrastive learning | suggested_concept | noise | 0.8 |
| vss blocks | mamba-unet, unet | suggested_concept | segmentation model | 0.8 |
| rho | beta | suggested_concept | sigma | 0.8 |
| stereology | gnr, vascular | suggested_concept | histomorphometry | 0.8 |
| knowledge distillation | dense retrieval, image segmentation | suggested_concept | transfer learning | 0.8 |
| relevance judgement | BM25 | suggested_concept | information retrieval | 0.8 |
| myelin | axon | suggested_concept | demyelination | 0.9 |

**Quality read:** plausible and mostly on-topic given each concept's neighbours (e.g. "relevance
judgement" + "BM25" → "information retrieval"; "myelin"/"axon" → "demyelination"; "cross-encoder" +
"sbert" → "sentence similarity"). A couple are weak/generic ("untrained" → "model", "rho" → "sigma"
— Greek-letter co-occurrence noise from a stats paper, not a real conceptual gap). **Every
suggestion came back `suggested_concept`** — this run's llama3.1:8b never chose `suggested_link` or
`thin_area`, and rating is flat at 0.8 (one at 0.9) — a calibration ceiling, not a bug: matches the
already-documented local-8B confidence-flattening finding from Node B / the reviewer (this box's
8B doesn't spread its own confidence). None of this blocks the ceiling's purpose (surface
candidates for a human to promote/dismiss); it does mean the `rating` column isn't yet a
discriminating signal on this model — a stronger model or an explicit calibration pass would be
the fix, not scoped to this sprint.

## Not touched (deliberately out of scope here)

- **`_DEFAULT_MIN_DEGREE = 3`** (`scripts/build_gaps.py`) — unrevisited. The corpus has grown
  materially since `gap_min_degree_2026-07.md` was written (26 curated concepts → 357; that
  baseline's own instructions say to re-derive Q1 if the corpus changes materially). Recomputing
  it is a `gaps.py` Tier-1 threshold question, not a Tier-2a stochastic-ceiling one — left for a
  separate pass so this baseline stays scoped to what SPRINT-005 shipped.
- No promotion/dismissal of any of the 12 rows above — they sit `status="surfaced"` in `data/gaps`
  (real, gitignored DB), for a human to curate.

## Re-running this baseline

```
python -m scripts.build_gaps --apply --suggest
```
Idempotent by design: a re-run with all 12 rows still `surfaced` replaces them with a fresh
suggestion (no dedup by content — same concept, possibly a different sampled response since
`temperature=0.0` but the model itself is stochastic across calls in practice); a promoted or
dismissed row survives untouched (`tests/integration/test_build_gaps.py::
test_promoted_stochastic_survives_rebuild_and_resuggest`).

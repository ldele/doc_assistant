<!-- status: active · updated: 2026-11-24 (the release benchmark's reference on the public home, H6) · class: append-only -->

# Stage profile on the public home: the release benchmark's reference (2026-11-24)

**What this is for.** `docs/RELEASE.md` § 3 runs `scripts/profile_stages.py` before each release
on the 28 public papers, so a release that makes a launch or a turn slower is seen before the tag.
This file is the reference for one machine. It is not a claim about any other machine: the corpus
size and the torch device change these numbers by an order of magnitude
(`stage_profile_2026-07-28.md`, `stage_profile_2026-07-29.md`).

| | |
|---|---|
| Code | `7d29dd6`, from a fresh clone (the same run as `release_0.8.0_chat_2026-11-24.md`) |
| Corpus | the 28 public papers of `download_corpus --demo`, 6,850 parent-child chunks |
| torch device | `cuda` (NVIDIA GeForce RTX 4070), torch 2.12.0+cu130 |
| Python / OS | 3.12.3 / Windows 11 |
| Command | `DOC_DATA_DIR=<public home> uv run python -m scripts.profile_stages -r 3` |
| Cost | $0; no model is called |

| Stage | Median (3 repeats) | Paid |
|---|---:|---|
| Cold launch, fresh process (`RAGPipeline()`) | 13.02 s | every launch |
| Embedder load, cold | 10.45 s | every launch |
| Read all chunks from Chroma | 255 ms | every launch |
| Build the keyword index | 108 ms | every launch |
| Embed the question | 5.2 ms | every turn |
| Vector search (k=20) | 33.9 ms | every turn |
| Keyword search (k=20) | 3.9 ms | every turn |
| Both arms and fusion | 37.5 ms | every turn |
| Retrieve, rerank and expand (10 sources) | 84.7 ms | every turn |

**How a release reads it.** On the same machine and corpus, a stage more than twice its median
here is a finding to explain before the tag, not a blocker by itself; a different machine records
its own reference first. Answer generation is not in this profile (the model's time dominates a
turn and depends on the model, not on the app).

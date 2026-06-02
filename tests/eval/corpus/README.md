# Public corpus — populated from arXiv, not committed

This directory is intentionally empty in git. The public demo corpus is the
literature behind doc_assistant's own design (RAG, dense retrieval, the BGE and
SPECTER2 embedders, BERT re-ranking, ColBERT, HyDE, LLM-as-a-judge, AI Usage
Cards) — all on arXiv.

Nothing is re-hosted here: arXiv papers carry a non-exclusive license that
permits downloading but not redistribution. So the corpus ships as a manifest
of arXiv IDs, and the script fetches the PDFs:

```bash
uv run python -m scripts.download_corpus            # -> data/sources/
```

See `tests/eval/corpus_manifest.yaml` for the 10 papers (pinned versions,
sha256, abstract links) and `tests/eval/cases.public.yaml` for the eval cases
written against them.

This is the project's reproducible benchmark corpus. See
`tests/eval/TESTING.md` for the testing strategy — what each scorer measures
and why.

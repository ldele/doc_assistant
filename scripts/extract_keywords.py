"""Extract content keywords from ingested documents (concept-skeleton vocabulary seed).

Deterministic corpus TF-IDF over each document's cached markdown — zero LLM, free, no new
dependency. Populates the ``keywords`` table (``source="extracted"``), which
``scripts/seed_concepts.py`` then surfaces as concept-skeleton vocabulary candidates for
``--promote``. Fixes KI-13 (the promote seam had no Keyword producer). Additive + idempotent
(Enrichment-Layer Pattern) — never mutates the chunk store.

Usage:
    python -m scripts.extract_keywords                     # dry-run all (rank, no writes)
    python -m scripts.extract_keywords --apply             # write Keyword rows + links
    python -m scripts.extract_keywords --doc <id>          # one document (IDF still corpus-wide)
    python -m scripts.extract_keywords --apply --force     # re-extract (clear old extracted)
    python -m scripts.extract_keywords --top-k 20          # override keywords-per-doc

Then: python -m scripts.seed_concepts            # list candidates
      python -m scripts.seed_concepts --promote "dense retrieval"
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant.config import (
    KEYWORD_CONTRASTIVE_MIN_CVALUE,
    KEYWORD_CORPUS_TOP_K,
    KEYWORD_MAX_DF_FRAC,
    KEYWORD_MIN_CHARS,
    KEYWORD_MIN_DF,
    KEYWORD_NGRAM_MAX,
    KEYWORD_WEIRDNESS_REF_CEILING,
    KEYWORDS_PER_DOC,
)
from doc_assistant.keywords import KeywordExtractionResult, extract_keywords

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _format_report(result: KeywordExtractionResult, *, apply: bool) -> str:
    out: list[str] = []
    out.append("=" * 76)
    out.append(f"Documents processed:     {result.n_documents}")
    out.append(f"Distinct keywords:       {result.n_distinct_keywords}")
    if apply:
        out.append(f"Keyword links written:   {result.total_written}")
    out.append("=" * 76)
    for doc in result.docs:
        head = f"\n{doc.filename}"
        if apply:
            head += f"  (+{doc.written} written)"
        out.append(head)
        for kw in doc.keywords:
            out.append(f"    {kw.score:6.2f}  {kw.term}  (tf={kw.tf}, df={kw.df})")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write Keyword rows (no LLM)")
    parser.add_argument("--force", action="store_true", help="Re-extract (clear old extracted)")
    parser.add_argument("--doc", default=None, metavar="ID", help="Only this document id")
    parser.add_argument(
        "--mode",
        choices=("per-doc", "corpus-band", "contrastive"),
        default="per-doc",
        help="per-doc = distinctive terms per document (TF-IDF); "
        "corpus-band = shared mid-document-frequency vocabulary; "
        "contrastive = termhood via C-value + reference-corpus weirdness (R3, recommended)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keywords kept (per document, or total for corpus-band)",
    )
    parser.add_argument(
        "--min-df", type=int, default=KEYWORD_MIN_DF, help="corpus-band: min document-frequency"
    )
    parser.add_argument(
        "--max-df-frac",
        type=float,
        default=KEYWORD_MAX_DF_FRAC,
        help="corpus-band: max document-frequency as a fraction of N documents",
    )
    args = parser.parse_args()

    mode = {"corpus-band": "corpus_band", "contrastive": "contrastive"}.get(args.mode, "per_doc")
    top_k = args.top_k
    if top_k is None:
        top_k = KEYWORDS_PER_DOC if mode == "per_doc" else KEYWORD_CORPUS_TOP_K

    result = extract_keywords(
        apply=args.apply,
        force=args.force,
        document_id=args.doc,
        top_k=top_k,
        ngram_max=KEYWORD_NGRAM_MAX,
        min_chars=KEYWORD_MIN_CHARS,
        mode=mode,
        min_df=args.min_df,
        max_df_frac=args.max_df_frac,
        ref_ceiling=KEYWORD_WEIRDNESS_REF_CEILING,
        min_cvalue=KEYWORD_CONTRASTIVE_MIN_CVALUE,
    )
    banner = f"mode={args.mode}  top_k={top_k}"
    if mode == "corpus_band":
        banner += f"  min_df={args.min_df}  max_df_frac={args.max_df_frac}"
    elif mode == "contrastive":
        banner += (
            f"  ref_ceiling={KEYWORD_WEIRDNESS_REF_CEILING}"
            f"  min_cvalue={KEYWORD_CONTRASTIVE_MIN_CVALUE}"
        )
    if args.apply and args.force:
        banner += f"  (swept {result.removed_orphans} orphaned rows)"
    print(banner)
    print(_format_report(result, apply=args.apply))
    if not args.apply:
        print("\nDry run (ranked, nothing written). Pass --apply to write Keyword rows.")
    else:
        print("\nDone. Next: python -m scripts.seed_concepts  (list + --promote candidates).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

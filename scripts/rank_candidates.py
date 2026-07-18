"""Rank mined keyword candidates for promotion — the review list, cheapest signal first.

This is **stage 0** of vocabulary curation: it runs *before* promotion, where
``seed_concepts.py --promote-all`` runs blind. That bulk run (2026-07-05) imported **672 of 688
keywords that appear in exactly one document**, because the keyword extractor scores
per-document salience, not cross-document vocabulary — and a singleton keyword can never form a
co-occurrence edge, so it lands in the skeleton as a permanently isolated node.

**Read-only. It ranks; it never promotes, excludes, or writes.** Promotion stays an explicit act
(``seed_concepts.py --promote``), per redesign Decision 1.

Signals, in descending trustworthiness:

* ``docs``     — distinct documents the keyword appears in. The primary signal: a
                 *cross*-document graph wants cross-document vocabulary. **Not a gate** —
                 ``pddl`` is a legitimate 1-document concept.
* ``artifact`` — deterministic regex (pure-digit token, single-char token, <=2 chars).
                 High precision.
* ``author?``  — **advisory only, ~1/3 precision on this corpus.** ``documents.authors``
                 holds whole citation strings, so it contains paper *titles* too. Never
                 auto-exclude on it; use it to order review, and let the LLM pass
                 (``curate_concepts.py --llm``) judge.

Usage:
    python -m scripts.rank_candidates                 # top candidates by document reach
    python -m scripts.rank_candidates --min-docs 2    # only cross-document candidates
    python -m scripts.rank_candidates --all           # the whole pool, including promoted
    python -m scripts.rank_candidates --limit 50
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant.concept_curation import rank_keyword_candidates

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-docs", type=int, default=1, help="only candidates in >= N documents (default 1)"
    )
    parser.add_argument(
        "--all", action="store_true", help="include already-promoted candidates (default: hide)"
    )
    parser.add_argument("--limit", type=int, default=40, help="rows to print (default 40)")
    args = parser.parse_args()

    ranked = rank_keyword_candidates()
    if not ranked:
        print("No mined keywords — run scripts/extract_keywords.py first.")
        return 0

    total = len(ranked)
    multi = sum(1 for c in ranked if c.doc_count >= 2)
    shown = [c for c in ranked if c.doc_count >= args.min_docs]
    if not args.all:
        shown = [c for c in shown if not c.promoted]

    print(f"Mined keywords: {total}  |  in >=2 documents: {multi}  ({multi / total:.1%})")
    print(
        f"Showing {min(len(shown), args.limit)} of {len(shown)} "
        f"(min-docs={args.min_docs}, promoted {'shown' if args.all else 'hidden'})\n"
    )
    print(f"  {'docs':>4}  {'flags':<18}  candidate")
    print(f"  {'-' * 4}  {'-' * 18}  {'-' * 44}")
    for c in shown[: args.limit]:
        flags = []
        if c.in_graph:
            flags.append("in-graph")
        elif c.promoted:
            flags.append("promoted")
        if c.artifact:
            flags.append("artifact")
        if c.author_like:
            flags.append("author?")
        print(f"  {c.doc_count:>4}  {','.join(flags):<18}  {c.name}")

    print('\nPromote one with:  python -m scripts.seed_concepts --promote "<name>"')
    print("Then rebuild:      python -m scripts.build_concept_skeleton --apply --enrich")
    print("\nSignals are advisory — `artifact` is precise, `author?` is ~1/3 precision here.")
    print("Nothing above is excluded automatically; a 1-document term can be a real concept.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Seed + curate the concept-skeleton vocabulary from existing Keyword rows.

Lists existing ``Keyword`` rows as vocabulary candidates (a Keyword is a *candidate only* —
never auto-written as a Concept, concept-graph redesign Decision 1) and promotes chosen
ones to curated ``Concept`` rows (+ a seed alias). Zero LLM, free. The skeleton
(``scripts/build_concept_skeleton.py``) is empty without a promoted vocabulary — promoting
a starter set is a real curation step, not a formality.

Usage:
    python -m scripts.seed_concepts                                # list candidates
    python -m scripts.seed_concepts --promote "RAG" --promote "BM25"
    python -m scripts.seed_concepts --promote-all                  # promote every keyword
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant.concept_skeleton import list_keyword_candidates, promote_keyword

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--promote",
        action="append",
        default=[],
        metavar="KEYWORD",
        help="Promote this keyword to a curated Concept (repeatable)",
    )
    parser.add_argument(
        "--promote-all",
        action="store_true",
        help="Promote every Keyword candidate to a Concept",
    )
    args = parser.parse_args()

    names = [c.name for c in list_keyword_candidates()] if args.promote_all else list(args.promote)

    if names:
        promoted = 0
        for name in names:
            concept_id = promote_keyword(name)
            if concept_id is None:
                print(f"  ! no Keyword named {name!r} — skipped")
            else:
                promoted += 1
                print(f"  + {name!r} -> concept {concept_id}")
        print(f"\nPromoted {promoted}/{len(names)} keyword(s) to curated concepts.")
        return 0

    candidates = list_keyword_candidates()
    if not candidates:
        print("No Keyword rows found — ingest documents first (keywords seed the vocabulary).")
        return 0
    print(f"{len(candidates)} keyword candidate(s) — a Concept is created only on --promote:")
    for c in candidates:
        mark = "[concept]" if c.promoted else "[candidate]"
        print(f"  {mark:<12} {c.name}")
    print('\nPromote with: python -m scripts.seed_concepts --promote "<keyword>"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

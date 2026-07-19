"""Seed + curate the concept-skeleton vocabulary (a glossary of curated concepts).

Two curation paths, both zero-LLM and free:

* **Promote a mined candidate.** ``Keyword`` rows (from ``scripts/extract_keywords.py``) are
  *candidates only* — never auto-written as a Concept (redesign Decision 1). ``--promote``
  turns a chosen one into a curated ``Concept`` (+ a seed alias).
* **Add directly (glossary entry).** ``--add`` curates a Concept by hand with its definition
  and synonyms — no mined ``Keyword`` needed. This is the recommended path: the concept graph
  is only as good as its nodes, and a small hand-picked set beats an auto-selected one (see
  ``tests/eval/baselines/rg001_concept_skeleton_2026-07-01.md``).

The skeleton (``scripts/build_concept_skeleton.py``) is empty without a promoted/added vocabulary.

Usage:
    python -m scripts.seed_concepts                                      # list mined candidates
    python -m scripts.seed_concepts --promote "dense retrieval"          # promote a candidate
    python -m scripts.seed_concepts --promote-all                        # promote every candidate
    python -m scripts.seed_concepts --add "BM25" --alias "Okapi BM25" \
        --define "A sparse lexical ranking function over term frequencies."   # curate directly
    python -m scripts.seed_concepts --glossary                           # print the glossary
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant.knowledge.concept_skeleton import (
    add_concept,
    list_keyword_candidates,
    load_glossary,
    promote_keyword,
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _print_glossary() -> int:
    entries = load_glossary()
    if not entries:
        print("Glossary is empty — add concepts with --add, or --promote a mined candidate.")
        return 0
    print(f"Glossary — {len(entries)} curated concept(s):")
    for e in entries:
        print(f"  {e.label}  [{e.source}]")
        if e.definition:
            print(f"      def: {e.definition}")
        if e.aliases:
            print(f"      aka: {', '.join(e.aliases)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--promote",
        action="append",
        default=[],
        metavar="KEYWORD",
        help="Promote this mined Keyword candidate to a curated Concept (repeatable)",
    )
    parser.add_argument(
        "--promote-all", action="store_true", help="Promote every Keyword candidate to a Concept"
    )
    parser.add_argument(
        "--add",
        metavar="LABEL",
        default=None,
        help="Directly curate a Concept (glossary entry) with this label — no Keyword needed",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="SYNONYM",
        help="A synonym/surface form for the --add concept (repeatable)",
    )
    parser.add_argument(
        "--define", metavar="TEXT", default=None, help="Definition (gloss) for the --add concept"
    )
    parser.add_argument(
        "--glossary",
        action="store_true",
        help="Print the curated glossary (labels + definitions + aliases)",
    )
    args = parser.parse_args()

    if args.glossary:
        return _print_glossary()

    if args.add:
        concept_id = add_concept(args.add, definition=args.define, aliases=args.alias)
        print(f"  + curated concept {args.add!r} -> {concept_id}")
        if args.alias:
            print(f"    aliases: {', '.join(args.alias)}")
        if args.define:
            print(f"    definition: {args.define}")
        return 0

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
        print("No Keyword rows found — run `extract_keywords --apply`, or curate with --add.")
        return 0
    print(f"{len(candidates)} keyword candidate(s) — a Concept is created only on --promote:")
    for c in candidates:
        mark = "[concept]" if c.promoted else "[candidate]"
        print(f"  {mark:<12} {c.name}")
    print('\nPromote with: python -m scripts.seed_concepts --promote "<keyword>"')
    print('Or curate directly: python -m scripts.seed_concepts --add "<label>" --define "..."')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

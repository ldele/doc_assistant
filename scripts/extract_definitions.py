"""Find sentences in the library that may define each concept (ROADMAP 93, ADR-053).

Deterministic and free: no model, no network. For every concept (or the ones named), looks for the
sentence where an author names the term, sentences shaped as a definition, and the first mention in
the documents that use it most — each copied verbatim with its document and chunk. With `--apply`
they are stored as *suggested* candidates; nothing is chosen for you, and nothing you chose,
dismissed or wrote yourself is touched. Why a candidate is graded the way it is:
`doc_assistant/knowledge/definitions.py`.

Usage:
    python -m scripts.extract_definitions                        # dry run: counts + a sample
    python -m scripts.extract_definitions --apply                # store the candidates
    python -m scripts.extract_definitions --label "hard negatives" --show 5
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant import config
from doc_assistant.knowledge.definitions import extract_definitions, passage_evidence

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="Store the candidates (default: dry run)"
    )
    parser.add_argument(
        "--label", action="append", default=[], help="Only this concept label (repeatable)"
    )
    parser.add_argument("--show", type=int, default=0, help="Print up to N candidates per concept")
    args = parser.parse_args()

    from doc_assistant.logging_config import configure_logging

    configure_logging(json=config.LOG_JSON, level=config.LOG_LEVEL)

    concept_ids = None
    labels: dict[str, str] = {}
    from sqlalchemy import select

    from doc_assistant.db.models import Concept
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        rows = session.execute(
            select(Concept.id, Concept.label).where(Concept.kind == "concept")
        ).all()
        labels = {str(i): str(label) for i, label in rows}
    if args.label:
        wanted = {w.casefold() for w in args.label}
        concept_ids = [i for i, label in labels.items() if label.casefold() in wanted]

    run = extract_definitions(concept_ids=concept_ids, apply=args.apply)
    by_form = {"coined": 0, "defining": 0, "first_mention": 0}
    by_grade = {"strong": 0, "some": 0, "thin": 0}
    for hits in run.hits.values():
        for h in hits:
            by_form[h.form] += 1
            by_grade[passage_evidence(h)["grade"]] += 1

    print("=" * 76)
    print(f"Concepts read:                  {run.n_concepts}")
    print(f"Concepts with a candidate:      {run.n_with_passages}")
    print(f"Candidates:                     {run.n_hits}")
    print(
        f"  by form:  coined {by_form['coined']} · defining {by_form['defining']} · "
        f"first mention {by_form['first_mention']}"
    )
    print(
        f"  by grade: strong {by_grade['strong']} · some {by_grade['some']} · "
        f"thin {by_grade['thin']}"
    )
    if run.applied:
        print(f"Stored (new):                   {run.n_added}")
    else:
        print("Nothing stored — dry run (--apply stores them as suggestions)")
    print("=" * 76)
    if args.show:
        for concept_id, hits in sorted(run.hits.items(), key=lambda kv: labels[kv[0]].casefold()):
            print(f"\n## {labels[concept_id]}")
            for h in hits[: args.show]:
                grade = passage_evidence(h)["grade"]
                text = " ".join(h.text.split())
                print(f"  [{grade:<6} {h.form:<13}] {text[:220]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

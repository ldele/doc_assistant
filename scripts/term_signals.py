"""Read what the library's prose says about each term (ROADMAP 93, ADR-053 decision 4).

Deterministic and free: no model, no network. For every row of the vocabulary, concepts and terms
alike: the long form the library gives beside a short one (and the reverse), the word most often
joined to a one-word label, how often it is followed by "et al.", and how often and in how many
documents the body prose uses it. What each signal rests on:
`doc_assistant/knowledge/term_signals.py`.

**This runner only reads.** A signal is offered, never applied: nothing here adds or marks a form,
renames a concept, promotes a term or deletes one, so there is no `--apply`.

Usage:
    python -m scripts.term_signals                       # counts
    python -m scripts.term_signals --show 20             # and the first rows of each signal
    python -m scripts.term_signals --label dbs --label pddl
    python -m scripts.term_signals --json signals.json   # every row, for a review page
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from doc_assistant import config
from doc_assistant.knowledge.term_signals import (
    TermSignals,
    load_term_signals,
    majority,
    signal_payload,
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label", action="append", default=[], help="Only this label (repeatable)"
    )
    parser.add_argument("--show", type=int, default=0, help="Print up to N rows per signal")
    parser.add_argument("--json", type=Path, default=None, help="Write every row to this file")
    args = parser.parse_args()

    from doc_assistant.logging_config import configure_logging

    configure_logging(json=config.LOG_JSON, level=config.LOG_LEVEL)

    rows = sorted(load_term_signals().values(), key=lambda s: s.label.casefold())
    if args.label:
        wanted = {w.casefold() for w in args.label}
        rows = [s for s in rows if s.label.casefold() in wanted]

    spelled = [s for s in rows if any(o.kind == "long" for o in s.offers)]
    abbreviated = [s for s in rows if any(o.kind == "short" for o in s.offers)]
    new_offers = sum(1 for s in rows for o in s.offers if not o.known)
    silent = [s for s in rows if s.mentions == 0]
    cited = [s for s in rows if s.cited]
    front = [s for s in rows if s.one_word and s.after and majority(s.after.count, s.mentions)]
    tail = [s for s in rows if s.one_word and s.before and majority(s.before.count, s.mentions)]

    print("=" * 76)
    print(f"Rows read:                                   {len(rows)}")
    print(f"Spelled out (the label is a short form):     {len(spelled)}")
    print(f"Abbreviated (the label is a long form):      {len(abbreviated)}")
    print(f"  forms offered that the row does not have:  {new_offers}")
    print(f"One word, same word after in half or more:   {len(front)}")
    print(f"One word, same word before in half or more:  {len(tail)}")
    print(f"Followed by 'et al.' in half or more:        {len(cited)}")
    print(f"Never in body prose:                         {len(silent)}")
    print("Nothing written: this runner only reads.")
    print("=" * 76)

    if args.show:
        _show("Spelled out / abbreviated", [s for s in rows if s.offers], args.show)
        _show("Same word after", front, args.show)
        _show("Same word before", tail, args.show)
        _show("Followed by 'et al.'", cited, args.show)
        _show("Never in body prose", silent, args.show)

    if args.json is not None:
        args.json.write_text(
            json.dumps([signal_payload(s) for s in rows], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"Wrote {len(rows)} rows to {args.json}")
    return 0


def _show(title: str, rows: list[TermSignals], limit: int) -> None:
    print(f"\n## {title} ({len(rows)})")
    for s in rows[:limit]:
        parts = [f"{s.mentions} mentions in {s.documents} documents"]
        if s.after:
            parts.append(f"then '{s.after.word}' {s.after.count}")
        if s.before:
            parts.append(f"after '{s.before.word}' {s.before.count}")
        if s.et_al:
            parts.append(f"'et al.' {s.et_al}")
        print(f"  {s.label}: " + " · ".join(parts))
        for o in s.offers:
            marks = [f"{o.documents} doc"]
            if not o.initials:
                marks.append("letters not on word starts")
            if o.known:
                marks.append("already a form")
            print(f"      {o.kind} form of '{o.beside}': {o.form}  ({', '.join(marks)})")


if __name__ == "__main__":
    raise SystemExit(main())

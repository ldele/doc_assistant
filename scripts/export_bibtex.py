"""Export the library as a BibTeX file (PR 1.5).

Writes to ``docs/library.bib`` by default, or to stdout with
``--stdout``. The output is regenerated wholesale on each run; do not
hand-edit ``docs/library.bib``.

Entry-type classification is heuristic; see ``doc_assistant.bibtex``
for the rules.

Usage::

    python -m scripts.export_bibtex                    # write docs/library.bib
    python -m scripts.export_bibtex --stdout           # print to stdout
    python -m scripts.export_bibtex --out custom.bib   # custom path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from doc_assistant.bibtex import export_bibtex
from doc_assistant.config import PROJECT_ROOT

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of a file")
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "library.bib"),
        help="Output path (default %(default)s)",
    )
    args = parser.parse_args()

    text = export_bibtex()
    if args.stdout:
        print(text)
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    entry_count = text.count("\n@")
    print(f"Wrote {entry_count} entries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

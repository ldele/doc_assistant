"""One-time backfill: put already-ingested demo-corpus papers into the demo folder (ADR-025 F3).

From now on, ingest assigns demo files automatically as they arrive
(``doc_assistant.demo_corpus.assign_new_documents``). This runner covers the documents that were
already in the library **before** that hook existed — it scans the sources dir for files whose
exact bytes match a ``collection: demo`` pin in ``tests/eval/corpus_manifest.yaml`` (the same
content-hash scan ``download_corpus --remove-demo`` uses, so it finds the same files even if they
were renamed) and adds their library rows to the folder.

**It refuses to run twice.** A second pass would re-add exactly the papers the user had removed by
hand, which is the ADR-013 user-wins violation this whole feature is shaped to avoid. ``--force``
overrides that, deliberately and loudly.

Nothing is deleted, no chunk store is touched, no LLM and no network are involved.

Usage::

    python -m scripts.backfill_demo_folder                  # plan only (default)
    python -m scripts.backfill_demo_folder --apply          # assign
    python -m scripts.backfill_demo_folder --apply --force  # run again anyway
    python -m scripts.backfill_demo_folder --dest data/sources   # scan elsewhere
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from doc_assistant import app_settings, demo_corpus
from doc_assistant.library import folder_document_ids

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the memberships (default: dry run — report only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="run again after a completed backfill (re-adds papers you removed by hand)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="sources directory to scan (default: the configured source dir)",
    )
    args = parser.parse_args()

    if args.force and not args.apply:
        parser.error("--force only makes sense with --apply")

    if not demo_corpus.load_demo_pins():
        print("No demo pins available — tests/eval/corpus_manifest.yaml is missing or has no")
        print("`collection: demo` entries. Nothing to back-fill.")
        return 0

    already_done = app_settings.demo_backfill_done()
    if already_done and args.apply and not args.force:
        print("The demo backfill has already run on this install; refusing to run it again.")
        print("Re-running would re-add demo papers you removed from the folder by hand.")
        print("New demo files are assigned automatically at ingest, so this is normally correct.")
        print("\nIf you really want a second pass: --apply --force")
        return 0

    dest = Path(args.dest) if args.dest else app_settings.get_source_dir()
    matches = demo_corpus.backfill_matches(dest)
    if not matches:
        print(f"No demo-collection files found in {dest} (content-hash match); nothing to do.")
        return 0

    folder = demo_corpus.resolve_demo_folder(create=False)
    members = set(folder_document_ids(folder.id)) if folder else set()

    # Counted per category rather than as "everything that isn't assignable", so a never-ingested
    # file is never reported as an existing member (it has no library row at all).
    assignable: list[str] = []
    ambiguous = not_ingested = already = 0
    for match in sorted(matches, key=lambda m: m.path.name):
        if match.ambiguous:
            ambiguous += 1
            state = "AMBIGUOUS (several library rows share this name) — skip; assign in the UI"
        elif match.document_id is None:
            not_ingested += 1
            state = "file only (never ingested) — ingest it and it joins automatically"
        elif match.document_id in members:
            already += 1
            state = "already a member"
        else:
            state = "would assign" if not args.apply else "assigning"
            assignable.append(match.document_id)
        print(f"[demo] {match.path.name}  ->  {state}")

    print("\n--- summary ---")
    print(f"  demo files matched   : {len(matches)}")
    print(f"  never ingested       : {not_ingested}")
    print(f"  already members      : {already}")
    if ambiguous:
        print(f"  skipped (ambiguous)  : {ambiguous} — assign via the Library UI")
    print(f"  to assign            : {len(assignable)}")
    if folder is not None:
        print(f"  folder               : {folder.name!r} ({folder.doc_count} documents)")

    if not assignable:
        print("\nNothing to assign.")
        if not_ingested:
            print(f"{not_ingested} demo file(s) are on disk but not in the library — run an")
            print("ingest and they will join the folder automatically.")
        return 0

    if not args.apply:
        target = demo_corpus.DEFAULT_FOLDER_NAME if folder is None else folder.name
        print(f"\nDry run — nothing written. Re-run with --apply to assign into {target!r}.")
        return 0

    if already_done and args.force:
        print("\n!! --force: the backfill had already completed. Any demo paper you removed")
        print("!! from the folder by hand is about to be put back.")

    result = demo_corpus.apply_assignments(assignable)
    app_settings.mark_demo_backfill_done()
    print(f"\nAssigned {len(result.added)} document(s) into {result.folder_name!r}.")
    print("Use it as a chat scope from the composer's folder selector (ADR-025 F2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

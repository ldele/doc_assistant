"""Detect and resolve duplicate Document rows caused by path+content hashing.

When a document is re-extracted with different content (e.g., switching from
PyMuPDF to Marker), it gets a new doc_hash and a second SQLite row appears.
This script merges duplicates by keeping the row with more chunks.
"""
from collections import defaultdict

from sqlalchemy import select, func

from doc_assistant.db.models import Document, IngestionEvent
from doc_assistant.db.session import session_scope


def main(apply: bool = False):
    with session_scope() as session:
        # Group documents by source path
        docs = session.execute(select(Document)).scalars().all()
        by_source = defaultdict(list)
        for doc in docs:
            by_source[doc.source_original].append(doc)

        duplicates = {src: rows for src, rows in by_source.items() if len(rows) > 1}

        if not duplicates:
            print("No duplicates found.")
            return

        print(f"Found {len(duplicates)} source paths with duplicates:\n")
        for source, rows in duplicates.items():
            print(f"  {source}")
            # Sort by chunk_count desc — keep the row with most chunks
            rows.sort(key=lambda d: -(d.chunk_count or 0))
            keeper = rows[0]
            stale = rows[1:]

            print(f"    KEEP:   id={keeper.id[:8]}  chunks={keeper.chunk_count}  "
                  f"health={keeper.extraction_health}  hash={keeper.doc_hash}")
            for s in stale:
                print(f"    REMOVE: id={s.id[:8]}  chunks={s.chunk_count}  "
                      f"health={s.extraction_health}  hash={s.doc_hash}")
            print()

        if not apply:
            print("--- DRY RUN: no changes made. Re-run with --apply to delete stale rows. ---")
            return

        # Apply mode: delete stale rows
        # We log an IngestionEvent on the keeper recording the merge
        removed = 0
        for source, rows in duplicates.items():
            rows.sort(key=lambda d: -(d.chunk_count or 0))
            keeper = rows[0]
            for stale in rows[1:]:
                event = IngestionEvent(
                    document_id=keeper.id,
                    event_type="dedupe_merge",
                    notes=f"Merged stale row {stale.id[:8]} (hash={stale.doc_hash}, "
                          f"chunks={stale.chunk_count}, health={stale.extraction_health})",
                )
                session.add(event)
                session.delete(stale)
                removed += 1
        print(f"Removed {removed} stale rows.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete stale rows. Without this flag, only reports.")
    args = parser.parse_args()
    main(apply=args.apply)
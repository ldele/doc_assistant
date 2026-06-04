"""Migrate from path+content hashing to content-only hashing.

For each document in SQLite:
1. Read the cached markdown (or re-extract if cache is missing).
2. Compute the new content-only hash.
3. Update SQLite doc_hash column.
4. Update Chroma metadata in both baseline and parent-child stores.
5. Merge any collisions (same content at different paths → keep one row).

Usage:
    python -m scripts.migrate_to_content_hash           # dry run
    python -m scripts.migrate_to_content_hash --apply   # apply changes
"""

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import select

from doc_assistant.config import CACHE_PATH, CHROMA_PATH, DOCS_PATH, PC_CHROMA_PATH
from doc_assistant.db.models import Document, IngestionEvent
from doc_assistant.db.session import session_scope

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def content_only_hash(text: str) -> str:
    """New hash: content only, no path."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def find_cached_text(doc: Document) -> str | None:
    """Try to read the cached markdown for a document."""
    # Try source_cache first
    if doc.source_cache:
        cache_path = Path(doc.source_cache)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

    # Fall back: derive cache path from source_original
    original = Path(doc.source_original)
    if original.exists():
        try:
            relative = original.relative_to(DOCS_PATH)
            derived_cache = CACHE_PATH / relative.with_suffix(".md")
            if derived_cache.exists():
                return derived_cache.read_text(encoding="utf-8")
        except ValueError:
            pass

    return None


def update_chroma_hashes(
    db: Chroma, old_hash: str, new_hash: str, store_name: str, apply: bool
) -> int:
    """Update doc_hash in all chunks matching old_hash. Returns count."""
    results = db.get(where={"doc_hash": old_hash}, include=["metadatas"])
    if not results["ids"]:
        return 0

    count = len(results["ids"])
    if apply:
        updated_metadatas = []
        for meta in results["metadatas"]:
            new_meta = dict(meta)
            new_meta["doc_hash"] = new_hash
            updated_metadatas.append(new_meta)
        # Use underlying collection for metadata-only update
        db._collection.update(
            ids=results["ids"],
            metadatas=updated_metadatas,
        )
    return count


def delete_chroma_chunks(db: Chroma, doc_hash: str, store_name: str) -> int:
    """Delete all chunks for a given hash. Returns count."""
    results = db.get(where={"doc_hash": doc_hash}, include=[])
    if not results["ids"]:
        return 0
    count = len(results["ids"])
    db.delete(ids=results["ids"])
    return count


def main(apply: bool = False) -> None:
    print("=" * 60)
    print("MIGRATION: path+content hash → content-only hash")
    print("=" * 60)
    mode = "APPLY" if apply else "DRY RUN"
    print(f"Mode: {mode}\n")

    # Load embeddings and Chroma stores
    print("Loading Chroma stores...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    pc_db = Chroma(persist_directory=PC_CHROMA_PATH, embedding_function=embeddings)

    # Phase 1: compute new hashes for all documents
    print("\nPhase 1: Computing new content-only hashes...\n")
    with session_scope() as session:
        docs = session.execute(select(Document)).scalars().all()

        if not docs:
            print("No documents in SQLite. Nothing to migrate.")
            return

        # Map: old_hash → (new_hash, document, cached_text)
        migrations: list[dict] = []
        errors: list[str] = []

        for doc in docs:
            text = find_cached_text(doc)
            if text is None:
                errors.append(
                    f"  SKIP {doc.filename} (id={doc.id[:8]}): "
                    f"no cached text found at {doc.source_cache}"
                )
                continue

            new_hash = content_only_hash(text)
            changed = new_hash != doc.doc_hash
            migrations.append(
                {
                    "doc": doc,
                    "old_hash": doc.doc_hash,
                    "new_hash": new_hash,
                    "changed": changed,
                }
            )

        # Report errors
        if errors:
            print(f"  {len(errors)} documents skipped (no cache):")
            for e in errors:
                print(e)
            print()

        # Report what would change
        changed = [m for m in migrations if m["changed"]]
        unchanged = [m for m in migrations if not m["changed"]]
        print(f"  Total documents: {len(docs)}")
        print(f"  Hash changes:    {len(changed)}")
        print(f"  Already correct: {len(unchanged)}")
        print(f"  Skipped:         {len(errors)}")

        if changed:
            print("\n  Changes:")
            for m in changed:
                print(f"    {m['doc'].filename:40s} {m['old_hash']} → {m['new_hash']}")

        # Phase 2: detect collisions (multiple docs → same new hash)
        print("\nPhase 2: Checking for dedup collisions...\n")
        by_new_hash: dict[str, list[dict]] = defaultdict(list)
        for m in migrations:
            by_new_hash[m["new_hash"]].append(m)

        collisions = {h: ms for h, ms in by_new_hash.items() if len(ms) > 1}

        if collisions:
            print(f"  {len(collisions)} collision(s) detected:")
            for new_hash, ms in collisions.items():
                print(f"\n    New hash: {new_hash}")
                # Keep the one with highest chunk_count
                ms.sort(key=lambda x: -(x["doc"].chunk_count or 0))
                keeper = ms[0]
                stale = ms[1:]
                print(
                    f"      KEEP:   {keeper['doc'].filename} "
                    f"(chunks={keeper['doc'].chunk_count}, "
                    f"id={keeper['doc'].id[:8]})"
                )
                for s in stale:
                    print(
                        f"      MERGE:  {s['doc'].filename} "
                        f"(chunks={s['doc'].chunk_count}, "
                        f"id={s['doc'].id[:8]})"
                    )
        else:
            print("  No collisions. Clean migration.")

        if not apply:
            print("\n--- DRY RUN: no changes made. Re-run with --apply. ---")
            return

        # Phase 3: Apply changes
        print("\nPhase 3: Applying changes...\n")

        # First, handle collisions: delete stale rows and their Chroma chunks
        merged = 0
        for _new_hash, ms in collisions.items():
            ms.sort(key=lambda x: -(x["doc"].chunk_count or 0))
            keeper = ms[0]
            for s in ms[1:]:
                stale_doc = s["doc"]
                # Delete Chroma chunks for the stale hash
                bc = delete_chroma_chunks(db, s["old_hash"], "baseline")
                pc = delete_chroma_chunks(pc_db, s["old_hash"], "parent-child")
                print(
                    f"  Merged {stale_doc.filename} into {keeper['doc'].filename} "
                    f"(removed {bc} baseline + {pc} pc chunks)"
                )

                # Log the merge event
                event = IngestionEvent(
                    document_id=keeper["doc"].id,
                    event_type="hash_migration_merge",
                    notes=(
                        f"Merged {stale_doc.id[:8]} "
                        f"({stale_doc.filename}, old_hash={s['old_hash']}) "
                        f"during content-only hash migration"
                    ),
                )
                session.add(event)
                session.delete(stale_doc)
                merged += 1
                # Mark as handled so we don't try to update it below
                s["merged"] = True

        # Now update hashes for remaining documents
        updated = 0
        for m in migrations:
            if m.get("merged"):
                continue
            if not m["changed"]:
                continue

            doc = m["doc"]
            old_hash = m["old_hash"]
            new_hash = m["new_hash"]

            # Update Chroma metadata
            bc = update_chroma_hashes(db, old_hash, new_hash, "baseline", apply=True)
            pc = update_chroma_hashes(pc_db, old_hash, new_hash, "parent-child", apply=True)

            # Update SQLite
            doc.doc_hash = new_hash

            # Log event
            event = IngestionEvent(
                document_id=doc.id,
                event_type="hash_migration",
                notes=f"Migrated {old_hash} → {new_hash} (content-only)",
            )
            session.add(event)
            updated += 1
            print(
                f"  Updated {doc.filename}: {old_hash} → {new_hash} "
                f"({bc} baseline + {pc} pc chunks)"
            )

        print(f"\nDone: {updated} updated, {merged} merged, {len(errors)} skipped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate from path+content to content-only document hashing."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, only reports what would change.",
    )
    args = parser.parse_args()
    main(apply=args.apply)

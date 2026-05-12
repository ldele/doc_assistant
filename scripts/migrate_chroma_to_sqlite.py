"""One-time migration: populate SQLite from existing Chroma metadata.

Reads chunks from Chroma, groups by filename, creates one Document row 
per unique file. Updates Chroma chunks in place to add `document_id` 
linking back to SQLite.

Idempotent: safe to run multiple times. Documents that already exist 
in SQLite (by doc_hash) are skipped.
"""
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import select

from doc_assistant.config import CHROMA_PATH, PC_CHROMA_PATH
from doc_assistant.db.models import Document, IngestionEvent
from doc_assistant.db.session import session_scope


def migrate_chroma(chroma_path: str, label: str) -> None:
    """Migrate documents from a single Chroma collection to SQLite."""
    print(f"\n=== Migrating {label}: {chroma_path} ===")

    if not Path(chroma_path).exists():
        print(f"  Skipped: path does not exist")
        return

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

    data = db.get(include=["metadatas"])
    metadatas = data["metadatas"]
    ids = data["ids"]
    print(f"  Loaded {len(metadatas)} chunks from Chroma")

    # Group chunks by doc_hash (the most reliable identifier we have)
    chunks_by_doc: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for chunk_id, meta in zip(ids, metadatas):
        if not meta:
            continue
        doc_hash = meta.get("doc_hash")
        if not doc_hash:
            continue
        chunks_by_doc[doc_hash].append((chunk_id, meta))

    print(f"  Found {len(chunks_by_doc)} unique documents")

    # For each document, create or skip a SQLite row
    created = skipped = 0
    chunk_updates: list[tuple[str, dict]] = []  # (chunk_id, new_metadata)

    with session_scope() as session:
        for doc_hash, chunk_list in chunks_by_doc.items():
            # Use the first chunk's metadata as representative
            first_meta = chunk_list[0][1]

            # Check if document already exists in SQLite
            existing = session.execute(
                select(Document).where(Document.doc_hash == doc_hash)
            ).scalar_one_or_none()

            if existing:
                document_id = existing.id
                skipped += 1
            else:
                document = Document(
                    filename=first_meta.get("filename", "unknown"),
                    source_original=first_meta.get("source_original", ""),
                    source_cache=first_meta.get("source_cache") or None,
                    doc_hash=doc_hash,
                    format=first_meta.get("format", "unknown"),
                    extractor_used=None,  # unknown from existing data
                    extraction_health=None,  # will be filled in Phase 3.2
                    chunk_count=len(chunk_list),
                    page_count=None,
                    extracted_at=datetime.utcnow(),  # approximate
                )
                session.add(document)
                session.flush()  # generate the UUID
                document_id = document.id

                # Log the migration as an ingestion event
                event = IngestionEvent(
                    document_id=document_id,
                    event_type="migrated_from_chroma",
                    extractor=None,
                    chunks_produced=len(chunk_list),
                    health_status=None,
                    notes=f"Migrated from {label}",
                )
                session.add(event)
                created += 1

            # Schedule chunk metadata updates: add document_id
            for chunk_id, meta in chunk_list:
                if meta.get("document_id") == document_id:
                    continue  # already linked
                new_meta = dict(meta)
                new_meta["document_id"] = document_id
                # Chroma can't store None values
                new_meta = {k: (v if v is not None else "") for k, v in new_meta.items()}
                chunk_updates.append((chunk_id, new_meta))

    print(f"  Created {created} new Document rows, skipped {skipped} existing")

    # Apply chunk metadata updates in batches
    if chunk_updates:
        print(f"  Updating {len(chunk_updates)} chunks with document_id...")
        batch_size = 500
        for i in range(0, len(chunk_updates), batch_size):
            batch = chunk_updates[i:i + batch_size]
            batch_ids = [cid for cid, _ in batch]
            batch_metas = [m for _, m in batch]
            db._collection.update(ids=batch_ids, metadatas=batch_metas)
            print(f"    Updated batch {i // batch_size + 1} ({len(batch)} chunks)")


def main():
    print("Migrating Chroma data into SQLite...")

    # Migrate both Chroma collections (baseline and parent-child)
    migrate_chroma(CHROMA_PATH, "baseline (data/chroma)")
    migrate_chroma(PC_CHROMA_PATH, "parent-child (data/chroma_pc)")

    # Final summary
    print("\n=== Summary ===")
    with session_scope() as session:
        doc_count = session.query(Document).count()
        event_count = session.query(IngestionEvent).count()
        print(f"  Total documents in SQLite: {doc_count}")
        print(f"  Total ingestion events:    {event_count}")


if __name__ == "__main__":
    main()
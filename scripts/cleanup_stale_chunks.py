"""Remove chunks from Chroma whose doc_hash has no corresponding SQLite row.

These are stale artifacts from earlier extractions (e.g., before re-running
with Marker, or before dedup_documents.py). The SQLite row was removed but
the chunks were left behind.
"""
import argparse
from collections import Counter

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import select

from doc_assistant.config import CHROMA_PATH, PC_CHROMA_PATH
from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope


def clean_store(path: str, label: str, apply: bool) -> int:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    db = Chroma(persist_directory=path, embedding_function=embeddings)
    data = db.get(include=["metadatas"])

    # Get all valid hashes from SQLite
    with session_scope() as session:
        valid_hashes = set(session.execute(select(Document.doc_hash)).scalars().all())

    # Find chunks with no SQLite row
    chunk_count_by_hash = Counter()
    chunk_ids_by_hash = {}
    for chunk_id, meta in zip(data["ids"], data["metadatas"]):
        if not meta or not meta.get("doc_hash"):
            continue
        h = meta["doc_hash"]
        chunk_count_by_hash[h] += 1
        chunk_ids_by_hash.setdefault(h, []).append(chunk_id)

    stale_hashes = set(chunk_count_by_hash.keys()) - valid_hashes
    if not stale_hashes:
        print(f"  {label}: no stale chunks")
        return 0

    total_stale = sum(chunk_count_by_hash[h] for h in stale_hashes)
    print(f"  {label}: {len(stale_hashes)} stale hashes, {total_stale} chunks")
    for h in stale_hashes:
        print(f"    {h}: {chunk_count_by_hash[h]} chunks")

    if not apply:
        return total_stale

    # Delete stale chunks
    for h in stale_hashes:
        db.delete(where={"doc_hash": h})
    print(f"  {label}: deleted {total_stale} stale chunks")
    return total_stale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete stale chunks. Without this, dry-run only.")
    args = parser.parse_args()

    print("Cleaning stale chunks from Chroma stores")
    print("=" * 60)

    baseline_removed = clean_store(CHROMA_PATH, "baseline", apply=args.apply)
    pc_removed = clean_store(PC_CHROMA_PATH, "parent-child", apply=args.apply)

    print(f"\nTotal: {baseline_removed + pc_removed} stale chunks")
    if not args.apply:
        print("(dry run; use --apply to actually delete)")


if __name__ == "__main__":
    main()
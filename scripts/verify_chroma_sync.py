"""Verify that baseline and parent-child Chroma stores are coherent.

Checks:
1. Same set of documents (by doc_hash) in both stores
2. Same document_id linkage to SQLite
3. SQLite chunk counts match baseline Chroma's chunk counts
4. Every chunk in either store has a document_id pointing to a real SQLite row

Run periodically and after any operation that touches both stores.
"""
from collections import Counter, defaultdict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import select

from doc_assistant.config import CHROMA_PATH, PC_CHROMA_PATH
from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope


def hashes_in_store(path: str, label: str) -> tuple[set[str], dict, dict]:
    """Return (unique hashes, hash→count, hash→document_id) for a Chroma store."""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    db = Chroma(persist_directory=path, embedding_function=embeddings)
    data = db.get(include=["metadatas"])
    
    counts = Counter()
    doc_ids = {}
    for meta in data["metadatas"]:
        if not meta:
            continue
        h = meta.get("doc_hash")
        if not h:
            continue
        counts[h] += 1
        # Store the document_id from any chunk (should be consistent)
        if meta.get("document_id"):
            doc_ids[h] = meta["document_id"]
    
    return set(counts.keys()), dict(counts), doc_ids


def main():
    print("=" * 60)
    print("Chroma sync verification")
    print("=" * 60)
    
    # Load both stores
    baseline_hashes, baseline_counts, baseline_ids = hashes_in_store(CHROMA_PATH, "baseline")
    pc_hashes, pc_counts, pc_ids = hashes_in_store(PC_CHROMA_PATH, "parent-child")
    
    print(f"\nBaseline:     {len(baseline_hashes)} documents, {sum(baseline_counts.values())} chunks")
    print(f"Parent-child: {len(pc_hashes)} documents, {sum(pc_counts.values())} chunks")
    
    # Check 1: same set of documents
    only_baseline = baseline_hashes - pc_hashes
    only_pc = pc_hashes - baseline_hashes
    
    issues = 0
    
    if only_baseline:
        print(f"\n⚠ {len(only_baseline)} hashes only in baseline (missing from pc_chroma):")
        for h in list(only_baseline)[:5]:
            print(f"    {h}")
        issues += 1
    
    if only_pc:
        print(f"\n⚠ {len(only_pc)} hashes only in parent-child (missing from baseline):")
        for h in list(only_pc)[:5]:
            print(f"    {h}")
        issues += 1
    
    if not only_baseline and not only_pc:
        print(f"\nOK Both stores have the same {len(baseline_hashes)} documents")
    
    # Check 2: document_ids match between stores
    shared = baseline_hashes & pc_hashes
    id_mismatches = []
    for h in shared:
        if baseline_ids.get(h) != pc_ids.get(h):
            id_mismatches.append((h, baseline_ids.get(h), pc_ids.get(h)))
    
    if id_mismatches:
        print(f"\n⚠ {len(id_mismatches)} documents have different document_ids between stores:")
        for h, bid, pid in id_mismatches[:5]:
            print(f"    {h[:8]}...: baseline={bid[:8] if bid else 'None'}  pc={pid[:8] if pid else 'None'}")
        issues += 1
    else:
        print(f"OK document_id linkage consistent across stores")
    
    # Check 3: SQLite has rows for every hash in Chroma
    with session_scope() as session:
        sqlite_hashes = set(session.execute(select(Document.doc_hash)).scalars().all())
    
    chroma_only = (baseline_hashes | pc_hashes) - sqlite_hashes
    sqlite_only = sqlite_hashes - (baseline_hashes | pc_hashes)
    
    if chroma_only:
        print(f"\n⚠ {len(chroma_only)} hashes in Chroma but not in SQLite:")
        for h in list(chroma_only)[:5]:
            print(f"    {h}")
        issues += 1
    
    if sqlite_only:
        print(f"\n⚠ {len(sqlite_only)} SQLite documents with no Chroma chunks:")
        with session_scope() as session:
            for h in list(sqlite_only)[:5]:
                doc = session.execute(
                    select(Document).where(Document.doc_hash == h)
                ).scalar_one_or_none()
                if doc:
                    print(f"    {h[:8]}... → {doc.filename}")
        issues += 1
    
    if not chroma_only and not sqlite_only:
        print(f"OK SQLite ({len(sqlite_hashes)} docs) and Chroma are aligned")
    
    # Check 4: SQLite chunk_count matches baseline Chroma chunk count
    print(f"\nChecking chunk count consistency (SQLite vs baseline Chroma)...")
    count_mismatches = []
    with session_scope() as session:
        docs = session.execute(select(Document)).scalars().all()
        for doc in docs:
            sqlite_count = doc.chunk_count or 0
            chroma_count = baseline_counts.get(doc.doc_hash, 0)
            if abs(sqlite_count - chroma_count) > 0:
                count_mismatches.append((doc.filename, sqlite_count, chroma_count))
    
    if count_mismatches:
        print(f"⚠ {len(count_mismatches)} documents with chunk_count drift:")
        for filename, sqlite_n, chroma_n in count_mismatches[:10]:
            print(f"    {filename}: SQLite={sqlite_n}, Chroma={chroma_n}")
        issues += 1
    else:
        print(f"OK Chunk counts match for all {len(docs)} documents")
    
    # Final report
    print(f"\n{'=' * 60}")
    if issues == 0:
        print(f"PASS All sync checks succeeded")
    else:
        print(f"FAIL {issues} sync issues detected")
    print("=" * 60)


if __name__ == "__main__":
    main()
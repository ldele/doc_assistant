"""Audit metadata coverage across the chunk store."""
import sys
print("Script starting...", flush=True)

from collections import Counter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from doc_assistant.config import CHROMA_PATH


def main():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    data = db.get(include=["metadatas"])
    metadatas = data["metadatas"]
    total = len(metadatas)

    print(f"\n=== Metadata audit: {total} chunks ===\n")

    # Coverage of each field
    fields = ["filename", "format", "page", "section", "doc_hash"]
    print("Field coverage:")
    for field in fields:
        present = sum(1 for m in metadatas if m and m.get(field) not in (None, "", []))
        pct = 100 * present / total if total else 0
        print(f"  {field:20s} {present:5d} / {total} ({pct:.1f}%)")

    # Format breakdown
    print("\nFormat distribution:")
    formats = Counter(m.get("format", "?") for m in metadatas if m)
    for fmt, count in formats.most_common():
        print(f"  {fmt:10s} {count}")

    # Document count
    print("\nUnique documents:")
    docs = Counter(m.get("filename", "?") for m in metadatas if m)
    print(f"  Total: {len(docs)}")
    print(f"  Chunks per doc: min={min(docs.values())}, "
          f"max={max(docs.values())}, "
          f"avg={sum(docs.values()) / len(docs):.1f}")

    # Section coverage examples
    print("\nMost common sections (top 15):")
    sections = Counter(m.get("section") for m in metadatas if m and m.get("section"))
    for section, count in sections.most_common(15):
        truncated = section[:60] + "..." if len(section) > 60 else section
        print(f"  {count:5d}  {truncated}")

    # Sections that look like noise (very long, very short, all caps)
    print("\nPossibly noisy sections (sample of 10):")
    noisy = [s for s in sections if len(s) > 100 or len(s) < 3]
    for section in noisy[:10]:
        print(f"  [len={len(section)}]  {section[:80]}")


if __name__ == "__main__":
    main()
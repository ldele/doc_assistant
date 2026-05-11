"""Clean up chunk metadata in the Chroma store.

Two-pass process:
  1. Dry run (default): report what would change, make no modifications.
  2. Apply (--apply): actually update the metadata.

Safe to run repeatedly. Operates on metadata only — no re-embedding.
"""
import sys
print("Script starting...", flush=True)
import argparse
import re
from collections import Counter, defaultdict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from doc_assistant.config import CHROMA_PATH

# ============================================================
# Classification rules
# ============================================================

REFERENCE_KEYWORDS = [
    "reference",       # "References", "References Cited"
    "bibliograph",     # "Bibliography"
    "works cited",
    "literature cited",
    "citations",
]

# Length bounds for a section name to be considered "real"
MIN_SECTION_LEN = 3
MAX_SECTION_LEN = 200

# A section is considered "document-title-as-section" if it appears
# on more than this fraction of a document's chunks
TITLE_AS_SECTION_THRESHOLD = 0.70


def is_reference_section(section: str) -> bool:
    """True if section looks like a references/bibliography section."""
    if not section:
        return False
    cleaned = re.sub(r"[^a-z\s]", "", section.lower()).strip()
    return any(kw in cleaned for kw in REFERENCE_KEYWORDS)


def is_garbage_section(section: str) -> bool:
    """True if section is clearly not a meaningful heading.

    Conservative: only flags clearly-broken sections.
    Long but valid section titles are kept.
    """
    if not section:
        return False
    s = section.strip()

    # Too short to be meaningful
    if len(s) < MIN_SECTION_LEN:
        return True

    # 200 chars is generous (typical subsection: 30-100 chars).
    if len(s) > MAX_SECTION_LEN:
        return True

    # Strip markdown/formatting characters for analysis
    stripped = re.sub(r"[\W_]", "", s)

    # No alphanumeric content at all (e.g., "**", "##")
    if not stripped:
        return True

    # Mostly digits (page numbers, footnote markers like "#7")
    digits = sum(1 for c in s if c.isdigit())
    letters = sum(1 for c in s if c.isalpha())
    if letters < 3:  # essentially no letters
        return True
    if digits > letters * 2:  # more than 2x as many digits as letters
        return True

    return False


def find_title_sections(chunks_by_doc: dict) -> dict:
    """For each document, find sections that appear on >70% of chunks.
    These are almost certainly the document title, not a real section."""
    title_sections = {}
    for filename, sections in chunks_by_doc.items():
        if not sections:
            continue
        counter = Counter(sections)
        total = len(sections)
        for section, count in counter.most_common(3):
            if section and count / total > TITLE_AS_SECTION_THRESHOLD:
                title_sections[filename] = section
                break
    return title_sections


# ============================================================
# Cleanup logic
# ============================================================

def classify_chunk(meta: dict, title_sections: dict) -> dict:
    """Return updates to apply to this chunk's metadata."""
    section = meta.get("section")
    filename = meta.get("filename", "")
    updates = {}

    # Decision 1: drop references from retrieval
    if is_reference_section(section):
        updates["keep_for_retrieval"] = False
        updates["section_issue"] = "reference"
        return updates

    # Decision 2: clear garbage sections
    if is_garbage_section(section):
        updates["section"] = None
        updates["section_issue"] = "garbage"
        return updates

    # Decision 3: clear title-as-section
    if filename in title_sections and section == title_sections[filename]:
        updates["section"] = None
        updates["section_issue"] = "title_as_section"
        return updates

    # No change needed
    return updates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, only reports what would change.",
    )
    args = parser.parse_args()

    print("Loading embeddings model (this can take 10-30 seconds on first run)...", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    print("Embeddings loaded.", flush=True)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    print("Loading chunks...")
    data = db.get(include=["metadatas"])
    ids = data["ids"]
    metadatas = data["metadatas"]
    total = len(ids)
    print(f"Loaded {total} chunks")

    # First pass: gather sections per document to detect title-as-section
    chunks_by_doc = defaultdict(list)
    for meta in metadatas:
        if meta:
            chunks_by_doc[meta.get("filename", "?")].append(meta.get("section"))

    title_sections = find_title_sections(chunks_by_doc)
    if title_sections:
        print(f"\nDetected {len(title_sections)} probable title-as-section patterns:")
        for fn, sec in title_sections.items():
            truncated = sec[:60] + "..." if len(sec) > 60 else sec
            print(f"  {fn}: {truncated}")

    # Second pass: classify each chunk and gather updates
    print("\nClassifying chunks...")
    pending_updates = []
    issue_counts = Counter()

    for chunk_id, meta in zip(ids, metadatas):
        if not meta:
            continue
        updates = classify_chunk(meta, title_sections)
        if updates:
            issue_counts[updates.get("section_issue", "unknown")] += 1
            pending_updates.append((chunk_id, updates))

    # Report
    print(f"\n=== Cleanup report ===")
    print(f"Total chunks:           {total}")
    print(f"Chunks needing updates: {len(pending_updates)}")
    print(f"\nBreakdown:")
    for issue, count in issue_counts.most_common():
        pct = 100 * count / total
        print(f"  {issue:25s} {count:5d} ({pct:.1f}%)")

    if not pending_updates:
        print("\nNothing to do.")
        return
    
    # Cleanup state
    print("\nCleanup state:")
    flagged = sum(1 for m in metadatas if m and m.get("keep_for_retrieval") is False)
    issues = Counter(m.get("section_issue") for m in metadatas if m and m.get("section_issue"))
    print(f"  Flagged as non-retrievable: {flagged}")
    if issues:
        print("  Cleanup issue tags:")
        for issue, count in issues.most_common():
            print(f"    {issue:25s} {count}")

    if not args.apply:
        print(f"\n--- DRY RUN: no changes applied. ---")
        print(f"Re-run with --apply to actually update metadata.")

        # Show a few examples of each issue type for sanity-checking
        examples_by_issue = defaultdict(list)
        for chunk_id, updates in pending_updates:
            issue = updates.get("section_issue", "unknown")
            if len(examples_by_issue[issue]) < 3:
                idx = ids.index(chunk_id)
                examples_by_issue[issue].append({
                    "filename": metadatas[idx].get("filename"),
                    "section": metadatas[idx].get("section"),
                    "updates": updates,
                })

        print(f"\n--- Sample changes ---")
        for issue, examples in examples_by_issue.items():
            print(f"\n[{issue}]")
            for ex in examples:
                sec = ex["section"]
                sec_display = (sec[:50] + "...") if sec and len(sec) > 50 else sec
                print(f"  {ex['filename']}")
                print(f"    section: {sec_display!r}")
                print(f"    -> {ex['updates']}")
        return

    # Apply mode
    print(f"\nApplying {len(pending_updates)} updates...")
    batch_size = 500
    for i in range(0, len(pending_updates), batch_size):
        batch = pending_updates[i:i + batch_size]
        batch_ids = [chunk_id for chunk_id, _ in batch]
        # Build updated metadata for each chunk: start from existing, apply changes
        batch_metadatas = []
        for chunk_id, updates in batch:
            idx = ids.index(chunk_id)
            new_meta = dict(metadatas[idx])
            new_meta.update(updates)
            # Chroma doesn't allow None values in metadata — replace with empty string
            new_meta = {k: (v if v is not None else "") for k, v in new_meta.items()}
            batch_metadatas.append(new_meta)

        db._collection.update(ids=batch_ids, metadatas=batch_metadatas)

    print("\nUpdates done.", flush=True)
    return

if __name__ == "__main__":
    main()
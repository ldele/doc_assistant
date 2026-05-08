import hashlib
import shutil
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm
import re

from doc_assistant.config import DOCS_PATH, CACHE_PATH, CHROMA_PATH, PDF_EXTRACTOR
from doc_assistant.extractors import extract_to_markdown, is_supported

PAGE_MARKER = re.compile(r"<!--\s*page:(\d+)\s*-->")
HEADING_MARKER = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)


def get_cache_path(original: Path) -> Path:
    relative = original.relative_to(DOCS_PATH)
    return CACHE_PATH / relative.with_suffix(".md")


def is_cache_fresh(original: Path, cached: Path) -> bool:
    if not cached.exists():
        return False
    return cached.stat().st_mtime >= original.stat().st_mtime


def load_or_extract(original: Path) -> str:
    cached = get_cache_path(original)
    if is_cache_fresh(original, cached):
        return cached.read_text(encoding="utf-8")

    print(f"  Extracting: {original.name}")
    text = extract_to_markdown(original, pdf_extractor=PDF_EXTRACTOR)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    return text


def doc_hash(text: str, source: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def get_indexed_hashes(db: Chroma) -> set[str]:
    data = db.get(include=["metadatas"])
    return {m.get("doc_hash") for m in data["metadatas"] if m and m.get("doc_hash")}


def cleanup_orphans(db: Chroma, also_clean_cache: bool = True):
    """Remove chunks (and optionally cached markdown) for files no longer in docs/."""
    data = db.get(include=["metadatas"])
    
    # Group metadatas by doc_hash
    hash_to_meta = {}
    for meta in data["metadatas"]:
        if meta and meta.get("doc_hash"):
            hash_to_meta[meta["doc_hash"]] = meta
    
    orphan_hashes = []
    orphan_caches = []
    for h, meta in hash_to_meta.items():
        original_path = Path(meta.get("source_original", ""))
        if not original_path.exists():
            orphan_hashes.append(h)
            cache_path = Path(meta.get("source_cache", ""))
            if cache_path.exists():
                orphan_caches.append(cache_path)
    
    if not orphan_hashes:
        return
    
    print(f"Cleaning up {len(orphan_hashes)} orphan documents...")
    for h in orphan_hashes:
        db.delete(where={"doc_hash": h})
    
    if also_clean_cache:
        for cache_path in orphan_caches:
            try:
                cache_path.unlink()
            except Exception as e:
                print(f"  Couldn't delete cache {cache_path.name}: {e}")
        print(f"  Removed {len(orphan_caches)} orphan cache files")


def load_documents() -> list[Document]:
    documents = []
    files = [p for p in DOCS_PATH.rglob("*") if p.is_file() and is_supported(p)]
    print(f"Found {len(files)} supported files")

    for path in files:
        try:
            text = load_or_extract(path)
            if not text.strip():
                print(f"  Skipping empty: {path.name}")
                continue

            documents.append(Document(
                page_content=text,
                metadata={
                    "source_original": str(path),
                    "source_cache": str(get_cache_path(path)),
                    "filename": path.name,
                    "format": path.suffix.lower().lstrip("."),
                    "doc_hash": doc_hash(text, str(path)),
                }
            ))
        except Exception as e:
            print(f"  Error on {path.name}: {e}")

    return documents


def extract_chunk_metadata(chunk_text: str, full_text: str, chunk_start: int) -> dict:
    """Find the nearest preceding heading and current page number."""
    # Find page number — last page marker at or before this chunk's start
    text_before = full_text[:chunk_start + len(chunk_text)]
    page_matches = list(PAGE_MARKER.finditer(text_before))
    page = int(page_matches[-1].group(1)) if page_matches else None

    # Find section — last heading at or before this chunk's start
    heading_matches = list(HEADING_MARKER.finditer(text_before))
    section = heading_matches[-1].group(2).strip() if heading_matches else None

    return {"page": page, "section": section}


def clean_chunk_text(text: str) -> str:
    """Remove page markers from displayed text (keep them only for metadata)."""
    return PAGE_MARKER.sub("", text).strip()


def process_one_document(path: Path, db: Chroma, splitter, indexed: set[str]) -> str:
    try:
        text = load_or_extract(path)
        if not text.strip():
            return "skipped"

        h = doc_hash(text, str(path))
        if h in indexed:
            return "skipped"

        # Split with positions tracked
        raw_chunks = splitter.split_text(text)
        if not raw_chunks:
            return "skipped"

        documents = []
        cursor = 0
        for i, chunk_text in enumerate(raw_chunks):
            chunk_start = text.find(chunk_text, cursor)
            if chunk_start == -1:
                chunk_start = cursor
            cursor = chunk_start + len(chunk_text)

            extra = extract_chunk_metadata(chunk_text, text, chunk_start)

            documents.append(Document(
                page_content=clean_chunk_text(chunk_text),
                metadata={
                    "source_original": str(path),
                    "source_cache": str(get_cache_path(path)),
                    "filename": path.name,
                    "format": path.suffix.lower().lstrip("."),
                    "doc_hash": h,
                    "chunk_index": i,
                    "page": extra["page"],
                    "section": extra["section"],
                }
            ))

        db.add_documents(documents)
        indexed.add(h)
        return "added"
    except Exception as e:
        print(f"\n  Error on {path.name}: {e}")
        return "error"


def main(force_rebuild: bool = False, skip_cleanup: bool = False):
    CACHE_PATH.mkdir(exist_ok=True)
    Path(CHROMA_PATH).mkdir(exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"batch_size": 32},
    )

    if force_rebuild:
        print("Force rebuild: clearing vector store...")
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        Path(CHROMA_PATH).mkdir(exist_ok=True)

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    if not skip_cleanup and not force_rebuild:
        cleanup_orphans(db)

    indexed = get_indexed_hashes(db)
    print(f"Already indexed: {len(indexed)} unique documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
    )

    files = [p for p in DOCS_PATH.rglob("*") if p.is_file() and is_supported(p)]
    print(f"Found {len(files)} supported files")

    stats = {"added": 0, "skipped": 0, "error": 0}
    for path in tqdm(files, desc="Processing"):
        result = process_one_document(path, db, splitter, indexed)
        stats[result] += 1

    print(f"\nDone: {stats['added']} added, {stats['skipped']} skipped, {stats['error']} errors")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe the vector store and re-embed everything")
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Skip the orphan cleanup pass")
    args = parser.parse_args()
    main(force_rebuild=args.rebuild, skip_cleanup=args.skip_cleanup)
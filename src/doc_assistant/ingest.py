import hashlib
import shutil
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm
import re
from datetime import datetime, UTC
from sqlalchemy import select

from doc_assistant.db.models import Document as DBDocument, IngestionEvent
from doc_assistant.db.session import session_scope

from doc_assistant.config import DOCS_PATH, CACHE_PATH, CHROMA_PATH, PC_CHROMA_PATH, PDF_EXTRACTOR
from doc_assistant.extractors import extract_to_markdown, is_supported

PAGE_MARKER = re.compile(r"<!--\s*page:(\d+)\s*-->")
HEADING_MARKER = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)

_pc_parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
)
_pc_child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)


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

def cleanup_orphans_sqlite(db_for_metadata: Chroma) -> list[str]:
    """Remove SQLite rows for documents whose source files are gone.
    Returns the list of orphan hashes (for downstream Chroma cleanup).
    """
    data = db_for_metadata.get(include=["metadatas"])
    hash_to_meta = {}
    for meta in data["metadatas"]:
        if meta and meta.get("doc_hash"):
            hash_to_meta[meta["doc_hash"]] = meta

    orphan_hashes = []
    for h, meta in hash_to_meta.items():
        original_path = Path(meta.get("source_original", ""))
        if not original_path.exists():
            orphan_hashes.append(h)

    if not orphan_hashes:
        return []

    print(f"Removing {len(orphan_hashes)} orphan documents from SQLite...")
    with session_scope() as session:
        for h in orphan_hashes:
            doc = session.execute(
                select(DBDocument).where(DBDocument.doc_hash == h)
            ).scalar_one_or_none()
            if doc:
                session.delete(doc)

    return orphan_hashes


def cleanup_orphans_chroma(db: Chroma, orphan_hashes: list[str], 
                            also_clean_cache: bool = False) -> None:
    """Delete chunks for orphan documents from a Chroma store."""
    if not orphan_hashes:
        return
    
    orphan_caches = []
    if also_clean_cache:
        data = db.get(include=["metadatas"])
        for meta in data["metadatas"]:
            if meta and meta.get("doc_hash") in orphan_hashes:
                cache_path = Path(meta.get("source_cache", ""))
                if cache_path.exists():
                    orphan_caches.append(cache_path)
    
    for h in orphan_hashes:
        db.delete(where={"doc_hash": h})
    
    if also_clean_cache:
        for cache_path in set(orphan_caches):
            try:
                cache_path.unlink()
            except Exception as e:
                print(f"  Couldn't delete cache {cache_path.name}: {e}")
        print(f"  Removed {len(set(orphan_caches))} orphan cache files")


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

    heading_matches = list(HEADING_MARKER.finditer(text_before))
    if heading_matches:
        raw_section = heading_matches[-1].group(2).strip()
        section = re.sub(r"[*_`]+", "", raw_section).strip()
        # Empty after stripping = not a real heading
        section = section if section else None
    else:
        section = None

    return {"page": page, "section": section}

def compute_health_signals(documents: list[Document], full_text: str) -> dict:
    """Compute signals for health classification from a list of chunks."""
    if not documents:
        return {
            "chunk_count": 0,
            "avg_chunk_length": 0.0,
            "section_detection_rate": 0.0,
            "reference_flagged_ratio": 0.0,
        }
    
    chunk_lengths = [len(d.page_content) for d in documents]
    sections_detected = sum(1 for d in documents if d.metadata.get("section"))
    # Reference flagging happens later in cleanup; not available at ingest time.
    # We'll leave this at 0 for now and compute it post-cleanup.
    
    return {
        "chunk_count": len(documents),
        "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
        "section_detection_rate": sections_detected / len(documents),
        "reference_flagged_ratio": 0.0,  # post-cleanup signal, not used at ingest
    }

def clean_chunk_text(text: str) -> str:
    """Remove page markers from displayed text (keep them only for metadata)."""
    return PAGE_MARKER.sub("", text).strip()

def upsert_document_in_sqlite(
    filename: str,
    source_original: str,
    source_cache: str | None,
    doc_hash: str,
    format: str,
    extractor_used: str,
    chunk_count: int,
    page_count: int | None = None,
    extraction_health: str | None = None,
) -> str:
    """Create or update a Document row in SQLite. Returns the document's UUID.

    If a document with the same doc_hash exists, return its ID and log a 
    re-ingestion event. Otherwise create a new Document row.
    """
    with session_scope() as session:
        existing = session.execute(
            select(DBDocument).where(DBDocument.doc_hash == doc_hash)
        ).scalar_one_or_none()

        if existing:
            # Re-ingestion of an existing document
            existing.chunk_count = chunk_count
            existing.extractor_used = extractor_used
            existing.extracted_at = datetime.now(UTC)
            if page_count is not None:
                existing.page_count = page_count

            event = IngestionEvent(
                document_id=existing.id,
                event_type="reextract",
                extractor=extractor_used,
                chunks_produced=chunk_count,
                health_status=extraction_health,
            )
            session.add(event)
            return existing.id
        else:
            # New document
            document = DBDocument(
                filename=filename,
                source_original=source_original,
                source_cache=source_cache,
                doc_hash=doc_hash,
                format=format,
                extractor_used=extractor_used,
                extraction_health=extraction_health,
                chunk_count=chunk_count,
                page_count=page_count,
                extracted_at=datetime.now(UTC),
            )
            session.add(document)
            session.flush()  # generate the UUID

            event = IngestionEvent(
                document_id=document.id,
                event_type="extract",
                extractor=extractor_used,
                chunks_produced=chunk_count,
                health_status=extraction_health,
            )
            session.add(event)
            return document.id


def build_parent_child_chunks(text: str, base_metadata: dict) -> list[Document]:
    """Produce child chunks each carrying its parent text in metadata."""
    parents = _pc_parent_splitter.split_text(text)
    children = []
    for parent_idx, parent_text in enumerate(parents):
        for child_idx, child_text in enumerate(_pc_child_splitter.split_text(parent_text)):
            meta = {**base_metadata, "parent_text": parent_text, "parent_index": parent_idx, "child_index": child_idx}
            children.append(Document(page_content=child_text, metadata=meta))
    return children


def process_one_document(path: Path, db: Chroma, pc_db: Chroma, splitter, indexed: set[str]) -> str:
    try:
        text = load_or_extract(path)
        if not text.strip():
            return "skipped"

        h = doc_hash(text, str(path))
        if h in indexed:
            # sync_sqlite_with_chroma(h, db)
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

        pages = [int(m.group(1)) for m in PAGE_MARKER.finditer(text)]
        page_count = max(pages) if pages else None

        # Compute health classification
        from doc_assistant.health import classify_document_health
        signals = compute_health_signals(documents, text)
        health = classify_document_health(
            chunk_count=signals["chunk_count"],
            avg_chunk_length=signals["avg_chunk_length"],
            page_count=page_count,
            section_detection_rate=signals["section_detection_rate"],
            format=path.suffix.lower().lstrip("."),
            reference_flagged_ratio=signals["reference_flagged_ratio"],
        )

        document_id = upsert_document_in_sqlite(
            filename=path.name,
            source_original=str(path),
            source_cache=str(get_cache_path(path)),
            doc_hash=h,
            format=path.suffix.lower().lstrip("."),
            extractor_used=PDF_EXTRACTOR,
            chunk_count=len(documents),
            page_count=page_count,
            extraction_health=health.status,
        )

        # Print a warning if anything's amiss
        if health.status != "healthy":
            print(f"  [{health.status.upper()}] {path.name}: {', '.join(health.reasons)}")

        # Stamp health and document_id onto chunks
        for doc in documents:
            doc.metadata["document_id"] = document_id
            doc.metadata["health"] = health.status

        db.add_documents(documents)

        pc_chunks = build_parent_child_chunks(text, {
            "source_original": str(path),
            "source_cache": str(get_cache_path(path)),
            "filename": path.name,
            "format": path.suffix.lower().lstrip("."),
            "doc_hash": h,
            "document_id": document_id,
            "health": health.status,
        })
        pc_db.add_documents(pc_chunks)

        indexed.add(h)
        return "added"
    except Exception as e:
        print(f"\n  Error on {path.name}: {e}")
        return "error"


def main(force_rebuild: bool = False, skip_cleanup: bool = False):
    CACHE_PATH.mkdir(exist_ok=True)
    Path(CHROMA_PATH).mkdir(exist_ok=True)
    Path(PC_CHROMA_PATH).mkdir(exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
    )

    if force_rebuild:
        print("Force rebuild: clearing vector stores and SQLite document records...")
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        shutil.rmtree(PC_CHROMA_PATH, ignore_errors=True)
        Path(CHROMA_PATH).mkdir(exist_ok=True)
        Path(PC_CHROMA_PATH).mkdir(exist_ok=True)
        with session_scope() as session:
            # Raw delete; relies on database-level ON DELETE CASCADE
            # to clear IngestionEvents, Citations, etc.
            session.execute(DBDocument.__table__.delete())

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    pc_db = Chroma(persist_directory=PC_CHROMA_PATH, embedding_function=embeddings)

    if not skip_cleanup and not force_rebuild:
        orphan_hashes = cleanup_orphans_sqlite(db)
        cleanup_orphans_chroma(db, orphan_hashes, also_clean_cache=True)
        cleanup_orphans_chroma(pc_db, orphan_hashes, also_clean_cache=False)

    indexed = get_indexed_hashes(db) & get_indexed_hashes(pc_db)
    print(f"Already indexed in both stores: {len(indexed)} unique documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
    )

    files = [p for p in DOCS_PATH.rglob("*") if p.is_file() and is_supported(p)]
    print(f"Found {len(files)} supported files")

    stats = {"added": 0, "skipped": 0, "error": 0}
    for path in tqdm(files, desc="Processing"):
        result = process_one_document(path, db, pc_db, splitter, indexed)
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
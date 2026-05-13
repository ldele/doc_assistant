"""One-time script: classify health for documents already in the library."""
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import select

from doc_assistant.config import CHROMA_PATH
from doc_assistant.db.models import Document, IngestionEvent
from doc_assistant.db.session import session_scope
from doc_assistant.health import classify_document_health


def main():
    # Load Chroma to compute signals from existing chunks
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    data = db.get(include=["metadatas", "documents"])

    # Group chunks by doc_hash
    chunks_by_doc = defaultdict(list)
    for chunk_id, meta, content in zip(data["ids"], data["metadatas"], data["documents"]):
        if meta and meta.get("doc_hash"):
            chunks_by_doc[meta["doc_hash"]].append({"meta": meta, "content": content})

    print(f"Found {len(chunks_by_doc)} unique documents in Chroma\n")

    classified = 0
    by_status = defaultdict(int)

    with session_scope() as session:
        for doc_hash, chunks in chunks_by_doc.items():
            # Find the SQLite row
            doc = session.execute(
                select(Document).where(Document.doc_hash == doc_hash)
            ).scalar_one_or_none()
            if not doc:
                print(f"  No SQLite row for hash {doc_hash[:8]}... skipping")
                continue

            # Compute signals
            chunk_lengths = [len(c["content"]) for c in chunks]
            avg_length = sum(chunk_lengths) / len(chunks)
            sections = sum(1 for c in chunks if c["meta"].get("section"))
            ref_flagged = sum(1 for c in chunks if c["meta"].get("keep_for_retrieval") is False)

            health = classify_document_health(
                chunk_count=len(chunks),
                avg_chunk_length=avg_length,
                page_count=doc.page_count,
                section_detection_rate=sections / len(chunks),
                format=doc.format,
                reference_flagged_ratio=ref_flagged / len(chunks),
            )

            doc.extraction_health = health.status
            event = IngestionEvent(
                document_id=doc.id,
                event_type="backfill_health",
                health_status=health.status,
                notes=f"score={health.score} | {'; '.join(health.reasons) if health.reasons else 'no issues'}",
            )
            session.add(event)

            classified += 1
            by_status[health.status] += 1
            
            marker = {"healthy": "  ", "marginal": "~ ", "broken": "! "}[health.status]
            print(f"{marker}{doc.filename:50s} {health.status:8s} score={health.score:3d}")
            if health.reasons:
                print(f"     reasons: {'; '.join(health.reasons)}")

    print(f"\n{classified} documents classified:")
    for status in ["healthy", "marginal", "broken"]:
        if status in by_status:
            print(f"  {status}: {by_status[status]}")


if __name__ == "__main__":
    main()
"""Show documents that aren't classified as healthy."""

from sqlalchemy import desc, select

from doc_assistant.db.models import Document, IngestionEvent
from doc_assistant.db.session import session_scope


def main():
    with session_scope() as session:
        docs = (
            session.execute(
                select(Document).where(Document.extraction_health.in_(["broken", "marginal"]))
            )
            .scalars()
            .all()
        )

        if not docs:
            print("No broken or marginal documents found.")
            return

        for doc in docs:
            print(f"[{doc.extraction_health.upper()}] {doc.filename}")
            print(f"  format={doc.format}  chunks={doc.chunk_count}  pages={doc.page_count}")
            event = session.execute(
                select(IngestionEvent)
                .where(IngestionEvent.document_id == doc.id)
                .where(IngestionEvent.event_type == "backfill_health")
                .order_by(desc(IngestionEvent.timestamp))
                .limit(1)
            ).scalar_one_or_none()
            if event and event.notes:
                print(f"  reasons: {event.notes}")
            print()


if __name__ == "__main__":
    main()

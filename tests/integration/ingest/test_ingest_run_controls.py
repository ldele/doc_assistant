"""A real ``ingest.main`` run, paused, stopped and resumed between documents (ROADMAP 31).

What is proved: a stop ends the loop after the document in flight, that document is in the
library and the rest are *not started*; a pause holds the loop between two documents while what
finished is already in the library, and a resume finishes the run; a failed file carries its
reason; a second run reads the first one's documents as unchanged; and a follow-up run over the
files a stop left indexes exactly those.

Deterministic and offline: fake embedder, temp data dirs, temp SQLite, serial extraction.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import config, ingest
from doc_assistant.db.models import Base
from doc_assistant.db.models import Document as DBDocument
from doc_assistant.ingest.run import NO_TEXT, IngestRun


@pytest.fixture
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Three readable markdown files in temp data dirs, with a fake embedder and a temp SQLite."""
    docs = tmp_path / "sources"
    for d in ("sources", "cache", "chroma", "chroma_pc"):
        (tmp_path / d).mkdir()
    monkeypatch.setattr(config, "DOCS_PATH", docs)
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "cache")
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "PC_CHROMA_PATH", str(tmp_path / "chroma_pc"))
    monkeypatch.setattr(
        ingest, "get_embeddings", lambda name=None: DeterministicFakeEmbedding(size=16)
    )
    for i, topic in enumerate(("retrieval", "reranking", "citations")):
        (docs / f"{i}-{topic}.md").write_text(
            f"# On {topic}\n\n" + f"A paragraph about {topic} and why it matters. " * 40,
            encoding="utf-8",
        )

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)
    try:
        yield docs
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _in_library() -> set[str]:
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        return set(session.execute(select(DBDocument.filename)).scalars())


def _files(library: Path) -> list[Path]:
    return sorted(library.glob("*.md"))


def test_a_stop_ends_the_run_after_the_document_in_flight(library: Path) -> None:
    run = IngestRun()

    def stop_during_the_first(done: int, total: int, current: str | None) -> None:
        if done == 0 and current is not None:
            run.stop()  # asked while the first document is about to be read

    stats = ingest.main(
        files=_files(library), workers=1, on_progress=stop_during_the_first, run=run
    )
    assert stats == {"added": 1, "skipped": 0, "error": 0}
    assert run.stopped
    assert _in_library() == {"0-retrieval.md"}
    counts = run.counts()
    assert (counts["added"], counts["not_started"]) == (1, 2)
    rows, _ = run.page(group="finished")
    assert rows[0].detail is not None and rows[0].detail.endswith("passages")

    # Index the remaining: exactly the two a stop left.
    follow_up = IngestRun()
    stats = ingest.main(files=run.paths("remaining"), workers=1, run=follow_up)
    assert stats == {"added": 2, "skipped": 0, "error": 0}
    assert _in_library() == {"0-retrieval.md", "1-reranking.md", "2-citations.md"}


def test_a_pause_holds_the_loop_between_documents_until_resumed(library: Path) -> None:
    run = IngestRun()

    def pause_during_the_first(done: int, total: int, current: str | None) -> None:
        if done == 0 and current is not None:
            run.pause()

    result: dict[str, dict[str, int]] = {}
    worker = threading.Thread(
        target=lambda: result.setdefault(
            "stats",
            ingest.main(
                files=_files(library), workers=1, on_progress=pause_during_the_first, run=run
            ),
        ),
        daemon=True,
    )
    worker.start()
    for _ in range(1000):
        if run.control() == "paused":
            break
        threading.Event().wait(0.01)
    assert run.control() == "paused"
    # What finished before the pause is already in the library; the rest waits.
    assert _in_library() == {"0-retrieval.md"}
    counts = run.counts()
    assert (counts["added"], counts["waiting"]) == (1, 2)

    run.resume()
    worker.join(timeout=60)
    assert result["stats"] == {"added": 3, "skipped": 0, "error": 0}
    assert not run.stopped and run.control() == "finishing"


def test_a_failed_file_says_why_and_a_second_run_reads_the_rest_as_unchanged(
    library: Path,
) -> None:
    (library / "broken.pdf").write_bytes(b"this is not a PDF at all")
    (library / "empty.md").write_text("", encoding="utf-8")
    run = IngestRun()
    ingest.main(files=sorted(library.iterdir()), workers=1, run=run)

    failed, total = run.page(group="failed")
    assert total == 2
    reasons = {r.name: r.detail for r in failed}
    assert reasons["empty.md"] == NO_TEXT
    assert reasons["broken.pdf"] is not None and reasons["broken.pdf"].startswith(
        "Could not read it: "
    )
    assert {p.name for p in run.paths("failed")} == {"broken.pdf", "empty.md"}

    again = IngestRun()
    ingest.main(files=_files(library), workers=1, run=again)
    rows, _ = again.page(group="finished")
    assert {r.state for r in rows} == {"unchanged"}
    assert {r.detail for r in rows} == {"Already in your library"}

"""Integration tests for the demo-corpus removal core (library.match/remove_pinned_sources).

Files are matched by content hash (rename-proof), library rows by filename; removal rides
``delete_document`` (ADR-014) with ``send2trash`` monkeypatched so nothing is really recycled.
Covers: content-not-name matching, the ingested/file-only/ambiguous triage, secondary-store
chunk sweep, ambiguous skip, and a refused trash failing one match without stopping the batch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.library import (
    SourcePin,
    match_pinned_sources,
    remove_pinned_sources,
)


class FakeChroma:
    """Chunk ids per doc_hash; `.get(where=...)` returns them, `.delete(ids=...)` records them."""

    def __init__(self, ids_by_hash: dict[str, list[str]]) -> None:
        self._ids = ids_by_hash
        self.deleted: list[str] = []

    def get(self, *, where: dict[str, Any], include: list[str]) -> dict[str, Any]:
        return {"ids": list(self._ids.get(where["doc_hash"], []))}

    def delete(self, *, ids: list[str]) -> None:
        self.deleted.extend(ids)


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    yield
    engine.dispose()


def _pin_for(content: bytes, filename: str) -> SourcePin:
    return SourcePin(filename, hashlib.sha256(content).hexdigest(), len(content))


def _seed_doc(filename: str, *, source_original: str, doc_hash: str) -> str:
    with session_scope() as session:
        doc = Document(
            filename=filename,
            source_original=source_original,
            doc_hash=doc_hash,
            format="pdf",
        )
        session.add(doc)
        session.flush()
        return str(doc.id)


def _exists(doc_id: str) -> bool:
    with session_scope() as session:
        return session.get(Document, doc_id) is not None


def test_match_is_by_content_not_name(temp_db: None, tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    content = b"demo paper bytes"
    (sources / "renamed_by_user.pdf").write_bytes(content)
    (sources / "my_own_paper.pdf").write_bytes(b"unrelated user document")

    matches = match_pinned_sources([_pin_for(content, "transformer_vaswani_2017.pdf")], sources)

    assert [m.path.name for m in matches] == ["renamed_by_user.pdf"]
    assert matches[0].document_id is None and matches[0].ambiguous is False


def test_match_links_the_ingested_row(temp_db: None, tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    content = b"ingested demo paper"
    path = sources / "resnet_he_2015.pdf"
    path.write_bytes(content)
    doc_id = _seed_doc("resnet_he_2015.pdf", source_original=str(path), doc_hash="hash-resnet")

    matches = match_pinned_sources([_pin_for(content, "resnet_he_2015.pdf")], sources)

    assert len(matches) == 1
    assert matches[0].document_id == doc_id


def test_match_flags_ambiguous_filename(temp_db: None, tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    content = b"twice-ingested name"
    path = sources / "ntm_graves_2014.pdf"
    path.write_bytes(content)
    _seed_doc("ntm_graves_2014.pdf", source_original=str(path), doc_hash="hash-a")
    _seed_doc("ntm_graves_2014.pdf", source_original=str(path), doc_hash="hash-b")

    matches = match_pinned_sources([_pin_for(content, "ntm_graves_2014.pdf")], sources)

    assert len(matches) == 1
    assert matches[0].ambiguous is True and matches[0].document_id is None


def test_remove_ingested_sweeps_row_file_and_both_stores(
    temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    content = b"ingested demo paper"
    path = sources / "gpipe_huang_2018.pdf"
    path.write_bytes(content)
    doc_id = _seed_doc("gpipe_huang_2018.pdf", source_original=str(path), doc_hash="hash-gpipe")
    live = FakeChroma({"hash-gpipe": ["c1", "c2"]})
    secondary = FakeChroma({"hash-gpipe": ["b1", "b2", "b3"]})
    trashed: list[str] = []

    def fake_trash(p: str) -> None:  # the real send2trash removes the file from its dir
        trashed.append(p)
        Path(p).unlink()

    monkeypatch.setattr("send2trash.send2trash", fake_trash)

    matches = match_pinned_sources([_pin_for(content, "gpipe_huang_2018.pdf")], sources)
    results = remove_pinned_sources(matches, [live, secondary])

    assert len(results) == 1
    r = results[0]
    assert r.deleted_document is True and r.trashed_file is True and r.failed is False
    assert r.chunks_removed == 5  # 2 live + 3 secondary
    assert live.deleted == ["c1", "c2"] and secondary.deleted == ["b1", "b2", "b3"]
    assert trashed == [str(path)]  # once, via delete_document's resolved source
    assert not _exists(doc_id)


def test_remove_file_only_goes_to_trash(
    temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    content = b"downloaded, never ingested"
    path = sources / "vlae_chen_2016.pdf"
    path.write_bytes(content)
    trashed: list[str] = []
    monkeypatch.setattr("send2trash.send2trash", lambda p: trashed.append(p))

    matches = match_pinned_sources([_pin_for(content, "vlae_chen_2016.pdf")], sources)
    results = remove_pinned_sources(matches, [FakeChroma({})])

    assert results[0].deleted_document is False and results[0].trashed_file is True
    assert trashed == [str(path)]


def test_remove_skips_ambiguous_and_leaves_everything(
    temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    content = b"ambiguous"
    path = sources / "dup.pdf"
    path.write_bytes(content)
    id_a = _seed_doc("dup.pdf", source_original=str(path), doc_hash="hash-a")
    id_b = _seed_doc("dup.pdf", source_original=str(path), doc_hash="hash-b")
    monkeypatch.setattr(
        "send2trash.send2trash", lambda p: pytest.fail("ambiguous match must not trash")
    )

    matches = match_pinned_sources([_pin_for(content, "dup.pdf")], sources)
    results = remove_pinned_sources(matches, [FakeChroma({})])

    assert results[0].skipped_ambiguous is True
    assert path.exists() and _exists(id_a) and _exists(id_b)


def test_refused_trash_fails_one_match_and_continues(
    temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    locked_content = b"locked demo paper"
    locked = sources / "locked.pdf"
    locked.write_bytes(locked_content)
    locked_id = _seed_doc("locked.pdf", source_original=str(locked), doc_hash="hash-locked")
    ok_content = b"removable demo paper"
    ok = sources / "removable.pdf"
    ok.write_bytes(ok_content)

    def trash(p: str) -> None:
        if p == str(locked):
            raise OSError("locked")

    monkeypatch.setattr("send2trash.send2trash", trash)

    matches = match_pinned_sources(
        [_pin_for(locked_content, "locked.pdf"), _pin_for(ok_content, "removable.pdf")], sources
    )
    results = remove_pinned_sources(matches, [FakeChroma({})])

    by_name = {r.filename: r for r in results}
    assert by_name["locked.pdf"].failed is True
    assert _exists(locked_id) and locked.exists()  # ADR-014: refused trash leaves the row intact
    assert by_name["removable.pdf"].failed is False and by_name["removable.pdf"].trashed_file

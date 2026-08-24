"""`library/add.py` — the review sheet's data source (AD2).

Two properties carry this module and are asserted hardest:

* **`inspect` mutates nothing a user would notice.** It is the whole basis of spec constraint 2
  ("nothing is copied, registered or indexed before the review sheet is shown and confirmed"). The
  one write it makes is a hash cache on rows it already had to read.
* **The library is not hashed unless a duplicate is actually possible.** Size is the discriminator
  and sha256 only ever confirms. Hashing every registered file on every inspect would be hundreds
  of megabytes on a 97-document corpus, and nothing in the output would differ — so the test
  counts reads rather than trusting the comment.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def temp_database(monkeypatch):
    """Isolated engine + session factory, mirroring tests/unit/test_library.py."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, connection_record):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    from doc_assistant.db import session as session_module

    monkeypatch.setattr(session_module, "_engine", engine)
    monkeypatch.setattr(
        session_module,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    from doc_assistant.db.models import Base

    Base.metadata.create_all(engine)
    yield path
    engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(path)


def _register(rel_path: str, size: int, *, sha: str | None = None) -> None:
    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.add(
            SourceFile(
                rel_path=rel_path,
                format=rel_path.rsplit(".", 1)[-1],
                size=size,
                mtime=0.0,
                source_sha256=sha,
            )
        )


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ============================================================
# Verdicts
# ============================================================


def test_a_supported_new_file_is_added(temp_database, tmp_path):
    from doc_assistant.library.add import inspect

    f = _write(tmp_path / "paper.pdf", b"%PDF-1.4 hello")
    (v,) = inspect([f], source_dir=tmp_path)
    assert v.verdict == "add"
    assert v.name == "paper.pdf"
    assert v.size == len(b"%PDF-1.4 hello")
    assert v.selected_by_default


@pytest.mark.parametrize(
    ("name", "hint"), [("old.doc", "DOCX"), ("paper.tex", "PDF"), ("book.mobi", "EPUB")]
)
def test_an_unsupported_file_carries_get_format_status_verbatim(
    temp_database, tmp_path, name, hint
):
    """The advisory names the conversion target; rewording it here would fork the message."""
    from doc_assistant.extractors import get_format_status
    from doc_assistant.library.add import inspect

    f = _write(tmp_path / name, b"data")
    (v,) = inspect([f], source_dir=tmp_path)
    assert v.verdict == "unsupported"
    assert v.advisory == get_format_status(Path(name))[1]
    assert hint in (v.advisory or "")
    assert not v.selected_by_default


def test_a_missing_path_is_reported_not_silently_dropped(temp_database, tmp_path):
    """A file moved between the drop and the inspect must appear in the sheet saying so."""
    from doc_assistant.library.add import inspect

    (v,) = inspect([tmp_path / "gone.pdf"], source_dir=tmp_path)
    assert v.verdict == "unreadable"
    assert v.advisory and "moved or renamed" in v.advisory
    assert not v.selected_by_default


def test_identical_bytes_under_a_different_name_read_as_duplicate(temp_database, tmp_path):
    """The point of hashing bytes rather than comparing names (ADR-046)."""
    from doc_assistant.library.add import inspect, sha256_file

    body = b"%PDF-1.4 " + b"x" * 500
    registered = _write(tmp_path / "cajal-1899.pdf", body)
    _register("cajal-1899.pdf", len(body), sha=sha256_file(registered))

    candidate = _write(tmp_path / "downloads" / "renamed-copy.pdf", body)
    (v,) = inspect([candidate], source_dir=tmp_path)
    assert v.verdict == "duplicate"
    assert v.duplicate_of == "cajal-1899.pdf"
    assert not v.selected_by_default


def test_same_size_but_different_bytes_is_not_a_duplicate(temp_database, tmp_path):
    """Size opens the question; sha256 answers it. Size alone would be a false positive."""
    from doc_assistant.library.add import inspect, sha256_file

    a = _write(tmp_path / "a.pdf", b"A" * 400)
    _register("a.pdf", 400, sha=sha256_file(a))
    b = _write(tmp_path / "b.pdf", b"B" * 400)

    (v,) = inspect([b], source_dir=tmp_path)
    assert v.verdict == "add"


# ============================================================
# The cost property — the library is not hashed without cause
# ============================================================


def test_no_registered_file_is_hashed_when_no_size_matches(temp_database, tmp_path, monkeypatch):
    """A 97-document corpus must not be read to inspect one unrelated file."""
    from doc_assistant.library import add as add_mod

    for i in range(20):
        _register(f"registered-{i}.pdf", 10_000 + i)

    hashed: list[Path] = []
    real = add_mod.sha256_file
    monkeypatch.setattr(add_mod, "sha256_file", lambda p: (hashed.append(p), real(p))[1])

    candidate = _write(tmp_path / "new.pdf", b"z" * 77)  # a size nothing shares
    (v,) = add_mod.inspect([candidate], source_dir=tmp_path)
    assert v.verdict == "add"
    assert hashed == [], f"hashed {len(hashed)} file(s) with no size collision"


def test_only_the_colliding_row_is_hashed(temp_database, tmp_path, monkeypatch):
    from doc_assistant.library import add as add_mod

    body = b"y" * 300
    _write(tmp_path / "match.pdf", body)
    _register("match.pdf", 300)  # no cached hash — must be computed
    for i in range(10):
        _register(f"other-{i}.pdf", 900 + i)

    hashed: list[Path] = []
    real = add_mod.sha256_file
    monkeypatch.setattr(add_mod, "sha256_file", lambda p: (hashed.append(p), real(p))[1])

    candidate = _write(tmp_path / "candidate.pdf", body)
    (v,) = add_mod.inspect([candidate], source_dir=tmp_path)
    assert v.verdict == "duplicate"
    # the colliding registered file + the candidate itself, and nothing else
    assert {p.name for p in hashed} == {"match.pdf", "candidate.pdf"}


def test_a_computed_hash_is_cached_so_the_next_inspect_is_cheaper(temp_database, tmp_path):
    from sqlalchemy import select

    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import inspect

    body = b"w" * 250
    _write(tmp_path / "reg.pdf", body)
    _register("reg.pdf", 250)
    inspect([_write(tmp_path / "cand.pdf", body)], source_dir=tmp_path)

    with session_scope() as session:
        row = session.execute(
            select(SourceFile).where(SourceFile.rel_path == "reg.pdf")
        ).scalar_one()
        assert row.source_sha256 is not None


def test_inspect_adds_no_registry_rows(temp_database, tmp_path):
    """Constraint 2, asserted rather than trusted: inspecting is not registering."""
    from sqlalchemy import func, select

    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import inspect

    _register("existing.pdf", 10)

    def count() -> int:
        with session_scope() as session:
            return int(session.execute(select(func.count()).select_from(SourceFile)).scalar_one())

    before = count()
    inspect(
        [_write(tmp_path / "a.pdf", b"a"), _write(tmp_path / "b.epub", b"b")], source_dir=tmp_path
    )
    assert count() == before


# ============================================================
# expand_paths — a dropped folder
# ============================================================


def test_a_dropped_folder_expands_recursively(temp_database, tmp_path):
    """Matches `registry.scan_sources` (`rglob`, no depth limit) — grill branch 3."""
    from doc_assistant.library.add import expand_paths

    _write(tmp_path / "top.pdf", b"1")
    _write(tmp_path / "a" / "mid.pdf", b"2")
    _write(tmp_path / "a" / "b" / "deep.pdf", b"3")

    names = {p.name for p in expand_paths([tmp_path])}
    assert names == {"top.pdf", "mid.pdf", "deep.pdf"}


def test_expansion_does_not_duplicate_a_file_named_twice(temp_database, tmp_path):
    """Selecting a folder and a file inside it is one gesture a user can easily make."""
    from doc_assistant.library.add import expand_paths

    f = _write(tmp_path / "a" / "one.pdf", b"1")
    assert len(expand_paths([tmp_path, f])) == 1


def test_expanding_nothing_yields_nothing(temp_database):
    from doc_assistant.library.add import expand_paths

    assert expand_paths([]) == []


# ============================================================
# sort_for_review / summarise
# ============================================================


def test_every_exception_sorts_above_every_clean_add(temp_database, tmp_path):
    """Grill branch 7: the sheet paginates, so page one must hold everything worth seeing."""
    from doc_assistant.library.add import FileVerdict, sort_for_review

    verdicts = [
        FileVerdict(path="1", name="ok1.pdf", verdict="add"),
        FileVerdict(path="2", name="dup.pdf", verdict="duplicate"),
        FileVerdict(path="3", name="ok2.pdf", verdict="add"),
        FileVerdict(path="4", name="bad.doc", verdict="unsupported"),
        FileVerdict(path="5", name="gone.pdf", verdict="unreadable"),
    ]
    kinds = [v.verdict for v in sort_for_review(verdicts)]
    assert kinds.index("add") > max(
        kinds.index("unsupported"), kinds.index("unreadable"), kinds.index("duplicate")
    )


def test_the_sort_is_stable_within_a_group(temp_database):
    from doc_assistant.library.add import FileVerdict, sort_for_review

    verdicts = [FileVerdict(path=str(i), name=f"{i}.pdf", verdict="add") for i in range(5)]
    assert [v.name for v in sort_for_review(verdicts)] == [f"{i}.pdf" for i in range(5)]


def test_summarise_counts_every_verdict_and_the_total(temp_database):
    from doc_assistant.library.add import FileVerdict, summarise

    counts = summarise(
        [
            FileVerdict(path="1", name="a", verdict="add"),
            FileVerdict(path="2", name="b", verdict="add"),
            FileVerdict(path="3", name="c", verdict="duplicate"),
        ]
    )
    assert counts["total"] == 3
    assert counts["add"] == 2
    assert counts["duplicate"] == 1


def test_summarising_nothing_reports_zero_rather_than_an_empty_dict(temp_database):
    from doc_assistant.library.add import summarise

    assert summarise([])["total"] == 0


# ============================================================
# AD3 — apply / undo
# ============================================================


def test_apply_copies_the_file_and_registers_it(temp_database, tmp_path):
    from sqlalchemy import select

    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import apply_add

    root = tmp_path / "library"
    src = _write(tmp_path / "inbox" / "paper.pdf", b"%PDF-1.4 body")

    result = apply_add([src], source_dir=root)
    assert [o.rel_path for o in result.added] == ["paper.pdf"]
    assert result.failed is None
    assert (root / "paper.pdf").read_bytes() == b"%PDF-1.4 body"
    assert src.exists(), "the user's original must not be moved"

    with session_scope() as session:
        row = session.execute(
            select(SourceFile).where(SourceFile.rel_path == "paper.pdf")
        ).scalar_one()
        assert row.origin == "copied"
        assert row.source_sha256 is not None


def test_a_name_collision_does_not_overwrite_the_existing_file(temp_database, tmp_path):
    """Two different papers can share a filename; ADR-043 keeps received content verbatim."""
    from doc_assistant.library.add import apply_add

    root = tmp_path / "library"
    _write(root / "paper.pdf", b"the original")
    src = _write(tmp_path / "inbox" / "paper.pdf", b"a different paper")

    (outcome,) = apply_add([src], source_dir=root).added
    assert outcome.rel_path == "paper-2.pdf"
    assert (root / "paper.pdf").read_bytes() == b"the original"
    assert (root / "paper-2.pdf").read_bytes() == b"a different paper"


def test_apply_stops_at_the_first_failure_and_says_what_it_did_not_try(temp_database, tmp_path):
    """Grill branch 6: keep-or-undo needs to know exactly what landed and what was untouched."""
    from doc_assistant.library.add import apply_add

    root = tmp_path / "library"
    good = _write(tmp_path / "in" / "a.pdf", b"a")
    missing = tmp_path / "in" / "gone.pdf"  # never created
    later = _write(tmp_path / "in" / "c.pdf", b"c")

    result = apply_add([good, missing, later], source_dir=root)
    assert result.stopped_early
    assert [o.rel_path for o in result.added] == ["a.pdf"]
    assert result.failed is not None and result.failed.name == "gone.pdf"
    assert result.not_attempted == [str(later)]
    assert not (root / "c.pdf").exists(), "a file after the failure must not be copied"


def test_undo_removes_exactly_what_apply_added(temp_database, tmp_path):
    from sqlalchemy import func, select

    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import apply_add, undo_add

    root = tmp_path / "library"
    srcs = [_write(tmp_path / "in" / f"{i}.pdf", bytes([i]) * 40) for i in range(3)]
    result = apply_add(srcs, source_dir=root)
    assert len(result.added) == 3

    undone = undo_add([o.rel_path for o in result.added if o.rel_path], source_dir=root)
    assert undone == 3
    assert list(root.glob("*.pdf")) == []
    for s in srcs:
        assert s.exists(), "undo must never touch the user's originals"
    with session_scope() as session:
        assert session.execute(select(func.count()).select_from(SourceFile)).scalar_one() == 0


def test_undo_refuses_to_delete_a_referenced_file(temp_database, tmp_path):
    """The ADR-014 amendment, enforced where it cannot be forgotten: undo owns copies only."""
    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import undo_add

    root = tmp_path / "library"
    kept = _write(root / "theirs.pdf", b"not ours")
    with session_scope() as session:
        session.add(
            SourceFile(rel_path="theirs.pdf", format="pdf", size=8, mtime=0.0, origin="referenced")
        )

    assert undo_add(["theirs.pdf"], source_dir=root) == 0
    assert kept.exists()


def test_reference_mode_refuses_loudly_rather_than_silently_copying(temp_database, tmp_path):
    """AD3b is not built. Accepting the value and doing a copy would be the worst outcome."""
    from doc_assistant.library.add import apply_add

    src = _write(tmp_path / "a.pdf", b"a")
    with pytest.raises(NotImplementedError, match="reference-in-place"):
        apply_add([src], mode="reference", source_dir=tmp_path / "library")


def test_applying_nothing_is_a_valid_empty_result(temp_database, tmp_path):
    from doc_assistant.library.add import apply_add

    result = apply_add([], source_dir=tmp_path / "library")
    assert result.added == [] and result.failed is None and result.not_attempted == []

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
from sqlalchemy import create_engine, event, select
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
    # AD3b: `SourceFile.root_id` carries a literal DEFAULT pointing at the library root, and the
    # FK is enforced — so the row has to exist before anything inserts. `init_db` guarantees this
    # in production (`_seed_library_root`); this fixture stands in for it.
    from doc_assistant.db.models import LIBRARY_ROOT_ID, SourceRoot

    with sessionmaker(bind=engine, future=True)() as s:
        s.add(SourceRoot(id=LIBRARY_ROOT_ID, path=str(Path(path).parent), kind="library"))
        s.commit()
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
    assert v.duplicate_of == "library:cajal-1899.pdf", "a key, not a bare rel_path (AD3b)"
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


def test_undo_never_deletes_a_referenced_file_but_does_drop_its_row(temp_database, tmp_path):
    """The ADR-014 amendment, enforced where it cannot be forgotten.

    Undoing a *reference* add has to remove the row — the user is rejecting the add — while
    leaving the file alone, because the app never owned it. Before AD3b undo refused the row
    outright, which after AD3b would have stranded it in the library forever.
    """
    from sqlalchemy import func, select

    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import apply_add, undo_add

    theirs = _write(tmp_path / "zotero" / "theirs.pdf", b"not ours")
    result = apply_add([theirs], mode="reference", source_dir=tmp_path / "library")
    key = result.added[0].key
    assert key is not None

    assert undo_add([key], source_dir=tmp_path / "library") == 1
    assert theirs.exists(), "undo must never delete a file the app does not own"
    with session_scope() as session:
        assert session.execute(select(func.count()).select_from(SourceFile)).scalar_one() == 0


def test_reference_mode_registers_in_place_and_copies_nothing(temp_database, tmp_path):
    """AD3b: the file is registered where it lives. Not one byte is written into the library."""
    from sqlalchemy import select

    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import apply_add

    theirs = _write(tmp_path / "zotero" / "paper.pdf", b"theirs")
    library = tmp_path / "library"

    result = apply_add([theirs], mode="reference", source_dir=library)

    assert result.failed is None and len(result.added) == 1
    assert theirs.exists(), "the original must stay put"
    assert list(library.rglob("*.pdf")) == [], "reference mode must copy nothing into the library"
    with session_scope() as session:
        row = session.execute(select(SourceFile)).scalar_one()
        assert row.origin == "referenced"
        assert row.rel_path == "paper.pdf"
        assert row.root_id != "library", "a referenced file gets its own root"


def test_a_duplicate_is_found_even_when_it_lives_under_a_referenced_root(temp_database, tmp_path):
    """AD3b regression guard: duplicate detection has to span roots.

    Before the root join, `_size_index` resolved every registered `rel_path` against the *library*
    folder — so a file registered under a referenced root resolved to a path that does not exist,
    the read failed, and the candidate came back a clean `add`. The library would then hold the
    same bytes twice, which is exactly what the duplicate gate exists to prevent.
    """
    from doc_assistant.library.add import apply_add, inspect

    library = tmp_path / "library"
    zotero = tmp_path / "zotero"
    original = _write(zotero / "paper.pdf", b"%PDF-1.4 the same bytes")
    apply_add([original], mode="reference", source_dir=library)

    # The same bytes arriving again under a different name, from somewhere else entirely.
    again = _write(tmp_path / "inbox" / "copy-of-paper.pdf", b"%PDF-1.4 the same bytes")
    (v,) = inspect([again], source_dir=library)

    assert v.verdict == "duplicate"
    assert v.duplicate_of is not None and v.duplicate_of.endswith(":paper.pdf")


def test_referencing_two_files_from_one_folder_makes_one_root(temp_database, tmp_path):
    """Roots are per-directory, so a twenty-paper Zotero folder does not mint twenty roots."""
    from sqlalchemy import select

    from doc_assistant.db.models import SourceRoot
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import apply_add

    a = _write(tmp_path / "zotero" / "a.pdf", b"a")
    b = _write(tmp_path / "zotero" / "b.pdf", b"bb")
    apply_add([a, b], mode="reference", source_dir=tmp_path / "library")

    with session_scope() as session:
        referenced = (
            session.execute(select(SourceRoot).where(SourceRoot.kind == "referenced"))
            .scalars()
            .all()
        )
        assert len(referenced) == 1


def test_a_file_already_inside_the_library_references_the_library_root(temp_database, tmp_path):
    """No second root pointing at the same folder — that would give one file two origins."""
    from sqlalchemy import select

    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import apply_add

    library = tmp_path / "library"
    inside = _write(library / "already.pdf", b"here")

    apply_add([inside], mode="reference", source_dir=library)

    with session_scope() as session:
        row = session.execute(select(SourceFile)).scalar_one()
        assert row.root_id == "library"
        assert row.origin == "referenced", (
            "the app still did not put it there, so it may not bin it"
        )


def test_applying_nothing_is_a_valid_empty_result(temp_database, tmp_path):
    from doc_assistant.library.add import apply_add

    result = apply_add([], source_dir=tmp_path / "library")
    assert result.added == [] and result.failed is None and result.not_attempted == []


# ============================================================
# Failure containment and the delete guard (review findings 1, 4, 5)
# ============================================================


def test_re_registering_a_file_is_a_reported_failure_not_an_exception(temp_database, tmp_path):
    """A DB constraint is a failure like any other — it must not escape with the whole report.

    `apply_add` caught `(OSError, ValueError)` only, so the registry's `(root_id, rel_path)`
    uniqueness raised `IntegrityError` straight out of the function. The caller lost `added`
    entirely, which is the one thing undo needs.
    """
    from doc_assistant.library.add import apply_add

    library = tmp_path / "library"
    paper = _write(tmp_path / "zotero" / "paper.pdf", b"%PDF-1.4 z")

    first = apply_add([paper], mode="reference", source_dir=library)
    assert len(first.added) == 1

    other = _write(tmp_path / "zotero" / "other.pdf", b"%PDF-1.4 o")
    again = apply_add([other, paper], mode="reference", source_dir=library)

    assert [o.name for o in again.added] == ["other.pdf"], "what landed is still reported"
    assert again.failed is not None and again.failed.name == "paper.pdf"
    assert "already registered" in (again.failed.error or ""), again.failed.error
    assert again.stopped_early


def test_a_copy_that_fails_to_register_does_not_survive_on_disk(temp_database, tmp_path):
    """The database rolls back; the filesystem does not — so the copy has to be undone by hand.

    An orphaned copy is invisible to `undo_add` (no row, no key) and perfectly visible to the
    next `scan_root`, which would adopt a document the user was told had failed.
    """
    from doc_assistant.library import add as add_mod

    library = tmp_path / "library"
    paper = _write(tmp_path / "in" / "paper.pdf", b"%PDF-1.4 p")

    def boom(_path):
        raise OSError("disk went away mid-add")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(add_mod, "sha256_file", boom)
        result = add_mod.apply_add([paper], source_dir=library)

    assert result.failed is not None
    assert list(library.glob("*.pdf")) == [], "the half-added copy must not be left behind"
    assert paper.exists(), "the user's original is never touched"


def test_undo_will_not_delete_a_file_outside_the_library_root(temp_database, tmp_path):
    """The confirmed cross-root deletion: a key naming a referenced file binned a library one.

    `undo_add` built the delete path as ``library / rel_path`` whatever root the row belonged to,
    so a referenced row marked ``copied`` reached across and removed an unrelated same-named
    document out of the library folder.
    """
    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest import registry
    from doc_assistant.library.add import apply_add, undo_add

    library = tmp_path / "library"
    added = _write(tmp_path / "zotero" / "paper.pdf", b"%PDF-1.4 mine")
    _write(tmp_path / "zotero" / "notes.pdf", b"%PDF-1.4 zotero notes")
    victim = _write(library / "notes.pdf", b"%PDF-1.4 a DIFFERENT, months-old document")

    apply_add([added], mode="reference", source_dir=library)
    with session_scope() as session:
        registry.scan_sources(session, library)
        row = session.execute(
            select(SourceFile).where(SourceFile.rel_path == "notes.pdf")
        ).scalars()
        keys = [registry.source_key(r.root_id, r.rel_path) for r in row if r.root_id != "library"]

    undo_add(keys, source_dir=library)
    assert victim.exists(), "a row under another root must never reach into the library folder"


def test_a_scan_never_claims_ownership_of_a_referenced_folder(temp_database, tmp_path):
    """`origin` decides whether delete may bin the file, so a scan must state it.

    It fell through to the column DEFAULT (``'copied'``), so referencing one paper out of a
    folder made the app claim every other document in it.
    """
    from doc_assistant.db.models import SourceFile, SourceRoot
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest import registry
    from doc_assistant.library.add import apply_add

    library = tmp_path / "library"
    added = _write(tmp_path / "zotero" / "paper.pdf", b"%PDF-1.4 mine")
    _write(tmp_path / "zotero" / "never-added.pdf", b"%PDF-1.4 not mine")

    apply_add([added], mode="reference", source_dir=library)
    with session_scope() as session:
        registry.scan_sources(session, library)
    with session_scope() as session:
        rows = session.execute(
            select(SourceFile.rel_path, SourceFile.origin, SourceRoot.kind).join(
                SourceRoot, SourceFile.root_id == SourceRoot.id
            )
        ).all()

    claimed = [rel for rel, origin, kind in rows if kind == "referenced" and origin == "copied"]
    assert claimed == [], f"the app claimed files it did not put there: {claimed}"


def test_undo_declines_to_delete_outside_its_window_but_still_drops_the_row(
    temp_database, tmp_path
):
    """Undo is an undo, not a delete-by-key for anything the app ever copied in.

    The row still goes — leaving a file the next scan re-registers is recoverable, deleting one
    is not (KI-49).
    """
    from datetime import timedelta

    from doc_assistant.db.models import SourceFile, _utcnow
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import UNDO_DELETE_WINDOW_SECONDS, apply_add, undo_add

    library = tmp_path / "library"
    paper = _write(tmp_path / "in" / "paper.pdf", b"%PDF-1.4 p")
    result = apply_add([paper], source_dir=library)
    key = result.added[0].key
    landed = library / "paper.pdf"
    assert landed.exists()

    with session_scope() as session:
        row = session.execute(select(SourceFile)).scalar_one()
        row.first_seen = _utcnow() - timedelta(seconds=UNDO_DELETE_WINDOW_SECONDS + 60)

    assert undo_add([key], source_dir=library) == 1, "the row still goes"
    assert landed.exists(), "a stale key must not bin a file the user has kept"


# ============================================================
# KI-51 — undo removes the document the add produced, and un-references an emptied root
# ============================================================


class _FakeChroma:
    """The two calls `purge_document_record` makes, and a record of what was deleted.

    A fake rather than a real Chroma (cpc §13): the assertion is *which ids undo asked to remove*,
    and a real store would add an embedding model, a directory and several seconds to learn
    nothing.
    """

    def __init__(self, ids_by_hash: dict[str, list[str]] | None = None) -> None:
        self._ids = ids_by_hash or {}
        self.deleted: list[str] = []

    def get(self, where=None, include=None):
        wanted = (where or {}).get("doc_hash")
        return {"ids": list(self._ids.get(wanted, []))}

    def delete(self, ids=None):
        self.deleted.extend(ids or [])


def _document_at(path, *, doc_hash: str, added_at=None) -> str:
    """Insert a `Document` row pointing at `path`, as a completed ingest would have left it."""
    from uuid import uuid4

    from doc_assistant.db.models import Document, _utcnow
    from doc_assistant.db.session import session_scope

    doc_id = str(uuid4())
    with session_scope() as session:
        session.add(
            Document(
                id=doc_id,
                filename=path.name,
                source_original=str(path),
                source_cache=None,
                doc_hash=doc_hash,
                format="pdf",
                extractor_used="pymupdf",
                extraction_health="ok",
                chunk_count=3,
                page_count=1,
                added_at=added_at or _utcnow(),
            )
        )
    return doc_id


def test_undo_removes_the_document_the_add_produced(temp_database, tmp_path):
    """KI-51 part 1. Undo used to drop the registry row and stop, leaving the library listing —
    and able to cite — a document whose file undo had just deleted."""
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import apply_add, undo_add

    library = tmp_path / "library"
    src = _write(tmp_path / "inbox" / "paper.pdf", b"%PDF-1.4 hello")
    result = apply_add([src], mode="copy", source_dir=library)
    key = result.added[0].key

    doc_id = _document_at(library / "paper.pdf", doc_hash="h-1")
    chroma = _FakeChroma({"h-1": ["c1", "c2", "c3"]})

    assert undo_add([key], source_dir=library, chroma_db=chroma) == 1
    assert not (library / "paper.pdf").exists(), "the copy still goes"
    with session_scope() as session:
        assert session.get(Document, doc_id) is None, "the document must go with the row"
    assert chroma.deleted == ["c1", "c2", "c3"], "its chunks must go too, or it stays retrievable"


def test_undo_leaves_a_document_it_cannot_prove_this_add_created(temp_database, tmp_path):
    """The guard. A path may already carry a document the user has had for months — under ADR-047
    a replacement even inherits its id — and undo must not destroy it on their behalf."""
    from datetime import timedelta

    from doc_assistant.db.models import Document, _utcnow
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import UNDO_DELETE_WINDOW_SECONDS, apply_add, undo_add

    library = tmp_path / "library"
    src = _write(tmp_path / "inbox" / "paper.pdf", b"%PDF-1.4 hello")
    result = apply_add([src], mode="copy", source_dir=library)
    key = result.added[0].key

    old = _utcnow() - timedelta(seconds=UNDO_DELETE_WINDOW_SECONDS + 60)
    doc_id = _document_at(library / "paper.pdf", doc_hash="h-old", added_at=old)
    chroma = _FakeChroma({"h-old": ["c1"]})

    assert undo_add([key], source_dir=library, chroma_db=chroma) == 1, "the row still goes"
    with session_scope() as session:
        assert session.get(Document, doc_id) is not None, "a pre-existing document must survive"
    assert chroma.deleted == [], "and keep its chunks"


def test_undo_without_a_chroma_handle_leaves_the_document_alone(temp_database, tmp_path):
    """Half-removing is worse than not starting: dropping the row while its chunks stayed in the
    index would leave the chunks retrievable with no document behind them."""
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import apply_add, undo_add

    library = tmp_path / "library"
    src = _write(tmp_path / "inbox" / "paper.pdf", b"%PDF-1.4 hello")
    key = apply_add([src], mode="copy", source_dir=library).added[0].key
    doc_id = _document_at(library / "paper.pdf", doc_hash="h-1")

    assert undo_add([key], source_dir=library) == 1
    with session_scope() as session:
        assert session.get(Document, doc_id) is not None


def test_undoing_the_last_file_un_references_the_root(temp_database, tmp_path):
    """KI-51 part 2. The root used to survive, so the next scan re-found the file as `new` and the
    next "index all" re-ingested exactly what the user had undone — without asking."""
    from doc_assistant.db.models import SourceRoot
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import apply_add, undo_add

    library = tmp_path / "library"
    src = _write(tmp_path / "mine" / "paper.pdf", b"%PDF-1.4 the user's own file")
    key = apply_add([src], mode="reference", source_dir=library).added[0].key

    with session_scope() as session:
        roots = session.execute(select(SourceRoot)).scalars().all()
        assert any(r.kind == "referenced" for r in roots), "precondition: a root was registered"

    assert undo_add([key], source_dir=library) == 1
    assert src.exists(), "the ADR-014 amendment still holds — the file is untouched"
    with session_scope() as session:
        roots = session.execute(select(SourceRoot)).scalars().all()
        assert not any(r.kind == "referenced" for r in roots), "the reference must be withdrawn"


def test_a_root_that_still_holds_files_is_kept(temp_database, tmp_path):
    """Only an *emptied* root goes.

    Undoing one file of several must not un-reference the whole folder.
    """
    from doc_assistant.db.models import SourceRoot
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import apply_add, undo_add

    library = tmp_path / "library"
    one = _write(tmp_path / "mine" / "one.pdf", b"%PDF-1.4 one")
    two = _write(tmp_path / "mine" / "two.pdf", b"%PDF-1.4 two")
    result = apply_add([one, two], mode="reference", source_dir=library)
    first = result.added[0].key

    assert undo_add([first], source_dir=library) == 1
    with session_scope() as session:
        roots = session.execute(select(SourceRoot)).scalars().all()
        assert any(r.kind == "referenced" for r in roots), "the other file still lives there"
    assert two.exists()


def test_the_library_root_is_never_dropped(temp_database, tmp_path):
    """An empty library is a normal state, not a stale reference — it is the app's own folder."""
    from doc_assistant.db.models import LIBRARY_ROOT_ID, SourceRoot
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.add import apply_add, undo_add

    library = tmp_path / "library"
    src = _write(tmp_path / "inbox" / "paper.pdf", b"%PDF-1.4 hello")
    key = apply_add([src], mode="copy", source_dir=library).added[0].key

    assert undo_add([key], source_dir=library) == 1
    with session_scope() as session:
        assert session.get(SourceRoot, LIBRARY_ROOT_ID) is not None

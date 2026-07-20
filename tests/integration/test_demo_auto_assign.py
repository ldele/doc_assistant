"""Integration tests for demo-corpus auto-assign (ADR-025 F3).

The database half of ``doc_assistant.demo_corpus`` plus the backfill runner: which documents land
in the folder, which ones never come back once removed (the ADR-013 user-wins guarantee), and the
run-once guard on the backfill. Temp file-backed SQLite + a temp manifest + a temp settings file —
no LLM, no model load, no network, no writes to the real data home.

Contract: ``docs/specs/feature-corpus-folders-demo.md`` (M1, M2, M5, M6, M7, M8).
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import app_settings, demo_corpus
from doc_assistant.db.models import Base, Document
from doc_assistant.db.session import session_scope
from doc_assistant.ingest.store import get_document_row_hashes
from doc_assistant.library import (
    delete_folder,
    folder_document_ids,
    remove_documents_from_folder,
    rename_folder,
)
from doc_assistant.sources_manifest import sha256_file

_DEMO_BYTES = b"%PDF-1.4 pretend this is alexnet"
_OTHER_BYTES = b"%PDF-1.4 a private paper of my own"


@dataclass
class Env:
    """A throwaway install: temp DB, temp sources dir, temp manifest, temp settings.json."""

    root: Path
    sources: Path
    demo_file: Path
    other_file: Path


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )

    sources = tmp_path / "sources"
    sources.mkdir()
    demo_file = sources / "alexnet.pdf"
    demo_file.write_bytes(_DEMO_BYTES)
    other_file = sources / "private_notes.pdf"
    other_file.write_bytes(_OTHER_BYTES)

    manifest = tmp_path / "corpus_manifest.yaml"
    manifest.write_text(
        textwrap.dedent(
            f"""\
            documents:
              - filename: "alexnet.pdf"
                collection: demo
                sha256: {sha256_file(demo_file)}
                bytes: {demo_file.stat().st_size}
            """
        ),
        encoding="utf-8",
    )

    # Never touch the real data home: settings.json holds the demo folder id + backfill flag.
    monkeypatch.setattr(demo_corpus, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    try:
        yield Env(root=tmp_path, sources=sources, demo_file=demo_file, other_file=other_file)
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _seed_doc(path: Path, doc_hash: str) -> str:
    """Commit a Document row the way ``process_one_document`` does, last step first."""
    with session_scope() as session:
        doc = Document(
            filename=path.name,
            source_original=str(path),
            doc_hash=doc_hash,
            format="pdf",
        )
        session.add(doc)
        session.flush()
        return str(doc.id)


def _demo_folder_id() -> str | None:
    folder = demo_corpus.resolve_demo_folder(create=False)
    return folder.id if folder else None


# --- the ingest hook -------------------------------------------------------------------------- #


def test_new_demo_document_joins_the_folder_and_a_private_one_does_not(env: Env) -> None:
    demo_id = _seed_doc(env.demo_file, "h-demo")
    other_id = _seed_doc(env.other_file, "h-other")

    result = demo_corpus.assign_new_documents({"h-demo", "h-other"})

    assert result.folder_id is not None
    assert result.added == [demo_id]
    assert folder_document_ids(result.folder_id) == [demo_id]
    assert other_id not in folder_document_ids(result.folder_id)


def test_a_removed_document_is_never_put_back_by_a_later_ingest(env: Env) -> None:
    """The ADR-013 user-wins guarantee (M1/M2), exercised through the real predicate.

    ``ingest.main`` hands the hook ``get_document_row_hashes()`` diffed around the processing
    loop. A re-ingest of an already-known file commits no new row, so the diff is empty and the
    hook is a no-op — which is *why* a hand-removed document stays removed. Testing the diff
    itself rather than a fixed argument is the point: keying the hook on
    ``process_one_document``'s "added" instead would re-add here, silently.
    """
    before_first = get_document_row_hashes()
    demo_id = _seed_doc(env.demo_file, "h-demo")
    demo_corpus.assign_new_documents(get_document_row_hashes() - before_first)

    folder_id = _demo_folder_id()
    assert folder_id is not None
    assert folder_document_ids(folder_id) == [demo_id]

    remove_documents_from_folder(folder_id, [demo_id])
    assert folder_document_ids(folder_id) == []

    # Second ingest run over the same, unchanged file: no new row, so nothing to consider.
    before_second = get_document_row_hashes()
    new_hashes = get_document_row_hashes() - before_second
    assert new_hashes == set()
    demo_corpus.assign_new_documents(new_hashes)

    assert folder_document_ids(folder_id) == []


def test_no_manifest_is_a_silent_no_op(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """A frozen build has no `tests/` — it must ingest normally and create no folder (M10)."""
    monkeypatch.setattr(demo_corpus, "MANIFEST_PATH", env.root / "absent.yaml")
    _seed_doc(env.demo_file, "h-demo")

    assert demo_corpus.assign_new_documents({"h-demo"}).folder_id is None
    assert _demo_folder_id() is None


def test_nothing_to_assign_creates_no_folder(env: Env) -> None:
    """An empty "Demo corpus" nobody asked for is the honest-empty rule inverted (M7)."""
    _seed_doc(env.other_file, "h-other")

    assert demo_corpus.assign_new_documents({"h-other"}).folder_id is None
    assert demo_corpus.assign_new_documents(set()).folder_id is None
    assert _demo_folder_id() is None


def test_hash_with_no_document_row_is_skipped(env: Env) -> None:
    assert demo_corpus.assign_new_documents({"h-never-committed"}).folder_id is None


def test_assignment_is_idempotent(env: Env) -> None:
    demo_id = _seed_doc(env.demo_file, "h-demo")

    first = demo_corpus.apply_assignments([demo_id])
    second = demo_corpus.apply_assignments([demo_id])

    assert first.added == [demo_id] and first.already_member == 0
    assert second.added == [] and second.already_member == 1
    assert folder_document_ids(first.folder_id or "") == [demo_id]


def test_ingest_hook_never_fails_an_otherwise_good_ingest(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documents are indexed and answerable either way — a folder problem is a warning."""
    from doc_assistant import ingest

    def _boom(_: object) -> None:
        raise RuntimeError("settings.json is on a dead network drive")

    monkeypatch.setattr(demo_corpus, "assign_new_documents", _boom)
    ingest._assign_demo_folder({"h-demo"})  # must not raise


# --- the folder pointer ----------------------------------------------------------------------- #


def test_renaming_the_folder_is_respected(env: Env) -> None:
    """ADR-025 promises an ordinary, renamable folder. A name-keyed lookup would quietly make a
    second "Demo corpus" on the next demo ingest (M5)."""
    demo_id = _seed_doc(env.demo_file, "h-demo")
    folder_id = demo_corpus.assign_new_documents({"h-demo"}).folder_id
    assert folder_id is not None
    rename_folder(folder_id, "Sutskever list")

    resolved = demo_corpus.resolve_demo_folder(create=True)

    assert resolved is not None
    assert resolved.id == folder_id
    assert resolved.name == "Sutskever list"
    assert folder_document_ids(folder_id) == [demo_id]


def test_deleting_the_folder_is_respected_until_a_new_demo_document_arrives(env: Env) -> None:
    """No tombstone (M6): the per-document removal is what never gets re-fought, and ingesting a
    *new* demo paper is a fresh action that reasonably wants a home."""
    _seed_doc(env.demo_file, "h-demo")
    folder_id = demo_corpus.assign_new_documents({"h-demo"}).folder_id
    assert folder_id is not None
    assert delete_folder(folder_id) is True

    assert demo_corpus.resolve_demo_folder(create=False) is None  # stale id → not resurrected

    second = env.sources / "resnet.pdf"
    second.write_bytes(_DEMO_BYTES)  # same bytes: still a pin match
    _seed_doc(second, "h-demo-2")
    again = demo_corpus.assign_new_documents({"h-demo-2"})

    assert again.folder_id is not None and again.folder_id != folder_id
    assert app_settings.get_demo_folder_id() == again.folder_id


def test_an_existing_hand_made_folder_of_the_same_name_is_adopted(env: Env) -> None:
    """`create_folder` is an idempotent get-or-create, so F3 never duplicates a user's folder."""
    from doc_assistant.library import create_folder

    mine = create_folder(demo_corpus.DEFAULT_FOLDER_NAME)
    _seed_doc(env.demo_file, "h-demo")

    assert demo_corpus.assign_new_documents({"h-demo"}).folder_id == mine.id


# --- the backfill runner ---------------------------------------------------------------------- #


def _run_backfill(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    import sys

    from scripts import backfill_demo_folder

    monkeypatch.setattr(sys, "argv", ["backfill_demo_folder", *argv])
    return backfill_demo_folder.main()


def test_backfill_dry_run_writes_nothing(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_doc(env.demo_file, "h-demo")

    assert _run_backfill(monkeypatch, "--dest", str(env.sources)) == 0

    assert "Dry run" in capsys.readouterr().out
    assert _demo_folder_id() is None
    assert app_settings.demo_backfill_done() is False


def test_backfill_assigns_then_refuses_to_run_twice(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second pass would re-add exactly the papers the user removed by hand (M8)."""
    demo_id = _seed_doc(env.demo_file, "h-demo")
    _seed_doc(env.other_file, "h-other")

    assert _run_backfill(monkeypatch, "--apply", "--dest", str(env.sources)) == 0
    folder_id = _demo_folder_id()
    assert folder_id is not None
    assert folder_document_ids(folder_id) == [demo_id]
    assert app_settings.demo_backfill_done() is True

    remove_documents_from_folder(folder_id, [demo_id])
    capsys.readouterr()

    assert _run_backfill(monkeypatch, "--apply", "--dest", str(env.sources)) == 0

    assert "refusing" in capsys.readouterr().out
    assert folder_document_ids(folder_id) == []


def test_backfill_force_overrides_the_run_once_guard(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    demo_id = _seed_doc(env.demo_file, "h-demo")
    _run_backfill(monkeypatch, "--apply", "--dest", str(env.sources))
    folder_id = _demo_folder_id()
    assert folder_id is not None
    remove_documents_from_folder(folder_id, [demo_id])
    capsys.readouterr()

    assert _run_backfill(monkeypatch, "--apply", "--force", "--dest", str(env.sources)) == 0

    out = capsys.readouterr().out
    assert "--force" in out and "put back" in out
    assert folder_document_ids(folder_id) == [demo_id]


def test_backfill_reports_a_never_ingested_demo_file_without_assigning_it(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The file is on disk but has no library row — ingest it and the hook does the rest.

    It must NOT count as an existing member (it has no row to be a member with), and a run that
    assigned nothing must NOT burn the run-once flag — otherwise downloading the demo corpus,
    back-filling before ingesting it, and then ingesting would leave the real backfill locked out.
    """
    assert _run_backfill(monkeypatch, "--apply", "--dest", str(env.sources)) == 0

    out = capsys.readouterr().out
    assert "never ingested" in out
    assert "already members      : 0" in out
    assert _demo_folder_id() is None
    assert app_settings.demo_backfill_done() is False

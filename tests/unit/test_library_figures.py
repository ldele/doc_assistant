"""Guard tests for the per-document figure browser (`library.figures`, L1b).

The panel's job is not "list the figures" — it is to show **which figures the assistant can
actually see**. A figure enters retrieval only once it has a VLM description
(`ingest.figure_units` filters on exactly that), so a browser that listed rows without that
distinction would show a page of images and leave the user guessing.

Pure reason-mapping tests plus the view over a temp SQLite; no Chroma, no model, no LLM.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.db.models import Base, Document, Figure
from doc_assistant.db.session import session_scope
from doc_assistant.library.figures import _reason, list_document_figures

DOC_HASH = "figbrowserhash"


@pytest.fixture
def db(tmp_path: Path) -> Iterator[str]:
    """Temp SQLite bound to the global session machinery; yields a seeded document id."""
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    try:
        # Commit before yielding — `list_document_figures` opens its own session, so a row
        # still inside an open transaction is invisible to it.
        with session_scope() as session:
            doc = Document(
                filename="paper.pdf",
                title="A Paper",
                source_original=str(tmp_path / "paper.pdf"),
                doc_hash=DOC_HASH,
                format="pdf",
            )
            session.add(doc)
            session.flush()
            doc_id = str(doc.id)
        yield doc_id
    finally:
        session_mod._engine, session_mod._SessionLocal = orig_engine, orig_factory
        engine.dispose()


def _add(doc_id: str, tmp_path: Path, **kw: object) -> None:
    """Insert one Figure row; `image`=True renders a real file so `has_image` is honest."""
    kw.setdefault("kind", "figure")  # NOT NULL in the schema
    image_path = None
    if kw.pop("image", False):
        png = tmp_path / f"fig{kw.get('page', 1)}.png"
        png.write_bytes(b"\x89PNG")
        image_path = str(png)
    with session_scope() as session:
        session.add(
            Figure(document_id=doc_id, doc_hash=DOC_HASH, image_path=image_path, **kw)  # type: ignore[arg-type]
        )


# ---- _reason: why a figure is not retrievable -------------------------------


def test_a_described_figure_is_retrievable() -> None:
    assert _reason("A bar chart.", None, True) is None


def test_a_described_figure_beats_a_stale_skip_reason() -> None:
    # Order matters: the description is the fact that decides retrievability. A skip reason
    # left over from an earlier pass must not mark a described figure unavailable.
    assert _reason("A bar chart.", "caption_sufficient", True) is None


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_blank_description_does_not_count(blank: str | None) -> None:
    assert _reason(blank, None, True) is not None


def test_caption_only_reads_as_no_image_not_as_undescribed() -> None:
    # A caption-only row can never be described, so "run the description pass" would be a
    # lie — it beats any skip reason recorded against it.
    assert "no image region" in str(_reason(None, "caption_sufficient", False))


def test_known_skip_reasons_are_translated_for_a_user() -> None:
    # The enum is an audit value; shown raw in a UI it reads as a defect rather than the
    # deliberate cost gate it is.
    assert _reason(None, "caption_sufficient", True) == (
        "Caption already describes it — no description needed"
    )
    assert _reason(None, "budget_exhausted", True) == "Per-document description budget reached"


def test_an_unknown_skip_reason_is_passed_through_not_swallowed() -> None:
    # A new enum value must still reach the user; mapping it to None would silently claim
    # the figure is retrievable.
    assert _reason(None, "some_new_reason", True) == "some_new_reason"


def test_a_failed_call_is_summarised_not_dumped_into_the_panel() -> None:
    # The recorded reason is an audit string of up to 400 chars (a pydantic message);
    # the panel gets the actionable sentence, the DB keeps the diagnosis.
    recorded = (
        "error: ValidationError: 1 validation error for FigureDescription key_quantities "
        "Input should be a valid list [type=list_type, input_value='100 mV, 40 m.mho/cm2']"
    )
    assert _reason(None, recorded, True) == (
        "Description attempt failed — it is retried by the next description pass"
    )


def test_not_yet_described_says_so() -> None:
    assert "Not described yet" in str(_reason(None, None, True))


# ---- list_document_figures --------------------------------------------------


def test_unknown_document_is_none_not_empty(db: str) -> None:
    # None is a 404; an empty list is an ordinary text-only document. Collapsing them would
    # make a deleted document look like one with no figures.
    assert list_document_figures("no-such-id") is None


def test_known_document_with_no_figures_is_empty_not_none(db: str) -> None:
    view = list_document_figures(db)
    assert view is not None
    assert view.figures == []
    assert view.total == 0


def test_figures_come_back_in_page_order(db: str, tmp_path: Path) -> None:
    _add(db, tmp_path, page=7, kind="chart", image=True)
    _add(db, tmp_path, page=2, kind="photo", image=True)
    view = list_document_figures(db)
    assert view is not None
    assert [f.page for f in view.figures] == [2, 7]


def test_counts_describe_the_state(db: str, tmp_path: Path) -> None:
    _add(
        db,
        tmp_path,
        page=1,
        caption="Figure 1. Described.",
        vlm_description="A chart.",
        image=True,
    )
    _add(db, tmp_path, page=2, caption="Figure 2. Not described.", image=True)
    _add(db, tmp_path, page=3, extraction_method="caption_only", caption="Figure 3.")
    view = list_document_figures(db)
    assert view is not None
    assert (view.total, view.retrievable_count, view.captioned_count) == (3, 1, 3)
    # The caption-only row never had an image, so it is not a broken one.
    assert view.missing_image_count == 0


def test_a_vanished_png_counts_as_missing(db: str, tmp_path: Path) -> None:
    # The figure dir being cleared (or the doc re-hashing) leaves a row pointing at nothing.
    # That is a real broken state and must be visible, unlike a caption-only row.
    _add(db, tmp_path, page=1, extraction_method="image_block", image=True)
    png = next(tmp_path.glob("fig1.png"))
    png.unlink()
    view = list_document_figures(db)
    assert view is not None
    assert view.figures[0].has_image is False
    assert view.missing_image_count == 1


def test_the_header_carries_the_document_identity(db: str, tmp_path: Path) -> None:
    view = list_document_figures(db)
    assert view is not None
    assert (view.filename, view.title) == ("paper.pdf", "A Paper")

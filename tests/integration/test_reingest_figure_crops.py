"""Figure crops: restoring the ones that went missing, and not destroying what survived (KI-50).

Two failures meet here, and they are opposites.

**KI-50** is the one that already happened: 723 of 811 cropped PNGs were gone from the reference
library while every row and every paid VLM description survived. The crop is the only part of a
figure that lives outside the database, so it is the only part that can go missing on its own —
and getting it back must not put the rest at risk. Hence `crops`: re-render the *recorded*
rectangle, touch no row.

**KI-55** is the one that was about to happen: `figures` rebuilt a document's rows from scratch and
wrote `vlm_description=None` into every one of them, so the cheapest-looking box in the re-run
dialog silently discarded the only expensive thing in the table — and, because retrieval admits a
figure on its description rather than its image, dropped those figures out of search too.

The guard that matters is not "descriptions are kept" on its own. It is that a description is kept
**only when the region is recognisably the same one**: a description pointing at a different
picture is worse than no description, which is the same rule the chunk locator lives by.

Deterministic and offline: a two-page PDF built in-test, a temp SQLite, a temp figure dir.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import config, reingest
from doc_assistant.db.models import Base, Document, Figure
from doc_assistant.db.session import session_scope
from doc_assistant.ingest import figures as figures_mod

DOC_HASH = "crophash0001"
DESCRIPTION = "A red square, rendered as a test fixture."


def _build_pdf(path: Path) -> None:
    """Two pages, each with one inserted raster image and a caption under it."""
    import pymupdf

    doc = pymupdf.open()
    for n in (1, 2):
        page = doc.new_page(width=300, height=400)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 80, 80))
        pix.set_rect(pix.irect, (200, 30 * n, 30))
        page.insert_image(pymupdf.Rect(40, 220, 260, 340), stream=pix.tobytes("png"))
        page.insert_text((40, 360), f"Figure {n}: a test figure caption.", fontsize=10)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )

    fig_root = tmp_path / "figures"
    monkeypatch.setattr(figures_mod, "FIGURE_DIR", fig_root)
    monkeypatch.setattr(config, "DOCS_PATH", tmp_path / "sources")

    pdf = tmp_path / "paper.pdf"
    _build_pdf(pdf)
    try:
        yield SimpleNamespace(tmp=tmp_path, pdf=pdf, fig_root=fig_root)
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _seed(env: SimpleNamespace, *, described: bool = True) -> str:
    """One document with real detected figures, cropped to disk, optionally described."""
    with session_scope() as session:
        doc = Document(
            filename="paper.pdf",
            source_original=str(env.pdf),
            doc_hash=DOC_HASH,
            format="pdf",
        )
        session.add(doc)
        session.flush()
        doc_id = str(doc.id)

    # Build the rows the way ingestion does, so the fixture is the real shape rather than a guess.
    regions = figures_mod.detect_figure_regions(str(env.pdf))
    assert regions, "the fixture PDF produced no figure regions"
    import pymupdf

    handle = pymupdf.open(str(env.pdf))
    try:
        with session_scope() as session:
            per_page: dict[int, int] = {}
            for region in regions:
                index = per_page.get(region.page, 0)
                per_page[region.page] = index + 1
                image_path = None
                if region.bbox is not None:
                    out = figures_mod.figure_image_path(DOC_HASH, region.page, index)
                    figures_mod.render_region(handle[region.page - 1], region.bbox, out, dpi=120)
                    image_path = str(out)
                bbox = region.bbox
                session.add(
                    Figure(
                        document_id=doc_id,
                        doc_hash=DOC_HASH,
                        page=region.page,
                        bbox_x0=bbox[0] if bbox else None,
                        bbox_y0=bbox[1] if bbox else None,
                        bbox_x1=bbox[2] if bbox else None,
                        bbox_y1=bbox[3] if bbox else None,
                        kind=region.kind,
                        caption=region.caption,
                        image_path=image_path,
                        extraction_method=region.extraction_method,
                        vlm_description=DESCRIPTION if described else None,
                    )
                )
    finally:
        handle.close()
    return doc_id


def _rows(document_id: str) -> list[Figure]:
    with session_scope() as session:
        return list(
            session.execute(
                select(Figure).where(Figure.document_id == document_id).order_by(Figure.page)
            ).scalars()
        )


def _crops(document_id: str) -> list[Path]:
    return [Path(str(r.image_path)) for r in _rows(document_id) if r.image_path]


# --- restoring a crop ------------------------------------------------------------------------ #


def test_a_missing_crop_is_restored_from_the_region_already_recorded(
    env: SimpleNamespace,
) -> None:
    doc_id = _seed(env)
    paths = _crops(doc_id)
    assert paths and all(p.exists() for p in paths)
    for p in paths:
        p.unlink()

    result = reingest.rerun([doc_id], ["crops"])

    assert result.outcomes[0].status == "ok", result.outcomes[0].detail
    assert f"{len(paths)} image(s) restored" in result.outcomes[0].detail
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_restoring_a_crop_changes_no_row_at_all(env: SimpleNamespace) -> None:
    """The whole point of the cheap part: it puts a file back and touches nothing else."""
    doc_id = _seed(env)
    before = [
        (r.id, r.page, r.caption, r.image_path, r.vlm_description, r.bbox_x0)
        for r in _rows(doc_id)
    ]
    for p in _crops(doc_id):
        p.unlink()

    reingest.rerun([doc_id], ["crops"])

    after = [
        (r.id, r.page, r.caption, r.image_path, r.vlm_description, r.bbox_x0)
        for r in _rows(doc_id)
    ]
    assert after == before
    assert all(r.vlm_description == DESCRIPTION for r in _rows(doc_id))


def test_restoring_twice_leaves_the_first_file_alone(env: SimpleNamespace) -> None:
    """A repair pass has to be safe to run over a whole library twice."""
    doc_id = _seed(env)
    for p in _crops(doc_id):
        p.unlink()
    reingest.rerun([doc_id], ["crops"])
    first = {p: p.read_bytes() for p in _crops(doc_id)}

    result = reingest.rerun([doc_id], ["crops"])

    assert result.outcomes[0].status == "skipped"
    assert "already on disk" in result.outcomes[0].detail
    assert {p: p.read_bytes() for p in _crops(doc_id)} == first


def test_a_region_the_pdf_no_longer_has_is_an_error_not_a_crash(env: SimpleNamespace) -> None:
    """A row pointing past the end of the file is a broken row; report it, do not raise."""
    doc_id = _seed(env)
    for p in _crops(doc_id):
        p.unlink()
    with session_scope() as session:
        row = session.execute(select(Figure).where(Figure.document_id == doc_id)).scalars().first()
        assert row is not None
        row.page = 99

    result = reingest.rerun([doc_id], ["crops"])

    assert result.outcomes[0].status == "error"
    assert "page 99" in result.outcomes[0].detail
    # The *other* page still got its crop back — one broken row must not cost the rest.
    assert any(p.exists() for p in _crops(doc_id))


def test_a_document_with_no_figures_is_skipped_with_a_reason(env: SimpleNamespace) -> None:
    with session_scope() as session:
        doc = Document(
            filename="empty.pdf",
            source_original=str(env.pdf),
            doc_hash="nofigures001",
            format="pdf",
        )
        session.add(doc)
        session.flush()
        doc_id = str(doc.id)

    result = reingest.rerun([doc_id], ["crops"])
    assert result.outcomes[0].status == "skipped"
    assert "no figure regions" in result.outcomes[0].detail


# --- not destroying what survived (KI-55) ----------------------------------------------------- #


def test_re_running_figures_keeps_the_descriptions(env: SimpleNamespace) -> None:
    """The guard for KI-55. Without the carry-over every description here comes back None."""
    doc_id = _seed(env)
    assert all(r.vlm_description == DESCRIPTION for r in _rows(doc_id))

    result = reingest.rerun([doc_id], ["figures"])

    assert result.outcomes[0].status == "ok"
    rows = _rows(doc_id)
    assert rows, "re-detection produced no rows"
    assert all(r.vlm_description == DESCRIPTION for r in rows), result.outcomes[0].detail
    assert "description(s) kept" in result.outcomes[0].detail


def test_a_description_is_not_carried_onto_a_region_that_moved(env: SimpleNamespace) -> None:
    """The other half of the guard: a rectangle that no longer matches gets nothing, and the run
    says so. A description on the wrong picture is worse than no description."""
    doc_id = _seed(env)
    with session_scope() as session:
        for row in session.execute(select(Figure).where(Figure.document_id == doc_id)).scalars():
            row.bbox_x0 = float(row.bbox_x0 or 0) + 50.0  # the recorded region is now elsewhere

    result = reingest.rerun([doc_id], ["figures"])

    assert result.outcomes[0].status == "ok"
    assert all(r.vlm_description is None for r in _rows(doc_id))
    assert "dropped (their regions changed)" in result.outcomes[0].detail


def test_the_identity_key_ignores_sub_point_bbox_noise() -> None:
    """A float round-trip through SQLite must not read as "a different figure"."""
    a = reingest._figure_identity(3, (10.0, 20.0, 30.0, 40.0), "Figure 1")
    b = reingest._figure_identity(3, (10.2, 19.8, 30.1, 40.0), "Figure 1")
    assert a == b
    assert a != reingest._figure_identity(4, (10.0, 20.0, 30.0, 40.0), "Figure 1")


def test_two_caption_only_rows_on_one_page_stay_distinct() -> None:
    """With no rectangle to compare, the caption is the only thing telling them apart."""
    first = reingest._figure_identity(2, None, "Figure 1: the first")
    second = reingest._figure_identity(2, None, "Figure 2: the second")
    assert first != second

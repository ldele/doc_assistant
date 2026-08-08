"""Integration test for Feature 4b figure extraction (the CLI round-trip).

Builds a tiny one-page PDF in-test (an inserted raster image + a
``Figure 1:`` caption), then drives ``scripts.extract_figures`` against a fresh
temp SQLite and a temp ``FIGURE_DIR``. Asserts detection + caption pairing,
that ``--apply`` writes one PNG and one ``Figure`` row, idempotency (a second
run without ``--force`` is a no-op; ``--force`` re-renders), and the
Enrichment-Layer invariant — no chunk-store / ``Document`` mutation.

Deterministic and offline: no corpus, no network, no LLM.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import extract_figures
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.db.models import Base, Document, Figure
from doc_assistant.db.session import session_scope
from doc_assistant.ingest import figures
from doc_assistant.ingest.figures import detect_figure_regions

CAPTION = "Figure 1: a test figure caption."
DOC_HASH = "testhash0001"


def _build_fixture_pdf(path: Path) -> None:
    """A 300x400 page: one inserted raster image (a figure) + a caption line."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 80, 80))
    pix.set_rect(pix.irect, (200, 30, 30))
    page.insert_image(pymupdf.Rect(40, 220, 260, 340), stream=pix.tobytes("png"))
    page.insert_text((40, 360), CAPTION, fontsize=10)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Temp SQLite bound to the global session machinery + temp FIGURE_DIR + fixture PDF."""
    db_path = tmp_path / "library.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    # expire_on_commit=False so the helper queries can return detached Figure rows
    # whose column attributes stay readable after the session closes.
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )

    fig_root = tmp_path / "figures"
    monkeypatch.setattr(figures, "FIGURE_DIR", fig_root)

    pdf = tmp_path / "paper.pdf"
    _build_fixture_pdf(pdf)

    try:
        yield SimpleNamespace(tmp=tmp_path, pdf=pdf, fig_root=fig_root)
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _seed_document(pdf_path: Path) -> str:
    with session_scope() as session:
        doc = Document(
            filename="paper.pdf",
            source_original=str(pdf_path),
            doc_hash=DOC_HASH,
            format="pdf",
        )
        session.add(doc)
        session.flush()
        return str(doc.id)


def _get_figures(document_id: str) -> list[Figure]:
    with session_scope() as session:
        return list(
            session.execute(select(Figure).where(Figure.document_id == document_id))
            .scalars()
            .all()
        )


def _run(document_id: str, pdf: Path, *, apply: bool, force: bool, dpi: int = 120) -> dict:
    return extract_figures._run_one(
        document_id,
        DOC_HASH,
        "paper.pdf",
        str(pdf),
        "pdf",
        apply=apply,
        force=force,
        dpi=dpi,
    )


# ============================================================
# Detection + pairing
# ============================================================


def test_detect_finds_one_region_with_caption(env: SimpleNamespace) -> None:
    regions = detect_figure_regions(str(env.pdf))
    assert len(regions) == 1
    region = regions[0]
    assert region.page == 1
    assert region.kind == "photo"
    assert region.bbox is not None
    assert region.extraction_method == "image_block"
    assert region.caption == CAPTION


# ============================================================
# --apply: rows + PNGs
# ============================================================


def test_apply_writes_one_png_and_one_row(env: SimpleNamespace) -> None:
    doc_id = _seed_document(env.pdf)
    row = _run(doc_id, env.pdf, apply=True, force=False)

    assert row["status"] == "ok"
    assert row["figures"] == 1
    assert row["rendered"] == 1

    figs = _get_figures(doc_id)
    assert len(figs) == 1
    fig = figs[0]
    assert fig.page == 1
    assert fig.kind == "photo"
    assert fig.caption == CAPTION
    assert fig.extraction_method == "image_block"
    assert fig.vlm_description is None
    assert fig.vlm_call_skipped_reason is None

    assert fig.image_path is not None
    png = Path(fig.image_path)
    assert png.exists()
    assert png.parent == env.fig_root / DOC_HASH


# ============================================================
# Idempotency
# ============================================================


def test_second_run_without_force_is_noop(env: SimpleNamespace) -> None:
    doc_id = _seed_document(env.pdf)
    _run(doc_id, env.pdf, apply=True, force=False)
    png = Path(_get_figures(doc_id)[0].image_path)
    mtime = png.stat().st_mtime_ns

    row = _run(doc_id, env.pdf, apply=True, force=False)
    assert row["status"] == "skipped"
    assert len(_get_figures(doc_id)) == 1  # no duplicate rows
    assert png.stat().st_mtime_ns == mtime  # PNG untouched


def test_force_reextracts_without_duplicating(env: SimpleNamespace) -> None:
    doc_id = _seed_document(env.pdf)
    _run(doc_id, env.pdf, apply=True, force=False)

    row = _run(doc_id, env.pdf, apply=True, force=True)
    assert row["status"] == "ok"
    assert row["rendered"] == 1
    assert len(_get_figures(doc_id)) == 1  # replaced, not appended


# ============================================================
# Enrichment-Layer invariant
# ============================================================


def test_no_document_or_chunk_store_mutation(env: SimpleNamespace) -> None:
    doc_id = _seed_document(env.pdf)

    def _doc_snapshot() -> tuple:
        with session_scope() as session:
            d = session.get(Document, doc_id)
            assert d is not None
            return (d.chunk_count, d.extraction_health, d.updated_at)

    before = _doc_snapshot()
    _run(doc_id, env.pdf, apply=True, force=False)
    after = _doc_snapshot()

    assert before == after  # the figure sidecar never writes Document columns
    with session_scope() as session:
        doc_count = session.execute(select(func.count()).select_from(Document)).scalar_one()
    assert doc_count == 1
    # No vector store is created by the figure pass (it never imports chromadb).
    assert not (env.tmp / "chroma").exists()
    assert not (env.tmp / "chroma_pc").exists()


def test_load_figure_image_paths_returns_only_existing_files(env: SimpleNamespace) -> None:
    """The UI lookup yields a path only for figures whose PNG is actually on disk."""
    doc_id = _seed_document(env.pdf)
    real_png = env.tmp / "real.png"
    real_png.write_bytes(b"\x89PNG\r\n")  # bytes are irrelevant — only existence is checked
    with session_scope() as session:
        present = Figure(
            document_id=doc_id, doc_hash=DOC_HASH, page=1, kind="figure", image_path=str(real_png)
        )
        gone = Figure(
            document_id=doc_id,
            doc_hash=DOC_HASH,
            page=2,
            kind="figure",
            image_path=str(env.tmp / "gone.png"),
        )
        caption_only = Figure(
            document_id=doc_id, doc_hash=DOC_HASH, page=3, kind="figure", image_path=None
        )
        session.add_all([present, gone, caption_only])
        session.flush()
        pid, gid, cid = str(present.id), str(gone.id), str(caption_only.id)

    paths = figures.load_figure_image_paths([pid, gid, cid])
    assert paths == {pid: str(real_png)}  # missing-file and caption-only excluded
    assert figures.load_figure_image_paths([]) == {}  # empty input → no DB hit


# ============================================================
# Page-scan ceiling + `--force` must be able to CLEAR (2026-08-08)
# ============================================================
# A scanned page is one full-page image, and with only an area floor every page of a
# scan became a "figure": `hebb_1949.pdf` produced 365 for 365 pages. Adding the ceiling
# exposed a second bug — `--force` was guarded by `and regions`, so a document that
# legitimately drops to ZERO kept its stale rows forever, which is precisely what left
# those 365 rows in place after the ceiling had correctly rejected all of them.


def _build_page_scan_pdf(path: Path) -> None:
    """A page that IS one image, with no caption — i.e. a scanned page."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 400))
    pix.set_rect(pix.irect, (128, 128, 128))
    page.insert_image(pymupdf.Rect(0, 0, 300, 400), stream=pix.tobytes("png"))
    doc.save(str(path))
    doc.close()


def test_page_scan_yields_no_figures(env: SimpleNamespace, tmp_path: Path) -> None:
    scan = tmp_path / "scan.pdf"
    _build_page_scan_pdf(scan)
    assert detect_figure_regions(str(scan)) == []


def test_force_clears_rows_when_detection_drops_to_zero(
    env: SimpleNamespace, tmp_path: Path
) -> None:
    doc_id = _seed_document(env.pdf)
    # 1. a real figure is detected and persisted
    row = _run(doc_id, env.pdf, apply=True, force=False)
    assert row["figures"] == 1
    assert len(_get_figures(doc_id)) == 1

    # 2. the same document now yields nothing (here: it is a page scan). Before the fix
    #    the runner reported "no-figures" while the old row silently survived.
    scan = tmp_path / "scan.pdf"
    _build_page_scan_pdf(scan)
    cleared = _run(doc_id, scan, apply=True, force=True)
    assert cleared["status"] == "no-figures"
    assert cleared["note"] == "cleared (0 figures)"
    assert _get_figures(doc_id) == [], "stale figure rows survived a --force re-run"


def test_without_force_a_zero_detection_leaves_rows_alone(
    env: SimpleNamespace, tmp_path: Path
) -> None:
    # The clear is `--force`-only: a plain re-run must stay idempotent and non-destructive.
    doc_id = _seed_document(env.pdf)
    _run(doc_id, env.pdf, apply=True, force=False)
    scan = tmp_path / "scan.pdf"
    _build_page_scan_pdf(scan)
    _run(doc_id, scan, apply=True, force=False)
    assert len(_get_figures(doc_id)) == 1

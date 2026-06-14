"""Integration test for Feature 4c — VLM figure description + chunk emission.

Drives ``scripts.describe_figures`` against a fresh temp SQLite with seeded 4b
``Figure`` rows and a **fake** describer (zero API cost), asserting the gating
(thin caption → describe, long caption → skip, no image → skip, budget → skip),
idempotency, and that ``ingest.figure_units`` materialises a described figure
into a ``chunk_type='figure'`` retrieval unit.

Deterministic and offline: no network, no real VLM, no Anthropic key.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from scripts import describe_figures
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import ingest
from doc_assistant.db.models import Base, Document, Figure
from doc_assistant.db.session import session_scope
from doc_assistant.figures import FigureDescription

DOC_HASH = "vlmhash0001"


class _FakeDescriber:
    """Counts calls; returns a canned description echoing the caption."""

    def __init__(self) -> None:
        self.calls = 0

    def describe(
        self, *, image_b64: str, media_type: str, caption: str, model: str, max_tokens: int
    ) -> FigureDescription:
        self.calls += 1
        return FigureDescription(
            figure_type="bar chart",
            summary=f"A figure described from caption {caption!r}.",
            key_quantities=["42%"],
            axes="x: model, y: accuracy",
            trend="increasing",
        )


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Path]:
    """Temp SQLite bound to the global session machinery. Yields tmp_path."""
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    try:
        yield tmp_path
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _seed(tmp_path: Path) -> str:
    """Seed one Document + three Figure rows; returns the document_id."""
    png = tmp_path / "fig.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n-fake-")  # bytes are read but the fake ignores them
    with session_scope() as session:
        doc = Document(
            filename="paper.pdf", source_original="paper.pdf", doc_hash=DOC_HASH, format="pdf"
        )
        session.add(doc)
        session.flush()
        doc_id = str(doc.id)
        session.add_all(
            [
                # eligible: has image, thin caption
                Figure(
                    document_id=doc_id,
                    doc_hash=DOC_HASH,
                    page=1,
                    kind="photo",
                    caption="Figure 1.",
                    image_path=str(png),
                    extraction_method="image_block",
                ),
                # skip: caption already long
                Figure(
                    document_id=doc_id,
                    doc_hash=DOC_HASH,
                    page=2,
                    kind="chart",
                    caption="x" * 400,
                    image_path=str(png),
                    extraction_method="drawing_union",
                ),
                # skip: caption-only (no image)
                Figure(
                    document_id=doc_id,
                    doc_hash=DOC_HASH,
                    page=3,
                    kind="figure",
                    caption="Figure 3.",
                    image_path=None,
                    extraction_method="caption_only",
                ),
            ]
        )
    return doc_id


def _figs_by_page(doc_id: str) -> dict[int, Figure]:
    with session_scope() as session:
        rows = session.execute(select(Figure).where(Figure.document_id == doc_id)).scalars().all()
    return {f.page: f for f in rows}


def _run(doc_id: str, fake: _FakeDescriber, *, force: bool = False, max_calls: int = 30) -> dict:
    return describe_figures._describe_doc(
        doc_id,
        "paper.pdf",
        apply=True,
        force=force,
        describer=fake,
        model="fake-model",
        max_calls=max_calls,
    )


def test_gating_describe_skip_skip(env: Path) -> None:
    doc_id = _seed(env)
    fake = _FakeDescriber()
    row = _run(doc_id, fake)

    assert fake.calls == 1  # only the thin-caption figure
    assert row["described"] == 1
    assert row["skipped"] == 2

    figs = _figs_by_page(doc_id)
    assert figs[1].vlm_description is not None
    assert "described from caption" in figs[1].vlm_description
    assert figs[1].vlm_call_skipped_reason is None
    assert figs[2].vlm_call_skipped_reason == "caption_sufficient"
    assert figs[2].vlm_description is None
    assert figs[3].vlm_call_skipped_reason == "no_image"


def test_idempotent_second_run_makes_no_calls(env: Path) -> None:
    doc_id = _seed(env)
    fake = _FakeDescriber()
    _run(doc_id, fake)
    assert fake.calls == 1
    # Second run without --force: the described figure is left alone.
    _run(doc_id, fake)
    assert fake.calls == 1


def test_force_redescribes(env: Path) -> None:
    doc_id = _seed(env)
    fake = _FakeDescriber()
    _run(doc_id, fake)
    _run(doc_id, fake, force=True)
    assert fake.calls == 2


def test_budget_zero_blocks_all_calls(env: Path) -> None:
    doc_id = _seed(env)
    fake = _FakeDescriber()
    row = _run(doc_id, fake, max_calls=0)
    assert fake.calls == 0
    figs = _figs_by_page(doc_id)
    assert figs[1].vlm_call_skipped_reason == "budget_exhausted"
    assert row["described"] == 0


def test_image_missing_is_recorded_not_fatal(env: Path) -> None:
    doc_id = _seed(env)
    # Point the eligible figure at a non-existent PNG.
    with session_scope() as session:
        fig = session.execute(
            select(Figure).where(Figure.document_id == doc_id, Figure.page == 1)
        ).scalar_one()
        fig.image_path = str(env / "missing.png")
    fake = _FakeDescriber()
    _run(doc_id, fake)
    assert fake.calls == 0
    assert _figs_by_page(doc_id)[1].vlm_call_skipped_reason == "image_missing"


def test_figure_units_materialises_described_figure(env: Path) -> None:
    doc_id = _seed(env)
    _run(doc_id, _FakeDescriber())

    units = ingest.figure_units(doc_id)
    assert len(units) == 1  # only the described figure becomes a chunk
    text, page, figure_id = units[0]
    assert page == 1
    assert "Figure 1." in text  # caption preserved
    assert "described from caption" in text  # VLM description appended
    assert figure_id  # links back to the Figure row

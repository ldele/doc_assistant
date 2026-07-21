"""E0.4 / GP-7 — pin the 0-document honest-degradation contract with a gate, not habit.

The `.claude/CONTEXT.md` robustness contract requires every module to degrade honestly at **0
documents** (never a crash). Read paths already do; two *build* paths did not, and nothing pinned
the contract — so it survived by habit. This is the missing empty-input guard: one parametrized
test over the four corpus-derived build entry points (wiki / epistemics / skeleton / gaps),
asserting each returns an honest empty result rather than raising.

Non-vacuous: today ``wiki``/``epistemics`` raise ``OperationalError`` on a never-migrated DB (the
two the E0.4 fix repairs); ``skeleton``/``gaps`` already degrade honestly on an empty-but-migrated
DB and this locks that behaviour in.

Two DB worlds, by necessity:
* **wiki / epistemics** run against a **never-migrated** DB (no tables at all) — the harshest 0-doc
  state, and exactly where they used to throw. Their only DB touch is a read (`load_doc_graph`) /
  the sidecar write (`_write_rows`), both now guarded.
* **skeleton / gaps** run against a **migrated-but-empty** DB (0 rows). They read the curated
  vocabulary / claim tables directly with no loader seam, so those tables must exist; the honest
  path is "0 concepts / 0 claims -> empty output", which this pins.

Deterministic + offline: no network, no real LLM, no Chroma.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.db.models import Base
from doc_assistant.knowledge import concept_skeleton as cs
from doc_assistant.knowledge import epistemics, gaps, wiki


def _bind_engine(engine: object) -> tuple[object, object]:
    orig = (session_mod._engine, session_mod._SessionLocal)
    session_mod._engine = engine  # type: ignore[assignment]
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    return orig


def _write_empty_skeleton(root: Path) -> None:
    """A structurally-valid skeleton.json with zero nodes/edges (the empty-corpus artifact)."""
    root.mkdir(parents=True, exist_ok=True)
    skeleton = cs.analyze_skeleton([], [], seed=42)
    (root / cs.SKELETON_NAME).write_text(
        json.dumps(cs.skeleton_to_dict(skeleton)), encoding="utf-8"
    )


@pytest.fixture
def empty_world(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """A temp SQLite bound to the global session machinery; migrated only if the case needs it."""
    migrate: bool = request.param
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    if migrate:
        Base.metadata.create_all(engine)
    orig_engine, orig_factory = _bind_engine(engine)
    monkeypatch.setattr("doc_assistant.config.CONCEPT_SKELETON_DIR", tmp_path / "skeleton")
    try:
        yield tmp_path
    finally:
        session_mod._engine = orig_engine  # type: ignore[assignment]
        session_mod._SessionLocal = orig_factory  # type: ignore[assignment]
        engine.dispose()


@pytest.mark.parametrize(
    ("empty_world", "case"),
    [
        pytest.param(False, "wiki", id="wiki-never-migrated"),
        pytest.param(False, "epistemics", id="epistemics-never-migrated"),
        pytest.param(True, "skeleton", id="skeleton-empty-migrated"),
        pytest.param(True, "gaps", id="gaps-empty-migrated"),
    ],
    indirect=["empty_world"],
)
def test_empty_input_degrades_honestly(
    empty_world: Path, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    skel_dir = empty_world / "skeleton"

    if case == "wiki":
        # Never-migrated DB: the read used to raise OperationalError; now it is an honest empty.
        assert wiki.load_doc_graph() == ([], [])

    elif case == "epistemics":
        # Never-migrated DB + an empty skeleton: the projection is empty and the sidecar write is
        # guarded, so build_epistemics returns an honest empty result instead of raising.
        _write_empty_skeleton(skel_dir)
        monkeypatch.setattr(epistemics, "load_doc_chunks", lambda: [])
        monkeypatch.setattr(epistemics, "load_pc_parent_chunks", lambda: [])  # E1.1 segmentation
        result = epistemics.build_epistemics(apply=True, skeleton_dir=skel_dir)
        assert result.rows == []
        assert result.n_nodes == 0

    elif case == "skeleton":
        # Migrated-but-empty DB (0 concepts, 0 chunks): an honest empty skeleton, no raise.
        monkeypatch.setattr(cs, "load_presence_inputs", lambda document_ids=None: [])
        result = cs.build_concept_skeleton(apply=True)
        assert result.n_concepts == 0
        assert result.n_edges == 0

    elif case == "gaps":
        # Migrated-but-empty DB + an empty skeleton: no concepts, no claims -> zero gaps, no raise.
        _write_empty_skeleton(skel_dir)
        result = gaps.build_gaps(apply=True, skeleton_dir=skel_dir, min_degree=1)
        assert result.gaps == []
    else:  # pragma: no cover - guards against a mistyped param
        raise AssertionError(f"unknown case {case!r}")

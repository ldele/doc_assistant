"""Guard for the additive-column migration (Chunk 2c added a column to a
pre-existing table — ``create_all`` does not do that).
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, inspect, text

from doc_assistant.db.migrations import _apply_additive_columns


def test_additive_column_added_to_preexisting_table(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'm.db'}", future=True)
    try:
        # Simulate the OLD schema: answer_reviews without failure_tag.
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE answer_reviews "
                    "(id VARCHAR PRIMARY KEY, answer_record_id VARCHAR)"
                )
            )
        assert "failure_tag" not in {
            c["name"] for c in inspect(engine).get_columns("answer_reviews")
        }

        _apply_additive_columns(engine)
        cols = {c["name"] for c in inspect(engine).get_columns("answer_reviews")}
        assert "failure_tag" in cols

        # Idempotent: a second pass is a no-op (no duplicate-column error).
        _apply_additive_columns(engine)
    finally:
        engine.dispose()


def test_r4_strength_json_added_to_preexisting_concept_edges(tmp_path: Path) -> None:
    # R4: an existing concept_edges (Node-A, pre-strength) gains strength_json in place.
    engine = create_engine(f"sqlite:///{tmp_path / 'edges.db'}", future=True)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE concept_edges "
                    "(id VARCHAR PRIMARY KEY, provenance_json TEXT, weight FLOAT)"
                )
            )
        assert "strength_json" not in {
            c["name"] for c in inspect(engine).get_columns("concept_edges")
        }

        _apply_additive_columns(engine)
        assert "strength_json" in {c["name"] for c in inspect(engine).get_columns("concept_edges")}

        _apply_additive_columns(engine)  # idempotent second pass
    finally:
        engine.dispose()


def test_alias_breadth_is_added_unset_and_leaves_labels_alone(tmp_path: Path) -> None:
    """ADR-054: a library that predates exact/broad gains `concept_aliases.breadth` in place.
    Every existing alias reads NULL — unclassified, which counts as exact, as it did before the
    column existed — and no label or alias text is touched (ADR-043)."""
    engine = create_engine(f"sqlite:///{tmp_path / 'aliases.db'}", future=True)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE concepts (id VARCHAR PRIMARY KEY, label VARCHAR)"))
            conn.execute(
                text(
                    "CREATE TABLE concept_aliases "
                    "(id VARCHAR PRIMARY KEY, concept_id VARCHAR, alias VARCHAR)"
                )
            )
            conn.execute(text("INSERT INTO concepts VALUES ('c1', 'knowledge distillation')"))
            conn.execute(text("INSERT INTO concept_aliases VALUES ('a1', 'c1', 'distillation')"))
            conn.execute(text("INSERT INTO concept_aliases VALUES ('a2', 'c1', 'dIN')"))

        added = _apply_additive_columns(engine)
        assert "concept_aliases.breadth" in added

        with engine.connect() as conn:
            aliases = conn.execute(
                text("SELECT id, alias, breadth FROM concept_aliases ORDER BY id")
            ).all()
            labels = conn.execute(text("SELECT id, label FROM concepts")).all()
        assert [tuple(r) for r in aliases] == [("a1", "distillation", None), ("a2", "dIN", None)]
        assert [tuple(r) for r in labels] == [("c1", "knowledge distillation")]

        assert "concept_aliases.breadth" not in _apply_additive_columns(engine)  # idempotent
    finally:
        engine.dispose()


def test_additive_migration_skips_absent_table(tmp_path: Path) -> None:
    # No answer_reviews table at all → migration is a clean no-op.
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}", future=True)
    try:
        _apply_additive_columns(engine)  # must not raise
    finally:
        engine.dispose()


def test_keyword_link_gains_score_and_rank_unset_and_keeps_its_links(tmp_path: Path) -> None:
    """Session 5c: a library that predates stored scores gains `document_keywords.score` and
    `.rank` in place. Every existing link survives and reads NULL — unranked, listed after ranked
    links by name — because the scores were never stored and there is nothing true to backfill."""
    engine = create_engine(f"sqlite:///{tmp_path / 'links.db'}", future=True)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE document_keywords "
                    "(document_id VARCHAR, keyword_id VARCHAR, "
                    "PRIMARY KEY (document_id, keyword_id))"
                )
            )
            conn.execute(text("INSERT INTO document_keywords VALUES ('d1', 'k1'), ('d1', 'k2')"))

        added = _apply_additive_columns(engine)
        assert {"document_keywords.score", "document_keywords.rank"} <= set(added)
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT keyword_id, score, rank FROM document_keywords ORDER BY keyword_id")
            ).all()
        assert [tuple(r) for r in rows] == [("k1", None, None), ("k2", None, None)]

        assert _apply_additive_columns(engine) == []  # idempotent second pass
    finally:
        engine.dispose()

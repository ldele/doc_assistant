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
        assert "strength_json" in {
            c["name"] for c in inspect(engine).get_columns("concept_edges")
        }

        _apply_additive_columns(engine)  # idempotent second pass
    finally:
        engine.dispose()


def test_additive_migration_skips_absent_table(tmp_path: Path) -> None:
    # No answer_reviews table at all → migration is a clean no-op.
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}", future=True)
    try:
        _apply_additive_columns(engine)  # must not raise
    finally:
        engine.dispose()

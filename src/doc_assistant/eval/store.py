"""DuckDB persistence for eval runs (generic).

Schema (locked, idempotent CREATE IF NOT EXISTS):

* ``runs`` — one row per ``Runner.run`` invocation
* ``case_results`` — one row per case per run
* ``scores`` — one row per scorer per case per run

DuckDB gives us OLAP-friendly aggregates over many runs without
needing a separate analytics layer. Storage cost is trivial at
personal-eval scale.

Connection model: one ``Store`` instance owns one connection. Open at
``__init__``, close via ``close()`` or context manager. The store is
not thread-safe — runs are sequential.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

import duckdb

from doc_assistant.eval.results import EvalResult


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS runs (
        id            VARCHAR PRIMARY KEY,
        started_at    TIMESTAMP NOT NULL,
        finished_at   TIMESTAMP,
        system_name   VARCHAR NOT NULL,
        config_json   VARCHAR,
        n_cases       INTEGER NOT NULL,
        note          VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS case_results (
        run_id        VARCHAR NOT NULL,
        case_id       VARCHAR NOT NULL,
        query         VARCHAR NOT NULL,
        answer        VARCHAR,
        citations_json VARCHAR,
        latency_ms    DOUBLE NOT NULL,
        token_input   INTEGER,
        token_output  INTEGER,
        error         VARCHAR,
        timestamp     TIMESTAMP NOT NULL,
        PRIMARY KEY (run_id, case_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scores (
        run_id        VARCHAR NOT NULL,
        case_id       VARCHAR NOT NULL,
        scorer_name   VARCHAR NOT NULL,
        value         DOUBLE NOT NULL,
        details_json  VARCHAR,
        scoreable     BOOLEAN,
        PRIMARY KEY (run_id, case_id, scorer_name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_scores_run ON scores(run_id, scorer_name)",
    "CREATE INDEX IF NOT EXISTS idx_case_results_run ON case_results(run_id)",
    # Migration for stores created before the scoreable column existed.
    # `ADD COLUMN IF NOT EXISTS` is a no-op when the column is already present
    # (fresh stores get it via CREATE TABLE above).
    "ALTER TABLE scores ADD COLUMN IF NOT EXISTS scoreable BOOLEAN",
    # One-time backfill for legacy rows: a "skipped" score was always written
    # with an "error" key in details_json (see ScoreResult.is_skipped).
    """
    UPDATE scores
    SET scoreable = (details_json NOT LIKE '%"error"%')
    WHERE scoreable IS NULL
    """,
]


class Store:
    """DuckDB-backed persistence for eval runs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        # ":memory:" stays in-memory; anything else opens a file (DuckDB creates it).
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        for stmt in _SCHEMA:
            self.conn.execute(stmt)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ============================================================
    # Writes
    # ============================================================

    def persist_run(
        self,
        results: list[EvalResult],
        *,
        system_name: str,
        config: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> str:
        """Persist one Runner.run() output. Returns the new run_id."""
        run_id = str(uuid.uuid4())
        started_at = min((r.timestamp for r in results), default=_utcnow())
        finished_at = max((r.timestamp for r in results), default=started_at)

        self.conn.begin()
        try:
            self.conn.execute(
                "INSERT INTO runs (id, started_at, finished_at, system_name, "
                "config_json, n_cases, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    started_at,
                    finished_at,
                    system_name,
                    json.dumps(config or {}),
                    len(results),
                    note,
                ],
            )
            for r in results:
                self.conn.execute(
                    "INSERT INTO case_results (run_id, case_id, query, answer, "
                    "citations_json, latency_ms, token_input, token_output, error, "
                    "timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        run_id,
                        r.case_id,
                        # The case query is recoverable from the eval set, but we
                        # snapshot it on the row so old runs stay self-describing
                        # if the eval set evolves.
                        r.output.raw.get("query", "") if r.output else "",
                        r.output.answer if r.output else None,
                        json.dumps(r.output.citations) if r.output else None,
                        r.latency_ms,
                        r.output.token_input if r.output else None,
                        r.output.token_output if r.output else None,
                        r.error,
                        r.timestamp,
                    ],
                )
                for s in r.scores:
                    self.conn.execute(
                        "INSERT INTO scores (run_id, case_id, scorer_name, value, "
                        "details_json, scoreable) VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            run_id,
                            r.case_id,
                            s.scorer_name,
                            s.value,
                            json.dumps(s.details),
                            not s.is_skipped,
                        ],
                    )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        return run_id

    # ============================================================
    # Reads
    # ============================================================

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, started_at, finished_at, system_name, n_cases, note "
            "FROM runs ORDER BY started_at DESC LIMIT ?",
            [limit],
        ).fetchall()
        return [
            {
                "id": r[0],
                "started_at": r[1],
                "finished_at": r[2],
                "system_name": r[3],
                "n_cases": r[4],
                "note": r[5],
            }
            for r in rows
        ]

    def scorer_means(self, run_id: str) -> dict[str, float]:
        """Mean score per scorer for one run, over scoreable cases only.

        Scorers that couldn't grade any case in this run (e.g., missing
        ``expected_answer`` for every case) are omitted from the result.
        Use ``scorer_stats`` to see the skipped counts.
        """
        rows = self.conn.execute(
            "SELECT scorer_name, AVG(value) FROM scores "
            "WHERE run_id = ? AND scoreable = TRUE "
            "GROUP BY scorer_name ORDER BY scorer_name",
            [run_id],
        ).fetchall()
        return {r[0]: float(r[1]) for r in rows}

    def scorer_stats(self, run_id: str) -> dict[str, dict[str, float | int | None]]:
        """Per-scorer aggregates: ``{scorer: {mean, n_scored, n_skipped}}``.

        ``mean`` is ``None`` when every case was skipped for that scorer.
        Includes every scorer that wrote at least one row, scored or not.
        """
        rows = self.conn.execute(
            """
            SELECT
                scorer_name,
                AVG(value) FILTER (WHERE scoreable = TRUE) AS mean,
                COUNT(*) FILTER (WHERE scoreable = TRUE) AS n_scored,
                COUNT(*) FILTER (WHERE scoreable = FALSE) AS n_skipped
            FROM scores
            WHERE run_id = ?
            GROUP BY scorer_name
            ORDER BY scorer_name
            """,
            [run_id],
        ).fetchall()
        return {
            r[0]: {
                "mean": float(r[1]) if r[1] is not None else None,
                "n_scored": int(r[2]),
                "n_skipped": int(r[3]),
            }
            for r in rows
        }

    def case_scores(self, run_id: str) -> list[dict[str, Any]]:
        """Per-case score breakdown for one run."""
        rows = self.conn.execute(
            "SELECT case_id, scorer_name, value FROM scores WHERE run_id = ? "
            "ORDER BY case_id, scorer_name",
            [run_id],
        ).fetchall()
        return [{"case_id": r[0], "scorer_name": r[1], "value": float(r[2])} for r in rows]

    def aggregate_runs(self, run_ids: list[str]) -> dict[str, dict[str, float | int | None]]:
        """Aggregate scoreable scores across multiple runs (e.g., N trials of the same eval).

        Returns ``{scorer_name: {mean, score_std, trial_mean_std, n_scored, n_skipped}}``.

        * ``mean`` — mean of all scoreable per-(case, trial) scores. Same as
          mean of per-trial means when each trial has the same case set.
        * ``score_std`` — sample std across all individual per-(case, trial)
          rows. **Conflates** "different cases score differently" with
          "the same case scores differently across trials". Large because
          the per-case spread dominates.
        * ``trial_mean_std`` — sample std across the N per-trial means.
          **This is what you want for "how reliable is one run's mean?"**
          A deterministic scorer (citation_overlap) returns 0.0 here even
          though score_std is non-zero. ``None`` when only one trial.
        * ``n_scored`` / ``n_skipped`` — total counts across all runs.

        Use case: ``--repeat N`` runs the eval N times, then this gives a
        proper measurement-reliability summary.
        """
        if not run_ids:
            return {}
        # `placeholders` is purely "?,?,?" — no user input ever flows into
        # the f-string. The actual run_ids are passed as bound parameters
        # via `self.conn.execute(..., run_ids)`. SQL injection impossible.
        placeholders = ",".join(["?"] * len(run_ids))

        # Query 1: per-(scorer, trial) means — used for trial_mean_std.
        per_trial_rows = self.conn.execute(
            f"""
            SELECT
                scorer_name,
                run_id,
                AVG(value) FILTER (WHERE scoreable = TRUE) AS trial_mean
            FROM scores
            WHERE run_id IN ({placeholders})
            GROUP BY scorer_name, run_id
            """,  # nosec B608 -- placeholders are bind markers, not user data
            run_ids,
        ).fetchall()
        # Group trial means by scorer for std calculation in Python.
        # DuckDB STDDEV is sample std (n-1); replicate that here.
        per_scorer_means: dict[str, list[float]] = {}
        for scorer_name, _run_id, tm in per_trial_rows:
            if tm is None:
                continue
            per_scorer_means.setdefault(scorer_name, []).append(float(tm))

        def _sample_std(values: list[float]) -> float | None:
            n = len(values)
            if n < 2:
                return None
            mean = sum(values) / n
            return float((sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5)

        # Query 2: overall per-scorer stats (mean, score_std, counts).
        overall_rows = self.conn.execute(
            f"""
            SELECT
                scorer_name,
                AVG(value)    FILTER (WHERE scoreable = TRUE) AS mean,
                STDDEV(value) FILTER (WHERE scoreable = TRUE) AS score_std,
                COUNT(*)      FILTER (WHERE scoreable = TRUE) AS n_scored,
                COUNT(*)      FILTER (WHERE scoreable = FALSE) AS n_skipped
            FROM scores
            WHERE run_id IN ({placeholders})
            GROUP BY scorer_name
            ORDER BY scorer_name
            """,  # nosec B608 -- placeholders are bind markers, not user data
            run_ids,
        ).fetchall()

        return {
            r[0]: {
                "mean": float(r[1]) if r[1] is not None else None,
                "score_std": float(r[2]) if r[2] is not None else None,
                "trial_mean_std": _sample_std(per_scorer_means.get(r[0], [])),
                "n_scored": int(r[3]),
                "n_skipped": int(r[4]),
            }
            for r in overall_rows
        }

    def flaky_cases(self, run_ids: list[str]) -> list[dict[str, Any]]:
        """Return cases that are scoreable in some trials and skipped in others.

        Signals an intermittent failure (API timeout, parse error,
        edge-case prompt). Sorted by (scorer_name, case_id).
        """
        if not run_ids:
            return []
        # See note in aggregate_runs — `placeholders` is "?,?,?" only.
        placeholders = ",".join(["?"] * len(run_ids))
        rows = self.conn.execute(
            f"""
            SELECT
                scorer_name,
                case_id,
                SUM(CASE WHEN scoreable = TRUE THEN 1 ELSE 0 END)  AS n_scored,
                SUM(CASE WHEN scoreable = FALSE THEN 1 ELSE 0 END) AS n_skipped
            FROM scores
            WHERE run_id IN ({placeholders})
            GROUP BY scorer_name, case_id
            HAVING n_scored > 0 AND n_skipped > 0
            ORDER BY scorer_name, case_id
            """,  # nosec B608 -- placeholders are bind markers, not user data
            run_ids,
        ).fetchall()
        return [
            {
                "scorer_name": r[0],
                "case_id": r[1],
                "n_scored": int(r[2]),
                "n_skipped": int(r[3]),
            }
            for r in rows
        ]

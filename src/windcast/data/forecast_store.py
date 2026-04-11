"""Forecast predictions SQLite store — upsert, query, coverage."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS forecasts (
    target_timestamp TEXT NOT NULL,
    horizon_h INTEGER NOT NULL,
    prediction_mw REAL NOT NULL,
    model_name TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT 'demand',
    dataset TEXT NOT NULL DEFAULT 'rte_france',
    created_at TEXT NOT NULL,
    PRIMARY KEY (target_timestamp, horizon_h, model_name)
);
"""


class ForecastStore:
    """SQLite-based forecast prediction store with upsert and temporal queries."""

    def __init__(self, db_path: Path) -> None:
        """Open or create SQLite database at db_path."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    _REQUIRED_KEYS = frozenset({"target_timestamp", "horizon_h", "prediction_mw", "model_name"})

    def upsert(self, forecasts: list[dict]) -> int:
        """Insert or replace forecast predictions.

        Args:
            forecasts: List of dicts with keys: target_timestamp, horizon_h,
                prediction_mw, model_name. Optional: domain, dataset, created_at.

        Returns:
            Number of rows upserted.

        Raises:
            ValueError: If any forecast dict is missing required keys.
        """
        if not forecasts:
            return 0

        missing = self._REQUIRED_KEYS - forecasts[0].keys()
        if missing:
            raise ValueError(f"Forecast dict missing required keys: {missing}")

        rows = [
            (
                f["target_timestamp"],
                f["horizon_h"],
                f["prediction_mw"],
                f["model_name"],
                f.get("domain", "demand"),
                f.get("dataset", "rte_france"),
                f.get("created_at", datetime.now(UTC).isoformat()),
            )
            for f in forecasts
        ]

        self._conn.executemany(
            "INSERT OR REPLACE INTO forecasts "
            "(target_timestamp, horizon_h, prediction_mw, model_name, domain, dataset, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        logger.info("Upserted %d forecast rows", len(rows))
        return len(rows)

    def query(
        self,
        start: str,
        end: str,
        horizon_h: int | None = None,
        model_name: str | None = None,
    ) -> pl.DataFrame:
        """Query forecasts for a date range.

        Args:
            start: Start date ISO "YYYY-MM-DD".
            end: End date ISO "YYYY-MM-DD".
            horizon_h: Optional filter on forecast horizon.
            model_name: Optional filter on model name.

        Returns:
            DataFrame with columns: target_timestamp, horizon_h, prediction_mw,
            model_name, domain, dataset, created_at.
        """
        sql = (
            "SELECT target_timestamp, horizon_h, prediction_mw, model_name, "
            "domain, dataset, created_at FROM forecasts "
            "WHERE target_timestamp >= ? AND target_timestamp <= ?"
        )
        params: list[str | int] = [start, end + "T23:59:59"]

        if horizon_h is not None:
            sql += " AND horizon_h = ?"
            params.append(horizon_h)

        if model_name is not None:
            sql += " AND model_name = ?"
            params.append(model_name)

        sql += " ORDER BY target_timestamp"

        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()

        if not rows:
            return pl.DataFrame(
                schema={
                    "target_timestamp": pl.Utf8,
                    "horizon_h": pl.Int64,
                    "prediction_mw": pl.Float64,
                    "model_name": pl.Utf8,
                    "domain": pl.Utf8,
                    "dataset": pl.Utf8,
                    "created_at": pl.Utf8,
                }
            )

        return pl.DataFrame(
            rows,
            schema={
                "target_timestamp": pl.Utf8,
                "horizon_h": pl.Int64,
                "prediction_mw": pl.Float64,
                "model_name": pl.Utf8,
                "domain": pl.Utf8,
                "dataset": pl.Utf8,
                "created_at": pl.Utf8,
            },
            orient="row",
        )

    def get_coverage(self) -> tuple[str, str] | None:
        """Return (min_date, max_date) ISO strings, or None if empty."""
        cursor = self._conn.execute(
            "SELECT MIN(target_timestamp), MAX(target_timestamp) FROM forecasts"
        )
        row = cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return (row[0], row[1])

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

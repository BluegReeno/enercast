"""Tests for windcast.data.forecast_store — SQLite forecast prediction store."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from windcast.data.forecast_store import ForecastStore


@pytest.fixture()
def store(tmp_path):
    """Create a temporary ForecastStore."""
    s = ForecastStore(tmp_path / "test_forecast.db")
    yield s
    s.close()


def _make_forecast(
    target: str = "2025-06-15T12:00:00+00:00",
    horizon_h: int = 24,
    prediction_mw: float = 45000.0,
    model_name: str = "enercast-rte_france-xgboost",
) -> dict:
    return {
        "target_timestamp": target,
        "horizon_h": horizon_h,
        "prediction_mw": prediction_mw,
        "model_name": model_name,
        "domain": "demand",
        "dataset": "rte_france",
        "created_at": datetime.now(UTC).isoformat(),
    }


class TestCreateAndUpsert:
    def test_insert_five_forecasts(self, store):
        forecasts = [
            _make_forecast(target=f"2025-06-15T{h:02d}:00:00+00:00", prediction_mw=40000 + h * 100)
            for h in range(5)
        ]
        count = store.upsert(forecasts)
        assert count == 5

    def test_upsert_empty_list(self, store):
        count = store.upsert([])
        assert count == 0


class TestQuery:
    def test_query_by_date_range(self, store):
        forecasts = [
            _make_forecast(target=f"2025-06-{d:02d}T12:00:00+00:00", prediction_mw=40000 + d * 100)
            for d in range(10, 20)
        ]
        store.upsert(forecasts)

        result = store.query("2025-06-12", "2025-06-15")
        assert len(result) == 4  # days 12, 13, 14, 15

    def test_query_by_horizon(self, store):
        forecasts = [
            _make_forecast(
                target="2025-06-15T12:00:00+00:00",
                horizon_h=h,
                prediction_mw=40000 + h * 100,
            )
            for h in [1, 6, 12, 24, 48]
        ]
        store.upsert(forecasts)

        result = store.query("2025-06-15", "2025-06-15", horizon_h=24)
        assert len(result) == 1
        assert result["horizon_h"][0] == 24

    def test_query_by_model_name(self, store):
        forecasts = [
            _make_forecast(model_name="model-a"),
            _make_forecast(
                target="2025-06-15T13:00:00+00:00",
                model_name="model-b",
            ),
        ]
        store.upsert(forecasts)

        result = store.query("2025-06-15", "2025-06-15", model_name="model-a")
        assert len(result) == 1


class TestUpsertIdempotent:
    def test_same_forecast_twice_no_duplicate(self, store):
        f = _make_forecast()
        store.upsert([f])
        store.upsert([f])  # Same PK

        result = store.query("2025-06-15", "2025-06-15")
        assert len(result) == 1

    def test_upsert_updates_prediction(self, store):
        f1 = _make_forecast(prediction_mw=40000)
        store.upsert([f1])

        f2 = _make_forecast(prediction_mw=42000)
        store.upsert([f2])

        result = store.query("2025-06-15", "2025-06-15")
        assert len(result) == 1
        assert result["prediction_mw"][0] == pytest.approx(42000.0)


class TestGetCoverage:
    def test_coverage_returns_min_max(self, store):
        forecasts = [
            _make_forecast(target="2025-06-10T00:00:00+00:00"),
            _make_forecast(target="2025-06-20T23:00:00+00:00"),
        ]
        store.upsert(forecasts)

        coverage = store.get_coverage()
        assert coverage is not None
        assert coverage[0] == "2025-06-10T00:00:00+00:00"
        assert coverage[1] == "2025-06-20T23:00:00+00:00"


class TestUpsertValidation:
    def test_missing_required_key_raises(self, store):
        bad = [{"horizon_h": 24, "prediction_mw": 40000}]  # missing target_timestamp, model_name
        with pytest.raises(ValueError, match="missing required keys"):
            store.upsert(bad)


class TestEmptyStore:
    def test_query_empty_returns_empty_df(self, store):
        result = store.query("2025-06-15", "2025-06-15")
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert "target_timestamp" in result.columns

    def test_coverage_empty_returns_none(self, store):
        assert store.get_coverage() is None

"""Tests for the inference pipeline (Pass 12)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from windcast.data.open_meteo import FORECAST_URL
from windcast.data.schema import QC_OK

# ---------------------------------------------------------------------------
# fetch_forecast_weather
# ---------------------------------------------------------------------------


class TestFetchForecastWeather:
    """Tests for the Open-Meteo Forecast API fetcher."""

    def test_uses_forecast_url(self):
        """Verify FORECAST_URL points to the live forecast endpoint."""
        assert FORECAST_URL == "https://api.open-meteo.com/v1/forecast"

    def test_params_include_forecast_days(self):
        """Verify fetch_forecast_weather passes forecast_days, not start_date/end_date."""
        from windcast.data.open_meteo import fetch_forecast_weather

        n_hours = 48
        variables = ["wind_speed_100m", "temperature_2m"]

        mock_hourly_var = MagicMock()
        mock_hourly_var.ValuesAsNumpy.return_value = np.random.rand(n_hours).astype(np.float32)

        mock_hourly = MagicMock()
        mock_hourly.Time.return_value = 1609459200
        mock_hourly.TimeEnd.return_value = 1609459200 + 3600 * n_hours
        mock_hourly.Interval.return_value = 3600
        mock_hourly.Variables.return_value = mock_hourly_var

        mock_response = MagicMock()
        mock_response.Hourly.return_value = mock_hourly

        mock_client = MagicMock()
        mock_client.weather_api.return_value = [mock_response]

        df = fetch_forecast_weather(
            latitude=52.4,
            longitude=-0.9,
            variables=variables,
            forecast_days=3,
            past_days=1,
            client=mock_client,
        )

        # Check the client was called with correct URL and params
        call_args = mock_client.weather_api.call_args
        assert call_args[0][0] == FORECAST_URL
        params = call_args[1].get("params") or call_args[0][1]
        assert "forecast_days" in params
        assert params["forecast_days"] == 3
        assert params["past_days"] == 1
        assert "start_date" not in params
        assert "end_date" not in params

        # Output is a Polars DataFrame with expected columns
        assert isinstance(df, pl.DataFrame)
        assert "timestamp_utc" in df.columns
        for var in variables:
            assert var in df.columns
        assert len(df) == n_hours

    def test_output_schema_matches_existing_fetchers(self):
        """Verify forecast fetcher output has same schema as archive fetcher."""
        from windcast.data.open_meteo import fetch_forecast_weather

        n_hours = 24
        variables = ["wind_speed_100m"]

        mock_hourly_var = MagicMock()
        mock_hourly_var.ValuesAsNumpy.return_value = np.random.rand(n_hours).astype(np.float32)

        mock_hourly = MagicMock()
        mock_hourly.Time.return_value = 1609459200
        mock_hourly.TimeEnd.return_value = 1609459200 + 3600 * n_hours
        mock_hourly.Interval.return_value = 3600
        mock_hourly.Variables.return_value = mock_hourly_var

        mock_response = MagicMock()
        mock_response.Hourly.return_value = mock_hourly

        mock_client = MagicMock()
        mock_client.weather_api.return_value = [mock_response]

        df = fetch_forecast_weather(
            latitude=52.4, longitude=-0.9, variables=variables, client=mock_client
        )

        assert df["timestamp_utc"].dtype == pl.Datetime("us", "UTC")
        assert df["wind_speed_100m"].dtype == pl.Float32


# ---------------------------------------------------------------------------
# build_inference_features
# ---------------------------------------------------------------------------


def _make_wind_actuals(n_rows: int = 50) -> pl.DataFrame:
    """Create synthetic wind actuals for testing."""
    base_ts = datetime(2026, 4, 10, 0, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp_utc": [base_ts + timedelta(minutes=10 * i) for i in range(n_rows)],
            "turbine_id": ["kwf1"] * n_rows,
            "active_power_kw": np.random.uniform(100, 2000, n_rows).tolist(),
            "wind_speed_ms": np.random.uniform(3, 15, n_rows).tolist(),
            "wind_direction_deg": np.random.uniform(0, 360, n_rows).tolist(),
            "qc_flag": [QC_OK] * n_rows,
        }
    ).cast({"timestamp_utc": pl.Datetime("us", "UTC")})


def _make_demand_actuals(n_rows: int = 200) -> pl.DataFrame:
    """Create synthetic demand actuals for testing."""
    base_ts = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp_utc": [base_ts + timedelta(hours=i) for i in range(n_rows)],
            "zone_id": ["FR"] * n_rows,
            "load_mw": np.random.uniform(30000, 70000, n_rows).tolist(),
            "is_holiday": [0] * n_rows,
            "qc_flag": [QC_OK] * n_rows,
        }
    ).cast({"timestamp_utc": pl.Datetime("us", "UTC")})


def _make_nwp(n_hours: int = 120, base_ts: datetime | None = None) -> pl.DataFrame:
    """Create synthetic NWP forecast for testing."""
    if base_ts is None:
        base_ts = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp_utc": [base_ts + timedelta(hours=i) for i in range(n_hours)],
            "wind_speed_100m": np.random.uniform(3, 15, n_hours).tolist(),
            "wind_direction_100m": np.random.uniform(0, 360, n_hours).tolist(),
            "temperature_2m": np.random.uniform(-5, 25, n_hours).tolist(),
            "wind_speed_10m": np.random.uniform(2, 10, n_hours).tolist(),
            "relative_humidity_2m": np.random.uniform(40, 95, n_hours).tolist(),
        }
    ).cast({"timestamp_utc": pl.Datetime("us", "UTC")})


class TestBuildInferenceFeatures:
    """Tests for the inference feature builder."""

    def test_wind_produces_single_row(self):
        """Verify wind inference produces exactly 1 row."""
        from scripts.inference import build_inference_features

        actuals = _make_wind_actuals(50)
        base_ts = actuals["timestamp_utc"].min()
        nwp = _make_nwp(120, base_ts=base_ts)

        result = build_inference_features(
            actuals_df=actuals,
            nwp_df=nwp,
            domain="wind",
            feature_set_name="wind_full",
            horizon=6,
            resolution_minutes=10,
        )
        assert len(result) == 1
        assert isinstance(result, pl.DataFrame)

    def test_wind_canonical_column_names(self):
        """Verify NWP columns use canonical names (no _h{N} suffix)."""
        from scripts.inference import build_inference_features

        actuals = _make_wind_actuals(50)
        base_ts = actuals["timestamp_utc"].min()
        nwp = _make_nwp(120, base_ts=base_ts)

        result = build_inference_features(
            actuals_df=actuals,
            nwp_df=nwp,
            domain="wind",
            feature_set_name="wind_full",
            horizon=6,
            resolution_minutes=10,
        )
        # Should have canonical names, not _h6 suffixes
        for col in result.columns:
            assert "_h6" not in col, f"Found horizon-suffixed column: {col}"

        # Should contain NWP canonical columns
        assert "nwp_wind_speed_100m" in result.columns
        assert "nwp_temperature_2m" in result.columns

    def test_wind_has_expected_feature_columns(self):
        """Verify the output contains feature columns from wind_full."""
        from scripts.inference import build_inference_features

        actuals = _make_wind_actuals(50)
        base_ts = actuals["timestamp_utc"].min()
        nwp = _make_nwp(120, base_ts=base_ts)

        result = build_inference_features(
            actuals_df=actuals,
            nwp_df=nwp,
            domain="wind",
            feature_set_name="wind_full",
            horizon=6,
            resolution_minutes=10,
        )
        # Key baseline features
        assert "wind_speed_ms" in result.columns
        assert "wind_dir_sin" in result.columns
        assert "active_power_kw_lag1" in result.columns

    def test_demand_produces_single_row(self):
        """Verify demand inference produces exactly 1 row."""
        from scripts.inference import build_inference_features

        actuals = _make_demand_actuals(200)
        base_ts = actuals["timestamp_utc"].min()
        nwp = _make_nwp(300, base_ts=base_ts)

        result = build_inference_features(
            actuals_df=actuals,
            nwp_df=nwp,
            domain="demand",
            feature_set_name="demand_full",
            horizon=24,
            resolution_minutes=60,
        )
        assert len(result) == 1

    def test_insufficient_actuals_raises(self):
        """Verify error when not enough rows for lag computation."""
        from scripts.inference import build_inference_features

        actuals = _make_wind_actuals(3)  # Too few for lags
        nwp = _make_nwp(24, base_ts=actuals["timestamp_utc"].min())

        with pytest.raises(ValueError, match="No valid feature rows"):
            build_inference_features(
                actuals_df=actuals,
                nwp_df=nwp,
                domain="wind",
                feature_set_name="wind_full",
                horizon=6,
                resolution_minutes=10,
            )


# ---------------------------------------------------------------------------
# Output JSON structure
# ---------------------------------------------------------------------------


class TestOutputFormat:
    """Tests for the output JSON structure."""

    def test_output_has_required_fields(self):
        """Verify output JSON contains all required fields."""
        from scripts.inference import UNIT_BY_DOMAIN
        from windcast.training.harness import build_horizon_desc

        # Simulate output construction
        output = {
            "domain": "wind",
            "dataset": "kelmarsh",
            "model": "models:/enercast-kelmarsh-xgboost@champion",
            "horizon_steps": 24,
            "horizon_desc": build_horizon_desc(24, 10),
            "target_timestamp_utc": datetime.now(UTC).isoformat(),
            "prediction": 1234.5,
            "unit": UNIT_BY_DOMAIN["wind"],
            "nwp_source": "Open-Meteo Forecast API",
            "generated_at": datetime.now(UTC).isoformat(),
        }

        required_fields = [
            "domain",
            "dataset",
            "model",
            "horizon_steps",
            "horizon_desc",
            "target_timestamp_utc",
            "prediction",
            "unit",
            "nwp_source",
            "generated_at",
        ]
        for field in required_fields:
            assert field in output, f"Missing field: {field}"

        # Validate JSON serializable
        json_str = json.dumps(output, default=str)
        parsed = json.loads(json_str)
        assert parsed["domain"] == "wind"
        assert parsed["unit"] == "kW"
        assert isinstance(parsed["prediction"], float)

    def test_unit_mapping(self):
        """Verify unit mapping for all domains."""
        from scripts.inference import UNIT_BY_DOMAIN

        assert UNIT_BY_DOMAIN["wind"] == "kW"
        assert UNIT_BY_DOMAIN["demand"] == "MW"
        assert UNIT_BY_DOMAIN["solar"] == "kW"

    def test_horizon_desc_formatting(self):
        """Verify horizon description formatting."""
        from windcast.training.harness import build_horizon_desc

        assert build_horizon_desc(6, 10) == "1h ahead"
        assert build_horizon_desc(24, 10) == "4h ahead"
        assert build_horizon_desc(24, 60) == "D+1"
        assert build_horizon_desc(3, 10) == "30 min ahead"

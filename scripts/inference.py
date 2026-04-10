"""Run inference: fetch live NWP → build features → load model → predict.

Closes the gap between experiment and production — a trained EnerCast model
becomes usable for real forecasting, not just backtesting.

Usage:
    uv run python scripts/inference.py --domain wind --horizon 6
    uv run python scripts/inference.py --domain demand --dataset rte_france --horizon 24
    uv run python scripts/inference.py --model-uri "runs:/{run_id}/model_h06"
    uv run python scripts/inference.py --serve-url http://localhost:5001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from windcast.config import DOMAIN_RESOLUTION, get_settings
from windcast.features.registry import get_feature_set
from windcast.features.weather import join_nwp_horizon_features
from windcast.training.harness import build_horizon_desc, resolve_horizon_features

logger = logging.getLogger(__name__)

UNIT_BY_DOMAIN: dict[str, str] = {
    "wind": "kW",
    "demand": "MW",
    "solar": "kW",
}


def build_inference_features(
    actuals_df: pl.DataFrame,
    nwp_df: pl.DataFrame,
    domain: str,
    feature_set_name: str,
    horizon: int,
    resolution_minutes: int,
) -> pl.DataFrame:
    """Build a single-row feature vector for inference.

    Reuses the same feature builders as training to guarantee feature parity.

    1. Join NWP at the target horizon onto the actuals tail.
    2. Build domain features (lags, rolling, cyclic, HDD/CDD).
    3. Resolve horizon-specific NWP columns → canonical names.
    4. Return a single-row DataFrame with only the model's input columns.

    Args:
        actuals_df: Recent actuals from processed Parquet (tail rows for lags).
        nwp_df: Fresh NWP forecast DataFrame (timestamp_utc + variables).
        domain: "wind", "demand", or "solar".
        feature_set_name: Feature set name from registry (e.g., "wind_full").
        horizon: Forecast horizon in steps.
        resolution_minutes: Minutes per step.

    Returns:
        Single-row DataFrame with canonical feature columns ready for prediction.
    """
    from windcast.features.demand import build_demand_features
    from windcast.features.wind import build_wind_features

    horizons = [horizon]

    if domain == "demand":
        # Demand full needs NWP joined BEFORE build_demand_features for HDD/CDD
        df = join_nwp_horizon_features(
            actuals_df,
            nwp_df,
            horizons=horizons,
            resolution_minutes=resolution_minutes,
        )
        df = build_demand_features(df, feature_set=feature_set_name)
    elif domain == "wind":
        df = build_wind_features(
            actuals_df,
            feature_set=feature_set_name,
            weather_df=nwp_df,
            horizons=horizons,
            resolution_minutes=resolution_minutes,
        )
    else:
        raise ValueError(f"Unsupported domain for inference: {domain!r}")

    # Resolve horizon-specific NWP columns to canonical names
    fs = get_feature_set(feature_set_name)
    actual_cols, rename_map = resolve_horizon_features(df.columns, fs.columns, horizon)

    # Filter to only the columns the model expects
    available = [c for c in actual_cols if c in df.columns]
    if not available:
        raise ValueError(
            f"No feature columns available after building. "
            f"Expected: {actual_cols[:5]}... Available: {df.columns[:10]}..."
        )

    result = df.select(available)
    if rename_map:
        result = result.rename(rename_map)

    # Drop rows with any null (lag/rolling warmup) and take the last valid row
    result = result.drop_nulls()
    if result.is_empty():
        raise ValueError(
            "No valid feature rows after dropping nulls. "
            "Ensure actuals_df has enough rows for lag computation."
        )

    return result.tail(1)


def _load_model_direct(model_uri: str) -> Any:
    """Load a model via MLflow pyfunc."""
    import mlflow.pyfunc

    logger.info("Loading model from %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)


def _predict_direct(model: Any, X: pl.DataFrame) -> float:
    """Run prediction via direct model loading."""
    X_pd = X.to_pandas()
    predictions = model.predict(X_pd)
    return float(predictions[0])


def _predict_server(serve_url: str, X: pl.DataFrame) -> float:
    """Run prediction via MLflow serving endpoint."""
    import requests

    X_pd = X.to_pandas()
    payload = {
        "dataframe_split": {
            "columns": X_pd.columns.tolist(),
            "data": X_pd.values.tolist(),
        }
    }

    url = f"{serve_url.rstrip('/')}/invocations"
    logger.info("POSTing to %s", url)
    resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    resp.raise_for_status()

    result = resp.json()
    if isinstance(result, dict) and "predictions" in result:
        return float(result["predictions"][0])
    if isinstance(result, list):
        return float(result[0])
    return float(result)


def _resolve_actuals_path(domain: str, dataset: str, settings: Any) -> Path:
    """Resolve the path to processed actuals Parquet."""
    if domain in ("demand", "solar"):
        return settings.processed_dir / f"{dataset}.parquet"
    return settings.processed_dir / "kelmarsh_kwf1.parquet"


def _tail_rows_for_domain(domain: str) -> int:
    """How many recent rows to load for lag/rolling computation."""
    if domain == "demand":
        return 200  # 168 (week lag) + buffer
    return 50  # 24 (day lag) + buffer for wind


def main() -> None:
    """Run inference pipeline."""
    parser = argparse.ArgumentParser(
        description="Run EnerCast inference: fetch NWP → build features → predict"
    )
    parser.add_argument(
        "--domain",
        choices=["wind", "demand", "solar"],
        default="wind",
        help="Domain: wind, demand, or solar. Default: wind",
    )
    parser.add_argument("--dataset", default=None, help="Dataset ID. Default: domain-specific")
    parser.add_argument(
        "--model-name",
        default=None,
        help="MLflow registered model name. Default: enercast-{dataset}-xgboost",
    )
    parser.add_argument(
        "--model-alias",
        default="champion",
        help="MLflow model alias. Default: champion",
    )
    parser.add_argument(
        "--model-uri",
        default=None,
        help="Direct MLflow model URI (overrides --model-name/--model-alias)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Forecast horizon in steps. Default: 24",
    )
    parser.add_argument(
        "--feature-set",
        default=None,
        help="Feature set name. Default: {domain}_full",
    )
    parser.add_argument(
        "--serve-url",
        default=None,
        help="MLflow serving URL (if set, POST instead of direct model load)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path for JSON. Default: stdout",
    )
    parser.add_argument(
        "--actuals-path",
        type=Path,
        default=None,
        help="Path to processed Parquet for lag features. Default: auto-resolve",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()

    # Resolve defaults
    domain = args.domain
    domain_dataset_defaults = {
        "wind": "kelmarsh",
        "demand": "rte_france",
        "solar": "pvdaq_system4",
    }
    dataset = args.dataset or domain_dataset_defaults[domain]
    feature_set = args.feature_set or f"{domain}_full"
    resolution_minutes = DOMAIN_RESOLUTION[domain]
    # Resolve weather config name (matches dataset for demand, "kelmarsh" for wind)
    domain_weather_map = {
        "wind": "kelmarsh",
        "demand": dataset,
        "solar": dataset,
    }
    weather_config_name = domain_weather_map[domain]

    # 1. Load model
    if args.model_uri:
        model_uri = args.model_uri
    else:
        model_name = args.model_name or f"enercast-{dataset}-xgboost"
        model_uri = f"models:/{model_name}@{args.model_alias}"

    model = None
    if not args.serve_url:
        model = _load_model_direct(model_uri)

    # 2. Load recent actuals
    actuals_path = args.actuals_path or _resolve_actuals_path(domain, dataset, settings)
    if not actuals_path.exists():
        logger.error("Actuals file not found: %s", actuals_path)
        sys.exit(1)

    tail_rows = _tail_rows_for_domain(domain)
    logger.info("Loading %d tail rows from %s", tail_rows, actuals_path)
    actuals_df = pl.read_parquet(actuals_path)
    actuals_df = actuals_df.sort("timestamp_utc").tail(tail_rows)
    last_actual_ts = actuals_df["timestamp_utc"].max()
    logger.info("Actuals: %d rows, last timestamp: %s", len(actuals_df), last_actual_ts)

    # 3. Fetch live NWP
    from windcast.weather import get_live_forecast

    logger.info("Fetching live NWP forecast for %s", weather_config_name)
    nwp_df = get_live_forecast(weather_config_name, forecast_days=3, past_days=2)
    logger.info("NWP forecast: %d rows, columns: %s", len(nwp_df), nwp_df.columns)

    # 4. Build features
    logger.info("Building features: %s, horizon=%d", feature_set, args.horizon)
    X = build_inference_features(
        actuals_df=actuals_df,
        nwp_df=nwp_df,
        domain=domain,
        feature_set_name=feature_set,
        horizon=args.horizon,
        resolution_minutes=resolution_minutes,
    )
    logger.info("Feature vector: %d columns: %s", len(X.columns), X.columns)

    # 5. Predict
    if args.serve_url:
        prediction = _predict_server(args.serve_url, X)
    else:
        assert model is not None, "No model loaded and no --serve-url provided"
        prediction = _predict_direct(model, X)

    # 6. Format output
    horizon_desc = build_horizon_desc(args.horizon, resolution_minutes)
    target_ts = last_actual_ts + timedelta(minutes=args.horizon * resolution_minutes)

    output = {
        "domain": domain,
        "dataset": dataset,
        "model": model_uri,
        "horizon_steps": args.horizon,
        "horizon_desc": horizon_desc,
        "target_timestamp_utc": target_ts.isoformat() if target_ts else None,
        "prediction": round(prediction, 1),
        "unit": UNIT_BY_DOMAIN.get(domain, "?"),
        "nwp_source": "Open-Meteo Forecast API",
        "generated_at": datetime.now(UTC).isoformat(),
    }

    output_json = json.dumps(output, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output_json)
        logger.info("Output written to %s", args.output)
    else:
        print(output_json)


if __name__ == "__main__":
    main()

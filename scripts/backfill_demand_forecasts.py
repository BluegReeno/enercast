"""Backfill demand forecasts — iterate over a date range, predict, store.

Generates model forecasts for historical timestamps and stores them in the
ForecastStore SQLite database. Idempotent: re-running skips existing forecasts.

Usage:
    uv run python scripts/backfill_demand_forecasts.py \
        --start 2025-01-01 --end 2025-01-03 --horizon 24

    uv run python scripts/backfill_demand_forecasts.py \
        --start 2025-01-01 --end 2026-04-11 \
        --model-name enercast-rte_france-xgboost
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = [1, 6, 12, 24, 48]
DEMAND_TAIL_ROWS = 200  # 168h week lag + buffer


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill demand forecasts into ForecastStore")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--horizon",
        type=int,
        nargs="*",
        default=None,
        help="Horizons to backfill (default: all [1,6,12,24,48])",
    )
    parser.add_argument(
        "--model-name",
        default="enercast-rte_france-xgboost",
        help="MLflow registered model name",
    )
    parser.add_argument("--model-alias", default="champion", help="MLflow model alias")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/forecast_store.db"),
        help="ForecastStore SQLite path",
    )
    parser.add_argument(
        "--actuals-path",
        type=Path,
        default=None,
        help="Path to processed actuals Parquet (default: auto)",
    )
    parser.add_argument("--batch-size", type=int, default=24, help="Hours per batch upsert")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    horizons = args.horizon or DEFAULT_HORIZONS
    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)

    logger.info(
        "Backfill: %s → %s, horizons=%s, model=%s@%s",
        args.start,
        args.end,
        horizons,
        args.model_name,
        args.model_alias,
    )

    # Lazy imports to avoid slow import on --help
    import mlflow
    import mlflow.pyfunc

    from windcast.config import get_settings
    from windcast.data.forecast_store import ForecastStore
    from windcast.weather import get_forecast_weather, get_live_forecast

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.inference import build_inference_features

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # 1. Load model
    model_uri = f"models:/{args.model_name}@{args.model_alias}"
    logger.info("Loading model: %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)

    # 2. Load actuals
    actuals_path = args.actuals_path or settings.processed_dir / "rte_france.parquet"
    if not actuals_path.exists():
        logger.error("Actuals file not found: %s", actuals_path)
        sys.exit(1)

    logger.info("Loading actuals from %s", actuals_path)
    all_actuals = pl.read_parquet(actuals_path).sort("timestamp_utc")
    actuals_timestamps = all_actuals["timestamp_utc"]
    actuals_min = actuals_timestamps.min()
    actuals_max = actuals_timestamps.max()
    logger.info("Actuals: %d rows, %s → %s", len(all_actuals), actuals_min, actuals_max)

    # 3. Load NWP — historical forecast weather for the backfill period
    # We need NWP from (start - max_horizon) to end
    max_horizon_h = max(horizons)
    nwp_start = start_date - timedelta(hours=max_horizon_h + 24)
    nwp_end = end_date + timedelta(hours=max_horizon_h + 24)

    # Clamp to historical forecast coverage (2022-01-01+)
    nwp_start_str = max(nwp_start.strftime("%Y-%m-%d"), "2022-01-01")
    nwp_end_str = nwp_end.strftime("%Y-%m-%d")

    logger.info("Loading NWP: %s → %s", nwp_start_str, nwp_end_str)

    # Use historical forecast for bulk, live forecast for recent dates
    today = datetime.now(UTC)
    live_cutoff = today - timedelta(days=5)

    nwp_parts: list[pl.DataFrame] = []

    # Historical NWP (if needed)
    hist_end_str = min(nwp_end_str, live_cutoff.strftime("%Y-%m-%d"))
    if nwp_start_str < hist_end_str:
        logger.info("Fetching historical NWP: %s → %s", nwp_start_str, hist_end_str)
        try:
            nwp_hist = get_forecast_weather("rte_france", nwp_start_str, hist_end_str)
            if not nwp_hist.is_empty():
                nwp_parts.append(nwp_hist)
        except Exception as e:
            logger.warning("Historical NWP fetch failed: %s", e)

    # Live NWP (for recent dates)
    if nwp_end > live_cutoff:
        logger.info("Fetching live NWP forecast")
        try:
            nwp_live = get_live_forecast("rte_france", forecast_days=7, past_days=5)
            if not nwp_live.is_empty():
                nwp_parts.append(nwp_live)
        except Exception as e:
            logger.warning("Live NWP fetch failed: %s", e)

    if not nwp_parts:
        logger.error("No NWP data available for the backfill period")
        sys.exit(1)

    nwp_df = pl.concat(nwp_parts).sort("timestamp_utc").unique("timestamp_utc")
    logger.info("NWP: %d rows, columns: %s", len(nwp_df), nwp_df.columns)

    # 4. Open forecast store
    store = ForecastStore(args.db_path)
    logger.info("ForecastStore: %s", args.db_path)

    # 5. Iterate over target timestamps (hourly)
    total_generated = 0
    total_skipped = 0
    batch: list[dict] = []
    now_iso = datetime.now(UTC).isoformat()

    current_ts = start_date
    total_hours = int((end_date - start_date).total_seconds() / 3600)

    while current_ts <= end_date:
        hours_done = int((current_ts - start_date).total_seconds() / 3600)
        if hours_done > 0 and hours_done % 100 == 0:
            logger.info(
                "Progress: %d/%d hours (%.0f%%)",
                hours_done,
                total_hours,
                100 * hours_done / max(total_hours, 1),
            )

        for h in horizons:
            # reference_time = when the forecast is "made" = target - horizon * resolution
            reference_time = current_ts - timedelta(hours=h)

            # Slice actuals up to reference_time using binary search (O(log n))
            cutoff_idx = actuals_timestamps.search_sorted(reference_time, side="right")
            start_idx = max(0, cutoff_idx - DEMAND_TAIL_ROWS)
            actuals_tail = all_actuals.slice(start_idx, cutoff_idx - start_idx)

            if len(actuals_tail) < 170:  # Need at least lag168 + a few rows
                total_skipped += 1
                continue

            try:
                features = build_inference_features(
                    actuals_df=actuals_tail,
                    nwp_df=nwp_df,
                    domain="demand",
                    feature_set_name="demand_full",
                    horizon=h,
                    resolution_minutes=60,
                    reference_time=reference_time,
                )

                features_pd = features.to_pandas()

                # Align integer dtypes with model signature
                if hasattr(model, "metadata") and model.metadata.signature:
                    for col_spec in model.metadata.signature.inputs:
                        if col_spec.name in features_pd.columns and col_spec.type == "integer":
                            features_pd[col_spec.name] = features_pd[col_spec.name].astype("int32")

                prediction = float(model.predict(features_pd, params={"horizon": h})[0])

                batch.append(
                    {
                        "target_timestamp": current_ts.isoformat(),
                        "horizon_h": h,
                        "prediction_mw": round(prediction, 1),
                        "model_name": args.model_name,
                        "domain": "demand",
                        "dataset": "rte_france",
                        "created_at": now_iso,
                    }
                )
                total_generated += 1

            except Exception as e:
                logger.debug("Skipped %s h%d: %s", current_ts.isoformat(), h, e)
                total_skipped += 1

        # Batch upsert
        if len(batch) >= args.batch_size * len(horizons):
            store.upsert(batch)
            batch.clear()

        current_ts += timedelta(hours=1)

    # Final flush
    if batch:
        store.upsert(batch)

    coverage = store.get_coverage()
    store.close()

    logger.info(
        "Backfill complete: %d forecasts generated, %d skipped. Coverage: %s",
        total_generated,
        total_skipped,
        coverage,
    )


if __name__ == "__main__":
    main()

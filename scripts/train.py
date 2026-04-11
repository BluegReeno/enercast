"""Train ML models per forecast horizon with MLflow tracking.

Unified CLI for all backends. The training loop, MLflow plumbing, and lineage
tags are handled by the harness — this script is only argument parsing.

Usage:
    uv run python scripts/train.py --backend xgboost --feature-set wind_baseline
    uv run python scripts/train.py --backend autogluon --domain demand --dataset rte_france
    uv run python scripts/train.py --backend xgboost --generation gen4 --nwp-source forecast
"""

import argparse
import logging
from pathlib import Path

from windcast.config import DATASETS, get_settings
from windcast.features import list_feature_sets
from windcast.training import XGBoostBackend, run_training


def main() -> None:
    """Run training pipeline."""
    parser = argparse.ArgumentParser(description="Train forecast models with MLflow tracking")
    parser.add_argument(
        "--backend",
        choices=["xgboost", "autogluon"],
        default="xgboost",
        help="ML backend. Default: xgboost",
    )
    parser.add_argument(
        "--domain",
        choices=["wind", "demand", "solar"],
        default="wind",
        help="Domain: wind, demand, or solar. Default: wind",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Features directory. Default: data/features/",
    )
    parser.add_argument(
        "--feature-set",
        default=None,
        choices=list_feature_sets(),
        help="Feature set to use. Default: domain-specific baseline",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset ID for file lookup. Default: domain-specific",
    )
    parser.add_argument("--turbine-id", default="kwf1", help="(Wind) Turbine ID. Default: kwf1")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name. Default: enercast-{dataset}",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Forecast horizons in steps. Default: from settings",
    )
    # Split config
    parser.add_argument(
        "--train-years",
        type=int,
        default=None,
        help="Training split in years. Default: from dataset config",
    )
    parser.add_argument(
        "--val-years",
        type=int,
        default=None,
        help="Validation split in years. Default: from dataset config",
    )
    # AutoGluon-specific
    parser.add_argument(
        "--time-limit",
        type=int,
        default=120,
        help="AutoGluon time limit per horizon in seconds. Default: 120",
    )
    parser.add_argument(
        "--presets",
        default="good_quality",
        choices=["best_quality", "high_quality", "good_quality", "medium_quality"],
        help="AutoGluon presets. Default: good_quality",
    )
    # Lineage tags
    parser.add_argument("--generation", default=None, help="Generation label (e.g. gen4)")
    parser.add_argument(
        "--nwp-source",
        default="forecast",
        help="NWP source: forecast, era5, none. Default: forecast",
    )
    parser.add_argument(
        "--data-quality",
        default="CLEAN",
        help="Data quality: CLEAN or LEAKED. Default: CLEAN",
    )
    parser.add_argument("--change-reason", default=None, help="Free-text change reason")
    # Model logging
    parser.add_argument(
        "--no-log-models",
        action="store_true",
        default=False,
        help="Skip logging model artifacts to MLflow (faster for experimentation)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    features_dir = args.features_dir or settings.features_dir
    horizons = args.horizons or settings.forecast_horizons
    domain = args.domain

    domain_feature_defaults = {
        "wind": "wind_baseline",
        "demand": "demand_baseline",
        "solar": "solar_baseline",
    }
    feature_set = args.feature_set or domain_feature_defaults[domain]

    domain_dataset_defaults = {
        "wind": "kelmarsh",
        "demand": "spain_demand",
        "solar": "pvdaq_system4",
    }
    dataset = args.dataset or domain_dataset_defaults[domain]
    experiment_name = args.experiment_name or f"enercast-{dataset}"

    # Resolve split config: CLI > dataset config > global settings
    dataset_cfg = DATASETS.get(dataset)
    resolved_train_years = (
        args.train_years
        or (getattr(dataset_cfg, "train_years", None) if dataset_cfg else None)
        or settings.train_years
    )
    resolved_val_years = (
        args.val_years
        or (getattr(dataset_cfg, "val_years", None) if dataset_cfg else None)
        or settings.val_years
    )

    if domain in ("demand", "solar"):
        parquet_path = features_dir / f"{dataset}_features.parquet"
    else:
        parquet_path = features_dir / f"kelmarsh_{args.turbine_id}.parquet"

    # Build backend
    if args.backend == "autogluon":
        from windcast.models.autogluon_model import AutoGluonConfig
        from windcast.training import AutoGluonBackend

        ag_config = AutoGluonConfig(presets=args.presets, time_limit=args.time_limit)
        backend = AutoGluonBackend(config=ag_config)
    else:
        backend = XGBoostBackend()

    run_training(
        backend=backend,
        domain=domain,
        dataset=dataset,
        feature_set_name=feature_set,
        features_path=parquet_path,
        experiment_name=experiment_name,
        horizons=horizons,
        turbine_id=args.turbine_id,
        generation=args.generation,
        nwp_source=args.nwp_source,
        data_quality=args.data_quality,
        change_reason=args.change_reason,
        train_years=resolved_train_years,
        val_years=resolved_val_years,
        log_models=not args.no_log_models,
    )


if __name__ == "__main__":
    main()

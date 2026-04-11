"""Promote a trained model to champion in the MLflow Model Registry.

Decoupled from training: evaluate and compare first, then promote.

Usage:
    # Promote a specific run
    uv run python scripts/promote_model.py --run-id <RUN_ID>

    # Auto-select best run from experiment by metric
    uv run python scripts/promote_model.py --experiment enercast-rte_france --metric h24_mae

    # List available parent runs
    uv run python scripts/promote_model.py --experiment enercast-rte_france --list
"""

from __future__ import annotations

import argparse
import logging
import sys

import mlflow
import pandas as pd

from windcast.config import get_settings

logger = logging.getLogger(__name__)


def list_parent_runs(experiment_name: str) -> None:
    """Print parent runs with key metrics for comparison."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="tags.`enercast.run_type` = 'parent'",
        output_format="pandas",
        order_by=["start_time DESC"],
    )
    assert isinstance(runs, pd.DataFrame)
    if runs.empty:
        logger.error("No parent runs found in experiment '%s'", experiment_name)
        sys.exit(1)

    print(f"\nParent runs in '{experiment_name}':\n")
    print(f"{'Run ID':<36} {'Name':<45} {'Backend':<12} {'Feature Set':<20} {'h24 MAE':>10}")
    print("-" * 130)
    for _, row in runs.iterrows():
        run_id = row["run_id"]
        name = row.get("tags.mlflow.runName", "?")
        backend = row.get("tags.enercast.backend", "?")
        fs = row.get("tags.enercast.feature_set", "?")
        mae = row.get("metrics.h24_mae")
        mae_str = f"{mae:,.0f}" if mae and mae == mae else "—"
        print(f"{run_id:<36} {name:<45} {backend:<12} {fs:<20} {mae_str:>10}")
    print()


def find_best_run(experiment_name: str, metric: str) -> str:
    """Find the parent run with the lowest value for the given metric."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="tags.`enercast.run_type` = 'parent'",
        output_format="pandas",
        order_by=[f"metrics.{metric} ASC"],
    )
    assert isinstance(runs, pd.DataFrame)
    if runs.empty:
        logger.error("No parent runs found in experiment '%s'", experiment_name)
        sys.exit(1)

    # Filter to runs that have the metric
    metric_col = f"metrics.{metric}"
    if metric_col not in runs.columns:
        logger.error(
            "Metric '%s' not found. Available: %s",
            metric,
            [c for c in runs.columns if c.startswith("metrics.")],
        )
        sys.exit(1)

    valid = runs[runs[metric_col].notna()]
    if valid.empty:
        logger.error("No runs have metric '%s'", metric)
        sys.exit(1)

    best = valid.iloc[0]
    run_id = best["run_id"]
    name = best.get("tags.mlflow.runName", "?")
    val = best[metric_col]
    logger.info("Best run by %s: %s (%s) = %.1f", metric, run_id[:8], name, val)
    return run_id


def promote_run(run_id: str, model_name: str, alias: str = "champion") -> None:
    """Register the HorizonRouter from a run and set the alias."""
    client = mlflow.tracking.MlflowClient()  # pyright: ignore[reportPrivateImportUsage]

    # Verify the run exists
    try:
        run = client.get_run(run_id)
    except Exception:
        logger.error("Run '%s' not found", run_id)
        sys.exit(1)

    run_name = run.data.tags.get("mlflow.runName", "?")
    logger.info("Promoting run: %s (%s)", run_id[:8], run_name)

    # Find horizon_router LoggedModel on this run (MLflow 3.x API)
    experiment_id = run.info.experiment_id
    logged_models = client.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=f"source_run_id = '{run_id}' AND name = 'horizon_router'",
    )
    if not logged_models:
        logger.error(
            "Run '%s' has no 'horizon_router' model. "
            "Was it trained with log_models=True?",
            run_id[:8],
        )
        sys.exit(1)

    # Register using the LoggedModel URI
    model_uri = logged_models[0].model_uri
    mv = mlflow.register_model(model_uri, model_name)
    logger.info("Registered %s v%s", model_name, mv.version)

    # Set alias
    client.set_registered_model_alias(model_name, alias, str(mv.version))
    logger.info("Set alias @%s → v%s", alias, mv.version)

    # Summary
    print(f"\n✓ Promoted: {model_name}@{alias} → v{mv.version}")
    print(f"  Run: {run_id[:8]}... ({run_name})")
    print(f"  Load with: mlflow.pyfunc.load_model('models:/{model_name}@{alias}')")
    print()


def main() -> None:
    """Promote a trained model to the MLflow Model Registry."""
    parser = argparse.ArgumentParser(
        description="Promote a trained model to champion (decoupled from training)"
    )
    parser.add_argument("--run-id", default=None, help="Parent run ID to promote")
    parser.add_argument(
        "--experiment", default=None, help="Experiment name (for --list or --metric)"
    )
    parser.add_argument(
        "--metric",
        default="h24_mae",
        help="Metric to minimize when auto-selecting best run. Default: h24_mae",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Registered model name. Default: from run tags (enercast-{dataset}-{backend})",
    )
    parser.add_argument(
        "--alias",
        default="champion",
        help="Alias to assign. Default: champion",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_runs",
        help="List parent runs in the experiment (no promotion)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # List mode
    if args.list_runs:
        if not args.experiment:
            parser.error("--list requires --experiment")
        list_parent_runs(args.experiment)
        return

    # Determine run_id
    if args.run_id:
        run_id = args.run_id
    elif args.experiment:
        run_id = find_best_run(args.experiment, args.metric)
    else:
        parser.error("Provide --run-id or --experiment")
        return  # unreachable but satisfies type checker

    # Determine model name
    if args.model_name:
        model_name = args.model_name
    else:
        # Infer from run tags
        client = mlflow.tracking.MlflowClient()  # pyright: ignore[reportPrivateImportUsage]
        run = client.get_run(run_id)
        dataset = run.data.params.get("dataset", "unknown")
        backend = run.data.tags.get("enercast.backend", "xgboost")
        model_name = f"enercast-{dataset}-{backend}"
        logger.info("Inferred model name: %s", model_name)

    promote_run(run_id, model_name, args.alias)


if __name__ == "__main__":
    main()

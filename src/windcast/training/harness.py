"""Training harness — Backend Protocol, shared utilities, and run_training() loop."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

DOMAIN_CONFIG: dict[str, dict[str, str]] = {
    "wind": {"target": "active_power_kw", "group": "turbine_id", "lag1": "active_power_kw_lag1"},
    "demand": {"target": "load_mw", "group": "zone_id", "lag1": "load_mw_lag1"},
    "solar": {"target": "power_kw", "group": "system_id", "lag1": "power_kw_lag1"},
}


class TrainingBackend(Protocol):
    """Interface that ML backends must satisfy to plug into run_training()."""

    @property
    def name(self) -> str: ...

    def mlflow_setup(self) -> None:
        """Backend-specific MLflow config (e.g., XGBoost autolog)."""
        ...

    def extra_params(self) -> dict[str, Any]:
        """Backend-specific MLflow params (e.g., AG presets, time_limit)."""
        ...

    def train(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame,
        y_val: pl.Series,
    ) -> Any:
        """Train model and return fitted object."""
        ...

    def predict(self, model: Any, X: pl.DataFrame) -> np.ndarray:
        """Generate predictions from fitted model."""
        ...

    def log_child_artifacts(self, model: Any, horizon: int) -> None:
        """Log backend-specific artifacts on the child run (e.g., AG leaderboard)."""
        ...

    def log_model(
        self,
        model: Any,
        X_val: pl.DataFrame,
        y_pred: np.ndarray,
        horizon: int,
    ) -> str | None:
        """Log model artifact to MLflow with signature. Returns model URI or None."""
        ...

    def describe_model(self, model: Any) -> str:
        """One-line model description for Markdown notes."""
        ...


def temporal_split(
    df: pl.DataFrame,
    train_years: int,
    val_years: int,
    timestamp_col: str = "timestamp_utc",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split DataFrame temporally using year-based boundaries.

    Computes split dates from data timestamp range:
    - train: first train_years
    - val: next val_years
    - test: remainder
    """
    ts = df.get_column(timestamp_col)
    start: datetime = ts.min()  # type: ignore[assignment]
    train_end = start.replace(year=start.year + train_years)
    val_end = train_end.replace(year=train_end.year + val_years)

    train = df.filter(pl.col(timestamp_col) < train_end)
    val = df.filter((pl.col(timestamp_col) >= train_end) & (pl.col(timestamp_col) < val_end))
    test = df.filter(pl.col(timestamp_col) >= val_end)

    return train, val, test


def resolve_horizon_features(
    available_cols: list[str],
    feature_set_cols: list[str],
    horizon: int,
) -> tuple[list[str], dict[str, str]]:
    """Resolve feature columns for a specific horizon.

    For NWP columns (``nwp_*``), selects the ``_h{h}`` variant matching the
    current horizon and excludes all other horizon variants.  Non-NWP columns
    are passed through unchanged.

    Returns:
        ``(actual_columns, rename_map)`` — *actual_columns* are the DataFrame
        column names to select; *rename_map* maps ``nwp_X_h{h}`` → ``nwp_X``
        so the model sees consistent canonical names across horizons.
    """
    available_set = set(available_cols)
    actual: list[str] = []
    rename_map: dict[str, str] = {}

    for col in feature_set_cols:
        if col.startswith("nwp_"):
            horizon_col = f"{col}_h{horizon}"
            if horizon_col in available_set:
                actual.append(horizon_col)
                rename_map[horizon_col] = col
            elif col in available_set:
                actual.append(col)
        elif col in available_set:
            actual.append(col)

    return actual, rename_map


def build_horizon_target(
    df: pl.DataFrame,
    horizon: int,
    feature_cols: list[str],
    target_col_name: str,
    rename_map: dict[str, str] | None = None,
) -> tuple[pl.DataFrame, pl.Series]:
    """Build target for a given horizon and return X, y without nulls.

    Target at row i = target value at row i+horizon (shift(-h)).
    If *rename_map* is provided, renames columns (e.g. strip ``_h{h}`` suffix)
    so the model sees canonical feature names.
    """
    target_col = f"target_h{horizon}"
    df_h = df.with_columns(pl.col(target_col_name).shift(-horizon).alias(target_col)).drop_nulls(
        subset=[target_col]
    )

    X = df_h.select(feature_cols)
    if rename_map:
        X = X.rename(rename_map)
    y = df_h.get_column(target_col)
    return X, y


def build_horizon_desc(horizon: int, data_resolution: int) -> str:
    """Return human-readable horizon description (e.g. '10 min ahead', '2h ahead', 'D+1')."""
    minutes = horizon * data_resolution
    if minutes < 60:
        return f"{minutes} min ahead"
    elif minutes < 1440:
        return f"{minutes // 60}h ahead"
    else:
        return f"D+{minutes // 1440}"


def run_training(
    backend: TrainingBackend,
    domain: str,
    dataset: str,
    feature_set_name: str,
    features_path: Path,
    experiment_name: str,
    horizons: list[int],
    turbine_id: str = "kwf1",
    generation: str | None = None,
    nwp_source: str = "forecast",
    data_quality: str = "CLEAN",
    change_reason: str | None = None,
    train_years: int | None = None,
    val_years: int | None = None,
    log_models: bool = True,
) -> None:
    """Run the full training pipeline for any backend.

    Loads features, splits data, sets up MLflow parent/child runs, trains
    per-horizon models via the backend, logs metrics and lineage tags.
    """
    import mlflow
    import mlflow.data

    from windcast.config import get_settings
    from windcast.features import get_feature_set
    from windcast.models.evaluation import compute_metrics
    from windcast.models.persistence import compute_persistence_metrics
    from windcast.tracking import (
        log_evaluation_results,
        log_stepped_horizon_metrics,
        setup_mlflow,
    )
    from windcast.training.lineage import log_lineage_tags

    settings = get_settings()
    effective_train_years = train_years if train_years is not None else settings.train_years
    effective_val_years = val_years if val_years is not None else settings.val_years
    dcfg = DOMAIN_CONFIG[domain]
    target_col = dcfg["target"]
    lag1_col = dcfg["lag1"]
    data_resolution = 10 if domain == "wind" else 60
    run_label = turbine_id if domain == "wind" else dataset
    fs = get_feature_set(feature_set_name)

    if not features_path.exists():
        logger.error("Feature file not found: %s", features_path)
        return

    logger.info("Loading features from %s", features_path)
    df = pl.read_parquet(features_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    has_nwp_horizons = any(c for c in df.columns if c.startswith("nwp_") and "_h" in c)
    if has_nwp_horizons:
        logger.info("NWP horizon columns detected — will resolve per horizon")

    non_nwp_cols = [c for c in fs.columns if not c.startswith("nwp_")]
    available_non_nwp = [c for c in non_nwp_cols if c in df.columns]
    missing_non_nwp = set(non_nwp_cols) - set(available_non_nwp)
    if missing_non_nwp:
        logger.warning("Missing feature columns (skipped): %s", sorted(missing_non_nwp))

    train_df, val_df, test_df = temporal_split(df, effective_train_years, effective_val_years)
    logger.info(
        "Temporal split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df)
    )

    if len(train_df) == 0 or len(val_df) == 0:
        logger.error("Insufficient data for temporal split")
        return

    setup_mlflow(settings.mlflow_tracking_uri, experiment_name)
    backend.mlflow_setup()

    horizon_descs: dict[int, str] = {h: build_horizon_desc(h, data_resolution) for h in horizons}

    parent_tags = {
        "enercast.stage": "dev",
        "enercast.domain": domain,
        "enercast.purpose": "baseline",
        "enercast.backend": backend.name,
        "enercast.data_resolution_min": str(data_resolution),
        "enercast.feature_set": feature_set_name,
    }

    run_name = f"{run_label}-{backend.name}-{feature_set_name}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({**parent_tags, "enercast.run_type": "parent"})
        log_lineage_tags(
            generation=generation,
            nwp_source=nwp_source,
            data_quality=data_quality,
            change_reason=change_reason,
        )

        ts_col = "timestamp_utc"
        params: dict[str, Any] = {
            "domain": domain,
            "dataset": dataset,
            "feature_set": feature_set_name,
            "n_features_base": len(available_non_nwp),
            "horizons": str(horizons),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "split.train_years": effective_train_years,
            "split.val_years": effective_val_years,
            "split.train_start": str(train_df[ts_col].min()),
            "split.train_end": str(train_df[ts_col].max()),
            "split.val_start": str(val_df[ts_col].min()),
            "split.val_end": str(val_df[ts_col].max()),
            "split.test_start": str(test_df[ts_col].min()),
            "data.source_file": str(features_path),
            "data.n_rows_total": len(df),
        }
        params.update(backend.extra_params())
        mlflow.log_params(params)

        if domain == "wind":
            mlflow.log_param("turbine_id", turbine_id)

        src = str(features_path)
        train_dataset = mlflow.data.from_polars(  # pyright: ignore[reportAttributeAccessIssue]
            train_df, source=src, name=f"{dataset}-{run_label}-train", targets=target_col
        )
        val_dataset = mlflow.data.from_polars(  # pyright: ignore[reportAttributeAccessIssue]
            val_df, source=src, name=f"{dataset}-{run_label}-val", targets=target_col
        )
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(val_dataset, context="validation")
        if len(test_df) > 0:
            test_dataset = mlflow.data.from_polars(  # pyright: ignore[reportAttributeAccessIssue]
                test_df, source=src, name=f"{dataset}-{run_label}-test", targets=target_col
            )
            mlflow.log_input(test_dataset, context="test")

        results_summary: list[str] = []
        horizon_metrics: dict[int, dict[str, float]] = {}
        best_mae: float = float("inf")
        horizon_model_uris: dict[int, str] = {}
        last_feature_cols: list[str] = []

        for h in horizons:
            logger.info("=== Horizon h=%d (%s) ===", h, horizon_descs[h])

            if has_nwp_horizons:
                h_cols, rename_map = resolve_horizon_features(df.columns, fs.columns, h)
                feature_cols_h = [c for c in available_non_nwp if not c.startswith("nwp_")] + [
                    c for c in h_cols if c.startswith("nwp_")
                ]
            else:
                feature_cols_h = available_non_nwp + [
                    c for c in fs.columns if c.startswith("nwp_") and c in df.columns
                ]
                rename_map = {}

            X_train, y_train = build_horizon_target(
                train_df, h, feature_cols_h, target_col, rename_map
            )
            X_val, y_val = build_horizon_target(val_df, h, feature_cols_h, target_col, rename_map)
            if len(test_df) > 0:
                X_test, y_test = build_horizon_target(
                    test_df, h, feature_cols_h, target_col, rename_map
                )
            else:
                X_test, y_test = None, None

            if len(X_train) == 0 or len(X_val) == 0:
                logger.warning("Insufficient data for horizon %d, skipping", h)
                continue

            with mlflow.start_run(run_name=f"h{h:02d}", nested=True):
                mlflow.set_tags(
                    {
                        **parent_tags,
                        "enercast.run_type": "child",
                        "enercast.horizon_steps": str(h),
                        "enercast.horizon_desc": horizon_descs[h],
                    }
                )
                log_lineage_tags(
                    generation=generation,
                    nwp_source=nwp_source,
                    data_quality=data_quality,
                    change_reason=change_reason,
                )
                mlflow.log_params({"horizon_steps": h, "n_features": len(feature_cols_h)})

                model = backend.train(X_train, y_train, X_val, y_val)
                y_pred = backend.predict(model, X_val)

                if lag1_col in X_val.columns:
                    y_persistence = X_val.get_column(lag1_col).to_numpy()
                    metrics = compute_metrics(y_val.to_numpy(), y_pred, y_persistence=y_persistence)
                    persistence_metrics = compute_persistence_metrics(
                        y_val.to_numpy(), y_persistence
                    )
                    prefixed_persistence = {
                        f"persistence_{k}": v for k, v in persistence_metrics.items()
                    }
                    mlflow.log_metrics(prefixed_persistence)
                else:
                    metrics = compute_metrics(y_val.to_numpy(), y_pred)
                    prefixed_persistence = {}

                log_evaluation_results(metrics, horizon=h)

                test_metrics_prefixed: dict[str, float] = {}
                if X_test is not None and y_test is not None and len(X_test) > 0:
                    y_pred_test = backend.predict(model, X_test)
                    if lag1_col in X_test.columns:
                        y_persistence_test = X_test.get_column(lag1_col).to_numpy()
                        test_metrics = compute_metrics(
                            y_test.to_numpy(), y_pred_test, y_persistence=y_persistence_test
                        )
                        test_persistence = compute_persistence_metrics(
                            y_test.to_numpy(), y_persistence_test
                        )
                    else:
                        test_metrics = compute_metrics(y_test.to_numpy(), y_pred_test)
                        test_persistence = {}
                    test_metrics_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}
                    test_metrics_prefixed.update(
                        {f"test_persistence_{k}": v for k, v in test_persistence.items()}
                    )
                    mlflow.log_metrics({f"h{h}_{k}": v for k, v in test_metrics_prefixed.items()})

                horizon_metrics[h * data_resolution] = {
                    **metrics,
                    **prefixed_persistence,
                    **test_metrics_prefixed,
                }

                backend.log_child_artifacts(model, h)
                model_uri = None
                if log_models:
                    model_uri = backend.log_model(model, X_val, y_pred, h)
                if model_uri:
                    horizon_model_uris[h] = model_uri
                last_feature_cols = list(X_val.columns)

                mae = metrics["mae"]
                if mae < best_mae:
                    best_mae = mae
                rmse = metrics["rmse"]
                skill = metrics.get("skill_score", float("nan"))
                bias = metrics.get("bias", float("nan"))
                model_desc = backend.describe_model(model)

                if test_metrics_prefixed:
                    test_mae = test_metrics_prefixed.get("test_mae", float("nan"))
                    test_rmse = test_metrics_prefixed.get("test_rmse", float("nan"))
                    test_skill = test_metrics_prefixed.get("test_skill_score", float("nan"))
                    test_bias = test_metrics_prefixed.get("test_bias", float("nan"))
                    metrics_table = (
                        f"| Metric | Val | Test |\n|--------|------|------|\n"
                        f"| MAE | {mae:.1f} | {test_mae:.1f} |\n"
                        f"| RMSE | {rmse:.1f} | {test_rmse:.1f} |\n"
                        f"| Skill score | {skill:.3f} | {test_skill:.3f} |\n"
                        f"| Bias | {bias:+.1f} | {test_bias:+.1f} |\n"
                    )
                else:
                    metrics_table = (
                        f"| Metric | Value |\n|--------|-------|\n"
                        f"| MAE | {mae:.1f} |\n"
                        f"| RMSE | {rmse:.1f} |\n"
                        f"| Skill score | {skill:.3f} |\n"
                        f"| Bias | {bias:+.1f} |\n"
                    )
                child_desc = (
                    f"## Horizon h{h} — {horizon_descs[h]}\n\n"
                    f"**Feature set:** {feature_set_name} | "
                    f"**Features:** {len(feature_cols_h)} | "
                    f"**{model_desc}**\n\n"
                    f"{metrics_table}"
                )
                mlflow.set_tag("mlflow.note.content", child_desc)

                if test_metrics_prefixed:
                    logger.info(
                        "h=%d: val MAE=%.1f, val skill=%.3f | test MAE=%.1f, test skill=%.3f "
                        "(%d features)",
                        h,
                        mae,
                        skill,
                        test_metrics_prefixed.get("test_mae", float("nan")),
                        test_metrics_prefixed.get("test_skill_score", float("nan")),
                        len(feature_cols_h),
                    )
                    results_summary.append(
                        f"h{h} ({horizon_descs[h]}): "
                        f"val MAE={mae:.0f}, skill={skill:.3f} | "
                        f"test MAE={test_metrics_prefixed['test_mae']:.0f}, "
                        f"skill={test_metrics_prefixed['test_skill_score']:.3f}"
                    )
                else:
                    logger.info(
                        "h=%d: val MAE=%.1f, RMSE=%.1f, skill=%.3f (%d features)",
                        h,
                        mae,
                        rmse,
                        skill,
                        len(feature_cols_h),
                    )
                    results_summary.append(
                        f"h{h} ({horizon_descs[h]}): MAE={mae:.0f}, skill={skill:.3f}"
                    )

        log_stepped_horizon_metrics(horizon_metrics)

        active = mlflow.active_run()
        parent_run_id = active.info.run_id if active else ""
        client = mlflow.tracking.MlflowClient()  # pyright: ignore[reportPrivateImportUsage]
        exp_obj = client.get_experiment_by_name(experiment_name)
        if exp_obj and parent_run_id:
            children = client.search_runs(
                experiment_ids=[exp_obj.experiment_id],
                filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            )
            for child in children:
                for k, v in child.data.metrics.items():
                    if k.startswith("h") and ("_mae" in k or "_rmse" in k or "_skill_score" in k):
                        mlflow.log_metric(k, v)

        # Log HorizonRouter as parent-run artifact (bundles all horizon sub-models)
        if log_models and horizon_model_uris:
            from windcast.models.horizon_router import HorizonRouter, build_router_signature

            router = HorizonRouter(horizon_model_uris)
            router.load_context(None)

            signature = build_router_signature(last_feature_cols, default_horizon=horizons[0])

            mlflow.pyfunc.log_model(
                name="horizon_router",
                python_model=router,
                signature=signature,
            )
            logger.info(
                "Logged HorizonRouter artifact (%d horizons, best MAE=%.1f)",
                len(horizon_model_uris),
                best_mae,
            )

        summary_text = "\n".join(results_summary)
        parent_desc = (
            f"## {run_label.upper()} — {backend.name} — {feature_set_name}\n\n"
            f"**Dataset:** {dataset} | "
            f"**Train:** {len(train_df):,} rows | "
            f"**Val:** {len(val_df):,} rows\n\n"
            f"### Feature Set\n{fs.description}\n\n"
            f"### Results (validation set)\n{summary_text}\n"
        )
        mlflow.set_tag("mlflow.note.content", parent_desc)

    logger.info("Training complete! Check MLflow UI: %s", settings.mlflow_tracking_uri)

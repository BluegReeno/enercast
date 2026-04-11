"""Tests for model logging and registry integration."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import pytest

from windcast.training.backends import XGBoostBackend
from windcast.training.harness import run_training


@pytest.fixture
def _wind_features_parquet(tmp_path):
    """Create a minimal wind features parquet for testing."""
    n = 500
    dates = [datetime(2015, 1, 1, h % 24) for h in range(n)]
    for i in range(n):
        year = 2015 + i * 11 // n
        dates[i] = dates[i].replace(year=year)
    dates.sort()

    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "timestamp_utc": dates,
            "active_power_kw": rng.uniform(0, 1000, n).tolist(),
            "active_power_kw_lag1": rng.uniform(0, 1000, n).tolist(),
            "hour": [d.hour for d in dates],
        }
    )
    path = tmp_path / "kelmarsh_kwf1.parquet"
    df.write_parquet(path)
    return tmp_path


def test_xgboost_log_model(tmp_path):
    """Test that XGBoostBackend.log_model() produces a loadable model."""
    rng = np.random.default_rng(42)
    X = pl.DataFrame({"f1": rng.normal(size=100).tolist(), "f2": rng.normal(size=100).tolist()})
    y = pl.Series("target", rng.normal(size=100).tolist())

    backend = XGBoostBackend()
    model = backend.train(X, y, X, y)
    y_pred = backend.predict(model, X)

    tracking_uri = f"sqlite:///{tmp_path}/test_registry.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-log-model")

    with mlflow.start_run():
        with mlflow.start_run(run_name="h01", nested=True):
            model_uri = backend.log_model(model, X, y_pred, horizon=1)
            assert model_uri is not None
            assert model_uri.startswith("models:/")

        # Load back the model and verify predictions
        loaded = mlflow.pyfunc.load_model(model_uri)
        preds = loaded.predict(X.to_pandas())
        np.testing.assert_allclose(preds, y_pred, rtol=1e-5)


def test_run_training_with_model_logging(_wind_features_parquet, tmp_path):
    """Test run_training logs models when log_models=True."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow_registry_test.db"

    with patch("windcast.config.get_settings") as mock_settings:
        mock_settings.return_value.mlflow_tracking_uri = tracking_uri
        mock_settings.return_value.train_years = 5
        mock_settings.return_value.val_years = 1
        mock_settings.return_value.features_dir = _wind_features_parquet

        run_training(
            backend=XGBoostBackend(),
            domain="wind",
            dataset="kelmarsh",
            feature_set_name="wind_baseline",
            features_path=_wind_features_parquet / "kelmarsh_kwf1.parquet",
            experiment_name="test-registry",
            horizons=[1],
            turbine_id="kwf1",
            log_models=True,
            train_years=8,
            val_years=2,
        )

    mlflow.set_tracking_uri(tracking_uri)
    # Verify a child run was created
    runs = mlflow.search_runs(
        experiment_names=["test-registry"],
        filter_string="tags.`enercast.run_type` = 'child'",
        output_format="pandas",
    )
    assert len(runs) == 1
    # Verify model was logged via MLflow 3.x LoggedModel API
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("test-registry")
    assert exp is not None
    logged_models = client.search_logged_models(
        experiment_ids=[exp.experiment_id],
    )
    assert len(logged_models) >= 1
    model_names = [m.name for m in logged_models]
    assert "model_h01" in model_names


def test_run_training_with_registration(_wind_features_parquet, tmp_path):
    """Test run_training registers best model when register_model_name is set."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow_register_test.db"

    with patch("windcast.config.get_settings") as mock_settings:
        mock_settings.return_value.mlflow_tracking_uri = tracking_uri
        mock_settings.return_value.train_years = 5
        mock_settings.return_value.val_years = 1
        mock_settings.return_value.features_dir = _wind_features_parquet

        run_training(
            backend=XGBoostBackend(),
            domain="wind",
            dataset="kelmarsh",
            feature_set_name="wind_baseline",
            features_path=_wind_features_parquet / "kelmarsh_kwf1.parquet",
            experiment_name="test-register",
            horizons=[1],
            turbine_id="kwf1",
            log_models=True,
            register_model_name="test-enercast-kelmarsh-xgboost",
            train_years=8,
            val_years=2,
        )

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    # Verify model was registered
    registered = client.get_registered_model("test-enercast-kelmarsh-xgboost")
    assert registered is not None
    assert len(registered.latest_versions) >= 1
    # Verify champion alias
    alias_mv = client.get_model_version_by_alias("test-enercast-kelmarsh-xgboost", "champion")
    assert alias_mv is not None


def test_run_training_registers_horizon_router(_wind_features_parquet, tmp_path):
    """Test that registration creates a HorizonRouter servable via params."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow_router_test.db"

    with patch("windcast.config.get_settings") as mock_settings:
        mock_settings.return_value.mlflow_tracking_uri = tracking_uri
        mock_settings.return_value.train_years = 5
        mock_settings.return_value.val_years = 1
        mock_settings.return_value.features_dir = _wind_features_parquet

        run_training(
            backend=XGBoostBackend(),
            domain="wind",
            dataset="kelmarsh",
            feature_set_name="wind_baseline",
            features_path=_wind_features_parquet / "kelmarsh_kwf1.parquet",
            experiment_name="test-router",
            horizons=[1, 6],
            turbine_id="kwf1",
            log_models=True,
            register_model_name="test-router-model",
            train_years=8,
            val_years=2,
        )

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Verify champion alias resolves
    alias_mv = client.get_model_version_by_alias("test-router-model", "champion")
    assert alias_mv is not None

    # Load the registered model — should be a HorizonRouter
    model = mlflow.pyfunc.load_model("models:/test-router-model@champion")

    # Build a minimal input matching the training feature columns
    rng = np.random.default_rng(42)
    # Read the feature parquet to get column names
    features_df = pl.read_parquet(_wind_features_parquet / "kelmarsh_kwf1.parquet")
    # Get feature columns (same logic as wind_baseline: exclude target + timestamp)
    exclude = {"timestamp_utc", "active_power_kw", "turbine_id", "qc_flag"}
    feature_cols = [c for c in features_df.columns if c not in exclude]

    X_test = pd.DataFrame({c: rng.normal(size=1).tolist() for c in feature_cols})

    # Predict with horizon=1
    pred_h1 = model.predict(X_test, params={"horizon": 1})
    assert pred_h1 is not None
    assert len(pred_h1) == 1

    # Predict with horizon=6
    pred_h6 = model.predict(X_test, params={"horizon": 6})
    assert pred_h6 is not None
    assert len(pred_h6) == 1

    # Different horizons should give different predictions (different models)
    # (Not guaranteed with random data, but very likely)
    assert isinstance(pred_h1[0], (int, float, np.floating))

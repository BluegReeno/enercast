"""EnerCast — Forecast Dashboard.

Live demo frontend for the WeatherNews presentation.
Generates real-time forecasts using trained models and fresh NWP data.

Usage:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import requests
import streamlit as st

# Add project root to path so `scripts.inference` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from windcast.config import DOMAIN_RESOLUTION, get_settings
from windcast.training.harness import build_horizon_desc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENTS_BY_DOMAIN: dict[str, str] = {
    "wind": "enercast-kelmarsh",
    "demand": "enercast-rte_france",
}
MODEL_NAME_BY_DOMAIN: dict[str, str] = {
    "wind": "enercast-kelmarsh-xgboost",
    "demand": "enercast-rte_france-xgboost",
}
DOMAIN_DEFAULTS: dict[str, str] = {"wind": "kelmarsh", "demand": "rte_france"}
UNIT_BY_DOMAIN: dict[str, str] = {"wind": "kW", "demand": "MW"}
WEATHER_CONFIG_MAP: dict[str, str] = {"kelmarsh": "kelmarsh", "rte_france": "rte_france"}
DEFAULT_HORIZONS: list[int] = [1, 6, 12, 24, 48]
SERVE_URL: str | None = os.environ.get("ENERCAST_SERVE_URL")

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EnerCast — Forecast Dashboard",
    page_icon="🌬️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached data fetchers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_parent_runs(experiment_name: str):
    """Query MLflow for parent runs in the given experiment."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    df = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="tags.`enercast.run_type` = 'parent'",
        output_format="pandas",
        order_by=["start_time DESC"],
    )
    return df


@st.cache_resource
def load_champion_model(model_name: str) -> Any:
    """Load the champion HorizonRouter model from the registry — cached."""
    from mlflow.exceptions import MlflowException

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        return mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
    except MlflowException:
        return None


@st.cache_resource
def load_models_for_run(parent_run_id: str) -> dict[int, object]:
    """Load all horizon models from a parent run's children — cached as singleton."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    parent = client.get_run(parent_run_id)
    children = client.search_runs(
        experiment_ids=[parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )

    models: dict[int, object] = {}
    for child in children:
        horizon = int(child.data.params.get("horizon_steps", 0))
        if horizon == 0:
            continue
        uri = f"runs:/{child.info.run_id}/model_h{horizon:02d}"
        models[horizon] = mlflow.pyfunc.load_model(uri)

    return models


def predict_via_server(serve_url: str, X_pd: pd.DataFrame, horizon: int) -> float:
    """POST to MLflow serving endpoint with params routing."""
    payload = {
        "dataframe_split": {
            "columns": X_pd.columns.tolist(),
            "data": X_pd.values.tolist(),
        },
        "params": {"horizon": horizon},
    }
    resp = requests.post(f"{serve_url.rstrip('/')}/invocations", json=payload)
    resp.raise_for_status()
    result = resp.json()
    if isinstance(result, dict) and "predictions" in result:
        return float(result["predictions"][0])
    if isinstance(result, list):
        return float(result[0])
    return float(result)


# ---------------------------------------------------------------------------
# Forecast runner
# ---------------------------------------------------------------------------


def run_forecast(
    domain: str,
    dataset: str,
    feature_set: str,
    *,
    router: Any | None = None,
    models: dict[int, Any] | None = None,
    horizons: list[int] | None = None,
    serve_url: str | None = None,
) -> list[dict]:
    """Run inference across all horizons. Returns list of result dicts.

    Three modes:
    - **serve_url**: POST to MLflow serving endpoint (HorizonRouter via REST)
    - **router**: in-process HorizonRouter, route via params={"horizon": h}
    - **models**: legacy per-horizon dict, one model per horizon
    """
    from scripts.inference import build_inference_features
    from windcast.weather import get_live_forecast

    settings = get_settings()
    resolution = DOMAIN_RESOLUTION[domain]
    weather_name = WEATHER_CONFIG_MAP[dataset]

    # Load actuals (tail for lags)
    if domain == "wind":
        actuals_path = settings.processed_dir / "kelmarsh_kwf1.parquet"
        tail_rows = 50
    else:
        actuals_path = settings.processed_dir / f"{dataset}.parquet"
        tail_rows = 200

    actuals = pl.read_parquet(actuals_path).sort("timestamp_utc").tail(tail_rows)
    last_ts = actuals["timestamp_utc"].max()
    if last_ts is None:
        raise ValueError(f"Empty actuals file: {actuals_path}")

    # Fetch live NWP
    nwp = get_live_forecast(weather_name, forecast_days=3, past_days=2)

    # Determine which horizons to iterate
    if models is not None:
        iter_horizons = sorted(models.keys())
    elif horizons is not None:
        iter_horizons = horizons
    else:
        iter_horizons = DEFAULT_HORIZONS

    results = []
    for h in iter_horizons:
        features = build_inference_features(
            actuals_df=actuals,
            nwp_df=nwp,
            domain=domain,
            feature_set_name=feature_set,
            horizon=h,
            resolution_minutes=resolution,
        )

        features_pd = features.to_pandas()

        if serve_url:
            prediction = predict_via_server(serve_url, features_pd, h)
        elif router is not None:
            prediction = float(router.predict(features_pd, params={"horizon": h})[0])
        elif models is not None and h in models:
            prediction = float(models[h].predict(features_pd)[0])
        else:
            continue

        offset_minutes = h * resolution
        target_ts = last_ts + dt.timedelta(minutes=offset_minutes)

        results.append(
            {
                "horizon": h,
                "horizon_desc": build_horizon_desc(h, resolution),
                "minutes_ahead": offset_minutes,
                "target_timestamp": target_ts,
                "prediction": prediction,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Plotly chart builder
# ---------------------------------------------------------------------------


def make_forecast_chart(
    results: list[dict],
    unit: str,
    domain: str,
) -> go.Figure:
    """Build a Plotly line chart: prediction vs forecast horizon."""
    fig = go.Figure()

    timestamps = [r["target_timestamp"] for r in results]
    predictions = [r["prediction"] for r in results]

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=predictions,
            mode="lines+markers+text",
            name="Forecast",
            line={"color": "#2ca02c", "width": 2},
            marker={"size": 10},
            text=[f"{p:,.0f}" for p in predictions],
            textposition="top center",
            textfont={"size": 10},
        )
    )

    target_label = "Active Power" if domain == "wind" else "Load"
    fig.update_layout(
        xaxis_title="Target Time (UTC)",
        yaxis_title=f"{target_label} ({unit})",
        hovermode="x unified",
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        height=400,
    )

    return fig


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("EnerCast")
    st.caption("Forecast Dashboard")

    # Domain selector
    domain = st.selectbox("Domain", list(EXPERIMENTS_BY_DOMAIN.keys()), format_func=str.title)
    assert isinstance(domain, str)
    dataset = DOMAIN_DEFAULTS[domain]
    experiment = EXPERIMENTS_BY_DOMAIN[domain]
    unit = UNIT_BY_DOMAIN[domain]
    model_name = MODEL_NAME_BY_DOMAIN[domain]

    # Mode selector
    mode = st.radio("Model source", ["Champion", "Run"], horizontal=True)

    router_model: Any = None
    run_models: dict[int, Any] | None = None
    feature_set = f"{domain}_full"
    runs_df: Any = pd.DataFrame()
    selected_name: str | None = None

    if mode == "Champion":
        router_model = load_champion_model(model_name)
        if router_model is None:
            st.warning(f"No champion model registered as '{model_name}'. Falling back to run mode.")
            mode = "Run"
        else:
            st.caption(f"**Model:** `{model_name}@champion`")
            if SERVE_URL:
                st.caption(f"**Serve URL:** `{SERVE_URL}`")

    if mode == "Run":
        # Load runs for this experiment
        runs_df = load_parent_runs(experiment)

        if runs_df.empty:
            st.error(f"No trained models found in experiment '{experiment}'")
            st.stop()

        # Model run selector
        name_col = "tags.mlflow.runName"
        run_names = runs_df[name_col].dropna().tolist()
        selected_name = st.selectbox("Model run", run_names)

        # Get selected run row
        selected_run = runs_df[runs_df[name_col] == selected_name].iloc[0]
        run_id = selected_run["run_id"]

        # Feature set (from run tags)
        feature_set = selected_run.get("tags.enercast.feature_set", f"{domain}_full")

        # Pre-load all horizon models for this run
        run_models = load_models_for_run(run_id)

        if not run_models:
            st.error(f"No horizon models found for run '{selected_name}'")
            st.stop()

        st.caption(f"**Horizons loaded:** {', '.join(f'h{h}' for h in sorted(run_models))}")

    # Generate button
    run_btn = st.button("Generate Forecast", type="primary", use_container_width=True)

    # Model metadata
    st.divider()
    st.subheader("Model Info")
    if mode == "Run" and not runs_df.empty and selected_name:
        selected_run = runs_df[runs_df["tags.mlflow.runName"] == selected_name].iloc[0]
        backend_tag = selected_run.get("tags.enercast.backend", "—")
        st.caption(f"**Backend:** {backend_tag}")
        st.caption(f"**Feature set:** {feature_set}")
        trained_date = selected_run["start_time"]
        if hasattr(trained_date, "strftime"):
            st.caption(f"**Trained:** {trained_date.strftime('%Y-%m-%d %H:%M')}")

        # Show validation MAE per horizon if available
        for h in DEFAULT_HORIZONS:
            mae_col = f"metrics.h{h}_mae"
            if mae_col in runs_df.columns:
                mae_val = selected_run.get(mae_col)
                if pd.notna(mae_val):
                    st.caption(f"Val MAE h{h}: **{mae_val:,.0f} {unit}**")
    else:
        st.caption(f"**Feature set:** {feature_set}")
        st.caption("**Mode:** Champion (registry)")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.header("Forecast Results")

# Handle button click — store results in session state
if run_btn:
    with st.spinner("Fetching NWP and running inference..."):
        try:
            results = run_forecast(
                domain=domain,
                dataset=dataset,
                feature_set=feature_set,
                router=router_model,
                models=run_models,
                serve_url=SERVE_URL if mode == "Champion" else None,
            )
            st.session_state["forecast_results"] = results
            st.session_state["forecast_domain"] = domain
            st.session_state["forecast_unit"] = unit
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.session_state.pop("forecast_results", None)

# Display results from session state (persists across reruns)
if "forecast_results" in st.session_state:
    results = st.session_state["forecast_results"]
    domain_display = st.session_state["forecast_domain"]
    unit_display = st.session_state["forecast_unit"]

    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Horizons", f"{len(results)} points")
    col2.metric("Domain", domain_display.title())
    col3.metric("Unit", unit_display)

    # Chart
    fig = make_forecast_chart(results, unit_display, domain_display)
    st.plotly_chart(fig, use_container_width=True)

    # Predictions table
    st.subheader("Predictions")
    table_data = [
        {
            "Horizon": r["horizon_desc"],
            "Target Time (UTC)": r["target_timestamp"].strftime("%Y-%m-%d %H:%M"),
            f"Prediction ({unit_display})": f"{r['prediction']:,.1f}",
        }
        for r in results
    ]
    st.table(table_data)

else:
    st.info("Select a model and horizons in the sidebar, then click **Generate Forecast**.")

"""EnerCast — Forecast Dashboard.

Live demo frontend for the WeatherNews presentation.
Generates real-time forecasts using trained models and fresh NWP data.

Usage:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import datetime as dt
import logging
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

logger = logging.getLogger(__name__)

# Add project root to path so `scripts.inference` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from windcast.config import DOMAIN_RESOLUTION, get_settings
from windcast.training.harness import build_horizon_desc

# ---------------------------------------------------------------------------
# Constants (physical/code-level, not MLflow state)
# ---------------------------------------------------------------------------

UNIT_BY_DOMAIN: dict[str, str] = {"wind": "kW", "demand": "MW", "solar": "kW"}
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


@st.cache_data(ttl=60)
def discover_available_models() -> list[dict]:
    """Discover domains with promoted champion models from MLflow registry.

    Queries all registered models, finds those with a @champion alias,
    and extracts domain/dataset metadata from the source run's tags.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    result: list[dict] = []
    for rm in client.search_registered_models():
        # Use get_model_version_by_alias (search_model_versions doesn't
        # populate aliases correctly with the SQLite backend in MLflow 3.x).
        try:
            champion_version = client.get_model_version_by_alias(rm.name, "champion")
        except mlflow.exceptions.MlflowException:
            continue

        # Read domain/dataset from the source run's tags.
        # enercast.dataset may not exist — infer from model name convention
        # "enercast-{dataset}-{backend}" or from experiment name "enercast-{dataset}".
        run = client.get_run(champion_version.run_id)
        domain = run.data.tags.get("enercast.domain", "")
        dataset = run.data.tags.get("enercast.dataset", "")
        feature_set = run.data.tags.get("enercast.feature_set", "")

        if not dataset:
            # Infer from model name: "enercast-rte_france-xgboost" → "rte_france"
            parts = rm.name.split("-")
            if len(parts) >= 3 and parts[0] == "enercast":
                dataset = "-".join(parts[1:-1])

        if not domain or not dataset:
            continue

        # Extract validation metrics from run
        metrics: dict[str, float] = {}
        for h in DEFAULT_HORIZONS:
            mae = run.data.metrics.get(f"h{h}_mae")
            skill = run.data.metrics.get(f"h{h}_skill_score")
            if mae is not None:
                metrics[f"h{h}_mae"] = mae
            if skill is not None:
                metrics[f"h{h}_skill"] = skill

        result.append(
            {
                "model_name": rm.name,
                "domain": domain,
                "dataset": dataset,
                "experiment": f"enercast-{dataset}",
                "feature_set": feature_set or f"{domain}_full",
                "backend": run.data.tags.get("enercast.backend", "unknown"),
                "trained": run.info.start_time,
                "metrics": metrics,
            }
        )

    return result


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


@st.cache_data(ttl=300)
def fetch_live_actuals(hours: int = 72) -> pl.DataFrame | None:
    """Fetch recent observed load from RTE API. Returns None if unavailable."""
    settings = get_settings()
    if not settings.rte_client_id or not settings.rte_client_secret:
        return None
    try:
        from windcast.data.rte_api import get_live_actuals

        return get_live_actuals(settings.rte_client_id, settings.rte_client_secret, hours=hours)
    except Exception as e:
        logger.warning("Could not fetch live actuals: %s", e)
        return None


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
    serve_url: str | None = None,
) -> list[dict]:
    """Run inference across all horizons using the champion HorizonRouter.

    Two modes:
    - **serve_url**: POST to MLflow serving endpoint (HorizonRouter via REST)
    - **router**: in-process HorizonRouter, route via params={"horizon": h}
    """
    from scripts.inference import build_inference_features
    from windcast.weather import get_live_forecast

    settings = get_settings()
    resolution = DOMAIN_RESOLUTION[domain]
    weather_name = WEATHER_CONFIG_MAP.get(dataset, dataset)

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

    results = []
    for h in DEFAULT_HORIZONS:
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
    actuals_df: pl.DataFrame | None = None,
) -> go.Figure:
    """Build a Plotly line chart: prediction vs forecast horizon.

    When *actuals_df* is provided (demand domain), a blue "Observed" trace is
    drawn before the green forecast trace, with a vertical dashed separator.
    """
    fig = go.Figure()

    # Actuals trace (observed load — demand domain only)
    if actuals_df is not None and not actuals_df.is_empty():
        fig.add_trace(
            go.Scatter(
                x=actuals_df["timestamp_utc"].to_list(),
                y=actuals_df["load_mw"].to_list(),
                mode="lines",
                name="Observed",
                line={"color": "#1f77b4", "width": 2},
            )
        )
        # Vertical separator at last observed timestamp
        last_obs = actuals_df["timestamp_utc"].max()
        if last_obs is not None:
            fig.add_vline(x=last_obs, line_dash="dash", line_color="gray", opacity=0.6)

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

    # Discover available champion models from MLflow registry
    available_models = discover_available_models()

    if not available_models:
        st.error("No champion models found in MLflow registry. Promote a model first.")
        st.code(
            "uv run python scripts/promote_model.py --experiment enercast-kelmarsh --metric h24_mae"
        )
        st.stop()

    # Build display labels: "Domain — Dataset (backend)"
    model_labels = [
        f"{m['domain'].title()} — {m['dataset']} ({m['backend']})" for m in available_models
    ]
    selected_idx = st.selectbox(
        "Model",
        range(len(model_labels)),
        format_func=lambda i: model_labels[i],
    )
    assert isinstance(selected_idx, int)
    selected_model = available_models[selected_idx]

    domain = selected_model["domain"]
    dataset = selected_model["dataset"]
    model_name = selected_model["model_name"]
    feature_set = selected_model["feature_set"]
    unit = UNIT_BY_DOMAIN.get(domain, "?")

    # Load champion model
    router_model = load_champion_model(model_name)
    if router_model is None:
        st.error(f"Failed to load champion model: {model_name}")
        st.stop()

    st.caption(f"**Model:** `{model_name}@champion`")
    if SERVE_URL:
        st.caption(f"**Serve URL:** `{SERVE_URL}`")

    # Generate button
    run_btn = st.button("Generate Forecast", type="primary", use_container_width=True)

    # Model metadata
    st.divider()
    st.subheader("Model Info")
    st.caption(f"**Backend:** {selected_model['backend']}")
    st.caption(f"**Feature set:** {feature_set}")
    trained_ts = selected_model.get("trained")
    if trained_ts:
        trained_dt = dt.datetime.fromtimestamp(trained_ts / 1000, tz=dt.UTC)
        st.caption(f"**Trained:** {trained_dt.strftime('%Y-%m-%d %H:%M')}")

    # Show validation MAE per horizon
    metrics = selected_model.get("metrics", {})
    for h in DEFAULT_HORIZONS:
        mae = metrics.get(f"h{h}_mae")
        if mae is not None:
            st.caption(f"Val MAE h{h}: **{mae:,.0f} {unit}**")

    # Live actuals availability indicator
    settings = get_settings()
    if settings.rte_client_id and domain == "demand":
        st.caption("**Live actuals:** Available (RTE API)")
    elif domain == "demand":
        st.caption("**Live actuals:** Not configured (set WINDCAST_RTE_CLIENT_ID)")

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
                serve_url=SERVE_URL,
            )
            st.session_state["forecast_results"] = results
            st.session_state["forecast_domain"] = domain
            st.session_state["forecast_unit"] = unit

            # Fetch live actuals for demand domain
            if domain == "demand":
                st.session_state["forecast_actuals"] = fetch_live_actuals()
            else:
                st.session_state["forecast_actuals"] = None
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            logger.exception("Forecast failed")
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
    actuals_df = st.session_state.get("forecast_actuals") if domain_display == "demand" else None
    fig = make_forecast_chart(results, unit_display, domain_display, actuals_df=actuals_df)
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
    st.info("Select a model in the sidebar, then click **Generate Forecast**.")

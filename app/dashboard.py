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
def fetch_actuals_for_timeline(days: int = 15) -> pl.DataFrame | None:
    """Fetch actuals for the timeline view — longer range than fetch_live_actuals."""
    settings = get_settings()
    if not settings.rte_client_id or not settings.rte_client_secret:
        return None
    try:
        from windcast.data.rte_api import RTEClient, fetch_load_history

        client = RTEClient(settings.rte_client_id, settings.rte_client_secret)
        end = dt.datetime.now(dt.UTC)
        start = end - dt.timedelta(days=days)
        return fetch_load_history(client, start, end, types="REALISED,D-1")
    except Exception as e:
        logger.warning("Could not fetch timeline actuals: %s", e)
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
# Forecast runner (wind + forward-looking demand)
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
# Demand timeline: backfill gaps + forward forecast
# ---------------------------------------------------------------------------


def run_demand_timeline(
    dataset: str,
    feature_set: str,
    model_name: str,
    router: Any,
    days: int = 15,
    serve_url: str | None = None,
) -> dict:
    """Generate the demand timeline: actuals + forward forecast.

    Stored forecasts are loaded separately via query_stored_forecasts() so the
    display_horizon dropdown can update without re-running inference.

    Returns dict with keys: actuals_df, forward_results, days, model_name.
    """
    # 1. Fetch actuals for the timeline
    actuals_df = fetch_actuals_for_timeline(days=days)

    # 2. Forward-looking forecast (from now)
    forward_results = run_forecast(
        domain="demand",
        dataset=dataset,
        feature_set=feature_set,
        router=router,
        serve_url=serve_url,
    )

    return {
        "actuals_df": actuals_df,
        "forward_results": forward_results,
        "days": days,
        "model_name": model_name,
    }


def query_stored_forecasts(
    model_name: str, days: int, display_horizon: int
) -> tuple[pl.DataFrame, float | None, float | None]:
    """Query ForecastStore for a given horizon and compute accuracy metrics.

    Called on every Streamlit rerun so the display_horizon dropdown works
    without re-running the full inference pipeline.

    Returns (stored_df, mae, mape).
    """
    from windcast.data.forecast_store import ForecastStore

    settings = get_settings()

    _empty_schema = {
        "target_timestamp": pl.Utf8,
        "horizon_h": pl.Int64,
        "prediction_mw": pl.Float64,
        "model_name": pl.Utf8,
        "domain": pl.Utf8,
        "dataset": pl.Utf8,
        "created_at": pl.Utf8,
    }

    store_path = settings.data_dir / "forecast_store.db"
    if not store_path.exists():
        return pl.DataFrame(schema=_empty_schema), None, None

    store = ForecastStore(store_path)
    end_date = dt.datetime.now(dt.UTC)
    start_date = end_date - dt.timedelta(days=days)
    stored_df = store.query(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        horizon_h=display_horizon,
        model_name=model_name,
    )
    store.close()

    return stored_df, None, None


# ---------------------------------------------------------------------------
# Plotly chart builders
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


def make_demand_timeline_chart(
    actuals_df: pl.DataFrame | None,
    stored_df: pl.DataFrame,
    forward_results: list[dict],
    display_horizon: int,
) -> go.Figure:
    """Build a 15-day demand timeline: actuals + stored forecasts + forward forecast."""
    fig = go.Figure()

    # 1. Actuals (blue line)
    if actuals_df is not None and not actuals_df.is_empty():
        fig.add_trace(
            go.Scatter(
                x=actuals_df["timestamp_utc"].to_list(),
                y=actuals_df["load_mw"].to_list(),
                mode="lines",
                name="Actuals",
                line={"color": "#1f77b4", "width": 2},
            )
        )

        # D-1 TSO forecast (orange dashed) if available
        if "forecast_mw" in actuals_df.columns:
            d1_data = actuals_df.drop_nulls("forecast_mw")
            if not d1_data.is_empty():
                fig.add_trace(
                    go.Scatter(
                        x=d1_data["timestamp_utc"].to_list(),
                        y=d1_data["forecast_mw"].to_list(),
                        mode="lines",
                        name="RTE D-1 Forecast",
                        line={"color": "#ff7f0e", "width": 1.5, "dash": "dash"},
                    )
                )

    # 2. Stored model forecasts (green line)
    if not stored_df.is_empty():
        forecast_ts = stored_df.with_columns(
            pl.col("target_timestamp").str.to_datetime(time_zone="UTC").alias("ts")
        ).sort("ts")

        fig.add_trace(
            go.Scatter(
                x=forecast_ts["ts"].to_list(),
                y=forecast_ts["prediction_mw"].to_list(),
                mode="lines",
                name=f"Model h{display_horizon} Forecast",
                line={"color": "#2ca02c", "width": 2},
            )
        )

    # 3. Vertical separator at "now"
    now = dt.datetime.now(dt.UTC)
    fig.add_vline(x=now, line_dash="dash", line_color="gray", opacity=0.6)

    # 4. Forward forecast (green dots ahead of now)
    if forward_results:
        fig.add_trace(
            go.Scatter(
                x=[r["target_timestamp"] for r in forward_results],
                y=[r["prediction"] for r in forward_results],
                mode="markers+text",
                name="Forward Forecast",
                marker={"color": "#2ca02c", "size": 10, "symbol": "circle"},
                text=[f"{r['prediction']:,.0f}" for r in forward_results],
                textposition="top center",
                textfont={"size": 9},
            )
        )

    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="Load (MW)",
        hovermode="x unified",
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        height=450,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
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

    # Demand-specific: timeline controls
    if domain == "demand":
        st.divider()
        timeline_days = st.slider("Days to show", min_value=7, max_value=30, value=15)
        display_horizon = st.selectbox(
            "Display horizon",
            DEFAULT_HORIZONS,
            index=DEFAULT_HORIZONS.index(24),
            format_func=lambda h: f"h{h} ({build_horizon_desc(h, 60)})",
        )
        assert isinstance(display_horizon, int)

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
        st.caption("**Live actuals:** Not configured (set RTE_CLIENT_ID)")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.header("Forecast Results")

# Handle button click — store results in session state
if run_btn:
    if domain == "demand":
        # Demand domain: 15-day timeline view
        with st.spinner("Fetching actuals, running inference..."):
            try:
                timeline = run_demand_timeline(
                    dataset=dataset,
                    feature_set=feature_set,
                    model_name=model_name,
                    router=router_model,
                    days=timeline_days,
                    serve_url=SERVE_URL,
                )
                st.session_state["demand_timeline"] = timeline
                st.session_state.pop("forecast_results", None)
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                logger.exception("Forecast failed")
                st.session_state.pop("demand_timeline", None)
    else:
        # Wind/solar domain: original 5-point forecast
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
                st.session_state["forecast_actuals"] = None
                st.session_state.pop("demand_timeline", None)
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                logger.exception("Forecast failed")
                st.session_state.pop("forecast_results", None)

# Drop stale demand state when the user switches to a non-demand domain
if domain != "demand":
    st.session_state.pop("demand_timeline", None)

# Display demand timeline
if domain == "demand" and "demand_timeline" in st.session_state:
    tl = st.session_state["demand_timeline"]
    dh = display_horizon  # read live from sidebar dropdown — updates on every rerun

    # Re-query stored forecasts for the current display_horizon (cheap SQLite query)
    stored_df, _, _ = query_stored_forecasts(tl["model_name"], tl["days"], dh)
    n_pts = len(stored_df)

    # Compute accuracy metrics where actuals and forecasts overlap
    mae = None
    mape = None
    actuals_df = tl["actuals_df"]
    if actuals_df is not None and not actuals_df.is_empty() and not stored_df.is_empty():
        forecast_for_join = stored_df.with_columns(
            pl.col("target_timestamp").str.to_datetime(time_zone="UTC").alias("timestamp_utc")
        ).select("timestamp_utc", "prediction_mw")
        merged = actuals_df.join(forecast_for_join, on="timestamp_utc", how="inner")
        if not merged.is_empty():
            errors = (merged["load_mw"] - merged["prediction_mw"]).abs()
            mae = errors.mean()
            pct_errors = errors / merged["load_mw"].abs().clip(lower_bound=1.0) * 100
            mape = pct_errors.mean()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Domain", "Demand")
    col2.metric("Stored Forecasts", f"{n_pts} points")

    if mae is not None:
        col3.metric(f"MAE (h{dh})", f"{mae:,.0f} MW")
    else:
        col3.metric(f"MAE (h{dh})", "N/A")

    if mape is not None:
        col4.metric(f"MAPE (h{dh})", f"{mape:.1f}%")
    else:
        col4.metric(f"MAPE (h{dh})", "N/A")

    # Timeline chart
    fig = make_demand_timeline_chart(
        actuals_df=actuals_df,
        stored_df=stored_df,
        forward_results=tl["forward_results"],
        display_horizon=dh,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forward predictions table
    if tl["forward_results"]:
        st.subheader("Forward Forecast")
        table_data = [
            {
                "Horizon": r["horizon_desc"],
                "Target Time (UTC)": r["target_timestamp"].strftime("%Y-%m-%d %H:%M"),
                "Prediction (MW)": f"{r['prediction']:,.1f}",
            }
            for r in tl["forward_results"]
        ]
        st.table(table_data)

    # Backfill hint
    if n_pts == 0:
        st.info(
            "No stored forecasts found. Run the backfill script to populate "
            "historical forecasts:\n\n"
            "```bash\n"
            "uv run python scripts/backfill_demand_forecasts.py "
            "--start 2025-01-01 --end 2026-04-11\n"
            "```"
        )

# Display wind/solar results from session state (persists across reruns)
elif "forecast_results" in st.session_state:
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

# Feature: Pass 13 — Streamlit Demo Dashboard

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files.

## Feature Description

Minimal Streamlit dashboard that serves as the **live demo frontend** for the WeatherNews presentation (April 14). The dashboard lets a user pick a domain (Wind/Demand), select a trained model run, choose a horizon, and generate a real-time forecast using fresh NWP data from Open-Meteo. It visualizes the forecast as a time-series chart and displays model metadata from MLflow in the sidebar.

This is the visual culmination of Passes 11-12: the model registry provides servable models, the inference pipeline provides feature building + prediction, and the dashboard wraps it all in a clickable interface.

## User Story

As a WN evaluator watching the live demo,
I want to see a dashboard where I pick a domain, click "Generate Forecast", and see a forecast chart with real NWP data,
So that I believe EnerCast is not just a training framework — it produces live, deployable predictions.

## Problem Statement

The inference pipeline (Pass 12) works via CLI (`scripts/inference.py`), but a live demo needs a visual interface. The presentation plan (slide 7 / deployment section) calls for a dashboard that proves the end-to-end path from experiment to live prediction.

## Solution Statement

Create a single-file Streamlit app (`app/dashboard.py`) that:
1. Queries MLflow for available trained model runs
2. Lets the user select domain, model run, and forecast horizon
3. On "Generate Forecast", calls the inference pipeline logic (reusing `build_inference_features()` and model loading from Pass 12)
4. Renders a Plotly line chart of predicted power/load across multiple horizons
5. Displays model metadata (backend, feature set, training date, validation MAE) in the sidebar

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: Medium
**Primary Systems Affected**: `app/` (new), `pyproject.toml` (new deps)
**Dependencies**: Streamlit, Plotly, MLflow (existing), inference pipeline (Pass 12)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ BEFORE IMPLEMENTING

- `scripts/inference.py` (lines 39-116) — **`build_inference_features()`**: the core function to reuse. Takes actuals + NWP + domain + feature_set + horizon + resolution → single-row DataFrame. Also see `_load_model_direct()` (line 119), `_predict_direct()` (line 127), `_resolve_actuals_path()` (line 159), `_tail_rows_for_domain()` (line 166).
- `scripts/inference.py` (lines 173-331) — **`main()` CLI flow**: shows the full inference pipeline. The dashboard reuses the same steps but with UI controls instead of CLI args.
- `scripts/compare_runs.py` (lines 96-116) — **`_fetch_parent_runs()`**: queries MLflow for parent runs with `tags.enercast.run_type = 'parent'`. Reuse this pattern for the sidebar run selector.
- `scripts/compare_runs.py` (lines 119-181) — **`_extract_horizon_metrics()`**: extracts `h{N}_mae` and `h{N}_skill_score` from runs DataFrame. Useful for sidebar model metadata display.
- `src/windcast/config.py` (lines 160-164) — **`DOMAIN_RESOLUTION`**: maps domain → minutes per step (wind=10, demand=60, solar=15).
- `src/windcast/config.py` (lines 168-208) — **`WindCastSettings`**: has `mlflow_tracking_uri`, `processed_dir`, `forecast_horizons`.
- `src/windcast/weather/__init__.py` (lines 248-309) — **`get_live_forecast()`**: fetches fresh NWP. Supports both single-point (Kelmarsh) and weighted (RTE France) configs.
- `src/windcast/features/registry.py` (lines 283-298) — **`get_feature_set()` / `list_feature_sets()`**: look up feature sets by name.
- `src/windcast/training/harness.py` (lines 94-125) — **`resolve_horizon_features()`**: selects horizon-specific NWP columns and creates rename map. Used by `build_inference_features()`.
- `src/windcast/training/harness.py` (lines 153-161) — **`build_horizon_desc()`**: converts steps + resolution to human-readable string (e.g., "4h ahead", "D+1").

### New Files to Create

- `app/dashboard.py` — Main Streamlit app (single file, ~250 lines)

### Files to Modify

- `pyproject.toml` — Add `streamlit` and `plotly` dependencies

### Patterns to Follow

**MLflow Run Query Pattern** (from `compare_runs.py:96-116`):
```python
filter_str = "tags.`enercast.run_type` = 'parent'"
df = mlflow.search_runs(
    experiment_names=experiment_names,
    filter_string=filter_str,
    output_format="pandas",
)
```

**Inference Flow Pattern** (from `scripts/inference.py:173-331`):
```python
# 1. Load model
model = mlflow.pyfunc.load_model(model_uri)
# 2. Load tail actuals
actuals = pl.read_parquet(actuals_path).tail(n_tail_rows)
# 3. Fetch live NWP
nwp = get_live_forecast(weather_config_name, forecast_days=3, past_days=2)
# 4. Build features
features = build_inference_features(actuals, nwp, domain, feature_set, horizon, resolution)
# 5. Predict
prediction = float(model.predict(features.to_pandas())[0])
```

**Domain Defaults** (from `scripts/inference.py`):
```python
DOMAIN_DEFAULTS = {"wind": "kelmarsh", "demand": "rte_france", "solar": "pvdaq_system4"}
UNIT_BY_DOMAIN = {"wind": "kW", "demand": "MW", "solar": "kW"}
WEATHER_CONFIG_MAP = {"kelmarsh": "kelmarsh", "rte_france": "rte_france", "pvdaq_system4": "pvdaq"}
```

**Horizon Description** (from `harness.py:153-161`):
```python
def build_horizon_desc(horizon: int, data_resolution: int) -> str:
    minutes = horizon * data_resolution
    if minutes < 60: return f"{minutes} min ahead"
    hours = minutes / 60
    if hours < 24: return f"{hours:.0f}h ahead" if hours == int(hours) else f"{hours:.1f}h ahead"
    days = hours / 24
    return f"D+{days:.0f}"
```

---

## IMPLEMENTATION PLAN

### Phase 1: Dependencies

Add `streamlit` and `plotly` to `pyproject.toml`. These are runtime deps (needed to run the dashboard). No other dependencies needed — MLflow, Polars, and the inference pipeline are already available.

### Phase 2: Dashboard App

Single-file Streamlit app with this structure:

```
┌──────────────────────────────────────────────────────────┐
│  SIDEBAR                    │  MAIN AREA                 │
│                             │                            │
│  EnerCast                   │  Forecast Results          │
│  ─────────                  │                            │
│  Domain: [Wind ▼]           │  ┌─────┐  ┌─────┐  ┌────┐│
│  Experiment: [enercast-...] │  │MAE  │  │Skill│  │Unit││
│  Model run: [kwf1-xgb...]  │  └─────┘  └─────┘  └────┘│
│  Horizons: ☑h1 ☑h6 ☑h12..  │                            │
│                             │  ┌──────────────────────┐  │
│  [Generate Forecast]        │  │                      │  │
│                             │  │  Plotly line chart    │  │
│  ─────────                  │  │  X = horizon          │  │
│  Model Info                 │  │  Y = prediction       │  │
│  Backend: xgboost           │  │                      │  │
│  Feature set: wind_full     │  └──────────────────────┘  │
│  Trained: 2026-04-10        │                            │
│  Val MAE h6: 181 kW         │  Predictions table         │
│                             │                            │
└──────────────────────────────────────────────────────────┘
```

**Key design decisions:**

1. **Multi-horizon forecast** — rather than a single horizon slider, use checkboxes for all 5 standard horizons [1, 6, 12, 24, 48]. The chart shows predicted values at each horizon as a scatter+line, giving a forecast curve over time. This is more visually compelling than a single-point prediction.

2. **Plotly over st.line_chart** — Plotly supports custom axis labels (kW/MW), hover tooltips, dashed lines, and works well beyond 5k rows. `st.line_chart` is Altair sugar with no control.

3. **Caching strategy**:
   - `@st.cache_resource` for MLflow model loading (singleton per run_id + horizon)
   - `@st.cache_data(ttl=300)` for MLflow run queries (refresh every 5 min)
   - `@st.cache_data(ttl=3600)` for NWP fetching (weather updates every ~6h, 1h cache is fine)

4. **Session state for results** — store forecast results in `st.session_state` so they persist across widget interactions (Streamlit buttons don't persist state).

5. **Error handling** — show `st.error()` with clear messages for: no registered models, NWP fetch failure, feature building failure, prediction failure.

### Phase 3: Testing

Light testing — this is a UI app for a demo, not a production service. Focus on:
- Import test (can the module be imported without errors)
- Verify the app runs with `streamlit run app/dashboard.py` (manual)

---

## STEP-BY-STEP TASKS

### Task 1: UPDATE `pyproject.toml` — add Streamlit and Plotly dependencies

- **ADD** `streamlit>=1.40` and `plotly>=5.0` to `[project] dependencies`
- **VALIDATE**: `uv sync` (installs new deps)
- **VALIDATE**: `uv run python -c "import streamlit; import plotly; print('OK')"`

### Task 2: CREATE `app/dashboard.py` — main Streamlit dashboard

- **CREATE** directory `app/` and file `app/dashboard.py`

- **IMPLEMENT** the app with these sections:

#### Section 1: Imports and Config (~20 lines)

```python
"""EnerCast — Forecast Dashboard.

Live demo frontend for the WeatherNews presentation.
Generates real-time forecasts using trained models and fresh NWP data.

Usage:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import datetime as dt

import mlflow
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from windcast.config import DOMAIN_RESOLUTION, get_settings
from windcast.training.harness import build_horizon_desc
from windcast.weather import get_live_forecast

# Lazy import to avoid circular deps / heavy loading at import time
# build_inference_features is imported inside the forecast function
```

Constants:
```python
EXPERIMENTS_BY_DOMAIN = {
    "wind": "enercast-kelmarsh",
    "demand": "enercast-rte_france",
}
DOMAIN_DEFAULTS = {"wind": "kelmarsh", "demand": "rte_france"}
UNIT_BY_DOMAIN = {"wind": "kW", "demand": "MW"}
WEATHER_CONFIG_MAP = {"kelmarsh": "kelmarsh", "rte_france": "rte_france"}
DEFAULT_HORIZONS = [1, 6, 12, 24, 48]
```

- **GOTCHA**: `st.set_page_config()` MUST be the first Streamlit call — before any other `st.*` call. Put it right after imports.

#### Section 2: Cached Data Fetchers (~30 lines)

```python
@st.cache_data(ttl=300)
def load_parent_runs(experiment_name: str) -> pd.DataFrame:
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
def load_model(run_id: str, horizon: int):
    """Load an MLflow pyfunc model — cached as singleton."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"runs:/{run_id}/model_h{horizon:02d}"
    return mlflow.pyfunc.load_model(uri)
```

- **PATTERN**: Mirrors `_fetch_parent_runs()` from `compare_runs.py:96-116`
- **GOTCHA**: Must call `mlflow.set_tracking_uri()` inside cached functions too, because Streamlit caching may run in a different context. Use `settings.mlflow_tracking_uri` (which is `"sqlite:///mlflow.db"`).

#### Section 3: Forecast Runner (~60 lines)

```python
def run_forecast(
    domain: str,
    dataset: str,
    run_id: str,
    horizons: list[int],
    feature_set: str,
) -> list[dict]:
    """Run inference across multiple horizons. Returns list of result dicts."""
    from scripts.inference import build_inference_features

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

    actuals = pl.read_parquet(actuals_path).tail(tail_rows)

    # Fetch live NWP
    nwp = get_live_forecast(weather_name, forecast_days=3, past_days=2)

    results = []
    for h in horizons:
        model = load_model(run_id, h)

        # Build features for this horizon
        features = build_inference_features(
            actuals_df=actuals,
            nwp_df=nwp,
            domain=domain,
            feature_set_name=feature_set,
            horizon=h,
            resolution_minutes=resolution,
        )

        # Predict
        prediction = float(model.predict(features.to_pandas())[0])

        # Compute target timestamp
        last_ts = actuals["timestamp_utc"].max()
        offset_minutes = h * resolution
        target_ts = last_ts + dt.timedelta(minutes=offset_minutes)

        results.append({
            "horizon": h,
            "horizon_desc": build_horizon_desc(h, resolution),
            "minutes_ahead": offset_minutes,
            "target_timestamp": target_ts,
            "prediction": prediction,
        })

    return results
```

- **PATTERN**: Mirrors `scripts/inference.py:main()` flow but loops over horizons
- **IMPORT**: `build_inference_features` is imported from `scripts.inference` — this works because `scripts/` is importable. If not, copy the function inline.
- **GOTCHA**: Import `build_inference_features` inside the function (lazy) to avoid heavy imports at module load time (faster Streamlit startup).
- **GOTCHA**: The import `from scripts.inference import build_inference_features` may fail if `scripts` is not on sys.path. Fallback: add `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` at module level, or use a relative import strategy. Test this during implementation.

#### Section 4: Plotly Chart Builder (~30 lines)

```python
def make_forecast_chart(
    results: list[dict],
    unit: str,
    domain: str,
) -> go.Figure:
    """Build a Plotly line chart: prediction vs forecast horizon."""
    fig = go.Figure()

    timestamps = [r["target_timestamp"] for r in results]
    predictions = [r["prediction"] for r in results]
    labels = [r["horizon_desc"] for r in results]

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=predictions,
        mode="lines+markers+text",
        name="Forecast",
        line=dict(color="#2ca02c", width=2),
        marker=dict(size=10),
        text=[f"{p:,.0f}" for p in predictions],
        textposition="top center",
        textfont=dict(size=10),
    ))

    target_label = "Active Power" if domain == "wind" else "Load"
    fig.update_layout(
        xaxis_title="Target Time (UTC)",
        yaxis_title=f"{target_label} ({unit})",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=40, b=40),
        height=400,
    )

    return fig
```

- **PATTERN**: Plotly `go.Scatter` with `lines+markers+text` mode for clear forecast visualization
- **KEY**: `hovermode="x unified"` shows all series at same x on hover

#### Section 5: Sidebar UI (~50 lines)

```python
with st.sidebar:
    st.title("EnerCast")
    st.caption("Forecast Dashboard")

    # Domain selector
    domain = st.selectbox("Domain", list(EXPERIMENTS_BY_DOMAIN.keys()), format_func=str.title)
    dataset = DOMAIN_DEFAULTS[domain]
    experiment = EXPERIMENTS_BY_DOMAIN[domain]
    unit = UNIT_BY_DOMAIN[domain]

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

    # Horizon checkboxes
    st.subheader("Horizons")
    selected_horizons = []
    cols = st.columns(len(DEFAULT_HORIZONS))
    for i, h in enumerate(DEFAULT_HORIZONS):
        with cols[i]:
            if st.checkbox(f"h{h}", value=True, key=f"h_{h}"):
                selected_horizons.append(h)

    # Generate button
    run_btn = st.button("Generate Forecast", type="primary", use_container_width=True)

    # Model metadata
    st.divider()
    st.subheader("Model Info")
    backend = selected_run.get("tags.enercast.backend", "—")
    st.caption(f"**Backend:** {backend}")
    st.caption(f"**Feature set:** {feature_set}")
    trained_date = selected_run["start_time"]
    if hasattr(trained_date, "strftime"):
        st.caption(f"**Trained:** {trained_date.strftime('%Y-%m-%d %H:%M')}")

    # Show validation MAE per horizon if available
    for h in DEFAULT_HORIZONS:
        mae_col = f"metrics.h{h}_mae"
        if mae_col in runs_df.columns:
            mae_val = selected_run.get(mae_col)
            if mae_val is not None and str(mae_val) != "nan":
                st.caption(f"Val MAE h{h}: **{mae_val:,.0f} {unit}**")
```

- **KEY**: `st.stop()` prevents the rest of the app from rendering if no runs found
- **KEY**: Horizon checkboxes let the user pick which horizons to forecast (all checked by default)
- **GOTCHA**: `st.button()` returns True only on the click frame — must store results in `st.session_state`

#### Section 6: Main Area (~40 lines)

```python
st.header("Forecast Results")

# Handle button click — store results in session state
if run_btn and selected_horizons:
    with st.spinner("Fetching NWP and running inference..."):
        try:
            results = run_forecast(
                domain=domain,
                dataset=dataset,
                run_id=run_id,
                horizons=selected_horizons,
                feature_set=feature_set,
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
    table_data = []
    for r in results:
        table_data.append({
            "Horizon": r["horizon_desc"],
            "Target Time (UTC)": r["target_timestamp"].strftime("%Y-%m-%d %H:%M"),
            f"Prediction ({unit_display})": f"{r['prediction']:,.1f}",
        })
    st.table(table_data)

else:
    st.info("Select a model and horizons in the sidebar, then click **Generate Forecast**.")
```

- **KEY**: Results stored in `st.session_state["forecast_results"]` so they persist when user interacts with other widgets
- **KEY**: `st.spinner()` shows a loading indicator during NWP fetch + inference (~5-10 seconds)
- **KEY**: `st.table()` for the predictions table (static, not interactive — cleaner for a demo)

### Task 3: Handle `scripts.inference` import path

The dashboard imports `build_inference_features` from `scripts/inference.py`. Since `scripts/` is not a Python package (no `__init__.py`), this requires a path fix.

**Option A (preferred)**: Add a `sys.path` insert at the top of `app/dashboard.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```
This adds the project root to the path, making `scripts.inference` importable.

**Option B (cleaner but more work)**: Move `build_inference_features` to `src/windcast/features/inference.py` and import from there. This is cleaner but creates a new module just for one function — not worth it for a demo.

- **VALIDATE**: `uv run python -c "import sys; sys.path.insert(0, '.'); from scripts.inference import build_inference_features; print('OK')"`

### Task 4: Validate the full stack

- **VALIDATE**: `uv sync` (installs streamlit + plotly)
- **VALIDATE**: `uv run ruff check src/ tests/ scripts/ app/`
- **VALIDATE**: `uv run ruff format --check src/ tests/ scripts/ app/`
- **VALIDATE**: `uv run pyright src/`
- **VALIDATE**: `uv run pytest tests/ -v` (no regressions)
- **VALIDATE** (manual): `cd /Users/renaud/Projects/windcast && uv run streamlit run app/dashboard.py` — should open browser with dashboard

---

## TESTING STRATEGY

### Manual Testing (primary — this is a demo UI)

1. **App launches**: `uv run streamlit run app/dashboard.py` opens without errors
2. **Wind domain**: Select Wind → select a model run → check all horizons → click Generate Forecast → chart renders with 5 prediction points
3. **Demand domain**: Switch to Demand → select RTE France run → Generate Forecast → chart renders with MW units
4. **Model metadata**: Sidebar shows backend, feature set, training date, and validation MAE per horizon
5. **Error handling**: If no models registered, shows error message instead of crashing
6. **Re-run persistence**: After generating a forecast, interact with sidebar widgets — forecast results should persist (session state)

### Automated Testing (minimal)

No unit tests for the Streamlit app itself (testing Streamlit apps requires `streamlit.testing` which is experimental and not worth the overhead for a demo). The core logic being reused (`build_inference_features`, `get_live_forecast`, `load_model`) is already tested in `tests/test_inference.py` and `tests/weather/`.

### Edge Cases

- No MLflow experiments created yet → `st.error()` with clear message
- NWP fetch fails (network issue) → `st.error()` with exception message
- Model not logged for a specific horizon → `st.error()` from MLflow
- Empty actuals Parquet → caught by `build_inference_features()` ValueError

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
uv run ruff check src/ tests/ scripts/ app/
uv run ruff format --check src/ tests/ scripts/ app/
uv run pyright src/
```

**Expected**: All pass with exit code 0

### Level 2: Unit Tests

```bash
uv run pytest tests/ -v
```

**Expected**: All existing 322+ tests pass (no regressions)

### Level 3: Dependency Installation

```bash
uv sync
uv run python -c "import streamlit; import plotly; print('OK')"
```

### Level 4: Manual Validation

```bash
# Launch dashboard
uv run streamlit run app/dashboard.py

# In browser:
# 1. Select Wind domain → pick model → Generate Forecast → verify chart
# 2. Select Demand domain → pick model → Generate Forecast → verify chart
# 3. Verify sidebar shows model metadata
# 4. Verify forecast persists when changing sidebar widgets
```

---

## ACCEPTANCE CRITERIA

- [ ] `streamlit` and `plotly` added to `pyproject.toml` dependencies
- [ ] `uv sync` installs all dependencies without errors
- [ ] `uv run streamlit run app/dashboard.py` launches the dashboard
- [ ] Wind domain: can select a model run and generate a multi-horizon forecast
- [ ] Demand domain: can select RTE France run and generate a forecast
- [ ] Plotly chart shows prediction values at each horizon with correct units (kW / MW)
- [ ] Sidebar displays model metadata: backend, feature set, training date, validation MAE
- [ ] Session state preserves forecast results across widget interactions
- [ ] Error messages shown for missing models or NWP fetch failures
- [ ] All validation commands pass with zero errors
- [ ] No regressions in existing test suite

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] `uv sync` succeeds
- [ ] `uv run ruff check src/ tests/ scripts/ app/` — 0 errors
- [ ] `uv run ruff format --check src/ tests/ scripts/ app/` — 0 errors
- [ ] `uv run pyright src/` — 0 errors
- [ ] `uv run pytest tests/ -v` — all pass
- [ ] Manual: Wind forecast generates and displays correctly
- [ ] Manual: Demand forecast generates and displays correctly
- [ ] Manual: Sidebar metadata displays correctly
- [ ] All acceptance criteria met

---

## NOTES

### Design Decision: Multi-Horizon Chart (not single-point)

Rather than showing a single prediction for one horizon, the dashboard forecasts across all selected horizons and plots them as a curve. This is more visually compelling for the WN demo and shows the model's performance degradation over time — which is the core story ("rearview mirror → windshield").

### Design Decision: Plotly over Altair/st.line_chart

`st.line_chart` is Altair sugar with no axis label control, no unit display, and a 5,000-row limit. Plotly gives us custom axis labels (kW/MW), hover tooltips, markers with value labels, and works at any scale.

### Design Decision: Single File (~250 lines)

The dashboard is one file — no pages, no components, no abstraction. This is a demo for a presentation. If EnerCast becomes a product, the dashboard would be split into pages (forecast, comparison, monitoring). For now, YAGNI.

### Design Decision: Reuse inference.py directly

Rather than duplicating the inference logic, the dashboard imports `build_inference_features` from `scripts/inference.py`. This guarantees feature parity with the CLI tool. The sys.path hack is acceptable for a demo app.

### Known Limitation: Actuals from Parquet

The dashboard uses the tail of the processed Parquet as "recent actuals" for lag features. In production, this would come from a real-time SCADA/load feed. For the demo, this is honest — we're showing the pipeline works, not pretending we have live feeds.

### Presentation Flow

For the live demo during the WN presentation:
1. Pre-launch the dashboard before the talk
2. Show Wind domain forecast first (Kelmarsh — their core business)
3. Switch to Demand domain (RTE France — proves cross-domain)
4. Highlight the sidebar metadata (MLflow integration, reproducibility)
5. Point out: "Same framework, different domain, zero code changes"

# Feature: Pass 12 — Inference Pipeline

The following plan should be complete, but validate documentation and codebase patterns before implementing.

## Feature Description

End-to-end inference script that fetches fresh NWP forecast data from Open-Meteo's Forecast API, builds a feature vector on the fly, loads a trained model from MLflow Model Registry, and returns structured JSON predictions per horizon. This closes the gap between experiment and production — a trained EnerCast model becomes usable for real forecasting, not just backtesting.

## User Story

As a WN evaluator watching the live demo,
I want to see EnerCast produce a forecast for tomorrow using fresh weather data,
So that I believe the framework doesn't stop at experiments — it's deployable.

## Problem Statement

Pass 11 logged models with MLflow signatures, but there's no way to run a forecast outside of the training loop. The missing piece: fetch fresh NWP → build features → call model → structured output. This is the building block for the Streamlit dashboard (Pass 13) and for any future cron/API consumer.

## Solution Statement

Create `scripts/inference.py` that:
1. Loads a registered model via `mlflow.pyfunc.load_model("models:/{name}@champion")`
2. Fetches fresh NWP from Open-Meteo's **Forecast API** (`api.open-meteo.com/v1/forecast`)
3. Builds the feature vector matching the model's expected input schema
4. Runs prediction and outputs structured JSON (one entry per horizon)

Also add `fetch_forecast_weather()` to the weather module for live NWP fetching (new API endpoint, distinct from archive and historical-forecast).

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: Medium
**Primary Systems Affected**: `weather/`, `features/`, `scripts/`
**Dependencies**: Open-Meteo Forecast API (free, no key), MLflow Model Registry (local)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ BEFORE IMPLEMENTING

- `src/windcast/data/open_meteo.py` (lines 1-167) — **Pattern for new `fetch_forecast_weather()`**. Mirror `fetch_historical_forecast_weather()` structure. Uses `build_client()` + `_response_to_polars()` (both reusable). Add `FORECAST_URL = "https://api.open-meteo.com/v1/forecast"`.
- `src/windcast/weather/__init__.py` (lines 210-244) — **`get_forecast_weather()` pattern**. Shows how weather module wraps the low-level fetcher. New `get_live_forecast()` function should follow this shape but use the Forecast API.
- `src/windcast/weather/registry.py` (lines 1-143) — **Weather configs per dataset**. `KELMARSH_WEATHER` has `wind_speed_100m`, `wind_direction_100m`, `temperature_2m`. `RTE_FRANCE_WEATHER` is a `WeightedWeatherConfig` with 8 cities. The inference script must handle both single-point and weighted configs.
- `src/windcast/features/weather.py` (lines 20-84) — **`join_nwp_horizon_features()`**. Joins NWP at each horizon offset. For inference, we need a single-horizon variant: just shift by the target horizon and rename to canonical names (no `_h{N}` suffix).
- `src/windcast/features/wind.py` (lines 17-79) — **`build_wind_features()`**. Shows the full feature build chain: QC filter → lags → rolling → cyclic → NWP join. For inference, lag/rolling features require recent actuals (loaded from processed Parquet).
- `src/windcast/features/demand.py` (lines 17-65) — **`build_demand_features()`**. Same pattern: lags → cyclic → rolling → HDD/CDD from NWP. Inference needs recent load history for lag computation.
- `src/windcast/features/registry.py` (lines 260-298) — **`FEATURE_REGISTRY` + `get_feature_set()`**. Lists all feature columns. The model's input signature matches canonical names (no `_h{N}`).
- `src/windcast/training/harness.py` (lines 94-125) — **`resolve_horizon_features()`**. Shows the `_h{N}` → canonical name rename logic used during training. Inference must produce the same canonical feature names.
- `src/windcast/training/backends.py` (lines 55-74) — **`XGBoostBackend.log_model()`**. Shows how models are logged with `infer_signature()`. The model URI pattern: `runs:/{run_id}/model_h{HH}` or `models:/{name}/{version}`.
- `src/windcast/models/autogluon_pyfunc.py` (lines 1-31) — **`AutoGluonPyfuncWrapper`**. Shows pyfunc pattern. Both XGBoost and AutoGluon models are loadable via `mlflow.pyfunc.load_model()`.
- `src/windcast/config.py` (lines 117-208) — **`DATASETS`, `DOMAIN_RESOLUTION`, `WindCastSettings`**. Dataset configs have `latitude`/`longitude`. `DOMAIN_RESOLUTION` maps domain → minutes per step.
- `scripts/train.py` (lines 1-201) — **CLI argument pattern**. Mirror `argparse` style: `--domain`, `--dataset`, `--horizons`. Domain defaults, dataset defaults.
- `scripts/build_features.py` (lines 1-251) — **Feature building orchestration**. Shows how weather is loaded + joined + features built. Inference reuses this flow but with live NWP.

### New Files to Create

- `scripts/inference.py` — Main inference CLI script
- `tests/test_inference.py` — Tests for inference pipeline helpers

### Files to Modify

- `src/windcast/data/open_meteo.py` — Add `FORECAST_URL` + `fetch_forecast_weather()`
- `src/windcast/weather/__init__.py` — Add `get_live_forecast()` + export

### Patterns to Follow

**CLI Pattern** (from `scripts/train.py`):
```python
parser = argparse.ArgumentParser(description="...")
parser.add_argument("--domain", choices=["wind", "demand", "solar"], default="wind")
parser.add_argument("--dataset", default=None)
# ... resolve defaults from domain
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
settings = get_settings()
```

**Weather Fetch Pattern** (from `open_meteo.py`):
```python
# Same client, same _response_to_polars(), different URL + params
responses = client.weather_api(FORECAST_URL, params=params)
return _response_to_polars(responses[0], variables)
```

**Model Loading Pattern** (MLflow pyfunc):
```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/enercast-kelmarsh-xgboost@champion")
predictions = model.predict(X_pandas_df)
```

---

## IMPLEMENTATION PLAN

### Phase 1: Weather Module — Live Forecast Fetcher

Add the ability to fetch **real-time NWP forecasts** from Open-Meteo's Forecast API. This is distinct from the existing archive (ERA5) and historical-forecast endpoints.

**Key decisions:**
- **Short cache TTL**: Live forecast data changes every 6h. Use `expire_after=3600` (1h) for the forecast client, not the default `-1` (never expire).
- **`forecast_days` param**: Default to 7 (Open-Meteo max is 16). The script will filter to the requested horizons.
- **`past_days` param**: Include 2 days of recent NWP for context features. This doesn't replace SCADA actuals for lag computation, but gives the model NWP-consistent recent data.
- **Reuse `_response_to_polars()`**: Same protobuf response format across all 3 APIs.

### Phase 2: Inference Feature Builder

Build a feature vector for a single forecast target from:
1. **Recent actuals** (from processed Parquet) — needed for lag and rolling features
2. **Fresh NWP** (from Forecast API) — needed for weather-driven features
3. **Calendar features** — computed on the fly from target timestamp

**Key challenge:** The training pipeline builds features over the full dataset, then the harness extracts per-horizon slices. For inference, we need to:
- Load the tail of the processed data (enough rows for the largest lag: 168 steps = 1 week for demand)
- Append a "future row" with the target timestamp
- Build features normally (lags computed from actuals, NWP joined at target horizon)
- Extract the single row for prediction

This approach reuses the existing feature builders (`build_wind_features`, `build_demand_features`) rather than duplicating logic.

### Phase 3: Inference Script

CLI that ties everything together. Two modes:
1. **Direct mode** (default): Load model via `mlflow.pyfunc.load_model()`, predict in-process
2. **Server mode** (`--serve-url`): POST to a running `mlflow models serve` endpoint

Direct mode is simpler and sufficient for the demo. Server mode validates the serving architecture from PRD.

### Phase 4: Testing

Unit tests for the new weather fetcher and inference feature builder. Integration test for the full pipeline (requires MLflow model to be registered — skip in CI with marker).

---

## STEP-BY-STEP TASKS

### Task 1: ADD `fetch_forecast_weather()` to `src/windcast/data/open_meteo.py`

- **IMPLEMENT**: Add `FORECAST_URL = "https://api.open-meteo.com/v1/forecast"` constant
- **IMPLEMENT**: Add `fetch_forecast_weather()` function:
  ```python
  def fetch_forecast_weather(
      latitude: float,
      longitude: float,
      variables: list[str] | None = None,
      forecast_days: int = 7,
      past_days: int = 0,
      client: openmeteo_requests.Client | None = None,
  ) -> pl.DataFrame:
  ```
  - Uses `FORECAST_URL`, passes `forecast_days` and `past_days` (NOT `start_date`/`end_date`)
  - Client should use `expire_after=3600` (1h cache for live data)
  - Reuses `_response_to_polars()` — same response format
- **PATTERN**: Mirror `fetch_historical_forecast_weather()` at line 81-136 of same file
- **GOTCHA**: Default `build_client()` uses `expire_after=-1` (never expire). Pass `expire_after=3600` for live forecast client to avoid stale predictions.
- **VALIDATE**: `uv run python -c "from windcast.data.open_meteo import fetch_forecast_weather; print('OK')"`

### Task 2: ADD `get_live_forecast()` to `src/windcast/weather/__init__.py`

- **IMPLEMENT**: Add a public `get_live_forecast()` function:
  ```python
  def get_live_forecast(
      config_name: str,
      forecast_days: int = 7,
      past_days: int = 0,
  ) -> pl.DataFrame:
  ```
  - Resolves weather config from registry (handles both `WeatherConfig` and `WeightedWeatherConfig`)
  - For single-point: calls `fetch_forecast_weather()` directly
  - For weighted (RTE France): fetches each point, applies `_weighted_mean()`
  - No SQLite caching for live forecasts (they're ephemeral)
- **IMPORTS**: Add `fetch_forecast_weather` import from `windcast.data.open_meteo`
- **UPDATE**: Add `get_live_forecast` to `__all__`
- **PATTERN**: Follow `get_forecast_weather()` at line 210-244 for structure, but simpler (no cache, no date validation)
- **VALIDATE**: `uv run python -c "from windcast.weather import get_live_forecast; print('OK')"`

### Task 3: CREATE `scripts/inference.py` — main inference script

- **IMPLEMENT**: Full inference CLI with these arguments:
  - `--domain` (wind/demand/solar, default: wind)
  - `--dataset` (default: domain-specific)
  - `--model-name` (MLflow registered model name, default: `enercast-{dataset}-xgboost`)
  - `--model-alias` (default: `champion`)
  - `--model-uri` (override: direct MLflow model URI like `runs:/{id}/model_h06`)
  - `--horizon` (single horizon in steps, default: 24)
  - `--feature-set` (default: `{domain}_full`)
  - `--serve-url` (optional: if set, POST to this URL instead of loading model directly)
  - `--output` (optional: path for JSON output file, default: stdout)
  - `--actuals-path` (optional: path to processed Parquet for lag features, default: auto-resolve from settings)

- **IMPLEMENT**: Core inference flow:
  1. **Load model**: `mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")` or `--model-uri`
  2. **Load recent actuals**: Read tail of processed Parquet (need enough rows for max lag — 168 for demand, 24 for wind). This provides the lag/rolling features.
  3. **Fetch live NWP**: Call `get_live_forecast(weather_config_name, forecast_days=3)`
  4. **Build features**: Call domain-specific builder (`build_wind_features` / `build_demand_features`) on recent actuals with NWP joined at the target horizon
  5. **Extract feature row**: Take the last row where all features are populated, select only the model's input columns (from feature set registry, canonical names without `_h{N}`)
  6. **Predict**: Call `model.predict(X_df.to_pandas())` (pyfunc expects pandas)
  7. **Format output**: JSON with prediction value, horizon, timestamp, model info, NWP source

- **IMPLEMENT**: Server mode (when `--serve-url` is set):
  - Format input as `dataframe_split` JSON (MLflow serving format)
  - POST to `{serve_url}/invocations`
  - Parse response

- **IMPLEMENT**: Output format:
  ```json
  {
    "domain": "wind",
    "dataset": "kelmarsh",
    "model": "enercast-kelmarsh-xgboost@champion",
    "horizon_steps": 24,
    "horizon_desc": "4h ahead",
    "target_timestamp_utc": "2026-04-11T14:00:00+00:00",
    "prediction": 1234.5,
    "unit": "kW",
    "nwp_source": "Open-Meteo Forecast API",
    "generated_at": "2026-04-11T10:00:00+00:00"
  }
  ```

- **PATTERN**: CLI follows `scripts/train.py` structure (argparse, logging, settings)
- **GOTCHA**: Feature columns must match the model's signature exactly (canonical names, correct order). Use model's input signature to validate: `model.metadata.get_input_schema()`.
- **GOTCHA**: Wind lag features use `.over("turbine_id")` — make sure the recent actuals DataFrame has the `turbine_id` column when building features.
- **GOTCHA**: Demand full features compute HDD/CDD from `nwp_temperature_2m_h1` — need the NWP horizon join BEFORE calling `build_demand_features()` (same as `build_features.py` line 184-189).
- **GOTCHA**: `build_horizon_desc()` from `harness.py` converts steps to human-readable. Reuse it.
- **VALIDATE**: `uv run python scripts/inference.py --help` (should show usage)
- **VALIDATE**: `uv run python scripts/inference.py --domain wind --horizon 6 --model-uri runs:/{any_run_id}/model_h06` (requires a trained model)

### Task 4: ADD unit for the inference feature builder

The feature-building-for-inference logic should be factored into a testable helper, either in the script or as a utility function. The core logic:

- **IMPLEMENT**: `build_inference_features()` helper (can live in the script or in a new `src/windcast/features/inference.py`):
  ```python
  def build_inference_features(
      actuals_df: pl.DataFrame,
      nwp_df: pl.DataFrame,
      domain: str,
      feature_set_name: str,
      horizon: int,
      resolution_minutes: int,
  ) -> pl.DataFrame:
  ```
  - Takes recent actuals + fresh NWP
  - Calls the domain feature builder
  - Joins NWP at the single target horizon
  - Renames `nwp_*_h{N}` → `nwp_*` (canonical names)
  - Returns a single-row DataFrame ready for prediction

- **PATTERN**: Reuses `resolve_horizon_features()` from `harness.py` for the rename logic
- **VALIDATE**: `uv run pytest tests/test_inference.py -v`

### Task 5: CREATE `tests/test_inference.py`

- **IMPLEMENT**: Tests for:
  1. `fetch_forecast_weather()` — mock the Open-Meteo client, verify params include `forecast_days`, verify output schema matches existing fetchers
  2. `build_inference_features()` — use synthetic actuals + NWP, verify output has correct columns and single row
  3. Output JSON structure — verify all required fields present
- **PATTERN**: Follow `tests/weather/test_provider.py` for mocking patterns
- **VALIDATE**: `uv run pytest tests/test_inference.py -v`

### Task 6: Validate full pipeline

- **VALIDATE**: `uv run ruff check src/ tests/ scripts/`
- **VALIDATE**: `uv run ruff format --check src/ tests/ scripts/`
- **VALIDATE**: `uv run pyright src/`
- **VALIDATE**: `uv run pytest tests/ -v`
- **VALIDATE**: (manual) `uv run python scripts/inference.py --domain wind --horizon 6` (requires registered model from Pass 11)

---

## TESTING STRATEGY

### Unit Tests

1. **`fetch_forecast_weather()`**: Mock the `openmeteo_requests.Client`, verify:
   - URL is `FORECAST_URL` (not archive)
   - Params include `forecast_days`, not `start_date`/`end_date`
   - Output is a Polars DataFrame with `timestamp_utc` + variable columns

2. **`get_live_forecast()`**: Mock the underlying fetcher, verify:
   - Single-point config returns direct result
   - Weighted config (RTE France) calls fetcher per point and applies weighted mean

3. **`build_inference_features()`**: Synthetic data, verify:
   - Output DataFrame has exactly the columns from the feature set
   - NWP columns are renamed to canonical names (no `_h{N}` suffix)
   - Lag/rolling features are computed from actuals

### Integration Tests (marked `slow`, require trained model)

4. **Full inference round-trip**: Load a real model from MLflow, run inference with mock NWP, verify prediction is a float.

### Edge Cases

- No registered model → clear error message
- Empty NWP response (network issue) → graceful failure
- Feature mismatch (model expects columns that aren't in the feature set) → validation error
- Wind domain without `turbine_id` in actuals → error

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/
```

### Level 2: Unit Tests

```bash
uv run pytest tests/test_inference.py -v
uv run pytest tests/ -v  # full suite, no regressions
```

### Level 3: Manual Validation

```bash
# Requires a registered model (from Pass 11 --register)
uv run python scripts/inference.py --domain wind --horizon 6
uv run python scripts/inference.py --domain demand --dataset rte_france --horizon 24
```

### Level 4: Server Mode Validation

```bash
# Terminal 1: start serving
mlflow models serve -m "models:/enercast-kelmarsh-xgboost@champion" -p 5001

# Terminal 2: test via script
uv run python scripts/inference.py --domain wind --horizon 6 --serve-url http://localhost:5001

# Terminal 2: test via curl (raw)
curl -s http://localhost:5001/invocations -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

---

## ACCEPTANCE CRITERIA

- [ ] `fetch_forecast_weather()` fetches from `api.open-meteo.com/v1/forecast` (live NWP)
- [ ] `get_live_forecast()` works for both single-point (Kelmarsh) and weighted (RTE France) configs
- [ ] `scripts/inference.py --help` shows clean usage
- [ ] `scripts/inference.py --domain wind --horizon 6` outputs valid JSON with prediction
- [ ] `scripts/inference.py --domain demand --dataset rte_france --horizon 24` works
- [ ] Server mode (`--serve-url`) POSTs correct `dataframe_split` format
- [ ] All validation commands pass with zero errors
- [ ] No regressions in existing 322+ tests
- [ ] Output JSON contains: domain, dataset, model, horizon, target_timestamp, prediction, unit

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] `uv run ruff check src/ tests/ scripts/` — 0 errors
- [ ] `uv run ruff format --check src/ tests/ scripts/` — 0 errors
- [ ] `uv run pyright src/` — 0 errors
- [ ] `uv run pytest tests/ -v` — all pass
- [ ] Manual inference test on wind domain
- [ ] Manual inference test on demand domain
- [ ] All acceptance criteria met

---

## NOTES

### Design Decision: Direct Load vs Serve

The script supports both `mlflow.pyfunc.load_model()` (direct) and POST to `/invocations` (server). For the WN demo, direct mode is safer (no separate process to manage). Server mode validates the production architecture.

### Design Decision: Feature Building via Existing Builders

Rather than building a separate "inference feature pipeline", we reuse `build_wind_features()` / `build_demand_features()` with recent actuals + fresh NWP. This guarantees feature parity between training and inference — the #1 source of production bugs in ML systems.

### Design Decision: No Caching for Live Forecasts

Live forecast data from Open-Meteo changes every ~6h. We use a short HTTP cache TTL (1h via `requests_cache`) but don't persist to SQLite like we do for historical data. Forecast data is ephemeral by nature.

### Known Limitation: Actuals Availability

For a true live forecast, we'd need real-time SCADA/load data feeds. For the demo, we use the tail of the processed Parquet as a proxy for "recent actuals". This is honest — the demo shows the *pipeline* works, not that we have live data feeds.

### Unit Mapping

| Domain | Target | Unit |
|--------|--------|------|
| wind | active_power_kw | kW |
| demand | load_mw | MW |
| solar | power_kw | kW |

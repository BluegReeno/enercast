# Feature: Demand Forecast Backfill + 15-Day Dashboard

The following plan should be complete. Validate documentation and codebase patterns before implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files.

## Feature Description

Transform the demand dashboard from a "single-shot 5-point forecast" into a **continuous forecast timeline** that shows the last 15 days of actual demand alongside stored model forecasts. A "Generate Forecast" button backfills missing forecasts up to today and computes the forward-looking forecast.

The goal is a WattCast-like experience: open the dashboard → see recent history + forecast overlay → visually compare model accuracy in near-real-time.

## User Story

As a demand forecasting engineer using the EnerCast dashboard,
I want to see 15 days of actual load with forecast overlay and generate new forecasts on demand,
So that I can visually assess model accuracy and demonstrate live forecasting capability.

## Problem Statement

The current dashboard generates a **one-shot forecast** (5 discrete horizon points from "now") with no persistence. There is no way to see how the model performed over recent days because:
1. No forecast results are stored — predictions vanish when the Streamlit session ends
2. Actuals are limited to ~24h from the RTE API (current `fetch_recent_load` implementation)
3. There is no backfill mechanism to generate historical forecasts

## Solution Statement

1. **Extend `rte_api.py`** with chunked historical fetch (2025-01 → today, 180-day chunks)
2. **Create `ForecastStore`** (SQLite) to persist forecast predictions with timestamps
3. **Create `backfill_demand_forecasts.py`** script — iterate over recent dates, build features from actuals + NWP, predict, store
4. **Refactor `_extract_nwp_at_horizon`** to accept an arbitrary reference timestamp (not just `now`)
5. **Rewrite dashboard demand view** — 15-day actuals line + forecast curve from store + "Generate Forecast" button that fills gaps

## Feature Metadata

**Feature Type**: Enhancement
**Estimated Complexity**: High
**Primary Systems Affected**: `rte_api.py`, `scripts/inference.py`, `app/dashboard.py`, new `ForecastStore`
**Dependencies**: Existing — httpx, polars, mlflow, streamlit, plotly. No new dependencies.

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ BEFORE IMPLEMENTING

- `src/windcast/data/rte_api.py` (full file) — Current RTEClient + fetch_recent_load. Extend with chunked fetch.
- `scripts/inference.py` (lines 38-164) — `_extract_nwp_at_horizon` (hardcoded to `now` on line 50) and `build_inference_features`. Refactor to accept `reference_time`.
- `src/windcast/features/demand.py` (full file) — `build_demand_features`: needs 200 rows of actuals for lag168 + buffer.
- `src/windcast/features/registry.py` (lines 90-101) — `DEMAND_FULL` columns: lags, cyclic, rolling, NWP (temp/wind/humidity), HDD/CDD.
- `src/windcast/weather/__init__.py` (lines 211-245) — `get_forecast_weather()` for historical NWP (≥2022-01-01). `get_live_forecast()` for fresh NWP.
- `src/windcast/weather/storage.py` (full file) — `WeatherStorage` SQLite pattern to mirror for ForecastStore.
- `app/dashboard.py` (full file) — Current dashboard. Complete rewrite of main area for demand domain.
- `src/windcast/config.py` (lines 79-88) — `RTE_FRANCE` config: zone_id="FR", Paris coords, 8y train / 2y val.
- `scripts/fetch_historical_forecasts.py` (full file) — Existing NWP backfill pattern (idempotent, chunked).
- `docs/rte-api-notes.md` — API reference: `short_term` max 186 days, history since 2014, REALISED + D-1.

### New Files to Create

- `src/windcast/data/forecast_store.py` — ForecastStore SQLite class (upsert, query, coverage)
- `scripts/backfill_demand_forecasts.py` — Batch forecast generation script
- `tests/data/test_forecast_store.py` — Unit tests for ForecastStore

### Patterns to Follow

**SQLite Storage Pattern** (mirror `weather/storage.py`):
```python
class ForecastStore:
    def __init__(self, db_path: Path) -> None: ...
    def upsert(self, forecasts: pl.DataFrame) -> int: ...
    def query(self, start: str, end: str, horizon_h: int | None = None) -> pl.DataFrame: ...
    def get_coverage(self) -> tuple[str, str] | None: ...
    def close(self) -> None: ...
```

**Chunked API Pattern** (from WattCast `rte.py` + `rte-api-notes.md`):
```python
def _chunk_dates(start: datetime, end: datetime, chunk_days: int = 180) -> list[tuple[datetime, datetime]]:
    """Split a date range into API-safe chunks."""
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks
```

**Naming Conventions**: snake_case functions, PascalCase classes, UPPER_SNAKE constants.

**Error Handling**: `logger.warning()` for recoverable, `raise ValueError()` for bad input. No silent failures.

**Logging**: `logger = logging.getLogger(__name__)` at module top.

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation — ForecastStore + RTE API Extension

Build the storage layer and the ability to fetch long historical actuals.

**Tasks:**
- Create `ForecastStore` SQLite class with upsert/query
- Add `_chunk_dates()` and `fetch_load_history()` to `rte_api.py`
- Add `fetch_tso_forecast_history()` to get D-1 forecasts for comparison

### Phase 2: Inference Refactor — Support Arbitrary Timestamps

Make `build_inference_features` work for any reference time, not just "now".

**Tasks:**
- Refactor `_extract_nwp_at_horizon` to accept `reference_time` parameter
- Update `build_inference_features` signature accordingly
- Ensure backward compatibility (default to `now` if not specified)

### Phase 3: Backfill Script — Batch Forecast Generation

Script that generates forecasts for a date range and stores them.

**Tasks:**
- Create `scripts/backfill_demand_forecasts.py`
- Load actuals from RTE API (chunked) or processed Parquet
- Load NWP from historical forecast DB
- Iterate day by day, build features, predict all horizons, store

### Phase 4: Dashboard Rewrite — 15-Day Timeline View

Replace the current 5-point scatter with a continuous timeline.

**Tasks:**
- Rewrite demand chart: 15 days actuals (blue) + stored forecasts (green) + optional D-1 TSO (orange)
- "Generate Forecast" button: backfill missing days + compute forward forecast
- Keep wind domain unchanged (existing behavior)

### Phase 5: Testing & Validation

**Tasks:**
- Unit tests for ForecastStore
- Unit tests for chunked API fetch
- Integration test for backfill pipeline
- Manual dashboard validation

---

## STEP-BY-STEP TASKS

### Task 1: CREATE `src/windcast/data/forecast_store.py`

**IMPLEMENT**: SQLite storage for forecast predictions.

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS forecasts (
    target_timestamp TEXT NOT NULL,      -- UTC ISO, the time being predicted
    horizon_h INTEGER NOT NULL,          -- horizon in hours (1, 6, 12, 24, 48)
    prediction_mw REAL NOT NULL,         -- predicted load in MW
    model_name TEXT NOT NULL,            -- e.g. "enercast-rte_france-xgboost"
    domain TEXT NOT NULL DEFAULT 'demand',
    dataset TEXT NOT NULL DEFAULT 'rte_france',
    created_at TEXT NOT NULL,            -- when the forecast was generated
    PRIMARY KEY (target_timestamp, horizon_h, model_name)
);
```

**Methods:**
- `__init__(self, db_path: Path)` — open/create DB, create table
- `upsert(self, forecasts: list[dict]) -> int` — INSERT OR REPLACE, return count
- `query(self, start: str, end: str, horizon_h: int | None = None, model_name: str | None = None) -> pl.DataFrame` — return matching forecasts as Polars DataFrame
- `get_coverage(self) -> tuple[str, str] | None` — min/max target_timestamp
- `close(self)` — close connection

**PATTERN**: Mirror `weather/storage.py` structure (lines 24-144).
**IMPORTS**: `sqlite3`, `polars`, `logging`, `pathlib.Path`
**GOTCHA**: Store timestamps as ISO strings (same as WeatherStorage). Query uses `>=` and `<=` with `T23:59:59` suffix for end date.

**VALIDATE**: `uv run pytest tests/data/test_forecast_store.py -v`

---

### Task 2: UPDATE `src/windcast/data/rte_api.py` — Add Chunked Historical Fetch

**IMPLEMENT**: Add `_chunk_dates()` helper and `fetch_load_history()` for long-range backfill.

**ADD** after `fetch_recent_load` (line 123):

```python
def _chunk_dates(
    start: datetime, end: datetime, chunk_days: int = 180
) -> list[tuple[datetime, datetime]]:
    """Split date range into API-safe chunks (max 186 days per call)."""

def fetch_load_history(
    client: RTEClient,
    start: datetime,
    end: datetime,
    types: str = "REALISED",
    chunk_days: int = 180,
    sleep_between: float = 0.5,
) -> pl.DataFrame:
    """Fetch load history in chunks, returning hourly DataFrame.
    
    Returns: (timestamp_utc, load_mw) — or (timestamp_utc, load_mw, forecast_mw) 
    if types includes D-1.
    """
```

**PATTERN**: WattCast `_chunk_dates` pattern + 0.5s courtesy sleep between chunks.
**IMPORTS**: Add `time` (for sleep), `datetime` already imported.
**GOTCHA**: 
- API max is 186 days, use 180 for safety margin.
- API dates need timezone offset (use `+01:00` or `+02:00` for CET/CEST). Safest: use UTC `+00:00`.
- Parse D-1 type separately: its values are in a different series entry (see `rte-api-notes.md` response structure).
- Resample 15-min → hourly via `group_by_dynamic("timestamp_utc", every="1h").agg(pl.col("load_mw").mean())`.

**VALIDATE**: `uv run pytest tests/data/test_rte_api.py -v`

---

### Task 3: UPDATE `scripts/inference.py` — Parametric Reference Time

**IMPLEMENT**: Make `_extract_nwp_at_horizon` and `build_inference_features` accept an optional `reference_time` parameter.

**CHANGE** `_extract_nwp_at_horizon` (line 38-68):
- Add parameter: `reference_time: datetime | None = None`
- Line 50: `target_time = (reference_time or datetime.now(UTC)) + timedelta(minutes=horizon * resolution_minutes)`

**CHANGE** `build_inference_features` (line 71-164):
- Add parameter: `reference_time: datetime | None = None`
- Pass it through to `_extract_nwp_at_horizon` call (line 143)

**GOTCHA**: Keep backward compatibility — existing callers (dashboard `run_forecast`) don't pass `reference_time`, so it defaults to `now`.

**VALIDATE**: `uv run pytest tests/test_inference.py -v`

---

### Task 4: CREATE `scripts/backfill_demand_forecasts.py`

**IMPLEMENT**: Batch forecast generation for demand domain.

**CLI interface:**
```bash
uv run python scripts/backfill_demand_forecasts.py \
    --start 2025-01-01 --end 2026-04-11 \
    --horizon 24 \
    --model-name enercast-rte_france-xgboost
```

**Algorithm:**
1. Parse args: `--start`, `--end`, `--horizon` (default: all [1,6,12,24,48]), `--model-name`, `--model-alias` (default: champion), `--db-path` (default: `data/forecast_store.db`)
2. Load champion model via `mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")`
3. **Load actuals**: 
   - Try processed Parquet first (`data/processed/rte_france.parquet`) for dates within its range
   - For dates beyond Parquet coverage: fetch from RTE API via `fetch_load_history()` and append
   - Sort by timestamp, ensure continuous hourly series
4. **Load NWP**: Use `get_forecast_weather("rte_france", start, end)` from `weather_forecast.db` for historical period. For recent dates (last ~7 days), use `get_live_forecast("rte_france")`.
5. **Iterate**: For each target timestamp (hourly, from start to end):
   - Check if forecast already exists in store (skip if so — idempotent)
   - Slice actuals tail (200 rows before target)
   - For each horizon h: call `build_inference_features(actuals_tail, nwp, "demand", "demand_full", h, 60, reference_time=target - h*60min)`
   - Predict via model router: `model.predict(features_pd, params={"horizon": h})`
   - Collect result: `{target_timestamp, horizon_h, prediction_mw, model_name, created_at}`
6. **Store**: Batch upsert to ForecastStore
7. **Log**: Progress every 100 timestamps, final summary

**PATTERN**: Mirror `scripts/fetch_historical_forecasts.py` (idempotent, logged, CLI args).
**IMPORTS**: `argparse`, `logging`, `polars`, `mlflow`, forecast_store, inference, weather, rte_api.
**GOTCHA**: 
- The reference_time for inference is NOT the target timestamp. It's `target_ts - horizon * resolution`. For h24 demand (hourly), reference = target - 24h. This is "when we would have made the forecast".
- NWP coverage: Historical Forecast API starts 2022-01-01. For 2025-2026, it's available. For live (last ~5 days), use `get_live_forecast`.
- Actuals must extend to at least `reference_time` for lags to work. Need 200 rows before reference_time.
- Memory: don't load all actuals at once for large ranges. Process in daily batches.

**VALIDATE**: `uv run python scripts/backfill_demand_forecasts.py --start 2025-01-01 --end 2025-01-03 --horizon 24` (small range test)

---

### Task 5: UPDATE `app/dashboard.py` — 15-Day Timeline View

**IMPLEMENT**: Rewrite the demand domain view to show a continuous timeline.

**New flow for demand domain:**

1. **Sidebar**: Keep model selector. Add "Days to show" slider (default: 15, range: 7-30). Remove date picker (irrelevant for timeline view).

2. **"Generate Forecast" button** — when clicked:
   a. Fetch actuals from RTE API: `fetch_load_history(client, now - N_days, now, types="REALISED")`
   b. Check ForecastStore coverage for the same period
   c. For any gaps in the store: run inference (same logic as backfill but just for missing timestamps)
   d. Store new forecasts
   e. Display timeline

3. **Chart** — Plotly figure with:
   - **Blue line**: Actuals (hourly load_mw from RTE API)
   - **Green line**: h24 forecasts (from ForecastStore, each point = what the model predicted 24h before)
   - **Orange dashed line** (optional): RTE D-1 forecast (from `type=D-1` in API call) for benchmark comparison
   - **Vertical dashed line**: "now" separator between past and future
   - **Green dots ahead**: Forward-looking forecast points (h1 to h48 from "now")
   - X-axis: timestamps. Y-axis: Load (MW).

4. **Metrics row**: 
   - MAE over displayed period (actuals vs stored h24 forecasts where both exist)
   - MAPE over displayed period
   - Number of forecast points displayed

5. **Wind domain**: Keep existing behavior unchanged (5-point scatter).

**PATTERN**: Current chart builder `make_forecast_chart()` (lines 251-306) — extend, don't rewrite from scratch.
**IMPORTS**: Add `forecast_store`, `rte_api.fetch_load_history`, `_chunk_dates`.
**GOTCHA**:
- RTE API fetch for 15 days = 1 chunk (< 180 days), fast.
- ForecastStore query returns all horizons — filter to h24 (or user-selected horizon) for the main line.
- Forward-looking forecasts (the green dots ahead of "now") still use `run_forecast()` existing logic.
- `st.cache_data(ttl=300)` for RTE API calls to avoid hammering the API on every Streamlit rerun.
- Session state: store fetched actuals + forecasts in `st.session_state` to survive Streamlit reruns.

**VALIDATE**: `streamlit run app/dashboard.py` — visual check: 15 days of blue actuals + green forecast line.

---

### Task 6: CREATE `tests/data/test_forecast_store.py`

**IMPLEMENT**: Unit tests for ForecastStore.

**Tests:**
- `test_create_and_upsert` — create store, insert 5 forecasts, verify count
- `test_query_by_date_range` — insert 10 days, query 3-day window, verify filtering
- `test_query_by_horizon` — insert multiple horizons, filter by horizon_h=24
- `test_upsert_idempotent` — insert same forecast twice, verify no duplicates (PK conflict handled)
- `test_get_coverage` — verify min/max timestamps
- `test_empty_store` — query on empty DB returns empty DataFrame, coverage returns None

**PATTERN**: Mirror `tests/weather/test_storage.py` if it exists, otherwise follow `tests/data/test_*.py` patterns.
**GOTCHA**: Use `tmp_path` pytest fixture for temporary DB files.

**VALIDATE**: `uv run pytest tests/data/test_forecast_store.py -v`

---

### Task 7: UPDATE `tests/data/test_rte_api.py` — Tests for Chunked Fetch

**IMPLEMENT**: Add tests for `_chunk_dates` and `fetch_load_history`.

**Tests:**
- `test_chunk_dates_single` — range < 180 days → 1 chunk
- `test_chunk_dates_multiple` — range = 400 days → 3 chunks
- `test_chunk_dates_exact` — range = 180 days → 1 chunk (boundary)
- `test_fetch_load_history_mock` — mock RTEClient.get, verify chunks are called, results concatenated

**VALIDATE**: `uv run pytest tests/data/test_rte_api.py -v`

---

## TESTING STRATEGY

### Unit Tests

- `tests/data/test_forecast_store.py` — CRUD operations, idempotency, filtering
- `tests/data/test_rte_api.py` — Chunking logic, API response parsing
- `tests/test_inference.py` — Verify `reference_time` parameter works correctly

### Integration Tests

- Backfill script with small date range (2 days) → verify ForecastStore populated
- Dashboard manual test: generate forecast, refresh, verify persistence

### Edge Cases

- Empty NWP for a timestamp → forecast skipped with warning, not crash
- RTE API returns partial data for a chunk → handle gracefully
- ForecastStore DB doesn't exist → created automatically
- Model not promoted (no champion) → clear error message in dashboard
- Actuals not available for recent hours (API lag) → forecast still generated with available lags

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
uv run ruff check src/ tests/ scripts/ app/
uv run ruff format --check src/ tests/ scripts/ app/
uv run pyright src/
```

### Level 2: Unit Tests

```bash
uv run pytest tests/data/test_forecast_store.py -v
uv run pytest tests/data/test_rte_api.py -v
uv run pytest tests/test_inference.py -v
uv run pytest tests/ -v  # full suite, no regressions
```

### Level 3: Integration Tests

```bash
# Small backfill test (2 days)
uv run python scripts/backfill_demand_forecasts.py --start 2025-01-01 --end 2025-01-03 --horizon 24

# Verify store populated
uv run python -c "
from windcast.data.forecast_store import ForecastStore
from pathlib import Path
store = ForecastStore(Path('data/forecast_store.db'))
print(store.get_coverage())
df = store.query('2025-01-01', '2025-01-03')
print(f'{len(df)} forecasts stored')
store.close()
"
```

### Level 4: Manual Validation

```bash
# Start dashboard
streamlit run app/dashboard.py

# Verify:
# 1. Select Demand model
# 2. Click "Generate Forecast" 
# 3. See 15 days of blue actuals + green forecast overlay
# 4. See MAE/MAPE metrics for the displayed period
# 5. Refresh page — data persists (from store, not session)
```

---

## ACCEPTANCE CRITERIA

- [ ] `ForecastStore` persists predictions in SQLite with upsert semantics
- [ ] `fetch_load_history()` fetches RTE actuals in 180-day chunks, unlimited range
- [ ] `build_inference_features()` accepts `reference_time` for arbitrary-date inference
- [ ] `backfill_demand_forecasts.py` generates forecasts for any date range, idempotent
- [ ] Dashboard shows 15 days of actuals (blue) + forecast overlay (green) for demand
- [ ] "Generate Forecast" fills gaps in store up to today + forward forecast
- [ ] Wind domain dashboard unchanged (no regression)
- [ ] All validation commands pass (ruff, pyright, pytest)
- [ ] MAE/MAPE computed on the displayed period where actuals and forecasts overlap

---

## COMPLETION CHECKLIST

- [ ] Task 1: ForecastStore created and tested
- [ ] Task 2: Chunked RTE API fetch working
- [ ] Task 3: Inference refactored for arbitrary timestamps
- [ ] Task 4: Backfill script working end-to-end
- [ ] Task 5: Dashboard shows 15-day timeline
- [ ] Task 6: ForecastStore unit tests passing
- [ ] Task 7: RTE API unit tests passing
- [ ] Full test suite: `uv run pytest tests/ -v` passes
- [ ] Lint: `uv run ruff check src/ tests/ scripts/ app/` clean
- [ ] Types: `uv run pyright src/` clean
- [ ] Manual: Dashboard visual validation

---

## NOTES

### Design Decision: SQLite vs Parquet for Forecast Store

SQLite chosen over Parquet because:
- Upsert semantics (idempotent backfill) — Parquet requires read-modify-write
- Query by date range + horizon without loading full file
- Consistent with `weather.db` / `weather_forecast.db` pattern in the project
- Dashboard reads are small (15 days × 5 horizons × 24 hours = ~1,800 rows)

### Design Decision: h24 as Primary Display Horizon

The dashboard main line shows h24 forecasts because:
- h24 = day-ahead, the most useful operational horizon
- Directly comparable to RTE D-1 forecast (the killer benchmark)
- User can select other horizons via dropdown

### Design Decision: RTE API for 2025+ Actuals

- Processed Parquet (`rte_france.parquet`) covers 2014-2024 (local éCO2mix files)
- For 2025+, actuals come from RTE API `consumption/v1/short_term?type=REALISED`
- Backfill script handles both sources seamlessly: Parquet for covered dates, API for gaps

### Risk: NWP Coverage Gap

- Historical Forecast API starts 2022-01-01
- For backfill 2025-01 → today: NWP is available (well within coverage)
- For live forecast (today + horizons): use `get_live_forecast()` (ephemeral, no cache)
- If NWP fetch fails for a timestamp, skip that forecast with warning (don't crash the backfill)

### Risk: RTE API Rate Limits

- No documented rate limits, but WattCast uses 0.5s courtesy sleep between chunks
- 15 days of actuals = 1 API call (well under 186-day limit)
- Full 2025 backfill = ~3 chunks ≈ 2 seconds of API time

### Backfill Duration Estimate

For 2025-01-01 → 2026-04-11 (~466 days):
- Actuals fetch: ~3 API chunks × 2s = ~6s
- NWP fetch: likely cached in `weather_forecast.db` if `fetch_historical_forecasts.py` was run
- Inference: ~466 days × 24h × 5 horizons = ~55,920 predictions
- At ~50 predictions/sec (in-process model) = ~19 minutes
- Optimization: batch by day (24 features + 5 predicts per day) = ~466 × 0.5s = ~4 minutes

### Forward Compatibility

The ForecastStore schema includes `domain` and `dataset` columns — ready for wind/solar forecast storage without schema migration.

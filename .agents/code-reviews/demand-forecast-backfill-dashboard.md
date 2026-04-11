# Code Review: Demand Forecast Backfill + 15-Day Dashboard

**Date**: 2026-04-11
**Reviewer**: Claude Code
**Feature**: Demand Forecast Backfill + 15-Day Dashboard Timeline

**Stats:**
- Files Modified: 4 (rte_api.py, inference.py, dashboard.py, test_rte_api.py)
- Files Added: 3 (forecast_store.py, backfill_demand_forecasts.py, test_forecast_store.py)
- Files Deleted: 0
- New lines: ~691
- Deleted lines: ~180

---

## Issues Found

```
severity: medium
file: scripts/backfill_demand_forecasts.py
line: 60
issue: --skip-existing flag defined but never used
detail: The CLI accepts --skip-existing (default True) but the code never
        queries the ForecastStore to check if a forecast already exists before
        generating it. The INSERT OR REPLACE semantics make the result correct
        (no duplicates), but the script wastes compute regenerating forecasts
        that already exist. For a 466-day backfill with 5 horizons (~56K
        predictions), this could mean ~4 minutes of unnecessary model inference.
suggestion: Either implement the skip logic by querying store.get_coverage()
            or store.query() before iterating, or remove the --skip-existing
            flag entirely since the upsert is inherently idempotent. Simplest
            fix: remove the flag, document idempotency in the docstring.
```

```
severity: medium
file: app/dashboard.py
line: 287
issue: Hardcoded ForecastStore path in run_demand_timeline
detail: store_path = Path("data/forecast_store.db") is hardcoded. All other
        data paths in the codebase are resolved via settings.data_dir or
        settings.processed_dir. If the working directory changes (e.g. running
        from a different location), this path will break silently (store_path
        won't exist → empty stored_df → no error, just missing data).
suggestion: Resolve via settings: store_path = get_settings().data_dir / "forecast_store.db"
            or add a FORECAST_STORE_DB constant consistent with the backfill
            script's --db-path default.
```

```
severity: low
file: app/dashboard.py
line: 132-144
issue: fetch_live_actuals function is now orphaned
detail: The old dashboard called fetch_live_actuals() for the demand domain's
        actuals overlay. The new code uses fetch_actuals_for_timeline() for
        the demand timeline, and the wind branch sets forecast_actuals=None.
        fetch_live_actuals is defined, cached, but never called from any code
        path in the current dashboard.
suggestion: Remove fetch_live_actuals or keep it if it's planned for wind
            domain live overlay in the future. If keeping, add a comment
            explaining its intended use.
```

```
severity: low
file: scripts/backfill_demand_forecasts.py
line: 188-189
issue: Repeated filter+tail on full DataFrame for every (timestamp, horizon) pair
detail: For each of the ~56K iterations, the code filters all_actuals with a
        boolean mask and takes .tail(200). Polars is fast at this, but the
        actuals DataFrame is sorted — a binary search for the cutoff index
        followed by a slice would be O(log n) vs O(n) per iteration.
suggestion: Pre-compute reference timestamps, use search_sorted or slice by
            index for O(1) tail extraction. For the current dataset size
            (96K rows), this is a ~2x speedup at most — not critical, but
            worth noting for larger datasets.
```

```
severity: low
file: src/windcast/data/forecast_store.py
line: 38
issue: upsert accepts list[dict] without validation
detail: If a dict is missing required keys (e.g., "target_timestamp"), the
        code will raise a KeyError with no context about which forecast
        caused the error. WeatherStorage avoids this by accepting a
        DataFrame (schema-validated). Not a bug per se, but the error
        message in a batch upsert of 120 items would be opaque.
suggestion: Accept for now — the only callers (backfill script and future
            dashboard backfill) construct dicts carefully. If this becomes
            a public API, consider accepting a DataFrame or validating
            required keys upfront with a clear error message.
```

---

## Positive Observations

- **SQL injection safe**: All queries use parameterized `?` placeholders — no string interpolation of user input into SQL.
- **Backward compatible**: `reference_time` parameter defaults to `None` (→ now) in inference.py — existing callers (dashboard `run_forecast`, CLI `main()`) continue working unchanged.
- **Consistent patterns**: ForecastStore faithfully mirrors WeatherStorage's structure (init/upsert/query/coverage/close), making the codebase predictable.
- **Good test coverage**: 10 new ForecastStore tests + 6 chunked fetch tests. All edge cases (empty store, idempotent upsert, boundary chunks) covered.
- **Wind domain regression-safe**: Dashboard cleanly branches demand vs wind — the wind path is untouched code.
- **MAPE division-by-zero protected**: `clip(lower_bound=1.0)` on load_mw prevents /0 in percentage error calculation.
- **Chunking is correct**: `_chunk_dates` generates contiguous non-overlapping ranges, and `unique("timestamp_utc")` handles any boundary duplicates from API overlap.

# Feature: Demand Forecast Backfill + 15-Day Dashboard

## Goal
Transform the demand dashboard from a single-shot 5-point forecast into a continuous forecast timeline showing 15 days of actuals with stored model forecasts.

## Context
- **PRD Reference**: `.claude/PRD.md` — Demand domain enhancement
- **Plan**: `.agents/plans/demand-forecast-backfill-dashboard.md`
- **Related Files**: rte_api.py, inference.py, dashboard.py, weather/storage.py

## Tasks

### Phase 1: Foundation — ForecastStore + RTE API Extension
- [x] Create `src/windcast/data/forecast_store.py` — SQLite storage with upsert/query ✓ 2026-04-11
- [x] Add `_chunk_dates()` and `fetch_load_history()` to `rte_api.py` ✓ 2026-04-11

### Phase 2: Inference Refactor
- [x] Refactor `_extract_nwp_at_horizon` to accept `reference_time` parameter ✓ 2026-04-11
- [x] Update `build_inference_features` to pass `reference_time` through ✓ 2026-04-11

### Phase 3: Backfill Script
- [x] Create `scripts/backfill_demand_forecasts.py` ✓ 2026-04-11

### Phase 4: Dashboard Rewrite
- [x] Rewrite demand chart: 15-day actuals + stored forecasts + Generate Forecast button ✓ 2026-04-11

### Phase 5: Testing & Validation
- [x] Create `tests/data/test_forecast_store.py` — Unit tests ✓ 2026-04-11
- [x] Add chunked fetch tests to `tests/data/test_rte_api.py` ✓ 2026-04-11
- [x] Run full validation: ruff + pyright + pytest ✓ 2026-04-11

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/windcast/data/forecast_store.py` | Create | SQLite forecast storage |
| `src/windcast/data/rte_api.py` | Modify | Add chunked historical fetch |
| `scripts/inference.py` | Modify | Add reference_time parameter |
| `scripts/backfill_demand_forecasts.py` | Create | Batch forecast generation |
| `app/dashboard.py` | Modify | 15-day timeline view |
| `tests/data/test_forecast_store.py` | Create | ForecastStore unit tests |
| `tests/data/test_rte_api.py` | Modify | Chunked fetch tests |

## Notes
- SQLite chosen over Parquet for upsert semantics (idempotent backfill)
- h24 as primary display horizon (day-ahead, comparable to RTE D-1)
- ForecastStore schema includes domain/dataset columns for forward compatibility

## Completion
- **Started**: 2026-04-11
- **Completed**: 2026-04-11
- **Commit**: (pending /commit)

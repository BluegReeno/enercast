# Feature: Dashboard Live Actuals for RTE France

## Goal
Add live observed load data from RTE eCO2mix API to the Streamlit dashboard for the demand domain.

## Context
- **Plan**: `.agents/plans/dashboard-live-actuals-rte.md`
- **Related Files**: `app/dashboard.py`, `src/windcast/config.py`, `src/windcast/data/rte_api.py` (new)

## Tasks

### Phase 1: Foundation
- [x] Task 1: Add httpx dependency to pyproject.toml ✓ 2026-04-11
- [x] Task 2: Add optional RTE credentials to WindCastSettings ✓ 2026-04-11
- [x] Task 3: Create sync RTE API client (`src/windcast/data/rte_api.py`) ✓ 2026-04-11

### Phase 2: Dashboard Integration
- [x] Task 4: Add `fetch_live_actuals()` cached function to dashboard ✓ 2026-04-11
- [x] Task 5: Modify `make_forecast_chart()` for optional actuals trace ✓ 2026-04-11
- [x] Task 6: Wire actuals into main area + sidebar indicator ✓ 2026-04-11

### Phase 3: Testing & Validation
- [x] Task 7: Create unit tests (`tests/data/test_rte_api.py`) ✓ 2026-04-11
- [x] Task 8: Full validation (ruff + pyright + pytest) ✓ 2026-04-11

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `pyproject.toml` | Modify | Add httpx dependency |
| `src/windcast/config.py` | Modify | Add optional RTE credentials |
| `src/windcast/data/rte_api.py` | Create | Sync RTE API client |
| `app/dashboard.py` | Modify | Add actuals fetching + chart overlay |
| `tests/data/test_rte_api.py` | Create | Unit tests for API client |

## Notes

## Completion
- **Started**: 2026-04-11
- **Completed**: 2026-04-11
- **Commit**: (pending /commit)

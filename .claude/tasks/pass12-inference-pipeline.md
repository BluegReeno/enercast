# Feature: Pass 12 — Inference Pipeline

## Goal
End-to-end inference script: fetch live NWP → build features → load model → structured JSON predictions.

## Context
- **PRD Reference**: `.claude/PRD.md` — Model Serving section
- **Plan**: `.claude/plans/pass12-inference-pipeline.md`
- **Related Files**: `src/windcast/data/open_meteo.py`, `src/windcast/weather/__init__.py`, `scripts/train.py`, `src/windcast/features/`

## Tasks

### Phase 1: Weather Module — Live Forecast Fetcher
- [x] Task 1: Add `fetch_forecast_weather()` to `src/windcast/data/open_meteo.py` ✓ 2026-04-10
- [x] Task 2: Add `get_live_forecast()` to `src/windcast/weather/__init__.py` ✓ 2026-04-10

### Phase 2: Inference Script
- [x] Task 3: Create `scripts/inference.py` with `build_inference_features()` helper ✓ 2026-04-10
- [x] Task 4: Create `tests/test_inference.py` ✓ 2026-04-10

### Phase 3: Validation
- [x] Task 5: Run full validation (ruff, pyright, pytest) ✓ 2026-04-10

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/windcast/data/open_meteo.py` | Modify | Add `FORECAST_URL` + `fetch_forecast_weather()` |
| `src/windcast/weather/__init__.py` | Modify | Add `get_live_forecast()` + export |
| `scripts/inference.py` | Create | Main inference CLI script |
| `tests/test_inference.py` | Create | Tests for inference pipeline |

## Notes
- Direct model loading mode (default) + server mode (`--serve-url`)
- Reuse existing feature builders for training/inference parity
- Short HTTP cache TTL (1h) for live forecast data

## Completion
- **Started**: 2026-04-10
- **Completed**: 2026-04-10
- **Commit**: (pending /commit)

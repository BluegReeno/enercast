# Feature: Pass 13 — Streamlit Demo Dashboard

## Goal
Create a minimal Streamlit dashboard for the WN live demo: pick domain/model/horizons → generate real-time forecast with Plotly chart + model metadata sidebar.

## Context
- **PRD Reference**: `.claude/PRD.md` Passes 11-13
- **Plan**: `.claude/plans/pass13-streamlit-dashboard.md`
- **Related Files**: `scripts/inference.py`, `scripts/compare_runs.py`, `src/windcast/config.py`, `src/windcast/weather/__init__.py`

## Tasks

### Phase 1: Dependencies
- [x] Add `streamlit>=1.40` and `plotly>=5.0` to `pyproject.toml` + `uv sync` ✓ 2026-04-11

### Phase 2: Dashboard Implementation
- [x] Create `app/dashboard.py` — single-file Streamlit app (~250 lines) ✓ 2026-04-11

### Phase 3: Validation
- [x] Lint, format, type-check, and test suite pass (336/336) ✓ 2026-04-11

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `pyproject.toml` | Modify | Add streamlit + plotly deps |
| `app/dashboard.py` | Create | Main Streamlit dashboard |

## Notes
- Import `build_inference_features` from `scripts.inference` via sys.path hack (acceptable for demo)
- Single file, no pages — YAGNI for a presentation demo

## Completion
- **Started**: 2026-04-11
- **Completed**: 2026-04-11
- **Commit**: (link to commit when done)

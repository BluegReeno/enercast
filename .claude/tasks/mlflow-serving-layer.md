# Feature: MLflow Native Multi-Horizon Serving

## Goal
Replace per-child-run model loading with a single `HorizonRouter` PythonModel wrapper that bundles all horizon models and routes via `params["horizon"]`, enabling native `mlflow models serve`.

## Context
- **PRD Reference**: `.claude/PRD.md` — Model Serving
- **Plan**: `.claude/plans/mlflow-serving-layer.md`
- **Related Files**: `src/windcast/models/`, `src/windcast/training/harness.py`, `app/dashboard.py`, `scripts/inference.py`

## Tasks

### Phase 1: HorizonRouter PythonModel
- [x] Task 1: Create `src/windcast/models/horizon_router.py` — PythonModel wrapper + `build_router_signature()` ✓ 2026-04-11

### Phase 2: Training Integration
- [x] Task 2: Update `src/windcast/training/harness.py` — collect horizon URIs, log HorizonRouter, register with champion alias ✓ 2026-04-11

### Phase 3: Consumer Refactor
- [x] Task 3: Update `app/dashboard.py` — champion mode with HorizonRouter + params routing ✓ 2026-04-11
- [x] Task 4: Update `scripts/inference.py` — add params support to both predict modes ✓ 2026-04-11

### Phase 4: Tests
- [x] Task 5: Create `tests/models/test_horizon_router.py` — router unit tests ✓ 2026-04-11
- [x] Task 6: Update `tests/training/test_model_registry.py` — router registration integration test ✓ 2026-04-11
- [x] Task 7: Update `tests/test_inference.py` — params in server payload test ✓ 2026-04-11

### Phase 5: Validation
- [x] Task 8: Full validation — ruff, pyright, pytest ✓ 2026-04-11

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/windcast/models/horizon_router.py` | Create | HorizonRouter PythonModel + signature builder |
| `src/windcast/training/harness.py` | Modify | Collect URIs, log router, register champion |
| `app/dashboard.py` | Modify | Champion mode, params routing, serve URL |
| `scripts/inference.py` | Modify | Add params to predict functions |
| `tests/models/test_horizon_router.py` | Create | Router unit tests |
| `tests/training/test_model_registry.py` | Modify | Router registration test |
| `tests/test_inference.py` | Modify | Server payload params test |

## Notes
- ParamSchema is REQUIRED — without it, params silently dropped (MLflow #18522)
- Use `runs:/` URIs (not `models:/`) inside the router — logged in same session before registry alias
- AutoGluon sub-models are already pyfunc wrappers — transparent chaining

## Completion
- **Started**: 2026-04-11
- **Completed**: 2026-04-11
- **Commit**: (pending /commit)

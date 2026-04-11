# Feature: Decouple Model Promotion from Training

## Goal
Separate "train & log" from "register & promote" — the clean MLflow workflow: train → evaluate → compare → promote.

## Context
- **Plan**: `.agents/plans/decouple-model-promotion.md`
- **Related Files**: `src/windcast/training/harness.py`, `scripts/train.py`, `scripts/promote_model.py`, `tests/training/test_model_registry.py`, `README.md`

## Tasks

### Phase 1: Always log HorizonRouter
- [x] Remove `register_model_name` param from `run_training()`, always log HorizonRouter when `log_models=True` ✓ 2026-04-11

### Phase 2: Clean up train.py
- [x] Remove `--register`, `--model-name` flags and `register_model_name` kwarg from `run_training()` call ✓ 2026-04-11

### Phase 3: Create promote_model.py
- [x] Create standalone `scripts/promote_model.py` with --run-id, --experiment, --list modes ✓ 2026-04-11

### Phase 4: Update tests
- [x] Update existing registration tests + add new promote test ✓ 2026-04-11

### Phase 5: Update README
- [x] Add promote_model.py to Quick Start workflow ✓ 2026-04-11

### Validation
- [x] ruff check + pyright clean ✓ 2026-04-11
- [x] All tests pass (356/356) ✓ 2026-04-11

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/windcast/training/harness.py` | Modify | Remove register_model_name, always log HorizonRouter |
| `scripts/train.py` | Modify | Remove --register and --model-name flags |
| `scripts/promote_model.py` | Create | Standalone promotion script |
| `tests/training/test_model_registry.py` | Modify | Update tests for new API |
| `README.md` | Modify | Add promote_model.py to Quick Start |

## Completion
- **Started**: 2026-04-11
- **Completed**: 2026-04-11
- **Commit**: (link to commit when done)

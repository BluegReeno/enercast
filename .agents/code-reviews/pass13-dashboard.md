# Code Review: Pass 13 — Streamlit Dashboard

**Date**: 2026-04-11
**Reviewer**: Claude Code
**Scope**: `app/dashboard.py` (new), `pyproject.toml` (modified)

---

**Stats:**

- Files Modified: 1 (`pyproject.toml`)
- Files Added: 1 (`app/dashboard.py`)
- Files Deleted: 0
- New lines: ~305
- Deleted lines: 0

---

## Issues

```
severity: critical
file: app/dashboard.py
line: 74
issue: load_model uses parent run_id but model artifacts are on child runs
detail: The sidebar selects a parent run (filtered by `enercast.run_type = 'parent'`),
        and `load_model()` constructs `runs:/{run_id}/model_h{horizon:02d}`. But models
        are logged on *child* (nested) runs, not the parent — see
        `src/windcast/training/backends.py:67-68`. This URI will 404 at runtime.
        The correct approach is to find the child run for each horizon via
        `MlflowClient.search_runs(filter_string="tags.mlflow.parentRunId = '{parent_id}'")`,
        extract the child run_id for the matching horizon, then load from there.
suggestion: Add a cached helper that maps (parent_run_id, horizon) → child_run_id,
            then load via `runs:/{child_run_id}/model_h{horizon:02d}`. Pattern exists
            in `scripts/evaluate.py:91-112` (_load_models_from_run).
```

```
severity: medium
file: app/dashboard.py
line: 126
issue: last_ts computed inside the loop but never changes
detail: `actuals["timestamp_utc"].max()` is constant for all horizons. Computing it
        inside the loop is wasteful (minor perf) but more importantly, if actuals were
        ever empty (edge case), this would fail repeatedly without a clear error at the
        right place.
suggestion: Move `last_ts = actuals["timestamp_utc"].max()` before the loop, and add
            a guard: `if last_ts is None: raise ValueError("Empty actuals")`.
```

```
severity: medium
file: app/dashboard.py
line: 246
issue: Fragile NaN check via string comparison
detail: `str(mae_val) != "nan"` works for float NaN but would miss `pd.NA` or numpy
        NaN edge cases. Using `pd.notna(mae_val)` is the standard pandas idiom and
        handles all NA types.
suggestion: Replace `if mae_val is not None and str(mae_val) != "nan":` with
            `if pd.notna(mae_val):` (add `import pandas as pd` at the top).
```

```
severity: low
file: app/dashboard.py
line: 99-104
issue: Duplicated actuals path / tail_rows logic from scripts/inference.py
detail: `_resolve_actuals_path()` and `_tail_rows_for_domain()` in
        `scripts/inference.py:159-170` implement the same logic. Not a bug, but
        if defaults change in one place they'll diverge.
suggestion: Acceptable for a demo app. If the dashboard outlives the presentation,
            refactor these helpers into a shared module.
```

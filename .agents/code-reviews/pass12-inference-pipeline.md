# Code Review: Pass 12 — Inference Pipeline

**Date**: 2026-04-10
**Reviewer**: Claude (automated)
**Scope**: Pass 12 inference pipeline implementation

## Stats

- Files Modified: 3 (`open_meteo.py`, `weather/__init__.py`, `STATUS.md`)
- Files Added: 2 (`scripts/inference.py`, `tests/test_inference.py`)
- Files Deleted: 0
- New lines: ~430
- Deleted lines: 1

---

## Issues Found

### Issue 1

```
severity: medium
file: scripts/inference.py
line: 302-303
issue: Redundant timestamp_utc.max() call — last_ts duplicates last_actual_ts
detail: Line 275 already computes `last_actual_ts = actuals_df["timestamp_utc"].max()` but
        line 302 recomputes `last_ts = actuals_df["timestamp_utc"].max()` again. This is a
        trivial cost (in-memory max) but is a clarity issue — two variables for the same value.
suggestion: Replace line 302 with `last_ts = last_actual_ts` or just use `last_actual_ts`
            directly on line 305.
```

### Issue 2

```
severity: medium
file: scripts/inference.py
line: 303
issue: Mid-function import of `timedelta`
detail: `from datetime import timedelta` is imported inside main() at line 303, but
        `from datetime import UTC, datetime` is already imported at the top of the file (line 19).
        timedelta should be imported alongside them at the top. Mid-function imports are the
        pattern for heavy optional deps (mlflow, requests), not stdlib.
suggestion: Add `timedelta` to the top-level import: `from datetime import UTC, datetime, timedelta`
            and remove line 303.
```

### Issue 3

```
severity: medium
file: scripts/inference.py
line: 298
issue: _predict_direct(model, X) called when model is None in serve mode
detail: When --serve-url is set, `model` stays None (line 261-263). The ternary on line 298
        correctly gates on `args.serve_url`, so `_predict_direct(model, X)` is only called
        when model is not None. However, the type system doesn't know this — `model` is
        `Any | None`, and `_predict_direct` takes `Any`. Not a runtime bug, but fragile:
        if someone rearranges the ternary or removes the guard, they get a cryptic
        `NoneType has no attribute predict` at runtime.
suggestion: Add an assertion before the ternary:
            `assert model is not None or args.serve_url, "No model loaded and no serve URL"`
            or restructure to make the None case impossible.
```

### Issue 4

```
severity: low
file: scripts/inference.py
line: 159
issue: _resolve_actuals_path uses hardcoded "kelmarsh_kwf1" for wind
detail: Wind domain always resolves to `kelmarsh_kwf1.parquet`, ignoring the `dataset`
        argument. If a second wind dataset is added (e.g., hill_of_towie), inference
        would silently use Kelmarsh data. The train.py script handles this with
        `--turbine-id` + dataset-aware paths.
suggestion: Accept this for now (single wind dataset in scope), but consider adding
            a --turbine-id arg or using the dataset name in the path when a second
            wind dataset becomes active.
```

### Issue 5

```
severity: low
file: scripts/inference.py
line: 273
issue: Full Parquet read then tail() is wasteful for large files
detail: `pl.read_parquet(actuals_path)` loads the entire processed Parquet into memory
        just to `.tail(N)` a few hundred rows. For Kelmarsh (~175k rows, ~5 MB) this is
        fine. For RTE France (~96k rows, <1 MB) also fine. But if datasets grow, this
        becomes unnecessary I/O.
suggestion: Acceptable for the demo. A future optimization could use Parquet row-group
            metadata to skip early rows, or store a "latest actuals" cache file.
```

### Issue 6

```
severity: low
file: tests/test_inference.py
line: 269
issue: Test imports build_horizon_desc from scripts.inference (re-export), not from source
detail: `from scripts.inference import UNIT_BY_DOMAIN, build_horizon_desc` — build_horizon_desc
        is not defined in scripts/inference.py, it's imported from windcast.training.harness.
        The test works because Python re-exports imports, but it's misleading to import
        a harness function via a script module. The test at line 317 correctly imports from
        windcast.training.harness.
suggestion: Change line 269 to `from windcast.training.harness import build_horizon_desc`
            and import UNIT_BY_DOMAIN separately: `from scripts.inference import UNIT_BY_DOMAIN`.
```

---

## Summary

No critical or high-severity issues. The implementation is clean and follows existing codebase patterns well. The 3 medium issues (redundant variable, mid-function import, None model guard) are worth fixing before commit. The 3 low issues are acceptable technical debt for a demo-scoped feature.

**Verdict: Fix medium issues, then ready for commit.**

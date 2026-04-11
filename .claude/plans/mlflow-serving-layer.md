# Feature: MLflow Native Multi-Horizon Serving

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files.

## Feature Description

Replace the dashboard's per-child-run model loading with a single MLflow `PythonModel` wrapper that bundles all 5 horizon models and routes via the `params` argument. This enables native `mlflow models serve` with zero custom server code — one model, one endpoint, one process, five horizons.

**Currently** the dashboard:
1. Lists all parent runs in the sidebar (user picks one manually)
2. Loads 5 child run models via `mlflow.pyfunc.load_model("runs:/{child_id}/model_h{NN}")`
3. Calls `model.predict()` in-process per horizon

**After this feature:**
1. Training logs a `HorizonRouter` pyfunc wrapper that contains all 5 horizon sub-models
2. The wrapper is registered as a single model `enercast-kelmarsh-xgboost@champion`
3. `mlflow models serve -m "models:/enercast-kelmarsh-xgboost@champion" --env-manager local -p 5001` starts a REST API
4. Any consumer (dashboard, cron, curl) POSTs to `/invocations` with `{"dataframe_split": {...}, "params": {"horizon": 24}}`
5. The dashboard can also load the wrapper in-process: `model.predict(X, params={"horizon": 24})`

**MLflow official pattern**: [Serving Multiple Models on a Single Endpoint with a Custom PyFunc Model](https://mlflow.org/docs/latest/traditional-ml/serving-multiple-models-with-pyfunc/index.html) — exact same architecture, uses `params` routing with `ParamSchema`.

## User Story

As an ML engineer deploying EnerCast to production,
I want a single registered MLflow model that serves all 5 forecast horizons,
So that `mlflow models serve` gives me a production-ready REST API with zero custom code.

## Problem Statement

The current dashboard loads models directly from MLflow child run artifacts. This has 3 issues:
1. **No champion isolation** — user must manually select a parent run; no guarantee it's the best model
2. **Not deployable** — 5 separate `load_model()` calls can't be served via `mlflow models serve`
3. **Tight coupling** — dashboard code mixes UI logic with model loading and MLflow child run traversal

## Solution Statement

Create a `HorizonRouter(mlflow.pyfunc.PythonModel)` wrapper that:
1. Stores references to per-horizon sub-model URIs at log time
2. Loads all sub-models eagerly in `load_context()` at serve/load time
3. Routes predictions via `params={"horizon": 24}` in `predict()`
4. Is logged with a `ModelSignature` including `ParamSchema([ParamSpec("horizon", "long", 1)])`

This is the canonical MLflow multi-model pattern (see DOWModel tutorial). One registered model, one `mlflow models serve` process, five internal sub-models.

## Feature Metadata

**Feature Type**: Enhancement
**Estimated Complexity**: Medium
**Primary Systems Affected**: `src/windcast/models/horizon_router.py` (new), `src/windcast/training/harness.py`, `app/dashboard.py`, `scripts/inference.py`
**Dependencies**: None (all within existing MLflow)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ BEFORE IMPLEMENTING

- `src/windcast/models/autogluon_pyfunc.py` — **Existing PythonModel wrapper pattern**. The `HorizonRouter` follows the same structure: `load_context()` + `predict()`. Read this file to match the style.
- `src/windcast/training/harness.py` (lines 304-408) — **Per-horizon training loop**. Each iteration logs a child model via `backend.log_model()` and returns a `model_uri`. We need to collect these URIs to pass to the `HorizonRouter`.
- `src/windcast/training/harness.py` (lines 489-497) — **Current registration logic**. Registers only best-MAE horizon. Must be replaced with `HorizonRouter` registration.
- `src/windcast/training/backends.py` (lines 52-74) — `XGBoostBackend.log_model()`. Returns `model_uri` string like `"models:/model_h24/1"`. These URIs become the sub-model references in the router.
- `src/windcast/training/backends.py` (lines 138-161) — `AutoGluonBackend.log_model()`. Same pattern, returns pyfunc URI.
- `app/dashboard.py` (lines 70-92) — `load_models_for_run()` — traverses child runs to load models. Replace with single `mlflow.pyfunc.load_model()` + `params`.
- `app/dashboard.py` (lines 99-157) — `run_forecast()` — calls `model.predict(features.to_pandas())` per horizon. Replace with `model.predict(features.to_pandas(), params={"horizon": h})`.
- `scripts/inference.py` (lines 119-156) — `_load_model_direct()` and `_predict_server()`. Both need `params` support.
- `tests/training/test_model_registry.py` — Existing registry tests. Pattern to follow for new tests.

### New Files to Create

- `src/windcast/models/horizon_router.py` — `HorizonRouter(PythonModel)` wrapper (~60 lines)
- `tests/models/test_horizon_router.py` — Unit tests for the router (~80 lines)

### Relevant Documentation

**MLflow Official Tutorial — Multi-Model Serving**:
- Guide: `https://mlflow.org/docs/latest/traditional-ml/serving-multiple-models-with-pyfunc/index.html`
- Notebook: `https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/serving-multiple-models-with-pyfunc/notebooks/MME_Tutorial/`

**MLflow Model Signatures with Params**:
- `https://mlflow.org/docs/latest/ml/model/signatures/`
- `ParamSchema` + `ParamSpec` from `mlflow.types.schema`

**Critical Gotcha — Issue #18522**: `params` are **silently ignored** if the model signature does not include a `ParamSchema`. MUST define `ParamSchema([ParamSpec("horizon", DataType.long, 1)])` in the `ModelSignature`.

**MLflow `models serve` REST format**:
```bash
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}, "params": {"horizon": 24}}'
```
Response: `{"predictions": [1234.5]}`

### Patterns to Follow

**Existing PythonModel pattern** (from `autogluon_pyfunc.py`):
```python
class AutoGluonPyfuncWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.predictor = TabularPredictor.load(context.artifacts["ag_predictor"])

    def predict(self, context, model_input, params=None):
        return self.predictor.predict(model_input).values
```

**MLflow DOWModel pattern** (from official tutorial):
```python
class DOWModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_uris):
        self.model_uris = model_uris

    def load_context(self, context):
        self.models = {
            dow: mlflow.sklearn.load_model(uri)
            for dow, uri in self.model_uris.items()
        }

    def predict(self, context, model_input, params):
        dow = params.get("dow")
        return self.models[dow].predict(model_input)
```

**Our adaptation** — same pattern, `horizon` instead of `dow`, XGBoost/AutoGluon instead of sklearn.

---

## IMPLEMENTATION PLAN

### Phase 1: HorizonRouter PythonModel

Create the pyfunc wrapper that bundles N horizon sub-models and routes via `params["horizon"]`.

### Phase 2: Training Integration

Modify `run_training()` to:
1. Collect all per-horizon model URIs during the child loop
2. Create a `HorizonRouter` with those URIs
3. Log it as a single pyfunc model with `ParamSchema`
4. Register it with `champion` alias

### Phase 3: Dashboard + Inference Refactor

Update consumers to use `params={"horizon": h}` instead of loading per-child models.

### Phase 4: Tests

Unit tests for HorizonRouter + integration test for registration + updated inference tests.

---

## STEP-BY-STEP TASKS

### Task 1: CREATE `src/windcast/models/horizon_router.py` — PythonModel wrapper

- **IMPLEMENT**: Create a `HorizonRouter(mlflow.pyfunc.PythonModel)` that wraps multiple horizon sub-models:

  ```python
  """Multi-horizon PythonModel wrapper for MLflow serving.

  Bundles N per-horizon models into a single servable artifact.
  Route predictions via params={"horizon": <int>}.

  Based on the official MLflow pattern:
  https://mlflow.org/docs/latest/traditional-ml/serving-multiple-models-with-pyfunc/
  """

  class HorizonRouter(mlflow.pyfunc.PythonModel):
      def __init__(self, horizon_model_uris: dict[int, str] | None = None):
          # horizon_model_uris: {1: "runs:/.../model_h01", 6: "runs:/.../model_h06", ...}
          # Stored at log time, resolved at load time.
          self.horizon_model_uris = horizon_model_uris or {}
          self.models: dict[int, Any] = {}

      def load_context(self, context):
          # Called once by mlflow.pyfunc.load_model() or mlflow models serve
          for horizon, uri in self.horizon_model_uris.items():
              self.models[horizon] = mlflow.pyfunc.load_model(uri)

      def predict(self, context, model_input, params=None):
          if params is None or "horizon" not in params:
              raise ValueError(
                  "params must include 'horizon' key. "
                  f"Available horizons: {sorted(self.models.keys())}"
              )
          horizon = int(params["horizon"])
          model = self.models.get(horizon)
          if model is None:
              raise ValueError(
                  f"No model for horizon {horizon}. "
                  f"Available: {sorted(self.models.keys())}"
              )
          return model.predict(model_input)
  ```

  Also add a helper function to build the `ModelSignature` with `ParamSchema`:

  ```python
  def build_router_signature(
      feature_columns: list[str],
      default_horizon: int = 1,
  ) -> ModelSignature:
      from mlflow.models import ModelSignature
      from mlflow.types.schema import ColSpec, DataType, ParamSchema, ParamSpec, Schema

      input_schema = Schema([ColSpec(DataType.double, col) for col in feature_columns])
      output_schema = Schema([ColSpec(DataType.double, "prediction")])
      params_schema = ParamSchema([
          ParamSpec("horizon", DataType.long, default_horizon),
      ])
      return ModelSignature(
          inputs=input_schema,
          outputs=output_schema,
          params=params_schema,
      )
  ```

- **PATTERN**: Mirror `autogluon_pyfunc.py` structure
- **GOTCHA**: The `horizon_model_uris` dict is serialized by MLflow's cloudpickle when logging the PythonModel. The URIs must be resolvable at load time (they point to logged model artifacts within MLflow runs, which persist). Do NOT use registry URIs (`models:/...`) here — use the `runs:/` URIs returned by `backend.log_model()`, because the wrapper is logged in the same training session before any registry alias is set.
- **GOTCHA**: `ParamSchema` is REQUIRED. Without it, `params` is silently `None` at predict time (MLflow issue #18522).
- **VALIDATE**: `uv run ruff check src/windcast/models/horizon_router.py && uv run pyright src/windcast/models/horizon_router.py`

### Task 2: UPDATE `src/windcast/training/harness.py` — log HorizonRouter + register

- **IMPLEMENT**: Modify `run_training()` to collect horizon model URIs and log a `HorizonRouter`:

  **Step 2a**: Add `horizon_model_uris: dict[int, str] = {}` before the horizon loop (around line 309).

  **Step 2b**: Inside the horizon loop, after `backend.log_model()` (line 401-403), collect the URI:
  ```python
  model_uri = None
  if log_models:
      model_uri = backend.log_model(model, X_val, y_pred, h)
  if model_uri:
      horizon_model_uris[h] = model_uri
  ```

  **Step 2c**: After the horizon loop (replace lines 489-497), log the `HorizonRouter` on the parent run and register it:
  ```python
  # Log multi-horizon router as a single servable model
  if log_models and horizon_model_uris:
      from windcast.models.horizon_router import HorizonRouter, build_router_signature

      router = HorizonRouter(horizon_model_uris)
      # Pre-warm so load_context can be called for signature validation
      router.load_context(None)

      # Build signature from the last horizon's feature columns
      # (all horizons share the same canonical feature names after renaming)
      last_h = max(horizon_model_uris.keys())
      if has_nwp_horizons:
          sig_cols, _ = resolve_horizon_features(df.columns, fs.columns, last_h)
          sig_feature_names = [
              rename_map.get(c, c) if (_, rename_map := resolve_horizon_features(df.columns, fs.columns, last_h)) else c
              for c in sig_cols
          ]
      else:
          sig_feature_names = available_non_nwp + [
              c for c in fs.columns if c.startswith("nwp_") and c in df.columns
          ]
      # Actually, use the canonical column names from the feature set
      sig_feature_names = [c for c in fs.columns if not c.startswith("nwp_") and c in df.columns]
      sig_feature_names += [c for c in fs.columns if c.startswith("nwp_")]

      signature = build_router_signature(sig_feature_names, default_horizon=horizons[0])

      router_uri = mlflow.pyfunc.log_model(
          name="horizon_router",
          python_model=router,
          signature=signature,
      )

      if register_model_name:
          mv = mlflow.register_model(router_uri.model_uri, register_model_name)
          client.set_registered_model_alias(register_model_name, "champion", str(mv.version))
          logger.info(
              "Registered %s v%s @champion (%d horizons, best MAE=%.1f)",
              register_model_name,
              mv.version,
              len(horizon_model_uris),
              best_mae,
          )
  ```

  **NOTE ON SIGNATURE COLUMNS**: The signature feature names must be the **canonical** names (e.g., `nwp_wind_speed_100m`, NOT `nwp_wind_speed_100m_h24`). This matches what the sub-models expect after `resolve_horizon_features()` renaming. The simplest approach: use the feature set's `.columns` list directly, filtering to only columns that exist. This is the same set that every horizon's model was trained on.

- **PATTERN**: Follows existing `mlflow.register_model()` + `set_registered_model_alias()` at line 489-497. The `HorizonRouter` replaces the single best-model registration.
- **GOTCHA**: `router.load_context(None)` pre-warms the router so sub-models are loaded when we infer the signature. The `context` parameter is unused in our `load_context` (we load from URIs, not from `context.artifacts`).
- **GOTCHA**: The feature column list for the signature should be the CANONICAL names from `fs.columns`, not the horizon-suffixed names. All sub-models expect the same canonical columns after renaming.
- **VALIDATE**: `uv run ruff check src/windcast/training/harness.py && uv run pyright src/windcast/training/harness.py`

### Task 3: UPDATE `app/dashboard.py` — use HorizonRouter with params

- **IMPLEMENT**: Simplify the dashboard to load a single `HorizonRouter` model and use `params`:

  **Step 3a**: Replace `load_models_for_run()` (lines 70-92) with a simpler loader that loads the router:
  ```python
  @st.cache_resource
  def load_router_model(model_uri: str) -> object:
      """Load the HorizonRouter model — cached as singleton."""
      settings = get_settings()
      mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
      return mlflow.pyfunc.load_model(model_uri)
  ```

  **Step 3b**: In `run_forecast()` (lines 99-157), replace the per-horizon `model.predict(features.to_pandas())` with:
  ```python
  prediction = float(router.predict(features.to_pandas(), params={"horizon": h})[0])
  ```

  **Step 3c**: In the sidebar, add two modes:
  - **Champion mode** (new default): loads from `models:/{experiment_model_name}@champion`. No run selector needed.
  - **Run mode** (existing): keeps the run selector for debugging/comparison. Falls back to current `load_models_for_run()` logic.

  Add constants:
  ```python
  MODEL_NAME_BY_DOMAIN: dict[str, str] = {
      "wind": "enercast-kelmarsh-xgboost",
      "demand": "enercast-rte_france-xgboost",
  }
  ```

  **Step 3d**: When in champion mode and `--serve-url` is set (env var `ENERCAST_SERVE_URL`), POST to `/invocations` instead of in-process predict:
  ```python
  def predict_via_server(serve_url: str, X_pd: pd.DataFrame, horizon: int) -> float:
      payload = {
          "dataframe_split": {
              "columns": X_pd.columns.tolist(),
              "data": X_pd.values.tolist(),
          },
          "params": {"horizon": horizon},
      }
      resp = requests.post(f"{serve_url}/invocations", json=payload)
      resp.raise_for_status()
      result = resp.json()
      if isinstance(result, dict) and "predictions" in result:
          return float(result["predictions"][0])
      return float(result[0])
  ```

- **PATTERN**: Mirror `_predict_server()` from `scripts/inference.py:134-156`, adding `"params"` to the payload.
- **GOTCHA**: Keep the existing direct-load fallback for cases where no champion model is registered yet (first run before training with `--register-model`).
- **VALIDATE**: `uv run ruff check app/dashboard.py && uv run pyright app/dashboard.py`

### Task 4: UPDATE `scripts/inference.py` — support params in both modes

- **IMPLEMENT**:

  **Step 4a**: Update `_predict_server()` (lines 134-156) to include `params` in the payload:
  ```python
  def _predict_server(serve_url: str, X: pl.DataFrame, horizon: int) -> float:
      X_pd = X.to_pandas()
      payload = {
          "dataframe_split": {
              "columns": X_pd.columns.tolist(),
              "data": X_pd.values.tolist(),
          },
          "params": {"horizon": horizon},
      }
      url = f"{serve_url.rstrip('/')}/invocations"
      resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
      resp.raise_for_status()
      result = resp.json()
      if isinstance(result, dict) and "predictions" in result:
          return float(result["predictions"][0])
      if isinstance(result, list):
          return float(result[0])
      return float(result)
  ```
  Note: `horizon` parameter added to the function signature.

  **Step 4b**: Update `_predict_direct()` (lines 127-130) to pass params:
  ```python
  def _predict_direct(model: Any, X: pl.DataFrame, horizon: int) -> float:
      X_pd = X.to_pandas()
      predictions = model.predict(X_pd, params={"horizon": horizon})
      return float(predictions[0])
  ```

  **Step 4c**: Update `main()` callers of `_predict_server` and `_predict_direct` to pass `args.horizon`.

  **Step 4d**: Update the `--model-uri` default resolution to point to the registered router model instead of per-horizon:
  ```python
  # Before: model_uri = f"models:/{model_name}@{args.model_alias}"
  # After: same URI, but now it resolves to the HorizonRouter, not a single-horizon model
  ```
  No change needed in the URI — the registered model IS the router now.

- **GOTCHA**: `_predict_direct` currently calls `model.predict(X_pd)` without `params`. When loading a `HorizonRouter`, this would fail because `params["horizon"]` is required. Adding `params={"horizon": horizon}` makes it work with both the old per-horizon models AND the new router (old models ignore unknown params).
- **GOTCHA**: Actually, old per-horizon models (loaded via `runs:/` URI) do NOT accept `params` — they're raw XGBoost models. So the code must handle both: if model is a `HorizonRouter` (loaded from registry), pass params; if it's a raw model (loaded from run), don't. Simplest approach: try with params, catch TypeError, retry without.

  Better approach: detect model type. If loaded from `models:/` registry URI, it's the router → pass params. If loaded from `runs:/` URI, it's a raw model → don't pass params. Check `args.model_uri` prefix.
- **VALIDATE**: `uv run ruff check scripts/inference.py && uv run pyright scripts/inference.py`

### Task 5: CREATE `tests/models/test_horizon_router.py`

- **IMPLEMENT**: Unit tests for `HorizonRouter`:
  ```python
  class TestHorizonRouter:
      def test_predict_routes_to_correct_horizon(self):
          """Verify params={"horizon": h} routes to the right sub-model."""
          # Create mock sub-models that return their horizon as prediction
          mock_models = {}
          for h in [1, 6, 24]:
              m = MagicMock()
              m.predict.return_value = np.array([float(h * 100)])
              mock_models[h] = m

          router = HorizonRouter()
          router.models = mock_models

          result = router.predict(None, pd.DataFrame({"x": [1.0]}), params={"horizon": 24})
          assert result[0] == 2400.0

      def test_predict_missing_horizon_raises(self):
          """Verify ValueError when horizon not in loaded models."""
          router = HorizonRouter()
          router.models = {1: MagicMock()}
          with pytest.raises(ValueError, match="No model for horizon 99"):
              router.predict(None, pd.DataFrame({"x": [1]}), params={"horizon": 99})

      def test_predict_no_params_raises(self):
          """Verify ValueError when params is None."""
          router = HorizonRouter()
          router.models = {1: MagicMock()}
          with pytest.raises(ValueError, match="params must include"):
              router.predict(None, pd.DataFrame({"x": [1]}))

      def test_build_router_signature_includes_params(self):
          """Verify signature has ParamSchema with horizon."""
          sig = build_router_signature(["wind_speed", "temp"], default_horizon=6)
          assert sig.params is not None
          param_names = [p.name for p in sig.params.params]
          assert "horizon" in param_names
  ```
- **VALIDATE**: `uv run pytest tests/models/test_horizon_router.py -v`

### Task 6: UPDATE `tests/training/test_model_registry.py` — test router registration

- **IMPLEMENT**: Add test `test_run_training_registers_horizon_router`:
  - Run `run_training()` with `register_model_name="test-router"` and `horizons=[1, 6]`
  - Verify model `test-router` is registered with `champion` alias
  - Load the registered model via `mlflow.pyfunc.load_model("models:/test-router@champion")`
  - Verify `model.predict(X, params={"horizon": 1})` returns predictions
  - Verify `model.predict(X, params={"horizon": 6})` returns predictions
- **PATTERN**: Follow existing `test_run_training_with_registration` at lines 111-144
- **VALIDATE**: `uv run pytest tests/training/test_model_registry.py -v`

### Task 7: UPDATE `tests/test_inference.py` — add params to tests

- **IMPLEMENT**: Update `TestBuildInferenceFeatures` and add a test for `_predict_server` with params:
  - `test_predict_server_includes_params` — mock `requests.post`, verify the payload includes `"params": {"horizon": ...}`
- **VALIDATE**: `uv run pytest tests/test_inference.py -v`

### Task 8: Full validation

- **VALIDATE**: Run the full validation suite:
  ```bash
  uv run ruff check src/ tests/ scripts/ app/
  uv run ruff format --check src/ tests/ scripts/ app/
  uv run pyright src/
  uv run pytest tests/ -v
  ```

---

## TESTING STRATEGY

### Unit Tests

- `tests/models/test_horizon_router.py` — Router routing logic, signature building, error handling
- `tests/training/test_model_registry.py` — Router registration + load + predict via registered model
- `tests/test_inference.py` — Updated `_predict_server` with params

### Integration Tests (Manual)

1. Train with registration:
   ```bash
   uv run python scripts/train.py --domain wind --feature-set wind_full \
     --register-model enercast-kelmarsh-xgboost
   ```
2. Serve via native MLflow:
   ```bash
   MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
   mlflow models serve -m "models:/enercast-kelmarsh-xgboost@champion" \
     --env-manager local -p 5001
   ```
3. Test with curl:
   ```bash
   curl -X POST http://localhost:5001/invocations \
     -H 'Content-Type: application/json' \
     -d '{"dataframe_split": {"columns": [...], "data": [[...]]}, "params": {"horizon": 24}}'
   ```
4. Test inference script:
   ```bash
   uv run python scripts/inference.py --domain wind --horizon 24 \
     --serve-url http://localhost:5001
   ```
5. Test dashboard (in-process champion mode):
   ```bash
   streamlit run app/dashboard.py
   ```

### Edge Cases

- `params=None` or missing `horizon` key → `ValueError` with available horizons
- Unknown horizon value → `ValueError` with available horizons
- No champion alias registered → `mlflow.exceptions.MlflowException` (clear error)
- Old per-horizon model URI in `--model-uri` (backward compat) → works without params
- `mlflow models serve` without `MLFLOW_TRACKING_URI` → fails to resolve `models:/` URI

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
uv run pytest tests/models/test_horizon_router.py -v
uv run pytest tests/training/test_model_registry.py -v
uv run pytest tests/test_inference.py -v
```

### Level 3: Full Test Suite

```bash
uv run pytest tests/ -v
```

### Level 4: Manual Validation

```bash
# 1. Train with registration
uv run python scripts/train.py --domain wind --feature-set wind_full \
  --register-model enercast-kelmarsh-xgboost

# 2. Serve with native MLflow
MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
mlflow models serve -m "models:/enercast-kelmarsh-xgboost@champion" \
  --env-manager local -p 5001

# 3. Health check
curl http://localhost:5001/health

# 4. Predict via REST
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["wind_speed_ms"], "data": [[8.5]]}, "params": {"horizon": 24}}'

# 5. Predict via CLI
uv run python scripts/inference.py --domain wind --horizon 24 --serve-url http://localhost:5001

# 6. Dashboard (in-process champion mode)
streamlit run app/dashboard.py
```

---

## ACCEPTANCE CRITERIA

- [ ] `HorizonRouter` PythonModel wraps N sub-models and routes via `params["horizon"]`
- [ ] `run_training()` with `--register-model` logs and registers the router with `champion` alias
- [ ] `mlflow models serve -m "models:/enercast-kelmarsh-xgboost@champion"` starts and serves all 5 horizons
- [ ] `POST /invocations` with `"params": {"horizon": 24}` returns correct prediction
- [ ] Dashboard loads champion router model in-process, no manual run selection needed
- [ ] Dashboard falls back to per-run loading when no champion is registered
- [ ] `scripts/inference.py --serve-url` works with the native MLflow serving endpoint
- [ ] All validation commands pass with zero errors
- [ ] No regressions in existing tests (322+ tests)

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully:
  - [ ] Level 1: ruff check, ruff format, pyright
  - [ ] Level 2: router tests, registry tests, inference tests
  - [ ] Level 3: full test suite
  - [ ] Level 4: manual serve + curl + dashboard validation
- [ ] Full test suite passes
- [ ] No linting errors
- [ ] No type checking errors
- [ ] All acceptance criteria met

---

## NOTES

### Why a PythonModel wrapper instead of custom FastAPI?

MLflow's `mlflow models serve` is production-grade (FastAPI under the hood, health checks, Docker support). Writing our own FastAPI server would duplicate all of that. The official MLflow pattern for multi-model serving is a `PythonModel` wrapper with `params` routing — one model, one endpoint, one process. Same Docker image deploys to AWS/GCP via `mlflow models build-docker`.

### The `step` parameter in `log_model()`

Unrelated to forecast horizons. It's a training checkpoint index for deep learning (epoch tracking). Default `step=0` is correct for our use case.

### `ParamSchema` is mandatory

Without `ParamSchema` in the `ModelSignature`, `params` are silently dropped (MLflow issue #18522). This is the #1 gotcha. The `build_router_signature()` helper ensures it's always included.

### Backward compatibility

- Old per-horizon model URIs (`runs:/.../model_h24`) still work via `--model-uri`
- The router model is additive — existing runs and child models are unchanged
- Dashboard gets champion mode as default but keeps run-selection as fallback

### AutoGluon sub-models

AutoGluon models are already logged as pyfunc wrappers (`AutoGluonPyfuncWrapper`). Loading them via `mlflow.pyfunc.load_model()` inside the `HorizonRouter.load_context()` works transparently — MLflow resolves the pyfunc chain automatically.

### Confidence: 9/10

This is the canonical MLflow pattern with an official tutorial. Main risk: signature column list must exactly match what sub-models expect (canonical names after NWP renaming). The test in Task 6 validates this end-to-end.

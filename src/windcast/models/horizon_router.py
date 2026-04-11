"""Multi-horizon PythonModel wrapper for MLflow serving.

Bundles N per-horizon models into a single servable artifact.
Route predictions via params={"horizon": <int>}.

Based on the official MLflow pattern:
https://mlflow.org/docs/latest/traditional-ml/serving-multiple-models-with-pyfunc/
"""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc  # pyright: ignore[reportPrivateImportUsage]
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, DataType, ParamSchema, ParamSpec, Schema


class HorizonRouter(mlflow.pyfunc.PythonModel):  # pyright: ignore[reportPrivateImportUsage]
    """Routes predictions to the correct per-horizon sub-model.

    At log time, stores horizon→model_uri mappings (e.g. ``{1: "runs:/.../model_h01"}``).
    At load time, ``load_context()`` eagerly loads all sub-models.
    At predict time, ``params["horizon"]`` selects the sub-model.
    """

    def __init__(self, horizon_model_uris: dict[int, str] | None = None) -> None:
        self.horizon_model_uris: dict[int, str] = horizon_model_uris or {}
        self.models: dict[int, Any] = {}

    def load_context(self, context: Any) -> None:
        """Load all sub-models from their logged URIs."""
        for horizon, uri in self.horizon_model_uris.items():
            self.models[horizon] = mlflow.pyfunc.load_model(uri)

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Route prediction to the sub-model for the requested horizon."""
        if params is None or "horizon" not in params:
            raise ValueError(
                "params must include 'horizon' key. "
                f"Available horizons: {sorted(self.models.keys())}"
            )
        horizon = int(params["horizon"])
        model = self.models.get(horizon)
        if model is None:
            raise ValueError(
                f"No model for horizon {horizon}. Available: {sorted(self.models.keys())}"
            )
        result = model.predict(model_input)
        if isinstance(result, np.ndarray):
            return result
        return np.asarray(result)


def build_router_signature(
    feature_columns: list[str],
    default_horizon: int = 1,
) -> ModelSignature:
    """Build a ModelSignature with ParamSchema for the horizon router.

    The ParamSchema is mandatory — without it, ``params`` is silently
    ``None`` at predict time (MLflow issue #18522).
    """
    input_schema = Schema([ColSpec(DataType.double, col) for col in feature_columns])
    output_schema = Schema([ColSpec(DataType.double, "prediction")])
    params_schema = ParamSchema(
        [
            ParamSpec("horizon", DataType.long, default_horizon),
        ]
    )
    return ModelSignature(
        inputs=input_schema,
        outputs=output_schema,
        params=params_schema,
    )

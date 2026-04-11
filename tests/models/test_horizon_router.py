"""Tests for HorizonRouter PythonModel wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from windcast.models.horizon_router import HorizonRouter, build_router_signature


class TestHorizonRouter:
    """Unit tests for HorizonRouter prediction routing."""

    def test_predict_routes_to_correct_horizon(self):
        """Verify params={"horizon": h} routes to the right sub-model."""
        mock_models = {}
        for h in [1, 6, 24]:
            m = MagicMock()
            m.predict.return_value = np.array([float(h * 100)])
            mock_models[h] = m

        router = HorizonRouter()
        router.models = mock_models

        df = pd.DataFrame({"x": [1.0]})
        result = router.predict(None, df, params={"horizon": 24})
        assert result[0] == 2400.0

        result = router.predict(None, df, params={"horizon": 1})
        assert result[0] == 100.0

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

    def test_predict_no_horizon_key_raises(self):
        """Verify ValueError when params dict lacks 'horizon' key."""
        router = HorizonRouter()
        router.models = {1: MagicMock()}
        with pytest.raises(ValueError, match="params must include"):
            router.predict(None, pd.DataFrame({"x": [1]}), params={"other": 1})

    def test_predict_returns_numpy_array(self):
        """Verify predict always returns numpy array."""
        m = MagicMock()
        m.predict.return_value = [42.0]  # list, not array

        router = HorizonRouter()
        router.models = {1: m}

        result = router.predict(None, pd.DataFrame({"x": [1.0]}), params={"horizon": 1})
        assert isinstance(result, np.ndarray)
        assert result[0] == 42.0


class TestBuildRouterSignature:
    """Tests for build_router_signature helper."""

    def test_includes_param_schema(self):
        """Verify signature has ParamSchema with horizon."""
        sig = build_router_signature(["wind_speed", "temp"], default_horizon=6)
        assert sig.params is not None
        param_names = [p.name for p in sig.params.params]
        assert "horizon" in param_names

    def test_input_schema_matches_columns(self):
        """Verify input schema has the right column names."""
        cols = ["wind_speed", "temperature", "power_lag1"]
        sig = build_router_signature(cols)
        input_names = [c.name for c in sig.inputs.inputs]
        assert input_names == cols

    def test_default_horizon_value(self):
        """Verify default horizon is set correctly."""
        sig = build_router_signature(["x"], default_horizon=12)
        horizon_param = next(p for p in sig.params.params if p.name == "horizon")
        assert horizon_param.default == 12

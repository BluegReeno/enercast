"""Tests for windcast.data.rte_api — sync RTE API client."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from windcast.data.rte_api import RTEClient, _parse_realised_values, fetch_recent_load

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_TOKEN_RESPONSE = {
    "access_token": "test-token-abc123",
    "token_type": "Bearer",
    "expires_in": 3600,
    "scope": "",
}


def _val(hh: str, mm: str, value: int) -> dict:
    """Build a single RTE API value entry."""
    from datetime import datetime, timedelta

    start = datetime.fromisoformat(f"2026-04-10T{hh}:{mm}:00+02:00")
    end = start + timedelta(minutes=15)
    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "value": value,
    }


REALISTIC_API_RESPONSE = {
    "short_term": [
        {
            "type": "REALISED",
            "start_date": "2026-04-10T00:00:00+02:00",
            "end_date": "2026-04-10T04:00:00+02:00",
            "values": [
                _val("00", "00", 44778),
                _val("00", "15", 44500),
                _val("00", "30", 44200),
                _val("00", "45", 43900),
                _val("01", "00", 43600),
                _val("01", "15", 43400),
                _val("01", "30", 43200),
                _val("01", "45", 43000),
            ],
        }
    ]
}


def _make_mock_client():
    """Create a mock httpx.Client context manager."""
    mock_http = MagicMock()
    mock_http.__enter__ = MagicMock(return_value=mock_http)
    mock_http.__exit__ = MagicMock(return_value=False)
    return mock_http


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------


class TestTokenRefresh:
    def test_token_fetched_on_first_call(self):
        client = RTEClient("test-id", "test-secret")
        mock_http = _make_mock_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_TOKEN_RESPONSE
        mock_http.post.return_value = mock_resp

        with patch("windcast.data.rte_api.httpx.Client", return_value=mock_http):
            client._ensure_token()

        assert client._token == "test-token-abc123"
        mock_http.post.assert_called_once()

    def test_token_reused_when_valid(self):
        client = RTEClient("test-id", "test-secret")
        client._token = "cached-token"
        client._token_expires_at = time.time() + 3600

        mock_http = _make_mock_client()
        with patch("windcast.data.rte_api.httpx.Client", return_value=mock_http):
            client._ensure_token()

        assert client._token == "cached-token"
        mock_http.post.assert_not_called()

    def test_token_refreshed_when_expired(self):
        client = RTEClient("test-id", "test-secret")
        client._token = "old-token"
        client._token_expires_at = time.time() - 10  # expired

        mock_http = _make_mock_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_TOKEN_RESPONSE
        mock_http.post.return_value = mock_resp

        with patch("windcast.data.rte_api.httpx.Client", return_value=mock_http):
            client._ensure_token()

        assert client._token == "test-token-abc123"
        mock_http.post.assert_called_once()


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseRealisedValues:
    def test_parses_realistic_response(self):
        values = REALISTIC_API_RESPONSE["short_term"][0]["values"]
        df = _parse_realised_values(values)

        assert df.columns == ["timestamp_utc", "load_mw"]
        assert df.schema["timestamp_utc"] == pl.Datetime("us", "UTC")
        assert df.schema["load_mw"] == pl.Float64

        # 8 x 15-min values → 2 hourly rows
        assert len(df) == 2

        # First hour: mean of 44778, 44500, 44200, 43900 = 44344.5
        first_load = df["load_mw"][0]
        assert first_load == pytest.approx(44344.5, rel=1e-3)

    def test_empty_values_returns_empty_df(self):
        df = _parse_realised_values([])

        assert df.columns == ["timestamp_utc", "load_mw"]
        assert len(df) == 0


# ---------------------------------------------------------------------------
# fetch_recent_load
# ---------------------------------------------------------------------------


class TestFetchRecentLoad:
    def test_fetches_and_parses(self):
        client = RTEClient("test-id", "test-secret")
        client._token = "valid-token"
        client._token_expires_at = time.time() + 3600

        mock_http = _make_mock_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = REALISTIC_API_RESPONSE
        mock_http.get.return_value = mock_resp

        with patch("windcast.data.rte_api.httpx.Client", return_value=mock_http):
            df = fetch_recent_load(client, hours=48)

        assert not df.is_empty()
        assert "timestamp_utc" in df.columns
        assert "load_mw" in df.columns

    def test_empty_response_returns_empty_df(self):
        client = RTEClient("test-id", "test-secret")
        client._token = "valid-token"
        client._token_expires_at = time.time() + 3600

        mock_http = _make_mock_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"short_term": []}
        mock_http.get.return_value = mock_resp

        with patch("windcast.data.rte_api.httpx.Client", return_value=mock_http):
            df = fetch_recent_load(client, hours=48)

        assert df.is_empty()
        assert df.columns == ["timestamp_utc", "load_mw"]

    def test_http_error_propagates(self):
        import httpx

        client = RTEClient("test-id", "test-secret")
        client._token = "valid-token"
        client._token_expires_at = time.time() + 3600

        mock_http = _make_mock_client()
        mock_http.get.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        with (
            patch("windcast.data.rte_api.httpx.Client", return_value=mock_http),
            pytest.raises(httpx.HTTPStatusError),
        ):
            fetch_recent_load(client, hours=48)

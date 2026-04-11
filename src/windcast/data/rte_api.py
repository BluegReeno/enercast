"""RTE Data API client — sync OAuth2 + live load fetching.

Ported from the wattcast async pattern to sync httpx.
Used by the dashboard to overlay recent observed load on forecast charts.
"""

from __future__ import annotations

import base64
import logging
import time
from datetime import UTC, datetime, timedelta

import httpx
import polars as pl

logger = logging.getLogger(__name__)

TOKEN_URL = "https://digital.iservices.rte-france.com/token/oauth/"
API_BASE = "https://digital.iservices.rte-france.com/open_api"
TOKEN_TTL_BUFFER = 60  # Renew 60s before actual expiry


def _rte_datetime(dt: datetime) -> str:
    """Format datetime for RTE API — ISO 8601 with timezone offset."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat(timespec="seconds")


class RTEClient:
    """Sync httpx client with automatic OAuth2 token refresh for RTE Data API."""

    def __init__(self, client_id: str, client_secret: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def _basic_auth_header(self) -> str:
        raw = f"{self._client_id}:{self._client_secret}"
        return f"Basic {base64.b64encode(raw.encode()).decode()}"

    def _ensure_token(self) -> None:
        if self._token and time.time() < self._token_expires_at:
            return
        with httpx.Client(timeout=30) as http:
            resp = http.post(
                TOKEN_URL,
                headers={"Authorization": self._basic_auth_header()},
            )
            resp.raise_for_status()
            data = resp.json()
        self._token = data["access_token"]
        self._token_expires_at = time.time() + data["expires_in"] - TOKEN_TTL_BUFFER
        logger.debug("RTE token refreshed, expires in %ds", data["expires_in"])

    def get(self, endpoint: str, params: dict[str, str] | None = None) -> dict:
        """Make an authenticated GET request with auto-refresh."""
        self._ensure_token()
        url = f"{API_BASE}/{endpoint}"
        with httpx.Client(timeout=30) as http:
            resp = http.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {self._token}"},
            )
            resp.raise_for_status()
            return resp.json()


def _parse_realised_values(values: list[dict]) -> pl.DataFrame:
    """Parse RTE REALISED values into a Polars DataFrame.

    Returns DataFrame with columns: timestamp_utc (Datetime[UTC]), load_mw (Float64).
    Resampled to hourly via group_by_dynamic.
    """
    if not values:
        return pl.DataFrame(
            schema={"timestamp_utc": pl.Datetime("us", "UTC"), "load_mw": pl.Float64}
        )

    rows = []
    for v in values:
        ts = datetime.fromisoformat(v["start_date"]).astimezone(UTC)
        rows.append({"timestamp_utc": ts, "load_mw": float(v["value"])})

    df = pl.DataFrame(rows).sort("timestamp_utc")

    # Resample to hourly (native resolution is 15-min)
    df = df.group_by_dynamic("timestamp_utc", every="1h").agg(pl.col("load_mw").mean())

    return df


def fetch_recent_load(client: RTEClient, hours: int = 48) -> pl.DataFrame:
    """Fetch recent REALISED load from RTE consumption API.

    Args:
        client: Authenticated RTEClient instance.
        hours: Number of hours of history to fetch. Default: 48.

    Returns:
        Polars DataFrame: (timestamp_utc, load_mw), hourly.
    """
    now = datetime.now(UTC)
    start = now - timedelta(hours=hours)

    data = client.get(
        "consumption/v1/short_term",
        params={
            "start_date": _rte_datetime(start),
            "end_date": _rte_datetime(now),
            "type": "REALISED",
        },
    )

    for series in data.get("short_term", []):
        if series.get("type") == "REALISED":
            return _parse_realised_values(series.get("values", []))

    logger.warning("No REALISED data in RTE API response")
    return pl.DataFrame(schema={"timestamp_utc": pl.Datetime("us", "UTC"), "load_mw": pl.Float64})


def _chunk_dates(
    start: datetime, end: datetime, chunk_days: int = 180
) -> list[tuple[datetime, datetime]]:
    """Split a date range into API-safe chunks (max 186 days per call)."""
    chunks: list[tuple[datetime, datetime]] = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


def fetch_load_history(
    client: RTEClient,
    start: datetime,
    end: datetime,
    types: str = "REALISED",
    chunk_days: int = 180,
    sleep_between: float = 0.5,
) -> pl.DataFrame:
    """Fetch load history in chunks, returning hourly DataFrame.

    Args:
        client: Authenticated RTEClient instance.
        start: Start datetime (UTC).
        end: End datetime (UTC).
        types: Comma-separated types (e.g. "REALISED", "REALISED,D-1").
        chunk_days: Days per API chunk (max 186, default 180 for safety).
        sleep_between: Courtesy sleep between chunks (seconds).

    Returns:
        Polars DataFrame: (timestamp_utc, load_mw) for REALISED.
        If types includes "D-1", also includes forecast_mw column.
    """
    chunks = _chunk_dates(start, end, chunk_days)
    all_realised: list[pl.DataFrame] = []
    all_d1: list[pl.DataFrame] = []
    include_d1 = "D-1" in types

    for i, (chunk_start, chunk_end) in enumerate(chunks):
        logger.info(
            "Fetching chunk %d/%d: %s → %s",
            i + 1,
            len(chunks),
            chunk_start.date(),
            chunk_end.date(),
        )

        data = client.get(
            "consumption/v1/short_term",
            params={
                "start_date": _rte_datetime(chunk_start),
                "end_date": _rte_datetime(chunk_end),
                "type": types,
            },
        )

        for series in data.get("short_term", []):
            if series.get("type") == "REALISED":
                df = _parse_realised_values(series.get("values", []))
                if not df.is_empty():
                    all_realised.append(df)
            elif series.get("type") == "D-1" and include_d1:
                df = _parse_d1_values(series.get("values", []))
                if not df.is_empty():
                    all_d1.append(df)

        if i < len(chunks) - 1 and sleep_between > 0:
            time.sleep(sleep_between)

    if not all_realised:
        cols: list[tuple[str, pl.DataType]] = [
            ("timestamp_utc", pl.Datetime("us", "UTC")),
            ("load_mw", pl.Float64()),
        ]
        if include_d1:
            cols.append(("forecast_mw", pl.Float64()))
        return pl.DataFrame(schema=dict(cols))

    result = pl.concat(all_realised).sort("timestamp_utc").unique("timestamp_utc")

    if include_d1 and all_d1:
        d1_df = pl.concat(all_d1).sort("timestamp_utc").unique("timestamp_utc")
        result = result.join(d1_df, on="timestamp_utc", how="left")

    logger.info("Fetched %d hourly rows (%s → %s)", len(result), start.date(), end.date())
    return result


def _parse_d1_values(values: list[dict]) -> pl.DataFrame:
    """Parse RTE D-1 forecast values into hourly DataFrame.

    Returns DataFrame with columns: timestamp_utc (Datetime[UTC]), forecast_mw (Float64).
    """
    if not values:
        return pl.DataFrame(
            schema={"timestamp_utc": pl.Datetime("us", "UTC"), "forecast_mw": pl.Float64}
        )

    rows = []
    for v in values:
        ts = datetime.fromisoformat(v["start_date"]).astimezone(UTC)
        rows.append({"timestamp_utc": ts, "forecast_mw": float(v["value"])})

    df = pl.DataFrame(rows).sort("timestamp_utc")
    df = df.group_by_dynamic("timestamp_utc", every="1h").agg(pl.col("forecast_mw").mean())
    return df


def get_live_actuals(client_id: str, client_secret: str, hours: int = 48) -> pl.DataFrame:
    """Convenience wrapper: create client + fetch recent load.

    Args:
        client_id: RTE API client ID.
        client_secret: RTE API client secret.
        hours: Number of hours of history. Default: 48.

    Returns:
        Polars DataFrame: (timestamp_utc, load_mw), hourly.
    """
    client = RTEClient(client_id, client_secret)
    return fetch_recent_load(client, hours=hours)

"""Open-Meteo historical weather data client."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import openmeteo_requests
import polars as pl
import requests_cache
from retry_requests import retry

logger = logging.getLogger(__name__)

WIND_VARIABLES: list[str] = [
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_speed_10m",
    "wind_direction_10m",
    "temperature_2m",
    "pressure_msl",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def build_client(
    cache_dir: str = ".cache",
    expire_after: int = -1,
    retries: int = 5,
    backoff_factor: float = 0.2,
) -> openmeteo_requests.Client:
    """Build a cached, auto-retrying Open-Meteo client."""
    cache_session = requests_cache.CachedSession(cache_dir, expire_after=expire_after)
    retry_session = retry(cache_session, retries=retries, backoff_factor=backoff_factor)
    return openmeteo_requests.Client(session=retry_session)  # type: ignore[arg-type]


def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
    client: openmeteo_requests.Client | None = None,
) -> pl.DataFrame:
    """Fetch hourly historical weather from Open-Meteo archive.

    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        start_date: Start date ISO format "YYYY-MM-DD".
        end_date: End date ISO format "YYYY-MM-DD".
        variables: Weather variables to fetch. Defaults to WIND_VARIABLES.
        client: Pre-built client. Creates new one if None.

    Returns:
        Polars DataFrame with timestamp_utc + weather variable columns.
    """
    if variables is None:
        variables = WIND_VARIABLES
    if client is None:
        client = build_client()

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": variables,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    responses = client.weather_api(ARCHIVE_URL, params=params)
    return _response_to_polars(responses[0], variables)


def fetch_historical_forecast_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
    client: openmeteo_requests.Client | None = None,
) -> pl.DataFrame:
    """Fetch hourly archived NWP *forecast* output from Open-Meteo.

    Unlike :func:`fetch_historical_weather` (ERA5 reanalysis = observed weather),
    this endpoint returns the model output that was actually *forecast* at the
    time — stitched from the short-range portion of successive NWP runs.
    Coverage starts 2022-01-01 for ICON; earlier for ECMWF IFS (2017+) and
    GFS (2021+).

    This is the function to use for honest val/test features where the model
    must not peek at future ground truth. See the ERA5 leak incident note in
    ``docs/WNchallenge/post-era5-fix-results.md`` for the motivation.

    Bias vs a fully-issued D+1 forecast: ~1°C RMSE at D+1, ~3°C at D+7 on
    temperature (measured in WattCast production, see
    ``wattcast/docs/delivery-time-weather-features.md``).

    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        start_date: Start date ISO format "YYYY-MM-DD".
        end_date: End date ISO format "YYYY-MM-DD".
        variables: Weather variables to fetch. Defaults to WIND_VARIABLES.
        client: Pre-built client. Creates new one if None.

    Returns:
        Polars DataFrame with timestamp_utc + weather variable columns — same
        schema as :func:`fetch_historical_weather` so downstream consumers are
        source-agnostic.

    # adapted from wattcast/src/wattcast/data/open_meteo.py::fetch_historical_forecast_weather
    """
    if variables is None:
        variables = WIND_VARIABLES
    if client is None:
        client = build_client()

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": variables,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    responses = client.weather_api(HISTORICAL_FORECAST_URL, params=params)
    return _response_to_polars(responses[0], variables)


def fetch_forecast_weather(
    latitude: float,
    longitude: float,
    variables: list[str] | None = None,
    forecast_days: int = 7,
    past_days: int = 0,
    client: openmeteo_requests.Client | None = None,
) -> pl.DataFrame:
    """Fetch hourly live NWP forecast from Open-Meteo Forecast API.

    Unlike the archive and historical-forecast endpoints, this returns the
    latest model run output — what the NWP actually predicts right now.
    Data changes every ~6h as new model runs become available.

    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        variables: Weather variables to fetch. Defaults to WIND_VARIABLES.
        forecast_days: Number of forecast days (1-16). Default: 7.
        past_days: Include N recent past days for context. Default: 0.
        client: Pre-built client. Creates one with 1h cache TTL if None.

    Returns:
        Polars DataFrame with timestamp_utc + weather variable columns.
    """
    if variables is None:
        variables = WIND_VARIABLES
    if client is None:
        client = build_client(expire_after=3600)

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "forecast_days": forecast_days,
        "past_days": past_days,
        "hourly": variables,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    responses = client.weather_api(FORECAST_URL, params=params)
    return _response_to_polars(responses[0], variables)


def _response_to_polars(response: Any, variables: list[str]) -> pl.DataFrame:
    """Parse an Open-Meteo protobuf response into a wide Polars DataFrame.

    Shared by :func:`fetch_historical_weather` and
    :func:`fetch_historical_forecast_weather` — both APIs return the same
    ``hourly`` shape.
    """
    hourly = response.Hourly()
    assert hourly is not None, "No hourly data in Open-Meteo response"

    start_ts = datetime.fromtimestamp(hourly.Time(), tz=UTC)
    end_ts = datetime.fromtimestamp(hourly.TimeEnd(), tz=UTC)
    interval_s = hourly.Interval()

    timestamps = pl.datetime_range(
        start=start_ts,
        end=end_ts,
        interval=f"{interval_s}s",
        eager=True,
        closed="left",
    )

    data: dict[str, pl.Series] = {"timestamp_utc": timestamps}
    for i, var_name in enumerate(variables):
        values = hourly.Variables(i).ValuesAsNumpy()
        data[var_name] = pl.Series(var_name, values, dtype=pl.Float32)

    return pl.DataFrame(data)

"""Weather data layer — fetch, cache, and serve NWP data for any site."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from windcast.weather.provider import (
    HistoricalForecastProvider,
    OpenMeteoProvider,
    WeatherProvider,
)
from windcast.weather.registry import (
    AnyWeatherConfig,
    WeatherConfig,
    WeatherPoint,
    WeightedWeatherConfig,
    get_weather_config,
    list_weather_configs,
)
from windcast.weather.storage import WeatherStorage

__all__ = [
    "DEFAULT_DB_PATH",
    "WEATHER_FORECAST_DB_PATH",
    "AnyWeatherConfig",
    "HistoricalForecastProvider",
    "OpenMeteoProvider",
    "WeatherConfig",
    "WeatherPoint",
    "WeatherProvider",
    "WeatherStorage",
    "WeightedWeatherConfig",
    "get_forecast_weather",
    "get_live_forecast",
    "get_weather",
    "get_weather_config",
    "get_weather_weighted",
    "list_weather_configs",
    "load_blended_weather",
    "load_weather_from_db",
]

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/weather.db")
WEATHER_FORECAST_DB_PATH = Path("data/weather_forecast.db")

ERA5_LAG_DAYS = 5
HISTORICAL_FORECAST_EARLIEST = "2022-01-01"


def _location_key(config: WeatherConfig) -> str:
    """Derive a deterministic location key from config coordinates."""
    return f"{config.latitude}_{config.longitude}"


def _clamp_end_date(end_date: str) -> str:
    """Clamp end_date to ERA5 availability (today - 5 days)."""
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    max_dt = datetime.now(UTC) - timedelta(days=ERA5_LAG_DAYS)

    if end_dt > max_dt:
        clamped = max_dt.strftime("%Y-%m-%d")
        logger.warning(
            "ERA5 has ~%d-day lag. Clamping end_date from %s to %s",
            ERA5_LAG_DAYS,
            end_date,
            clamped,
        )
        return clamped
    return end_date


def get_weather(
    config_name: str,
    start_date: str,
    end_date: str,
    db_path: Path = DEFAULT_DB_PATH,
    provider: WeatherProvider | None = None,
) -> pl.DataFrame:
    """Fetch weather data for a registered config, with SQLite caching.

    Dispatches on config type:

    - :class:`WeatherConfig` (single-point): returns variable columns for one
      lat/lon, as cached in the SQLite store.
    - :class:`WeightedWeatherConfig` (multi-point): fetches each point, then
      returns a population-weighted national mean DataFrame — one row per
      timestamp, one column per variable.

    Args:
        config_name: Registered weather config name (e.g., "kelmarsh").
        start_date: Start date ISO "YYYY-MM-DD".
        end_date: End date ISO "YYYY-MM-DD".
        db_path: Path to SQLite cache file.
        provider: Weather data provider. Defaults to OpenMeteoProvider.

    Returns:
        Polars DataFrame: timestamp_utc + weather variable columns (hourly).
    """
    config = get_weather_config(config_name)
    end_date = _clamp_end_date(end_date)

    if provider is None:
        provider = OpenMeteoProvider()

    if isinstance(config, WeightedWeatherConfig):
        return get_weather_weighted(config, start_date, end_date, db_path, provider)

    storage = WeatherStorage(db_path)
    loc_key = _location_key(config)

    try:
        _fetch_missing(storage, provider, config, loc_key, start_date, end_date)
        return storage.query(loc_key, start_date, end_date, config.variables)
    finally:
        storage.close()


def get_weather_weighted(
    config: WeightedWeatherConfig,
    start_date: str,
    end_date: str,
    db_path: Path = DEFAULT_DB_PATH,
    provider: WeatherProvider | None = None,
) -> pl.DataFrame:
    """Fetch a multi-point config and return a weighted national DataFrame.

    Each point is cached independently under its own ``{lat}_{lon}`` location
    key, so subsequent calls hit the cache. The result is the weight-normalised
    sum across points per timestamp, per variable — the same pattern as
    wattcast's ``compute_national_temperature``.
    """
    if provider is None:
        provider = OpenMeteoProvider()
    end_date = _clamp_end_date(end_date)

    storage = WeatherStorage(db_path)
    per_point_dfs: list[tuple[float, pl.DataFrame]] = []

    try:
        for point in config.points:
            # Build a single-point WeatherConfig stub to reuse _fetch_missing.
            point_cfg = WeatherConfig(
                name=f"{config.name}:{point.name}",
                latitude=point.latitude,
                longitude=point.longitude,
                variables=config.variables,
            )
            loc_key = _location_key(point_cfg)
            _fetch_missing(storage, provider, point_cfg, loc_key, start_date, end_date)
            df = storage.query(loc_key, start_date, end_date, config.variables)
            if df.is_empty():
                logger.warning(
                    "Empty NWP slice for point %s (%s,%s)",
                    point.name,
                    point.latitude,
                    point.longitude,
                )
                continue
            per_point_dfs.append((point.weight, df))
    finally:
        storage.close()

    if not per_point_dfs:
        logger.warning("No NWP data fetched for any point in %s", config.name)
        return pl.DataFrame(schema={"timestamp_utc": pl.Datetime("us", "UTC")})

    return _weighted_mean(per_point_dfs, config.variables)


def _weighted_mean(
    per_point_dfs: list[tuple[float, pl.DataFrame]],
    variables: list[str],
) -> pl.DataFrame:
    """Combine per-point DataFrames into a single weighted-mean wide DataFrame.

    Each input frame must have ``timestamp_utc`` + variable columns. Missing
    values are handled per (timestamp, variable): weights are renormalised over
    only the points that contributed a value, so gaps in one city don't bias
    the national mean.
    """
    # Label each frame by point index, stack vertically
    labelled: list[pl.DataFrame] = []
    for idx, (weight, df) in enumerate(per_point_dfs):
        labelled.append(
            df.with_columns(
                pl.lit(idx).alias("_point_idx"),
                pl.lit(weight).alias("_w"),
            )
        )
    stacked = pl.concat(labelled, how="vertical_relaxed")

    # For each (timestamp, variable), compute sum(w*x) / sum(w) ignoring null x
    agg_exprs = []
    for var in variables:
        if var not in stacked.columns:
            continue
        weighted_val = (pl.col("_w") * pl.col(var)).filter(pl.col(var).is_not_null()).sum()
        weight_sum = pl.col("_w").filter(pl.col(var).is_not_null()).sum()
        agg_exprs.append((weighted_val / weight_sum).alias(var))

    national = stacked.group_by("timestamp_utc").agg(*agg_exprs).sort("timestamp_utc")
    return national


def get_forecast_weather(
    config_name: str,
    start_date: str,
    end_date: str,
    db_path: Path = WEATHER_FORECAST_DB_PATH,
    provider: WeatherProvider | None = None,
) -> pl.DataFrame:
    """Fetch archived NWP *forecast* weather for a registered config.

    Thin wrapper around :func:`get_weather` that defaults the provider to
    :class:`HistoricalForecastProvider` and the cache path to
    :data:`WEATHER_FORECAST_DB_PATH`. Works for both single-point and
    weighted multi-point configs (dispatch happens inside ``get_weather``).

    Use this for val/test periods where honest forecast-time features are
    required. Raises if ``start_date`` is before the Open-Meteo
    historical-forecast coverage start (2022-01-01).
    """
    if start_date < HISTORICAL_FORECAST_EARLIEST:
        raise ValueError(
            "Historical forecast API coverage starts "
            f"{HISTORICAL_FORECAST_EARLIEST}; got start_date={start_date}. "
            "Use get_weather() (ERA5) for earlier periods or blend mode."
        )

    if provider is None:
        provider = HistoricalForecastProvider()

    return get_weather(
        config_name=config_name,
        start_date=start_date,
        end_date=end_date,
        db_path=db_path,
        provider=provider,
    )


def get_live_forecast(
    config_name: str,
    forecast_days: int = 7,
    past_days: int = 0,
) -> pl.DataFrame:
    """Fetch live NWP forecast for a registered weather config.

    Uses the Open-Meteo Forecast API (latest model run). No SQLite caching —
    live forecasts are ephemeral by nature (updated every ~6h). HTTP-level
    caching uses a 1h TTL via ``requests_cache``.

    Supports both single-point (:class:`WeatherConfig`) and multi-point
    (:class:`WeightedWeatherConfig`) configs.

    Args:
        config_name: Registered weather config name (e.g., "kelmarsh").
        forecast_days: Number of forecast days (1-16). Default: 7.
        past_days: Include N recent past days. Default: 0.

    Returns:
        Polars DataFrame: timestamp_utc + weather variable columns (hourly).
    """
    from windcast.data.open_meteo import build_client, fetch_forecast_weather

    config = get_weather_config(config_name)
    client = build_client(expire_after=3600)

    if isinstance(config, WeightedWeatherConfig):
        per_point_dfs: list[tuple[float, pl.DataFrame]] = []
        for point in config.points:
            df = fetch_forecast_weather(
                latitude=point.latitude,
                longitude=point.longitude,
                variables=config.variables,
                forecast_days=forecast_days,
                past_days=past_days,
                client=client,
            )
            if df.is_empty():
                logger.warning(
                    "Empty forecast for point %s (%s,%s)",
                    point.name,
                    point.latitude,
                    point.longitude,
                )
                continue
            per_point_dfs.append((point.weight, df))

        if not per_point_dfs:
            logger.warning("No forecast data for any point in %s", config.name)
            return pl.DataFrame(schema={"timestamp_utc": pl.Datetime("us", "UTC")})

        return _weighted_mean(per_point_dfs, config.variables)

    return fetch_forecast_weather(
        latitude=config.latitude,
        longitude=config.longitude,
        variables=config.variables,
        forecast_days=forecast_days,
        past_days=past_days,
        client=client,
    )


def load_weather_from_db(weather_name: str, db_path: Path) -> pl.DataFrame | None:
    """Load cached NWP weather for a registered config from a single SQLite db.

    Supports both single-point :class:`WeatherConfig` and multi-point
    :class:`WeightedWeatherConfig`. Returns ``None`` if the db has no coverage
    for the config's location(s). Unlike :func:`get_weather`, this does not
    hit any remote API — it only reads what is already cached in ``db_path``.
    """
    wcfg = get_weather_config(weather_name)

    if isinstance(wcfg, WeightedWeatherConfig):
        first_point = wcfg.points[0]
        storage = WeatherStorage(db_path)
        try:
            coverage = storage.get_coverage(f"{first_point.latitude}_{first_point.longitude}")
        finally:
            storage.close()
        if coverage is None:
            logger.error(
                "No weather data in %s for %s (first point %s)",
                db_path,
                weather_name,
                first_point.name,
            )
            return None
        start, end = coverage[0][:10], coverage[1][:10]
        df = _read_weighted_from_cache(wcfg, start, end, db_path)
        logger.info(
            "Loaded weighted NWP: %d rows, %d variables, %d points from %s",
            len(df),
            len(df.columns) - 1,
            len(wcfg.points),
            db_path,
        )
        return df

    storage = WeatherStorage(db_path)
    try:
        coverage = storage.get_coverage(f"{wcfg.latitude}_{wcfg.longitude}")
        if coverage is None:
            logger.error("No weather data in %s for %s", db_path, weather_name)
            return None
        start, end = coverage[0][:10], coverage[1][:10]
        df = storage.query(f"{wcfg.latitude}_{wcfg.longitude}", start, end, wcfg.variables)
    finally:
        storage.close()
    logger.info(
        "Loaded NWP: %d rows, %d variables from %s",
        len(df),
        len(df.columns) - 1,
        db_path,
    )
    return df


def _read_weighted_from_cache(
    config: WeightedWeatherConfig,
    start_date: str,
    end_date: str,
    db_path: Path,
) -> pl.DataFrame:
    """Cache-only read for a weighted config — no remote fetch.

    Mirrors the aggregation in :func:`get_weather_weighted` but skips the
    :func:`_fetch_missing` step so it can be used against either
    ``weather.db`` or ``weather_forecast.db`` without hitting the network.
    """
    storage = WeatherStorage(db_path)
    per_point_dfs: list[tuple[float, pl.DataFrame]] = []
    try:
        for point in config.points:
            loc_key = f"{point.latitude}_{point.longitude}"
            df = storage.query(loc_key, start_date, end_date, config.variables)
            if df.is_empty():
                logger.warning(
                    "Empty NWP slice for point %s (%s,%s) in %s",
                    point.name,
                    point.latitude,
                    point.longitude,
                    db_path,
                )
                continue
            per_point_dfs.append((point.weight, df))
    finally:
        storage.close()

    if not per_point_dfs:
        return pl.DataFrame(schema={"timestamp_utc": pl.Datetime("us", "UTC")})

    return _weighted_mean(per_point_dfs, config.variables)


def load_blended_weather(
    weather_name: str,
    era5_db: Path,
    forecast_db: Path,
    cutoff: str,
) -> pl.DataFrame | None:
    """Blend ERA5 and archived-forecast weather at a timestamp cutoff.

    Rows with ``timestamp_utc < cutoff`` come from *era5_db* and rows with
    ``timestamp_utc >= cutoff`` come from *forecast_db*. Columns must match
    between the two sources (same variables / same config).

    This is the mode to use for honest val/test features while still getting
    long ERA5 train history — mirrors WattCast's production pattern.

    Args:
        weather_name: Registered weather config name.
        era5_db: Path to the ERA5 SQLite cache.
        forecast_db: Path to the historical-forecast SQLite cache.
        cutoff: ISO date string, e.g. ``"2022-01-01"``.

    Returns:
        A wide NWP DataFrame with ``timestamp_utc`` + variable columns,
        sorted by timestamp, or ``None`` if both sources are empty.
    """
    era5_df = load_weather_from_db(weather_name, era5_db)
    forecast_df = load_weather_from_db(weather_name, forecast_db)

    if era5_df is None and forecast_df is None:
        logger.error("Blend mode: neither ERA5 nor forecast db has data for %s", weather_name)
        return None
    if era5_df is None:
        logger.warning("Blend mode: ERA5 db empty, using forecast db only")
        return forecast_df
    if forecast_df is None:
        logger.warning("Blend mode: forecast db empty, using ERA5 db only")
        return era5_df

    cutoff_dt = datetime.fromisoformat(cutoff).replace(tzinfo=UTC)
    era5_pre = era5_df.filter(pl.col("timestamp_utc") < cutoff_dt)
    forecast_post = forecast_df.filter(pl.col("timestamp_utc") >= cutoff_dt)

    # Align columns (both sources should have the same schema, but guard anyway)
    common_cols = [c for c in era5_pre.columns if c in forecast_post.columns]
    if "timestamp_utc" not in common_cols:
        common_cols = ["timestamp_utc", *common_cols]
    era5_pre = era5_pre.select(common_cols)
    forecast_post = forecast_post.select(common_cols)

    blended = pl.concat([era5_pre, forecast_post], how="vertical_relaxed").sort("timestamp_utc")
    logger.info(
        "Blended NWP: %d rows pre-%s (ERA5) + %d rows >=%s (forecast) = %d total",
        len(era5_pre),
        cutoff,
        len(forecast_post),
        cutoff,
        len(blended),
    )
    return blended


def _fetch_missing(
    storage: WeatherStorage,
    provider: WeatherProvider,
    config: WeatherConfig,
    loc_key: str,
    start_date: str,
    end_date: str,
) -> None:
    """Fetch only date ranges not already in the cache."""
    coverage = storage.get_coverage(loc_key)

    if coverage is None:
        logger.info("No cached data for %s — fetching full range", config.name)
        df = provider.fetch(
            config.latitude, config.longitude, start_date, end_date, config.variables
        )
        storage.upsert(loc_key, df)
        return

    cached_min, cached_max = coverage
    # Compare date portions only (cached timestamps are full ISO with time)
    cached_min_date = cached_min[:10]
    cached_max_date = cached_max[:10]

    # Fetch data before cache range
    if start_date < cached_min_date:
        logger.info("Fetching pre-cache gap: %s to %s", start_date, cached_min_date)
        df = provider.fetch(
            config.latitude, config.longitude, start_date, cached_min_date, config.variables
        )
        storage.upsert(loc_key, df)

    # Fetch data after cache range
    if end_date > cached_max_date:
        logger.info("Fetching post-cache gap: %s to %s", cached_max_date, end_date)
        df = provider.fetch(
            config.latitude, config.longitude, cached_max_date, end_date, config.variables
        )
        storage.upsert(loc_key, df)

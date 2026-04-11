# RTE Data API — Complete Reference

**Last tested**: 2026-04-11
**Status**: `consumption/v1/short_term` integrated in `rte_api.py`. Other endpoints documented for future use.
**Credentials**: `RTE_CLIENT_ID` / `RTE_CLIENT_SECRET` in `.env` (shared with wattcast)

---

## Authentication

```
POST https://digital.iservices.rte-france.com/token/oauth/
Authorization: Basic base64(client_id:client_secret)
→ { access_token, token_type: "Bearer", expires_in: 3600 }
```

- TTL: 3600 s (1 hour), refresh 60 s before expiry
- No scope required
- Register at https://data.rte-france.com → Create Application → Subscribe to each API individually

---

## Subscribed Endpoints (tested 2026-04-11)

### `consumption/v1/short_term` ✅

**The main endpoint for our use case.** Real-time actuals + TSO forecasts.

| Parameter | Values | Notes |
|-----------|--------|-------|
| `start_date` | ISO 8601 with TZ | Required as pair |
| `end_date` | ISO 8601 with TZ | Required as pair |
| `type` | `REALISED,D-1,D-2,ID` | Comma-separated, all returned if omitted |

**Types available:**

| Type | Resolution | Description |
|------|-----------|-------------|
| `REALISED` | 15 min (96/day) | Metered actual consumption |
| `D-1` | 15 min (96/day) | Day-ahead forecast (RTE official) |
| `D-2` | 30 min (48/day) | 2-day-ahead forecast |
| `ID` | 15 min (96/day) | Intraday forecast |
| `CORRECTED` | — | Exists but returned 0 values in tests |

**Limits:**
- **Max range per call: 186 days** (190 → HTTP 400 `CONSUMPTION_SHORTTERM_F03`)
- **History: since 2014** (tested 2014-01-01 ✅, older dates may return 0 values)
- **Freshness: ~minutes** (`updated_date` near real-time for current day)

**Value structure:**
```json
{
  "start_date": "2026-04-10T00:00:00+02:00",
  "end_date": "2026-04-10T00:15:00+02:00",
  "updated_date": "2026-04-10T23:49:33+02:00",
  "value": 44909
}
```

**Chunking strategy for backfill:**
- 2014 → 2026 (~4,400 days) = ~24 chunks of 180 days
- ~24 API calls, ~2-3 min with 0.5 s courtesy sleep
- Pattern: `_chunk_dates(start, end, chunk_days=180)` (WattCast uses 150, but 180 is safe)

---

### `consumption/v1/weekly_forecasts` ✅

Medium-range load forecasts (D+3 to D+9).

| Parameter | Values |
|-----------|--------|
| `start_date` | ISO 8601 with TZ |
| `end_date` | ISO 8601 with TZ |

- **Resolution**: 30 min (48 values/day)
- **Max range per call**: 155 days
- **History**: since 2004-12-23
- **No new forecasts on weekends** — each series has an `updated_date`
- **Bonus data**: `peak` object with `peak_hour`, `value` (MW), `temperature` (°C), `temperature_deviation` (°C)

**Response structure:**
```json
{
  "weekly_forecasts": [
    {
      "start_date": "2026-04-07T00:00:00+02:00",
      "end_date": "2026-04-08T00:00:00+02:00",
      "updated_date": "2026-04-03T11:15:03+02:00",
      "peak": {
        "peak_hour": "2026-04-07T08:30:00+02:00",
        "value": 52000,
        "temperature": 10.9,
        "temperature_deviation": 5.2
      },
      "values": [
        {"start_date": "...", "end_date": "...", "value": 43378}
      ]
    }
  ]
}
```

**Use case**: Could serve as a 2nd TSO benchmark (week-ahead vs our h48 model), or as a feature input for longer-horizon models. Not currently integrated.

---

### `consumption/v1/annual_forecasts` ✅

Seasonal/annual outlook — weekly resolution.

- **Resolution**: weekly (52 values/year)
- **Range**: min 1 calendar year, max 6 years
- **History**: since 2015
- **Content**: `average_load_saturday_to_friday`, `average_load_monday_to_sunday`, min/max
- **Use case**: Contextual only (presentation slide). Not useful for hourly forecasting.

---

### `wholesale_market/v3/france_power_exchanges` ✅

Spot market prices and volumes.

- **Resolution**: 15 min (96 values/day)
- **Content**: `value` (volume MWh), `price` (€/MWh)
- **Includes next-day prices** (published D-1 at noon)
- **Use case**: `price_eur_mwh` feature for price-aware demand models. Not currently integrated but ready to plug in.

---

### `generation_forecast/v3/forecasts` ✅

Day-ahead generation forecasts by production type.

- Used by WattCast for nuclear/renewable generation forecasting
- Not needed for demand-only pipeline

---

### `actual_generation/v1/actual_generations_per_production_type` ✅

Actual generation by fuel type (nuclear, wind, solar, gas, etc.).

- Used by WattCast for generation mix analysis
- Could be useful as features for demand models (generation mix → demand context)

---

## Not Subscribed (HTTP 403)

These require separate subscriptions on the RTE portal:

| Endpoint | Status | Notes |
|----------|--------|-------|
| `consolidated_consumption/v1/consolidated_power_consumption` | 403 | Definitive 30-min data since 1996 |
| `consolidated_consumption/v1/consolidated_energy_consumption` | 403 | Definitive daily data since 1996 |
| `ecowatt/v5/signals` | 403 | Grid stress signals |
| `wholesale_market/v3/epex_france_power_auction` | 403 | Separate subscription |
| `wholesale_market/v3/eod_france_power_exchanges` | 403 | Separate subscription |
| `wholesale_market/v3/clearing_prices` | 403 | Separate subscription |

**To subscribe**: Portal → API Catalog → find the API → "Souscrire". Each API is a separate subscription. No wildcard/grouping.

**Note on `consolidated_consumption`**: Would give higher-quality definitive data (vs REALISED which can be revised), but `REALISED` is sufficient for our use case — the differences are minimal for hourly aggregates.

---

## Comparison: Local Files vs API

| Criterion | Files `data/ecomix-rte/*.zip` | API `short_term` |
|-----------|-------------------------------|-------------------|
| **Resolution** | 30 min (consumption) | **15 min** |
| **History** | 2014-2024 (local) | **2014 → today** |
| **Real-time** | No | **Yes (~minutes lag)** |
| **Quality** | Définitif (best) | REALISED (revisions possible) |
| **Credentials** | No | Yes (OAuth2) |
| **Network** | No | Yes |
| **Reproducibility** | **Total** (files versioned) | Depends on server |
| **TSO D-1 forecast** | In files (`Prévision J-1`) | In API (`type=D-1`) |

**Conclusion**: Files for reproducible training/validation. API for 2025+ live data and dashboard.

---

## Implementation in WindCast

### Current (`src/windcast/data/rte_api.py`)

- `RTEClient` — sync OAuth2 client (ported from WattCast async pattern)
- `fetch_recent_load(client, hours=48)` — fetches REALISED from `short_term`
- `get_live_actuals(client_id, secret, hours=48)` — convenience wrapper
- Resamples 15-min → hourly via `group_by_dynamic`

### Needed for dashboard backfill

- `_chunk_dates(start, end, chunk_days=180)` — split long ranges into API-safe chunks
- `fetch_load_history(client, start, end)` — chunked REALISED + D-1 fetch
- `ForecastStore` — SQLite table for prediction results

### Reference: WattCast pattern (`../wattcast/src/wattcast/data/rte.py`)

- Async `RTEClient` with `httpx.AsyncClient`
- `_chunk_dates(start, end, chunk_days=150)` for load, 20 for generation
- `collect_load()` backfills 2019-2025 by chunks → Supabase PostgreSQL
- Fetches `REALISED` + `D-1` in single call via `type=REALISED,D-1`

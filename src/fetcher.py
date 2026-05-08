"""
fetcher.py — incremental fetch of SMARD + Berlin weather, merged to hourly UTC.

Reads:
    data/external/smard_weather_2024_to_mar2026.parquet   (existing cache,
                                                           if present)
Writes:
    same path, with new hourly rows appended; deduplicated by timestamp.

Behaviour:
    - First run (no cache): full pull from 2024-01-01 to today's hour.
    - Subsequent runs:       fetch only from cache_max + 1h to today.
    - Idempotent: safe to re-run; duplicates collapse by `timestamp`.

SMARD (region DE-LU, hourly):
    Generation by source (11 series — nuclear excluded since Germany's last
    reactors shut down in April 2023 and the series is zero for this window):
        lignite, wind_offshore, hydro, other_conv, other_renew, biomass,
        wind_onshore, solar, hard_coal, pumped_gen, natural_gas
    Consumption:
        total_load, residual_load, pumped_consumption
    Price:
        day_ahead  (EUR/MWh, day-ahead auction DE/LU bidding zone)

Weather (Open-Meteo Historical Weather API, ERA5 reanalysis, ~9 km grid):
    Single location — Berlin (52.52 N, 13.405 E). Good proxy for
    temperature-driven load. Less representative for wind (concentrated on
    the north coast) and solar (stronger in the south).

Note: ERA5 has ~5 day delay for real-time. The Open-Meteo bridge auto-fills
recent days from the forecast model, but the very latest 1-2 days may show
NaN — that's fine; the dashboard handles missing values gracefully.

Usage:
    python src/fetcher.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from smard_client import (
    FILTERS_GENERATION, FILTERS_CONSUMPTION, FILTERS_PRICES, fetch_many,
)
from weather_client import fetch_germany_weather, GERMAN_CITIES

ROOT     = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / 'data' / 'external' / 'smard_weather_2024_to_mar2026.parquet'

# Earliest date we want in the cache. Cap on first-run pulls.
DEFAULT_START = datetime(2024, 1, 1)


def existing_cache() -> pd.DataFrame | None:
    if not OUT_PATH.exists():
        return None
    df = pd.read_parquet(OUT_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


def fetch_window(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch SMARD + weather for [start, end), merge on hourly UTC timestamp."""
    print('=' * 60)
    print(f'fetching SMARD {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M}')
    print('=' * 60)
    series = {
        **{k: v for k, v in FILTERS_GENERATION.items() if k != 'nuclear'},
        'total_load':         FILTERS_CONSUMPTION['total_load'],
        'residual_load':      FILTERS_CONSUMPTION['residual_load'],
        'pumped_consumption': FILTERS_CONSUMPTION['pumped_consumption'],
        'day_ahead':          FILTERS_PRICES['day_ahead_price_de_lu'],
    }
    smard = fetch_many(series, start=start, end=end,
                       region='DE-LU', resolution='hour')
    print(f'  SMARD shape: {smard.shape}')

    print('\n' + '=' * 60)
    print(f'fetching weather (Berlin) {start:%Y-%m-%d} -> {end:%Y-%m-%d}')
    print('=' * 60)
    # weather_client.fetch_germany_weather treats end as inclusive.
    wx = fetch_germany_weather(start, end, cities={'Berlin': GERMAN_CITIES['Berlin']})
    print(f'  weather shape: {wx.shape}')

    smard['timestamp'] = pd.to_datetime(smard['timestamp'], utc=True)
    wx['timestamp']    = pd.to_datetime(wx['timestamp'],    utc=True)
    merged = smard.merge(wx, on='timestamp', how='left')
    return merged


def main():
    cache = existing_cache()
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0,
                                                  microsecond=0, tzinfo=None)

    if cache is None or cache.empty:
        start = DEFAULT_START
        print(f'[fetcher] no cache → full pull from {start}')
    else:
        start = (cache['timestamp'].max()
                 + pd.Timedelta(hours=1)).to_pydatetime().replace(tzinfo=None)
        print(f'[fetcher] cache ends at {cache["timestamp"].max()} '
              f'→ fetching from {start}')

    end = now_utc
    if end <= start:
        print('[fetcher] cache is already up to date; nothing to do.')
        return cache

    new = fetch_window(start, end)
    if new.empty:
        print('[fetcher] no new rows returned; cache unchanged.')
        return cache

    if cache is None or cache.empty:
        combined = new
    else:
        combined = (pd.concat([cache, new], ignore_index=True)
                      .drop_duplicates(subset='timestamp', keep='last')
                      .sort_values('timestamp')
                      .reset_index(drop=True))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)

    print(f'\n[fetcher] wrote {OUT_PATH}')
    print(f'[fetcher] cache now spans '
          f'{combined["timestamp"].min()} -> {combined["timestamp"].max()} '
          f'({len(combined):,} rows)')
    return combined


if __name__ == '__main__':
    main()

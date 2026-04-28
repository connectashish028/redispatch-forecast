"""
Fetch German electricity market data (SMARD) + Berlin weather (Open-Meteo),
merge on hourly UTC timestamp, save as a single parquet.

Window:
    2024-01-01 through 2026-03-31 (inclusive), hourly resolution, UTC timestamps.

SMARD (region DE-LU, hourly):
    Generation by source (11 series — nuclear excluded since Germany's last
    reactors shut down in April 2023 and the series is zero for this window):
        lignite, wind_offshore, hydro, other_conv, other_renew, biomass,
        wind_onshore, solar, hard_coal, pumped_gen, natural_gas
    Consumption:
        total_load, residual_load, pumped_consumption
    Price:
        day_ahead  (EUR/MWh, day-ahead auction DE/LU bidding zone)

Weather (Open-Meteo Historical Weather API, ERA5 reanalysis, ~9 km resolution):
    Single location — Berlin (52.52 N, 13.405 E). Good proxy for temperature-driven
    load. Less representative for wind (concentrated on the north coast) and solar
    (stronger in the south). See weather_client.GERMAN_CITIES to switch or expand.
    All 32 hourly variables, prefixed `wx_` in the output columns:
        temperature/humidity, precipitation, pressure, clouds (low/mid/high),
        wind at 10 m and 100 m, five solar radiation variables, soil, ET₀, etc.

Output:
    smard_weather_2024_to_mar2026.parquet
    — one row per hour, columns = timestamp + SMARD series + wx_* weather.

Usage:
    python fetcher.py

Depends on: smard_client.py, weather_client.py in the same directory.
"""

from datetime import datetime
import pandas as pd

from smard_client import (
    FILTERS_GENERATION, FILTERS_CONSUMPTION, FILTERS_PRICES, fetch_many,
)
from weather_client import fetch_germany_weather, GERMAN_CITIES


def main():
    start = datetime(2024, 1, 1)
    end   = datetime(2026, 4, 1)  # exclusive upper bound → includes all of March 2026

    # ---- SMARD ----
    print("=" * 60)
    print("Fetching SMARD data...")
    print("=" * 60)
    series = {
        **{k: v for k, v in FILTERS_GENERATION.items() if k != "nuclear"},
        "total_load":         FILTERS_CONSUMPTION["total_load"],
        "residual_load":      FILTERS_CONSUMPTION["residual_load"],
        "pumped_consumption": FILTERS_CONSUMPTION["pumped_consumption"],
        "day_ahead":          FILTERS_PRICES["day_ahead_price_de_lu"],
    }
    smard = fetch_many(series, start=start, end=end,
                       region="DE-LU", resolution="hour")
    print(f"  SMARD shape: {smard.shape}")

    # ---- Weather ----
    print("\n" + "=" * 60)
    print("Fetching weather data (Berlin only)...")
    print("=" * 60)
    # Weather API takes end_date inclusive, so pass March 31 not April 1
    berlin_only = {"Berlin": GERMAN_CITIES["Berlin"]}
    wx = fetch_germany_weather(start, datetime(2026, 3, 31), cities=berlin_only)
    print(f"  Weather shape: {wx.shape}")

    # ---- Merge ----
    # Both sides use UTC timestamps; weather is hourly on-the-dot, SMARD hourly
    # values are labeled at the top of the hour too. Merge is exact.
    print("\nMerging...")
    smard["timestamp"] = pd.to_datetime(smard["timestamp"], utc=True)
    wx["timestamp"]    = pd.to_datetime(wx["timestamp"], utc=True)
    merged = smard.merge(wx, on="timestamp", how="left")

    print(f"\nMerged shape: {merged.shape}")
    print(f"Range: {merged['timestamp'].min()} -> {merged['timestamp'].max()}")
    print(f"Columns ({len(merged.columns)}): {list(merged.columns)}")
    print(merged.head())

    out_path = "smard_weather_2024_to_mar2026.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"\nsaved: data/external/smard_weather_2024_to_mar2026.parquet")

    return merged


if __name__ == "__main__":
    main()
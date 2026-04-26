"""
SMARD.de API client — fetch German electricity market data programmatically.

No auth required. All data under CC BY 4.0.

Endpoints:
  1. Index:      https://www.smard.de/app/chart_data/{filter}/{region}/index_{resolution}.json
                 -> list of available chunk-start timestamps (ms since epoch)
  2. Timeseries: https://www.smard.de/app/chart_data/{filter}/{region}/{filter}_{region}_{resolution}_{timestamp}.json
                 -> [timestamp_ms, value] rows for that chunk

Chunk size depends on resolution:
  - quarterhour / hour  -> 1 week per chunk
  - day / week / month / year -> 1 year per chunk
"""

from __future__ import annotations
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta  # pip install python-dateutil

BASE = "https://www.smard.de/app/chart_data"

# ---- Filter IDs (the "Main category" + "Data category" dropdowns on smard.de) ----
FILTERS_GENERATION = {
    "lignite":       1223,
    "nuclear":       1224,
    "wind_offshore": 1225,
    "hydro":         1226,
    "other_conv":    1227,
    "other_renew":   1228,
    "biomass":       4066,
    "wind_onshore":  4067,
    "solar":         4068,
    "hard_coal":     4069,
    "pumped_gen":    4070,
    "natural_gas":   4071,
}

FILTERS_CONSUMPTION = {
    "total_load":         410,
    "residual_load":      4359,
    "pumped_consumption": 4387,
}

FILTERS_PRICES = {
    "day_ahead_price_de_lu": 4169,
}

REGIONS = ("DE", "DE-LU", "DE-AT-LU", "AT", "LU",
           "50Hertz", "Amprion", "TenneT", "TransnetBW")

RESOLUTIONS = ("quarterhour", "hour", "day", "week", "month", "year")


# -------- low-level --------
def list_timestamps(filter_id: int, region: str, resolution: str) -> list[int]:
    url = f"{BASE}/{filter_id}/{region}/index_{resolution}.json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()["timestamps"]


def fetch_chunk(filter_id: int, region: str, resolution: str, ts: int) -> list[list]:
    url = f"{BASE}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{ts}.json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()["series"]


# -------- range fetch --------
def fetch_range(
    filter_id: int,
    region: str = "DE-LU",
    resolution: str = "quarterhour",
    start: datetime | None = None,
    end: datetime | None = None,
    polite_sleep: float = 0.0,
) -> pd.DataFrame:
    """Returns a DataFrame with columns [timestamp (UTC), value]. `value` may contain NaN."""
    all_ts = sorted(list_timestamps(filter_id, region, resolution))
    start_ms = int(start.replace(tzinfo=timezone.utc).timestamp() * 1000) if start else 0
    end_ms   = int(end.replace(tzinfo=timezone.utc).timestamp() * 1000) if end else 10**15

    # Keep chunks whose window overlaps [start_ms, end_ms]: chunk starts
    # at-or-before end_ms, and the next chunk starts after start_ms.
    needed = []
    for i, t in enumerate(all_ts):
        if t > end_ms:
            break
        next_t = all_ts[i + 1] if i + 1 < len(all_ts) else float("inf")
        if next_t > start_ms:
            needed.append(t)

    rows = []
    for t in needed:
        try:
            rows.extend(fetch_chunk(filter_id, region, resolution, t))
        except requests.HTTPError as e:
            print(f"  skipped chunk {t}: {e}")
        if polite_sleep:
            time.sleep(polite_sleep)

    df = pd.DataFrame(rows, columns=["timestamp_ms", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df[(df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)]
    return df[["timestamp", "value"]].reset_index(drop=True)


def fetch_last_years(
    filter_id: int,
    years: int = 3,
    region: str = "DE-LU",
    resolution: str = "quarterhour",
) -> pd.DataFrame:
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - relativedelta(years=years)
    return fetch_range(filter_id, region, resolution, start, end)


def fetch_between(
    filter_id: int,
    start: datetime,
    end: datetime,
    region: str = "DE-LU",
    resolution: str = "quarterhour",
) -> pd.DataFrame:
    """Convenience wrapper for an explicit [start, end] window."""
    return fetch_range(filter_id, region, resolution, start, end)


def fetch_many(
    series: dict[str, int],
    start: datetime | None = None,
    end: datetime | None = None,
    years: int | None = None,
    region: str = "DE-LU",
    resolution: str = "quarterhour",
) -> pd.DataFrame:
    """Pull multiple series and return a wide DataFrame, one column per series name.

    Provide either an explicit (start, end) window, or `years` for a rolling window
    ending now. If both are given, (start, end) wins.
    """
    if start is None or end is None:
        if years is None:
            raise ValueError("Either (start, end) or years must be provided.")
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        start = end - relativedelta(years=years)

    out = None
    for name, fid in series.items():
        print(f"fetching {name} (id={fid})...")
        df = fetch_range(fid, region=region, resolution=resolution,
                         start=start, end=end)
        df = df.rename(columns={"value": name}).set_index("timestamp")
        out = df if out is None else out.join(df, how="outer")
    return out.reset_index()


if __name__ == "__main__":
    # ---- Jan 2024 through March 2026, hourly, DE/LU ----
    start = datetime(2024, 1, 1)
    end   = datetime(2026, 4, 1)  # exclusive upper bound → includes all of March 2026

    series = {
        **FILTERS_GENERATION,           # all 12 generation sources
        "total_load":  FILTERS_CONSUMPTION["total_load"],
        "day_ahead":   FILTERS_PRICES["day_ahead_price_de_lu"],
    }

    df = fetch_many(series, start=start, end=end,
                    region="DE-LU", resolution="hour")
    print(df.head())
    print(f"\nshape: {df.shape}")
    print(f"range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    df.to_parquet("smard_2024_to_mar2026.parquet", index=False)
    # df.to_csv("smard_2024_to_mar2026.csv", index=False)
    print("saved: smard_2024_to_mar2026.parquet")
"""
Open-Meteo weather client — population-weighted German average.

Uses the free Historical Weather API (ERA5 reanalysis, ~9 km resolution, 1940-present,
~5 day delay for real-time). No API key, CC BY 4.0.

Endpoint: https://archive-api.open-meteo.com/v1/archive

The API automatically bridges to the forecast model for the most recent ~5 days,
so you get continuous coverage through yesterday.
"""

from __future__ import annotations
import time
import requests
import pandas as pd
from datetime import datetime

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Top German cities by population with (lat, lon, population in millions).
# Population values are ~2023 official figures; weights are computed from these,
# so exact precision doesn't matter — only the ratios do.
GERMAN_CITIES = {
    "Berlin":       (52.5200, 13.4050, 3.755),
    "Hamburg":      (53.5511, 9.9937,  1.892),
    "Munich":       (48.1351, 11.5820, 1.512),
    "Cologne":      (50.9375, 6.9603,  1.086),
    "Frankfurt":    (50.1109, 8.6821,  0.773),
    "Stuttgart":    (48.7758, 9.1829,  0.633),
    "Dusseldorf":   (51.2277, 6.7735,  0.629),
    "Leipzig":      (51.3397, 12.3731, 0.617),
    "Dortmund":     (51.5136, 7.4653,  0.593),
    "Essen":        (51.4556, 7.0116,  0.584),
    "Bremen":       (53.0793, 8.8017,  0.566),
    "Dresden":      (51.0504, 13.7373, 0.563),
    "Hannover":     (52.3759, 9.7320,  0.546),
    "Nuremberg":    (49.4521, 11.0767, 0.523),
}

# All the hourly variables Open-Meteo's archive exposes that are relevant for
# energy/load analysis. Grouped for readability — the API accepts one flat list.
HOURLY_VARIABLES = [
    # --- temperature & humidity ---
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    # --- precipitation ---
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    # --- pressure ---
    "pressure_msl",
    "surface_pressure",
    # --- clouds ---
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    # --- wind (10m = standard, 100m = roughly wind-turbine hub height) ---
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",
    # --- solar radiation (the key drivers for PV output) ---
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
    "terrestrial_radiation",
    # --- soil / evapotranspiration ---
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
    # --- misc ---
    "is_day",
    "sunshine_duration",
]


def _get_with_retry(url: str, params: dict, max_retries: int = 6) -> dict:
    """GET with exponential backoff on 429 / 5xx. Honours Retry-After when sent."""
    delay = 5.0
    for attempt in range(max_retries):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            wait = float(r.headers.get("Retry-After", delay))
            print(f"    {r.status_code}; sleeping {wait:.0f}s "
                  f"(attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            delay = min(delay * 2, 120)
            continue
        r.raise_for_status()
    r.raise_for_status()  # final failure
    return {}


def _fetch_window(
    lat: float, lon: float,
    start: datetime, end: datetime,
    variables: list[str],
) -> pd.DataFrame:
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date":   end.strftime("%Y-%m-%d"),
        "hourly":     ",".join(variables),
        "timezone":   "UTC",
    }
    data = _get_with_retry(ARCHIVE_URL, params)["hourly"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    return df.drop(columns=["time"]).set_index("timestamp")


def fetch_city(
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    variables: list[str] = HOURLY_VARIABLES,
    chunk_years: int = 1,
    between_chunk_sleep: float = 2.0,
) -> pd.DataFrame:
    """Fetch hourly weather for a single location, chunked by year to stay under
    Open-Meteo's per-request payload limits. Returns UTC-indexed DataFrame."""
    parts = []
    cur = start
    while cur <= end:
        chunk_end = min(
            datetime(cur.year + chunk_years, 1, 1) - pd.Timedelta(days=1),
            end,
        )
        parts.append(_fetch_window(lat, lon, cur, chunk_end, variables))
        cur = chunk_end + pd.Timedelta(days=1)
        if cur <= end and between_chunk_sleep:
            time.sleep(between_chunk_sleep)
    return pd.concat(parts).sort_index()


def fetch_germany_weather(
    start: datetime,
    end: datetime,
    cities: dict = GERMAN_CITIES,
    variables: list[str] = HOURLY_VARIABLES,
    polite_sleep: float = 3.0,
) -> pd.DataFrame:
    """Fetch weather for each city, then compute a population-weighted average.

    Returns a long wide DataFrame with one column per variable (UTC timestamp index
    reset to a column).
    """
    # Pull each city
    per_city: dict[str, pd.DataFrame] = {}
    weights = {}
    total_pop = sum(pop for _, _, pop in cities.values())
    for name, (lat, lon, pop) in cities.items():
        print(f"  fetching weather for {name}...")
        try:
            per_city[name] = fetch_city(lat, lon, start, end, variables)
            weights[name] = pop / total_pop
        except Exception as e:
            print(f"    FAILED for {name}: {e}. Skipping this city.")
        if polite_sleep:
            time.sleep(polite_sleep)

    if not per_city:
        raise RuntimeError("All city fetches failed — check network / rate limits.")

    # Re-normalize weights across the cities we actually got
    w_sum = sum(weights.values())
    weights = {k: v / w_sum for k, v in weights.items()}

    # Align on timestamp, then weighted-average each variable.
    # Build a DataFrame per variable with one column per city, then dot with weights.
    any_city = next(iter(per_city.values()))
    index = any_city.index
    out = pd.DataFrame(index=index)

    for var in variables:
        # Some variables may not be present in the response for certain date ranges.
        cols = []
        ws = []
        for name, df in per_city.items():
            if var in df.columns:
                cols.append(df[var].reindex(index))
                ws.append(weights[name])
        if not cols:
            continue
        stacked = pd.concat(cols, axis=1)
        stacked.columns = list(range(len(cols)))
        ws_arr = pd.Series(ws, index=stacked.columns)
        # Weighted mean that ignores NaNs per-row (re-normalize weights of available cities).
        mask = stacked.notna()
        weighted = (stacked.fillna(0) * ws_arr).sum(axis=1)
        norm = (mask * ws_arr).sum(axis=1)
        out[var] = (weighted / norm).where(norm > 0)

    out = out.reset_index()
    # Prefix to distinguish from SMARD columns when merged later
    out = out.rename(columns={c: f"wx_{c}" for c in out.columns if c != "timestamp"})
    return out


if __name__ == "__main__":
    start = datetime(2024, 1, 1)
    end   = datetime(2026, 3, 31)
    wx = fetch_germany_weather(start, end)
    print(wx.head())
    print(f"\nshape: {wx.shape}")
    print(f"columns: {list(wx.columns)}")
    wx.to_parquet("weather_de_2024_to_mar2026.parquet", index=False)
    print("saved: weather_de_2024_to_mar2026.parquet")
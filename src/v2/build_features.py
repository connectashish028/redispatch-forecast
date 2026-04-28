"""
build_features.py - hourly feature table for the day-ahead redispatch model.

Reads:
  data/processed/ts_15min_wide.parquet                  (raw 15-min activity per town,
                                                         dominant reasons only:
                                                         Netzengpass + Netzengpass I)
  data/external/smard_weather_2024_to_mar2026.parquet   (national SMARD + weather, hourly)
  data/external/towns_geo.parquet                       (lat/lon per town)

Writes:
  data/processed/features.parquet                       (hourly x town, labels + features)

Design choices
--------------
* Hourly grain, not 15-min. The day-ahead forecast horizon makes intra-hour
  resolution fake precision: weather/SMARD inputs are hourly anyway, and 24h
  forecast skill at 15-min granularity is below the noise floor.
* Labels are forward windows starting strictly after ts (ts+1h .. ts+H).
* Every lag feature is shifted by 24h before any rolling window, so the entire
  feature row at ts is knowable at T_fcst = ts - 24h. No leakage at 24h horizon.
* Warmup: the first 7d+24h are dropped (lag-7d window incomplete).
* Cooldown: the last 24h are dropped (y_24h window incomplete).
* v1 uses SMARD/weather actuals as proxies for day-ahead forecasts. The
  train/serve skew is bounded (~80-85% accuracy for 24h wind-speed forecasts).
  v2: swap to SMARD forecast filter_ids + Open-Meteo previous-forecast archive.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parents[2]
WIDE_IN   = ROOT / 'data' / 'processed' / 'ts_15min_wide.parquet'
EXT_IN    = ROOT / 'data' / 'external'  / 'smard_weather_2024_to_mar2026.parquet'
GEO_IN    = ROOT / 'data' / 'external'  / 'towns_geo.parquet'
OUT_PATH  = ROOT / 'data' / 'processed' / 'features.parquet'

LAG_H     = 24                                            # forecast lead time (h)
HORIZONS  = {'y_1h': 1, 'y_6h': 6, 'y_24h': 24}           # forward windows (h)
LOOKBACKS = {'active_24h_lag24': 24, 'active_7d_lag24': 24*7}

SMARD_COLS = ['residual_load', 'total_load', 'wind_onshore', 'wind_offshore',
              'solar', 'day_ahead']
WX_COLS    = ['wx_wind_speed_100m', 'wx_wind_speed_10m',
              'wx_wind_direction_100m', 'wx_wind_gusts_10m',
              'wx_shortwave_radiation', 'wx_temperature_2m',
              'wx_cloud_cover', 'wx_precipitation']


def load_hourly_active() -> pd.DataFrame:
    """15-min wide -> hourly any-active wide. bool DataFrame indexed by hour."""
    wide = pd.read_parquet(WIDE_IN)
    hourly = (wide > 0).resample('1h').max().astype(bool)
    return hourly


def fwd_any(M: np.ndarray, h: int) -> np.ndarray:
    """For each (t, town): any True in M[t+1 .. t+h]."""
    T, _ = M.shape
    out = np.zeros_like(M, dtype=bool)
    for k in range(1, h + 1):
        if k < T:
            out[:T - k] |= M[k:]
    return out


def back_count(M: np.ndarray, lookback: int, lag: int) -> np.ndarray:
    """Count of True in M[t - lag - lookback : t - lag] for each (t, town).
    Cumsum trick: O(T*N), no time loop. NaN-safe (input is bool)."""
    T, N = M.shape
    csum = np.zeros((T + 1, N), dtype=np.int32)
    csum[1:] = np.cumsum(M.astype(np.int32), axis=0)
    out = np.zeros((T, N), dtype=np.int16)
    t  = np.arange(T)
    lo = t - lag - lookback
    hi = t - lag
    ok = lo >= 0
    out[ok] = (csum[hi[ok]] - csum[lo[ok]]).astype(np.int16)
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    ts = df['ts']
    df['hour']       = ts.dt.hour.astype('int8')
    df['dow']        = ts.dt.dayofweek.astype('int8')
    df['month']      = ts.dt.month.astype('int8')
    df['is_weekend'] = (df['dow'] >= 5).astype('int8')
    try:
        import holidays
        de_h = holidays.Germany(years=range(2024, 2027))
        df['is_de_holiday'] = ts.dt.date.map(lambda d: d in de_h).astype('int8')
    except Exception as e:
        print(f'  (holidays unavailable: {e}; is_de_holiday=0)')
        df['is_de_holiday'] = np.int8(0)
    return df


def main():
    print('loading hourly active matrix...')
    hourly_active = load_hourly_active()
    T, N = hourly_active.shape
    towns = list(hourly_active.columns)
    times = hourly_active.index
    print(f'  hourly: {T:,} hours x {N} towns')
    print(f'  range : {times[0]}  ->  {times[-1]}')

    M = hourly_active.values   # bool (T, N)

    print('computing labels (forward windows)...')
    labels = {name: fwd_any(M, h) for name, h in HORIZONS.items()}

    print('computing lag features (24h shift + count windows)...')
    lags = {name: back_count(M, lb, LAG_H) for name, lb in LOOKBACKS.items()}

    print('stacking to long (ts, town)...')
    ts_col   = np.repeat(times.values, N)
    town_col = np.tile(towns, T)
    base = pd.DataFrame({
        'ts':        ts_col,
        'town':      town_col,
        'is_active': M.reshape(-1),
    })
    for name, arr in labels.items():
        base[name] = arr.reshape(-1).astype('int8')
    for name, arr in lags.items():
        base[name] = arr.reshape(-1)

    # warmup/cooldown trim
    warmup_h   = LAG_H + max(LOOKBACKS.values())   # 24 + 168 = 192h
    cooldown_h = max(HORIZONS.values())            # 24h
    t_lo = times[warmup_h]
    t_hi = times[T - cooldown_h - 1]
    base = base[(base['ts'] >= t_lo) & (base['ts'] <= t_hi)].reset_index(drop=True)
    print(f'  trimmed warmup({warmup_h}h) + cooldown({cooldown_h}h): {len(base):,} rows')
    print(f'  usable range: {t_lo}  ->  {t_hi}')

    print('joining SMARD + weather...')
    ext = pd.read_parquet(EXT_IN)
    ext['ts'] = pd.to_datetime(ext['timestamp']).dt.tz_convert('UTC').dt.tz_localize(None)
    have   = [c for c in SMARD_COLS + WX_COLS if c in ext.columns]
    miss   = set(SMARD_COLS + WX_COLS) - set(have)
    if miss:
        print(f'  WARN: cache missing {miss}')
    ext = ext[['ts'] + have]
    base = base.merge(ext, on='ts', how='left')
    nan_share = base[have[0]].isna().mean()
    print(f'  SMARD/wx NaN share post-join: {nan_share:.4%}')
    if nan_share > 0:
        base = base.sort_values(['town', 'ts']).reset_index(drop=True)
        base[have] = base.groupby('town')[have].ffill()

    print('joining geography (lat, lon)...')
    geo = pd.read_parquet(GEO_IN)[['town', 'lat', 'lon']]
    base = base.merge(geo, on='town', how='left')
    n_no_geo = base['lat'].isna().sum()
    if n_no_geo:
        print(f'  WARN: {n_no_geo} rows lack geocoding')

    print('adding calendar features...')
    base = add_calendar(base)

    # dtype downcasts
    for c in have + ['lat', 'lon']:
        if c in base.columns:
            base[c] = base[c].astype('float32')
    for c in LOOKBACKS:
        base[c] = base[c].astype('int16')
    base['town']      = base['town'].astype('category')
    base['is_active'] = base['is_active'].astype('int8')

    print(f'\nfinal shape: {base.shape[0]:,} rows x {base.shape[1]} cols')
    print('positive label rates:')
    for k in HORIZONS:
        print(f'  {k}: {base[k].mean():.4f}')
    print('column dtypes:')
    print(base.dtypes)

    print(f'\\nwriting data/processed/features.parquet...')
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(OUT_PATH, index=False)
    print('done.')


if __name__ == '__main__':
    main()

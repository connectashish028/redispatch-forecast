"""
score_today.py - generate redispatch predictions for one date.

Reads:
  data/processed/features.parquet     (built by build_features.py)
  data/external/towns_geo.parquet
  models/model_y_{1h,6h,24h}.txt
  models/calibrator_y_{1h,6h,24h}.joblib
  models/feature_cols.json

Writes:
  data/predictions/predictions_<YYYY-MM-DD>.parquet
  data/predictions/latest.parquet     (copy of the most recent run)

Usage:
  python src/score_today.py                          # tomorrow (UTC)
  python src/score_today.py --date 2026-02-15        # explicit date
  python src/score_today.py --threshold 50 --quiet   # alert-style: silent + 50% bar
  python src/score_today.py --date 2026-02-15 --out custom/dir/

Exit codes:
  0  success
  1  no features available for that date (upstream pipeline not run / data gap)
  2  user error (bad date format, missing model files)

NOTE on v1 scope:
  This script scores against features.parquet, which is built from cached
  *actual* SMARD + weather values (build_features.py). It works end-to-end
  for any date already in the feature table (the test window). To score a
  truly-future date, item 2 in the v1 deployment plan adds a live-data
  fetcher upstream; this script's interface and outputs do not change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parents[2]
FEAT       = ROOT / 'data' / 'processed' / 'features.parquet'
GEO        = ROOT / 'data' / 'external'  / 'towns_geo.parquet'
MODELS_DIR = ROOT / 'models'
OUT_DIR    = ROOT / 'data' / 'predictions'

HORIZONS = ['y_1h', 'y_6h', 'y_24h']
LOW_DATA_POS_THRESHOLD = 5         # mirror the dashboard convention


def _model_version() -> str:
    """Short hash over model artifact bytes — proxy for 'which trained model is this?'.
    Stable across re-runs unless the artifacts change. Better than mtime."""
    h = hashlib.sha1()
    for f in sorted(MODELS_DIR.glob('model_y_*.txt')):
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


def _low_data_towns() -> set[str]:
    """Towns with too few positives in the test slice to trust per-town predictions.
    Computed at scoring time so it follows the data without needing a separate file."""
    feats = pd.read_parquet(FEAT, columns=['ts', 'town', 'y_24h'])
    test = feats[feats['ts'] >= pd.Timestamp('2026-01-01')]
    pos = test.groupby('town', observed=True)['y_24h'].sum()
    return set(pos[pos <= LOW_DATA_POS_THRESHOLD].index.astype(str))


def score(target_date: pd.Timestamp) -> pd.DataFrame:
    """Run the three models over all towns × all hours for target_date.
    Returns a DataFrame with predictions + metadata, ready for parquet write."""
    # late import so a missing model file fails *after* arg parsing, not at import time
    import joblib
    import lightgbm as lgb

    # 1. load feature column order (single source of truth for inference)
    fcols_path = MODELS_DIR / 'feature_cols.json'
    if not fcols_path.exists():
        print(f'[error] missing {fcols_path}. Did you run train.py?', file=sys.stderr)
        sys.exit(2)                                           # config error
    with open(fcols_path) as f:
        fcols = json.load(f)

    # 2. load models + calibrators
    models, cals = {}, {}
    for h in HORIZONS:
        mp = MODELS_DIR / f'model_{h}.txt'
        cp = MODELS_DIR / f'calibrator_{h}.joblib'
        if not mp.exists() or not cp.exists():
            print(f'[error] missing model artifacts for {h}: {mp.name} / {cp.name}',
                  file=sys.stderr)
            sys.exit(2)                                       # config error
        models[h] = lgb.Booster(model_file=str(mp))
        cals[h]   = joblib.load(cp)

    # 3. load features for target_date
    raw = pd.read_parquet(FEAT)
    sub = raw[(raw['ts'] >= target_date) &
              (raw['ts'] <  target_date + pd.Timedelta(days=1))].copy()
    if sub.empty:
        avail_lo = raw['ts'].min(); avail_hi = raw['ts'].max()
        print(
            f'[error] no features for {target_date.date()}. '
            f'Feature table covers {avail_lo} to {avail_hi}.\n'
            f'Either pick a date in that range, or run the upstream pipeline '
            f'(build_timeseries.py + build_features.py) for the date you want.',
            file=sys.stderr,
        )
        sys.exit(1)                                           # data not ready

    # 4. one-hot encode town to match training-time schema
    raw_town = sub['town'].copy()
    town_oh = pd.get_dummies(sub['town'], prefix='town', dtype='int8')
    sub = pd.concat([sub.drop(columns='town'), town_oh], axis=1)
    # any training town column missing in this slice must be added as zeros
    for c in fcols:
        if c not in sub.columns:
            sub[c] = np.int8(0)
    X = sub[fcols].values

    # 5. score
    out = pd.DataFrame({'ts': sub['ts'].values, 'town': raw_town.values})
    for h in HORIZONS:
        raw_p = models[h].predict(X)
        cal_p = cals[h].predict(raw_p)
        out[f'p_{h.split("_")[1]}']      = cal_p.astype('float32')
        out[f'p_{h.split("_")[1]}_raw']  = raw_p.astype('float32')

    # 6. join geography
    geo = pd.read_parquet(GEO)[['town', 'lat', 'lon']]
    out = out.merge(geo, on='town', how='left')

    # 7. metadata columns
    out['is_low_data']        = out['town'].isin(_low_data_towns())
    out['prediction_made_at'] = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    out['model_version']      = _model_version()

    # 8. column order — keys first, predictions, geo, metadata
    cols = ['ts', 'town', 'lat', 'lon',
            'p_1h', 'p_6h', 'p_24h',
            'p_1h_raw', 'p_6h_raw', 'p_24h_raw',
            'is_low_data', 'prediction_made_at', 'model_version']
    return out[cols].sort_values(['ts', 'town']).reset_index(drop=True)


def print_summary(df: pd.DataFrame, threshold_pct: int) -> None:
    threshold = threshold_pct / 100
    n_alerts = int((df['p_24h'] >= threshold).sum())
    n_high   = int((df['p_24h'] >= 0.5).sum())

    print('=' * 70)
    print(f'date          : {df["ts"].dt.date.iloc[0]}')
    print(f'rows scored   : {len(df):,}  '
          f'({df["town"].nunique()} towns × {df["ts"].nunique()} hours)')
    print(f'model version : {df["model_version"].iloc[0]}')
    print(f'scored at     : {df["prediction_made_at"].iloc[0]:%Y-%m-%d %H:%M:%S} UTC')
    print('-' * 70)
    print(f'rows above {threshold_pct:>2}% (alerts) : {n_alerts:,}')
    print(f'rows above 50% (high risk) : {n_high:,}')
    print(f'highest p_24h              : {df["p_24h"].max():.1%}')
    print(f'grid avg p_24h             : {df["p_24h"].mean():.1%}')

    print('-' * 70)
    print('Top 10 (town, hour) by Next 24 hours probability:')
    top = df.nlargest(10, 'p_24h')[['ts', 'town', 'p_1h', 'p_6h', 'p_24h', 'is_low_data']]
    for _, r in top.iterrows():
        flag = '⚠' if r['is_low_data'] else ' '
        print(f'  {r["ts"]:%H:%M}  {flag} {r["town"][:30]:<30s}  '
              f'p_1h={r["p_1h"]:>5.1%}  p_6h={r["p_6h"]:>5.1%}  p_24h={r["p_24h"]:>5.1%}')
    print('=' * 70)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--date', type=str, default=None,
                    help='YYYY-MM-DD; defaults to tomorrow (UTC).')
    ap.add_argument('--out', type=Path, default=OUT_DIR,
                    help=f'Output dir; default {OUT_DIR}.')
    ap.add_argument('--threshold', type=int, default=30,
                    help='Alert threshold percentage for the summary; default 30.')
    ap.add_argument('--quiet', action='store_true',
                    help='Skip the printed summary; only write the parquet file.')
    args = ap.parse_args()

    if args.date is None:
        target = pd.Timestamp(datetime.now(timezone.utc).date()) + pd.Timedelta(days=1)
    else:
        try:
            target = pd.Timestamp(args.date).normalize()
        except ValueError:
            print(f'[error] could not parse --date {args.date!r} as YYYY-MM-DD',
                  file=sys.stderr)
            sys.exit(2)                                       # user error

    args.out.mkdir(parents=True, exist_ok=True)

    df = score(target)

    out_path    = args.out / f'predictions_{target.date()}.parquet'
    latest_path = args.out / 'latest.parquet'
    df.to_parquet(out_path, index=False)
    shutil.copyfile(out_path, latest_path)

    if not args.quiet:
        print_summary(df, args.threshold)
        print(f'wrote {out_path}')
        print(f'wrote {latest_path}  (copy of the most recent run)')


if __name__ == '__main__':
    main()

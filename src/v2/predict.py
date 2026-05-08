"""
predict.py - day-ahead redispatch probability for a single town.

Usage as library:
    from predict import predict_day
    df = predict_day('Husum', '2026-02-15')
    # df has 24 rows: ts, p_1h, p_6h, p_24h, plus the realized y_* if available.

Usage as CLI:
    python src/predict.py --town "Husum" --date 2026-02-15
    python src/predict.py --town "Husum" --date 2026-02-15 --plot

Notes
-----
* For now, features at any target hour are read from data/processed/features.parquet
  - i.e. we predict on dates already seen in the offline pipeline. To predict
  for a date strictly in the future, build_features.py would need a "forecast"
  variant that fetches live SMARD + Open-Meteo day-ahead forecast values.
* p_1h(ts)  = probability of any redispatch in the hour starting at ts
  p_6h(ts)  = probability of any redispatch in the next 6 hours after ts
  p_24h(ts) = probability of any redispatch in the next 24 hours after ts
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT       = Path(__file__).resolve().parents[2]
FEAT       = ROOT / 'data' / 'processed' / 'features.parquet'
MODELS_DIR = ROOT / 'models'

HORIZONS = ['y_1h', 'y_6h', 'y_24h']

# lazy-loaded module-level caches so repeated calls are fast
_FEATURES = None
_MODELS   = None
_CALS     = None
_FCOLS    = None


def _load():
    global _FEATURES, _MODELS, _CALS, _FCOLS
    if _FEATURES is None:
        _FEATURES = pd.read_parquet(FEAT)
        # one-hot encode town the same way train.py did
        town_oh = pd.get_dummies(_FEATURES['town'], prefix='town', dtype='int8')
        _FEATURES = pd.concat([_FEATURES.drop(columns='town'), town_oh], axis=1)
        # restore town as a non-feature key column for filtering
        _FEATURES['town'] = pd.read_parquet(FEAT, columns=['town'])['town'].values
    if _MODELS is None:
        _MODELS = {h: lgb.Booster(model_file=str(MODELS_DIR / f'model_{h}.txt'))
                   for h in HORIZONS}
        _CALS   = {h: joblib.load(MODELS_DIR / f'calibrator_{h}.joblib')
                   for h in HORIZONS}
        with open(MODELS_DIR / 'feature_cols.json') as f:
            _FCOLS = json.load(f)
    return _FEATURES, _MODELS, _CALS, _FCOLS


def predict_day(town: str, date: str | pd.Timestamp) -> pd.DataFrame:
    """Return hour-by-hour redispatch probabilities for `town` on `date`."""
    feats, models, cals, fcols = _load()
    d  = pd.Timestamp(date).normalize()
    lo = d
    hi = d + pd.Timedelta(hours=23)

    sub = feats[(feats['town'] == town) &
                (feats['ts'] >= lo) &
                (feats['ts'] <= hi)].copy()
    if sub.empty:
        towns = sorted(feats['town'].unique())
        ts_min, ts_max = feats['ts'].min(), feats['ts'].max()
        raise ValueError(
            f'No features for town={town!r} on {d.date()}. '
            f'Available date range: {ts_min} -> {ts_max}. '
            f'Town in dataset: {town in towns}.'
        )

    X = sub[fcols].copy()
    out = sub[['ts']].reset_index(drop=True)
    for h in HORIZONS:
        raw   = models[h].predict(X.values)
        cal   = cals[h].predict(raw)
        out[f'p_{h.split("_")[1]}']     = cal
        out[f'p_{h.split("_")[1]}_raw'] = raw
        if h in sub.columns:
            out[h] = sub[h].values
    out.insert(1, 'town', town)
    return out


def plot_day(df: pd.DataFrame, town: str, date: str, save: Path | None = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = df['ts'].dt.hour
    ax.plot(x, df['p_1h'],  marker='o', label='P(active in next 1h)',  color='#d62728')
    ax.plot(x, df['p_6h'],  marker='s', label='P(active in next 6h)',  color='#ff7f0e')
    ax.plot(x, df['p_24h'], marker='^', label='P(active in next 24h)', color='#1f77b4')

    # overlay realized labels if available (for backtests)
    for h, color in zip(['y_1h', 'y_6h', 'y_24h'], ['#d62728', '#ff7f0e', '#1f77b4']):
        if h in df.columns:
            mask = df[h] == 1
            if mask.any():
                ax.scatter(x[mask], df.loc[mask, f'p_{h.split("_")[1]}'],
                           s=140, facecolors='none', edgecolors=color, linewidths=2,
                           label=f'{h}=1 (actual)' if h == 'y_1h' else None, zorder=5)

    ax.set_xticks(range(0, 24))
    ax.set_xlabel('hour of day (UTC)')
    ax.set_ylabel('probability')
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.set_title(f'{town}  -  redispatch probability  -  {pd.Timestamp(date).date()}')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
        print(f'saved {save}')
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--town', required=True)
    p.add_argument('--date', required=True, help='YYYY-MM-DD')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--save', type=Path, default=None, help='png path for plot')
    args = p.parse_args()

    df = predict_day(args.town, args.date)
    print(df.to_string(index=False, float_format='%.3f'))
    if 'y_1h' in df.columns:
        actual_24h = df['y_24h'].max() if 'y_24h' in df.columns else None
        if actual_24h is not None:
            print(f'\nactual: any redispatch on this day in this town? {bool(actual_24h)}')

    if args.plot or args.save:
        plot_day(df, args.town, args.date, save=args.save)


if __name__ == '__main__':
    main()

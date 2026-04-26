"""
review_model.py - end-to-end diagnostic of the trained models.

Checks:
  1. Calibration   — predicted decile vs actual frequency
  2. Geography     — predicted intensity correlates with actual intensity per town
  3. Per-town PR-AUC distribution
  4. Day-level signal — daily mean prediction correlates with daily actual rate
  5. Spread / constant-prediction check
  6. Spot checks on busy (town, date) pairs
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT       = Path(__file__).resolve().parents[1]
FEAT       = ROOT / 'data' / 'processed' / 'features.parquet'
MODELS_DIR = ROOT / 'models'
HORIZONS   = ['y_1h', 'y_6h', 'y_24h']

print('loading features + models...')
df = pd.read_parquet(FEAT)
town_oh = pd.get_dummies(df['town'], prefix='town', dtype='int8')
df = pd.concat([df.drop(columns='town'), town_oh], axis=1)
df['town'] = pd.read_parquet(FEAT, columns=['town'])['town'].values

with open(MODELS_DIR / 'feature_cols.json') as f:
    fcols = json.load(f)

models = {h: lgb.Booster(model_file=str(MODELS_DIR / f'model_{h}.txt')) for h in HORIZONS}
cals   = {h: joblib.load(MODELS_DIR / f'calibrator_{h}.joblib') for h in HORIZONS}

test = df[df['ts'] >= pd.Timestamp('2026-01-01')].copy()
print(f'test rows: {len(test):,}, towns: {test["town"].nunique()}, '
      f'date range: {test["ts"].min()} -> {test["ts"].max()}')

# predict for the test set
X = test[fcols].values
for h in HORIZONS:
    raw = models[h].predict(X)
    cal = cals[h].predict(raw)
    test[f'p_{h.split("_")[1]}'] = cal

# --------------------------------------------
# 1. Calibration table — bin predicted prob into deciles, check actual rate
# --------------------------------------------
print('\n' + '='*72)
print('1. CALIBRATION  (predicted vs actual rate, in deciles of prediction)')
print('='*72)
for h in HORIZONS:
    pcol = 'p_' + h.split('_')[1]
    bins = pd.qcut(test[pcol], 10, duplicates='drop', labels=False)
    cal_df = test.groupby(bins).agg(
        pred_mean=(pcol, 'mean'),
        actual_mean=(h, 'mean'),
        n=(pcol, 'size'),
    )
    miscal = (cal_df['pred_mean'] - cal_df['actual_mean']).abs().mean()
    print(f'\n--- {h}  (mean abs miscalibration across deciles: {miscal:.3f})')
    cal_df['gap'] = cal_df['pred_mean'] - cal_df['actual_mean']
    print(cal_df.round(3).to_string())

# --------------------------------------------
# 2. Geographic sanity — predicted vs actual per-town intensity
# --------------------------------------------
print('\n' + '='*72)
print('2. GEOGRAPHIC SANITY  (per-town predicted-mean vs actual-rate)')
print('='*72)
for h in HORIZONS:
    pcol = 'p_' + h.split('_')[1]
    by_town = test.groupby('town').agg(
        pred=(pcol, 'mean'),
        actual=(h, 'mean'),
        n=('ts', 'size'),
    )
    rho_p = by_town['pred'].corr(by_town['actual'])
    rho_s = by_town['pred'].rank().corr(by_town['actual'].rank())
    print(f'\n--- {h}: Pearson corr={rho_p:.3f}, Spearman={rho_s:.3f}')
    print('top-10 actual:')
    print(by_town.nlargest(10, 'actual')[['actual','pred','n']].round(3).to_string())
    print('\ntop-10 predicted:')
    print(by_town.nlargest(10, 'pred')[['pred','actual','n']].round(3).to_string())

# --------------------------------------------
# 3. Per-town PR-AUC distribution
# --------------------------------------------
print('\n' + '='*72)
print('3. PER-TOWN PR-AUC DISTRIBUTION  (y_24h)')
print('='*72)
rows = []
for tw, grp in test.groupby('town'):
    if grp['y_24h'].sum() < 5:    # skip towns with too few positives
        continue
    rows.append({
        'town': tw,
        'pos': int(grp['y_24h'].sum()),
        'base_rate': grp['y_24h'].mean(),
        'pr_auc': average_precision_score(grp['y_24h'], grp['p_24h']),
    })
ptaux = pd.DataFrame(rows).sort_values('pr_auc', ascending=False)
ptaux['lift'] = ptaux['pr_auc'] / ptaux['base_rate']
print(f'eligible towns (>=5 positives in test): {len(ptaux)}')
print(f'  mean PR-AUC : {ptaux["pr_auc"].mean():.3f}')
print(f'  median lift : {ptaux["lift"].median():.2f}x')
print(f'  P(lift>1)   : {(ptaux["lift"]>1).mean():.1%}')
print(f'\nbest 5:')
print(ptaux.head(5).round(3).to_string(index=False))
print(f'\nworst 5:')
print(ptaux.tail(5).round(3).to_string(index=False))

# --------------------------------------------
# 4. Day-level signal — daily mean prediction vs daily actual rate
# --------------------------------------------
print('\n' + '='*72)
print('4. DAY-LEVEL SIGNAL  (daily mean prediction vs daily actual rate)')
print('='*72)
test['date'] = test['ts'].dt.date
for h in HORIZONS:
    pcol = 'p_' + h.split('_')[1]
    by_day = test.groupby('date').agg(pred=(pcol,'mean'), actual=(h,'mean'))
    rho = by_day['pred'].corr(by_day['actual'])
    rho_s = by_day['pred'].rank().corr(by_day['actual'].rank())
    print(f'  {h}: Pearson={rho:.3f}, Spearman={rho_s:.3f}, n_days={len(by_day)}')

# --------------------------------------------
# 5. Spread / constant-prediction check
# --------------------------------------------
print('\n' + '='*72)
print('5. PREDICTION SPREAD  (is the model just predicting the same number?)')
print('='*72)
for h in HORIZONS:
    pcol = 'p_' + h.split('_')[1]
    p = test[pcol]
    print(f'  {h}: min={p.min():.4f}  p1={p.quantile(0.01):.3f}  '
          f'p50={p.median():.3f}  p99={p.quantile(0.99):.3f}  max={p.max():.3f}  '
          f'std={p.std():.3f}')

# --------------------------------------------
# 6. Spot checks on a few (town, date) combos
# --------------------------------------------
print('\n' + '='*72)
print('6. SPOT CHECKS  (busiest test (town, date) pairs)')
print('='*72)
top_pairs = (test[test['y_24h']==1]
             .groupby(['town', test['date']])
             .size().rename('n_active_hours_24h_window')
             .sort_values(ascending=False).head(10).reset_index())
print(top_pairs.to_string(index=False))

print('\nFor the top-3 of those, here are the day-mean predicted vs actual rates:')
for _, r in top_pairs.head(3).iterrows():
    sub = test[(test['town']==r['town']) & (test['date']==r['date'])]
    p24_mean = sub['p_24h'].mean(); p24_max = sub['p_24h'].max()
    a24      = sub['y_24h'].mean()
    print(f'  {r["town"]:25s}  {r["date"]}   '
          f'p_24h mean={p24_mean:.2%}  max={p24_max:.2%}   '
          f'actual_24h_rate={a24:.2%}')

# --------------------------------------------
# Final verdict line
# --------------------------------------------
print('\n' + '='*72)
print('DONE.')

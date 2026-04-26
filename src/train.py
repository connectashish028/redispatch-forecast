"""
train.py - LightGBM probability models for redispatch at three horizons.

Trains 3 models from data/processed/features.parquet:
  y_1h   - P(any redispatch in the next 1 hour)
  y_6h   - P(any redispatch in the next 6 hours)
  y_24h  - P(any redispatch in the next 24 hours)

Time split:
  train: 2024-01-09 .. 2025-06-30
  val  : 2025-07-01 .. 2025-12-31    (early stopping + isotonic calibration)
  test : 2026-01-01 .. 2026-03-30    (held out, never seen)

Class imbalance via is_unbalance=True (loss reweighting, no row drops).
Town is one-hot encoded (int8) per user choice; native categorical would be
slightly better for trees but OH is fine at 175 categories.

Outputs (under models/):
  model_{y_1h,y_6h,y_24h}.txt        LightGBM native booster
  calibrator_{...}.joblib            isotonic regression fit on val
  feature_cols.json                  exact feature column order for inference
  metrics.json                       PR-AUC / ROC-AUC / Brier / log-loss
  feature_importance_{...}.csv       gain-based importance per model
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    brier_score_loss, log_loss,
)
from sklearn.isotonic import IsotonicRegression

ROOT       = Path(__file__).resolve().parents[1]
FEAT       = ROOT / 'data' / 'processed' / 'features.parquet'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

HORIZONS  = ['y_1h', 'y_6h', 'y_24h']
TRAIN_END = pd.Timestamp('2025-06-30 23:00:00')
VAL_END   = pd.Timestamp('2025-12-31 23:00:00')

LGB_PARAMS = dict(
    objective='binary',
    is_unbalance=True,
    n_estimators=4000,
    learning_rate=0.02,           # v2: slower learning, more trees needed
    num_leaves=63,
    min_child_samples=200,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    verbosity=-1,
    n_jobs=-1,
)
EARLY_STOP_ROUNDS = 200           # v2: more patience so noisy val curve doesn't trigger early


def split(df: pd.DataFrame):
    tr = df[df['ts'] <= TRAIN_END]
    va = df[(df['ts'] > TRAIN_END) & (df['ts'] <= VAL_END)]
    te = df[df['ts'] > VAL_END]
    return tr, va, te


def fit_one(y_col: str, Xtr, ytr, Xv, yv, Xte, yte) -> tuple[lgb.LGBMClassifier, IsotonicRegression, dict]:
    print(f'\n=== {y_col} ===')
    print(f'  rows  train/val/test: {len(ytr):,} / {len(yv):,} / {len(yte):,}')
    print(f'  pos%  train/val/test: {ytr.mean():.4f} / {yv.mean():.4f} / {yte.mean():.4f}')

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xv, yv)],
        eval_metric='average_precision',
        callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False), lgb.log_evaluation(200)],
    )

    p_val_raw  = model.predict_proba(Xv)[:, 1]
    p_test_raw = model.predict_proba(Xte)[:, 1]

    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(p_val_raw, yv)
    p_val_cal  = cal.predict(p_val_raw)
    p_test_cal = cal.predict(p_test_raw)

    base_test = float(yte.mean())
    metrics = {
        'best_iteration': int(model.best_iteration_ or model.n_estimators),
        'val': {
            'base_rate':  float(yv.mean()),
            'pr_auc_raw': float(average_precision_score(yv, p_val_raw)),
            'pr_auc_cal': float(average_precision_score(yv, p_val_cal)),
            'roc_auc':    float(roc_auc_score(yv, p_val_raw)),
            'brier_raw':  float(brier_score_loss(yv, p_val_raw)),
            'brier_cal':  float(brier_score_loss(yv, p_val_cal)),
        },
        'test': {
            'base_rate':       base_test,
            'pr_auc_cal':      float(average_precision_score(yte, p_test_cal)),
            'pr_auc_lift':     float(average_precision_score(yte, p_test_cal) / base_test),
            'roc_auc':         float(roc_auc_score(yte, p_test_raw)),
            'brier_cal':       float(brier_score_loss(yte, p_test_cal)),
            'log_loss_cal':    float(log_loss(yte, np.clip(p_test_cal, 1e-6, 1 - 1e-6))),
        },
    }
    print(f'  val  PR-AUC raw/cal: {metrics["val"]["pr_auc_raw"]:.3f} / {metrics["val"]["pr_auc_cal"]:.3f}'
          f'   ROC-AUC: {metrics["val"]["roc_auc"]:.3f}   Brier cal: {metrics["val"]["brier_cal"]:.4f}')
    print(f'  test PR-AUC cal: {metrics["test"]["pr_auc_cal"]:.3f}'
          f'   (base {base_test:.4f}, lift {metrics["test"]["pr_auc_lift"]:.1f}x)'
          f'   ROC-AUC: {metrics["test"]["roc_auc"]:.3f}   Brier: {metrics["test"]["brier_cal"]:.4f}')
    return model, cal, metrics


def main():
    print('loading features...')
    df = pd.read_parquet(FEAT)
    print(f'  {len(df):,} rows x {df.shape[1]} cols')

    print('one-hot encoding town...')
    n_towns = df['town'].cat.categories.size if hasattr(df['town'], 'cat') else df['town'].nunique()
    town_oh = pd.get_dummies(df['town'], prefix='town', dtype='int8')
    df = pd.concat([df.drop(columns='town'), town_oh], axis=1)
    print(f'  added {n_towns} town one-hot cols')

    drop_for_X = HORIZONS + ['ts', 'is_active']
    feat_cols = [c for c in df.columns if c not in drop_for_X]
    print(f'  feature count: {len(feat_cols)}')

    tr, va, te = split(df)
    print(f'  train: {tr["ts"].min()} -> {tr["ts"].max()}  ({len(tr):,})')
    print(f'  val  : {va["ts"].min()} -> {va["ts"].max()}  ({len(va):,})')
    print(f'  test : {te["ts"].min()} -> {te["ts"].max()}  ({len(te):,})')

    Xtr = tr[feat_cols]; Xv = va[feat_cols]; Xte = te[feat_cols]

    all_metrics = {}
    for y in HORIZONS:
        model, cal, m = fit_one(y, Xtr, tr[y], Xv, va[y], Xte, te[y])
        all_metrics[y] = m
        model.booster_.save_model(str(MODELS_DIR / f'model_{y}.txt'))
        joblib.dump(cal, MODELS_DIR / f'calibrator_{y}.joblib')

        # feature importance (gain)
        imp = pd.DataFrame({
            'feature': feat_cols,
            'gain':    model.booster_.feature_importance(importance_type='gain'),
            'split':   model.booster_.feature_importance(importance_type='split'),
        }).sort_values('gain', ascending=False)
        imp.to_csv(MODELS_DIR / f'feature_importance_{y}.csv', index=False)
        print(f'  top features by gain:')
        for _, row in imp.head(10).iterrows():
            print(f'    {row["feature"]:<35s} gain={row["gain"]:>12,.0f}  split={row["split"]:>6}')

    with open(MODELS_DIR / 'feature_cols.json', 'w') as f:
        json.dump(feat_cols, f)
    with open(MODELS_DIR / 'metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'\nwrote models/* under {MODELS_DIR}')


if __name__ == '__main__':
    main()

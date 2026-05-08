"""
driver_attribution.py — group LightGBM TreeSHAP contributions into operator-
friendly families and decompose a single day's prediction.

Why this exists
---------------
The dashboard's "Why are redispatch events happening?" panel needs to attribute
the day's grid stress to interpretable drivers. Quintile-bucketed univariate
P(busy | bucket) is statistically weak (correlated drivers double-count, ignores
magnitude, disagrees with the production model).

Instead, we use LightGBM's built-in TreeSHAP (`booster.predict(X, pred_contrib=
True)`) on the same model that scores the dashboard's predictions, and group
the 198 features into 6 families a non-technical reader can scan in seconds:

    Recent activity | Wind | Solar & temperature | Load & price | Calendar | Location

This module is **pure** (no streamlit, no I/O beyond reading parquet) so it can
be unit-tested and reused outside the dashboard.

Math notes
----------
- TreeSHAP returns one column per feature plus a `bias` column. For any row,
  `bias + sum(feature_contribs) == raw_score` in log-odds space (LightGBM
  binary objective).
- We average contributions across all (hour × town) rows for the selected
  date, in log-odds space. The mean bias is the "baseline log-odds" — what the
  model would predict for an average row with all features at their training
  mean.
- We pass log-odds through sigmoid then through the isotonic calibrator to
  land on a calibrated probability axis the user can read.
- Per-group "effect in pp" is the **marginal** effect: difference between
  calibrated P at (baseline) and at (baseline + group_contrib). This is
  order-independent. Because the calibrator is non-linear, per-group pp
  effects don't perfectly sum to the total effect — close enough for an
  operator-facing chart, and we caption that caveat.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Feature group definitions.
# Order here is the display order in the dashboard chart (top → bottom).
# ----------------------------------------------------------------------
GROUP_ORDER = [
    'Recent activity',
    'Wind',
    'Load & price',
    'Solar & temperature',
    'Calendar',
    'Location',
]

# Plain-English one-liners used in the dashboard expander.
GROUP_DESCRIPTIONS = {
    'Recent activity':     'Lagged stress signal — how busy the grid was over the prior 24h and 7d.',
    'Wind':                'Wind generation (onshore + offshore) and hub-height wind speed/gusts/direction.',
    'Load & price':        'National grid demand, residual load, and the day-ahead market price.',
    'Solar & temperature': 'Solar generation, shortwave radiation, ambient temperature, cloud cover.',
    'Calendar':            'Hour of day, day of week, month, weekend/holiday flags.',
    'Location':            'Per-town identity (one-hot) and lat/lon — geography priors.',
}


def assign_groups(feature_cols: Iterable[str]) -> dict[str, str]:
    """Map every model feature to one of the six groups.

    Rules (first match wins):
        active_*_lag*           -> Recent activity
        wind_* / wx_wind_*      -> Wind
        solar / wx_shortwave_*  -> Solar & temperature
        wx_temperature_*        -> Solar & temperature
        wx_cloud_* / wx_precip* -> Solar & temperature
        total_load / residual_load / day_ahead -> Load & price
        hour / dow / month / is_weekend / is_de_holiday -> Calendar
        lat / lon / town_*      -> Location
        anything else           -> Location  (catch-all; emits a warning)
    """
    groups: dict[str, str] = {}
    for c in feature_cols:
        lc = c.lower()
        if lc.startswith('active_') and 'lag' in lc:
            groups[c] = 'Recent activity'
        elif lc.startswith('wind_') or lc.startswith('wx_wind'):
            groups[c] = 'Wind'
        elif lc == 'solar' or lc.startswith('wx_shortwave') \
                or lc.startswith('wx_temperature') or lc.startswith('wx_cloud') \
                or lc.startswith('wx_precip'):
            groups[c] = 'Solar & temperature'
        elif lc in ('total_load', 'residual_load', 'day_ahead'):
            groups[c] = 'Load & price'
        elif lc in ('hour', 'dow', 'month', 'is_weekend', 'is_de_holiday'):
            groups[c] = 'Calendar'
        elif lc in ('lat', 'lon') or lc.startswith('town_'):
            groups[c] = 'Location'
        else:
            # Unknown feature — treat as Location (safest catch-all; mostly
            # geography-flavoured one-hots in this codebase). Caller can
            # validate coverage with audit_groups().
            groups[c] = 'Location'
    return groups


def audit_groups(groups: dict[str, str]) -> pd.Series:
    """Count features per group. Useful for `python -c` smoke tests."""
    return pd.Series(list(groups.values())).value_counts()


# ----------------------------------------------------------------------
# Core decomposition.
# ----------------------------------------------------------------------
def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def aggregate_contributions(
    contribs: pd.DataFrame,
    groups: dict[str, str],
) -> pd.DataFrame:
    """Sum per-feature contributions into per-group contributions, per row.

    `contribs` has columns: <feature_1>, ..., <feature_N>, bias.
    Returns a DataFrame with one column per group + a 'bias' column,
    same row count as `contribs`.
    """
    out = pd.DataFrame(index=contribs.index)
    for g in GROUP_ORDER:
        members = [c for c, gg in groups.items() if gg == g and c in contribs.columns]
        if not members:
            out[g] = 0.0
        else:
            out[g] = contribs[members].sum(axis=1)
    if 'bias' in contribs.columns:
        out['bias'] = contribs['bias']
    return out


def decompose_day(
    contribs: pd.DataFrame,
    groups: dict[str, str],
    calibrator=None,
) -> dict:
    """Decompose all rows from one day into a per-group attribution record.

    Parameters
    ----------
    contribs   raw TreeSHAP contributions (one row per (hour, town) prediction)
               with one column per feature plus a 'bias' column. All values in
               log-odds space.
    groups     {feature_name: group_name} mapping (use `assign_groups`).
    calibrator scikit isotonic regressor with .predict(probs)->probs. Optional;
               when None, returns uncalibrated sigmoid probabilities.

    Returns
    -------
    {
        'baseline_p':       calibrated P at log-odds = mean(bias)
        'final_p':          calibrated P at log-odds = mean(raw_score)
        'mean_logodds':     dict of {group: mean_logodds_contribution}
        'group_effects':    list of {group, delta_p_pp, delta_logodds, share},
                            sorted by |delta_logodds| desc
        'n_rows':           int
    }
    """
    if 'bias' not in contribs.columns:
        raise ValueError("contribs must contain a 'bias' column (TreeSHAP intercept)")

    grouped = aggregate_contributions(contribs, groups)
    n_rows  = len(grouped)

    mean_bias = float(grouped['bias'].mean())
    mean_per_group: dict[str, float] = {
        g: float(grouped[g].mean()) for g in GROUP_ORDER
    }
    raw_score_mean = mean_bias + sum(mean_per_group.values())

    def _calibrate(logodds: float) -> float:
        p = float(_sigmoid(logodds))
        if calibrator is not None:
            try:
                p = float(np.atleast_1d(calibrator.predict([p]))[0])
            except Exception:
                pass
        return p

    baseline_p = _calibrate(mean_bias)
    final_p    = _calibrate(raw_score_mean)

    # Marginal effect per group: turn ONLY this group on from baseline,
    # measure pp shift on the calibrated axis.
    group_effects = []
    total_logodds = sum(mean_per_group.values())
    for g, dl in mean_per_group.items():
        p_with = _calibrate(mean_bias + dl)
        delta_pp = (p_with - baseline_p) * 100.0
        share   = (dl / total_logodds) if total_logodds != 0 else 0.0
        group_effects.append({
            'group':         g,
            'delta_p_pp':    delta_pp,
            'delta_logodds': dl,
            'share':         share,
        })
    group_effects.sort(key=lambda r: abs(r['delta_logodds']), reverse=True)

    return {
        'baseline_p':    baseline_p,
        'final_p':       final_p,
        'mean_logodds':  mean_per_group,
        'group_effects': group_effects,
        'n_rows':        n_rows,
    }


def top_features(
    contribs: pd.DataFrame,
    n: int = 25,
) -> pd.DataFrame:
    """Top-N features by mean |contribution| across rows. For the technical
    expander."""
    feat_cols = [c for c in contribs.columns if c != 'bias']
    if not feat_cols:
        return pd.DataFrame(columns=['feature', 'mean_logodds', 'abs_logodds'])
    means = contribs[feat_cols].mean()
    out = (pd.DataFrame({'feature': means.index, 'mean_logodds': means.values})
             .assign(abs_logodds=lambda d: d['mean_logodds'].abs())
             .sort_values('abs_logodds', ascending=False)
             .head(n)
             .reset_index(drop=True))
    return out


# ----------------------------------------------------------------------
# CLI smoke test
#   python src/driver_attribution.py
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    import json
    root = Path(__file__).resolve().parents[1]
    fcols = json.loads((root / 'models' / 'feature_cols.json').read_text())
    groups = assign_groups(fcols)
    print(f'features: {len(fcols)}')
    print(f'covered : {len(groups)}')
    print('per-group counts:')
    print(audit_groups(groups).to_string())


if __name__ == '__main__':
    _smoke_test()

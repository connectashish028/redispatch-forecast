"""
Build the 15-min redispatch time series, wide pivot (one column per town).

Reads:  data/raw/shn_operations_last_2y/*.parquet
Writes:
  data/processed/ts_15min_wide.parquet   one row per 15-min slot,
                                         one column per town = active op count
                                         (filtered to dominant reasons:
                                          'Netzengpass I' + 'Netzengpass')
  data/processed/ts_15min_long.parquet   long form (ts, town, active_ops, ...)
                                         kept for downstream feature work

Pipeline:
  1. Load raw asset-level rows, window to [2024-01-01, 2026-04-01).
  2. Drop dead columns, 'UW' placeholder location.
  3. Dedup to op-level on (start, end, location).
  4. Filter duration (0, 1 week].
  5. Explode comma-separated multi-town locations.
  6. Expand each op over its 15-min slots.
  7. Filter to dominant reasons (Netzengpass I, Netzengpass).
  8. Pivot to ts x town with active-op counts.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT          = Path(__file__).resolve().parents[1]
OPS_DIR       = ROOT / 'data' / 'raw' / 'shn_operations_last_2y'
OUT_DIR       = ROOT / 'data' / 'processed'
OUT_WIDE      = OUT_DIR / 'ts_15min_wide.parquet'
OUT_LONG      = OUT_DIR / 'ts_15min_long.parquet'

START_CUTOFF  = pd.Timestamp('2024-01-01')
# END_CUTOFF is auto-detected in load_and_clean(): one day past the latest op
# we have on disk, so build_timeseries always covers everything fetched so far
# without manual edits. Override with a hard date here only if you need a fixed
# evaluation window.
END_CUTOFF    = None
WEEK_MIN      = 7 * 24 * 60
SLOT          = pd.Timedelta(minutes=15)
TEST_REASON   = 'Funktionsnachweis'
KEEP_REASONS  = ('Netzengpass I', 'Netzengpass')   # dominant reasons only


def load_and_clean() -> pd.DataFrame:
    files = sorted(OPS_DIR.glob('chunk_*.parquet'))
    if not files:
        raise FileNotFoundError(f'no parquet chunks under data/raw/shn_operations_last_2y/')
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # ISO8601 covers both 'YYYY-MM-DD HH:MM:SS' and 'YYYY-MM-DDTHH:MM:SS' so
    # historical chunks (space) and freshly-fetched chunks (T) coexist cleanly.
    raw['start'] = pd.to_datetime(raw['start'], format='ISO8601')
    raw['end']   = pd.to_datetime(raw['end'],   format='ISO8601')

    # Auto-detect upper bound: one day past the latest op we have, floored to the
    # day boundary. Falls back to the module-level constant if the user pinned one.
    upper = END_CUTOFF if END_CUTOFF is not None \
        else (raw['start'].max().normalize() + pd.Timedelta(days=1))
    print(f'  window: {START_CUTOFF} -> {upper}')
    df = raw[(raw['start'] >= START_CUTOFF) & (raw['start'] < upper)].copy()
    df = df.drop(columns=['srId', 'Reason'], errors='ignore')
    df = df[df['location'] != 'UW']

    df_ops = df.drop_duplicates(subset=['start', 'end', 'location']).reset_index(drop=True)
    df_ops['duration_min'] = (df_ops['end'] - df_ops['start']).dt.total_seconds() / 60
    df_ops = df_ops[(df_ops['duration_min'] > 0) & (df_ops['duration_min'] <= WEEK_MIN)].reset_index(drop=True)

    df_ops['town_raw']  = df_ops['location'].str.replace(r'^UW\s+', '', regex=True).str.strip()
    df_ops['town_list'] = df_ops['town_raw'].str.split(',').apply(
        lambda xs: [t.replace('UW ', '').strip() for t in xs]
    )
    df_ops['is_test'] = df_ops['reason'].eq(TEST_REASON)

    df_town = (df_ops
               .explode('town_list')
               .rename(columns={'town_list': 'town'})
               .reset_index(drop=True))
    df_town = df_town[df_town['town'].str.len() > 0].reset_index(drop=True)

    cols = ['start', 'end', 'duration_min', 'town',
            'reason', 'liabilityOfCompensation', 'controlStage', 'is_test']
    return df_town[cols]


def expand_to_slots(ops: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Expand each op to one row per overlapping 15-min slot with per-slot overlap_min."""
    slot_ns  = np.int64(SLOT.value)
    start_ns = ops['start'].values.astype('datetime64[ns]').astype('int64')
    end_ns   = ops['end'].values.astype('datetime64[ns]').astype('int64')
    first_slot_ns = (start_ns // slot_ns) * slot_ns
    last_slot_ns  = ((end_ns - 1) // slot_ns) * slot_ns
    n_slots       = ((last_slot_ns - first_slot_ns) // slot_ns + 1).astype('int64')

    idx   = np.repeat(np.arange(len(ops), dtype=np.int64), n_slots)
    offs  = np.arange(n_slots.sum(), dtype=np.int64) - np.repeat(n_slots.cumsum() - n_slots, n_slots)
    slot_starts = np.repeat(first_slot_ns, n_slots) + offs * slot_ns

    expanded = ops.iloc[idx].reset_index(drop=True)
    expanded['ts']      = pd.to_datetime(slot_starts, unit='ns')
    expanded['_op_idx'] = idx

    slot_end_ns = slot_starts + slot_ns
    seg_start   = np.maximum(slot_starts, np.repeat(start_ns, n_slots))
    seg_end     = np.minimum(slot_end_ns, np.repeat(end_ns,   n_slots))
    expanded['overlap_min'] = (seg_end - seg_start) / 1e9 / 60.0
    return expanded, n_slots


def aggregate_long(expanded: pd.DataFrame) -> pd.DataFrame:
    """Long form: one row per (ts, town). Vectorised — no lambdas inside groupby."""
    exp = expanded.assign(
        _n_netzengpass_i = expanded['reason'].eq('Netzengpass I').astype('int32'),
        _n_netzengpass   = expanded['reason'].eq('Netzengpass').astype('int32'),
        _n_eeg           = expanded['liabilityOfCompensation'].eq('EEG').astype('int32'),
        _n_enwg          = expanded['liabilityOfCompensation'].eq('EnWG').astype('int32'),
    )
    out = exp.groupby(['ts', 'town'], sort=True, observed=True).agg(
        overlap_min     = ('overlap_min',      'sum'),
        active_ops      = ('overlap_min',      'size'),
        n_netzengpass_i = ('_n_netzengpass_i', 'sum'),
        n_netzengpass   = ('_n_netzengpass',   'sum'),
        n_eeg           = ('_n_eeg',           'sum'),
        n_enwg          = ('_n_enwg',          'sum'),
    ).reset_index()
    out['overlap_min'] = out['overlap_min'].clip(upper=15.0).astype('float32')
    out['is_active']   = out['overlap_min'] > 0
    for c in ('active_ops', 'n_netzengpass_i', 'n_netzengpass', 'n_eeg', 'n_enwg'):
        out[c] = out[c].astype('int16')
    return out


def pivot_wide(expanded: pd.DataFrame) -> pd.DataFrame:
    """Wide pivot: rows=ts (full 15-min grid), cols=town, values=active op count.

    Only ops with reason in KEEP_REASONS are counted, so each cell is
    'how many concurrent Netzengpass(/I) ops are active in town X at time t'.
    """
    keep = expanded[expanded['reason'].isin(KEEP_REASONS)]
    counts = (keep.groupby(['ts', 'town'], sort=True, observed=True)
                   .size()
                   .rename('active_ops')
                   .reset_index())

    wide = counts.pivot(index='ts', columns='town', values='active_ops')

    # full 15-min grid so gaps are explicit zeros, not missing rows.
    # Upper bound is the latest slot present in the data — auto-extends as we
    # fetch new chunks, no manual edit needed.
    upper = expanded['ts'].max()
    full_idx = pd.date_range(START_CUTOFF, upper, freq=SLOT)
    wide = wide.reindex(full_idx).fillna(0).astype('int16')
    wide.index.name = 'ts'
    wide.columns.name = None
    return wide


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('loading + cleaning ops...')
    ops = load_and_clean()
    print(f'  town-exploded ops: {len(ops):,}')
    print(f'  unique towns     : {ops["town"].nunique()}')
    print(f'  window           : {ops["start"].min()}  ->  {ops["end"].max()}')

    print('expanding to 15-min slots...')
    expanded, _ = expand_to_slots(ops)
    print(f'  op-slot rows: {len(expanded):,}')

    slot_sum  = expanded.groupby('_op_idx')['overlap_min'].sum()
    recon_err = slot_sum.values - ops['duration_min'].values
    print(f'  reconstruction err: max={np.abs(recon_err).max():.4f} min, '
          f'mean={np.abs(recon_err).mean():.5f} min')

    expanded = expanded.drop(columns='_op_idx')

    print('aggregating long form (ts, town)...')
    ts_long = aggregate_long(expanded)
    print(f'  long rows: {len(ts_long):,}')
    ts_long.to_parquet(OUT_LONG, index=False)
    print(f'  wrote data/processed/ts_15min_long.parquet')

    print('pivoting to wide (ts x town, dominant reasons only)...')
    print(f'  reasons kept: {KEEP_REASONS}')
    wide = pivot_wide(expanded)
    print(f'  wide shape: {wide.shape[0]:,} slots x {wide.shape[1]} towns')
    print(f'  ts range  : {wide.index.min()}  ->  {wide.index.max()}')
    print(f'  total active town-slots (>0): {int((wide > 0).values.sum()):,}')
    wide.to_parquet(OUT_WIDE)
    print(f'  wrote data/processed/ts_15min_wide.parquet')

    print('\n=== top 10 towns by # of active slots (wide, dominant reasons) ===')
    print((wide > 0).sum().sort_values(ascending=False).head(10))

    print('\n=== sample: 5 busiest timestamps (sum of concurrent ops across towns) ===')
    busiest = wide.sum(axis=1).sort_values(ascending=False).head(5)
    print(busiest)


if __name__ == '__main__':
    main()

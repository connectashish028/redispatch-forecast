"""
build_daily_summary.py — roll the per-day TreeSHAP contribution parquets up
into a single `contributions_daily_summary.parquet` (one row per date).

Why
---
Per-day contributions parquets are 600+ KB each (4,200 rows × 200 cols of
log-odds floats). Pushing 800+ of them to git would balloon the repo by
~500 MB. The dashboard's "Why?" bar chart only needs the **mean per-group
contribution per day**, which compresses to ~50 KB total.

What this writes
----------------
data/predictions/contributions_daily_summary.parquet
    columns: date, n_rows, mean_bias, <one column per driver group>
    one row per date

Modes
-----
--rebuild    re-scan every contributions_*.parquet and rewrite the summary
             (idempotent; safe to run any time)
--append-date YYYY-MM-DD
             read a single contributions_<date>.parquet, compute its row,
             upsert into the existing summary. Used by the daily cron.

Usage
-----
    python src/build_daily_summary.py --rebuild
    python src/build_daily_summary.py --append-date 2026-05-08
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).resolve().parents[1]
PRED_DIR  = ROOT / 'data' / 'predictions'
MODELS    = ROOT / 'models'
SUMMARY   = PRED_DIR / 'contributions_daily_summary.parquet'

sys.path.insert(0, str(ROOT / 'src'))
from driver_attribution import assign_groups, GROUP_ORDER  # noqa: E402


def _load_groups() -> dict[str, str]:
    fcols = json.loads((MODELS / 'feature_cols.json').read_text())
    return assign_groups(fcols)


def _summarize_one(path: Path, groups: dict[str, str]) -> dict:
    """Read a contributions_<date>.parquet and return one summary row."""
    df = pd.read_parquet(path)
    if 'bias' not in df.columns:
        raise ValueError(f"{path.name} missing 'bias' column")
    date = path.stem.replace('contributions_', '')
    rec = {
        'date':      date,
        'n_rows':    len(df),
        'mean_bias': float(df['bias'].mean()),
    }
    for grp in GROUP_ORDER:
        members = [c for c, g in groups.items() if g == grp and c in df.columns]
        rec[grp] = float(df[members].sum(axis=1).mean()) if members else 0.0
    return rec


def rebuild() -> pd.DataFrame:
    groups = _load_groups()
    files = sorted(PRED_DIR.glob('contributions_2*.parquet'))
    files = [f for f in files if 'daily_summary' not in f.name and 'latest' not in f.name]
    if not files:
        print('[summary] no contributions_*.parquet files found; nothing to do.')
        return pd.DataFrame()
    print(f'[summary] reading {len(files)} files...')
    rows = [_summarize_one(f, groups) for f in files]
    out = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SUMMARY, compression='zstd', compression_level=9, index=False)
    print(f'[summary] wrote {SUMMARY.relative_to(ROOT)} '
          f'({SUMMARY.stat().st_size:,} bytes, {len(out)} rows)')
    return out


def append_date(date_str: str) -> pd.DataFrame:
    groups = _load_groups()
    path = PRED_DIR / f'contributions_{date_str}.parquet'
    if not path.exists():
        print(f'[summary] missing {path.name}; nothing to append.', file=sys.stderr)
        sys.exit(1)
    new_row = _summarize_one(path, groups)

    if SUMMARY.exists():
        df = pd.read_parquet(SUMMARY)
        df = df[df['date'] != date_str]                     # upsert
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df = df.sort_values('date').reset_index(drop=True)
    df.to_parquet(SUMMARY, compression='zstd', compression_level=9, index=False)
    print(f'[summary] upserted {date_str} into {SUMMARY.relative_to(ROOT)} '
          f'({SUMMARY.stat().st_size:,} bytes, {len(df)} rows)')
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--rebuild', action='store_true',
                   help='re-scan all contributions_*.parquet and rewrite summary')
    g.add_argument('--append-date', type=str,
                   help='upsert a single date into the existing summary')
    args = ap.parse_args()

    if args.rebuild:
        rebuild()
    else:
        append_date(args.append_date)


if __name__ == '__main__':
    main()

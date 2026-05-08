"""
trim_to_last_6_months.py — one-shot cleanup that trims the repo's
historical SHN ops data to the last 180 days. Keeps the daily refresh
runtime predictable on the 7 GB hosted runner.

What it does
------------
1. Reads every chunk under data/raw/shn_operations_last_2y/.
2. Filters rows to the last `--days` (default 180) days.
3. Writes a single consolidated `chunk_bootstrap.parquet` for the
   pre-cron window, leaving daily date-named chunks alone.
4. Deletes the original numbered chunks (chunk_NNNNN.parquet).
5. Trims data/predictions/contributions_daily_summary.parquet to the
   same window.

Re-runnable: chunk_bootstrap.parquet is overwritten, daily summary is
filtered in place. Run again any time the cutoff slides forward (e.g.
once a quarter to drop the oldest month).

Usage
-----
    python tools/trim_to_last_6_months.py            # 180 days
    python tools/trim_to_last_6_months.py --days 90  # last quarter
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
RAW_DIR  = ROOT / 'data' / 'raw' / 'shn_operations_last_2y'
SUMMARY  = ROOT / 'data' / 'predictions' / 'contributions_daily_summary.parquet'


def trim_chunks(cutoff: pd.Timestamp) -> None:
    numbered = sorted(RAW_DIR.glob('chunk_0*.parquet'))
    if not numbered:
        print(f'[trim] no numbered chunks under {RAW_DIR}; skipping')
        return

    print(f'[trim] reading {len(numbered)} numbered chunks...')
    frames = []
    for f in numbered:
        df = pd.read_parquet(f)
        df['start'] = pd.to_datetime(df['start'], format='ISO8601', errors='coerce')
        frames.append(df[df['start'] >= cutoff])
    bootstrap = pd.concat(frames, ignore_index=True).drop_duplicates()
    bootstrap['start'] = bootstrap['start'].astype(str)        # back to string for parquet
    print(f'[trim] kept {len(bootstrap):,} rows >= {cutoff.date()} '
          f'(out of {sum(len(f) for f in [pd.read_parquet(p) for p in numbered]):,})')

    out = RAW_DIR / 'chunk_bootstrap.parquet'
    bootstrap.to_parquet(out, index=False)
    print(f'[trim] wrote {out.relative_to(ROOT)} ({out.stat().st_size:,} bytes)')

    print(f'[trim] deleting {len(numbered)} original numbered chunks...')
    for f in numbered:
        f.unlink()
    print('[trim] done.')


def trim_summary(cutoff: pd.Timestamp) -> None:
    if not SUMMARY.exists():
        print(f'[trim] summary not present; skipping')
        return
    df = pd.read_parquet(SUMMARY)
    before = len(df)
    df = df[pd.to_datetime(df['date']) >= cutoff].reset_index(drop=True)
    df.to_parquet(SUMMARY, compression='zstd', compression_level=9, index=False)
    print(f'[trim] daily summary: {before} -> {len(df)} rows '
          f'({SUMMARY.stat().st_size:,} bytes)')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', type=int, default=6,
                    help='Cutoff = first of this month - N months. '
                         'Aligns with build_timeseries WINDOW_MONTHS.')
    ap.add_argument('--days', type=int, default=None,
                    help='Override the months-rounded cutoff with an exact '
                         'today - DAYS rolling cutoff.')
    args = ap.parse_args()

    if args.days is not None:
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=args.days)
        print(f'[trim] cutoff = {cutoff.date()} (today - {args.days} days)')
    else:
        cutoff = (pd.Timestamp.today().normalize().replace(day=1)
                  - pd.DateOffset(months=args.months))
        print(f'[trim] cutoff = {cutoff.date()} '
              f'(first of month - {args.months} months)')
    trim_chunks(cutoff)
    trim_summary(cutoff)


if __name__ == '__main__':
    main()

"""
fetch_shn_ops.py - incremental fetch of SHN redispatch operations.

Source:
    https://redispatch-run.azurewebsites.net/api/operations/get
    Public Connect+ Redispatch 2.0 API. JSON, 500 rows per chunk, ordered newest-first.

What it does:
    1. Scans data/raw/shn_operations_last_2y/*.parquet for the most recent op start.
    2. Fetches API chunks 1, 2, 3, ... in order (newest first).
    3. Stops when a chunk's oldest op is at or before the most recent local op.
    4. Deduplicates against existing op `id`s.
    5. Writes new ops to data/raw/shn_operations_last_2y/chunk_<YYYY-MM-DD>.parquet
       (date-named, so it coexists with the historical chunk_NNNNN.parquet files).

Idempotent:
    Re-running on the same day overwrites the day's chunk file with the same
    content. Multiple runs per day are safe.

Exit codes:
    0  success (new data fetched, OR nothing new to fetch)
    1  data / network error
    2  config error (e.g. missing data dir)

Usage:
    python src/fetch_shn_ops.py                     # incremental from latest local
    python src/fetch_shn_ops.py --max-chunks 10     # safety cap on chunks (default 50)
    python src/fetch_shn_ops.py --since 2026-04-20  # force start date instead of auto-detect
    python src/fetch_shn_ops.py --dry-run           # fetch + report, do not write
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT      = Path(__file__).resolve().parents[1]
RAW_DIR   = ROOT / 'data' / 'raw' / 'shn_operations_last_2y'

API_URL    = 'https://redispatch-run.azurewebsites.net/api/operations/get'
API_PARAMS = {
    'networkoperator': 'shn',
    'type':            'finished',
    'orderDirection':  'desc',
    'orderBy':         'start',
}
EXPECTED_COLS = [
    'id', 'operationId', 'srId', 'start', 'end', 'duration',
    'location', 'locationBottleneck', 'controlStage', 'reason', 'Reason',
    'assetKey', 'requestor', 'liabilityOfCompensation',
    'DSOKey', 'inducingNO', 'KalKID',
]
POLITE_SLEEP_S = 1.5     # Connect+ API is slow (~20s/chunk); be polite


def latest_local_start() -> tuple[pd.Timestamp, set[int]]:
    """Find max op `start` across existing chunks + collect all known ids.
    Returns (Timestamp.min, empty_set) if no chunks exist yet."""
    files = sorted(RAW_DIR.glob('chunk_*.parquet'))
    if not files:
        return pd.Timestamp.min, set()

    latest = pd.Timestamp.min
    known_ids: set[int] = set()
    for f in files:
        df = pd.read_parquet(f, columns=['id', 'start'])
        # Historical chunks store `start` as ISO strings; new chunks too (we align).
        df['start'] = pd.to_datetime(df['start'], format='ISO8601')
        latest = max(latest, df['start'].max())
        known_ids.update(df['id'].dropna().astype('int64').tolist())
    return latest, known_ids


def fetch_chunk(chunk_nr: int, timeout: int = 120) -> dict:
    """One paginated API call. Raises requests.HTTPError on 4xx/5xx."""
    r = requests.get(API_URL, params={**API_PARAMS, 'chunkNr': chunk_nr}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_until(latest_local: pd.Timestamp, max_chunks: int) -> pd.DataFrame:
    """Walk chunks 1..N, stop once we've reached operations older than latest_local."""
    rows: list[pd.DataFrame] = []
    for n in range(1, max_chunks + 1):
        print(f'[fetch] chunkNr={n} ...', end=' ', flush=True)
        try:
            j = fetch_chunk(n)
        except requests.HTTPError as e:
            print(f'HTTP error: {e}')
            sys.exit(1)
        except requests.RequestException as e:
            print(f'network error: {e}')
            sys.exit(1)

        ops = j.get('operations') or []
        if not ops:
            print('empty page, stopping.')
            break

        df = pd.DataFrame(ops)
        df['start'] = pd.to_datetime(df['start'], format='ISO8601')
        df['end']   = pd.to_datetime(df['end'],   format='ISO8601')
        rows.append(df)

        oldest = df['start'].min()
        newest = df['start'].max()
        print(f'{len(df)} ops | {oldest:%Y-%m-%d %H:%M} -> {newest:%Y-%m-%d %H:%M}')

        # API returned ops older than what we already have -> we've caught up.
        if oldest <= latest_local:
            print(f'[fetch] reached cutoff ({latest_local}); stopping.')
            break

        if n < max_chunks:
            time.sleep(POLITE_SLEEP_S)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--max-chunks', type=int, default=50,
                    help='Safety cap on API pagination depth (default 50 = 25k ops).')
    ap.add_argument('--since', type=str, default=None,
                    help='YYYY-MM-DD; override auto-detected cutoff.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Fetch and report, do not write the chunk file.')
    args = ap.parse_args()

    if not RAW_DIR.exists():
        print(f'[error] {RAW_DIR} does not exist', file=sys.stderr)
        sys.exit(2)

    if args.since:
        try:
            latest = pd.Timestamp(args.since)
            known_ids: set[int] = set()
            print(f'[scan] using --since {latest}, skipping local id-scan')
        except ValueError:
            print(f'[error] could not parse --since {args.since!r}', file=sys.stderr)
            sys.exit(2)
    else:
        print('[scan] reading local chunks for latest start + known ids ...')
        latest, known_ids = latest_local_start()
        print(f'[scan] latest local start: {latest}')
        print(f'[scan] known ids        : {len(known_ids):,}')

    print(f'[fetch] starting (max {args.max_chunks} chunks)')
    df = fetch_until(latest, args.max_chunks)
    if df.empty:
        print('[fetch] nothing returned')
        return

    # Schema sanity
    missing = set(EXPECTED_COLS) - set(df.columns)
    if missing:
        print(f'[error] API response missing expected columns: {missing}', file=sys.stderr)
        sys.exit(1)
    df = df[EXPECTED_COLS]

    # Filter to genuinely new ops
    if known_ids:
        new_df = df[~df['id'].astype('int64').isin(known_ids)].copy()
    else:
        new_df = df.copy()

    # Also strictly require start > latest, in case ids overlap a re-published op
    new_df = new_df[new_df['start'] > latest].reset_index(drop=True)

    print(f'[diff] fetched {len(df):,} rows; {len(new_df):,} are new')

    if new_df.empty:
        print('[done] no new operations to write')
        return

    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    out_path = RAW_DIR / f'chunk_{today}.parquet'

    # Capture date range BEFORE we coerce timestamps back to strings.
    range_lo = new_df['start'].min()
    range_hi = new_df['start'].max()

    if args.dry_run:
        print(f'[dry-run] would write {len(new_df):,} ops to {out_path}')
        print(f'[dry-run] start range: {range_lo} -> {range_hi}')
        return

    # Schema alignment with historical chunks (sampled from chunk_00001.parquet):
    #   - `start` and `end` are stored as 'YYYY-MM-DD HH:MM:SS' strings (space, not T)
    #   - text columns use pandas StringDtype, not object
    # Match exactly so build_timeseries.py sees uniform format across all chunks.
    new_df['start'] = new_df['start'].dt.strftime('%Y-%m-%d %H:%M:%S')
    new_df['end']   = new_df['end'].dt.strftime('%Y-%m-%d %H:%M:%S')
    text_cols = [c for c in EXPECTED_COLS
                 if c not in ('id', 'duration', 'controlStage')]
    for c in text_cols:
        new_df[c] = new_df[c].astype('string')

    new_df.to_parquet(out_path, index=False)
    print(f'[done] wrote {len(new_df):,} ops to {out_path}')
    print(f'[done] start range: {range_lo} -> {range_hi}')


if __name__ == '__main__':
    main()

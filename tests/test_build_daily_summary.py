"""
Tests for src/build_daily_summary.py — the script that rolls per-day
TreeSHAP parquets into the lightweight summary that ships in git.

These are integration tests: they need the rest of the project layout
(models/feature_cols.json) but no external data — we synthesise a tiny
contributions parquet and round-trip it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_pred_dir(tmp_path, root, feature_cols):
    """A throwaway predictions directory with one synthetic contributions
    parquet, plus a copy of feature_cols.json so build_daily_summary can
    find the schema. We monkey-patch ROOT to point at this scratch tree."""
    pred = tmp_path / 'data' / 'predictions'
    pred.mkdir(parents=True)
    models = tmp_path / 'models'
    models.mkdir()
    shutil.copy(root / 'models' / 'feature_cols.json',
                models / 'feature_cols.json')

    # Two days, six (hour×town) rows each.
    rng = np.random.default_rng(7)
    for date in ['2026-01-01', '2026-01-02']:
        n = 6
        data = {
            'ts':   pd.date_range(f'{date} 00:00', periods=n, freq='1h'),
            'town': ['A', 'B', 'C'] * 2,
        }
        for c in feature_cols:
            data[c] = rng.normal(0, 0.02, n).astype('float32')
        data['bias'] = np.full(n, -2.5, dtype='float32')
        pd.DataFrame(data).to_parquet(
            pred / f'contributions_{date}.parquet', index=False
        )

    return tmp_path


def _run_with_scratch_root(tmp_path, args):
    """Run build_daily_summary.main() with sys.argv + ROOT pointed at tmp_path."""
    import sys
    import importlib

    sys.argv = ['build_daily_summary.py'] + args

    # Reload the module so the ROOT module-level constants pick up our patched paths.
    if 'build_daily_summary' in sys.modules:
        del sys.modules['build_daily_summary']
    import build_daily_summary as bds                              # noqa: WPS433
    bds.ROOT     = tmp_path
    bds.PRED_DIR = tmp_path / 'data' / 'predictions'
    bds.MODELS   = tmp_path / 'models'
    bds.SUMMARY  = bds.PRED_DIR / 'contributions_daily_summary.parquet'
    importlib.reload  # noqa: B015 — keep linter quiet
    if 'rebuild' in args[0]:
        bds.rebuild()
    else:
        bds.append_date(args[1])
    return bds.SUMMARY


class TestRebuild:
    def test_writes_one_row_per_input_file(self, tmp_pred_dir):
        out_path = _run_with_scratch_root(tmp_pred_dir, ['--rebuild'])
        assert out_path.exists()
        df = pd.read_parquet(out_path)
        assert len(df) == 2
        assert set(df['date']) == {'2026-01-01', '2026-01-02'}

    def test_summary_is_small(self, tmp_pred_dir):
        out_path = _run_with_scratch_root(tmp_pred_dir, ['--rebuild'])
        assert out_path.stat().st_size < 50_000, \
            'daily summary should be lightweight (~few KB per row)'

    def test_summary_columns_include_all_groups(self, tmp_pred_dir):
        out_path = _run_with_scratch_root(tmp_pred_dir, ['--rebuild'])
        df = pd.read_parquet(out_path)
        from driver_attribution import GROUP_ORDER
        for grp in GROUP_ORDER:
            assert grp in df.columns, f'missing group column: {grp}'
        assert 'date'      in df.columns
        assert 'n_rows'    in df.columns
        assert 'mean_bias' in df.columns

    def test_n_rows_matches_input(self, tmp_pred_dir):
        out_path = _run_with_scratch_root(tmp_pred_dir, ['--rebuild'])
        df = pd.read_parquet(out_path)
        assert (df['n_rows'] == 6).all()


class TestAppendDate:
    def test_appends_new_date(self, tmp_pred_dir, feature_cols):
        # First, rebuild the existing 2 days
        out_path = _run_with_scratch_root(tmp_pred_dir, ['--rebuild'])
        # Add a third day's contributions parquet
        pred_dir = tmp_pred_dir / 'data' / 'predictions'
        rng = np.random.default_rng(99)
        n = 6
        data = {
            'ts':   pd.date_range('2026-01-03 00:00', periods=n, freq='1h'),
            'town': ['A', 'B', 'C'] * 2,
        }
        for c in feature_cols:
            data[c] = rng.normal(0, 0.02, n).astype('float32')
        data['bias'] = np.full(n, -2.5, dtype='float32')
        pd.DataFrame(data).to_parquet(
            pred_dir / 'contributions_2026-01-03.parquet', index=False
        )
        out_path = _run_with_scratch_root(
            tmp_pred_dir, ['--append-date', '2026-01-03']
        )
        df = pd.read_parquet(out_path)
        assert len(df) == 3
        assert '2026-01-03' in set(df['date'])
        # Existing dates preserved
        assert {'2026-01-01', '2026-01-02', '2026-01-03'} == set(df['date'])

    def test_upsert_replaces_existing_date(self, tmp_pred_dir, feature_cols):
        _run_with_scratch_root(tmp_pred_dir, ['--rebuild'])
        # Now overwrite 2026-01-01's contributions parquet with different values
        pred_dir = tmp_pred_dir / 'data' / 'predictions'
        rng = np.random.default_rng(123)
        n = 6
        data = {
            'ts':   pd.date_range('2026-01-01 00:00', periods=n, freq='1h'),
            'town': ['A', 'B', 'C'] * 2,
        }
        for c in feature_cols:
            data[c] = rng.normal(5, 0.02, n).astype('float32')   # very different
        data['bias'] = np.full(n, -2.5, dtype='float32')
        pd.DataFrame(data).to_parquet(
            pred_dir / 'contributions_2026-01-01.parquet', index=False
        )
        out_path = _run_with_scratch_root(
            tmp_pred_dir, ['--append-date', '2026-01-01']
        )
        df = pd.read_parquet(out_path)
        assert len(df) == 2, 'upsert should not duplicate the date'
        # Wind contribution should reflect the new (much larger) values
        assert df.loc[df['date'] == '2026-01-01', 'Wind'].abs().iloc[0] > 1.0

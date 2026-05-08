"""Shared pytest fixtures + path setup."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


@pytest.fixture(scope='session')
def root() -> Path:
    return ROOT


@pytest.fixture(scope='session')
def feature_cols() -> list[str]:
    """Production feature schema. Tests read this rather than hard-coding 198
    feature names — so adding a feature in train.py shouldn't break tests."""
    p = ROOT / 'models' / 'feature_cols.json'
    if not p.exists():
        pytest.skip('models/feature_cols.json not present')
    return json.loads(p.read_text())


@pytest.fixture
def synthetic_contribs(feature_cols) -> pd.DataFrame:
    """Build a tiny but structurally-valid contributions DataFrame for unit
    tests. Two timestamps × three towns × all features + bias. Values are
    crafted so per-group sums are easy to assert against."""
    rng = np.random.default_rng(42)
    n = 6
    data = {
        'ts':   pd.date_range('2026-01-01', periods=n, freq='1h'),
        'town': ['A', 'B', 'C'] * 2,
    }
    for c in feature_cols:
        data[c] = rng.normal(0, 0.01, n).astype('float32')
    data['bias'] = np.full(n, -2.5, dtype='float32')
    return pd.DataFrame(data)

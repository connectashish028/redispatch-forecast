"""
Unit tests for src/driver_attribution.py — the pure functions powering
the dashboard's "Why?" panel. These are the math invariants that, if
they break, will silently produce wrong attribution. Worth pinning.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from driver_attribution import (
    GROUP_DESCRIPTIONS,
    GROUP_ORDER,
    aggregate_contributions,
    assign_groups,
    audit_groups,
    decompose_day,
    top_features,
)


# ---------------------------------------------------------------------------
# Group assignment — every feature must land somewhere, no orphans.
# ---------------------------------------------------------------------------
class TestAssignGroups:
    def test_full_coverage(self, feature_cols):
        groups = assign_groups(feature_cols)
        assert len(groups) == len(feature_cols), 'every feature gets a group'
        assert set(groups.values()).issubset(set(GROUP_ORDER)), \
            'no group outside the defined six'

    def test_known_features_route_correctly(self):
        groups = assign_groups([
            'wind_onshore', 'wx_wind_speed_100m', 'wx_wind_gusts_10m',
            'solar', 'wx_temperature_2m', 'wx_cloud_cover', 'wx_precipitation',
            'total_load', 'residual_load', 'day_ahead',
            'active_24h_lag24', 'active_7d_lag24',
            'hour', 'dow', 'month', 'is_weekend', 'is_de_holiday',
            'lat', 'lon', 'town_Husum',
        ])
        assert groups['wind_onshore']        == 'Wind'
        assert groups['wx_wind_speed_100m']  == 'Wind'
        assert groups['wx_wind_gusts_10m']   == 'Wind'
        assert groups['solar']               == 'Solar & temperature'
        assert groups['wx_temperature_2m']   == 'Solar & temperature'
        assert groups['wx_cloud_cover']      == 'Solar & temperature'
        assert groups['wx_precipitation']    == 'Solar & temperature'
        assert groups['total_load']          == 'Load & price'
        assert groups['residual_load']       == 'Load & price'
        assert groups['day_ahead']           == 'Load & price'
        assert groups['active_24h_lag24']    == 'Recent activity'
        assert groups['active_7d_lag24']     == 'Recent activity'
        assert groups['hour']                == 'Calendar'
        assert groups['is_weekend']          == 'Calendar'
        assert groups['is_de_holiday']       == 'Calendar'
        assert groups['lat']                 == 'Location'
        assert groups['town_Husum']          == 'Location'

    def test_unknown_feature_falls_back_to_location(self):
        groups = assign_groups(['some_unknown_feature_xyz'])
        assert groups['some_unknown_feature_xyz'] == 'Location'

    def test_audit_returns_per_group_counts(self, feature_cols):
        counts = audit_groups(assign_groups(feature_cols))
        assert isinstance(counts, pd.Series)
        assert counts.sum() == len(feature_cols)


# ---------------------------------------------------------------------------
# Aggregation — per-group sum must equal sum of its member features.
# ---------------------------------------------------------------------------
class TestAggregateContributions:
    def test_group_sums_match_member_sums(self, synthetic_contribs, feature_cols):
        groups = assign_groups(feature_cols)
        out = aggregate_contributions(synthetic_contribs, groups)

        for grp in GROUP_ORDER:
            members = [c for c, g in groups.items() if g == grp]
            if not members:
                continue
            expected = synthetic_contribs[members].sum(axis=1).values
            np.testing.assert_allclose(
                out[grp].values, expected, rtol=1e-5,
                err_msg=f'group sum mismatch for {grp}',
            )

    def test_bias_passed_through(self, synthetic_contribs, feature_cols):
        groups = assign_groups(feature_cols)
        out = aggregate_contributions(synthetic_contribs, groups)
        np.testing.assert_array_equal(
            out['bias'].values, synthetic_contribs['bias'].values
        )

    def test_total_equals_sum_of_features(self, synthetic_contribs, feature_cols):
        """Most important: bias + sum of all groups == bias + sum of all features.
        This is the invariant TreeSHAP itself guarantees, and we must preserve it."""
        groups = assign_groups(feature_cols)
        out = aggregate_contributions(synthetic_contribs, groups)
        feat_only = [c for c in synthetic_contribs.columns
                     if c not in ('ts', 'town', 'bias')]
        per_row_total_via_groups = out[GROUP_ORDER].sum(axis=1).values
        per_row_total_via_features = synthetic_contribs[feat_only].sum(axis=1).values
        np.testing.assert_allclose(
            per_row_total_via_groups, per_row_total_via_features, rtol=1e-5
        )


# ---------------------------------------------------------------------------
# Daily decomposition — the call site the dashboard cares about.
# ---------------------------------------------------------------------------
class TestDecomposeDay:
    def test_required_keys(self, synthetic_contribs, feature_cols):
        groups = assign_groups(feature_cols)
        feat_and_bias = [c for c in synthetic_contribs.columns
                         if c not in ('ts', 'town')]
        result = decompose_day(synthetic_contribs[feat_and_bias], groups)
        for key in ('baseline_p', 'final_p', 'mean_logodds',
                    'group_effects', 'n_rows'):
            assert key in result, f'missing key: {key}'

    def test_baseline_is_sigmoid_of_mean_bias(self, synthetic_contribs, feature_cols):
        groups = assign_groups(feature_cols)
        feat_and_bias = [c for c in synthetic_contribs.columns
                         if c not in ('ts', 'town')]
        result = decompose_day(synthetic_contribs[feat_and_bias], groups)
        mean_bias = float(synthetic_contribs['bias'].mean())
        expected = 1.0 / (1.0 + math.exp(-mean_bias))
        assert abs(result['baseline_p'] - expected) < 1e-6

    def test_final_p_within_unit_interval(self, synthetic_contribs, feature_cols):
        groups = assign_groups(feature_cols)
        feat_and_bias = [c for c in synthetic_contribs.columns
                         if c not in ('ts', 'town')]
        result = decompose_day(synthetic_contribs[feat_and_bias], groups)
        assert 0.0 <= result['final_p'] <= 1.0
        assert 0.0 <= result['baseline_p'] <= 1.0

    def test_group_effects_sorted_by_abs_logodds(self, synthetic_contribs, feature_cols):
        groups = assign_groups(feature_cols)
        feat_and_bias = [c for c in synthetic_contribs.columns
                         if c not in ('ts', 'town')]
        result = decompose_day(synthetic_contribs[feat_and_bias], groups)
        abs_dls = [abs(e['delta_logodds']) for e in result['group_effects']]
        assert abs_dls == sorted(abs_dls, reverse=True)

    def test_group_effects_cover_all_groups(self, synthetic_contribs, feature_cols):
        groups = assign_groups(feature_cols)
        feat_and_bias = [c for c in synthetic_contribs.columns
                         if c not in ('ts', 'town')]
        result = decompose_day(synthetic_contribs[feat_and_bias], groups)
        seen = {e['group'] for e in result['group_effects']}
        assert seen == set(GROUP_ORDER)

    def test_n_rows_matches_input(self, synthetic_contribs, feature_cols):
        groups = assign_groups(feature_cols)
        feat_and_bias = [c for c in synthetic_contribs.columns
                         if c not in ('ts', 'town')]
        result = decompose_day(synthetic_contribs[feat_and_bias], groups)
        assert result['n_rows'] == len(synthetic_contribs)

    def test_missing_bias_raises(self, synthetic_contribs, feature_cols):
        groups = assign_groups(feature_cols)
        no_bias = synthetic_contribs.drop(columns=['ts', 'town', 'bias'])
        with pytest.raises(ValueError, match='bias'):
            decompose_day(no_bias, groups)


# ---------------------------------------------------------------------------
# Top-N features — the technical drill-down expander's data source.
# ---------------------------------------------------------------------------
class TestTopFeatures:
    def test_returns_n_rows(self, synthetic_contribs, feature_cols):
        feat_only = [c for c in synthetic_contribs.columns
                     if c not in ('ts', 'town', 'bias')]
        out = top_features(synthetic_contribs[['bias'] + feat_only], n=10)
        assert len(out) == 10
        assert list(out.columns) == ['feature', 'mean_logodds', 'abs_logodds']

    def test_sorted_by_abs_descending(self, synthetic_contribs, feature_cols):
        feat_only = [c for c in synthetic_contribs.columns
                     if c not in ('ts', 'town', 'bias')]
        out = top_features(synthetic_contribs[['bias'] + feat_only], n=20)
        abs_vals = out['abs_logodds'].values
        assert (abs_vals[:-1] >= abs_vals[1:]).all()

    def test_handles_empty_input(self):
        empty = pd.DataFrame({'bias': [0.0]})
        out = top_features(empty, n=5)
        assert out.empty

    def test_n_larger_than_input_returns_all(self, synthetic_contribs, feature_cols):
        feat_only = [c for c in synthetic_contribs.columns
                     if c not in ('ts', 'town', 'bias')]
        out = top_features(synthetic_contribs[['bias'] + feat_only], n=100_000)
        assert len(out) == len(feat_only)


# ---------------------------------------------------------------------------
# Group descriptions — make sure the dashboard's narrative copy stays in sync
# with the group definitions.
# ---------------------------------------------------------------------------
class TestGroupDescriptions:
    def test_description_for_every_group(self):
        for grp in GROUP_ORDER:
            assert grp in GROUP_DESCRIPTIONS, f'missing description for {grp}'
            assert GROUP_DESCRIPTIONS[grp].strip(), 'description is empty'

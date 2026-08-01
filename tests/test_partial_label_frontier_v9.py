"""Tests for the exploratory v0.9 partial-label evidence frontier."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.partial_label_frontier_v9 import (
    adaptive_bottleneck_cover_curve,
    adaptive_random_active_disagreement_curve,
    audit_curve_from_order,
    first_crossing,
    nonadaptive_initial_coverage_order,
    retrospective_oracle_minimum_queries,
    stable_hash_order,
    summarize_hash_thresholds,
)
from src.analysis.partial_identification import (
    realized_accuracy_advantage,
    sharp_partial_accuracy_envelope,
)


def _example():
    quantum = np.array([0, 1, 1, 0, 1, 0], dtype=np.int8)
    classical = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )
    labels = np.array([0, 0, 1, 1, 0, 0], dtype=np.int8)
    return quantum, classical, labels


def test_fixed_order_curve_matches_sharp_subset_formula_at_every_prefix():
    quantum, classical, labels = _example()
    order = np.array([4, 1, 5, 0, 3, 2])
    curve = audit_curve_from_order(quantum, classical, labels, order)
    for n_observed, row in curve.iterrows():
        indices = order[:n_observed]
        exact = sharp_partial_accuracy_envelope(
            quantum, classical, indices, labels[indices]
        )
        assert row.lower == pytest.approx(exact.lower)
        assert row.upper == pytest.approx(exact.upper)
    assert np.all(np.diff(curve.upper) <= 1e-15)
    realized = realized_accuracy_advantage(labels, quantum, classical)
    assert curve.iloc[-1].lower == pytest.approx(realized)
    assert curve.iloc[-1].upper == pytest.approx(realized)


def test_adaptive_policy_is_a_permutation_and_does_not_peek_at_first_query():
    quantum, classical, labels = _example()
    tie_order = stable_hash_order(len(labels), "unit", "adaptive")
    curve_a, order_a = adaptive_bottleneck_cover_curve(
        quantum, classical, labels, tie_order
    )
    altered = 1 - labels
    curve_b, order_b = adaptive_bottleneck_cover_curve(
        quantum, classical, altered, tie_order
    )
    assert order_a[0] == order_b[0]
    assert np.array_equal(np.sort(order_a), np.arange(len(labels)))
    assert np.array_equal(np.sort(order_b), np.arange(len(labels)))
    assert np.all(np.diff(curve_a.upper) <= 1e-15)
    assert np.all(np.diff(curve_b.upper) <= 1e-15)


def test_random_active_baseline_is_a_permutation_and_does_not_peek_first():
    quantum, classical, labels = _example()
    tie_order = stable_hash_order(len(labels), "unit", "random-active")
    curve_a, order_a = adaptive_random_active_disagreement_curve(
        quantum, classical, labels, tie_order
    )
    curve_b, order_b = adaptive_random_active_disagreement_curve(
        quantum, classical, 1 - labels, tie_order
    )
    assert order_a[0] == order_b[0]
    assert np.array_equal(np.sort(order_a), np.arange(len(labels)))
    assert np.array_equal(np.sort(order_b), np.arange(len(labels)))
    assert np.all(np.diff(curve_a.upper) <= 1e-15)
    assert np.all(np.diff(curve_b.upper) <= 1e-15)


def test_nonadaptive_coverage_order_uses_predictions_not_labels():
    quantum, classical, labels = _example()
    tie_order = stable_hash_order(len(labels), "unit", "coverage")
    order_a = nonadaptive_initial_coverage_order(
        quantum, classical, tie_order
    )
    order_b = nonadaptive_initial_coverage_order(
        quantum, classical, tie_order
    )
    assert np.array_equal(order_a, order_b)
    assert np.array_equal(np.sort(order_a), np.arange(len(labels)))

    disagreement = classical != quantum[:, None]
    total = disagreement.sum(axis=0)
    active = (total > 0) & (total == total[total > 0].min())
    coverage = disagreement[:, active].sum(axis=1)
    assert np.all(np.diff(coverage[order_a]) <= 0)


@pytest.mark.parametrize("threshold", [0.0, 1.0 / 6.0, 0.5])
def test_retrospective_oracle_matches_exhaustive_subset_search(threshold):
    quantum, classical, labels = _example()
    exact = -1
    for size in range(len(labels) + 1):
        reached = False
        for subset in combinations(range(len(labels)), size):
            indices = np.asarray(subset, dtype=np.int64)
            envelope = sharp_partial_accuracy_envelope(
                quantum, classical, indices, labels[indices]
            )
            if envelope.upper <= threshold + 1e-15:
                reached = True
                break
        if reached:
            exact = size
            break
    assert retrospective_oracle_minimum_queries(
        quantum, classical, labels, threshold
    ) == exact


def test_hash_order_and_threshold_crossing_are_deterministic():
    first = stable_hash_order(20, "case", 7)
    second = stable_hash_order(20, "case", 7)
    other = stable_hash_order(20, "case", 8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    quantum, classical, labels = _example()
    curve = audit_curve_from_order(
        quantum, classical, labels, stable_hash_order(len(labels), "threshold")
    )
    assert first_crossing(curve, float(curve.upper.iloc[0])) == 0
    assert first_crossing(curve, -2.0) == -1


def test_not_reached_hash_draws_are_not_counted_as_early_certificates():
    draws = pd.DataFrame(
        {
            "case": ["c", "c"],
            "group": ["g", "g"],
            "model": ["svc", "svc"],
            "quantum_stratum": ["entangling_zz", "entangling_zz"],
            "tier": ["full_115", "full_115"],
            "threshold": [0.01, 0.01],
            "n_labels": [-1, -1],
            "n_informative_counterexamples": [-1, -1],
        }
    )
    summary = summarize_hash_thresholds(draws).iloc[0]
    assert summary.n_reached == 0
    assert summary.probability_reached == 0
    assert summary.probability_by_25 == 0
    assert np.isnan(summary.median_n_labels)

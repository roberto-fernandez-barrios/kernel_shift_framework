"""Unit tests for the frozen v0.9 partial-identification estimands."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.analysis.partial_identification import (
    finite_family_population_correction,
    realized_accuracy_advantage,
    realized_balanced_accuracy_advantage,
    sharp_accuracy_envelope,
    sharp_balanced_accuracy_envelope,
    sharp_partial_accuracy_envelope,
    train_target_transport_counterexample,
)


def _all_binary_labels(n: int):
    for values in itertools.product((0, 1), repeat=n):
        yield np.asarray(values, dtype=np.int8)


def test_accuracy_envelope_matches_exhaustive_label_search():
    quantum = np.array([0, 0, 1, 1, 1], dtype=np.int8)
    classical = np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=np.int8,
    )
    envelope = sharp_accuracy_envelope(quantum, classical)
    realized = np.array(
        [realized_accuracy_advantage(y, quantum, classical) for y in _all_binary_labels(5)]
    )
    assert envelope.lower == pytest.approx(realized.min())
    assert envelope.upper == pytest.approx(realized.max())
    assert envelope.lower == pytest.approx(-envelope.disagreement_max)
    assert envelope.upper == pytest.approx(envelope.disagreement_min)


def test_accuracy_upper_is_attained_by_quantum_labels_and_lower_by_farthest():
    quantum = np.array([0, 1, 1, 0])
    classical = np.array([[0, 1], [1, 0], [0, 0], [0, 1]])
    envelope = sharp_accuracy_envelope(quantum, classical)
    assert realized_accuracy_advantage(quantum, quantum, classical) == pytest.approx(
        envelope.upper
    )
    farthest = classical[:, envelope.farthest_indices[0]]
    assert realized_accuracy_advantage(farthest, quantum, classical) == pytest.approx(
        envelope.lower
    )


def test_partial_accuracy_envelope_matches_all_label_completions():
    quantum = np.array([0, 0, 1, 1, 1], dtype=np.int8)
    classical = np.array(
        [[0, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]],
        dtype=np.int8,
    )
    observed_indices = np.array([1, 3])
    observed_labels = np.array([0, 1])
    envelope = sharp_partial_accuracy_envelope(
        quantum, classical, observed_indices, observed_labels
    )
    values = []
    for labels in _all_binary_labels(len(quantum)):
        if np.array_equal(labels[observed_indices], observed_labels):
            values.append(realized_accuracy_advantage(labels, quantum, classical))
    assert envelope.lower == pytest.approx(min(values))
    assert envelope.upper == pytest.approx(max(values))
    assert envelope.n_observed == 2


def test_partial_accuracy_endpoints_join_zero_and_full_label_regimes():
    quantum = np.array([0, 1, 1, 0])
    classical = np.array([[0, 1], [1, 0], [0, 0], [0, 1]])
    labels = np.array([0, 0, 1, 1])
    zero = sharp_partial_accuracy_envelope(
        quantum, classical, np.array([], dtype=int), np.array([], dtype=int)
    )
    label_free = sharp_accuracy_envelope(quantum, classical)
    assert zero.lower == pytest.approx(label_free.lower)
    assert zero.upper == pytest.approx(label_free.upper)

    full = sharp_partial_accuracy_envelope(
        quantum, classical, np.arange(len(labels)), labels
    )
    realized = realized_accuracy_advantage(labels, quantum, classical)
    assert full.lower == pytest.approx(realized)
    assert full.upper == pytest.approx(realized)


def test_partial_accuracy_upper_is_monotone_under_label_revelation():
    quantum = np.array([0, 1, 1, 0, 1])
    classical = np.array(
        [[0, 1], [1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.int8
    )
    labels = np.array([0, 0, 1, 1, 0])
    uppers = []
    for n_observed in range(len(labels) + 1):
        indices = np.arange(n_observed)
        envelope = sharp_partial_accuracy_envelope(
            quantum, classical, indices, labels[indices]
        )
        uppers.append(envelope.upper)
    assert np.all(np.diff(uppers) <= 1e-15)


def test_train_only_geometry_does_not_determine_target_prediction_operator():
    example = train_target_transport_counterexample(n_train=3, regularization=0.2)
    np.testing.assert_array_equal(
        example.joint_zero_transport[:3, :3],
        example.joint_max_transport[:3, :3],
    )
    np.testing.assert_array_equal(np.diag(example.joint_zero_transport), np.ones(4))
    np.testing.assert_array_equal(np.diag(example.joint_max_transport), np.ones(4))
    assert np.linalg.eigvalsh(example.joint_zero_transport).min() >= -1e-12
    assert np.linalg.eigvalsh(example.joint_max_transport).min() >= -1e-12
    assert example.operator_difference == pytest.approx(1.0 / 1.6)
    assert not np.array_equal(
        example.prediction_operator_zero, example.prediction_operator_max
    )


def test_exact_bacc_milp_matches_exhaustive_fixed_prevalence_search():
    quantum = np.array([0, 0, 1, 1, 1, 0], dtype=np.int8)
    classical = np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.int8,
    )
    exact = sharp_balanced_accuracy_envelope(
        quantum,
        classical,
        n_positive=3,
        integral=True,
    )
    realized = []
    for labels in _all_binary_labels(len(quantum)):
        if labels.sum() == 3:
            realized.append(
                realized_balanced_accuracy_advantage(labels, quantum, classical)
            )
    assert exact.lower == pytest.approx(min(realized), abs=1e-10)
    assert exact.upper == pytest.approx(max(realized), abs=1e-10)
    assert exact.n_positive == 3
    assert exact.n_signatures <= len(quantum)


def test_bacc_lp_relaxation_contains_integer_envelope():
    quantum = np.array([0, 0, 1, 1, 1])
    classical = np.array([[0, 1], [1, 1], [1, 0], [0, 1], [1, 0]])
    integer = sharp_balanced_accuracy_envelope(
        quantum, classical, n_positive=2, integral=True
    )
    relaxed = sharp_balanced_accuracy_envelope(
        quantum, classical, n_positive=2, integral=False
    )
    assert relaxed.lower <= integer.lower + 1e-10
    assert relaxed.upper >= integer.upper - 1e-10


def test_population_correction_matches_frozen_q1000_value():
    correction = finite_family_population_correction(115, 500, delta=0.05)
    assert correction == pytest.approx(0.091836, abs=1e-6)


@pytest.mark.parametrize("n_positive", [0, 4])
def test_bacc_rejects_single_class_targets(n_positive):
    quantum = np.array([0, 0, 1, 1])
    classical = np.array([[0], [1], [1], [0]])
    with pytest.raises(ValueError, match="strictly between"):
        sharp_balanced_accuracy_envelope(
            quantum, classical, n_positive=n_positive, integral=True
        )

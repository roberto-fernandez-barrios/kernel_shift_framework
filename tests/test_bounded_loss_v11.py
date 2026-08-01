"""Exact-enumeration gates for the v1.1 bounded-loss identified set."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.analysis.partial_identification import (
    sharp_bounded_loss_envelope,
    sharp_partial_accuracy_envelope,
)


def _all_labels(n_target: int, n_labels: int):
    for values in itertools.product(range(n_labels), repeat=n_target):
        yield np.asarray(values, dtype=np.int64)


def _advantage(
    labels: np.ndarray,
    quantum_losses: np.ndarray,
    classical_losses: np.ndarray,
) -> float:
    rows = np.arange(len(labels))
    quantum_risk = np.mean(quantum_losses[rows, labels])
    classical_risks = np.mean(classical_losses[rows, :, labels], axis=0)
    return float(np.min(classical_risks) - quantum_risk)


def _zero_one_losses(
    quantum: np.ndarray, classical: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([0, 1], dtype=np.int8)
    quantum_losses = (quantum[:, None] != labels[None, :]).astype(float)
    classical_losses = (
        classical[:, :, None] != labels[None, None, :]
    ).astype(float)
    return quantum_losses, classical_losses


def test_general_bounded_loss_matches_exhaustive_brier_search():
    quantum_prob = np.array([0.10, 0.35, 0.75, 0.90])
    classical_prob = np.array(
        [
            [0.20, 0.70, 0.05],
            [0.50, 0.25, 0.65],
            [0.60, 0.85, 0.45],
            [0.80, 0.55, 0.95],
        ]
    )
    labels = np.array([0.0, 1.0])
    quantum_losses = (quantum_prob[:, None] - labels[None, :]) ** 2
    classical_losses = (
        classical_prob[:, :, None] - labels[None, None, :]
    ) ** 2

    envelope = sharp_bounded_loss_envelope(quantum_losses, classical_losses)
    values = np.array(
        [
            _advantage(y, quantum_losses, classical_losses)
            for y in _all_labels(4, 2)
        ]
    )
    assert envelope.lower == pytest.approx(values.min(), abs=1e-10)
    assert envelope.upper == pytest.approx(values.max(), abs=1e-10)
    assert _advantage(
        envelope.upper_label_indices, quantum_losses, classical_losses
    ) == pytest.approx(envelope.upper, abs=1e-10)


def test_general_partial_label_envelope_matches_multiclass_enumeration():
    # Arbitrary bounded loss tables demonstrate that no common label need
    # maximize every witness contrast, so the upper max--min is genuinely used.
    quantum_losses = np.array(
        [
            [0.05, 0.40, 0.80],
            [0.70, 0.10, 0.55],
            [0.60, 0.35, 0.15],
        ]
    )
    classical_losses = np.array(
        [
            [[0.20, 0.30, 0.65], [0.10, 0.75, 0.40]],
            [[0.50, 0.25, 0.60], [0.80, 0.15, 0.35]],
            [[0.45, 0.50, 0.25], [0.70, 0.20, 0.10]],
        ]
    )
    observed_indices = np.array([1])
    observed_labels = np.array([2])
    envelope = sharp_bounded_loss_envelope(
        quantum_losses,
        classical_losses,
        observed_indices,
        observed_labels,
    )
    values = []
    for labels in _all_labels(3, 3):
        if labels[1] == 2:
            values.append(_advantage(labels, quantum_losses, classical_losses))
    assert envelope.lower == pytest.approx(min(values), abs=1e-10)
    assert envelope.upper == pytest.approx(max(values), abs=1e-10)
    assert envelope.upper_label_indices[1] == 2


def test_zero_one_specialization_recovers_closed_form_accuracy_envelope():
    quantum = np.array([0, 0, 1, 1, 1], dtype=np.int8)
    classical = np.array(
        [[0, 1, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]],
        dtype=np.int8,
    )
    observed_indices = np.array([1, 3])
    observed_labels = np.array([0, 1])
    quantum_losses, classical_losses = _zero_one_losses(quantum, classical)
    general = sharp_bounded_loss_envelope(
        quantum_losses,
        classical_losses,
        observed_indices,
        observed_labels,
    )
    accuracy = sharp_partial_accuracy_envelope(
        quantum, classical, observed_indices, observed_labels
    )
    assert general.lower == pytest.approx(accuracy.lower, abs=1e-10)
    assert general.upper == pytest.approx(accuracy.upper, abs=1e-10)


def test_general_envelope_contracts_under_any_label_revelation_order():
    quantum_prob = np.array([0.15, 0.45, 0.70, 0.85])
    classical_prob = np.array(
        [[0.25, 0.75], [0.55, 0.30], [0.65, 0.90], [0.60, 0.95]]
    )
    label_values = np.array([0.0, 1.0])
    quantum_losses = (quantum_prob[:, None] - label_values[None, :]) ** 2
    classical_losses = (
        classical_prob[:, :, None] - label_values[None, None, :]
    ) ** 2
    realized_labels = np.array([0, 1, 1, 0])
    lowers = []
    uppers = []
    for n_observed in range(5):
        indices = np.arange(n_observed)
        envelope = sharp_bounded_loss_envelope(
            quantum_losses,
            classical_losses,
            indices,
            realized_labels[indices],
        )
        lowers.append(envelope.lower)
        uppers.append(envelope.upper)
    assert np.all(np.diff(lowers) >= -1e-10)
    assert np.all(np.diff(uppers) <= 1e-10)
    assert lowers[-1] == pytest.approx(uppers[-1], abs=1e-10)


def test_bounded_loss_validation_rejects_out_of_range_values():
    quantum_losses = np.array([[0.0, 1.2], [0.2, 0.8]])
    classical_losses = np.zeros((2, 1, 2))
    with pytest.raises(ValueError, match="outside loss_bounds"):
        sharp_bounded_loss_envelope(quantum_losses, classical_losses)

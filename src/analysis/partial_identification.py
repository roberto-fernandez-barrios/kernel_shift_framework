"""Sharp target-batch identification of relative classifier performance.

The zero-label functions in this module never require target labels.  The
explicit partial-label helper accepts only an audited label subset and leaves
all other target labels unrestricted.

For a quantum hard classifier ``q`` and a finite classical family ``C``, the
accuracy advantage over the best classical member is

    Delta(y) = Acc(q; y) - max_j Acc(C_j; y).

Its sharp target-batch endpoints over otherwise unrestricted labels are the
negative maximum and the minimum quantum--classical disagreement rates.  For
balanced accuracy at a fixed class count, exact finite-batch endpoints are
obtained by mixed-integer linear programming over joint prediction
signatures.

More generally, fixed predictions and any additive bounded loss on a finite
label space induce a finite max--min identification problem.  Its lower
endpoint separates over target items and classical witnesses; its upper
endpoint is an exact mixed-integer linear program.  Zero--one accuracy is the
special case in which assigning every unknown label to the quantum hard
prediction maximizes all witness contrasts simultaneously, giving the simple
closed forms above.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Sequence

import numpy as np


def _binary_vector(values: np.ndarray | Sequence[int], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, found {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only binary predictions")
    return array.astype(np.int8, copy=False)


def _classical_matrix(
    values: np.ndarray | Sequence[Sequence[int]],
    n_target: int,
) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != n_target:
        raise ValueError(
            "classical_predictions must have shape "
            f"({n_target}, n_models), found {matrix.shape}"
        )
    if matrix.shape[1] == 0:
        raise ValueError("the classical reference family must not be empty")
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError("classical_predictions must be binary")
    return matrix.astype(np.int8, copy=False)


def disagreement_rates(
    quantum_predictions: np.ndarray | Sequence[int],
    classical_predictions: np.ndarray | Sequence[Sequence[int]],
) -> np.ndarray:
    """Return one empirical target disagreement rate per classical model."""
    quantum = _binary_vector(quantum_predictions, "quantum_predictions")
    classical = _classical_matrix(classical_predictions, len(quantum))
    return np.mean(classical != quantum[:, None], axis=0, dtype=np.float64)


@dataclass(frozen=True)
class AccuracyEnvelope:
    """Sharp finite-target accuracy envelope relative to a classical family."""

    lower: float
    upper: float
    disagreement_min: float
    disagreement_max: float
    nearest_indices: tuple[int, ...]
    farthest_indices: tuple[int, ...]
    n_target: int
    n_classical: int


@dataclass(frozen=True)
class PartialAccuracyEnvelope:
    """Sharp accuracy envelope after auditing a subset of target labels."""

    lower: float
    upper: float
    lower_indices: tuple[int, ...]
    upper_indices: tuple[int, ...]
    observed_signed_counts: np.ndarray
    remaining_disagreement_counts: np.ndarray
    n_observed: int
    n_target: int
    n_classical: int


@dataclass(frozen=True)
class BoundedLossEnvelope:
    """Sharp advantage envelope for an additive bounded loss.

    Advantage is defined as the best classical empirical loss minus the
    quantum empirical loss, so positive values favour the quantum model.
    ``upper_label_indices`` records one target-label completion attaining the
    exact upper endpoint; audited entries are set to their observed labels.
    """

    lower: float
    upper: float
    lower_indices: tuple[int, ...]
    upper_indices: tuple[int, ...]
    upper_label_indices: np.ndarray
    observed_signed_losses: np.ndarray
    n_observed: int
    n_target: int
    n_labels: int
    n_classical: int
    upper_status: int


def _loss_arrays(
    quantum_losses: np.ndarray | Sequence[Sequence[float]],
    classical_losses: np.ndarray | Sequence[Sequence[Sequence[float]]],
    loss_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    quantum = np.asarray(quantum_losses, dtype=float)
    classical = np.asarray(classical_losses, dtype=float)
    if quantum.ndim != 2 or quantum.shape[0] == 0 or quantum.shape[1] < 2:
        raise ValueError(
            "quantum_losses must have shape (n_target, n_labels) with "
            "n_target > 0 and n_labels >= 2"
        )
    expected = (quantum.shape[0], quantum.shape[1])
    if (
        classical.ndim != 3
        or classical.shape[0] != expected[0]
        or classical.shape[2] != expected[1]
        or classical.shape[1] == 0
    ):
        raise ValueError(
            "classical_losses must have shape "
            f"({expected[0]}, n_models, {expected[1]}) with n_models > 0; "
            f"found {classical.shape}"
        )
    if not np.isfinite(quantum).all() or not np.isfinite(classical).all():
        raise ValueError("loss arrays must contain only finite values")
    lower, upper = map(float, loss_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("loss_bounds must be finite and ordered")
    tolerance = 1e-12
    if (
        quantum.min() < lower - tolerance
        or quantum.max() > upper + tolerance
        or classical.min() < lower - tolerance
        or classical.max() > upper + tolerance
    ):
        raise ValueError("loss arrays contain values outside loss_bounds")
    return quantum, classical


def sharp_bounded_loss_envelope(
    quantum_losses: np.ndarray | Sequence[Sequence[float]],
    classical_losses: np.ndarray | Sequence[Sequence[Sequence[float]]],
    observed_indices: np.ndarray | Sequence[int] = (),
    observed_label_indices: np.ndarray | Sequence[int] = (),
    *,
    loss_bounds: tuple[float, float] = (0.0, 1.0),
    atol: float = 1e-10,
) -> BoundedLossEnvelope:
    """Compute sharp partial-label endpoints for any bounded additive loss.

    ``quantum_losses[i, y]`` is the loss of the fixed quantum prediction on
    target item ``i`` if its label is ``y``.  The corresponding classical
    array has shape ``(n_target, n_models, n_labels)``.  Labels are represented
    by integer column indices; arbitrary finite label values can therefore be
    encoded before calling this function.

    For loss contrast ``a_ij(y) = loss(C_j, y) - loss(Q, y)``, the lower
    endpoint is

    ``min_j [S_j + sum_i min_y a_ij(y)] / n``.

    The upper endpoint is the exact finite max--min over all completions of
    the unaudited labels and is solved by a binary linear program.  Both
    endpoints are attained, so every assumption-free interval based only on
    these fixed losses and audited labels must contain them.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    quantum, classical = _loss_arrays(
        quantum_losses, classical_losses, loss_bounds
    )
    n_target, n_labels = quantum.shape
    n_classical = classical.shape[1]
    indices = np.asarray(observed_indices)
    labels = np.asarray(observed_label_indices)
    if indices.ndim != 1 or labels.ndim != 1:
        raise ValueError(
            "observed_indices and observed_label_indices must be one-dimensional"
        )
    if len(indices) != len(labels):
        raise ValueError(
            "observed_indices and observed_label_indices have different lengths"
        )
    if not len(indices):
        indices = indices.astype(np.int64)
    if not len(labels):
        labels = labels.astype(np.int64)
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("observed_indices must be integers")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("observed_label_indices must be integers")
    indices = indices.astype(np.int64, copy=False)
    labels = labels.astype(np.int64, copy=False)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("observed_indices must be unique")
    if len(indices) and (indices.min() < 0 or indices.max() >= n_target):
        raise ValueError("observed_indices lie outside the target batch")
    if len(labels) and (labels.min() < 0 or labels.max() >= n_labels):
        raise ValueError("observed_label_indices lie outside the finite label space")

    contrast = classical - quantum[:, None, :]
    observed_mask = np.zeros(n_target, dtype=bool)
    observed_mask[indices] = True
    signed = np.zeros(n_classical, dtype=float)
    if len(indices):
        signed = np.sum(
            contrast[indices, :, labels], axis=0, dtype=np.float64
        )
    unknown_indices = np.flatnonzero(~observed_mask)

    lower_candidates = signed.copy()
    if len(unknown_indices):
        lower_candidates += np.sum(
            np.min(contrast[unknown_indices], axis=2), axis=0, dtype=np.float64
        )
    lower_candidates /= float(n_target)
    lower = float(np.min(lower_candidates))
    lower_indices = tuple(
        int(i)
        for i in np.flatnonzero(np.isclose(lower_candidates, lower, atol=atol))
    )

    completed_labels = np.full(n_target, -1, dtype=np.int64)
    completed_labels[indices] = labels
    if not len(unknown_indices):
        witness_values = signed / float(n_target)
        upper = float(np.min(witness_values))
        upper_indices = tuple(
            int(i)
            for i in np.flatnonzero(np.isclose(witness_values, upper, atol=atol))
        )
        return BoundedLossEnvelope(
            lower=lower,
            upper=upper,
            lower_indices=lower_indices,
            upper_indices=upper_indices,
            upper_label_indices=completed_labels,
            observed_signed_losses=signed,
            n_observed=len(indices),
            n_target=n_target,
            n_labels=n_labels,
            n_classical=n_classical,
            upper_status=0,
        )

    # Variables are one-hot labels z_(i,y) for every unaudited item, followed
    # by the max--min value t.  Maximize t subject to each witness contrast
    # being at least t.
    n_unknown = len(unknown_indices)
    n_binary = n_unknown * n_labels
    objective = np.zeros(n_binary + 1, dtype=float)
    objective[-1] = -1.0
    integrality = np.append(np.ones(n_binary, dtype=int), 0)
    variable_bounds = Bounds(
        np.append(np.zeros(n_binary), -np.inf),
        np.append(np.ones(n_binary), np.inf),
    )

    one_hot = np.zeros((n_unknown, n_binary + 1), dtype=float)
    for row in range(n_unknown):
        start = row * n_labels
        one_hot[row, start : start + n_labels] = 1.0
    label_constraint = LinearConstraint(
        one_hot, np.ones(n_unknown), np.ones(n_unknown)
    )

    witness_matrix = np.zeros((n_classical, n_binary + 1), dtype=float)
    unknown_contrast = contrast[unknown_indices]
    for witness in range(n_classical):
        witness_matrix[witness, :n_binary] = -unknown_contrast[
            :, witness, :
        ].reshape(-1) / float(n_target)
    witness_matrix[:, -1] = 1.0
    witness_constraint = LinearConstraint(
        witness_matrix,
        np.full(n_classical, -np.inf),
        signed / float(n_target),
    )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=variable_bounds,
        constraints=(label_constraint, witness_constraint),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success:
        raise RuntimeError(
            f"bounded-loss upper optimization failed with status {result.status}: "
            f"{result.message}"
        )
    assignments = result.x[:n_binary].reshape(n_unknown, n_labels)
    unknown_labels = np.argmax(assignments, axis=1).astype(np.int64)
    completed_labels[unknown_indices] = unknown_labels
    upper_values = signed + np.sum(
        contrast[unknown_indices, :, unknown_labels], axis=0, dtype=np.float64
    )
    upper_values /= float(n_target)
    upper = float(np.min(upper_values))
    upper_indices = tuple(
        int(i) for i in np.flatnonzero(np.isclose(upper_values, upper, atol=atol))
    )
    if not np.isclose(upper, -float(result.fun), atol=atol):
        raise RuntimeError("MILP solution and reconstructed upper endpoint disagree")

    return BoundedLossEnvelope(
        lower=lower,
        upper=upper,
        lower_indices=lower_indices,
        upper_indices=upper_indices,
        upper_label_indices=completed_labels,
        observed_signed_losses=signed,
        n_observed=len(indices),
        n_target=n_target,
        n_labels=n_labels,
        n_classical=n_classical,
        upper_status=int(result.status),
    )


def sharp_accuracy_envelope(
    quantum_predictions: np.ndarray | Sequence[int],
    classical_predictions: np.ndarray | Sequence[Sequence[int]],
    *,
    atol: float = 1e-15,
) -> AccuracyEnvelope:
    """Compute the sharp lower and upper accuracy-advantage endpoints.

    The endpoints are sharp over every possible hard labeling of the observed
    target batch.  In a finite batch the attainable values between the two
    endpoints can be discrete, so the return value is an envelope rather than
    a claim that every intermediate real number is attainable.
    """
    quantum = _binary_vector(quantum_predictions, "quantum_predictions")
    classical = _classical_matrix(classical_predictions, len(quantum))
    rates = disagreement_rates(quantum, classical)
    d_min = float(np.min(rates))
    d_max = float(np.max(rates))
    nearest = tuple(int(i) for i in np.flatnonzero(np.isclose(rates, d_min, atol=atol)))
    farthest = tuple(int(i) for i in np.flatnonzero(np.isclose(rates, d_max, atol=atol)))
    return AccuracyEnvelope(
        lower=-d_max,
        upper=d_min,
        disagreement_min=d_min,
        disagreement_max=d_max,
        nearest_indices=nearest,
        farthest_indices=farthest,
        n_target=len(quantum),
        n_classical=classical.shape[1],
    )


def sharp_partial_accuracy_envelope(
    quantum_predictions: np.ndarray | Sequence[int],
    classical_predictions: np.ndarray | Sequence[Sequence[int]],
    observed_indices: np.ndarray | Sequence[int],
    observed_labels: np.ndarray | Sequence[int],
    *,
    atol: float = 1e-15,
) -> PartialAccuracyEnvelope:
    """Compute sharp endpoints given only an audited target-label subset.

    For witness ``j``, ``S_j`` is its observed signed accuracy difference
    against the quantum classifier and ``R_j`` is its remaining unlabelled
    disagreement count.  The sharp endpoints are

    ``min_j (S_j - R_j) / n`` and ``min_j (S_j + R_j) / n``.

    Choosing every unaudited label equal to the quantum prediction attains
    the upper endpoint simultaneously for all witnesses.  Choosing them equal
    to a lower-endpoint witness attains the lower endpoint.  The result is
    therefore exact for binary and multiclass hard classification; this
    implementation validates binary inputs because the surrounding study is
    binary.
    """
    quantum = _binary_vector(quantum_predictions, "quantum_predictions")
    classical = _classical_matrix(classical_predictions, len(quantum))
    indices = np.asarray(observed_indices)
    labels = np.asarray(observed_labels)
    if indices.ndim != 1 or labels.ndim != 1:
        raise ValueError("observed_indices and observed_labels must be one-dimensional")
    if len(indices) != len(labels):
        raise ValueError("observed_indices and observed_labels have different lengths")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("observed_indices must be integers")
    indices = indices.astype(np.int64, copy=False)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("observed_indices must be unique")
    if len(indices) and (indices.min() < 0 or indices.max() >= len(quantum)):
        raise ValueError("observed_indices lie outside the target batch")
    if len(labels) and not np.isin(labels, (0, 1)).all():
        raise ValueError("observed_labels must be binary")
    labels = labels.astype(np.int8, copy=False)

    observed = np.zeros(len(quantum), dtype=bool)
    observed[indices] = True
    signed = np.zeros(classical.shape[1], dtype=np.int64)
    if len(indices):
        quantum_correct = quantum[indices] == labels
        classical_correct = classical[indices] == labels[:, None]
        signed = np.sum(
            quantum_correct[:, None].astype(np.int8)
            - classical_correct.astype(np.int8),
            axis=0,
            dtype=np.int64,
        )
    remaining = np.sum(
        classical[~observed] != quantum[~observed, None],
        axis=0,
        dtype=np.int64,
    )
    lower_candidates = (signed - remaining) / float(len(quantum))
    upper_candidates = (signed + remaining) / float(len(quantum))
    lower = float(np.min(lower_candidates))
    upper = float(np.min(upper_candidates))
    lower_indices = tuple(
        int(i)
        for i in np.flatnonzero(np.isclose(lower_candidates, lower, atol=atol))
    )
    upper_indices = tuple(
        int(i)
        for i in np.flatnonzero(np.isclose(upper_candidates, upper, atol=atol))
    )
    return PartialAccuracyEnvelope(
        lower=lower,
        upper=upper,
        lower_indices=lower_indices,
        upper_indices=upper_indices,
        observed_signed_counts=signed,
        remaining_disagreement_counts=remaining,
        n_observed=len(indices),
        n_target=len(quantum),
        n_classical=classical.shape[1],
    )


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions == labels))


def realized_accuracy_advantage(
    labels: np.ndarray | Sequence[int],
    quantum_predictions: np.ndarray | Sequence[int],
    classical_predictions: np.ndarray | Sequence[Sequence[int]],
) -> float:
    """Evaluate the realized advantage after target labels are unlocked."""
    target = _binary_vector(labels, "labels")
    quantum = _binary_vector(quantum_predictions, "quantum_predictions")
    if len(target) != len(quantum):
        raise ValueError("labels and quantum_predictions have different lengths")
    classical = _classical_matrix(classical_predictions, len(target))
    q_accuracy = _accuracy(quantum, target)
    c_accuracy = max(_accuracy(classical[:, j], target) for j in range(classical.shape[1]))
    return q_accuracy - c_accuracy


def _balanced_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    positive = labels == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("balanced accuracy requires both target classes")
    tpr = np.mean(predictions[positive] == 1)
    tnr = np.mean(predictions[negative] == 0)
    return float(0.5 * (tpr + tnr))


def realized_balanced_accuracy_advantage(
    labels: np.ndarray | Sequence[int],
    quantum_predictions: np.ndarray | Sequence[int],
    classical_predictions: np.ndarray | Sequence[Sequence[int]],
) -> float:
    """Evaluate realized BAcc advantage after target labels are unlocked."""
    target = _binary_vector(labels, "labels")
    quantum = _binary_vector(quantum_predictions, "quantum_predictions")
    if len(target) != len(quantum):
        raise ValueError("labels and quantum_predictions have different lengths")
    classical = _classical_matrix(classical_predictions, len(target))
    q_bacc = _balanced_accuracy(quantum, target)
    c_bacc = max(
        _balanced_accuracy(classical[:, j], target)
        for j in range(classical.shape[1])
    )
    return q_bacc - c_bacc


@dataclass(frozen=True)
class PredictionSignatures:
    patterns: np.ndarray
    counts: np.ndarray
    inverse: np.ndarray


@dataclass(frozen=True)
class TrainTargetTransportCounterexample:
    """Normalized PSD kernels sharing train geometry but not target transport."""

    joint_zero_transport: np.ndarray
    joint_max_transport: np.ndarray
    train_block: np.ndarray
    target_train_zero: np.ndarray
    target_train_max: np.ndarray
    prediction_operator_zero: np.ndarray
    prediction_operator_max: np.ndarray
    operator_difference: float


def train_target_transport_counterexample(
    n_train: int = 2,
    regularization: float = 0.1,
) -> TrainTargetTransportCounterexample:
    """Construct identical ``K_TT`` with distinct normalized PSD extensions.

    The first target is orthogonal to every training feature; in the second
    extension it duplicates the first training feature.  Both joint Gram
    matrices have unit diagonal and are positive semidefinite.  For KRR with
    the convention ``K_TT + n * regularization * I``, their target prediction
    operators differ by exactly ``1 / (1 + n * regularization)`` in spectral
    norm despite having identical train-only geometry.
    """
    if int(n_train) != n_train or n_train <= 0:
        raise ValueError("n_train must be a positive integer")
    if regularization < 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    n_train = int(n_train)
    joint_zero = np.eye(n_train + 1, dtype=float)
    joint_max = np.eye(n_train + 1, dtype=float)
    joint_max[0, n_train] = 1.0
    joint_max[n_train, 0] = 1.0
    train = np.eye(n_train, dtype=float)
    target_zero = np.zeros((1, n_train), dtype=float)
    target_max = np.zeros((1, n_train), dtype=float)
    target_max[0, 0] = 1.0
    inverse = np.linalg.inv(
        train + n_train * float(regularization) * np.eye(n_train)
    )
    operator_zero = target_zero @ inverse
    operator_max = target_max @ inverse
    difference = float(np.linalg.norm(operator_max - operator_zero, ord=2))
    return TrainTargetTransportCounterexample(
        joint_zero_transport=joint_zero,
        joint_max_transport=joint_max,
        train_block=train,
        target_train_zero=target_zero,
        target_train_max=target_max,
        prediction_operator_zero=operator_zero,
        prediction_operator_max=operator_max,
        operator_difference=difference,
    )


def prediction_signatures(
    quantum_predictions: np.ndarray | Sequence[int],
    classical_predictions: np.ndarray | Sequence[Sequence[int]],
) -> PredictionSignatures:
    """Compress equal joint hard-prediction rows into unique signatures."""
    quantum = _binary_vector(quantum_predictions, "quantum_predictions")
    classical = _classical_matrix(classical_predictions, len(quantum))
    joint = np.column_stack((quantum, classical)).astype(np.int8, copy=False)
    patterns, inverse, counts = np.unique(
        joint,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return PredictionSignatures(
        patterns=patterns,
        counts=counts.astype(np.int64, copy=False),
        inverse=inverse.astype(np.int64, copy=False),
    )


def _bacc_difference_affine(
    q_by_signature: np.ndarray,
    c_by_signature: np.ndarray,
    counts: np.ndarray,
    n_positive: int,
) -> tuple[float, np.ndarray]:
    """Return constant and coefficients for BAcc(Q)-BAcc(C).

    The variables are the numbers of positive labels assigned to each joint
    prediction signature.
    """
    n_target = int(np.sum(counts))
    n_negative = n_target - int(n_positive)
    if n_positive <= 0 or n_negative <= 0:
        raise ValueError("n_positive must leave both classes present")
    difference = q_by_signature.astype(float) - c_by_signature.astype(float)
    constant = float(np.sum(-difference * counts) / (2.0 * n_negative))
    coefficients = 0.5 * difference * (
        1.0 / float(n_positive) + 1.0 / float(n_negative)
    )
    return constant, coefficients


@dataclass(frozen=True)
class BalancedAccuracyEnvelope:
    lower: float
    upper: float
    lower_witness_index: int
    n_target: int
    n_positive: int
    n_classical: int
    n_signatures: int
    integral: bool
    upper_status: int
    lower_statuses: tuple[int, ...]


def sharp_balanced_accuracy_envelope(
    quantum_predictions: np.ndarray | Sequence[int],
    classical_predictions: np.ndarray | Sequence[Sequence[int]],
    *,
    n_positive: int,
    integral: bool = True,
) -> BalancedAccuracyEnvelope:
    """Compute prevalence-conditional sharp BAcc endpoints by MILP or LP.

    ``integral=True`` gives the exact finite-batch result over binary labels.
    ``integral=False`` relaxes signature-positive counts to continuous masses
    and is appropriate as a distributional sensitivity.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    quantum = _binary_vector(quantum_predictions, "quantum_predictions")
    classical = _classical_matrix(classical_predictions, len(quantum))
    n_target = len(quantum)
    if not 0 < int(n_positive) < n_target:
        raise ValueError("n_positive must be strictly between zero and n_target")

    signatures = prediction_signatures(quantum, classical)
    patterns = signatures.patterns
    counts = signatures.counts.astype(float)
    q_signature = patterns[:, 0]
    c_signatures = patterns[:, 1:]
    n_signature = len(counts)

    constants: list[float] = []
    coefficients: list[np.ndarray] = []
    for j in range(classical.shape[1]):
        constant, coefficient = _bacc_difference_affine(
            q_signature,
            c_signatures[:, j],
            counts,
            int(n_positive),
        )
        constants.append(constant)
        coefficients.append(coefficient)

    integrality = np.ones(n_signature, dtype=int) if integral else np.zeros(n_signature, dtype=int)
    count_constraint = LinearConstraint(
        np.ones((1, n_signature)),
        lb=float(n_positive),
        ub=float(n_positive),
    )
    count_bounds = Bounds(np.zeros(n_signature), counts)
    options = {"presolve": True}
    if integral:
        options["mip_rel_gap"] = 0.0

    lower_values: list[float] = []
    lower_statuses: list[int] = []
    for constant, coefficient in zip(constants, coefficients):
        result = milp(
            c=coefficient,
            integrality=integrality,
            bounds=count_bounds,
            constraints=count_constraint,
            options=options,
        )
        if not result.success:
            raise RuntimeError(
                f"BAcc lower optimization failed with status {result.status}: "
                f"{result.message}"
            )
        lower_values.append(float(constant + result.fun))
        lower_statuses.append(int(result.status))
    lower_witness = int(np.argmin(lower_values))

    # Maximize t subject to t <= constant_j + coefficient_j @ k for every j.
    objective = np.zeros(n_signature + 1)
    objective[-1] = -1.0
    upper_integrality = np.append(integrality, 0)
    upper_bounds = Bounds(
        np.append(np.zeros(n_signature), -np.inf),
        np.append(counts, np.inf),
    )
    equality = LinearConstraint(
        np.append(np.ones(n_signature), 0.0)[None, :],
        lb=float(n_positive),
        ub=float(n_positive),
    )
    inequalities = np.column_stack((-np.vstack(coefficients), np.ones(len(constants))))
    upper_constraint = LinearConstraint(
        inequalities,
        lb=np.full(len(constants), -np.inf),
        ub=np.asarray(constants),
    )
    upper_result = milp(
        c=objective,
        integrality=upper_integrality,
        bounds=upper_bounds,
        constraints=(equality, upper_constraint),
        options=options,
    )
    if not upper_result.success:
        raise RuntimeError(
            f"BAcc upper optimization failed with status {upper_result.status}: "
            f"{upper_result.message}"
        )

    return BalancedAccuracyEnvelope(
        lower=float(lower_values[lower_witness]),
        upper=float(-upper_result.fun),
        lower_witness_index=lower_witness,
        n_target=n_target,
        n_positive=int(n_positive),
        n_classical=classical.shape[1],
        n_signatures=n_signature,
        integral=bool(integral),
        upper_status=int(upper_result.status),
        lower_statuses=tuple(lower_statuses),
    )


def finite_family_population_correction(
    n_models: int,
    n_target: int,
    delta: float = 0.05,
) -> float:
    """Two-sided finite-family Hoeffding correction from the frozen spec."""
    if n_models <= 0 or n_target <= 0:
        raise ValueError("n_models and n_target must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    return sqrt(log(2.0 * n_models / delta) / (2.0 * n_target))

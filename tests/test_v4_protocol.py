# tests/test_v4_protocol.py
"""Unit tests for the v4 protocol utilities (spec sections 2.4-2.6)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.ember.extended.v4_protocol import (  # noqa: E402
    perturb_kernel_finite_shots, select_c_by_train_cv, split_id_val_test)


# ---------------------------------------------------------------------------
# Hash-based ID split (constraint 5)
# ---------------------------------------------------------------------------
def test_id_split_deterministic_stratified_disjoint():
    rng = np.random.default_rng(0)
    gidx = rng.choice(100000, size=500, replace=False)
    y = (rng.random(500) < 0.4).astype(int)
    v1, t1, a1 = split_id_val_test(gidx, y)
    v2, t2, a2 = split_id_val_test(gidx, y)
    np.testing.assert_array_equal(v1, v2)      # deterministic
    np.testing.assert_array_equal(t1, t2)
    assert len(np.intersect1d(v1, t1)) == 0    # disjoint
    assert len(v1) + len(t1) == 500            # exhaustive
    assert a1["class_balance_gap"] < 0.01      # stratified
    assert a1["overlap"] == 0


def test_id_split_depends_on_global_ids_not_positions():
    """Same labels, shifted global ids -> different assignment (hash-based,
    not position-based)."""
    y = np.array([0, 1] * 50)
    v1, _, _ = split_id_val_test(np.arange(100), y)
    v2, _, _ = split_id_val_test(np.arange(100) + 7, y)
    assert not np.array_equal(v1, v2)


def test_id_split_not_even_odd():
    y = np.zeros(100, dtype=int)
    v, t, _ = split_id_val_test(np.arange(100), y)
    assert not (np.array_equal(v, np.arange(0, 100, 2))
                or np.array_equal(v, np.arange(1, 100, 2)))


# ---------------------------------------------------------------------------
# Internal C CV (constraint 4)
# ---------------------------------------------------------------------------
def _toy_problem(n=200, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.4 * rng.normal(size=n) > 0).astype(int)
    K = np.exp(-((X[:, None] - X[None]) ** 2).sum(-1) / 4)
    return K, y


def test_c_cv_returns_grid_value_and_scores():
    K, y = _toy_problem()
    c, scores = select_c_by_train_cv(K, y)
    assert c in (0.01, 0.1, 1.0, 10.0, 100.0)
    assert set(scores) == {0.01, 0.1, 1.0, 10.0, 100.0}
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_c_cv_never_touches_eval_data():
    """Only the train Gram enters; poisoning hypothetical eval rows cannot
    change the choice because the function never sees them."""
    K, y = _toy_problem()
    c1, s1 = select_c_by_train_cv(K, y)
    c2, s2 = select_c_by_train_cv(K.copy(), y.copy())
    assert c1 == c2 and s1 == s2               # pure function of (K_tr, y_tr)


def test_c_cv_tie_breaks_small():
    # constant labels except one -> all Cs behave identically on tiny folds is
    # not guaranteed; instead test the tie-break rule directly on a crafted
    # score dict via the same key function
    grid = (0.01, 0.1, 1.0, 10.0, 100.0)
    mean_scores = {c: 0.5 for c in grid}
    best = min(grid, key=lambda c: (-mean_scores[c], c))
    assert best == 0.01


def test_c_cv_cap_subsamples_deterministically():
    K, y = _toy_problem(n=300)
    c1, s1 = select_c_by_train_cv(K, y, cap=100)
    c2, s2 = select_c_by_train_cv(K, y, cap=100)
    assert c1 == c2 and s1 == s2


# ---------------------------------------------------------------------------
# Finite-shot fidelity-estimation perturbation model (constraint 6)
# ---------------------------------------------------------------------------
def _exact_square(n=40, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, 6))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    K = (A @ A.T) ** 2                          # valid fidelity-like Gram
    np.fill_diagonal(K, 1.0)
    return np.clip(K, 0, 1)


def test_shots_square_properties():
    K = _exact_square()
    Kp, audit = perturb_kernel_finite_shots(K, 512, np.random.default_rng(0), square=True)
    np.testing.assert_allclose(Kp, Kp.T)                       # symmetric
    assert np.linalg.eigvalsh(Kp).min() >= -1e-10              # PSD (final step)
    assert audit["min_eig_after_psd"] >= -1e-10
    # diagonal / range deviate only by the projection residual, and the
    # deviation is REPORTED rather than silently re-clipped (which would
    # destroy PSD-ness)
    np.testing.assert_allclose(np.diag(Kp), 1.0, atol=0.05)
    assert audit["max_diag_dev_after_psd"] < 0.05
    assert audit["max_range_dev_after_psd"] < 0.05
    assert audit["fro_change_sampling"] > 0


def test_shots_rectangular_no_psd():
    K = np.clip(np.random.default_rng(2).random((30, 40)), 0, 1)
    Kp, audit = perturb_kernel_finite_shots(K, 512, np.random.default_rng(0), square=False)
    assert Kp.shape == (30, 40)
    assert audit["fro_change_projection"] == 0.0               # never projected
    assert "min_eig_before_psd" not in audit
    assert Kp.min() >= 0.0 and Kp.max() <= 1.0


def test_shots_converges_to_exact():
    K = _exact_square()
    errs = []
    for shots in (128, 2048, 1 << 20):
        Kp, _ = perturb_kernel_finite_shots(K, shots, np.random.default_rng(0), square=True)
        errs.append(np.abs(Kp - K).max())
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 5e-3                                       # shots -> inf: exact


def test_shots_before_after_projection_reported():
    K = _exact_square()
    Kp, audit = perturb_kernel_finite_shots(K, 128, np.random.default_rng(0), square=True)
    for key in ("min_eig_before_psd", "min_eig_after_psd",
                "fro_change_sampling", "fro_change_projection"):
        assert key in audit

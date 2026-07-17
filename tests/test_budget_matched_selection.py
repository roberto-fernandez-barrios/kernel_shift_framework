# tests/test_budget_matched_selection.py
"""Unit tests for the GATE 1 budget-matched selection machinery (spec section 4)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analysis.budget_matched_selection import (  # noqa: E402
    FamilyMatrices, expected_max_of_k, p1_for_subsets, parse_kernel,
    sample_kernel_blocked, sample_kernel_stratified, sample_uniform)

RNG = np.random.default_rng(7)


def make_fm(R=12, C=9, nan_frac=0.1, seed=0) -> FamilyMatrices:
    rng = np.random.default_rng(seed)
    X = rng.random((R, C))
    Y = rng.random((R, C))
    mask = rng.random((R, C)) < nan_frac
    X[mask] = np.nan
    Y[mask] = np.nan
    runs = pd.DataFrame({"setting": ["s%d" % (i // 3) for i in range(R)],
                         "qs": [i % 3 for i in range(R)], "seed": 0})
    runs["group"] = "g"
    cfgs = pd.DataFrame({"cfg": [f"k{i}" for i in range(C)],
                         "shape": [f"sh{i // 3}" for i in range(C)],
                         "scale": 1.0, "dim": 4})
    return FamilyMatrices(runs=runs, cfgs=cfgs, X_sel=X.astype(np.float32),
                          Y_ood=Y, Y_extra={})


# ---------------------------------------------------------------------------
def test_parse_cfg_grammars():
    assert parse_kernel("laplacian_med") == ("laplacian_med", 1.0)
    assert parse_kernel("laplacian_med_x0.3") == ("laplacian_med", 0.3)
    assert parse_kernel("rbf_gscale_x10") == ("rbf_gscale", 10.0)
    assert parse_kernel("zz_r2_full__as2") == ("zz_r2_full", 2.0)
    assert parse_kernel("zz_r1_full") == ("zz_r1_full", 1.0)
    assert parse_kernel("pauli_xz_r1_full__as0.5") == ("pauli_xz_r1_full", 0.5)


def test_p1_subset_matches_bruteforce():
    fm = make_fm()
    for _ in range(50):
        k = int(RNG.integers(2, 8))
        cols = RNG.choice(fm.X_sel.shape[1], size=k, replace=False)[None, :]
        got = p1_for_subsets(fm, cols)[:, 0]
        for r in range(fm.X_sel.shape[0]):
            xs = fm.X_sel[r, cols[0]]
            if np.isnan(xs).all():
                assert np.isnan(got[r])
                continue
            win = cols[0][np.nanargmax(np.where(np.isnan(xs), -np.inf, xs))]
            expect = fm.Y_ood[r, win]
            if np.isnan(expect):
                assert np.isnan(got[r])
            else:
                assert got[r] == pytest.approx(expect)


def test_full_pool_equals_per_run_argmax():
    """col_idx = all columns must reproduce plain P1 (the software-validation
    regression required by spec constraint 7)."""
    fm = make_fm(nan_frac=0.0)
    got = p1_for_subsets(fm, np.arange(fm.X_sel.shape[1])[None, :])[:, 0]
    expect = fm.Y_ood[np.arange(len(got)), fm.X_sel.argmax(axis=1)]
    np.testing.assert_allclose(got, expect, rtol=1e-6)


def test_all_nan_run_yields_nan():
    fm = make_fm()
    fm.X_sel[3, :] = np.nan
    got = p1_for_subsets(fm, np.arange(fm.X_sel.shape[1])[None, :])[:, 0]
    assert np.isnan(got[3])


# ---------------------------------------------------------------------------
def test_uniform_scheme_properties():
    idx = sample_uniform(np.random.default_rng(0), n_pool=115, k=60, B=200)
    assert idx.shape == (200, 60)
    for row in idx:
        assert len(np.unique(row)) == 60
        assert row.min() >= 0 and row.max() < 115


def test_stratified_scheme_counts():
    shapes = np.repeat([f"sh{i}" for i in range(23)], 5)   # 115 candidates
    idx = sample_kernel_stratified(np.random.default_rng(0), shapes, k=60, B=50)
    assert idx.shape == (50, 60)
    for row in idx:
        assert len(np.unique(row)) == 60
        counts = pd.Series(shapes[row]).value_counts()
        assert set(counts.unique()) <= {2, 3}
        assert counts.sum() == 60


def test_blocked_scheme_is_whole_blocks():
    shapes = np.repeat([f"sh{i}" for i in range(23)], 5)
    idx = sample_kernel_blocked(np.random.default_rng(0), shapes, k=60, B=50)
    assert idx.shape == (50, 60)
    for row in idx:
        picked = pd.Series(shapes[row]).value_counts()
        assert len(picked) == 12 and (picked == 5).all()


def test_blocked_requires_multiple():
    shapes = np.repeat([f"sh{i}" for i in range(23)], 5)
    with pytest.raises(AssertionError):
        sample_kernel_blocked(np.random.default_rng(0), shapes, k=58, B=5)


# ---------------------------------------------------------------------------
def test_expected_max_of_k_closed_form():
    vals = np.array([0.1, 0.2, 0.3, 0.4])
    # k = n -> deterministic max
    assert expected_max_of_k(vals, 4) == pytest.approx(0.4)
    # k = 1 -> plain mean
    assert expected_max_of_k(vals, 1) == pytest.approx(0.25)
    # k = 2: E[max] = sum over pairs of max / C(4,2) = (0.2+0.3+0.4+0.3+0.4+0.4)/6
    assert expected_max_of_k(vals, 2) == pytest.approx(2.0 / 6)
    # k > n undefined
    assert np.isnan(expected_max_of_k(vals, 5))


def test_expected_max_matches_monte_carlo():
    vals = np.random.default_rng(1).random(30)
    mc = np.mean([np.max(np.random.default_rng(i).choice(vals, 7, replace=False))
                  for i in range(4000)])
    assert expected_max_of_k(vals, 7) == pytest.approx(mc, abs=5e-3)


def test_reproducibility_same_seed():
    shapes = np.repeat([f"sh{i}" for i in range(23)], 5)
    a = sample_kernel_blocked(np.random.default_rng(42), shapes, 60, 20)
    b = sample_kernel_blocked(np.random.default_rng(42), shapes, 60, 20)
    np.testing.assert_array_equal(a, b)

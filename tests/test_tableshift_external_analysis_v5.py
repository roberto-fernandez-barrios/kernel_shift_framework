"""Tests for controlled/oracle TableShift v5 estimands."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analysis.tableshift_external_v5 import (  # noqa: E402
    kernel_block_subsets,
    selected_ood,
)


def classical_pool():
    rows = []
    for kernel_idx in range(23):
        for dim in (4, 6, 8, 10, 12):
            rows.append(
                {
                    "family": "classical_ext",
                    "kernel": f"k{kernel_idx:02d}",
                    "dim": dim,
                    "validation": kernel_idx + dim / 100,
                    "ood_test": kernel_idx / 100,
                }
            )
    return pd.DataFrame(rows)


def test_equal_budget_is_twelve_whole_kernel_blocks():
    pool = classical_pool()
    subsets = kernel_block_subsets(pool, n_resamples=200, seed=3)
    ordered = pool.sort_values(["kernel", "dim"]).reset_index(drop=True)
    assert subsets.shape == (200, 60)
    for subset in subsets:
        counts = ordered.iloc[subset].kernel.value_counts()
        assert len(counts) == 12
        assert (counts == 5).all()


def test_selected_ood_uses_separate_selection_vector():
    selection = np.array([0.9, 0.1, 0.2])
    ood = np.array([0.0, 0.8, 0.7])
    subsets = np.array([[0, 1], [1, 2]])
    got = selected_ood(selection, ood, subsets)
    np.testing.assert_allclose(got, [0.0, 0.7])
    # An oracle deliberately selects with OOD itself.
    oracle = selected_ood(ood, ood, subsets)
    np.testing.assert_allclose(oracle, [0.8, 0.8])


def test_pool_gate_rejects_missing_kernel_block():
    pool = classical_pool().iloc[:-1]
    with pytest.raises(ValueError):
        kernel_block_subsets(pool)

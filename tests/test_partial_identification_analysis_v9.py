"""Tests for the v0.9 label-free analysis orchestration."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.partial_identification_v9 import (
    N_FRONTIER_PERMUTATIONS,
    block_frontier_draws,
    stable_rng,
)


def _arrays() -> dict[str, np.ndarray]:
    quantum = np.array([0, 1, 0, 1], dtype=np.int8)
    kernels = np.repeat([f"k{i:02d}" for i in range(23)], 5)
    classical = np.tile(quantum[:, None], (1, 115))
    # Make later blocks progressively worse while preserving five dims/block.
    for block in range(23):
        if block % 4:
            classical[block % 4, block * 5 : (block + 1) * 5] ^= 1
    return {
        "quantum_prediction": quantum,
        "classical_predictions": classical,
        "classical_kernels": kernels,
    }


def test_frontier_draws_are_nested_complete_and_deterministic(monkeypatch):
    monkeypatch.setattr(
        "scripts.analysis.partial_identification_v9.N_FRONTIER_PERMUTATIONS", 7
    )
    metadata = {"run": "frozen-run", "group": "g"}
    first = block_frontier_draws(_arrays(), metadata, "case", "svc")
    second = block_frontier_draws(_arrays(), metadata, "case", "svc")
    assert first.equals(second)
    assert len(first) == 7 * 3
    assert set(first.budget) == {30, 60, 115}
    wide = first.pivot(index="replicate", columns="budget", values="accuracy_upper")
    assert (wide[60] <= wide[30]).all()
    assert (wide[115] <= wide[60]).all()


def test_stable_rng_separates_tokens():
    a = stable_rng("a").integers(0, 2**32, size=20)
    a_again = stable_rng("a").integers(0, 2**32, size=20)
    b = stable_rng("b").integers(0, 2**32, size=20)
    np.testing.assert_array_equal(a, a_again)
    assert not np.array_equal(a, b)


def test_frozen_frontier_uses_5000_permutations():
    assert N_FRONTIER_PERMUTATIONS == 5_000

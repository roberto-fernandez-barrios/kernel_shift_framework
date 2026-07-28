"""Tests for label-blind, nested TableShift v5 sampling."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data.export_tableshift_v5 import (  # noqa: E402
    feature_role,
    hashed_order,
    row_digest,
    validate_source_splits,
)


def test_hash_order_is_deterministic_and_seed_specific():
    positions = np.arange(3000)
    first = hashed_order("task", "train", 42, positions)
    second = hashed_order("task", "train", 42, positions)
    other = hashed_order("task", "train", 123, positions)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_q1000_is_prefix_of_q2000():
    positions = np.arange(3000)
    order = hashed_order("task", "train", 42, positions)
    q1000 = positions[order[:1000]]
    q2000 = positions[order[:2000]]
    np.testing.assert_array_equal(q1000, q2000[:1000])


def test_digest_contract():
    assert row_digest("t", "train", 42, 7) == (
        "d0447279c28f31fb2c7381c285cbe013261869d9d75e9b5e07a964bb7bb3a1aa"
    )


def test_split_overlap_gate():
    validate_source_splits(
        {
            "train": np.array([0, 1]),
            "validation": np.array([2]),
            "id_test": np.array([3]),
            "ood_test": np.array([4]),
        }
    )
    with pytest.raises(RuntimeError):
        validate_source_splits(
            {
                "train": np.array([0, 1]),
                "validation": np.array([1, 2]),
                "id_test": np.array([3]),
                "ood_test": np.array([4]),
            }
        )


def test_feature_roles_preserve_categorical_codes():
    assert feature_role(np.dtype("float64")) == "numeric"
    assert feature_role(np.dtype("bool")) == "categorical"
    assert feature_role(np.dtype("O")) == "categorical"

"""Unit tests for train-only TableShift v5 representation and kernels."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.tableshift.run_external_validation_v5 import (  # noqa: E402
    fidelity_blocks,
    prepare_representations,
)


def synthetic_frames():
    rng = np.random.default_rng(4)
    frames = {}
    for split, n in {"train": 60, "validation": 20, "id_test": 20, "ood_test": 30}.items():
        frame = pd.DataFrame(
            {
                **{f"x{i}": rng.normal(size=n) for i in range(8)},
                "cat_a": rng.choice(["a", "b", None], size=n),
                "cat_b": rng.choice(["c", "d", "e"], size=n),
                "constant": 1,
            }
        )
        frames[split] = frame
    # A target-only extreme must not change fitted train ranges.
    frames["ood_test"].loc[:, "x0"] = 10_000
    frames["ood_test"].loc[:, "cat_b"] = "unseen"
    schema = pd.DataFrame(
        {
            "task": "toy",
            "column": list(frames["train"].columns),
            "dtype": ["float64"] * 8 + ["object", "object", "int64"],
            "role": ["numeric"] * 8 + ["categorical", "categorical", "numeric"],
        }
    )
    return frames, schema


def test_train_only_representation_and_constant_removal():
    frames, schema = synthetic_frames()
    representations, audit = prepare_representations(frames, schema, dims=(4,))
    assert audit["fit_split"] == "train"
    assert audit["target_used"] is False
    assert audit["n_dropped_constant_features"] == 1
    train = representations[4]["train"]
    assert train.shape == (60, 4)
    assert np.isclose(train.min(axis=0), 0.0).all()
    assert np.isclose(train.max(axis=0), np.pi).all()
    # Applying train-fitted MinMax to a shifted target split may leave [0, pi].
    assert representations[4]["ood_test"].max() > np.pi


def test_cpu_fidelity_blocks_match_direct_definition():
    rng = np.random.default_rng(2)
    train = rng.normal(size=(8, 16)) + 1j * rng.normal(size=(8, 16))
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    test = rng.normal(size=(5, 16)) + 1j * rng.normal(size=(5, 16))
    test /= np.linalg.norm(test, axis=1, keepdims=True)
    got = fidelity_blocks(
        {"train": train.astype(np.complex64), "ood_test": test.astype(np.complex64)},
        "cpu",
    )
    expected = np.abs(test.conj() @ train.T) ** 2
    np.testing.assert_allclose(got["ood_test"], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(got["train"], got["train"].T, atol=1e-12)

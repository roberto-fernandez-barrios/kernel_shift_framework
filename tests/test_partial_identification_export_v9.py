"""Focused tests for the v0.9 per-example prediction exporter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.experiments.export_partial_identification_v9 import (
    customary_kernel,
    fit_predict,
    select_p1_candidate,
)


def test_p1_candidate_uses_stable_lexicographic_tie_break():
    summary = pd.DataFrame(
        [
            {"family": "quantum", "model": "svc", "split": "id_val", "cfg": "z", "balanced_accuracy": 0.8},
            {"family": "quantum", "model": "svc", "split": "id_val", "cfg": "a", "balanced_accuracy": 0.8},
            {"family": "quantum", "model": "svc", "split": "ood_test", "cfg": "a", "balanced_accuracy": 1.0},
        ]
    )
    assert select_p1_candidate(summary, "quantum", "svc").cfg == "a"


def test_customary_reference_is_exactly_linear_and_rbf_variants():
    assert customary_kernel("linear")
    assert customary_kernel("rbf_gscale")
    assert customary_kernel("rbf_gscale_x0.1")
    assert not customary_kernel("laplacian_med")
    assert not customary_kernel("poly2")


def test_svc_fit_predict_returns_hard_predictions_and_scores():
    x_train = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    x_target = np.array([[-1.5], [1.5]])
    train_gram = x_train @ x_train.T
    target_gram = x_target @ x_train.T
    prediction, score = fit_predict(
        "svc",
        train_gram,
        target_gram,
        np.array([0, 0, 1, 1]),
        1.0,
    )
    np.testing.assert_array_equal(prediction, [0, 1])
    assert score.shape == (2,)
    assert set(np.unique(prediction)) <= {0, 1}


def test_svc_fit_predict_requires_frozen_regularization():
    gram = np.eye(4)
    with pytest.raises(ValueError, match="c_selected"):
        fit_predict("svc", gram, gram, np.array([0, 0, 1, 1]), None)

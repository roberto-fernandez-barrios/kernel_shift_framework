"""Tests for the v0.9 finite-shot prediction extension."""
from __future__ import annotations

import numpy as np

from scripts.experiments.export_shot_predictions_v9 import fit_shot_svc


def test_fit_shot_svc_returns_frozen_grid_c_and_target_predictions():
    rng = np.random.default_rng(5)
    x_train = rng.normal(size=(60, 3))
    y_train = (x_train[:, 0] > 0).astype(int)
    x_target = rng.normal(size=(12, 3))
    train = x_train @ x_train.T
    target = x_target @ x_train.T
    c_selected, prediction, score = fit_shot_svc(train, target, y_train)
    assert c_selected in {0.01, 0.1, 1.0, 10.0, 100.0}
    assert prediction.shape == (12,)
    assert score.shape == (12,)
    assert set(np.unique(prediction)) <= {0, 1}

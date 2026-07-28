"""Regression tests for the frozen v5 specification-curve aggregation."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analysis.specification_curve_v5 import (  # noqa: E402
    aggregate_setting_deltas,
    dataset_equal_summary,
    dataset_for_group,
)


def test_dataset_mapping():
    assert dataset_for_group("ember_m1") == "ember"
    assert dataset_for_group("unsw_dos_natural_cur") == "unsw"
    assert dataset_for_group("toniot_scanning_m2_centroid") == "toniot"
    with pytest.raises(ValueError):
        dataset_for_group("unknown_group")


def test_setting_then_group_aggregation_avoids_run_weighting():
    rows = pd.DataFrame(
        {
            "group": ["ember_m1"] * 4,
            "setting": ["a", "a", "a", "b"],
            "model": ["svc"] * 4,
            "delta": [0.0, 0.0, 0.0, 1.0],
        }
    )
    got = aggregate_setting_deltas(rows)
    assert got.loc[0, "delta"] == pytest.approx(0.5)


def test_dataset_equal_weighting():
    rows = pd.DataFrame(
        {
            "spec_id": ["S1"] * 4,
            "spec_label": ["x"] * 4,
            "model": ["svc"] * 4,
            "group": ["ember_m1", "unsw_a", "unsw_b", "toniot_a"],
            "delta": [0.0, 0.3, 0.9, 0.0],
        }
    )
    summary, per_dataset = dataset_equal_summary(rows)
    # UNSW contributes its within-dataset mean (0.6), not twice the weight.
    assert summary.loc[0, "dataset_equal_delta"] == pytest.approx(0.2)
    assert summary.loc[0, "n_datasets"] == 3
    assert set(per_dataset.dataset) == {"ember", "unsw", "toniot"}

"""Regression tests for the v0.6.0 reviewer-driven corrections."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analysis.budget_scheme_sensitivity_v6 import summarize_schemes  # noqa: E402
from scripts.analysis.mechanism_robustness_v4 import (  # noqa: E402
    match_one_to_one,
    match_with_replacement,
    summarize_rank_matches,
)
from scripts.experiments.run_shots_mc_v6 import (  # noqa: E402
    select_p1_winner,
    stable_measurement_seed,
)
from scripts.reproduce_v6 import STAGES as REPRODUCE_V6_STAGES  # noqa: E402
from src.analysis.source_datasets import source_dataset_for_group  # noqa: E402


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("ember_m1", "ember"),
        ("ember_m2", "ember"),
        ("unsw_dos_natural_cur", "unsw"),
        ("unsw_dos_m2_centroid", "unsw"),
        ("unsw_recon_natural_cur", "unsw"),
        ("unsw_recon_m2_centroid", "unsw"),
        ("toniot_scanning_natural_cur", "toniot"),
        ("toniot_scanning_m2_centroid", "toniot"),
    ],
)
def test_canonical_source_dataset_mapping(group, expected):
    assert source_dataset_for_group(group) == expected


def test_unknown_source_dataset_fails_loudly():
    with pytest.raises(ValueError, match="Unrecognized"):
        source_dataset_for_group("mystery_case")


def test_budget_scheme_summary_averages_groups_within_source():
    rows = []
    groups = [
        "ember_m1",
        "ember_m2",
        "unsw_dos_natural_cur",
        "unsw_recon_natural_cur",
        "toniot_scanning_natural_cur",
    ]
    values = {
        "ember_m1": 0.1,
        "ember_m2": 0.3,
        "unsw_dos_natural_cur": 0.4,
        "unsw_recon_natural_cur": 0.8,
        "toniot_scanning_natural_cur": 0.7,
    }
    for model in ("svc", "gpc"):
        for scheme in ("kernel_blocked", "uniform", "kernel_stratified"):
            offset = {"kernel_blocked": 0.0, "uniform": 0.01,
                      "kernel_stratified": -0.01}[scheme]
            for group in groups:
                rows.append(
                    {
                        "group": group,
                        "model": model,
                        "scheme": scheme,
                        "budget": 60,
                        "delta_median": values[group] + offset,
                        "select_col": "id_val",
                    }
                )
    by_dataset, overall, variation = summarize_schemes(pd.DataFrame(rows))
    # source means: EMBER=.2, UNSW=.6, ToN-IoT=.7; equal-source mean=.5
    primary = overall[
        (overall.model == "svc") & (overall.scheme == "kernel_blocked")
    ].iloc[0]
    assert primary.dataset_equal_delta == pytest.approx(0.5)
    assert primary.n_datasets == 3
    assert set(by_dataset.dataset) == {"ember", "unsw", "toniot"}
    assert variation.scheme_range.max() == pytest.approx(0.02)


def _matching_frame() -> pd.DataFrame:
    rows = []
    for run in ("run_a", "run_b"):
        for family, ranks, scores in (
            ("quantum", [1.0, 2.0], [0.70, 0.72]),
            ("classical_ext", [1.1, 2.1, 3.0], [0.71, 0.74, 0.73]),
        ):
            for i, (rank, score) in enumerate(zip(ranks, scores)):
                rows.append(
                    {
                        "run": run,
                        "group": "ember_m1",
                        "dim": 4,
                        "family": family,
                        "kernel": f"{family}_{i}",
                        "spec_train_eff_rank": rank,
                        "balanced_accuracy": score,
                    }
                )
    return pd.DataFrame(rows)


def test_rank_matching_discloses_replacement_and_one_to_one_uniqueness():
    frame = _matching_frame()
    nearest = match_with_replacement(frame)
    one_to_one = match_one_to_one(frame)
    assert set(nearest.match_method) == {"nearest_with_replacement"}
    assert set(one_to_one.match_method) == {"one_to_one"}
    assert not one_to_one.duplicated(["run", "dim", "q_kernel"]).any()
    assert not one_to_one.duplicated(["run", "dim", "c_kernel"]).any()
    assert (nearest.rank_ratio >= 1).all()

    summary = summarize_rank_matches(nearest, one_to_one)
    assert set(summary.caliper) == {"1.10", "1.25", "1.50", "2.00", "unfiltered"}
    assert set(summary.match_method) == {"nearest_with_replacement", "one_to_one"}
    assert (summary.retained_fraction <= 1).all()


def test_finite_shot_seed_is_stable_and_block_specific():
    args = ("run", "kernel", 12, 512, 7)
    seed = stable_measurement_seed(*args, "train")
    assert seed == stable_measurement_seed(*args, "train")
    assert seed != stable_measurement_seed(*args, "ood_square")
    assert 0 <= seed < 2**64


def test_p1_tie_break_matches_lexicographic_candidate_order():
    candidates = pd.DataFrame(
        {
            "cfg": ["zz__svc__d10", "pauli__svc__d12", "lower__svc__d4"],
            "balanced_accuracy": [0.9, 0.9, 0.8],
        }
    )
    assert select_p1_winner(candidates).cfg == "pauli__svc__d12"


def test_corrected_confirmatory_endpoint_equals_specification_s10():
    root = Path(__file__).resolve().parents[1] / "results/v6"
    hierarchical = pd.read_csv(root / "inference/hierarchical_effects.csv")
    primary = hierarchical[
        (hierarchical.variant == "budget60")
        & (hierarchical.scope == "dataset_equal_mean")
    ].set_index("model")
    specification = pd.read_csv(
        root / "specification_curve/specification_summary.csv"
    )
    s10 = specification[specification.spec_id == "S10"].set_index("model")
    for model in ("svc", "gpc"):
        assert primary.loc[model, "effect"] == pytest.approx(
            s10.loc[model, "dataset_equal_delta"], abs=2e-15
        )
        assert primary.loc[model, "n_independent_clusters"] == 3
        assert s10.loc[model, "n_datasets"] == 3


def test_v6_reproduction_uses_confirmatory_budget_inputs():
    primary_command = REPRODUCE_V6_STAGES["analysis"][0]
    assert "results/v4/budget_confirmatory" in primary_command
    assert "results/v4/budget" not in primary_command

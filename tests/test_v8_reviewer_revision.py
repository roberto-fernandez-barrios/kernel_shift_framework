"""Tests for the frozen v0.8 reviewer-revision analyses."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analysis.circuit_resources_v8 import (  # noqa: E402
    circuit_table,
    shot_resource_table,
)
from scripts.analysis.reviewer_revision_v8 import (  # noqa: E402
    dataset_equal_summary,
    factorial_axis_contrasts,
    factorial_pairwise_interactions,
    stable_rng,
)
from scripts.experiments.run_shots_mc_v6 import FIXED_RUNS  # noqa: E402
from src.experiments.ember.extended.run_classical_extensions import (  # noqa: E402
    v8_shortcut_feature_mask,
    v8_svc_eval_rows,
)


def _fidelity(feature_map, x, xp) -> float:
    parameters = sorted(feature_map.parameters, key=lambda parameter: parameter.name)
    left = Statevector.from_instruction(
        feature_map.assign_parameters(dict(zip(parameters, x)))
    )
    right = Statevector.from_instruction(
        feature_map.assign_parameters(dict(zip(parameters, xp)))
    )
    return float(abs(left.inner(right)) ** 2)


@pytest.mark.parametrize("kind", ["pauli_xz", "zmap"])
def test_local_feature_map_fidelity_factorizes(kind):
    x = np.array([0.21, 0.73, 1.19])
    xp = np.array([0.42, 0.51, 1.41])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        if kind == "pauli_xz":
            full = PauliFeatureMap(
                3,
                reps=1,
                paulis=["X", "Z"],
                entanglement="full",
            )
            one = lambda: PauliFeatureMap(
                1,
                reps=1,
                paulis=["X", "Z"],
                entanglement="full",
            )
        else:
            full = ZFeatureMap(3, reps=2)
            one = lambda: ZFeatureMap(1, reps=2)
        product = np.prod(
            [_fidelity(one(), [x[j]], [xp[j]]) for j in range(3)]
        )
        assert _fidelity(full, x, xp) == pytest.approx(product, abs=1e-12)


def test_circuit_audit_detects_only_zz_entanglement():
    table = circuit_table()
    four = table[table.dimension_qubits == 4].set_index("feature_map")
    assert four.loc["zz_r1_full", "feature_map_cx_gates"] == 12
    assert four.loc["zz_r2_full", "feature_map_cx_gates"] == 24
    assert four.loc["pauli_xz_r1_full", "feature_map_cx_gates"] == 0
    assert four.loc["zmap_r2", "feature_map_cx_gates"] == 0
    assert bool(four.loc["pauli_xz_r1_full", "product_factorized"])
    assert not bool(four.loc["zz_r1_full", "product_factorized"])


def test_finite_shot_resource_count_matches_sampled_blocks():
    table = shot_resource_table()
    first = table.iloc[0]
    assert first.distinct_fidelity_estimates_per_case_replicate == 1_624_250
    assert first.id_holdout_to_train == 500_000
    assert "id_validation_to_train" not in table.columns
    assert "id_test_to_train" not in table.columns
    assert first.total_shots_per_case_replicate == 1_624_250 * 128
    entangling_cases = sum(run.kernel.startswith("zz_") for run in FIXED_RUNS)
    assert first.n_fixed_cases == len(FIXED_RUNS) == 8
    assert first.n_entangling_fixed_cases == entangling_cases == 4
    assert first.n_product_fixed_cases == len(FIXED_RUNS) - entangling_cases
    assert (
        first.projected_shots_entangling_cases_only
        == first.projected_shots_full_sensitivity / 2
    )
    assert bool(first.product_factorization_can_bypass_circuit_sampling)
    assert first.n_measurement_replicates == 30


def test_stable_rng_is_label_deterministic_and_separated():
    a = stable_rng("a").integers(0, 2**31, size=20)
    a_again = stable_rng("a").integers(0, 2**31, size=20)
    b = stable_rng("b").integers(0, 2**31, size=20)
    np.testing.assert_array_equal(a, a_again)
    assert not np.array_equal(a, b)


def _write_inventory(directory: Path, names: list[str]) -> None:
    directory.mkdir()
    (directory / "meta_export.json").write_text(
        json.dumps({"feature_names": names}),
        encoding="utf-8",
    )


def test_shortcut_masks_follow_frozen_feature_rules(tmp_path):
    unsw = tmp_path / "unsw_dos"
    unsw_names = [
        "dur",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "is_sm_ips_ports",
    ]
    _write_inventory(unsw, unsw_names)
    keep, _, removed = v8_shortcut_feature_mask(unsw, len(unsw_names))
    assert removed == unsw_names[1:]
    np.testing.assert_array_equal(keep, [True, False, False, False])

    toniot = tmp_path / "toniot_scanning"
    toniot_names = [
        "duration",
        "src_port",
        "dst_port",
        "proto_tcp",
        "service",
        "service_http",
        "conn_state_SF",
    ]
    _write_inventory(toniot, toniot_names)
    keep, _, removed = v8_shortcut_feature_mask(toniot, len(toniot_names))
    assert removed == toniot_names[1:6]
    np.testing.assert_array_equal(
        keep,
        [True, False, False, False, False, False, True],
    )


def test_v8_svc_rows_distinguish_fixed_and_train_cv():
    rng = np.random.default_rng(11)
    x_train = rng.normal(size=(60, 3))
    x_eval = rng.normal(size=(30, 3))
    y_train = (x_train[:, 0] > 0).astype(int)
    y_eval = (x_eval[:, 0] > 0).astype(int)
    gram_train = x_train @ x_train.T
    gram_eval = x_eval @ x_train.T
    blocks = {
        "train": gram_train,
        "id_val": gram_eval[:10],
        "id_test": gram_eval[10:20],
        "ood_test": gram_eval[20:],
    }
    labels = {
        "id_val": y_eval[:10],
        "id_test": y_eval[10:20],
        "ood_test": y_eval[20:],
    }
    rows = v8_svc_eval_rows(
        "linear",
        "classical_ext",
        3,
        blocks,
        labels,
        y_train,
        ("fixed_c1", "train_cv"),
    )
    assert len(rows) == 6
    assert {row["regularization"] for row in rows} == {
        "fixed_c1",
        "train_cv",
    }
    fixed = [row for row in rows if row["regularization"] == "fixed_c1"]
    assert {row["c_selected"] for row in fixed} == {1.0}


def test_factorial_contrasts_are_paired_over_other_axes():
    rows = []
    for regularization in ("fixed_c1", "train_cv"):
        for selection in ("ood_test", "id_val"):
            for reference in ("customary", "extended"):
                for budget_mode in ("native", "equal_count"):
                    effect = (
                        (1 if regularization == "train_cv" else 0)
                        + (2 if selection == "id_val" else 0)
                        + (4 if reference == "extended" else 0)
                        + (8 if budget_mode == "equal_count" else 0)
                    )
                    rows.append({
                        "regularization": regularization,
                        "selection": selection,
                        "reference": reference,
                        "budget_mode": budget_mode,
                        "dataset_equal_effect": effect,
                    })
    contrasts = factorial_axis_contrasts(pd.DataFrame(rows))
    means = contrasts[
        contrasts.contrast_scope == "mean_over_other_factorial_axes"
    ].set_index("axis")
    assert means.loc["regularization", "paired_change"] == pytest.approx(1)
    assert means.loc["selection", "paired_change"] == pytest.approx(2)
    assert means.loc["reference", "paired_change"] == pytest.approx(4)
    assert means.loc["budget_mode", "paired_change"] == pytest.approx(8)


def test_factorial_pairwise_interaction_recovers_known_term():
    rows = []
    for regularization in ("fixed_c1", "train_cv"):
        for selection in ("ood_test", "id_val"):
            for reference in ("customary", "extended"):
                for budget_mode in ("native", "equal_count"):
                    regularized = regularization == "train_cv"
                    id_selected = selection == "id_val"
                    effect = (
                        int(regularized)
                        + 2 * int(id_selected)
                        + 4 * int(reference == "extended")
                        + 8 * int(budget_mode == "equal_count")
                        + 3 * int(regularized and id_selected)
                    )
                    rows.append({
                        "regularization": regularization,
                        "selection": selection,
                        "reference": reference,
                        "budget_mode": budget_mode,
                        "dataset_equal_effect": effect,
                    })
    interactions = factorial_pairwise_interactions(pd.DataFrame(rows))
    means = interactions[
        interactions.interaction_scope
        == "mean_over_remaining_factorial_axes"
    ]
    by_pair = {
        frozenset((row.axis_a, row.axis_b)): row.difference_in_differences
        for row in means.itertuples()
    }
    assert by_pair[frozenset(("regularization", "selection"))] == pytest.approx(3)
    for pair, value in by_pair.items():
        if pair != frozenset(("regularization", "selection")):
            assert value == pytest.approx(0)


def test_dataset_equal_summary_does_not_weight_groups_equally():
    groups = pd.DataFrame(
        {
            "group": [
                "unsw_dos_natural_cur",
                "unsw_dos_m2_centroid",
                "unsw_recon_natural_cur",
                "unsw_recon_m2_centroid",
                "toniot_scanning_natural_cur",
                "toniot_scanning_m2_centroid",
            ],
            "effect": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        }
    )
    summary = dataset_equal_summary(groups, [])
    assert summary.loc[0, "dataset_equal_effect"] == pytest.approx(0.5)
    assert summary.loc[0, "n_source_datasets"] == 2

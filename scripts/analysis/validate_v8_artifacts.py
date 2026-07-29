"""Fail-fast integrity gates for the v0.8 reviewer-revision artifacts."""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(r".+__q1000_id500_ood500__qs\d+__s\d+$")
EXPECTED_GROUPS = {
    "ember_m1",
    "ember_m2",
    "unsw_dos_natural_cur",
    "unsw_dos_m2_centroid",
    "unsw_recon_natural_cur",
    "unsw_recon_m2_centroid",
    "toniot_scanning_natural_cur",
    "toniot_scanning_m2_centroid",
}
EXPECTED_NETWORK_GROUPS = EXPECTED_GROUPS - {"ember_m1", "ember_m2"}


def discover_outputs(
    roots: tuple[Path, ...],
    filename: str,
) -> list[Path]:
    return sorted(
        path
        for root in roots
        for path in root.glob(f"*/{filename}")
        if RUN_RE.fullmatch(path.parent.name)
    )


def validate_campaign_outputs(roots: tuple[Path, ...]) -> None:
    fixed = discover_outputs(roots, "summary_v8_fixedc.csv")
    shortcut = discover_outputs(roots, "summary_v8_shortcut.csv")
    if len(fixed) != 360:
        raise ValueError(
            f"fixed-C campaign incomplete: expected 360 summaries, found {len(fixed)}"
        )
    if len(shortcut) != 270:
        raise ValueError(
            "shortcut campaign incomplete: "
            f"expected 270 summaries, found {len(shortcut)}"
        )

    for path in fixed:
        frame = pd.read_csv(path)
        if len(frame) != 525:
            raise ValueError(f"{path}: expected 525 rows, found {len(frame)}")
        if set(frame.regularization) != {"fixed_c1"}:
            raise ValueError(f"{path}: fixed-C regularization label mismatch")
        if set(frame.split) != {"id_val", "id_test", "ood_test"}:
            raise ValueError(f"{path}: split coverage mismatch")
        if set(frame.id_split_hash_salt) != {"ksf-v4-idsplit::"}:
            raise ValueError(f"{path}: ID split hash salt mismatch")
        v4_path = path.with_name("summary_v4.csv")
        if not v4_path.is_file():
            raise FileNotFoundError(v4_path)
        v4 = pd.read_csv(
            v4_path,
            usecols=["family", "model", "cfg", "kernel", "split"],
        )
        v4 = v4[v4.model == "svc"]
        key_columns = ["family", "model", "cfg", "kernel", "split"]
        fixed_keys = frame[key_columns].sort_values(
            key_columns,
            kind="stable",
        ).reset_index(drop=True)
        v4_keys = v4[key_columns].sort_values(
            key_columns,
            kind="stable",
        ).reset_index(drop=True)
        if not fixed_keys.equals(v4_keys):
            raise ValueError(
                f"{path}: fixed-C candidate/split inventory differs from v4"
            )
        audit = json.loads(path.with_name("audit_v8fixed.json").read_text())
        if (
            audit["mode"] != "v8fixed"
            or audit["feature_policy"] != "full_v4_features"
            or audit["n_removed_features"] != 0
        ):
            raise ValueError(f"{path}: fixed-C audit mismatch")
        v4_audit_path = path.with_name("idsplit_audit_v4.csv")
        if not v4_audit_path.is_file():
            raise FileNotFoundError(v4_audit_path)
        v4_audit = pd.read_csv(v4_audit_path).iloc[0]
        for field in (
            "n_id",
            "n_val",
            "n_test",
            "pos_rate_val",
            "pos_rate_test",
            "overlap",
            "class_balance_gap",
        ):
            if not np.isclose(float(audit[field]), float(v4_audit[field])):
                raise ValueError(
                    f"{path}: fixed-C ID-split audit differs on {field}"
                )

    for path in shortcut:
        frame = pd.read_csv(path)
        if len(frame) != 1050:
            raise ValueError(f"{path}: expected 1,050 rows, found {len(frame)}")
        if set(frame.regularization) != {"fixed_c1", "train_cv"}:
            raise ValueError(f"{path}: shortcut regularization coverage mismatch")
        audit = json.loads(path.with_name("audit_v8shortcut.json").read_text())
        if audit["mode"] != "v8shortcut":
            raise ValueError(f"{path}: shortcut audit mode mismatch")
        input_count = int(audit["n_input_features"])
        removed_count = int(audit["n_removed_features"])
        if input_count == 39 and removed_count != 3:
            raise ValueError(f"{path}: UNSW removal count must be 3")
        if input_count == 88 and removed_count != 14:
            raise ValueError(f"{path}: ToN-IoT removal count must be 14")
        if input_count not in {39, 88}:
            raise ValueError(f"{path}: unexpected network feature inventory")


def _read(root: Path, filename: str) -> pd.DataFrame:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path}: empty artifact")
    return frame


def _require_finite(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: non-finite reported value")


def validate_analysis_outputs(root: Path) -> None:
    strata = _read(root, "quantum_strata_summary.csv")
    if len(strata) != 3 or set(strata.stratum) != {
        "all_quantum_maps",
        "entangling_zz",
        "separable_product",
    }:
        raise ValueError("quantum-stratum summary coverage mismatch")
    if dict(zip(strata.stratum, strata.budget)) != {
        "all_quantum_maps": 60,
        "entangling_zz": 30,
        "separable_product": 30,
    }:
        raise ValueError("quantum-stratum budgets mismatch")
    if not (strata.n_source_datasets == 3).all():
        raise ValueError("quantum-stratum source-dataset coverage mismatch")

    strata_groups = _read(root, "quantum_strata_group_effects.csv")
    if len(strata_groups) != 24 or set(strata_groups.group) != EXPECTED_GROUPS:
        raise ValueError("quantum-stratum group coverage mismatch")
    if strata_groups.simultaneous.astype(bool).any():
        raise ValueError("pointwise quantum-stratum intervals marked simultaneous")
    strata_runs = _read(root, "quantum_strata_run_effects.csv")
    strata_clusters = _read(root, "quantum_strata_cluster_effects.csv")
    if len(strata_runs) != 3 * 1080 or len(strata_clusters) != 3 * 8 * 5:
        raise ValueError("quantum-stratum run/cluster coverage mismatch")

    winners = _read(root, "quantum_winner_composition.csv")
    if len(winners) != 16:
        raise ValueError("quantum winner-composition coverage mismatch")
    totals = winners.groupby(["model", "selector"]).n_winners.sum()
    if not (totals == 1080).all():
        raise ValueError("quantum winner counts must sum to 1,080 per endpoint")
    product_p1 = (
        winners[
            (winners.selector == "id_val")
            & (winners.map_stratum == "separable_product")
        ]
        .groupby("model")
        .n_winners.sum()
        .to_dict()
    )
    if product_p1 != {"gpc": 348, "svc": 594}:
        raise ValueError("frozen P1' product-map winner counts changed")

    factorial = _read(root, "factorial_summary.csv")
    if len(factorial) != 16:
        raise ValueError("factorial must contain 16 endpoints")
    for column, levels in {
        "regularization": {"fixed_c1", "train_cv"},
        "selection": {"ood_test", "id_val"},
        "reference": {"customary", "extended"},
        "budget_mode": {"native", "equal_count"},
    }.items():
        if set(factorial[column]) != levels:
            raise ValueError(f"factorial axis {column} is incomplete")
    if not (factorial.n_source_datasets == 3).all():
        raise ValueError("factorial source-dataset coverage mismatch")

    factorial_groups = _read(root, "factorial_group_effects.csv")
    if len(factorial_groups) != 128:
        raise ValueError("factorial group table must contain 16 x 8 rows")
    if set(factorial_groups.group) != EXPECTED_GROUPS:
        raise ValueError("factorial fixed-case coverage mismatch")
    factorial_runs = _read(root, "factorial_run_effects.csv")
    factorial_clusters = _read(root, "factorial_cluster_effects.csv")
    if len(factorial_runs) != 16 * 360 or len(factorial_clusters) != 16 * 8 * 5:
        raise ValueError("factorial run/cluster coverage mismatch")

    contrasts = _read(root, "factorial_axis_contrasts.csv")
    means = contrasts[
        contrasts.contrast_scope == "mean_over_other_factorial_axes"
    ]
    if len(contrasts) != 36 or len(means) != 4:
        raise ValueError("factorial contrast coverage mismatch")
    if set(means.axis) != {
        "regularization",
        "selection",
        "reference",
        "budget_mode",
    }:
        raise ValueError("factorial mean axis contrasts are incomplete")

    interactions = _read(root, "factorial_pairwise_interactions.csv")
    interaction_means = interactions[
        interactions.interaction_scope
        == "mean_over_remaining_factorial_axes"
    ]
    if len(interactions) != 30 or len(interaction_means) != 6:
        raise ValueError("factorial pairwise-interaction coverage mismatch")
    observed_pairs = {
        frozenset((row.axis_a, row.axis_b))
        for row in interaction_means.itertuples()
    }
    expected_pairs = {
        frozenset(pair)
        for pair in itertools.combinations(
            (
                "regularization",
                "selection",
                "reference",
                "budget_mode",
            ),
            2,
        )
    }
    if observed_pairs != expected_pairs:
        raise ValueError("factorial pairwise interactions are incomplete")

    shortcut = _read(root, "shortcut_ablation_group_effects.csv")
    if len(shortcut) != 6 or set(shortcut.group) != EXPECTED_NETWORK_GROUPS:
        raise ValueError("shortcut-ablation group coverage mismatch")
    shortcut_datasets = _read(root, "shortcut_ablation_dataset_effects.csv")
    if (
        len(shortcut_datasets) != 2
        or set(shortcut_datasets.dataset) != {"unsw", "toniot"}
        or int(shortcut_datasets.n_groups.sum()) != 6
    ):
        raise ValueError("shortcut-ablation source-dataset coverage mismatch")
    shortcut_summary = _read(root, "shortcut_ablation_summary.csv")
    if (
        len(shortcut_summary) != 1
        or shortcut_summary.iloc[0].aggregation != "source_dataset_equal"
        or int(shortcut_summary.iloc[0].n_source_datasets) != 2
    ):
        raise ValueError("shortcut-ablation summary mismatch")
    shortcut_row = shortcut_summary.iloc[0]
    for column in (
        "original_effect",
        "ablated_effect",
        "ablation_change",
    ):
        if not np.isclose(
            shortcut_row[column],
            shortcut_datasets[column].mean(),
            atol=1e-12,
        ):
            raise ValueError(
                f"shortcut-ablation source-equal {column} mismatch"
            )
    if not np.isclose(
        shortcut_row.ablated_effect - shortcut_row.original_effect,
        shortcut_row.ablation_change,
        atol=1e-12,
    ):
        raise ValueError("shortcut-ablation paired change is inconsistent")
    shortcut_runs = _read(root, "shortcut_ablation_run_effects.csv")
    shortcut_clusters = _read(root, "shortcut_ablation_cluster_effects.csv")
    shortcut_changes = _read(root, "shortcut_ablation_change_clusters.csv")
    if (
        len(shortcut_runs) != 270
        or len(shortcut_clusters) != 2 * 6 * 5
        or len(shortcut_changes) != 6 * 5
    ):
        raise ValueError("shortcut-ablation run/cluster coverage mismatch")

    clusters = _read(root, "primary_cluster_values.csv")
    if len(clusters) != 80:
        raise ValueError("primary cluster table must contain 2 x 8 x 5 rows")
    if set(clusters.model) != {"svc", "gpc"}:
        raise ValueError("primary cluster classifier coverage mismatch")
    if set(clusters.group) != EXPECTED_GROUPS:
        raise ValueError("primary cluster group coverage mismatch")
    if not (clusters.groupby(["model", "group"]).size() == 5).all():
        raise ValueError("each primary interval must expose five cluster values")

    _require_finite(
        strata,
        ["dataset_equal_effect", "min_dataset_effect", "max_dataset_effect"],
        "quantum strata",
    )
    _require_finite(
        factorial,
        ["dataset_equal_effect", "min_dataset_effect", "max_dataset_effect"],
        "factorial",
    )
    _require_finite(
        shortcut,
        [
            "effect",
            "ci_lo",
            "ci_hi",
            "original_effect",
            "original_ci_lo",
            "original_ci_hi",
            "ablated_effect",
            "ablated_ci_lo",
            "ablated_ci_hi",
            "ablation_change",
        ],
        "shortcut ablation",
    )
    _require_finite(
        shortcut_summary,
        [
            "original_effect",
            "ablated_effect",
            "ablation_change",
            "min_dataset_change",
            "max_dataset_change",
        ],
        "shortcut-ablation source-dataset summary",
    )


def validate_resource_outputs(root: Path) -> None:
    circuits = _read(root, "circuit_resources.csv")
    if len(circuits) != 20:
        raise ValueError("circuit table must contain 4 maps x 5 dimensions")
    product = circuits[circuits.map_stratum == "separable_product"]
    entangling = circuits[circuits.map_stratum == "entangling_zz"]
    if len(product) != 10 or len(entangling) != 10:
        raise ValueError("circuit strata coverage mismatch")
    if (product.feature_map_cx_gates != 0).any():
        raise ValueError("separable product maps contain unexpected CX gates")
    if (entangling.feature_map_cx_gates <= 0).any():
        raise ValueError("entangling ZZ maps lack CX gates")
    if circuits.device_routing.astype(bool).any():
        raise ValueError("logical circuit audit must not claim device routing")

    shots = _read(root, "finite_shot_resources.csv")
    levels = shots[shots.shots_per_fidelity != "all_levels"].copy()
    if set(levels.shots_per_fidelity.astype(int)) != {128, 512, 2048, 8192}:
        raise ValueError("finite-shot resource levels mismatch")
    if not (
        levels.distinct_fidelity_estimates_per_case_replicate == 1_624_250
    ).all():
        raise ValueError("finite-shot fidelity count mismatch")
    if levels.hardware_executed.astype(bool).any():
        raise ValueError("resource sensitivity must not claim hardware execution")
    if (
        not (levels.n_entangling_fixed_cases == 4).all()
        or not (levels.n_product_fixed_cases == 4).all()
    ):
        raise ValueError("finite-shot circuit strata must contain four cases each")
    if not levels.product_factorization_can_bypass_circuit_sampling.astype(
        bool
    ).all():
        raise ValueError("product-map classical bypass is not documented")
    if not np.allclose(
        levels.projected_shots_entangling_cases_only,
        levels.projected_shots_full_sensitivity / 2,
    ):
        raise ValueError("entangling-only shot projection mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/v8/reviewer_revision"),
    )
    parser.add_argument(
        "--result-roots",
        type=Path,
        nargs="+",
        default=[
            Path("results/ember_shift/extended_kernels"),
            Path("results/netflow/extended_kernels"),
        ],
    )
    args = parser.parse_args()
    validate_campaign_outputs(tuple(args.result_roots))
    print("[ok] complete fixed-C and shortcut campaigns")
    validate_analysis_outputs(args.root)
    print("[ok] quantum-stratum, factorial, shortcut, and cluster outputs")
    validate_resource_outputs(args.root)
    print("[ok] circuit and finite-shot resource outputs")
    print("[ok] all v0.8.0 reviewer-revision artifact gates passed")


if __name__ == "__main__":
    main()

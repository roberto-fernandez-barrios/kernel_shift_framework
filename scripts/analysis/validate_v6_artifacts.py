"""Fail-fast release gates for the reviewer-corrected v0.6.0 artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.shots_mc_v6 import validate as validate_shots
from scripts.experiments.run_shots_mc_v6 import (
    FIXED_RUNS,
    locate_run,
    matches_frozen_text_sha256,
)


def validate_corrected_estimand(root: Path) -> None:
    hierarchical = pd.read_csv(root / "inference/hierarchical_effects.csv")
    primary = hierarchical[
        (hierarchical.variant == "budget60")
        & (hierarchical.scope == "dataset_equal_mean")
    ].set_index("model")
    specification = pd.read_csv(
        root / "specification_curve/specification_summary.csv"
    )
    s10 = specification[specification.spec_id == "S10"].set_index("model")
    if set(primary.index) != {"svc", "gpc"} or set(s10.index) != {"svc", "gpc"}:
        raise ValueError("S10 endpoint coverage is incomplete")
    np.testing.assert_allclose(
        primary.loc[["svc", "gpc"], "effect"],
        s10.loc[["svc", "gpc"], "dataset_equal_delta"],
        rtol=0,
        atol=2e-15,
    )
    if not (primary.n_independent_clusters == 3).all():
        raise ValueError("dataset-equal endpoint must contain three source datasets")
    if not (s10.n_datasets == 3).all():
        raise ValueError("specification S10 must contain three source datasets")


def validate_budget_and_rank(root: Path) -> None:
    budget = pd.read_csv(root / "budget/scheme_summary.csv")
    if set(budget.scheme) != {"kernel_blocked", "uniform", "kernel_stratified"}:
        raise ValueError("budget scheme sensitivity is incomplete")
    if set(budget.model) != {"svc", "gpc"} or not (budget.n_datasets == 3).all():
        raise ValueError("budget source-dataset coverage is incomplete")
    primary = budget[budget.scheme == "kernel_blocked"]
    if not primary.is_primary.all():
        raise ValueError("kernel_blocked must be marked primary")

    rank = pd.read_csv(root / "mechanism/rank_matching_sensitivity.csv")
    if set(rank.match_method) != {"nearest_with_replacement", "one_to_one"}:
        raise ValueError("rank-matching method sensitivity is incomplete")
    if set(rank.caliper.astype(str)) != {
        "1.10", "1.25", "1.50", "2.00", "unfiltered"
    }:
        raise ValueError("rank-matching caliper sensitivity is incomplete")
    if not rank.retained_fraction.between(0, 1).all():
        raise ValueError("invalid rank-matching retained fraction")

    pairs = pd.read_csv(root / "mechanism/rank_matched_one_to_one_pairs.csv")
    if pairs.duplicated(["run", "dim", "q_kernel"]).any():
        raise ValueError("one-to-one matching reuses a quantum candidate")
    if pairs.duplicated(["run", "dim", "c_kernel"]).any():
        raise ValueError("one-to-one matching reuses a classical candidate")


def validate_finite_shots(root: Path, result_roots: tuple[Path, ...]) -> None:
    run_dir = root / "shots_mc/runs"
    if list(run_dir.glob("*.partial.csv")):
        raise ValueError("incomplete finite-shot checkpoints remain")
    files = sorted(run_dir.glob("*.csv"))
    if len(files) != len(FIXED_RUNS):
        raise ValueError("finite-shot fixed-run coverage is incomplete")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    validate_shots(frame)
    seed_cells = frame[
        ["run", "shots", "replicate", "stable_seed_train"]
    ].drop_duplicates()
    if len(seed_cells) != 8 * 4 * 30:
        raise ValueError("unexpected finite-shot seed-cell count")
    if seed_cells.stable_seed_train.nunique() != len(seed_cells):
        raise ValueError("finite-shot train seeds collide")

    for fixed in FIXED_RUNS:
        result_dir = locate_run(fixed.run, result_roots)
        summary_path = result_dir / "summary_v4.csv"
        if not matches_frozen_text_sha256(summary_path, fixed.summary_sha256):
            raise ValueError(f"summary hash changed for {fixed.run}")
        summary = pd.read_csv(summary_path)
        exact = summary[
            (summary.family == "quantum")
            & (summary.model == "svc")
            & (summary.kernel == fixed.kernel)
            & (summary["dim"] == fixed.dim)
            & summary.split.isin(["id_val", "id_test", "ood_test"])
        ].set_index("split")
        run_rows = frame[frame.run == fixed.run]
        for split, output_column in (
            ("id_val", "exact_id_val_bacc"),
            ("id_test", "exact_id_test_bacc"),
            ("ood_test", "exact_ood_test_bacc"),
        ):
            observed = run_rows[output_column].unique()
            if len(observed) != 1:
                raise ValueError(f"non-constant exact endpoint for {fixed.run}/{split}")
            np.testing.assert_allclose(
                observed[0], exact.loc[split, "balanced_accuracy"], rtol=0, atol=1e-10
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/v6"))
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
    validate_corrected_estimand(args.root)
    print("[ok] corrected three-source S10 estimand")
    validate_budget_and_rank(args.root)
    print("[ok] budget and rank-matching sensitivities")
    validate_finite_shots(args.root, tuple(args.result_roots))
    print("[ok] repeated finite-shot artifacts and exact references")
    print("[ok] all v0.6.0 analysis artifact gates passed")


if __name__ == "__main__":
    main()

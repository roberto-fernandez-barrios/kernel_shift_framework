"""Aggregate the frozen repeated finite-shot sensitivity for v0.7.0."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.run_shots_mc_v6 import (
    FIXED_RUNS,
    N_REPLICATES,
    PROJECTION_CONDITIONS,
    SHOTS,
)


METRICS = (
    "selected_c",
    "id_val_bacc",
    "id_test_bacc",
    "ood_test_bacc",
    "ood_difference_from_exact",
    "absolute_ood_difference",
    "train_effective_rank_ratio",
    "kta_ood_difference",
    "train_min_eig_before_psd",
    "train_frac_negative_eig",
    "train_fro_change_sampling",
    "train_fro_change_projection",
    "ood_fro_change_sampling",
    "ood_fro_change_projection",
    "condition_train_fro_change_from_sampled",
    "condition_id_val_fro_change_from_sampled",
    "condition_id_test_fro_change_from_sampled",
    "condition_ood_test_fro_change_from_sampled",
    "condition_ood_square_fro_change_from_sampled",
)


def validate(frame: pd.DataFrame) -> None:
    expected_rows = (
        len(FIXED_RUNS) * len(SHOTS) * N_REPLICATES * len(PROJECTION_CONDITIONS)
    )
    if len(frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(frame)}")
    key = ["run", "shots", "replicate", "projection_condition"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate Monte Carlo cells")
    if frame.run.nunique() != len(FIXED_RUNS):
        raise ValueError("incomplete fixed-run coverage")
    if set(frame.shots) != set(SHOTS):
        raise ValueError("incomplete shot-count coverage")
    if set(frame.projection_condition) != set(PROJECTION_CONDITIONS):
        raise ValueError("incomplete projection-condition coverage")
    counts = frame.groupby(["run", "shots", "projection_condition"]).size()
    if not (counts == N_REPLICATES).all():
        raise ValueError("every conditional cell must have 30 measurement replicates")


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, group in frame.groupby(
        ["run", "group", "dataset", "shots", "projection_condition"],
        observed=True,
    ):
        row = dict(
            zip(
                ["run", "group", "dataset", "shots", "projection_condition"],
                keys,
            )
        )
        row["n_measurement_replicates"] = len(group)
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            row[f"{metric}_q975"] = float(np.quantile(values, 0.975))
        rows.append(row)
    by_group = pd.DataFrame(rows)

    across_rows = []
    for keys, group in by_group.groupby(
        ["shots", "projection_condition"], observed=True
    ):
        shots, condition = keys
        row = {
            "shots": shots,
            "projection_condition": condition,
            "n_fixed_scenario_groups": group.group.nunique(),
        }
        for metric in (
            "ood_difference_from_exact_median",
            "absolute_ood_difference_median",
            "train_effective_rank_ratio_median",
            "kta_ood_difference_median",
            "condition_train_fro_change_from_sampled_median",
            "condition_ood_square_fro_change_from_sampled_median",
        ):
            values = group[metric].to_numpy(float)
            row[f"{metric}_across_group_median"] = float(np.median(values))
            row[f"{metric}_across_group_min"] = float(values.min())
            row[f"{metric}_across_group_max"] = float(values.max())
        across_rows.append(row)
    return by_group, pd.DataFrame(across_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path, default=Path("results/v6/shots_mc/runs")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/v6/shots_mc")
    )
    args = parser.parse_args()
    files = sorted(args.run_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"no completed run CSVs found in {args.run_dir}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    validate(frame)
    by_group, across = summarize(frame)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_group.to_csv(args.out_dir / "monte_carlo_by_group.csv", index=False)
    across.to_csv(args.out_dir / "descriptive_across_groups.csv", index=False)
    print("=== Repeated finite-shot sensitivity across fixed scenario-groups ===")
    display = across[
        [
            "shots",
            "projection_condition",
            "ood_difference_from_exact_median_across_group_median",
            "ood_difference_from_exact_median_across_group_min",
            "ood_difference_from_exact_median_across_group_max",
            "absolute_ood_difference_median_across_group_median",
            "train_effective_rank_ratio_median_across_group_median",
        ]
    ]
    print(display.round(6).to_string(index=False))


if __name__ == "__main__":
    main()

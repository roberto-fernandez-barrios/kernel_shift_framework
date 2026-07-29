"""Summarize the frozen equal-budget sampling-scheme sensitivity.

The confirmatory analysis selects the best classical candidate from a
60-candidate subsample. ``kernel_blocked`` is primary because it mirrors the
quantum pool's 12 kernel shapes crossed with all five dimensions. ``uniform``
and ``kernel_stratified`` are descriptive sensitivity schemes.

Aggregation is source-dataset equal: scenario-groups are averaged within
EMBER, UNSW-NB15, and ToN-IoT before the three source datasets receive equal
weight.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.source_datasets import source_dataset_for_group


SCHEMES = ("kernel_blocked", "uniform", "kernel_stratified")
MODELS = ("svc", "gpc")


def summarize_schemes(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return source-dataset, overall, and per-group sensitivity summaries."""
    required = {"group", "model", "scheme", "budget", "delta_median", "select_col"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"budget table lacks columns: {sorted(missing)}")
    if set(frame.scheme) != set(SCHEMES):
        raise ValueError(
            f"expected schemes {SCHEMES}, found {tuple(sorted(frame.scheme.unique()))}"
        )
    if set(frame.model) != set(MODELS):
        raise ValueError(
            f"expected models {MODELS}, found {tuple(sorted(frame.model.unique()))}"
        )
    if frame.duplicated(["group", "model", "scheme"]).any():
        raise ValueError("duplicated group/model/scheme rows")
    if frame.budget.nunique() != 1 or int(frame.budget.iloc[0]) != 60:
        raise ValueError("the frozen confirmatory budget must be 60")
    if set(frame.select_col) != {"id_val"}:
        raise ValueError("the frozen confirmatory endpoint must select on id_val")

    expected_groups = set(frame.group)
    for model in MODELS:
        for scheme in SCHEMES:
            observed = set(frame[(frame.model == model) & (frame.scheme == scheme)].group)
            if observed != expected_groups:
                raise ValueError(f"incomplete {model}/{scheme} group coverage")

    work = frame.copy()
    work["dataset"] = work.group.map(source_dataset_for_group)
    by_dataset = (
        work.groupby(["model", "scheme", "dataset"], as_index=False, observed=True)
        .agg(
            delta=("delta_median", "mean"),
            n_groups=("group", "nunique"),
        )
    )
    overall = (
        by_dataset.groupby(["model", "scheme"], as_index=False, observed=True)
        .agg(
            dataset_equal_delta=("delta", "mean"),
            min_dataset_delta=("delta", "min"),
            max_dataset_delta=("delta", "max"),
            n_datasets=("dataset", "nunique"),
        )
    )
    overall["is_primary"] = overall.scheme.eq("kernel_blocked")

    primary = (
        work[work.scheme == "kernel_blocked"]
        .set_index(["group", "model"])
        .delta_median
    )
    variation = (
        work.groupby(["group", "model"], as_index=False, observed=True)
        .agg(
            min_delta=("delta_median", "min"),
            max_delta=("delta_median", "max"),
        )
    )
    variation["scheme_range"] = variation.max_delta - variation.min_delta
    variation["primary_delta"] = [
        primary.loc[(group, model)]
        for group, model in variation[["group", "model"]].itertuples(index=False)
    ]
    alternatives = work[work.scheme != "kernel_blocked"].copy()
    alternatives["abs_difference_from_primary"] = [
        abs(row.delta_median - primary.loc[(row.group, row.model)])
        for row in alternatives.itertuples()
    ]
    max_abs = (
        alternatives.groupby(["group", "model"], observed=True)
        .abs_difference_from_primary.max()
    )
    variation["max_abs_difference_from_primary"] = [
        max_abs.loc[(group, model)]
        for group, model in variation[["group", "model"]].itertuples(index=False)
    ]
    return by_dataset, overall, variation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/v4/budget_confirmatory/resamples_by_group.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/v6/budget"))
    args = parser.parse_args()

    by_dataset, overall, variation = summarize_schemes(pd.read_csv(args.input))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_dataset.to_csv(args.out_dir / "scheme_by_dataset.csv", index=False)
    overall.to_csv(args.out_dir / "scheme_summary.csv", index=False)
    variation.to_csv(args.out_dir / "scheme_variation_by_group.csv", index=False)

    print("=== Dataset-equal budget-scheme sensitivity ===")
    print(overall.round(6).to_string(index=False))
    print("\n=== Largest group-level scheme range ===")
    print(
        variation.sort_values("scheme_range", ascending=False)
        .head(10)
        .round(6)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

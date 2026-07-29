"""Build the frozen v5 S1--S10 specification curve.

This script only aggregates versioned legacy/v4 summaries.  It does not
recompute kernels or alter the v4 experiment grid.  The specification order is
fixed in docs/EXTERNAL_VALIDATION_SPEC.md and is never sorted by effect size.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.source_datasets import source_dataset_for_group


SPECIFICATIONS = (
    ("S1", "Legacy oracle; C=1; linear/RBF; native"),
    ("S2", "Legacy ID-selected; C=1; linear/RBF; native"),
    ("S3", "Legacy oracle; C=1; extended; native"),
    ("S4", "Legacy ID-selected; C=1; extended; native"),
    ("S5", "v4 oracle; train-CV; linear/RBF; native"),
    ("S6", "v4 ID-validation; train-CV; linear/RBF; native"),
    ("S7", "v4 oracle; train-CV; extended; native"),
    ("S8", "v4 ID-validation; train-CV; extended; native"),
    ("S9", "v4 oracle; train-CV; extended; equal budget"),
    ("S10", "v4 ID-validation; train-CV; extended; equal budget"),
)
SPEC_LABELS = dict(SPECIFICATIONS)
MODELS = ("svc", "gpc")


def dataset_for_group(group: str) -> str:
    """Map the eight scenario groups to the three source datasets."""
    return source_dataset_for_group(group)


def _validate_complete(
    frame: pd.DataFrame,
    keys: list[str],
    expected: set[tuple],
    source: str,
) -> None:
    observed = set(map(tuple, frame[keys].drop_duplicates().itertuples(index=False, name=None)))
    missing = expected - observed
    if missing:
        preview = ", ".join(map(str, sorted(missing)[:5]))
        raise ValueError(f"{source} is incomplete; missing {len(missing)} cells: {preview}")


def aggregate_setting_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    """Average runs within setting, then settings within scenario group."""
    required = {"group", "setting", "model", "delta"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"run-level table lacks columns: {sorted(missing)}")
    per_setting = (
        rows.groupby(["group", "setting", "model"], as_index=False, observed=True)
        .delta.mean()
    )
    return (
        per_setting.groupby(["group", "model"], as_index=False, observed=True)
        .delta.mean()
    )


def dataset_equal_summary(by_group: pd.DataFrame) -> pd.DataFrame:
    """Average groups within dataset, then give each dataset equal weight."""
    frame = by_group.copy()
    frame["dataset"] = frame.group.map(dataset_for_group)
    per_dataset = (
        frame.groupby(["spec_id", "spec_label", "model", "dataset"],
                      as_index=False, observed=True)
        .delta.mean()
    )
    summary = (
        per_dataset.groupby(["spec_id", "spec_label", "model"],
                            as_index=False, observed=True)
        .delta.mean()
        .rename(columns={"delta": "dataset_equal_delta"})
    )
    summary["n_datasets"] = per_dataset.groupby(
        ["spec_id", "spec_label", "model"], observed=True
    ).size().to_numpy()
    summary["min_dataset_delta"] = per_dataset.groupby(
        ["spec_id", "spec_label", "model"], observed=True
    ).delta.min().to_numpy()
    summary["max_dataset_delta"] = per_dataset.groupby(
        ["spec_id", "spec_label", "model"], observed=True
    ).delta.max().to_numpy()
    return summary, per_dataset


def load_legacy(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    legacy = pd.concat(frames, ignore_index=True)
    required = {
        "setting", "group", "model", "fam", "p1_ood_mean", "p3_ood_mean",
    }
    missing = required - set(legacy)
    if missing:
        raise ValueError(f"legacy summaries lack columns: {sorted(missing)}")
    if legacy.duplicated(["setting", "model", "fam"]).any():
        raise ValueError("legacy summaries contain duplicated setting/model/family cells")

    expected = {
        (setting, model, family)
        for setting in legacy.setting.unique()
        for model in MODELS
        for family in ("quantum", "classical_orig", "classical_ext")
    }
    _validate_complete(
        legacy, ["setting", "model", "fam"], expected, "legacy summaries"
    )
    wide = legacy.pivot(
        index=["setting", "group", "model"],
        columns="fam",
        values=["p1_ood_mean", "p3_ood_mean"],
    )
    output = []
    definitions = (
        ("S1", "p3_ood_mean", "classical_orig"),
        ("S2", "p1_ood_mean", "classical_orig"),
        ("S3", "p3_ood_mean", "classical_ext"),
        ("S4", "p1_ood_mean", "classical_ext"),
    )
    for spec_id, metric, comparator in definitions:
        delta = wide[(metric, "quantum")] - wide[(metric, comparator)]
        tmp = delta.rename("delta").reset_index()
        by_group = aggregate_setting_deltas(tmp)
        by_group["spec_id"] = spec_id
        output.append(by_group)
    return pd.concat(output, ignore_index=True)


def load_v4_native(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    required = {
        "group", "model",
        "p1_ood_delta_vs_classical_orig",
        "oracle_ood_delta_vs_classical_orig",
        "p1_ood_delta_vs_classical_ext",
        "oracle_ood_delta_vs_classical_ext",
    }
    missing = required - set(source)
    if missing:
        raise ValueError(f"v4 group summary lacks columns: {sorted(missing)}")
    if source.duplicated(["group", "model"]).any():
        raise ValueError("v4 group summary contains duplicated group/model cells")

    definitions = (
        ("S5", "oracle_ood_delta_vs_classical_orig"),
        ("S6", "p1_ood_delta_vs_classical_orig"),
        ("S7", "oracle_ood_delta_vs_classical_ext"),
        ("S8", "p1_ood_delta_vs_classical_ext"),
    )
    output = []
    for spec_id, column in definitions:
        tmp = source[["group", "model", column]].rename(columns={column: "delta"})
        tmp["spec_id"] = spec_id
        output.append(tmp)
    return pd.concat(output, ignore_index=True)


def load_equal_budget(path: Path, spec_id: str) -> pd.DataFrame:
    source = pd.read_csv(path)
    by_group = aggregate_setting_deltas(source)
    by_group["spec_id"] = spec_id
    return by_group


def validate_final(by_group: pd.DataFrame) -> pd.DataFrame:
    result = by_group.copy()
    result["spec_label"] = result.spec_id.map(SPEC_LABELS)
    if result.spec_label.isna().any():
        unknown = sorted(result.loc[result.spec_label.isna(), "spec_id"].unique())
        raise ValueError(f"Unknown specification IDs: {unknown}")
    groups = sorted(result.group.unique())
    expected = {
        (spec_id, model, group)
        for spec_id, _ in SPECIFICATIONS
        for model in MODELS
        for group in groups
    }
    _validate_complete(
        result, ["spec_id", "model", "group"], expected, "specification curve"
    )
    if result.duplicated(["spec_id", "model", "group"]).any():
        raise ValueError("specification curve contains duplicated cells")
    if len(groups) != 8:
        raise ValueError(f"expected eight scenario groups, found {len(groups)}")
    return result


def plot_curve(summary: pd.DataFrame, by_group: pd.DataFrame, output: Path) -> None:
    """Plot group-level points and the dataset-equal mean in frozen order."""
    order = [spec_id for spec_id, _ in SPECIFICATIONS]
    # The figure is placed at manuscript text width.  Keep the native canvas
    # close to that width so labels remain at least approximately 8 pt after
    # typesetting rather than being down-scaled from a presentation-size plot.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), sharey=True)
    colors = {"svc": "#1f5a7a", "gpc": "#9a4f20"}
    for ax, model in zip(axes, MODELS):
        group_part = by_group[by_group.model == model]
        summary_part = summary[summary.model == model].set_index("spec_id").loc[order]
        for y, spec_id in enumerate(order):
            vals = group_part.loc[group_part.spec_id == spec_id, "delta"]
            ax.scatter(vals, np.full(len(vals), y), s=16, color="#a6adb4",
                       alpha=0.72, linewidths=0, zorder=2)
        ax.scatter(
            summary_part.dataset_equal_delta,
            np.arange(len(order)),
            marker="D",
            s=43,
            color=colors[model],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--", zorder=1)
        ax.grid(axis="x", color="#e4e6e8", linewidth=0.7)
        ax.set_title(model.upper(), fontsize=9)
        ax.set_xlabel(
            r"$\Delta_{\mathrm{OOD}}$ (quantum $-$ classical)", fontsize=8
        )
        ax.tick_params(axis="x", labelsize=7.5)
    axes[0].invert_yaxis()
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels(
        [f"{sid}  {SPEC_LABELS[sid]}" for sid in order], fontsize=7.5
    )
    fig.suptitle("Specification curve: protocol choices reverse the apparent effect",
                 fontsize=9.5)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def build_outputs(
    legacy_paths: list[Path],
    v4_native_path: Path,
    oracle_equal_path: Path,
    honest_equal_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_group = pd.concat(
        [
            load_legacy(legacy_paths),
            load_v4_native(v4_native_path),
            load_equal_budget(oracle_equal_path, "S9"),
            load_equal_budget(honest_equal_path, "S10"),
        ],
        ignore_index=True,
    )
    by_group = validate_final(by_group)
    summary, by_dataset = dataset_equal_summary(by_group)
    order = {spec_id: i for i, (spec_id, _) in enumerate(SPECIFICATIONS)}
    for frame in (by_group, by_dataset, summary):
        frame["_spec_order"] = frame.spec_id.map(order)
        frame.sort_values(["_spec_order", "model"], inplace=True)
        frame.drop(columns="_spec_order", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return by_group, by_dataset, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy",
        type=Path,
        nargs="+",
        default=[
            Path("results/honest_selection/ember_main__by_setting.csv"),
            Path("results/honest_selection/netflow_main__by_setting.csv"),
        ],
    )
    parser.add_argument(
        "--v4-native",
        type=Path,
        default=Path("results/v4/family_comparison/group_summary.csv"),
    )
    parser.add_argument(
        "--oracle-equal",
        type=Path,
        default=Path("results/v5/specification_curve/oracle_budget/"
                     "p1_runs__budget60.csv"),
    )
    parser.add_argument(
        "--honest-equal",
        type=Path,
        default=Path("results/v4/budget_confirmatory/p1_runs__budget60.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/v5/specification_curve"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("manuscript/fig_v5_specification.pdf"),
    )
    args = parser.parse_args()

    by_group, by_dataset, summary = build_outputs(
        args.legacy, args.v4_native, args.oracle_equal, args.honest_equal
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_group.to_csv(args.out_dir / "specification_by_group.csv", index=False)
    by_dataset.to_csv(args.out_dir / "specification_by_dataset.csv", index=False)
    summary.to_csv(args.out_dir / "specification_summary.csv", index=False)
    plot_curve(summary, by_group, args.figure)
    print(summary[["spec_id", "model", "dataset_equal_delta"]].to_string(index=False))
    print(f"[OK] wrote {args.out_dir} and {args.figure}")


if __name__ == "__main__":
    main()

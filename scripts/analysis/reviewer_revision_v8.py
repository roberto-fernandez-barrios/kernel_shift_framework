"""Reviewer-motivated v0.8 analyses frozen in docs/REVIEWER_REVISION_SPEC_V8.md.

The script produces:

* entangling-ZZ versus separable-product quantum-family sensitivities;
* the within-v4 2x2x2x2 SVC factorial;
* the q1000 port/protocol/service-field ablation sensitivity;
* winner-composition counts and the five cluster values behind each primary
  security interval.

The v0.4 primary endpoint is not replaced. These are fixed-case,
reviewer-motivated sensitivities. Resampling distributions quantify candidate
budget sensitivity and are never confidence intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.budget_matched_selection import (  # noqa: E402
    annotate,
    build_matrices,
    load_combined,
    p1_for_subsets,
    sample_kernel_blocked,
    wide_metrics,
)
from scripts.analysis.hierarchical_effect_estimation import (  # noqa: E402
    cluster_t_ci,
    nested_cluster_means,
    parse_setting,
)
from scripts.analysis.honest_selection_analysis import (  # noqa: E402
    group_label,
)
from src.analysis.source_datasets import source_dataset_for_group  # noqa: E402


ROOT_SEED = 20260729
N_RESAMPLES = 5000
ENTANGLING = {"zz_r1_full", "zz_r2_full"}
SEPARABLE = {"pauli_xz_r1_full", "zmap_r2"}
ORIGINAL_PREFIXES = ("linear", "rbf_gscale")
RUN_RE = re.compile(r"(?P<setting>.+)__qs(?P<qs>\d+)__s(?P<seed>\d+)$")


def stable_rng(label: str, root_seed: int = ROOT_SEED) -> np.random.Generator:
    digest = hashlib.sha256(f"ksf-v8::{root_seed}::{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def base_quantum_map(kernel: str) -> str:
    return kernel.split("__as", 1)[0]


def load_v4(roots: list[tuple[str, Path]]) -> pd.DataFrame:
    return load_combined(roots, [], base_file="summary_v4.csv")


def load_v8(
    roots: list[tuple[str, Path]],
    filename: str,
    regularization: str | None = None,
) -> pd.DataFrame:
    frames = []
    for tag, root in roots:
        for directory in sorted(root.iterdir()):
            match = RUN_RE.match(directory.name)
            path = directory / filename
            if not match or not path.exists():
                continue
            frame = pd.read_csv(path)
            if regularization is not None:
                frame = frame[frame.regularization == regularization]
            frame = frame[[
                "family",
                "model",
                "cfg",
                "kernel",
                "split",
                "balanced_accuracy",
            ]].copy()
            frame["setting"] = match.group("setting")
            frame["qs"] = int(match.group("qs"))
            frame["seed"] = int(match.group("seed"))
            frame["root"] = tag
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {filename} summaries found")
    output = pd.concat(frames, ignore_index=True)
    keys = ["setting", "qs", "seed", "model", "cfg", "split"]
    if output.duplicated(keys).any():
        raise ValueError(f"{filename} contains duplicate cells")
    return annotate(output)


def family_matrices(
    frame: pd.DataFrame,
    selector: str,
    model: str = "svc",
):
    wide = wide_metrics(frame, selector)
    metadata = frame[["cfg", "kernel", "shape", "scale", "dim"]].drop_duplicates(
        "cfg"
    )
    wide["kernel"] = wide.cfg.map(metadata.set_index("cfg").kernel)
    if wide.kernel.isna().any():
        raise ValueError("kernel metadata missing for one or more configurations")
    quantum = build_matrices(
        wide,
        model,
        wide.family == "quantum",
        metadata,
    )
    classical_extended = build_matrices(
        wide,
        model,
        wide.family == "classical_ext",
        metadata,
    )
    classical_original = build_matrices(
        wide,
        model,
        (wide.family == "classical_ext")
        & wide.kernel.str.startswith(ORIGINAL_PREFIXES),
        metadata,
    )
    return quantum, classical_original, classical_extended


def assert_aligned(*families) -> None:
    reference = families[0].runs[["setting", "qs", "seed", "group"]]
    for family in families[1:]:
        if not reference.equals(
            family.runs[["setting", "qs", "seed", "group"]]
        ):
            raise ValueError("run sets or ordering differ between families")


def full_endpoint(family) -> np.ndarray:
    columns = np.arange(family.X_sel.shape[1])[None, :]
    return p1_for_subsets(family, columns)[:, 0]


def blocked_expected_endpoint(
    family,
    budget: int,
    label: str,
    n_resamples: int = N_RESAMPLES,
) -> np.ndarray:
    kernels = family.cfgs.kernel.to_numpy()
    columns = sample_kernel_blocked(
        stable_rng(label),
        kernels,
        budget,
        n_resamples,
    )
    return np.nanmean(p1_for_subsets(family, columns), axis=1)


def add_design_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    master_size = output.setting.map(parse_setting)
    output["ms"] = [value[0] for value in master_size]
    output["size"] = [value[1] for value in master_size]
    output["dataset"] = output.group.map(source_dataset_for_group)
    return output


def aggregate_fixed_groups(
    run_effects: pd.DataFrame,
    id_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    cluster_rows = []
    for keys, group in run_effects.groupby(id_columns + ["group"], sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identifiers = dict(zip(id_columns + ["group"], keys))
        cluster_values = nested_cluster_means(group)
        qsplit_levels = sorted(group.qs.unique())
        if len(cluster_values) != 5 or len(qsplit_levels) != 5:
            raise ValueError(f"{identifiers}: expected five q-split clusters")
        effect, lo, hi = cluster_t_ci(cluster_values)
        group_rows.append({
            **identifiers,
            "effect": effect,
            "ci_lo": lo,
            "ci_hi": hi,
            "ci_method": "pointwise_conditional_cluster_t",
            "n_qsplit_clusters": 5,
            "simultaneous": False,
        })
        for qsplit, value in zip(qsplit_levels, cluster_values):
            cluster_rows.append({
                **identifiers,
                "qs": qsplit,
                "cluster_effect": value,
            })

    groups = pd.DataFrame(group_rows)
    clusters = pd.DataFrame(cluster_rows)
    return groups, clusters


def dataset_equal_summary(
    group_effects: pd.DataFrame,
    id_columns: list[str],
) -> pd.DataFrame:
    frame = group_effects.copy()
    frame["dataset"] = frame.group.map(source_dataset_for_group)
    per_dataset = (
        frame.groupby(id_columns + ["dataset"], as_index=False, observed=True)
        .effect.mean()
    )
    if not id_columns:
        return pd.DataFrame(
            [
                {
                    "dataset_equal_effect": float(per_dataset.effect.mean()),
                    "min_dataset_effect": float(per_dataset.effect.min()),
                    "max_dataset_effect": float(per_dataset.effect.max()),
                    "n_source_datasets": int(per_dataset.dataset.nunique()),
                }
            ]
        )
    return (
        per_dataset.groupby(id_columns, as_index=False, observed=True)
        .agg(
            dataset_equal_effect=("effect", "mean"),
            min_dataset_effect=("effect", "min"),
            max_dataset_effect=("effect", "max"),
            n_source_datasets=("dataset", "nunique"),
        )
    )


def make_entanglement_outputs(
    v4: pd.DataFrame,
    output_dir: Path,
) -> None:
    quantum, _, classical = family_matrices(v4, "id_val", model="svc")
    assert_aligned(quantum, classical)
    base_maps = quantum.cfgs.kernel.map(base_quantum_map)
    strata = {
        "all_quantum_maps": np.ones(len(base_maps), dtype=bool),
        "entangling_zz": base_maps.isin(ENTANGLING).to_numpy(),
        "separable_product": base_maps.isin(SEPARABLE).to_numpy(),
    }
    run_tables = []
    for stratum, mask in strata.items():
        columns = np.flatnonzero(mask)
        budget = len(columns)
        if budget not in (30, 60):
            raise ValueError(f"{stratum}: unexpected quantum budget {budget}")
        quantum_endpoint = p1_for_subsets(quantum, columns[None, :])[:, 0]
        classical_endpoint = blocked_expected_endpoint(
            classical,
            budget,
            f"entanglement::{stratum}::classical",
        )
        run = quantum.runs.copy()
        run["stratum"] = stratum
        run["budget"] = budget
        run["quantum_ood"] = quantum_endpoint
        run["classical_expected_ood"] = classical_endpoint
        run["delta"] = quantum_endpoint - classical_endpoint
        run_tables.append(run)
    runs = add_design_columns(pd.concat(run_tables, ignore_index=True))
    groups, clusters = aggregate_fixed_groups(runs, ["stratum", "budget"])
    summary = dataset_equal_summary(groups, ["stratum", "budget"])
    runs.to_csv(output_dir / "quantum_strata_run_effects.csv", index=False)
    groups.to_csv(output_dir / "quantum_strata_group_effects.csv", index=False)
    clusters.to_csv(
        output_dir / "quantum_strata_cluster_effects.csv", index=False
    )
    summary.to_csv(output_dir / "quantum_strata_summary.csv", index=False)


def make_winner_composition(v4: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    kernel_by_cfg = v4[["cfg", "kernel"]].drop_duplicates("cfg").set_index(
        "cfg"
    ).kernel
    for model in ("svc", "gpc"):
        for selector in ("id_val", "ood_test"):
            wide = wide_metrics(v4, selector)
            quantum = wide[
                (wide.model == model) & (wide.family == "quantum")
            ].copy()
            quantum["kernel"] = quantum.cfg.map(kernel_by_cfg)
            quantum["base_map"] = quantum.kernel.map(base_quantum_map)
            quantum = quantum.sort_values(
                ["setting", "qs", "seed", "cfg"],
                kind="stable",
            )
            chosen = quantum.loc[
                quantum.groupby(["setting", "qs", "seed"]).sel_metric.idxmax()
            ]
            counts = chosen.base_map.value_counts()
            for base_map, count in counts.items():
                rows.append({
                    "model": model,
                    "selector": selector,
                    "base_map": base_map,
                    "map_stratum": (
                        "entangling_zz"
                        if base_map in ENTANGLING
                        else "separable_product"
                    ),
                    "n_winners": int(count),
                    "n_runs": int(len(chosen)),
                    "winner_fraction": float(count / len(chosen)),
                })
    pd.DataFrame(rows).sort_values(
        ["model", "selector", "map_stratum", "base_map"]
    ).to_csv(output_dir / "quantum_winner_composition.csv", index=False)


def factorial_cell(
    frame: pd.DataFrame,
    regularization: str,
    selector: str,
    reference: str,
    budget_mode: str,
) -> pd.DataFrame:
    quantum, classical_original, classical_extended = family_matrices(
        frame,
        selector,
        model="svc",
    )
    classical = (
        classical_original if reference == "customary" else classical_extended
    )
    assert_aligned(quantum, classical)
    quantum_endpoint = full_endpoint(quantum)
    classical_endpoint = full_endpoint(classical)

    if budget_mode == "equal_count":
        if reference == "customary":
            quantum_endpoint = blocked_expected_endpoint(
                quantum,
                30,
                "factorial::paired_quantum30",
            )
        else:
            classical_endpoint = blocked_expected_endpoint(
                classical,
                60,
                "factorial::paired_classical60",
            )
    elif budget_mode != "native":
        raise ValueError(budget_mode)

    output = quantum.runs.copy()
    output["regularization"] = regularization
    output["selection"] = selector
    output["reference"] = reference
    output["budget_mode"] = budget_mode
    output["quantum_ood"] = quantum_endpoint
    output["classical_ood"] = classical_endpoint
    output["delta"] = quantum_endpoint - classical_endpoint
    return output


def factorial_axis_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    axes = {
        "regularization": ("fixed_c1", "train_cv"),
        "selection": ("ood_test", "id_val"),
        "reference": ("customary", "extended"),
        "budget_mode": ("native", "equal_count"),
    }
    rows = []
    for axis, (level_from, level_to) in axes.items():
        other = [column for column in axes if column != axis]
        wide = summary.pivot_table(
            index=other,
            columns=axis,
            values="dataset_equal_effect",
        ).reset_index()
        for _, row in wide.iterrows():
            rows.append({
                "axis": axis,
                "from_level": level_from,
                "to_level": level_to,
                **{column: row[column] for column in other},
                "paired_change": row[level_to] - row[level_from],
            })
    output = pd.DataFrame(rows)
    means = (
        output.groupby(["axis", "from_level", "to_level"], as_index=False)
        .paired_change.mean()
    )
    means["contrast_scope"] = "mean_over_other_factorial_axes"
    output["contrast_scope"] = "individual_paired_cell"
    return pd.concat([output, means], ignore_index=True, sort=False)


def factorial_pairwise_interactions(summary: pd.DataFrame) -> pd.DataFrame:
    axes = {
        "regularization": ("fixed_c1", "train_cv"),
        "selection": ("ood_test", "id_val"),
        "reference": ("customary", "extended"),
        "budget_mode": ("native", "equal_count"),
    }
    rows = []
    for axis_a, axis_b in itertools.combinations(axes, 2):
        a_from, a_to = axes[axis_a]
        b_from, b_to = axes[axis_b]
        other = [
            column for column in axes if column not in {axis_a, axis_b}
        ]
        wide = summary.pivot_table(
            index=other,
            columns=[axis_a, axis_b],
            values="dataset_equal_effect",
        ).reset_index()
        for _, row in wide.iterrows():
            interaction = (
                row[(a_to, b_to)]
                - row[(a_from, b_to)]
                - row[(a_to, b_from)]
                + row[(a_from, b_from)]
            )
            rows.append({
                "axis_a": axis_a,
                "axis_a_from": a_from,
                "axis_a_to": a_to,
                "axis_b": axis_b,
                "axis_b_from": b_from,
                "axis_b_to": b_to,
                **{column: row[column] for column in other},
                "difference_in_differences": interaction,
                "interaction_scope": "individual_paired_cell",
            })
    output = pd.DataFrame(rows)
    means = (
        output.groupby(
            [
                "axis_a",
                "axis_a_from",
                "axis_a_to",
                "axis_b",
                "axis_b_from",
                "axis_b_to",
            ],
            as_index=False,
        )
        .difference_in_differences.mean()
    )
    means["interaction_scope"] = "mean_over_remaining_factorial_axes"
    return pd.concat([output, means], ignore_index=True, sort=False)


def make_factorial_outputs(
    v4: pd.DataFrame,
    fixed: pd.DataFrame,
    output_dir: Path,
) -> None:
    run_tables = []
    for regularization, frame in (
        ("fixed_c1", fixed),
        ("train_cv", v4[v4.setting.str.contains("__q1000_")]),
    ):
        for selector in ("ood_test", "id_val"):
            for reference in ("customary", "extended"):
                for budget_mode in ("native", "equal_count"):
                    run_tables.append(
                        factorial_cell(
                            frame,
                            regularization,
                            selector,
                            reference,
                            budget_mode,
                        )
                    )
    identifiers = [
        "regularization",
        "selection",
        "reference",
        "budget_mode",
    ]
    runs = add_design_columns(pd.concat(run_tables, ignore_index=True))
    groups, clusters = aggregate_fixed_groups(runs, identifiers)
    summary = dataset_equal_summary(groups, identifiers)
    contrasts = factorial_axis_contrasts(summary)
    interactions = factorial_pairwise_interactions(summary)
    runs.to_csv(output_dir / "factorial_run_effects.csv", index=False)
    groups.to_csv(output_dir / "factorial_group_effects.csv", index=False)
    clusters.to_csv(output_dir / "factorial_cluster_effects.csv", index=False)
    summary.to_csv(output_dir / "factorial_summary.csv", index=False)
    contrasts.to_csv(output_dir / "factorial_axis_contrasts.csv", index=False)
    interactions.to_csv(
        output_dir / "factorial_pairwise_interactions.csv",
        index=False,
    )


def primary_endpoints_for_runs(
    frame: pd.DataFrame,
    draw_label: str,
) -> pd.DataFrame:
    quantum, _, classical = family_matrices(frame, "id_val", model="svc")
    assert_aligned(quantum, classical)
    quantum_endpoint = full_endpoint(quantum)
    classical_endpoint = blocked_expected_endpoint(
        classical,
        60,
        draw_label,
    )
    output = quantum.runs.copy()
    output["quantum_ood"] = quantum_endpoint
    output["classical_ood"] = classical_endpoint
    output["delta"] = quantum_endpoint - classical_endpoint
    return output


def make_shortcut_outputs(
    v4: pd.DataFrame,
    shortcut: pd.DataFrame,
    output_dir: Path,
) -> None:
    original_frame = v4[
        v4.setting.str.contains("__q1000_")
        & ~v4.setting.str.startswith(("m1_", "m2_"))
    ]
    original = primary_endpoints_for_runs(
        original_frame,
        "shortcut::paired_classical60",
    )
    ablated = primary_endpoints_for_runs(
        shortcut,
        "shortcut::paired_classical60",
    )
    keys = ["setting", "qs", "seed", "group"]
    paired = original.merge(
        ablated,
        on=keys,
        suffixes=("_original", "_ablated"),
        validate="one_to_one",
    )
    paired["delta"] = paired.delta_ablated
    paired["ablation_change"] = paired.delta_ablated - paired.delta_original
    paired = add_design_columns(paired)

    effect_input = paired[keys + [
        "ms",
        "size",
        "dataset",
        "delta",
    ]]
    groups, clusters = aggregate_fixed_groups(effect_input, [])
    original_input = paired[keys + [
        "ms",
        "size",
        "dataset",
        "delta_original",
    ]].rename(columns={"delta_original": "delta"})
    original_groups, original_clusters = aggregate_fixed_groups(
        original_input,
        [],
    )
    original_groups = original_groups.rename(
        columns={
            "effect": "original_effect",
            "ci_lo": "original_ci_lo",
            "ci_hi": "original_ci_hi",
        }
    )
    change_input = paired[keys + [
        "ms",
        "size",
        "dataset",
        "ablation_change",
    ]].rename(columns={"ablation_change": "delta"})
    change_groups, change_clusters = aggregate_fixed_groups(change_input, [])
    change_groups = change_groups.rename(
        columns={
            "effect": "ablation_change",
            "ci_lo": "change_ci_lo",
            "ci_hi": "change_ci_hi",
        }
    )
    groups = groups.merge(
        original_groups[[
            "group",
            "original_effect",
            "original_ci_lo",
            "original_ci_hi",
        ]],
        on="group",
        validate="one_to_one",
    )
    groups = groups.merge(
        change_groups[[
            "group",
            "ablation_change",
            "change_ci_lo",
            "change_ci_hi",
        ]],
        on="group",
        validate="one_to_one",
    )
    groups["ablated_effect"] = groups["effect"]
    groups["ablated_ci_lo"] = groups["ci_lo"]
    groups["ablated_ci_hi"] = groups["ci_hi"]
    groups["dataset"] = groups.group.map(source_dataset_for_group)
    per_dataset = (
        groups.groupby("dataset", as_index=False, observed=True)
        .agg(
            original_effect=("original_effect", "mean"),
            ablated_effect=("ablated_effect", "mean"),
            ablation_change=("ablation_change", "mean"),
            n_groups=("group", "nunique"),
        )
    )
    summary = pd.DataFrame(
        [
            {
                "aggregation": "source_dataset_equal",
                "n_source_datasets": int(per_dataset.dataset.nunique()),
                "original_effect": float(per_dataset.original_effect.mean()),
                "ablated_effect": float(per_dataset.ablated_effect.mean()),
                "ablation_change": float(per_dataset.ablation_change.mean()),
                "min_dataset_change": float(
                    per_dataset.ablation_change.min()
                ),
                "max_dataset_change": float(
                    per_dataset.ablation_change.max()
                ),
            }
        ]
    )
    paired.to_csv(output_dir / "shortcut_ablation_run_effects.csv", index=False)
    groups.to_csv(output_dir / "shortcut_ablation_group_effects.csv", index=False)
    per_dataset.to_csv(
        output_dir / "shortcut_ablation_dataset_effects.csv", index=False
    )
    summary.to_csv(
        output_dir / "shortcut_ablation_summary.csv", index=False
    )
    pd.concat(
        [
            original_clusters.assign(endpoint="original_effect"),
            clusters.assign(endpoint="ablated_effect"),
        ],
        ignore_index=True,
    ).to_csv(
        output_dir / "shortcut_ablation_cluster_effects.csv", index=False
    )
    change_clusters.assign(endpoint="ablation_change").to_csv(
        output_dir / "shortcut_ablation_change_clusters.csv", index=False
    )


def make_primary_cluster_values(output_dir: Path) -> None:
    source = pd.read_csv(
        "results/v4/budget_confirmatory/p1_runs__budget60.csv"
    )
    source = source.rename(
        columns={"p1_ood_classical_exp_budget": "p1_ood_classical"}
    )
    source = add_design_columns(source)
    rows = []
    for (model, group), frame in source.groupby(["model", "group"]):
        qsplit_levels = sorted(frame.qs.unique())
        values = nested_cluster_means(frame)
        for qsplit, value in zip(qsplit_levels, values):
            rows.append({
                "model": model,
                "group": group,
                "qs": qsplit,
                "cluster_effect": value,
                "n_lower_level_rows": int((frame.qs == qsplit).sum()),
                "interval_role": (
                    "one_of_five_values_underlying_pointwise_conditional_"
                    "cluster_t_interval"
                ),
            })
    pd.DataFrame(rows).to_csv(
        output_dir / "primary_cluster_values.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "ember=results/ember_shift/extended_kernels",
            "netflow=results/netflow/extended_kernels",
        ],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/v8/reviewer_revision"),
    )
    args = parser.parse_args()
    roots = [
        (item.split("=", 1)[0], Path(item.split("=", 1)[1]))
        for item in args.roots
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    v4 = load_v4(roots)
    fixed = load_v8(roots, "summary_v8_fixedc.csv")
    shortcut = load_v8(
        roots,
        "summary_v8_shortcut.csv",
        regularization="train_cv",
    )
    make_entanglement_outputs(v4, args.out_dir)
    make_winner_composition(v4, args.out_dir)
    make_factorial_outputs(v4, fixed, args.out_dir)
    make_shortcut_outputs(v4, shortcut, args.out_dir)
    make_primary_cluster_values(args.out_dir)
    print(f"[OK] wrote v0.8 reviewer-revision outputs to {args.out_dir}")


if __name__ == "__main__":
    main()

"""Generate the v0.8 circuit-aware and within-generation sensitivity figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd


GROUP_ORDER = (
    "ember_m1",
    "ember_m2",
    "unsw_dos_natural_cur",
    "unsw_dos_m2_centroid",
    "unsw_recon_natural_cur",
    "unsw_recon_m2_centroid",
    "toniot_scanning_natural_cur",
    "toniot_scanning_m2_centroid",
)
NETWORK_ORDER = GROUP_ORDER[2:]
GROUP_LABELS = {
    "ember_m1": "EMBER m1",
    "ember_m2": "EMBER m2",
    "unsw_dos_natural_cur": "UNSW-DoS campaign",
    "unsw_dos_m2_centroid": "UNSW-DoS constructed",
    "unsw_recon_natural_cur": "UNSW-Recon campaign",
    "unsw_recon_m2_centroid": "UNSW-Recon constructed",
    "toniot_scanning_natural_cur": "ToN-IoT campaign",
    "toniot_scanning_m2_centroid": "ToN-IoT constructed",
}
DATASET_FOR_GROUP = {
    "ember_m1": "EMBER",
    "ember_m2": "EMBER",
    "unsw_dos_natural_cur": "UNSW-NB15",
    "unsw_dos_m2_centroid": "UNSW-NB15",
    "unsw_recon_natural_cur": "UNSW-NB15",
    "unsw_recon_m2_centroid": "UNSW-NB15",
    "toniot_scanning_natural_cur": "ToN-IoT",
    "toniot_scanning_m2_centroid": "ToN-IoT",
}
DATASET_COLORS = {
    "EMBER": "#0072B2",
    "UNSW-NB15": "#E69F00",
    "ToN-IoT": "#009E73",
}
STRATA = (
    ("all_quantum_maps", "All maps"),
    ("entangling_zz", "Entangling ZZ"),
    ("separable_product", "Product maps"),
)
FACTORIAL_ROWS = (
    ("fixed_c1", "ood_test", "$C=1$, OOD oracle"),
    ("fixed_c1", "id_val", "$C=1$, ID validation"),
    ("train_cv", "ood_test", "Train-CV, OOD oracle"),
    ("train_cv", "id_val", "Train-CV, ID validation"),
)
FACTORIAL_COLUMNS = (
    ("customary", "native", "Customary\nnative"),
    ("customary", "equal_count", "Customary\nequal count"),
    ("extended", "native", "Extended\nnative"),
    ("extended", "equal_count", "Extended\nequal count"),
)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )


def _panel_quantum_strata(
    axis: plt.Axes,
    groups: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    axis.axhline(0, color="0.45", linestyle="--", linewidth=0.8, zorder=1)
    offsets = np.linspace(-0.18, 0.18, len(GROUP_ORDER))
    for x, (stratum, _) in enumerate(STRATA):
        subset = groups[groups.stratum == stratum].set_index("group")
        for offset, group in zip(offsets, GROUP_ORDER):
            row = subset.loc[group]
            dataset = DATASET_FOR_GROUP[group]
            axis.scatter(
                x + offset,
                row.effect,
                s=18,
                color=DATASET_COLORS[dataset],
                alpha=0.78,
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
                label=dataset if x == 0 and group in {
                    "ember_m1",
                    "unsw_dos_natural_cur",
                    "toniot_scanning_natural_cur",
                } else None,
            )
        aggregate = summary[summary.stratum == stratum].iloc[0]
        axis.scatter(
            x,
            aggregate.dataset_equal_effect,
            marker="D",
            s=48,
            color="black",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
            label="3-source equal mean" if x == 0 else None,
        )
    axis.set_xticks(
        np.arange(len(STRATA)),
        [label for _, label in STRATA],
    )
    axis.set_ylabel(r"OOD balanced accuracy $\Delta$" "\n(quantum $-$ classical)")
    axis.set_title("Quantum-map strata at matched candidate count")
    axis.grid(axis="y", color="0.91", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=2, loc="best", handletextpad=0.4)


def _panel_shortcut(axis: plt.Axes, groups: pd.DataFrame) -> None:
    axis.axvline(0, color="0.45", linestyle="--", linewidth=0.8, zorder=1)
    subset = groups.set_index("group")
    y = np.arange(len(NETWORK_ORDER))
    for position, group in zip(y, NETWORK_ORDER):
        row = subset.loc[group]
        original = row.original_effect
        ablated = row.ablated_effect
        axis.plot(
            [original, ablated],
            [position, position],
            color="0.72",
            linewidth=1.1,
            zorder=1,
        )
        axis.scatter(
            original,
            position,
            marker="o",
            s=28,
            facecolor="white",
            edgecolor="#0072B2",
            linewidth=1.0,
            zorder=3,
            label="Original features" if position == 0 else None,
        )
        axis.scatter(
            ablated,
            position,
            marker="s",
            s=28,
            color="#D55E00",
            zorder=3,
            label="Fields removed" if position == 0 else None,
        )
    axis.set_yticks(y, [GROUP_LABELS[group] for group in NETWORK_ORDER])
    axis.set_ylim(len(NETWORK_ORDER) - 0.5, -0.5)
    axis.set_xlabel(r"OOD balanced accuracy $\Delta$")
    axis.set_title("Port/protocol/service-field ablation")
    axis.grid(axis="x", color="0.91", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="best")


def _panel_factorial(
    axis: plt.Axes,
    summary: pd.DataFrame,
) -> plt.cm.ScalarMappable:
    matrix = np.empty((len(FACTORIAL_ROWS), len(FACTORIAL_COLUMNS)))
    for row_index, (regularization, selection, _) in enumerate(FACTORIAL_ROWS):
        for column_index, (reference, budget_mode, _) in enumerate(
            FACTORIAL_COLUMNS
        ):
            cell = summary[
                (summary.regularization == regularization)
                & (summary.selection == selection)
                & (summary.reference == reference)
                & (summary.budget_mode == budget_mode)
            ]
            if len(cell) != 1:
                raise ValueError(
                    "factorial cell missing or duplicated: "
                    f"{regularization}/{selection}/{reference}/{budget_mode}"
                )
            matrix[row_index, column_index] = cell.iloc[0].dataset_equal_effect

    limit = max(0.01, float(np.max(np.abs(matrix))))
    normalizer = colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    image = axis.imshow(
        matrix,
        cmap="RdBu_r",
        norm=normalizer,
        aspect="auto",
        interpolation="nearest",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            rgba = image.cmap(image.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axis.text(
                column_index,
                row_index,
                f"{value:+.4f}",
                ha="center",
                va="center",
                color="black" if luminance > 0.55 else "white",
                fontsize=7.2,
            )
    axis.set_xticks(
        np.arange(len(FACTORIAL_COLUMNS)),
        [label for _, _, label in FACTORIAL_COLUMNS],
    )
    axis.set_yticks(
        np.arange(len(FACTORIAL_ROWS)),
        [label for _, _, label in FACTORIAL_ROWS],
    )
    axis.set_title(
        "Within-v4 factorial: three-source-equal fixed-case effects",
        pad=5,
    )
    axis.tick_params(length=0)
    return image


def make_figure(input_dir: Path, output_path: Path) -> None:
    strata_groups = pd.read_csv(input_dir / "quantum_strata_group_effects.csv")
    strata_summary = pd.read_csv(input_dir / "quantum_strata_summary.csv")
    factorial = pd.read_csv(input_dir / "factorial_summary.csv")
    shortcut = pd.read_csv(input_dir / "shortcut_ablation_group_effects.csv")

    _configure_style()
    figure = plt.figure(figsize=(7.2, 6.7), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.05, 1.0))
    axis_a = figure.add_subplot(grid[0, 0])
    axis_c = figure.add_subplot(grid[0, 1])
    axis_b = figure.add_subplot(grid[1, :])

    _panel_quantum_strata(axis_a, strata_groups, strata_summary)
    _panel_shortcut(axis_c, shortcut)
    image = _panel_factorial(axis_b, factorial)
    colorbar = figure.colorbar(
        image,
        ax=axis_b,
        orientation="vertical",
        fraction=0.025,
        pad=0.025,
    )
    colorbar.set_label(r"$\Delta$ (quantum $-$ classical)")

    for label, axis in zip(("a", "b", "c"), (axis_a, axis_c, axis_b)):
        axis.text(
            -0.10 if axis is axis_b else -0.16,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"[ok] wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/v8/reviewer_revision"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manuscript/fig_v8_reviewer_revision.pdf"),
    )
    args = parser.parse_args()
    make_figure(args.input_dir, args.output)


if __name__ == "__main__":
    main()

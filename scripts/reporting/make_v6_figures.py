"""Generate reviewer-requested v0.6.0 manuscript figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_LABELS = {
    "ember_m1": "EMBER $m1$",
    "ember_m2": "EMBER $m2$",
    "unsw_dos_natural_cur": "UNSW-DoS campaign",
    "unsw_dos_m2_centroid": "UNSW-DoS constructed",
    "unsw_recon_natural_cur": "UNSW-Recon campaign",
    "unsw_recon_m2_centroid": "UNSW-Recon constructed",
    "toniot_scanning_natural_cur": "ToN-IoT campaign",
    "toniot_scanning_m2_centroid": "ToN-IoT constructed",
}
GROUP_ORDER = tuple(GROUP_LABELS)
CONDITIONS = (
    ("pre_psd", "Sampled, pre-PSD", "#D55E00", "s"),
    ("post_psd", "PSD-projected", "#0072B2", "o"),
)


def make_shots_mc_figure(summary_path: Path, output_path: Path) -> None:
    data = pd.read_csv(summary_path)
    shots = np.array([128, 512, 2048, 8192])
    x = np.arange(len(shots))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        2, 4, figsize=(11.7, 6.2), sharex=True, sharey=True, constrained_layout=True
    )
    for panel, (axis, group) in enumerate(zip(axes.ravel(), GROUP_ORDER)):
        subset = data[data.group == group]
        for condition, label, color, marker in CONDITIONS:
            rows = (
                subset[subset.projection_condition == condition]
                .set_index("shots")
                .loc[shots]
            )
            median = rows.ood_difference_from_exact_median.to_numpy(float)
            lower = rows.ood_difference_from_exact_q025.to_numpy(float)
            upper = rows.ood_difference_from_exact_q975.to_numpy(float)
            axis.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0)
            axis.plot(
                x,
                median,
                color=color,
                marker=marker,
                markersize=3.8,
                linewidth=1.15,
                label=label,
            )
        axis.axhline(0, color="0.35", linewidth=0.8, linestyle="--")
        axis.set_title(GROUP_LABELS[group], fontsize=8.5, pad=3)
        axis.text(
            0.02,
            0.96,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            fontsize=9,
        )
        axis.set_xticks(x, [str(value) for value in shots])
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.set_axisbelow(True)

    for axis in axes[:, 0]:
        axis.set_ylabel("OOD balanced-accuracy\nchange from exact")
    for axis in axes[1, :]:
        axis.set_xlabel("Shots per fidelity entry")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
        frameon=False,
        handlelength=2.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"[ok] wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/v6/shots_mc/monte_carlo_by_group.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manuscript/fig_v6_shots_mc.pdf"),
    )
    args = parser.parse_args()
    make_shots_mc_figure(args.summary, args.output)


if __name__ == "__main__":
    main()

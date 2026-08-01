"""Create the manuscript figure for the prospective v1.0 Gate-2 replication."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUPS = (
    ("acsfoodstamps", "svc", "ACS Food Stamps\nSVC"),
    ("acsfoodstamps", "gpc", "ACS Food Stamps\nGPC"),
    ("brfss_diabetes", "svc", "BRFSS Diabetes\nSVC"),
    ("brfss_diabetes", "gpc", "BRFSS Diabetes\nGPC"),
)
TASK_COLORS = {"acsfoodstamps": "#2C7FB8", "brfss_diabetes": "#D55E00"}
MODEL_MARKERS = {"svc": "o", "gpc": "s"}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figure(root: Path, output: Path) -> None:
    zero = pd.read_csv(root / "zero_label_and_realized.csv")
    adaptive = pd.read_csv(root / "adaptive_thresholds.csv")
    draws = pd.read_csv(root / "comparison_draws.csv")
    oracle = pd.read_csv(root / "retrospective_oracle_thresholds.csv")
    zero = zero[zero.tier.eq("full_115")]
    adaptive = adaptive[
        adaptive.tier.eq("full_115") & np.isclose(adaptive.threshold, 0.010)
    ]
    draws = draws[
        draws.tier.eq("full_115") & np.isclose(draws.threshold, 0.010)
    ]
    oracle = oracle[
        oracle.tier.eq("full_115") & np.isclose(oracle.threshold, 0.010)
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.2),
        sharey=True,
        gridspec_kw={"width_ratios": (0.92, 1.38)},
        constrained_layout=True,
    )
    y = np.arange(len(GROUPS))[::-1].astype(float)
    jitter = np.linspace(-0.18, 0.18, 5)

    ax = axes[0]
    for ypos, (task, model, _) in zip(y, GROUPS):
        part = zero[zero.task.eq(task) & zero.model.eq(model)].sort_values("seed")
        if len(part) != 5:
            raise RuntimeError(f"expected five zero-label seeds for {task}/{model}")
        values = part.zero_label_upper.to_numpy(dtype=float)
        ax.scatter(
            values,
            ypos + jitter,
            s=34,
            marker=MODEL_MARKERS[model],
            color=TASK_COLORS[task],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.plot(
            [float(np.min(values)), float(np.max(values))],
            [ypos, ypos],
            color=TASK_COLORS[task],
            alpha=0.35,
            linewidth=1.1,
            zorder=1,
        )
        ax.scatter(
            [float(np.median(values))],
            [ypos],
            marker="|",
            s=100,
            color="#111111",
            linewidth=1.4,
            zorder=4,
        )
    ax.axvline(0.010, color="#555555", linestyle="--", linewidth=1.0)
    ax.text(0.0115, 3.43, "1-point threshold", color="#555555", va="top")
    ax.set_yticks(y, [item[2] for item in GROUPS])
    ax.set_xlabel("Sharp upper endpoint at zero labels")
    ax.set_xlim(-0.004, 0.182)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.6)
    ax.set_title("a  Locked label-free certificates", loc="left", fontweight="bold")

    ax = axes[1]
    offsets = {
        "retrospective_label_oracle": -0.19,
        "adaptive_bottleneck_cover": -0.06,
        "random_active_disagreement": 0.07,
        "nonadaptive_initial_coverage": 0.20,
    }
    for ypos, (task, model, _) in zip(y, GROUPS):
        adaptive_part = adaptive[
            adaptive.task.eq(task) & adaptive.model.eq(model)
        ].sort_values("seed")
        oracle_part = oracle[oracle.task.eq(task) & oracle.model.eq(model)].sort_values(
            "seed"
        )
        ax.scatter(
            adaptive_part.n_labels,
            np.full(5, ypos + offsets["adaptive_bottleneck_cover"]),
            s=31,
            marker=MODEL_MARKERS[model],
            color=TASK_COLORS[task],
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        ax.scatter(
            oracle_part.n_labels,
            np.full(5, ypos + offsets["retrospective_label_oracle"]),
            s=38,
            marker="|",
            color="#111111",
            linewidth=1.2,
            zorder=4,
        )
        for policy, marker, color in (
            ("random_active_disagreement", "D", "#666666"),
            ("nonadaptive_initial_coverage", "^", "#999999"),
        ):
            part = draws[
                draws.task.eq(task)
                & draws.model.eq(model)
                & draws.policy.eq(policy)
            ]
            values = part.n_labels.to_numpy(dtype=float)
            values = values[values >= 0]
            median = float(np.median(values))
            low, high = np.quantile(values, (0.025, 0.975))
            ax.errorbar(
                median,
                ypos + offsets[policy],
                xerr=np.asarray([[median - low], [high - median]]),
                fmt=marker,
                markersize=4.1,
                markerfacecolor="white",
                color=color,
                elinewidth=0.8,
                capsize=2.0,
                zorder=3,
            )
    ax.axvline(50, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(100, color="#AAAAAA", linestyle=":", linewidth=0.8)
    ax.text(50, 3.43, "50-label cell criterion", color="#666666", ha="right", va="top")
    ax.set_xscale("symlog", linthresh=10, linscale=0.9, base=10)
    ax.set_xticks([0, 2, 5, 10, 25, 50, 100, 200, 500])
    ax.set_xticklabels(["0", "2", "5", "10", "25", "50", "100", "200", "500"])
    ax.set_xlim(-1.2, 550)
    ax.set_xlabel("Labels needed for upper endpoint $\\leq 0.010$")
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.6)
    ax.set_title("b  Prospective contraction after label opening", loc="left", fontweight="bold")
    handles = [
        plt.Line2D([], [], marker="|", markersize=8, linestyle="", color="#111111", label="exact label oracle"),
        plt.Line2D([], [], marker="o", linestyle="", color=TASK_COLORS["acsfoodstamps"], label="adaptive; five seeds"),
        plt.Line2D([], [], marker="D", markersize=4, markerfacecolor="white", linestyle="", color="#666666", label="random active"),
        plt.Line2D([], [], marker="^", markersize=5, markerfacecolor="white", linestyle="", color="#999999", label="initial coverage"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2)
    fig.text(
        0.5,
        -0.012,
        "Points are the five frozen sampling seeds; comparator bars span the 2.5th-97.5th percentiles over 1,000 seed-order combinations.",
        ha="center",
        va="top",
        fontsize=7.3,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("results/v10/gate2_prospective/audit"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manuscript/fig_v10_gate2_prospective.pdf"),
    )
    args = parser.parse_args()
    configure_style()
    make_figure(args.audit_root, args.output)
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()

"""Create manuscript figures for the v0.9 partial-identification analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASE_LABELS = {
    "00_ember_m1": "EMBER\nm1",
    "01_ember_m2": "EMBER\nm2",
    "02_toniot_scanning_m2_centroid": "ToN-IoT\nconstr.",
    "03_toniot_scanning_natural_cur": "ToN-IoT\ncampaign",
    "04_unsw_dos_m2_centroid": "UNSW\nDoS constr.",
    "05_unsw_dos_natural_cur": "UNSW\nDoS campaign",
    "06_unsw_recon_m2_centroid": "UNSW\nRecon constr.",
    "07_unsw_recon_natural_cur": "UNSW\nRecon campaign",
}
SOURCE_COLORS = {
    "EMBER": "#6F4C9B",
    "ToN-IoT": "#1B9E77",
    "UNSW": "#2C7FB8",
}
STRATUM_COLORS = {"entangling_zz": "#0072B2", "product_map": "#D55E00"}
MODEL_MARKERS = {"svc": "o", "gpc": "s"}
MODEL_LABELS = {"svc": "SVC", "gpc": "GPC"}


def source_for_case(case: str) -> str:
    if "ember" in case:
        return "EMBER"
    if "toniot" in case:
        return "ToN-IoT"
    return "UNSW"


def linestyle_for_case(case: str) -> str:
    """Distinguish paired shifts without spending a second colour channel."""
    if "natural_cur" in case or case.endswith("m2"):
        return "--"
    return "-"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_frontiers(root: Path) -> pd.DataFrame:
    frames = []
    for model in ("svc", "gpc"):
        path = root / "partial_label_frontier" / model / "evidence_frontier_summary.csv"
        frame = pd.read_csv(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_certificate_figure(analysis_root: Path, output: Path) -> None:
    exact = pd.read_csv(analysis_root / "analysis" / "sharp_accuracy_envelopes.csv")
    exact = exact[exact.tier.eq("full_115")].copy()
    frontier = load_frontiers(analysis_root)
    frontier = frontier[
        frontier.tier.eq("full_115") & np.isclose(frontier.threshold, 0.01)
    ].copy()
    merged = exact.merge(
        frontier[
            [
                "case",
                "model",
                "n_labels",
                "random_active_median_n_labels",
                "random_active_q025_n_labels",
                "random_active_q975_n_labels",
                "initial_coverage_median_n_labels",
                "initial_coverage_q025_n_labels",
                "initial_coverage_q975_n_labels",
                "oracle_min_n_labels",
                "quantum_stratum",
            ]
        ],
        on=["case", "model"],
        validate="one_to_one",
    )
    cases = list(CASE_LABELS)
    x = np.arange(len(cases), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), constrained_layout=True)

    ax = axes[0]
    for model, offset in (("svc", -0.14), ("gpc", 0.14)):
        frame = merged[merged.model.eq(model)].set_index("case").loc[cases]
        for index, (_, row) in enumerate(frame.iterrows()):
            ax.scatter(
                x[index] + offset,
                row.accuracy_upper,
                s=43,
                marker=MODEL_MARKERS[model],
                color=STRATUM_COLORS[row.quantum_stratum],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
    ax.axhline(0.01, color="#555555", linestyle="--", linewidth=1.0)
    ax.text(7.42, 0.0115, "1-point threshold", ha="right", va="bottom", color="#555555")
    ax.set_xticks(x, [CASE_LABELS[case] for case in cases])
    ax.set_ylabel("Sharp upper endpoint at zero labels")
    ax.set_ylim(-0.003, 0.098)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_title("a  Label-free advantage remaining possible", loc="left", fontweight="bold")

    ax = axes[1]
    for model, offset in (("svc", -0.18), ("gpc", 0.18)):
        frame = merged[merged.model.eq(model)].set_index("case").loc[cases]
        for index, (_, row) in enumerate(frame.iterrows()):
            xpos = x[index] + offset
            ax.errorbar(
                xpos + 0.025,
                row.random_active_median_n_labels,
                yerr=np.asarray(
                    [
                        [
                            row.random_active_median_n_labels
                            - row.random_active_q025_n_labels
                        ],
                        [
                            row.random_active_q975_n_labels
                            - row.random_active_median_n_labels
                        ],
                    ]
                ),
                fmt="D",
                markersize=3.4,
                markerfacecolor="white",
                color="#666666",
                elinewidth=0.7,
                capsize=1.8,
                zorder=2,
            )
            ax.errorbar(
                xpos + 0.075,
                row.initial_coverage_median_n_labels,
                yerr=np.asarray(
                    [
                        [
                            row.initial_coverage_median_n_labels
                            - row.initial_coverage_q025_n_labels
                        ],
                        [
                            row.initial_coverage_q975_n_labels
                            - row.initial_coverage_median_n_labels
                        ],
                    ]
                ),
                fmt="^",
                markersize=4.0,
                markerfacecolor="white",
                color="#999999",
                elinewidth=0.7,
                capsize=1.8,
                zorder=2,
            )
            ax.scatter(
                xpos - 0.025,
                row.n_labels,
                s=38,
                marker=MODEL_MARKERS[model],
                color=STRATUM_COLORS[row.quantum_stratum],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.scatter(
                xpos - 0.075,
                row.oracle_min_n_labels,
                s=42,
                marker="_",
                color="#111111",
                linewidth=1.5,
                zorder=4,
            )
    ax.set_xticks(x, [CASE_LABELS[case] for case in cases])
    ax.set_ylabel("Labels needed for upper endpoint $\\leq 0.01$")
    ax.set_yscale("symlog", linthresh=10, linscale=0.9, base=10)
    ax.set_yticks([0, 5, 10, 20, 50, 100, 200, 500])
    ax.set_yticklabels(["0", "5", "10", "20", "50", "100", "200", "500"])
    ax.set_ylim(-1.5, 550)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_title("b  Prediction-aware acquisition controls", loc="left", fontweight="bold")
    policy_handles = [
        plt.Line2D([], [], marker="_", markersize=8, linestyle="", color="#111111", label="label oracle"),
        plt.Line2D([], [], marker="o", linestyle="", color="#0072B2", label="adaptive coverage"),
        plt.Line2D([], [], marker="D", markersize=4, markerfacecolor="white", linestyle="", color="#666666", label="random active"),
        plt.Line2D([], [], marker="^", markersize=5, markerfacecolor="white", linestyle="", color="#999999", label="initial coverage"),
    ]
    ax.legend(handles=policy_handles, loc="upper left", ncol=2, frameon=False)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#333333", label="SVC"),
        plt.Line2D([], [], marker="s", linestyle="", color="#333333", label="GPC"),
        plt.Line2D([], [], marker="o", linestyle="", color=STRATUM_COLORS["entangling_zz"], label="entangling ZZ"),
        plt.Line2D([], [], marker="o", linestyle="", color=STRATUM_COLORS["product_map"], label="product map"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_label_curve_figure(analysis_root: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharex=True, sharey=True, constrained_layout=True)
    for ax, model in zip(axes, ("svc", "gpc")):
        path = analysis_root / "partial_label_frontier" / model / "adaptive_curves.csv"
        frame = pd.read_csv(path)
        frame = frame[frame.tier.eq("full_115") & frame.n_observed.le(50)]
        for case in CASE_LABELS:
            part = frame[frame.case.eq(case)].sort_values("n_observed")
            source = source_for_case(case)
            label = CASE_LABELS[case].replace("\n", " ")
            ax.plot(
                part.n_observed,
                part.upper,
                color=SOURCE_COLORS[source],
                linestyle=linestyle_for_case(case),
                alpha=0.82,
                linewidth=1.35,
                label=label,
            )
        ax.axhline(0.01, color="#555555", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#999999", linestyle=":", linewidth=0.8)
        ax.set_title(f"{MODEL_LABELS[model]} selected quantum models", fontweight="bold")
        ax.set_xlabel("Audited target labels out of 500")
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axes[0].set_ylabel("Sharp upper endpoint for accuracy advantage")
    axes[0].set_ylim(-0.075, 0.10)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_shot_figure(analysis_root: Path, output: Path) -> None:
    path = analysis_root / "shot_analysis" / "shot_emulation_across_groups.csv"
    frame = pd.read_csv(path)
    frame = frame[frame.tier.eq("full_115")]
    conditions = {
        "pre_psd": ("Unprojected", "#009E73"),
        "independent_square_psd": ("Independent PSD", "#D55E00"),
        "nystrom_psd": ("Nystrom PSD", "#0072B2"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), sharex=True, constrained_layout=True)
    metrics = (
        (
            "across_case_median_median_accuracy_upper_change",
            "across_case_min_median_accuracy_upper_change",
            "across_case_max_median_accuracy_upper_change",
            "Change in zero-label upper endpoint",
        ),
        (
            "across_case_median_median_realized_bacc_advantage_change",
            "across_case_min_median_realized_bacc_advantage_change",
            "across_case_max_median_realized_bacc_advantage_change",
            "Change in realized BAcc advantage",
        ),
    )
    for ax, (median_col, min_col, max_col, ylabel) in zip(axes, metrics):
        for condition, (label, color) in conditions.items():
            part = frame[frame.projection_condition.eq(condition)].sort_values("shots")
            ax.plot(part.shots, part[median_col], marker="o", color=color, label=label)
            ax.fill_between(part.shots, part[min_col], part[max_col], color=color, alpha=0.10)
        ax.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 512, 2048, 8192], ["128", "512", "2,048", "8,192"])
        ax.set_xlabel("Shots per fidelity")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axes[0].set_title("a  Apparent non-emulability", loc="left", fontweight="bold")
    axes[1].set_title("b  Realized performance", loc="left", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("results/v9/partial_identification"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("manuscript"))
    args = parser.parse_args()
    configure_style()
    make_certificate_figure(args.analysis_root, args.output_dir / "fig_v9_certificate.pdf")
    make_label_curve_figure(args.analysis_root, args.output_dir / "fig_v9_label_frontier.pdf")
    make_shot_figure(args.analysis_root, args.output_dir / "fig_v9_shot_emulation.pdf")
    print(f"[ok] wrote v0.9 figures under {args.output_dir}")


if __name__ == "__main__":
    main()

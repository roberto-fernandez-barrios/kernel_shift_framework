# scripts/reporting/make_v2_figures.py
"""
Generate the print figures of the v2 manuscript from the frozen result CSVs.

Design notes: Okabe-Ito colorblind-safe palette with fixed category
assignment (classical = blue #0072B2, quantum = vermillion #D55E00,
neutral reference lines in gray); thin marks; direct labels only where they
carry information; single axis per panel; vector PDF output.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("manuscript")
C_CLASSICAL = "#0072B2"   # Okabe-Ito blue
C_QUANTUM = "#D55E00"     # Okabe-Ito vermillion
C_NEUTRAL = "#7f7f7f"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

CLASSICAL_KERNELS = ["linear", "poly2", "poly3", "rbf_gscale", "matern25_med", "matern15_med", "laplacian_med"]
QUANTUM_KERNELS = ["zz_r1_full", "zz_r2_full", "pauli_xz_r1_full", "zmap_r2"]
KLABEL = {
    "linear": "Linear", "poly2": "Poly-2", "poly3": "Poly-3", "rbf_gscale": "RBF",
    "matern25_med": "Matérn-5/2", "matern15_med": "Matérn-3/2", "laplacian_med": "Laplacian",
    "zz_r1_full": "ZZ-r1", "zz_r2_full": "ZZ-r2", "pauli_xz_r1_full": "PauliXZ", "zmap_r2": "Z-map",
}


# ----------------------------------------------------------------------------
# Figure 1: the four-movement arc (per-setting OOD deltas, Best-by-OOD)
# ----------------------------------------------------------------------------
def fig_arc() -> None:
    ember = pd.read_csv("results/ember_shift/extended_kernels/family_comparison_by_setting.csv")
    sweep = pd.read_csv("results/ember_shift/bandwidth_sweep/family_comparison_by_setting.csv")
    net = pd.read_csv("results/netflow/family_comparison_by_setting.csv")

    groups = [
        ("EMBER\nvs lin+RBF", ember, "ood_delta_q_vs_orig"),
        ("EMBER\nvs extended", ember, "ood_delta_q_vs_ext"),
        ("EMBER, bw-sym.\nvs extended", sweep, "ood_delta_q_vs_ext"),
        ("Network flows\nvs extended", net, "ood_delta_q_vs_ext"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, model, title in [(axes[0], "svc", "SVC"), (axes[1], "gpc", "GP classifier")]:
        ax.axhline(0, color=C_NEUTRAL, lw=0.8, ls="--", zorder=1)
        for i, (label, df, col) in enumerate(groups):
            vals = df[df.model == model][col].to_numpy()
            x = i + rng.uniform(-0.14, 0.14, size=vals.size)
            ax.scatter(x, vals, s=9, alpha=0.55, color=C_QUANTUM, edgecolors="none", zorder=2)
            ax.hlines(np.median(vals), i - 0.24, i + 0.24, color="black", lw=1.4, zorder=3)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[0] for g in groups], fontsize=6.8)
        ax.set_title(title)
        ax.set_xlim(-0.5, len(groups) - 0.5)
    axes[0].set_ylabel(r"$\Delta_{\mathrm{OOD}}$ (quantum $-$ best classical)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_v2_arc.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[✓] fig_v2_arc.pdf")


# ----------------------------------------------------------------------------
# Figure 2: dose-response mechanism scatter (effective rank vs alignment survival)
# ----------------------------------------------------------------------------
def fig_mechanism_scatter() -> None:
    frames = []
    for d in ["results/kernel_geometry/grid_ext_classical", "results/kernel_geometry/grid"]:
        for p in sorted(Path(d).glob("kernel_geometry__*.csv")):
            if "__qs42" not in p.name and not re.search(r"qs42", p.name):
                continue
            frames.append(pd.read_csv(p))
    geo = pd.concat(frames, ignore_index=True)
    geo = geo.drop_duplicates(subset=["setting_label", "kernel", "dim"])
    geo["kta_gain"] = geo["kta_ood"] - geo["kta_id"]

    agg = geo.groupby("kernel", as_index=False).agg(
        eff_rank=("spec_train_eff_rank", "mean"), kta_gain=("kta_gain", "mean"),
        er_lo=("spec_train_eff_rank", lambda s: s.quantile(0.25)),
        er_hi=("spec_train_eff_rank", lambda s: s.quantile(0.75)),
        kg_lo=("kta_gain", lambda s: s.quantile(0.25)),
        kg_hi=("kta_gain", lambda s: s.quantile(0.75)),
    )

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.axhline(0, color=C_NEUTRAL, lw=0.8, ls="--", zorder=1)
    for _, r in agg.iterrows():
        if r.kernel in CLASSICAL_KERNELS:
            color, marker = C_CLASSICAL, "o"
        elif r.kernel in QUANTUM_KERNELS:
            color, marker = C_QUANTUM, "^"
        else:
            continue
        ax.errorbar(r.eff_rank, r.kta_gain,
                    xerr=[[r.eff_rank - r.er_lo], [r.er_hi - r.eff_rank]],
                    yerr=[[r.kta_gain - r.kg_lo], [r.kg_hi - r.kta_gain]],
                    fmt=marker, ms=5, color=color, elinewidth=0.7, capsize=0, zorder=3)
        offsets = {"linear": (2, -18), "rbf_gscale": (5, -3), "laplacian_med": (-12, 9),
                   "zz_r2_full": (3, 8), "zz_r1_full": (5, -11), "pauli_xz_r1_full": (5, -12),
                   "zmap_r2": (8, 1), "matern15_med": (-38, 9), "matern25_med": (-20, -14),
                   "poly2": (2, 6), "poly3": (12, -10)}
        dx, dy = offsets.get(r.kernel, (3, 3))
        ax.annotate(KLABEL[r.kernel], (r.eff_rank, r.kta_gain),
                    textcoords="offset points", xytext=(dx, dy), fontsize=6.5, color="#333333")
    ax.set_xscale("log")
    ax.set_xlabel("Effective rank of training kernel (log)")
    ax.set_ylabel(r"Alignment survival: $\Delta$KTA (OOD $-$ ID)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=C_CLASSICAL, ms=5, label="Classical"),
        plt.Line2D([], [], marker="^", ls="", color=C_QUANTUM, ms=5, label="Quantum"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_v2_dose_response.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[✓] fig_v2_dose_response.pdf")


# ----------------------------------------------------------------------------
# Figure 3: cross-dataset within-setting correlation distributions
# ----------------------------------------------------------------------------
def fig_mechanism_law() -> None:
    df = pd.read_csv("results/mechanism/mechanism_within_setting_correlations.csv")
    dlabel = {"ember": "EMBER", "toniot_scanning": "ToN-IoT\nScanning",
              "unsw_dos": "UNSW\nDoS", "unsw_recon": "UNSW\nRecon."}
    order = ["ember", "toniot_scanning", "unsw_dos", "unsw_recon"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, col, title in [
        (axes[0], "rho_kta_ood", "KTA on OOD (alignment survival)"),
        (axes[1], "rho_eff_rank", "Effective rank (train-only)"),
    ]:
        ax.axhline(0, color=C_NEUTRAL, lw=0.8, ls="--", zorder=1)
        data, positions = [], []
        for i, ds in enumerate(order):
            for j, model in enumerate(["svc", "gpc"]):
                vals = df[(df.dataset == ds) & (df.model == model)][col].to_numpy()
                data.append(vals)
                positions.append(i + (j - 0.5) * 0.34)
        parts = ax.boxplot(
            data, positions=positions, widths=0.26, showfliers=False, patch_artist=True,
            medianprops=dict(color="black", lw=1.2),
            boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
        )
        for k, box in enumerate(parts["boxes"]):
            box.set_facecolor("#9ecae1" if k % 2 == 0 else "#fdae6b")
            box.set_edgecolor("#555555")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([dlabel[d] for d in order], fontsize=7)
        ax.set_title(title)
        ax.set_ylim(-1.05, 1.05)
    axes[0].set_ylabel(r"Within-setting Spearman $\rho$" + "\nvs OOD balanced accuracy")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#9ecae1", ec="#555555", lw=0.6, label="SVC"),
               plt.Rectangle((0, 0), 1, 1, fc="#fdae6b", ec="#555555", lw=0.6, label="GP classifier")]
    axes[1].legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_v2_mechanism_law.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[✓] fig_v2_mechanism_law.pdf")


if __name__ == "__main__":
    fig_arc()
    fig_mechanism_scatter()
    fig_mechanism_law()

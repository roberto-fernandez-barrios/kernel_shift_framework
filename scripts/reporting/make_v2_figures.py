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

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
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
                    fmt=marker, ms=6, color=color, elinewidth=0.6, capsize=0,
                    alpha=0.4, zorder=3)
        ax.errorbar(r.eff_rank, r.kta_gain, fmt=marker, ms=6, color=color, zorder=4)

    # Direct labels: the three lowest-rank classical kernels are collectively
    # labeled (their values are nearly identical); ambiguous labels get leader lines.
    a = agg.set_index("kernel")

    def label(kernel, text, dx, dy, leader=False):
        r = a.loc[kernel]
        arrow = dict(arrowstyle="-", lw=0.5, color="#999999", shrinkA=0, shrinkB=3) if leader else None
        ax.annotate(text, (r.eff_rank, r.kta_gain), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7, color="#333333", arrowprops=arrow,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.4), zorder=6)

    label("poly2", "Linear, Poly-2/3", 6, 14, leader=True)
    label("rbf_gscale", "RBF", 8, -4, leader=True)
    label("matern25_med", "Matérn-5/2", -8, -22, leader=True)
    label("matern15_med", "Matérn-3/2", -28, 16, leader=True)
    label("laplacian_med", "Laplacian", -14, 13, leader=True)
    label("pauli_xz_r1_full", "PauliXZ", 10, -16, leader=True)
    label("zmap_r2", "Z-map", 12, 4, leader=True)
    label("zz_r1_full", "ZZ-r1", 8, -12)
    label("zz_r2_full", "ZZ-r2", 8, 8)

    ax.set_xscale("log")
    ax.set_xlim(0.85, 220)
    ax.set_ylim(-0.055, 0.085)
    ax.set_xlabel("Effective rank of the training kernel (log scale)")
    ax.set_ylabel(r"Alignment survival: $\Delta$KTA (OOD $-$ ID)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=C_CLASSICAL, ms=6, label="Classical"),
        plt.Line2D([], [], marker="^", ls="", color=C_QUANTUM, ms=6, label="Quantum"),
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


# ----------------------------------------------------------------------------
# Figure: protocol schematic (kernel-swap pipeline)
# ----------------------------------------------------------------------------
def fig_protocol() -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, text, edge="#555555", face="#f7f7f7", fs=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    fc=face, ec=edge, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#222222")

    PAD = 0.7  # visual expansion of FancyBboxPatch (boxstyle pad) in data units

    def arrow(x0, y0, x1, y1):
        # Endpoints are given at the *visual* border of the boxes (nominal
        # edge +/- PAD), so heads and tails touch the drawn outline exactly.
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#555555",
                                    shrinkA=0.5, shrinkB=0.5, mutation_scale=9))

    box(1, 26, 16, 13, "Benchmarks\nEMBER (malware)\nUNSW / ToN-IoT\n(network flows)", fs=6.5)
    box(21, 26, 17, 13, "Shift construction\n$m1$ $\\cdot$ $m2$ $\\cdot$ natural\n$T,\\ S_{\\mathrm{ID}},\\ S_{\\mathrm{OOD}}$\n15 runs / setting", fs=6.5)
    box(42, 26, 16, 13, "Shared embedding\nMaxAbs, SVD($d$),\nangles $[0,\\pi]$", fs=6.5)

    box(62, 34, 18, 7, "Classical kernels\n7 + $\\gamma$ sweep", edge=C_CLASSICAL, fs=6.5)
    box(62, 24, 18, 7, "Quantum kernels\n4 maps, 3 scales", edge=C_QUANTUM, fs=6.5)

    box(84, 26, 15, 13, "Same classifier\nSVC / GPC\n(precomputed $K$)", fs=6.5)

    box(6, 3, 40, 12, "Gram-matrix geometry\neffective rank $\\cdot$ KTA (ID/OOD) $\\cdot$ $g(K_C{\\to}K_Q)$\n$\\Rightarrow$ mechanism analysis", face="#fdf3ec", edge=C_QUANTUM, fs=6.5)
    box(53, 3, 46, 12, "Family-internal selection on run means\nBest-by-OOD / Best-by-ID\npaired Wilcoxon + Holm over settings", fs=6.5)

    arrow(17 + PAD, 32.5, 21 - PAD, 32.5)
    arrow(38 + PAD, 32.5, 42 - PAD, 32.5)
    arrow(58 + PAD, 34.5, 62 - PAD, 37.5)
    arrow(58 + PAD, 30.5, 62 - PAD, 27.5)
    arrow(80 + PAD, 37.5, 84 - PAD, 35.5)
    arrow(80 + PAD, 27.5, 84 - PAD, 29.5)
    arrow(68, 24 - PAD, 30, 15 + PAD)
    arrow(91.5, 26 - PAD, 80, 15 + PAD)

    fig.savefig(OUT / "fig_v2_protocol.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[✓] fig_v2_protocol.pdf")


# ----------------------------------------------------------------------------
# Figure: predictive uncertainty under shift (GPC)
# ----------------------------------------------------------------------------
def fig_uncertainty() -> None:
    ember = pd.read_csv("results/ember_shift/extended_kernels/family_comparison_by_setting.csv")
    net = pd.read_csv("results/netflow/family_comparison_by_setting.csv")
    fams = ["classical_orig", "classical_ext", "quantum"]
    flabel = {"classical_orig": "lin+RBF", "classical_ext": "extended", "quantum": "quantum"}
    fcolor = {"classical_orig": "#9ecae1", "classical_ext": C_CLASSICAL, "quantum": C_QUANTUM}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    rng = np.random.default_rng(1)
    for ax, metric, title in [
        (axes[0], "entropy", "Predictive entropy rises under shift"),
        (axes[1], "ece", "Calibration degrades moderately"),
    ]:
        ax.axhline(0, color=C_NEUTRAL, lw=0.8, ls="--", zorder=1)
        pos = 0.0
        ticks, ticklabels = [], []
        for gname, df in [("EMBER", ember), ("Network flows", net)]:
            r = df[df.model == "gpc"]
            for fam in fams:
                vals = (r[f"gpc_{metric}_ood_{fam}"] - r[f"gpc_{metric}_id_{fam}"]).to_numpy()
                x = pos + rng.uniform(-0.13, 0.13, size=vals.size)
                ax.scatter(x, vals, s=8, alpha=0.55, color=fcolor[fam], edgecolors="none", zorder=2)
                ax.hlines(np.median(vals), pos - 0.25, pos + 0.25, color="black", lw=1.3, zorder=3)
                ticks.append(pos)
                ticklabels.append(flabel[fam])
                pos += 1.45
            pos += 1.1
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=6.5, rotation=20, ha="right")
        ax.set_title(title, pad=4)
        mid = ((ticks[0] + ticks[2]) / 2, (ticks[3] + ticks[5]) / 2)
        for m, g in zip(mid, ["EMBER", "Network flows"]):
            ax.annotate(g, (m, -0.30), xycoords=("data", "axes fraction"),
                        ha="center", fontsize=7.2, color="#555555", annotation_clip=False)
    axes[0].set_ylabel("OOD $-$ ID (per setting)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_v2_uncertainty.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[✓] fig_v2_uncertainty.pdf")


if __name__ == "__main__":
    fig_arc()
    fig_mechanism_scatter()
    fig_mechanism_law()
    fig_protocol()
    fig_uncertainty()

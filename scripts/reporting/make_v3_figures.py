# scripts/reporting/make_v3_figures.py
"""
Print figures for the revised (honest-selection) manuscript.

Design: Okabe-Ito colorblind-safe palette, classical = blue #0072B2,
quantum = vermillion #D55E00, neutral reference lines gray; thin marks; vector
PDF. Every figure traces to a closure-analysis CSV under results/.

Figures:
  fig_v3_continuum   dose-response continuum: effective rank orders alignment
                     regardless of family; fidelity maps sit mid-continuum.
  fig_v3_reversal    honest-selection deltas: quantum beats linear+RBF but not
                     the geometry-matched extended family, on both modalities.
  fig_v3_crossfit    non-circular mechanism: cross-fit and label-free rho.
  fig_v3_protocol    kernel-swap schematic (honest-selection wording).
  fig_v3_uncertainty predictive uncertainty under shift (GPC).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("manuscript")
C_CLASSICAL = "#0072B2"
C_QUANTUM = "#D55E00"
C_NEUTRAL = "#7f7f7f"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

QUANTUM = {"zz_r1_full", "zz_r2_full", "pauli_xz_r1_full", "zmap_r2"}
QLABEL = {"zz_r1_full": "ZZ-r1", "zz_r2_full": "ZZ-r2",
          "pauli_xz_r1_full": "PauliXZ", "zmap_r2": "Z-map"}


# ---------------------------------------------------------------------------
# Figure 1 (central): the cross-family dose-response continuum
# ---------------------------------------------------------------------------
def fig_continuum() -> None:
    df = pd.read_csv("results/dose_response/kernel_table.csv")
    df["is_q"] = df.kernel.isin(QUANTUM)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, col, ylab in [
        (axes[0], "kta_ood", "OOD kernel-target alignment"),
        (axes[1], "kta_survival", r"Alignment survival $\Delta$KTA (OOD $-$ ID)"),
    ]:
        if col == "kta_survival":
            ax.axhline(0, color=C_NEUTRAL, lw=0.8, ls="--", zorder=1)
        cl = df[~df.is_q]
        q = df[df.is_q]
        ax.scatter(cl.eff_rank, cl[col], s=26, color=C_CLASSICAL, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3, label="Classical")
        ax.scatter(q.eff_rank, q[col], s=52, color=C_QUANTUM, marker="^",
                   edgecolors="white", linewidths=0.5, zorder=4, label="Quantum (fidelity)")
        for _, r in q.iterrows():
            ax.annotate(QLABEL[r.kernel], (r.eff_rank, r[col]), textcoords="offset points",
                        xytext=(5, 5), fontsize=6.3, color=C_QUANTUM, zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("Effective rank of training kernel (log)")
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, loc="lower right", handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_v3_continuum.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_v3_continuum.pdf")


# ---------------------------------------------------------------------------
# Figure 2: the honest-selection reversal
# ---------------------------------------------------------------------------
def _deltas(tag: str, model: str, proto: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(f"results/honest_selection/{tag}__by_setting.csv")
    df = df[df.model == model]
    piv = df.pivot_table(index="setting", columns="fam", values=f"{proto}_ood_mean")
    return {"orig": (piv["quantum"] - piv["classical_orig"]).dropna().to_numpy(),
            "ext": (piv["quantum"] - piv["classical_ext"]).dropna().to_numpy()}


def fig_reversal() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    rng = np.random.default_rng(0)
    cats = [("EMBER\nvs lin+RBF", "ember_sweep_full", "orig"),
            ("EMBER\nvs extended", "ember_sweep_full", "ext"),
            ("Net flows\nvs lin+RBF", "netflow_sweep_full", "orig"),
            ("Net flows\nvs extended", "netflow_sweep_full", "ext")]
    for ax, model, title in [(axes[0], "svc", "SVC"), (axes[1], "gpc", "GPC")]:
        ax.axhline(0, color=C_NEUTRAL, lw=0.8, ls="--", zorder=1)
        for i, (label, tag, ref) in enumerate(cats):
            vals = _deltas(tag, model, "p1")[ref]
            color = C_QUANTUM if ref == "orig" else C_CLASSICAL
            x = i + rng.uniform(-0.14, 0.14, size=vals.size)
            ax.scatter(x, vals, s=11, alpha=0.6, color=color, edgecolors="none", zorder=2)
            ax.hlines(np.median(vals), i - 0.26, i + 0.26, color="black", lw=1.5, zorder=3)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c[0] for c in cats], fontsize=6.6)
        ax.set_title(title)
        ax.set_xlim(-0.5, len(cats) - 0.5)
    axes[0].set_ylabel(r"$\Delta_{\mathrm{OOD}}$ P1: quantum $-$ classical")
    fig.tight_layout()
    fig.savefig(OUT / "fig_v3_reversal.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_v3_reversal.pdf")


# ---------------------------------------------------------------------------
# Figure 3: non-circular mechanism controls
# ---------------------------------------------------------------------------
def fig_crossfit() -> None:
    df = pd.read_csv("results/mechanism_crossfit/unit_correlations.csv")
    df = df[df.scope == "all"]
    order = ["ember_m1", "ember_m2", "unsw_dos_natural_cur", "unsw_recon_natural_cur",
             "toniot_scanning_natural_cur"]
    dlab = {"ember_m1": "EMBER\nm1", "ember_m2": "EMBER\nm2",
            "unsw_dos_natural_cur": "UNSW\nDoS", "unsw_recon_natural_cur": "UNSW\nRecon.",
            "toniot_scanning_natural_cur": "ToN-IoT\nScan."}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, cols, title in [
        (axes[0], ("rho_ktaA_svc", "rho_ktaA_gpc"),
         r"Cross-fit $\rho$(KTA$_A$, acc$_B$), disjoint halves"),
        (axes[1], ("rho_rank_svc", "rho_rank_gpc"),
         r"Label-free $\rho$(eff. rank, acc)"),
    ]:
        ax.axhline(0, color=C_NEUTRAL, lw=0.8, ls="--", zorder=1)
        data, positions, faces = [], [], []
        for i, ds in enumerate(order):
            for j, c in enumerate(cols):
                vals = df[df.group == ds][c].dropna().to_numpy()
                data.append(vals)
                positions.append(i + (j - 0.5) * 0.34)
                faces.append("#9ecae1" if j == 0 else "#fdae6b")
        parts = ax.boxplot(data, positions=positions, widths=0.26, showfliers=False,
                           patch_artist=True, medianprops=dict(color="black", lw=1.2),
                           boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
                           capprops=dict(lw=0.6))
        for box, fc in zip(parts["boxes"], faces):
            box.set_facecolor(fc)
            box.set_edgecolor("#555555")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([dlab[d] for d in order], fontsize=6.8)
        ax.set_title(title, fontsize=7.6)
        ax.set_ylim(-1.05, 1.05)
    axes[0].set_ylabel(r"Within-unit Spearman $\rho$")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#9ecae1", ec="#555555", lw=0.6, label="SVC"),
               plt.Rectangle((0, 0), 1, 1, fc="#fdae6b", ec="#555555", lw=0.6, label="GPC")]
    axes[1].legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_v3_crossfit.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_v3_crossfit.pdf")


# ---------------------------------------------------------------------------
# Figure 4: protocol schematic (honest-selection wording)
# ---------------------------------------------------------------------------
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

    PAD = 0.7

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#555555",
                                    shrinkA=0.5, shrinkB=0.5, mutation_scale=9))

    box(1, 26, 16, 13, "Benchmarks\nEMBER (malware)\nUNSW / ToN-IoT\n(network flows)", fs=6.5)
    box(21, 26, 17, 13, "Shift construction\n$m1$ $\\cdot$ $m2$ $\\cdot$ natural\n$T,\\ S_{\\mathrm{ID}},\\ S_{\\mathrm{OOD}}$\n15 runs / setting", fs=6.5)
    box(42, 26, 16, 13, "Shared embedding\nMaxAbs, SVD($d$),\nangles $[0,\\pi]$", fs=6.5)
    box(62, 34, 18, 7, "Classical kernels\n7 + length-scale sweep", edge=C_CLASSICAL, fs=6.3)
    box(62, 24, 18, 7, "Quantum kernels\n4 maps, 3 scales", edge=C_QUANTUM, fs=6.5)
    box(84, 26, 15, 13, "Same classifier\nSVC / GPC\n(precomputed $K$)", fs=6.5)
    box(6, 3, 40, 12, "Gram-matrix geometry\neffective rank $\\cdot$ KTA (ID/OOD)\n$\\Rightarrow$ mechanism + cross-fit controls", face="#fdf3ec", edge=C_QUANTUM, fs=6.3)
    box(53, 3, 46, 12, "Family-internal honest selection\nP1 deployment (ID) $\\cdot$ P2 cross-seed\nhierarchical permutation test", fs=6.4)

    arrow(17 + PAD, 32.5, 21 - PAD, 32.5)
    arrow(38 + PAD, 32.5, 42 - PAD, 32.5)
    arrow(58 + PAD, 34.5, 62 - PAD, 37.5)
    arrow(58 + PAD, 30.5, 62 - PAD, 27.5)
    arrow(80 + PAD, 37.5, 84 - PAD, 35.5)
    arrow(80 + PAD, 27.5, 84 - PAD, 29.5)
    arrow(68, 24 - PAD, 30, 15 + PAD)
    arrow(91.5, 26 - PAD, 80, 15 + PAD)
    fig.savefig(OUT / "fig_v3_protocol.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_v3_protocol.pdf")


# ---------------------------------------------------------------------------
# Figure 5: predictive uncertainty under shift (GPC) -- unchanged finding
# ---------------------------------------------------------------------------
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
    fig.savefig(OUT / "fig_v3_uncertainty.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_v3_uncertainty.pdf")


if __name__ == "__main__":
    fig_continuum()
    fig_reversal()
    fig_crossfit()
    fig_protocol()
    fig_uncertainty()

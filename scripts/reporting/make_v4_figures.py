# scripts/reporting/make_v4_figures.py
"""
Print figures for the v4 (honest-selection) manuscript. Okabe-Ito palette,
classical = blue #0072B2, quantum = vermillion #D55E00, neutral gray. Vector
PDF. Every figure traces to a results/v4/ CSV. Framing F1.

  fig_v4_honest      per-group honest P1' OOD delta vs both references, with
                     conditional intervals -- deltas sit at/below zero.
  fig_v4_rankmatched at matched effective rank, quantum minus classical delta
                     is negative in every group (geometry, not quantumness).
  fig_v4_mechanism   the geometry-OOD association is regime-dependent.
  fig_v4_shots       finite-shot stability: KTA survives, effective rank does not.
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

ORDER = ["ember_m1", "ember_m2", "unsw_dos_natural_cur", "unsw_dos_m2_centroid",
         "unsw_recon_natural_cur", "unsw_recon_m2_centroid",
         "toniot_scanning_natural_cur", "toniot_scanning_m2_centroid"]
SHORT = {"ember_m1": "EMBER m1", "ember_m2": "EMBER m2",
         "unsw_dos_natural_cur": "UNSW-DoS drift", "unsw_dos_m2_centroid": "UNSW-DoS m2",
         "unsw_recon_natural_cur": "UNSW-Recon drift", "unsw_recon_m2_centroid": "UNSW-Recon m2",
         "toniot_scanning_natural_cur": "ToN-IoT drift", "toniot_scanning_m2_centroid": "ToN-IoT m2"}


def fig_honest() -> None:
    g = pd.read_csv("results/v4/family_comparison/group_summary.csv")
    hier = pd.read_csv("results/v4/family_comparison/inference/hierarchical_effects.csv")
    hier = hier[hier.stratum == "all"].set_index(["variant", "model", "scope"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    y = np.arange(len(ORDER))
    for ax, model, title in [(axes[0], "svc", "SVC"), (axes[1], "gpc", "GPC")]:
        ax.axvline(0, color=C_NEUTRAL, lw=0.8, ls="--", zorder=1)
        gm = g[g.model == model].set_index("group")
        for i, grp in enumerate(ORDER):
            r = gm.loc[grp]
            # vs extended (primary) with conditional interval
            try:
                ci = hier.loc[("vs_classical_ext", model, grp)]
                ax.plot([ci.ci_lo, ci.ci_hi], [i, i], color=C_CLASSICAL, lw=1.2, zorder=2)
            except KeyError:
                pass
            ax.scatter(r.p1_ood_delta_vs_classical_ext, i, s=30, color=C_CLASSICAL,
                       zorder=3, label="vs extended" if i == 0 else None)
            ax.scatter(r.p1_ood_delta_vs_classical_orig, i, s=24, marker="D",
                       facecolors="none", edgecolors=C_QUANTUM, linewidths=1.0,
                       zorder=3, label="vs linear+RBF" if i == 0 else None)
        ax.set_yticks(y)
        ax.set_title(title)
        ax.set_xlabel(r"$\Delta_{\mathrm{OOD}}$ P1$'$: quantum $-$ classical")
    axes[0].set_yticklabels([SHORT[g_] for g_ in ORDER])
    axes[0].set_ylim(len(ORDER) - 0.5, -0.5)   # top-to-bottom = ORDER, once (shared)
    axes[0].legend(frameon=False, loc="lower left", fontsize=7)
    fig.suptitle("No robust quantum advantage under honest, budget-matched selection",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_v4_honest.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_v4_honest.pdf")


def fig_rankmatched() -> None:
    rk = pd.read_csv("results/v4/mechanism/rank_matched_summary.csv").set_index("group")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.axvline(0, color=C_NEUTRAL, lw=0.8, ls="--", zorder=1)
    y = np.arange(len(ORDER))
    for i, grp in enumerate(ORDER):
        r = rk.loc[grp]
        ax.barh(i, r.median_delta, color=C_CLASSICAL, alpha=0.85, height=0.6, zorder=2)
        ax.text(r.median_delta - 0.0004, i, f"{100*r.frac_quantum_better:.0f}%",
                va="center", ha="right", fontsize=6.2, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[g] for g in ORDER])
    ax.invert_yaxis()
    ax.set_xlabel(r"median $\Delta_{\mathrm{OOD}}$ at matched effective rank"
                  "\n(quantum $-$ nearest-rank classical)")
    ax.set_title("At matched geometry, classical kernels are at least as accurate")
    fig.tight_layout()
    fig.savefig(OUT / "fig_v4_rankmatched.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_v4_rankmatched.pdf")


def fig_mechanism() -> None:
    m = pd.read_csv("results/v4/mechanism/summary_by_group.csv")
    m = m[m.scope == "all"].set_index("group")
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.axhline(0, color=C_NEUTRAL, lw=0.8, ls="--", zorder=1)
    x = np.arange(len(ORDER))
    ax.plot(x, [m.loc[g].median_rho_spec_train_eff_rank for g in ORDER], "o-",
            color=C_CLASSICAL, lw=1.0, ms=5, label="effective rank")
    ax.plot(x, [m.loc[g].median_rho_kta_ood for g in ORDER], "^-",
            color=C_QUANTUM, lw=1.0, ms=5, label="KTA on OOD")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[g] for g in ORDER], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel(r"within-unit Spearman $\rho$ vs OOD accuracy")
    ax.set_title("The geometry--OOD association is regime-dependent")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(-0.4, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_v4_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_v4_mechanism.pdf")


def fig_shots() -> None:
    geo = pd.read_csv("results/v4/shots/geometry_stability.csv")
    acc = pd.read_csv("results/v4/shots/accuracy_stability.csv")
    acc = acc[(acc.model == "svc") & (acc.split == "ood_test")].groupby("shots").mean_abs_dev.mean()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    shots = geo.shots.to_numpy()
    ax.plot(shots, geo.eff_rank_ratio_median, "o-", color=C_CLASSICAL, lw=1.0, ms=5,
            label="eff. rank inflation (ratio)")
    ax.fill_between(shots, geo.eff_rank_ratio_p5, geo.eff_rank_ratio_p95,
                    color=C_CLASSICAL, alpha=0.15)
    ax.axhline(1.0, color=C_NEUTRAL, lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("shots per kernel entry")
    ax.set_ylabel("effective-rank ratio (shots / exact)", color=C_CLASSICAL)
    ax.tick_params(axis="y", labelcolor=C_CLASSICAL)
    ax2 = ax.twinx()
    ax2.plot(shots, geo.kta_ood_dev_median, "^-", color=C_QUANTUM, lw=1.0, ms=5,
             label="KTA$_{OOD}$ deviation")
    ax2.plot(shots, acc.reindex(shots).to_numpy(), "s--", color="#555555", lw=1.0, ms=4,
             label="OOD accuracy deviation")
    ax2.set_ylabel("median |deviation from exact|")
    ax2.spines["top"].set_visible(False)
    ax.set_title("Finite shots inflate effective rank; KTA and accuracy survive")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=6.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_v4_shots.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_v4_shots.pdf")


def fig_uncertainty() -> None:
    """GPC predictive-uncertainty change ID->OOD, for the P1'-selected config
    of each family (from summary_v4). Computed under honest selection."""
    import glob
    import os
    rows = []
    for f in glob.glob("results/*/extended_kernels/*/summary_v4.csv"):
        run = os.path.basename(os.path.dirname(f))
        grp = ("ember_" + run.split("__")[0].split("_")[0]
               if run.startswith(("m1_", "m2_"))
               else run.split("__")[0] + "_" + run.split("__")[1])
        ds = "ember" if grp.startswith("ember") else grp.rsplit("_m", 1)[0].rsplit("_natural", 1)[0]
        s = pd.read_csv(f)
        g = s[s.model == "gpc"]
        for fam, mask in [("classical", g.family == "classical_ext"),
                          ("quantum", g.family == "quantum")]:
            sub = g[mask]
            w = sub.pivot_table(index="cfg", columns="split",
                                values=["balanced_accuracy", "ece", "mean_predictive_entropy"])
            sel = w["balanced_accuracy"]["id_val"].idxmax()   # P1' selection
            for metric in ("ece", "mean_predictive_entropy"):
                rows.append({"dataset": ds, "family": fam, "metric": metric,
                             "delta": w[metric].loc[sel, "ood_test"] - w[metric].loc[sel, "id_test"]})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    datasets = ["ember", "unsw_dos", "unsw_recon", "toniot_scanning"]
    dlab = {"ember": "EMBER", "unsw_dos": "UNSW-DoS", "unsw_recon": "UNSW-Recon",
            "toniot_scanning": "ToN-IoT"}
    for ax, metric, title in [
        ("mean_predictive_entropy", "mean_predictive_entropy", "Predictive entropy rises under shift"),
        ("ece", "ece", "Calibration error changes moderately")]:
        a = axes[0] if metric == "mean_predictive_entropy" else axes[1]
        a.axhline(0, color=C_NEUTRAL, lw=0.8, ls="--")
        for j, fam in enumerate(("classical", "quantum")):
            vals = [df[(df.dataset == d) & (df.family == fam) & (df.metric == metric)].delta.mean()
                    for d in datasets]
            x = np.arange(len(datasets)) + (j - 0.5) * 0.32
            a.bar(x, vals, width=0.3, color=C_CLASSICAL if fam == "classical" else C_QUANTUM,
                  label=fam if metric == "mean_predictive_entropy" else None)
        a.set_xticks(range(len(datasets)))
        a.set_xticklabels([dlab[d] for d in datasets], fontsize=7, rotation=20, ha="right")
        a.set_title(title, fontsize=8)
    axes[0].set_ylabel("OOD $-$ ID (mean over settings)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_v4_uncertainty.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_v4_uncertainty.pdf")


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
    P = 0.7

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#555555",
                                    shrinkA=0.5, shrinkB=0.5, mutation_scale=9))
    box(1, 26, 16, 13, "Benchmarks\nEMBER (malware)\nUNSW / ToN-IoT\n(network flows)", fs=6.5)
    box(21, 26, 17, 13, "Shift construction\nsparsity $\\cdot$ centroid $\\cdot$\nheld-out campaign\n$T, S_{\\mathrm{ID}}, S_{\\mathrm{OOD}}$", fs=6.3)
    box(42, 26, 16, 13, "Shared embedding\nMaxAbs, SVD($d$),\nangles $[0,\\pi]$", fs=6.5)
    box(62, 34, 18, 7, "Classical kernels\n23 geometries", edge=C_CLASSICAL, fs=6.5)
    box(62, 24, 18, 7, "Quantum kernels\n12 fidelity maps", edge=C_QUANTUM, fs=6.5)
    box(84, 26, 15, 13, "Same classifier\nSVC / GPC\n$C$ tuned on train", fs=6.3)
    box(6, 3, 40, 12, "Gram-matrix geometry\neffective rank $\\cdot$ KTA (ID/OOD)\n$\\Rightarrow$ association analysis + shots model", face="#fdf3ec", edge=C_QUANTUM, fs=6.2)
    box(53, 3, 46, 12, "Honest selection at equal budget\nP1$'$: select on ID-validation\nfixed case studies $\\cdot$ conditional intervals", fs=6.3)
    arrow(17 + P, 32.5, 21 - P, 32.5)
    arrow(38 + P, 32.5, 42 - P, 32.5)
    arrow(58 + P, 34.5, 62 - P, 37.5)
    arrow(58 + P, 30.5, 62 - P, 27.5)
    arrow(80 + P, 37.5, 84 - P, 35.5)
    arrow(80 + P, 27.5, 84 - P, 29.5)
    arrow(68, 24 - P, 30, 15 + P)
    arrow(91.5, 26 - P, 80, 15 + P)
    fig.savefig(OUT / "fig_v4_protocol.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_v4_protocol.pdf")


if __name__ == "__main__":
    fig_honest()
    fig_rankmatched()
    fig_mechanism()
    fig_shots()
    fig_uncertainty()
    fig_protocol()

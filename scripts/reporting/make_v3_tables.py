# scripts/reporting/make_v3_tables.py
"""
LaTeX tables for the revised (honest-selection) manuscript.

Every number traces to a closure-analysis CSV under results/:
  honest_selection/<tag>__stats.csv     per-scenario deltas (Wilcoxon)
  honest_selection/hier_stats.csv       hierarchy-aware permutation test + LODO
  dose_response/kernel_table.csv        cross-family geometry continuum
  mechanism_crossfit/{summary_by_group,null_summary}.csv   circularity controls

Outputs booktabs bodies under results/tables_v3/.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("results/tables_v3")
MODEL = {"svc": "SVC", "gpc": "GPC"}
REF = {"classical_orig": "linear+RBF", "classical_ext": "extended classical"}
PROTO = {"p1": "P1 deployment", "p2": "P2 cross-seed", "p3": "P3 oracle"}
DS_GROUPS = {
    "ember": ["ember_m1", "ember_m2"],
    "netflow": ["unsw_dos_natural_cur", "unsw_recon_natural_cur",
                "toniot_scanning_natural_cur"],
}
GROUP_LABEL = {
    "ember_m1": "EMBER m1 (sparsity)", "ember_m2": "EMBER m2 (centroid)",
    "unsw_dos_natural_cur": "UNSW-NB15 DoS", "unsw_recon_natural_cur": "UNSW-NB15 Recon.",
    "toniot_scanning_natural_cur": "ToN-IoT Scanning",
}


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "--"
    if p >= 0.01:
        return f"${p:.2f}$"
    exp = int(f"{p:.0e}".split("e")[1])
    mant = p / (10 ** exp)
    return f"${mant:.1f}\\times 10^{{{exp}}}$"


def hier_headline(tag: str) -> str:
    """Hierarchy-aware inference: T = mean-over-datasets of mean-per-dataset delta."""
    h = pd.read_csv("results/honest_selection/hier_stats.csv")
    h = h[(h.tag == tag) & (h.stat == "hier_T")]
    lines = []
    for ref in ("classical_orig", "classical_ext"):
        first = True
        for proto in ("p1", "p2", "p3"):
            for model in ("svc", "gpc"):
                r = h[(h.reference == ref) & (h.protocol == proto) & (h.model == model)]
                if r.empty:
                    continue
                r = r.iloc[0]
                refcell = f"\\multirow{{6}}{{*}}{{{REF[ref]}}}" if first else ""
                first = False
                lodo = (f"$[{r.lodo_min:+.4f}, {r.lodo_max:+.4f}]$"
                        if pd.notna(r.lodo_min) else "--")
                lines.append(
                    f"{refcell} & {PROTO[proto]} & {MODEL[model]} & "
                    f"${r.mean_delta:+.4f}$ & {fmt_p(r.perm_p)} & {lodo} \\\\")
        lines.append("\\midrule" if ref == "classical_orig" else "")
    return "\n".join(l for l in lines if l)


def per_scenario(tag: str, dataset: str, ref: str = "classical_ext") -> str:
    """Per-scenario deltas with paired Wilcoxon (appendix detail)."""
    s = pd.read_csv(f"results/honest_selection/{tag}__stats.csv")
    s = s[(s.reference == ref) & (s.scope.isin(DS_GROUPS[dataset]))]
    lines = []
    for proto in ("p1", "p2"):
        first = True
        for model in ("svc", "gpc"):
            for grp in DS_GROUPS[dataset]:
                r = s[(s.protocol == proto) & (s.model == model) & (s.scope == grp)]
                if r.empty:
                    continue
                r = r.iloc[0]
                pcell = f"\\multirow{{6}}{{*}}{{{PROTO[proto]}}}" if first else ""
                first = False
                lines.append(
                    f"{pcell} & {MODEL[model]} & {GROUP_LABEL[grp]} & "
                    f"{int(r.wins)}/{int(r.n_settings)} & ${r.mean_delta:+.4f}$ & "
                    f"${r.median_delta:+.4f}$ & {fmt_p(r.wilcoxon_p)} \\\\")
        lines.append("\\midrule" if proto == "p1" else "")
    return "\n".join(l for l in lines if l)


KERNEL_LABEL = {
    "linear": "Linear", "rbf_gscale": "RBF", "poly2": "Poly-2", "poly3": "Poly-3",
    "laplacian_med": "Laplacian", "matern15_med": "Mat\\'ern-3/2", "matern25_med": "Mat\\'ern-5/2",
    "zz_r1_full": "ZZ-r1", "zz_r2_full": "ZZ-r2", "pauli_xz_r1_full": "PauliXZ", "zmap_r2": "Z-map",
}


def kernel_label(k: str) -> str:
    if k in KERNEL_LABEL:
        return KERNEL_LABEL[k]
    for base, lab in KERNEL_LABEL.items():
        if k.startswith(base + "_x"):
            return f"{lab}$\\times${k.split('_x')[-1]}"
    return k.replace("_", "\\_")


def dose_response_table() -> str:
    """Fidelity maps within the classical geometry continuum, sorted by eff. rank."""
    df = pd.read_csv("results/dose_response/kernel_table.csv").sort_values("eff_rank")
    lines = []
    for _, r in df.iterrows():
        fam = "Q" if r.family == "quantum" else "C"
        mark = "\\;$\\star$" if r.family == "quantum" else ""
        lines.append(
            f"{kernel_label(r.kernel)}{mark} & {fam} & {r.eff_rank:.1f} & "
            f"${r.kta_ood:+.3f}$ & ${r.kta_survival:+.3f}$ \\\\")
    return "\n".join(lines)


def crossfit_table() -> str:
    """Anti-circularity controls per dataset (scope=all)."""
    nul = pd.read_csv("results/mechanism_crossfit/null_summary.csv")
    summ = pd.read_csv("results/mechanism_crossfit/summary_by_group.csv")
    nul = nul[nul.scope == "all"].set_index("group")
    summ = summ[summ.scope == "all"].set_index("group")
    order = ["ember_m1", "ember_m2", "unsw_dos_natural_cur", "unsw_dos_m2_centroid",
             "unsw_recon_natural_cur", "unsw_recon_m2_centroid",
             "toniot_scanning_natural_cur", "toniot_scanning_m2_centroid"]
    glab = dict(GROUP_LABEL)
    glab.update({"unsw_dos_m2_centroid": "UNSW-NB15 DoS (m2)",
                 "unsw_recon_m2_centroid": "UNSW-NB15 Recon. (m2)",
                 "toniot_scanning_m2_centroid": "ToN-IoT Scanning (m2)"})
    lines = []
    for g in order:
        if g not in nul.index or g not in summ.index:
            continue
        n, s = nul.loc[g], summ.loc[g]
        # cross-fit rho: average svc/gpc for compactness of the headline column
        cf = (s.median_rho_ktaA_svc + s.median_rho_ktaA_gpc) / 2
        rk = (s.median_rho_rank_svc + s.median_rho_rank_gpc) / 2
        lines.append(
            f"{glab.get(g, g)} & {n.median_kta_ood:.3f} & {n.median_null_mean:.3f} & "
            f"{100*n.frac_cells_significant:.0f}\\% & ${cf:+.2f}$ & ${rk:+.2f}$ \\\\")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    writes = {
        "table_honest_hier_ember.tex": hier_headline("ember_sweep_full"),
        "table_honest_hier_netflow.tex": hier_headline("netflow_sweep_full"),
        "table_honest_scenario_ember.tex": per_scenario("ember_sweep_full", "ember"),
        "table_honest_scenario_netflow.tex": per_scenario("netflow_sweep_full", "netflow"),
        "table_dose_response.tex": dose_response_table(),
        "table_crossfit.tex": crossfit_table(),
    }
    for name, body in writes.items():
        (OUT / name).write_text(body + "\n", encoding="utf-8")
        print(f"[ok] {name} ({body.count(chr(10)) + 1} rows)")


if __name__ == "__main__":
    main()

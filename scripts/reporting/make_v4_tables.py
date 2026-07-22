# scripts/reporting/make_v4_tables.py
"""
LaTeX tables for the v4 (honest-selection) manuscript. Every number traces to
a results/v4/ CSV produced by the confirmatory pipeline. Framing F1: no robust
family-level advantage survives matched budget and honest selection.

Outputs booktabs bodies under results/v4/tables/.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("results/v4/tables")
MODEL = {"svc": "SVC", "gpc": "GPC"}
GLAB = {
    "ember_m1": "EMBER $m1$", "ember_m2": "EMBER $m2$",
    "unsw_dos_natural_cur": "UNSW-DoS (drift)", "unsw_dos_m2_centroid": "UNSW-DoS ($m2$)",
    "unsw_recon_natural_cur": "UNSW-Recon (drift)", "unsw_recon_m2_centroid": "UNSW-Recon ($m2$)",
    "toniot_scanning_natural_cur": "ToN-IoT (drift)", "toniot_scanning_m2_centroid": "ToN-IoT ($m2$)",
}
ORDER = ["ember_m1", "ember_m2", "unsw_dos_natural_cur", "unsw_dos_m2_centroid",
         "unsw_recon_natural_cur", "unsw_recon_m2_centroid",
         "toniot_scanning_natural_cur", "toniot_scanning_m2_centroid"]


def _sci(x: float) -> str:
    return f"${x:+.4f}$"


def headline_table() -> str:
    """Per-group honest P1' OOD delta vs both references, both classifiers,
    with the conditional interval from the GATE 2 estimator."""
    g = pd.read_csv("results/v4/family_comparison/group_summary.csv").set_index(["group", "model"])
    hier = pd.read_csv("results/v4/family_comparison/inference/hierarchical_effects.csv")
    hier = hier[(hier.stratum == "all")].set_index(["variant", "model", "scope"])
    lines = []
    for grp in ORDER:
        first = True
        for model in ("svc", "gpc"):
            if (grp, model) not in g.index:
                continue
            r = g.loc[(grp, model)]
            try:
                ci = hier.loc[("vs_classical_ext", model, grp)]
                ci_s = f"$[{ci.ci_lo:+.4f}, {ci.ci_hi:+.4f}]$"
            except KeyError:
                ci_s = "--"
            gcell = f"\\multirow{{2}}{{*}}{{{GLAB[grp]}}}" if first else ""
            first = False
            lines.append(
                f"{gcell} & {MODEL[model]} & "
                f"{_sci(r.p1_ood_delta_vs_classical_orig)} & "
                f"{_sci(r.p1_ood_delta_vs_classical_ext)} & {ci_s} & "
                f"{_sci(r.p1_idtest_delta_vs_classical_ext)} \\\\")
        lines.append("\\addlinespace")
    return "\n".join(l for l in lines if l)


def summary_row() -> str:
    """Dataset-equal-weighted descriptive means + LODO for the abstract/text."""
    h = pd.read_csv("results/v4/family_comparison/inference/hierarchical_effects.csv")
    h = h[(h.scope == "dataset_equal_mean") & (h.stratum == "all")]
    lines = []
    for _, r in h.iterrows():
        ref = r.variant.replace("vs_classical_", "").replace("orig", "linear+RBF").replace("ext", "extended")
        lines.append(f"{ref} & {MODEL[r.model]} & {_sci(r.effect)} & "
                     f"$[{r.jackknife_min:+.4f}, {r.jackknife_max:+.4f}]$ \\\\")
    return "\n".join(lines)


def mechanism_table() -> str:
    m = pd.read_csv("results/v4/mechanism/summary_by_group.csv")
    m = m[m.scope == "all"].set_index("group")
    rk = pd.read_csv("results/v4/mechanism/rank_matched_summary.csv").set_index("group")
    lines = []
    for grp in ORDER:
        if grp not in m.index:
            continue
        r, k = m.loc[grp], rk.loc[grp]
        lines.append(
            f"{GLAB[grp]} & {r.median_rho_spec_train_eff_rank:+.2f} & "
            f"{r.median_partial_eff_rank:+.2f} & {r.median_rho_kta_ood:+.2f} & "
            f"{k.median_delta:+.4f} & {100 * k.frac_quantum_better:.0f}\\% \\\\")
    return "\n".join(lines)


def shots_table() -> str:
    acc = pd.read_csv("results/v4/shots/accuracy_stability.csv")
    acc = acc[(acc.model == "svc") & (acc.split == "ood_test")].groupby("shots").mean_abs_dev.mean()
    geo = pd.read_csv("results/v4/shots/geometry_stability.csv").set_index("shots")
    sel = pd.read_csv("results/v4/shots/selection_stability_summary.csv")
    sel = sel[sel.model == "svc"].set_index("shots")
    lines = []
    for s in (128, 512, 2048, 8192):
        lines.append(
            f"{s} & {acc.loc[s]:.4f} & {geo.loc[s].kta_ood_dev_median:.4f} & "
            f"{geo.loc[s].eff_rank_ratio_median:.2f} & "
            f"{100 * sel.loc[s].frac_same_selection:.0f}\\% & "
            f"{sel.loc[s].mean_regret:+.4f} \\\\")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    writes = {
        "table_headline.tex": headline_table(),
        "table_summary.tex": summary_row(),
        "table_mechanism.tex": mechanism_table(),
        "table_shots.tex": shots_table(),
    }
    for name, body in writes.items():
        (OUT / name).write_text(body + "\n", encoding="utf-8")
        print(f"[ok] {name}")


if __name__ == "__main__":
    main()

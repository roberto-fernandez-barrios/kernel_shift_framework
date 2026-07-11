# scripts/reporting/make_v2_tables.py
"""
Generate the LaTeX tables of the v2 manuscript from the frozen result CSVs.

Outputs booktabs table bodies under results/tables_v2/ so that every number
in the paper is traceable to a script run over the public artifact.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("results/tables_v2")

FAMILY_LABELS = {"classical_orig": "linear+RBF", "classical_ext": "extended classical"}
MODEL_LABELS = {"svc": "SVC", "gpc": "GP classifier"}

COMPARISONS = {
    "ember": ("results/ember_shift/extended_kernels/family_comparison_wilcoxon.csv", 18),
    "netflow": ("results/netflow/family_comparison_wilcoxon.csv", 54),
    "sweep_ember": ("results/ember_shift/bandwidth_sweep/family_comparison_wilcoxon.csv", 18),
    "sweep_netflow": ("results/netflow_bandwidth_sweep/family_comparison_wilcoxon.csv", 27),
}


def fmt_p(p: float) -> str:
    if p >= 0.01:
        return f"${p:.2f}$"
    exp = int(f"{p:.0e}".split("e")[1])
    mant = p / (10 ** exp)
    return f"${mant:.1f}\\times 10^{{{exp}}}$"


def family_table(name: str, csv_path: str, n: int) -> str:
    df = pd.read_csv(csv_path)
    lines = []
    for view, view_label in [("ood", "Best-by-OOD"), ("id", "Best-by-ID")]:
        first = True
        for model in ["svc", "gpc"]:
            for ref in ["classical_orig", "classical_ext"]:
                r = df[(df.model == model) & (df.view == view) & (df.reference == ref)]
                if r.empty:
                    continue
                r = r.iloc[0]
                view_cell = f"\\multirow{{4}}{{*}}{{{view_label}}}" if first else ""
                first = False
                lines.append(
                    f"{view_cell} & {MODEL_LABELS[model]} & {FAMILY_LABELS[ref]} & "
                    f"{int(r.wins)}/{int(r.n_settings)} & "
                    f"${r.mean_delta:+.4f}$ & ${r.median_delta:+.4f}$ & "
                    f"{fmt_p(r.wilcoxon_p_holm)} & "
                    f"{int(r.effect_gt1_quantum)} \\\\"
                )
        lines.append("\\midrule" if view == "ood" else "")
    body = "\n".join(l for l in lines if l)
    return body


def mechanism_table() -> str:
    df = pd.read_csv("results/mechanism/mechanism_summary_by_dataset.csv")
    label = {
        "ember": "EMBER", "toniot_scanning": "ToN-IoT Scanning",
        "unsw_dos": "UNSW-NB15 DoS", "unsw_recon": "UNSW-NB15 Recon.",
    }
    lines = []
    for _, r in df.iterrows():
        lines.append(
            f"{label[r.dataset]} & {MODEL_LABELS[r.model]} & {int(r.n_settings)} & "
            f"{r.rho_eff_rank_median:.2f} & {100*r.rho_eff_rank_pos_frac:.0f}\\% & "
            f"{r.rho_kta_ood_median:.2f} & {100*r.rho_kta_ood_pos_frac:.0f}\\% \\\\"
        )
    return "\n".join(lines)


def dose_response_table() -> str:
    df = pd.read_csv("results/kernel_geometry/ext_classical_geometry_summary.csv")
    label = {
        "linear": "Linear", "poly2": "Polynomial (2)", "poly3": "Polynomial (3)",
        "rbf_gscale": "RBF (scale)", "matern25_med": "Mat\\'ern $5/2$",
        "matern15_med": "Mat\\'ern $3/2$", "laplacian_med": "Laplacian",
    }
    lines = []
    for _, r in df.iterrows():
        lines.append(
            f"{label[r.kernel]} & {r.eff_rank_mean:.2f} & "
            f"${r.kta_gain_ood_mean:+.3f}$ & {100*r.kta_gain_pos_frac:.0f}\\% \\\\"
        )
    return "\n".join(lines)


KERNEL_SHORT = {
    "linear": "Linear", "rbf_gscale": "RBF", "poly2": "Poly-2", "poly3": "Poly-3",
    "laplacian_med": "Laplacian", "matern15_med": "Mat\\'ern-3/2", "matern25_med": "Mat\\'ern-5/2",
    "zz_r1_full": "ZZ-r1", "zz_r2_full": "ZZ-r2", "pauli_xz_r1_full": "PauliXZ", "zmap_r2": "Z-map",
}
SIZE_SHORT = {"size_q1000_id500_ood500": "S", "size_q2000_id1000_ood1000": "M", "size_q4000_id1800_ood1800": "L"}
VARIANT_SHORT = {
    "m1_hist_byteent": "m1", "m2_hist_byteent": "m2",
    "unsw_dos__m2_centroid": "U-DoS/m2c", "unsw_dos__natural_cur": "U-DoS/nat",
    "unsw_recon__m2_centroid": "U-Rec/m2c", "unsw_recon__natural_cur": "U-Rec/nat",
    "toniot_scanning__m2_centroid": "T-Scan/m2c", "toniot_scanning__natural_cur": "T-Scan/nat",
}


def cfg_short(cfg: str) -> str:
    parts = cfg.split("__")
    dim = parts[-1]
    kernel = parts[0]
    suffix = ""
    for p in parts[1:-1]:
        if p.startswith("as"):
            suffix = f"@{p[2:]}"
    return f"{KERNEL_SHORT.get(kernel, kernel)}{suffix} ({dim})"


def appendix_setting_tables(name: str, by_setting_csv: str) -> None:
    df = pd.read_csv(by_setting_csv)
    for model in ["svc", "gpc"]:
        r = df[df.model == model].sort_values(["variant", "master_seed", "size_tag"])
        lines = []
        for _, row in r.iterrows():
            lines.append(
                f"{VARIANT_SHORT.get(row.variant, row.variant)} & {row.master_seed} & "
                f"{SIZE_SHORT[row.size_tag]} & "
                f"{cfg_short(row.ood_best_classical_ext_cfg)} & "
                f"{cfg_short(row.ood_best_quantum_cfg)} & "
                f"${row.ood_best_classical_ext_bacc:.3f}\\pm{row.ood_best_classical_ext_std:.3f}$ & "
                f"${row.ood_best_quantum_bacc:.3f}\\pm{row.ood_best_quantum_std:.3f}$ & "
                f"${row.ood_delta_q_vs_ext:+.3f}$ & ${row.ood_effect_q_vs_ext:+.2f}$ \\\\"
            )
        out = OUT / f"appendix_{name}_{model}.tex"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[✓] {out.name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table_dose_response.tex").write_text(dose_response_table() + "\n", encoding="utf-8")
    print("[✓] table_dose_response.tex")
    appendix_setting_tables("ember", "results/ember_shift/extended_kernels/family_comparison_by_setting.csv")
    appendix_setting_tables("netflow", "results/netflow/family_comparison_by_setting.csv")
    for name, (csv_path, n) in COMPARISONS.items():
        p = Path(csv_path)
        if not p.exists():
            print(f"[skip] {name}: missing {csv_path}")
            continue
        body = family_table(name, csv_path, n)
        (OUT / f"table_family_{name}.tex").write_text(body + "\n", encoding="utf-8")
        print(f"[✓] table_family_{name}.tex")
    (OUT / "table_mechanism.tex").write_text(mechanism_table() + "\n", encoding="utf-8")
    print("[✓] table_mechanism.tex")


if __name__ == "__main__":
    main()

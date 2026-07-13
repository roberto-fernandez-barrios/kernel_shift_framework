# scripts/analysis/preprocessing_ablation_analysis.py
"""
Preprocessing ablation (Phase B2): do the classical-kernel conclusions depend
on the angular [0, pi] representation the quantum circuits require?

Per run dir that has summary_classical_plainrep.csv, compares against the
frozen angular rows of the same run:
  * rank stability: Spearman correlation between the classical kernels'
    OOD-accuracy ordering under the plain vs angular representation,
    per (setting, model, dim);
  * per-kernel OOD balanced-accuracy delta (plain - angular);
  * geometry: train effective rank per kernel under both representations
    (plain from geometry_plainrep.csv; angular from the frozen
    kernel-geometry CSVs where available).

Output: results/preproc_ablation/{rank_stability.csv, kernel_deltas.csv,
geometry_comparison.csv} plus a console digest.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUN_RE = re.compile(r"(?P<setting>.+)__qs(?P<qs>\d+)__s(?P<seed>\d+)$")
ROOTS = [Path("results/ember_shift/extended_kernels"), Path("results/netflow/extended_kernels")]


def main() -> None:
    stab_rows, delta_rows, geo_rows = [], [], []
    for root in ROOTS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            m = RUN_RE.match(d.name)
            fp = d / "summary_classical_plainrep.csv"
            if not m or not fp.exists():
                continue
            plain = pd.read_csv(fp)
            plain = plain[plain.split == "ood_test"]
            ang = pd.read_csv(d / "extended_kernels_qsplits__summary.csv")
            ang = ang[(ang.split == "ood_test") & (ang.family == "classical_ext")]
            ang = ang[ang.kernel.isin(plain.kernel.unique())]
            merged = plain.merge(ang, on=["model", "dim", "kernel"],
                                 suffixes=("_plain", "_ang"))
            for (model, dim), g in merged.groupby(["model", "dim"]):
                if len(g) < 4:
                    continue
                rho, _ = stats.spearmanr(g.balanced_accuracy_plain, g.balanced_accuracy_ang)
                stab_rows.append({"setting": m.group("setting"), "qs": int(m.group("qs")),
                                  "seed": int(m.group("seed")), "model": model, "dim": dim,
                                  "rank_rho": float(rho)})
            dd = merged.groupby(["model", "kernel"], as_index=False).apply(
                lambda g: pd.Series({
                    "delta_ood": float((g.balanced_accuracy_plain - g.balanced_accuracy_ang).mean())}),
                include_groups=False)
            dd["setting"], dd["qs"], dd["seed"] = m.group("setting"), int(m.group("qs")), int(m.group("seed"))
            delta_rows.append(dd)
            fg = d / "geometry_plainrep.csv"
            if fg.exists():
                g = pd.read_csv(fg)
                g["setting"], dd_qs = m.group("setting"), int(m.group("qs"))
                g["qs"], g["seed"] = dd_qs, int(m.group("seed"))
                geo_rows.append(g)

    out = Path("results/preproc_ablation")
    out.mkdir(parents=True, exist_ok=True)
    stab = pd.DataFrame(stab_rows)
    deltas = pd.concat(delta_rows, ignore_index=True) if delta_rows else pd.DataFrame()
    geo = pd.concat(geo_rows, ignore_index=True) if geo_rows else pd.DataFrame()
    stab.to_csv(out / "rank_stability.csv", index=False)
    deltas.to_csv(out / "kernel_deltas.csv", index=False)
    geo.to_csv(out / "geometry_comparison.csv", index=False)

    if not stab.empty:
        print("Rank stability (Spearman plain vs angular OOD ordering):")
        print(stab.groupby("model").rank_rho.describe()[["count", "50%", "25%", "75%"]].round(3).to_string())
    if not deltas.empty:
        print("\nMean OOD delta (plain - angular) per kernel:")
        print(deltas.groupby(["model", "kernel"]).delta_ood.mean().round(4).to_string())
    if not geo.empty:
        print("\nPlain-representation train effective rank (mean per kernel, dim=12):")
        g12 = geo[geo.dim == geo.dim.max()]
        print(g12.groupby("kernel").spec_train_eff_rank.mean().round(2).to_string())


if __name__ == "__main__":
    main()

# scripts/analysis/csens_analysis.py
"""
SVC regularization sensitivity (Phase B3/C): does the kernel-family verdict
depend on the fixed C=1 operating point?

Per run dir with summary_csens.csv, pools the C grid with the frozen C=1 rows
and reports:
  * ordering stability: Spearman correlation between the kernels' OOD ordering
    at C=1 and at each other C, per (setting, model=svc, dim);
  * a C-tuned deployment protocol (P1C): per run and family, select
    (kernel, dim, C) jointly by ID-test balanced accuracy, report OOD —
    the family comparison when C is tuned instead of fixed;
  * per-family best-OOD as a function of C (descriptive curves).

Output: results/csens/{ordering_stability.csv, p1c_by_setting.csv,
family_curves.csv} + console digest.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUN_RE = re.compile(r"(?P<setting>.+)__qs(?P<qs>\d+)__s(?P<seed>\d+)$")
ROOTS = [Path("results/ember_shift/extended_kernels"), Path("results/netflow/extended_kernels")]
ORIG_PREFIXES = ("linear", "rbf_gscale")


def fam_of(row_family: str, kernel: str) -> str:
    return "quantum" if row_family == "quantum" else "classical_ext"


def load_run(d: Path) -> pd.DataFrame | None:
    fc = d / "summary_csens.csv"
    if not fc.exists():
        return None
    cs = pd.read_csv(fc)
    base = pd.read_csv(d / "extended_kernels_qsplits__summary.csv")
    base = base[base.model == "svc"].copy()
    base["svc_c"] = 1.0
    cols = ["family", "model", "dim", "cfg", "kernel", "split", "svc_c", "balanced_accuracy"]
    return pd.concat([cs[cols], base[cols]], ignore_index=True)


def main() -> None:
    ord_rows, p1c_rows, curve_rows = [], [], []
    for root in ROOTS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            m = RUN_RE.match(d.name)
            if not m:
                continue
            df = load_run(d)
            if df is None:
                continue
            meta = {"setting": m.group("setting"), "qs": int(m.group("qs")),
                    "seed": int(m.group("seed"))}
            ood = df[df.split == "ood_test"]
            # ordering stability vs C=1
            for dim, g in ood.groupby("dim"):
                piv = g.pivot_table(index="kernel", columns="svc_c", values="balanced_accuracy")
                if 1.0 not in piv.columns:
                    continue
                for c in piv.columns:
                    if c == 1.0 or piv[c].isna().all():
                        continue
                    rho, _ = stats.spearmanr(piv[1.0], piv[c], nan_policy="omit")
                    ord_rows.append({**meta, "dim": dim, "svc_c": float(c),
                                     "rho_vs_C1": float(rho), "n_kernels": piv[c].notna().sum()})
            # P1C: joint (kernel, dim, C) selection on ID per family
            w = df.pivot_table(index=["family", "kernel", "dim", "svc_c"],
                               columns="split", values="balanced_accuracy").reset_index()
            w["fam"] = [fam_of(f, k) for f, k in zip(w.family, w.kernel)]
            for fam, g in w.groupby("fam"):
                sel = g.loc[g.id_test.idxmax()]
                p1c_rows.append({**meta, "fam": fam, "sel_c": float(sel.svc_c),
                                 "sel_kernel": sel.kernel, "p1c_ood": float(sel.ood_test),
                                 "p1c_id": float(sel.id_test)})
            # descriptive: best OOD per family per C
            for (fam, c), g in ood.assign(fam=[fam_of(f, k) for f, k in zip(ood.family, ood.kernel)]) \
                                  .groupby(["fam", "svc_c"]):
                curve_rows.append({**meta, "fam": fam, "svc_c": float(c),
                                   "best_ood": float(g.balanced_accuracy.max())})

    out = Path("results/csens")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ord_rows).to_csv(out / "ordering_stability.csv", index=False)
    p1c = pd.DataFrame(p1c_rows)
    p1c.to_csv(out / "p1c_by_setting.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "family_curves.csv", index=False)

    if ord_rows:
        o = pd.DataFrame(ord_rows)
        print("Kernel-ordering stability vs C=1 (median Spearman rho by C):")
        print(o.groupby("svc_c").rho_vs_C1.median().round(3).to_string())
    if not p1c.empty and {"quantum", "classical_ext"} <= set(p1c.fam.unique()):
        runmean = p1c.groupby(["setting", "fam"], as_index=False)[["p1c_ood"]].mean()
        piv = runmean.pivot(index="setting", columns="fam", values="p1c_ood")
        delta = (piv["quantum"] - piv["classical_ext"]).dropna()
        print(f"\nP1C (C-tuned deployment) quantum - classical_ext: "
              f"mean {delta.mean():+.4f}, wins {(delta > 0).sum()}/{len(delta)}")
        print("Selected C distribution (quantum / classical):")
        print(p1c.groupby(["fam", "sel_c"]).size().to_string())


if __name__ == "__main__":
    main()

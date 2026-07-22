# scripts/analysis/honest_family_comparison_v4.py
"""
v4 honest family comparison (the manuscript's headline table source).

For every run, family, and classifier, selects one configuration by the P1'
deployment protocol (best ID-validation balanced accuracy; each config already
at its internally CV-tuned C) and reports the selected config's ID-test and
OOD-test balanced accuracy. Aggregates the quantum-minus-classical delta per
scenario-group against two references:

  classical_orig  = linear + RBF(scale) variants  (the customary weak baseline)
  classical_ext   = all 23 classical geometries    (the geometry-matched family)

Also emits the oracle (Best-by-OOD / Best-by-ID) columns for the appendix, so
the reader can see exactly how much of the apparent advantage was selection
optimism. No p-values (spec constraint 1); this feeds the conditional-interval
estimator (hierarchical_effect_estimation.py) for uncertainty.

Output -> results/v4/family_comparison/
"""
from __future__ import annotations

import argparse
import os
import re
from glob import glob

import numpy as np
import pandas as pd

ORIG_PREFIX = ("linear", "rbf_gscale")


def group_label(run: str) -> str:
    tok = run.split("__")
    if tok[0].startswith(("m1_", "m2_")):
        return f"ember_{tok[0].split('_')[0]}"
    return f"{tok[0]}_{tok[1]}"


def select(sub: pd.DataFrame, by: str, evalcol: str) -> float:
    return float(sub.loc[sub[by].idxmax()][evalcol])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "results/ember_shift/extended_kernels", "results/netflow/extended_kernels"])
    ap.add_argument("--out-dir", default="results/v4/family_comparison")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for root in args.roots:
        for f in sorted(glob(os.path.join(root, "*", "summary_v4.csv"))):
            run = os.path.basename(os.path.dirname(f))
            grp = group_label(run)
            qs = int(re.search(r"qs(\d+)", run).group(1))
            seed = int(re.search(r"__s(\d+)$", run).group(1))
            s = pd.read_csv(f)
            for model in ("svc", "gpc"):
                sm = s[s.model == model]
                w = sm.pivot_table(index="cfg", columns="split",
                                   values="balanced_accuracy")
                meta = sm.drop_duplicates("cfg").set_index("cfg")
                w["kernel"], w["family"] = meta.kernel, meta.family
                fams = {
                    "classical_orig": w[w.kernel.str.startswith(ORIG_PREFIX)],
                    "classical_ext": w[w.family == "classical_ext"],
                    "quantum": w[w.family == "quantum"],
                }
                rec = {"group": grp, "run": run, "qs": qs, "seed": seed, "model": model}
                for name, sub in fams.items():
                    if sub.empty:
                        continue
                    rec[f"{name}__p1_ood"] = select(sub, "id_val", "ood_test")
                    rec[f"{name}__p1_idtest"] = select(sub, "id_val", "id_test")
                    rec[f"{name}__oracle_ood"] = select(sub, "ood_test", "ood_test")
                    rec[f"{name}__oracle_idtest"] = select(sub, "id_test", "id_test")
                rows.append(rec)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(args.out_dir, "selected_by_run.csv"), index=False)

    # per-run deltas -> GATE 2 interface files
    for ref in ("classical_orig", "classical_ext"):
        d = runs[["group", "run", "qs", "seed", "model"]].copy()
        d["p1_ood_quantum"] = runs["quantum__p1_ood"]
        d["p1_ood_classical"] = runs[f"{ref}__p1_ood"]
        d["setting"] = runs.run.str.replace(r"__qs\d+__s\d+$", "", regex=True)
        d["delta"] = d.p1_ood_quantum - d.p1_ood_classical
        d.to_csv(os.path.join(args.out_dir, f"p1_runs__vs_{ref}.csv"), index=False)

    # group summary (descriptive means; intervals come from the GATE 2 estimator)
    summ = []
    for (grp, model), g in runs.groupby(["group", "model"]):
        row = {"group": grp, "model": model, "n_runs": len(g)}
        for ref in ("classical_orig", "classical_ext"):
            row[f"p1_ood_delta_vs_{ref}"] = float(
                (g["quantum__p1_ood"] - g[f"{ref}__p1_ood"]).mean())
            row[f"p1_idtest_delta_vs_{ref}"] = float(
                (g["quantum__p1_idtest"] - g[f"{ref}__p1_idtest"]).mean())
            row[f"oracle_ood_delta_vs_{ref}"] = float(
                (g["quantum__oracle_ood"] - g[f"{ref}__oracle_ood"]).mean())
        summ.append(row)
    ss = pd.DataFrame(summ)
    ss.to_csv(os.path.join(args.out_dir, "group_summary.csv"), index=False)

    print("Honest P1' OOD delta (quantum - reference), by group:")
    show = ss[["group", "model", "p1_ood_delta_vs_classical_orig",
               "p1_ood_delta_vs_classical_ext"]]
    print(show.round(4).to_string(index=False))
    print("\nDataset-equal-weighted means (descriptive):")
    ds = ss.assign(dataset=ss.group.map(
        lambda x: "ember" if x.startswith("ember") else x.rsplit("_m", 1)[0].rsplit("_natural", 1)[0]))
    for ref in ("classical_orig", "classical_ext"):
        for metric in (f"p1_ood_delta_vs_{ref}", f"p1_idtest_delta_vs_{ref}"):
            m = ds.groupby("model").apply(
                lambda d: d.groupby("dataset")[metric].mean().mean(), include_groups=False)
            print(f"  {metric}: svc {m['svc']:+.4f}, gpc {m['gpc']:+.4f}")


if __name__ == "__main__":
    main()

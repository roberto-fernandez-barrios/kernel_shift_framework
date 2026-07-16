# scripts/analysis/mechanism_crossfit_analysis.py
"""
Non-circular mechanism controls from the ktanull cross-fit (Phase A3/C).

Each run dir carries mechanism_crossfit.csv, one row per (kernel, dim), with:
  kta_ood_full          OOD KTA on the full OOD split (uses OOD labels)
  kta_null_*            label-permutation null for that KTA (mean/std, and
                        exceed_frac = P(|null| >= |observed|))
  kta_ood_halfA         OOD KTA measured on OOD half A only
  eff_rank_train        train Gram effective rank (label-free)
  bacc_{svc,gpc}_halfB  balanced accuracy on OOD half B (disjoint from A)

Two controls the reviewer asked for:
  1. Permutation null: is the observed OOD alignment above what random labels
     give? We report the median exceed_frac (near 0 = alignment is real).
  2. Cross-fitting: KTA on half A vs accuracy on the disjoint half B removes the
     circularity of correlating KTA and accuracy on the same labels. We report
     within-(run, dim) Spearman rho(kta_ood_halfA, bacc_halfB) and the
     label-free rho(eff_rank_train, bacc_halfB), per scope, by dataset group.

Output: results/mechanism_crossfit/{unit_correlations.csv, summary_by_group.csv,
null_summary.csv} plus a console digest.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUN_RE = re.compile(r"(?P<setting>.+)__qs(?P<qs>\d+)__s(?P<seed>\d+)$")
ROOTS = [Path("results/ember_shift/extended_kernels"), Path("results/netflow/extended_kernels")]


def group_label(setting: str) -> str:
    toks = setting.split("__")
    if toks[0].startswith(("m1_", "m2_")):
        return f"ember_{toks[0].split('_')[0]}"
    return f"{toks[0]}_{toks[1]}"


def scope_mask(g: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all":
        return pd.Series(True, index=g.index)
    if scope == "classical":
        return g.family.str.startswith("classical")
    return g.family == "quantum"


def load() -> pd.DataFrame:
    frames = []
    for root in ROOTS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            m = RUN_RE.match(d.name)
            f = d / "mechanism_crossfit.csv"
            if not m or not f.exists():
                continue
            df = pd.read_csv(f)
            df["setting"] = m.group("setting")
            df["qs"] = int(m.group("qs"))
            df["seed"] = int(m.group("seed"))
            df["group"] = group_label(m.group("setting"))
            frames.append(df)
    if not frames:
        raise SystemExit("No mechanism_crossfit.csv found")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = load()

    # 1. permutation-null significance of the observed OOD alignment
    null_rows = []
    for (grp, scope), g in [((grp, sc), df[(df.group == grp) & scope_mask(df, sc)]) for grp in
                            df.group.unique() for sc in ("all", "classical", "quantum")]:
        if g.empty:
            continue
        null_rows.append({
            "group": grp, "scope": scope, "n_cells": len(g),
            "median_kta_ood": float(g.kta_ood_full.median()),
            "median_null_mean": float(g.kta_null_mean.median()),
            "median_exceed_frac": float(g.kta_null_exceed_frac.median()),
            "frac_cells_significant": float((g.kta_null_exceed_frac < 0.05).mean())})
    null_summary = pd.DataFrame(null_rows)

    # 2. cross-fitted (non-circular) within-unit correlations
    unit_rows = []
    for (setting, qs, seed), g in df.groupby(["setting", "qs", "seed"]):
        for dim, gd in g.groupby("dim"):
            for scope in ("all", "classical", "quantum"):
                sub = gd[scope_mask(gd, scope)]
                if sub.kernel.nunique() < 3:
                    continue
                rec = {"setting": setting, "qs": qs, "seed": seed, "dim": dim,
                       "scope": scope, "group": group_label(setting), "n_kernels": len(sub)}
                for model in ("svc", "gpc"):
                    b = sub[f"bacc_{model}_halfB"]
                    rec[f"rho_ktaA_{model}"] = float(
                        stats.spearmanr(sub.kta_ood_halfA, b).statistic)
                    rec[f"rho_rank_{model}"] = float(
                        stats.spearmanr(sub.eff_rank_train, b).statistic)
                unit_rows.append(rec)
    units = pd.DataFrame(unit_rows)

    out = Path("results/mechanism_crossfit")
    out.mkdir(parents=True, exist_ok=True)
    null_summary.to_csv(out / "null_summary.csv", index=False)
    units.to_csv(out / "unit_correlations.csv", index=False)

    metric_cols = [c for c in units.columns if c.startswith("rho_")]
    summary = units.groupby(["group", "scope"]).agg(
        n_units=("dim", "size"),
        **{f"median_{c}": (c, "median") for c in metric_cols},
        **{f"fracpos_{c}": (c, lambda s: float((s > 0).mean())) for c in metric_cols},
    ).reset_index()
    summary.to_csv(out / "summary_by_group.csv", index=False)

    print("Permutation null (scope=all): median observed OOD KTA vs null, and "
          "fraction of cells significant at 5%:")
    print(null_summary[null_summary.scope == "all"][
        ["group", "median_kta_ood", "median_null_mean", "median_exceed_frac",
         "frac_cells_significant"]].round(4).to_string(index=False))
    print("\nCross-fit rho(kta_ood_halfA, bacc_halfB) — KTA and accuracy on "
          "DISJOINT OOD halves (scope=all):")
    print(summary[summary.scope == "all"][
        ["group", "n_units", "median_rho_ktaA_svc", "fracpos_rho_ktaA_svc",
         "median_rho_ktaA_gpc", "fracpos_rho_ktaA_gpc"]].round(3).to_string(index=False))
    print("\nLabel-free rho(eff_rank_train, bacc_halfB) (scope=all):")
    print(summary[summary.scope == "all"][
        ["group", "median_rho_rank_svc", "fracpos_rho_rank_svc",
         "median_rho_rank_gpc", "fracpos_rho_rank_gpc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

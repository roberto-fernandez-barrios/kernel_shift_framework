# scripts/analysis/mechanism_controls.py
"""
Circularity controls for the geometry-mechanism analysis.

The original analysis correlates kta_ood (computed WITH OOD labels) against
OOD balanced accuracy (evaluated on the same labels). This script separates:

  Diagnostic predictors (use OOD labels)  : kta_ood, kta_drop_id_to_ood
  Honest predictors (train/ID data only)  : kta_train, kta_id,
                                            spec_train_eff_rank
  Label-free novelty                      : ood_novelty_ratio

and reports within-setting Spearman correlations with OOD balanced accuracy
per scope (all kernels / classical only / quantum only), aggregated by
dataset group. If the honest predictors carry signal, the mechanism has a
predictive component; kta_ood alone is reported as diagnostic association.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

GEO_SOURCES = [
    ("ember", Path("results/kernel_geometry/grid")),
    ("netflow", Path("results/netflow/kernel_geometry")),
]
ACC_ROOTS = {
    "ember": Path("results/ember_shift/extended_kernels"),
    "netflow": Path("results/netflow/extended_kernels"),
}
PREDICTORS = {
    "kta_ood": "diagnostic",
    "kta_drop_id_to_ood": "diagnostic",
    "kta_train": "honest",
    "kta_id": "honest",
    "spec_train_eff_rank": "honest",
    "ood_novelty_ratio": "label_free",
}
GEO_RE = re.compile(r"kernel_geometry__(?P<setting>.+?)__qs(?P<qs>\d+)(?:__s\d+)?\.csv$")


def group_label(setting: str) -> str:
    toks = setting.split("__")
    if toks[0].startswith(("m1_", "m2_")):
        return f"ember_{toks[0].split('_')[0]}"
    return f"{toks[0]}_{toks[1]}"


def load_geometry() -> pd.DataFrame:
    frames = []
    for src, root in GEO_SOURCES:
        for f in sorted(root.glob("kernel_geometry__*.csv")):
            m = GEO_RE.search(f.name)
            if not m:
                continue
            df = pd.read_csv(f, usecols=lambda c: c in
                             {"dim", "kernel", "family"} | set(PREDICTORS))
            df["setting"] = m.group("setting")
            df["qs"] = int(m.group("qs"))
            df["source"] = src
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_accuracy(source: str, settings: set[str]) -> pd.DataFrame:
    root = ACC_ROOTS[source]
    frames = []
    for d in sorted(root.iterdir()):
        m = re.match(r"(.+)__qs(\d+)__s(\d+)$", d.name)
        f = d / "extended_kernels_qsplits__summary.csv"
        if not m or m.group(1) not in settings or not f.exists():
            continue
        df = pd.read_csv(f, usecols=["model", "dim", "kernel", "split", "balanced_accuracy"])
        df = df[df.split == "ood_test"].drop(columns="split")
        df["setting"], df["qs"] = m.group(1), int(m.group(2))
        frames.append(df)
    acc = pd.concat(frames, ignore_index=True)
    return acc.groupby(["setting", "qs", "model", "kernel", "dim"], as_index=False)[
        "balanced_accuracy"].mean()  # mean over model seeds


def scope_mask(g: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all":
        return pd.Series(True, index=g.index)
    if scope == "classical":
        return g.family.str.startswith("classical")
    return g.family == "quantum"


def main() -> None:
    geo = load_geometry()
    out_rows = []
    for source in geo.source.unique():
        gsrc = geo[geo.source == source]
        acc = load_accuracy(source, set(gsrc.setting.unique()))
        merged = gsrc.merge(acc, on=["setting", "qs", "kernel", "dim"], how="inner")
        for (setting, qs, model), g in merged.groupby(["setting", "qs", "model"]):
            for scope in ("all", "classical", "quantum"):
                sub = g[scope_mask(g, scope)]
                if sub.kernel.nunique() < 3:
                    continue
                for pred, kind in PREDICTORS.items():
                    if pred not in sub or sub[pred].isna().all():
                        continue
                    rho, _ = stats.spearmanr(sub[pred], sub.balanced_accuracy)
                    out_rows.append({"source": source, "setting": setting, "qs": qs,
                                     "model": model, "scope": scope, "predictor": pred,
                                     "kind": kind, "rho": float(rho), "n_cells": len(sub),
                                     "group": group_label(setting)})
    res = pd.DataFrame(out_rows)
    out = Path("results/mechanism_controls")
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "unit_correlations.csv", index=False)

    summary = res.groupby(["group", "model", "scope", "predictor", "kind"]).agg(
        n_units=("rho", "size"), median_rho=("rho", "median"),
        frac_pos=("rho", lambda s: float((s > 0).mean()))).reset_index()
    summary.to_csv(out / "summary_by_group.csv", index=False)

    console = summary[summary.scope == "all"].pivot_table(
        index=["group", "model"], columns="predictor", values="median_rho")
    print("Median within-unit Spearman rho vs OOD balanced accuracy (scope=all):")
    print(console.round(2).to_string())
    print("\nHonest predictors, frac(rho>0) by scope:")
    hon = summary[summary.kind == "honest"].pivot_table(
        index=["group", "model"], columns=["predictor", "scope"], values="frac_pos")
    print(hon.round(2).to_string())


if __name__ == "__main__":
    main()

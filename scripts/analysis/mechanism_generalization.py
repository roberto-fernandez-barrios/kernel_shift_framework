# scripts/analysis/mechanism_generalization.py
"""
Cross-dataset test of the mechanistic hypothesis: within each setting, do the
kernel-geometry descriptors (effective rank of the train kernel, OOD kernel-
target alignment) predict which kernel cells generalize better OOD?

For every setting and classifier we join, per (kernel, dim) cell, the geometry
descriptors with the OOD balanced accuracy of the extended-kernels runner,
compute within-setting Spearman correlations, and summarize their distribution
per dataset. A mechanism that holds across EMBER (static PE malware) and the
network-flow scenarios (UNSW-NB15, ToN-IoT; synthetic and natural drift) is a
dataset-general regularity, not an EMBER artifact.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

RX_EXT = re.compile(r"^(?P<setting>.+?__ms\d+__q\d+_id\d+_ood\d+__qs\d+)(?:__s(?P<mseed>\d+))?$")


def load_geometry(dirs: List[Path]) -> pd.DataFrame:
    frames = []
    for d in dirs:
        for p in sorted(d.glob("kernel_geometry__*.csv")):
            df = pd.read_csv(p)
            # normalize label to the qs level (geometry may or may not carry __s)
            df["setting"] = df["setting_label"].str.replace(r"__s\d+$", "", regex=True)
            frames.append(df[["setting", "kernel", "dim", "spec_train_eff_rank", "kta_ood", "kta_train"]])
    geo = pd.concat(frames, ignore_index=True)
    return geo.drop_duplicates(subset=["setting", "kernel", "dim"], keep="first")


def load_extended(root: Path, mseed: int = 42) -> pd.DataFrame:
    """All q-split seeds are used (geometry exists per q-split); the model seed
    is fixed to the one the geometry embeddings were computed with."""
    frames = []
    for p in sorted(root.glob("*/extended_kernels_qsplits__summary.csv")):
        m = RX_EXT.match(p.parent.name)
        if not m:
            continue
        if m.group("mseed") is not None and int(m.group("mseed")) != mseed:
            continue
        df = pd.read_csv(p)
        df = df[df.split == "ood_test"]
        df["setting"] = m.group("setting")
        frames.append(df[["setting", "kernel", "dim", "model", "balanced_accuracy"]])
    return pd.concat(frames, ignore_index=True)


def dataset_of(setting: str) -> str:
    if setting.startswith("m1_hist") or setting.startswith("m2_hist"):
        return "ember"
    return setting.split("__")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results/mechanism"))
    args = ap.parse_args()

    geo = load_geometry([
        Path("results/kernel_geometry/grid"),
        Path("results/kernel_geometry/grid_ext_classical"),
        Path("results/netflow/kernel_geometry"),
    ])
    ext = pd.concat([
        load_extended(Path("results/ember_shift/extended_kernels")),
        load_extended(Path("results/netflow/extended_kernels")),
    ], ignore_index=True)

    joined = ext.merge(geo, on=["setting", "kernel", "dim"], how="inner")
    joined["dataset"] = joined["setting"].map(dataset_of)

    rows: List[Dict] = []
    for (setting, model), g in joined.groupby(["setting", "model"]):
        if g.shape[0] < 10:
            continue
        rho_rank, _ = stats.spearmanr(g.spec_train_eff_rank, g.balanced_accuracy)
        rho_kta, _ = stats.spearmanr(g.kta_ood, g.balanced_accuracy)
        rows.append({
            "setting": setting, "model": model, "dataset": dataset_of(setting),
            "n_cells": int(g.shape[0]),
            "rho_eff_rank": float(rho_rank),
            "rho_kta_ood": float(rho_kta),
        })
    res = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_dir / "mechanism_within_setting_correlations.csv", index=False)

    pd.set_option("display.width", 250)
    print("=== Within-setting Spearman(geometry, OOD balanced accuracy), by dataset ===")
    summary = res.groupby(["dataset", "model"]).agg(
        n_settings=("setting", "nunique"),
        cells_per_setting=("n_cells", "median"),
        rho_eff_rank_median=("rho_eff_rank", "median"),
        rho_eff_rank_pos_frac=("rho_eff_rank", lambda s: (s > 0).mean()),
        rho_kta_ood_median=("rho_kta_ood", "median"),
        rho_kta_ood_pos_frac=("rho_kta_ood", lambda s: (s > 0).mean()),
    ).round(3)
    print(summary.to_string())
    summary.to_csv(args.out_dir / "mechanism_summary_by_dataset.csv")
    print(f"\n[✓] Wrote {args.out_dir}")


if __name__ == "__main__":
    main()

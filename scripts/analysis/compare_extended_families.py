# scripts/analysis/compare_extended_families.py
"""
Family-level comparison for the extended-kernels experiments (workstream B).

For each of the 18 principal settings and each classifier (SVC, Laplace-GPC),
apply the paper's family-internal Best-by-OOD / Best-by-ID selection to three
kernel families:

  - classical_orig: linear, rbf_gscale            (the paper's baseline set)
  - classical_ext:  + poly2/3, laplacian, matern  (the QMI-requested set)
  - quantum:        the four fidelity maps

Runs are aggregated first: per (setting, cfg, split) the balanced accuracy is
averaged over all available (qsplit seed, model seed) runs, and selection
operates on the run means — the same logic as the paper. Deltas are reported
with the paper's effect-size proxy delta / sqrt(std_q^2 + std_c^2).

Also summarizes GPC uncertainty behavior under shift (ECE and predictive
entropy, ID vs OOD) for the Best-by-OOD configuration of each family.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

CLASSICAL_ORIG = ["linear", "rbf_gscale"]
CLASSICAL_EXT = ["linear", "rbf_gscale", "poly2", "poly3", "laplacian_med", "matern15_med", "matern25_med"]
QUANTUM = ["zz_r1_full", "zz_r2_full", "pauli_xz_r1_full", "zmap_r2"]

RX_LABEL = re.compile(
    r"^(?P<variant>.+?)__ms(?P<ms>\d+)__q(?P<qtag>\d+_id\d+_ood\d+)__qs(?P<qs>\d+)(?:__s(?P<mseed>\d+))?$"
)

PROB_METRICS = ["log_loss", "brier", "ece", "mean_predictive_entropy"]


def load_all(root: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(root.glob("*/extended_kernels_qsplits__summary.csv")):
        m = RX_LABEL.match(p.parent.name)
        if not m:
            continue
        df = pd.read_csv(p)
        df["variant"] = m.group("variant")
        df["master_seed"] = int(m.group("ms"))
        df["size_tag"] = "size_q" + m.group("qtag")
        df["qsplit_seed"] = int(m.group("qs"))
        df["model_seed"] = int(m.group("mseed") or 42)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No summaries found under {root}")
    return pd.concat(frames, ignore_index=True)


def aggregate_runs(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant", "master_seed", "size_tag", "model", "kernel", "dim", "cfg", "split"]
    agg = data.groupby(keys, as_index=False).agg(
        bacc_mean=("balanced_accuracy", "mean"),
        bacc_std=("balanced_accuracy", "std"),
        n_runs=("balanced_accuracy", "size"),
        **{f"{m}_mean": (m, "mean") for m in PROB_METRICS},
    )
    agg["bacc_std"] = agg["bacc_std"].fillna(0.0)
    return agg


def best_by(g: pd.DataFrame, kernels: List[str], select_split: str) -> pd.Series:
    sub = g[(g.kernel.isin(kernels)) & (g.split == select_split)]
    return sub.loc[sub.bacc_mean.idxmax()]


def metric_at(g: pd.DataFrame, cfg: str, split: str, col: str) -> float:
    r = g[(g.cfg == cfg) & (g.split == split)]
    return float(r[col].iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/ember_shift/extended_kernels"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/ember_shift/extended_kernels"))
    args = ap.parse_args()

    raw = load_all(args.root)
    n_runs_by_setting = raw.groupby(["variant", "master_seed", "size_tag"]).apply(
        lambda g: g[["qsplit_seed", "model_seed"]].drop_duplicates().shape[0], include_groups=False
    )
    data = aggregate_runs(raw)

    rows: List[Dict] = []
    for (variant, ms, size), g_setting in data.groupby(["variant", "master_seed", "size_tag"]):
        for model in sorted(g_setting.model.unique()):
            g = g_setting[g_setting.model == model]
            row: Dict = {
                "variant": variant, "master_seed": ms, "size_tag": size, "model": model,
                "n_runs": int(n_runs_by_setting.loc[(variant, ms, size)]),
            }
            for view, sel_split in [("ood", "ood_test"), ("id", "id_test")]:
                fams = {
                    "classical_orig": best_by(g, CLASSICAL_ORIG, sel_split),
                    "classical_ext": best_by(g, CLASSICAL_EXT, sel_split),
                    "quantum": best_by(g, QUANTUM, sel_split),
                }
                for fam, b in fams.items():
                    row[f"{view}_best_{fam}_cfg"] = b.cfg
                    row[f"{view}_best_{fam}_bacc"] = float(b.bacc_mean)
                    row[f"{view}_best_{fam}_std"] = float(b.bacc_std)
                for ref in ["classical_orig", "classical_ext"]:
                    short = "orig" if ref.endswith("orig") else "ext"
                    delta = row[f"{view}_best_quantum_bacc"] - row[f"{view}_best_{ref}_bacc"]
                    pooled = np.sqrt(row[f"{view}_best_quantum_std"] ** 2 + row[f"{view}_best_{ref}_std"] ** 2)
                    row[f"{view}_delta_q_vs_{short}"] = delta
                    row[f"{view}_effect_q_vs_{short}"] = delta / pooled if pooled > 0 else np.nan

                if view == "ood" and model == "gpc":
                    for fam, b in fams.items():
                        row[f"gpc_ece_id_{fam}"] = metric_at(g, b.cfg, "id_test", "ece_mean")
                        row[f"gpc_ece_ood_{fam}"] = metric_at(g, b.cfg, "ood_test", "ece_mean")
                        row[f"gpc_entropy_id_{fam}"] = metric_at(g, b.cfg, "id_test", "mean_predictive_entropy_mean")
                        row[f"gpc_entropy_ood_{fam}"] = metric_at(g, b.cfg, "ood_test", "mean_predictive_entropy_mean")
            rows.append(row)

    res = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_dir / "family_comparison_by_setting.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"Runs per setting: min={res.n_runs.min()}, max={res.n_runs.max()}\n")

    for model in sorted(res.model.unique()):
        r = res[res.model == model]
        n = len(r)
        print(f"=== {model.upper()} (n={n} settings) ===")
        for view in ["ood", "id"]:
            for short, ref in [("orig", "classical_orig"), ("ext", "classical_ext")]:
                d = r[f"{view}_delta_q_vs_{short}"]
                e = r[f"{view}_effect_q_vs_{short}"]
                print(
                    f"  Best-by-{view.upper()} quantum vs {ref:15s}: wins {int((d > 0).sum())}/{n} | "
                    f"mean Δ {d.mean():+.4f} | median Δ {d.median():+.4f} | effect>1 in {int((e.abs() > 1).sum())} "
                    f"(of which quantum-favorable {int((e > 1).sum())})"
                )
        if model == "gpc":
            print("  GPC uncertainty (Best-by-OOD configs, mean over settings):")
            for fam in ["classical_orig", "classical_ext", "quantum"]:
                d_ece = (r[f"gpc_ece_ood_{fam}"] - r[f"gpc_ece_id_{fam}"]).mean()
                d_ent = (r[f"gpc_entropy_ood_{fam}"] - r[f"gpc_entropy_id_{fam}"]).mean()
                print(f"    {fam:15s}: ECE OOD-ID {d_ece:+.4f} | entropy OOD-ID {d_ent:+.4f}")
        print()

    print("=== Best-by-OOD selected kernels (classical_ext family) ===")
    print(res.groupby("model")["ood_best_classical_ext_cfg"].value_counts().to_string())
    print(f"\n[✓] Wrote {args.out_dir / 'family_comparison_by_setting.csv'}")


if __name__ == "__main__":
    main()

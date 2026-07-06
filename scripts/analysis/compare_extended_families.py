# scripts/analysis/compare_extended_families.py
"""
Family-level comparison for the extended-kernels experiments (workstream B).

For each of the 18 principal settings and each classifier (SVC, Laplace-GPC),
apply the paper's family-internal Best-by-OOD / Best-by-ID selection to three
kernel families:

  - classical_orig: linear, rbf_gscale            (the paper's baseline set)
  - classical_ext:  + poly2/3, laplacian, matern  (the QMI-requested set)
  - quantum:        the four fidelity maps

and report per-setting winners and aggregate win counts for
quantum-vs-classical_orig (continuity check against the paper) and
quantum-vs-classical_ext (the honest test).

Also summarizes GPC uncertainty behavior under shift (ECE and predictive
entropy, ID vs OOD) for the Best-by-OOD configuration of each family.

Note: these runs use one q-split seed and one model seed per setting (no
repeated-run averaging), so wins are point comparisons, not the paper's
15-run means. Treat counts as a robustness probe, not headline numbers.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

CLASSICAL_ORIG = ["linear", "rbf_gscale"]
CLASSICAL_EXT = ["linear", "rbf_gscale", "poly2", "poly3", "laplacian_med", "matern15_med", "matern25_med"]
QUANTUM = ["zz_r1_full", "zz_r2_full", "pauli_xz_r1_full", "zmap_r2"]

RX_LABEL = re.compile(r"^(?P<variant>m\d_.+?)__ms(?P<ms>\d+)__q(?P<qtag>\d+_id\d+_ood\d+)__qs(?P<qs>\d+)$")


def load_all(root: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(root.glob("*/extended_kernels_qsplits__summary.csv")):
        label = p.parent.name
        m = RX_LABEL.match(label)
        if not m:
            continue
        df = pd.read_csv(p)
        df["variant"] = m.group("variant")
        df["master_seed"] = int(m.group("ms"))
        df["size_tag"] = "size_q" + m.group("qtag")
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No summaries found under {root}")
    return pd.concat(frames, ignore_index=True)


def best_by(df: pd.DataFrame, kernels: List[str], select_split: str) -> pd.Series:
    """Best (kernel, dim) within a family by balanced accuracy on select_split."""
    sub = df[(df.kernel.isin(kernels)) & (df.split == select_split)]
    best = sub.loc[sub.balanced_accuracy.idxmax()]
    return best


def metric_at(df: pd.DataFrame, cfg: str, split: str, col: str) -> float:
    r = df[(df.cfg == cfg) & (df.split == split)]
    return float(r[col].iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/ember_shift/extended_kernels"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/ember_shift/extended_kernels"))
    args = ap.parse_args()

    data = load_all(args.root)
    keys = ["variant", "master_seed", "size_tag"]
    rows: List[Dict] = []

    for (variant, ms, size), g_setting in data.groupby(keys):
        for model in sorted(g_setting.model.unique()):
            g = g_setting[g_setting.model == model]
            row: Dict = {"variant": variant, "master_seed": ms, "size_tag": size, "model": model}
            for view, sel_split in [("ood", "ood_test"), ("id", "id_test")]:
                fams = {
                    "classical_orig": best_by(g, CLASSICAL_ORIG, sel_split),
                    "classical_ext": best_by(g, CLASSICAL_EXT, sel_split),
                    "quantum": best_by(g, QUANTUM, sel_split),
                }
                for fam, b in fams.items():
                    row[f"{view}_best_{fam}_cfg"] = b.cfg
                    row[f"{view}_best_{fam}_bacc"] = float(b.balanced_accuracy)
                row[f"{view}_delta_q_vs_orig"] = row[f"{view}_best_quantum_bacc"] - row[f"{view}_best_classical_orig_bacc"]
                row[f"{view}_delta_q_vs_ext"] = row[f"{view}_best_quantum_bacc"] - row[f"{view}_best_classical_ext_bacc"]

                if view == "ood" and model == "gpc":
                    for fam, b in fams.items():
                        row[f"gpc_ece_id_{fam}"] = metric_at(g, b.cfg, "id_test", "ece")
                        row[f"gpc_ece_ood_{fam}"] = metric_at(g, b.cfg, "ood_test", "ece")
                        row[f"gpc_entropy_id_{fam}"] = metric_at(g, b.cfg, "id_test", "mean_predictive_entropy")
                        row[f"gpc_entropy_ood_{fam}"] = metric_at(g, b.cfg, "ood_test", "mean_predictive_entropy")
            rows.append(row)

    res = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_dir / "family_comparison_by_setting.csv", index=False)

    pd.set_option("display.width", 250)
    n_settings = res.groupby("model").size()
    print(f"Settings loaded per model: {dict(n_settings)}\n")

    for model in sorted(res.model.unique()):
        r = res[res.model == model]
        n = len(r)
        print(f"=== {model.upper()} (n={n} settings) ===")
        for view in ["ood", "id"]:
            wins_orig = int((r[f"{view}_delta_q_vs_orig"] > 0).sum())
            wins_ext = int((r[f"{view}_delta_q_vs_ext"] > 0).sum())
            print(
                f"  Best-by-{view.upper()}: quantum > classical_orig in {wins_orig}/{n} "
                f"(mean delta {r[f'{view}_delta_q_vs_orig'].mean():+.4f}) | "
                f"quantum > classical_ext in {wins_ext}/{n} "
                f"(mean delta {r[f'{view}_delta_q_vs_ext'].mean():+.4f})"
            )
        if model == "gpc":
            print("  GPC uncertainty (Best-by-OOD configs, mean over settings):")
            for fam in ["classical_orig", "classical_ext", "quantum"]:
                d_ece = (r[f"gpc_ece_ood_{fam}"] - r[f"gpc_ece_id_{fam}"]).mean()
                d_ent = (r[f"gpc_entropy_ood_{fam}"] - r[f"gpc_entropy_id_{fam}"]).mean()
                print(f"    {fam:15s}: ECE OOD-ID {d_ece:+.4f} | entropy OOD-ID {d_ent:+.4f}")
        print()

    # Which extended kernels actually get selected under Best-by-OOD?
    print("=== Best-by-OOD selected kernels (classical_ext family) ===")
    sel = res.groupby("model")["ood_best_classical_ext_cfg"].value_counts()
    print(sel.to_string())
    print(f"\n[✓] Wrote {args.out_dir / 'family_comparison_by_setting.csv'}")


if __name__ == "__main__":
    main()

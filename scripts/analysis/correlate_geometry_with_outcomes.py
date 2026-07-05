# scripts/analysis/correlate_geometry_with_outcomes.py
"""
Cross the kernel-geometry descriptors (results/kernel_geometry/grid) with the
paper's Best-by-OOD outcomes (results/tables/table_18settings_best_ood__with_std.csv).

Two questions:
  1. Which geometry quantities correlate with the observed OOD delta
     (quantum - classical) across the 18 principal settings?
  2. Does any a-priori-computable quantity separate the classical-favorable
     settings from the quantum-favorable ones? (Reported as Mann-Whitney AUC
     plus exact separation counts; n=18 with 4 positives, so this is a
     hypothesis test, not a classifier benchmark.)

Features come in two flavors:
  - selected: conditioned on the configurations actually selected by
    Best-by-OOD (diagnostic value),
  - family: computed from family-level aggregates only, available without
    knowing the winner (predictive value).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

CLASSICAL = ["linear", "rbf_gscale"]
QUANTUM = ["zz_r1_full", "zz_r2_full", "pauli_xz_r1_full", "zmap_r2"]

RX_LABEL = re.compile(r"^(?P<variant>m\d_.+?)__ms(?P<ms>\d+)__q(?P<qtag>\d+_id\d+_ood\d+)__qs(?P<qs>\d+)$")


def load_geometry(grid_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(grid_dir.glob("kernel_geometry__*.csv")):
        df = pd.read_csv(p)
        m = RX_LABEL.match(df["setting_label"].iloc[0])
        if not m:
            raise RuntimeError(f"Unparseable setting label in {p}")
        df["variant_name"] = m.group("variant")
        df["master_seed"] = int(m.group("ms"))
        df["size_tag"] = "size_q" + m.group("qtag")
        df["qsplit_seed"] = int(m.group("qs"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pick(geo: pd.DataFrame, kernel: str, dim: int) -> pd.Series:
    r = geo[(geo.kernel == kernel) & (geo.dim == dim)]
    if len(r) != 1:
        raise RuntimeError(f"Expected 1 geometry row for {kernel}/d{dim}, got {len(r)}")
    return r.iloc[0]


def setting_features(geo: pd.DataFrame, out_row: pd.Series) -> Dict[str, float]:
    """geo: geometry rows for one setting (all kernels x dims)."""
    c_cfg = str(out_row["cfg_classical"])
    c_dim = int(out_row["dim_classical"])
    q_kernel, q_dim_s = str(out_row["cfg_quantum"]).split("__d")
    q_dim = int(q_dim_s)

    sel_c = pick(geo, c_cfg, c_dim)
    sel_q = pick(geo, q_kernel, q_dim)

    f: Dict[str, float] = {}

    # --- selected-configuration features (diagnostic) ---
    f["sel_gd_lam1e-4"] = float(sel_q[f"gd_vs_{c_cfg}_lam0.0001"])
    f["sel_gd_lam1e-2"] = float(sel_q[f"gd_vs_{c_cfg}_lam0.01"])
    f["sel_kta_ood_q_minus_c"] = float(sel_q["kta_ood"] - sel_c["kta_ood"])
    f["sel_kta_drop_q_minus_c"] = float(sel_q["kta_drop_id_to_ood"] - sel_c["kta_drop_id_to_ood"])
    f["sel_eff_rank_q"] = float(sel_q["spec_train_eff_rank"])
    f["sel_novelty_ratio_q_minus_c"] = float(sel_q["ood_novelty_ratio"] - sel_c["ood_novelty_ratio"])

    # --- family-level features (a priori: no knowledge of the winner) ---
    gc = geo[geo.family == "classical"]
    gq = geo[geo.family == "quantum"]
    f["fam_kta_ood_best_q_minus_best_c"] = float(gq.kta_ood.max() - gc.kta_ood.max())
    f["fam_kta_id_best_q_minus_best_c"] = float(gq.kta_id.max() - gc.kta_id.max())
    f["fam_kta_train_best_q_minus_best_c"] = float(gq.kta_train.max() - gc.kta_train.max())
    f["fam_gd_median_lam1e-4"] = float(
        np.median(np.concatenate([gq[f"gd_vs_{c}_lam0.0001"].to_numpy() for c in CLASSICAL]))
    )
    f["fam_eff_rank_max_q"] = float(gq.spec_train_eff_rank.max())
    f["fam_novelty_ratio_mean_q_minus_c"] = float(gq.ood_novelty_ratio.mean() - gc.ood_novelty_ratio.mean())

    # Fully a-priori variant: select the quantum config by TRAIN kernel-target
    # alignment only (no ID/OOD information), then read off its effective rank.
    # If this discriminates the classical-favorable settings, the criterion is
    # computable before deployment.
    q_by_train_kta = gq.loc[gq.kta_train.idxmax()]
    f["apriori_eff_rank_q_at_best_train_kta"] = float(q_by_train_kta["spec_train_eff_rank"])
    f["apriori_gd1e-2_vs_rbf_at_best_train_kta"] = float(q_by_train_kta["gd_vs_rbf_gscale_lam0.01"])
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", type=Path, default=Path("results/kernel_geometry/grid"))
    ap.add_argument("--outcomes", type=Path, default=Path("results/tables/table_18settings_best_ood__with_std.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/kernel_geometry"))
    args = ap.parse_args()

    geo_all = load_geometry(args.grid_dir)
    outcomes = pd.read_csv(args.outcomes)
    qsplit_seeds = sorted(geo_all.qsplit_seed.unique())

    # Features per (setting, qsplit seed).
    per_seed_rows: List[Dict[str, float]] = []
    for _, out_row in outcomes.iterrows():
        key = (out_row["variant_name"], int(out_row["master_seed"]), out_row["size_tag"])
        for qs in qsplit_seeds:
            geo = geo_all[
                (geo_all.variant_name == key[0])
                & (geo_all.master_seed == key[1])
                & (geo_all.size_tag == key[2])
                & (geo_all.qsplit_seed == qs)
            ]
            if geo.empty:
                raise RuntimeError(f"No geometry rows for setting {key} qs={qs}")
            f = setting_features(geo, out_row)
            f.update(
                variant=key[0], master_seed=key[1], size_tag=key[2], qsplit_seed=qs,
                delta_ood=float(out_row["delta_ood_bal_acc_mean"]),
                winner=str(out_row["winner_ood_bal_acc_mean"]),
            )
            per_seed_rows.append(f)

    df_seeds = pd.DataFrame(per_seed_rows)
    feature_cols = [c for c in df_seeds.columns if c.startswith(("sel_", "fam_", "apriori_"))]

    # Main analysis on the qsplit-seed mean (variability-aware, one point per setting).
    keys = ["variant", "master_seed", "size_tag"]
    df = (
        df_seeds.groupby(keys, as_index=False)
        .agg({**{c: "mean" for c in feature_cols}, "delta_ood": "first", "winner": "first"})
    )

    # 1) correlation with the continuous OOD delta
    corr_rows = []
    y = df["delta_ood"].to_numpy()
    is_classical_win = (df["winner"] == "classical").to_numpy()
    for c in feature_cols:
        x = df[c].to_numpy()
        rho_s, p_s = stats.spearmanr(x, y)
        r_p, p_p = stats.pearsonr(x, y)
        # 2) separation of the 4 classical-favorable settings (Mann-Whitney AUC)
        u = stats.mannwhitneyu(x[~is_classical_win], x[is_classical_win], alternative="two-sided")
        auc = u.statistic / (np.sum(~is_classical_win) * np.sum(is_classical_win))
        # exact separation: does a single threshold classify all 18?
        lo_q, hi_q = x[~is_classical_win].min(), x[~is_classical_win].max()
        lo_c, hi_c = x[is_classical_win].min(), x[is_classical_win].max()
        perfectly_separable = bool(hi_c < lo_q or hi_q < lo_c)
        corr_rows.append({
            "feature": c,
            "spearman_rho_vs_delta_ood": round(float(rho_s), 3),
            "spearman_p": round(float(p_s), 4),
            "pearson_r": round(float(r_p), 3),
            "pearson_p": round(float(p_p), 4),
            "auc_quantum_vs_classical_wins": round(float(auc), 3),
            "mannwhitney_p": round(float(u.pvalue), 4),
            "perfect_threshold_separation": perfectly_separable,
        })

    corr = pd.DataFrame(corr_rows).sort_values("mannwhitney_p")

    # Per-qsplit-seed stability of the separation (AUC per seed, per feature).
    stab_rows = []
    for c in feature_cols:
        aucs = []
        for qs in qsplit_seeds:
            sub = df_seeds[df_seeds.qsplit_seed == qs]
            x = sub[c].to_numpy()
            cls = (sub["winner"] == "classical").to_numpy()
            u = stats.mannwhitneyu(x[~cls], x[cls], alternative="two-sided")
            aucs.append(u.statistic / (np.sum(~cls) * np.sum(cls)))
        stab_rows.append({
            "feature": c,
            "auc_mean_across_seeds": round(float(np.mean(aucs)), 3),
            "auc_min": round(float(np.min(aucs)), 3),
            "auc_max": round(float(np.max(aucs)), 3),
            "n_seeds": len(aucs),
        })
    stab = pd.DataFrame(stab_rows).sort_values("auc_mean_across_seeds", ascending=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "geometry_features_by_setting.csv", index=False)
    df_seeds.to_csv(args.out_dir / "geometry_features_by_setting_and_qsplit.csv", index=False)
    corr.to_csv(args.out_dir / "geometry_outcome_correlations.csv", index=False)
    stab.to_csv(args.out_dir / "geometry_outcome_auc_stability.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"=== Per-setting features (18 principal settings, mean over qsplit seeds {qsplit_seeds}) ===")
    show = ["variant", "master_seed", "size_tag", "winner", "delta_ood"] + feature_cols
    print(df[show].round(3).to_string(index=False))
    print("\n=== Correlations & separation (on qsplit-seed means) ===")
    print(corr.to_string(index=False))
    print("\n=== AUC stability across individual qsplit seeds ===")
    print(stab.to_string(index=False))
    print(f"\n[✓] Wrote {args.out_dir / 'geometry_features_by_setting.csv'}")
    print(f"[✓] Wrote {args.out_dir / 'geometry_features_by_setting_and_qsplit.csv'}")
    print(f"[✓] Wrote {args.out_dir / 'geometry_outcome_correlations.csv'}")
    print(f"[✓] Wrote {args.out_dir / 'geometry_outcome_auc_stability.csv'}")


if __name__ == "__main__":
    main()

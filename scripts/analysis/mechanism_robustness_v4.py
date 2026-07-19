# scripts/analysis/mechanism_robustness_v4.py
"""
v4 mechanism robustness controls (PLAN.md section 10; spec section 8 wording).

Permitted thesis: effective rank and OOD kernel-target alignment are ROBUSTLY
ASSOCIATED with OOD performance. Not permitted without further evidence: a
single causal geometric mechanism. This script quantifies how far the
association survives controls that the causal reading would require:

  1. within-unit Spearman correlations (all kernels / classical-only /
     quantum-only) of eff_rank_train and KTA_OOD vs OOD balanced accuracy --
     KTA_OOD and KTA_survival (= KTA_OOD - KTA_ID) reported SEPARATELY;
  2. partial rank correlations controlling for family, dimension, and length
     scale (residualized Spearman);
  3. leave-one-dataset-out predictive check: rank the kernels of the held-out
     dataset by a geometry model fit on the other three; report held-out
     Spearman between predicted and actual OOD ordering;
  4. rank-matched family comparison: quantum vs classical configs paired by
     nearest effective rank within (run, dim); does the family label carry
     residual OOD-accuracy information once rank is matched?

Input: per-run geometry_v4.csv + summary_v4.csv (SVC rows; GPC as secondary).
Output -> results/v4/mechanism/.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUN_RE = re.compile(r"^(?P<setting>.+)__qs(?P<qs>\d+)__s(?P<seed>\d+)$")


def group_label(setting: str) -> str:
    toks = setting.split("__")
    if toks[0].startswith(("m1_", "m2_")):
        return f"ember_{toks[0].split('_')[0]}"
    return f"{toks[0]}_{toks[1]}"


def dataset_of(group: str) -> str:
    return "ember" if group.startswith("ember") else group.rsplit("_m", 1)[0].rsplit("_natural", 1)[0]


def parse_scale(kernel: str) -> float:
    if "__as" in kernel:
        return float(kernel.split("__as", 1)[1])
    m = re.match(r"^.+_x([0-9.]+)$", kernel)
    return float(m.group(1)) if m else 1.0


def load(roots: list[Path], limit: int | None = None) -> pd.DataFrame:
    frames = []
    for root in roots:
        for d in sorted(root.iterdir()):
            m = RUN_RE.match(d.name) if d.is_dir() else None
            fg, fs = d / "geometry_v4.csv", d / "summary_v4.csv"
            if not m or not fg.exists() or not fs.exists():
                continue
            g = pd.read_csv(fg)
            s = pd.read_csv(fs)
            s = s[s.split == "ood_test"][["family", "model", "kernel", "dim",
                                          "balanced_accuracy"]]
            df = g.merge(s, on=["family", "kernel", "dim"], how="inner")
            df["run"], df["group"] = d.name, group_label(m.group("setting"))
            frames.append(df)
            if limit and len(frames) >= limit:
                break
        if limit and len(frames) >= limit:
            break
    out = pd.concat(frames, ignore_index=True)
    out["dataset"] = out.group.map(dataset_of)
    out["scale"] = out.kernel.map(parse_scale)
    out["is_quantum"] = (out.family == "quantum").astype(float)
    return out


def scope_mask(g: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all":
        return pd.Series(True, index=g.index)
    if scope == "classical":
        return g.family != "quantum"
    return g.family == "quantum"


def partial_spearman(y: np.ndarray, x: np.ndarray, Z: np.ndarray) -> float:
    """Spearman of rank-residuals after linear control for Z (ranked)."""
    def resid(v):
        vr = stats.rankdata(v).astype(float)
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(vr)), Z]), vr, rcond=None)
        return vr - np.column_stack([np.ones(len(vr)), Z]) @ beta
    rx, ry = resid(x), resid(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", type=Path,
                    default=[Path("results/ember_shift/extended_kernels"),
                             Path("results/netflow/extended_kernels")])
    ap.add_argument("--out-dir", type=Path, default=Path("results/v4/mechanism"))
    ap.add_argument("--model", default="svc", choices=["svc", "gpc"])
    ap.add_argument("--limit-runs", type=int, default=None,
                    help="smoke-testing only: cap the number of runs loaded")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.roots, args.limit_runs)
    df = df[df.model == args.model]
    print(f"[mech] {df.run.nunique()} runs, {df.kernel.nunique()} kernels, model={args.model}")

    # 1 + 2: within-unit correlations, plain and partial ---------------------
    unit_rows = []
    for (run, dim), g in df.groupby(["run", "dim"]):
        Z = np.column_stack([stats.rankdata(g.is_quantum),
                             stats.rankdata(np.log(g.scale))])
        for scope in ("all", "classical", "quantum"):
            sub = g[scope_mask(g, scope)]
            if sub.kernel.nunique() < 4:
                continue
            row = {"run": run, "group": g.group.iloc[0], "dataset": g.dataset.iloc[0],
                   "dim": dim, "scope": scope, "n_kernels": len(sub)}
            for pred in ("spec_train_eff_rank", "kta_ood", "kta_survival"):
                row[f"rho_{pred}"] = float(stats.spearmanr(
                    sub[pred], sub.balanced_accuracy).statistic)
            if scope == "all":
                Zs = np.column_stack([stats.rankdata(sub.is_quantum),
                                      stats.rankdata(np.log(sub.scale))])
                row["partial_rho_eff_rank"] = partial_spearman(
                    sub.balanced_accuracy.to_numpy(),
                    sub.spec_train_eff_rank.to_numpy(), Zs)
                row["partial_rho_kta_ood"] = partial_spearman(
                    sub.balanced_accuracy.to_numpy(), sub.kta_ood.to_numpy(), Zs)
            unit_rows.append(row)
    units = pd.DataFrame(unit_rows)
    units.to_csv(args.out_dir / "unit_correlations.csv", index=False)

    summ = units.groupby(["group", "scope"]).agg(
        n_units=("dim", "size"),
        **{f"median_{c}": (c, "median") for c in
           ["rho_spec_train_eff_rank", "rho_kta_ood", "rho_kta_survival"]},
        **{f"fracpos_{c}": (c, lambda s: float((s > 0).mean())) for c in
           ["rho_spec_train_eff_rank", "rho_kta_ood", "rho_kta_survival"]},
        median_partial_eff_rank=("partial_rho_eff_rank", "median"),
        median_partial_kta_ood=("partial_rho_kta_ood", "median"),
    ).reset_index()
    summ.to_csv(args.out_dir / "summary_by_group.csv", index=False)

    # 3: leave-one-dataset-out predictive ordering ---------------------------
    lodo_rows = []
    per_cell = df.groupby(["dataset", "run", "dim", "kernel"]).agg(
        eff=("spec_train_eff_rank", "mean"), bacc=("balanced_accuracy", "mean")
    ).reset_index()
    for held in per_cell.dataset.unique():
        tr = per_cell[per_cell.dataset != held]
        te = per_cell[per_cell.dataset == held]
        # geometry model = mean OOD-accuracy rank of each kernel's log-eff-rank
        # decile learned on the training datasets
        tr = tr.assign(bin=pd.qcut(np.log(tr.eff), 10, labels=False, duplicates="drop"))
        bin_score = tr.groupby("bin").bacc.mean()
        edges = pd.qcut(np.log(tr.eff), 10, retbins=True, duplicates="drop")[1]
        te_bin = np.clip(np.digitize(np.log(te.eff), edges) - 1, 0, len(bin_score) - 1)
        pred = bin_score.reindex(te_bin).to_numpy()
        rho = float(stats.spearmanr(pred, te.bacc).statistic)
        lodo_rows.append({"held_out_dataset": held, "n_cells": len(te),
                          "spearman_pred_vs_actual": rho})
    lodo = pd.DataFrame(lodo_rows)
    lodo.to_csv(args.out_dir / "lodo_predictive.csv", index=False)

    # 4: rank-matched family comparison --------------------------------------
    match_rows = []
    for (run, dim), g in df.groupby(["run", "dim"]):
        q = g[g.family == "quantum"]
        c = g[g.family != "quantum"]
        if q.empty or c.empty:
            continue
        for _, qr in q.iterrows():
            ci = (c.spec_train_eff_rank - qr.spec_train_eff_rank).abs().idxmin()
            cr = c.loc[ci]
            ratio = max(qr.spec_train_eff_rank, cr.spec_train_eff_rank) / \
                max(1e-9, min(qr.spec_train_eff_rank, cr.spec_train_eff_rank))
            match_rows.append({"run": run, "group": g.group.iloc[0], "dim": dim,
                               "q_kernel": qr.kernel, "c_kernel": cr.kernel,
                               "rank_ratio": float(ratio),
                               "delta_bacc_q_minus_c": float(qr.balanced_accuracy
                                                             - cr.balanced_accuracy)})
    matched = pd.DataFrame(match_rows)
    matched.to_csv(args.out_dir / "rank_matched_pairs.csv", index=False)
    tight = matched[matched.rank_ratio <= 1.25]
    msum = tight.groupby("group").agg(
        n_pairs=("delta_bacc_q_minus_c", "size"),
        mean_delta=("delta_bacc_q_minus_c", "mean"),
        median_delta=("delta_bacc_q_minus_c", "median"),
        frac_quantum_better=("delta_bacc_q_minus_c", lambda s: float((s > 0).mean()))
    ).reset_index()
    msum.to_csv(args.out_dir / "rank_matched_summary.csv", index=False)

    print("\n[mech] within-unit medians (scope=all):")
    print(summ[summ.scope == "all"][["group", "median_rho_spec_train_eff_rank",
                                     "median_rho_kta_ood", "median_rho_kta_survival",
                                     "median_partial_eff_rank",
                                     "median_partial_kta_ood"]]
          .round(3).to_string(index=False))
    print("\n[mech] LODO predictive ordering (geometry fit on 3 datasets, "
          "evaluated on the 4th):")
    print(lodo.round(3).to_string(index=False))
    print("\n[mech] rank-matched (ratio<=1.25) quantum-minus-classical OOD delta:")
    print(msum.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

# scripts/analysis/shots_stability_analysis.py
"""
Finite-shot fidelity-estimation perturbation analysis (spec constraint 6 and
PLAN.md section 17). Consumes the completed shots subset (q1000, model seed
42, all 5 q-split seeds -- 120 runs) and quantifies which statevector-exact
conclusions survive finite estimation budgets.

This is NOT a hardware simulation: no device noise model is involved, only
the statistical perturbation of estimating each kernel entry from a finite
number of Bernoulli fidelity samples.

Reported per shots level (vs the exact statevector reference of the SAME
runs from summary_v4.csv):
  1. accuracy stability : per (run, config, model) |bacc_shots - bacc_exact|
     on id_val / id_test / ood_test;
  2. geometry stability : effective-rank ratio and KTA_OOD delta;
  3. selection stability: does the P1' (argmax on id_val) quantum winner
     within the run change under perturbation, and how much OOD accuracy is
     lost by deploying the shots-selected config;
  4. PSD projection diagnostics before/after (min eigenvalue, Frobenius
     change from sampling vs from projection).

Outputs -> results/v4/shots/.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RUN_RE = re.compile(r"^(?P<setting>.+)__qs(?P<qs>\d+)__s(?P<seed>\d+)$")


def group_label(setting: str) -> str:
    toks = setting.split("__")
    if toks[0].startswith(("m1_", "m2_")):
        return f"ember_{toks[0].split('_')[0]}"
    return f"{toks[0]}_{toks[1]}"


def load(roots: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shot_frames, exact_frames, geo_frames = [], [], []
    for root in roots:
        for d in sorted(root.iterdir()):
            m = RUN_RE.match(d.name) if d.is_dir() else None
            f = d / "shots_v4.csv"
            if not m or not f.exists():
                continue
            sh = pd.read_csv(f)
            sh["run"], sh["group"] = d.name, group_label(m.group("setting"))
            shot_frames.append(sh[sh.model != "geometry"])
            geo_frames.append(sh[sh.model == "geometry"].copy())
            ex = pd.read_csv(d / "summary_v4.csv")
            ex = ex[ex.family == "quantum"]
            ex["run"], ex["group"] = d.name, group_label(m.group("setting"))
            exact_frames.append(ex)
    return (pd.concat(shot_frames, ignore_index=True),
            pd.concat(exact_frames, ignore_index=True),
            pd.concat(geo_frames, ignore_index=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", type=Path,
                    default=[Path("results/ember_shift/extended_kernels"),
                             Path("results/netflow/extended_kernels")])
    ap.add_argument("--out-dir", type=Path, default=Path("results/v4/shots"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    shots, exact, geo = load(args.roots)
    n_runs = shots.run.nunique()
    print(f"[shots] {n_runs} runs, {shots.shots.nunique()} shot levels, "
          f"{shots.kernel.nunique()} quantum kernels")

    # 1. accuracy stability vs exact ---------------------------------------
    key = ["run", "cfg", "model", "split"]
    merged = shots.merge(exact[key + ["balanced_accuracy"]], on=key,
                         suffixes=("", "_exact"))
    merged["abs_dev"] = (merged.balanced_accuracy - merged.balanced_accuracy_exact).abs()
    acc = merged.groupby(["group", "shots", "model", "split"]).agg(
        n=("abs_dev", "size"), mean_abs_dev=("abs_dev", "mean"),
        p95_abs_dev=("abs_dev", lambda s: float(np.percentile(s, 95))),
        mean_exact=("balanced_accuracy_exact", "mean"),
        mean_shots=("balanced_accuracy", "mean")).reset_index()
    acc.to_csv(args.out_dir / "accuracy_stability.csv", index=False)

    # 2. geometry stability -------------------------------------------------
    # exact geometry reference comes from geometry_v4.csv of the same runs
    geo_ex_frames = []
    for root in args.roots:
        for d in sorted(root.iterdir()):
            f = d / "geometry_v4.csv"
            if (d / "shots_v4.csv").exists() and f.exists():
                g = pd.read_csv(f)
                g = g[g.family == "quantum"]
                g["run"] = d.name
                geo_ex_frames.append(g)
    geo_ex = pd.concat(geo_ex_frames, ignore_index=True)
    gm = geo.merge(geo_ex[["run", "kernel", "dim", "spec_train_eff_rank", "kta_ood"]],
                   on=["run", "kernel", "dim"], suffixes=("", "_exact"))
    gm["eff_rank_ratio"] = gm.spec_train_eff_rank / gm.spec_train_eff_rank_exact
    gm["kta_ood_dev"] = (gm.kta_ood - gm.kta_ood_exact).abs()
    gstab = gm.groupby("shots").agg(
        n=("eff_rank_ratio", "size"),
        eff_rank_ratio_median=("eff_rank_ratio", "median"),
        eff_rank_ratio_p5=("eff_rank_ratio", lambda s: float(np.percentile(s, 5))),
        eff_rank_ratio_p95=("eff_rank_ratio", lambda s: float(np.percentile(s, 95))),
        kta_ood_dev_median=("kta_ood_dev", "median"),
        kta_ood_dev_p95=("kta_ood_dev", lambda s: float(np.percentile(s, 95)))).reset_index()
    gstab.to_csv(args.out_dir / "geometry_stability.csv", index=False)

    # 3. selection stability (P1' within the quantum family) ---------------
    sel_rows = []
    ex_sel = exact[exact.split == "id_val"]
    ex_ood = exact[exact.split == "ood_test"].set_index(["run", "cfg", "model"])
    for (run, model), g in ex_sel.groupby(["run", "model"]):
        best_exact = g.loc[g.balanced_accuracy.idxmax()]
        ood_exact = ex_ood.loc[(run, best_exact.cfg, model)].balanced_accuracy
        sh_run = shots[(shots.run == run) & (shots.model == model)]
        for s, gs in sh_run[sh_run.split == "id_val"].groupby("shots"):
            best_shots = gs.loc[gs.balanced_accuracy.idxmax()]
            # deploy the shots-selected config, evaluate at EXACT kernel OOD
            try:
                ood_dep = ex_ood.loc[(run, best_shots.cfg, model)].balanced_accuracy
            except KeyError:
                continue
            sel_rows.append({"run": run, "group": g.group.iloc[0], "model": model,
                             "shots": s, "same_selection": best_shots.cfg == best_exact.cfg,
                             "ood_exact_selection": float(ood_exact),
                             "ood_shots_selection": float(ood_dep),
                             "selection_regret": float(ood_exact - ood_dep)})
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(args.out_dir / "selection_stability.csv", index=False)
    sel_sum = sel.groupby(["shots", "model"]).agg(
        n=("same_selection", "size"),
        frac_same_selection=("same_selection", "mean"),
        mean_regret=("selection_regret", "mean"),
        p95_regret=("selection_regret", lambda s: float(np.percentile(s, 95)))).reset_index()
    sel_sum.to_csv(args.out_dir / "selection_stability_summary.csv", index=False)

    # 4. PSD diagnostics ----------------------------------------------------
    psd = shots.dropna(subset=["min_eig_before_psd"]).groupby("shots").agg(
        min_eig_before_median=("min_eig_before_psd", "median"),
        fro_sampling_median=("fro_change_sampling", "median"),
        fro_projection_median=("fro_change_projection", "median")).reset_index()
    psd["projection_vs_sampling"] = psd.fro_projection_median / psd.fro_sampling_median
    psd.to_csv(args.out_dir / "psd_diagnostics.csv", index=False)

    print("\n[shots] accuracy deviation vs exact (mean |dev| in OOD bacc, svc):")
    a = acc[(acc.model == "svc") & (acc.split == "ood_test")]
    print(a.pivot_table(index="group", columns="shots", values="mean_abs_dev")
          .round(4).to_string())
    print("\n[shots] geometry stability:")
    print(gstab.round(3).to_string(index=False))
    print("\n[shots] P1' selection stability:")
    print(sel_sum.round(4).to_string(index=False))
    print("\n[shots] PSD projection diagnostics:")
    print(psd.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

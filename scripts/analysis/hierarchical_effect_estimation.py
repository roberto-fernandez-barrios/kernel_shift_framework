# scripts/analysis/hierarchical_effect_estimation.py
"""
GATE 2 of the v4 revision (docs/ANALYSIS_SPEC_V4.md section 5): effect
estimation that respects the experimental dependence structure. Replaces the
withdrawn sign-flip permutation of hierarchical_stats.py (legacy).

Framing (author constraints 1-2): the 8 scenario-groups are FIXED CASE
STUDIES. No global population p-value of any kind is computed. Reported per
(variant, model, scenario-group): the P1 quantum-minus-classical OOD delta
with an interval that quantifies CONDITIONAL PIPELINE-REALIZATION UNCERTAINTY
-- conditional on the benchmark pools and the experimental design; it is not
inference about a population of datasets or shifts.

Method: the only near-independent replication axis inside a group is the
q-split seed (5 levels, <=1.2% EMBER / <=7.8% netflow OOD sample overlap;
results/v4/audit/). Master seed is not a replicate (EMBER m1 OOD pools are
identical across master seeds) and size is a fixed design factor (netflow
trains are nested across sizes). Each group's effect is therefore summarized
by 5 q-split cluster means built from nested means (size -> master seed ->
model seed), with a t-interval on the 5 cluster means (Ibragimov-Muller style,
used here as a conditional sensitivity), a BCa bootstrap over cluster means as
secondary, and a leave-one-qsplit-out jackknife as diagnostic. Cross-group:
per-case effects, descriptive dataset-equal-weighted mean, between-case
heterogeneity, and leave-one-dataset-out ranges -- all descriptive.

Input: run-level P1 files from budget_matched_selection.py
(p1_runs__{full,matched60,budget60}.csv). Outputs under results/v4/inference/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BCA_SEED = 20260719
N_BOOT = 9999
ASSUMPTIONS = ("qsplit clusters ~independent given fixed benchmark pools "
               "(audited overlap <=1.2% EMBER, <=7.8% netflow OOD); cluster means "
               "approx normal; master seed and size are fixed strata inside "
               "clusters; CONDITIONAL pipeline-realization uncertainty -- not "
               "inference about a population of datasets or shifts")


def dataset_of(group: str) -> str:
    return group.rsplit("_m", 1)[0].rsplit("_natural", 1)[0].replace("ember", "ember") \
        if not group.startswith("ember") else "ember"


def parse_setting(setting: str) -> tuple[str, str]:
    """Return (master_seed, size) from '...__ms123__q1000_id500_ood500'."""
    import re
    m = re.search(r"__ms(\d+)__q(\d+)_", setting)
    return m.group(1), m.group(2)


def nested_cluster_means(g: pd.DataFrame) -> np.ndarray:
    """One value per q-split cluster: mean over sizes of (mean over master
    seeds of (mean over model seeds)) -- unbalanced data never reweights."""
    vals = []
    for _, gq in g.groupby("qs"):
        by_size = gq.groupby("size").apply(
            lambda d: d.groupby("ms").delta.mean().mean(), include_groups=False)
        vals.append(float(by_size.mean()))
    return np.asarray(vals)


def cluster_t_ci(T: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    m = float(T.mean())
    if len(T) < 2 or np.allclose(T, T[0]):
        return m, m, m
    half = stats.t.ppf(1 - alpha / 2, len(T) - 1) * T.std(ddof=1) / np.sqrt(len(T))
    return m, m - half, m + half


def bca_ci(T: np.ndarray, rng: np.random.Generator,
           alpha: float = 0.05) -> tuple[float, float, str]:
    if np.allclose(T, T[0]):
        return float(T[0]), float(T[0]), "degenerate"
    n = len(T)
    boots = rng.choice(T, size=(N_BOOT, n), replace=True).mean(axis=1)
    obs = T.mean()
    frac = np.clip((boots < obs).mean(), 1 / (N_BOOT + 1), 1 - 1 / (N_BOOT + 1))
    z0 = stats.norm.ppf(frac)
    jack = np.array([np.delete(T, i).mean() for i in range(n)])
    d = jack.mean() - jack
    denom = 6 * (d ** 2).sum() ** 1.5
    a = (d ** 3).sum() / denom if denom > 0 else 0.0
    z = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
    adj = stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.percentile(boots, 100 * adj)
    return float(lo), float(hi), "bca"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-dir", type=Path, default=Path("results/v4/budget"))
    ap.add_argument("--variants", nargs="+", default=["full", "matched60", "budget60"])
    ap.add_argument("--out-dir", type=Path, default=Path("results/v4/inference"))
    ap.add_argument("--seed", type=int, default=BCA_SEED)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows, icc_rows = [], []
    for variant in args.variants:
        f = args.p1_dir / f"p1_runs__{variant}.csv"
        if not f.exists():
            print(f"[skip] {f} missing")
            continue
        df = pd.read_csv(f)
        if variant == "budget60":
            df = df.rename(columns={"p1_ood_classical_exp_budget": "p1_ood_classical"})
        ms_size = df.setting.map(parse_setting)
        df["ms"], df["size"] = [t[0] for t in ms_size], [t[1] for t in ms_size]
        df["dataset"] = df.group.map(dataset_of)

        for model, dm in df.groupby("model"):
            group_effects = {}
            for grp, g in dm.groupby("group"):
                qs_levels = sorted(g.qs.unique())
                if len(qs_levels) != 5:
                    raise SystemExit(f"{variant}/{model}/{grp}: expected 5 qsplit "
                                     f"clusters, found {len(qs_levels)}")
                T = nested_cluster_means(g)
                eff, lo, hi = cluster_t_ci(T)
                blo, bhi, bmethod = bca_ci(T, rng)
                jack = np.array([np.delete(T, i).mean() for i in range(len(T))])
                group_effects[grp] = eff
                # variance components (method of moments, descriptive)
                sd_qs = float(g.groupby("qs").delta.mean().std(ddof=1))
                sd_ms = float(g.groupby("ms").delta.mean().std(ddof=1))
                sd_size = float(g.groupby("size").delta.mean().std(ddof=1))
                sd_res = float(g.delta.std(ddof=1))
                icc_rows.append({"variant": variant, "model": model, "group": grp,
                                 "sd_between_qs": sd_qs, "sd_between_ms": sd_ms,
                                 "sd_between_size": sd_size, "sd_run_level": sd_res,
                                 "n_runs": len(g)})
                rows.append({
                    "variant": variant, "model": model, "scope": grp, "stratum": "all",
                    "effect": eff, "ci_lo": lo, "ci_hi": hi, "ci_method": "cluster_t",
                    "bca_lo": blo, "bca_hi": bhi, "bca_method": bmethod,
                    "jackknife_min": float(jack.min()), "jackknife_max": float(jack.max()),
                    "unit_of_resampling": "qsplit_seed",
                    "n_independent_clusters": len(T), "n_lower_level_obs": len(g),
                    "n_settings": g.setting.nunique(),
                    "method": "t-interval on 5 qsplit-cluster nested means "
                              "(Ibragimov-Muller, conditional sensitivity)",
                    "assumptions": ASSUMPTIONS})
                # per-size strata (design-factor view)
                for size, gs in g.groupby("size"):
                    Ts = nested_cluster_means(gs)
                    es, ls, hs = cluster_t_ci(Ts)
                    rows.append({
                        "variant": variant, "model": model, "scope": grp,
                        "stratum": f"q{size}", "effect": es, "ci_lo": ls, "ci_hi": hs,
                        "ci_method": "cluster_t", "bca_lo": np.nan, "bca_hi": np.nan,
                        "bca_method": "", "jackknife_min": np.nan, "jackknife_max": np.nan,
                        "unit_of_resampling": "qsplit_seed",
                        "n_independent_clusters": len(Ts), "n_lower_level_obs": len(gs),
                        "n_settings": gs.setting.nunique(),
                        "method": "t-interval on qsplit-cluster means within size stratum",
                        "assumptions": ASSUMPTIONS})

            # descriptive cross-group summaries (no p-values, per constraint 1)
            ge = pd.Series(group_effects)
            ds_means = ge.groupby(ge.index.map(dataset_of)).mean()
            eq_mean = float(ds_means.mean())
            lodo = {d: float(ds_means.drop(d).mean()) for d in ds_means.index}
            rows.append({
                "variant": variant, "model": model, "scope": "dataset_equal_mean",
                "stratum": "all", "effect": eq_mean, "ci_lo": np.nan, "ci_hi": np.nan,
                "ci_method": "descriptive", "bca_lo": np.nan, "bca_hi": np.nan,
                "bca_method": "", "jackknife_min": min(lodo.values()),
                "jackknife_max": max(lodo.values()),
                "unit_of_resampling": "none (descriptive)",
                "n_independent_clusters": len(ds_means),
                "n_lower_level_obs": len(ge), "n_settings": int(dm.setting.nunique()),
                "method": "dataset-equal-weighted mean of case-study effects; "
                          "jackknife columns hold the leave-one-dataset-out range",
                "assumptions": "descriptive summary of fixed case studies; "
                               "no population inference"})
            rows.append({
                "variant": variant, "model": model, "scope": "heterogeneity",
                "stratum": "all", "effect": float(ge.max() - ge.min()),
                "ci_lo": float(ge.min()), "ci_hi": float(ge.max()),
                "ci_method": "range", "bca_lo": np.nan, "bca_hi": np.nan,
                "bca_method": "", "jackknife_min": np.nan,
                "jackknife_max": float(ge.std(ddof=1)),
                "unit_of_resampling": "none (descriptive)",
                "n_independent_clusters": len(ge), "n_lower_level_obs": len(ge),
                "n_settings": int(dm.setting.nunique()),
                "method": "between-case heterogeneity: effect=range, ci=[min,max], "
                          "jackknife_max=SD of case effects",
                "assumptions": "descriptive; case studies are fixed, not sampled"})

    out = pd.DataFrame(rows)
    assert not any(c.endswith("_p") or c == "p" for c in out.columns), \
        "no p-value columns allowed (spec constraint 1)"
    out.to_csv(args.out_dir / "hierarchical_effects.csv", index=False)
    pd.DataFrame(icc_rows).to_csv(args.out_dir / "icc_diagnostics.csv", index=False)

    head = out[(out.stratum == "all") & ~out.scope.isin(["heterogeneity"])]
    print("\n[gate2] case-study effects (conditional pipeline-realization "
          "uncertainty; no population p-values):")
    print(head[["variant", "model", "scope", "effect", "ci_lo", "ci_hi",
                "jackknife_min", "jackknife_max"]].to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()

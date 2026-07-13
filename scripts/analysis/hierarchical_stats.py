# scripts/analysis/hierarchical_stats.py
"""
Hierarchy-aware inference for the family comparisons, replacing pooled
Wilcoxon over correlated settings.

For each (tag, protocol, model, reference) the setting-level paired deltas
(quantum - classical) are grouped by scenario/variant and by dataset. We report:

  * per-group mean delta with a bootstrap CI (resampling settings),
  * a sign-flip permutation test on the hierarchical statistic
    T = mean over datasets of (mean over that dataset's settings),
    which weights datasets equally instead of counting settings as
    independent replicates,
  * leave-one-dataset-out (LODO) values of T.

Input: results/honest_selection/<tag>__by_setting.csv (from
honest_selection_analysis.py). Output: results/honest_selection/hier_stats.csv.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260713)
N_PERM = 10000
N_BOOT = 10000


def dataset_label(group: str) -> str:
    return group.split("_m")[0].split("_natural")[0]  # ember_m1->ember, unsw_dos_m2..->unsw_dos


def hier_T(df: pd.DataFrame) -> float:
    return float(df.groupby("dataset").delta.mean().mean())


def perm_p(df: pd.DataFrame) -> float:
    obs = hier_T(df)
    d = df.delta.to_numpy()
    ds = df.dataset.to_numpy()
    uniq = np.unique(ds)
    count = 0
    for _ in range(N_PERM):
        flip = RNG.choice([-1.0, 1.0], size=len(d))
        perm = d * flip
        t = np.mean([perm[ds == u].mean() for u in uniq])
        if abs(t) >= abs(obs):
            count += 1
    return (count + 1) / (N_PERM + 1)


def boot_ci(x: np.ndarray) -> tuple[float, float]:
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    root = Path("results/honest_selection")
    rows = []
    for f in sorted(root.glob("*__by_setting.csv")):
        tag = f.name.replace("__by_setting.csv", "")
        res = pd.read_csv(f)
        q = res[res.fam == "quantum"].set_index(["setting", "model"])
        for ref in ("classical_orig", "classical_ext"):
            c = res[res.fam == ref].set_index(["setting", "model"])
            for proto in ("p1", "p2", "p3"):
                col = f"{proto}_ood_mean"
                dd = (q[col] - c[col]).dropna().rename("delta").reset_index()
                dd["group"] = dd.setting.map(
                    res.drop_duplicates("setting").set_index("setting").group)
                dd["dataset"] = dd.group.map(dataset_label)
                for model, dm in dd.groupby("model"):
                    # per-group bootstrap CIs
                    for gname, dg in dm.groupby("group"):
                        lo, hi = boot_ci(dg.delta.to_numpy())
                        rows.append({"tag": tag, "protocol": proto, "model": model,
                                     "reference": ref, "scope": gname,
                                     "n": len(dg), "mean_delta": dg.delta.mean(),
                                     "ci_lo": lo, "ci_hi": hi, "stat": "group_mean"})
                    # hierarchical permutation + LODO
                    T = hier_T(dm)
                    p = perm_p(dm)
                    lodo = {u: hier_T(dm[dm.dataset != u]) for u in dm.dataset.unique()
                            if dm.dataset.nunique() > 1}
                    rows.append({"tag": tag, "protocol": proto, "model": model,
                                 "reference": ref, "scope": "HIER",
                                 "n": len(dm), "mean_delta": T,
                                 "ci_lo": np.nan, "ci_hi": np.nan,
                                 "stat": "hier_T", "perm_p": p,
                                 "lodo_min": min(lodo.values()) if lodo else np.nan,
                                 "lodo_max": max(lodo.values()) if lodo else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(root / "hier_stats.csv", index=False)
    hier = out[out.stat == "hier_T"]
    print(hier[["tag", "protocol", "model", "reference", "mean_delta",
                "perm_p", "lodo_min", "lodo_max"]].to_string(
        index=False, formatters={"mean_delta": "{:+.4f}".format,
                                 "lodo_min": "{:+.4f}".format,
                                 "lodo_max": "{:+.4f}".format,
                                 "perm_p": "{:.4f}".format}))


if __name__ == "__main__":
    main()

# scripts/analysis/honest_selection_analysis.py
"""
Honest (no-test-label) selection protocols, recomputed from per-run summaries.

Protocols, applied identically to every kernel family:
  P1 deployment  : per run, select the config with the best ID-test balanced
                   accuracy; report that config's OOD balanced accuracy.
                   No OOD labels enter the selection.
  P2 cross-seed  : per setting, 5-fold over q-split seeds: select the config
                   with the best mean OOD balanced accuracy on 4 seeds,
                   evaluate it on the held-out seed's runs. OOD labels are
                   used only from splits disjoint from the evaluation split.
  P3 oracle      : the paper's original Best-by-OOD (select and evaluate on
                   the same OOD mean) — retained as an upper bound only.

Families: quantum, classical_ext (all classical), classical_orig (linear+RBF,
including bandwidth-sweep RBF variants by prefix).

Outputs per root: <out>/<tag>__by_setting.csv and <tag>__stats.csv, plus a
console summary grouped by scenario/variant.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ORIG_PREFIXES = ("linear", "rbf_gscale")
RUN_RE = re.compile(r"(?P<setting>.+)__qs(?P<qs>\d+)__s(?P<seed>\d+)$")


def load_runs(root: Path) -> pd.DataFrame:
    frames = []
    for d in sorted(root.iterdir()):
        f = d / "extended_kernels_qsplits__summary.csv"
        m = RUN_RE.match(d.name)
        if not m or not f.exists():
            continue
        df = pd.read_csv(f, usecols=["family", "model", "cfg", "kernel", "split", "balanced_accuracy"])
        df["setting"] = m.group("setting")
        df["qs"] = int(m.group("qs"))
        df["seed"] = int(m.group("seed"))
        frames.append(df)
    if not frames:
        raise SystemExit(f"No run summaries under {root}")
    out = pd.concat(frames, ignore_index=True)
    return out


def group_label(setting: str) -> str:
    """Scenario/variant group for per-benchmark statistics."""
    toks = setting.split("__")
    if toks[0].startswith(("m1_", "m2_")):
        return f"ember_{toks[0].split('_')[0]}"           # ember_m1 / ember_m2
    return f"{toks[0]}_{toks[1]}"                          # e.g. unsw_dos_natural_cur


def family_view(df: pd.DataFrame, fam: str) -> pd.DataFrame:
    if fam == "quantum":
        return df[df.family == "quantum"]
    if fam == "classical_ext":
        return df[df.family == "classical_ext"]
    if fam == "classical_orig":
        return df[(df.family == "classical_ext") & df.kernel.str.startswith(ORIG_PREFIXES)]
    raise ValueError(fam)


def wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (setting, qs, seed, model, cfg) with id/ood bacc columns."""
    w = df.pivot_table(index=["setting", "qs", "seed", "model", "cfg"],
                       columns="split", values="balanced_accuracy").reset_index()
    return w.rename(columns={"id_test": "bacc_id", "ood_test": "bacc_ood"}).sort_values(
        ["setting", "qs", "seed", "model", "cfg"]).reset_index(drop=True)


def p1_deployment(w: pd.DataFrame) -> pd.DataFrame:
    idx = w.groupby(["setting", "qs", "seed", "model"])["bacc_id"].idxmax()
    sel = w.loc[idx]
    agg = sel.groupby(["setting", "model"]).agg(
        p1_ood_mean=("bacc_ood", "mean"), p1_ood_std=("bacc_ood", "std"),
        p1_id_mean=("bacc_id", "mean"), p1_n_runs=("bacc_ood", "size"))
    return agg.reset_index()


def p2_cross_seed(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, model), g in w.groupby(["setting", "model"]):
        table = g.pivot_table(index="cfg", columns="qs", values="bacc_ood", aggfunc="mean")
        fold_vals = []
        for held in table.columns:
            others = [c for c in table.columns if c != held]
            if not others:
                continue
            cfg_sel = table[others].mean(axis=1).idxmax()
            v = table.loc[cfg_sel, held]
            if pd.notna(v):
                fold_vals.append(float(v))
        rows.append({"setting": setting, "model": model,
                     "p2_ood_mean": float(np.mean(fold_vals)) if fold_vals else np.nan,
                     "p2_n_folds": len(fold_vals)})
    return pd.DataFrame(rows)


def p3_oracle(w: pd.DataFrame) -> pd.DataFrame:
    mean_ood = w.groupby(["setting", "model", "cfg"])["bacc_ood"].mean().reset_index()
    idx = mean_ood.groupby(["setting", "model"])["bacc_ood"].idxmax()
    sel = mean_ood.loc[idx].rename(columns={"bacc_ood": "p3_ood_mean", "cfg": "p3_cfg"})
    return sel[["setting", "model", "p3_ood_mean", "p3_cfg"]]


def analyze_root(root: Path, out_dir: Path, tag: str) -> pd.DataFrame:
    df = load_runs(root)
    per_family = []
    for fam in ("quantum", "classical_ext", "classical_orig"):
        sub = family_view(df, fam)
        w = wide(sub)
        n_cfg = sub.groupby("model")["cfg"].nunique().to_dict()
        merged = p1_deployment(w).merge(p2_cross_seed(w), on=["setting", "model"]) \
                                 .merge(p3_oracle(w), on=["setting", "model"])
        merged["fam"] = fam
        merged["n_candidates"] = merged["model"].map(n_cfg)
        per_family.append(merged)
    res = pd.concat(per_family, ignore_index=True)
    res["group"] = res["setting"].map(group_label)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / f"{tag}__by_setting.csv", index=False)

    # Paired deltas quantum - classical reference, per protocol
    stats_rows = []
    q = res[res.fam == "quantum"].set_index(["setting", "model"])
    for ref in ("classical_orig", "classical_ext"):
        c = res[res.fam == ref].set_index(["setting", "model"])
        for proto in ("p1", "p2", "p3"):
            col = f"{proto}_ood_mean"
            delta = (q[col] - c[col]).dropna().rename("delta").reset_index()
            delta["group"] = delta["setting"].map(group_label)
            for model in delta.model.unique():
                dm = delta[delta.model == model]
                scopes = [("ALL", dm)] + [(gname, dg) for gname, dg in dm.groupby("group")]
                for gname, dg in scopes:
                    d = dg.delta.to_numpy()
                    if len(d) < 2 or np.allclose(d, 0):
                        p = np.nan
                    else:
                        p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
                    stats_rows.append({
                        "tag": tag, "protocol": proto, "model": model, "reference": ref,
                        "scope": gname, "n_settings": len(d), "wins": int((d > 0).sum()),
                        "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
                        "wilcoxon_p": p})
    st = pd.DataFrame(stats_rows)
    st.to_csv(out_dir / f"{tag}__stats.csv", index=False)
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="pairs tag=path, e.g. ember_main=results/ember_shift/extended_kernels")
    ap.add_argument("--out-dir", type=Path, default=Path("results/honest_selection"))
    args = ap.parse_args()
    for spec in args.roots:
        tag, path = spec.split("=", 1)
        st = analyze_root(Path(path), args.out_dir, tag)
        view = st[st.scope != "ALL"] if st.scope.nunique() > 2 else st
        print(f"\n=== {tag} ===")
        print(view.to_string(index=False,
              formatters={"mean_delta": "{:+.4f}".format, "median_delta": "{:+.4f}".format,
                          "wilcoxon_p": lambda p: "nan" if pd.isna(p) else f"{p:.2e}"}))


if __name__ == "__main__":
    main()

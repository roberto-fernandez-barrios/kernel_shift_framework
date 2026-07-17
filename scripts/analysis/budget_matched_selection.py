# scripts/analysis/budget_matched_selection.py
"""
GATE 1 of the v4 revision (docs/ANALYSIS_SPEC_V4.md section 4): does the
family-level P1 verdict survive EQUAL candidate budgets?

The published comparison selects the best-of-family over 115 classical kernel
geometries vs 60 quantum ones; under argmax selection the larger pool enjoys a
mechanical advantage. Primary analysis (author constraint 3): repeated
subsampling of the classical pool to the quantum budget with P1 selection
inside each subsample, plus performance-vs-budget curves for BOTH families,
expected top-k, family mean/median (selection-free), and normalized AUC-B.
The fixed "matched60" mirror design is a SECONDARY sensitivity (its
composition could be considered post hoc).

The resampling distribution quantifies BUDGET SENSITIVITY. It is not a
confidence interval; conditional uncertainty is estimated downstream by
hierarchical_effect_estimation.py on this script's run-level outputs.

Outputs under --out-dir (default results/v4/budget/):
  coverage.csv                  candidate pools and budget class per group
  resamples_by_group.csv        budget-sensitivity distribution of the delta
  resamples_by_setting.csv     ... at setting level
  curves.csv                    E[P1-OOD] vs budget B for both families
  family_quality.csv            selection-free mean/median OOD + AUC-B + E[max-of-k]
  matched60_by_setting.csv      fixed mirror design (secondary sensitivity)
  p1_runs__{full,matched60,budget60}.csv   run-level interface for GATE 2

Selection column is configurable: legacy id_test (--select-col bacc_id, the
software-validation pass, spec constraint 7) or the v4 ID-validation column
once the recompute lands (--select-col bacc_id_val).
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.honest_selection_analysis import load_runs, group_label  # noqa: E402

SPEC_SEED = 20260715
CFG_RE = re.compile(r"^(?P<kernel>.+?)__(?P<model>svc|gpc)(?:_C(?P<C>[0-9.]+))?__d(?P<dim>\d+)$")
CURVE_BUDGETS = (5, 10, 20, 30, 40, 50, 60)
MATCHED_SHAPES = ("rbf_gscale", "laplacian_med", "matern15_med", "matern25_med")
MATCHED_SCALES = (0.3, 1.0, 3.0)


def parse_kernel(kernel: str) -> tuple[str, float]:
    """Split a kernel token into (shape, scale): 'laplacian_med_x0.3' -> 1 case,
    'zz_r2_full__as2' -> the quantum case, otherwise scale 1.0."""
    if "__as" in kernel:
        shape, s = kernel.split("__as", 1)
        return shape, float(s)
    m = re.match(r"^(?P<shape>.+?)_x(?P<s>[0-9.]+)$", kernel)
    if m:
        return m.group("shape"), float(m.group("s"))
    return kernel, 1.0


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    ks = df.kernel.map(parse_kernel)
    df = df.assign(shape=[k[0] for k in ks], scale=[k[1] for k in ks])
    cfg_m = df.cfg.str.extract(CFG_RE)
    df["dim"] = cfg_m["dim"].astype(int)
    df["C"] = cfg_m["C"].astype(float).fillna(1.0)
    df["group"] = df.setting.map(group_label)
    return df


def load_combined(roots: list[tuple[str, Path]], extra_files: list[str],
                  base_file: str = "extended_kernels_qsplits__summary.csv") -> pd.DataFrame:
    frames = []
    for tag, root in roots:
        d = load_runs(root, extra_files, base_file=base_file)
        d["root"] = tag
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    key = ["setting", "qs", "seed", "model", "cfg", "split"]
    dup = df[df.duplicated(key, keep=False)]
    if len(dup):
        spread = dup.groupby(key).balanced_accuracy.agg(lambda s: s.max() - s.min())
        bad = spread[spread > 1e-9]
        if len(bad):
            print(f"[WARN] {len(bad)} duplicated keys DISAGREE across roots "
                  f"(max spread {bad.max():.2e}); keeping first occurrence")
    df = df.drop_duplicates(key, keep="first")
    return annotate(df)


@dataclass
class FamilyMatrices:
    runs: pd.DataFrame          # canonical (setting, qs, seed) rows + group
    cfgs: pd.DataFrame          # canonical cfg rows + shape/scale/dim
    X_sel: np.ndarray           # (R, C) selection metric (float32, NaN kept)
    Y_ood: np.ndarray           # (R, C) OOD balanced accuracy (float64)
    Y_extra: dict[str, np.ndarray]  # optional extra eval columns, e.g. bacc_id_test


def wide_metrics(df: pd.DataFrame, select_split: str) -> pd.DataFrame:
    w = df.pivot_table(index=["setting", "qs", "seed", "model", "family", "cfg"],
                       columns="split", values="balanced_accuracy").reset_index()
    if select_split not in w.columns:
        raise SystemExit(f"selection split '{select_split}' not present in the data "
                         f"(available: {[c for c in w.columns if '_' in str(c)]})")
    return w.rename(columns={select_split: "sel_metric", "ood_test": "bacc_ood"})


def build_matrices(w: pd.DataFrame, model: str, fam_mask: pd.Series,
                   cfg_meta: pd.DataFrame) -> FamilyMatrices:
    sub = w[(w.model == model) & fam_mask].copy()
    runs = (sub[["setting", "qs", "seed"]].drop_duplicates()
            .sort_values(["setting", "qs", "seed"]).reset_index(drop=True))
    runs["group"] = runs.setting.map(group_label)
    cfgs = sorted(sub.cfg.unique())
    cfg_ix = {c: i for i, c in enumerate(cfgs)}
    run_ix = {t: i for i, t in enumerate(map(tuple, runs[["setting", "qs", "seed"]].values))}
    X = np.full((len(runs), len(cfgs)), np.nan, dtype=np.float64)
    Y = np.full_like(X, np.nan)
    ri = sub[["setting", "qs", "seed"]].apply(tuple, axis=1).map(run_ix).to_numpy()
    ci = sub.cfg.map(cfg_ix).to_numpy()
    X[ri, ci] = sub.sel_metric.to_numpy()
    Y[ri, ci] = sub.bacc_ood.to_numpy()
    meta = cfg_meta.set_index("cfg").loc[cfgs].reset_index()
    return FamilyMatrices(runs=runs, cfgs=meta, X_sel=X.astype(np.float32), Y_ood=Y,
                          Y_extra={})


def p1_for_subsets(fm: FamilyMatrices, col_idx: np.ndarray, chunk: int = 256,
                   y: np.ndarray | None = None) -> np.ndarray:
    """P1 selection restricted to sampled candidate columns.

    col_idx: (B, k) int array. Returns (R, B) OOD bacc of the per-run winner
    (argmax of the selection metric within the subsample); NaN where a run has
    no valid candidate. Evaluation matrix defaults to fm.Y_ood."""
    Xm = np.where(np.isnan(fm.X_sel), -np.inf, fm.X_sel)
    Y = fm.Y_ood if y is None else y
    R = Xm.shape[0]
    B, k = col_idx.shape
    out = np.full((R, B), np.nan)
    for b0 in range(0, B, chunk):
        cols = col_idx[b0:b0 + chunk]                       # (b, k)
        sub = Xm[:, cols]                                   # (R, b, k)
        sel = sub.argmax(axis=2)                            # (R, b)
        picked = cols[np.arange(cols.shape[0])[None, :], sel]
        vals = np.take_along_axis(Y, picked, axis=1)
        dead = np.isneginf(sub).all(axis=2)
        vals[dead] = np.nan
        out[:, b0:b0 + cols.shape[0]] = vals
    return out


# ---------------------------------------------------------------------------
# Sampling schemes (spec section 4; RNG spawned per scheme so schemes are
# independent of each other and of execution order)
# ---------------------------------------------------------------------------
def sample_uniform(rng, n_pool: int, k: int, B: int) -> np.ndarray:
    return np.argsort(rng.random((B, n_pool)), axis=1)[:, :k]


def sample_kernel_stratified(rng, shapes: np.ndarray, k: int, B: int) -> np.ndarray:
    """Per-shape strata; each resample takes 2 or 3 members per shape so the
    total is exactly k (generalizes to the reduced 7-shape/20-budget branch)."""
    uniq = np.unique(shapes)
    per_shape = {s: np.flatnonzero(shapes == s) for s in uniq}
    base = k // len(uniq)
    n_extra = k - base * len(uniq)
    out = np.empty((B, k), dtype=np.int64)
    for b in range(B):
        extra = rng.choice(len(uniq), size=n_extra, replace=False)
        take = np.full(len(uniq), base)
        take[extra] += 1
        cols = [rng.choice(per_shape[s], size=min(t, len(per_shape[s])), replace=False)
                for s, t in zip(uniq, take)]
        flat = np.concatenate(cols)
        while len(flat) < k:   # top up if a stratum was smaller than its quota
            pool = np.setdiff1d(np.arange(len(shapes)), flat)
            flat = np.append(flat, rng.choice(pool))
        out[b] = np.sort(flat[:k])
    return out


def sample_kernel_blocked(rng, shapes: np.ndarray, k: int, B: int) -> np.ndarray:
    """Whole-shape blocks: choose shapes and keep ALL their dims -- the
    structural mirror of the quantum 12-maps-by-5-dims pool. Requires k to be
    a multiple of the block size."""
    uniq = np.unique(shapes)
    blocks = [np.flatnonzero(shapes == s) for s in uniq]
    per = len(blocks[0])
    assert all(len(b) == per for b in blocks), "unbalanced shape blocks"
    n_blocks = k // per
    assert n_blocks * per == k, f"budget {k} not a multiple of block size {per}"
    pick = np.argsort(rng.random((B, len(uniq))), axis=1)[:, :n_blocks]
    lut = np.stack(blocks)                                   # (n_shapes, per)
    return lut[pick].reshape(B, k)


SCHEMES = {"uniform": sample_uniform,
           "kernel_stratified": sample_kernel_stratified,
           "kernel_blocked": sample_kernel_blocked}


def expected_max_of_k(values: np.ndarray, k: int) -> float:
    """Closed-form E[max of k draws without replacement] from the finite pool
    of per-candidate values (selection-free ceiling; NaNs dropped)."""
    v = np.sort(values[~np.isnan(values)])
    n = len(v)
    if k > n or n == 0:
        return np.nan
    i = np.arange(n)
    from scipy.special import comb
    w = comb(i, k - 1) / comb(n, k)
    return float((w * v).sum())


def per_setting_mean(fm: FamilyMatrices, mat: np.ndarray) -> tuple[pd.Index, np.ndarray]:
    g = fm.runs.groupby("setting").indices
    settings = pd.Index(sorted(g))
    out = np.stack([np.nanmean(mat[g[s]], axis=0) for s in settings])
    return settings, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "ember=results/ember_shift/extended_kernels",
        "ember_bw=results/ember_shift/bandwidth_sweep",
        "netflow=results/netflow/extended_kernels",
        "netflow_bw=results/netflow_bandwidth_sweep/extended_kernels"])
    ap.add_argument("--extra-files", nargs="*", default=["summary_classical_lsweep.csv"])
    ap.add_argument("--base-file", default="extended_kernels_qsplits__summary.csv",
                    help="per-run base summary; the confirmatory v4 pass uses "
                         "summary_v4.csv (with --extra-files '' --select-col id_val)")
    ap.add_argument("--select-col", choices=["id_test", "id_val"], default="id_test",
                    help="id_test = legacy software-validation pass (spec constraint 7); "
                         "id_val = confirmatory v4 protocol")
    ap.add_argument("--out-dir", type=Path, default=Path("results/v4/budget"))
    ap.add_argument("--n-resamples", type=int, default=5000)
    ap.add_argument("--n-curve-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SPEC_SEED)
    ap.add_argument("--models", nargs="+", default=["svc", "gpc"])
    args = ap.parse_args()

    roots = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in args.roots]
    extra = [f for f in args.extra_files if f]
    df = load_combined(roots, extra, base_file=args.base_file)
    cfg_meta = df[["cfg", "kernel", "shape", "scale", "dim"]].drop_duplicates("cfg")
    w = wide_metrics(df, args.select_col)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    root_rng = np.random.default_rng(args.seed)

    cov_rows, grp_rows, set_rows, curve_rows, qual_rows = [], [], [], [], []
    m60_rows, p1_full_rows, p1_m60_rows, p1_b60_rows = [], [], [], []

    for model in args.models:
        fm_c = build_matrices(w, model, w.family == "classical_ext", cfg_meta)
        fm_q = build_matrices(w, model, w.family == "quantum", cfg_meta)
        # align runs across families
        common = fm_c.runs.merge(fm_q.runs, on=["setting", "qs", "seed", "group"])
        assert len(common) == len(fm_c.runs) == len(fm_q.runs), \
            "run sets differ between families"

        # quantum P1 at full pool (fixed reference)
        q_full = p1_for_subsets(fm_q, np.arange(fm_q.X_sel.shape[1])[None, :])[:, 0]
        c_full = p1_for_subsets(fm_c, np.arange(fm_c.X_sel.shape[1])[None, :])[:, 0]

        for grp, gruns in fm_c.runs.groupby("group"):
            rows = gruns.index.to_numpy()
            # group-local candidate pools (drop cfgs never evaluated in this group)
            c_alive = np.flatnonzero(~np.isnan(fm_c.X_sel[rows]).all(axis=0))
            q_alive = np.flatnonzero(~np.isnan(fm_q.X_sel[rows]).all(axis=0))
            n_c, n_q = len(c_alive), len(q_alive)
            budget = n_q
            cov_rows.append({"group": grp, "model": model, "n_classical": n_c,
                             "n_quantum": n_q, "budget": budget,
                             "budget_class": "full" if n_c >= 100 else "reduced"})
            # scheme strata = full kernel token (23 balanced blocks of 5 dims in
            # full-coverage groups; 7 blocks in reduced ones). The collapsed
            # 'shape' would merge scale variants into unbalanced strata.
            kernels_c = fm_c.cfgs.iloc[c_alive]["kernel"].to_numpy()
            q_grp = np.nanmean(q_full[rows])
            settings_in_grp = gruns.setting.unique()

            # --- primary: equal-budget resampling, three schemes -------------
            for scheme, fn in SCHEMES.items():
                rng = root_rng.spawn(1)[0]
                if scheme == "uniform":
                    local = fn(rng, n_c, budget, args.n_resamples)
                elif scheme == "kernel_stratified":
                    local = fn(rng, kernels_c, budget, args.n_resamples)
                else:
                    per = 5  # dims per kernel block
                    if budget % per or len(np.unique(kernels_c)) * per != n_c:
                        local = None
                    else:
                        local = fn(rng, kernels_c, budget, args.n_resamples)
                if local is None:
                    continue
                cols = c_alive[local]
                mat = p1_for_subsets(fm_c, cols)[rows]           # (Rg, B), gruns order
                # per-setting means then group mean per resample
                gsets = gruns.reset_index(drop=True).groupby("setting").indices
                per_set = np.stack([np.nanmean(mat[ix], axis=0) for _, ix in gsets.items()])
                per_set_q = np.array([np.nanmean(q_full[rows][ix]) for _, ix in gsets.items()])
                delta_set = per_set_q[:, None] - per_set          # (S, B)
                delta_grp = delta_set.mean(axis=0)                # (B,)
                grp_rows.append({
                    "group": grp, "model": model, "scheme": scheme, "budget": budget,
                    "B": len(delta_grp), "n_settings": len(gsets),
                    "q_p1_ood": q_grp,
                    "c_p1_ood_median": float(np.median(per_set.mean(axis=0))),
                    "delta_median": float(np.median(delta_grp)),
                    "delta_p2.5": float(np.percentile(delta_grp, 2.5)),
                    "delta_p97.5": float(np.percentile(delta_grp, 97.5)),
                    "frac_resamples_delta_lt0": float((delta_grp < 0).mean()),
                    "delta_full_pool": float(np.nanmean(q_full[rows]) - np.nanmean(c_full[rows])),
                    "select_col": args.select_col, "seed": args.seed})
                for si, (s, _) in enumerate(gsets.items()):
                    set_rows.append({
                        "group": grp, "setting": s, "model": model, "scheme": scheme,
                        "budget": budget, "delta_median": float(np.median(delta_set[si])),
                        "delta_p2.5": float(np.percentile(delta_set[si], 2.5)),
                        "delta_p97.5": float(np.percentile(delta_set[si], 97.5)),
                        "frac_lt0": float((delta_set[si] < 0).mean())})
                if scheme == "kernel_blocked":
                    # run-level expected classical P1 at equal budget -> GATE 2
                    exp_c = np.nanmean(mat, axis=1)
                    for rloc, (_, rr) in zip(range(len(rows)), gruns.iterrows()):
                        p1_b60_rows.append({
                            "group": grp, "setting": rr.setting, "qs": rr.qs,
                            "seed": rr.seed, "model": model,
                            "p1_ood_quantum": float(q_full[gruns.index[rloc]]),
                            "p1_ood_classical_exp_budget": float(exp_c[rloc]),
                            "delta": float(q_full[gruns.index[rloc]] - exp_c[rloc]),
                            "scheme": scheme, "budget": budget})

            # --- budget curves + family quality ------------------------------
            for fam, fmx, alive in [("classical_ext", fm_c, c_alive),
                                    ("quantum", fm_q, q_alive)]:
                pool = len(alive)
                per_cand = np.nanmean(fmx.Y_ood[rows][:, alive], axis=0)
                qual_rows.append({
                    "group": grp, "model": model, "family": fam, "n_pool": pool,
                    "family_mean_ood": float(np.nanmean(per_cand)),
                    "family_median_ood": float(np.nanmedian(per_cand)),
                    **{f"expected_max_ood_k{k}": expected_max_of_k(per_cand, k)
                       for k in CURVE_BUDGETS if k <= pool}})
                curve_pts = []
                for Bb in CURVE_BUDGETS:
                    if Bb > pool:
                        continue
                    rng = root_rng.spawn(1)[0]
                    cols = alive[sample_uniform(rng, pool, Bb, args.n_curve_resamples)]
                    mat = p1_for_subsets(fmx, cols)[rows]
                    val = float(np.nanmean(mat))
                    curve_pts.append((Bb, val))
                    curve_rows.append({"group": grp, "model": model, "family": fam,
                                       "budget": Bb, "e_p1_ood": val,
                                       "p2.5": float(np.nanpercentile(np.nanmean(mat, axis=0), 2.5)),
                                       "p97.5": float(np.nanpercentile(np.nanmean(mat, axis=0), 97.5))})
                if fam == "classical_ext" and pool > 60:
                    curve_rows.append({"group": grp, "model": model, "family": fam,
                                       "budget": pool,
                                       "e_p1_ood": float(np.nanmean(c_full[rows])),
                                       "p2.5": np.nan, "p97.5": np.nan})
                if len(curve_pts) >= 2:
                    bs, vs = zip(*curve_pts)
                    auc = float(np.trapezoid(vs, bs) / (bs[-1] - bs[0]))
                    qual_rows[-1]["auc_b"] = auc

            # --- matched60 secondary sensitivity ----------------------------
            is_matched = (fm_c.cfgs["shape"].isin(MATCHED_SHAPES)
                          & fm_c.cfgs["scale"].isin(MATCHED_SCALES)).to_numpy()
            m_cols = np.intersect1d(np.flatnonzero(is_matched), c_alive)
            if len(m_cols) == len(MATCHED_SHAPES) * len(MATCHED_SCALES) * 5:
                m_c = p1_for_subsets(fm_c, m_cols[None, :])[:, 0]
                for rloc, (_, rr) in zip(range(len(rows)), gruns.iterrows()):
                    ridx = gruns.index[rloc]
                    p1_m60_rows.append({"group": grp, "setting": rr.setting, "qs": rr.qs,
                                        "seed": rr.seed, "model": model,
                                        "p1_ood_quantum": float(q_full[ridx]),
                                        "p1_ood_classical": float(m_c[ridx]),
                                        "delta": float(q_full[ridx] - m_c[ridx])})
                for s, ix in gruns.groupby("setting").indices.items():
                    ridx = gruns.index[ix]
                    m60_rows.append({"group": grp, "setting": s, "model": model,
                                     "q_p1_ood": float(np.nanmean(q_full[ridx])),
                                     "c_p1_ood": float(np.nanmean(m_c[ridx])),
                                     "delta": float(np.nanmean(q_full[ridx]) - np.nanmean(m_c[ridx]))})

            # --- full-pool run-level interface -------------------------------
            for rloc, (_, rr) in zip(range(len(rows)), gruns.iterrows()):
                ridx = gruns.index[rloc]
                p1_full_rows.append({"group": grp, "setting": rr.setting, "qs": rr.qs,
                                     "seed": rr.seed, "model": model,
                                     "p1_ood_quantum": float(q_full[ridx]),
                                     "p1_ood_classical": float(c_full[ridx]),
                                     "delta": float(q_full[ridx] - c_full[ridx]),
                                     "n_classical": n_c, "n_quantum": n_q})

    pd.DataFrame(cov_rows).to_csv(args.out_dir / "coverage.csv", index=False)
    pd.DataFrame(grp_rows).to_csv(args.out_dir / "resamples_by_group.csv", index=False)
    pd.DataFrame(set_rows).to_csv(args.out_dir / "resamples_by_setting.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(args.out_dir / "curves.csv", index=False)
    pd.DataFrame(qual_rows).to_csv(args.out_dir / "family_quality.csv", index=False)
    pd.DataFrame(m60_rows).to_csv(args.out_dir / "matched60_by_setting.csv", index=False)
    pd.DataFrame(p1_full_rows).to_csv(args.out_dir / "p1_runs__full.csv", index=False)
    pd.DataFrame(p1_m60_rows).to_csv(args.out_dir / "p1_runs__matched60.csv", index=False)
    pd.DataFrame(p1_b60_rows).to_csv(args.out_dir / "p1_runs__budget60.csv", index=False)

    g = pd.DataFrame(grp_rows)
    print(f"\n[gate1] budget-matched deltas (select={args.select_col}; "
          "budget-sensitivity distribution, NOT a CI):")
    print(g[["group", "model", "scheme", "budget", "delta_median",
             "delta_p2.5", "delta_p97.5", "frac_resamples_delta_lt0",
             "delta_full_pool"]].to_string(index=False,
          float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()

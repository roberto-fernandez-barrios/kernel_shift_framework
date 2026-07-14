# scripts/analysis/dose_response_law.py
"""
The cross-family dose-response law (revised mechanism result).

Pools the length-scale-sweep geometry (geometry_lsweep.csv: seven base
classical kernels, twelve length-scale variants, four fidelity maps, all on
the same embedding) and asks whether a single geometric axis — the effective
rank of the training Gram — orders kernel-target alignment under shift
REGARDLESS of family.

Reports, per (run, dim) unit:
  rho(eff_rank, kta_ood), rho(eff_rank, kta_survival)
and the same correlations computed within the classical family only, so the
law cannot be an artefact of the quantum/classical split. Also emits the
per-kernel table used by the manuscript figure and the position of the
fidelity maps within the classical continuum.

Output: results/dose_response/{unit_correlations.csv, kernel_table.csv}
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd
from scipy import stats

RUN_RE = re.compile(r"([^\\/]+)[\\/]geometry_lsweep\.csv$")


def load() -> pd.DataFrame:
    frames = []
    for f in glob.glob("results/*/**/geometry_lsweep.csv", recursive=True):
        m = RUN_RE.search(f)
        if not m:
            continue
        df = pd.read_csv(f)
        df["run"] = m.group(1)
        df["dataset"] = "ember" if df.run.iloc[0].startswith(("m1_", "m2_")) \
            else df.run.iloc[0].split("__")[0]
        frames.append(df)
    if not frames:
        raise SystemExit("No geometry_lsweep.csv found")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = load()
    rows = []
    for (run, dim), g in df.groupby(["run", "dim"]):
        for scope, sub in (("all", g), ("classical_only", g[g.family != "quantum"])):
            if sub.kernel.nunique() < 4:
                continue
            rows.append({
                "run": run, "dataset": g.dataset.iloc[0], "dim": dim, "scope": scope,
                "n_kernels": sub.kernel.nunique(),
                "rho_rank_ktaood": stats.spearmanr(sub.spec_train_eff_rank, sub.kta_ood).statistic,
                "rho_rank_survival": stats.spearmanr(sub.spec_train_eff_rank, sub.kta_survival).statistic,
            })
    units = pd.DataFrame(rows)
    out = Path("results/dose_response")
    out.mkdir(parents=True, exist_ok=True)
    units.to_csv(out / "unit_correlations.csv", index=False)

    ktab = df.groupby(["family", "kernel"]).agg(
        eff_rank=("spec_train_eff_rank", "mean"),
        kta_ood=("kta_ood", "mean"),
        kta_survival=("kta_survival", "mean"),
        n=("kta_ood", "size")).reset_index().sort_values("eff_rank")
    ktab.to_csv(out / "kernel_table.csv", index=False)

    print(f"units: {len(units)} over {df.run.nunique()} runs, "
          f"{df.kernel.nunique()} kernels, datasets: {sorted(df.dataset.unique())}\n")
    print("Dose-response law (Spearman within run x dim):")
    print(units.groupby(["dataset", "scope"]).agg(
        n_units=("rho_rank_ktaood", "size"),
        median_rho_ktaood=("rho_rank_ktaood", "median"),
        frac_pos_ktaood=("rho_rank_ktaood", lambda s: (s > 0).mean()),
        median_rho_survival=("rho_rank_survival", "median"),
        frac_pos_survival=("rho_rank_survival", lambda s: (s > 0).mean()),
    ).round(3).to_string())

    q = ktab[ktab.family == "quantum"]
    c = ktab[ktab.family != "quantum"]
    print("\nPosition of the fidelity maps in the classical continuum:")
    for _, r in q.iterrows():
        above = int((c.eff_rank > r.eff_rank).sum())
        better_align = int((c.kta_ood > r.kta_ood).sum())
        print(f"  {r.kernel:18s} rank {r.eff_rank:7.1f}  kta_ood {r.kta_ood:.3f}  "
              f"| classical kernels with higher rank: {above}/{len(c)}, "
              f"with better OOD alignment: {better_align}/{len(c)}")


if __name__ == "__main__":
    main()

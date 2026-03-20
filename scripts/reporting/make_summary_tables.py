#!/usr/bin/env python3
"""
Generate manuscript-ready summary tables from root-level aggregated result CSVs.

Inputs
------
- AGG_ROOT_mean_std_metrics__by_variant_seed_and_size.csv
- AGG_ROOT_mean_std_drop__by_variant_seed_and_size.csv

Outputs
-------
1) table_18settings_best_ood__with_std.csv
   Per (variant_name, master_seed, size_tag):
   best classical vs best quantum by OOD balanced accuracy (mean),
   including std, n_runs, deltas, effect-size proxy, and winner.

2) table_18settings_best_id__with_std.csv
   Per (variant_name, master_seed, size_tag):
   best classical vs best quantum by ID balanced accuracy (mean),
   including std, n_runs, deltas, effect-size proxy, and winner.

3) table_18settings_best_drop__with_std.csv
   Per (variant_name, master_seed, size_tag):
   best classical vs best quantum by smallest ID→OOD balanced-accuracy drop,
   including std, n_runs, deltas, effect-size proxy, and winner.

4) table_90cells_dimlevel_best_ood__with_std.csv
   Per (variant_name, master_seed, size_tag, dim):
   best classical vs best quantum by OOD balanced accuracy (mean),
   including std, n_runs, deltas, effect-size proxy, and winner.

5) table_topN_cases_ood__with_std.csv
   Top-N rows by OOD balanced accuracy per family (classical / quantum).

Effect-size proxy
-----------------
effect = (mean_q - mean_c) / sqrt(std_q^2 + std_c^2)

For DROP metrics (lower is better), the effect is negative when quantum is better.
This is a quick descriptive proxy, not a formal statistical test.

Example
-------
python scripts/reporting/make_summary_tables.py \
  --metrics results/pipeline_validation/AGG_ROOT_mean_std_metrics__by_variant_seed_and_size.csv \
  --drops results/pipeline_validation/AGG_ROOT_mean_std_drop__by_variant_seed_and_size.csv \
  --outdir results/pipeline_validation/tables \
  --topn 10 \
  --round 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQ_METRICS_COLS = {
    "variant_name",
    "master_seed",
    "size_tag",
    "family",
    "dim",
    "cfg",
    "n_runs",
    "id_bal_acc_mean",
    "id_bal_acc_std",
    "ood_bal_acc_mean",
    "ood_bal_acc_std",
}

REQ_DROPS_COLS = {
    "variant_name",
    "master_seed",
    "size_tag",
    "family",
    "dim",
    "cfg",
    "n_runs",
    "drop_bal_acc_mean",
    "drop_bal_acc_std",
}


def ensure_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"[ERROR] Missing columns in {name}: {missing}")


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def coerce_nullable_int(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def best_per_group(
    df: pd.DataFrame,
    group_cols: list[str],
    sort_col: str,
    ascending: bool,
) -> pd.DataFrame:
    """
    Return one row per (group_cols + family): the best row under sort_col.
    """
    ordered = df.sort_values(sort_col, ascending=ascending).copy()
    return ordered.groupby(group_cols + ["family"], as_index=False).head(1)


def pivot_family_comparison(
    best_rows: pd.DataFrame,
    index_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    """
    Pivot family rows into one comparison row with columns such as:
      <metric>_classical, <metric>_quantum
    """
    pivoted = best_rows.pivot_table(
        index=index_cols,
        columns="family",
        values=value_cols,
        aggfunc="first",
    )
    pivoted.columns = [f"{metric}_{family}" for metric, family in pivoted.columns]
    return pivoted.reset_index()


def add_higher_is_better_winner(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    q_col = f"{metric}_quantum"
    c_col = f"{metric}_classical"
    out = df.copy()
    out[f"delta_{metric}"] = out[q_col] - out[c_col]
    out[f"winner_{metric}"] = np.where(
        out[q_col] > out[c_col],
        "quantum",
        np.where(out[q_col] < out[c_col], "classical", "tie"),
    )
    return out


def add_lower_is_better_winner(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    q_col = f"{metric}_quantum"
    c_col = f"{metric}_classical"
    out = df.copy()
    out[f"delta_{metric}"] = out[q_col] - out[c_col]
    out[f"winner_{metric}"] = np.where(
        out[q_col] < out[c_col],
        "quantum",
        np.where(out[q_col] > out[c_col], "classical", "tie"),
    )
    return out


def add_effect_proxy(
    df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Adds:
      effect_<mean_col> = (mean_q - mean_c) / sqrt(std_q^2 + std_c^2)
    """
    out = df.copy()
    mean_q = out[f"{mean_col}_quantum"].astype(float)
    mean_c = out[f"{mean_col}_classical"].astype(float)
    std_q = out[f"{std_col}_quantum"].astype(float)
    std_c = out[f"{std_col}_classical"].astype(float)

    denom = np.sqrt(np.maximum(std_q**2 + std_c**2, eps))
    out[f"effect_{mean_col}"] = (mean_q - mean_c) / denom
    return out


def round_float_columns(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out


def print_effect_summary(df: pd.DataFrame, effect_col: str, label: str) -> None:
    if effect_col not in df.columns:
        return
    values = pd.to_numeric(df[effect_col], errors="coerce")
    n = int(values.notna().sum())
    if n == 0:
        return
    gt_10 = int((values > 1.0).sum())
    gt_05 = int((values > 0.5).sum())
    lt_n10 = int((values < -1.0).sum())
    print(f"[Effect] {label}: n={n} | >1.0: {gt_10} | >0.5: {gt_05} | <-1.0: {lt_n10}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate manuscript-ready summary tables from aggregated root-level CSVs."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to AGG_ROOT_mean_std_metrics__by_variant_seed_and_size.csv",
    )
    parser.add_argument(
        "--drops",
        type=str,
        required=True,
        help="Path to AGG_ROOT_mean_std_drop__by_variant_seed_and_size.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/tables",
        help="Directory where output CSV tables will be written.",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Top-N rows by OOD balanced accuracy to keep per family.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=4,
        help="Decimal places for float outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    metrics_path = Path(args.metrics)
    drops_path = Path(args.drops)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(metrics_path)
    drops_df = pd.read_csv(drops_path)

    ensure_columns(metrics_df, REQ_METRICS_COLS, metrics_path.name)
    ensure_columns(drops_df, REQ_DROPS_COLS, drops_path.name)

    metrics_df = metrics_df[metrics_df["family"].isin(["classical", "quantum"])].copy()
    drops_df = drops_df[drops_df["family"].isin(["classical", "quantum"])].copy()

    metrics_df = coerce_numeric(
        metrics_df,
        [
            "master_seed",
            "dim",
            "n_runs",
            "id_bal_acc_mean",
            "id_bal_acc_std",
            "ood_bal_acc_mean",
            "ood_bal_acc_std",
        ],
    )
    drops_df = coerce_numeric(
        drops_df,
        [
            "master_seed",
            "dim",
            "n_runs",
            "drop_bal_acc_mean",
            "drop_bal_acc_std",
        ],
    )

    metrics_df = coerce_nullable_int(metrics_df, ["master_seed", "dim", "n_runs"])
    drops_df = coerce_nullable_int(drops_df, ["master_seed", "dim", "n_runs"])

    idx_18 = ["variant_name", "master_seed", "size_tag"]

    # 1) 18 settings — best OOD per family
    best_ood_18 = best_per_group(metrics_df, idx_18, sort_col="ood_bal_acc_mean", ascending=False)
    cmp_ood_18 = pivot_family_comparison(
        best_ood_18,
        index_cols=idx_18,
        value_cols=[
            "cfg",
            "dim",
            "n_runs",
            "id_bal_acc_mean",
            "id_bal_acc_std",
            "ood_bal_acc_mean",
            "ood_bal_acc_std",
        ],
    )
    cmp_ood_18 = add_higher_is_better_winner(cmp_ood_18, "ood_bal_acc_mean")
    cmp_ood_18 = add_higher_is_better_winner(cmp_ood_18, "id_bal_acc_mean")
    cmp_ood_18 = add_effect_proxy(cmp_ood_18, mean_col="ood_bal_acc_mean", std_col="ood_bal_acc_std")
    cmp_ood_18 = cmp_ood_18[
        [
            "variant_name",
            "master_seed",
            "size_tag",
            "cfg_classical",
            "dim_classical",
            "n_runs_classical",
            "id_bal_acc_mean_classical",
            "id_bal_acc_std_classical",
            "ood_bal_acc_mean_classical",
            "ood_bal_acc_std_classical",
            "cfg_quantum",
            "dim_quantum",
            "n_runs_quantum",
            "id_bal_acc_mean_quantum",
            "id_bal_acc_std_quantum",
            "ood_bal_acc_mean_quantum",
            "ood_bal_acc_std_quantum",
            "delta_ood_bal_acc_mean",
            "effect_ood_bal_acc_mean",
            "winner_ood_bal_acc_mean",
            "delta_id_bal_acc_mean",
            "winner_id_bal_acc_mean",
        ]
    ].sort_values(["variant_name", "master_seed", "size_tag"])
    cmp_ood_18 = round_float_columns(cmp_ood_18, args.round)
    out_ood_18 = outdir / "table_18settings_best_ood__with_std.csv"
    cmp_ood_18.to_csv(out_ood_18, index=False)

    # 1b) 18 settings — best ID per family
    best_id_18 = best_per_group(metrics_df, idx_18, sort_col="id_bal_acc_mean", ascending=False)
    cmp_id_18 = pivot_family_comparison(
        best_id_18,
        index_cols=idx_18,
        value_cols=[
            "cfg",
            "dim",
            "n_runs",
            "id_bal_acc_mean",
            "id_bal_acc_std",
            "ood_bal_acc_mean",
            "ood_bal_acc_std",
        ],
    )
    cmp_id_18 = add_higher_is_better_winner(cmp_id_18, "id_bal_acc_mean")
    cmp_id_18 = add_higher_is_better_winner(cmp_id_18, "ood_bal_acc_mean")
    cmp_id_18 = add_effect_proxy(cmp_id_18, mean_col="id_bal_acc_mean", std_col="id_bal_acc_std")
    cmp_id_18 = cmp_id_18[
        [
            "variant_name",
            "master_seed",
            "size_tag",
            "cfg_classical",
            "dim_classical",
            "n_runs_classical",
            "id_bal_acc_mean_classical",
            "id_bal_acc_std_classical",
            "ood_bal_acc_mean_classical",
            "ood_bal_acc_std_classical",
            "cfg_quantum",
            "dim_quantum",
            "n_runs_quantum",
            "id_bal_acc_mean_quantum",
            "id_bal_acc_std_quantum",
            "ood_bal_acc_mean_quantum",
            "ood_bal_acc_std_quantum",
            "delta_id_bal_acc_mean",
            "effect_id_bal_acc_mean",
            "winner_id_bal_acc_mean",
            "delta_ood_bal_acc_mean",
            "winner_ood_bal_acc_mean",
        ]
    ].sort_values(["variant_name", "master_seed", "size_tag"])
    cmp_id_18 = round_float_columns(cmp_id_18, args.round)
    out_id_18 = outdir / "table_18settings_best_id__with_std.csv"
    cmp_id_18.to_csv(out_id_18, index=False)

    # 2) 18 settings — best DROP (lower is better) per family
    best_drop_18 = best_per_group(drops_df, idx_18, sort_col="drop_bal_acc_mean", ascending=True)
    cmp_drop_18 = pivot_family_comparison(
        best_drop_18,
        index_cols=idx_18,
        value_cols=["cfg", "dim", "n_runs", "drop_bal_acc_mean", "drop_bal_acc_std"],
    )
    cmp_drop_18 = add_lower_is_better_winner(cmp_drop_18, "drop_bal_acc_mean")
    cmp_drop_18 = add_effect_proxy(cmp_drop_18, mean_col="drop_bal_acc_mean", std_col="drop_bal_acc_std")
    cmp_drop_18 = cmp_drop_18[
        [
            "variant_name",
            "master_seed",
            "size_tag",
            "cfg_classical",
            "dim_classical",
            "n_runs_classical",
            "drop_bal_acc_mean_classical",
            "drop_bal_acc_std_classical",
            "cfg_quantum",
            "dim_quantum",
            "n_runs_quantum",
            "drop_bal_acc_mean_quantum",
            "drop_bal_acc_std_quantum",
            "delta_drop_bal_acc_mean",
            "effect_drop_bal_acc_mean",
            "winner_drop_bal_acc_mean",
        ]
    ].sort_values(["variant_name", "master_seed", "size_tag"])
    cmp_drop_18 = round_float_columns(cmp_drop_18, args.round)
    out_drop_18 = outdir / "table_18settings_best_drop__with_std.csv"
    cmp_drop_18.to_csv(out_drop_18, index=False)

    # 3) 90 cells — best OOD per (setting, dim)
    idx_90 = ["variant_name", "master_seed", "size_tag", "dim"]
    best_ood_90 = best_per_group(metrics_df, idx_90, sort_col="ood_bal_acc_mean", ascending=False)
    cmp_ood_90 = pivot_family_comparison(
        best_ood_90,
        index_cols=idx_90,
        value_cols=[
            "cfg",
            "n_runs",
            "id_bal_acc_mean",
            "id_bal_acc_std",
            "ood_bal_acc_mean",
            "ood_bal_acc_std",
        ],
    )
    cmp_ood_90 = add_higher_is_better_winner(cmp_ood_90, "ood_bal_acc_mean")
    cmp_ood_90 = add_higher_is_better_winner(cmp_ood_90, "id_bal_acc_mean")
    cmp_ood_90 = add_effect_proxy(cmp_ood_90, mean_col="ood_bal_acc_mean", std_col="ood_bal_acc_std")
    cmp_ood_90 = cmp_ood_90[
        [
            "variant_name",
            "master_seed",
            "size_tag",
            "dim",
            "cfg_classical",
            "n_runs_classical",
            "id_bal_acc_mean_classical",
            "id_bal_acc_std_classical",
            "ood_bal_acc_mean_classical",
            "ood_bal_acc_std_classical",
            "cfg_quantum",
            "n_runs_quantum",
            "id_bal_acc_mean_quantum",
            "id_bal_acc_std_quantum",
            "ood_bal_acc_mean_quantum",
            "ood_bal_acc_std_quantum",
            "delta_ood_bal_acc_mean",
            "effect_ood_bal_acc_mean",
            "winner_ood_bal_acc_mean",
            "delta_id_bal_acc_mean",
            "winner_id_bal_acc_mean",
        ]
    ].sort_values(["variant_name", "master_seed", "size_tag", "dim"])
    cmp_ood_90 = round_float_columns(cmp_ood_90, args.round)
    out_ood_90 = outdir / "table_90cells_dimlevel_best_ood__with_std.csv"
    cmp_ood_90.to_csv(out_ood_90, index=False)

    # 4) Top-N by OOD per family
    topn = int(args.topn)
    top = (
        metrics_df[
            [
                "variant_name",
                "master_seed",
                "size_tag",
                "family",
                "dim",
                "cfg",
                "n_runs",
                "id_bal_acc_mean",
                "id_bal_acc_std",
                "ood_bal_acc_mean",
                "ood_bal_acc_std",
            ]
        ]
        .sort_values("ood_bal_acc_mean", ascending=False)
        .groupby("family", as_index=False)
        .head(topn)
        .reset_index(drop=True)
    )
    top = round_float_columns(top, args.round)
    out_top = outdir / f"table_top{topn}_cases_ood__with_std.csv"
    top.to_csv(out_top, index=False)

    # Console summaries
    wins_ood_18 = cmp_ood_18["winner_ood_bal_acc_mean"].value_counts().to_dict()
    wins_id_18 = cmp_id_18["winner_id_bal_acc_mean"].value_counts().to_dict()
    wins_drop_18 = cmp_drop_18["winner_drop_bal_acc_mean"].value_counts().to_dict()
    wins_ood_90 = cmp_ood_90["winner_ood_bal_acc_mean"].value_counts().to_dict()

    print("\n[✓] Wrote:")
    print(f" - {out_ood_18}")
    print(f" - {out_id_18}")
    print(f" - {out_drop_18}")
    print(f" - {out_ood_90}")
    print(f" - {out_top}")

    print("\n[Summary] OOD winner (18 settings):", wins_ood_18)
    print("[Summary] ID winner  (18 settings, best-by-ID selection):", wins_id_18)
    print("[Summary] DROP winner (18 settings, lower is better):", wins_drop_18)
    print("[Summary] OOD winner (dim-level):", wins_ood_90)

    print_effect_summary(cmp_ood_18, "effect_ood_bal_acc_mean", "OOD (18 settings)")
    print_effect_summary(cmp_id_18, "effect_id_bal_acc_mean", "ID (18 settings)")
    print_effect_summary(cmp_drop_18, "effect_drop_bal_acc_mean", "DROP (18 settings)")

    classical_wins = cmp_ood_18[cmp_ood_18["winner_ood_bal_acc_mean"] == "classical"].copy()
    if not classical_wins.empty:
        print("\n[Info] Settings where CLASSICAL wins (best OOD per setting):")
        print(
            classical_wins[
                [
                    "variant_name",
                    "master_seed",
                    "size_tag",
                    "ood_bal_acc_mean_classical",
                    "ood_bal_acc_std_classical",
                    "cfg_classical",
                    "dim_classical",
                    "ood_bal_acc_mean_quantum",
                    "ood_bal_acc_std_quantum",
                    "cfg_quantum",
                    "dim_quantum",
                    "delta_ood_bal_acc_mean",
                    "effect_ood_bal_acc_mean",
                ]
            ].to_string(index=False)
        )

    if "effect_ood_bal_acc_mean" in cmp_ood_18.columns:
        top5 = cmp_ood_18.sort_values("effect_ood_bal_acc_mean", ascending=False).head(5)
        print("\n[Info] Top-5 settings by effect_ood (largest positive):")
        print(
            top5[
                [
                    "variant_name",
                    "master_seed",
                    "size_tag",
                    "delta_ood_bal_acc_mean",
                    "effect_ood_bal_acc_mean",
                    "winner_ood_bal_acc_mean",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
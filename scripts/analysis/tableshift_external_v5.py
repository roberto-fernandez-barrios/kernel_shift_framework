"""Analyze the frozen v5 TableShift external validation.

All model selection for the controlled endpoint uses `validation` only.  The
same-test OOD endpoints are emitted as explicitly non-deployable diagnostics.
Uncertainty is conditional seed-cluster uncertainty; no population p-value is
computed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL_SEED = 20260729
N_BUDGET_RESAMPLES = 5000
N_BOOTSTRAP = 9999
TASKS = ("college_scorecard", "diabetes_readmission", "acsincome")
STRATA = ("q1000", "q2000")
SEEDS = (42, 123, 999, 7, 2024)
MODELS = ("svc", "gpc")


def kernel_block_subsets(
    classical: pd.DataFrame,
    n_resamples: int = N_BUDGET_RESAMPLES,
    seed: int = PROTOCOL_SEED,
) -> np.ndarray:
    """Sample 12 whole 5-dimension kernel blocks: 60 of 115 candidates."""
    ordered = classical.sort_values(["kernel", "dim"]).reset_index(drop=True)
    kernels = sorted(ordered.kernel.unique())
    if len(kernels) != 23:
        raise ValueError(f"expected 23 classical kernels, found {len(kernels)}")
    blocks = []
    for kernel in kernels:
        idx = np.flatnonzero(ordered.kernel.eq(kernel).to_numpy())
        if len(idx) != 5 or set(ordered.iloc[idx].dim) != {4, 6, 8, 10, 12}:
            raise ValueError(f"classical block {kernel} is not five-dimensional")
        blocks.append(idx)
    rng = np.random.default_rng(seed)
    chosen = np.argsort(rng.random((n_resamples, len(blocks))), axis=1)[:, :12]
    lookup = np.stack(blocks)
    return lookup[chosen].reshape(n_resamples, 60)


def selected_ood(
    selection: np.ndarray, ood: np.ndarray, subsets: np.ndarray
) -> np.ndarray:
    selection = np.asarray(selection)
    ood = np.asarray(ood)
    local_winner = selection[subsets].argmax(axis=1)
    winner = subsets[np.arange(len(subsets)), local_winner]
    return ood[winner]


def _model_wide(summary: pd.DataFrame, model: str, regularization: str) -> pd.DataFrame:
    part = summary[
        summary.model.eq(model) & summary.regularization.eq(regularization)
    ].copy()
    if part.duplicated(["family", "kernel", "dim", "split"]).any():
        raise ValueError("duplicated candidate/split rows in unit summary")
    wide = part.pivot(
        index=["family", "kernel", "dim", "candidate_id"],
        columns="split",
        values="balanced_accuracy",
    ).reset_index()
    required = {"validation", "id_test", "ood_test"}
    if not required.issubset(wide):
        raise ValueError(f"unit summary lacks splits {sorted(required - set(wide))}")
    return wide.sort_values(["family", "kernel", "dim"]).reset_index(drop=True)


def unit_effect(summary: pd.DataFrame, model: str) -> dict:
    """Compute honest, oracle, weak-fixed, and contraction effects for one unit."""
    regularization = "train_cv" if model == "svc" else "not_applicable"
    wide = _model_wide(summary, model, regularization)
    quantum = wide[wide.family.eq("quantum")].reset_index(drop=True)
    classical = wide[wide.family.eq("classical_ext")].reset_index(drop=True)
    if len(quantum) != 60 or len(classical) != 115:
        raise ValueError(
            f"incomplete {model} pools: quantum={len(quantum)}, "
            f"classical={len(classical)}"
        )

    q_honest_idx = int(quantum.validation.to_numpy().argmax())
    q_oracle_idx = int(quantum.ood_test.to_numpy().argmax())
    subsets = kernel_block_subsets(classical)
    classical = classical.sort_values(["kernel", "dim"]).reset_index(drop=True)
    c_honest = selected_ood(
        classical.validation.to_numpy(), classical.ood_test.to_numpy(), subsets
    )
    c_oracle = selected_ood(
        classical.ood_test.to_numpy(), classical.ood_test.to_numpy(), subsets
    )
    honest = float(quantum.loc[q_honest_idx, "ood_test"] - c_honest.mean())
    oracle = float(quantum.loc[q_oracle_idx, "ood_test"] - c_oracle.mean())

    weak_regularization = "fixed_c1" if model == "svc" else "not_applicable"
    weak = _model_wide(summary, model, weak_regularization)
    weak_q = weak[weak.family.eq("quantum")].reset_index(drop=True)
    weak_c = weak[
        weak.family.eq("classical_ext")
        & weak.kernel.isin(["linear", "rbf_gscale"])
    ].reset_index(drop=True)
    if len(weak_q) != 60 or len(weak_c) != 10:
        raise ValueError(
            f"incomplete weak pools: quantum={len(weak_q)}, classical={len(weak_c)}"
        )
    weak_q_idx = int(weak_q.ood_test.to_numpy().argmax())
    weak_c_idx = int(weak_c.ood_test.to_numpy().argmax())
    weak_delta = float(
        weak_q.loc[weak_q_idx, "ood_test"] - weak_c.loc[weak_c_idx, "ood_test"]
    )
    return {
        "model": model,
        "honest_delta": honest,
        "oracle_equal_delta": oracle,
        "weak_fixed_oracle_delta": weak_delta,
        "contraction": weak_delta - honest,
        "quantum_honest_candidate": quantum.loc[q_honest_idx, "candidate_id"],
        "quantum_oracle_candidate": quantum.loc[q_oracle_idx, "candidate_id"],
        "quantum_weak_candidate": weak_q.loc[weak_q_idx, "candidate_id"],
        "classical_weak_candidate": weak_c.loc[weak_c_idx, "candidate_id"],
        "classical_honest_expected_ood": float(c_honest.mean()),
        "classical_oracle_expected_ood": float(c_oracle.mean()),
        "budget_resamples": len(subsets),
    }


def load_run_effects(root: Path) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for stratum in STRATA:
            for seed in SEEDS:
                path = root / task / stratum / f"seed_{seed}" / "summary_v5.csv"
                if not path.exists():
                    raise FileNotFoundError(f"missing external unit: {path}")
                summary = pd.read_csv(path)
                for model in MODELS:
                    row = unit_effect(summary, model)
                    row.update({"task": task, "stratum": stratum, "seed": seed})
                    rows.append(row)
    return pd.DataFrame(rows)


def percentile_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    draws = values[indices].mean(axis=1)
    return tuple(map(float, np.percentile(draws, [2.5, 97.5])))


def aggregate_effects(run_effects: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "honest_delta", "oracle_equal_delta", "weak_fixed_oracle_delta",
        "contraction",
    )
    rows = []
    root_rng = np.random.default_rng(PROTOCOL_SEED)
    for model in MODELS:
        model_rows = run_effects[run_effects.model.eq(model)]
        for task in TASKS:
            task_rows = model_rows[model_rows.task.eq(task)]
            for stratum in STRATA:
                values = task_rows[task_rows.stratum.eq(stratum)].set_index("seed")
                if set(values.index) != set(SEEDS):
                    raise ValueError(f"incomplete seed clusters: {task}/{stratum}/{model}")
                for metric in metrics:
                    lo, hi = percentile_ci(
                        values.loc[list(SEEDS), metric].to_numpy(),
                        root_rng,
                    )
                    rows.append(
                        {
                            "scope": "task_size",
                            "task": task,
                            "stratum": stratum,
                            "model": model,
                            "metric": metric,
                            "effect": float(values[metric].mean()),
                            "ci_lo": lo,
                            "ci_hi": hi,
                            "n_seed_clusters": len(SEEDS),
                        }
                    )
            per_seed = (
                task_rows.groupby("seed", observed=True)[list(metrics)].mean()
                .loc[list(SEEDS)]
            )
            for metric in metrics:
                lo, hi = percentile_ci(per_seed[metric].to_numpy(), root_rng)
                rows.append(
                    {
                        "scope": "task",
                        "task": task,
                        "stratum": "sizes_equal",
                        "model": model,
                        "metric": metric,
                        "effect": float(per_seed[metric].mean()),
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "n_seed_clusters": len(SEEDS),
                    }
                )

        per_task_seed = (
            model_rows.groupby(["task", "seed"], observed=True)[list(metrics)]
            .mean()
            .reset_index()
        )
        for metric in metrics:
            task_arrays = [
                per_task_seed[per_task_seed.task.eq(task)]
                .set_index("seed").loc[list(SEEDS), metric].to_numpy()
                for task in TASKS
            ]
            draw_means = []
            for values in task_arrays:
                indices = root_rng.integers(
                    0, len(values), size=(N_BOOTSTRAP, len(values))
                )
                draw_means.append(values[indices].mean(axis=1))
            draws = np.stack(draw_means).mean(axis=0)
            task_means = np.asarray([values.mean() for values in task_arrays])
            leave_one_out = [
                float(np.delete(task_means, i).mean()) for i in range(len(TASKS))
            ]
            rows.append(
                {
                    "scope": "task_equal",
                    "task": "all",
                    "stratum": "sizes_equal",
                    "model": model,
                    "metric": metric,
                    "effect": float(task_means.mean()),
                    "ci_lo": float(np.percentile(draws, 2.5)),
                    "ci_hi": float(np.percentile(draws, 97.5)),
                    "n_seed_clusters": len(SEEDS) * len(TASKS),
                    "leave_one_task_out_min": min(leave_one_out),
                    "leave_one_task_out_max": max(leave_one_out),
                }
            )
    return pd.DataFrame(rows)


def classify_external(aggregate: pd.DataFrame) -> dict:
    svc = aggregate[aggregate.model.eq("svc")]
    overall = svc[
        svc.scope.eq("task_equal") & svc.metric.isin(["honest_delta", "contraction"])
    ].set_index("metric")
    honest = float(overall.loc["honest_delta", "effect"])
    contraction = float(overall.loc["contraction", "effect"])
    size_honest = svc[
        svc.scope.eq("task_size") & svc.metric.eq("honest_delta")
    ]
    task_positive_both = 0
    positive_tasks = []
    for task in TASKS:
        cells = size_honest[size_honest.task.eq(task)]
        if len(cells) == 2 and (cells.ci_lo > 0).all():
            task_positive_both += 1
            positive_tasks.append(task)

    if contraction > 0 and honest <= 0 and task_positive_both < 2:
        category = "replicated_protocol_sensitivity"
    elif honest > 0 and task_positive_both >= 2:
        category = "failure_to_generalize"
    elif task_positive_both >= 1:
        category = "regime_specific_boundary"
    else:
        category = "mixed_inconclusive"
    return {
        "category": category,
        "task_equal_svc_honest_delta": honest,
        "task_equal_svc_contraction": contraction,
        "tasks_positive_in_both_sizes": positive_tasks,
        "population_p_value_reported": False,
        "interpretation_rule": "docs/EXTERNAL_VALIDATION_SPEC.md section 7",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results/v5/external/runs"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/v5/external/analysis"),
    )
    args = parser.parse_args()
    run_effects = load_run_effects(args.runs_root)
    aggregate = aggregate_effects(run_effects)
    interpretation = classify_external(aggregate)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_effects.to_csv(args.out_dir / "run_effects.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_effects.csv", index=False)
    (args.out_dir / "interpretation.json").write_text(
        json.dumps(interpretation, indent=2), encoding="utf-8"
    )
    print(
        aggregate[aggregate.scope.eq("task_equal")][
            ["model", "metric", "effect", "ci_lo", "ci_hi"]
        ].to_string(index=False)
    )
    print(json.dumps(interpretation, indent=2))


if __name__ == "__main__":
    main()

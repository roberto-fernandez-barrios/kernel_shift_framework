"""Explore the sharp classical-search x target-label evidence frontier.

This is a declared post-unlock extension of the frozen v0.9 pilot.  It never
uses an unaudited target label to select the next adaptive query.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.partial_identification import (  # noqa: E402
    realized_accuracy_advantage,
)


ROOT = "ksf-v9-label-audit-20260731"
THRESHOLDS = (0.005, 0.010, 0.020)
N_HASH_DRAWS = 200
CURVE_BUDGETS = (0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash_order(n_items: int, *parts: object) -> np.ndarray:
    """Return a platform-independent SHA-256 permutation."""
    keys = []
    prefix = "|".join(str(part) for part in (ROOT, *parts))
    for index in range(n_items):
        digest = hashlib.sha256(f"{prefix}|{index}".encode("utf-8")).digest()
        keys.append((digest, index))
    return np.asarray([index for _, index in sorted(keys)], dtype=np.int64)


def _audit_state(
    quantum: np.ndarray, classical: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    disagreement = classical != quantum[:, None]
    signed = np.zeros(classical.shape[1], dtype=np.int64)
    remaining = disagreement.sum(axis=0, dtype=np.int64)
    return disagreement, signed, remaining


def _endpoints(signed: np.ndarray, remaining: np.ndarray, n: int) -> tuple[float, float]:
    return (
        float(np.min(signed - remaining) / n),
        float(np.min(signed + remaining) / n),
    )


def _reveal(
    index: int,
    label: int,
    quantum: np.ndarray,
    disagreement: np.ndarray,
    signed: np.ndarray,
    remaining: np.ndarray,
) -> None:
    affected = disagreement[index]
    remaining[affected] -= 1
    if label == quantum[index]:
        signed[affected] += 1
    else:
        signed[affected] -= 1


def audit_curve_from_order(
    quantum: np.ndarray,
    classical: np.ndarray,
    labels: np.ndarray,
    order: Iterable[int],
) -> pd.DataFrame:
    """Return the exact partial-label envelope along a fixed query order."""
    order = np.asarray(tuple(order), dtype=np.int64)
    n = len(quantum)
    if order.shape != (n,) or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("order must be a permutation of all target indices")
    disagreement, signed, remaining = _audit_state(quantum, classical)
    rows = []
    n_quantum_errors = 0
    n_union_observed = 0
    n_informative_counterexamples = 0
    for n_observed in range(n + 1):
        lower, upper = _endpoints(signed, remaining, n)
        rows.append(
            {
                "n_observed": n_observed,
                "lower": lower,
                "upper": upper,
                "region_width": upper - lower,
                "n_quantum_errors_observed": n_quantum_errors,
                "n_disagreement_union_observed": n_union_observed,
                "n_informative_counterexamples": n_informative_counterexamples,
            }
        )
        if n_observed < n:
            index = int(order[n_observed])
            in_union = bool(disagreement[index].any())
            quantum_error = bool(labels[index] != quantum[index])
            n_quantum_errors += int(quantum_error)
            n_union_observed += int(in_union)
            n_informative_counterexamples += int(in_union and quantum_error)
            _reveal(
                index,
                int(labels[index]),
                quantum,
                disagreement,
                signed,
                remaining,
            )
    return pd.DataFrame(rows)


def adaptive_bottleneck_cover_curve(
    quantum: np.ndarray,
    classical: np.ndarray,
    labels: np.ndarray,
    tie_order: Iterable[int],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Audit reducible bottleneck witnesses using only revealed outcomes."""
    tie_order = np.asarray(tuple(tie_order), dtype=np.int64)
    n = len(quantum)
    if tie_order.shape != (n,) or not np.array_equal(np.sort(tie_order), np.arange(n)):
        raise ValueError("tie_order must be a permutation of all target indices")
    tie_rank = np.empty(n, dtype=np.int64)
    tie_rank[tie_order] = np.arange(n)
    disagreement, signed, remaining = _audit_state(quantum, classical)
    observed = np.zeros(n, dtype=bool)
    chosen: list[int] = []
    rows = []
    n_quantum_errors = 0
    n_union_observed = 0
    n_informative_counterexamples = 0
    for n_observed in range(n + 1):
        lower, upper = _endpoints(signed, remaining, n)
        rows.append(
            {
                "n_observed": n_observed,
                "lower": lower,
                "upper": upper,
                "region_width": upper - lower,
                "n_quantum_errors_observed": n_quantum_errors,
                "n_disagreement_union_observed": n_union_observed,
                "n_informative_counterexamples": n_informative_counterexamples,
            }
        )
        if n_observed == n:
            break
        reducible = remaining > 0
        if reducible.any():
            witness_upper = signed + remaining
            bottleneck = np.min(witness_upper[reducible])
            active = reducible & (witness_upper == bottleneck)
            coverage = disagreement[:, active].sum(axis=1, dtype=np.int64)
            coverage[observed] = -1
            best_coverage = int(np.max(coverage))
            candidates = np.flatnonzero(coverage == best_coverage)
        else:
            candidates = np.flatnonzero(~observed)
        index = int(candidates[np.argmin(tie_rank[candidates])])
        chosen.append(index)
        observed[index] = True
        in_union = bool(disagreement[index].any())
        quantum_error = bool(labels[index] != quantum[index])
        n_quantum_errors += int(quantum_error)
        n_union_observed += int(in_union)
        n_informative_counterexamples += int(in_union and quantum_error)
        _reveal(
            index,
            int(labels[index]),
            quantum,
            disagreement,
            signed,
            remaining,
        )
    return pd.DataFrame(rows), np.asarray(chosen, dtype=np.int64)


def adaptive_random_active_disagreement_curve(
    quantum: np.ndarray,
    classical: np.ndarray,
    labels: np.ndarray,
    tie_order: Iterable[int],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Randomly audit only disagreements with a current bottleneck witness.

    The SHA-256 order supplies the randomization. Unlike the coverage policy,
    all unaudited points disagreeing with at least one active reducible witness
    are equally eligible, so no query is spent on a point that cannot change a
    currently actionable bottleneck.
    """
    tie_order = np.asarray(tuple(tie_order), dtype=np.int64)
    n = len(quantum)
    if tie_order.shape != (n,) or not np.array_equal(np.sort(tie_order), np.arange(n)):
        raise ValueError("tie_order must be a permutation of all target indices")
    tie_rank = np.empty(n, dtype=np.int64)
    tie_rank[tie_order] = np.arange(n)
    disagreement, signed, remaining = _audit_state(quantum, classical)
    observed = np.zeros(n, dtype=bool)
    chosen: list[int] = []
    rows = []
    n_quantum_errors = 0
    n_union_observed = 0
    n_informative_counterexamples = 0
    for n_observed in range(n + 1):
        lower, upper = _endpoints(signed, remaining, n)
        rows.append(
            {
                "n_observed": n_observed,
                "lower": lower,
                "upper": upper,
                "region_width": upper - lower,
                "n_quantum_errors_observed": n_quantum_errors,
                "n_disagreement_union_observed": n_union_observed,
                "n_informative_counterexamples": n_informative_counterexamples,
            }
        )
        if n_observed == n:
            break
        reducible = remaining > 0
        if reducible.any():
            witness_upper = signed + remaining
            bottleneck = np.min(witness_upper[reducible])
            active = reducible & (witness_upper == bottleneck)
            relevant = disagreement[:, active].any(axis=1) & ~observed
            candidates = np.flatnonzero(relevant)
        else:
            candidates = np.flatnonzero(~observed)
        index = int(candidates[np.argmin(tie_rank[candidates])])
        chosen.append(index)
        observed[index] = True
        in_union = bool(disagreement[index].any())
        quantum_error = bool(labels[index] != quantum[index])
        n_quantum_errors += int(quantum_error)
        n_union_observed += int(in_union)
        n_informative_counterexamples += int(in_union and quantum_error)
        _reveal(
            index,
            int(labels[index]),
            quantum,
            disagreement,
            signed,
            remaining,
        )
    return pd.DataFrame(rows), np.asarray(chosen, dtype=np.int64)


def nonadaptive_initial_coverage_order(
    quantum: np.ndarray,
    classical: np.ndarray,
    tie_order: Iterable[int],
) -> np.ndarray:
    """Preorder points by coverage of the zero-label bottleneck witnesses."""
    tie_order = np.asarray(tuple(tie_order), dtype=np.int64)
    n = len(quantum)
    if tie_order.shape != (n,) or not np.array_equal(np.sort(tie_order), np.arange(n)):
        raise ValueError("tie_order must be a permutation of all target indices")
    tie_rank = np.empty(n, dtype=np.int64)
    tie_rank[tie_order] = np.arange(n)
    disagreement, _, remaining = _audit_state(quantum, classical)
    reducible = remaining > 0
    if reducible.any():
        bottleneck = np.min(remaining[reducible])
        active = reducible & (remaining == bottleneck)
        coverage = disagreement[:, active].sum(axis=1, dtype=np.int64)
    else:
        coverage = np.zeros(n, dtype=np.int64)
    return np.lexsort((tie_rank, -coverage)).astype(np.int64, copy=False)


def retrospective_oracle_minimum_queries(
    quantum: np.ndarray,
    classical: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> int:
    """Return the exact ex-post minimum queries needed to cross a threshold.

    This oracle knows every target label and is therefore unattainable during
    an honest audit. For witness j, crossing requires enough audited
    disagreements favouring that classical prediction to make
    D_j - 2 b_j <= n * threshold. The minimum feasible requirement over
    witnesses is an exact lower bound on every sequential acquisition policy.
    """
    quantum = np.asarray(quantum, dtype=np.int8)
    classical = np.asarray(classical, dtype=np.int8)
    labels = np.asarray(labels, dtype=np.int8)
    if quantum.ndim != 1 or labels.shape != quantum.shape:
        raise ValueError("quantum and labels must be equal-length vectors")
    if classical.ndim != 2 or classical.shape[0] != len(quantum):
        raise ValueError("classical must have one target row per quantum prediction")
    disagreement = classical != quantum[:, None]
    total = disagreement.sum(axis=0, dtype=np.int64)
    classical_favouring = disagreement & (classical == labels[:, None])
    available = classical_favouring.sum(axis=0, dtype=np.int64)
    raw = (total.astype(float) - len(quantum) * float(threshold)) / 2.0
    required = np.maximum(0, np.ceil(raw - 1e-12).astype(np.int64))
    feasible = available >= required
    if not feasible.any():
        return -1
    return int(np.min(required[feasible]))


def first_crossing(curve: pd.DataFrame, threshold: float) -> int:
    crossed = curve.loc[curve.upper <= threshold, "n_observed"]
    if crossed.empty:
        return -1
    return int(crossed.iloc[0])


def quantum_stratum(kernel: str) -> str:
    return "entangling_zz" if kernel.startswith("zz_") else "product_map"


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def summarize_hash_thresholds(draws: pd.DataFrame) -> pd.DataFrame:
    keys = ["case", "group", "model", "quantum_stratum", "tier", "threshold"]
    rows = []
    for key, frame in draws.groupby(keys, sort=True):
        all_values = frame.n_labels.to_numpy(dtype=float)
        reached = all_values >= 0
        values = all_values[reached]
        informative = frame.n_informative_counterexamples.to_numpy(dtype=float)[reached]
        record = dict(zip(keys, key))
        if len(values):
            quantiles = {
                "median_n_labels": float(np.median(values)),
                "q025_n_labels": float(np.quantile(values, 0.025)),
                "q975_n_labels": float(np.quantile(values, 0.975)),
                "min_n_labels": int(np.min(values)),
                "max_n_labels": int(np.max(values)),
                "median_n_informative_counterexamples": float(np.median(informative)),
                "q025_n_informative_counterexamples": float(
                    np.quantile(informative, 0.025)
                ),
                "q975_n_informative_counterexamples": float(
                    np.quantile(informative, 0.975)
                ),
            }
        else:
            quantiles = {
                "median_n_labels": np.nan,
                "q025_n_labels": np.nan,
                "q975_n_labels": np.nan,
                "min_n_labels": np.nan,
                "max_n_labels": np.nan,
                "median_n_informative_counterexamples": np.nan,
                "q025_n_informative_counterexamples": np.nan,
                "q975_n_informative_counterexamples": np.nan,
            }
        record.update(
            {
                "n_draws": len(all_values),
                "n_reached": int(reached.sum()),
                "probability_reached": float(np.mean(reached)),
                **quantiles,
            }
        )
        for budget in (25, 50, 100, 200):
            record[f"probability_by_{budget}"] = float(
                np.mean(reached & (all_values <= budget))
            )
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("results/v9/partial_identification/predictions"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/v9/partial_identification/partial_label_frontier"),
    )
    parser.add_argument("--model", choices=("svc", "gpc"), default="svc")
    args = parser.parse_args()
    cases = sorted(path for path in args.prediction_root.iterdir() if path.is_dir())
    if len(cases) != 8:
        raise RuntimeError(f"expected eight exact cases, found {len(cases)}")

    adaptive_curves = []
    adaptive_thresholds = []
    hash_threshold_draws = []
    hash_curve_draws = []
    random_active_threshold_draws = []
    nonadaptive_coverage_threshold_draws = []
    oracle_thresholds = []
    hashes = []
    for case in cases:
        prediction_path = case / f"predictions_{args.model}.npz"
        label_path = case / "evaluation_labels.npz"
        metadata_path = case / f"metadata_{args.model}.json"
        hashes.extend(
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (prediction_path, label_path, metadata_path)
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(prediction_path, allow_pickle=False) as archive:
            quantum = archive["quantum_prediction"].astype(np.int8)
            classical_all = archive["classical_predictions"].astype(np.int8)
            customary = archive["customary_mask"].astype(bool)
        with np.load(label_path, allow_pickle=False) as archive:
            labels = archive["target_labels"].astype(np.int8)
        strata = quantum_stratum(metadata["quantum_winner"]["kernel"])
        tiers = {
            "customary_30": classical_all[:, customary],
            "full_115": classical_all,
        }
        hash_orders = [
            stable_hash_order(len(labels), case.name, "hash_all", draw)
            for draw in range(N_HASH_DRAWS)
        ]
        random_active_orders = [
            stable_hash_order(len(labels), case.name, "random_active", draw)
            for draw in range(N_HASH_DRAWS)
        ]
        coverage_tie_orders = [
            stable_hash_order(len(labels), case.name, "initial_coverage", draw)
            for draw in range(N_HASH_DRAWS)
        ]
        for tier, classical in tiers.items():
            identity = {
                "status": "post-unlock exploratory",
                "case": case.name,
                "run": metadata["run"],
                "group": metadata["group"],
                "model": args.model,
                "quantum_kernel": metadata["quantum_winner"]["kernel"],
                "quantum_stratum": strata,
                "tier": tier,
                "n_target": len(labels),
                "n_classical": classical.shape[1],
                "realized_accuracy_advantage": realized_accuracy_advantage(
                    labels, quantum, classical
                ),
            }
            tie_order = stable_hash_order(
                len(labels), case.name, tier, "adaptive_bottleneck_cover"
            )
            adaptive, chosen = adaptive_bottleneck_cover_curve(
                quantum, classical, labels, tie_order
            )
            if not np.array_equal(np.sort(chosen), np.arange(len(labels))):
                raise RuntimeError("adaptive policy did not query every target point")
            for key, value in identity.items():
                adaptive[key] = value
            adaptive["policy"] = "adaptive_bottleneck_cover"
            adaptive_curves.append(adaptive)
            for threshold in THRESHOLDS:
                crossing = first_crossing(adaptive, threshold)
                informative = (
                    -1
                    if crossing < 0
                    else int(
                        adaptive.loc[
                            adaptive.n_observed == crossing,
                            "n_informative_counterexamples",
                        ].iloc[0]
                    )
                )
                adaptive_thresholds.append(
                    {
                        **identity,
                        "policy": "adaptive_bottleneck_cover",
                        "threshold": threshold,
                        "n_labels": crossing,
                        "n_informative_counterexamples": informative,
                    }
                )
                oracle_thresholds.append(
                    {
                        **identity,
                        "policy": "retrospective_label_oracle",
                        "threshold": threshold,
                        "n_labels": retrospective_oracle_minimum_queries(
                            quantum, classical, labels, threshold
                        ),
                    }
                )

            for draw, order in enumerate(hash_orders):
                curve = audit_curve_from_order(quantum, classical, labels, order)
                for threshold in THRESHOLDS:
                    crossing = first_crossing(curve, threshold)
                    informative = (
                        -1
                        if crossing < 0
                        else int(
                            curve.loc[
                                curve.n_observed == crossing,
                                "n_informative_counterexamples",
                            ].iloc[0]
                        )
                    )
                    hash_threshold_draws.append(
                        {
                            **identity,
                            "policy": "hash_all",
                            "draw": draw,
                            "threshold": threshold,
                            "n_labels": crossing,
                            "n_informative_counterexamples": informative,
                        }
                    )
                selected = curve[curve.n_observed.isin(CURVE_BUDGETS)].copy()
                selected["draw"] = draw
                for key, value in identity.items():
                    selected[key] = value
                selected["policy"] = "hash_all"
                hash_curve_draws.append(selected)

                random_curve, random_order = (
                    adaptive_random_active_disagreement_curve(
                        quantum,
                        classical,
                        labels,
                        random_active_orders[draw],
                    )
                )
                if not np.array_equal(
                    np.sort(random_order), np.arange(len(labels))
                ):
                    raise RuntimeError(
                        "random-active policy did not query every target point"
                    )
                coverage_order = nonadaptive_initial_coverage_order(
                    quantum,
                    classical,
                    coverage_tie_orders[draw],
                )
                coverage_curve = audit_curve_from_order(
                    quantum,
                    classical,
                    labels,
                    coverage_order,
                )
                for policy, policy_curve, destination in (
                    (
                        "random_active_disagreement",
                        random_curve,
                        random_active_threshold_draws,
                    ),
                    (
                        "nonadaptive_initial_coverage",
                        coverage_curve,
                        nonadaptive_coverage_threshold_draws,
                    ),
                ):
                    for threshold in THRESHOLDS:
                        crossing = first_crossing(policy_curve, threshold)
                        informative = (
                            -1
                            if crossing < 0
                            else int(
                                policy_curve.loc[
                                    policy_curve.n_observed == crossing,
                                    "n_informative_counterexamples",
                                ].iloc[0]
                            )
                        )
                        destination.append(
                            {
                                **identity,
                                "policy": policy,
                                "draw": draw,
                                "threshold": threshold,
                                "n_labels": crossing,
                                "n_informative_counterexamples": informative,
                            }
                        )

    adaptive_curves_frame = pd.concat(adaptive_curves, ignore_index=True)
    adaptive_thresholds_frame = pd.DataFrame(adaptive_thresholds)
    hash_threshold_draws_frame = pd.DataFrame(hash_threshold_draws)
    hash_threshold_summary = summarize_hash_thresholds(hash_threshold_draws_frame)
    random_active_threshold_draws_frame = pd.DataFrame(
        random_active_threshold_draws
    )
    random_active_threshold_summary = summarize_hash_thresholds(
        random_active_threshold_draws_frame
    )
    nonadaptive_coverage_threshold_draws_frame = pd.DataFrame(
        nonadaptive_coverage_threshold_draws
    )
    nonadaptive_coverage_threshold_summary = summarize_hash_thresholds(
        nonadaptive_coverage_threshold_draws_frame
    )
    oracle_thresholds_frame = pd.DataFrame(oracle_thresholds)
    hash_curve_draws_frame = pd.concat(hash_curve_draws, ignore_index=True)
    curve_keys = [
        "case",
        "group",
        "model",
        "quantum_stratum",
        "tier",
        "n_observed",
    ]
    hash_curve_summary = (
        hash_curve_draws_frame.groupby(curve_keys, sort=True)
        .agg(
            median_lower=("lower", "median"),
            q025_lower=("lower", lambda x: x.quantile(0.025)),
            q975_lower=("lower", lambda x: x.quantile(0.975)),
            median_upper=("upper", "median"),
            q025_upper=("upper", lambda x: x.quantile(0.025)),
            q975_upper=("upper", lambda x: x.quantile(0.975)),
        )
        .reset_index()
    )
    summary_keys = [
        "case",
        "group",
        "model",
        "quantum_stratum",
        "tier",
        "threshold",
    ]

    def prefixed_summary(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return frame.rename(
            columns={
                column: f"{prefix}_{column}"
                for column in frame.columns
                if column not in summary_keys
            }
        )

    frontier_summary = adaptive_thresholds_frame.merge(
        hash_threshold_summary,
        on=summary_keys,
        suffixes=("_adaptive", "_hash"),
        validate="one_to_one",
    )
    frontier_summary = frontier_summary.merge(
        prefixed_summary(
            random_active_threshold_summary,
            "random_active",
        ),
        on=summary_keys,
        validate="one_to_one",
    ).merge(
        prefixed_summary(
            nonadaptive_coverage_threshold_summary,
            "initial_coverage",
        ),
        on=summary_keys,
        validate="one_to_one",
    ).merge(
        oracle_thresholds_frame[summary_keys + ["n_labels"]].rename(
            columns={"n_labels": "oracle_min_n_labels"}
        ),
        on=summary_keys,
        validate="one_to_one",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "adaptive_curves": args.out_dir / "adaptive_curves.csv",
        "adaptive_thresholds": args.out_dir / "adaptive_thresholds.csv",
        "hash_threshold_draws": args.out_dir / "hash_threshold_draws.csv",
        "hash_threshold_summary": args.out_dir / "hash_threshold_summary.csv",
        "hash_curve_summary": args.out_dir / "hash_curve_summary.csv",
        "random_active_threshold_draws": (
            args.out_dir / "random_active_threshold_draws.csv"
        ),
        "random_active_threshold_summary": (
            args.out_dir / "random_active_threshold_summary.csv"
        ),
        "nonadaptive_coverage_threshold_draws": (
            args.out_dir / "nonadaptive_coverage_threshold_draws.csv"
        ),
        "nonadaptive_coverage_threshold_summary": (
            args.out_dir / "nonadaptive_coverage_threshold_summary.csv"
        ),
        "retrospective_oracle_thresholds": (
            args.out_dir / "retrospective_oracle_thresholds.csv"
        ),
        "frontier_summary": args.out_dir / "evidence_frontier_summary.csv",
    }
    frames = {
        "adaptive_curves": adaptive_curves_frame,
        "adaptive_thresholds": adaptive_thresholds_frame,
        "hash_threshold_draws": hash_threshold_draws_frame,
        "hash_threshold_summary": hash_threshold_summary,
        "hash_curve_summary": hash_curve_summary,
        "random_active_threshold_draws": random_active_threshold_draws_frame,
        "random_active_threshold_summary": random_active_threshold_summary,
        "nonadaptive_coverage_threshold_draws": (
            nonadaptive_coverage_threshold_draws_frame
        ),
        "nonadaptive_coverage_threshold_summary": (
            nonadaptive_coverage_threshold_summary
        ),
        "retrospective_oracle_thresholds": oracle_thresholds_frame,
        "frontier_summary": frontier_summary,
    }
    for name, path in outputs.items():
        write_csv(frames[name], path)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "specification": "docs/PARTIAL_IDENTIFICATION_SPEC_V9.md#9",
                "status": "post-unlock exploratory",
                "model": args.model,
                "root": ROOT,
                "n_hash_draws": N_HASH_DRAWS,
                "comparison_policies": {
                    "random_active_disagreement": (
                        "dynamic uniform SHA order among unaudited disagreements "
                        "with current reducible bottleneck witnesses"
                    ),
                    "nonadaptive_initial_coverage": (
                        "fixed descending coverage of zero-label bottleneck "
                        "witnesses with SHA tie-breaking"
                    ),
                    "retrospective_label_oracle": (
                        "exact ex-post minimum number of favorable revealed "
                        "outcomes needed by any single witness"
                    ),
                },
                "thresholds": THRESHOLDS,
                "curve_budgets": CURVE_BUDGETS,
                "input_hashes": hashes,
                "outputs": {str(path): sha256_file(path) for path in outputs.values()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    display = frontier_summary[
        [
            "case",
            "model",
            "quantum_stratum",
            "tier",
            "threshold",
            "n_labels",
            "random_active_median_n_labels",
            "random_active_q025_n_labels",
            "random_active_q975_n_labels",
            "initial_coverage_median_n_labels",
            "initial_coverage_q025_n_labels",
            "initial_coverage_q975_n_labels",
            "median_n_labels",
            "q025_n_labels",
            "q975_n_labels",
            "oracle_min_n_labels",
        ]
    ]
    print(display.to_string(index=False))
    print(f"[ok] wrote partial-label frontier under {args.out_dir}")


if __name__ == "__main__":
    main()

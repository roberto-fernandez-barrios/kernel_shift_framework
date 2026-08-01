"""Open the frozen v1.0 target labels once and apply the prespecified audit.

The aggregate prediction lock is verified before this script records the
label-opening event.  Every policy, hash root, threshold, tier, and outcome
category is fixed in the immutable v1.0 protocol.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.partial_label_frontier_v9 import (
    adaptive_bottleneck_cover_curve,
    adaptive_random_active_disagreement_curve,
    audit_curve_from_order,
    first_crossing,
    nonadaptive_initial_coverage_order,
    retrospective_oracle_minimum_queries,
)
from src.analysis.partial_identification import (
    realized_accuracy_advantage,
    realized_balanced_accuracy_advantage,
    sharp_accuracy_envelope,
    sharp_balanced_accuracy_envelope,
)


PROTOCOL_ROOT = "ksf-v10-gate2-prospective-20260801"
SPEC_SHA256 = "3a8318d92d4af2aeeaf0c0edb069c3be59f31da6d1ee50fb6a6256e9d9d280b0"
ROOT = Path("results/v10/gate2_prospective")
LOCK_PATH = ROOT / "aggregate_prediction_lock_manifest.json"
OUT = ROOT / "audit"
THRESHOLDS = (0.005, 0.010, 0.020)
N_DRAWS = 200
MODELS = ("svc", "gpc")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def stable_hash_order(n_items: int, *parts: object) -> np.ndarray:
    prefix = "|".join(str(part) for part in (PROTOCOL_ROOT, *parts))
    keyed = []
    for index in range(n_items):
        digest = hashlib.sha256(f"{prefix}|{index}".encode("utf-8")).digest()
        keyed.append((digest, index))
    return np.asarray([index for _, index in sorted(keyed)], dtype=np.int64)


def quantum_stratum(kernel: str) -> str:
    return "entangling_zz" if kernel.startswith("zz_") else "product_map"


def metric_values(
    labels: np.ndarray, quantum: np.ndarray, classical: np.ndarray
) -> dict[str, float]:
    q_accuracy = float(np.mean(quantum == labels))
    c_accuracy = np.mean(classical == labels[:, None], axis=0)
    positive = labels == 1
    negative = ~positive
    q_bacc = 0.5 * (
        float(np.mean(quantum[positive] == 1))
        + float(np.mean(quantum[negative] == 0))
    )
    c_bacc = 0.5 * (
        np.mean(classical[positive] == 1, axis=0)
        + np.mean(classical[negative] == 0, axis=0)
    )
    return {
        "quantum_accuracy": q_accuracy,
        "best_classical_accuracy": float(np.max(c_accuracy)),
        "realized_accuracy_advantage": realized_accuracy_advantage(
            labels, quantum, classical
        ),
        "quantum_balanced_accuracy": q_bacc,
        "best_classical_balanced_accuracy": float(np.max(c_bacc)),
        "realized_balanced_accuracy_advantage": (
            realized_balanced_accuracy_advantage(labels, quantum, classical)
        ),
    }


def crossing_record(curve: pd.DataFrame, threshold: float) -> tuple[int, int]:
    crossing = first_crossing(curve, threshold)
    if crossing < 0:
        return -1, -1
    informative = int(
        curve.loc[
            curve.n_observed == crossing, "n_informative_counterexamples"
        ].iloc[0]
    )
    return crossing, informative


def summarize_draws(draws: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "model", "quantum_stratum", "tier", "policy", "threshold"]
    rows = []
    for key, frame in draws.groupby(keys, sort=True):
        raw = frame.n_labels.to_numpy(dtype=float)
        reached = raw >= 0
        values = raw[reached]
        record = dict(zip(keys, key))
        record.update(
            {
                "n_seed_draws": len(raw),
                "probability_reached": float(np.mean(reached)),
                "median_n_labels": float(np.median(values)) if len(values) else np.nan,
                "q025_n_labels": (
                    float(np.quantile(values, 0.025)) if len(values) else np.nan
                ),
                "q975_n_labels": (
                    float(np.quantile(values, 0.975)) if len(values) else np.nan
                ),
                "min_n_labels": float(np.min(values)) if len(values) else np.nan,
                "max_n_labels": float(np.max(values)) if len(values) else np.nan,
            }
        )
        for budget in (25, 50, 100):
            record[f"probability_by_{budget}"] = float(
                np.mean(reached & (raw <= budget))
            )
        rows.append(record)
    return pd.DataFrame(rows)


def verify_lock() -> tuple[dict[str, Any], str]:
    sidecar = LOCK_PATH.with_suffix(LOCK_PATH.suffix + ".sha256")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    observed = sha256_file(LOCK_PATH)
    if observed != expected:
        raise RuntimeError("aggregate prediction-lock manifest hash mismatch")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["specification_sha256"] != SPEC_SHA256:
        raise RuntimeError("aggregate lock points to a different specification")
    if lock["target_labels_opened_for_analysis"] is not False:
        raise RuntimeError("aggregate lock was not created before label opening")
    if int(lock["n_available_units"]) != 10:
        raise RuntimeError("frozen two-task replication must have ten available units")
    for unit in lock["units"]:
        for item in unit["files"]:
            path = Path(item["path"])
            if sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"locked input changed: {path}")
    return lock, observed


def classify_outcome(adaptive: pd.DataFrame, available_tasks: list[str]) -> dict[str, Any]:
    primary = adaptive[
        adaptive.tier.eq("full_115") & np.isclose(adaptive.threshold, 0.010)
    ].copy()
    if len(primary) != len(available_tasks) * 5 * 2:
        raise RuntimeError("primary outcome table is incomplete")
    if (primary.n_labels < 0).any():
        primary_values = primary.n_labels.replace(-1, 501)
    else:
        primary_values = primary.n_labels
    primary["outcome_n_labels"] = primary_values
    task_model = (
        primary.groupby(["task", "model"], sort=True)
        .outcome_n_labels.median()
        .rename("median_n_labels")
        .reset_index()
    )
    n_cells = len(task_model)
    overall_median = float(np.median(primary.outcome_n_labels))
    max_seed = int(primary.outcome_n_labels.max())
    n_task_model_at_most_50 = int((task_model.median_n_labels <= 50).sum())
    max_task_model_median = float(task_model.median_n_labels.max())

    technically_limited = len(available_tasks) == 2
    strong = (
        n_task_model_at_most_50 == n_cells
        and overall_median <= 25
        and max_seed <= 100
    )
    if technically_limited:
        partial = (
            n_task_model_at_most_50 == n_cells
            and overall_median <= 50
            and max_task_model_median <= 100
        )
        suffix = "__technically_limited_two_task_replication"
    else:
        partial = (
            n_task_model_at_most_50 >= 5
            and overall_median <= 50
            and max_task_model_median <= 100
        )
        suffix = ""
    if strong:
        category = "strong_prospective_transfer" + suffix
    elif partial:
        category = "partial_prospective_transfer" + suffix
    else:
        category = "failure_to_transfer" + suffix
    return {
        "outcome_category": category,
        "technically_limited": technically_limited,
        "available_tasks": available_tasks,
        "n_task_classifier_cells": n_cells,
        "n_task_classifier_medians_at_most_50": n_task_model_at_most_50,
        "overall_seed_level_median_n_labels": overall_median,
        "maximum_seed_level_n_labels": max_seed,
        "maximum_task_classifier_median_n_labels": max_task_model_median,
        "task_classifier_medians": task_model.to_dict(orient="records"),
        "primary_threshold": 0.010,
        "primary_tier": "full_115",
    }


def main() -> None:
    complete_path = OUT / "AUDIT_COMPLETE.json"
    if complete_path.is_file():
        print(f"[skip] prospective audit already complete: {complete_path}")
        return
    lock, lock_hash = verify_lock()
    opened_path = OUT / "AUDIT_OPENED.json"
    if not opened_path.exists():
        atomic_json(
            opened_path,
            {
                "status": "target_labels_opened_for_single_prespecified_audit",
                "opened_utc": datetime.now(timezone.utc).isoformat(),
                "aggregate_prediction_lock": LOCK_PATH.as_posix(),
                "aggregate_prediction_lock_sha256": lock_hash,
                "audit_script": Path(__file__).as_posix(),
                "audit_script_sha256": sha256_file(Path(__file__)),
                "protocol_root": PROTOCOL_ROOT,
            },
        )
    elif json.loads(opened_path.read_text(encoding="utf-8"))[
        "aggregate_prediction_lock_sha256"
    ] != lock_hash:
        raise RuntimeError("label-opening record belongs to a different lock")

    available_tasks = list(lock["available_tasks"])
    available_units = [unit for unit in lock["units"] if unit["status"] == "locked"]
    zero_rows: list[dict[str, Any]] = []
    bacc_rows: list[dict[str, Any]] = []
    adaptive_rows: list[pd.DataFrame] = []
    adaptive_threshold_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []

    for unit_index, unit in enumerate(available_units, start=1):
        task = str(unit["task"])
        seed = int(unit["seed"])
        unit_dir = ROOT / "prediction_locks" / task / f"seed_{seed}"
        label_path = ROOT / "sealed_labels" / task / f"seed_{seed}" / "evaluation_labels.npz"
        with np.load(label_path, allow_pickle=False) as archive:
            labels = archive["target_labels"].astype(np.int8)
            label_indices = archive["target_indices"].astype(np.int64)
        if labels.shape != (500,) or set(np.unique(labels).tolist()) != {0, 1}:
            raise RuntimeError(f"invalid opened labels: {label_path}")

        manifest = json.loads(
            (unit_dir / "prediction_lock_manifest.json").read_text(encoding="utf-8")
        )
        for model in MODELS:
            with np.load(unit_dir / f"predictions_{model}.npz", allow_pickle=False) as archive:
                indices = archive["target_indices"].astype(np.int64)
                quantum = archive["quantum_prediction"].astype(np.int8)
                classical_all = archive["classical_predictions"].astype(np.int8)
                customary = archive["customary_mask"].astype(bool)
            if not np.array_equal(indices, label_indices):
                raise RuntimeError(f"prediction/label row-order mismatch: {task}/{seed}/{model}")
            kernel = str(manifest["models"][model]["quantum_winner"]["kernel"])
            stratum = quantum_stratum(kernel)
            for tier, classical in (
                ("customary_30", classical_all[:, customary]),
                ("full_115", classical_all),
            ):
                identity = {
                    "task": task,
                    "seed": seed,
                    "model": model,
                    "quantum_kernel": kernel,
                    "quantum_stratum": stratum,
                    "tier": tier,
                    "n_target": len(labels),
                    "n_classical": classical.shape[1],
                }
                envelope = sharp_accuracy_envelope(quantum, classical)
                metrics = metric_values(labels, quantum, classical)
                zero_rows.append(
                    {
                        **identity,
                        "zero_label_lower": envelope.lower,
                        "zero_label_upper": envelope.upper,
                        "disagreement_min": envelope.disagreement_min,
                        "disagreement_max": envelope.disagreement_max,
                        "n_positive": int(labels.sum()),
                        **metrics,
                    }
                )

                bacc = sharp_balanced_accuracy_envelope(
                    quantum,
                    classical,
                    n_positive=int(labels.sum()),
                    integral=True,
                )
                bacc_rows.append(
                    {
                        **identity,
                        "status": "retrospective_realized_prevalence_sensitivity",
                        "n_positive": int(labels.sum()),
                        "n_signatures": bacc.n_signatures,
                        "bacc_lower": bacc.lower,
                        "bacc_upper": bacc.upper,
                        "realized_balanced_accuracy_advantage": metrics[
                            "realized_balanced_accuracy_advantage"
                        ],
                    }
                )

                tie_order = stable_hash_order(
                    len(labels), task, seed, model, tier, "adaptive_bottleneck_cover"
                )
                adaptive, chosen = adaptive_bottleneck_cover_curve(
                    quantum, classical, labels, tie_order
                )
                if not np.array_equal(np.sort(chosen), np.arange(len(labels))):
                    raise RuntimeError("adaptive policy did not exhaust the target")
                for key, value in identity.items():
                    adaptive[key] = value
                adaptive["policy"] = "adaptive_bottleneck_cover"
                adaptive_rows.append(adaptive)
                for threshold in THRESHOLDS:
                    n_labels, informative = crossing_record(adaptive, threshold)
                    adaptive_threshold_rows.append(
                        {
                            **identity,
                            "policy": "adaptive_bottleneck_cover",
                            "threshold": threshold,
                            "n_labels": n_labels,
                            "n_informative_counterexamples": informative,
                        }
                    )
                    oracle_rows.append(
                        {
                            **identity,
                            "policy": "retrospective_label_oracle",
                            "threshold": threshold,
                            "n_labels": retrospective_oracle_minimum_queries(
                                quantum, classical, labels, threshold
                            ),
                        }
                    )

                for draw in range(N_DRAWS):
                    random_tie = stable_hash_order(
                        len(labels),
                        task,
                        seed,
                        model,
                        tier,
                        "random_active_disagreement",
                        draw,
                    )
                    random_curve, _ = adaptive_random_active_disagreement_curve(
                        quantum, classical, labels, random_tie
                    )
                    coverage_tie = stable_hash_order(
                        len(labels),
                        task,
                        seed,
                        model,
                        tier,
                        "nonadaptive_initial_coverage",
                        draw,
                    )
                    coverage_order = nonadaptive_initial_coverage_order(
                        quantum, classical, coverage_tie
                    )
                    coverage_curve = audit_curve_from_order(
                        quantum, classical, labels, coverage_order
                    )
                    hash_order = stable_hash_order(
                        len(labels), task, seed, model, tier, "hash_all", draw
                    )
                    hash_curve = audit_curve_from_order(
                        quantum, classical, labels, hash_order
                    )
                    for policy, curve in (
                        ("random_active_disagreement", random_curve),
                        ("nonadaptive_initial_coverage", coverage_curve),
                        ("hash_all", hash_curve),
                    ):
                        for threshold in THRESHOLDS:
                            n_labels, informative = crossing_record(curve, threshold)
                            draw_rows.append(
                                {
                                    **identity,
                                    "policy": policy,
                                    "draw": draw,
                                    "threshold": threshold,
                                    "n_labels": n_labels,
                                    "n_informative_counterexamples": informative,
                                }
                            )
        print(f"[audit] {unit_index}/{len(available_units)} units complete", flush=True)

    zero = pd.DataFrame(zero_rows)
    bacc = pd.DataFrame(bacc_rows)
    adaptive_curves = pd.concat(adaptive_rows, ignore_index=True)
    adaptive_thresholds = pd.DataFrame(adaptive_threshold_rows)
    oracle = pd.DataFrame(oracle_rows)
    draws = pd.DataFrame(draw_rows)
    draw_summary = summarize_draws(draws)
    outcome = classify_outcome(adaptive_thresholds, available_tasks)

    outputs = {
        "zero_label_and_realized": zero,
        "balanced_accuracy_sensitivity": bacc,
        "adaptive_curves": adaptive_curves,
        "adaptive_thresholds": adaptive_thresholds,
        "retrospective_oracle_thresholds": oracle,
        "comparison_draws": draws,
        "comparison_summary": draw_summary,
    }
    output_paths = {}
    for name, frame in outputs.items():
        path = OUT / f"{name}.csv"
        atomic_csv(frame, path)
        output_paths[path.as_posix()] = sha256_file(path)
    outcome_path = OUT / "prospective_outcome.json"
    atomic_json(
        outcome_path,
        {
            **outcome,
            "classified_utc": datetime.now(timezone.utc).isoformat(),
            "aggregate_prediction_lock_sha256": lock_hash,
            "specification_sha256": SPEC_SHA256,
        },
    )
    output_paths[outcome_path.as_posix()] = sha256_file(outcome_path)
    atomic_json(
        complete_path,
        {
            "status": "prespecified_prospective_audit_complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "aggregate_prediction_lock_sha256": lock_hash,
            "audit_opening_record_sha256": sha256_file(opened_path),
            "outcome_category": outcome["outcome_category"],
            "outputs": output_paths,
        },
    )
    print(json.dumps(outcome, indent=2))
    print(f"[complete] wrote prospective audit under {OUT}")


if __name__ == "__main__":
    main()

"""Reconstruct per-example predictions for the frozen v0.6 shot replicates.

The v0.6 artifacts contain aggregate metrics only. This v0.9 extension uses
the identical fixed cases, block-specific SHA-256 seeds, shot counts, and PSD
conditions, then requires every reconstructed selected C and OOD BAcc to match
the archived v0.6 row before exporting hard predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.experiments.export_partial_identification_v9 import (  # noqa: E402
    atomic_json,
    atomic_savez,
)
from scripts.experiments.run_shots_mc_v6 import (  # noqa: E402
    FIXED_RUNS,
    N_REPLICATES,
    PROJECTION_CONDITIONS,
    SHOTS,
    build_exact_blocks,
    group_for_run,
    locate_run,
    nystrom_psd_extension,
    sha256_file,
    stable_measurement_seed,
)
from src.experiments.ember.extended.v4_protocol import (  # noqa: E402
    sample_kernel_finite_shots,
    select_c_by_train_cv,
)


def fit_shot_svc(
    train_gram: np.ndarray,
    target_gram: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    c_selected, _ = select_c_by_train_cv(train_gram, y_train)
    model = SVC(
        kernel="precomputed",
        C=float(c_selected),
        class_weight="balanced",
    ).fit(train_gram, y_train)
    prediction = model.predict(target_gram).astype(np.int8)
    score = np.asarray(model.decision_function(target_gram), dtype=np.float64).ravel()
    return float(c_selected), prediction, score


def archived_row(
    archive: pd.DataFrame,
    shots: int,
    replicate: int,
    condition: str,
) -> pd.Series:
    hit = archive[
        (archive.shots.astype(int) == shots)
        & (archive.replicate.astype(int) == replicate)
        & (archive.projection_condition == condition)
    ]
    if len(hit) != 1:
        raise RuntimeError(
            f"expected one archived row for shots={shots}, replicate={replicate}, "
            f"condition={condition}; found {len(hit)}"
        )
    return hit.iloc[0]


def run_one(
    run_index: int,
    result_roots: tuple[Path, ...],
    v6_root: Path,
    prediction_root: Path,
    output_root: Path,
    force: bool,
) -> Path:
    fixed = FIXED_RUNS[run_index]
    group = group_for_run(fixed.run)
    case_name = f"{run_index:02d}_{group}"
    output_dir = output_root / case_name
    output_path = output_dir / "shot_predictions_svc.npz"
    metadata_path = output_dir / "shot_metadata_svc.json"
    if output_path.is_file() and metadata_path.is_file() and not force:
        print(f"[skip] complete {output_dir}", flush=True)
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    v6_path = v6_root / f"{case_name}.csv"
    if not v6_path.is_file():
        raise FileNotFoundError(v6_path)
    archived = pd.read_csv(v6_path)
    expected_rows = len(SHOTS) * N_REPLICATES * len(PROJECTION_CONDITIONS)
    if len(archived) != expected_rows:
        raise RuntimeError(f"{v6_path} has {len(archived)} rows, expected {expected_rows}")

    result_dir = locate_run(fixed.run, result_roots)
    exact_blocks, exact_metadata = build_exact_blocks(fixed, result_dir)
    y_train = exact_metadata["y_train"]
    y_ood = exact_metadata["y_eval"]["ood_test"]
    n_target = len(y_ood)

    exact_model = SVC(
        kernel="precomputed",
        C=float(fixed.exact_c),
        class_weight="balanced",
    ).fit(exact_blocks["train"], y_train)
    exact_prediction = exact_model.predict(exact_blocks["ood_test"]).astype(np.int8)
    exact_score = np.asarray(
        exact_model.decision_function(exact_blocks["ood_test"]), dtype=np.float64
    ).ravel()
    exact_bacc = float(balanced_accuracy_score(y_ood, exact_prediction))
    archived_exact = float(archived.exact_ood_test_bacc.iloc[0])
    if abs(exact_bacc - archived_exact) > 1e-12:
        raise RuntimeError(
            f"{case_name}: exact BAcc {exact_bacc} != archived {archived_exact}"
        )

    shape = (len(SHOTS), N_REPLICATES, len(PROJECTION_CONDITIONS), n_target)
    predictions = np.empty(shape, dtype=np.int8)
    scores = np.empty(shape, dtype=np.float64)
    selected_cs = np.empty(shape[:-1], dtype=np.float64)
    baccs = np.empty(shape[:-1], dtype=np.float64)
    audit_rows: list[dict[str, Any]] = []
    started = time.time()
    for shot_index, shots in enumerate(SHOTS):
        for replicate in range(N_REPLICATES):
            sampled: dict[str, np.ndarray] = {}
            projected: dict[str, np.ndarray] = {}
            for block in ("train", "id_val", "id_test", "ood_test", "ood_square"):
                seed = stable_measurement_seed(
                    fixed.run,
                    fixed.kernel,
                    fixed.dim,
                    shots,
                    replicate,
                    block,
                )
                pre, post, _ = sample_kernel_finite_shots(
                    exact_blocks[block],
                    shots,
                    np.random.default_rng(seed),
                    square=block in {"train", "ood_square"},
                )
                sampled[block] = pre
                projected[block] = post
            nystrom, _ = nystrom_psd_extension(sampled)
            condition_blocks = {
                "pre_psd": sampled,
                "independent_square_psd": projected,
                "nystrom_psd": nystrom,
            }
            for condition_index, condition in enumerate(PROJECTION_CONDITIONS):
                blocks = condition_blocks[condition]
                c_selected, prediction, score = fit_shot_svc(
                    blocks["train"],
                    blocks["ood_test"],
                    y_train,
                )
                bacc = float(balanced_accuracy_score(y_ood, prediction))
                expected = archived_row(archived, shots, replicate, condition)
                c_error = c_selected - float(expected.selected_c)
                bacc_error = bacc - float(expected.ood_test_bacc)
                if c_error != 0.0 or abs(bacc_error) > 1e-12:
                    raise RuntimeError(
                        f"{case_name}/{shots}/{replicate}/{condition}: "
                        f"C error={c_error}, BAcc error={bacc_error}"
                    )
                predictions[shot_index, replicate, condition_index] = prediction
                scores[shot_index, replicate, condition_index] = score
                selected_cs[shot_index, replicate, condition_index] = c_selected
                baccs[shot_index, replicate, condition_index] = bacc
                audit_rows.append(
                    {
                        "shots": shots,
                        "replicate": replicate,
                        "projection_condition": condition,
                        "selected_c": c_selected,
                        "selected_c_error": c_error,
                        "ood_bacc": bacc,
                        "ood_bacc_error": bacc_error,
                    }
                )
        print(
            f"[{group}] shots={shots} complete "
            f"({(time.time() - started) / 60:.1f} min)",
            flush=True,
        )

    atomic_savez(
        output_path,
        shots=np.asarray(SHOTS, dtype=np.int32),
        projection_conditions=np.asarray(PROJECTION_CONDITIONS, dtype=str),
        exact_quantum_prediction=exact_prediction,
        exact_quantum_score=exact_score,
        shot_quantum_predictions=predictions,
        shot_quantum_scores=scores,
        selected_cs=selected_cs,
        ood_baccs=baccs,
    )
    audit_path = output_dir / "shot_integrity_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    atomic_json(
        metadata_path,
        {
            "specification": "docs/PARTIAL_IDENTIFICATION_SPEC_V9.md",
            "status": "post-hoc exploratory extension of frozen v0.6 replicates",
            "run_index": run_index,
            "run": fixed.run,
            "group": group,
            "kernel": fixed.kernel,
            "dim": fixed.dim,
            "exact_c": fixed.exact_c,
            "exact_ood_bacc": exact_bacc,
            "n_target": n_target,
            "n_rows": len(audit_rows),
            "source_v6_path": str(v6_path),
            "source_v6_sha256": sha256_file(v6_path),
            "prediction_sha256": sha256_file(output_path),
            "audit_sha256": sha256_file(audit_path),
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"[ok] wrote {output_dir} in {(time.time() - started) / 60:.1f} min")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-index", type=int, required=True, choices=range(len(FIXED_RUNS))
    )
    parser.add_argument(
        "--result-roots",
        type=Path,
        nargs="+",
        default=[
            Path("results/ember_shift/extended_kernels"),
            Path("results/netflow/extended_kernels"),
        ],
    )
    parser.add_argument(
        "--v6-root", type=Path, default=Path("results/v6/shots_mc/runs")
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("results/v9/partial_identification/predictions"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/v9/partial_identification/shot_predictions"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_one(
        args.run_index,
        tuple(args.result_roots),
        args.v6_root,
        args.prediction_root,
        args.out_dir,
        args.force,
    )


if __name__ == "__main__":
    main()

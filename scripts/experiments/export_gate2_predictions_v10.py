"""Prospectively lock Gate-2 predictions before opening target labels.

This is stage 1 of the frozen v1.0 protocol.  It deliberately never computes
or prints a target-label metric, a disagreement, a prevalence, or an audit
curve.  Target labels are copied to a physically separate sealed tree and are
then removed from the model-fitting state.  Resume-safe candidate checkpoints
contain predictions and validation-only selection quantities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.experiments.export_partial_identification_v9 import customary_kernel
from src.experiments.ember.extended.run_classical_extensions import (
    V4_ANGLE_SCALES,
    v4_classical_kernels,
)
from src.experiments.ember.extended.run_ember_extended_kernels_qsplits import (
    ClassicalKernelFactory,
    LaplaceGPC,
)
from src.experiments.ember.extended.v4_protocol import (
    C_GRID_FULL,
    select_c_by_train_cv,
)
from src.experiments.ember.quantum.run_ember_quantum_kernel_sparsity_shift_qsplits import (
    DEFAULT_QUANTUM_CONFIGS,
    build_feature_map,
    compute_statevectors_batch,
)
from src.experiments.tableshift.run_external_validation_v5 import (
    DIMS,
    PROTOCOL_SEED,
    fidelity_blocks,
    load_unit,
    prepare_representations,
    resolve_kernel_backend,
)


SPECIFICATION = Path("docs/GATE2_PROSPECTIVE_REPLICATION_SPEC_V10.md")
FROZEN_SPEC_SHA256 = "3a8318d92d4af2aeeaf0c0edb069c3be59f31da6d1ee50fb6a6256e9d9d280b0"
TASKS = ("brfss_diabetes", "acsfoodstamps", "nhanes_lead")
SEEDS = (42, 123, 999, 7, 2024)
MODELS = ("svc", "gpc")
N_TARGET = 500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _predict_both(
    blocks: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_validation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit frozen SVC/GPC and return validation metrics plus target decisions."""
    c_selected, c_scores = select_c_by_train_cv(
        blocks["train"],
        y_train,
        grid=C_GRID_FULL,
        folds=5,
        seed=PROTOCOL_SEED,
    )
    svc = SVC(
        kernel="precomputed",
        C=float(c_selected),
        class_weight="balanced",
    ).fit(blocks["train"], y_train)
    svc_validation = svc.predict(blocks["validation"]).astype(np.int8)
    svc_target = svc.predict(blocks["ood_test"]).astype(np.int8)

    gpc = LaplaceGPC().fit(blocks["train"], y_train)
    gpc_validation_prob = np.asarray(
        gpc.predict_proba(
            blocks["validation"], np.ones(blocks["validation"].shape[0])
        ),
        dtype=np.float64,
    ).ravel()
    gpc_target_prob = np.asarray(
        gpc.predict_proba(
            blocks["ood_test"], np.ones(blocks["ood_test"].shape[0])
        ),
        dtype=np.float64,
    ).ravel()
    gpc_validation = (gpc_validation_prob >= 0.5).astype(np.int8)
    gpc_target = (gpc_target_prob >= 0.5).astype(np.int8)
    return {
        "svc_target_prediction": svc_target,
        "gpc_target_prediction": gpc_target,
        "svc_validation_bacc": np.array(
            balanced_accuracy_score(y_validation, svc_validation), dtype=np.float64
        ),
        "gpc_validation_bacc": np.array(
            balanced_accuracy_score(y_validation, gpc_validation), dtype=np.float64
        ),
        "svc_c_selected": np.array(c_selected, dtype=np.float64),
        "svc_c_cv_score": np.array(c_scores[c_selected], dtype=np.float64),
    }


def _checkpoint(
    path: Path,
    family: str,
    kernel: str,
    dim: int,
    blocks: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_validation: np.ndarray,
    target_row_sha256: str,
) -> None:
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["spec_sha256"].item()) == FROZEN_SPEC_SHA256
                and str(saved["target_row_sha256"].item()) == target_row_sha256
                and str(saved["family"].item()) == family
                and str(saved["kernel"].item()) == kernel
                and int(saved["dim"].item()) == dim
                and saved["svc_target_prediction"].shape == (N_TARGET,)
                and saved["gpc_target_prediction"].shape == (N_TARGET,)
            ):
                return
        raise RuntimeError(f"incompatible candidate checkpoint: {path}")
    result = _predict_both(blocks, y_train, y_validation)
    atomic_savez(
        path,
        spec_sha256=np.array(FROZEN_SPEC_SHA256),
        target_row_sha256=np.array(target_row_sha256),
        family=np.array(family),
        kernel=np.array(kernel),
        dim=np.array(dim, dtype=np.int16),
        **result,
    )


def _candidate_path(cache_dir: Path, family: str, kernel: str, dim: int) -> Path:
    return cache_dir / f"{family}__{safe_name(kernel)}__d{dim}.npz"


def _validate_spec() -> None:
    observed = sha256_file(SPECIFICATION)
    if observed != FROZEN_SPEC_SHA256:
        raise RuntimeError(
            f"frozen specification changed: expected {FROZEN_SPEC_SHA256}, got {observed}"
        )


def _read_source_positions(export_root: Path, task: str, seed: int) -> np.ndarray:
    path = export_root / task / f"seed_{seed}" / "ood_test.csv"
    positions = pd.read_csv(path, usecols=["__source_position__"], nrows=N_TARGET)[
        "__source_position__"
    ].to_numpy(dtype=np.int64)
    if positions.shape != (N_TARGET,) or len(np.unique(positions)) != N_TARGET:
        raise RuntimeError(f"invalid target source positions in {path}")
    return positions


def _assemble_unit(
    task: str,
    seed: int,
    unit_dir: Path,
    label_dir: Path,
    source_positions: np.ndarray,
    preprocess_audit: dict[str, Any],
    backend: str,
    elapsed_seconds: float,
) -> Path:
    cache_dir = unit_dir / "candidate_cache"
    classical = []
    for dim in DIMS:
        for kernel in v4_classical_kernels():
            path = _candidate_path(cache_dir, "classical_ext", kernel, dim)
            if not path.is_file():
                raise RuntimeError(f"missing classical checkpoint: {path}")
            classical.append((kernel, dim, path))
    if len(classical) != 115:
        raise RuntimeError(f"expected 115 classical checkpoints, found {len(classical)}")
    classical.sort(key=lambda item: f"{item[0]}__svc__d{item[1]}")

    quantum = []
    for dim in DIMS:
        for scale in V4_ANGLE_SCALES:
            suffix = "" if scale == 1.0 else f"__as{scale:g}"
            for config in DEFAULT_QUANTUM_CONFIGS:
                kernel = str(config["id"]) + suffix
                path = _candidate_path(cache_dir, "quantum", kernel, dim)
                if not path.is_file():
                    raise RuntimeError(f"missing quantum checkpoint: {path}")
                quantum.append((kernel, dim, path))
    if len(quantum) != 60:
        raise RuntimeError(f"expected 60 quantum checkpoints, found {len(quantum)}")

    classical_kernels = np.array([item[0] for item in classical], dtype=str)
    classical_dims = np.array([item[1] for item in classical], dtype=np.int16)
    customary_mask = np.array([customary_kernel(value) for value in classical_kernels])
    if int(customary_mask.sum()) != 30:
        raise RuntimeError(f"expected customary_30, found {customary_mask.sum()}")

    manifest_models: dict[str, Any] = {}
    for model in MODELS:
        q_ordered = sorted(quantum, key=lambda item: f"{item[0]}__{model}__d{item[1]}")
        q_validation = []
        for _, _, path in q_ordered:
            with np.load(path, allow_pickle=False) as saved:
                q_validation.append(float(saved[f"{model}_validation_bacc"].item()))
        winner_index = int(np.nanargmax(np.asarray(q_validation, dtype=np.float64)))
        winner_kernel, winner_dim, winner_path = q_ordered[winner_index]
        with np.load(winner_path, allow_pickle=False) as saved:
            quantum_prediction = saved[f"{model}_target_prediction"].astype(np.int8)
            winner_c = (
                float(saved["svc_c_selected"].item()) if model == "svc" else None
            )
        classical_predictions = np.empty((N_TARGET, 115), dtype=np.int8)
        classical_cs = np.full(115, np.nan, dtype=np.float64)
        for index, (_, _, path) in enumerate(classical):
            with np.load(path, allow_pickle=False) as saved:
                classical_predictions[:, index] = saved[
                    f"{model}_target_prediction"
                ].astype(np.int8)
                if model == "svc":
                    classical_cs[index] = float(saved["svc_c_selected"].item())

        cfgs = np.array(
            [f"{kernel}__{model}__d{dim}" for kernel, dim, _ in classical],
            dtype=str,
        )
        prediction_path = unit_dir / f"predictions_{model}.npz"
        atomic_savez(
            prediction_path,
            target_indices=source_positions,
            quantum_prediction=quantum_prediction,
            classical_predictions=classical_predictions,
            classical_cfgs=cfgs,
            classical_kernels=classical_kernels,
            classical_dims=classical_dims,
            classical_c_selected=classical_cs,
            customary_mask=customary_mask,
        )
        model_meta = {
            "model": model,
            "n_target": N_TARGET,
            "n_classical": 115,
            "n_customary": 30,
            "quantum_winner": {
                "cfg": f"{winner_kernel}__{model}__d{winner_dim}",
                "kernel": winner_kernel,
                "dim": winner_dim,
                "c_selected": winner_c,
                "selection_split": "validation",
                "selection_balanced_accuracy": q_validation[winner_index],
            },
            "prediction_artifact": prediction_path.as_posix(),
            "prediction_sha256": sha256_file(prediction_path),
        }
        atomic_json(unit_dir / f"metadata_{model}.json", model_meta)
        manifest_models[model] = model_meta

    label_path = label_dir / "evaluation_labels.npz"
    if not label_path.is_file():
        raise RuntimeError(f"sealed label artifact disappeared: {label_path}")
    manifest = {
        "status": "prospective_predictions_locked_before_target_audit",
        "specification": SPECIFICATION.as_posix(),
        "specification_sha256": FROZEN_SPEC_SHA256,
        "task": task,
        "stratum": "q1000",
        "seed": seed,
        "target_role": "ood_test",
        "n_target": N_TARGET,
        "target_indices_sha256": sha256_array(source_positions),
        "sealed_label_artifact": label_path.as_posix(),
        "sealed_label_artifact_sha256": sha256_file(label_path),
        "target_labels_opened_for_analysis": False,
        "kernel_backend": backend,
        "preprocessing": preprocess_audit,
        "models": manifest_models,
        "elapsed_seconds": elapsed_seconds,
    }
    manifest_path = unit_dir / "prediction_lock_manifest.json"
    atomic_json(manifest_path, manifest)
    return manifest_path


def run_unit(args: argparse.Namespace) -> Path:
    _validate_spec()
    started = time.time()
    unit_dir = args.out_root / args.task / f"seed_{args.seed}"
    label_dir = args.label_root / args.task / f"seed_{args.seed}"
    final_manifest = unit_dir / "prediction_lock_manifest.json"
    if final_manifest.is_file() and not args.force:
        print(f"[skip] locked {args.task}/seed_{args.seed}", flush=True)
        return final_manifest

    X, labels, schema = load_unit(
        args.export_root,
        args.schema_root / f"tableshift_schema_{args.task}.csv",
        args.task,
        "q1000",
        args.seed,
    )
    source_positions = _read_source_positions(args.export_root, args.task, args.seed)
    target_labels = labels.pop("ood_test").astype(np.int8)
    labels.pop("id_test", None)
    label_path = label_dir / "evaluation_labels.npz"
    if label_path.is_file():
        with np.load(label_path, allow_pickle=False) as sealed:
            if not (
                np.array_equal(sealed["target_indices"], source_positions)
                and np.array_equal(sealed["target_labels"], target_labels)
            ):
                raise RuntimeError(f"sealed labels changed for {args.task}/seed_{args.seed}")
    else:
        atomic_savez(
            label_path,
            target_indices=source_positions,
            target_labels=target_labels,
        )
    atomic_json(
        label_dir / "sealed_label_manifest.json",
        {
            "status": "sealed_for_prediction_lock_stage",
            "task": args.task,
            "seed": args.seed,
            "n_target": N_TARGET,
            "label_artifact_sha256": sha256_file(label_path),
            "target_indices_sha256": sha256_array(source_positions),
            "labels_inspected_for_target_analysis": False,
        },
    )
    del target_labels

    try:
        representations, preprocess_audit = prepare_representations(X, schema, dims=DIMS)
    except ValueError as error:
        message = str(error)
        if "non-constant post-encoding features; need at least 12" not in message:
            raise
        failure_path = unit_dir / "technical_unavailability_manifest.json"
        atomic_json(
            failure_path,
            {
                "status": "technical_unavailability_before_model_execution",
                "permitted_gate": "minimum-feature failure",
                "specification": SPECIFICATION.as_posix(),
                "specification_sha256": FROZEN_SPEC_SHA256,
                "task": args.task,
                "stratum": "q1000",
                "seed": args.seed,
                "reason": message,
                "n_model_candidates_executed": 0,
                "target_labels_opened_for_analysis": False,
                "sealed_label_artifact": label_path.as_posix(),
                "sealed_label_artifact_sha256": sha256_file(label_path),
            },
        )
        print(
            f"[unavailable] {args.task}/seed_{args.seed}: minimum-feature gate "
            "failed before model execution",
            flush=True,
        )
        return failure_path
    representations = {
        dim: {
            role: values
            for role, values in embedded.items()
            if role in {"train", "validation", "ood_test"}
        }
        for dim, embedded in representations.items()
    }
    backend = resolve_kernel_backend(args.backend)
    cache_dir = unit_dir / "candidate_cache"
    target_row_sha256 = sha256_array(source_positions)
    completed = 0

    with threadpool_limits(limits=args.threads):
        for dim in DIMS:
            embedded = representations[dim]
            factory = ClassicalKernelFactory(embedded["train"], seed=PROTOCOL_SEED)
            for kernel in v4_classical_kernels():
                path = _candidate_path(cache_dir, "classical_ext", kernel, dim)
                if not path.is_file():
                    blocks = {
                        split: factory.block(kernel, values, embedded["train"])
                        for split, values in embedded.items()
                    }
                    _checkpoint(
                        path,
                        "classical_ext",
                        kernel,
                        dim,
                        blocks,
                        labels["train"],
                        labels["validation"],
                        target_row_sha256,
                    )
                    del blocks
                completed += 1

            for scale in V4_ANGLE_SCALES:
                scaled = (
                    embedded
                    if scale == 1.0
                    else {split: values * scale for split, values in embedded.items()}
                )
                suffix = "" if scale == 1.0 else f"__as{scale:g}"
                for config in DEFAULT_QUANTUM_CONFIGS:
                    kernel = str(config["id"]) + suffix
                    path = _candidate_path(cache_dir, "quantum", kernel, dim)
                    if not path.is_file():
                        feature_map = build_feature_map(config, feature_dim=dim)
                        states = {
                            split: compute_statevectors_batch(
                                values, feature_map, dtype=np.complex64
                            )
                            for split, values in scaled.items()
                        }
                        block_backend = backend
                        try:
                            blocks = fidelity_blocks(states, backend)
                        except Exception:
                            if backend != "cuda":
                                raise
                            blocks = fidelity_blocks(states, "cpu")
                            block_backend = "cpu_fallback"
                        _checkpoint(
                            path,
                            "quantum",
                            kernel,
                            dim,
                            blocks,
                            labels["train"],
                            labels["validation"],
                            target_row_sha256,
                        )
                        del states, blocks
                    completed += 1
            print(
                f"[lock] {args.task}/seed_{args.seed} d{dim}: "
                f"{completed}/175 candidates checkpointed",
                flush=True,
            )

    manifest_path = _assemble_unit(
        args.task,
        args.seed,
        unit_dir,
        label_dir,
        source_positions,
        preprocess_audit,
        backend,
        time.time() - started,
    )
    print(
        f"[locked] {args.task}/seed_{args.seed} without target-label analysis "
        f"({(time.time() - started) / 60:.1f} min)",
        flush=True,
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path("data/raw/tableshift/v10/exports"),
    )
    parser.add_argument(
        "--schema-root",
        type=Path,
        default=Path("results/v10/gate2_prospective/acquisition"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/v10/gate2_prospective/prediction_locks"),
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path("results/v10/gate2_prospective/sealed_labels"),
    )
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be >=1")
    run_unit(args)


if __name__ == "__main__":
    main()

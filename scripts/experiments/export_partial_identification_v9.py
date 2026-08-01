"""Export per-example target predictions for the frozen v0.9 pilot.

The expensive v4 grids archive aggregate metrics.  This runner reconstructs
only the eight q1000 cases frozen for the v0.6 shot analysis, exports the
P1'-selected quantum classifier and the complete 115-candidate classical
reference family, and verifies every reconstructed OOD metric against its
frozen v4 summary.

Target labels are written to a physically separate evaluation artifact.  The
partial-identification analysis consumes only ``predictions_<model>.npz`` and
``metadata_<model>.json``.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.experiments.run_phase_b_driver import resolve  # noqa: E402
from scripts.experiments.run_shots_mc_v6 import (  # noqa: E402
    FIXED_RUNS,
    group_for_run,
    locate_run,
    matches_frozen_text_sha256,
    parse_quantum_kernel,
    sha256_file,
)
from src.experiments.ember.extended.run_classical_extensions import (  # noqa: E402
    v4_classical_kernels,
)
from src.experiments.ember.extended.run_ember_extended_kernels_qsplits import (  # noqa: E402
    ClassicalKernelFactory,
    LaplaceGPC,
)
from src.experiments.ember.quantum.run_ember_quantum_kernel_sparsity_shift_qsplits import (  # noqa: E402
    DEFAULT_QUANTUM_CONFIGS,
    build_feature_map,
    compute_statevectors_batch,
    kernel_block_abs2,
    load_indices,
    make_embedding_pipeline,
)


MODELS = ("svc", "gpc")
TOLERANCE = 1e-10


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_p1_candidate(summary: pd.DataFrame, family: str, model: str) -> pd.Series:
    """Apply the v4 stable lexicographic tie break on ID-validation BAcc."""
    candidates = summary[
        (summary.family == family)
        & (summary.model == model)
        & (summary.split == "id_val")
    ].sort_values("cfg", kind="stable")
    if candidates.empty or candidates.balanced_accuracy.isna().all():
        raise ValueError(f"no valid {family}/{model} ID-validation candidates")
    position = int(
        np.nanargmax(candidates.balanced_accuracy.to_numpy(dtype=np.float64))
    )
    return candidates.iloc[position]


def customary_kernel(kernel: str) -> bool:
    """Frozen customary 30-candidate reference: linear plus all RBF blocks."""
    return kernel == "linear" or kernel.startswith("rbf_gscale")


def fit_predict(
    model_name: str,
    train_gram: np.ndarray,
    target_gram: np.ndarray,
    y_train: np.ndarray,
    c_selected: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "svc":
        if c_selected is None or not np.isfinite(c_selected):
            raise ValueError("SVC requires a finite frozen c_selected")
        model = SVC(
            kernel="precomputed",
            C=float(c_selected),
            class_weight="balanced",
        ).fit(train_gram, y_train)
        prediction = model.predict(target_gram).astype(np.int8)
        score = np.asarray(model.decision_function(target_gram), dtype=np.float64).ravel()
        return prediction, score
    if model_name == "gpc":
        model = LaplaceGPC().fit(train_gram, y_train)
        probability = np.asarray(
            model.predict_proba(target_gram, np.ones(target_gram.shape[0])),
            dtype=np.float64,
        ).ravel()
        return (probability >= 0.5).astype(np.int8), probability
    raise ValueError(f"unknown model {model_name}")


def expected_ood_row(summary: pd.DataFrame, cfg: str) -> pd.Series:
    row = summary[(summary.cfg == cfg) & (summary.split == "ood_test")]
    if len(row) != 1:
        raise RuntimeError(f"expected one frozen OOD row for {cfg}, found {len(row)}")
    return row.iloc[0]


def audit_prediction(
    cfg: str,
    prediction: np.ndarray,
    labels: np.ndarray,
    expected: pd.Series,
) -> dict[str, Any]:
    observed_accuracy = float(accuracy_score(labels, prediction))
    observed_bacc = float(balanced_accuracy_score(labels, prediction))
    expected_accuracy = float(expected.accuracy)
    expected_bacc = float(expected.balanced_accuracy)
    accuracy_error = observed_accuracy - expected_accuracy
    bacc_error = observed_bacc - expected_bacc
    if abs(accuracy_error) > TOLERANCE or abs(bacc_error) > TOLERANCE:
        raise RuntimeError(
            f"{cfg}: reconstructed OOD metrics differ from frozen summary: "
            f"accuracy error={accuracy_error}, BAcc error={bacc_error}"
        )
    return {
        "cfg": cfg,
        "family": str(expected.family),
        "model": str(expected.model),
        "observed_accuracy": observed_accuracy,
        "expected_accuracy": expected_accuracy,
        "accuracy_error": accuracy_error,
        "observed_balanced_accuracy": observed_bacc,
        "expected_balanced_accuracy": expected_bacc,
        "balanced_accuracy_error": bacc_error,
    }


def _embedding(
    X: np.ndarray,
    indices: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    dim: int,
    seed: int,
) -> dict[str, np.ndarray]:
    pipeline = make_embedding_pipeline(
        dim=dim,
        select_k=None,
        use_scaling=True,
        angle_min=0.0,
        angle_max=float(np.pi),
        seed=seed,
    )
    pipeline.fit(np.asarray(X[indices["train"]]), labels["train"])
    return {
        role: np.asarray(
            pipeline.transform(np.asarray(X[indices[role]])),
            dtype=np.float64,
        )
        for role in ("train", "ood_test")
    }


def _quantum_blocks(
    embedded: dict[str, np.ndarray],
    kernel: str,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    base_kernel, angle_scale = parse_quantum_kernel(kernel)
    config = next(
        candidate for candidate in DEFAULT_QUANTUM_CONFIGS if candidate["id"] == base_kernel
    )
    feature_map = build_feature_map(config, feature_dim=dim)
    train_states = compute_statevectors_batch(
        embedded["train"] * angle_scale,
        feature_map,
        dtype=np.complex64,
    )
    target_states = compute_statevectors_batch(
        embedded["ood_test"] * angle_scale,
        feature_map,
        dtype=np.complex64,
    )
    return (
        kernel_block_abs2(train_states, train_states, out_dtype=np.float64),
        kernel_block_abs2(target_states, train_states, out_dtype=np.float64),
    )


def run_one(
    run_index: int,
    roots: tuple[Path, ...],
    output_root: Path,
    models: tuple[str, ...],
    force: bool,
) -> Path:
    fixed = FIXED_RUNS[run_index]
    result_dir = locate_run(fixed.run, roots)
    resolved = resolve(fixed.run)
    if resolved is None:
        raise RuntimeError(f"cannot resolve raw inputs for {fixed.run}")
    input_dir, splits_dir, model_seed = resolved
    if model_seed != 42:
        raise RuntimeError(f"v0.9 frozen pilot expected model seed 42, found {model_seed}")

    case_dir = output_root / f"{run_index:02d}_{group_for_run(fixed.run)}"
    complete = all(
        (case_dir / f"predictions_{model}.npz").is_file()
        and (case_dir / f"metadata_{model}.json").is_file()
        for model in models
    )
    if complete and not force:
        print(f"[skip] complete {case_dir}", flush=True)
        return case_dir
    case_dir.mkdir(parents=True, exist_ok=True)

    summary_path = result_dir / "summary_v4.csv"
    if not matches_frozen_text_sha256(summary_path, fixed.summary_sha256):
        raise RuntimeError(
            f"{fixed.run}: summary hash {sha256_file(summary_path)} does not match "
            "the previously frozen v0.6 hash"
        )
    summary = pd.read_csv(summary_path)
    classical_meta: dict[str, pd.DataFrame] = {}
    quantum_winners: dict[str, pd.Series] = {}
    for model_name in models:
        meta = summary[
            (summary.family == "classical_ext")
            & (summary.model == model_name)
            & (summary.split == "ood_test")
        ].sort_values("cfg", kind="stable")
        if len(meta) != 115 or meta.cfg.nunique() != 115:
            raise RuntimeError(
                f"{fixed.run}/{model_name}: expected 115 classical candidates, "
                f"found {len(meta)} rows and {meta.cfg.nunique()} unique cfgs"
            )
        classical_meta[model_name] = meta.reset_index(drop=True)
        quantum_winners[model_name] = select_p1_candidate(summary, "quantum", model_name)

    X = np.load(input_dir / "X.npy", mmap_mode="r")
    y = np.load(input_dir / "y.npy").astype(np.int64).ravel()
    indices = {
        role: load_indices(splits_dir / f"{role}_idx.npy")
        for role in ("train", "ood_test")
    }
    labels = {role: y[index] for role, index in indices.items()}
    n_target = len(indices["ood_test"])
    if n_target != 500:
        raise RuntimeError(f"v0.9 frozen pilot expected 500 OOD rows, found {n_target}")

    embeddings: dict[int, dict[str, np.ndarray]] = {}
    for dim in (4, 6, 8, 10, 12):
        embeddings[dim] = _embedding(X, indices, labels, dim, model_seed)

    predictions = {
        model_name: np.empty((n_target, 115), dtype=np.int8)
        for model_name in models
    }
    scores = {
        model_name: np.empty((n_target, 115), dtype=np.float64)
        for model_name in models
    }
    audits: list[dict[str, Any]] = []
    classical_kernels = v4_classical_kernels()
    if len(classical_kernels) != 23:
        raise RuntimeError(f"expected 23 v4 classical blocks, found {len(classical_kernels)}")

    started = time.time()
    for dim in (4, 6, 8, 10, 12):
        embedded = embeddings[dim]
        factory = ClassicalKernelFactory(embedded["train"], seed=model_seed)
        for kernel in classical_kernels:
            train_gram = factory.block(kernel, embedded["train"], embedded["train"])
            target_gram = factory.block(kernel, embedded["ood_test"], embedded["train"])
            for model_name in models:
                meta = classical_meta[model_name]
                hit = meta[(meta.kernel == kernel) & (meta.dim.astype(int) == dim)]
                if len(hit) != 1:
                    raise RuntimeError(
                        f"{fixed.run}/{model_name}/{kernel}/d{dim}: expected one candidate"
                    )
                row = hit.iloc[0]
                position = int(hit.index[0])
                c_selected = float(row.c_selected) if model_name == "svc" else None
                prediction, score = fit_predict(
                    model_name,
                    train_gram,
                    target_gram,
                    labels["train"],
                    c_selected,
                )
                predictions[model_name][:, position] = prediction
                scores[model_name][:, position] = score
                audits.append(
                    audit_prediction(str(row.cfg), prediction, labels["ood_test"], row)
                )
        print(
            f"[{group_for_run(fixed.run)}] classical d{dim} complete "
            f"({(time.time() - started) / 60:.1f} min)",
            flush=True,
        )

    for model_name in models:
        winner = quantum_winners[model_name]
        dim = int(winner.dim)
        train_gram, target_gram = _quantum_blocks(
            embeddings[dim],
            str(winner.kernel),
            dim,
        )
        c_selected = float(winner.c_selected) if model_name == "svc" else None
        q_prediction, q_score = fit_predict(
            model_name,
            train_gram,
            target_gram,
            labels["train"],
            c_selected,
        )
        expected = expected_ood_row(summary, str(winner.cfg))
        audits.append(
            audit_prediction(str(winner.cfg), q_prediction, labels["ood_test"], expected)
        )

        meta = classical_meta[model_name]
        cfgs = meta.cfg.astype(str).to_numpy(dtype=str)
        kernels = meta.kernel.astype(str).to_numpy(dtype=str)
        dims = meta.dim.astype(int).to_numpy(dtype=np.int16)
        customary_mask = np.array([customary_kernel(value) for value in kernels])
        if int(customary_mask.sum()) != 30:
            raise RuntimeError(
                f"{fixed.run}/{model_name}: customary tier has {customary_mask.sum()} candidates"
            )

        prediction_path = case_dir / f"predictions_{model_name}.npz"
        atomic_savez(
            prediction_path,
            target_indices=indices["ood_test"].astype(np.int64),
            quantum_prediction=q_prediction,
            quantum_score=q_score,
            classical_predictions=predictions[model_name],
            classical_scores=scores[model_name],
            classical_cfgs=cfgs,
            classical_kernels=kernels,
            classical_dims=dims,
            customary_mask=customary_mask,
        )
        metadata = {
            "specification": "docs/PARTIAL_IDENTIFICATION_SPEC_V9.md",
            "status": "post-hoc exploratory",
            "run_index": run_index,
            "run": fixed.run,
            "group": group_for_run(fixed.run),
            "model": model_name,
            "target_role": "ood_test",
            "threadpool_limit": 2,
            "n_target": n_target,
            "n_classical": 115,
            "n_customary": 30,
            "quantum_winner": {
                "cfg": str(winner.cfg),
                "kernel": str(winner.kernel),
                "dim": dim,
                "c_selected": c_selected,
                "selection_split": "id_val",
                "selection_balanced_accuracy": float(winner.balanced_accuracy),
            },
            "summary_v4_sha256": sha256_file(summary_path),
            "target_indices_sha256": sha256_array(indices["ood_test"].astype(np.int64)),
            "prediction_artifact_sha256": sha256_file(prediction_path),
            "elapsed_seconds_at_write": time.time() - started,
        }
        atomic_json(case_dir / f"metadata_{model_name}.json", metadata)

    atomic_savez(
        case_dir / "evaluation_labels.npz",
        target_indices=indices["ood_test"].astype(np.int64),
        target_labels=labels["ood_test"].astype(np.int8),
    )
    audit_frame = pd.DataFrame(audits).sort_values(
        ["model", "family", "cfg"], kind="stable"
    )
    temporary_audit = case_dir / "integrity_audit.csv.tmp"
    audit_frame.to_csv(temporary_audit, index=False)
    temporary_audit.replace(case_dir / "integrity_audit.csv")
    atomic_json(
        case_dir / "case_manifest.json",
        {
            "run_index": run_index,
            "run": fixed.run,
            "group": group_for_run(fixed.run),
            "models": list(models),
            "n_integrity_rows": len(audit_frame),
            "max_abs_accuracy_error": float(audit_frame.accuracy_error.abs().max()),
            "max_abs_balanced_accuracy_error": float(
                audit_frame.balanced_accuracy_error.abs().max()
            ),
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"[ok] wrote {case_dir} in {(time.time() - started) / 60:.1f} min", flush=True)
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-index",
        type=int,
        required=True,
        choices=range(len(FIXED_RUNS)),
    )
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=[
            Path("results/ember_shift/extended_kernels"),
            Path("results/netflow/extended_kernels"),
        ],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/v9/partial_identification/predictions"),
    )
    parser.add_argument("--models", choices=MODELS, nargs="+", default=["svc"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    # The frozen v4 Windows run is prediction-sensitive at an SVC boundary to
    # BLAS thread scheduling. The v0.9 execution-integrity amendment fixes two
    # thread and retains exact aggregate-metric equality as a hard gate.
    with threadpool_limits(limits=2):
        run_one(
            args.run_index,
            tuple(args.roots),
            args.out_dir,
            tuple(args.models),
            args.force,
        )


if __name__ == "__main__":
    main()

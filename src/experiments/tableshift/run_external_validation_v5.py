"""Run one frozen v5 TableShift task/size/seed unit.

The exporter in scripts/data/export_tableshift_v5.py fixes the published
TableShift split and the label-blind nested sample.  This runner fits every
representation transform on `train` only, evaluates the frozen 60-quantum and
115-classical geometry pools, and writes resume-safe per-configuration rows.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC

from src.experiments.ember.extended.run_classical_extensions import (
    V4_ANGLE_SCALES,
    v4_classical_kernels,
)
from src.experiments.ember.extended.run_ember_extended_kernels_qsplits import (
    ClassicalKernelFactory,
    LaplaceGPC,
    probabilistic_metrics,
)
from src.experiments.ember.extended.v4_protocol import (
    C_GRID_FULL,
    select_c_by_train_cv,
)
from src.experiments.ember.quantum.run_ember_quantum_kernel_sparsity_shift_qsplits import (
    DEFAULT_QUANTUM_CONFIGS,
    build_feature_map,
    compute_statevectors_batch,
    eval_split,
)


PROTOCOL_SEED = 20260729
DIMS = (4, 6, 8, 10, 12)
SIZES = {
    "q1000": {"train": 1000, "validation": 250, "id_test": 250, "ood_test": 500},
    "q2000": {"train": 2000, "validation": 500, "id_test": 500, "ood_test": 1000},
}
RESERVED_COLUMNS = {"__target__", "__source_position__", "__sample_rank__"}
SUMMARY_COLUMNS = (
    "task", "stratum", "seed", "family", "kernel", "dim", "candidate_id",
    "model", "regularization", "c_selected", "c_cv_score", "split",
    "accuracy", "balanced_accuracy", "f1_macro", "f1_pos", "roc_auc",
    "pr_auc", "log_loss", "brier", "ece", "mean_predictive_entropy",
    "fit_seconds", "kernel_backend",
)


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_unit(
    export_root: Path,
    schema_path: Path,
    task: str,
    stratum: str,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], pd.DataFrame]:
    if stratum not in SIZES:
        raise ValueError(f"unknown stratum {stratum}")
    schema = pd.read_csv(schema_path)
    if set(schema.task) != {task}:
        raise ValueError(f"schema {schema_path} does not describe task {task}")
    feature_columns = schema.column.tolist()
    X: dict[str, pd.DataFrame] = {}
    y: dict[str, np.ndarray] = {}
    source_positions: dict[str, np.ndarray] = {}
    for split, n_rows in SIZES[stratum].items():
        path = export_root / task / f"seed_{seed}" / f"{split}.csv"
        frame = pd.read_csv(path, nrows=n_rows)
        missing = set(feature_columns) - set(frame)
        if missing:
            raise ValueError(f"{path} lacks schema columns {sorted(missing)}")
        if len(frame) != n_rows:
            raise ValueError(f"{path} has {len(frame)} rows; expected {n_rows}")
        labels = frame["__target__"].to_numpy(dtype=np.int64)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"class-presence gate failed for {path}")
        X[split] = frame[feature_columns].copy()
        y[split] = labels
        source_positions[split] = frame["__source_position__"].to_numpy(dtype=np.int64)
    names = tuple(SIZES[stratum])
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if len(np.intersect1d(source_positions[left], source_positions[right])):
                raise ValueError(f"sampled splits overlap: {left}/{right}")
    return X, y, schema


def _coerce_semantic_types(
    X: dict[str, pd.DataFrame], schema: pd.DataFrame
) -> tuple[list[str], list[str]]:
    numeric = schema.loc[schema.role == "numeric", "column"].tolist()
    categorical = schema.loc[schema.role == "categorical", "column"].tolist()
    if set(numeric).intersection(categorical):
        raise ValueError("schema assigns a feature to multiple roles")
    if set(numeric + categorical) != set(schema.column):
        raise ValueError("schema contains unsupported or missing roles")
    for split in X:
        for column in numeric:
            X[split][column] = pd.to_numeric(X[split][column], errors="coerce")
        for column in categorical:
            values = X[split][column]
            X[split][column] = values.where(values.isna(), values.astype(str))
    return numeric, categorical


def prepare_representations(
    X: dict[str, pd.DataFrame],
    schema: pd.DataFrame,
    dims: Iterable[int] = DIMS,
    seed: int = PROTOCOL_SEED,
) -> tuple[dict[int, dict[str, np.ndarray]], dict]:
    """Fit train-only imputation/encoding/scaling/SVD/angle transforms."""
    X = {split: frame.copy() for split, frame in X.items()}
    numeric, categorical = _coerce_semantic_types(X, schema)
    train = X["train"]
    kept = [
        column for column in schema.column
        if train[column].nunique(dropna=False) > 1
    ]
    if not kept:
        raise ValueError("every feature is constant on training")
    numeric = [column for column in numeric if column in kept]
    categorical = [column for column in categorical if column in kept]
    X = {split: frame[kept] for split, frame in X.items()}

    transformers = []
    if numeric:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric)
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    encoder = ColumnTransformer(
        transformers, remainder="drop", sparse_threshold=1.0
    )
    encoded = {"train": encoder.fit_transform(X["train"])}
    for split in X:
        if split != "train":
            encoded[split] = encoder.transform(X[split])
    maxabs = MaxAbsScaler()
    encoded["train"] = maxabs.fit_transform(encoded["train"])
    for split in X:
        if split != "train":
            encoded[split] = maxabs.transform(encoded[split])

    n_encoded = int(encoded["train"].shape[1])
    dims = tuple(map(int, dims))
    if n_encoded < max(dims):
        raise ValueError(
            f"only {n_encoded} non-constant post-encoding features; "
            f"need at least {max(dims)}"
        )

    output: dict[int, dict[str, np.ndarray]] = {}
    svd_variance = {}
    for dim in dims:
        svd = TruncatedSVD(n_components=dim, random_state=seed)
        reduced = {
            "train": svd.fit_transform(encoded["train"]),
        }
        for split in X:
            if split != "train":
                reduced[split] = svd.transform(encoded[split])
        standard = StandardScaler()
        reduced["train"] = standard.fit_transform(reduced["train"])
        for split in X:
            if split != "train":
                reduced[split] = standard.transform(reduced[split])
        angle = MinMaxScaler(feature_range=(0.0, float(np.pi)))
        reduced["train"] = angle.fit_transform(reduced["train"])
        for split in X:
            if split != "train":
                reduced[split] = angle.transform(reduced[split])
        output[dim] = {
            split: np.asarray(values, dtype=np.float64)
            for split, values in reduced.items()
        }
        svd_variance[str(dim)] = float(svd.explained_variance_ratio_.sum())

    audit = {
        "n_raw_features": int(len(schema)),
        "n_train_nonconstant_features": int(len(kept)),
        "n_dropped_constant_features": int(len(schema) - len(kept)),
        "n_numeric_features": int(len(numeric)),
        "n_categorical_features": int(len(categorical)),
        "n_post_encoding_features": n_encoded,
        "kept_columns": kept,
        "svd_explained_variance_ratio_sum": svd_variance,
        "fit_split": "train",
        "target_used": False,
        "protocol_seed": seed,
    }
    return output, audit


def resolve_kernel_backend(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unknown kernel backend {requested}")
    if requested == "cpu":
        return "cpu"
    try:
        import cupy as cp
        _ = cp.cuda.runtime.getDeviceCount()
        # Exercise cuBLAS now so auto never fails halfway through a unit.
        probe = cp.ones((2, 2), dtype=cp.complex64)
        _ = probe @ probe
        cp.cuda.Stream.null.synchronize()
        return "cuda"
    except Exception:
        if requested == "cuda":
            raise
        return "cpu"


def fidelity_blocks(
    states: dict[str, np.ndarray], backend: str
) -> dict[str, np.ndarray]:
    """Return exact fidelity blocks against train, using optional CUDA GEMM."""
    if backend == "cuda":
        import cupy as cp

        train = cp.asarray(states["train"])
        output = {}
        for split, values in states.items():
            device = train if split == "train" else cp.asarray(values)
            overlaps = device.conj() @ train.T
            kernel = cp.abs(overlaps) ** 2
            output[split] = cp.asnumpy(kernel).astype(np.float64, copy=False)
            del device, overlaps, kernel
        output["train"] = (output["train"] + output["train"].T) / 2.0
        cp.get_default_memory_pool().free_all_blocks()
        return output

    train = states["train"]
    output = {}
    for split, values in states.items():
        overlaps = values.conj() @ train.T
        output[split] = (np.abs(overlaps) ** 2).real.astype(np.float64)
    output["train"] = (output["train"] + output["train"].T) / 2.0
    return output


def _model_rows(
    task: str,
    stratum: str,
    seed: int,
    family: str,
    kernel: str,
    dim: int,
    blocks: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    model_name: str,
    regularization: str,
    kernel_backend: str,
) -> list[dict]:
    c_scores: dict[float, float] = {}
    if model_name == "svc":
        if regularization == "train_cv":
            c_selected, c_scores = select_c_by_train_cv(
                blocks["train"],
                labels["train"],
                grid=C_GRID_FULL,
                folds=5,
                seed=PROTOCOL_SEED,
            )
        elif regularization == "fixed_c1":
            c_selected = 1.0
        else:
            raise ValueError(f"unknown SVC regularization {regularization}")
        model = SVC(
            kernel="precomputed", C=c_selected, class_weight="balanced"
        )
    elif model_name == "gpc":
        if regularization != "not_applicable":
            raise ValueError("GPC regularization must be not_applicable")
        c_selected = np.nan
        model = LaplaceGPC()
    else:
        raise ValueError(f"unknown model {model_name}")

    started = time.time()
    model.fit(blocks["train"], labels["train"])
    fit_seconds = time.time() - started
    rows = []
    candidate_id = f"{kernel}__d{dim}"
    for split in ("validation", "id_test", "ood_test"):
        if model_name == "svc":
            prediction = model.predict(blocks[split]).astype(np.int64)
            scores = np.asarray(model.decision_function(blocks[split])).ravel()
            metrics = eval_split(labels[split], prediction, scores)
        else:
            probabilities = model.predict_proba(
                blocks[split], np.ones(blocks[split].shape[0])
            )
            prediction = (probabilities >= 0.5).astype(np.int64)
            metrics = eval_split(labels[split], prediction, probabilities)
            metrics.update(probabilistic_metrics(labels[split], probabilities))
        row = {
            "task": task,
            "stratum": stratum,
            "seed": seed,
            "family": family,
            "kernel": kernel,
            "dim": dim,
            "candidate_id": candidate_id,
            "model": model_name,
            "regularization": regularization,
            "c_selected": c_selected,
            "c_cv_score": c_scores.get(c_selected, np.nan),
            "split": split,
            "fit_seconds": fit_seconds,
            "kernel_backend": kernel_backend,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_pos": metrics["f1_pos"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        }
        for metric in ("log_loss", "brier", "ece", "mean_predictive_entropy"):
            row[metric] = metrics.get(metric)
        rows.append(row)
    return rows


def _is_complete(
    summary: pd.DataFrame,
    family: str,
    kernel: str,
    dim: int,
    model: str,
    regularization: str,
) -> bool:
    if summary.empty:
        return False
    mask = (
        summary.family.eq(family)
        & summary.kernel.eq(kernel)
        & summary.dim.eq(dim)
        & summary.model.eq(model)
        & summary.regularization.eq(regularization)
    )
    return set(summary.loc[mask, "split"]) == {"validation", "id_test", "ood_test"}


def _required_jobs(family: str, kernel: str, models: tuple[str, ...]):
    for model in models:
        regularization = "train_cv" if model == "svc" else "not_applicable"
        yield model, regularization
    if "svc" in models and (
        family == "quantum" or kernel in {"linear", "rbf_gscale"}
    ):
        yield "svc", "fixed_c1"


def run_unit(args) -> None:
    unit_dir = args.out_root / args.task / args.stratum / f"seed_{args.seed}"
    summary_path = unit_dir / "summary_v5.csv"
    X, labels, schema = load_unit(
        args.export_root, args.schema, args.task, args.stratum, args.seed
    )
    representations, preprocess_audit = prepare_representations(
        X, schema, dims=args.dims
    )
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "preprocess_audit.json").write_text(
        json.dumps(preprocess_audit, indent=2), encoding="utf-8"
    )
    if args.preflight_only:
        print(
            f"[OK] preflight {args.task}/{args.stratum}/seed_{args.seed}: "
            f"{preprocess_audit['n_post_encoding_features']} encoded features"
        )
        return

    backend = resolve_kernel_backend(args.backend)
    models = tuple(args.models)
    summary = (
        pd.read_csv(summary_path)
        if summary_path.exists()
        else pd.DataFrame(columns=SUMMARY_COLUMNS)
    )

    def evaluate(family, kernel, dim, blocks):
        nonlocal summary
        for model, regularization in _required_jobs(family, kernel, models):
            if _is_complete(summary, family, kernel, dim, model, regularization):
                continue
            new_rows = _model_rows(
                args.task, args.stratum, args.seed, family, kernel, dim,
                blocks, labels, model, regularization, backend,
            )
            summary = pd.concat(
                [summary, pd.DataFrame(new_rows)], ignore_index=True
            )
            summary = summary.drop_duplicates(
                ["family", "kernel", "dim", "model", "regularization", "split"],
                keep="last",
            )
            atomic_to_csv(summary, summary_path)

    for dim in args.dims:
        embedded = representations[dim]
        factory = ClassicalKernelFactory(embedded["train"], seed=PROTOCOL_SEED)
        if "classical_ext" in args.families:
            for kernel in v4_classical_kernels():
                jobs = list(_required_jobs("classical_ext", kernel, models))
                if all(
                    _is_complete(summary, "classical_ext", kernel, dim, model, reg)
                    for model, reg in jobs
                ):
                    continue
                blocks = {
                    split: factory.block(kernel, values, embedded["train"])
                    for split, values in embedded.items()
                }
                evaluate("classical_ext", kernel, dim, blocks)

        if "quantum" in args.families:
            for scale in V4_ANGLE_SCALES:
                scaled = (
                    embedded
                    if scale == 1.0
                    else {split: values * scale for split, values in embedded.items()}
                )
                suffix = "" if scale == 1.0 else f"__as{scale:g}"
                for config in DEFAULT_QUANTUM_CONFIGS:
                    kernel = config["id"] + suffix
                    jobs = list(_required_jobs("quantum", kernel, models))
                    if all(
                        _is_complete(summary, "quantum", kernel, dim, model, reg)
                        for model, reg in jobs
                    ):
                        continue
                    feature_map = build_feature_map(config, feature_dim=dim)
                    states = {
                        split: compute_statevectors_batch(
                            values, feature_map, dtype=np.complex64
                        )
                        for split, values in scaled.items()
                    }
                    blocks = fidelity_blocks(states, backend)
                    evaluate("quantum", kernel, dim, blocks)
                    del states, blocks
        print(
            f"[OK] {args.task}/{args.stratum}/seed_{args.seed} dim={dim}; "
            f"{len(summary)} rows",
            flush=True,
        )
    print(f"[OK] complete unit: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--stratum", choices=tuple(SIZES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path("data/raw/tableshift/v5/exports"),
    )
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/v5/external/runs"),
    )
    parser.add_argument(
        "--models", nargs="+", choices=("svc", "gpc"), default=["svc", "gpc"]
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("classical_ext", "quantum"),
        default=["classical_ext", "quantum"],
    )
    parser.add_argument(
        "--dims", type=int, nargs="+", choices=DIMS, default=list(DIMS)
    )
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run_unit(args)


if __name__ == "__main__":
    main()

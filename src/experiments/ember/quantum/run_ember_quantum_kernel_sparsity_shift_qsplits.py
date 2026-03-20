# src/experiments/ember/quantum/run_ember_quantum_kernel_sparsity_shift_qsplits.py
from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)



# ----------------------------
# Optional quantum dependency
# ----------------------------
def _require_qiskit():
    try:
        from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap
        from qiskit.quantum_info import Statevector
        return ZZFeatureMap, PauliFeatureMap, ZFeatureMap, Statevector
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "qiskit is required to run the quantum experiments. Install the pinned environment "
            "from requirements.txt or environment.yml (expected: qiskit==2.3.1)."
        ) from exc


# ----------------------------
# Defaults
# ----------------------------
DEFAULT_IN_DIR = Path("data/processed/ember")
DEFAULT_OUT_DIR = Path("results/ember_shift/quantum_kernel_sparsity")

DEFAULT_SEED = 42
DEFAULT_DIMS = [4, 6]

DEFAULT_SELECT_K: Optional[int] = None
DEFAULT_USE_SCALING = True
DEFAULT_ANGLE_MIN = 0.0
DEFAULT_ANGLE_MAX = float(np.pi)

DEFAULT_SVC_C = 1.0
DEFAULT_CLASS_WEIGHT = "balanced"

DEFAULT_THRESH_SOURCE = "train"
DEFAULT_THRESH_CRITERION = "balanced_accuracy"
DEFAULT_THRESH_GRID = 401

DEFAULT_QUANTUM_CONFIGS = [
    {"id": "zz_r1_full", "map_type": "zz", "reps": 1, "entanglement": "full"},
    {"id": "zz_r2_full", "map_type": "zz", "reps": 2, "entanglement": "full"},
    {"id": "pauli_xz_r1_full", "map_type": "pauli", "reps": 1, "entanglement": "full", "paulis": ["X", "Z"]},
    {"id": "zmap_r2", "map_type": "z", "reps": 2},
]


# ----------------------------
# Split structure
# ----------------------------
@dataclass(frozen=True)
class Split:
    name: str
    idx_path: Path


# ----------------------------
# Split / safety helpers
# ----------------------------
def load_indices(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    arr = np.load(path).astype(np.int64).ravel()
    if arr.size == 0:
        raise RuntimeError(f"Empty indices array: {path}")
    return arr


def _assert_disjoint_fast(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    # O(n log n) but avoids Python sets.
    a = np.asarray(a, dtype=np.int64).ravel()
    b = np.asarray(b, dtype=np.int64).ravel()
    inter = np.intersect1d(a, b, assume_unique=False)
    if inter.size:
        raise RuntimeError(f"Overlap between {name_a} and {name_b}: {int(inter.size)} indices")


def _infer_expected_qsizes_from_splits_dir(splits_dir: Path) -> Optional[Dict[str, int]]:
    m = re.search(r"_q(\d+)_id(\d+)_ood(\d+)_seed(\d+)\b", splits_dir.name)
    if not m:
        return None
    q, idn, ood = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return {"train": q, "id_test": idn, "ood_test": ood}


def _load_expected_qsizes_from_meta(splits_dir: Path) -> Optional[Dict[str, int]]:
    p = splits_dir / "meta_q.json"
    if not p.exists():
        return None
    meta = json.loads(p.read_text(encoding="utf-8"))
    sizes = meta.get("actual_sizes") or meta.get("requested_sizes") or meta.get("sizes") or {}
    n_train = sizes.get("n_train")
    n_id = sizes.get("n_id") or sizes.get("n_id_test")
    n_ood = sizes.get("n_ood") or sizes.get("n_ood_test")
    if n_train is None or n_id is None or n_ood is None:
        return None
    return {"train": int(n_train), "id_test": int(n_id), "ood_test": int(n_ood)}


# ----------------------------
# Metrics helpers
# ----------------------------
def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def safe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, scores))


def scores_from_precomputed_svc(model: SVC, K: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "decision_function"):
        s = model.decision_function(K)
        return np.asarray(s).ravel()
    return None


def predict_with_threshold(scores: np.ndarray, thr: float) -> np.ndarray:
    return (scores >= thr).astype(np.int64)


def best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    criterion: str,
    grid_size: int,
) -> float:
    if scores.size == 0:
        return 0.0
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if lo == hi:
        return lo

    thresholds = np.linspace(lo, hi, int(grid_size))
    best_thr = float(thresholds[0])
    best_val = -1e18

    for thr in thresholds:
        y_pred = predict_with_threshold(scores, float(thr))
        if criterion == "balanced_accuracy":
            val = balanced_accuracy_score(y_true, y_pred)
        elif criterion == "f1_pos":
            val = f1_score(y_true, y_pred, pos_label=1)
        else:
            raise ValueError(f"Unknown threshold criterion: {criterion}")
        if val > best_val:
            best_val = float(val)
            best_thr = float(thr)

    return float(best_thr)


def eval_split(y_true: np.ndarray, y_pred: np.ndarray, scores: Optional[np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    out["f1_pos"] = float(f1_score(y_true, y_pred, pos_label=1))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["confusion_matrix"] = cm.tolist()

    if scores is not None:
        out["roc_auc"] = safe_roc_auc(y_true, scores)
        out["pr_auc"] = safe_pr_auc(y_true, scores)
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None

    return out


def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


# ----------------------------
# Embedding pipeline
# ----------------------------
def make_embedding_pipeline(
    dim: int,
    select_k: Optional[int],
    use_scaling: bool,
    angle_min: float,
    angle_max: float,
    seed: int,
) -> Pipeline:
    steps: List[Tuple[str, Any]] = []

    if select_k is not None:
        steps.append(("selectk", SelectKBest(chi2, k=int(select_k))))

    # For sparse-ish EMBER, MaxAbsScaler is safe and cheap.
    steps.append(("scale_maxabs", MaxAbsScaler()))
    steps.append(("svd", TruncatedSVD(n_components=int(dim), random_state=seed)))

    if use_scaling:
        steps.append(("std", StandardScaler()))
        steps.append(("minmax", MinMaxScaler(feature_range=(angle_min, angle_max))))

    return Pipeline(steps)


# ----------------------------
# Quantum kernel utilities
# ----------------------------
def build_feature_map(cfg: Dict[str, Any], feature_dim: int):
    ZZFeatureMap, PauliFeatureMap, ZFeatureMap, _ = _require_qiskit()

    t = cfg["map_type"]  # zz | pauli | z
    reps = int(cfg.get("reps", 1))
    ent = cfg.get("entanglement", "full")

    if t == "zz":
        return ZZFeatureMap(feature_dimension=feature_dim, reps=reps, entanglement=ent)
    if t == "pauli":
        paulis = cfg.get("paulis", ["X", "Z"])
        return PauliFeatureMap(feature_dimension=feature_dim, reps=reps, paulis=paulis, entanglement=ent)
    if t == "z":
        return ZFeatureMap(feature_dimension=feature_dim, reps=reps)

    raise ValueError(f"Unknown map_type: {t}")


def _statevector_dim(num_qubits: int) -> int:
    return 2 ** int(num_qubits)


def compute_statevectors_batch(X_angles: np.ndarray, feature_map, *, dtype=np.complex64) -> np.ndarray:
    """
    No list/vstack. Returns (batch, 2**dim) complex.
    """
    *_, Statevector = _require_qiskit()

    X_angles = np.asarray(X_angles)
    n = int(X_angles.shape[0])
    dim_sv = _statevector_dim(int(feature_map.num_qubits))
    SV = np.empty((n, dim_sv), dtype=dtype)

    for i, x in enumerate(X_angles):
        qc = feature_map.assign_parameters(x)
        SV[i] = Statevector.from_instruction(qc).data.astype(dtype, copy=False)

    return SV


def kernel_block_abs2(SV_X: np.ndarray, SV_Z: np.ndarray, *, out_dtype=np.float32) -> np.ndarray:
    """
    Computes |<x|z>|^2 for blocks:
      overlaps = SV_X.conj() @ SV_Z.T   => complex
      K = abs(overlaps)^2              => real
    Returns float32 by default.
    """
    overlaps = SV_X.conj() @ SV_Z.T
    K = (np.abs(overlaps) ** 2).real
    return K.astype(out_dtype, copy=False)


def estimate_kernel_mb(n_rows: int, n_cols: int, dtype: np.dtype) -> float:
    return (n_rows * n_cols * np.dtype(dtype).itemsize) / (1024.0**2)


def make_memmap(path: Path, shape: Tuple[int, int], dtype: np.dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(str(path), mode="w+", dtype=dtype, shape=shape)


def fit_qsvc_with_memmap_kernel(
    SV_train: np.ndarray,
    y_train: np.ndarray,
    *,
    svc_c: float,
    class_weight: Optional[str],
    out_dir: Path,
    kernel_dtype: np.dtype,
    kernel_block_rows: int,
) -> Tuple[SVC, float, float, Path]:
    """
    Builds K_train on disk (memmap) by blocks to keep RAM stable, then fits SVC(kernel="precomputed").
    Returns model + timings + path to memmap file.
    """
    n_train = int(SV_train.shape[0])
    K_path = out_dir / "K_train.memmap"
    K = make_memmap(K_path, shape=(n_train, n_train), dtype=kernel_dtype)

    tK = time.time()
    # Block over rows; each block computes overlaps with full SV_train (still manageable)
    for i0 in range(0, n_train, int(kernel_block_rows)):
        i1 = min(n_train, i0 + int(kernel_block_rows))
        K[i0:i1, :] = kernel_block_abs2(SV_train[i0:i1], SV_train, out_dtype=kernel_dtype)
        K.flush()
    kernel_train_s = time.time() - tK

    tfit = time.time()
    svm_q = SVC(kernel="precomputed", C=svc_c, class_weight=class_weight)
    svm_q.fit(K, y_train)
    svc_fit_s = time.time() - tfit

    return svm_q, float(kernel_train_s), float(svc_fit_s), K_path


def predict_scores_streaming(
    model: SVC,
    SV_split_angles: np.ndarray,
    feature_map,
    SV_train: np.ndarray,
    *,
    kernel_dtype: np.dtype,
    sv_batch: int,
    kernel_out_block_rows: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], float, float]:
    """
    Streaming prediction: never holds full K_split in memory.
    Computes SV for split in batches, then K_batch vs SV_train, then predicts/scores.
    Returns (y_pred, scores_or_None, sv_seconds_total, kernel_seconds_total).
    """
    n_split = int(SV_split_angles.shape[0])
    y_pred = np.empty(n_split, dtype=np.int64)
    scores_all: Optional[np.ndarray] = None
    can_score = hasattr(model, "decision_function")

    if can_score:
        scores_all = np.empty(n_split, dtype=np.float64)

    sv_total = 0.0
    k_total = 0.0

    # We compute SV in sv_batch chunks, then kernel in (same) chunks
    bs = max(1, int(sv_batch))
    for i0 in range(0, n_split, bs):
        i1 = min(n_split, i0 + bs)

        tsv = time.time()
        SV_b = compute_statevectors_batch(SV_split_angles[i0:i1], feature_map, dtype=np.complex64)
        sv_total += (time.time() - tsv)

        # Kernel for this batch vs all train
        # Optionally further split by rows (usually not needed because SV_b already small).
        tK = time.time()
        K_b = kernel_block_abs2(SV_b, SV_train, out_dtype=kernel_dtype)
        k_total += (time.time() - tK)

        y_pred[i0:i1] = model.predict(K_b).astype(np.int64)

        if can_score and scores_all is not None:
            s = model.decision_function(K_b)
            scores_all[i0:i1] = np.asarray(s).ravel()

    return y_pred, scores_all, float(sv_total), float(k_total)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "EMBER sparsity/score shift — Quantum kernel (QSVC via statevector kernel) experiments "
            "(reads q-splits; optimized to avoid RAM OOM via memmap + streaming)."
        )
    )

    ap.add_argument("--in-dir", type=str, default=str(DEFAULT_IN_DIR))
    ap.add_argument("--splits-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))

    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dims", type=int, nargs="+", default=DEFAULT_DIMS)

    ap.add_argument("--select-k", type=int, default=-1, help="If >0, use SelectKBest(chi2,k). Else disabled.")
    ap.add_argument("--no-selectk", action="store_true", help="Force disable SelectKBest even if --select-k>0.")
    ap.add_argument(
        "--validate-nonneg-chi2",
        action="store_true",
        help="If set, error if using SelectKBest(chi2) while train features contain negatives.",
    )

    ap.add_argument("--no-scaling", action="store_true")
    ap.add_argument("--angle-min", type=float, default=DEFAULT_ANGLE_MIN)
    ap.add_argument("--angle-max", type=float, default=DEFAULT_ANGLE_MAX)

    ap.add_argument("--svc-c", type=float, default=DEFAULT_SVC_C)
    ap.add_argument("--class-weight", type=str, default=DEFAULT_CLASS_WEIGHT)

    ap.add_argument("--thresh-source", type=str, default=DEFAULT_THRESH_SOURCE, choices=["train", "id_test"])
    ap.add_argument(
        "--thresh-criterion",
        type=str,
        default=DEFAULT_THRESH_CRITERION,
        choices=["balanced_accuracy", "f1_pos"],
    )
    ap.add_argument("--thresh-grid", type=int, default=DEFAULT_THRESH_GRID)
    ap.add_argument("--no-thresholding", action="store_true")

    ap.add_argument(
        "--enforce-qsplits",
        action="store_true",
        help="Validate split sizes match expected q-split sizes (from dir name or meta_q.json).",
    )

    ap.add_argument("--mmap", action="store_true", help="Load X.npy with mmap_mode='r' (recommended).")

    ap.add_argument(
        "--quantum-configs-json",
        type=str,
        default="",
        help="Optional path to JSON list of quantum configs. If empty, uses DEFAULT_QUANTUM_CONFIGS.",
    )

    # --- RAM/OOM hardening knobs ---
    ap.add_argument("--dry-run", action="store_true", help="Print estimated memory and exit (no training).")
    ap.add_argument("--max-train", type=int, default=5000, help="Abort if |train| exceeds this.")
    ap.add_argument(
        "--kernel-dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Kernel dtype. float32 saves RAM/disk and is usually fine.",
    )
    ap.add_argument("--max-kernel-mb", type=int, default=2000, help="Abort if estimated K_train > this MB.")
    ap.add_argument(
        "--kernel-block-rows",
        type=int,
        default=512,
        help="Rows per kernel block when building K_train (trade speed vs peak RAM).",
    )
    ap.add_argument(
        "--sv-batch",
        type=int,
        default=128,
        help="Batch size for computing statevectors on eval splits (streaming).",
    )

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed)
    dims = [int(d) for d in args.dims]

    select_k = int(args.select_k)
    if args.no_selectk or select_k <= 0:
        select_k_opt: Optional[int] = None
    else:
        select_k_opt = select_k

    use_scaling = not args.no_scaling
    angle_min = float(args.angle_min)
    angle_max = float(args.angle_max)

    class_weight = None if (args.class_weight.lower() in ["none", "null"]) else args.class_weight
    svc_c = float(args.svc_c)

    kernel_dtype = np.float32 if args.kernel_dtype == "float32" else np.float64

    qcfgs = DEFAULT_QUANTUM_CONFIGS
    if args.quantum_configs_json.strip():
        p = Path(args.quantum_configs_json)
        if not p.exists():
            raise FileNotFoundError(f"Missing quantum configs JSON: {p}")
        qcfgs = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(qcfgs, list) or not qcfgs:
            raise RuntimeError("quantum-configs-json must be a non-empty JSON list of configs")

    # Load data
    X_path = in_dir / "X.npy"
    y_path = in_dir / "y.npy"
    if not X_path.exists():
        raise FileNotFoundError(f"Missing X file: {X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing y file: {y_path}")

    X = np.load(X_path, mmap_mode="r" if args.mmap else None)
    y = np.load(y_path).astype(np.int64).ravel()

    n = int(X.shape[0])
    if y.shape[0] != n:
        raise RuntimeError(f"Shape mismatch: X rows={n}, y len={y.shape[0]}")

    uniq = set(np.unique(y).tolist())
    if not uniq.issubset({0, 1}):
        raise RuntimeError(f"y has unexpected labels: {sorted(list(uniq))}. Expected subset of {{0,1}}")

    # Load q-splits (+ optional low/high)
    splits: List[Split] = [
        Split("train", splits_dir / "train_idx.npy"),
        Split("id_test", splits_dir / "id_test_idx.npy"),
        Split("ood_test", splits_dir / "ood_test_idx.npy"),
    ]

    ood_low = splits_dir / "ood_low_idx.npy"
    ood_high = splits_dir / "ood_high_idx.npy"
    if ood_low.exists() and ood_high.exists():
        splits.extend([Split("ood_low", ood_low), Split("ood_high", ood_high)])

    idx: Dict[str, np.ndarray] = {s.name: load_indices(s.idx_path) for s in splits}

    # Range checks
    for name, arr in idx.items():
        if arr.min() < 0 or arr.max() >= n:
            raise RuntimeError(f"Split {name} out of range: min={arr.min()}, max={arr.max()}, n={n}")

    # Disjointness checks
    _assert_disjoint_fast(idx["train"], idx["id_test"], "train", "id_test")
    _assert_disjoint_fast(idx["train"], idx["ood_test"], "train", "ood_test")
    _assert_disjoint_fast(idx["id_test"], idx["ood_test"], "id_test", "ood_test")

    # Optional: enforce expected q sizes
    if args.enforce_qsplits:
        expected = _infer_expected_qsizes_from_splits_dir(splits_dir) or _load_expected_qsizes_from_meta(splits_dir)
        if expected is None:
            raise RuntimeError(
                "Could not infer expected q-split sizes from folder name or meta_q.json, but --enforce-qsplits was set."
            )
        for k in ["train", "id_test", "ood_test"]:
            if idx[k].size != int(expected[k]):
                raise RuntimeError(
                    f"Bad splits: {k} has {idx[k].size} but expected {expected[k]}. "
                    f"Are you pointing to master splits instead of q-splits? splits_dir={splits_dir}"
                )

    # Airbags: QSVC kernel is O(n_train^2)
    n_train = int(idx["train"].size)
    if n_train > int(args.max_train):
        raise RuntimeError(f"train size {n_train} > --max-train {args.max_train}. Wrong splits?")

    est_k_mb = estimate_kernel_mb(n_train, n_train, kernel_dtype)
    if est_k_mb > int(args.max_kernel_mb):
        raise RuntimeError(
            f"Estimated K_train size ~ {est_k_mb:.1f} MB > --max-kernel-mb {args.max_kernel_mb}. "
            "Reduce train size or use smaller q-splits."
        )

    if args.dry_run:
        print("[DRY RUN]")
        print(f"n_train={n_train}, kernel_dtype={args.kernel_dtype}, est_K_train={est_k_mb:.1f} MB")
        print(f"splits sizes: " + ", ".join([f"{k}={int(v.size)}" for k, v in idx.items()]))
        return

    # Optional: validate chi2 nonnegativity on TRAIN
    if select_k_opt is not None and args.validate_nonneg_chi2:
        Xtr = np.asarray(X[idx["train"]])
        mn = float(np.min(Xtr))
        if mn < 0.0:
            raise RuntimeError(
                f"SelectKBest(chi2) requires non-negative features. Found min(X_train)={mn}. "
                "Disable SelectKBest (--no-selectk) or fix features."
            )

    # Results holders
    all_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    # Note: we do NOT cache SV for non-train splits. That’s what used to explode RAM.
    for dim in dims:
        dim_key = f"dim_{dim}"
        all_results[dim_key] = {}

        print("=" * 80)
        print(
            f"[*] Fit embedding (dim={dim}) on TRAIN only | select_k={select_k_opt} | "
            f"scaling={use_scaling} | angle=({angle_min},{angle_max})"
        )

        embed = make_embedding_pipeline(
            dim=dim,
            select_k=select_k_opt,
            use_scaling=use_scaling,
            angle_min=angle_min,
            angle_max=angle_max,
            seed=seed,
        )

        t0 = time.time()
        embed.fit(np.asarray(X[idx["train"]]), y[idx["train"]])  # chi2 uses y
        embed_fit_s = time.time() - t0

        # Precompute embeddings for TRAIN (needed many times)
        X_train_ang = np.asarray(embed.transform(np.asarray(X[idx["train"]])), dtype=np.float64)
        y_train = y[idx["train"]]

        for qcfg in qcfgs:
            qid = qcfg["id"]
            cfg_key = f"{qid}__d{dim}"

            feature_map = build_feature_map(qcfg, feature_dim=dim)

            print("-" * 80)
            print(
                f"[*] Quantum config: {cfg_key} | map_type={qcfg['map_type']} reps={int(qcfg.get('reps', 1))} "
                f"ent={qcfg.get('entanglement', None)} | K_train≈{est_k_mb:.1f}MB ({args.kernel_dtype})"
            )

            # ---- TRAIN SV ----
            tsv = time.time()
            SV_train = compute_statevectors_batch(X_train_ang, feature_map, dtype=np.complex64)
            sv_train_s = time.time() - tsv

            # ---- Train kernel via memmap + fit ----
            # Put each cfg under its own folder so K_train.memmap doesn't collide.
            cfg_out_dir = out_dir / f"{dim_key}" / cfg_key
            cfg_out_dir.mkdir(parents=True, exist_ok=True)

            svm_q, kernel_train_s, svc_fit_s, K_path = fit_qsvc_with_memmap_kernel(
                SV_train=SV_train,
                y_train=y_train,
                svc_c=svc_c,
                class_weight=class_weight,
                out_dir=cfg_out_dir,
                kernel_dtype=kernel_dtype,
                kernel_block_rows=int(args.kernel_block_rows),
            )

            # Thresholding (train/id_test) — done streaming, no K stored.
            thr_value: Optional[float] = None
            kernel_src_s: Optional[float] = None
            sv_src_s: Optional[float] = None

            if not args.no_thresholding:
                src = args.thresh_source
                X_src_ang = np.asarray(embed.transform(np.asarray(X[idx[src]])), dtype=np.float64)
                y_src = y[idx[src]]

                tpred = time.time()
                y_pred_src, scores_src, sv_s, k_s = predict_scores_streaming(
                    model=svm_q,
                    SV_split_angles=X_src_ang,
                    feature_map=feature_map,
                    SV_train=SV_train,
                    kernel_dtype=kernel_dtype,
                    sv_batch=int(args.sv_batch),
                    kernel_out_block_rows=int(args.kernel_block_rows),
                )
                _ = y_pred_src  # not used for threshold selection
                sv_src_s = sv_s
                kernel_src_s = k_s
                _ = time.time() - tpred

                if scores_src is not None:
                    thr_value = best_threshold(
                        y_true=y_src,
                        scores=scores_src,
                        criterion=args.thresh_criterion,
                        grid_size=int(args.thresh_grid),
                    )

            all_results[dim_key][cfg_key] = {
                "config": {
                    "dim": dim,
                    "family": "quantum",
                    "model": "qsvc",
                    "quantum_id": qid,
                    "map_type": qcfg["map_type"],
                    "reps": int(qcfg.get("reps", 1)),
                    "entanglement": qcfg.get("entanglement", None),
                    "paulis": qcfg.get("paulis", None),
                    "seed": seed,
                    "select_k_features": select_k_opt,
                    "use_scaling": bool(use_scaling),
                    "angle_range": [angle_min, angle_max] if use_scaling else None,
                    "svc_C": svc_c,
                    "class_weight": class_weight,
                    "kernel": {
                        "kernel_dtype": str(np.dtype(kernel_dtype)),
                        "K_train_estimated_mb": float(est_k_mb),
                        "K_train_memmap_path": str(K_path),
                        "kernel_block_rows": int(args.kernel_block_rows),
                        "sv_batch": int(args.sv_batch),
                    },
                    "thresholding": None
                    if args.no_thresholding
                    else {
                        "thresh_source": args.thresh_source,
                        "thresh_criterion": args.thresh_criterion,
                        "thr_value": thr_value,
                        "thresh_grid": int(args.thresh_grid),
                    },
                    "timings": {
                        "embedding_fit_seconds": float(embed_fit_s),
                        "sv_train_seconds": float(sv_train_s),
                        "kernel_train_seconds": float(kernel_train_s),
                        "svc_fit_seconds": float(svc_fit_s),
                        "sv_src_seconds": None if sv_src_s is None else float(sv_src_s),
                        "kernel_src_seconds": None if kernel_src_s is None else float(kernel_src_s),
                    },
                    "sizes": {k: int(v.size) for k, v in idx.items()},
                },
                "splits": {},
                "thr_splits": {},
                "degradation": {},
                "thr_degradation": {},
            }

            # Evaluate splits streaming (no K stored)
            for split_name, split_idx in idx.items():
                X_s_ang = np.asarray(embed.transform(np.asarray(X[split_idx])), dtype=np.float64)
                y_s = y[split_idx]

                tpred = time.time()
                y_pred, scores, sv_s, k_s = predict_scores_streaming(
                    model=svm_q,
                    SV_split_angles=X_s_ang,
                    feature_map=feature_map,
                    SV_train=SV_train,
                    kernel_dtype=kernel_dtype,
                    sv_batch=int(args.sv_batch),
                    kernel_out_block_rows=int(args.kernel_block_rows),
                )
                _ = time.time() - tpred

                m = eval_split(y_s, y_pred, scores)
                m["sv_seconds"] = float(sv_s)
                m["kernel_seconds"] = float(k_s)
                all_results[dim_key][cfg_key]["splits"][split_name] = m

                summary_rows.append(
                    {
                        "family": "quantum",
                        "dim": dim,
                        "model": "qsvc",
                        "cfg": cfg_key,
                        "split": split_name,
                        "fit_seconds": float(embed_fit_s + sv_train_s + kernel_train_s + svc_fit_s),
                        "accuracy": m["accuracy"],
                        "balanced_accuracy": m["balanced_accuracy"],
                        "f1_macro": m["f1_macro"],
                        "f1_pos": m["f1_pos"],
                        "roc_auc": m["roc_auc"],
                        "pr_auc": m["pr_auc"],
                        "sv_seconds": float(sv_s),
                        "kernel_seconds": float(k_s),
                        "thr_value": thr_value,
                    }
                )

                if (not args.no_thresholding) and (thr_value is not None) and (scores is not None):
                    y_thr = predict_with_threshold(scores, float(thr_value))
                    tm = eval_split(y_s, y_thr, scores)
                    all_results[dim_key][cfg_key]["thr_splits"][split_name] = tm

                    summary_rows.append(
                        {
                            "family": "quantum",
                            "dim": dim,
                            "model": "qsvc",
                            "cfg": cfg_key,
                            "split": f"{split_name}__thr",
                            "fit_seconds": float(embed_fit_s + sv_train_s + kernel_train_s + svc_fit_s),
                            "accuracy": tm["accuracy"],
                            "balanced_accuracy": tm["balanced_accuracy"],
                            "f1_macro": tm["f1_macro"],
                            "f1_pos": tm["f1_pos"],
                            "roc_auc": tm["roc_auc"],
                            "pr_auc": tm["pr_auc"],
                            "sv_seconds": float(sv_s),
                            "kernel_seconds": float(k_s),
                            "thr_value": thr_value,
                        }
                    )

            # Degradation (default)
            if "id_test" in all_results[dim_key][cfg_key]["splits"] and "ood_test" in all_results[dim_key][cfg_key]["splits"]:
                idm = all_results[dim_key][cfg_key]["splits"]["id_test"]
                oodm = all_results[dim_key][cfg_key]["splits"]["ood_test"]
                all_results[dim_key][cfg_key]["degradation"] = {
                    "bal_acc_id_minus_ood": delta(idm["balanced_accuracy"], oodm["balanced_accuracy"]),
                    "f1_macro_id_minus_ood": delta(idm["f1_macro"], oodm["f1_macro"]),
                    "f1_pos_id_minus_ood": delta(idm["f1_pos"], oodm["f1_pos"]),
                    "roc_auc_id_minus_ood": delta(idm["roc_auc"], oodm["roc_auc"]),
                    "pr_auc_id_minus_ood": delta(idm["pr_auc"], oodm["pr_auc"]),
                }

            # Degradation (thresholded)
            if (not args.no_thresholding) and all_results[dim_key][cfg_key]["thr_splits"]:
                if "id_test" in all_results[dim_key][cfg_key]["thr_splits"] and "ood_test" in all_results[dim_key][cfg_key]["thr_splits"]:
                    tid = all_results[dim_key][cfg_key]["thr_splits"]["id_test"]
                    tood = all_results[dim_key][cfg_key]["thr_splits"]["ood_test"]
                    all_results[dim_key][cfg_key]["thr_degradation"] = {
                        "bal_acc_id_minus_ood": delta(tid["balanced_accuracy"], tood["balanced_accuracy"]),
                        "f1_macro_id_minus_ood": delta(tid["f1_macro"], tood["f1_macro"]),
                        "f1_pos_id_minus_ood": delta(tid["f1_pos"], tood["f1_pos"]),
                        "roc_auc_id_minus_ood": delta(tid["roc_auc"], tood["roc_auc"]),
                        "pr_auc_id_minus_ood": delta(tid["pr_auc"], tood["pr_auc"]),
                    }

            print("[OK] done")

    out_json = out_dir / "baseline_quantum_ember_sparsity_qsplits.json"
    out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"[✓] Wrote {out_json}")

    df = pd.DataFrame(summary_rows)
    out_csv = out_dir / "baseline_quantum_ember_sparsity_qsplits__summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] Wrote {out_csv}")

    # Console ranking: smallest drop in balanced accuracy (ID -> OOD)
    base = df[df["split"].isin(["id_test", "ood_test"])].copy()
    if not base.empty:
        id_df = base[base["split"] == "id_test"][["family", "dim", "cfg", "balanced_accuracy", "roc_auc", "pr_auc"]].rename(
            columns={
                "balanced_accuracy": "balanced_accuracy_id_test",
                "roc_auc": "roc_auc_id_test",
                "pr_auc": "pr_auc_id_test",
            }
        )
        ood_df = base[base["split"] == "ood_test"][["family", "dim", "cfg", "balanced_accuracy", "roc_auc", "pr_auc"]].rename(
            columns={
                "balanced_accuracy": "balanced_accuracy_ood_test",
                "roc_auc": "roc_auc_ood_test",
                "pr_auc": "pr_auc_ood_test",
            }
        )
        m = id_df.merge(ood_df, on=["family", "dim", "cfg"], how="inner")
        m["drop_bal_acc"] = m["balanced_accuracy_id_test"] - m["balanced_accuracy_ood_test"]
        m = m.sort_values("drop_bal_acc", ascending=True)

        print("\nRobustness ranking (default predict) ID->OOD (smallest drop):")
        print(m.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

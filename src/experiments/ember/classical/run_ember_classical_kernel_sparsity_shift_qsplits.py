# src/experiments/ember/classical/run_ember_classical_kernel_sparsity_shift_qsplits.py
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.svm import SVC


# ----------------------------
# Defaults
# ----------------------------
DEFAULT_IN_DIR = Path("data/processed/ember")
DEFAULT_OUT_DIR = Path("results/ember_shift/classical_kernel_sparsity")

DEFAULT_SEED = 42
DEFAULT_DIMS = [4, 6]

# EMBER usually does not need SelectKBest aggressively; keep optional.
DEFAULT_SELECT_K: Optional[int] = None  # e.g. 2000 if you want to try
DEFAULT_USE_SCALING = True
DEFAULT_ANGLE_MIN = 0.0
DEFAULT_ANGLE_MAX = float(np.pi)

DEFAULT_SVC_C = 1.0
DEFAULT_CLASS_WEIGHT = "balanced"

DEFAULT_KERNELS = ["linear", "rbf"]
DEFAULT_GAMMA = "scale"

DEFAULT_THRESH_SOURCE = "train"  # train or id_test
DEFAULT_THRESH_CRITERION = "balanced_accuracy"
DEFAULT_THRESH_GRID = 401

DEFAULT_KERNEL_NORMALIZE = True


# ----------------------------
# Split structure
# ----------------------------
@dataclass(frozen=True)
class Split:
    name: str
    idx_path: Path


# ----------------------------
# Utils
# ----------------------------
def load_indices(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    arr = np.load(path).astype(np.int64).ravel()
    if arr.size == 0:
        raise RuntimeError(f"Empty indices array: {path}")
    return arr


def _assert_disjoint(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    sa, sb = set(a.tolist()), set(b.tolist())
    inter = sa & sb
    if inter:
        raise RuntimeError(f"Overlap between {name_a} and {name_b}: {len(inter)} indices")


def _infer_expected_qsizes_from_splits_dir(splits_dir: Path) -> Optional[Dict[str, int]]:
    """
    Try to infer expected q-split sizes from folder name like:
      splits_sparsity_q1000_id500_ood500_seed42
    Returns dict or None if pattern not matched.
    """
    m = re.search(r"_q(\d+)_id(\d+)_ood(\d+)_seed(\d+)\b", splits_dir.name)
    if not m:
        return None
    q, idn, ood = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return {"train": q, "id_test": idn, "ood_test": ood}


def _load_expected_qsizes_from_meta(splits_dir: Path) -> Optional[Dict[str, int]]:
    """
    If meta_q.json exists, use it.
    """
    p = splits_dir / "meta_q.json"
    if not p.exists():
        return None
    meta = json.loads(p.read_text(encoding="utf-8"))
    sizes = meta.get("sizes", {})
    # tolerate both keys (n_id vs n_id_test) if present
    n_train = sizes.get("n_train", None)
    n_id = sizes.get("n_id", sizes.get("n_id_test", None))
    n_ood = sizes.get("n_ood", sizes.get("n_ood_test", None))
    if n_train is None or n_id is None or n_ood is None:
        return None
    return {"train": int(n_train), "id_test": int(n_id), "ood_test": int(n_ood)}


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


def diag_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


# ----------------------------
# Embedding pipeline (DrebinRed-aligned)
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

    # SelectKBest(chi2) requires non-negative features.
    if select_k is not None:
        steps.append(("selectk", SelectKBest(chi2, k=int(select_k))))

    # MaxAbs is safe and aligns with sparse pipelines
    steps.append(("scale_maxabs", MaxAbsScaler()))
    steps.append(("svd", TruncatedSVD(n_components=int(dim), random_state=seed)))

    if use_scaling:
        steps.append(("std", StandardScaler()))
        steps.append(("minmax", MinMaxScaler(feature_range=(angle_min, angle_max))))

    return Pipeline(steps)


# ----------------------------
# Kernel helpers (precomputed SVC)
# ----------------------------
def parse_gamma(gamma_str: str) -> Any:
    g = gamma_str.strip().lower()
    if g in ["scale", "auto"]:
        return g
    return float(g)


def gamma_scale_from_X(X: np.ndarray) -> float:
    var = float(np.var(X))
    d = int(X.shape[1])
    eps = 1e-12
    return 1.0 / (max(var, eps) * max(d, 1))


def gamma_auto_from_X(X: np.ndarray) -> float:
    return 1.0 / max(int(X.shape[1]), 1)


def kernel_linear(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A @ B.T).astype(np.float64, copy=False)


def kernel_rbf(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    dist2 = A2 + B2 - 2.0 * (A @ B.T)
    dist2 = np.maximum(dist2, 0.0)
    return np.exp(-gamma * dist2).astype(np.float64, copy=False)


def kernel_diag_linear(A: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", A, A).astype(np.float64, copy=False)


def kernel_diag_rbf(n: int) -> np.ndarray:
    return np.ones((n,), dtype=np.float64)


def normalize_kernel_train(K_train: np.ndarray, d_train: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    inv_sqrt = 1.0 / np.sqrt(np.maximum(d_train, eps))
    return K_train * inv_sqrt[:, None] * inv_sqrt[None, :]


def normalize_kernel_test(K_test: np.ndarray, d_test: np.ndarray, d_train: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    inv_sqrt_test = 1.0 / np.sqrt(np.maximum(d_test, eps))
    inv_sqrt_train = 1.0 / np.sqrt(np.maximum(d_train, eps))
    return K_test * inv_sqrt_test[:, None] * inv_sqrt_train[None, :]


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "EMBER sparsity/score shift — Classical kernels (SVC precomputed) experiments "
            "(reads q-splits; no internal subsampling). Fair kernel normalization ON by default."
        )
    )

    ap.add_argument("--in-dir", type=str, default=str(DEFAULT_IN_DIR))
    ap.add_argument("--splits-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))

    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dims", type=int, nargs="+", default=DEFAULT_DIMS)

    # SelectKBest
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

    ap.add_argument("--kernels", type=str, nargs="+", default=DEFAULT_KERNELS, choices=["linear", "rbf"])
    ap.add_argument("--gamma", type=str, default=str(DEFAULT_GAMMA))

    ap.add_argument("--thresh-source", type=str, default=DEFAULT_THRESH_SOURCE, choices=["train", "id_test"])
    ap.add_argument(
        "--thresh-criterion", type=str, default=DEFAULT_THRESH_CRITERION, choices=["balanced_accuracy", "f1_pos"]
    )
    ap.add_argument("--thresh-grid", type=int, default=DEFAULT_THRESH_GRID)
    ap.add_argument("--no-thresholding", action="store_true")

    ap.add_argument(
        "--no-kernel-normalize",
        action="store_true",
        help="Disable kernel normalization K'_{ij}=K_{ij}/sqrt(K_{ii}K_{jj}). Default is normalized (fair vs quantum).",
    )
    ap.add_argument(
        "--print-kernel-diag-stats",
        action="store_true",
        help="Print diag(K_train) stats before/after normalization for each cfg/dim.",
    )

    # Safety: ensure splits are q-splits, not master splits
    ap.add_argument(
        "--enforce-qsplits",
        action="store_true",
        help="If set, validate that split sizes match expected q-split sizes (from dir name or meta_q.json).",
    )

    # Memory: mmap X.npy
    ap.add_argument(
        "--mmap",
        action="store_true",
        help="Load X.npy with mmap_mode='r' (recommended if EMBER X is large).",
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

    class_weight = None if args.class_weight.lower() in ["none", "null"] else args.class_weight
    svc_c = float(args.svc_c)

    kernels = [k.lower() for k in args.kernels]
    gamma_cfg = parse_gamma(args.gamma)
    kernel_normalize = not args.no_kernel_normalize

    # Load data (EMBER: dense)
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

    # Disjointness checks (core splits)
    _assert_disjoint(idx["train"], idx["id_test"], "train", "id_test")
    _assert_disjoint(idx["train"], idx["ood_test"], "train", "ood_test")
    _assert_disjoint(idx["id_test"], idx["ood_test"], "id_test", "ood_test")

    # Optional: enforce q-split sizes (avoid accidentally pointing to master splits)
    if args.enforce_qsplits:
        expected = _infer_expected_qsizes_from_splits_dir(splits_dir)
        if expected is None:
            expected = _load_expected_qsizes_from_meta(splits_dir)
        if expected is None:
            raise RuntimeError(
                "Could not infer expected q-split sizes from folder name or meta_q.json, "
                "but --enforce-qsplits was set."
            )

        for k in ["train", "id_test", "ood_test"]:
            if idx[k].size != int(expected[k]):
                raise RuntimeError(
                    f"Bad splits: {k} has {idx[k].size} but expected {expected[k]}. "
                    f"Are you pointing to master splits instead of q-splits? splits_dir={splits_dir}"
                )

    # If using chi2, optionally validate non-negativity on TRAIN
    if select_k_opt is not None and args.validate_nonneg_chi2:
        Xtr = np.asarray(X[idx["train"]])
        mn = float(np.min(Xtr))
        if mn < 0.0:
            raise RuntimeError(
                f"SelectKBest(chi2) requires non-negative features. Found min(X_train)={mn}. "
                "Disable SelectKBest (--no-selectk) or fix features."
            )

    all_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []
    emb_cache: Dict[Tuple[int, str], np.ndarray] = {}

    for dim in dims:
        dim_key = f"dim_{dim}"
        all_results[dim_key] = {}

        print("=" * 80)
        print(
            f"[*] Fit embedding (dim={dim}) on TRAIN only | select_k={select_k_opt} | "
            f"scaling={use_scaling} | angle=({angle_min},{angle_max}) | kernel_normalize={kernel_normalize}"
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
        embed.fit(np.asarray(X[idx["train"]]), y[idx["train"]])
        embed_fit_s = time.time() - t0

        def get_emb(split_name: str) -> np.ndarray:
            k = (dim, split_name)
            if k in emb_cache:
                return emb_cache[k]
            Z = embed.transform(np.asarray(X[idx[split_name]]))
            Z = np.asarray(Z, dtype=np.float64)
            emb_cache[k] = Z
            return Z

        X_train = get_emb("train")
        y_train = y[idx["train"]]

        gamma_val: Optional[float] = None
        gamma_label = ""
        if "rbf" in kernels:
            if gamma_cfg == "scale":
                gamma_val = gamma_scale_from_X(X_train)
                gamma_label = "scale"
            elif gamma_cfg == "auto":
                gamma_val = gamma_auto_from_X(X_train)
                gamma_label = "auto"
            else:
                gamma_val = float(gamma_cfg)
                gamma_label = str(gamma_cfg)

        for kname in kernels:
            cfg_key = f"{kname}" if kname != "rbf" else f"rbf_g{gamma_label}"

            print("-" * 80)
            print(f"[*] Classical kernel config: {cfg_key} | dim={dim}")

            # Train kernel
            tK0 = time.time()
            if kname == "linear":
                K_train = kernel_linear(X_train, X_train)
            elif kname == "rbf":
                assert gamma_val is not None
                K_train = kernel_rbf(X_train, X_train, gamma=float(gamma_val))
            else:
                raise ValueError(f"Unknown kernel: {kname}")
            kernel_train_s = time.time() - tK0

            # Normalize train kernel
            tN0 = time.time()
            if kernel_normalize:
                d_train = np.diag(K_train).astype(np.float64, copy=False)
                if args.print_kernel_diag_stats:
                    st_pre = diag_stats(d_train)
                    print(
                        f"[diag pre] mean={st_pre['mean']:.6g} std={st_pre['std']:.6g} "
                        f"min={st_pre['min']:.6g} max={st_pre['max']:.6g}"
                    )
                K_train = normalize_kernel_train(K_train, d_train)
                if args.print_kernel_diag_stats:
                    st_post = diag_stats(np.diag(K_train))
                    print(
                        f"[diag post] mean={st_post['mean']:.6g} std={st_post['std']:.6g} "
                        f"min={st_post['min']:.6g} max={st_post['max']:.6g}"
                    )
            else:
                d_train = np.diag(K_train).astype(np.float64, copy=False)
            kernel_norm_train_s = time.time() - tN0

            # Fit SVC
            t1 = time.time()
            svc = SVC(kernel="precomputed", C=svc_c, class_weight=class_weight)
            svc.fit(K_train, y_train)
            svc_fit_s = time.time() - t1

            # Thresholding (paper-safe: choose threshold on train or id_test)
            thr_value: Optional[float] = None
            kernel_src_s: Optional[float] = None
            if not args.no_thresholding:
                src = args.thresh_source
                X_src = get_emb(src)
                y_src = y[idx[src]]

                tKs = time.time()
                if kname == "linear":
                    K_src = kernel_linear(X_src, X_train)
                else:
                    K_src = kernel_rbf(X_src, X_train, gamma=float(gamma_val))  # type: ignore[arg-type]
                kernel_src_s = time.time() - tKs

                if kernel_normalize:
                    if kname == "linear":
                        d_src = kernel_diag_linear(X_src)
                    else:
                        d_src = kernel_diag_rbf(X_src.shape[0])
                    K_src = normalize_kernel_test(K_src, d_src, d_train)

                src_scores = scores_from_precomputed_svc(svc, K_src)
                if src_scores is not None:
                    thr_value = best_threshold(
                        y_true=y_src,
                        scores=src_scores,
                        criterion=args.thresh_criterion,
                        grid_size=int(args.thresh_grid),
                    )

            all_results[dim_key][cfg_key] = {
                "config": {
                    "dim": dim,
                    "family": "classical",
                    "kernel": kname,
                    "gamma": None if kname != "rbf" else float(gamma_val),  # type: ignore[arg-type]
                    "gamma_mode": None if kname != "rbf" else gamma_label,
                    "gamma_computed_from": None if kname != "rbf" else "X_train_embedding",
                    "seed": seed,
                    "select_k_features": select_k_opt,
                    "use_scaling": bool(use_scaling),
                    "angle_range": [angle_min, angle_max] if use_scaling else None,
                    "kernel_normalize": bool(kernel_normalize),
                    "svc_C": svc_c,
                    "class_weight": class_weight,
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
                        "kernel_train_seconds": float(kernel_train_s),
                        "kernel_norm_train_seconds": float(kernel_norm_train_s),
                        "svc_fit_seconds": float(svc_fit_s),
                        "kernel_src_seconds": None if kernel_src_s is None else float(kernel_src_s),
                    },
                    "sizes": {k: int(v.size) for k, v in idx.items()},
                },
                "splits": {},
                "thr_splits": {},
                "degradation": {},
                "thr_degradation": {},
            }

            # Evaluate all splits
            for split_name in idx.keys():
                X_s = get_emb(split_name)
                y_s = y[idx[split_name]]

                tKs = time.time()
                if kname == "linear":
                    K_s = kernel_linear(X_s, X_train)
                    d_s = kernel_diag_linear(X_s)
                else:
                    K_s = kernel_rbf(X_s, X_train, gamma=float(gamma_val))  # type: ignore[arg-type]
                    d_s = kernel_diag_rbf(X_s.shape[0])

                if kernel_normalize:
                    K_s = normalize_kernel_test(K_s, d_s, d_train)

                kernel_s = time.time() - tKs

                y_pred = svc.predict(K_s)
                scores = scores_from_precomputed_svc(svc, K_s)

                m = eval_split(y_s, y_pred, scores)
                m["kernel_seconds"] = float(kernel_s)
                all_results[dim_key][cfg_key]["splits"][split_name] = m

                summary_rows.append(
                    {
                        "family": "classical",
                        "dim": dim,
                        "model": "svc",
                        "cfg": cfg_key,
                        "split": split_name,
                        "fit_seconds": float(embed_fit_s + kernel_train_s + kernel_norm_train_s + svc_fit_s),
                        "accuracy": m["accuracy"],
                        "balanced_accuracy": m["balanced_accuracy"],
                        "f1_macro": m["f1_macro"],
                        "f1_pos": m["f1_pos"],
                        "roc_auc": m["roc_auc"],
                        "pr_auc": m["pr_auc"],
                        "sv_seconds": None,  # align with quantum schema
                        "thr_value": thr_value,
                        "kernel_seconds": float(kernel_s),
                    }
                )

                # Thresholded evaluation (if enabled and we found thr)
                if (not args.no_thresholding) and (thr_value is not None) and (scores is not None):
                    y_thr = predict_with_threshold(scores, float(thr_value))
                    tm = eval_split(y_s, y_thr, scores)
                    all_results[dim_key][cfg_key]["thr_splits"][split_name] = tm

                    summary_rows.append(
                        {
                            "family": "classical",
                            "dim": dim,
                            "model": "svc",
                            "cfg": cfg_key,
                            "split": f"{split_name}__thr",
                            "fit_seconds": float(embed_fit_s + kernel_train_s + kernel_norm_train_s + svc_fit_s),
                            "accuracy": tm["accuracy"],
                            "balanced_accuracy": tm["balanced_accuracy"],
                            "f1_macro": tm["f1_macro"],
                            "f1_pos": tm["f1_pos"],
                            "roc_auc": tm["roc_auc"],
                            "pr_auc": tm["pr_auc"],
                            "sv_seconds": None,
                            "thr_value": thr_value,
                            "kernel_seconds": float(kernel_s),
                        }
                    )

            # Degradation ID -> OOD (no thr)
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

            # Degradation ID -> OOD (thr)
            if (not args.no_thresholding) and all_results[dim_key][cfg_key]["thr_splits"]:
                if (
                    "id_test" in all_results[dim_key][cfg_key]["thr_splits"]
                    and "ood_test" in all_results[dim_key][cfg_key]["thr_splits"]
                ):
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

    out_json = out_dir / "baseline_classical_kernel_ember_sparsity_qsplits.json"
    out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"[✓] Wrote {out_json}")

    df = pd.DataFrame(summary_rows)
    out_csv = out_dir / "baseline_classical_kernel_ember_sparsity_qsplits__summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] Wrote {out_csv}")

    # Console ranking: smallest drop in balanced accuracy (ID -> OOD)
    base = df[df["split"].isin(["id_test", "ood_test"])].copy()
    if not base.empty:
        id_df = base[base["split"] == "id_test"][["family", "dim", "cfg", "balanced_accuracy"]].rename(
            columns={"balanced_accuracy": "balanced_accuracy_id_test"}
        )
        ood_df = base[base["split"] == "ood_test"][["family", "dim", "cfg", "balanced_accuracy"]].rename(
            columns={"balanced_accuracy": "balanced_accuracy_ood_test"}
        )
        m = id_df.merge(ood_df, on=["family", "dim", "cfg"], how="inner")
        m["drop_bal_acc"] = m["balanced_accuracy_id_test"] - m["balanced_accuracy_ood_test"]
        m = m.sort_values("drop_bal_acc", ascending=True)
        print("\nRobustness ranking (smallest drop) ID->OOD:")
        print(m.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

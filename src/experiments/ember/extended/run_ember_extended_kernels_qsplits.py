# src/experiments/ember/extended/run_ember_extended_kernels_qsplits.py
"""
Extended kernel-swap experiments: broader classical kernel family and a
probabilistic classifier, under the same controlled protocol as the paper.

Additions relative to the original runners:
  - Classical kernels beyond linear/RBF: polynomial (deg 2, 3), Laplacian and
    Matern (nu = 3/2, 5/2), with train-only median-heuristic length scales.
  - A Gaussian Process classifier (binary Laplace approximation, Rasmussen &
    Williams 2006, Alg. 3.1/3.2) consuming the same precomputed kernels, which
    provides calibrated predictive probabilities and hence uncertainty
    quantification under shift (log-loss, Brier, ECE, predictive entropy on
    ID vs OOD).

Everything else is held fixed: embedding pipeline, splits, metrics and
family-internal selection logic are identical to the original experiments.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.svm import SVC

from src.experiments.ember.classical.run_ember_classical_kernel_sparsity_shift_qsplits import (
    gamma_scale_from_X,
    kernel_linear,
    kernel_rbf,
)
from src.experiments.ember.quantum.run_ember_quantum_kernel_sparsity_shift_qsplits import (
    DEFAULT_QUANTUM_CONFIGS,
    build_feature_map,
    compute_statevectors_batch,
    eval_split,
    kernel_block_abs2,
    load_indices,
    make_embedding_pipeline,
)

DEFAULT_IN_DIR = Path("data/processed/ember")
DEFAULT_OUT_DIR = Path("results/ember_shift/extended_kernels")
DEFAULT_DIMS = [4, 6, 8, 10, 12]
DEFAULT_SVC_C = 1.0

CLASSICAL_KERNELS = [
    "linear",
    "rbf_gscale",
    "poly2",
    "poly3",
    "laplacian_med",
    "matern15_med",
    "matern25_med",
]


# ----------------------------
# Kernel constructions (train-only hyperparameters, unit diagonal)
# ----------------------------
def _cosine_normalize(K: np.ndarray, diag_a: np.ndarray, diag_b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    inv_a = 1.0 / np.sqrt(np.maximum(diag_a, eps))
    inv_b = 1.0 / np.sqrt(np.maximum(diag_b, eps))
    return K * inv_a[:, None] * inv_b[None, :]


def _poly_raw(A: np.ndarray, B: np.ndarray, gamma: float, coef0: float, degree: int) -> np.ndarray:
    return (gamma * (A @ B.T) + coef0) ** degree


def _median_heuristic(X_tr: np.ndarray, metric: str, max_pairs: int = 2000, seed: int = 0) -> float:
    """Median pairwise distance on (a subsample of) the training set."""
    rng = np.random.default_rng(seed)
    n = X_tr.shape[0]
    sub = X_tr if n <= max_pairs else X_tr[rng.choice(n, max_pairs, replace=False)]
    D = cdist(sub, sub, metric=metric)
    vals = D[np.triu_indices_from(D, k=1)]
    med = float(np.median(vals))
    return max(med, 1e-12)


def _matern(D: np.ndarray, ls: float, nu: str) -> np.ndarray:
    r = D / ls
    if nu == "1.5":
        s = np.sqrt(3.0) * r
        return (1.0 + s) * np.exp(-s)
    if nu == "2.5":
        s = np.sqrt(5.0) * r
        return (1.0 + s + s**2 / 3.0) * np.exp(-s)
    raise ValueError(f"Unsupported Matern nu: {nu}")


class ClassicalKernelFactory:
    """Builds unit-diagonal kernel blocks; all hyperparameters fit on train only."""

    def __init__(self, X_tr: np.ndarray, seed: int):
        self.X_tr = X_tr
        d = X_tr.shape[1]
        self.hp: Dict[str, float] = {
            "rbf_gamma": gamma_scale_from_X(X_tr),
            "poly_gamma": 1.0 / max(d, 1),
            "poly_coef0": 1.0,
            "l1_median": _median_heuristic(X_tr, "cityblock", seed=seed),
            "l2_median": _median_heuristic(X_tr, "euclidean", seed=seed),
        }

    def block(self, name: str, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        hp = self.hp
        if name.startswith("rbf_gscale_x"):
            # Bandwidth-sweep variant: gamma = scale-gamma * factor.
            factor = float(name.rsplit("x", 1)[-1])
            return kernel_rbf(A, B, hp["rbf_gamma"] * factor)
        if name == "linear":
            K = kernel_linear(A, B)
            da = np.einsum("ij,ij->i", A, A)
            db = np.einsum("ij,ij->i", B, B)
            return _cosine_normalize(K, da, db)
        if name == "rbf_gscale":
            return kernel_rbf(A, B, hp["rbf_gamma"])
        if name in ("poly2", "poly3"):
            deg = 2 if name == "poly2" else 3
            K = _poly_raw(A, B, hp["poly_gamma"], hp["poly_coef0"], deg)
            da = (hp["poly_gamma"] * np.einsum("ij,ij->i", A, A) + hp["poly_coef0"]) ** deg
            db = (hp["poly_gamma"] * np.einsum("ij,ij->i", B, B) + hp["poly_coef0"]) ** deg
            return _cosine_normalize(K, da, db)
        if name.startswith("laplacian_med_x"):
            # Length-scale sweep variant: l1 = median * factor.
            factor = float(name.rsplit("x", 1)[-1])
            return np.exp(-cdist(A, B, "cityblock") / (hp["l1_median"] * factor))
        if name.startswith("matern15_med_x"):
            factor = float(name.rsplit("x", 1)[-1])
            return _matern(cdist(A, B, "euclidean"), hp["l2_median"] * factor, "1.5")
        if name.startswith("matern25_med_x"):
            factor = float(name.rsplit("x", 1)[-1])
            return _matern(cdist(A, B, "euclidean"), hp["l2_median"] * factor, "2.5")
        if name == "laplacian_med":
            return np.exp(-cdist(A, B, "cityblock") / hp["l1_median"])
        if name == "matern15_med":
            return _matern(cdist(A, B, "euclidean"), hp["l2_median"], "1.5")
        if name == "matern25_med":
            return _matern(cdist(A, B, "euclidean"), hp["l2_median"], "2.5")
        raise ValueError(f"Unknown classical kernel: {name}")


# ----------------------------
# GP classifier (binary Laplace approximation, precomputed kernel)
# Rasmussen & Williams (2006), Algorithms 3.1 and 3.2, logistic link.
# ----------------------------
class LaplaceGPC:
    def __init__(self, jitter: float = 1e-8, max_iter: int = 100, tol: float = 1e-6):
        self.jitter = jitter
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.tanh(0.5 * z))

    def fit(self, K_tr: np.ndarray, y01: np.ndarray) -> "LaplaceGPC":
        t = np.asarray(y01, dtype=np.float64).ravel()  # targets in {0,1}
        n = t.shape[0]
        K = np.asarray(K_tr, dtype=np.float64) + self.jitter * np.eye(n)

        f = np.zeros(n)
        obj_prev = -np.inf
        for _ in range(self.max_iter):
            pi = self._sigmoid(f)
            W = np.maximum(pi * (1.0 - pi), 1e-12)
            sW = np.sqrt(W)
            B = np.eye(n) + (sW[:, None] * K) * sW[None, :]
            L = np.linalg.cholesky(B)
            b = W * f + (t - pi)
            v = np.linalg.solve(L, sW * (K @ b))
            a = b - sW * np.linalg.solve(L.T, v)
            f = K @ a
            # Laplace objective (up to constants): -0.5 aᵀf + log p(t|f)
            obj = -0.5 * float(a @ f) + float(np.sum(t * f - np.logaddexp(0.0, f)))
            if abs(obj - obj_prev) < self.tol:
                break
            obj_prev = obj

        pi = self._sigmoid(f)
        self._sW = np.sqrt(np.maximum(pi * (1.0 - pi), 1e-12))
        self._L = np.linalg.cholesky(
            np.eye(n) + (self._sW[:, None] * K) * self._sW[None, :]
        )
        self._grad = t - pi  # d log p(t|f) / df at the mode
        return self

    def predict_proba(self, K_star_tr: np.ndarray, k_star_diag: np.ndarray) -> np.ndarray:
        """K_star_tr: (m, n_train) cross-kernel; k_star_diag: (m,) prior variances."""
        f_mean = K_star_tr @ self._grad
        v = np.linalg.solve(self._L, self._sW[:, None] * K_star_tr.T)
        f_var = np.maximum(k_star_diag - np.einsum("ij,ij->j", v, v), 1e-12)
        # Logistic-probit approximation for the predictive integral.
        kappa = 1.0 / np.sqrt(1.0 + np.pi * f_var / 8.0)
        return self._sigmoid(kappa * f_mean)


# ----------------------------
# Probabilistic metrics
# ----------------------------
def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        conf = float(p[m].mean())
        acc = float(y_true[m].mean())
        ece += (m.sum() / p.size) * abs(acc - conf)
    return float(ece)


def probabilistic_metrics(y_true: np.ndarray, p_pos: np.ndarray) -> Dict[str, float]:
    p = np.clip(p_pos, 1e-12, 1.0 - 1e-12)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return {
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
        "ece": expected_calibration_error(y_true, p),
        "mean_predictive_entropy": float(ent.mean()),
    }


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Extended kernels + probabilistic classifier on EMBER q-splits.")
    ap.add_argument("--in-dir", type=str, default=str(DEFAULT_IN_DIR))
    ap.add_argument("--splits-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dims", type=int, nargs="+", default=DEFAULT_DIMS)
    ap.add_argument("--models", type=str, nargs="+", default=["svc", "gpc"], choices=["svc", "gpc"])
    ap.add_argument("--classical-kernels", type=str, nargs="+", default=CLASSICAL_KERNELS)
    ap.add_argument("--include-quantum", action="store_true",
                    help="Also evaluate the four paper quantum kernels under the same models.")
    ap.add_argument("--rbf-gamma-factors", type=float, nargs="+", default=[],
                    help="Extra RBF variants with gamma = scale-gamma * factor (classical bandwidth sweep).")
    ap.add_argument("--quantum-angle-scales", type=float, nargs="+", default=[1.0],
                    help="Angle-scaling factors for the quantum feature maps (quantum bandwidth sweep).")
    ap.add_argument("--svc-c", type=float, default=DEFAULT_SVC_C)
    ap.add_argument("--mmap", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(in_dir / "X.npy", mmap_mode="r" if args.mmap else None)
    y = np.load(in_dir / "y.npy").astype(np.int64).ravel()

    idx = {
        "train": load_indices(splits_dir / "train_idx.npy"),
        "id_test": load_indices(splits_dir / "id_test_idx.npy"),
        "ood_test": load_indices(splits_dir / "ood_test_idx.npy"),
    }
    y_by_split = {k: y[v] for k, v in idx.items()}

    all_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for dim in [int(d) for d in args.dims]:
        dim_key = f"dim_{dim}"
        all_results[dim_key] = {}
        t_dim = time.time()

        embed = make_embedding_pipeline(
            dim=dim, select_k=None, use_scaling=True,
            angle_min=0.0, angle_max=float(np.pi), seed=int(args.seed),
        )
        embed.fit(np.asarray(X[idx["train"]]), y_by_split["train"])
        X_emb = {k: np.asarray(embed.transform(np.asarray(X[v])), dtype=np.float64) for k, v in idx.items()}

        # Assemble kernel blocks per kernel name: train, and cross blocks per eval split.
        kernel_sets: Dict[str, Dict[str, Any]] = {}

        factory = ClassicalKernelFactory(X_emb["train"], seed=int(args.seed))
        classical_names = list(args.classical_kernels) + [
            f"rbf_gscale_x{f:g}" for f in args.rbf_gamma_factors
        ]
        for kname in classical_names:
            blocks = {
                "family": "classical_ext",
                "train": factory.block(kname, X_emb["train"], X_emb["train"]),
            }
            for split in ["id_test", "ood_test"]:
                blocks[split] = factory.block(kname, X_emb[split], X_emb["train"])
            kernel_sets[kname] = blocks

        if args.include_quantum:
            for scale in [float(s) for s in args.quantum_angle_scales]:
                X_q = X_emb if scale == 1.0 else {k: v * scale for k, v in X_emb.items()}
                suffix = "" if scale == 1.0 else f"__as{scale:g}"
                for qcfg in DEFAULT_QUANTUM_CONFIGS:
                    fmap = build_feature_map(qcfg, feature_dim=dim)
                    sv = {k: compute_statevectors_batch(X_q[k], fmap, dtype=np.complex64) for k in X_q}
                    blocks = {
                        "family": "quantum",
                        "train": kernel_block_abs2(sv["train"], sv["train"], out_dtype=np.float64),
                    }
                    for split in ["id_test", "ood_test"]:
                        blocks[split] = kernel_block_abs2(sv[split], sv["train"], out_dtype=np.float64)
                    kernel_sets[qcfg["id"] + suffix] = blocks

        for kname, blocks in kernel_sets.items():
            K_tr = blocks["train"]
            for model_name in args.models:
                cfg_key = f"{kname}__{model_name}__d{dim}"
                t0 = time.time()

                if model_name == "svc":
                    model = SVC(kernel="precomputed", C=float(args.svc_c), class_weight="balanced")
                    model.fit(K_tr, y_by_split["train"])
                else:
                    model = LaplaceGPC().fit(K_tr, y_by_split["train"])

                fit_seconds = time.time() - t0
                res: Dict[str, Any] = {
                    "config": {
                        "dim": dim, "kernel": kname, "family": blocks["family"],
                        "model": model_name, "seed": int(args.seed),
                        "hyperparams": factory.hp if blocks["family"] == "classical_ext" else None,
                        "fit_seconds": float(fit_seconds),
                    },
                    "splits": {},
                    "degradation": {},
                }

                for split in ["id_test", "ood_test"]:
                    K_s = blocks[split]
                    y_s = y_by_split[split]
                    if model_name == "svc":
                        y_pred = model.predict(K_s).astype(np.int64)
                        scores = np.asarray(model.decision_function(K_s)).ravel()
                        m = eval_split(y_s, y_pred, scores)
                    else:
                        # All kernels here have unit diagonal by construction.
                        p_pos = model.predict_proba(K_s, np.ones(K_s.shape[0]))
                        y_pred = (p_pos >= 0.5).astype(np.int64)
                        m = eval_split(y_s, y_pred, p_pos)
                        m.update(probabilistic_metrics(y_s, p_pos))
                    res["splits"][split] = m

                    row = {
                        "family": blocks["family"], "model": model_name, "dim": dim,
                        "cfg": cfg_key, "kernel": kname, "split": split,
                        "fit_seconds": float(fit_seconds),
                        "accuracy": m["accuracy"],
                        "balanced_accuracy": m["balanced_accuracy"],
                        "f1_macro": m["f1_macro"], "f1_pos": m["f1_pos"],
                        "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                    }
                    for pm in ["log_loss", "brier", "ece", "mean_predictive_entropy"]:
                        row[pm] = m.get(pm)
                    summary_rows.append(row)

                idm, oodm = res["splits"]["id_test"], res["splits"]["ood_test"]
                res["degradation"] = {
                    "bal_acc_id_minus_ood": float(idm["balanced_accuracy"] - oodm["balanced_accuracy"]),
                }
                if model_name == "gpc":
                    res["degradation"]["ece_ood_minus_id"] = float(oodm["ece"] - idm["ece"])
                    res["degradation"]["entropy_ood_minus_id"] = float(
                        oodm["mean_predictive_entropy"] - idm["mean_predictive_entropy"]
                    )
                all_results[dim_key][cfg_key] = res

        print(f"[OK] dim={dim} done in {time.time() - t_dim:.1f}s "
              f"({len(kernel_sets)} kernels x {len(args.models)} models)")

    out_json = out_dir / "extended_kernels_qsplits.json"
    out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    df = pd.DataFrame(summary_rows)
    out_csv = out_dir / "extended_kernels_qsplits__summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_json}")
    print(f"[OK] Wrote {out_csv}")


if __name__ == "__main__":
    main()

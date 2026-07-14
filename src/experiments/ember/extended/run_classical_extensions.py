# src/experiments/ember/extended/run_classical_extensions.py
"""
Phase-B/C extension runner: adds, on top of an existing run directory,

  lsweep   : Laplacian/Matern length-scale variants (l = factor * median) on
             the shared angular representation -> summary_classical_lsweep.csv
  ablation : all base classical kernels on a NON-angular representation
             (MaxAbs -> SVD -> Standard, no [0, pi] mapping), with train-Gram
             geometry descriptors -> summary_classical_plainrep.csv +
             geometry_plainrep.csv
  csens    : SVC regularization sweep (C grid) for the base classical kernels
             on the angular representation; optionally also for the quantum
             feature maps (--include-quantum, Phase C)
             -> summary_csens.csv

Reuses the embedding, kernel factory, models, and metrics of
run_ember_extended_kernels_qsplits so every number is comparable with the
frozen runs. Deterministic given (splits-dir, seed, dims).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import SVC

from src.experiments.ember.extended.run_ember_extended_kernels_qsplits import (
    ClassicalKernelFactory,
    LaplaceGPC,
    CLASSICAL_KERNELS,
    probabilistic_metrics,
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

LS_FACTORS = [0.1, 0.3, 3.0, 10.0]
C_GRID = [0.01, 0.1, 10.0, 100.0]

OUT_FILES = {"lsweep": "summary_classical_lsweep.csv",
             "ablation": "summary_classical_plainrep.csv",
             "csens": "summary_csens.csv",
             "ktanull": "mechanism_crossfit.csv",
             "lsweep_geo": "geometry_lsweep.csv"}
N_PERM = 100


def make_plain_pipeline(dim: int, seed: int) -> Pipeline:
    """MaxAbs -> SVD -> Standard, without the [0, pi] angle mapping."""
    return Pipeline([
        ("scale_maxabs", MaxAbsScaler()),
        ("svd", TruncatedSVD(n_components=int(dim), random_state=seed)),
        ("std", StandardScaler()),
    ])


def centered_kta(K: np.ndarray, y: np.ndarray) -> float:
    yy = np.where(np.asarray(y) > 0, 1.0, -1.0)
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    num = float(yy @ Kc @ yy)
    den = float(n * np.linalg.norm(Kc, "fro"))
    return num / max(den, 1e-12)


def eff_rank(K: np.ndarray) -> float:
    ev = np.linalg.eigvalsh((K + K.T) / 2.0)
    ev = np.clip(ev, 0.0, None)
    s = ev.sum()
    if s <= 0:
        return 1.0
    p = ev / s
    p = p[p > 1e-15]
    return float(np.exp(-(p * np.log(p)).sum()))


def kta_terms(K: np.ndarray) -> tuple[np.ndarray, float]:
    """Doubly centered kernel and its Frobenius norm (label-independent)."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    return Kc, float(np.linalg.norm(Kc, "fro"))


def kta_with(Kc: np.ndarray, fro: float, y: np.ndarray) -> float:
    yy = np.where(np.asarray(y) > 0, 1.0, -1.0)
    return float(yy @ Kc @ yy) / max(len(yy) * fro, 1e-12)


def geometry_row(kname: str, family: str, dim: int, K_tr: np.ndarray,
                 K_id: np.ndarray, K_ood: np.ndarray,
                 y_by_split: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Mechanism descriptors on the true square Gram of each split."""
    kta_id = centered_kta(K_id, y_by_split["id_test"])
    kta_ood = centered_kta(K_ood, y_by_split["ood_test"])
    return {
        "family": family, "kernel": kname, "dim": dim,
        "spec_train_eff_rank": eff_rank(K_tr),
        "kta_train": centered_kta(K_tr, y_by_split["train"]),
        "kta_id": kta_id,
        "kta_ood": kta_ood,
        "kta_survival": kta_ood - kta_id,
    }


def ktanull_rows(kname: str, family: str, dim: int, K_oo: np.ndarray,
                 blocks: Dict[str, np.ndarray], y_by_split: Dict[str, np.ndarray],
                 K_tr: np.ndarray, seed: int) -> Dict[str, Any]:
    """Permutation null for OOD KTA + cross-fitted mechanism cell.

    KTA is measured on one half of the OOD split (A); balanced accuracy of
    models trained on the training split is measured on the other half (B),
    so the mechanism correlation never reuses the same labels.
    """
    rng = np.random.default_rng(seed + dim)
    y_ood = y_by_split["ood_test"]
    n = len(y_ood)
    # full-OOD KTA + permutation null (Kc fixed; only labels permute)
    Kc, fro = kta_terms(K_oo)
    kta_full = kta_with(Kc, fro, y_ood)
    null = np.array([kta_with(Kc, fro, rng.permutation(y_ood)) for _ in range(N_PERM)])
    # cross-fit halves
    perm = rng.permutation(n)
    a_idx, b_idx = perm[: n // 2], perm[n // 2:]
    Kc_a, fro_a = kta_terms(K_oo[np.ix_(a_idx, a_idx)])
    kta_a = kta_with(Kc_a, fro_a, y_ood[a_idx])
    row: Dict[str, Any] = {
        "family": family, "kernel": kname, "dim": dim,
        "kta_ood_full": kta_full,
        "kta_null_mean": float(null.mean()), "kta_null_std": float(null.std()),
        "kta_null_exceed_frac": float((np.abs(null) >= abs(kta_full)).mean()),
        "kta_ood_halfA": kta_a,
        "eff_rank_train": eff_rank(K_tr),
    }
    from sklearn.metrics import balanced_accuracy_score
    for model_name in ("svc", "gpc"):
        if model_name == "svc":
            model = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
            model.fit(K_tr, y_by_split["train"])
            y_pred = model.predict(blocks["ood_test"][b_idx]).astype(np.int64)
        else:
            model = LaplaceGPC().fit(K_tr, y_by_split["train"])
            p = model.predict_proba(blocks["ood_test"][b_idx], np.ones(len(b_idx)))
            y_pred = (p >= 0.5).astype(np.int64)
        row[f"bacc_{model_name}_halfB"] = float(
            balanced_accuracy_score(y_ood[b_idx], y_pred))
    return row


def eval_model_rows(kname: str, family: str, model_name: str, dim: int,
                    blocks: Dict[str, np.ndarray], y_by_split: Dict[str, np.ndarray],
                    svc_c: float) -> List[Dict[str, Any]]:
    K_tr = blocks["train"]
    t0 = time.time()
    if model_name == "svc":
        model = SVC(kernel="precomputed", C=svc_c, class_weight="balanced")
        model.fit(K_tr, y_by_split["train"])
    else:
        model = LaplaceGPC().fit(K_tr, y_by_split["train"])
    fit_seconds = time.time() - t0
    c_tag = "" if (model_name != "svc" or svc_c == 1.0) else f"_C{svc_c:g}"
    cfg_key = f"{kname}__{model_name}{c_tag}__d{dim}"
    rows = []
    for split in ("id_test", "ood_test"):
        K_s, y_s = blocks[split], y_by_split[split]
        if model_name == "svc":
            y_pred = model.predict(K_s).astype(np.int64)
            scores = np.asarray(model.decision_function(K_s)).ravel()
            m = eval_split(y_s, y_pred, scores)
        else:
            p_pos = model.predict_proba(K_s, np.ones(K_s.shape[0]))
            y_pred = (p_pos >= 0.5).astype(np.int64)
            m = eval_split(y_s, y_pred, p_pos)
            m.update(probabilistic_metrics(y_s, p_pos))
        row = {"family": family, "model": model_name, "dim": dim, "cfg": cfg_key,
               "kernel": kname, "split": split, "svc_c": svc_c if model_name == "svc" else np.nan,
               "fit_seconds": float(fit_seconds),
               "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
               "f1_macro": m["f1_macro"], "f1_pos": m["f1_pos"],
               "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"]}
        for pm in ("log_loss", "brier", "ece", "mean_predictive_entropy"):
            row[pm] = m.get(pm)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    ap.add_argument("--mode", choices=list(OUT_FILES), required=True)
    ap.add_argument("--include-quantum", action="store_true",
                    help="csens only: also sweep C for the quantum feature maps.")
    args = ap.parse_args()

    X = np.load(args.in_dir / "X.npy", mmap_mode="r")
    y = np.load(args.in_dir / "y.npy").astype(np.int64).ravel()
    idx = {k: load_indices(args.splits_dir / f"{k}_idx.npy")
           for k in ("train", "id_test", "ood_test")}
    y_by_split = {k: y[v] for k, v in idx.items()}

    summary_rows: List[Dict[str, Any]] = []
    geo_rows: List[Dict[str, Any]] = []

    for dim in args.dims:
        t_dim = time.time()
        if args.mode == "ablation":
            embed = make_plain_pipeline(dim, args.seed)
        else:
            embed = make_embedding_pipeline(dim=dim, select_k=None, use_scaling=True,
                                            angle_min=0.0, angle_max=float(np.pi),
                                            seed=args.seed)
        embed.fit(np.asarray(X[idx["train"]]), y_by_split["train"])
        X_emb = {k: np.asarray(embed.transform(np.asarray(X[v])), dtype=np.float64)
                 for k, v in idx.items()}
        factory = ClassicalKernelFactory(X_emb["train"], seed=args.seed)

        if args.mode == "lsweep":
            kernels = [f"{base}_x{f:g}" for base in
                       ("laplacian_med", "matern15_med", "matern25_med") for f in LS_FACTORS]
        elif args.mode == "lsweep_geo":
            kernels = list(CLASSICAL_KERNELS) + [
                f"{base}_x{f:g}" for base in
                ("laplacian_med", "matern15_med", "matern25_med") for f in LS_FACTORS]
        else:
            kernels = list(CLASSICAL_KERNELS)

        family = "classical_ext_plain" if args.mode == "ablation" else "classical_ext"
        for kname in kernels:
            blocks = {"train": factory.block(kname, X_emb["train"], X_emb["train"])}
            for split in ("id_test", "ood_test"):
                blocks[split] = factory.block(kname, X_emb[split], X_emb["train"])
            if args.mode == "lsweep_geo":
                summary_rows.append(geometry_row(
                    kname, family, dim, blocks["train"],
                    factory.block(kname, X_emb["id_test"], X_emb["id_test"]),
                    factory.block(kname, X_emb["ood_test"], X_emb["ood_test"]),
                    y_by_split))
                continue
            if args.mode == "ktanull":
                K_oo = factory.block(kname, X_emb["ood_test"], X_emb["ood_test"])
                summary_rows.append(ktanull_rows(kname, family, dim, K_oo, blocks,
                                                 y_by_split, blocks["train"], args.seed))
                continue
            if args.mode == "csens":
                for c in C_GRID:
                    summary_rows += eval_model_rows(kname, family, "svc", dim, blocks, y_by_split, c)
            else:
                for model_name in ("svc", "gpc"):
                    summary_rows += eval_model_rows(kname, family, model_name, dim,
                                                    blocks, y_by_split, 1.0)
            if args.mode == "ablation":
                geo_rows.append({
                    "dim": dim, "kernel": kname, "family": family,
                    "representation": "plain",
                    "spec_train_eff_rank": eff_rank(blocks["train"]),
                    "kta_train": centered_kta(blocks["train"], y_by_split["train"]),
                })

        if args.mode in ("csens", "ktanull", "lsweep_geo") and args.include_quantum:
            for qcfg in DEFAULT_QUANTUM_CONFIGS:
                fmap = build_feature_map(qcfg, feature_dim=dim)
                sv = {k: compute_statevectors_batch(X_emb[k], fmap, dtype=np.complex64)
                      for k in X_emb}
                blocks = {"train": kernel_block_abs2(sv["train"], sv["train"], out_dtype=np.float64)}
                for split in ("id_test", "ood_test"):
                    blocks[split] = kernel_block_abs2(sv[split], sv["train"], out_dtype=np.float64)
                if args.mode == "lsweep_geo":
                    summary_rows.append(geometry_row(
                        qcfg["id"], "quantum", dim, blocks["train"],
                        kernel_block_abs2(sv["id_test"], sv["id_test"], out_dtype=np.float64),
                        kernel_block_abs2(sv["ood_test"], sv["ood_test"], out_dtype=np.float64),
                        y_by_split))
                    continue
                if args.mode == "ktanull":
                    K_oo = kernel_block_abs2(sv["ood_test"], sv["ood_test"], out_dtype=np.float64)
                    summary_rows.append(ktanull_rows(qcfg["id"], "quantum", dim, K_oo,
                                                     blocks, y_by_split, blocks["train"],
                                                     args.seed))
                    continue
                for c in C_GRID:
                    summary_rows += eval_model_rows(qcfg["id"], "quantum", "svc", dim,
                                                    blocks, y_by_split, c)
        print(f"[OK] dim={dim} done in {time.time() - t_dim:.1f}s ({args.mode})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.out_dir / OUT_FILES[args.mode], index=False)
    if geo_rows:
        pd.DataFrame(geo_rows).to_csv(args.out_dir / "geometry_plainrep.csv", index=False)
    print(f"[OK] Wrote {args.out_dir / OUT_FILES[args.mode]}")


if __name__ == "__main__":
    main()

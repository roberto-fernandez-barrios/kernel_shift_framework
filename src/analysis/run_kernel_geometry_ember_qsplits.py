# src/analysis/run_kernel_geometry_ember_qsplits.py
"""
Mechanistic kernel-geometry analysis under the controlled kernel-swap protocol.

For a given q-split (train / id_test / ood_test) and each embedding dimension,
this script builds the exact same classical and quantum kernels used by the
experiment runners (same embedding pipeline, same feature maps, same gamma
convention) and computes geometry descriptors intended to *explain* the
observed ID/OOD behavior rather than re-measure accuracy:

  - spectral summary of K_train (top-eigenvalue shares, effective rank),
  - kernel concentration (off-diagonal statistics) on train / ID / OOD blocks,
  - centered kernel-target alignment (KTA) on train / ID / OOD,
  - OOD-to-train vs ID-to-train similarity ratio per kernel,
  - geometric difference g(K_C -> K_Q) between each classical and each quantum
    train kernel (Huang et al., 2021), over a regularization sweep.

The kernel construction is imported from the experiment runners so that the
analysis is guaranteed to describe the very kernels the paper evaluates.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.experiments.ember.extended.run_ember_extended_kernels_qsplits import (
    CLASSICAL_KERNELS as EXTENDED_CLASSICAL_KERNELS,
    ClassicalKernelFactory,
)
from src.experiments.ember.quantum.run_ember_quantum_kernel_sparsity_shift_qsplits import (
    DEFAULT_QUANTUM_CONFIGS,
    build_feature_map,
    compute_statevectors_batch,
    kernel_block_abs2,
    load_indices,
    make_embedding_pipeline,
)

DEFAULT_IN_DIR = Path("data/processed/ember")
DEFAULT_OUT_DIR = Path("results/kernel_geometry")
DEFAULT_DIMS = [4, 6, 8, 10, 12]
DEFAULT_GD_LAMBDAS = [1e-6, 1e-4, 1e-2]


# ----------------------------
# Kernel construction (mirrors the runners)
# ----------------------------
def classical_kernel_blocks(
    factory: ClassicalKernelFactory,
    name: str,
    X_tr: np.ndarray,
    X_id: np.ndarray,
    X_ood: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Unit-diagonal kernel blocks for a classical kernel, delegating to the
    extended-runner factory so both analyses share one construction (all
    hyperparameters are fit on train only)."""
    return {
        "train": factory.block(name, X_tr, X_tr),
        "id_cross": factory.block(name, X_id, X_tr),
        "ood_cross": factory.block(name, X_ood, X_tr),
        "id_within": factory.block(name, X_id, X_id),
        "ood_within": factory.block(name, X_ood, X_ood),
    }


def quantum_kernel_blocks(
    qcfg: Dict[str, Any],
    dim: int,
    X_tr: np.ndarray,
    X_id: np.ndarray,
    X_ood: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Fidelity kernel blocks |<phi(x)|phi(z)>|^2 (unit diagonal by construction)."""
    fmap = build_feature_map(qcfg, feature_dim=dim)
    sv_tr = compute_statevectors_batch(X_tr, fmap, dtype=np.complex64)
    sv_id = compute_statevectors_batch(X_id, fmap, dtype=np.complex64)
    sv_ood = compute_statevectors_batch(X_ood, fmap, dtype=np.complex64)
    return {
        "train": kernel_block_abs2(sv_tr, sv_tr, out_dtype=np.float64),
        "id_cross": kernel_block_abs2(sv_id, sv_tr, out_dtype=np.float64),
        "ood_cross": kernel_block_abs2(sv_ood, sv_tr, out_dtype=np.float64),
        "id_within": kernel_block_abs2(sv_id, sv_id, out_dtype=np.float64),
        "ood_within": kernel_block_abs2(sv_ood, sv_ood, out_dtype=np.float64),
    }


# ----------------------------
# Geometry descriptors
# ----------------------------
def psd_eigvals(K: np.ndarray) -> np.ndarray:
    """Eigenvalues of a symmetric kernel matrix, clipped at 0, descending."""
    w = np.linalg.eigvalsh((K + K.T) / 2.0)
    w = np.clip(w, 0.0, None)
    return w[::-1]


def spectral_summary(K: np.ndarray, prefix: str) -> Dict[str, float]:
    w = psd_eigvals(K)
    total = float(w.sum())
    if total <= 0.0:
        return {f"{prefix}_top1_share": np.nan}
    p = w / total
    nz = p[p > 1e-15]
    entropy = float(-(nz * np.log(nz)).sum())
    out = {
        f"{prefix}_top1_share": float(p[0]),
        f"{prefix}_top5_share": float(p[:5].sum()),
        f"{prefix}_top20_share": float(p[:20].sum()),
        f"{prefix}_eff_rank": float(np.exp(entropy)),
    }
    return out


def offdiag_stats(K: np.ndarray, prefix: str, square: bool = True) -> Dict[str, float]:
    """Mean/std of kernel entries. For square within-blocks the diagonal is excluded."""
    if square:
        mask = ~np.eye(K.shape[0], dtype=bool)
        vals = K[mask]
    else:
        vals = K.ravel()
    return {
        f"{prefix}_mean": float(vals.mean()),
        f"{prefix}_std": float(vals.std()),
    }


def centered_kta(K: np.ndarray, y01: np.ndarray) -> float:
    """Centered kernel-target alignment between K and the label kernel yy^T."""
    y = np.where(np.asarray(y01).ravel() == 1, 1.0, -1.0)
    n = y.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    num = float(y @ Kc @ y)
    denom = float(np.linalg.norm(Kc, "fro")) * n
    if denom <= 0.0:
        return float("nan")
    return num / denom


def _top_eigval(M: np.ndarray) -> float:
    M = (M + M.T) / 2.0
    if M.shape[0] > 500:
        try:
            from scipy.sparse.linalg import eigsh

            return float(eigsh(M, k=1, which="LA", return_eigenvectors=False)[0])
        except Exception:
            pass
    return float(np.linalg.eigvalsh(M)[-1])


def geometric_differences(
    classical_train: Dict[str, np.ndarray],
    quantum_train: Dict[str, np.ndarray],
    lambdas: List[float],
) -> Dict[str, Dict[str, float]]:
    """g(K_C -> K_Q) = sqrt(|| sqrt(K_Q) (K_C + lam*n*I)^{-1} sqrt(K_Q) ||_inf),
    following Huang et al. (2021), with both kernels normalized to unit diagonal
    (hence trace n) and a relative regularizer lam scaled by n.

    Large g means the quantum geometry can deviate substantially from what the
    classical kernel can express; g ~ sqrt(n) is the saturation scale.

    Eigendecompositions are computed once per kernel and reused across the
    (classical, quantum, lambda) grid. Returns {q_name: {column: g}}.
    """
    if not quantum_train:
        return {}
    n = next(iter(classical_train.values())).shape[0]

    sqrt_q: Dict[str, np.ndarray] = {}
    for qname, K_q in quantum_train.items():
        wq, Vq = np.linalg.eigh((K_q + K_q.T) / 2.0)
        wq = np.clip(wq, 0.0, None)
        sqrt_q[qname] = (Vq * np.sqrt(wq)) @ Vq.T

    out: Dict[str, Dict[str, float]] = {qname: {} for qname in quantum_train}
    for cname, K_c in classical_train.items():
        wc, Vc = np.linalg.eigh((K_c + K_c.T) / 2.0)
        wc = np.clip(wc, 0.0, None)
        for lam in lambdas:
            inv_Kc = (Vc / (wc + lam * n)) @ Vc.T
            for qname, S_q in sqrt_q.items():
                M = S_q @ inv_Kc @ S_q
                top = _top_eigval(M)
                out[qname][f"gd_vs_{cname}_lam{lam:g}"] = float(np.sqrt(max(top, 0.0)))
    return out


def kernel_descriptor_row(
    blocks: Dict[str, np.ndarray],
    y_tr: np.ndarray,
    y_id: np.ndarray,
    y_ood: np.ndarray,
) -> Dict[str, float]:
    row: Dict[str, float] = {}
    row.update(spectral_summary(blocks["train"], "spec_train"))
    row.update(spectral_summary(blocks["ood_within"], "spec_ood"))
    row.update(offdiag_stats(blocks["train"], "k_train", square=True))
    row.update(offdiag_stats(blocks["id_cross"], "k_id2train", square=False))
    row.update(offdiag_stats(blocks["ood_cross"], "k_ood2train", square=False))

    # How "novel" does the OOD set look relative to train, per kernel geometry?
    id_mean = row["k_id2train_mean"]
    row["ood_novelty_ratio"] = row["k_ood2train_mean"] / id_mean if id_mean > 0 else np.nan

    row["kta_train"] = centered_kta(blocks["train"], y_tr)
    row["kta_id"] = centered_kta(blocks["id_within"], y_id)
    row["kta_ood"] = centered_kta(blocks["ood_within"], y_ood)
    row["kta_drop_id_to_ood"] = row["kta_id"] - row["kta_ood"]
    return row


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Kernel-geometry analysis on EMBER q-splits (kernel-swap protocol).")
    ap.add_argument("--in-dir", type=str, default=str(DEFAULT_IN_DIR))
    ap.add_argument("--splits-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dims", type=int, nargs="+", default=DEFAULT_DIMS)
    ap.add_argument("--gd-lambdas", type=float, nargs="+", default=DEFAULT_GD_LAMBDAS)
    ap.add_argument("--setting-label", type=str, default="", help="Free-form label (variant/master-seed/size) stored in the output rows.")
    ap.add_argument("--classical-kernels", type=str, nargs="+", default=["linear", "rbf_gscale"],
                    choices=EXTENDED_CLASSICAL_KERNELS)
    ap.add_argument("--quantum-configs-json", type=str, default="",
                    help="JSON list of quantum configs; pass a file containing [] to skip quantum kernels.")
    ap.add_argument("--save-spectra", action="store_true", help="Also store full train-kernel spectra as .npz.")
    ap.add_argument("--mmap", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qcfgs = DEFAULT_QUANTUM_CONFIGS
    if args.quantum_configs_json.strip():
        qcfgs = json.loads(Path(args.quantum_configs_json).read_text(encoding="utf-8"))

    X = np.load(in_dir / "X.npy", mmap_mode="r" if args.mmap else None)
    y = np.load(in_dir / "y.npy").astype(np.int64).ravel()

    idx = {
        "train": load_indices(splits_dir / "train_idx.npy"),
        "id_test": load_indices(splits_dir / "id_test_idx.npy"),
        "ood_test": load_indices(splits_dir / "ood_test_idx.npy"),
    }
    y_tr, y_id, y_ood = (y[idx[k]] for k in ["train", "id_test", "ood_test"])

    rows: List[Dict[str, Any]] = []
    spectra: Dict[str, np.ndarray] = {}

    for dim in [int(d) for d in args.dims]:
        t0 = time.time()
        embed = make_embedding_pipeline(
            dim=dim, select_k=None, use_scaling=True,
            angle_min=0.0, angle_max=float(np.pi), seed=int(args.seed),
        )
        embed.fit(np.asarray(X[idx["train"]]), y_tr)
        X_tr = np.asarray(embed.transform(np.asarray(X[idx["train"]])), dtype=np.float64)
        X_id = np.asarray(embed.transform(np.asarray(X[idx["id_test"]])), dtype=np.float64)
        X_ood = np.asarray(embed.transform(np.asarray(X[idx["ood_test"]])), dtype=np.float64)

        classical_train_blocks: Dict[str, np.ndarray] = {}
        all_blocks: Dict[str, Dict[str, np.ndarray]] = {}

        factory = ClassicalKernelFactory(X_tr, seed=int(args.seed))
        for cname in args.classical_kernels:
            blocks = classical_kernel_blocks(factory, cname, X_tr, X_id, X_ood)
            all_blocks[cname] = blocks
            classical_train_blocks[cname] = blocks["train"]

        for qcfg in qcfgs:
            all_blocks[qcfg["id"]] = quantum_kernel_blocks(qcfg, dim, X_tr, X_id, X_ood)

        quantum_train_blocks = {
            k: v["train"] for k, v in all_blocks.items() if k not in classical_train_blocks
        }
        gd = geometric_differences(
            classical_train_blocks, quantum_train_blocks, [float(l) for l in args.gd_lambdas]
        )

        for kname, blocks in all_blocks.items():
            family = "classical" if kname in classical_train_blocks else "quantum"
            row: Dict[str, Any] = {
                "setting_label": args.setting_label or splits_dir.name,
                "splits_dir": str(splits_dir),
                "dim": dim,
                "kernel": kname,
                "family": family,
                "n_train": int(idx["train"].size),
            }
            row.update(kernel_descriptor_row(blocks, y_tr, y_id, y_ood))

            # Geometric difference of each quantum kernel w.r.t. each classical one.
            if family == "quantum":
                row.update(gd[kname])

            rows.append(row)
            if args.save_spectra:
                spectra[f"{kname}__d{dim}"] = psd_eigvals(blocks["train"])

        print(f"[OK] dim={dim} done in {time.time() - t0:.1f}s ({len(all_blocks)} kernels)")

    df = pd.DataFrame(rows)
    label = (args.setting_label or splits_dir.name).replace("/", "_")
    out_csv = out_dir / f"kernel_geometry__{label}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] Wrote {out_csv}")

    if args.save_spectra:
        out_npz = out_dir / f"kernel_geometry_spectra__{label}.npz"
        np.savez_compressed(out_npz, **spectra)
        print(f"[✓] Wrote {out_npz}")


if __name__ == "__main__":
    main()

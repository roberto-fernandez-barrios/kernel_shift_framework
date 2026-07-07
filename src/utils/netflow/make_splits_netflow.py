# src/utils/netflow/make_splits_netflow.py
"""
Master shift-splits for netflow ref/cur exports, mirroring the EMBER master
layout (train_idx / id_test_idx / ood_test_idx [/ ood_low_idx / ood_high_idx]
+ meta.json) so make_qsplits_from_master and all runners work unchanged.

Variants:
  - m2_centroid: within the class-balanced REFERENCE pool, fix a provisional
    train pool, compute class-conditional centroids in a train-only
    MaxAbs+TruncatedSVD space, and take within-class low/high distance tails
    as the OOD pool (faithful analog of the paper's m2 mechanism).
  - m1_nonzero: same structure with the score = number of non-zero features
    (analog of the paper's m1 sparsity mechanism).
  - natural_cur: train/ID pools from the reference pool, OOD pool from the
    class-balanced CURRENT pool (natural drift; no low/high tails).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler

VARIANTS = ["m2_centroid", "m1_nonzero", "natural_cur"]
DEFAULT_TAIL_FRAC = 0.10
DEFAULT_TRAIN_POOL_FRAC = 0.50
DEFAULT_SVD_DIM = 16


def balanced_subset(y: np.ndarray, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Class-balanced subset of idx (downsample the majority class)."""
    idx0 = idx[y[idx] == 0]
    idx1 = idx[y[idx] == 1]
    n = min(idx0.size, idx1.size)
    if n == 0:
        raise RuntimeError("A class is empty in the requested pool")
    out = np.concatenate([
        rng.choice(idx0, size=n, replace=False),
        rng.choice(idx1, size=n, replace=False),
    ])
    rng.shuffle(out)
    return out


def stratified_split(y: np.ndarray, idx: np.ndarray, frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    parts_a, parts_b = [], []
    for c in (0, 1):
        idx_c = idx[y[idx] == c]
        idx_c = rng.permutation(idx_c)
        k = int(round(frac * idx_c.size))
        parts_a.append(idx_c[:k])
        parts_b.append(idx_c[k:])
    a, b = np.concatenate(parts_a), np.concatenate(parts_b)
    rng.shuffle(a)
    rng.shuffle(b)
    return a, b


def within_class_tails(
    y: np.ndarray, idx: np.ndarray, score: np.ndarray, tail_frac: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per class, split idx into (low tail, high tail, middle) by score quantiles."""
    lows, highs, mids = [], [], []
    for c in (0, 1):
        idx_c = idx[y[idx] == c]
        s = score[idx_c]
        lo_thr, hi_thr = np.quantile(s, [tail_frac, 1.0 - tail_frac])
        lows.append(idx_c[s <= lo_thr])
        highs.append(idx_c[s >= hi_thr])
        mids.append(idx_c[(s > lo_thr) & (s < hi_thr)])
    return np.concatenate(lows), np.concatenate(highs), np.concatenate(mids)


def main() -> None:
    ap = argparse.ArgumentParser(description="Master shift splits for netflow exports.")
    ap.add_argument("--in-dir", type=Path, required=True, help="Directory with X.npy, y.npy, meta_export.json")
    ap.add_argument("--out-dir", type=Path, required=True, help="Master split directory to create")
    ap.add_argument("--variant", type=str, required=True, choices=VARIANTS)
    ap.add_argument("--master-seed", type=int, required=True)
    ap.add_argument("--tail-frac", type=float, default=DEFAULT_TAIL_FRAC)
    ap.add_argument("--train-pool-frac", type=float, default=DEFAULT_TRAIN_POOL_FRAC)
    ap.add_argument("--svd-dim", type=int, default=DEFAULT_SVD_DIM)
    args = ap.parse_args()

    meta_export = json.loads((args.in_dir / "meta_export.json").read_text(encoding="utf-8"))
    n_ref = int(meta_export["n_ref"])
    y = np.load(args.in_dir / "y.npy").astype(np.int64).ravel()
    rng = np.random.default_rng(int(args.master_seed))

    ref_idx = np.arange(n_ref, dtype=np.int64)
    cur_idx = np.arange(n_ref, y.size, dtype=np.int64)
    ref_bal = balanced_subset(y, ref_idx, rng)

    extra: Dict[str, object] = {}

    if args.variant == "natural_cur":
        train_pool, id_pool = stratified_split(y, ref_bal, args.train_pool_frac, rng)
        ood_pool = balanced_subset(y, cur_idx, rng)
        low = high = None
    else:
        train_pool, rest = stratified_split(y, ref_bal, args.train_pool_frac, rng)
        if args.variant == "m1_nonzero":
            X = np.load(args.in_dir / "X.npy", mmap_mode="r")
            score = np.zeros(y.size, dtype=np.float64)
            score[rest] = (np.asarray(X[rest]) != 0).sum(axis=1).astype(np.float64)
        else:  # m2_centroid
            X = np.load(args.in_dir / "X.npy", mmap_mode="r")
            X_tr = np.asarray(X[train_pool], dtype=np.float64)
            scaler = MaxAbsScaler().fit(X_tr)
            dim = min(int(args.svd_dim), X_tr.shape[1] - 1)
            svd = TruncatedSVD(n_components=dim, random_state=int(args.master_seed)).fit(scaler.transform(X_tr))
            Z_tr = svd.transform(scaler.transform(X_tr))
            centroids = {c: Z_tr[y[train_pool] == c].mean(axis=0) for c in (0, 1)}
            Z_rest = svd.transform(scaler.transform(np.asarray(X[rest], dtype=np.float64)))
            score = np.zeros(y.size, dtype=np.float64)
            dists = np.empty(rest.size, dtype=np.float64)
            for c in (0, 1):
                m = y[rest] == c
                dists[m] = np.linalg.norm(Z_rest[m] - centroids[c], axis=1)
            score[rest] = dists
            extra["svd_dim"] = dim
        low, high, id_pool = within_class_tails(y, rest, score, args.tail_frac)
        ood_pool = np.concatenate([low, high])
        rng.shuffle(ood_pool)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "train_idx.npy", train_pool)
    np.save(args.out_dir / "id_test_idx.npy", id_pool)
    np.save(args.out_dir / "ood_test_idx.npy", ood_pool)
    if low is not None:
        np.save(args.out_dir / "ood_low_idx.npy", low)
        np.save(args.out_dir / "ood_high_idx.npy", high)

    def _cc(idx: np.ndarray) -> Dict[str, int]:
        return {"benign": int((y[idx] == 0).sum()), "attack": int((y[idx] == 1).sum())}

    meta = {
        "kind": "netflow_master_split",
        "variant": args.variant,
        "master_seed": int(args.master_seed),
        "in_dir": str(args.in_dir),
        "params": {
            "tail_frac": args.tail_frac,
            "train_pool_frac": args.train_pool_frac,
            **extra,
        },
        "pool_sizes": {
            "train": int(train_pool.size),
            "id_test": int(id_pool.size),
            "ood_test": int(ood_pool.size),
            **({"ood_low": int(low.size), "ood_high": int(high.size)} if low is not None else {}),
        },
        "class_counts": {
            "train": _cc(train_pool), "id_test": _cc(id_pool), "ood_test": _cc(ood_pool),
        },
        "export_meta": meta_export,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[✓] Master splits ({args.variant}, ms={args.master_seed}) -> {args.out_dir}")
    print(f"    pools: {meta['pool_sizes']} | class counts: {meta['class_counts']}")


if __name__ == "__main__":
    main()

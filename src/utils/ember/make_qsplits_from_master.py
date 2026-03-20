# src/utils/ember/make_qsplits_from_master.py
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_npy(path: Path) -> str:
    # after np.save, hashing file is simplest and robust
    return _sha256_file(path)


def subsample(arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int64).ravel()
    if n <= 0:
        return np.empty((0,), dtype=np.int64)
    if n >= arr.size:
        # keep deterministic order? master idx already shuffled; keep as-is
        return arr
    return rng.choice(arr, size=n, replace=False)


def _assert_disjoint(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    sa, sb = set(a.tolist()), set(b.tolist())
    inter = sa & sb
    if inter:
        raise RuntimeError(f"[qsplit] Overlap between {name_a} and {name_b}: {len(inter)} indices")


def _load_idx(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return np.load(path).astype(np.int64).ravel()


def main():
    ap = argparse.ArgumentParser(description="Generate q-splits from master EMBER shift splits (leakage-safe).")
    ap.add_argument("--src", type=str, default="data/processed/ember/splits_sparsity")
    ap.add_argument("--dst-root", type=str, default="data/processed/ember")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-id", type=int, default=500)
    ap.add_argument("--n-ood", type=int, default=500)

    ap.add_argument(
        "--use-low-high",
        action="store_true",
        help="If set, build ood_test from subsampled low+high (half+half), enforcing exact size and disjointness.",
    )
    ap.add_argument(
        "--strict-sizes",
        action="store_true",
        help="Fail if requested sizes exceed master split sizes (instead of returning full). Recommended for paper runs.",
    )
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(f"--src does not exist: {src}")

    dst = Path(args.dst_root) / f"splits_sparsity_q{args.n_train}_id{args.n_id}_ood{args.n_ood}_seed{args.seed}"
    dst.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))

    # Load master splits
    train_full = _load_idx(src / "train_idx.npy")
    id_full = _load_idx(src / "id_test_idx.npy")
    ood_full_path = src / "ood_test_idx.npy"

    if args.strict_sizes:
        if args.n_train > train_full.size:
            raise RuntimeError(f"Requested n_train={args.n_train} > master train size={train_full.size}")
        if args.n_id > id_full.size:
            raise RuntimeError(f"Requested n_id={args.n_id} > master id size={id_full.size}")

    # TRAIN + ID
    train_q = subsample(train_full, int(args.n_train), rng)
    id_q = subsample(id_full, int(args.n_id), rng)

    _assert_disjoint(train_q, id_q, "train_q", "id_q")

    np.save(dst / "train_idx.npy", train_q)
    np.save(dst / "id_test_idx.npy", id_q)

    # OOD
    used_low_high = False
    low_q = None
    high_q = None

    low_path = src / "ood_low_idx.npy"
    high_path = src / "ood_high_idx.npy"

    if args.use_low_high and low_path.exists() and high_path.exists():
        low_full = _load_idx(low_path)
        high_full = _load_idx(high_path)

        # Safety: low/high should be disjoint in a correct master split
        _assert_disjoint(low_full, high_full, "ood_low_full", "ood_high_full")

        if args.strict_sizes:
            n_half = int(args.n_ood) // 2
            if n_half > low_full.size:
                raise RuntimeError(f"Requested low half={n_half} > master low size={low_full.size}")
            if (int(args.n_ood) - n_half) > high_full.size:
                raise RuntimeError(
                    f"Requested high half={(int(args.n_ood) - n_half)} > master high size={high_full.size}"
                )

        n_half = int(args.n_ood) // 2
        low_q = subsample(low_full, n_half, rng)

        # sample high WITHOUT overlapping low_q (paranoid safety even if master is clean)
        low_set = set(low_q.tolist())
        high_candidates = np.array([i for i in high_full.tolist() if i not in low_set], dtype=np.int64)
        high_q = subsample(high_candidates, int(args.n_ood) - n_half, rng)

        # Now enforce exact size
        ood_q = np.concatenate([low_q, high_q])
        if ood_q.size != int(args.n_ood):
            raise RuntimeError(f"OOD size mismatch: got {ood_q.size} but requested {int(args.n_ood)}")

        # shuffle final ood
        rng.shuffle(ood_q)

        used_low_high = True
        np.save(dst / "ood_low_idx.npy", low_q)
        np.save(dst / "ood_high_idx.npy", high_q)
        np.save(dst / "ood_test_idx.npy", ood_q)

    else:
        if not ood_full_path.exists():
            raise FileNotFoundError(f"Missing {ood_full_path}")
        ood_full = _load_idx(ood_full_path)

        if args.strict_sizes and args.n_ood > ood_full.size:
            raise RuntimeError(f"Requested n_ood={args.n_ood} > master ood size={ood_full.size}")

        ood_q = subsample(ood_full, int(args.n_ood), rng)
        np.save(dst / "ood_test_idx.npy", ood_q)

        # copy low/high if present (full, not subsampled) – optional but useful
        for name in ["ood_low_idx.npy", "ood_high_idx.npy"]:
            p = src / name
            if p.exists():
                np.save(dst / name, np.load(p))

    # Final disjointness checks
    _assert_disjoint(train_q, ood_q, "train_q", "ood_q")
    _assert_disjoint(id_q, ood_q, "id_q", "ood_q")

    # Hash outputs
    out_hashes = {
        "train_idx_sha256": _sha256_npy(dst / "train_idx.npy"),
        "id_test_idx_sha256": _sha256_npy(dst / "id_test_idx.npy"),
        "ood_test_idx_sha256": _sha256_npy(dst / "ood_test_idx.npy"),
    }
    if used_low_high:
        out_hashes["ood_low_idx_sha256"] = _sha256_npy(dst / "ood_low_idx.npy")
        out_hashes["ood_high_idx_sha256"] = _sha256_npy(dst / "ood_high_idx.npy")

    meta: Dict[str, object] = {
        "dataset": "ember",
        "seed": int(args.seed),
        "source_splits": str(src),
        "requested_sizes": {"n_train": int(args.n_train), "n_id": int(args.n_id), "n_ood": int(args.n_ood)},
        "actual_sizes": {"n_train": int(train_q.size), "n_id": int(id_q.size), "n_ood": int(ood_q.size)},
        "use_low_high": bool(args.use_low_high),
        "used_low_high": bool(used_low_high),
        "strict_sizes": bool(args.strict_sizes),
        "inputs": {
            "train_full_size": int(train_full.size),
            "id_full_size": int(id_full.size),
            "ood_full_size": int(_load_idx(ood_full_path).size) if ood_full_path.exists() else None,
            "master_meta_exists": bool((src / "meta.json").exists()),
            **({"master_meta_sha256": _sha256_file(src / "meta.json")} if (src / "meta.json").exists() else {}),
        },
        "output_fingerprints": out_hashes,
    }

    (dst / "meta_q.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[✓] Wrote q-splits to: {dst}")


if __name__ == "__main__":
    main()

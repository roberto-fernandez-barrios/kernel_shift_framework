# src/utils/ember/make_splits_ember_sparsity.py
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np

# Optional dependency (only needed for dist_to_train_* when --svd-dim > 0)
try:
    from sklearn.decomposition import TruncatedSVD
except Exception:
    TruncatedSVD = None


# ----------------------------
# Utilities
# ----------------------------
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _pos_rate(y: np.ndarray, idx: np.ndarray) -> float:
    if idx.size == 0:
        return float("nan")
    return float((y[idx] == 1).mean())


def _score_stats(score: np.ndarray, idx: np.ndarray) -> Dict[str, float]:
    if idx.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan}
    v = score[idx]
    return {
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
    }


def _split_stats_by_class(score: np.ndarray, y: np.ndarray, idx: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for cls in (0, 1):
        cls_idx = idx[y[idx] == cls]
        out[f"class_{cls}"] = _score_stats(score, cls_idx)
    return out


def _assert_disjoint(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    sa, sb = set(a.tolist()), set(b.tolist())
    inter = sa & sb
    if inter:
        raise RuntimeError(f"Overlap between {name_a} and {name_b}: {len(inter)} indices")


def _save_npy(path: Path, arr: np.ndarray) -> str:
    np.save(path, arr)
    return _sha256_file(path)


def _load_feature_names(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Export it when building X.npy to map columns like hist_0..hist_255."
        )
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise RuntimeError(f"{path} must be a JSON list[str].")
    return names


def _columns_indices_by_prefix(names: List[str], prefix: str, n_bins: int) -> np.ndarray:
    want = {f"{prefix}{i}" for i in range(int(n_bins))}
    idx = [i for i, n in enumerate(names) if n in want]
    if len(idx) != int(n_bins):
        sample = [n for n in names if n.startswith(prefix)][:10]
        raise RuntimeError(
            f"Expected {n_bins} columns named {prefix}0..{prefix}{n_bins-1}, found {len(idx)}.\n"
            f"Example existing '{prefix}*' columns: {sample}\n"
            f"Check feature_names.json ordering / naming."
        )
    return np.array(idx, dtype=np.int64)


def _compute_score_dense_feature_block(
    X: np.ndarray,
    cols_idx: np.ndarray,
    mode: str,
    eps: float,
    batch_size: int,
) -> np.ndarray:
    n = int(X.shape[0])
    score = np.empty(n, dtype=np.int64)

    # For non-negative hist/byteent, (H > eps) is OK. If you later add signed features, switch to abs(H) > eps.
    if batch_size <= 0:
        H = X[:, cols_idx]
        if mode == "nnz":
            score[:] = (H > eps).sum(axis=1).astype(np.int64)
        elif mode == "zeros":
            score[:] = (np.abs(H) <= eps).sum(axis=1).astype(np.int64)
        else:
            raise ValueError(f"Unknown mode={mode}")
        return score

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        H = X[start:end, cols_idx]
        if mode == "nnz":
            score[start:end] = (H > eps).sum(axis=1).astype(np.int64)
        elif mode == "zeros":
            score[start:end] = (np.abs(H) <= eps).sum(axis=1).astype(np.int64)
        else:
            raise ValueError(f"Unknown mode={mode}")

    return score


def _median_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.median(a) - np.median(b))


def _cliffs_delta(a: np.ndarray, b: np.ndarray, max_pairs: int, rng: np.random.Generator) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")

    max_pairs = int(max_pairs)
    if max_pairs <= 0:
        return float("nan")

    side = max(1, int(np.sqrt(max_pairs)))
    m = int(min(a.size, side))
    n = int(min(b.size, side))
    if m <= 0 or n <= 0:
        return float("nan")

    a_s = rng.choice(a, size=m, replace=False) if a.size > m else a
    b_s = rng.choice(b, size=n, replace=False) if b.size > n else b

    gt = (a_s[:, None] > b_s[None, :]).sum()
    lt = (a_s[:, None] < b_s[None, :]).sum()
    denom = a_s.size * b_s.size
    return float((gt - lt) / denom)


def _stratified_split_two_way(
    indices: np.ndarray,
    y: np.ndarray,
    frac_a: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    idx0 = indices[y[indices] == 0].copy()
    idx1 = indices[y[indices] == 1].copy()

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_a = int(np.floor(len(idx0) * frac_a))
    n1_a = int(np.floor(len(idx1) * frac_a))

    A = np.concatenate([idx0[:n0_a], idx1[:n1_a]])
    B = np.concatenate([idx0[n0_a:], idx1[n1_a:]])

    rng.shuffle(A)
    rng.shuffle(B)
    return A, B


def _stratified_split_n(
    indices: np.ndarray,
    y: np.ndarray,
    n_a: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split indices into (A, B) stratified by y with |A| == n_a (best-effort exact).
    Uses proportional allocation per class, then adjusts for rounding.
    """
    indices = np.asarray(indices, dtype=np.int64)
    n_a = int(n_a)
    if n_a < 0 or n_a > indices.size:
        raise RuntimeError(f"Invalid n_a={n_a} for indices size={indices.size}")

    idx0 = indices[y[indices] == 0].copy()
    idx1 = indices[y[indices] == 1].copy()

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0 = idx0.size
    n1 = idx1.size
    tot = n0 + n1
    if tot != indices.size:
        raise RuntimeError("Internal: class partition size mismatch")

    if n_a == 0:
        A = np.array([], dtype=np.int64)
        B = indices.copy()
        rng.shuffle(B)
        return A, B

    if n_a == tot:
        A = indices.copy()
        B = np.array([], dtype=np.int64)
        rng.shuffle(A)
        return A, B

    # proportional targets
    p0 = n0 / tot if tot > 0 else 0.0
    n0_a = int(round(n_a * p0))
    n0_a = max(0, min(n0_a, n0))
    n1_a = n_a - n0_a

    # clamp & adjust if overflow
    if n1_a < 0:
        n1_a = 0
        n0_a = n_a
    if n1_a > n1:
        n1_a = n1
        n0_a = n_a - n1_a

    # final sanity
    if n0_a < 0 or n0_a > n0 or n1_a < 0 or n1_a > n1 or (n0_a + n1_a) != n_a:
        # fallback: safe floor-based split
        frac = n_a / tot
        return _stratified_split_two_way(indices, y, frac, rng=rng)

    A = np.concatenate([idx0[:n0_a], idx1[:n1_a]])
    B = np.concatenate([idx0[n0_a:], idx1[n1_a:]])
    rng.shuffle(A)
    rng.shuffle(B)
    return A, B


# ----------------------------
# OOD builders
# ----------------------------
def _make_ood_extremes_within_class(
    score: np.ndarray,
    y: np.ndarray,
    frac_each_side: float,
    rng: np.random.Generator,
    *,
    candidate_idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Within-class extremes, with optional candidate restriction.
    If candidate_idx is provided, extremes are taken ONLY within that subset.

    NOTE: This version uses fraction-per-class (classic M1 behavior).
    For M2 "paper-safe exact sizes" we use _make_ood_extremes_within_class_counts().
    """
    low_parts = []
    high_parts = []
    thr_by_class: Dict[str, Dict[str, float]] = {}

    if candidate_idx is None:
        candidate_idx = np.arange(y.shape[0], dtype=np.int64)
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)

    for cls in (0, 1):
        cls_idx = candidate_idx[y[candidate_idx] == cls]
        if cls_idx.size == 0:
            raise RuntimeError(f"Class {cls} has zero samples in candidate set.")

        order_cls = cls_idx[np.argsort(score[cls_idx])]
        k = int(np.floor(len(order_cls) * float(frac_each_side)))
        if k <= 0:
            raise RuntimeError(
                f"Not enough samples in class {cls} to take extremes with frac_each_side={frac_each_side}"
            )

        low = order_cls[:k]
        high = order_cls[-k:]

        low_parts.append(low)
        high_parts.append(high)

        thr_by_class[f"class_{cls}"] = {
            "ood_low_max_score": float(np.max(score[low])) if low.size else float("nan"),
            "ood_high_min_score": float(np.min(score[high])) if high.size else float("nan"),
        }

    low_ood = np.concatenate(low_parts)
    high_ood = np.concatenate(high_parts)

    ood_idx = np.unique(np.concatenate([low_ood, high_ood]))
    rng.shuffle(ood_idx)
    return ood_idx, low_ood, high_ood, thr_by_class


def _make_ood_extremes_within_class_counts(
    score: np.ndarray,
    y: np.ndarray,
    *,
    candidate_idx: np.ndarray,
    n_ood_total: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
    """
    Paper-safe builder: choose EXACT total OOD size (best-effort), within-class,
    selecting low/high extremes per class using counts (not fractions).

    We allocate OOD per class proportional to class presence in candidate_idx.
    Within each class, we split into low/high: low gets floor(n/2), high gets remainder.
    """
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    n_ood_total = int(n_ood_total)
    if n_ood_total <= 0:
        raise RuntimeError(f"Invalid n_ood_total={n_ood_total}")
    if n_ood_total >= candidate_idx.size:
        raise RuntimeError(f"n_ood_total={n_ood_total} must be < candidate size={candidate_idx.size}")

    # class pools
    cls0 = candidate_idx[y[candidate_idx] == 0]
    cls1 = candidate_idx[y[candidate_idx] == 1]
    n0 = int(cls0.size)
    n1 = int(cls1.size)
    if n0 == 0 or n1 == 0:
        raise RuntimeError("Candidate set must contain both classes for within-class OOD.")

    # allocate per class
    p0 = n0 / (n0 + n1)
    n0_ood = int(round(n_ood_total * p0))
    n0_ood = max(1, min(n0_ood, n0 - 1))  # keep at least 1 and leave at least 1 non-OOD
    n1_ood = n_ood_total - n0_ood
    n1_ood = max(1, min(n1_ood, n1 - 1))
    # adjust to keep sum close to n_ood_total (best-effort)
    n0_ood = n_ood_total - n1_ood

    if n0_ood <= 0 or n1_ood <= 0:
        raise RuntimeError(f"Could not allocate OOD per class: n0_ood={n0_ood}, n1_ood={n1_ood}")

    # low/high split
    n0_low = n0_ood // 2
    n0_high = n0_ood - n0_low
    n1_low = n1_ood // 2
    n1_high = n1_ood - n1_low

    if n0_low <= 0 or n0_high <= 0 or n1_low <= 0 or n1_high <= 0:
        raise RuntimeError("Not enough OOD budget per class to split into low/high (increase ood_test_frac or data size).")

    low_parts = []
    high_parts = []
    thr_by_class: Dict[str, Dict[str, float]] = {}
    counts_by_class: Dict[str, Dict[str, int]] = {}

    for cls, cls_idx, n_low, n_high in [
        (0, cls0, n0_low, n0_high),
        (1, cls1, n1_low, n1_high),
    ]:
        order_cls = cls_idx[np.argsort(score[cls_idx])]  # low->high
        if (n_low + n_high) > order_cls.size:
            raise RuntimeError(f"Class {cls}: requested {n_low+n_high} OOD but only {order_cls.size} candidates.")
        low = order_cls[:n_low]
        high = order_cls[-n_high:]

        low_parts.append(low)
        high_parts.append(high)

        thr_by_class[f"class_{cls}"] = {
            "ood_low_max_score": float(np.max(score[low])) if low.size else float("nan"),
            "ood_high_min_score": float(np.min(score[high])) if high.size else float("nan"),
        }
        counts_by_class[f"class_{cls}"] = {
            "n_candidate": int(order_cls.size),
            "n_ood_total": int(n_low + n_high),
            "n_ood_low": int(n_low),
            "n_ood_high": int(n_high),
        }

    low_ood = np.concatenate(low_parts)
    high_ood = np.concatenate(high_parts)
    ood_idx = np.unique(np.concatenate([low_ood, high_ood]))
    rng.shuffle(ood_idx)
    return ood_idx, low_ood, high_ood, thr_by_class, counts_by_class


def _make_ood_extremes_global(
    score: np.ndarray,
    y: np.ndarray,
    frac_each_side: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    n = int(score.shape[0])
    k = int(np.floor(n * float(frac_each_side)))
    if k <= 0:
        raise RuntimeError(f"Not enough samples to take global extremes with frac_each_side={frac_each_side}")
    if 2 * k >= n:
        raise RuntimeError(
            f"Global extremes would overlap: n={n}, k={k} (2*k >= n). Reduce --ood-extreme-frac-each-side."
        )

    order = np.argsort(score)
    low_ood = order[:k]
    high_ood = order[-k:]

    if np.intersect1d(low_ood, high_ood).size > 0:
        raise RuntimeError("Unexpected overlap between low/high in global extremes (should not happen).")

    ood_idx = np.unique(np.concatenate([low_ood, high_ood]))
    rng.shuffle(ood_idx)

    thr_by_class: Dict[str, Dict[str, float]] = {}
    for cls in (0, 1):
        low_cls = low_ood[y[low_ood] == cls]
        high_cls = high_ood[y[high_ood] == cls]
        thr_by_class[f"class_{cls}"] = {
            "ood_low_max_score": float(np.max(score[low_cls])) if low_cls.size else float("nan"),
            "ood_high_min_score": float(np.min(score[high_cls])) if high_cls.size else float("nan"),
        }

    return ood_idx, low_ood, high_ood, thr_by_class


def _compute_dist_to_train_score_within_class(
    X: np.ndarray,
    y: np.ndarray,
    train_ref_idx: np.ndarray,
    *,
    svd_dim: int,
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    """
    Score = distance to class centroid computed on a reference TRAIN set (within class).
    - Build centroids mu0/mu1 from train_ref_idx.
    - For each sample x, score = ||x - mu_{y(x)}||_2.

    If svd_dim > 0:
      - Fit SVD on reference train only (no peeking).
      - Transform full X in batches to control memory.
      - Distances computed in projected space.

    IMPORTANT improvement:
      - Distances computed via norms + dot:
            ||x - mu||^2 = ||x||^2 + ||mu||^2 - 2 x·mu
        This avoids big temporaries ((Xi - mu)**2) for large feature dims.
    """
    X_ref = X

    # Optional SVD projection (fit on train_ref only)
    if svd_dim and int(svd_dim) > 0:
        if TruncatedSVD is None:
            raise RuntimeError("scikit-learn is required for --svd-dim but it's not available.")
        svd_dim = int(svd_dim)
        svd = TruncatedSVD(n_components=svd_dim, random_state=int(rng.integers(0, 2**31 - 1)))

        Z_train = svd.fit_transform(X[train_ref_idx]).astype(np.float32, copy=False)

        Z = np.empty((X.shape[0], svd_dim), dtype=np.float32)
        Z[train_ref_idx] = Z_train

        all_idx = np.arange(X.shape[0], dtype=np.int64)
        mask = np.ones(X.shape[0], dtype=bool)
        mask[train_ref_idx] = False
        rest = all_idx[mask]

        bs = max(1, int(batch_size))
        for start in range(0, rest.size, bs):
            chunk = rest[start : start + bs]
            Z[chunk] = svd.transform(X[chunk]).astype(np.float32, copy=False)

        X_ref = Z

    # centroids per class using train_ref_idx
    c0_idx = train_ref_idx[y[train_ref_idx] == 0]
    c1_idx = train_ref_idx[y[train_ref_idx] == 1]
    if c0_idx.size == 0 or c1_idx.size == 0:
        raise RuntimeError("Reference train must contain both classes for dist_to_train_within_class.")

    mu0 = X_ref[c0_idx].mean(axis=0).astype(np.float32, copy=False)
    mu1 = X_ref[c1_idx].mean(axis=0).astype(np.float32, copy=False)
    mu0_norm = float(np.dot(mu0, mu0))
    mu1_norm = float(np.dot(mu1, mu1))

    n = int(X_ref.shape[0])
    score = np.empty(n, dtype=np.float32)
    bs = max(1, int(batch_size))

    for start in range(0, n, bs):
        end = min(n, start + bs)
        Xi = X_ref[start:end].astype(np.float32, copy=False)
        yi = y[start:end]

        xnorm = (Xi * Xi).sum(axis=1)  # (b,)

        xdot0 = Xi @ mu0
        xdot1 = Xi @ mu1

        d2_0 = xnorm + mu0_norm - 2.0 * xdot0
        d2_1 = xnorm + mu1_norm - 2.0 * xdot1

        d2_0 = np.maximum(d2_0, 0.0)
        d2_1 = np.maximum(d2_1, 0.0)

        d0 = np.sqrt(d2_0, dtype=np.float32)
        d1 = np.sqrt(d2_1, dtype=np.float32)

        score[start:end] = np.where(yi == 0, d0, d1).astype(np.float32, copy=False)

    return score


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Make EMBER splits:\n"
            " - OOD by score extremes (within-class/global)\n"
            " - OR OOD by dist-to-train within-class (paper-safe: anchor train_ref is the final train)\n"
            " - Train/ID from remaining pool (stratified)\n"
            "Includes runner compatibility flags."
        )
    )

    ap.add_argument("--in-dir", type=str, default="data/processed/ember")
    ap.add_argument("--out-dir", type=str, default="data/processed/ember/splits_sparsity")

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--id-test-frac", type=float, default=0.15)
    ap.add_argument("--ood-test-frac", type=float, default=0.15)

    # Runner modes (compat)
    ap.add_argument("--ood-mode", type=str, default="score_extremes_within_class")
    ap.add_argument("--strict-fracs", action="store_true", help="(compat) Alias of --strict-ood-frac.")

    # Score-extremes params
    ap.add_argument("--ood-extreme-frac-each-side", type=float, default=0.075)

    # dist_to_train params (compat)
    ap.add_argument("--svd-dim", type=int, default=0, help="(dist_to_train*) TruncatedSVD dim, 0 disables.")
    ap.add_argument(
        "--save-provisional",
        action="store_true",
        help="(compat) Save provisional artifacts used to compute dist_to_train scores.",
    )

    ap.add_argument("--max-allowed-posrate-drift", type=float, default=0.02)

    ap.add_argument("--feature-names", type=str, default="feature_names.json")

    ap.add_argument("--use-hist", action="store_true")
    ap.add_argument("--hist-prefix", type=str, default="hist_")
    ap.add_argument("--hist-bins", type=int, default=256)

    ap.add_argument("--use-byteent", action="store_true")
    ap.add_argument("--byteent-prefix", type=str, default="byteent_")
    ap.add_argument("--byteent-bins", type=int, default=256)

    ap.add_argument("--score-mode", type=str, default="nnz", choices=["nnz", "zeros"])
    ap.add_argument("--eps", type=float, default=0.0)

    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--batch-size", type=int, default=20000)

    ap.add_argument("--max-cliffs-pairs", type=int, default=2000000)
    ap.add_argument("--save-id-pool", action="store_true")
    ap.add_argument("--strict-ood-frac", action="store_true")
    ap.add_argument("--max-cliffs-pairs-cap", type=int, default=5000000)

    args = ap.parse_args()

    if bool(args.strict_fracs):
        args.strict_ood_frac = True

    # normalize aliases for runner strings
    ood_mode_raw = str(args.ood_mode).strip()
    aliases = {
        # score extremes
        "within_class": "score_extremes_within_class",
        "score_extremes_within_class": "score_extremes_within_class",
        "score_extremes_global": "score_extremes_global",
        "global": "score_extremes_global",
        "m1": "score_extremes_within_class",
        "m2": "dist_to_train_within_class",
        # dist_to_train
        "dist_to_train_within_class": "dist_to_train_within_class",
        "dist_to_train": "dist_to_train_within_class",
    }
    ood_mode = aliases.get(ood_mode_raw, ood_mode_raw)

    supported = {"score_extremes_within_class", "score_extremes_global", "dist_to_train_within_class"}
    if ood_mode not in supported:
        raise RuntimeError(f"Unsupported --ood-mode '{args.ood_mode}'. Supported: {sorted(list(supported))}")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_path = in_dir / "X.npy"
    y_path = in_dir / "y.npy"
    feat_names_path = in_dir / args.feature_names

    if not X_path.exists():
        raise FileNotFoundError(f"Missing X file: {X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing y file: {y_path}")

    total_frac = float(args.train_frac + args.id_test_frac + args.ood_test_frac)
    if abs(total_frac - 1.0) > 1e-9:
        raise RuntimeError(f"Fractions must sum to 1.0. Got {total_frac}")

    X = np.load(X_path, mmap_mode="r" if bool(args.mmap) else None)
    y = np.load(y_path).astype(np.int64).ravel()

    n = int(X.shape[0])
    if y.shape[0] != n:
        raise RuntimeError(f"Shape mismatch: X has n={n} rows but y has {y.shape[0]} entries")

    uniq = set(np.unique(y).tolist())
    if not uniq.issubset({0, 1}):
        raise RuntimeError(f"y has unexpected labels: {sorted(list(uniq))}. Expected subset of {{0,1}}")

    # Targets (paper-safe, sum exactly to n)
    n_train_target = int(round(float(args.train_frac) * n))
    n_id_target = int(round(float(args.id_test_frac) * n))
    n_ood_target = int(n - n_train_target - n_id_target)
    if n_train_target <= 0 or n_id_target <= 0 or n_ood_target <= 0:
        raise RuntimeError(
            f"Invalid targets: n_train={n_train_target}, n_id={n_id_target}, n_ood={n_ood_target} (n={n})."
        )

    # feature names only needed for nnz/zeros score-extremes modes
    names: Optional[List[str]] = None
    cols_idx: Optional[np.ndarray] = None
    blocks_used: List[Dict[str, object]] = []

    if ood_mode in {"score_extremes_within_class", "score_extremes_global"}:
        if not args.use_hist and not args.use_byteent:
            raise RuntimeError("Select at least one score block: --use-hist and/or --use-byteent")

        names = _load_feature_names(feat_names_path)
        if len(names) != int(X.shape[1]):
            raise RuntimeError(f"feature_names has {len(names)} but X has {X.shape[1]} columns")

        cols: List[int] = []
        if bool(args.use_hist):
            hist_idx = _columns_indices_by_prefix(names, prefix=str(args.hist_prefix), n_bins=int(args.hist_bins))
            cols.extend(hist_idx.tolist())
            blocks_used.append(
                {"block": "hist", "prefix": str(args.hist_prefix), "bins": int(args.hist_bins), "n_cols": int(hist_idx.size)}
            )
        if bool(args.use_byteent):
            be_idx = _columns_indices_by_prefix(names, prefix=str(args.byteent_prefix), n_bins=int(args.byteent_bins))
            cols.extend(be_idx.tolist())
            blocks_used.append(
                {"block": "byteent", "prefix": str(args.byteent_prefix), "bins": int(args.byteent_bins), "n_cols": int(be_idx.size)}
            )

        cols_idx_raw = np.array(cols, dtype=np.int64)
        cols_idx = np.unique(cols_idx_raw)
        if cols_idx.size != cols_idx_raw.size:
            raise RuntimeError("Duplicate columns detected across selected blocks (overlap).")

    rng_split = _rng(int(args.seed))
    rng_eff = _rng(int(args.seed) + 99991)

    expected_ood = 2.0 * float(args.ood_extreme_frac_each_side)

    if ood_mode in {"score_extremes_within_class", "score_extremes_global"}:
        if abs(expected_ood - float(args.ood_test_frac)) > 1e-9 and not bool(args.strict_ood_frac):
            print(
                "[WARN] ood_test_frac does not match 2*ood_extreme_frac_each_side.\n"
                f"       ood_test_frac={float(args.ood_test_frac):.6f}, "
                f"2*ood_extreme_frac_each_side={expected_ood:.6f}",
                file=sys.stderr,
            )
        if bool(args.strict_ood_frac):
            if abs(expected_ood - float(args.ood_test_frac)) > 1e-9:
                raise RuntimeError(
                    f"Expected ood_test_frac == 2*ood_extreme_frac_each_side == {expected_ood:.6f}, "
                    f"but got ood_test_frac={float(args.ood_test_frac):.6f}."
                )

    # ----------------------------
    # Build OOD + train/id pool
    # ----------------------------
    ood_counts_by_class: Optional[Dict[str, Dict[str, int]]] = None

    if ood_mode in {"score_extremes_within_class", "score_extremes_global"}:
        assert cols_idx is not None
        score = _compute_score_dense_feature_block(
            X=X,
            cols_idx=cols_idx,
            mode=str(args.score_mode),
            eps=float(args.eps),
            batch_size=int(args.batch_size),
        ).astype(np.float32)

        if ood_mode == "score_extremes_within_class":
            ood_idx, low_ood, high_ood, thr_by_class = _make_ood_extremes_within_class(
                score=score,
                y=y,
                frac_each_side=float(args.ood_extreme_frac_each_side),
                rng=rng_split,
            )
        else:
            ood_idx, low_ood, high_ood, thr_by_class = _make_ood_extremes_global(
                score=score,
                y=y,
                frac_each_side=float(args.ood_extreme_frac_each_side),
                rng=rng_split,
            )

        # ID pool is the rest (after removing OOD)
        mask = np.ones(n, dtype=bool)
        mask[ood_idx] = False
        id_pool = np.where(mask)[0]

        # Train/ID-test from ID pool (stratified) using fractions (legacy behavior)
        id_pool_frac = id_pool.size / n
        train_frac_within_id = float(args.train_frac) / float(id_pool_frac)
        if not (0.0 < train_frac_within_id < 1.0):
            raise RuntimeError(
                f"Invalid train_frac_within_id={train_frac_within_id:.4f}. "
                f"Check train_frac={args.train_frac} and OOD size={ood_idx.size}/{n}."
            )
        train_idx, id_test_idx = _stratified_split_two_way(id_pool, y, train_frac_within_id, rng=rng_split)

    else:
        # ----------------------------
        # dist_to_train_within_class (M2) - PAPER-SAFE
        # ----------------------------
        all_idx = np.arange(n, dtype=np.int64)

        # 1) Fix final TRAIN FIRST (paper-safe): train_ref is the final train
        train_ref, rest = _stratified_split_n(all_idx, y, n_train_target, rng=rng_split)

        # 2) Compute dist-to-train score (centroids/SVD fitted on train_ref only)
        score = _compute_dist_to_train_score_within_class(
            X=X,
            y=y,
            train_ref_idx=train_ref,
            svd_dim=int(args.svd_dim),
            rng=rng_split,
            batch_size=int(args.batch_size),
        )

        if bool(args.save_provisional):
            _save_npy(out_dir / "provisional_train_ref_idx.npy", train_ref)
            _save_npy(out_dir / "provisional_rest_idx.npy", rest)

        # 3) Choose OOD from REST ONLY (so OOD never overlaps with train_ref)
        ood_idx, low_ood, high_ood, thr_by_class, ood_counts_by_class = _make_ood_extremes_within_class_counts(
            score=score,
            y=y,
            candidate_idx=rest,
            n_ood_total=n_ood_target,
            rng=rng_split,
        )
        blocks_used = [{"block": "dist_to_train_within_class", "svd_dim": int(args.svd_dim)}]

        _assert_disjoint(train_ref, ood_idx, "train_ref", "ood_test")

        # 4) ID-test from the remaining pool (REST \ OOD), ensuring it cannot contain train_ref
        rest_mask = np.ones(rest.size, dtype=bool)
        # map to a set for quick filtering
        ood_set = set(ood_idx.tolist())
        remaining = np.array([ix for ix in rest.tolist() if ix not in ood_set], dtype=np.int64)
        if remaining.size <= 0:
            raise RuntimeError("No remaining samples after selecting OOD from rest (check targets / data size).")

        id_test_idx, _unused = _stratified_split_n(remaining, y, n_id_target, rng=rng_split)

        train_idx = train_ref  # final, fixed
        id_pool = remaining    # (optional) for diagnostics / effects

    # Disjointness (final)
    _assert_disjoint(train_idx, id_test_idx, "train", "id_test")
    _assert_disjoint(train_idx, ood_idx, "train", "ood_test")
    _assert_disjoint(id_test_idx, ood_idx, "id_test", "ood_test")

    # Prevalence drift check
    pr_total = float((y == 1).mean())
    pr_train = _pos_rate(y, train_idx)
    pr_id = _pos_rate(y, id_test_idx)
    pr_ood = _pos_rate(y, ood_idx)

    max_drift = float(args.max_allowed_posrate_drift)

    def _drift_ok(p: float) -> bool:
        return abs(p - pr_total) <= max_drift

    if not (_drift_ok(pr_train) and _drift_ok(pr_id) and _drift_ok(pr_ood)):
        raise RuntimeError(
            "Positive-rate drift too large between splits.\n"
            f"total={pr_total:.6f}, train={pr_train:.6f}, id_test={pr_id:.6f}, ood_test={pr_ood:.6f}\n"
            f"allowed_abs_drift={max_drift:.6f}\n"
            "Tip: reduce --ood-test-frac / --ood-extreme-frac-each-side, use within-class, or adjust score."
        )

    # Save indices
    train_hash = _save_npy(out_dir / "train_idx.npy", train_idx)
    id_hash = _save_npy(out_dir / "id_test_idx.npy", id_test_idx)
    ood_hash = _save_npy(out_dir / "ood_test_idx.npy", ood_idx)
    ood_low_hash = _save_npy(out_dir / "ood_low_idx.npy", low_ood)
    ood_high_hash = _save_npy(out_dir / "ood_high_idx.npy", high_ood)

    id_pool_hash: Optional[str] = None
    if bool(args.save_id_pool):
        id_pool_hash = _save_npy(out_dir / "id_pool_idx.npy", id_pool)

    # Diagnostics / effect sizes
    score_id_pool = score[id_pool]
    score_ood = score[ood_idx]
    score_low = score[low_ood]
    score_high = score[high_ood]

    max_pairs = min(int(args.max_cliffs_pairs), int(args.max_cliffs_pairs_cap))

    effects_overall = {
        "median_diff_ood_vs_idpool": _median_diff(score_ood, score_id_pool),
        "median_diff_low_vs_idpool": _median_diff(score_low, score_id_pool),
        "median_diff_high_vs_idpool": _median_diff(score_high, score_id_pool),
        "cliffs_delta_ood_vs_idpool": _cliffs_delta(score_ood, score_id_pool, max_pairs, rng_eff),
        "cliffs_delta_low_vs_idpool": _cliffs_delta(score_low, score_id_pool, max_pairs, rng_eff),
        "cliffs_delta_high_vs_idpool": _cliffs_delta(score_high, score_id_pool, max_pairs, rng_eff),
    }

    effects_by_class: Dict[str, Dict[str, float]] = {}
    for cls in (0, 1):
        id_cls = id_pool[y[id_pool] == cls]
        ood_cls = ood_idx[y[ood_idx] == cls]
        low_cls = low_ood[y[low_ood] == cls]
        high_cls = high_ood[y[high_ood] == cls]

        sid = score[id_cls]
        sood = score[ood_cls]
        slow = score[low_cls]
        shigh = score[high_cls]

        effects_by_class[f"class_{cls}"] = {
            "median_diff_ood_vs_idpool": _median_diff(sood, sid),
            "median_diff_low_vs_idpool": _median_diff(slow, sid),
            "median_diff_high_vs_idpool": _median_diff(shigh, sid),
            "cliffs_delta_ood_vs_idpool": _cliffs_delta(sood, sid, max_pairs, rng_eff),
            "cliffs_delta_low_vs_idpool": _cliffs_delta(slow, sid, max_pairs, rng_eff),
            "cliffs_delta_high_vs_idpool": _cliffs_delta(shigh, sid, max_pairs, rng_eff),
        }

    ood_low_max_score_global = float(np.max(score[low_ood])) if low_ood.size else float("nan")
    ood_high_min_score_global = float(np.min(score[high_ood])) if high_ood.size else float("nan")

    meta = {
        "seed": int(args.seed),
        "split_type": f"ember_{ood_mode}",
        "fractions_target": {
            "train": float(args.train_frac),
            "id_test": float(args.id_test_frac),
            "ood_test": float(args.ood_test_frac),
        },
        "targets_exact": {
            "n_total": int(n),
            "n_train_target": int(n_train_target),
            "n_id_target": int(n_id_target),
            "n_ood_target": int(n_ood_target),
        },
        "ood_definition": {
            "name": ood_mode,
            "blocks_used": blocks_used,
            "score_mode": str(args.score_mode),
            "eps": float(args.eps),
            "ood_extreme_frac_each_side": float(args.ood_extreme_frac_each_side),
            "ood_total_expected_from_extremes": expected_ood,
            "strict_ood_frac": bool(args.strict_ood_frac),
            "svd_dim": int(args.svd_dim),
            "save_provisional": bool(args.save_provisional),
            "m2_candidate_restriction": ("rest_only" if ood_mode == "dist_to_train_within_class" else None),
            "m2_exact_ood_counts_by_class": ood_counts_by_class,
        },
        "input_fingerprints": {
            "X_path": str(X_path),
            "y_path": str(y_path),
            "feature_names_path": str(feat_names_path),
            "X_sha256": _sha256_file(X_path),
            "y_sha256": _sha256_file(y_path),
            "feature_names_sha256": _sha256_file(feat_names_path) if feat_names_path.exists() else None,
        },
        "output_fingerprints": {
            "train_idx_sha256": train_hash,
            "id_test_idx_sha256": id_hash,
            "ood_test_idx_sha256": ood_hash,
            "ood_low_idx_sha256": ood_low_hash,
            "ood_high_idx_sha256": ood_high_hash,
            **({"id_pool_idx_sha256": id_pool_hash} if id_pool_hash is not None else {}),
        },
        "sizes": {
            "n_total": int(n),
            "n_train": int(train_idx.size),
            "n_id_test": int(id_test_idx.size),
            "n_ood_test": int(ood_idx.size),
            "n_ood_low": int(low_ood.size),
            "n_ood_high": int(high_ood.size),
            "id_pool_size": int(id_pool.size),
        },
        "pos_rates": {
            "total": pr_total,
            "train": pr_train,
            "id_test": pr_id,
            "ood_test": pr_ood,
            "allowed_abs_drift": float(max_drift),
        },
        "score_stats": {
            "global": _score_stats(score, np.arange(n)),
            "train": _score_stats(score, train_idx),
            "id_test": _score_stats(score, id_test_idx),
            "ood_test": _score_stats(score, ood_idx),
            "ood_low": _score_stats(score, low_ood),
            "ood_high": _score_stats(score, high_ood),
            "by_class": {
                "global": _split_stats_by_class(score, y, np.arange(n)),
                "train": _split_stats_by_class(score, y, train_idx),
                "id_test": _split_stats_by_class(score, y, id_test_idx),
                "ood_test": _split_stats_by_class(score, y, ood_idx),
                "ood_low": _split_stats_by_class(score, y, low_ood),
                "ood_high": _split_stats_by_class(score, y, high_ood),
            },
            "ood_thresholds": {
                "global": {"ood_low_max_score": ood_low_max_score_global, "ood_high_min_score": ood_high_min_score_global},
                "by_class": thr_by_class,
            },
        },
        "effects": {"overall": effects_overall, "by_class": effects_by_class},
        "runtime": {"mmap": bool(args.mmap), "batch_size": int(args.batch_size), "max_cliffs_pairs_used": int(max_pairs)},
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[OK] Wrote splits to:", out_dir)
    print(json.dumps(meta["pos_rates"], indent=2))
    print("OOD thresholds (global):", meta["score_stats"]["ood_thresholds"]["global"])
    if ood_mode == "dist_to_train_within_class":
        print("[M2] targets:", meta["targets_exact"])
        if ood_counts_by_class is not None:
            print("[M2] ood counts by class:", json.dumps(ood_counts_by_class, indent=2))


if __name__ == "__main__":
    main()

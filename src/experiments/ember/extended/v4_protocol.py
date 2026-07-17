# src/experiments/ember/extended/v4_protocol.py
"""
v4 protocol utilities (docs/ANALYSIS_SPEC_V4.md): deterministic hash-based
ID-validation/ID-test split, internal C selection by train-only CV, and the
finite-shot fidelity-estimation perturbation model.

All three are pure functions of their inputs plus spec-frozen seeds/salts, so
every run of the v4 recompute is reproducible bit-for-bit.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

ID_SPLIT_SALT = "ksf-v4-idsplit::"       # spec section 6
C_CV_SEED = 20260717
C_GRID_FULL = (0.01, 0.1, 1.0, 10.0, 100.0)
CV_FOLDS = 5
CV_TRAIN_CAP = 2000                       # spec cost amendment (selection only)


# ---------------------------------------------------------------------------
# 1. Hash-based ID-validation / ID-test split (author constraint 5)
# ---------------------------------------------------------------------------
def _stable_hash(global_index: int) -> int:
    h = hashlib.sha256(f"{ID_SPLIT_SALT}{int(global_index)}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def split_id_val_test(id_indices: np.ndarray, y_id: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split the positions 0..n-1 of the ID split into validation/test halves.

    Deterministic and stratified: within each class, positions are ordered by
    the SHA-256 hash of their GLOBAL row index (salted per spec) and assigned
    alternately starting with validation. Returns (val_pos, test_pos, audit).
    """
    id_indices = np.asarray(id_indices).ravel()
    y_id = np.asarray(y_id).ravel()
    assert len(id_indices) == len(y_id)
    val_pos, test_pos = [], []
    for cls in np.unique(y_id):
        pos = np.flatnonzero(y_id == cls)
        order = np.argsort([_stable_hash(g) for g in id_indices[pos]], kind="stable")
        ranked = pos[order]
        val_pos.append(ranked[0::2])
        test_pos.append(ranked[1::2])
    val = np.sort(np.concatenate(val_pos))
    test = np.sort(np.concatenate(test_pos))
    assert len(np.intersect1d(val, test)) == 0
    assert len(val) + len(test) == len(y_id)
    audit = {
        "n_id": int(len(y_id)), "n_val": int(len(val)), "n_test": int(len(test)),
        "pos_rate_val": float(y_id[val].mean()), "pos_rate_test": float(y_id[test].mean()),
        "overlap": 0,
        "class_balance_gap": float(abs(y_id[val].mean() - y_id[test].mean())),
    }
    return val, test, audit


# ---------------------------------------------------------------------------
# 2. Internal C selection by train-only CV (author constraint 4)
# ---------------------------------------------------------------------------
def select_c_by_train_cv(K_tr: np.ndarray, y_tr: np.ndarray,
                         grid: tuple[float, ...] = C_GRID_FULL,
                         folds: int = CV_FOLDS, seed: int = C_CV_SEED,
                         cap: int = CV_TRAIN_CAP) -> tuple[float, dict[float, float]]:
    """Pick C for one kernel configuration using ONLY the training Gram.

    Stratified k-fold over train rows; per fold, fit on the fold-train
    submatrix and score balanced accuracy on the fold-validation blocks.
    Never touches id_val / id_test / ood_test. When n_train > cap, the CV
    operates on a seeded stratified subsample of the train rows (selection
    only; final fits use the full train). Ties break toward smaller C.
    """
    y_tr = np.asarray(y_tr).ravel()
    n = len(y_tr)
    rows = np.arange(n)
    if n > cap:
        rng = np.random.default_rng(seed)
        keep = []
        for cls in np.unique(y_tr):
            pos = np.flatnonzero(y_tr == cls)
            k = max(1, int(round(cap * len(pos) / n)))
            keep.append(rng.choice(pos, size=min(k, len(pos)), replace=False))
        rows = np.sort(np.concatenate(keep))
    Ksub, ysub = K_tr[np.ix_(rows, rows)], y_tr[rows]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: dict[float, list[float]] = {c: [] for c in grid}
    for tr, va in skf.split(Ksub, ysub):
        K_fold = Ksub[np.ix_(tr, tr)]
        K_val = Ksub[np.ix_(va, tr)]
        for c in grid:
            m = SVC(kernel="precomputed", C=c, class_weight="balanced")
            m.fit(K_fold, ysub[tr])
            scores[c].append(balanced_accuracy_score(ysub[va], m.predict(K_val)))
    mean_scores = {c: float(np.mean(v)) for c, v in scores.items()}
    best = min((c for c in grid), key=lambda c: (-mean_scores[c], c))
    return best, mean_scores


# ---------------------------------------------------------------------------
# 3. Finite-shot fidelity-estimation perturbation model (author constraint 6)
# ---------------------------------------------------------------------------
def perturb_kernel_finite_shots(K: np.ndarray, shots: int, rng: np.random.Generator,
                                square: bool) -> tuple[np.ndarray, dict]:
    """Perturb an exact fidelity Gram block as if each entry were estimated
    from `shots` Bernoulli fidelity samples. NOT a hardware simulation: no
    device noise model, no transpilation -- a statistical estimation
    perturbation of the exact statevector kernel.

    Square (train x train) blocks: sample the upper triangle, mirror to keep
    exact symmetry, fix the diagonal at 1, clip to [0,1], then project to the
    PSD cone by eigenvalue clipping. Rectangular (test x train) blocks:
    binomial sampling + clipping only (no symmetrization or PSD projection is
    defined for them). Returns (K_perturbed, audit) where audit carries the
    before/after-projection diagnostics required by the spec.
    """
    Kc = np.clip(np.asarray(K, dtype=np.float64), 0.0, 1.0)
    audit: dict = {"shots": int(shots), "square": bool(square)}
    if square:
        n = Kc.shape[0]
        iu = np.triu_indices(n, k=1)
        est = rng.binomial(shots, Kc[iu]) / shots
        Ks = np.eye(n)
        Ks[iu] = est
        Ks[(iu[1], iu[0])] = est
        Ks = np.clip(Ks, 0.0, 1.0)
        ev = np.linalg.eigvalsh(Ks)
        audit["min_eig_before_psd"] = float(ev.min())
        audit["frac_negative_eig"] = float((ev < -1e-12).mean())
        if ev.min() < -1e-12:
            # PSD projection is the FINAL step (spec constraint 6 order):
            # clipping entries after projecting would reintroduce negative
            # eigenvalues. Residual deviations of the diagonal/range are
            # reported instead of silently re-clipped away.
            w, V = np.linalg.eigh(Ks)
            Kp = (V * np.clip(w, 0.0, None)) @ V.T
            Kp = (Kp + Kp.T) / 2.0
        else:
            Kp = Ks
        audit["min_eig_after_psd"] = float(np.linalg.eigvalsh(Kp).min())
        audit["max_diag_dev_after_psd"] = float(np.abs(np.diag(Kp) - 1.0).max())
        audit["max_range_dev_after_psd"] = float(
            max(0.0, -Kp.min(), Kp.max() - 1.0))
        audit["fro_change_sampling"] = float(np.linalg.norm(Ks - Kc, "fro"))
        audit["fro_change_projection"] = float(np.linalg.norm(Kp - Ks, "fro"))
        return Kp, audit
    est = rng.binomial(shots, Kc) / shots
    Kp = np.clip(est, 0.0, 1.0)
    audit["fro_change_sampling"] = float(np.linalg.norm(Kp - Kc, "fro"))
    audit["fro_change_projection"] = 0.0
    return Kp, audit

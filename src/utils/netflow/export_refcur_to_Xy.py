# src/utils/netflow/export_refcur_to_Xy.py
"""
Export a reference/current CSV pair (network-flow scenario) to the X.npy/y.npy
format consumed by the experiment runners.

Rows are stored reference-first, current-after; the boundary index is recorded
in meta_export.json so split makers can address the two pools by index range.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pool(csv_path: Path, label_col: str, pos_label: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise RuntimeError(f"{csv_path}: missing label column '{label_col}'")
    y = (df[label_col].astype(str).str.upper() == pos_label.upper()).to_numpy(dtype=np.int64)
    feats = df.drop(columns=[label_col])
    non_numeric = [c for c in feats.columns if not np.issubdtype(feats[c].dtype, np.number)]
    if non_numeric:
        raise RuntimeError(f"{csv_path}: non-numeric feature columns {non_numeric}; expected numeric-only pools")
    X = feats.to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).sum())
        raise RuntimeError(f"{csv_path}: {n_bad} non-finite feature values")
    return X, y, list(feats.columns)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export ref/cur netflow CSVs to X.npy / y.npy.")
    ap.add_argument("--ref-csv", type=Path, required=True)
    ap.add_argument("--cur-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--label-col", type=str, default="Label")
    ap.add_argument("--pos-label", type=str, default="ATTACK")
    args = ap.parse_args()

    X_ref, y_ref, cols_ref = load_pool(args.ref_csv, args.label_col, args.pos_label)
    X_cur, y_cur, cols_cur = load_pool(args.cur_csv, args.label_col, args.pos_label)
    if cols_ref != cols_cur:
        raise RuntimeError("Feature columns differ between ref and cur CSVs")

    X = np.concatenate([X_ref, X_cur], axis=0)
    y = np.concatenate([y_ref, y_cur], axis=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "X.npy", X)
    np.save(args.out_dir / "y.npy", y)

    meta = {
        "kind": "netflow_refcur",
        "ref_csv": str(args.ref_csv),
        "cur_csv": str(args.cur_csv),
        "ref_csv_sha256": _sha256_file(args.ref_csv),
        "cur_csv_sha256": _sha256_file(args.cur_csv),
        "n_ref": int(X_ref.shape[0]),
        "n_cur": int(X_cur.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": cols_ref,
        "label_col": args.label_col,
        "pos_label": args.pos_label,
        "class_counts": {
            "ref": {"benign": int((y_ref == 0).sum()), "attack": int((y_ref == 1).sum())},
            "cur": {"benign": int((y_cur == 0).sum()), "attack": int((y_cur == 1).sum())},
        },
    }
    (args.out_dir / "meta_export.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[✓] Exported X{X.shape} y{y.shape} -> {args.out_dir}")


if __name__ == "__main__":
    main()

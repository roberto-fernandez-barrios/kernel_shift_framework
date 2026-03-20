# src/utils/ember/export_ember_jsonl_to_Xy.py
from __future__ import annotations

import argparse
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------------
# Feature schema (fixed order, paper-safe)
# ----------------------------
STR_KEYS = ["numstrings", "avlength", "printables", "entropy", "paths"]
GEN_KEYS = ["size", "vsize", "has_debug", "has_relocations", "has_resources", "has_signature", "has_tls"]

N_HIST = 256
N_BYTEENT = 256

FEATURE_NAMES: List[str] = (
    [f"hist_{i}" for i in range(N_HIST)]
    + [f"byteent_{i}" for i in range(N_BYTEENT)]
    + [f"strings_{k}" for k in STR_KEYS]
    + [f"general_{k}" for k in GEN_KEYS]
)
N_FEATURES = len(FEATURE_NAMES)


def _extract_row(sample: Dict) -> Optional[Tuple[int, np.ndarray]]:
    """
    Returns (label, x_row) or None if label not in {0,1}.
    x_row is float32 vector with fixed schema defined above.
    """
    label = sample.get("label", -1)
    if label not in (0, 1):
        return None

    x = np.zeros((N_FEATURES,), dtype=np.float32)

    # 1) histogram[256]
    hist = sample.get("histogram", None)
    if isinstance(hist, list) and len(hist) == N_HIST:
        x[0:N_HIST] = np.asarray(hist, dtype=np.float32)

    # 2) byteentropy[256]
    be = sample.get("byteentropy", None)
    if isinstance(be, list) and len(be) == N_BYTEENT:
        x[N_HIST:N_HIST + N_BYTEENT] = np.asarray(be, dtype=np.float32)

    # 3) strings stats
    strings = sample.get("strings", {})
    base = N_HIST + N_BYTEENT
    for i, k in enumerate(STR_KEYS):
        v = strings.get(k, 0.0) if isinstance(strings, dict) else 0.0
        if isinstance(v, bool):
            v = float(int(v))
        if not isinstance(v, (int, float)):
            v = 0.0
        x[base + i] = float(v)

    # 4) general stats/flags
    general = sample.get("general", {})
    base2 = base + len(STR_KEYS)
    for i, k in enumerate(GEN_KEYS):
        v = general.get(k, 0.0) if isinstance(general, dict) else 0.0
        if isinstance(v, bool):
            v = float(int(v))
        if not isinstance(v, (int, float)):
            v = 0.0
        x[base2 + i] = float(v)

    return int(label), x


def _iter_jsonl_lines(paths: List[str]):
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield p, line


def main():
    ap = argparse.ArgumentParser(description="Export EMBER train_features_*.jsonl to X.npy/y.npy (no scaling, no subset).")
    ap.add_argument("--ember-dir", type=str, default="data/raw/ember/extracted/ember2018",
                    help="Dir con train_features_*.jsonl")
    ap.add_argument("--pattern", type=str, default="train_features_*.jsonl")
    ap.add_argument("--out-dir", type=str, default="data/processed/ember",
                    help="Output dir para X.npy, y.npy, feature_names.json")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    ap.add_argument("--max-samples", type=int, default=0,
                    help="0 = sin límite. Si >0, corta al llegar a ese número total de muestras (tras filtrar labels).")
    ap.add_argument("--shuffle", action="store_true",
                    help="Baraja al final (requiere cargar todo en memoria). Si no, mantiene el orden de lectura.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict", action="store_true",
                    help="Si está activo, exige que histogram y byteentropy tengan longitud 256; si no, descarta la muestra.")
    args = ap.parse_args()

    ember_dir = Path(args.ember_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_paths = sorted(glob.glob(str(ember_dir / args.pattern)))
    if not jsonl_paths:
        raise FileNotFoundError(f"No encontré ficheros con patrón {args.pattern} en {ember_dir}")

    # Recolección
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    n_read = 0
    n_used = 0
    n_skipped_label = 0
    n_skipped_strict = 0

    for p, line in _iter_jsonl_lines(jsonl_paths):
        n_read += 1
        try:
            sample = json.loads(line)
        except Exception:
            continue

        label = sample.get("label", -1)
        if label not in (0, 1):
            n_skipped_label += 1
            continue

        # strict check if requested
        if args.strict:
            hist = sample.get("histogram", None)
            be = sample.get("byteentropy", None)
            if not (isinstance(hist, list) and len(hist) == 256 and isinstance(be, list) and len(be) == 256):
                n_skipped_strict += 1
                continue

        out = _extract_row(sample)
        if out is None:
            n_skipped_label += 1
            continue

        lab, x = out
        X_rows.append(x)
        y_rows.append(lab)
        n_used += 1

        if args.max_samples and n_used >= int(args.max_samples):
            break

        # log ligero cada cierto tiempo
        if n_used % 50000 == 0:
            print(f"[*] used={n_used} read_lines={n_read} (last_file={os.path.basename(p)})")

    if n_used == 0:
        raise RuntimeError("No se extrajo ninguna muestra (label 0/1). Revisa ruta/formatos.")

    # Stack
    X = np.stack(X_rows, axis=0)
    y = np.asarray(y_rows, dtype=np.int64)

    # optional shuffle at end (to remove any file/order artefacts)
    if args.shuffle:
        rng = np.random.default_rng(int(args.seed))
        perm = rng.permutation(X.shape[0])
        X = X[perm]
        y = y[perm]

    # dtype
    if args.dtype == "float64":
        X = X.astype(np.float64, copy=False)
    else:
        X = X.astype(np.float32, copy=False)

    # Save
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    (out_dir / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8")

    meta = {
        "ember_dir": str(ember_dir),
        "pattern": args.pattern,
        "n_features": int(N_FEATURES),
        "feature_schema": {
            "hist": 256,
            "byteentropy": 256,
            "strings": STR_KEYS,
            "general": GEN_KEYS,
        },
        "read_lines": int(n_read),
        "used": int(n_used),
        "skipped_label": int(n_skipped_label),
        "skipped_strict": int(n_skipped_strict),
        "pos": int((y == 1).sum()),
        "neg": int((y == 0).sum()),
        "dtype": str(X.dtype),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "max_samples": int(args.max_samples),
        "strict": bool(args.strict),
        "note": "NO scaling here. Fit any scaler/SVD only on TRAIN inside experiment runners.",
    }
    (out_dir / "meta_export.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[OK] Exported:")
    print(f"  X: {out_dir / 'X.npy'} shape={X.shape} dtype={X.dtype}")
    print(f"  y: {out_dir / 'y.npy'} shape={y.shape} pos={(y==1).sum()} neg={(y==0).sum()}")
    print(f"  names: {out_dir / 'feature_names.json'} n_features={len(FEATURE_NAMES)}")
    print(f"  meta: {out_dir / 'meta_export.json'}")


if __name__ == "__main__":
    main()

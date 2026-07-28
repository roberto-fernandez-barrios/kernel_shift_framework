"""Export frozen, label-blind TableShift v5 subsamples.

Run this script inside the official TableShift container while mounting the
pinned TableShift source at the front of PYTHONPATH.  TableShift establishes
the published domain splits; this exporter bypasses its learned feature
preprocessing and writes only the maximum frozen subsample for each seed.
The q1000 stratum is the prefix of q2000 and is therefore nested by design.

Example (paths are illustrative):

  python /workspace/scripts/data/export_tableshift_v5.py \
      --task acsincome \
      --cache-dir /workspace/data/raw/tableshift/v5/cache/acsincome \
      --export-dir /workspace/data/raw/tableshift/v5/exports/acsincome \
      --audit-dir /workspace/results/v5/audit
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 123, 999, 7, 2024)
SPLIT_LIMITS = {
    "train": {"q1000": 1000, "q2000": 2000},
    "validation": {"q1000": 250, "q2000": 500},
    "id_test": {"q1000": 250, "q2000": 500},
    "ood_test": {"q1000": 500, "q2000": 1000},
}
SOURCE_COMMIT = "fca9429814703a07e3902d005d46563a207b7f0a"


def row_digest(task: str, split: str, seed: int, source_position: int) -> str:
    token = f"ksf-v5::{task}::{split}::{seed}::{source_position}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hashed_order(
    task: str, split: str, seed: int, source_positions: np.ndarray
) -> np.ndarray:
    """Return split-local positions ordered without consulting outcomes."""
    positions = np.asarray(source_positions)
    if positions.ndim != 1 or len(np.unique(positions)) != len(positions):
        raise ValueError("source positions must be a one-dimensional unique array")
    keys = np.asarray(
        [row_digest(task, split, seed, int(pos)) for pos in positions],
        dtype="U64",
    )
    return np.argsort(keys, kind="stable")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_digest(values: np.ndarray) -> str:
    payload = "\n".join(map(str, np.asarray(values).tolist())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_source_splits(split_positions: dict[str, np.ndarray]) -> None:
    names = tuple(SPLIT_LIMITS)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = np.intersect1d(
                split_positions[left], split_positions[right], assume_unique=False
            )
            if len(overlap):
                raise RuntimeError(
                    f"TableShift source splits {left}/{right} overlap "
                    f"at {len(overlap)} positions"
                )


def feature_role(dtype) -> str:
    """Preserve TableShift's semantic dtype across the CSV interchange."""
    if (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    ):
        return "categorical"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    raise TypeError(f"unsupported TableShift feature dtype: {dtype}")


def export_task(
    task: str,
    cache_dir: Path,
    export_dir: Path,
    audit_dir: Path,
) -> None:
    # Deferred imports keep the deterministic sampler testable without the
    # heavyweight TableShift environment.
    from tableshift import get_dataset
    from tableshift.core.features import PreprocessorConfig

    # TableShift's pinned CollegeScorecardDataSource unconditionally invokes
    # the Kaggle CLI, even when its public CC0 archive has already been placed
    # in the exact expected cache path.  In that one case, suppress only the
    # redundant download call; parsing, schema, target construction, and
    # published splitting remain the pinned TableShift implementation.
    if task == "college_scorecard":
        expected = cache_dir / "kaggle" / "college-scorecard" / "Scorecard.csv"
        if expected.exists():
            from tableshift.core.data_source import CollegeScorecardDataSource
            CollegeScorecardDataSource._download_if_not_cached = lambda self: None

    config = PreprocessorConfig(
        categorical_features="passthrough",
        numeric_features="passthrough",
        passthrough_columns="all",
        dropna=None,
        sub_illegal_chars=False,
    )
    dataset = get_dataset(
        task,
        cache_dir=str(cache_dir),
        preprocessor_config=config,
        initialize_data=True,
        use_cached=False,
    )

    available = set(dataset.splits)
    missing = set(SPLIT_LIMITS) - available
    if missing:
        raise RuntimeError(f"TableShift task {task} lacks splits {sorted(missing)}")

    split_positions = {
        split: np.asarray(dataset.splits[split], dtype=np.int64)
        for split in SPLIT_LIMITS
    }
    validate_source_splits(split_positions)
    audit_rows: list[dict] = []
    schema_rows: list[dict] | None = None

    for split, limits in SPLIT_LIMITS.items():
        X, y, _, _ = dataset.get_pandas(split)
        current_schema = [
            {"task": task, "column": column, "dtype": str(dtype),
             "role": feature_role(dtype)}
            for column, dtype in X.dtypes.items()
        ]
        if schema_rows is None:
            schema_rows = current_schema
        elif current_schema != schema_rows:
            raise RuntimeError(f"feature schema changes across splits for {task}")
        X = X.reset_index(drop=True)
        y = pd.Series(np.asarray(y).ravel()).reset_index(drop=True)
        source_positions = split_positions[split]
        if len(X) != len(y) or len(X) != len(source_positions):
            raise RuntimeError(f"misaligned X/y/source positions for {task}/{split}")
        if len(X) < limits["q2000"]:
            raise RuntimeError(
                f"{task}/{split} has {len(X)} rows; needs {limits['q2000']}"
            )
        labels = set(pd.unique(y.dropna()))
        if not labels.issubset({0, 1, 0.0, 1.0}) or len(labels) != 2:
            raise RuntimeError(
                f"{task}/{split} is not binary with both labels present: {labels}"
            )

        for seed in SEEDS:
            order = hashed_order(task, split, seed, source_positions)
            selected_local = order[:limits["q2000"]]
            selected_positions = source_positions[selected_local]
            selected_y = y.iloc[selected_local].astype(np.int8).reset_index(drop=True)
            selected_X = X.iloc[selected_local].reset_index(drop=True)

            out = selected_X.copy()
            for reserved in ("__target__", "__source_position__", "__sample_rank__"):
                if reserved in out:
                    raise RuntimeError(f"reserved export column already exists: {reserved}")
            out["__target__"] = selected_y
            out["__source_position__"] = selected_positions
            out["__sample_rank__"] = np.arange(len(out), dtype=np.int64)
            seed_dir = export_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            out.to_csv(seed_dir / f"{split}.csv", index=False)

            for stratum, n_rows in limits.items():
                y_prefix = selected_y.iloc[:n_rows]
                pos_prefix = selected_positions[:n_rows]
                counts = y_prefix.value_counts().to_dict()
                if len(counts) != 2:
                    raise RuntimeError(
                        f"class-presence gate failed: {task}/{split}/{seed}/{stratum}"
                    )
                audit_rows.append(
                    {
                        "task": task,
                        "split": split,
                        "seed": seed,
                        "stratum": stratum,
                        "source_n": len(X),
                        "sample_n": n_rows,
                        "class_0_n": int(counts.get(0, 0)),
                        "class_1_n": int(counts.get(1, 0)),
                        "positions_sha256": sequence_digest(pos_prefix),
                        "first_rank_digest": row_digest(
                            task, split, seed, int(pos_prefix[0])
                        ),
                        "tableshift_commit": SOURCE_COMMIT,
                        "ood_validation_accessed": False,
                    }
                )

    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).sort_values(
        ["task", "split", "seed", "stratum"]
    ).to_csv(audit_dir / f"tableshift_sampling_{task}.csv", index=False)
    pd.DataFrame(schema_rows).to_csv(
        audit_dir / f"tableshift_schema_{task}.csv", index=False
    )

    source_rows = []
    for path in sorted(p for p in cache_dir.rglob("*") if p.is_file()):
        source_rows.append(
            {
                "task": task,
                "relative_path": path.relative_to(cache_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "tableshift_commit": SOURCE_COMMIT,
            }
        )
    if not source_rows:
        raise RuntimeError(f"no cached source files found below {cache_dir}")
    pd.DataFrame(source_rows).to_csv(
        audit_dir / f"tableshift_source_files_{task}.csv", index=False
    )
    print(
        f"[OK] exported {task}: {len(audit_rows)} audited sample prefixes, "
        f"{len(source_rows)} source files"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    export_task(args.task, args.cache_dir, args.export_dir, args.audit_dir)


if __name__ == "__main__":
    main()

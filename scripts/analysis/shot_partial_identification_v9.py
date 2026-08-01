"""Quantify finite-shot changes in target-domain quantum non-emulability."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.partial_identification import (  # noqa: E402
    realized_accuracy_advantage,
    realized_balanced_accuracy_advantage,
    sharp_accuracy_envelope,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantum_stratum(kernel: str) -> str:
    return "entangling_zz" if kernel.startswith("zz_") else "product_map"


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def case_rows(exact_case: Path, shot_case: Path) -> tuple[pd.DataFrame, list[dict]]:
    exact_path = exact_case / "predictions_svc.npz"
    label_path = exact_case / "evaluation_labels.npz"
    shot_path = shot_case / "shot_predictions_svc.npz"
    metadata_path = shot_case / "shot_metadata_svc.json"
    with np.load(exact_path, allow_pickle=False) as archive:
        quantum_exact = archive["quantum_prediction"]
        classical = archive["classical_predictions"]
        cfgs = archive["classical_cfgs"].astype(str)
        customary = archive["customary_mask"].astype(bool)
    with np.load(label_path, allow_pickle=False) as archive:
        labels = archive["target_labels"]
    with np.load(shot_path, allow_pickle=False) as archive:
        shots = archive["shots"].astype(int)
        conditions = archive["projection_conditions"].astype(str)
        np.testing.assert_array_equal(
            quantum_exact, archive["exact_quantum_prediction"]
        )
        quantum_shot = archive["shot_quantum_predictions"]
        ood_baccs = archive["ood_baccs"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    tiers = {
        "customary_30": np.flatnonzero(customary),
        "full_115": np.arange(classical.shape[1]),
    }
    exact: dict[str, dict[str, Any]] = {}
    for tier, indices in tiers.items():
        reference = classical[:, indices]
        envelope = sharp_accuracy_envelope(quantum_exact, reference)
        exact[tier] = {
            "indices": indices,
            "upper": envelope.upper,
            "nearest": int(indices[envelope.nearest_indices[0]]),
            "accuracy": realized_accuracy_advantage(labels, quantum_exact, reference),
            "bacc": realized_balanced_accuracy_advantage(labels, quantum_exact, reference),
        }

    rows: list[dict[str, Any]] = []
    for shot_i, shot_count in enumerate(shots):
        for replicate in range(quantum_shot.shape[1]):
            for condition_i, condition in enumerate(conditions):
                prediction = quantum_shot[shot_i, replicate, condition_i]
                self_disagreement = float(np.mean(prediction != quantum_exact))
                for tier, baseline in exact.items():
                    indices = baseline["indices"]
                    reference = classical[:, indices]
                    envelope = sharp_accuracy_envelope(prediction, reference)
                    nearest = int(indices[envelope.nearest_indices[0]])
                    realized_accuracy = realized_accuracy_advantage(
                        labels, prediction, reference
                    )
                    realized_bacc = realized_balanced_accuracy_advantage(
                        labels, prediction, reference
                    )
                    rows.append(
                        {
                            "status": "post-hoc exploratory finite-shot extension",
                            "case": exact_case.name,
                            "run": metadata["run"],
                            "group": metadata["group"],
                            "quantum_kernel": metadata["kernel"],
                            "quantum_stratum": quantum_stratum(metadata["kernel"]),
                            "tier": tier,
                            "shots": int(shot_count),
                            "replicate": replicate,
                            "projection_condition": condition,
                            "self_disagreement_from_exact": self_disagreement,
                            "exact_accuracy_upper": baseline["upper"],
                            "shot_accuracy_upper": envelope.upper,
                            "accuracy_upper_change": envelope.upper - baseline["upper"],
                            "exact_nearest_cfg": str(cfgs[baseline["nearest"]]),
                            "shot_nearest_cfg": str(cfgs[nearest]),
                            "witness_identity_stable": bool(nearest == baseline["nearest"]),
                            "exact_realized_accuracy_advantage": baseline["accuracy"],
                            "shot_realized_accuracy_advantage": realized_accuracy,
                            "realized_accuracy_advantage_change": realized_accuracy
                            - baseline["accuracy"],
                            "exact_realized_bacc_advantage": baseline["bacc"],
                            "shot_realized_bacc_advantage": realized_bacc,
                            "realized_bacc_advantage_change": realized_bacc
                            - baseline["bacc"],
                            "shot_quantum_ood_bacc": float(
                                ood_baccs[shot_i, replicate, condition_i]
                            ),
                        }
                    )
    hashes = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in (exact_path, label_path, shot_path, metadata_path)
    ]
    return pd.DataFrame(rows), hashes


def summarize_by_group(rows: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "case",
        "run",
        "group",
        "quantum_kernel",
        "quantum_stratum",
        "tier",
        "shots",
        "projection_condition",
    ]
    metrics = [
        "self_disagreement_from_exact",
        "exact_accuracy_upper",
        "shot_accuracy_upper",
        "accuracy_upper_change",
        "realized_accuracy_advantage_change",
        "realized_bacc_advantage_change",
        "shot_quantum_ood_bacc",
    ]
    output: list[dict[str, Any]] = []
    for key, group in rows.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        record["n_replicates"] = len(group)
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            record[f"median_{metric}"] = float(np.median(values))
            record[f"q025_{metric}"] = float(np.quantile(values, 0.025))
            record[f"q975_{metric}"] = float(np.quantile(values, 0.975))
        record["witness_identity_stability"] = float(
            group.witness_identity_stable.mean()
        )
        output.append(record)
    return pd.DataFrame(output)


def summarize_across_groups(group_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["tier", "shots", "projection_condition"]
    metrics = [
        "median_self_disagreement_from_exact",
        "median_shot_accuracy_upper",
        "median_accuracy_upper_change",
        "median_realized_accuracy_advantage_change",
        "median_realized_bacc_advantage_change",
    ]
    output: list[dict[str, Any]] = []
    for key, group in group_summary.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        record["n_fixed_cases"] = len(group)
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            record[f"across_case_median_{metric}"] = float(np.median(values))
            record[f"across_case_min_{metric}"] = float(np.min(values))
            record[f"across_case_max_{metric}"] = float(np.max(values))
        output.append(record)
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exact-root",
        type=Path,
        default=Path("results/v9/partial_identification/predictions"),
    )
    parser.add_argument(
        "--shot-root",
        type=Path,
        default=Path("results/v9/partial_identification/shot_predictions"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/v9/partial_identification/shot_analysis"),
    )
    args = parser.parse_args()
    exact_cases = sorted(path for path in args.exact_root.iterdir() if path.is_dir())
    shot_cases = sorted(path for path in args.shot_root.iterdir() if path.is_dir())
    if len(exact_cases) != 8 or len(shot_cases) != 8:
        raise RuntimeError(
            f"expected eight exact and shot cases, found {len(exact_cases)} and {len(shot_cases)}"
        )
    if [path.name for path in exact_cases] != [path.name for path in shot_cases]:
        raise RuntimeError("exact and shot case names differ")

    frames: list[pd.DataFrame] = []
    input_hashes: list[dict] = []
    for exact_case, shot_case in zip(exact_cases, shot_cases):
        frame, hashes = case_rows(exact_case, shot_case)
        frames.append(frame)
        input_hashes.extend(hashes)
    rows = pd.concat(frames, ignore_index=True)
    groups = summarize_by_group(rows)
    across = summarize_across_groups(groups)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "replicates": args.out_dir / "shot_emulation_replicates.csv",
        "groups": args.out_dir / "shot_emulation_by_group.csv",
        "across": args.out_dir / "shot_emulation_across_groups.csv",
    }
    write_csv(rows, paths["replicates"])
    write_csv(groups, paths["groups"])
    write_csv(across, paths["across"])
    manifest_path = args.out_dir / "shot_analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "specification": "docs/PARTIAL_IDENTIFICATION_SPEC_V9.md",
                "status": "post-hoc exploratory finite-shot extension",
                "n_fixed_cases": 8,
                "n_replicate_rows": len(rows),
                "input_hashes": input_hashes,
                "outputs": {str(path): sha256_file(path) for path in paths.values()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(across.to_string(index=False))
    print(f"[ok] wrote finite-shot partial-identification analysis under {args.out_dir}")


if __name__ == "__main__":
    main()

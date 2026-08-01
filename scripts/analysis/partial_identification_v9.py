"""Analyze the frozen v0.9 target-domain partial-identification pilot.

Two deliberately separate modes prevent target labels from entering the
primary certificate:

``label-free``
    reads only prediction artifacts and writes sharp accuracy envelopes plus
    frozen block-budget frontier draws;

``unlock-evaluation``
    requires an already-complete label-free artifact, then reads the separate
    evaluation-label file to verify realized containment and solve the exact
    prevalence-conditional BAcc MILPs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.partial_identification import (  # noqa: E402
    finite_family_population_correction,
    realized_accuracy_advantage,
    realized_balanced_accuracy_advantage,
    sharp_accuracy_envelope,
    sharp_balanced_accuracy_envelope,
)


SPEC_PATH = Path("docs/PARTIAL_IDENTIFICATION_SPEC_V9.md")
TAUS = (0.005, 0.010, 0.020)
FRONTIER_BUDGETS = (30, 60, 115)
N_FRONTIER_PERMUTATIONS = 5_000
FRONTIER_SEED_ROOT = "ksf-v9-frontier-20260731"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rng(*tokens: str) -> np.random.Generator:
    label = "::".join((FRONTIER_SEED_ROOT, *tokens))
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(seed)


def load_prediction_artifact(
    case_dir: Path,
    model: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    prediction_path = case_dir / f"predictions_{model}.npz"
    metadata_path = case_dir / f"metadata_{model}.json"
    if not prediction_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"incomplete {model} artifact under {case_dir}")
    with np.load(prediction_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["prediction_artifact_sha256"] != sha256_file(prediction_path):
        raise RuntimeError(f"prediction hash mismatch for {prediction_path}")
    required = {
        "target_indices",
        "quantum_prediction",
        "classical_predictions",
        "classical_cfgs",
        "classical_kernels",
        "classical_dims",
        "customary_mask",
    }
    missing = required.difference(arrays)
    if missing:
        raise RuntimeError(f"{prediction_path} is missing arrays {sorted(missing)}")
    n_target = len(arrays["quantum_prediction"])
    if arrays["classical_predictions"].shape != (n_target, 115):
        raise RuntimeError(
            f"unexpected classical prediction shape {arrays['classical_predictions'].shape}"
        )
    if int(np.sum(arrays["customary_mask"])) != 30:
        raise RuntimeError("customary mask must contain exactly 30 candidates")
    return arrays, metadata


def _tier_indices(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "customary_30": np.flatnonzero(arrays["customary_mask"]),
        "full_115": np.arange(arrays["classical_predictions"].shape[1]),
    }


def envelope_row(
    case_dir: Path,
    model: str,
    tier: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    indices: np.ndarray,
) -> dict[str, Any]:
    classical = arrays["classical_predictions"][:, indices]
    envelope = sharp_accuracy_envelope(arrays["quantum_prediction"], classical)
    nearest_local = envelope.nearest_indices[0]
    farthest_local = envelope.farthest_indices[0]
    nearest = int(indices[nearest_local])
    farthest = int(indices[farthest_local])
    row: dict[str, Any] = {
        "status": "post-hoc exploratory",
        "case": case_dir.name,
        "run": metadata["run"],
        "group": metadata["group"],
        "model": model,
        "tier": tier,
        "n_target": envelope.n_target,
        "n_classical": envelope.n_classical,
        "accuracy_lower": envelope.lower,
        "accuracy_upper": envelope.upper,
        "disagreement_min": envelope.disagreement_min,
        "disagreement_max": envelope.disagreement_max,
        "n_nearest_ties": len(envelope.nearest_indices),
        "n_farthest_ties": len(envelope.farthest_indices),
        "nearest_cfg": str(arrays["classical_cfgs"][nearest]),
        "nearest_kernel": str(arrays["classical_kernels"][nearest]),
        "nearest_dim": int(arrays["classical_dims"][nearest]),
        "farthest_cfg": str(arrays["classical_cfgs"][farthest]),
        "farthest_kernel": str(arrays["classical_kernels"][farthest]),
        "farthest_dim": int(arrays["classical_dims"][farthest]),
        "quantum_cfg": metadata["quantum_winner"]["cfg"],
        "quantum_kernel": metadata["quantum_winner"]["kernel"],
        "quantum_dim": metadata["quantum_winner"]["dim"],
        "population_correction_95_theoretical_only": (
            finite_family_population_correction(envelope.n_classical, envelope.n_target)
        ),
    }
    for tau in TAUS:
        token = f"falsifies_{tau:.3f}".replace(".", "p")
        row[token] = bool(envelope.upper < tau)
    return row


def block_frontier_draws(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    case: str,
    model: str,
) -> pd.DataFrame:
    """Frozen nested whole-block budget sensitivity for one case/model."""
    quantum = arrays["quantum_prediction"]
    classical = arrays["classical_predictions"]
    disagreement = np.mean(classical != quantum[:, None], axis=0)
    kernels = arrays["classical_kernels"].astype(str)
    unique_blocks = np.unique(kernels)
    if len(unique_blocks) != 23:
        raise RuntimeError(f"expected 23 classical blocks, found {len(unique_blocks)}")
    block_min = {
        block: float(np.min(disagreement[kernels == block])) for block in unique_blocks
    }
    rng = stable_rng(metadata["run"], model)
    rows: list[dict[str, Any]] = []
    for replicate in range(N_FRONTIER_PERMUTATIONS):
        ordering = rng.permutation(unique_blocks)
        for budget, n_blocks in ((30, 6), (60, 12), (115, 23)):
            selected = ordering[:n_blocks]
            upper = min(block_min[block] for block in selected)
            rows.append(
                {
                    "status": "post-hoc exploratory budget sensitivity",
                    "case": case,
                    "run": metadata["run"],
                    "group": metadata["group"],
                    "model": model,
                    "replicate": replicate,
                    "budget": budget,
                    "n_blocks": n_blocks,
                    "accuracy_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def frontier_summary(draws: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["case", "run", "group", "model", "budget", "n_blocks"]
    for key, group in draws.groupby(keys, sort=True):
        values = group.accuracy_upper.to_numpy(dtype=float)
        row = dict(zip(keys, key))
        row.update(
            {
                "n_permutations": len(values),
                "mean_accuracy_upper": float(np.mean(values)),
                "median_accuracy_upper": float(np.median(values)),
                "q025_accuracy_upper": float(np.quantile(values, 0.025)),
                "q975_accuracy_upper": float(np.quantile(values, 0.975)),
            }
        )
        for tau in TAUS:
            token = f"probability_upper_below_{tau:.3f}".replace(".", "p")
            row[token] = float(np.mean(values < tau))
        rows.append(row)
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_label_free(prediction_root: Path, output_dir: Path, models: tuple[str, ...]) -> None:
    if not SPEC_PATH.is_file():
        raise FileNotFoundError(SPEC_PATH)
    case_dirs = sorted(path for path in prediction_root.iterdir() if path.is_dir())
    if len(case_dirs) != 8:
        raise RuntimeError(f"expected eight frozen pilot cases, found {len(case_dirs)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    envelope_rows: list[dict[str, Any]] = []
    frontier_frames: list[pd.DataFrame] = []
    input_hashes: list[dict[str, str]] = []
    for case_dir in case_dirs:
        for model in models:
            arrays, metadata = load_prediction_artifact(case_dir, model)
            prediction_path = case_dir / f"predictions_{model}.npz"
            input_hashes.append({"path": str(prediction_path), "sha256": sha256_file(prediction_path)})
            for tier, indices in _tier_indices(arrays).items():
                envelope_rows.append(
                    envelope_row(case_dir, model, tier, arrays, metadata, indices)
                )
            frontier_frames.append(
                block_frontier_draws(arrays, metadata, case_dir.name, model)
            )

    envelopes = pd.DataFrame(envelope_rows).sort_values(
        ["model", "group", "tier"], kind="stable"
    )
    draws = pd.concat(frontier_frames, ignore_index=True)
    summary = frontier_summary(draws)
    envelope_path = output_dir / "sharp_accuracy_envelopes.csv"
    draw_path = output_dir / "frontier_draws.csv"
    summary_path = output_dir / "frontier_summary.csv"
    write_csv(envelopes, envelope_path)
    write_csv(draws, draw_path)
    write_csv(summary, summary_path)
    manifest = {
        "specification": str(SPEC_PATH),
        "specification_sha256": sha256_file(SPEC_PATH),
        "status": "post-hoc exploratory",
        "mode": "label-free",
        "models": list(models),
        "n_cases": len(case_dirs),
        "n_envelopes": len(envelopes),
        "n_frontier_draws": len(draws),
        "frontier_seed_root": FRONTIER_SEED_ROOT,
        "frontier_permutations": N_FRONTIER_PERMUTATIONS,
        "target_labels_read": False,
        "input_hashes": input_hashes,
        "outputs": {
            str(path): sha256_file(path)
            for path in (envelope_path, draw_path, summary_path)
        },
    }
    write_json(manifest, output_dir / "label_free_manifest.json")
    print(envelopes.to_string(index=False), flush=True)
    print(f"[ok] wrote label-free v0.9 artifacts under {output_dir}", flush=True)


def run_unlocked(prediction_root: Path, output_dir: Path, models: tuple[str, ...]) -> None:
    manifest_path = output_dir / "label_free_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("unlock-evaluation requires a completed label-free manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path_text, expected_hash in manifest["outputs"].items():
        path = Path(path_text)
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"label-free output changed before unlock: {path}")

    rows: list[dict[str, Any]] = []
    label_hashes: list[dict[str, str]] = []
    case_dirs = sorted(path for path in prediction_root.iterdir() if path.is_dir())
    for case_dir in case_dirs:
        label_path = case_dir / "evaluation_labels.npz"
        with np.load(label_path, allow_pickle=False) as label_archive:
            target_indices = label_archive["target_indices"]
            target_labels = label_archive["target_labels"]
        label_hashes.append({"path": str(label_path), "sha256": sha256_file(label_path)})
        n_positive = int(np.sum(target_labels))
        for model in models:
            arrays, metadata = load_prediction_artifact(case_dir, model)
            np.testing.assert_array_equal(target_indices, arrays["target_indices"])
            for tier, indices in _tier_indices(arrays).items():
                classical = arrays["classical_predictions"][:, indices]
                accuracy_envelope = sharp_accuracy_envelope(
                    arrays["quantum_prediction"], classical
                )
                realized_accuracy = realized_accuracy_advantage(
                    target_labels, arrays["quantum_prediction"], classical
                )
                if not accuracy_envelope.lower - 1e-12 <= realized_accuracy <= accuracy_envelope.upper + 1e-12:
                    raise RuntimeError("realized accuracy advantage escaped its sharp envelope")

                exact_bacc = sharp_balanced_accuracy_envelope(
                    arrays["quantum_prediction"],
                    classical,
                    n_positive=n_positive,
                    integral=True,
                )
                relaxed_bacc = sharp_balanced_accuracy_envelope(
                    arrays["quantum_prediction"],
                    classical,
                    n_positive=n_positive,
                    integral=False,
                )
                realized_bacc = realized_balanced_accuracy_advantage(
                    target_labels, arrays["quantum_prediction"], classical
                )
                if not exact_bacc.lower - 1e-10 <= realized_bacc <= exact_bacc.upper + 1e-10:
                    raise RuntimeError("realized BAcc advantage escaped its exact envelope")
                rows.append(
                    {
                        "status": "post-hoc exploratory labels unlocked after certificate",
                        "case": case_dir.name,
                        "run": metadata["run"],
                        "group": metadata["group"],
                        "model": model,
                        "tier": tier,
                        "n_target": len(target_labels),
                        "n_positive": n_positive,
                        "prevalence": n_positive / len(target_labels),
                        "realized_accuracy_advantage": realized_accuracy,
                        "accuracy_lower": accuracy_envelope.lower,
                        "accuracy_upper": accuracy_envelope.upper,
                        "accuracy_upper_slack": accuracy_envelope.upper - realized_accuracy,
                        "realized_bacc_advantage": realized_bacc,
                        "bacc_exact_lower": exact_bacc.lower,
                        "bacc_exact_upper": exact_bacc.upper,
                        "bacc_exact_upper_slack": exact_bacc.upper - realized_bacc,
                        "bacc_relaxed_lower": relaxed_bacc.lower,
                        "bacc_relaxed_upper": relaxed_bacc.upper,
                        "n_prediction_signatures": exact_bacc.n_signatures,
                        "bacc_exact_lower_witness_index": exact_bacc.lower_witness_index,
                        "bacc_exact_upper_solver_status": exact_bacc.upper_status,
                    }
                )
    output = pd.DataFrame(rows).sort_values(["model", "group", "tier"], kind="stable")
    output_path = output_dir / "unlocked_accuracy_bacc_envelopes.csv"
    write_csv(output, output_path)
    write_json(
        {
            "specification": str(SPEC_PATH),
            "status": "post-hoc exploratory",
            "mode": "unlock-evaluation",
            "label_free_manifest_sha256": sha256_file(manifest_path),
            "label_hashes": label_hashes,
            "output": {str(output_path): sha256_file(output_path)},
        },
        output_dir / "unlocked_manifest.json",
    )
    print(output.to_string(index=False), flush=True)
    print(f"[ok] wrote unlocked v0.9 evaluation under {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("label-free", "unlock-evaluation"), required=True)
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("results/v9/partial_identification/predictions"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/v9/partial_identification/analysis"),
    )
    parser.add_argument("--models", choices=("svc", "gpc"), nargs="+", default=["svc"])
    args = parser.parse_args()
    if args.mode == "label-free":
        run_label_free(args.prediction_root, args.out_dir, tuple(args.models))
    else:
        run_unlocked(args.prediction_root, args.out_dir, tuple(args.models))


if __name__ == "__main__":
    main()

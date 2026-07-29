"""Run the frozen v0.6.0 repeated finite-shot sensitivity.

One exact P1'-selected quantum SVC configuration is fixed for each of the
eight security scenario-groups. Each fidelity block is independently sampled
at four shot counts for 30 stable SHA-256-derived measurement replicates.
Both the sampled indefinite Gram matrix and its PSD projection are evaluated.

This is a conditional measurement-sampling Monte Carlo experiment, not a
hardware or device-noise simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.run_phase_b_driver import resolve
from src.analysis.source_datasets import source_dataset_for_group
from src.experiments.ember.extended.run_classical_extensions import eff_rank
from src.experiments.ember.extended.v4_protocol import (
    sample_kernel_finite_shots,
    select_c_by_train_cv,
    split_id_val_test,
)
from src.experiments.ember.quantum.run_ember_quantum_kernel_sparsity_shift_qsplits import (
    DEFAULT_QUANTUM_CONFIGS,
    build_feature_map,
    compute_statevectors_batch,
    kernel_block_abs2,
    load_indices,
    make_embedding_pipeline,
)


SHOTS = (128, 512, 2048, 8192)
N_REPLICATES = 30
PROJECTION_CONDITIONS = ("pre_psd", "post_psd")


@dataclass(frozen=True)
class FixedRun:
    run: str
    kernel: str
    dim: int
    exact_c: float
    summary_sha256: str


FIXED_RUNS = (
    FixedRun(
        "m1_hist_byteent__ms42__q1000_id500_ood500__qs42__s42",
        "pauli_xz_r1_full__as2", 12, 10,
        "4c0b525d29ca3a3d912e5a97176f59d3ee396a34646ff16ea6365943867b0942",
    ),
    FixedRun(
        "m2_hist_byteent__ms42__q1000_id500_ood500__qs42__s42",
        "pauli_xz_r1_full__as2", 12, 100,
        "275c892415d1cfd92451c89bc4a81ea92c7e02ffba5197c815392519e5a9cc83",
    ),
    FixedRun(
        "toniot_scanning__m2_centroid__ms42__q1000_id500_ood500__qs42__s42",
        "zz_r2_full__as2", 6, 10,
        "783a0b798819024cdcf53393be74ba8b33c5ce3730bf440bea6dc4c49e4aae17",
    ),
    FixedRun(
        "toniot_scanning__natural_cur__ms42__q1000_id500_ood500__qs42__s42",
        "pauli_xz_r1_full__as2", 12, 10,
        "89f40af91873929966cfc3002e779385a5fcc4b19ea65e87b513ea61f9e1ef70",
    ),
    FixedRun(
        "unsw_dos__m2_centroid__ms42__q1000_id500_ood500__qs42__s42",
        "zmap_r2__as0.5", 12, 100,
        "307d051fc5098ee05c978fb955acc761756482054e0c1ca38c8b24c61f1d583d",
    ),
    FixedRun(
        "unsw_dos__natural_cur__ms42__q1000_id500_ood500__qs42__s42",
        "zz_r2_full__as0.5", 4, 100,
        "c1f869aaaeaeb705969ef857ba2ed7f259b07bd4bc0b4cfabace961f5cec3e60",
    ),
    FixedRun(
        "unsw_recon__m2_centroid__ms42__q1000_id500_ood500__qs42__s42",
        "zz_r2_full__as0.5", 4, 100,
        "f939251143c98f732f710f7f56db27291e39e7323bde9458a31684c4f568114d",
    ),
    FixedRun(
        "unsw_recon__natural_cur__ms42__q1000_id500_ood500__qs42__s42",
        "zz_r1_full__as0.5", 6, 10,
        "4f0cc9268a97073784b85389a841d54fba81c9483a29067f86c41c40e17d36a9",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_measurement_seed(
    run: str,
    kernel: str,
    dim: int,
    shots: int,
    replicate: int,
    block: str,
) -> int:
    token = (
        f"ksf-v6-shots::{run}::{kernel}::{dim}::{shots}::{replicate}::{block}"
    )
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")


def group_for_run(run: str) -> str:
    tokens = run.split("__")
    if tokens[0].startswith(("m1_", "m2_")):
        return f"ember_{tokens[0].split('_', 1)[0]}"
    return f"{tokens[0]}_{tokens[1]}"


def parse_quantum_kernel(kernel: str) -> tuple[str, float]:
    if "__as" in kernel:
        base, scale = kernel.split("__as", 1)
        return base, float(scale)
    return kernel, 1.0


def centered_kta_fast(K: np.ndarray, y: np.ndarray) -> float:
    """Centered kernel-target alignment without constructing the H matrix."""
    matrix = np.asarray(K, dtype=np.float64)
    centered = (
        matrix
        - matrix.mean(axis=0, keepdims=True)
        - matrix.mean(axis=1, keepdims=True)
        + matrix.mean()
    )
    labels = np.where(np.asarray(y) > 0, 1.0, -1.0)
    numerator = float(labels @ centered @ labels)
    denominator = float(len(labels) * np.linalg.norm(centered, "fro"))
    return numerator / max(denominator, 1e-12)


def evaluate_svc(
    train: np.ndarray,
    y_train: np.ndarray,
    eval_blocks: dict[str, np.ndarray],
    y_eval: dict[str, np.ndarray],
) -> tuple[float, float, dict[str, float]]:
    selected_c, cv_scores = select_c_by_train_cv(train, y_train)
    model = SVC(kernel="precomputed", C=selected_c, class_weight="balanced")
    model.fit(train, y_train)
    scores = {
        split: float(
            balanced_accuracy_score(y_eval[split], model.predict(eval_blocks[split]))
        )
        for split in ("id_val", "id_test", "ood_test")
    }
    return selected_c, float(cv_scores[selected_c]), scores


def locate_run(run: str, roots: tuple[Path, ...]) -> Path:
    hits = [root / run for root in roots if (root / run).is_dir()]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one result directory for {run}, found {hits}")
    return hits[0]


def select_p1_winner(candidates: pd.DataFrame) -> pd.Series:
    """Mirror the original P1' tie-break: lexicographically sorted cfg, first max."""
    ordered = candidates.sort_values("cfg", kind="stable").reset_index(drop=True)
    if ordered.empty or ordered.balanced_accuracy.isna().all():
        raise ValueError("no valid quantum P1' candidates")
    return ordered.iloc[
        int(np.nanargmax(ordered.balanced_accuracy.to_numpy(dtype=float)))
    ]


def build_exact_blocks(fixed: FixedRun, result_dir: Path) -> tuple[dict, dict]:
    resolved = resolve(fixed.run)
    if resolved is None:
        raise RuntimeError(f"cannot resolve inputs for {fixed.run}")
    input_dir, splits_dir, model_seed = resolved
    if model_seed != 42:
        raise RuntimeError(f"frozen run must use model seed 42, found {model_seed}")

    summary_path = result_dir / "summary_v4.csv"
    observed_hash = sha256_file(summary_path)
    if observed_hash != fixed.summary_sha256:
        raise RuntimeError(
            f"{fixed.run}: summary hash {observed_hash} != frozen {fixed.summary_sha256}"
        )
    summary = pd.read_csv(summary_path)
    candidates = summary[
        (summary.family == "quantum")
        & (summary.model == "svc")
        & (summary.split == "id_val")
    ]
    winner = select_p1_winner(candidates)
    if (
        winner.kernel != fixed.kernel
        or int(winner.dim) != fixed.dim
        or float(winner.c_selected) != fixed.exact_c
    ):
        raise RuntimeError(f"{fixed.run}: frozen winner no longer matches {winner.to_dict()}")

    X = np.load(input_dir / "X.npy", mmap_mode="r")
    y = np.load(input_dir / "y.npy").astype(np.int64).ravel()
    indices = {
        split: load_indices(splits_dir / f"{split}_idx.npy")
        for split in ("train", "id_test", "ood_test")
    }
    labels = {split: y[idx] for split, idx in indices.items()}
    val_pos, test_pos, split_audit = split_id_val_test(
        indices["id_test"], labels["id_test"]
    )
    y_eval = {
        "id_val": labels["id_test"][val_pos],
        "id_test": labels["id_test"][test_pos],
        "ood_test": labels["ood_test"],
    }

    embedding = make_embedding_pipeline(
        dim=fixed.dim,
        select_k=None,
        use_scaling=True,
        angle_min=0.0,
        angle_max=float(np.pi),
        seed=model_seed,
    )
    embedding.fit(np.asarray(X[indices["train"]]), labels["train"])
    embedded = {
        split: np.asarray(
            embedding.transform(np.asarray(X[idx])), dtype=np.float64
        )
        for split, idx in indices.items()
    }
    base_kernel, angle_scale = parse_quantum_kernel(fixed.kernel)
    config = next(cfg for cfg in DEFAULT_QUANTUM_CONFIGS if cfg["id"] == base_kernel)
    feature_map = build_feature_map(config, feature_dim=fixed.dim)
    statevectors = {
        split: compute_statevectors_batch(
            values * angle_scale, feature_map, dtype=np.complex64
        )
        for split, values in embedded.items()
    }
    train = kernel_block_abs2(
        statevectors["train"], statevectors["train"], out_dtype=np.float64
    )
    id_full = kernel_block_abs2(
        statevectors["id_test"], statevectors["train"], out_dtype=np.float64
    )
    ood = kernel_block_abs2(
        statevectors["ood_test"], statevectors["train"], out_dtype=np.float64
    )
    ood_square = kernel_block_abs2(
        statevectors["ood_test"], statevectors["ood_test"], out_dtype=np.float64
    )
    blocks = {
        "train": train,
        "id_val": id_full[val_pos],
        "id_test": id_full[test_pos],
        "ood_test": ood,
        "ood_square": ood_square,
    }
    metadata = {
        "y_train": labels["train"],
        "y_eval": y_eval,
        "split_audit": split_audit,
        "summary_winner": winner.to_dict(),
    }
    return blocks, metadata


def run_one(fixed: FixedRun, result_dir: Path, out_dir: Path, force: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = FIXED_RUNS.index(fixed)
    final_path = out_dir / f"{index:02d}_{group_for_run(fixed.run)}.csv"
    partial_path = final_path.with_suffix(".partial.csv")
    if final_path.exists() and not force:
        existing = pd.read_csv(final_path)
        expected = len(SHOTS) * N_REPLICATES * len(PROJECTION_CONDITIONS)
        if len(existing) == expected:
            print(f"[skip] complete {final_path}", flush=True)
            return final_path
        raise RuntimeError(f"incomplete existing output: {final_path}")

    started = time.time()
    exact_blocks, metadata = build_exact_blocks(fixed, result_dir)
    y_train = metadata["y_train"]
    y_eval = metadata["y_eval"]
    exact_c, exact_cv, exact_scores = evaluate_svc(
        exact_blocks["train"],
        y_train,
        {key: exact_blocks[key] for key in ("id_val", "id_test", "ood_test")},
        y_eval,
    )
    if exact_c != fixed.exact_c:
        raise RuntimeError(f"recomputed C {exact_c} != frozen C {fixed.exact_c}")
    summary_winner = metadata["summary_winner"]
    if abs(exact_scores["id_val"] - float(summary_winner["balanced_accuracy"])) > 1e-10:
        raise RuntimeError("recomputed exact ID-validation endpoint differs from summary_v4")

    exact_eff_rank = eff_rank(exact_blocks["train"])
    exact_kta_ood = centered_kta_fast(exact_blocks["ood_square"], y_eval["ood_test"])
    group = group_for_run(fixed.run)
    dataset = source_dataset_for_group(group)
    rows: list[dict] = []
    for shots in SHOTS:
        for replicate in range(N_REPLICATES):
            sampled: dict[str, np.ndarray] = {}
            projected: dict[str, np.ndarray] = {}
            audits: dict[str, dict] = {}
            for block in ("train", "id_val", "id_test", "ood_test", "ood_square"):
                seed = stable_measurement_seed(
                    fixed.run, fixed.kernel, fixed.dim, shots, replicate, block
                )
                pre, post, audit = sample_kernel_finite_shots(
                    exact_blocks[block],
                    shots,
                    np.random.default_rng(seed),
                    square=block in {"train", "ood_square"},
                )
                sampled[block], projected[block], audits[block] = pre, post, audit

            for condition, condition_blocks in (
                ("pre_psd", sampled),
                ("post_psd", projected),
            ):
                selected_c, cv_score, scores = evaluate_svc(
                    condition_blocks["train"],
                    y_train,
                    {
                        key: condition_blocks[key]
                        for key in ("id_val", "id_test", "ood_test")
                    },
                    y_eval,
                )
                train_eff_rank = audits["train"][
                    "effective_rank_before_psd"
                    if condition == "pre_psd"
                    else "effective_rank_after_psd"
                ]
                kta_ood = centered_kta_fast(
                    condition_blocks["ood_square"], y_eval["ood_test"]
                )
                row = {
                    "run": fixed.run,
                    "group": group,
                    "dataset": dataset,
                    "kernel": fixed.kernel,
                    "dim": fixed.dim,
                    "shots": shots,
                    "replicate": replicate,
                    "projection_condition": condition,
                    "selected_c": selected_c,
                    "c_cv_score": cv_score,
                    "id_val_bacc": scores["id_val"],
                    "id_test_bacc": scores["id_test"],
                    "ood_test_bacc": scores["ood_test"],
                    "exact_c": exact_c,
                    "exact_c_cv_score": exact_cv,
                    "exact_id_val_bacc": exact_scores["id_val"],
                    "exact_id_test_bacc": exact_scores["id_test"],
                    "exact_ood_test_bacc": exact_scores["ood_test"],
                    "ood_difference_from_exact": (
                        scores["ood_test"] - exact_scores["ood_test"]
                    ),
                    "absolute_ood_difference": abs(
                        scores["ood_test"] - exact_scores["ood_test"]
                    ),
                    "train_effective_rank": train_eff_rank,
                    "exact_train_effective_rank": exact_eff_rank,
                    "train_effective_rank_ratio": train_eff_rank / exact_eff_rank,
                    "kta_ood": kta_ood,
                    "exact_kta_ood": exact_kta_ood,
                    "kta_ood_difference": kta_ood - exact_kta_ood,
                    "train_min_eig_before_psd": audits["train"][
                        "min_eig_before_psd"
                    ],
                    "train_min_eig_after_psd": audits["train"][
                        "min_eig_after_psd"
                    ],
                    "train_frac_negative_eig": audits["train"][
                        "frac_negative_eig"
                    ],
                    "train_fro_change_sampling": audits["train"][
                        "fro_change_sampling"
                    ],
                    "train_fro_change_projection": audits["train"][
                        "fro_change_projection"
                    ],
                    "ood_min_eig_before_psd": audits["ood_square"][
                        "min_eig_before_psd"
                    ],
                    "ood_min_eig_after_psd": audits["ood_square"][
                        "min_eig_after_psd"
                    ],
                    "ood_frac_negative_eig": audits["ood_square"][
                        "frac_negative_eig"
                    ],
                    "ood_fro_change_sampling": audits["ood_square"][
                        "fro_change_sampling"
                    ],
                    "ood_fro_change_projection": audits["ood_square"][
                        "fro_change_projection"
                    ],
                    "stable_seed_train": stable_measurement_seed(
                        fixed.run,
                        fixed.kernel,
                        fixed.dim,
                        shots,
                        replicate,
                        "train",
                    ),
                }
                rows.append(row)
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            if (replicate + 1) % 5 == 0:
                print(
                    f"[{group}] shots={shots} rep={replicate + 1}/{N_REPLICATES} "
                    f"elapsed={(time.time() - started) / 60:.1f} min",
                    flush=True,
                )

    output = pd.DataFrame(rows)
    expected = len(SHOTS) * N_REPLICATES * len(PROJECTION_CONDITIONS)
    if len(output) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(output)}")
    output.to_csv(final_path, index=False)
    partial_path.unlink(missing_ok=True)
    metadata_path = final_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "frozen_run": asdict(fixed),
                "split_audit": metadata["split_audit"],
                "exact_scores": exact_scores,
                "exact_effective_rank": exact_eff_rank,
                "exact_kta_ood": exact_kta_ood,
                "n_rows": len(output),
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ok] wrote {final_path} in {(time.time() - started) / 60:.1f} min")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-index", type=int, required=True, choices=range(len(FIXED_RUNS))
    )
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=[
            Path("results/ember_shift/extended_kernels"),
            Path("results/netflow/extended_kernels"),
        ],
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/v6/shots_mc/runs")
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    fixed = FIXED_RUNS[args.run_index]
    result_dir = locate_run(fixed.run, tuple(args.roots))
    run_one(fixed, result_dir, args.out_dir, args.force)


if __name__ == "__main__":
    main()

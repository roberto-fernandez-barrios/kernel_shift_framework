"""Fail-closed audit of the complete frozen v5 TableShift grid."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.analysis.tableshift_external_v5 import SEEDS, STRATA, TASKS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_summary(summary: pd.DataFrame, label: str) -> dict:
    keys = ["family", "kernel", "dim", "model", "regularization", "split"]
    if summary.duplicated(keys).any():
        raise ValueError(f"{label}: duplicated result keys")
    if len(summary) != 1260:
        raise ValueError(f"{label}: expected 1260 rows, found {len(summary)}")
    if summary.balanced_accuracy.isna().any():
        raise ValueError(f"{label}: missing primary metric")
    if not summary.balanced_accuracy.between(0, 1).all():
        raise ValueError(f"{label}: primary metric outside [0,1]")
    expected = {
        ("quantum", "svc", "train_cv"): 60 * 3,
        ("quantum", "svc", "fixed_c1"): 60 * 3,
        ("quantum", "gpc", "not_applicable"): 60 * 3,
        ("classical_ext", "svc", "train_cv"): 115 * 3,
        ("classical_ext", "svc", "fixed_c1"): 10 * 3,
        ("classical_ext", "gpc", "not_applicable"): 115 * 3,
    }
    observed = summary.groupby(
        ["family", "model", "regularization"], observed=True
    ).size().to_dict()
    if observed != expected:
        raise ValueError(f"{label}: pool mismatch\nexpected={expected}\nobserved={observed}")
    if set(summary.dim) != {4, 6, 8, 10, 12}:
        raise ValueError(f"{label}: incomplete dimensions")
    if set(summary.split) != {"validation", "id_test", "ood_test"}:
        raise ValueError(f"{label}: incomplete or prohibited splits")
    classical_backends = set(
        summary.loc[summary.family.eq("classical_ext"), "kernel_backend"]
    )
    if classical_backends != {"analytic_cpu"}:
        raise ValueError(f"{label}: invalid classical backend metadata {classical_backends}")
    quantum_backends = set(
        summary.loc[summary.family.eq("quantum"), "kernel_backend"]
    )
    if not quantum_backends.issubset({"cuda", "cpu", "cpu_fallback"}):
        raise ValueError(f"{label}: invalid quantum backends {quantum_backends}")
    return {
        "n_rows": len(summary),
        "n_candidates_quantum": 60,
        "n_candidates_classical": 115,
        "quantum_backends": sorted(quantum_backends),
    }


def main() -> None:
    root = Path("results/v5/external/runs")
    audit_dir = Path("results/v5/audit")
    unit_rows = []
    for task in TASKS:
        sampling_path = audit_dir / f"tableshift_sampling_{task}.csv"
        sampling = pd.read_csv(sampling_path)
        if len(sampling) != 40 or sampling.ood_validation_accessed.astype(bool).any():
            raise ValueError(f"{task}: invalid sampling audit")
        if (sampling[["class_0_n", "class_1_n"]] <= 0).any().any():
            raise ValueError(f"{task}: class-presence gate failed")
        if not (audit_dir / f"tableshift_source_files_{task}.csv").exists():
            raise FileNotFoundError(f"{task}: source hashes absent")

        for stratum in STRATA:
            for seed in SEEDS:
                unit = root / task / stratum / f"seed_{seed}"
                preprocess_path = unit / "preprocess_audit.json"
                summary_path = unit / "summary_v5.csv"
                preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
                if (
                    preprocess["fit_split"] != "train"
                    or preprocess["target_used"] is not False
                    or preprocess["n_post_encoding_features"] < 12
                ):
                    raise ValueError(f"{task}/{stratum}/seed_{seed}: preprocessing gate")
                details = audit_summary(
                    pd.read_csv(summary_path), f"{task}/{stratum}/seed_{seed}"
                )
                unit_rows.append(
                    {
                        "task": task,
                        "stratum": stratum,
                        "seed": seed,
                        "summary_sha256": sha256_file(summary_path),
                        "preprocess_sha256": sha256_file(preprocess_path),
                        **details,
                    }
                )
    unit_frame = pd.DataFrame(unit_rows)
    unit_frame.to_csv(audit_dir / "external_grid_units.csv", index=False)
    report = {
        "status": "pass",
        "n_tasks": len(TASKS),
        "n_units": len(unit_frame),
        "n_summary_rows": int(unit_frame.n_rows.sum()),
        "ood_validation_accessed": False,
        "target_used_by_preprocessing": False,
        "population_p_value_computed": False,
        "unit_manifest_sha256": sha256_file(audit_dir / "external_grid_units.csv"),
    }
    (audit_dir / "external_grid_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

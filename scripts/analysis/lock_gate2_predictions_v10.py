"""Verify and freeze the aggregate v1.0 prediction lock.

This stage hashes sealed-label files as opaque bytes; it never loads their
contents.  The output is the mandatory gate before the one-time label audit.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SPEC = Path("docs/GATE2_PROSPECTIVE_REPLICATION_SPEC_V10.md")
SPEC_SHA256 = "3a8318d92d4af2aeeaf0c0edb069c3be59f31da6d1ee50fb6a6256e9d9d280b0"
ROOT = Path("results/v10/gate2_prospective")
TASKS = ("brfss_diabetes", "acsfoodstamps", "nhanes_lead")
SEEDS = (42, 123, 999, 7, 2024)
MODELS = ("svc", "gpc")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_available(task: str, seed: int) -> dict[str, Any]:
    unit = ROOT / "prediction_locks" / task / f"seed_{seed}"
    manifest_path = unit / "prediction_lock_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["specification_sha256"] != SPEC_SHA256:
        raise RuntimeError(f"spec hash mismatch: {manifest_path}")
    if manifest["target_labels_opened_for_analysis"] is not False:
        raise RuntimeError(f"target labels already marked opened: {manifest_path}")
    if manifest["task"] != task or int(manifest["seed"]) != seed:
        raise RuntimeError(f"unit identity mismatch: {manifest_path}")
    label_path = Path(manifest["sealed_label_artifact"])
    if sha256_file(label_path) != manifest["sealed_label_artifact_sha256"]:
        raise RuntimeError(f"opaque sealed-label hash mismatch: {label_path}")

    files: list[dict[str, Any]] = []
    for model in MODELS:
        prediction_path = unit / f"predictions_{model}.npz"
        expected = manifest["models"][model]["prediction_sha256"]
        observed = sha256_file(prediction_path)
        if observed != expected:
            raise RuntimeError(f"prediction hash mismatch: {prediction_path}")
        with np.load(prediction_path, allow_pickle=False) as prediction:
            if prediction["quantum_prediction"].shape != (500,):
                raise RuntimeError(f"bad quantum shape: {prediction_path}")
            if prediction["classical_predictions"].shape != (500, 115):
                raise RuntimeError(f"bad classical shape: {prediction_path}")
            if int(prediction["customary_mask"].sum()) != 30:
                raise RuntimeError(f"bad customary tier: {prediction_path}")
            values = np.unique(
                np.concatenate(
                    [
                        prediction["quantum_prediction"].ravel(),
                        prediction["classical_predictions"].ravel(),
                    ]
                )
            )
            if not set(values.tolist()).issubset({0, 1}):
                raise RuntimeError(f"non-binary predictions: {prediction_path}")
        files.append(
            {
                "role": f"predictions_{model}",
                "path": prediction_path.as_posix(),
                "sha256": observed,
            }
        )
    files.extend(
        [
            {
                "role": "unit_manifest",
                "path": manifest_path.as_posix(),
                "sha256": sha256_file(manifest_path),
            },
            {
                "role": "sealed_labels_opaque",
                "path": label_path.as_posix(),
                "sha256": sha256_file(label_path),
            },
        ]
    )
    return {"task": task, "seed": seed, "status": "locked", "files": files}


def verify_unavailable(task: str, seed: int) -> dict[str, Any]:
    unit = ROOT / "prediction_locks" / task / f"seed_{seed}"
    path = unit / "technical_unavailability_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["specification_sha256"] != SPEC_SHA256:
        raise RuntimeError(f"spec hash mismatch: {path}")
    if manifest["permitted_gate"] != "minimum-feature failure":
        raise RuntimeError(f"unexpected technical gate: {path}")
    if int(manifest["n_model_candidates_executed"]) != 0:
        raise RuntimeError(f"unavailable unit executed a model: {path}")
    if manifest["target_labels_opened_for_analysis"] is not False:
        raise RuntimeError(f"unavailable labels marked opened: {path}")
    cache = unit / "candidate_cache"
    if cache.exists() and any(cache.glob("*.npz")):
        raise RuntimeError(f"unavailable unit contains candidate predictions: {cache}")
    label_path = Path(manifest["sealed_label_artifact"])
    if sha256_file(label_path) != manifest["sealed_label_artifact_sha256"]:
        raise RuntimeError(f"opaque label hash mismatch: {label_path}")
    return {
        "task": task,
        "seed": seed,
        "status": "technical_unavailability",
        "permitted_gate": manifest["permitted_gate"],
        "reason": manifest["reason"],
        "files": [
            {"path": path.as_posix(), "sha256": sha256_file(path)},
            {
                "path": label_path.as_posix(),
                "sha256": sha256_file(label_path),
                "role": "sealed_labels_opaque",
            },
        ],
    }


def main() -> None:
    if sha256_file(SPEC) != SPEC_SHA256:
        raise RuntimeError("frozen v1.0 specification hash changed")
    if (ROOT / "audit" / "AUDIT_COMPLETE.json").exists():
        raise RuntimeError("refusing to recreate prediction lock after label audit")

    units = []
    available_tasks = []
    unavailable_tasks = []
    for task in TASKS:
        statuses = []
        task_units = []
        for seed in SEEDS:
            unit = ROOT / "prediction_locks" / task / f"seed_{seed}"
            if (unit / "prediction_lock_manifest.json").is_file():
                task_units.append(verify_available(task, seed))
                statuses.append("locked")
            elif (unit / "technical_unavailability_manifest.json").is_file():
                task_units.append(verify_unavailable(task, seed))
                statuses.append("technical_unavailability")
            else:
                raise FileNotFoundError(f"unit has neither lock nor failure: {unit}")
        if len(set(statuses)) != 1:
            raise RuntimeError(f"mixed availability across seeds for {task}: {statuses}")
        if statuses[0] == "locked":
            available_tasks.append(task)
        else:
            unavailable_tasks.append(task)
        units.extend(task_units)

    if len(available_tasks) < 2:
        availability = "technically_incomplete"
    elif unavailable_tasks:
        availability = "technically_limited_two_task_replication"
    else:
        availability = "complete_three_task_replication"
    payload = {
        "status": "aggregate_prediction_lock_complete_before_label_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "specification": SPEC.as_posix(),
        "specification_sha256": SPEC_SHA256,
        "protocol_root": "ksf-v10-gate2-prospective-20260801",
        "availability": availability,
        "available_tasks": available_tasks,
        "unavailable_tasks": unavailable_tasks,
        "n_available_units": sum(unit["status"] == "locked" for unit in units),
        "n_technical_unavailability_units": sum(
            unit["status"] == "technical_unavailability" for unit in units
        ),
        "target_labels_opened_for_analysis": False,
        "units": units,
    }
    output = ROOT / "aggregate_prediction_lock_manifest.json"
    atomic_json(output, payload)
    digest = sha256_file(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    os.replace(temporary, sidecar)
    print(
        f"[locked] aggregate manifest {digest}; "
        f"available={len(available_tasks)}, unavailable={len(unavailable_tasks)}"
    )


if __name__ == "__main__":
    main()

"""Verify and rebuild the prospective v1.0 Gate-2 release artifacts."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable
ORDER = ("export", "lock", "analysis", "report", "audit")
TASKS = ("brfss_diabetes", "acsfoodstamps", "nhanes_lead")
SEEDS = (42, 123, 999, 7, 2024)
ROOT = Path("results/v10/gate2_prospective")


def run(arguments: list[str]) -> None:
    command = [PYTHON, *arguments]
    print("[run]", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(f"[reproduce_v10] failed: {' '.join(arguments)}")


def preflight() -> None:
    required = (
        Path("docs/GATE2_PROSPECTIVE_REPLICATION_SPEC_V10.md"),
        Path("docs/GATE2_PROSPECTIVE_REPLICATION_FREEZE_V10.json"),
        Path("data/raw/tableshift/v10/exports"),
        Path("results/v10/gate2_prospective/acquisition"),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("[reproduce_v10] missing inputs:\n- " + "\n- ".join(missing))


def export() -> None:
    for task in TASKS:
        for seed in SEEDS:
            run(
                [
                    "-m",
                    "scripts.experiments.export_gate2_predictions_v10",
                    "--task",
                    task,
                    "--seed",
                    str(seed),
                ]
            )


def lock() -> None:
    aggregate = ROOT / "aggregate_prediction_lock_manifest.json"
    if aggregate.exists():
        sidecar = aggregate.with_suffix(aggregate.suffix + ".sha256")
        if not sidecar.exists():
            raise SystemExit("[reproduce_v10] aggregate lock lacks SHA-256 sidecar")
        print("[skip] immutable aggregate prediction lock already exists", flush=True)
        return
    run(["-m", "scripts.analysis.lock_gate2_predictions_v10"])


def analysis() -> None:
    run(["-m", "scripts.analysis.audit_gate2_prospective_v10"])


def report() -> None:
    run(["-m", "scripts.reporting.make_v10_gate2_prospective_figure"])


def audit() -> None:
    run(["-m", "pytest", "tests/test_gate2_prospective_v10.py", "-q"])
    run(["-m", "scripts.analysis.validate_manuscript_v10"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*ORDER, "all"), default="all")
    args = parser.parse_args()
    preflight()
    stages = ORDER if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"\n===== stage: {stage} =====", flush=True)
        globals()[stage]()
    print("\n[OK] reproduce_v10 complete:", ", ".join(stages))


if __name__ == "__main__":
    main()

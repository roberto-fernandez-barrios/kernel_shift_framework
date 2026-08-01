"""Rebuild and validate the v0.9 partial-identification artifacts.

The export stage reconstructs exact hard predictions for the eight frozen
q1000 cases and the existing v0.6 finite-shot replicates. The remaining
stages are inexpensive deterministic analyses.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable
ORDER = ("export", "analysis", "report", "audit")


def run(command: list[str]) -> None:
    print("[run]", " ".join(command), flush=True)
    completed = subprocess.run([PYTHON, *command], check=False)
    if completed.returncode:
        raise SystemExit(f"[reproduce_v9] failed: {' '.join(command)}")


def preflight() -> None:
    required = (
        Path("docs/PARTIAL_IDENTIFICATION_SPEC_V9.md"),
        Path("results/v4/budget_confirmatory/p1_runs__budget60.csv"),
        Path("results/v6/shots_mc"),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("[reproduce_v9] missing inputs:\n- " + "\n- ".join(missing))


def export() -> None:
    for index in range(8):
        run(
            [
                "-m",
                "scripts.experiments.export_partial_identification_v9",
                "--run-index",
                str(index),
                "--models",
                "svc",
                "gpc",
            ]
        )
    for index in range(8):
        run(
            [
                "-m",
                "scripts.experiments.export_shot_predictions_v9",
                "--run-index",
                str(index),
            ]
        )


def analysis() -> None:
    run(
        [
            "-m",
            "scripts.analysis.partial_identification_v9",
            "--mode",
            "label-free",
            "--models",
            "svc",
            "gpc",
        ]
    )
    run(
        [
            "-m",
            "scripts.analysis.partial_identification_v9",
            "--mode",
            "unlock-evaluation",
            "--models",
            "svc",
            "gpc",
        ]
    )
    for model in ("svc", "gpc"):
        run(
            [
                "-m",
                "scripts.analysis.partial_label_frontier_v9",
                "--model",
                model,
                "--out-dir",
                f"results/v9/partial_identification/partial_label_frontier/{model}",
            ]
        )
    run(["-m", "scripts.analysis.shot_partial_identification_v9"])


def report() -> None:
    run(["-m", "scripts.reporting.make_v9_partial_identification_figures"])


def audit() -> None:
    tests = sorted(str(path) for path in Path("tests").glob("test_*v9.py"))
    if not tests:
        raise SystemExit("[reproduce_v9] no v0.9 tests found")
    run(["-m", "pytest", *tests, "-q"])
    run(["-m", "scripts.analysis.validate_manuscript_v9"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*ORDER, "all"), default="all")
    args = parser.parse_args()
    preflight()
    stages = ORDER if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"\n===== stage: {stage} =====", flush=True)
        globals()[stage]()
    print("\n[OK] reproduce_v9 complete:", ", ".join(stages))


if __name__ == "__main__":
    main()

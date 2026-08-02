"""Rebuild and validate the v0.8 circuit and protocol derived artifacts.

This command consumes the versioned v4 summaries plus the frozen
``summary_v8_fixedc.csv`` and ``summary_v8_shortcut.csv`` campaign outputs. It
does not reacquire raw benchmarks or rerun the kernel/model grids.

Usage:
  python scripts/reproduce_v8.py --stage analysis
  python scripts/reproduce_v8.py --stage report
  python scripts/reproduce_v8.py --stage audit
  python scripts/reproduce_v8.py --stage all
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable
RUN_RE = re.compile(r".+__q1000_id500_ood500__qs\d+__s\d+$")
RESULT_ROOTS = (
    Path("results/ember_shift/extended_kernels"),
    Path("results/netflow/extended_kernels"),
)
STAGES = {
    "analysis": [
        ["-m", "scripts.analysis.circuit_resources_v8"],
        ["-m", "scripts.analysis.reviewer_revision_v8"],
    ],
    "report": [
        ["-m", "scripts.reporting.make_v8_figures"],
        ["-m", "scripts.reporting.make_v8_tables"],
    ],
    "audit": [
        ["-m", "scripts.analysis.validate_v6_artifacts"],
        ["-m", "scripts.analysis.validate_v8_artifacts"],
        ["-m", "scripts.analysis.validate_manuscript_v8"],
    ],
}
ORDER = ("analysis", "report", "audit")


def _count_campaign_files(filename: str) -> int:
    return sum(
        1
        for root in RESULT_ROOTS
        for path in root.glob(f"*/{filename}")
        if RUN_RE.fullmatch(path.parent.name)
    )


def preflight() -> None:
    required = [
        Path("docs/REVIEWER_REVISION_SPEC_V8.md"),
        Path("results/v4/budget_confirmatory/p1_runs__budget60.csv"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    campaign_counts = {
        "summary_v8_fixedc.csv": (360, _count_campaign_files("summary_v8_fixedc.csv")),
        "summary_v8_shortcut.csv": (
            270,
            _count_campaign_files("summary_v8_shortcut.csv"),
        ),
    }
    for filename, (expected, observed) in campaign_counts.items():
        if observed != expected:
            missing.append(
                f"{filename}: expected {expected} q1000 runs, found {observed}"
            )
    if missing:
        raise SystemExit("[reproduce_v8] preflight failed:\n- " + "\n- ".join(missing))
    print(
        "[reproduce_v8] preflight OK: frozen protocol and complete "
        "360/270 campaigns present"
    )


def run_stage(stage: str) -> None:
    print(f"\n===== stage: {stage} =====")
    for command in STAGES[stage]:
        print("[run]", " ".join(command))
        completed = subprocess.run([PYTHON, *command], check=False)
        if completed.returncode:
            raise SystemExit(
                f"[reproduce_v8] stage '{stage}' failed at: {' '.join(command)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*ORDER, "all"), default="all")
    args = parser.parse_args()
    preflight()
    stages = ORDER if args.stage == "all" else (args.stage,)
    for stage in stages:
        run_stage(stage)
    print("\n[OK] reproduce_v8 complete:", ", ".join(stages))


if __name__ == "__main__":
    main()

"""Reproduce the frozen v5 specification curve and external validation.

This command consumes the versioned v4 artifacts and the 30 audited TableShift
unit summaries.  It does not reacquire raw benchmark data or rerun the expensive
kernel grid.

Usage:
  python scripts/reproduce_v5.py --stage audit
  python scripts/reproduce_v5.py --stage analysis
  python scripts/reproduce_v5.py --stage report
  python scripts/reproduce_v5.py --stage all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PY = sys.executable
RUNS_ROOT = Path("results/v5/external/runs")
STAGES = {
    "audit": [
        ["-m", "scripts.analysis.audit_tableshift_v5"],
    ],
    "analysis": [
        ["-m", "scripts.analysis.specification_curve_v5"],
        ["-m", "scripts.analysis.tableshift_external_v5"],
    ],
    "report": [
        ["-m", "scripts.reporting.make_v5_figures"],
    ],
}
ORDER = ("audit", "analysis", "report")


def preflight() -> None:
    summaries = sorted(RUNS_ROOT.glob("*/*/seed_*/summary_v5.csv"))
    if len(summaries) != 30:
        raise SystemExit(
            f"[reproduce_v5] expected 30 TableShift summaries, found "
            f"{len(summaries)}; finish scripts/experiments/"
            "run_tableshift_v5_all.py first"
        )
    print(f"[reproduce_v5] preflight OK: {len(summaries)} unit summaries present")


def run_stage(stage: str) -> None:
    print(f"\n===== stage: {stage} =====")
    for command in STAGES[stage]:
        print("[run]", " ".join(command))
        completed = subprocess.run([PY, *command], check=False)
        if completed.returncode != 0:
            raise SystemExit(
                f"[reproduce_v5] stage '{stage}' failed at: "
                f"{' '.join(command)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*ORDER, "all"), default="all")
    args = parser.parse_args()
    preflight()
    stages = ORDER if args.stage == "all" else (args.stage,)
    for stage in stages:
        run_stage(stage)
    print("\n[OK] reproduce_v5 complete:", ", ".join(stages))


if __name__ == "__main__":
    main()

"""Verify the non-empirical v1.1 novelty consolidation."""
from __future__ import annotations

import argparse
import subprocess
import sys


PYTHON = sys.executable
ORDER = ("theory", "manuscript")


def run(arguments: list[str]) -> None:
    command = [PYTHON, *arguments]
    print("[run]", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(f"[reproduce_v11] failed: {' '.join(arguments)}")


def theory() -> None:
    run(["-m", "pytest", "tests/test_bounded_loss_v11.py", "-q"])


def manuscript() -> None:
    run(["-m", "scripts.analysis.validate_manuscript_v11"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*ORDER, "all"), default="all")
    args = parser.parse_args()
    stages = ORDER if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"\n===== stage: {stage} =====", flush=True)
        globals()[stage]()
    print("\n[OK] reproduce_v11 complete:", ", ".join(stages))


if __name__ == "__main__":
    main()

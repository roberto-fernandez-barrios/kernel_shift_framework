"""Reproduce and validate the reviewer-driven v0.6.0 derived artifacts.

This command consumes versioned per-run summaries.  It does not reacquire raw
benchmarks or rerun the expensive kernel grids.  The repeated finite-shot run
CSVs are likewise treated as frozen experiment outputs; ``analysis`` rebuilds
their summaries and figures.

Usage:
  python scripts/reproduce_v6.py --stage analysis
  python scripts/reproduce_v6.py --stage report
  python scripts/reproduce_v6.py --stage audit
  python scripts/reproduce_v6.py --stage all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PY = sys.executable
STAGES = {
    "analysis": [
        [
            "-m",
            "scripts.analysis.hierarchical_effect_estimation",
            "--p1-dir",
            "results/v4/budget_confirmatory",
            "--out-dir",
            "results/v6/inference",
        ],
        [
            "-m",
            "scripts.analysis.hierarchical_effect_estimation",
            "--p1-dir",
            "results/v4/family_comparison",
            "--variants",
            "vs_classical_orig",
            "vs_classical_ext",
            "--out-dir",
            "results/v6/family_comparison/inference",
        ],
        [
            "-m",
            "scripts.analysis.specification_curve_v5",
            "--out-dir",
            "results/v6/specification_curve",
            "--figure",
            "manuscript/fig_v6_specification.pdf",
        ],
        ["-m", "scripts.analysis.budget_scheme_sensitivity_v6"],
        [
            "-m",
            "scripts.analysis.mechanism_robustness_v4",
            "--out-dir",
            "results/v6/mechanism",
        ],
        ["-m", "scripts.analysis.shots_mc_v6"],
    ],
    "report": [
        ["-m", "scripts.reporting.make_v6_figures"],
        [
            "-c",
            "from scripts.reporting.make_v4_figures import "
            "fig_rankmatched; fig_rankmatched()",
        ],
        ["-m", "scripts.reporting.make_readme_assets"],
    ],
    "audit": [
        ["-m", "scripts.analysis.validate_v6_artifacts"],
    ],
}
ORDER = ("analysis", "report", "audit")


def preflight() -> None:
    required = [
        Path("docs/REVIEWER_REVISION_SPEC_V6.md"),
        Path("results/v4/budget_confirmatory/p1_runs__budget60.csv"),
        Path("results/v4/family_comparison/p1_runs__vs_classical_ext.csv"),
        Path("results/v4/budget_confirmatory/resamples_by_group.csv"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    shot_csvs = sorted(Path("results/v6/shots_mc/runs").glob("*.csv"))
    shot_manifests = sorted(Path("results/v6/shots_mc/runs").glob("*.json"))
    if len(shot_csvs) != 8 or len(shot_manifests) != 8:
        missing.append(
            "results/v6/shots_mc/runs: expected 8 CSVs and 8 manifests, "
            f"found {len(shot_csvs)} and {len(shot_manifests)}"
        )
    if missing:
        raise SystemExit("[reproduce_v6] preflight failed:\n- " + "\n- ".join(missing))
    print("[reproduce_v6] preflight OK: frozen inputs and 8 shot runs present")


def run_stage(stage: str) -> None:
    print(f"\n===== stage: {stage} =====")
    for command in STAGES[stage]:
        print("[run]", " ".join(command))
        completed = subprocess.run([PY, *command], check=False)
        if completed.returncode:
            raise SystemExit(
                f"[reproduce_v6] stage '{stage}' failed at: {' '.join(command)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*ORDER, "all"), default="all")
    args = parser.parse_args()
    preflight()
    stages = ORDER if args.stage == "all" else (args.stage,)
    for stage in stages:
        run_stage(stage)
    print("\n[OK] reproduce_v6 complete:", ", ".join(stages))


if __name__ == "__main__":
    main()

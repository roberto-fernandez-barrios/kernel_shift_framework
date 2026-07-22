# scripts/reproduce_v4.py
"""
Master reproduction command for the v0.4 methodological revision
(docs/ANALYSIS_SPEC_V4.md). Regenerates the confirmatory analysis, tables, and
figures from the frozen per-run summaries. The heavy recompute (Phase 3, the
1080 quantum runs) is NOT re-run here; its outputs are the versioned
summary_v4.csv files, whose provenance is recorded by the audit stage.

Usage:
  python scripts/reproduce_v4.py --stage audit       # GATE 0 provenance + hashes
  python scripts/reproduce_v4.py --stage analysis    # confirmatory gates + mechanism + shots
  python scripts/reproduce_v4.py --stage report      # v4 tables + figures
  python scripts/reproduce_v4.py --stage all         # the three above, in order

Each stage validates its inputs and fails loudly (never silently overwrites v0.3
outputs, which live outside results/v4/). Run from the repository root with the
project environment active.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable

STAGES = {
    "audit": [
        ["scripts/analysis/audit_v4.py"],
    ],
    "analysis": [
        ["scripts/analysis/budget_matched_selection.py",
         "--roots", "ember=results/ember_shift/extended_kernels",
         "netflow=results/netflow/extended_kernels",
         "--base-file", "summary_v4.csv", "--extra-files", "--select-col", "id_val",
         "--out-dir", "results/v4/budget_confirmatory"],
        ["scripts/analysis/hierarchical_effect_estimation.py",
         "--p1-dir", "results/v4/budget_confirmatory",
         "--variants", "budget60", "full",
         "--out-dir", "results/v4/inference_confirmatory"],
        ["scripts/analysis/honest_family_comparison_v4.py"],
        ["scripts/analysis/hierarchical_effect_estimation.py",
         "--p1-dir", "results/v4/family_comparison",
         "--variants", "vs_classical_orig", "vs_classical_ext",
         "--out-dir", "results/v4/family_comparison/inference"],
        ["scripts/analysis/mechanism_robustness_v4.py", "--model", "svc"],
        ["scripts/analysis/shots_stability_analysis.py"],
    ],
    "report": [
        ["scripts/reporting/make_v4_tables.py"],
        ["scripts/reporting/make_v4_figures.py"],
    ],
}
ORDER = ["audit", "analysis", "report"]


def preflight() -> None:
    n = len(list(Path("results/ember_shift/extended_kernels").glob("*/summary_v4.csv"))) \
        + len(list(Path("results/netflow/extended_kernels").glob("*/summary_v4.csv")))
    if n != 1080:
        raise SystemExit(f"[reproduce_v4] expected 1080 summary_v4.csv, found {n}; "
                         "run the Phase-3 recompute (scripts/experiments/run_v4_all.py) first")
    print(f"[reproduce_v4] preflight OK: {n} run summaries present")


def run_stage(stage: str) -> None:
    print(f"\n===== stage: {stage} =====")
    for cmd in STAGES[stage]:
        full = [PY, *cmd]
        print("[run]", " ".join(cmd))
        r = subprocess.run(full)
        if r.returncode != 0:
            raise SystemExit(f"[reproduce_v4] stage '{stage}' failed at: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=[*ORDER, "all"], default="all")
    args = ap.parse_args()
    preflight()
    stages = ORDER if args.stage == "all" else [args.stage]
    for s in stages:
        run_stage(s)
    print("\n[OK] reproduce_v4 complete:", ", ".join(stages))


if __name__ == "__main__":
    main()

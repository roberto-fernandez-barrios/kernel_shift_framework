# scripts/experiments/run_v4_all.py
"""Sequential v4-recompute batch for one shard (docs/ANALYSIS_SPEC_V4.md).

Batch 1 runs the finite-shot subset first (q1000, model seed 42 -- the spec
subset) with --shots; batch 2 covers every remaining run without shots. The
driver is resume-safe on summary_v4.csv, so batch 2 skips batch-1 dirs and a
killed fleet resumes with the same command.
"""
from __future__ import annotations

import subprocess
import sys

ROOTS = ["results/ember_shift/extended_kernels", "results/netflow/extended_kernels"]

BATCHES = [
    ["--roots", *ROOTS, "--mode", "v4", "--filter", "q1000", "__s42",
     "--runner-args=--shots"],
    # size-ordered so the q1000+q2000 confirmatory stratum (spec section 7
    # fallback) completes first; q4000 (the slow stratum) runs last
    ["--roots", *ROOTS, "--mode", "v4", "--filter", "q1000"],
    ["--roots", *ROOTS, "--mode", "v4", "--filter", "q2000"],
    ["--roots", *ROOTS, "--mode", "v4", "--filter", "q4000"],
]


def main() -> None:
    shard = sys.argv[1]
    for extra in BATCHES:
        cmd = [sys.executable, "scripts/experiments/run_phase_b_driver.py",
               *extra, "--shard", shard]
        print(f"[batch] {' '.join(extra)}", flush=True)
        p = subprocess.run(cmd)
        if p.returncode != 0:
            raise SystemExit(p.returncode)
    print(f"[OK] v4 recompute shard {shard} complete", flush=True)


if __name__ == "__main__":
    main()

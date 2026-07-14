# scripts/experiments/run_phase_b_all.py
"""Sequential Phase-B/C batch for one shard: lsweep -> ablation -> csens -> ktanull."""
from __future__ import annotations

import subprocess
import sys

BATCHES = [
    ["--roots", "results/ember_shift/bandwidth_sweep",
     "results/netflow_bandwidth_sweep/extended_kernels", "--mode", "lsweep"],
    ["--roots", "results/ember_shift/extended_kernels", "--mode", "ablation"],
    ["--roots", "results/netflow/extended_kernels", "--mode", "ablation", "--filter", "unsw_dos"],
    ["--roots", "results/ember_shift/extended_kernels",
     "results/netflow/extended_kernels", "--mode", "csens"],
    ["--roots", "results/ember_shift/extended_kernels", "--mode", "ktanull",
     "--include-quantum", "--filter", "q1000"],
    ["--roots", "results/netflow/extended_kernels", "--mode", "ktanull",
     "--include-quantum", "--filter", "q1000"],
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
    print(f"[OK] Phase B/C shard {shard} complete", flush=True)


if __name__ == "__main__":
    main()

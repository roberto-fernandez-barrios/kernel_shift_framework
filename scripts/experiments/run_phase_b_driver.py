# scripts/experiments/run_phase_b_driver.py
"""
Drives run_classical_extensions over existing run directories.

For each run dir under --roots whose extension output is missing, reconstructs
the (in-dir, splits-dir, seed) triple from the directory label and invokes the
extension runner as a subprocess. Resume-safe; shardable with --shard i/N
(runs where index % N == i).

Labels:
  EMBER   : {variant}__ms{ms}__q{n}_id{i}_ood{o}__qs{qs}__s{s}
            in-dir  data/processed/ember
            splits  data/processed/ember/splits_sparsity__{variant}__ms{ms}__h*/
                    qsplits/splits_sparsity_q{n}_id{i}_ood{o}_seed{qs}
  netflow : {scenario}__{variant}__ms{ms}__q{n}_id{i}_ood{o}__qs{qs}__s{s}
            in-dir  data/processed/netflow/{scenario}
            splits  data/processed/netflow/{scenario}/splits_{variant}__ms{ms}/
                    qsplits/splits_sparsity_q{n}_id{i}_ood{o}_seed{qs}
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from src.experiments.ember.extended.run_classical_extensions import OUT_FILES

LABEL_RE = re.compile(
    r"^(?P<head>.+)__ms(?P<ms>\d+)__q(?P<n>\d+)_id(?P<i>\d+)_ood(?P<o>\d+)__qs(?P<qs>\d+)__s(?P<s>\d+)$")
NETFLOW_SCENARIOS = ("unsw_dos", "unsw_recon", "toniot_scanning")


def resolve(label: str) -> tuple[Path, Path, int] | None:
    m = LABEL_RE.match(label)
    if not m:
        return None
    head, ms, qs = m.group("head"), m.group("ms"), m.group("qs")
    qname = f"splits_sparsity_q{m.group('n')}_id{m.group('i')}_ood{m.group('o')}_seed{qs}"
    if head.startswith(NETFLOW_SCENARIOS):
        scenario, variant = head.split("__", 1)
        in_dir = Path(f"data/processed/netflow/{scenario}")
        splits = in_dir / f"splits_{variant}__ms{ms}" / "qsplits" / qname
    else:
        in_dir = Path("data/processed/ember")
        hits = sorted(in_dir.glob(f"splits_sparsity__{head}__ms{ms}__h*"))
        if not hits:
            return None
        splits = hits[0] / "qsplits" / qname
    return in_dir, splits, int(m.group("s"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--mode", choices=list(OUT_FILES), required=True)
    ap.add_argument("--include-quantum", action="store_true")
    ap.add_argument("--shard", type=str, default="0/1", help="i/N")
    ap.add_argument("--dims", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    args = ap.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    run_dirs = []
    for root in args.roots:
        run_dirs += sorted(d for d in root.iterdir()
                           if d.is_dir() and LABEL_RE.match(d.name)
                           and (d / "extended_kernels_qsplits__summary.csv").exists())
    todo = [d for k, d in enumerate(run_dirs)
            if k % shard_n == shard_i and not (d / OUT_FILES[args.mode]).exists()]
    print(f"[plan] shard {args.shard}: {len(todo)} of {len(run_dirs)} run dirs pending ({args.mode})")

    t0 = time.time()
    for j, d in enumerate(todo):
        r = resolve(d.name)
        if r is None:
            print(f"[skip] cannot resolve {d.name}")
            continue
        in_dir, splits, seed = r
        if not splits.exists():
            print(f"[skip] missing splits {splits}")
            continue
        cmd = [sys.executable, "-m", "src.experiments.ember.extended.run_classical_extensions",
               "--in-dir", str(in_dir), "--splits-dir", str(splits), "--out-dir", str(d),
               "--seed", str(seed), "--dims", *map(str, args.dims), "--mode", args.mode]
        if args.include_quantum:
            cmd.append("--include-quantum")
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"[FAIL] {d.name}\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
            raise SystemExit(1)
        print(f"[{j + 1}/{len(todo)}] {d.name} ok ({(time.time() - t0) / 60:.1f} min total)")
    print(f"[OK] shard {args.shard} complete: {len(todo)} runs in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

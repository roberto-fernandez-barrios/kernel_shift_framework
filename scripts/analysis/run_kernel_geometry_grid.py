# scripts/analysis/run_kernel_geometry_grid.py
"""
Driver for the kernel-geometry analysis over the paper's 18-setting grid.

Steps:
  1. Ensure master splits exist (delegates to the paper orchestrator with
     --master-only, so variant configs, hashing and seeds are identical to the
     experiment runs).
  2. Generate q-splits per (variant, master seed, size, qsplit seed) with the
     same module the orchestrator uses.
  3. Run src.analysis.run_kernel_geometry_ember_qsplits per setting.

Defaults reproduce the paper protocol with a single representative q-split
seed (42); pass --qsplit-seeds 42 123 999 7 2024 for the full repeated-run
structure.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

PAPER_SIZES: List[Tuple[int, int, int]] = [(1000, 500, 500), (2000, 1000, 1000), (4000, 1800, 1800)]
DEFAULT_VARIANTS = ["m1_hist_byteent", "m2_hist_byteent"]
DEFAULT_MASTER_SEEDS = [42, 123, 999]
DEFAULT_DIMS = [4, 6, 8, 10, 12]


def run(cmd: List[str], log_path: Path) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[CMD] {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        tail = "".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-30:])
        raise RuntimeError(f"Command failed (exit {p.returncode}). Log tail:\n{tail}")


def find_master_dir(splits_root: Path, variant: str, master_seed: int) -> Path:
    hits = sorted(splits_root.glob(f"splits_sparsity__{variant}__ms{master_seed}__h*"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one master dir for {variant}/ms{master_seed}, found: {hits}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Kernel-geometry analysis over the 18-setting paper grid.")
    ap.add_argument("--splits-root", type=Path, default=Path("data/processed/ember"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/kernel_geometry/grid"))
    ap.add_argument("--log-dir", type=Path, default=Path("results/kernel_geometry/_logs"))
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--master-seeds", nargs="+", type=int, default=DEFAULT_MASTER_SEEDS)
    ap.add_argument("--qsplit-seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)
    ap.add_argument("--sizes", nargs="+", default=[f"{a},{b},{c}" for a, b, c in PAPER_SIZES],
                    help="Triples n_train,n_id,n_ood")
    ap.add_argument("--skip-masters", action="store_true", help="Assume master splits already exist.")
    ap.add_argument("--save-spectra", action="store_true")
    ap.add_argument("--classical-kernels", nargs="+", default=None,
                    help="Pass through to the analysis module (default: its own default).")
    ap.add_argument("--quantum-configs-json", type=str, default="",
                    help="Pass through; a file containing [] skips quantum kernels.")
    args = ap.parse_args()

    sizes = []
    for s in args.sizes:
        a, b, c = (int(x) for x in s.split(","))
        sizes.append((a, b, c))

    py = sys.executable
    t_start = time.time()

    # 1) Master splits via the paper orchestrator (identical config + hashing).
    if not args.skip_masters:
        run(
            [py, "scripts/ember/run_compare_q_vs_c_full.py",
             "--master-only",
             "--variants", *args.variants,
             "--master-seeds", *[str(s) for s in args.master_seeds],
             "--splits-root-dir", str(args.splits_root)],
            args.log_dir / "masters.log",
        )

    # 2) + 3) q-splits and geometry analysis per setting.
    n_settings = 0
    for variant in args.variants:
        for ms in args.master_seeds:
            master_dir = find_master_dir(args.splits_root, variant, ms)
            qsplits_root = master_dir / "qsplits"
            qsplits_root.mkdir(parents=True, exist_ok=True)

            for (n_train, n_id, n_ood) in sizes:
                for qs in args.qsplit_seeds:
                    qname = f"splits_sparsity_q{n_train}_id{n_id}_ood{n_ood}_seed{qs}"
                    qdir = qsplits_root / qname
                    if not (qdir / "train_idx.npy").exists():
                        run(
                            [py, "-m", "src.utils.ember.make_qsplits_from_master",
                             "--src", str(master_dir),
                             "--dst-root", str(qsplits_root),
                             "--seed", str(qs),
                             "--n-train", str(n_train),
                             "--n-id", str(n_id),
                             "--n-ood", str(n_ood),
                             "--use-low-high"],
                            args.log_dir / f"qsplits__{variant}__ms{ms}.log",
                        )

                    label = f"{variant}__ms{ms}__q{n_train}_id{n_id}_ood{n_ood}__qs{qs}"
                    out_csv = args.out_dir / f"kernel_geometry__{label}.csv"
                    if out_csv.exists():
                        print(f"[SKIP] {label} (already done)")
                        n_settings += 1
                        continue

                    t0 = time.time()
                    cmd = [py, "-m", "src.analysis.run_kernel_geometry_ember_qsplits",
                           "--splits-dir", str(qdir),
                           "--out-dir", str(args.out_dir),
                           "--setting-label", label,
                           "--dims", *[str(d) for d in args.dims],
                           "--mmap"]
                    if args.save_spectra:
                        cmd.append("--save-spectra")
                    if args.classical_kernels:
                        cmd += ["--classical-kernels", *args.classical_kernels]
                    if args.quantum_configs_json:
                        cmd += ["--quantum-configs-json", args.quantum_configs_json]
                    run(cmd, args.log_dir / f"geometry__{label}.log")
                    n_settings += 1
                    print(f"[OK] {label} in {time.time() - t0:.0f}s "
                          f"({n_settings} settings done, {time.time() - t_start:.0f}s total)")

    print(f"[✓] Grid complete: {n_settings} setting-qsplit combos in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()

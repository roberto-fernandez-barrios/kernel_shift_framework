# scripts/netflow/run_netflow_grid.py
"""
Workstream C driver: instantiate the kernel-swap protocol on network-flow
datasets (UNSW-NB15, ToN-IoT scenarios prepared for Paper 2).

Per scenario: export ref/cur pools -> master shift splits (m2_centroid analog
+ natural ref->cur drift) -> q-splits (shared module) -> extended kernel
runner (SVC + Laplace-GPC, classical extended + quantum) and kernel-geometry
analysis (mechanism test on a new modality).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PAPER_SIZES: List[Tuple[int, int, int]] = [(1000, 500, 500), (2000, 1000, 1000), (4000, 1800, 1800)]
DEFAULT_VARIANTS = ["m2_centroid", "natural_cur"]
DEFAULT_MASTER_SEEDS = [42, 123, 999]
DEFAULT_DIMS = [4, 6, 8, 10, 12]

SCENARIOS: Dict[str, Tuple[str, str]] = {
    "unsw_dos": ("unsw_nb15/unsw_ref_no_dos_binary.csv", "unsw_nb15/unsw_cur_dos_binary.csv"),
    "unsw_recon": ("unsw_nb15/unsw_ref_no_reconnaissance_binary.csv", "unsw_nb15/unsw_cur_reconnaissance_binary.csv"),
    "toniot_scanning": ("ton_iot_q1_gate/ton_iot_ref_no_scanning_binary.csv", "ton_iot_q1_gate/ton_iot_cur_scanning_binary.csv"),
}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper2-data", type=Path, default=Path(r"C:\Users\masteria.DOMINE\RF\paper_2\data\processed"))
    ap.add_argument("--data-root", type=Path, default=Path("data/processed/netflow"))
    ap.add_argument("--results-root", type=Path, default=Path("results/netflow"))
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()), choices=list(SCENARIOS.keys()))
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--master-seeds", nargs="+", type=int, default=DEFAULT_MASTER_SEEDS)
    ap.add_argument("--qsplit-seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--model-seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)
    ap.add_argument("--sizes", nargs="+", default=[f"{a},{b},{c}" for a, b, c in PAPER_SIZES])
    ap.add_argument("--skip-geometry", action="store_true")
    ap.add_argument("--skip-extended", action="store_true")
    ap.add_argument("--rbf-gamma-factors", nargs="+", type=float, default=[])
    ap.add_argument("--quantum-angle-scales", nargs="+", type=float, default=[])
    args = ap.parse_args()

    sizes = [tuple(int(x) for x in s.split(",")) for s in args.sizes]
    py = sys.executable
    log_dir = args.results_root / "_logs"
    t_start = time.time()
    n_done = 0

    for scenario in args.scenarios:
        ref_rel, cur_rel = SCENARIOS[scenario]
        data_dir = args.data_root / scenario

        # 1) export
        if not (data_dir / "X.npy").exists():
            run(
                [py, "-m", "src.utils.netflow.export_refcur_to_Xy",
                 "--ref-csv", str(args.paper2_data / ref_rel),
                 "--cur-csv", str(args.paper2_data / cur_rel),
                 "--out-dir", str(data_dir)],
                log_dir / f"export__{scenario}.log",
            )

        for variant in args.variants:
            for ms in args.master_seeds:
                master_dir = data_dir / f"splits_{variant}__ms{ms}"

                # 2) master splits
                if not (master_dir / "meta.json").exists():
                    run(
                        [py, "-m", "src.utils.netflow.make_splits_netflow",
                         "--in-dir", str(data_dir),
                         "--out-dir", str(master_dir),
                         "--variant", variant,
                         "--master-seed", str(ms)],
                        log_dir / f"master__{scenario}__{variant}__ms{ms}.log",
                    )

                use_low_high = variant != "natural_cur"
                qsplits_root = master_dir / "qsplits"

                for (n_train, n_id, n_ood) in sizes:
                    for qs in args.qsplit_seeds:
                        qname = f"splits_sparsity_q{n_train}_id{n_id}_ood{n_ood}_seed{qs}"
                        qdir = qsplits_root / qname

                        # 3) q-splits (shared, dataset-agnostic module)
                        if not (qdir / "train_idx.npy").exists():
                            cmd = [py, "-m", "src.utils.ember.make_qsplits_from_master",
                                   "--src", str(master_dir),
                                   "--dst-root", str(qsplits_root),
                                   "--seed", str(qs),
                                   "--n-train", str(n_train),
                                   "--n-id", str(n_id),
                                   "--n-ood", str(n_ood),
                                   "--strict-sizes"]
                            if use_low_high:
                                cmd.append("--use-low-high")
                            run(cmd, log_dir / f"qsplits__{scenario}__{variant}__ms{ms}.log")

                        # 4) extended runner + geometry
                        for mseed in args.model_seeds:
                            label = f"{scenario}__{variant}__ms{ms}__q{n_train}_id{n_id}_ood{n_ood}__qs{qs}__s{mseed}"

                            if not args.skip_extended:
                                out_dir = args.results_root / "extended_kernels" / label
                                if not (out_dir / "extended_kernels_qsplits__summary.csv").exists():
                                    t0 = time.time()
                                    cmd = [py, "-m", "src.experiments.ember.extended.run_ember_extended_kernels_qsplits",
                                           "--in-dir", str(data_dir),
                                           "--splits-dir", str(qdir),
                                           "--out-dir", str(out_dir),
                                           "--seed", str(mseed),
                                           "--dims", *[str(d) for d in args.dims],
                                           "--include-quantum",
                                           "--mmap"]
                                    if args.rbf_gamma_factors:
                                        cmd += ["--rbf-gamma-factors", *[f"{f:g}" for f in args.rbf_gamma_factors]]
                                    if args.quantum_angle_scales:
                                        cmd += ["--quantum-angle-scales", *[f"{s:g}" for s in args.quantum_angle_scales]]
                                    run(cmd, log_dir / f"extended__{label}.log")
                                    print(f"[OK] extended {label} in {time.time() - t0:.0f}s")

                            if not args.skip_geometry and mseed == args.model_seeds[0]:
                                geo_dir = args.results_root / "kernel_geometry"
                                if not (geo_dir / f"kernel_geometry__{label}.csv").exists():
                                    t0 = time.time()
                                    run(
                                        [py, "-m", "src.analysis.run_kernel_geometry_ember_qsplits",
                                         "--in-dir", str(data_dir),
                                         "--splits-dir", str(qdir),
                                         "--out-dir", str(geo_dir),
                                         "--setting-label", label,
                                         "--dims", *[str(d) for d in args.dims],
                                         "--classical-kernels", "linear", "rbf_gscale", "laplacian_med", "matern15_med",
                                         "--save-spectra",
                                         "--mmap"],
                                        log_dir / f"geometry__{label}.log",
                                    )
                                    print(f"[OK] geometry {label} in {time.time() - t0:.0f}s")

                        n_done += 1
                        print(f"[SETTING DONE] {scenario}/{variant}/ms{ms}/{n_train} "
                              f"({n_done} settings, {(time.time() - t_start) / 60:.1f} min total)")

    print(f"[OK] Netflow grid complete: {n_done} settings in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()

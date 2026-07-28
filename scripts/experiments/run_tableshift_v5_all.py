"""Resume-safe driver for the frozen v5 TableShift external grid.

The default command runs all three fixed tasks, both nested sizes, five seeds,
both classifier families, and both geometry families.  Partial invocations are
safe: every unit-level runner skips completed configuration/model cells.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


TASKS = ("college_scorecard", "diabetes_readmission", "acsincome")
STRATA = ("q1000", "q2000")
SEEDS = (42, 123, 999, 7, 2024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--strata", nargs="+", choices=STRATA, default=list(STRATA))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--models", nargs="+", choices=("svc", "gpc"),
                        default=["svc", "gpc"])
    parser.add_argument("--families", nargs="+",
                        choices=("classical_ext", "quantum"),
                        default=["classical_ext", "quantum"])
    parser.add_argument("--dims", nargs="+", type=int,
                        choices=(4, 6, 8, 10, 12), default=[4, 6, 8, 10, 12])
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--blas-threads", type=int, default=None,
        help="threads per worker for MKL/OpenMP/OpenBLAS; useful with --workers > 1",
    )
    parser.add_argument(
        "--retries", type=int, default=2,
        help="resume and retry a failed unit this many times",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    jobs = []
    for task in args.tasks:
        schema = Path(f"results/v5/audit/tableshift_schema_{task}.csv")
        if not schema.exists():
            raise SystemExit(f"missing acquisition schema: {schema}")
        for stratum in args.strata:
            for seed in args.seeds:
                command = [
                    sys.executable,
                    "-m",
                    "src.experiments.tableshift.run_external_validation_v5",
                    "--task", task,
                    "--stratum", stratum,
                    "--seed", str(seed),
                    "--schema", str(schema),
                    "--backend", args.backend,
                    "--models", *args.models,
                    "--families", *args.families,
                    "--dims", *map(str, args.dims),
                ]
                if args.preflight_only:
                    command.append("--preflight-only")
                jobs.append((task, stratum, seed, command))

    def run(job):
        task, stratum, seed, command = job
        label = f"{task}/{stratum}/seed_{seed}"
        print(f"\n[v5] start {label} models={','.join(args.models)}", flush=True)
        environment = os.environ.copy()
        if args.blas_threads is not None:
            if args.blas_threads < 1:
                raise ValueError("--blas-threads must be >= 1")
            for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[variable] = str(args.blas_threads)
        failures = []
        for attempt in range(args.retries + 1):
            completed = subprocess.run(
                command, check=False, text=True, capture_output=True, env=environment
            )
            if completed.returncode == 0:
                print(f"[v5] complete {label}\n{completed.stdout}", flush=True)
                return label
            failures.append(
                f"attempt {attempt + 1}: code={completed.returncode}\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
            print(
                f"[v5] retry {label} after failed attempt {attempt + 1}",
                flush=True,
            )
        raise RuntimeError(f"{label} exhausted retries\n" + "\n".join(failures))

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()

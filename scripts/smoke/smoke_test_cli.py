from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMANDS = [
    [sys.executable, "-m", "src.utils.ember.extract_ember_raw", "--help"],
    [sys.executable, "-m", "src.utils.ember.export_ember_jsonl_to_Xy", "--help"],
    [sys.executable, "-m", "src.utils.ember.make_splits_ember_sparsity", "--help"],
    [sys.executable, "-m", "src.utils.ember.make_qsplits_from_master", "--help"],
    [sys.executable, "scripts/ember/run_compare_q_vs_c_full.py", "--help"],
    [sys.executable, "-m", "src.experiments.ember.classical.run_ember_classical_kernel_sparsity_shift_qsplits", "--help"],
    [sys.executable, "-m", "src.experiments.ember.quantum.run_ember_quantum_kernel_sparsity_shift_qsplits", "--help"],
]


def main() -> int:
    report = []
    any_fail = False
    for cmd in COMMANDS:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        item = {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout_head": proc.stdout.splitlines()[:5],
            "stderr_head": proc.stderr.splitlines()[:8],
        }
        report.append(item)
        if proc.returncode != 0:
            any_fail = True

    out_path = ROOT / "docs" / "smoke_test_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_path}")
    for item in report:
        status = "PASS" if item["returncode"] == 0 else "FAIL"
        print(f"[{status}] {item['command']}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

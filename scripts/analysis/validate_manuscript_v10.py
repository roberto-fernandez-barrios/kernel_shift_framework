"""Fail-fast scientific, chronology, and format gates for the v1.0 paper."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.validate_manuscript_v9 import (
    MAIN,
    SUPPLEMENT,
    uncommented,
    validate_main,
    validate_supplement,
    validate_v9,
)


ROOT = Path("results/v10/gate2_prospective")
FIGURE = Path("manuscript/fig_v10_gate2_prospective.pdf")
SPEC_SHA256 = "3a8318d92d4af2aeeaf0c0edb069c3be59f31da6d1ee50fb6a6256e9d9d280b0"
REQUIRED_MAIN = (
    r"\label{sec:res_gate2_prospective}",
    r"\label{sec:methods_gate2_prospective}",
    r"\includegraphics[width=\textwidth]{fig_v10_gate2_prospective.pdf}",
    r"\label{fig:v10_gate2}",
    "technically limited",
    "All 20 prospective realized accuracy effects were negative",
    "All prospectively selected quantum winners are product maps",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_v10() -> None:
    main = MAIN.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    active = uncommented(main)
    missing = [fragment for fragment in REQUIRED_MAIN if fragment not in active]
    if missing:
        raise ValueError("missing v1.0 manuscript fragments: " + ", ".join(missing))
    if not FIGURE.is_file():
        raise FileNotFoundError(f"missing prospective figure: {FIGURE}")
    if "Supplementary Results: Prospective Gate-2 replication" not in supplement:
        raise ValueError("prospective Gate-2 supplementary section is missing")

    abstract_match = re.search(r"\\abstract\{(.*?)\}\s*\n", active, re.DOTALL)
    if not abstract_match:
        raise ValueError("cannot locate abstract")
    words = re.findall(r"\b[\w$'-]+\b", abstract_match.group(1))
    if len(words) > 200:
        raise ValueError(f"abstract has {len(words)} words; expected at most 200")

    spec = Path("docs/GATE2_PROSPECTIVE_REPLICATION_SPEC_V10.md")
    if sha256_file(spec) != SPEC_SHA256:
        raise ValueError("frozen v1.0 specification hash changed")
    lock_path = ROOT / "aggregate_prediction_lock_manifest.json"
    lock_hash = sha256_file(lock_path)
    sidecar = lock_path.with_suffix(lock_path.suffix + ".sha256")
    if lock_hash != sidecar.read_text(encoding="ascii").split()[0]:
        raise ValueError("aggregate prediction-lock sidecar mismatch")
    outcome = json.loads((ROOT / "audit/prospective_outcome.json").read_text())
    expected = "strong_prospective_transfer__technically_limited_two_task_replication"
    if outcome["outcome_category"] != expected:
        raise ValueError(f"unexpected prospective category: {outcome['outcome_category']}")

    zero = pd.read_csv(ROOT / "audit/zero_label_and_realized.csv")
    zero = zero[zero.tier.eq("full_115")]
    if len(zero) != 20:
        raise ValueError(f"expected 20 prospective full-family cells, found {len(zero)}")
    if not (zero.realized_accuracy_advantage < 0).all():
        raise ValueError("not every prospective realized accuracy effect is negative")
    if int((zero.zero_label_upper <= 0.010 + 1e-12).sum()) != 11:
        raise ValueError("zero-label primary crossing count changed")

    adaptive = pd.read_csv(ROOT / "audit/adaptive_thresholds.csv")
    primary = adaptive[
        adaptive.tier.eq("full_115") & np.isclose(adaptive.threshold, 0.010)
    ]
    if float(primary.n_labels.median()) != 0.0 or int(primary.n_labels.max()) != 76:
        raise ValueError("prospective adaptive headline changed")
    medians = sorted(
        primary.groupby(["task", "model"]).n_labels.median().to_numpy(dtype=float)
    )
    if medians != [0.0, 0.0, 3.0, 3.0]:
        raise ValueError(f"task-classifier medians changed: {medians}")

    if "A precommitted external replication would be needed" in active:
        raise ValueError("obsolete pre-replication limitation remains in main text")
    print(
        f"[ok] v1.0 chronology, outcome, figure, supplement, and abstract gates "
        f"({len(words)} abstract words)"
    )


def main() -> None:
    validate_main(MAIN.read_text(encoding="utf-8"))
    validate_supplement(SUPPLEMENT.read_text(encoding="utf-8"))
    validate_v9()
    validate_v10()
    print("[ok] all v1.0 manuscript gates passed")


if __name__ == "__main__":
    main()

"""Fail-fast theory, preservation, positioning, and release gates for v1.1."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.validate_manuscript_v10 import (
    MAIN,
    SUPPLEMENT,
    uncommented,
    validate_main,
    validate_supplement,
    validate_v9,
    validate_v10,
)


SPEC = Path("docs/V11_CONSOLIDATION_SPEC.md")
SPEC_SHA256 = "9e54f8a2905992913213df43884bcf0d63ca35684e235ed7aaf1b2400d41b5e3"
BIBLIOGRAPHY = Path("manuscript/sn-bibliography.bib")
COVER = Path("manuscript/cover_letter_npjqi.md")
FRONTIER = Path(
    "results/v9/partial_identification/analysis/frontier_summary.csv"
)

REQUIRED_MAIN = (
    r"\label{thm:bounded_loss}",
    "Sharp bounded-loss region and information optimality",
    "any assumption-free interval",
    "reference-breadth--target-supervision evidence frontier",
    "30, 60, and 115 candidates",
    "internally locked prospective replication",
    "This was not a public preregistration",
    r"\cite{madani2004covalidation}",
    r"\cite{okanovic2025modelselector}",
    r"\cite{shen2026vanishing}",
)

PRESERVATION_FRAGMENTS = (
    r"\label{tab:headline}",
    r"\label{fig:honest}",
    r"\label{tab:circuits}",
    r"\label{fig:v8_revision}",
    r"\label{fig:external}",
    r"\label{fig:v9_shots}",
    "Within-v4 sensitivities separate the evaluation bundle",
    "Geometry is associated with robustness only partially",
    "The quantum pool mixes entangling and product kernels",
    "Protocol sensitivity is prospectively corroborated on external domain shifts",
    "Finite shots can create distinctness without useful advantage",
)

REQUIRED_BIB = (
    "madani2004covalidation",
    "okanovic2025modelselector",
    "shen2026vanishing",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_v11() -> None:
    main = MAIN.read_text(encoding="utf-8")
    active = uncommented(main)
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")
    cover = COVER.read_text(encoding="utf-8")

    if sha256_file(SPEC) != SPEC_SHA256:
        raise ValueError("frozen v1.1 consolidation specification changed")
    missing = [fragment for fragment in REQUIRED_MAIN if fragment not in active]
    if missing:
        raise ValueError("missing v1.1 manuscript fragments: " + ", ".join(missing))
    missing_preserved = [
        fragment for fragment in PRESERVATION_FRAGMENTS if fragment not in active
    ]
    if missing_preserved:
        raise ValueError(
            "v1.1 contribution-preservation failure: "
            + ", ".join(missing_preserved)
        )
    if "supp:tab_v11_breadth" not in supplement:
        raise ValueError("v1.1 Supplementary reference-breadth table is missing")
    missing_bib = [
        key
        for key in REQUIRED_BIB
        if not re.search(rf"@\w+\{{{re.escape(key)},", bibliography)
    ]
    if missing_bib:
        raise ValueError("missing v1.1 bibliography entries: " + ", ".join(missing_bib))

    expected_title = (
        "Sharp Target-Domain Certificates for Quantum-Kernel Advantage "
        "under Distribution Shift"
    )
    cover_plain = cover.replace("**", "")
    if expected_title not in cover_plain:
        raise ValueError("npj cover letter does not use the v1.1 manuscript title")
    for fragment in (
        "bounded loss",
        "information can be uniformly smaller",
        "internally locked",
        "rather than publicly preregistered",
        "finite-shot noise can manufacture predictive distinctness",
    ):
        if fragment not in cover:
            raise ValueError(f"cover letter is missing v1.1 positioning: {fragment}")
    if "Evaluation Choices Shape Apparent" in cover or "v0.8.0 Zenodo record" in cover:
        raise ValueError("cover letter retains obsolete v0.8 submission framing")

    frontier = pd.read_csv(FRONTIER)
    medians = frontier.groupby("budget").median_accuracy_upper.median()
    hits = (
        frontier.assign(hit=frontier.median_accuracy_upper.le(0.010 + 1e-12))
        .groupby("budget")
        .hit.sum()
    )
    expected_medians = {30: 0.042, 60: 0.034, 115: 0.034}
    expected_hits = {30: 3, 60: 4, 115: 4}
    if set(medians.index) != set(expected_medians):
        raise ValueError(f"unexpected reference breadths: {sorted(medians.index)}")
    for budget, expected in expected_medians.items():
        if not np.isclose(float(medians.loc[budget]), expected, atol=1e-12):
            raise ValueError(f"breadth-{budget} median changed")
        if int(hits.loc[budget]) != expected_hits[budget]:
            raise ValueError(f"breadth-{budget} threshold count changed")

    code = Path("src/analysis/partial_identification.py").read_text(encoding="utf-8")
    if "def sharp_bounded_loss_envelope(" not in code:
        raise ValueError("bounded-loss implementation is missing")
    if 'version = "1.1.1"' not in Path("pyproject.toml").read_text(encoding="utf-8"):
        raise ValueError("pyproject version is not 1.1.1")
    if 'version: "1.1.1"' not in Path("CITATION.cff").read_text(encoding="utf-8"):
        raise ValueError("CITATION.cff version is not 1.1.1")
    print("[ok] v1.1 theory, positioning, breadth, preservation, and cover gates")


def main() -> None:
    validate_main(MAIN.read_text(encoding="utf-8"))
    validate_supplement(SUPPLEMENT.read_text(encoding="utf-8"))
    validate_v9()
    validate_v10()
    validate_v11()
    print("[ok] all v1.1 manuscript gates passed")


if __name__ == "__main__":
    main()

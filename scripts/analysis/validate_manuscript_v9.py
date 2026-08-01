"""Fail-fast scientific and format gates for the v0.9 manuscript."""
from __future__ import annotations

import re
from pathlib import Path

from scripts.analysis.validate_manuscript_v8 import (
    MAIN,
    SUPPLEMENT,
    uncommented,
    validate_main,
    validate_supplement,
)


FIGURES = (
    Path("manuscript/fig_v9_certificate.pdf"),
    Path("manuscript/fig_v9_label_frontier.pdf"),
    Path("manuscript/fig_v9_shot_emulation.pdf"),
)
REQUIRED_MAIN = (
    r"\label{thm:zero_label}",
    r"\label{thm:partial_label}",
    r"\label{sec:methods_identification}",
    r"\label{sec:methods_frontier}",
    r"\label{sec:methods_bacc}",
    r"\label{sec:methods_transport}",
    r"\includegraphics[width=\textwidth]{fig_v9_certificate.pdf}",
    r"\includegraphics[width=0.86\textwidth]{fig_v9_shot_emulation.pdf}",
    "post-hoc",
    "exploratory",
)
REQUIRED_BIB = (
    "kossen2021activetesting",
    "musgrove2023discordant",
    "polo2024partialidentification",
    "thabet2026disagree",
)


def validate_v9() -> None:
    main = MAIN.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    bibliography = Path("manuscript/sn-bibliography.bib").read_text(encoding="utf-8")
    active = uncommented(main)

    missing = [fragment for fragment in REQUIRED_MAIN if fragment not in active]
    if missing:
        raise ValueError("missing v0.9 manuscript fragments: " + ", ".join(missing))
    missing_figures = [str(path) for path in FIGURES if not path.is_file()]
    if missing_figures:
        raise FileNotFoundError("missing v0.9 figures: " + ", ".join(missing_figures))
    missing_bib = [
        key
        for key in REQUIRED_BIB
        if not re.search(rf"@\w+\{{{re.escape(key)},", bibliography)
    ]
    if missing_bib:
        raise ValueError("missing v0.9 bibliography entries: " + ", ".join(missing_bib))
    if "Supplementary Results: Target-label evidence frontier" not in supplement:
        raise ValueError("v0.9 evidence-frontier supplement is missing")
    if "Supplementary Results: Prevalence-conditional balanced accuracy" not in supplement:
        raise ValueError("v0.9 balanced-accuracy supplement is missing")
    if "fig_v9_label_frontier.pdf" not in active + "\n" + supplement:
        raise ValueError("v0.9 label-frontier figure is missing from article and supplement")

    labels = set(re.findall(r"\\label\{([^{}]+)\}", active))
    refs = set(re.findall(r"\\ref\{([^{}]+)\}", active))
    unresolved_internal = sorted(refs - labels)
    if unresolved_internal:
        raise ValueError(
            "main source has unresolved internal references: "
            + ", ".join(unresolved_internal)
        )
    print("[ok] v0.9 theorem, chronology, figure, bibliography, and reference gates")


def main() -> None:
    validate_main(MAIN.read_text(encoding="utf-8"))
    validate_supplement(SUPPLEMENT.read_text(encoding="utf-8"))
    validate_v9()
    print("[ok] all v0.9 manuscript gates passed")


if __name__ == "__main__":
    main()

"""Generate manuscript-facing LaTeX rows for the v0.8 sensitivities.

The files contain table rows only. Captions, labels, and interpretation remain
in ``manuscript/supplementary.tex`` so that scientific qualifications are
reviewable next to the generated values.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FACTORIAL_ROWS = (
    ("fixed_c1", "ood_test", r"$C=1$, OOD oracle"),
    ("fixed_c1", "id_val", r"$C=1$, ID validation"),
    ("train_cv", "ood_test", r"Train-CV, OOD oracle"),
    ("train_cv", "id_val", r"Train-CV, ID validation"),
)
FACTORIAL_COLUMNS = (
    ("customary", "native"),
    ("customary", "equal_count"),
    ("extended", "native"),
    ("extended", "equal_count"),
)
AXIS_LABELS = {
    "regularization": r"Regularization: $C=1\to$ train-CV",
    "selection": r"Selection: OOD oracle $\to$ ID validation",
    "reference": r"Reference: customary $\to$ extended",
    "budget_mode": r"Budget: native $\to$ equal count",
}
GROUP_ORDER = (
    "unsw_dos_natural_cur",
    "unsw_dos_m2_centroid",
    "unsw_recon_natural_cur",
    "unsw_recon_m2_centroid",
    "toniot_scanning_natural_cur",
    "toniot_scanning_m2_centroid",
)
GROUP_LABELS = {
    "unsw_dos_natural_cur": "UNSW-DoS (campaign)",
    "unsw_dos_m2_centroid": "UNSW-DoS (constructed)",
    "unsw_recon_natural_cur": "UNSW-Recon (campaign)",
    "unsw_recon_m2_centroid": "UNSW-Recon (constructed)",
    "toniot_scanning_natural_cur": "ToN-IoT (campaign)",
    "toniot_scanning_m2_centroid": "ToN-IoT (constructed)",
}
MAP_ORDER = (
    "zz_r1_full",
    "zz_r2_full",
    "pauli_xz_r1_full",
    "zmap_r2",
)
MAP_LABELS = {
    "zz_r1_full": "ZZ, 1 rep.",
    "zz_r2_full": "ZZ, 2 reps.",
    "pauli_xz_r1_full": "Pauli-X/Z, 1 rep.",
    "zmap_r2": "Z, 2 reps.",
}
STRATUM_LABELS = {
    "entangling_zz": "entangling ZZ",
    "separable_product": "separable product",
}


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def interval(effect: float, lo: float, hi: float) -> str:
    return (
        f"${signed(effect)}\\;[{signed(lo)},{signed(hi)}]$"
    )


def _factorial_rows(frame: pd.DataFrame) -> str:
    lines = []
    for regularization, selection, label in FACTORIAL_ROWS:
        values = []
        for reference, budget_mode in FACTORIAL_COLUMNS:
            cell = frame[
                (frame.regularization == regularization)
                & (frame.selection == selection)
                & (frame.reference == reference)
                & (frame.budget_mode == budget_mode)
            ]
            if len(cell) != 1:
                raise ValueError(
                    "missing or duplicated factorial cell: "
                    f"{regularization}/{selection}/{reference}/{budget_mode}"
                )
            values.append(
                f"${signed(cell.iloc[0].dataset_equal_effect)}$"
            )
        lines.append(f"{label} & " + " & ".join(values) + r" \\")
    return "\n".join(lines) + "\n"


def _contrast_rows(
    contrasts: pd.DataFrame,
    interactions: pd.DataFrame,
) -> str:
    mean_contrasts = contrasts[
        contrasts.contrast_scope == "mean_over_other_factorial_axes"
    ].set_index("axis")
    mean_interactions = interactions[
        interactions.interaction_scope
        == "mean_over_remaining_factorial_axes"
    ]
    if set(mean_contrasts.index) != set(AXIS_LABELS):
        raise ValueError("factorial mean-contrast axes are incomplete")
    if len(mean_interactions) != 6:
        raise ValueError("expected six mean pairwise interactions")

    lines = [
        f"{AXIS_LABELS[axis]} & ${signed(mean_contrasts.loc[axis, 'paired_change'])}$ \\\\"
        for axis in (
            "regularization",
            "selection",
            "reference",
            "budget_mode",
        )
    ]
    lines.append(r"\addlinespace")
    for row in mean_interactions.itertuples():
        label = (
            "Interaction: "
            + row.axis_a.replace("_", " ")
            + r" $\times$ "
            + row.axis_b.replace("_", " ")
        )
        lines.append(
            f"{label} & ${signed(row.difference_in_differences)}$ \\\\"
        )
    return "\n".join(lines) + "\n"


def _shortcut_rows(
    groups: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    indexed = groups.set_index("group")
    if set(indexed.index) != set(GROUP_ORDER):
        raise ValueError("shortcut group coverage is incomplete")
    lines = []
    for group in GROUP_ORDER:
        row = indexed.loc[group]
        lines.append(
            f"{GROUP_LABELS[group]} & "
            + interval(
                row.original_effect,
                row.original_ci_lo,
                row.original_ci_hi,
            )
            + " & "
            + interval(
                row.ablated_effect,
                row.ablated_ci_lo,
                row.ablated_ci_hi,
            )
            + " & "
            + interval(
                row.ablation_change,
                row.change_ci_lo,
                row.change_ci_hi,
            )
            + r" \\"
        )
    if len(summary) != 1:
        raise ValueError("shortcut source-dataset summary must contain one row")
    row = summary.iloc[0]
    lines.extend(
        [
            r"\midrule",
            "Two-source-dataset-equal"
            f" & ${signed(row.original_effect)}$"
            f" & ${signed(row.ablated_effect)}$"
            f" & ${signed(row.ablation_change)}"
            f"\\;[{signed(row.min_dataset_change)},"
            f"{signed(row.max_dataset_change)}]$ \\\\",
        ]
    )
    return "\n".join(lines) + "\n"


def _winner_rows(winners: pd.DataFrame) -> str:
    keys = ["base_map", "selector", "model"]
    if winners.duplicated(keys).any() or len(winners) != 16:
        raise ValueError("winner-composition cells are incomplete or duplicated")
    indexed = winners.set_index(keys)
    lines = []
    for base_map in MAP_ORDER:
        if base_map not in set(winners.base_map):
            raise ValueError(f"winner-composition map missing: {base_map}")
        map_rows = winners[winners.base_map == base_map]
        stratum_values = set(map_rows.map_stratum)
        if len(stratum_values) != 1:
            raise ValueError(f"inconsistent stratum for {base_map}")
        values = []
        for selector, model in (
            ("id_val", "svc"),
            ("id_val", "gpc"),
            ("ood_test", "svc"),
            ("ood_test", "gpc"),
        ):
            row = indexed.loc[(base_map, selector, model)]
            values.append(
                f"{int(row.n_winners)} ({100 * row.winner_fraction:.1f}\\%)"
            )
        lines.append(
            f"{MAP_LABELS[base_map]} & "
            f"{STRATUM_LABELS[next(iter(stratum_values))]} & "
            + " & ".join(values)
            + r" \\"
        )
    return "\n".join(lines) + "\n"


def make_tables(input_dir: Path, output_dir: Path) -> None:
    factorial = pd.read_csv(input_dir / "factorial_summary.csv")
    contrasts = pd.read_csv(input_dir / "factorial_axis_contrasts.csv")
    interactions = pd.read_csv(
        input_dir / "factorial_pairwise_interactions.csv"
    )
    shortcut = pd.read_csv(
        input_dir / "shortcut_ablation_group_effects.csv"
    )
    shortcut_summary = pd.read_csv(
        input_dir / "shortcut_ablation_summary.csv"
    )
    winners = pd.read_csv(input_dir / "quantum_winner_composition.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "v8_factorial_rows.tex": _factorial_rows(factorial),
        "v8_factorial_contrast_rows.tex": _contrast_rows(
            contrasts,
            interactions,
        ),
        "v8_shortcut_rows.tex": _shortcut_rows(
            shortcut,
            shortcut_summary,
        ),
        "v8_winner_rows.tex": _winner_rows(winners),
    }
    for filename, text in outputs.items():
        path = output_dir / filename
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"[ok] wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/v8/reviewer_revision"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manuscript/generated"),
    )
    args = parser.parse_args()
    make_tables(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

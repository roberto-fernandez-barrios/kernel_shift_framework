"""Generate v5 manuscript figures from frozen machine-readable summaries."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_LABELS = {
    "college_scorecard": "College Scorecard",
    "diabetes_readmission": "Diabetes readmission",
    "acsincome": "ACS income",
}
STRATUM_LABELS = {"q1000": "1k", "q2000": "2k"}
METRICS = (
    ("weak_fixed_oracle_delta", "Weak baseline + fixed C + oracle", "^", "#ba4b3d"),
    ("oracle_equal_delta", "Equal budget + train-CV + oracle", "s", "#d09026"),
    ("honest_delta", "Equal budget + train-CV + validation", "o", "#1f6685"),
)


def external_figure(aggregate: pd.DataFrame, output: Path) -> None:
    rows = [
        (task, stratum)
        for task in TASK_LABELS
        for stratum in ("q1000", "q2000")
    ]
    labels = [
        f"{TASK_LABELS[task]} ({STRATUM_LABELS[stratum]})"
        for task, stratum in rows
    ] + ["Task-equal mean"]
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.4), sharey=True)
    offsets = (-0.20, 0.0, 0.20)
    for ax, model in zip(axes, ("svc", "gpc")):
        part = aggregate[aggregate.model.eq(model)]
        for offset, (metric, label, marker, color) in zip(offsets, METRICS):
            effects, lows, highs = [], [], []
            for task, stratum in rows:
                cell = part[
                    part.scope.eq("task_size")
                    & part.task.eq(task)
                    & part.stratum.eq(stratum)
                    & part.metric.eq(metric)
                ]
                if len(cell) != 1:
                    raise ValueError(
                        f"missing aggregate cell: {task}/{stratum}/{model}/{metric}"
                    )
                effects.append(float(cell.effect.iloc[0]))
                lows.append(float(cell.ci_lo.iloc[0]))
                highs.append(float(cell.ci_hi.iloc[0]))
            overall = part[
                part.scope.eq("task_equal") & part.metric.eq(metric)
            ]
            if len(overall) != 1:
                raise ValueError(f"missing task-equal cell: {model}/{metric}")
            effects.append(float(overall.effect.iloc[0]))
            lows.append(float(overall.ci_lo.iloc[0]))
            highs.append(float(overall.ci_hi.iloc[0]))
            effects = np.asarray(effects)
            errors = np.vstack([effects - lows, np.asarray(highs) - effects])
            y = np.arange(len(labels)) + offset
            ax.errorbar(
                effects, y, xerr=errors, fmt=marker, markersize=5.5,
                color=color, ecolor=color, elinewidth=0.9, capsize=2.0,
                markeredgecolor="white", markeredgewidth=0.4, label=label,
            )
        ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.9)
        ax.grid(axis="x", color="#e2e5e7", linewidth=0.7)
        ax.set_title(model.upper())
        ax.set_xlabel("Quantum minus classical OOD balanced accuracy")
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8.5)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.02), fontsize=8.5,
    )
    fig.suptitle(
        "External domain shifts separate oracle and deployable conclusions",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path("results/v5/external/analysis/aggregate_effects.csv"),
    )
    parser.add_argument(
        "--external-output",
        type=Path,
        default=Path("manuscript/fig_v5_external.pdf"),
    )
    args = parser.parse_args()
    aggregate = pd.read_csv(args.aggregate)
    external_figure(aggregate, args.external_output)
    print(f"[OK] wrote {args.external_output}")


if __name__ == "__main__":
    main()

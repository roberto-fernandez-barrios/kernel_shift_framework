"""Tests for v5 external figure completeness gates."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.reporting.make_v5_figures import (  # noqa: E402
    METRICS,
    TASK_LABELS,
    external_figure,
)


def test_external_figure_smoke(tmp_path):
    rows = []
    for model in ("svc", "gpc"):
        for task in TASK_LABELS:
            for stratum in ("q1000", "q2000"):
                for idx, (metric, *_rest) in enumerate(METRICS):
                    rows.append(
                        {
                            "scope": "task_size", "task": task,
                            "stratum": stratum, "model": model, "metric": metric,
                            "effect": idx / 100, "ci_lo": idx / 100 - 0.01,
                            "ci_hi": idx / 100 + 0.01,
                        }
                    )
        for idx, (metric, *_rest) in enumerate(METRICS):
            rows.append(
                {
                    "scope": "task_equal", "task": "all",
                    "stratum": "sizes_equal", "model": model, "metric": metric,
                    "effect": idx / 100, "ci_lo": idx / 100 - 0.01,
                    "ci_hi": idx / 100 + 0.01,
                }
            )
    output = tmp_path / "external.pdf"
    external_figure(pd.DataFrame(rows), output)
    assert output.exists() and output.stat().st_size > 1000

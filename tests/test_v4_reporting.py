"""Regression tests for the frozen v4 manuscript headline outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reporting.make_v4_tables import MODEL, ORDER, _sci, headline_table  # noqa: E402


def test_headline_table_uses_equal_budget_confirmatory_effects(monkeypatch):
    """Protect against relabelling the full 115-vs-60 pool as budget matched."""
    monkeypatch.chdir(ROOT)
    rendered = headline_table()
    data_lines = [line for line in rendered.splitlines() if line != r"\addlinespace"]

    effects = pd.read_csv(
        ROOT / "results/v4/inference_confirmatory/hierarchical_effects.csv"
    )
    effects = effects[
        (effects["variant"] == "budget60")
        & (effects["stratum"] == "all")
        & (effects["scope"].isin(ORDER))
    ].set_index(["scope", "model"])

    assert len(data_lines) == len(ORDER) * len(MODEL)
    for group_index, group in enumerate(ORDER):
        for model_index, model in enumerate(("svc", "gpc")):
            line = data_lines[2 * group_index + model_index]
            effect = effects.loc[(group, model)]
            expected_ci = f"$[{effect.ci_lo:+.4f}, {effect.ci_hi:+.4f}]$"

            assert _sci(effect.effect) in line
            assert expected_ci in line
            assert line.count("&") == 4  # five manuscript columns

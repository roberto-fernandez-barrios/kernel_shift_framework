"""Unit-level checks for the v5 external-grid audit."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analysis.audit_tableshift_v5 import audit_summary  # noqa: E402


def make_summary():
    rows = []
    pools = (
        ("quantum", "svc", "train_cv", 60),
        ("quantum", "svc", "fixed_c1", 60),
        ("quantum", "gpc", "not_applicable", 60),
        ("classical_ext", "svc", "train_cv", 115),
        ("classical_ext", "svc", "fixed_c1", 10),
        ("classical_ext", "gpc", "not_applicable", 115),
    )
    for family, model, regularization, n_candidates in pools:
        for candidate in range(n_candidates):
            dim = (4, 6, 8, 10, 12)[candidate % 5]
            for split in ("validation", "id_test", "ood_test"):
                rows.append(
                    {
                        "family": family, "kernel": f"k{candidate // 5}",
                        "dim": dim, "model": model,
                        "regularization": regularization, "split": split,
                        "balanced_accuracy": 0.5,
                        "kernel_backend": (
                            "cuda" if family == "quantum" else "analytic_cpu"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_complete_summary_passes():
    details = audit_summary(make_summary(), "toy")
    assert details["n_rows"] == 1260


def test_duplicate_summary_fails():
    frame = make_summary()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        audit_summary(frame, "toy")

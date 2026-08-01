"""Tests for finite-shot non-emulability summaries."""
from __future__ import annotations

import pandas as pd

from scripts.analysis.shot_partial_identification_v9 import (
    quantum_stratum,
    summarize_across_groups,
    summarize_by_group,
)


def test_quantum_stratum_uses_actual_zz_prefix():
    assert quantum_stratum("zz_r2_full__as2") == "entangling_zz"
    assert quantum_stratum("pauli_xz_r1_full__as2") == "product_map"
    assert quantum_stratum("zmap_r2__as0.5") == "product_map"


def test_shot_summaries_preserve_fixed_case_units():
    rows = []
    for case in ("a", "b"):
        for replicate in range(3):
            rows.append(
                {
                    "case": case,
                    "run": case,
                    "group": case,
                    "quantum_kernel": "zz_r1_full",
                    "quantum_stratum": "entangling_zz",
                    "tier": "full_115",
                    "shots": 128,
                    "projection_condition": "pre_psd",
                    "self_disagreement_from_exact": 0.1 + 0.01 * replicate,
                    "exact_accuracy_upper": 0.02,
                    "shot_accuracy_upper": 0.04 + 0.01 * replicate,
                    "accuracy_upper_change": 0.02 + 0.01 * replicate,
                    "realized_accuracy_advantage_change": 0.0,
                    "realized_bacc_advantage_change": 0.0,
                    "shot_quantum_ood_bacc": 0.7,
                    "witness_identity_stable": replicate == 0,
                }
            )
    groups = summarize_by_group(pd.DataFrame(rows))
    assert len(groups) == 2
    assert set(groups.n_replicates) == {3}
    across = summarize_across_groups(groups)
    assert len(across) == 1
    assert across.iloc[0].n_fixed_cases == 2
    assert across.iloc[0].across_case_median_median_accuracy_upper_change == 0.03

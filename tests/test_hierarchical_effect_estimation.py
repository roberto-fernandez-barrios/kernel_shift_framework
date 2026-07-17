# tests/test_hierarchical_effect_estimation.py
"""Unit tests for the GATE 2 conditional effect estimation (spec section 5)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analysis.hierarchical_effect_estimation import (  # noqa: E402
    bca_ci, cluster_t_ci, dataset_of, nested_cluster_means, parse_setting)


def make_group(qs_levels=(7, 42, 123), sizes=("1000", "2000"), ms=("42", "123"),
               seeds=(42, 123), base=0.01) -> pd.DataFrame:
    rows = []
    for q in qs_levels:
        for sz in sizes:
            for m in ms:
                for s in seeds:
                    rows.append({"qs": q, "size": sz, "ms": m, "seed": s,
                                 "delta": base + 0.001 * q / 1000})
    return pd.DataFrame(rows)


def test_parse_setting():
    assert parse_setting("m1_hist_byteent__ms123__q1000_id500_ood500") == ("123", "1000")
    assert parse_setting("unsw_dos__natural_cur__ms42__q4000_id1800_ood1800") == ("42", "4000")


def test_dataset_of():
    assert dataset_of("ember_m1") == "ember"
    assert dataset_of("ember_m2") == "ember"
    assert dataset_of("unsw_dos_natural_cur") == "unsw_dos"
    assert dataset_of("unsw_dos_m2_centroid") == "unsw_dos"
    assert dataset_of("toniot_scanning_m2_centroid") == "toniot_scanning"
    assert dataset_of("toniot_scanning_natural_cur") == "toniot_scanning"


def test_nested_means_balanced_equals_plain_mean():
    g = make_group()
    T = nested_cluster_means(g)
    assert len(T) == 3
    for q, t in zip((7, 42, 123), T):
        assert t == pytest.approx(g[g.qs == q].delta.mean())


def test_nested_means_unbalanced_does_not_reweight():
    """Deleting one model-seed run must not shift the cluster mean when the
    remaining cell mean is unchanged (regression for nested vs pooled means)."""
    g = make_group()
    # duplicate one (size, ms) cell many times with the SAME delta value: a
    # pooled mean would overweight it; nested means must be invariant.
    cell = g[(g.qs == 7) & (g["size"] == "1000") & (g.ms == "42")]
    g2 = pd.concat([g] + [cell] * 5, ignore_index=True)
    T1, T2 = nested_cluster_means(g), nested_cluster_means(g2)
    np.testing.assert_allclose(T1, T2, atol=1e-12)


def test_cluster_t_matches_scipy():
    T = np.array([0.01, 0.02, 0.015, 0.005, 0.03])
    eff, lo, hi = cluster_t_ci(T)
    half = stats.t.ppf(0.975, 4) * T.std(ddof=1) / np.sqrt(5)
    assert eff == pytest.approx(T.mean())
    assert lo == pytest.approx(T.mean() - half)
    assert hi == pytest.approx(T.mean() + half)


def test_cluster_t_degenerate():
    T = np.full(5, 0.02)
    eff, lo, hi = cluster_t_ci(T)
    assert eff == lo == hi == pytest.approx(0.02)


def test_bca_degenerate_and_direction():
    rng = np.random.default_rng(0)
    lo, hi, method = bca_ci(np.full(5, 0.02), rng)
    assert method == "degenerate" and lo == hi == pytest.approx(0.02)
    # right-skewed cluster means -> BCa interval shifted vs percentile symmetry
    T = np.array([0.0, 0.001, 0.002, 0.003, 0.05])
    lo, hi, method = bca_ci(T, np.random.default_rng(1))
    assert method == "bca" and lo < T.mean() < hi


def test_output_schema_has_required_columns_and_no_p(tmp_path):
    """End-to-end smoke on a synthetic p1_runs file; asserts the Sol-spec
    metadata columns exist and no p-value column is emitted."""
    runs = []
    for grp, ds in [("ember_m1", "ember"), ("unsw_dos_natural_cur", "unsw_dos")]:
        for qs in (7, 42, 123, 999, 2024):
            for ms in ("42", "123", "999"):
                for size in ("1000", "2000", "4000"):
                    for seed in (42, 123, 999):
                        setting = (f"{'m1_hist_byteent' if ds == 'ember' else 'unsw_dos__natural_cur'}"
                                   f"__ms{ms}__q{size}_id500_ood500")
                        runs.append({"group": grp, "setting": setting, "qs": qs,
                                     "seed": seed, "model": "svc",
                                     "p1_ood_quantum": 0.8, "p1_ood_classical": 0.79,
                                     "delta": 0.01 + 0.0001 * qs / 2024})
    (tmp_path / "p1_runs__full.csv").write_text(pd.DataFrame(runs).to_csv(index=False))
    import subprocess
    r = subprocess.run([sys.executable,
                        str(Path(__file__).resolve().parents[1]
                            / "scripts/analysis/hierarchical_effect_estimation.py"),
                        "--p1-dir", str(tmp_path), "--variants", "full",
                        "--out-dir", str(tmp_path / "out")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    out = pd.read_csv(tmp_path / "out" / "hierarchical_effects.csv")
    for col in ["unit_of_resampling", "n_independent_clusters",
                "n_lower_level_obs", "method", "assumptions"]:
        assert col in out.columns
    assert not any(c == "p" or c.endswith("_p") or "pvalue" in c for c in out.columns)
    assert (out[out.scope == "ember_m1"].n_independent_clusters == 5).all()
    assert "conditional" in out.iloc[0].assumptions.lower()
    assert set(out.scope) >= {"ember_m1", "unsw_dos_natural_cur",
                              "dataset_equal_mean", "heterogeneity"}


def test_requires_five_clusters(tmp_path):
    runs = []
    for qs in (7, 42, 123):   # only 3 clusters -> must fail loudly
        runs.append({"group": "ember_m1",
                     "setting": "m1_hist_byteent__ms42__q1000_id500_ood500",
                     "qs": qs, "seed": 42, "model": "svc",
                     "p1_ood_quantum": 0.8, "p1_ood_classical": 0.79, "delta": 0.01})
    (tmp_path / "p1_runs__full.csv").write_text(pd.DataFrame(runs).to_csv(index=False))
    import subprocess
    r = subprocess.run([sys.executable,
                        str(Path(__file__).resolve().parents[1]
                            / "scripts/analysis/hierarchical_effect_estimation.py"),
                        "--p1-dir", str(tmp_path), "--variants", "full",
                        "--out-dir", str(tmp_path / "out")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "expected 5 qsplit" in (r.stderr + r.stdout)

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results/v10/gate2_prospective")
SPEC = Path("docs/GATE2_PROSPECTIVE_REPLICATION_SPEC_V10.md")
SPEC_SHA256 = "3a8318d92d4af2aeeaf0c0edb069c3be59f31da6d1ee50fb6a6256e9d9d280b0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_frozen_specification_and_aggregate_lock_hashes():
    assert sha256_file(SPEC) == SPEC_SHA256
    lock = ROOT / "aggregate_prediction_lock_manifest.json"
    expected = lock.with_suffix(lock.suffix + ".sha256").read_text().split()[0]
    assert sha256_file(lock) == expected


def test_prediction_lock_precedes_recorded_label_opening():
    lock = json.loads((ROOT / "aggregate_prediction_lock_manifest.json").read_text())
    opened = json.loads((ROOT / "audit/AUDIT_OPENED.json").read_text())
    assert lock["target_labels_opened_for_analysis"] is False
    assert datetime.fromisoformat(lock["created_utc"]) < datetime.fromisoformat(
        opened["opened_utc"]
    )
    assert opened["aggregate_prediction_lock_sha256"] == sha256_file(
        ROOT / "aggregate_prediction_lock_manifest.json"
    )


def test_task_availability_matches_frozen_technical_gate():
    lock = json.loads((ROOT / "aggregate_prediction_lock_manifest.json").read_text())
    assert lock["available_tasks"] == ["brfss_diabetes", "acsfoodstamps"]
    assert lock["unavailable_tasks"] == ["nhanes_lead"]
    assert lock["n_available_units"] == 10
    assert lock["n_technical_unavailability_units"] == 5
    for seed in (42, 123, 999, 7, 2024):
        unit = ROOT / f"prediction_locks/nhanes_lead/seed_{seed}"
        failure = json.loads((unit / "technical_unavailability_manifest.json").read_text())
        assert failure["permitted_gate"] == "minimum-feature failure"
        assert failure["n_model_candidates_executed"] == 0
        cache = unit / "candidate_cache"
        assert not cache.exists() or not list(cache.glob("*.npz"))


def test_every_available_prediction_lock_has_complete_pools():
    for task in ("brfss_diabetes", "acsfoodstamps"):
        for seed in (42, 123, 999, 7, 2024):
            unit = ROOT / f"prediction_locks/{task}/seed_{seed}"
            assert len(list((unit / "candidate_cache").glob("*.npz"))) == 175
            for model in ("svc", "gpc"):
                with np.load(unit / f"predictions_{model}.npz", allow_pickle=False) as data:
                    assert data["quantum_prediction"].shape == (500,)
                    assert data["classical_predictions"].shape == (500, 115)
                    assert int(data["customary_mask"].sum()) == 30


def test_prespecified_strong_transfer_category_recomputes():
    outcome = json.loads((ROOT / "audit/prospective_outcome.json").read_text())
    assert outcome["outcome_category"] == (
        "strong_prospective_transfer__technically_limited_two_task_replication"
    )
    adaptive = pd.read_csv(ROOT / "audit/adaptive_thresholds.csv")
    primary = adaptive[
        adaptive.tier.eq("full_115") & np.isclose(adaptive.threshold, 0.010)
    ]
    task_model = primary.groupby(["task", "model"]).n_labels.median()
    assert len(task_model) == 4
    assert (task_model <= 50).all()
    assert float(primary.n_labels.median()) <= 25
    assert int(primary.n_labels.max()) <= 100
    assert sorted(task_model.tolist()) == [0.0, 0.0, 3.0, 3.0]


def test_prospective_headlines_and_product_map_scope():
    zero = pd.read_csv(ROOT / "audit/zero_label_and_realized.csv")
    zero = zero[zero.tier.eq("full_115")]
    assert len(zero) == 20
    assert int((zero.zero_label_upper <= 0.010 + 1e-12).sum()) == 11
    assert (zero.realized_accuracy_advantage < 0).all()
    assert np.isclose(zero.realized_accuracy_advantage.min(), -0.156)
    assert np.isclose(zero.realized_accuracy_advantage.max(), -0.002)
    assert set(zero.quantum_stratum) == {"product_map"}

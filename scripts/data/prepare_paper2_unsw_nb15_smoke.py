# Copied verbatim from the Paper-2 artifact (paper_2/src/analysis/prepare_paper2_unsw_nb15_smoke.py)
# so the netflow ref/cur pools are reproducible from the public raw datasets.
# Semantics documented in docs/NETFLOW_POOLS.md.
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw/unsw_nb15/Training and Testing Sets")
OUT_DIR = Path("data/processed/unsw_nb15")

TRAIN_PATH = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = RAW_DIR / "UNSW_NB15_testing-set.csv"

SCENARIOS = {
    "dos": "DoS",
    "reconnaissance": "Reconnaissance",
}


def clean_attack_cat(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep numeric UNSW-NB15 flow features only.

    For the first smoke we deliberately avoid one-hot encoding proto/service/state
    to keep the setup close to the CICIDS tabular numeric pipeline and reduce
    preprocessing degrees of freedom.
    """
    drop_cols = {"id", "label", "attack_cat"}
    candidate_cols = [c for c in df.columns if c not in drop_cols]

    numeric = df[candidate_cols].select_dtypes(include=[np.number]).copy()

    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    for col in numeric.columns:
        if numeric[col].isna().any():
            numeric[col] = numeric[col].fillna(numeric[col].median())

    return numeric


def make_binary(df: pd.DataFrame, attack_category: str | None, exclude_attack_from_reference: bool) -> pd.DataFrame:
    df = df.copy()
    df["attack_cat"] = clean_attack_cat(df["attack_cat"])

    if attack_category is None:
        selected = df
    else:
        if exclude_attack_from_reference:
            selected = df[df["attack_cat"] != attack_category].copy()
        else:
            selected = df[df["attack_cat"].isin(["Normal", attack_category])].copy()

    x = prepare_features(selected)

    label = np.where(
        selected["attack_cat"].to_numpy(dtype=str) == "Normal",
        "BENIGN",
        "ATTACK",
    )

    x["Label"] = label
    return x


def summarize(path: Path) -> dict:
    df = pd.read_csv(path)
    label_counts = df["Label"].value_counts().to_dict()
    return {
        "path": str(path),
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "features": int(len(df.columns) - 1),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[LOAD]", TRAIN_PATH)
    train = pd.read_csv(TRAIN_PATH, low_memory=False)

    print("[LOAD]", TEST_PATH)
    test = pd.read_csv(TEST_PATH, low_memory=False)

    train["attack_cat"] = clean_attack_cat(train["attack_cat"])
    test["attack_cat"] = clean_attack_cat(test["attack_cat"])

    print()
    print("[TRAIN attack_cat]")
    print(train["attack_cat"].value_counts().to_string())

    print()
    print("[TEST attack_cat]")
    print(test["attack_cat"].value_counts().to_string())

    manifest = {
        "description": "UNSW-NB15 processed binary scenarios for Paper 2 smoke external validation.",
        "raw_train": str(TRAIN_PATH),
        "raw_test": str(TEST_PATH),
        "feature_policy": "numeric_only_drop_id_label_attack_cat_proto_service_state",
        "scenarios": {},
    }

    for scenario_name, attack_category in SCENARIOS.items():
        print("=" * 100)
        print("[SCENARIO]", scenario_name, "attack_category=", attack_category)

        ref = make_binary(
            train,
            attack_category=attack_category,
            exclude_attack_from_reference=True,
        )

        cur = make_binary(
            test,
            attack_category=attack_category,
            exclude_attack_from_reference=False,
        )

        ref_path = OUT_DIR / f"unsw_ref_no_{scenario_name}_binary.csv"
        cur_path = OUT_DIR / f"unsw_cur_{scenario_name}_binary.csv"

        ref.to_csv(ref_path, index=False)
        cur.to_csv(cur_path, index=False)

        ref_summary = summarize(ref_path)
        cur_summary = summarize(cur_path)

        manifest["scenarios"][scenario_name] = {
            "attack_category": attack_category,
            "reference": ref_summary,
            "current": cur_summary,
        }

        print("[REFERENCE]", ref_summary)
        print("[CURRENT]", cur_summary)

    manifest_path = OUT_DIR / "unsw_nb15_smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 100)
    print("Saved manifest:", manifest_path)


if __name__ == "__main__":
    main()

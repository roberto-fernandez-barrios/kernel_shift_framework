# Copied verbatim from the Paper-2 artifact (paper_2/src/analysis/prepare_paper2_ton_iot_q1_gate.py)
# so the netflow ref/cur pools are reproducible from the public raw datasets.
# Semantics documented in docs/NETFLOW_POOLS.md.
from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/ton_iot/train_test_network.csv")
OUT_DIR = Path("data/processed/ton_iot_q1_gate")

SCENARIOS = {
    "ddos": "ddos",
    "scanning": "scanning",
    "injection": "injection",
}

RANDOM_SEED = 12345

DROP_ALWAYS = {
    "label",
    "type",
    "src_ip",
    "dst_ip",
    "dns_query",
    "ssl_subject",
    "ssl_issuer",
    "http_uri",
    "http_user_agent",
    "http_orig_mime_types",
    "http_resp_mime_types",
    "weird_name",
    "weird_addl",
}

CATEGORICAL_CANDIDATES = [
    "proto",
    "service",
    "conn_state",
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rejected",
    "ssl_version",
    "ssl_cipher",
    "ssl_resumed",
    "ssl_established",
    "http_trans_depth",
    "http_method",
    "http_version",
    "weird_notice",
]

MAX_CARDINALITY = 50


def sanitize_col(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def normalize_type(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def split_by_type(df: pd.DataFrame) -> tuple[set[int], set[int]]:
    """
    Split rows into reference/current pools by type.

    This prevents using the same normal rows in both reference and current.
    Target attack rows are later placed only in current for each scenario.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    ref_idx: set[int] = set()
    cur_idx: set[int] = set()

    for attack_type, group in df.groupby("type_norm"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)

        cut = len(idx) // 2
        ref_idx.update(idx[:cut].tolist())
        cur_idx.update(idx[cut:].tolist())

    return ref_idx, cur_idx


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    numeric_cols = []
    categorical_cols = []
    skipped_categoricals = {}

    for col in df.columns:
        if col in DROP_ALWAYS or col == "type_norm":
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    for col in CATEGORICAL_CANDIDATES:
        if col not in df.columns:
            continue

        nunique = int(df[col].astype(str).str.strip().nunique(dropna=False))
        if nunique <= MAX_CARDINALITY:
            categorical_cols.append(col)
        else:
            skipped_categoricals[col] = nunique

    x_num = df[numeric_cols].copy()
    x_num = x_num.replace([np.inf, -np.inf], np.nan)

    for col in x_num.columns:
        if x_num[col].isna().any():
            med = x_num[col].median()
            if pd.isna(med):
                med = 0.0
            x_num[col] = x_num[col].fillna(med)

    x_num.columns = [sanitize_col(c) for c in x_num.columns]

    if categorical_cols:
        x_cat = df[categorical_cols].copy()
        for col in x_cat.columns:
            x_cat[col] = (
                x_cat[col]
                .astype(str)
                .str.strip()
                .replace({"": "__MISSING__", "nan": "__MISSING__", "None": "__MISSING__"})
            )

        x_ohe = pd.get_dummies(
            x_cat,
            columns=categorical_cols,
            prefix=[sanitize_col(c) for c in categorical_cols],
            dtype=np.uint8,
        )
        x_ohe.columns = [sanitize_col(c) for c in x_ohe.columns]
        x = pd.concat([x_num, x_ohe], axis=1)
    else:
        x = x_num

    info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "skipped_categoricals": skipped_categoricals,
        "n_features": int(x.shape[1]),
    }

    return x, info


def write_scenario(
    df: pd.DataFrame,
    x: pd.DataFrame,
    ref_pool: set[int],
    cur_pool: set[int],
    scenario_name: str,
    target_type: str,
) -> dict:
    normal_idx = set(df.index[df["type_norm"] == "normal"].tolist())
    target_idx = set(df.index[df["type_norm"] == target_type].tolist())

    # Reference:
    # - normal rows from reference pool
    # - non-target attack rows from reference pool
    ref_idx = [
        i for i in sorted(ref_pool)
        if df.at[i, "type_norm"] != target_type
    ]

    # Current:
    # - normal rows from current pool
    # - all target attack rows
    cur_idx = sorted((normal_idx.intersection(cur_pool)).union(target_idx))

    ref = x.loc[ref_idx].copy()
    cur = x.loc[cur_idx].copy()

    ref["Label"] = np.where(
        df.loc[ref_idx, "type_norm"].to_numpy(dtype=str) == "normal",
        "BENIGN",
        "ATTACK",
    )

    cur["Label"] = np.where(
        df.loc[cur_idx, "type_norm"].to_numpy(dtype=str) == "normal",
        "BENIGN",
        "ATTACK",
    )

    ref_path = OUT_DIR / f"ton_iot_ref_no_{scenario_name}_binary.csv"
    cur_path = OUT_DIR / f"ton_iot_cur_{scenario_name}_binary.csv"

    ref.to_csv(ref_path, index=False)
    cur.to_csv(cur_path, index=False)

    return {
        "scenario": scenario_name,
        "target_type": target_type,
        "reference_path": str(ref_path),
        "current_path": str(cur_path),
        "reference_rows": int(len(ref)),
        "current_rows": int(len(cur)),
        "reference_label_counts": {str(k): int(v) for k, v in ref["Label"].value_counts().to_dict().items()},
        "current_label_counts": {str(k): int(v) for k, v in cur["Label"].value_counts().to_dict().items()},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[LOAD]", RAW_PATH)
    df = pd.read_csv(RAW_PATH, low_memory=False)

    if "label" not in df.columns or "type" not in df.columns:
        raise ValueError("Expected columns 'label' and 'type' were not found.")

    df["type_norm"] = normalize_type(df["type"])

    print()
    print("[TYPE COUNTS]")
    print(df["type_norm"].value_counts().to_string())

    print()
    print("[LABEL COUNTS]")
    print(df["label"].value_counts().to_string())

    print()
    print("[BUILD FEATURES]")
    x, feature_info = build_feature_matrix(df)

    print("[FEATURE INFO]")
    print(json.dumps(feature_info, indent=2))

    print()
    print("[SPLIT]")
    ref_pool, cur_pool = split_by_type(df)
    print("ref_pool:", len(ref_pool))
    print("cur_pool:", len(cur_pool))

    manifest = {
        "description": "ToN-IoT Network Q1-gate processed binary scenarios for Paper 2.",
        "raw_path": str(RAW_PATH),
        "split_policy": "per-type 50/50 split; target attack only in current; normal split avoids reference/current overlap",
        "feature_policy": "numeric plus selected low-cardinality categorical one-hot; high-cardinality identifiers dropped",
        "feature_info": feature_info,
        "type_counts": {str(k): int(v) for k, v in df["type_norm"].value_counts().to_dict().items()},
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "scenarios": {},
    }

    for scenario_name, target_type in SCENARIOS.items():
        print("=" * 100)
        print("[SCENARIO]", scenario_name, "target=", target_type)

        summary = write_scenario(
            df=df,
            x=x,
            ref_pool=ref_pool,
            cur_pool=cur_pool,
            scenario_name=scenario_name,
            target_type=target_type,
        )

        manifest["scenarios"][scenario_name] = summary
        print(json.dumps(summary, indent=2))

    manifest_path = OUT_DIR / "ton_iot_q1_gate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 100)
    print("Saved manifest:", manifest_path)


if __name__ == "__main__":
    main()

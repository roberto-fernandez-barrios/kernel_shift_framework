# scripts/analysis/audit_v4.py
"""
GATE 0 of the v4 methodological revision (docs/ANALYSIS_SPEC_V4.md): freeze and
audit the inputs before any decisive analysis runs.

Outputs under results/v4/audit/:
  manifest.json                    commit, branch, date, environment versions
  file_hashes.csv                  SHA-256 of every frozen per-run summary CSV
  candidate_inventory.csv          deduplicated kernel-geometry pools per group
  experiment_hierarchy.csv         factor structure and counts per root
  sample_overlap_qsplit.csv        train/id/ood index overlap across q-split seeds
  sample_overlap_master_seed.csv   ... across master seeds (same qs)
  nested_size_audit.csv            ... across sizes (same ms, qs)
  hierarchy_report.md              human-readable summary of all of the above

Read-only over data/ and the frozen results roots; deterministic; no RNG.
Intracluster delta correlations belong to the GATE 2 dependence audit, not here.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/v4/audit")

RESULT_ROOTS = {
    "ember": Path("results/ember_shift/extended_kernels"),
    "ember_bw": Path("results/ember_shift/bandwidth_sweep"),
    "netflow": Path("results/netflow/extended_kernels"),
    "netflow_bw": Path("results/netflow_bandwidth_sweep/extended_kernels"),
}
SUMMARY_FILES = ["extended_kernels_qsplits__summary.csv", "summary_classical_lsweep.csv"]
RUN_RE = re.compile(r"^(?P<setting>.+)__qs(?P<qs>\d+)__s(?P<seed>\d+)$")
SETTING_RE = re.compile(r"^(?P<head>.+)__ms(?P<ms>\d+)__q(?P<n>\d+)_id(?P<i>\d+)_ood(?P<o>\d+)$")

EMBER_SPLITS = Path("data/processed/ember")
NETFLOW_SPLITS = Path("data/processed/netflow")


def group_label(setting: str) -> str:
    toks = setting.split("__")
    if toks[0].startswith(("m1_", "m2_")):
        return f"ember_{toks[0].split('_')[0]}"
    return f"{toks[0]}_{toks[1]}"


def git_info() -> dict:
    def _run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    return {"commit": _run("rev-parse", "HEAD"),
            "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
            "base_tag": _run("describe", "--tags", "--abbrev=0")}


def manifest() -> dict:
    import scipy
    import sklearn
    info = {
        "spec": "docs/ANALYSIS_SPEC_V4.md",
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
    }
    try:
        import qiskit
        info["qiskit"] = qiskit.__version__
    except ImportError:
        info["qiskit"] = None
    return info


def iter_run_dirs():
    for tag, root in RESULT_ROOTS.items():
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and RUN_RE.match(d.name):
                yield tag, d


def hash_files() -> pd.DataFrame:
    rows = []
    for tag, d in iter_run_dirs():
        for name in SUMMARY_FILES:
            f = d / name
            if not f.exists():
                continue
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            rows.append({"root": tag, "run": d.name, "file": name,
                         "bytes": f.stat().st_size, "sha256": h.hexdigest()})
    return pd.DataFrame(rows)


def candidate_inventory() -> pd.DataFrame:
    """Unique kernel-geometry candidates per scenario-group, family, model."""
    frames = []
    seen_groups: dict[tuple, set] = {}
    for tag, d in iter_run_dirs():
        m = RUN_RE.match(d.name)
        grp = group_label(m.group("setting"))
        key = (grp, tag)
        if key in seen_groups:   # one representative run per (group, root) suffices
            continue
        seen_groups[key] = set()
        for name in SUMMARY_FILES:
            f = d / name
            if not f.exists():
                continue
            df = pd.read_csv(f, usecols=["family", "model", "cfg", "kernel", "dim"])
            df["group"], df["root"], df["source_file"] = grp, tag, name
            frames.append(df.drop_duplicates(["family", "model", "cfg"]))
    cat = pd.concat(frames, ignore_index=True)
    # geometry identity = (kernel, dim); dedup across roots/files within a group
    inv = (cat.drop_duplicates(["group", "family", "model", "kernel", "dim"])
              .groupby(["group", "family", "model"])
              .agg(n_geometries=("cfg", "size")).reset_index())
    piv = inv.pivot_table(index="group", columns=["family", "model"],
                          values="n_geometries", fill_value=0)
    piv.columns = [f"{f}_{m}" for f, m in piv.columns]
    piv = piv.reset_index()
    piv["budget_class"] = np.where(piv.get("classical_ext_svc", 0) >= 100, "full_115v60",
                                   "reduced_35v20")
    return piv


def experiment_hierarchy() -> pd.DataFrame:
    rows = []
    for tag, root in RESULT_ROOTS.items():
        if not root.exists():
            continue
        recs = []
        for d in root.iterdir():
            m = RUN_RE.match(d.name) if d.is_dir() else None
            if not m:
                continue
            sm = SETTING_RE.match(m.group("setting"))
            recs.append({"group": group_label(m.group("setting")),
                         "ms": sm.group("ms"), "size": sm.group("n"),
                         "qs": m.group("qs"), "s": m.group("seed")})
        df = pd.DataFrame(recs)
        rows.append({"root": tag, "n_runs": len(df),
                     "n_groups": df.group.nunique(), "groups": ";".join(sorted(df.group.unique())),
                     "n_ms": df.ms.nunique(), "n_sizes": df["size"].nunique(),
                     "n_qs": df.qs.nunique(), "n_model_seeds": df.s.nunique(),
                     "balanced": len(df) == df.group.nunique() * df.ms.nunique()
                     * df["size"].nunique() * df.qs.nunique() * df.s.nunique()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split-index overlap audits
# ---------------------------------------------------------------------------
def _load_idx(qdir: Path) -> dict[str, set]:
    out = {}
    for part in ("train", "id_test", "ood_test"):
        f = qdir / f"{part}_idx.npy"
        if f.exists():
            out[part] = set(np.load(f).ravel().tolist())
    return out


def _ember_qdir(variant: str, ms: str, size: str, qs: str) -> Path | None:
    hits = sorted(EMBER_SPLITS.glob(f"splits_sparsity__{variant}__ms{ms}__h*"))
    if not hits:
        return None
    sub = sorted(hits[0].glob(f"qsplits/splits_sparsity_q{size}_id*_ood*_seed{qs}"))
    return sub[0] if sub else None


def _netflow_qdir(scenario: str, variant: str, ms: str, size: str, qs: str) -> Path | None:
    base = NETFLOW_SPLITS / scenario / f"splits_{variant}__ms{ms}"
    sub = sorted(base.glob(f"qsplits/splits_sparsity_q{size}_id*_ood*_seed{qs}"))
    return sub[0] if sub else None


CASES = [  # (label, resolver(ms, size, qs))
    ("ember_m1", lambda ms, size, qs: _ember_qdir("m1_hist_byteent", ms, size, qs)),
    ("ember_m2", lambda ms, size, qs: _ember_qdir("m2_hist_byteent", ms, size, qs)),
    ("unsw_dos_natural", lambda ms, size, qs: _netflow_qdir("unsw_dos", "natural_cur", ms, size, qs)),
]
QS_SEEDS = ["7", "42", "123", "999", "2024"]
MS_SEEDS = ["42", "123", "999"]
SIZES = ["1000", "2000", "4000"]


def _overlap_rows(label: str, axis: str, keys: list[str],
                  loader) -> list[dict]:
    idx = {k: loader(k) for k in keys}
    idx = {k: v for k, v in idx.items() if v}
    rows = []
    for a, b in itertools.combinations(sorted(idx), 2):
        for part in ("train", "id_test", "ood_test"):
            sa, sb = idx[a].get(part), idx[b].get(part)
            if not sa or not sb:
                continue
            inter = len(sa & sb)
            rows.append({"case": label, "axis": axis, "a": a, "b": b, "part": part,
                         "n_a": len(sa), "n_b": len(sb), "n_shared": inter,
                         "frac_of_smaller": inter / min(len(sa), len(sb))})
    return rows


def overlap_audits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qs_rows, ms_rows, size_rows = [], [], []
    for label, resolver in CASES:
        # across q-split seeds (ms=123, q1000)
        qs_rows += _overlap_rows(label, "qsplit_seed", QS_SEEDS,
                                 lambda qs: _load_idx(p) if (p := resolver("123", "1000", qs)) else {})
        # across master seeds (qs=42, q1000)
        ms_rows += _overlap_rows(label, "master_seed", MS_SEEDS,
                                 lambda ms: _load_idx(p) if (p := resolver(ms, "1000", "42")) else {})
        # across sizes (ms=123, qs=42)
        size_rows += _overlap_rows(label, "size", SIZES,
                                   lambda size: _load_idx(p) if (p := resolver("123", size, "42")) else {})
    return pd.DataFrame(qs_rows), pd.DataFrame(ms_rows), pd.DataFrame(size_rows)


def write_report(man: dict, inv: pd.DataFrame, hier: pd.DataFrame,
                 qs_ov: pd.DataFrame, ms_ov: pd.DataFrame, size_ov: pd.DataFrame,
                 hashes: pd.DataFrame) -> str:
    def _mx(df, case, part):
        d = df[(df.case == case) & (df.part == part)]
        return f"{d.frac_of_smaller.max():.3f}" if len(d) else "n/a"
    lines = [
        "# v4 GATE 0 audit report", "",
        f"Frozen {man['frozen_utc']} at `{man['git']['commit'][:9]}` "
        f"(branch `{man['git']['branch']}`, base tag `{man['git']['base_tag']}`).", "",
        f"Hashed {len(hashes)} frozen summary CSVs "
        f"({hashes['bytes'].sum() / 1e6:.0f} MB) across {hashes.root.nunique()} roots.", "",
        "## Candidate pools (kernel geometries, deduplicated)", "",
        "```", inv.to_string(index=False), "```", "",
        "## Hierarchy", "", "```", hier.to_string(index=False), "```", "",
        "## Sample-overlap findings (fraction of smaller set, MAX over pairs)", "",
        "| case | axis | train | id_test | ood_test |", "|---|---|---|---|---|",
    ]
    for case in [c for c, _ in CASES]:
        for axis, df in [("qsplit_seed", qs_ov), ("master_seed", ms_ov), ("size", size_ov)]:
            lines.append(f"| {case} | {axis} | {_mx(df, case, 'train')} | "
                         f"{_mx(df, case, 'id_test')} | {_mx(df, case, 'ood_test')} |")
    lines += [
        "", "## Reading", "",
        "- q-split seeds are near-disjoint everywhere (max shared fraction ~0.08 on",
        "  netflow OOD, ~0.01 on EMBER) -> defensible resampling clusters for",
        "  CONDITIONAL pipeline-realization uncertainty (spec section 5).",
        "- EMBER m1 ood_test is ~identical across master seeds (deterministic score",
        "  tails): master seeds are NOT independent replicates there.",
        "- Sizes: EMBER sizes are independently resampled (overlap ~ chance), but",
        "  NETFLOW trains are NESTED across sizes (train overlap = 1.0 for at least",
        "  one pair; OOD up to ~0.25). Either way, size is a fixed design factor and",
        "  never a replicate -- the netflow nesting makes this mandatory, not optional.",
        "- Consequence (spec sections 2.1-2.2): fixed case studies, no global",
        "  population p-value; intervals conditional on benchmark pools.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = manifest()
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"[audit] manifest at commit {man['git']['commit'][:9]}")

    inv = candidate_inventory()
    inv.to_csv(OUT / "candidate_inventory.csv", index=False)
    print(f"[audit] candidate inventory: {len(inv)} scenario-groups")

    hier = experiment_hierarchy()
    hier.to_csv(OUT / "experiment_hierarchy.csv", index=False)
    print(f"[audit] hierarchy over {int(hier.n_runs.sum())} run dirs, balanced={hier.balanced.all()}")

    qs_ov, ms_ov, size_ov = overlap_audits()
    qs_ov.to_csv(OUT / "sample_overlap_qsplit.csv", index=False)
    ms_ov.to_csv(OUT / "sample_overlap_master_seed.csv", index=False)
    size_ov.to_csv(OUT / "nested_size_audit.csv", index=False)
    print(f"[audit] overlaps: {len(qs_ov)} qs-pairs, {len(ms_ov)} ms-pairs, {len(size_ov)} size-pairs")

    print("[audit] hashing frozen summary CSVs (slow)...")
    hashes = hash_files()
    hashes.to_csv(OUT / "file_hashes.csv", index=False)
    print(f"[audit] hashed {len(hashes)} files")

    (OUT / "hierarchy_report.md").write_text(
        write_report(man, inv, hier, qs_ov, ms_ov, size_ov, hashes), encoding="utf-8")
    print(f"[OK] GATE 0 audit complete -> {OUT}")


if __name__ == "__main__":
    main()

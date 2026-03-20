# scripts/ember/run_compare_q_vs_c_full.py
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("Este script requiere pandas. Instala con: pip install pandas") from e


# =============================================================================
# Defaults / Config
# =============================================================================
DEFAULT_QSPLIT_SEEDS = [42, 123, 999, 7, 2024]
DEFAULT_MODEL_SEEDS = [42, 123, 999]
DEFAULT_MASTER_SEEDS = [42, 123, 999]

DEFAULT_N_TRAIN = 1000
DEFAULT_N_ID = 500
DEFAULT_N_OOD = 500

DEFAULT_DIMS = [4, 6, 8, 10, 12]

SIZE_PRESETS: Dict[str, List[Tuple[int, int, int]]] = {
    "baseline": [(1000, 500, 500)],
    "minimal": [(1000, 500, 500), (2000, 1000, 1000)],
    "paper": [(1000, 500, 500), (2000, 1000, 1000), (4000, 1800, 1800)],
    "ood_only": [(1000, 500, 500), (1000, 500, 1000), (1000, 500, 1800)],
}
DEFAULT_SIZES_PRESET = "paper"

DEFAULT_USE_THRESHOLDING = False
DEFAULT_THRESH_SOURCE = "train"  # train | id_test
DEFAULT_THRESH_CRITERION = "balanced_accuracy"  # balanced_accuracy | f1_pos
DEFAULT_THRESH_GRID = 401

DEFAULT_USE_PARALLEL = False
DEFAULT_WORKERS = 2

DEFAULT_SKIP_QSPLITS_IF_EXISTS = True
DEFAULT_CLEAN_RESULTS_BEFORE_RUN = False
DEFAULT_FAIL_IF_SPLITS_MISMATCH = True
DEFAULT_SKIP_RUN_IF_SUMMARY_EXISTS = True

DEFAULT_CLASSICAL_KERNEL_NORMALIZE = True
DEFAULT_BEST_CRITERION = "tradeoff"  # tradeoff | robust | ood

DEFAULT_WRITE_GLOBAL_JSON = True
DEFAULT_GLOBAL_JSON_FILENAME = "GLOBAL_results_simple.json"

# ---- Paths (EMBER) ----
DEFAULT_SPLITS_ROOT_DIR = Path("data/processed/ember")
DEFAULT_RESULTS_ROOT_DIR = Path("results/ember_shift/compare_q_vs_c")
DEFAULT_LOG_ROOT_DIR = Path("results/_logs_compare_q_vs_c_full_ember")

# ---- Modules (EMBER) ----
DEFAULT_MAKE_MASTER_SPLITS_MOD = "src.utils.ember.make_splits_ember_sparsity"
DEFAULT_MAKE_QSPLITS_MOD = "src.utils.ember.make_qsplits_from_master"
DEFAULT_RUN_Q_MOD = "src.experiments.ember.quantum.run_ember_quantum_kernel_sparsity_shift_qsplits"
DEFAULT_RUN_C_MOD = "src.experiments.ember.classical.run_ember_classical_kernel_sparsity_shift_qsplits"

# ---- Master splits variants (OOD protocols) ----
DEFAULT_MASTER_TAG_PREFIX = "splits_sparsity__"

MASTER_VARIANTS: Dict[str, Dict[str, object]] = {
    "m1_hist_byteent": {
        "ood_mode": "score_extremes_within_class",
        "use_hist": True,
        "use_byteent": True,
        "ood_test_frac": 0.15,
        "ood_extreme_frac_each_side": 0.075,
        "score_mode": "nnz",
        "eps": 0.0,
        "require_low_high": True,
    },
    "m2_hist_byteent": {
        "ood_mode": "dist_to_train_within_class",
        "use_hist": True,
        "use_byteent": True,
        "ood_test_frac": 0.15,
        "svd_dim": 128,
        "centroid_robust": False,
        "require_low_high": False,
    },
}

DEFAULT_VARIANTS = list(MASTER_VARIANTS.keys())

# =============================================================================
# Logging / Execution defaults
# =============================================================================
DEFAULT_CONTINUE_ON_ERROR = False
DEFAULT_LOG_TAIL_LINES = 200  # si falla, adjunta las últimas N líneas del log al error


# =============================================================================
# Helpers
# =============================================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _stable_json_dumps(x: object) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cfg_hash_short(d: Dict[str, object], n: int = 8) -> str:
    h = hashlib.sha256(_stable_json_dumps(d).encode("utf-8")).hexdigest()
    return h[:n]


def qsplit_name(n_train: int, n_id: int, n_ood: int, seed: int) -> str:
    return f"splits_sparsity_q{n_train}_id{n_id}_ood{n_ood}_seed{seed}"


def size_tag(n_train: int, n_id: int, n_ood: int) -> str:
    return f"size_q{n_train}_id{n_id}_ood{n_ood}"


def parse_size_triplet(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Formato inválido '{s}'. Usa n_train,n_id,n_ood (ej: 1000,500,500).")
    try:
        a, b, c = (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        raise argparse.ArgumentTypeError(f"Formato inválido '{s}'. Deben ser enteros.")
    if a <= 0 or b <= 0 or c <= 0:
        raise argparse.ArgumentTypeError(f"Formato inválido '{s}'. Deben ser >0.")
    return (a, b, c)


def resolve_size_grid(args: argparse.Namespace) -> List[Tuple[int, int, int]]:
    if args.size_grid and len(args.size_grid) > 0:
        return [parse_size_triplet(x) for x in args.size_grid]

    if args.sizes_preset:
        preset = args.sizes_preset.strip()
        if preset not in SIZE_PRESETS:
            raise SystemExit(f"--sizes-preset '{preset}' no existe. Opciones: {sorted(SIZE_PRESETS.keys())}")
        return SIZE_PRESETS[preset]

    if DEFAULT_SIZES_PRESET in SIZE_PRESETS:
        return SIZE_PRESETS[DEFAULT_SIZES_PRESET]

    return [(int(args.n_train), int(args.n_id), int(args.n_ood))]


def split_sizes_from_dir(splits_dir: Path) -> Dict[str, Optional[int]]:
    files = ["train_idx.npy", "id_test_idx.npy", "ood_test_idx.npy", "ood_low_idx.npy", "ood_high_idx.npy"]
    out: Dict[str, Optional[int]] = {}
    for f in files:
        p = splits_dir / f
        if not p.exists():
            out[f] = None
            continue
        try:
            arr = np.load(p, mmap_mode="r")
            out[f] = int(arr.size)
        except Exception:
            out[f] = None
    return out


def load_meta_q(splits_dir: Path) -> Optional[dict]:
    p = splits_dir / "meta_q.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def _extract_sizes_from_meta_q(meta: dict) -> Optional[Dict[str, int]]:
    if not isinstance(meta, dict):
        return None
    sizes = meta.get("actual_sizes") or meta.get("requested_sizes") or meta.get("sizes") or {}
    if not isinstance(sizes, dict):
        return None

    n_train = sizes.get("n_train")
    n_id = sizes.get("n_id") or sizes.get("n_id_test")
    n_ood = sizes.get("n_ood") or sizes.get("n_ood_test")
    if n_train is None or n_id is None or n_ood is None:
        return None

    try:
        return {"n_train": int(n_train), "n_id": int(n_id), "n_ood": int(n_ood)}
    except Exception:
        return None


def assert_qsplit_valid(
    splits_dir: Path,
    expected_train: int,
    expected_id: int,
    expected_ood: int,
    expected_low: int,
    expected_high: int,
    fail_if_mismatch: bool,
    require_low_high: bool,
    context: str = "",
) -> bool:
    if not splits_dir.exists():
        msg = f"No existe splits-dir: {splits_dir}"
        if fail_if_mismatch:
            raise FileNotFoundError(msg)
        print(f"[WARN] {msg}")
        return False

    sizes = split_sizes_from_dir(splits_dir)
    t = sizes["train_idx.npy"]
    i = sizes["id_test_idx.npy"]
    o = sizes["ood_test_idx.npy"]
    lo = sizes["ood_low_idx.npy"]
    hi = sizes["ood_high_idx.npy"]

    print(f"[SPLITS] {splits_dir} train={t} id={i} ood={o} low={lo} high={hi} {context}")

    ok = True
    errs: List[str] = []

    if t != expected_train:
        ok = False
        errs.append(f"train_idx.npy size={t} expected={expected_train}")
    if i != expected_id:
        ok = False
        errs.append(f"id_test_idx.npy size={i} expected={expected_id}")
    if o != expected_ood:
        ok = False
        errs.append(f"ood_test_idx.npy size={o} expected={expected_ood}")

    if require_low_high:
        if lo != expected_low:
            ok = False
            errs.append(f"ood_low_idx.npy size={lo} expected={expected_low}")
        if hi != expected_high:
            ok = False
            errs.append(f"ood_high_idx.npy size={hi} expected={expected_high}")

    meta = load_meta_q(splits_dir)
    if meta is None:
        print(f"[WARN] meta_q.json no encontrado en {splits_dir} (no es fatal).")
    else:
        ms = _extract_sizes_from_meta_q(meta)
        if ms is None:
            print(f"[WARN] meta_q.json existe pero no se pudo validar (estructura inesperada).")
        else:
            if int(ms["n_train"]) != expected_train:
                ok = False
                errs.append(f"meta_q.n_train={ms['n_train']} expected={expected_train}")
            if int(ms["n_id"]) != expected_id:
                ok = False
                errs.append(f"meta_q.n_id={ms['n_id']} expected={expected_id}")
            if int(ms["n_ood"]) != expected_ood:
                ok = False
                errs.append(f"meta_q.n_ood={ms['n_ood']} expected={expected_ood}")

    if not ok:
        msg = f"[SPLITS-MISMATCH] {splits_dir} :: " + " | ".join(errs)
        if fail_if_mismatch:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")
        return False

    return True


def _read_tail_lines(path: Path, n: int) -> str:
    if n <= 0:
        return ""
    try:
        # lectura eficiente del final para logs medianos: (suficiente para debug)
        txt = path.read_text(encoding="utf-8", errors="replace")
        lines = txt.splitlines()
        tail = lines[-n:] if len(lines) > n else lines
        return "\n".join(tail)
    except Exception:
        return ""


def run_cmd_to_log(
    title: str,
    log_path: Path,
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    *,
    tail_lines_on_error: int = DEFAULT_LOG_TAIL_LINES,
) -> None:
    """
    Ejecuta cmd y guarda un log EN STREAMING (no captura stdout/stderr entero en RAM).
    - Mucho más robusto cuando los scripts escupen mucho output.
    - En error, el RuntimeError incluye el log y el tail para debug rápido.
    """
    ensure_dir(log_path.parent)

    printable_cmd = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n[RUN] {title}")
    print(f"      Log: {log_path}")
    print(f"      Cmd: {printable_cmd}")

    if dry_run:
        return

    merged_env = os.environ.copy()
    merged_env["PYTHONIOENCODING"] = "utf-8"
    if env:
        merged_env.update(env)

    # Escribimos cabecera y luego streameamos stdout/stderr al mismo archivo.
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(f"===== {now_str()} =====\n")
        f.write(f"[TITLE] {title}\n")
        f.write(f"[CMD] {printable_cmd}\n")
        f.write("----- STREAM (stdout+stderr) -----\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=merged_env,
            shell=False,
            bufsize=1,  # line-buffered (best-effort)
        )

        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
        p.wait()

        f.write("\n")
        f.write(f"===== EXITCODE: {p.returncode} =====\n")

    if p.returncode != 0:
        tail = _read_tail_lines(log_path, tail_lines_on_error)
        extra = f"\n\n----- LOG TAIL (last {tail_lines_on_error} lines) -----\n{tail}\n" if tail else ""
        raise RuntimeError(f"Comando falló (exitcode={p.returncode}). Revisa: {log_path}{extra}")


def has_summary_csv(out_dir: Path) -> bool:
    return any(out_dir.glob("*__summary.csv"))


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def write_manifest(results_root: Path, args_namespace: argparse.Namespace, extra: Optional[dict] = None) -> None:
    ensure_dir(results_root)
    manifest = {
        "timestamp": now_str(),
        "python": sys.executable,
        "argv": sys.argv,
        "args": vars(args_namespace),
        "extra": extra or {},
        "git": {},
    }

    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
        if r.returncode == 0:
            manifest["git"]["commit"] = r.stdout.strip()
    except Exception:
        pass

    out = results_root / "MANIFEST_compare_q_vs_c.json"
    out.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(f"[OK] wrote {out}")


# =============================================================================
# Master splits orchestration
# =============================================================================
def ensure_master_splits(
    *,
    variant_name: str,
    variant: Dict[str, object],
    make_master_mod: str,
    in_dir: Path,
    out_root: Path,
    tag_prefix: str,
    master_seed: int,
    dry_run: bool,
    force: bool,
    log_root: Path,
) -> Path:
    """
    Genera (o reusa) splits maestro para (variante + master_seed + config-hash).
    Devuelve la carpeta de master splits.
    """
    vhash = cfg_hash_short(variant, n=8)
    master_dir = out_root / f"{tag_prefix}{variant_name}__ms{int(master_seed)}__h{vhash}"

    ok = (master_dir / "meta.json").exists() and (master_dir / "train_idx.npy").exists() and (master_dir / "ood_test_idx.npy").exists()
    if ok and not force:
        print(f"[OK] Master splits existen -> {master_dir}")
        return master_dir

    ensure_dir(master_dir)
    ensure_dir(log_root)

    ood_mode = str(variant["ood_mode"])
    use_hist = bool(variant.get("use_hist", False))
    use_byteent = bool(variant.get("use_byteent", False))

    cmd = [
        sys.executable,
        "-m",
        make_master_mod,
        "--in-dir",
        str(in_dir),
        "--out-dir",
        str(master_dir),
        "--seed",
        str(int(master_seed)),
        "--ood-mode",
        ood_mode,
        "--mmap",
        "--strict-fracs",
    ]

    if use_hist:
        cmd += ["--use-hist"]
    if use_byteent:
        cmd += ["--use-byteent"]

    if ood_mode == "score_extremes_within_class":
        ood_test_frac = float(variant.get("ood_test_frac", 0.15))
        extreme = float(variant.get("ood_extreme_frac_each_side", ood_test_frac / 2.0))
        score_mode = str(variant.get("score_mode", "nnz"))
        eps = float(variant.get("eps", 0.0))
        cmd += [
            "--ood-test-frac",
            str(ood_test_frac),
            "--ood-extreme-frac-each-side",
            str(extreme),
            "--score-mode",
            score_mode,
            "--eps",
            str(eps),
        ]
    else:
        ood_test_frac = float(variant.get("ood_test_frac", 0.15))
        svd_dim = int(variant.get("svd_dim", 128))
        centroid_robust = bool(variant.get("centroid_robust", False))
        cmd += [
            "--ood-test-frac",
            str(ood_test_frac),
            "--svd-dim",
            str(svd_dim),
            "--save-provisional",
        ]
        if centroid_robust:
            cmd += ["--centroid-robust"]

    log = log_root / f"_master_{variant_name}__ms{int(master_seed)}__h{vhash}.log"
    run_cmd_to_log(
        title=f"make_master_splits {variant_name} (ms={master_seed})",
        log_path=log,
        cmd=cmd,
        dry_run=dry_run,
    )

    if not dry_run and not (master_dir / "meta.json").exists():
        raise RuntimeError(f"Master splits no generados correctamente (falta meta.json) en {master_dir}")

    return master_dir


# =============================================================================
# Core execution (one combo)
# =============================================================================
@dataclass(frozen=True)
class RunConfig:
    splits_root: Path
    results_root: Path
    log_root: Path
    n_train: int
    n_id: int
    n_ood: int
    expected_low: int
    expected_high: int
    dims: List[int]
    use_thresholding: bool
    thresh_source: str
    thresh_criterion: str
    thresh_grid: int
    classical_kernel_normalize: bool
    skip_run_if_summary_exists: bool
    run_q_mod: str
    run_c_mod: str
    fail_if_splits_mismatch: bool
    require_low_high: bool
    dry_run: bool
    continue_on_error: bool


def run_one_combo(cfg: RunConfig, qsplit_seed: int, model_seed: int) -> None:
    qname = qsplit_name(cfg.n_train, cfg.n_id, cfg.n_ood, qsplit_seed)
    splits_dir = cfg.splits_root / qname

    # En dry-run queremos previsualizar los comandos del grid aunque los q-splits
    # todavía no existan. La validación estricta de ficheros reales solo debe
    # ejecutarse en corridas no-dry.
    if not cfg.dry_run:
        assert_qsplit_valid(
            splits_dir=splits_dir,
            expected_train=cfg.n_train,
            expected_id=cfg.n_id,
            expected_ood=cfg.n_ood,
            expected_low=cfg.expected_low,
            expected_high=cfg.expected_high,
            fail_if_mismatch=cfg.fail_if_splits_mismatch,
            require_low_high=cfg.require_low_high,
            context=f"(before-train qsplit={qsplit_seed} model-seed={model_seed})",
        )
    else:
        print(
            f"[DRY] Skip q-split validation before train -> {splits_dir} "
            f"(qsplit={qsplit_seed}, model-seed={model_seed})"
        )

    dims_args = ["--dims"] + [str(d) for d in cfg.dims]

    if cfg.use_thresholding:
        thr_args = [
            "--thresh-source",
            cfg.thresh_source,
            "--thresh-criterion",
            cfg.thresh_criterion,
            "--thresh-grid",
            str(cfg.thresh_grid),
        ]
    else:
        thr_args = ["--no-thresholding"]

    classical_fair_args: List[str] = []
    if not cfg.classical_kernel_normalize:
        classical_fair_args += ["--no-kernel-normalize"]

    # Quantum
    out_q = cfg.results_root / f"qsplit{qsplit_seed}" / f"seed{model_seed}" / "quantum"
    ensure_dir(out_q)
    log_q = cfg.log_root / f"quantum_qsplit{qsplit_seed}_seed{model_seed}.log"

    if cfg.skip_run_if_summary_exists and has_summary_csv(out_q):
        print(f"[SKIP] QUANTUM already has summary -> {out_q}")
    else:
        cmd_q = [
            sys.executable,
            "-m",
            cfg.run_q_mod,
            "--splits-dir",
            str(splits_dir),
            "--out-dir",
            str(out_q),
            "--seed",
            str(model_seed),
        ] + dims_args + thr_args

        try:
            run_cmd_to_log(
                title=f"QUANTUM qsplit={qsplit_seed} model-seed={model_seed}",
                log_path=log_q,
                cmd=cmd_q,
                dry_run=cfg.dry_run,
            )
        except Exception as e:
            msg = f"[FAIL] QUANTUM qsplit={qsplit_seed} seed={model_seed} :: {e}"
            if cfg.continue_on_error:
                print(msg)
            else:
                raise

    # Classical
    out_c = cfg.results_root / f"qsplit{qsplit_seed}" / f"seed{model_seed}" / "classical"
    ensure_dir(out_c)
    log_c = cfg.log_root / f"classical_qsplit{qsplit_seed}_seed{model_seed}.log"

    if cfg.skip_run_if_summary_exists and has_summary_csv(out_c):
        print(f"[SKIP] CLASSICAL already has summary -> {out_c}")
    else:
        cmd_c = [
            sys.executable,
            "-m",
            cfg.run_c_mod,
            "--splits-dir",
            str(splits_dir),
            "--out-dir",
            str(out_c),
            "--seed",
            str(model_seed),
        ] + dims_args + thr_args + classical_fair_args

        try:
            run_cmd_to_log(
                title=f"CLASSICAL qsplit={qsplit_seed} model-seed={model_seed} (kernel_normalize={cfg.classical_kernel_normalize})",
                log_path=log_c,
                cmd=cmd_c,
                dry_run=cfg.dry_run,
            )
        except Exception as e:
            msg = f"[FAIL] CLASSICAL qsplit={qsplit_seed} seed={model_seed} :: {e}"
            if cfg.continue_on_error:
                print(msg)
            else:
                raise


# =============================================================================
# Aggregation + Context indexing (robust)
# =============================================================================
RX_SIZE = re.compile(r"size_q(?P<n_train>\d+)_id(?P<n_id>\d+)_ood(?P<n_ood>\d+)", re.IGNORECASE)


def parse_size_tag_from_path(tag: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    m = RX_SIZE.search(tag or "")
    if not m:
        return (None, None, None)
    return (int(m.group("n_train")), int(m.group("n_id")), int(m.group("n_ood")))


def build_rx_for_paths(master_tag_prefix: str) -> Tuple[re.Pattern, re.Pattern]:
    prefix = re.escape(str(master_tag_prefix))
    rx_runkey_detail = re.compile(
        r"^" + prefix + r"(?P<variant>.+?)__ms(?P<master_seed>\d+)__h(?P<vhash>[0-9a-fA-F]+)$",
        re.IGNORECASE,
    )
    rx_path = re.compile(
        r"(?:^|[\\/])"
        r"(?P<run_key>" + prefix + r"[^\\/]+__ms\d+__h[0-9a-fA-F]{6,64})"
        r"[\\/]+(?P<size_tag>size_q\d+_id\d+_ood\d+)"
        r"[\\/]+qsplit(?P<qsplit>\d+)[\\/]+seed(?P<seed>\d+)"
        r"[\\/]+(?P<family>quantum|classical)",
        re.IGNORECASE,
    )
    return rx_path, rx_runkey_detail


def parse_runkey_details(run_key: str, rx_runkey_detail: re.Pattern) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    m = rx_runkey_detail.match(run_key or "")
    if not m:
        return (None, None, None)
    try:
        return (m.group("variant"), int(m.group("master_seed")), m.group("vhash"))
    except Exception:
        return (None, None, None)


def load_all_summary_rows(results_root: Path, *, master_tag_prefix: str) -> pd.DataFrame:
    rx_path, rx_runkey_detail = build_rx_for_paths(master_tag_prefix)

    files = list(results_root.rglob("*__summary.csv"))
    if not files:
        raise RuntimeError(f"No encontré ningún *__summary.csv en {results_root} (¿no se ejecutaron los runs?).")

    rows: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        p = str(f)

        m = rx_path.search(p)
        if not m:
            print(f"[WARN] summary path no matchea RX_PATH -> skip: {p}")
            continue

        run_key = str(m.group("run_key"))
        variant_name, master_seed, vhash = parse_runkey_details(run_key, rx_runkey_detail)

        qsplit_seed = int(m.group("qsplit"))
        model_seed = int(m.group("seed"))
        family_path = m.group("family").lower()
        size_tag_ = m.group("size_tag")

        n_train, n_id, n_ood = parse_size_tag_from_path(size_tag_)

        df["file"] = str(f)
        df["run_key"] = run_key
        df["variant_name"] = variant_name
        df["master_seed"] = master_seed
        df["variant_hash"] = vhash

        df["qsplit_seed"] = qsplit_seed
        df["model_seed"] = model_seed
        df["size_tag"] = size_tag_
        df["n_train"] = n_train
        df["n_id"] = n_id
        df["n_ood"] = n_ood

        if "family" not in df.columns or df["family"].isna().all():
            df["family"] = family_path
        else:
            df["family"] = df["family"].fillna(family_path)

        rows.append(df)

    if not rows:
        raise RuntimeError(
            f"No pude cargar summaries válidos desde {results_root} (RX_PATH no matcheó ninguno). "
            f"Tip: revisa que master_tag_prefix='{master_tag_prefix}' coincide con tu layout."
        )

    out = pd.concat(rows, ignore_index=True)

    if "dim" in out.columns:
        out["dim"] = pd.to_numeric(out["dim"], errors="coerce").astype("Int64")
    for c in ["qsplit_seed", "model_seed", "master_seed", "n_train", "n_id", "n_ood"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    out["family"] = out["family"].astype(str).str.lower()
    out["size_tag"] = out["size_tag"].astype(str)

    if "cfg" not in out.columns or "split" not in out.columns:
        raise RuntimeError("Tus *__summary.csv deben tener columnas 'cfg' y 'split'.")

    for col in ["accuracy", "balanced_accuracy", "f1_macro", "f1_pos", "roc_auc", "pr_auc", "thr_value"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def build_run_drops(base: pd.DataFrame) -> pd.DataFrame:
    gcols = [
        "run_key",
        "variant_name",
        "master_seed",
        "variant_hash",
        "size_tag",
        "qsplit_seed",
        "model_seed",
        "family",
        "dim",
        "cfg",
    ]
    for c in gcols:
        if c not in base.columns:
            base[c] = None

    parts: List[dict] = []
    for _, grp in base.groupby(gcols, dropna=False):
        id_row = grp.loc[grp["split"] == "id_test"]
        ood_row = grp.loc[grp["split"] == "ood_test"]
        if id_row.empty or ood_row.empty:
            continue
        id1 = id_row.iloc[0]
        ood1 = ood_row.iloc[0]

        def fget(series: pd.Series, key: str) -> float:
            v = series.get(key, np.nan)
            try:
                return float(v)
            except Exception:
                return float("nan")

        id_bal = fget(id1, "balanced_accuracy")
        ood_bal = fget(ood1, "balanced_accuracy")

        id_f1p = fget(id1, "f1_pos")
        ood_f1p = fget(ood1, "f1_pos")

        id_pr = fget(id1, "pr_auc")
        ood_pr = fget(ood1, "pr_auc")

        id_roc = fget(id1, "roc_auc")
        ood_roc = fget(ood1, "roc_auc")

        parts.append(
            {
                "run_key": str(id1.get("run_key") or ""),
                "variant_name": id1.get("variant_name"),
                "master_seed": int(id1["master_seed"]) if "master_seed" in id1 and not pd.isna(id1["master_seed"]) else None,
                "variant_hash": id1.get("variant_hash"),
                "size_tag": str(id1.get("size_tag") or ""),
                "n_train": int(id1["n_train"]) if "n_train" in id1 and not pd.isna(id1["n_train"]) else None,
                "n_id": int(id1["n_id"]) if "n_id" in id1 and not pd.isna(id1["n_id"]) else None,
                "n_ood": int(id1["n_ood"]) if "n_ood" in id1 and not pd.isna(id1["n_ood"]) else None,
                "qsplit_seed": int(id1["qsplit_seed"]) if "qsplit_seed" in id1 and not pd.isna(id1["qsplit_seed"]) else None,
                "model_seed": int(id1["model_seed"]) if "model_seed" in id1 and not pd.isna(id1["model_seed"]) else None,
                "family": str(id1.get("family") or ""),
                "dim": int(id1["dim"]) if "dim" in id1 and not pd.isna(id1["dim"]) else None,
                "cfg": str(id1.get("cfg") or ""),
                "id_bal_acc": id_bal,
                "ood_bal_acc": ood_bal,
                "drop_bal_acc": id_bal - ood_bal,
                "id_f1_pos": id_f1p,
                "ood_f1_pos": ood_f1p,
                "drop_f1_pos": id_f1p - ood_f1p,
                "id_pr_auc": id_pr,
                "ood_pr_auc": ood_pr,
                "drop_pr_auc": (id_pr - ood_pr) if (not np.isnan(id_pr) and not np.isnan(ood_pr)) else np.nan,
                "id_roc_auc": id_roc,
                "ood_roc_auc": ood_roc,
                "drop_roc_auc": (id_roc - ood_roc) if (not np.isnan(id_roc) and not np.isnan(ood_roc)) else np.nan,
            }
        )

    if not parts:
        raise RuntimeError("AGG_runs_drops quedó vacío. Revisa logs (¿faltan id_test/ood_test en summaries?).")

    return pd.DataFrame(parts)


def agg_mean_std(
    run_drops: pd.DataFrame,
    *,
    group_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    g = list(group_cols)

    agg_drop = (
        run_drops.groupby(g, as_index=False, dropna=False)
        .agg(
            n_runs=("drop_bal_acc", "count"),
            drop_bal_acc_mean=("drop_bal_acc", "mean"),
            drop_bal_acc_std=("drop_bal_acc", "std"),
        )
    ).fillna({"drop_bal_acc_std": 0.0})

    agg_metrics = (
        run_drops.groupby(g, as_index=False, dropna=False)
        .agg(
            n_runs=("id_bal_acc", "count"),
            id_bal_acc_mean=("id_bal_acc", "mean"),
            id_bal_acc_std=("id_bal_acc", "std"),
            ood_bal_acc_mean=("ood_bal_acc", "mean"),
            ood_bal_acc_std=("ood_bal_acc", "std"),
            id_f1_pos_mean=("id_f1_pos", "mean"),
            id_f1_pos_std=("id_f1_pos", "std"),
            ood_f1_pos_mean=("ood_f1_pos", "mean"),
            ood_f1_pos_std=("ood_f1_pos", "std"),
        )
    ).fillna(
        {
            "id_bal_acc_std": 0.0,
            "ood_bal_acc_std": 0.0,
            "id_f1_pos_std": 0.0,
            "ood_f1_pos_std": 0.0,
        }
    )

    joined = agg_metrics.merge(agg_drop, on=g, how="inner", suffixes=("", "_drop"))
    joined["tradeoff_score"] = joined["ood_bal_acc_mean"] - joined["drop_bal_acc_mean"]
    return agg_drop, agg_metrics, joined


def write_topk(joined: pd.DataFrame, out_txt: Path, *, topk: int = 10, context_cols: Optional[List[str]] = None) -> None:
    ensure_dir(out_txt.parent)
    ctx = context_cols or []
    ctx = [c for c in ctx if c in joined.columns]

    def _safe_int(x, default: str = "NA") -> str:
        try:
            if pd.isna(x):
                return default
            return str(int(x))
        except Exception:
            return default

    def ctx_str(r: pd.Series) -> str:
        if not ctx:
            return ""
        bits = []
        for c in ctx:
            v = r.get(c, "")
            if pd.isna(v):
                v = ""
            bits.append(f"{c}={v}")
        return " | " + " ".join(bits) if bits else ""

    lines: List[str] = []
    for fam in ["quantum", "classical"]:
        sub = joined[joined["family"] == fam].copy()
        if sub.empty:
            continue

        lines.append(f"===== FAMILY={fam} :: TOP-{topk} Robustez (min drop_bal_acc_mean) =====")
        top_rob = sub.sort_values(by=["drop_bal_acc_mean", "ood_bal_acc_mean"], ascending=[True, False], kind="mergesort").head(topk)
        for _, r in top_rob.iterrows():
            lines.append(
                f"dim={_safe_int(r.get('dim'))} | {r.get('cfg','')}"
                f"{ctx_str(r)}"
                f" | drop={r['drop_bal_acc_mean']:.4f}±{r['drop_bal_acc_std']:.4f}"
                f" | OOD={r['ood_bal_acc_mean']:.4f}±{r['ood_bal_acc_std']:.4f}"
                f" | ID={r['id_bal_acc_mean']:.4f}±{r['id_bal_acc_std']:.4f}"
            )

        lines.append("")
        lines.append(f"===== FAMILY={fam} :: TOP-{topk} Trade-off (max tradeoff_score) =====")
        top_trade = sub.sort_values(by=["tradeoff_score", "ood_bal_acc_mean", "drop_bal_acc_mean"], ascending=[False, False, True], kind="mergesort").head(topk)
        for _, r in top_trade.iterrows():
            lines.append(
                f"dim={_safe_int(r.get('dim'))} | {r.get('cfg','')}"
                f"{ctx_str(r)}"
                f" | tradeoff={r['tradeoff_score']:.4f}"
                f" | drop={r['drop_bal_acc_mean']:.4f}±{r['drop_bal_acc_std']:.4f}"
                f" | OOD={r['ood_bal_acc_mean']:.4f}±{r['ood_bal_acc_std']:.4f}"
            )
        lines.append("")

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def write_global_json(results_root: Path, filename: str, only_baseline_json: bool) -> None:
    rx = re.compile(
        r"qsplit(?P<qsplit>\d+)[\\/]+seed(?P<seed>\d+)[\\/]+(?P<family>quantum|classical)",
        re.IGNORECASE,
    )

    def safe_load(p: Path) -> Optional[dict]:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            try:
                return json.loads(p.read_text(encoding="utf-8-sig"))
            except Exception:
                return None

    files: List[Path] = []
    for p in results_root.rglob("*.json"):
        if p.name in {"meta_q.json", "MANIFEST_compare_q_vs_c.json"}:
            continue
        if p.name.startswith("AGG_") or p.name.startswith("GLOBAL_"):
            continue
        if only_baseline_json and not p.name.startswith("baseline_"):
            continue
        files.append(p)

    out = {"meta": {"results_root": str(results_root), "n_files": len(files)}, "runs": []}

    for p in files:
        m = rx.search(str(p))
        qsplit_seed = int(m.group("qsplit")) if m else None
        model_seed = int(m.group("seed")) if m else None
        family = m.group("family").lower() if m else None
        payload = safe_load(p)
        out["runs"].append(
            {
                "file": str(p),
                "qsplit_seed": qsplit_seed,
                "model_seed": model_seed,
                "family": family,
                "payload": payload,
            }
        )

    (results_root / filename).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] wrote {results_root / filename}")


def write_across_sizes_best(results_root: Path, run_drops: pd.DataFrame, dims: List[int], criterion: str) -> None:
    agg = (
        run_drops.groupby(["size_tag", "family", "dim", "cfg"], as_index=False, dropna=False)
        .agg(
            n_runs=("drop_bal_acc", "count"),
            drop_bal_acc_mean=("drop_bal_acc", "mean"),
            ood_bal_acc_mean=("ood_bal_acc", "mean"),
            id_bal_acc_mean=("id_bal_acc", "mean"),
        )
    )
    agg["tradeoff_score"] = agg["ood_bal_acc_mean"] - agg["drop_bal_acc_mean"]

    best_rows = []
    for sz in sorted(agg["size_tag"].dropna().unique()):
        for fam in ["quantum", "classical"]:
            for d in dims:
                sub = agg[(agg["size_tag"] == sz) & (agg["family"] == fam) & (agg["dim"] == d)].copy()
                if sub.empty:
                    continue

                if criterion == "robust":
                    sub = sub.sort_values(by=["drop_bal_acc_mean", "ood_bal_acc_mean"], ascending=[True, False])
                elif criterion == "ood":
                    sub = sub.sort_values(by=["ood_bal_acc_mean", "drop_bal_acc_mean"], ascending=[False, True])
                else:
                    sub = sub.sort_values(by=["tradeoff_score", "ood_bal_acc_mean"], ascending=[False, False])

                r = sub.iloc[0]
                best_rows.append(
                    {
                        "size_tag": sz,
                        "family": fam,
                        "dim": int(d),
                        "best_cfg": str(r["cfg"]),
                        "n_runs": int(r["n_runs"]),
                        "id_bal_acc_mean": float(r["id_bal_acc_mean"]),
                        "ood_bal_acc_mean": float(r["ood_bal_acc_mean"]),
                        "drop_bal_acc_mean": float(r["drop_bal_acc_mean"]),
                        "tradeoff_score": float(r["tradeoff_score"]),
                    }
                )

    df_best = pd.DataFrame(best_rows)
    out = results_root / "AGG_best_by_family_dim_and_size.csv"
    df_best.sort_values(by=["size_tag", "dim", "family"]).to_csv(out, index=False)
    print(f"[OK] {out}")


def write_index_experiment_context(out_path: Path, rows: List[dict]) -> None:
    if not rows:
        print("[WARN] INDEX_experiment_context.csv not written (no rows).")
        return
    df = pd.DataFrame(rows)

    preferred = [
        "run_key",
        "variant_name",
        "master_seed",
        "variant_hash",
        "ood_mode",
        "require_low_high",
        "size_tag",
        "n_train",
        "n_id",
        "n_ood",
        "qsplit_seed",
        "model_seed",
        "dims",
        "use_thresholding",
        "thresh_source",
        "thresh_criterion",
        "thresh_grid",
        "classical_kernel_normalize",
        "master_dir",
        "qsplits_root",
        "results_dir_size",
        "logs_dir_size",
        "out_q_dir",
        "out_c_dir",
        "has_summary_q",
        "has_summary_c",
        "summary_count_q",
        "summary_count_c",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols].sort_values(by=[c for c in ["variant_name", "master_seed", "size_tag", "qsplit_seed", "model_seed"] if c in df.columns])
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] wrote {out_path}")


def write_agg_long_runs(out_path: Path, run_drops: pd.DataFrame) -> None:
    ensure_dir(out_path.parent)
    run_drops.sort_values(
        by=["variant_name", "master_seed", "size_tag", "family", "dim", "cfg", "qsplit_seed", "model_seed"],
        kind="mergesort",
    ).to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] wrote {out_path}")


def write_contextual_summaries(root: Path, run_drops: pd.DataFrame) -> None:
    ensure_dir(root)

    # 1) Global absoluto
    g1 = ["family", "dim", "cfg"]
    agg_drop, agg_metrics, joined = agg_mean_std(run_drops, group_cols=g1)

    agg_drop.sort_values(by=["family", "drop_bal_acc_mean"]).to_csv(root / "AGG_ROOT_mean_std_drop__global.csv", index=False)
    agg_metrics.sort_values(by=["family", "ood_bal_acc_mean"], ascending=[True, False]).to_csv(root / "AGG_ROOT_mean_std_metrics__global.csv", index=False)
    joined.sort_values(by=["family", "tradeoff_score"], ascending=[True, False]).to_csv(root / "AGG_ROOT_ranking_tradeoff__global.csv", index=False)
    write_topk(joined, root / "AGG_ROOT_topK__global.txt", topk=15)

    # 2) Por run_key + size_tag
    if "run_key" in run_drops.columns:
        g2 = ["run_key", "size_tag", "family", "dim", "cfg"]
        a2_drop, a2_metrics, a2_joined = agg_mean_std(run_drops, group_cols=g2)
        a2_drop.sort_values(by=["run_key", "size_tag", "family", "drop_bal_acc_mean"]).to_csv(
            root / "AGG_ROOT_mean_std_drop__by_runkey_and_size.csv", index=False
        )
        a2_metrics.sort_values(by=["run_key", "size_tag", "family", "ood_bal_acc_mean"], ascending=[True, True, True, False]).to_csv(
            root / "AGG_ROOT_mean_std_metrics__by_runkey_and_size.csv", index=False
        )
        a2_joined.sort_values(by=["run_key", "size_tag", "family", "tradeoff_score"], ascending=[True, True, True, False]).to_csv(
            root / "AGG_ROOT_ranking_tradeoff__by_runkey_and_size.csv", index=False
        )
        write_topk(
            a2_joined,
            root / "AGG_ROOT_topK__by_runkey_and_size.txt",
            topk=10,
            context_cols=["run_key", "size_tag"],
        )

    # 3) Por variant/master_seed + size_tag
    if "variant_name" in run_drops.columns and "master_seed" in run_drops.columns:
        g3 = ["variant_name", "master_seed", "size_tag", "family", "dim", "cfg"]
        a3_drop, a3_metrics, a3_joined = agg_mean_std(run_drops, group_cols=g3)
        a3_drop.sort_values(by=["variant_name", "master_seed", "size_tag", "family", "drop_bal_acc_mean"]).to_csv(
            root / "AGG_ROOT_mean_std_drop__by_variant_seed_and_size.csv", index=False
        )
        a3_metrics.sort_values(by=["variant_name", "master_seed", "size_tag", "family", "ood_bal_acc_mean"], ascending=[True, True, True, True, False]).to_csv(
            root / "AGG_ROOT_mean_std_metrics__by_variant_seed_and_size.csv", index=False
        )
        a3_joined.sort_values(by=["variant_name", "master_seed", "size_tag", "family", "tradeoff_score"], ascending=[True, True, True, True, False]).to_csv(
            root / "AGG_ROOT_ranking_tradeoff__by_variant_seed_and_size.csv", index=False
        )
        write_topk(
            a3_joined,
            root / "AGG_ROOT_topK__by_variant_seed_and_size.txt",
            topk=10,
            context_cols=["variant_name", "master_seed", "size_tag"],
        )


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run EMBER quantum+classical grid with:\n"
            "  (variant + master_seed) master split generation -> q-splits -> experiments -> aggregation.\n"
            "Includes: variants sweep + master-seeds sweep + size sweep.\n"
            "Writes per-size AGGs + per-run across-sizes + ROOT summaries + context index."
        )
    )

    # Seeds
    ap.add_argument("--qsplit-seeds", nargs="+", type=int, default=DEFAULT_QSPLIT_SEEDS)
    ap.add_argument("--model-seeds", nargs="+", type=int, default=DEFAULT_MODEL_SEEDS)

    # master seeds sweep
    ap.add_argument("--master-seeds", nargs="+", type=int, default=DEFAULT_MASTER_SEEDS)
    ap.add_argument("--master-seed", type=int, default=None, help="(legacy) equivalente a --master-seeds X")

    # Size sweep
    ap.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    ap.add_argument("--n-id", type=int, default=DEFAULT_N_ID)
    ap.add_argument("--n-ood", type=int, default=DEFAULT_N_OOD)
    ap.add_argument("--size-grid", nargs="+", default=None, help="Lista de triples n_train,n_id,n_ood (ej: 1000,500,500 2000,1000,1000).")
    ap.add_argument("--sizes-preset", type=str, default=None, help=f"Preset de tamaños: {', '.join(sorted(SIZE_PRESETS.keys()))}. Default={DEFAULT_SIZES_PRESET}")

    # Dims
    ap.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)

    # Thresholding
    ap.add_argument("--use-thresholding", action="store_true", default=DEFAULT_USE_THRESHOLDING)
    ap.add_argument("--thresh-source", type=str, default=DEFAULT_THRESH_SOURCE)
    ap.add_argument("--thresh-criterion", type=str, default=DEFAULT_THRESH_CRITERION)
    ap.add_argument("--thresh-grid", type=int, default=DEFAULT_THRESH_GRID)

    # Parallel
    ap.add_argument("--use-parallel", action="store_true", default=DEFAULT_USE_PARALLEL)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    # Caching / strictness
    ap.add_argument("--no-skip-qsplits-if-exists", action="store_true", default=False, help="Fuerza REGENERAR q-splits aunque ya existan (por master).")
    ap.add_argument("--clean-results-before-run", action="store_true", default=DEFAULT_CLEAN_RESULTS_BEFORE_RUN)
    ap.add_argument("--no-fail-if-splits-mismatch", action="store_true", default=False)

    ap.add_argument(
        "--skip-run-if-summary-exists",
        dest="skip_run_if_summary_exists",
        action="store_true",
        default=DEFAULT_SKIP_RUN_IF_SUMMARY_EXISTS,
        help="(default) Salta runs si ya hay *__summary.csv en el out_dir.",
    )
    ap.add_argument("--no-skip-run-if-summary-exists", dest="skip_run_if_summary_exists", action="store_false", help="Desactiva el skip por summary existente.")

    ap.add_argument("--no-classical-kernel-normalize", action="store_true", default=False)
    ap.add_argument("--best-criterion", type=str, default=DEFAULT_BEST_CRITERION, choices=["tradeoff", "robust", "ood"])

    ap.add_argument("--no-write-global-json", action="store_true", default=False)
    ap.add_argument("--global-json-filename", type=str, default=DEFAULT_GLOBAL_JSON_FILENAME)

    gjson = ap.add_mutually_exclusive_group()
    gjson.add_argument("--global-json-only-baseline", action="store_true", default=True, help="(default) El merge global solo incluye JSON que empiecen por 'baseline_'.")
    gjson.add_argument("--global-json-include-all", action="store_true", default=False, help="Incluye cualquier JSON no-AGG (no filtra por baseline_).")

    # Paths
    ap.add_argument("--splits-root-dir", type=Path, default=DEFAULT_SPLITS_ROOT_DIR)
    ap.add_argument("--results-root-dir", type=Path, default=DEFAULT_RESULTS_ROOT_DIR)
    ap.add_argument("--log-root-dir", type=Path, default=DEFAULT_LOG_ROOT_DIR)

    # input to make_master_splits
    ap.add_argument(
        "--master-in-dir",
        type=Path,
        default=DEFAULT_SPLITS_ROOT_DIR,
        help="Directorio de entrada para make_splits_ember_sparsity (--in-dir). Pon aquí tu carpeta real de XY/features si no es data/processed/ember.",
    )

    # Modules
    ap.add_argument("--make-master-splits-mod", type=str, default=DEFAULT_MAKE_MASTER_SPLITS_MOD)
    ap.add_argument("--make-qsplits-mod", type=str, default=DEFAULT_MAKE_QSPLITS_MOD)
    ap.add_argument("--run-q-mod", type=str, default=DEFAULT_RUN_Q_MOD)
    ap.add_argument("--run-c-mod", type=str, default=DEFAULT_RUN_C_MOD)

    # Master variants control
    ap.add_argument("--variants", nargs="+", type=str, default=DEFAULT_VARIANTS, help=f"Variantes master a ejecutar. Disponibles: {', '.join(sorted(MASTER_VARIANTS.keys()))}")
    ap.add_argument("--master-tag-prefix", type=str, default=DEFAULT_MASTER_TAG_PREFIX)

    ap.add_argument("--no-run-master-splits", dest="run_master_splits", action="store_false", default=True, help="Si lo pasas, NO genera master splits (asume que ya existen).")
    ap.add_argument("--force-master-regenerate", action="store_true", default=False, help="Fuerza regenerar los master splits aunque existan.")
    ap.add_argument("--master-only", action="store_true", default=False, help="Solo genera los master splits (para variantes x master-seeds) y sale.")

    ap.add_argument("--no-require-low-high", action="store_true", default=False, help="Desactiva globalmente low/high (NO recomendado salvo que cambies tooling).")
    ap.add_argument("--dry-run", action="store_true", default=False, help="No ejecuta nada: imprime comandos y valida paths/args.")

    # NEW: error handling
    ap.add_argument("--continue-on-error", action="store_true", default=DEFAULT_CONTINUE_ON_ERROR, help="Si un run falla, lo reporta y continúa (útil en barridos grandes).")
    ap.add_argument("--log-tail-lines", type=int, default=DEFAULT_LOG_TAIL_LINES, help="Líneas finales del log a adjuntar en el error (cuando falla un cmd).")

    args = ap.parse_args()

    if args.master_seed is not None:
        args.master_seeds = [int(args.master_seed)]

    skip_qsplits_if_exists = not args.no_skip_qsplits_if_exists
    fail_if_mismatch = not args.no_fail_if_splits_mismatch
    classical_kernel_normalize = not args.no_classical_kernel_normalize
    write_global_json_flag = not args.no_write_global_json
    only_baseline_json = bool(args.global_json_only_baseline) and (not args.global_json_include_all)

    require_low_high_global = not args.no_require_low_high
    size_grid = resolve_size_grid(args)

    index_rows: List[dict] = []

    print(f"Python: {sys.executable}")
    print(f"[CFG] variants                   = {args.variants}")
    print(f"[CFG] master_seeds               = {args.master_seeds}")
    print(f"[CFG] qsplit_seeds               = {args.qsplit_seeds}")
    print(f"[CFG] model_seeds                = {args.model_seeds}")
    print(f"[CFG] dims                       = {args.dims}")
    print(f"[CFG] use_thresholding           = {args.use_thresholding} ({args.thresh_source}/{args.thresh_criterion}/grid={args.thresh_grid})")
    print(f"[CFG] skip_run_if_summary_exists = {args.skip_run_if_summary_exists}")
    print(f"[CFG] classical_kernel_normalize = {classical_kernel_normalize}")
    print(f"[CFG] best_criterion             = {args.best_criterion}")
    print(f"[CFG] use_parallel               = {args.use_parallel} (workers={args.workers})")
    print(f"[CFG] size_grid                  = {size_grid}")
    print(f"[CFG] require_low_high (global)  = {require_low_high_global}")
    print(f"[CFG] dry_run                    = {args.dry_run}")
    print(f"[CFG] continue_on_error          = {args.continue_on_error}")
    print(f"[CFG] log_tail_lines             = {args.log_tail_lines}")
    print(f"[CFG] global_json_only_baseline  = {only_baseline_json}")
    print(f"[CFG] run_master_splits          = {args.run_master_splits} (force={args.force_master_regenerate})")
    print(f"[CFG] master_only                = {args.master_only}")
    print(f"[CFG] master_in_dir              = {args.master_in_dir}")
    print(f"[CFG] master_tag_prefix          = {args.master_tag_prefix}")

    if args.clean_results_before_run and args.results_root_dir.exists():
        if args.dry_run:
            print(f"[DRY] would delete results dir: {args.results_root_dir}")
        else:
            print(f"[CLEAN] borrando results dir: {args.results_root_dir}")
            shutil.rmtree(args.results_root_dir, ignore_errors=True)

    ensure_dir(args.results_root_dir)
    ensure_dir(args.log_root_dir)

    for v in args.variants:
        if v not in MASTER_VARIANTS:
            raise SystemExit(f"Variante '{v}' no existe. Disponibles: {sorted(MASTER_VARIANTS.keys())}")

    write_manifest(
        args.results_root_dir,
        args,
        extra={
            "size_grid": size_grid,
            "variants": args.variants,
            "master_seeds": args.master_seeds,
            "note": "Root manifest (variants + master_seeds + size sweep).",
        },
    )

    # -------------------------------------------------------------------------
    # Loop (variant, master_seed)
    # -------------------------------------------------------------------------
    for variant_name in args.variants:
        variant = MASTER_VARIANTS[variant_name]
        vhash = cfg_hash_short(variant, n=8)
        ood_mode_str = str(variant.get("ood_mode"))

        for master_seed in args.master_seeds:
            print(f"\n==============================")
            print(f"=== VARIANT={variant_name} | MASTER_SEED={master_seed} | h{vhash} ===")
            print(f"==============================")

            print(f"\n=== [0/3] Master splits (OOD protocol) ===")
            master_dir = args.splits_root_dir / f"{args.master_tag_prefix}{variant_name}__ms{int(master_seed)}__h{vhash}"

            if args.run_master_splits:
                master_dir = ensure_master_splits(
                    variant_name=variant_name,
                    variant=variant,
                    make_master_mod=str(args.make_master_splits_mod),
                    in_dir=args.master_in_dir,
                    out_root=args.splits_root_dir,
                    tag_prefix=str(args.master_tag_prefix),
                    master_seed=int(master_seed),
                    dry_run=bool(args.dry_run),
                    force=bool(args.force_master_regenerate),
                    log_root=args.log_root_dir,
                )
            else:
                if not master_dir.exists():
                    raise SystemExit(f"Master splits no existe y --no-run-master-splits activo: {master_dir}")

            if args.master_only:
                continue

            # q-splits live under master_dir/qsplits
            splits_root_variant = master_dir / "qsplits"
            ensure_dir(splits_root_variant)

            require_low_high = bool(require_low_high_global) and bool(variant.get("require_low_high", True))
            expected_low_fn = lambda n_ood: int(np.floor(n_ood / 2))
            expected_high_fn = lambda n_ood: int(n_ood - int(np.floor(n_ood / 2)))

            run_key = f"{args.master_tag_prefix}{variant_name}__ms{int(master_seed)}__h{vhash}"
            run_results_root = args.results_root_dir / run_key
            run_log_root = args.log_root_dir / run_key
            ensure_dir(run_results_root)
            ensure_dir(run_log_root)

            write_manifest(
                run_results_root,
                args,
                extra={
                    "variant": variant_name,
                    "master_seed": int(master_seed),
                    "variant_hash": vhash,
                    "ood_mode": ood_mode_str,
                    "master_dir": str(master_dir),
                    "qsplits_root": str(splits_root_variant),
                    "size_grid": size_grid,
                    "require_low_high": require_low_high,
                },
            )

            # -----------------------------------------------------------------
            # Size sweep
            # -----------------------------------------------------------------
            for (n_train, n_id, n_ood) in size_grid:
                tag = size_tag(n_train, n_id, n_ood)
                results_dir_size = run_results_root / tag
                logs_dir_size = run_log_root / tag

                ensure_dir(results_dir_size)
                ensure_dir(logs_dir_size)

                write_manifest(
                    results_dir_size,
                    args,
                    extra={
                        "variant": variant_name,
                        "master_seed": int(master_seed),
                        "variant_hash": vhash,
                        "ood_mode": ood_mode_str,
                        "master_dir": str(master_dir),
                        "qsplits_root": str(splits_root_variant),
                        "size": {"n_train": n_train, "n_id": n_id, "n_ood": n_ood},
                        "tag": tag,
                        "require_low_high": require_low_high,
                    },
                )

                expected_low = int(expected_low_fn(n_ood))
                expected_high = int(expected_high_fn(n_ood))

                # 1) q-splits
                print(f"\n=== [SIZE {tag}] [1/3] Generando q-splits ===")
                for qs in args.qsplit_seeds:
                    name = qsplit_name(n_train, n_id, n_ood, qs)
                    dst = splits_root_variant / name
                    ensure_dir(dst)

                    already_ok = False
                    if skip_qsplits_if_exists and (dst / "train_idx.npy").exists():
                        try:
                            already_ok = assert_qsplit_valid(
                                splits_dir=dst,
                                expected_train=n_train,
                                expected_id=n_id,
                                expected_ood=n_ood,
                                expected_low=expected_low,
                                expected_high=expected_high,
                                fail_if_mismatch=fail_if_mismatch,
                                require_low_high=require_low_high,
                                context=f"(pre-check seed={qs} variant={variant_name} ms={master_seed})",
                            )
                        except Exception as e:
                            already_ok = False
                            print(f"[WARN] QSplit existente pero inválido; se regenerará. Motivo: {e}")

                    if already_ok:
                        print(f"[OK] QSplit ya existe y valida -> skip generación: {dst}")
                        continue

                    log = logs_dir_size / f"make_qsplits_seed{qs}.log"
                    cmd = [
                        sys.executable,
                        "-m",
                        str(args.make_qsplits_mod),
                        "--src",
                        str(master_dir),
                        "--dst-root",
                        str(splits_root_variant),
                        "--seed",
                        str(qs),
                        "--n-train",
                        str(n_train),
                        "--n-id",
                        str(n_id),
                        "--n-ood",
                        str(n_ood),
                    ]
                    if require_low_high:
                        cmd += ["--use-low-high"]

                    run_cmd_to_log(
                        title=f"make_qsplits seed={qs} ({tag}) [variant={variant_name} ms={master_seed}]",
                        log_path=log,
                        cmd=cmd,
                        dry_run=args.dry_run,
                        tail_lines_on_error=int(args.log_tail_lines),
                    )

                    if not args.dry_run:
                        assert_qsplit_valid(
                            splits_dir=dst,
                            expected_train=n_train,
                            expected_id=n_id,
                            expected_ood=n_ood,
                            expected_low=expected_low,
                            expected_high=expected_high,
                            fail_if_mismatch=True,
                            require_low_high=require_low_high,
                            context=f"(post-gen seed={qs} variant={variant_name} ms={master_seed})",
                        )

                # 2) runs
                print(f"\n=== [SIZE {tag}] [2/3] Ejecutando grid (Q + C) ===")

                cfg = RunConfig(
                    splits_root=splits_root_variant,
                    results_root=results_dir_size,
                    log_root=logs_dir_size,
                    n_train=n_train,
                    n_id=n_id,
                    n_ood=n_ood,
                    expected_low=expected_low,
                    expected_high=expected_high,
                    dims=list(args.dims),
                    use_thresholding=bool(args.use_thresholding),
                    thresh_source=str(args.thresh_source),
                    thresh_criterion=str(args.thresh_criterion),
                    thresh_grid=int(args.thresh_grid),
                    classical_kernel_normalize=bool(classical_kernel_normalize),
                    skip_run_if_summary_exists=bool(args.skip_run_if_summary_exists),
                    run_q_mod=str(args.run_q_mod),
                    run_c_mod=str(args.run_c_mod),
                    fail_if_splits_mismatch=bool(fail_if_mismatch),
                    require_low_high=bool(require_low_high),
                    dry_run=bool(args.dry_run),
                    continue_on_error=bool(args.continue_on_error),
                )

                combos = [(qs, ms) for qs in args.qsplit_seeds for ms in args.model_seeds]

                failures: List[str] = []

                if not args.use_parallel:
                    for qs, ms in combos:
                        try:
                            run_one_combo(cfg, qs, ms)
                        except Exception as e:
                            failures.append(f"(qsplit={qs}, seed={ms}) :: {e}")
                            if not args.continue_on_error:
                                raise
                else:
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    max_workers = max(1, int(args.workers))
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        futs = {ex.submit(run_one_combo, cfg, qs, ms): (qs, ms) for (qs, ms) in combos}
                        for f in as_completed(futs):
                            qs, ms = futs[f]
                            try:
                                f.result()
                            except Exception as e:
                                failures.append(f"(qsplit={qs}, seed={ms}) :: {e}")
                                if not args.continue_on_error:
                                    raise

                if failures:
                    print(f"[WARN] Hubo {len(failures)} fallos en {run_key}/{tag}.")
                    for x in failures[:10]:
                        print(f"  - {x}")
                    if len(failures) > 10:
                        print(f"  ... (+{len(failures)-10} más)")
                    if not args.continue_on_error:
                        raise RuntimeError("Fallos en grid (y continue_on_error=False). Revisa el listado anterior.")

                # 2.5) index rows (always)
                for qs, ms in combos:
                    out_q_dir = results_dir_size / f"qsplit{qs}" / f"seed{ms}" / "quantum"
                    out_c_dir = results_dir_size / f"qsplit{qs}" / f"seed{ms}" / "classical"
                    if args.dry_run:
                        has_q = None
                        has_c = None
                        cnt_q = None
                        cnt_c = None
                    else:
                        has_q = has_summary_csv(out_q_dir)
                        has_c = has_summary_csv(out_c_dir)
                        cnt_q = len(list(out_q_dir.glob("*__summary.csv"))) if out_q_dir.exists() else 0
                        cnt_c = len(list(out_c_dir.glob("*__summary.csv"))) if out_c_dir.exists() else 0

                    index_rows.append(
                        {
                            "run_key": run_key,
                            "variant_name": variant_name,
                            "master_seed": int(master_seed),
                            "variant_hash": vhash,
                            "ood_mode": ood_mode_str,
                            "require_low_high": bool(require_low_high),
                            "size_tag": tag,
                            "n_train": int(n_train),
                            "n_id": int(n_id),
                            "n_ood": int(n_ood),
                            "qsplit_seed": int(qs),
                            "model_seed": int(ms),
                            "dims": ",".join(str(d) for d in args.dims),
                            "use_thresholding": bool(args.use_thresholding),
                            "thresh_source": str(args.thresh_source),
                            "thresh_criterion": str(args.thresh_criterion),
                            "thresh_grid": int(args.thresh_grid),
                            "classical_kernel_normalize": bool(classical_kernel_normalize),
                            "master_dir": str(master_dir),
                            "qsplits_root": str(splits_root_variant),
                            "results_dir_size": str(results_dir_size),
                            "logs_dir_size": str(logs_dir_size),
                            "out_q_dir": str(out_q_dir),
                            "out_c_dir": str(out_c_dir),
                            "has_summary_q": has_q,
                            "has_summary_c": has_c,
                            "summary_count_q": cnt_q,
                            "summary_count_c": cnt_c,
                        }
                    )

                # 3) per-size aggregation
                print(f"\n=== [SIZE {tag}] [3/3] Agregación ===")
                if args.dry_run:
                    print(f"[DRY] aggregation skipped for {run_key}/{tag}")
                    continue

                all_rows = load_all_summary_rows(results_dir_size, master_tag_prefix=args.master_tag_prefix)
                base = all_rows[all_rows["split"].isin(["id_test", "ood_test"])].copy()
                if base.empty:
                    raise RuntimeError(f"[{run_key}/{tag}] No hay filas base (id_test/ood_test) en summaries. Revisa runs.")

                run_drops = build_run_drops(base)

                out_runs = results_dir_size / "AGG_runs_drops.csv"
                run_drops.sort_values(by=["family", "dim", "cfg", "qsplit_seed", "model_seed"]).to_csv(out_runs, index=False)
                print(f"[OK] {out_runs}")

                agg_drop, agg_metrics, joined = agg_mean_std(run_drops, group_cols=["family", "dim", "cfg"])

                out_drop = results_dir_size / "AGG_mean_std_drop_by_family.csv"
                agg_drop.sort_values(by=["family", "drop_bal_acc_mean"]).to_csv(out_drop, index=False)
                print(f"[OK] {out_drop}")

                out_metrics = results_dir_size / "AGG_mean_std_metrics_by_family.csv"
                agg_metrics.sort_values(by=["family", "ood_bal_acc_mean"], ascending=[True, False]).to_csv(out_metrics, index=False)
                print(f"[OK] {out_metrics}")

                out_trade = results_dir_size / "AGG_ranking_tradeoff_by_family.csv"
                joined.sort_values(by=["family", "tradeoff_score"], ascending=[True, False]).to_csv(out_trade, index=False)
                print(f"[OK] {out_trade}")

                out_top = results_dir_size / "AGG_topK_by_family.txt"
                write_topk(joined, out_top, topk=10)
                print(f"[OK] {out_top}")

                if write_global_json_flag:
                    print(f"\n=== [SIZE {tag}] [EXTRA] JSON global simple (merge) ===")
                    write_global_json(results_dir_size, args.global_json_filename, only_baseline_json=only_baseline_json)

                print(f"\n[OK] [RUN {run_key}] [SIZE {tag}] TODO LISTO")
                print(f"Logs:    {logs_dir_size}")
                print(f"Results: {results_dir_size}")

            # run-level across-sizes aggregation
            print(f"\n=== [RUN {run_key}] [GLOBAL] Agregación across sizes ===")
            if args.dry_run:
                print("[DRY] run global aggregation skipped")
            else:
                try:
                    all_rows_v = load_all_summary_rows(run_results_root, master_tag_prefix=args.master_tag_prefix)
                    base_v = all_rows_v[all_rows_v["split"].isin(["id_test", "ood_test"])].copy()
                    run_drops_v = build_run_drops(base_v)

                    out_v_runs = run_results_root / "AGG_runs_drops_across_sizes.csv"
                    run_drops_v.to_csv(out_v_runs, index=False)
                    print(f"[OK] {out_v_runs}")

                    write_across_sizes_best(
                        results_root=run_results_root,
                        run_drops=run_drops_v,
                        dims=[int(d) for d in args.dims],
                        criterion=args.best_criterion,
                    )

                    a2_drop, a2_metrics, a2_joined = agg_mean_std(run_drops_v, group_cols=["size_tag", "family", "dim", "cfg"])
                    a2_drop.to_csv(run_results_root / "AGG_mean_std_drop__by_size.csv", index=False)
                    a2_metrics.to_csv(run_results_root / "AGG_mean_std_metrics__by_size.csv", index=False)
                    a2_joined.to_csv(run_results_root / "AGG_ranking_tradeoff__by_size.csv", index=False)
                    write_topk(a2_joined, run_results_root / "AGG_topK__by_size.txt", topk=10, context_cols=["size_tag"])

                except Exception as e:
                    print(f"[WARN] run across-sizes aggregation skipped ({run_key}): {e}")

    # -------------------------------------------------------------------------
    # ROOT-level summaries (everything)
    # -------------------------------------------------------------------------
    print("\n=== [ROOT] Agregación across EVERYTHING (variants + master_seeds + sizes) ===")

    if args.dry_run:
        write_index_experiment_context(args.results_root_dir / "INDEX_experiment_context.csv", index_rows)
        print("\n[OK] DRY-RUN COMPLETADO (index escrito)")
        return 0

    try:
        all_rows_root = load_all_summary_rows(args.results_root_dir, master_tag_prefix=args.master_tag_prefix)
        base_root = all_rows_root[all_rows_root["split"].isin(["id_test", "ood_test"])].copy()
        run_drops_root = build_run_drops(base_root)

        out_root_runs = args.results_root_dir / "AGG_runs_drops_across_all.csv"
        run_drops_root.to_csv(out_root_runs, index=False)
        print(f"[OK] {out_root_runs}")

        out_long = args.results_root_dir / "AGG_long_runs.csv"
        write_agg_long_runs(out_long, run_drops_root)

        write_contextual_summaries(args.results_root_dir, run_drops_root)

    except Exception as e:
        print(f"[WARN] Root across-all aggregation skipped: {e}")

    write_index_experiment_context(args.results_root_dir / "INDEX_experiment_context.csv", index_rows)

    print("\n[OK] COMPLETADO (variants + master_seeds + size sweep)")
    print(f"Root results: {args.results_root_dir}")
    print(f"Root logs:    {args.log_root_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# Ejemplos
# =============================================================================
# 1) Corrida rápida:
#   python .\scripts\ember\run_compare_q_vs_c_full.py ^
#     --variants m1_hist_byteent m2_hist_byteent ^
#     --master-seeds 42 123 ^
#     --sizes-preset baseline ^
#     --qsplit-seeds 42 123 ^
#     --model-seeds 42 123 ^
#     --dims 4 6 8 ^
#     --use-parallel --workers 4
#
# 2) Paper-ish completo:
#   python .\scripts\ember\run_compare_q_vs_c_full.py ^
#     --variants m1_hist_byteent m2_hist_byteent ^
#     --master-seeds 42 123 999 ^
#     --sizes-preset paper ^
#     --use-parallel --workers 4
#
# 3) Solo master splits:
#   python .\scripts\ember\run_compare_q_vs_c_full.py ^
#     --master-only --force-master-regenerate ^
#     --variants m1_hist_byteent m2_hist_byteent ^
#     --master-seeds 42 123 999
#
# 4) Dry-run:
#   python .\scripts\ember\run_compare_q_vs_c_full.py --dry-run
#
# 5) Si quieres que siga aunque reviente alguno:
#   python .\scripts\ember\run_compare_q_vs_c_full.py --continue-on-error --use-parallel --workers 4

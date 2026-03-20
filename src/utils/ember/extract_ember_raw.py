from __future__ import annotations

import argparse
import gzip
import shutil
import tarfile
import time
from pathlib import Path


def _is_within_directory(base: Path, target: Path) -> bool:
    """
    Protección TarSlip robusta:
    - Evita que un miembro del tar escriba fuera de out_dir.
    - Usa relative_to() para evitar falsos positivos por prefijos (p.ej. /out vs /outside).
    """
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def safe_extract_tar(archive: Path, out_dir: Path, *, forbid_links: bool = True) -> None:
    """
    Extrae un tar/tar.* de forma segura:
    - Valida path traversal (TarSlip).
    - (Opcional) Bloquea symlinks/hardlinks para evitar escrituras indirectas fuera.
    - Normaliza separadores y fuerza el nombre normalizado en el miembro antes de extraer.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()

        for m in members:
            # Defensa extra: evita symlinks/hardlinks (raro en datasets, pero mejor blindado)
            if forbid_links and (m.issym() or m.islnk()):
                raise RuntimeError(f"[SECURITY] symlink/hardlink not allowed in tar: {m.name}")

            # Normaliza separadores por si el tar trae backslashes
            name = m.name.replace("\\", "/")

            # Fuerza el nombre normalizado (para que extractall use lo que hemos validado)
            m.name = name

            dest = out_dir / name
            if not _is_within_directory(out_dir, dest):
                raise RuntimeError(f"[SECURITY] Tar path traversal detected: {name}")

        print(f"[INFO] tar members: {len(members)}")
        tar.extractall(out_dir)


def extract_gz(archive: Path, out_file: Path) -> None:
    """Extrae un .gz suelto (p.ej. .jsonl.gz) a un fichero plano."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(archive, "rb") as f_in, out_file.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def _suffix_str(p: Path) -> str:
    return "".join(p.suffixes).lower()


def _dir_has_files(p: Path) -> bool:
    return p.exists() and any(p.iterdir())


def _clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract EMBER raw archives safely (tar/tar.* or gz).")
    ap.add_argument(
        "--raw-archive",
        type=Path,
        default=Path("data/raw/ember/ember_dataset_2018_2.tar.bz2"),
        help="Ruta al archivo raw (.tar, .tar.gz, .tar.bz2 o .gz suelto).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/ember/extracted"),
        help="Directorio destino para la extracción.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-extrae aunque out-dir ya tenga contenido (por defecto limpia out-dir antes).",
    )
    ap.add_argument(
        "--force-no-clean",
        action="store_true",
        help="Si se usa con --force, NO limpia out-dir (extrae encima). No recomendado para reproducibilidad.",
    )
    ap.add_argument(
        "--allow-links",
        action="store_true",
        help="Permite symlinks/hardlinks dentro del tar (no recomendado).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="No extrae nada. Solo imprime detección de formato y paths.",
    )
    args = ap.parse_args()

    raw = args.raw_archive
    out_dir = args.out_dir

    if not raw.exists():
        raise FileNotFoundError(f"Missing raw archive: {raw}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if _dir_has_files(out_dir):
        if not args.force:
            print(f"[SKIP] out-dir already has content: {out_dir} (use --force to re-extract)")
            return
        # force
        if not args.force_no_clean:
            print(f"[CLEAN] removing out-dir content: {out_dir}")
            _clean_dir(out_dir)
        else:
            print(f"[WARN] --force-no-clean enabled: extracting on top of existing files in {out_dir}")

    suffix = _suffix_str(raw)
    is_tar = tarfile.is_tarfile(raw)
    is_gz = suffix.endswith(".gz") and not is_tar  # gz suelto (no tar.gz)

    forbid_links = not args.allow_links

    t0 = time.time()
    print(f"[INFO] archive={raw}")
    print(f"[INFO] suffix={suffix}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] is_tar={is_tar} | is_gz={is_gz}")
    print(f"[INFO] forbid_links={forbid_links}")
    print(f"[INFO] dry_run={args.dry_run}")

    if args.dry_run:
        print("[DRY] no extraction performed.")
        return

    if is_tar:
        safe_extract_tar(raw, out_dir, forbid_links=forbid_links)
        dt = time.time() - t0
        print(f"[OK] Extracted TAR archive to: {out_dir} ({dt:.1f}s)")
        return

    if is_gz:
        out_file = out_dir / raw.stem  # quita .gz
        extract_gz(raw, out_file)
        dt = time.time() - t0
        print(f"[OK] Extracted GZ to: {out_file} ({dt:.1f}s)")
        return

    raise RuntimeError(f"Unsupported archive format: {suffix}")


if __name__ == "__main__":
    main()

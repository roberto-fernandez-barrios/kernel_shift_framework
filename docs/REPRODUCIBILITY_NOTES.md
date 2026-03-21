# Reproducibility notes

This repository contains the manuscript-facing aggregated CSV files and table exports used to support the main empirical claims, together with the code paths needed to inspect, validate, and rerun the experimental pipeline in reduced form.

## What is reproducible from this snapshot

From this repository snapshot, a reader can reproduce or directly inspect:

- the repository structure and CLI entry points,
- the controlled kernel-swap methodology,
- the mapping between included result artifacts and manuscript-facing tables,
- the package-installation path defined by `pyproject.toml`,
- the recommended environment setup via `environment.yml`,
- the pinned top-level pip setup via `requirements.txt`,
- reduced real end-to-end validation of the pipeline,
- reporting-table regeneration from included aggregate artifacts.

## Why this snapshot is stronger than a minimal code supplement

Compared with a bare paper supplement, this release includes:

- explicit environment files,
- reusable scripts for extraction, export, split generation, orchestration, and reporting,
- smoke-test support,
- package-installation validation in CI,
- CI coverage on **Python 3.11 and 3.12**,
- manuscript-facing aggregate and table artifacts already organized in dedicated result folders.

## Dependency views in this repository

This repository intentionally ships multiple dependency views because they serve different reproducibility goals:

- `environment.yml` — recommended Conda environment for normal use
- `requirements.txt` — pinned top-level pip environment
- `requirements-compat.txt` — looser compatibility envelope
- `environment.lock.yml` — more explicit Conda snapshot, if needed
- `requirements.lock.txt` — more explicit pip snapshot, if needed

Normal use:

- use `environment.yml` for the standard Conda path,
- use `requirements.txt` for the standard pip path,
- use the lockfiles only when you specifically want a closer archival rerun.

## What still remains outside strict archival reproducibility

The following would strengthen *exact* archival reruns further, but are not required for this repository to function as a reproducible public software artifact:

- a fully hashed transitive pip lock generated from the exact machine used for the manuscript-facing runs,
- a fully explicit Conda export from that same final environment,
- detailed hardware/runtime notes for readers who want stricter comparison of the quantum branch runtime context,
- final legal confirmation that the chosen open-source license matches the actual rights holder.

## Important note on release archives and DOI records

The repository may evolve after a public archival release. Zenodo DOIs refer to specific archived release snapshots, not to the moving `main` branch.

That means:

- the GitHub default branch may contain newer documentation or metadata than an older archived DOI snapshot,
- readers seeking exact software citation should use the DOI associated with the specific public release they are citing.

## Important note on `--dry-run`

The `--dry-run` mode of `scripts/ember/run_compare_q_vs_c_full.py` should be understood as a **pipeline preview mode**.

It is useful for checking orchestration intent and command generation, but it is not a replacement for reduced real execution on prepared data.

## Python-version note

The recommended local setup is centered on **Python 3.12** for simplicity and consistency.

At the same time, public CI validates package installation and smoke/reporting paths on **Python 3.11 and 3.12**, which provides a stronger public compatibility signal than a single-version local recommendation.

## Practical conclusion

This repository should be understood as a **reproducible research framework with reduced end-to-end validation and release-grade reporting artifacts**, rather than as an exact machine-image capture of the original full paper-scale runtime environment.

That level of reproducibility is appropriate for a public research release and materially stronger than a repository that provides only code without validated execution paths or organized result artifacts.

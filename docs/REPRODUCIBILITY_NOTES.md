# Reproducibility Notes

This repository includes both the **code required to execute the experimental pipeline** and the **canonical final result artifacts** used to support the manuscript-level claims.

## Reproducibility level achieved in this release

This release has been validated through a **reduced end-to-end execution** of the workflow, including:

- EMBER extraction
- JSONL export to `X.npy / y.npy`
- master split generation
- q-split generation
- reduced classical execution
- reduced quantum execution
- reduced end-to-end orchestration
- final table generation from aggregated root-level CSVs

This is sufficient to demonstrate that the repository is operational as a reproducible research artifact.

## Canonical versioned outputs

The repository should treat the following as the **canonical final result artifacts**:

- `results/aggregated/`
- `results/tables/`

These are the outputs that should remain under version control because they are compact, interpretable, and directly useful for readers and reviewers.

## Intentionally excluded artifacts

The following classes of artifacts are intentionally excluded from version control because they are large, local, regenerable, or debug-oriented:

- extracted EMBER raw data
- processed `X.npy / y.npy` arrays
- generated split directories
- sanity validation directories
- local pipeline-validation internals
- memmaps
- local logs
- regenerable smoke-test reports

This separation is deliberate: it keeps the public repository focused on **reproducibility, readability, and auditability**, not on storing every intermediate runtime artifact.

## Environment files provided

The repository provides three complementary environment descriptions:

- `requirements.txt` — pinned top-level environment
- `requirements-compat.txt` — looser compatibility envelope
- `environment.yml` — Conda environment entry point

These are sufficient for practical reproducibility.

## Optional stronger closure

For an archival-grade release, the following optional additions would further strengthen reproducibility:

- `environment.lock.yml` exported from the validated machine
- `requirements.lock.txt` from `pip freeze`
- brief hardware/runtime notes for the quantum branch
- archival DOI and release tag metadata

These are improvements, not blockers.

## Dry-run note

The `--dry-run` mode of `scripts/ember/run_compare_q_vs_c_full.py` should be understood as a **preview of the execution plan**, not as a substitute for a real reduced run.

In this repository state, both have been validated:

- the reduced real run confirms operational behavior
- the corrected `--dry-run` confirms preview integrity

## Practical conclusion

The repository now offers a sound level of practical reproducibility for public release:

- the workflow is executable
- the final reporting stage is executable
- the canonical outputs are preserved
- non-essential runtime internals are excluded

That is the right balance for a publication-facing research repository.

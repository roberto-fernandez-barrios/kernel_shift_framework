# Validation status

This repository has been validated beyond static inspection and beyond CLI-only checks. The public codebase supports a **reduced but real end-to-end execution path**, a **reporting path**, and a **package-installation path** validated through continuous integration.

## Validation summary

### 1. Package and CI validation
The repository is structured as an installable Python package via `pyproject.toml`.

Current CI validates:

- package installation via `pip install .`,
- CLI smoke-test execution,
- reporting on versioned aggregate inputs,
- support on **Python 3.11 and 3.12**.

This means the public automation now validates the **package itself**, not only a manually assembled local environment.

### 2. CLI validation
The smoke-test script completed successfully:

```bash
python scripts/smoke/smoke_test_cli.py
```

Validated entry points include:

- `python -m src.utils.ember.extract_ember_raw --help`
- `python -m src.utils.ember.export_ember_jsonl_to_Xy --help`
- `python -m src.utils.ember.make_splits_ember_sparsity --help`
- `python -m src.utils.ember.make_qsplits_from_master --help`
- `python scripts/ember/run_compare_q_vs_c_full.py --help`
- `python -m src.experiments.ember.classical.run_ember_classical_kernel_sparsity_shift_qsplits --help`
- `python -m src.experiments.ember.quantum.run_ember_quantum_kernel_sparsity_shift_qsplits --help`

The smoke test writes:

- `docs/smoke_test_report.json`

### 3. Data-preparation validation
The following stages were executed successfully on a real EMBER archive:

- raw archive extraction,
- JSONL to `X.npy / y.npy` export.

Observed outputs include:

- `data/processed/ember/X.npy`
- `data/processed/ember/y.npy`
- `data/processed/ember/feature_names.json`
- `data/processed/ember/meta_export.json`

### 4. Split-generation validation
Both master-shift variants were exercised successfully:

- `m1_hist_byteent`
- `m2_hist_byteent`

The q-split generation stage was also executed successfully on reduced sanity settings.

Validated scripts:

- `src.utils.ember.make_splits_ember_sparsity`
- `src.utils.ember.make_qsplits_from_master`

### 5. Model-run validation
Both experiment branches were executed successfully on reduced q-splits:

- classical branch: `src.experiments.ember.classical.run_ember_classical_kernel_sparsity_shift_qsplits`
- quantum branch: `src.experiments.ember.quantum.run_ember_quantum_kernel_sparsity_shift_qsplits`

This validates:

- train-only preprocessing,
- q-split consumption,
- classical kernel execution,
- quantum kernel execution,
- summary JSON / CSV generation.

### 6. Orchestrator validation
The orchestration script was validated in both modes:

1. **real reduced end-to-end run**
2. **clean `--dry-run` preview mode**

Validated script:

- `scripts/ember/run_compare_q_vs_c_full.py`

The reduced real run completed successfully for a minimal sanity configuration, confirming that the orchestration path remains operational on real artifacts.

The `--dry-run` mode was also verified as a genuine preview mode rather than a failing pre-execution path.

### 7. Reporting validation
The reporting stage was validated successfully.

Validated script:

- `scripts/reporting/make_summary_tables.py`

Validated outputs include:

- `table_18settings_best_ood__with_std.csv`
- `table_18settings_best_id__with_std.csv`
- `table_18settings_best_drop__with_std.csv`
- `table_90cells_dimlevel_best_ood__with_std.csv`
- `table_top10_cases_ood__with_std.csv`

## What has been validated overall

The repository has been validated at the following levels:

- package installation,
- repository structure,
- module imports,
- CLI entry points,
- raw data preparation,
- split generation,
- reduced classical execution,
- reduced quantum execution,
- reduced end-to-end orchestration,
- reporting / table generation.

## What this document does not claim

This document does **not** claim that the full paper-scale experiment matrix was re-executed as part of the final packaging pass.

That is not required to establish that the released repository is functional, reproducible in a reduced end-to-end sense, and suitable as a public research software artifact.

## Practical conclusion

The repository should be considered **functionally validated for public release** as a reproducible research codebase.

Any remaining work is release hygiene rather than core debugging, for example:

- metadata polish,
- citation polish,
- optional environment-lock refinement,
- optional future-proofing patches for ecosystem warnings or deprecations.

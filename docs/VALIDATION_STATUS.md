# Validation Status

This repository has been validated beyond CLI-level checks. The current release candidate has been exercised through a **real reduced end-to-end run**, a **working dry-run preview**, and a **validated reporting stage**.

## Validation scope

Validation was performed at five levels:

1. **CLI integrity** — module imports and entry points
2. **Data preparation** — extraction and EMBER export
3. **Split generation** — master splits and q-splits
4. **Model execution** — reduced classical and quantum runs
5. **Pipeline orchestration and reporting** — reduced end-to-end orchestration plus final table generation

This is sufficient to treat the repository as a **functionally validated reproducible research codebase**.

## 1. CLI validation

The smoke-test script completed successfully:

```bash
python scripts/smoke/smoke_test_cli.py
```

Validated entry points:

- `python -m src.utils.ember.extract_ember_raw --help`
- `python -m src.utils.ember.export_ember_jsonl_to_Xy --help`
- `python -m src.utils.ember.make_splits_ember_sparsity --help`
- `python -m src.utils.ember.make_qsplits_from_master --help`
- `python scripts/ember/run_compare_q_vs_c_full.py --help`
- `python -m src.experiments.ember.classical.run_ember_classical_kernel_sparsity_shift_qsplits --help`
- `python -m src.experiments.ember.quantum.run_ember_quantum_kernel_sparsity_shift_qsplits --help`

## 2. Data-preparation validation

The following steps were executed successfully on a real EMBER archive:

- raw archive extraction with `extract_ember_raw.py`
- JSONL to `X.npy / y.npy` export with `export_ember_jsonl_to_Xy.py`

Observed outputs:

- `data/processed/ember/X.npy`
- `data/processed/ember/y.npy`
- `data/processed/ember/feature_names.json`
- `data/processed/ember/meta_export.json`

## 3. Split-generation validation

Both master-shift variants were executed successfully:

- `m1_hist_byteent`
- `m2_hist_byteent`

Reduced q-split generation was also executed successfully on sanity settings.

Validated scripts:

- `src.utils.ember.make_splits_ember_sparsity`
- `src.utils.ember.make_qsplits_from_master`

## 4. Model-run validation

Both experiment branches were executed successfully on reduced q-splits:

- classical branch: `src.experiments.ember.classical.run_ember_classical_kernel_sparsity_shift_qsplits`
- quantum branch: `src.experiments.ember.quantum.run_ember_quantum_kernel_sparsity_shift_qsplits`

This validates:

- train-only preprocessing
- q-split consumption
- classical kernel execution
- quantum kernel execution
- summary JSON / CSV generation

## 5. Orchestrator validation

The orchestration script was validated in both modes:

1. **real reduced end-to-end run**
2. **clean `--dry-run` preview mode**

Validated script:

- `scripts/ember/run_compare_q_vs_c_full.py`

The reduced real run completed successfully for:

- variants: `m1_hist_byteent`, `m2_hist_byteent`
- master seeds: `42`
- q-split seeds: `42`
- model seeds: `42`
- size grid: `(200, 100, 100)`
- dimensions: `4`

The `--dry-run` path was corrected and verified so that it now behaves as a true preview of the pipeline rather than failing during pre-execution q-split validation.

## 6. Reporting validation

The reporting stage was also validated successfully.

Validated script:

- `scripts/reporting/make_summary_tables.py`

Validated outputs:

- `table_18settings_best_ood__with_std.csv`
- `table_18settings_best_id__with_std.csv`
- `table_18settings_best_drop__with_std.csv`
- `table_90cells_dimlevel_best_ood__with_std.csv`
- `table_top10_cases_ood__with_std.csv`

## What this validation does not claim

This document does **not** claim that the full paper-scale experiment was re-executed as part of the final repository-packaging pass.

That is not required to demonstrate that the repository itself is operational, coherent, and reproducible at a reduced end-to-end level.

## Practical conclusion

The repository should now be considered **functionally validated for public release**.

The remaining work before publication is no longer core debugging. It is release-quality finishing work:

- final artifact selection
- final metadata review
- CI confirmation
- optional environment lock export
- optional maintenance patches for future warnings and deprecations

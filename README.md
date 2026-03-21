# Controlled Kernel Evaluation under Distribution Shift

[![CI](https://github.com/roberto-fernandez-barrios/kernel_shift_framework/actions/workflows/ci.yml/badge.svg)](https://github.com/roberto-fernandez-barrios/kernel_shift_framework/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/roberto-fernandez-barrios/kernel_shift_framework)](https://github.com/roberto-fernandez-barrios/kernel_shift_framework/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19149199.svg)](https://doi.org/10.5281/zenodo.19149199)

A reproducible experimental framework for **controlled kernel comparison under distribution shift**.

**Case study implemented in this release:** EMBER static malware detection  
**Associated manuscript:**  
**_Comparing Quantum and Classical Kernels under Distribution Shift: A Controlled Kernel-Swap Study on EMBER Malware Detection_**

This repository should be read as a **reproducible experimental framework**, not as a flat code supplement. Its central design principle is a **controlled kernel-swap protocol** in which the dataset, split logic, preprocessing, classifier family, repeated-run structure, and reporting views are held fixed while only the **kernel** changes.

In this release, the protocol is instantiated on EMBER malware detection to compare strong classical kernels against fidelity-based quantum kernels under explicit distribution shift.

---

## Why this repository exists

The framework addresses a narrow and explicit question:

> **When the evaluation protocol is controlled, does kernel choice materially affect in-distribution fit and out-of-distribution robustness?**

This repository does **not** claim hardware quantum advantage, universal superiority of quantum methods, or complexity-theoretic separation. Its purpose is empirical and protocol-driven: under a fixed classifier and a fixed evaluation pipeline, does changing only the kernel alter performance under shift?

---

## Controlled comparison design

Across model families, the following components are kept fixed:

- dataset and exported feature representation,
- master split construction,
- q-split subsampling logic,
- train-only preprocessing,
- classifier family: `SVC(kernel="precomputed", class_weight="balanced")`,
- evaluation metrics and summary views,
- repeated-run design.

The only intentionally changing factor is the **kernel**.

```text
same data
same train / ID / OOD protocol
same train-only preprocessing
same classifier family
same repeated-run design
different kernel
```

That protocol-level fairness is the main organizing principle of this repository.

---

## Case study included in this release

### Dataset
- **EMBER** static PE malware dataset

### Exported representation
The export stage builds a fixed **524-dimensional** representation composed of:

- `256` histogram features,
- `256` byte-entropy features,
- `5` string statistics,
- `7` general PE statistics / flags.

### Shift variants
Two explicit OOD protocols are implemented:

- **`m1_hist_byteent`**: OOD from within-class score extremes over histogram + byte-entropy features
- **`m2_hist_byteent`**: OOD from within-class distance-to-train extremes anchored on the final training split

### Shared learning setup
- **classifier:** `SVC(kernel="precomputed", class_weight="balanced")`
- **shared preprocessing:** `MaxAbsScaler` + `TruncatedSVD`
- **dimension sweep:** `d ∈ {4, 6, 8, 10, 12}`

### Classical kernels
- linear
- RBF (`gamma="scale"`)

### Quantum kernels
- `ZZFeatureMap`, `reps=1`, `entanglement="full"`
- `ZZFeatureMap`, `reps=2`, `entanglement="full"`
- `PauliFeatureMap(['X', 'Z'])`, `reps=1`, `entanglement="full"`
- `ZFeatureMap`, `reps=2`

### Repeated-run design
- **master seeds:** `42, 123, 999`
- **q-split seeds:** `42, 123, 999, 7, 2024`
- **model seeds:** `42, 123, 999`
- **runs per evaluated configuration:** `5 × 3 = 15`

### Paper size preset
- `S = (1000, 500, 500)`
- `M = (2000, 1000, 1000)`
- `L = (4000, 1800, 1800)`

This yields:

- **18 principal settings** = `2 variants × 3 master seeds × 3 sizes`
- **90 setting–dimension cells** = `18 × 5 dimensions`

---

## Snapshot results included in this release

Under the robustness-oriented **Best-by-OOD** view, the selected quantum model achieves higher OOD balanced accuracy in **14 / 18** principal settings, with:

- **mean OOD gain:** `+0.0189`
- **median OOD gain:** `+0.0255`
- **settings with gain larger than the combined variability scale:** `9 / 18`

Under the separability-oriented **Best-by-ID** view, the selected quantum model improves ID balanced accuracy in **18 / 18** principal settings, with:

- **mean ID gain:** `+0.0737`
- **median ID gain:** `+0.0788`

At the finer dimension level, quantum kernels are favorable in **78 / 90** setting–dimension cells under Best-by-OOD selection.

**Important note.** The paper-scale claims above refer to the full aggregated release artifacts in `results/aggregated/` and `results/tables/`. Reduced outputs produced during local validation, such as `results/pipeline_validation/...`, are intended to validate the pipeline and should not be treated as replacements for the manuscript-facing tables.

---

## Quick start

### Recommended path: Conda

This repository targets a clean **Python 3.12** environment for the recommended local setup.

```bash
conda env create -f environment.yml
conda activate kernel-shift-framework
```

### Alternative path: virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install .
```

---

## Dependency files at a glance

This repository intentionally ships multiple dependency views, each for a different purpose:

- `environment.yml`  
  Recommended **Conda** environment for typical use of this release.

- `requirements.txt`  
  Pinned **top-level pip environment** for this release.

- `requirements-compat.txt`  
  Looser compatibility envelope for users who need a less restrictive install.

- `environment.lock.yml`  
  More explicit Conda environment snapshot for archival or exact reruns, when needed.

- `requirements.lock.txt`  
  More explicit pip environment snapshot for archival or exact reruns, when needed.

In short:

- use **`environment.yml`** for the normal Conda path,
- use **`requirements.txt`** for the normal pip path,
- use the **lockfiles** when you specifically want a closer archival rerun of the recorded software snapshot.

---

## Package installation and CI coverage

The repository is structured as an installable Python package via `pyproject.toml`.

Current CI validates:

- package installation via `pip install .`,
- CLI smoke-test execution,
- reporting on versioned aggregated outputs,
- compatibility on **Python 3.11 and 3.12**.

This means the public automation now validates the **package**, not just a manually assembled local environment.

---

## Expected data layout

```text
data/
├── raw/ember/
│   ├── ember_dataset_2018_2.tar.bz2
│   └── extracted/
└── processed/ember/
```

This repository does **not** redistribute the EMBER dataset itself. The intended workflow is to obtain the raw archive externally and run the preparation pipeline locally.

---

## Repository layout

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── environment.yml
├── environment.lock.yml
├── requirements.txt
├── requirements.lock.txt
├── requirements-compat.txt
├── src/
│   ├── utils/ember/
│   └── experiments/ember/
├── scripts/
│   ├── ember/
│   ├── reporting/
│   └── smoke/
├── results/
│   ├── aggregated/
│   └── tables/
├── paper/
│   ├── sn-article.tex
│   └── sn-bibliography.bib
├── docs/
│   ├── FINAL_RELEASE_CHECKLIST.md
│   ├── REPRODUCIBILITY_NOTES.md
│   ├── VALIDATION_STATUS.md
│   └── LEGAL_RELEASE_NOTE.md
└── data/
```

Top-level intent:

- `src/` contains reusable code and experiment logic
- `scripts/` contains runnable orchestration and reporting entry points
- `results/` contains aggregated outputs and manuscript-facing tables
- `paper/` contains the manuscript source
- `docs/` contains supporting release and reproducibility notes

---

## CLI sanity checks

From the repository root:

```bash
python -m src.utils.ember.extract_ember_raw --help
python -m src.utils.ember.export_ember_jsonl_to_Xy --help
python -m src.utils.ember.make_splits_ember_sparsity --help
python -m src.utils.ember.make_qsplits_from_master --help
python scripts/ember/run_compare_q_vs_c_full.py --help
python scripts/smoke/smoke_test_cli.py
```

These commands validate module layout and entry points. They do **not** by themselves guarantee data availability or full numerical reproduction.

---

## Validation status

A repeatable CLI smoke test is included:

```bash
python scripts/smoke/smoke_test_cli.py
```

This writes:

```text
docs/smoke_test_report.json
```

In addition to CLI validation, the repository has been manually validated on a reduced but real end-to-end run:

- EMBER raw extraction succeeded
- JSONL → `X.npy / y.npy` export succeeded
- both master split variants (`m1_hist_byteent`, `m2_hist_byteent`) succeeded
- q-splits were generated successfully
- classical and quantum sanity runs completed successfully
- the comparison orchestrator completed a reduced real run successfully
- `--dry-run` behaves as a true preview mode
- the summary-table generator completed successfully on validation aggregates

Taken together, the repository is validated at four levels:

1. **package level** — installable via `pip install .`
2. **CLI level** — entry points and module layout
3. **runtime sanity level** — reduced real classical and quantum execution
4. **orchestration/reporting level** — reduced end-to-end pipeline including table generation

---

## End-to-end workflow

### 1) Extract the raw EMBER archive safely

```bash
python -m src.utils.ember.extract_ember_raw \
  --raw-archive data/raw/ember/ember_dataset_2018_2.tar.bz2 \
  --out-dir data/raw/ember/extracted
```

### 2) Export EMBER JSONL into NumPy arrays

```bash
python -m src.utils.ember.export_ember_jsonl_to_Xy \
  --ember-dir data/raw/ember/extracted/ember2018 \
  --out-dir data/processed/ember \
  --dtype float32
```

Main outputs:

- `data/processed/ember/X.npy`
- `data/processed/ember/y.npy`
- `data/processed/ember/feature_names.json`
- `data/processed/ember/meta_export.json`

### 3) Generate a master split manually (optional)

Example for `m1_hist_byteent`:

```bash
python -m src.utils.ember.make_splits_ember_sparsity \
  --in-dir data/processed/ember \
  --out-dir data/processed/ember/splits_sparsity__m1_hist_byteent__ms42 \
  --seed 42 \
  --ood-mode score_extremes_within_class \
  --use-hist \
  --use-byteent \
  --ood-test-frac 0.15 \
  --ood-extreme-frac-each-side 0.075 \
  --score-mode nnz \
  --eps 0.0 \
  --mmap \
  --strict-fracs
```

Example for `m2_hist_byteent`:

```bash
python -m src.utils.ember.make_splits_ember_sparsity \
  --in-dir data/processed/ember \
  --out-dir data/processed/ember/splits_sparsity__m2_hist_byteent__ms42 \
  --seed 42 \
  --ood-mode dist_to_train_within_class \
  --use-hist \
  --use-byteent \
  --ood-test-frac 0.15 \
  --svd-dim 128 \
  --save-provisional \
  --mmap \
  --strict-fracs
```

### 4) Generate q-splits manually (optional)

```bash
python -m src.utils.ember.make_qsplits_from_master \
  --src data/processed/ember/splits_sparsity__m1_hist_byteent__ms42 \
  --dst-root data/processed/ember \
  --seed 42 \
  --n-train 1000 \
  --n-id 500 \
  --n-ood 500 \
  --use-low-high \
  --strict-sizes
```

### 5) Run the manuscript-oriented comparison pipeline

```bash
python scripts/ember/run_compare_q_vs_c_full.py \
  --master-in-dir data/processed/ember \
  --variants m1_hist_byteent m2_hist_byteent \
  --master-seeds 42 123 999 \
  --qsplit-seeds 42 123 999 7 2024 \
  --model-seeds 42 123 999 \
  --sizes-preset paper \
  --dims 4 6 8 10 12 \
  --best-criterion tradeoff
```

This is the main path used to generate master splits, q-splits, model runs, summaries, aggregated views, and manifests.

### 6) Preview the full pipeline without execution

```bash
python scripts/ember/run_compare_q_vs_c_full.py \
  --dry-run \
  --master-in-dir data/processed/ember \
  --variants m1_hist_byteent m2_hist_byteent \
  --master-seeds 42 \
  --qsplit-seeds 42 \
  --model-seeds 42 \
  --size-grid 200,100,100 \
  --dims 4 \
  --best-criterion tradeoff
```

The `--dry-run` mode is a **pipeline preview**, not a substitute for real numerical validation.

---

## Reporting and manuscript-facing tables

Once root-level aggregated CSVs are available, generate manuscript-facing tables with:

```bash
python scripts/reporting/make_summary_tables.py \
  --metrics results/aggregated/AGG_ROOT_mean_std_metrics__by_variant_seed_and_size.csv \
  --drops results/aggregated/AGG_ROOT_mean_std_drop__by_variant_seed_and_size.csv \
  --outdir results/tables \
  --topn 10 \
  --round 4
```

This writes:

- `results/tables/table_18settings_best_ood__with_std.csv`
- `results/tables/table_18settings_best_id__with_std.csv`
- `results/tables/table_18settings_best_drop__with_std.csv`
- `results/tables/table_90cells_dimlevel_best_ood__with_std.csv`
- `results/tables/table_top10_cases_ood__with_std.csv`

For local sanity validation, you may also run the same script on reduced validation aggregates such as `results/pipeline_validation/...`, but those outputs should be read as **sanity tables**, not as replacements for the paper-scale results.

---

## Output mapping

### Aggregated outputs
- `results/aggregated/AGG_ROOT_mean_std_metrics__by_variant_seed_and_size.csv`  
  principal-setting mean/std metrics
- `results/aggregated/AGG_ROOT_mean_std_drop__by_variant_seed_and_size.csv`  
  robustness-drop summaries by variant, seed, and size
- `results/aggregated/AGG_ROOT_ranking_tradeoff__by_variant_seed_and_size.csv`  
  family-level ranking summaries under the tradeoff view
- `results/aggregated/AGG_ROOT_topK__by_variant_seed_and_size.txt`  
  human-readable top-K ranking summaries

### Manuscript-facing tables
- `results/tables/table_18settings_best_ood__with_std.csv`  
  main Best-by-OOD principal-setting table
- `results/tables/table_18settings_best_id__with_std.csv`  
  Best-by-ID principal-setting table
- `results/tables/table_18settings_best_drop__with_std.csv`  
  principal-setting robustness-drop table
- `results/tables/table_90cells_dimlevel_best_ood__with_std.csv`  
  dimension-level Best-by-OOD table
- `results/tables/table_top10_cases_ood__with_std.csv`  
  top OOD cases

---

## Selection views

Two selection views are kept explicit.

### Best-by-OOD
Within each family, select the configuration with the strongest OOD performance. This is the primary robustness-oriented view and the main basis for the headline robustness claim.

### Best-by-ID
Within each family, select the configuration with the strongest ID performance. This is a separability-oriented view and answers a different question from the robustness claim.

Keeping both views explicit prevents robustness claims from being smuggled in through an ID-only selection strategy.

---

## Reproducibility status

This repository snapshot is suitable for:

- methodology inspection,
- result traceability,
- reduced real execution validation,
- package-level installation checks,
- manuscript-facing reporting.

The core scientific pipeline is already released and archived. Any future refinements should be interpreted as metadata, packaging, or presentation improvements rather than changes to the validated experimental design.

---

## Supporting documentation

- `docs/REPRODUCIBILITY_NOTES.md`
- `docs/VALIDATION_STATUS.md`
- `docs/LEGAL_RELEASE_NOTE.md`
- `docs/FINAL_RELEASE_CHECKLIST.md`

---

## Citation

Software citation metadata is provided in `CITATION.cff`.

If you use this repository, please cite the archived software release associated with the public Zenodo record below.

### Archived software release
- DOI: `10.5281/zenodo.19149199`

### Recommended software citation

```bibtex
@software{fernandez_barrios_2026_kernel_shift_framework,
  author  = {Fernández-Barrios, Roberto and Pastor-López, Iker and González-Santocildes, Asier and Garcia Bringas, Pablo},
  title   = {Controlled Kernel Evaluation under Distribution Shift},
  year    = {2026},
  version = {0.1.1},
  doi     = {10.5281/zenodo.19149199},
  url     = {https://doi.org/10.5281/zenodo.19149199}
}
```

If relevant to your work, please also cite the accompanying manuscript:

**_Comparing Quantum and Classical Kernels under Distribution Shift: A Controlled Kernel-Swap Study on EMBER Malware Detection_**

---

## License

This repository is distributed under the **BSD 3-Clause License**. See `LICENSE`.

---

## Scope note

This repository is a **controlled empirical comparison framework**. It should not be interpreted as evidence of universal quantum advantage, and it does not attempt to make claims beyond the experimental scope described above.
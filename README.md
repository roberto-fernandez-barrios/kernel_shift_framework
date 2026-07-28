# Controlled Kernel Evaluation under Distribution Shift

[![CI](https://github.com/roberto-fernandez-barrios/kernel_shift_framework/actions/workflows/ci.yml/badge.svg)](https://github.com/roberto-fernandez-barrios/kernel_shift_framework/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/roberto-fernandez-barrios/kernel_shift_framework)](https://github.com/roberto-fernandez-barrios/kernel_shift_framework/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19147649.svg)](https://doi.org/10.5281/zenodo.19147649)

Reproducible framework for the **controlled comparison of quantum and classical kernels under distribution shift** — the artifact behind the manuscript:

> **Evaluation Choices Determine Apparent Quantum-Kernel Robustness under Distribution Shift**

Within each experimental setting, the classifier, preprocessing, and splits are held fixed; **only the kernel changes**. Every configuration's regularization is tuned on training data alone, configurations are selected without out-of-distribution labels, and both families search an equal candidate budget.

![Protocol](docs/assets/fig_v4_protocol.png)

## The study at a glance

- **4 benchmark scenarios, 2 modalities**: EMBER (static PE malware), UNSW-NB15 (DoS, Reconnaissance) and ToN-IoT (Scanning) network flows.
- **3 shift mechanisms**: label-conditional sparsity-tail and train-centroid-tail stress tests, plus a **held-out attack-campaign shift** on network traffic (unseen attack campaign + capture-partition change).
- **2 classifier families** consuming the same precomputed Gram matrices: SVC and a Laplace-approximation **Gaussian process classifier** (whose calibration under shift is assessed, not assumed).
- **35 kernel geometries** (23 classical + 12 fidelity feature maps) with symmetric length-scale tuning and per-configuration `C` tuned by cross-validation on train; **equal candidate budgets**.
- **8 scenario-groups × 15 pipeline realizations**, evaluated under **honest ID-validation selection** (P1′), reported as **fixed case studies with conditional intervals** — no population *p*-value.
- **Finite-shot fidelity-estimation model** to test how far the statevector-exact geometry transfers to finite measurement.

## Key findings

![No robust advantage under honest, budget-matched selection](docs/assets/fig_v4_honest.png)

1. Under the **test-peeking oracle selection and fixed regularization** that dominate the optimistic literature, fidelity kernels appear to beat linear+RBF baselines by up to $+0.037$ OOD balanced accuracy.
2. Once each configuration's `C` is tuned on training data alone, selection uses **no OOD labels** (an ID-validation split), and budgets are matched, no robust family-level advantage remains. Against the equal-budget extended classical family, the dataset-equal-weighted $\Delta_{\mathrm{OOD}}$ is **-0.005 for SVC** and **-0.002 for GPC**; against linear+RBF it is **-0.004** and **+0.004**, respectively. The primary SVC comparison is classical-favoured or conditionally tied in **7 of 8 scenario-groups**.
3. **At matched effective rank the classical kernels are at least as accurate in every scenario-group.** The geometry that carries information about robustness (effective rank, OOD alignment) is a **regime-dependent association**, not a governing law, and it favours neither family.
4. A **finite-shot analysis** shows the quantum kernels' geometry is a statevector-exact idealization: effective rank inflates under finite estimation, while alignment and accuracy survive.

**Bottom line:** under equal candidate budgets and no-OOD-label selection, fidelity-based quantum kernels show **no robust out-of-distribution advantage** over well-tuned classical kernels; a short-length-scale Laplacian is a strong, inexpensive baseline.

![At matched geometry, classical kernels are at least as accurate](docs/assets/fig_v4_rankmatched.png)

## Repository layout

```text
src/
  utils/ember/       EMBER export + master/q-split construction
  utils/netflow/     network-flow export + shift constructions (m2-centroid, natural)
  experiments/       kernel-swap runners (classical, quantum, extended+GPC)
  analysis/          kernel-geometry descriptors (eff. rank, KTA, geometric difference)
scripts/
  ember/  netflow/   grid drivers (settings x seeds x sizes)
  analysis/          family comparisons, fixed-case inference, mechanism tests
  reporting/         every table and figure of the paper, generated from results/
results/             frozen per-run summaries and v4 confirmatory outputs
manuscript/          main and Supplementary LaTeX/PDF, figures, cover letters
```

## Reproducing

```bash
conda env create -f environment.yml && conda activate kernel-shift-framework

# Rebuild the provenance audit, confirmatory analysis, tables, and figures
# from the 1080 frozen v4 per-run summaries.
python scripts/reproduce_v4.py --stage all

# Verify the protocol utilities and reporting traceability.
python -m pytest -q
```

The master command deliberately does not rerun the expensive Phase-3 experiment grid; it validates and consumes the versioned `summary_v4.csv` files. To recompute that grid from the source benchmark data, use `scripts/experiments/run_v4_all.py` after placing EMBER 2018 feature-version-2 data under `data/raw/ember/` and preparing the public UNSW-NB15 and ToN-IoT inputs documented under `src/utils/netflow/`. The frozen analysis contract is [`docs/ANALYSIS_SPEC_V4.md`](docs/ANALYSIS_SPEC_V4.md).

## Manuscript and citation

The main manuscript and Supplementary Information live in [`manuscript/`](manuscript/) (Springer Nature-compatible main source plus a standalone Supplementary file). If you use this software, please cite it via [`CITATION.cff`](CITATION.cff) (Zenodo DOI: [10.5281/zenodo.19147649](https://doi.org/10.5281/zenodo.19147649)).

## License

BSD-3-Clause. Benchmark datasets remain subject to their original licenses.

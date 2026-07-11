# Cover letter — EPJ Quantum Technology

Dear Editors,

We submit for your consideration our manuscript "Quantum and Classical Kernels
under Distribution Shift: Kernel Geometry Governs Out-of-Distribution
Robustness" for publication in EPJ Quantum Technology as a Research article.

**Why EPJ Quantum Technology.** The manuscript addresses a question central to
the practical deployment of quantum kernel methods, within the journal's scope
of quantum information and computation: when, and why, do fidelity-based
quantum kernels generalize better than classical kernels once the data
distribution shifts? We answer it with what we believe is the most tightly
controlled and broadest benchmark of its kind: a kernel-swap protocol (fixed
classifier, preprocessing, splits, selection logic, and --- crucially ---
symmetric bandwidth-tuning freedom for both families) instantiated on four
public security benchmarks across two data modalities and three drift
mechanisms including natural regime drift, two classifier families (SVC and a
Laplace-approximation Gaussian process classifier with calibrated
uncertainty), and 72 principal settings with 15 repeated runs each, evaluated
with paired Wilcoxon tests under Holm correction.

The central contribution is a *design principle for quantum kernel methods
under drift*: out-of-distribution accuracy is governed by the survival of
kernel-target alignment under shift, a property enabled by the high effective
rank of the Gram matrices that quantum feature maps induce natively. This
mechanism holds across all four datasets (positive within-setting correlation
in 89--99% of 360 setting-seed units), predicts which classical kernels are
competitive, and extends the known importance of quantum kernel bandwidth
(Shaydulin & Wild, PRA 2022; Canatar et al., TMLR 2023) from in-distribution
generalization to robustness --- showing that asymmetric tuning freedom biases
quantum-versus-classical comparisons against the quantum family. On drifting
network traffic under the probabilistic classifier, the quantum family wins
all 54 settings against a strong extended classical family. We believe this
complements the empirical quantum-kernel evaluations EPJ Quantum Technology
has recently published (e.g., quantum-kernel anomaly detection and
quantum-kernel learning studies) with a mechanism-level account.

All quantum kernels are computed by exact statevector simulation; the paper
makes no hardware claim and states this scope explicitly. The complete
artifact (code, split definitions, results, and the scripts generating every
table and figure) is archived on Zenodo (DOI: 10.5281/zenodo.19152497).

**Declarations.** The authors have no competing interests. All authors have
approved the manuscript for submission. The content has not been published,
and is not under consideration, elsewhere; a preprint of an earlier,
substantially narrower version is available on arXiv and will be updated in
accordance with journal policy.

**Suggested reviewers** (researchers whose published work is directly adjacent
to this study; to be confirmed/completed before submission):
- [Candidato 1 — p. ej. autor/a de trabajos de bandwidth en quantum kernels]
- [Candidato 2 — p. ej. autor/a de benchmarking de QML]
- [Candidato 3 — p. ej. autor/a de quantum kernels aplicados a detección]

Thank you for your consideration.

On behalf of all authors,
Roberto Fernández-Barrios
University of Deusto, Bilbao, Spain

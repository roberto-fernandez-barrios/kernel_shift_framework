# v6 reviewer-driven correction and finite-shot Monte Carlo specification

**Status: FROZEN on 2026-07-29 before any v6 finite-shot Monte Carlo
outcome was computed.**

This specification starts from the immutable `v0.5.0` release at merge commit
`f042a680b1cd71546242edbc40b9054ec5382caa`. It responds to an independent
pre-submission review of the manuscript for the npj Quantum Information
Collection "Quantum machine learning: understanding capabilities,
limitations, and perspectives for quantum advantage."

The v4 and v5 raw experimental outputs remain immutable. Corrected and new
derived outputs are written below `results/v6/`. The objective is correction
and stronger reporting, not selection of a more favourable result.

## 1. Corrected cross-dataset estimand

Every security analysis that is described as "dataset-equal" must use the
three source datasets:

- every `ember_*` scenario-group maps to `ember`;
- every `unsw_*` scenario-group maps to `unsw`;
- every `toniot_*` scenario-group maps to `toniot`.

Within a model and endpoint, scenario-group effects are averaged within source
dataset and the three source-dataset means receive equal weight. The
leave-one-dataset-out range omits one of these three source datasets at a time.
No script may treat UNSW-NB15 DoS and Reconnaissance as separate datasets.

The corrected hierarchical summaries, specification curve, tables, figures,
and prose must agree exactly. Automated tests will assert the three-dataset
mapping and the equality of the corrected confirmatory S10 endpoint and the
S10 specification-curve endpoint.

The geometry portability analysis will likewise leave out one of the three
source datasets. A model is therefore trained on two source datasets and
evaluated on the third. This is a reviewer-requested corrective analysis, not
a population-generalization claim.

## 2. Candidate-budget reporting

The existing v4 resamples are reused without alteration.

- `kernel_blocked` is the primary equal-budget scheme because it samples
  whole five-dimension classical kernel blocks, structurally mirroring the
  quantum pool's feature-map-by-dimension blocks.
- `uniform` and `kernel_stratified` are mandatory sensitivity schemes.
- The manuscript will distinguish equality in candidate count from equality
  in search-space structure.
- A complete scheme-by-scenario-group table will report all three schemes for
  SVC and GPC. No scheme may be removed because of its result.

## 3. Effective-rank matching sensitivities

The original nearest-neighbour matching is observational and reuses classical
candidates. The revised report must state both facts.

The existing with-replacement pairs will be summarized at the complete set of
calipers `rank_ratio <= {1.10, 1.25, 1.50, 2.00}`, plus the unfiltered set.
For every caliper, report retained counts, retained fractions, rank
discrepancies, and quantum-minus-classical OOD balanced-accuracy differences.

A second reviewer-requested sensitivity will use one-to-one minimum-cost
assignment within each `(run, dimension)` cell. The cost is absolute
log-effective-rank difference. Because the quantum and classical pool sizes
can differ, the smaller family is fully matched and candidates in the larger
family may remain unmatched. Results will be reported both unfiltered and at
the same four calipers. No causal interpretation is permitted.

## 4. Repeated finite-shot perturbation

This is an explicitly post-confirmatory Monte Carlo sensitivity. It estimates
measurement-sampling variability conditional on eight fixed exact Gram
matrices; it is not a hardware simulation and does not support population
inference.

### Fixed subset

The subset contains one `q1000`, master-seed 42, q-split-seed 42,
model-seed 42 run per security scenario-group. Within each run, the fixed
configuration is the exact-statevector quantum SVC candidate selected by P1'
on ID-validation in the versioned `summary_v4.csv`.

| Run | Fixed kernel | Dimension | Exact selected C | `summary_v4.csv` SHA-256 |
|---|---|---:|---:|---|
| `m1_hist_byteent__ms42__q1000_id500_ood500__qs42__s42` | `pauli_xz_r1_full__as2` | 12 | 10 | `4c0b525d29ca3a3d912e5a97176f59d3ee396a34646ff16ea6365943867b0942` |
| `m2_hist_byteent__ms42__q1000_id500_ood500__qs42__s42` | `pauli_xz_r1_full__as2` | 12 | 100 | `275c892415d1cfd92451c89bc4a81ea92c7e02ffba5197c815392519e5a9cc83` |
| `toniot_scanning__m2_centroid__ms42__q1000_id500_ood500__qs42__s42` | `zz_r2_full__as2` | 6 | 10 | `783a0b798819024cdcf53393be74ba8b33c5ce3730bf440bea6dc4c49e4aae17` |
| `toniot_scanning__natural_cur__ms42__q1000_id500_ood500__qs42__s42` | `pauli_xz_r1_full__as2` | 12 | 10 | `89f40af91873929966cfc3002e779385a5fcc4b19ea65e87b513ea61f9e1ef70` |
| `unsw_dos__m2_centroid__ms42__q1000_id500_ood500__qs42__s42` | `zmap_r2__as0.5` | 12 | 100 | `307d051fc5098ee05c978fb955acc761756482054e0c1ca38c8b24c61f1d583d` |
| `unsw_dos__natural_cur__ms42__q1000_id500_ood500__qs42__s42` | `zz_r2_full__as0.5` | 4 | 100 | `c1f869aaaeaeb705969ef857ba2ed7f259b07bd4bc0b4cfabace961f5cec3e60` |
| `unsw_recon__m2_centroid__ms42__q1000_id500_ood500__qs42__s42` | `zz_r2_full__as0.5` | 4 | 100 | `f939251143c98f732f710f7f56db27291e39e7323bde9458a31684c4f568114d` |
| `unsw_recon__natural_cur__ms42__q1000_id500_ood500__qs42__s42` | `zz_r1_full__as0.5` | 6 | 10 | `4f0cc9268a97073784b85389a841d54fba81c9483a29067f86c41c40e17d36a9` |

### Monte Carlo design

- Shot counts: `{128, 512, 2048, 8192}`.
- Measurement replicates: 30 per `(run, shots)` using replicate identifiers
  `0..29`.
- Every block receives a separate NumPy generator whose seed is derived from
  the first eight bytes of SHA-256 over
  `ksf-v6-shots::<run>::<kernel>::<dimension>::<shots>::<replicate>::<block>`.
  Python's process-randomized `hash()` is prohibited.
- Train and OOD-square fidelity blocks are symmetrized with exact unit
  diagonal after binomial sampling. Both the sampled pre-projection matrix and
  its positive-semidefinite projection are retained.
- ID-validation, ID-test, and OOD-test rectangular blocks receive binomial
  sampling and clipping only.
- SVC `C` is reselected from the frozen grid by five-fold training-only CV for
  every sampled training matrix and projection condition. The final SVC fit
  uses the full sampled training block.

### Required outputs

For each run, shots, replicate, and projection condition, report:

- selected `C`;
- ID-validation, ID-test, and OOD-test balanced accuracy;
- signed OOD difference from the exact fixed-configuration endpoint;
- absolute OOD difference;
- training effective-rank ratio relative to exact;
- OOD centered-alignment difference relative to exact;
- minimum eigenvalue, negative-eigenvalue fraction, and Frobenius changes due
  separately to sampling and PSD projection.

Aggregates report medians and central 95% Monte Carlo intervals across the 30
measurement replicates within each fixed Gram matrix. Across the eight fixed
scenario-groups, only descriptive ranges and medians are permitted.

The original one-draw selection-instability result is retained only as
historical exploratory evidence. It must not be described as a Monte Carlo
estimate, and its signed difference must not be called "accuracy lost" or
"regret."

## 5. Statistical language

- Security intervals: "95% conditional cluster-t interval over five q-split
  seed clusters (`n=5`), conditional on the benchmark pools and design."
- External intervals: "95% conditional seed-cluster interval over five
  deterministic subsampling seeds per task (`n=5`)."
- Fifteen pipeline realizations are never called fifteen independent
  observations.
- The frozen external category name `replicated_protocol_sensitivity` remains
  machine-readable. Narrative text uses "prospectively corroborated across
  three fixed external tasks."

## 6. Manuscript and supplementary structure

All descriptions needed to understand or reproduce the study will appear in
the main manuscript Methods, with topical subheadings. The Supplementary
Information will contain extended results, full tables, diagnostics, and
derivations, but no section or note presented as Supplementary Methods.

The revised central contribution is:

> distribution shift + target-label-free selection + train-only
> regularization + structurally matched search budgets + specification curve
> + prospectively frozen external corroboration.

Geometry claims are associational. The manuscript may state that both
families reach similar informative geometries; it may not state that geometry
"rather than quantumness" causally carries information.

## 7. Editorial and reproducibility gates

- Use the standard numbered Nature reference style and remove unresolved
  bibliographic placeholders.
- Cite the published EPJ Quantum Technology version of spectral phase
  encoding.
- Cite both the project concept DOI and the immutable DOI of the exact release
  used for submission.
- Add `testpaths = ["tests"]` to the pytest configuration.
- The cover letter will be concise, state interest in the Collection, and
  identify the earlier public draft artifact and its overlap once its exact
  identifier is confirmed.
- Recompile and visually inspect every page of the main and supplementary
  PDFs after the final substantive revision.

## 8. Release rule

The corrected release will be `v0.6.0`. It may be published only if:

1. all v6 analysis gates pass;
2. the complete test suite passes on Python 3.11 and 3.12 CI;
3. both PDFs compile without unresolved references or placeholders;
4. visual inspection finds no clipping, overlap, broken glyphs, or illegible
   figure labels;
5. the GitHub tag, release assets, and Zenodo version resolve to the same
   commit.

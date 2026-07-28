# v5 external-validation and specification-curve protocol

**Status: FROZEN on 2026-07-29, before loading any TableShift task or
computing any external quantum/classical result.**

Prepared against repository commit
`d86fdfbaf41e5562f6178ec34b1f415198c53c0f`. The commit that first adds this
document is the public freeze record. The v4 analysis and its conclusions
remain immutable; all work described here is an independent extension under
`results/v5/`.

## 1. Questions

This extension asks two questions.

1. Across every already-computed, logically valid analysis specification, how
   do target-label access, regularization, baseline strength, and candidate
   budget change the reported quantum-minus-classical OOD effect?
2. Does the protocol sensitivity observed on the three security datasets
   replicate on predefined real-world distribution shifts outside
   cybersecurity?

The external outcome of interest is not whether the quantum or classical
family wins. The primary outcome is whether the apparent family-level effect
changes when a same-test OOD oracle with weak references is replaced by
train-only regularization, target-label-free selection, and equal candidate
budgets.

## 2. Source benchmark and fixed tasks

The source is TableShift at commit
`fca9429814703a07e3902d005d46563a207b7f0a`, using its predefined domain
splits. The three fixed tasks are:

| Identifier | Domain | Published split variable |
|---|---|---|
| `college_scorecard` | education | institution type |
| `diabetes_readmission` | health | admission source |
| `acsincome` | socioeconomic policy | geographic region |

They were chosen before external execution because they are public,
binary-classification tasks from different source domains with different
real-world split variables. Published TableShift baseline results were known
when the tasks were selected; no quantum-kernel result on these tasks was
available or inspected.

If and only if a fixed task cannot be acquired, parsed, or represented with at
least twelve non-constant post-encoding features, substitutions are attempted
in this order:

1. `brfss_diabetes`;
2. `acsfoodstamps`;
3. `nhanes_lead`.

A substitution and its technical reason must be recorded before any model is
run on the replacement. Performance is never a valid substitution reason.

The TableShift `ood_validation` split is ignored. No feature or label from it
may enter preprocessing, tuning, selection, analysis, or reporting.

## 3. Frozen sampling design

The predefined TableShift split roles are preserved:

- `train`: fit preprocessing, kernels, classifiers, and internal SVC
  regularization;
- `validation`: select the deployed kernel configuration;
- `id_test`: report in-distribution generalization only;
- `ood_test`: final target evaluation and the explicitly non-deployable oracle
  diagnostic.

Two fixed sample-size strata are used:

| Stratum | train | validation | ID test | OOD test |
|---|---:|---:|---:|---:|
| `q1000` | 1000 | 250 | 250 | 500 |
| `q2000` | 2000 | 500 | 500 | 1000 |

Subsampling seeds are `42, 123, 999, 7, 2024`. Within each source split, rows
are ordered by SHA-256 of

`"ksf-v5::<task>::<split>::<seed>::<source-row-position>"`

and the first required rows are taken. Sampling is label blind and the
`q1000` sample is nested in `q2000` for a given task, split, and seed. After
sampling, an automated gate verifies that both labels occur in every split. A
failure triggers the technical substitution rule above; it does not trigger a
new seed.

The five subsampling seeds are the conditional resampling clusters. Sample
size is a fixed, nested design factor and is never treated as an independent
replicate.

## 4. Frozen representation

All transformations are fitted on the sampled training split only.

1. Columns that are constant on training are removed.
2. Numeric columns receive median imputation.
3. Boolean, categorical, and string columns receive most-frequent imputation
   and one-hot encoding with unknown categories ignored.
4. The combined representation is MaxAbs scaled.
5. Truncated SVD maps it to `d in {4, 6, 8, 10, 12}` with random state
   `20260729`.
6. Each SVD representation is standardized and mapped to `[0, pi]` using
   training-fitted transforms.

No target label is used by preprocessing or representation learning.

## 5. Frozen kernel and classifier pools

The pools are identical to v4.

### Quantum family

Four fidelity feature maps are crossed with the five dimensions and angle
scales `{0.5, 1.0, 2.0}`, giving 60 candidate geometries:

- `zz_r1_full`;
- `zz_r2_full`;
- `pauli_xz_r1_full`;
- `zmap_r2`.

Kernels are exact statevector fidelities. This extension does not introduce a
hardware claim.

### Classical family

The 23 v4 classical kernel shapes/scales are crossed with the five dimensions,
giving 115 candidate geometries. They include linear, RBF, polynomial,
Laplacian, and Matern kernels plus the frozen length-scale variants.

### Classifiers

- SVC is primary. For every candidate separately, `C` is selected from
  `{0.01, 0.1, 1, 10, 100}` by mean balanced accuracy in five stratified
  training-only folds. Fold seed: `20260729`; ties choose the smaller `C`.
- The Laplace Gaussian-process classifier with the frozen v4 hyperparameters
  is secondary.

The decision threshold is the classifier default. No threshold is tuned on
validation or OOD data.

## 6. External estimands

Balanced accuracy is primary. All deltas are quantum minus classical.

### Primary controlled estimand

For each task, size, seed, and classifier:

1. select the best quantum candidate using `validation`;
2. repeatedly sample an equal-size, whole-kernel-block subset of the classical
   pool;
3. select the best classical candidate using `validation` within each
   subsample;
4. evaluate both selected candidates once on `ood_test`.

The equal-budget procedure uses 5000 kernel-blocked resamples and seed
`20260729`. The run-level classical endpoint is the expected selected OOD
accuracy over those resamples, exactly as in v4.

### Oracle diagnostic

The same equal-budget procedure is repeated with selection and evaluation on
`ood_test`. It is labelled same-test oracle everywhere and is never presented
as deployable evidence.

### Protocol-contraction estimand

For each unit,

`contraction = delta_oracle_weak_fixed - delta_honest_equal_budget_tuned`.

The weak-fixed endpoint uses the original linear/RBF classical reference,
fixed `C=1`, and same-test OOD selection. This endpoint is diagnostic. The
controlled endpoint is the primary scientific result.

### Aggregation and uncertainty

- average seeds within each task and size;
- average the two size strata within task;
- give each task equal weight in the external summary;
- report per-task effects and task-equal means;
- estimate conditional uncertainty by resampling the five seed clusters
  within task with 9999 draws, seed `20260729`;
- report leave-one-task-out ranges;
- report no population p-value.

## 7. Frozen interpretation

The external extension is classified after all outputs pass audit.

- **Replicated protocol sensitivity:** the task-equal SVC contraction is
  positive, the task-equal controlled SVC delta is non-positive, and no two
  tasks show a positive controlled effect whose conditional interval excludes
  zero in both size strata.
- **Regime-specific boundary:** at least one task shows a positive controlled
  effect whose conditional interval excludes zero in both sizes, while the
  task-equal result does not satisfy the failure condition below.
- **Failure to generalize:** the task-equal controlled SVC delta is positive
  and at least two tasks have positive controlled effects whose intervals
  exclude zero in both sizes.
- **Mixed/inconclusive:** every other outcome.

The manuscript must report the observed category. No task, size, seed,
classifier, kernel, or metric may be removed because its result is
inconvenient.

## 8. Frozen specification curve

The specification curve is descriptive and contains every endpoint below for
which the frozen artifacts provide complete paired data.

| ID | Generation | Selection | SVC regularization | Classical reference | Budget |
|---|---|---|---|---|---|
| S1 | legacy | same-test OOD oracle | fixed `C=1` | linear/RBF | native |
| S2 | legacy | ID-selected | fixed `C=1` | linear/RBF | native |
| S3 | legacy | same-test OOD oracle | fixed `C=1` | extended | native |
| S4 | legacy | ID-selected | fixed `C=1` | extended | native |
| S5 | v4 | same-test OOD oracle | train-CV | linear/RBF | native |
| S6 | v4 | ID-validation | train-CV | linear/RBF | native |
| S7 | v4 | same-test OOD oracle | train-CV | extended | native |
| S8 | v4 | ID-validation | train-CV | extended | native |
| S9 | v4 | same-test OOD oracle | train-CV | extended | equal |
| S10 | v4 | ID-validation | train-CV | extended | equal |

SVC and GPC are displayed separately; the regularization column is not
applicable to GPC. Specifications are ordered as above, never by effect size.
The headline is the dataset-equal mean, with every scenario-group shown.
There is no specification-level hypothesis test and no cherry-picking of a
preferred path.

The curve may use only versioned legacy summaries and v4 `summary_v4.csv`
files. It may not recompute or alter the v4 experiment grid.

## 9. Leakage and provenance gates

Before external evaluation:

1. record the TableShift source commit and source-file hashes;
2. record deterministic sampled row positions and class-presence gates;
3. verify split disjointness;
4. verify every transform was fitted on training only;
5. verify selection code cannot access `ood_test` in controlled mode;
6. write all outputs below `results/v5/`, never into v4 directories;
7. make the pipeline resume-safe and deterministic.

Raw TableShift data are not redistributed. The release contains acquisition
instructions, source identifiers, row-position hashes, code, and aggregated
outputs.

## 10. Reporting constraints

Prohibited regardless of outcome:

- presenting the oracle as a valid deployment protocol;
- describing the three external tasks as a random population sample;
- claiming hardware validation or quantum speedup;
- hiding a conflicting external result;
- folding the external tasks into the frozen v4 inference as though they had
  been planned together;
- changing this document after the first external result exists, except for a
  clearly dated erratum that cannot depend on performance.


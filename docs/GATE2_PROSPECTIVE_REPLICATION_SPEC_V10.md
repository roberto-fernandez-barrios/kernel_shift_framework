# v1.0 prospective Gate-2 replication protocol

**Status: FROZEN on 2026-08-01 before acquiring any of the three tasks,
computing a model prediction, or opening a target label.**

Protocol root: `ksf-v10-gate2-prospective-20260801`.

This document may not be edited after its SHA-256 freeze manifest is written.
Any correction must be recorded in a separately dated erratum without
deleting this file. All outcomes, including acquisition or replication
failure, must be retained.

## 1. Question and estimand

The experiment prospectively tests whether the v0.9 sharp target-domain
certificate and prediction-aware target audit transfer to tasks whose QML
predictions and target labels have not been inspected.

For a fixed quantum classifier `q`, a prespecified classical family
`C_B={c_1,...,c_M}`, and a target batch of `n=500`, the accuracy advantage is

```text
Delta_B(y) = Acc(q;y) - max_j Acc(c_j;y).
```

At audited label subset `L`, the exact upper endpoint is

```text
U_B(L) = min_j [S_j(L) + R_j(L)] / n.
```

The primary materiality threshold is `tau=0.010`. Thresholds `0.005` and
`0.020` are prespecified secondary sensitivities.

## 2. Frozen tasks

The source is the pinned TableShift checkout at commit
`fca9429814703a07e3902d005d46563a207b7f0a`, preserving its published domain
splits. The tasks are exactly the three public fallbacks nominated in the
pre-outcome v5 protocol but never acquired or executed in this project:

| Task | Domain shift | Role in old protocol |
|---|---|---|
| `brfss_diabetes` | race | fallback 1 |
| `acsfoodstamps` | geographic Census division | fallback 2 |
| `nhanes_lead` | poverty | fallback 3 |

Before this freeze, no directory named for any task existed under the project
TableShift exports, audit outputs, external results, or v9 prediction results.
Published TableShift benchmark results may be known; no quantum-kernel result,
prediction matrix, target prevalence, or target label for these project
samples has been inspected.

All three tasks are attempted. A task may be declared technically unavailable
only for download, parser, binary-label, split-cardinality, class-presence, or
minimum-feature failure. It is never replaced because of a model outcome. No
new task may be added after freeze.

## 3. Frozen samples and split roles

Only `q1000` is used:

| Split | Rows | Allowed use |
|---|---:|---|
| `train` | 1000 | preprocessing, fitting, SVC train-CV |
| `validation` | 250 | fixed P1-prime quantum selection |
| `id_test` | 250 | integrity/context only |
| `ood_test` | 500 | target predictions, then sealed label audit |

The five deterministic sampling seeds are `42, 123, 999, 7, 2024`. Rows are
ordered label-blind by the existing v5 SHA-256 sampler. `ood_validation` is
never loaded. The 15 task-seed units are fixed design cases; seeds describe
deterministic subsampling sensitivity, not a population sample.

## 4. Frozen representation, models, and references

The v5 train-only representation is reused verbatim: semantic imputation and
one-hot encoding, MaxAbs scaling, TruncatedSVD dimensions `{4,6,8,10,12}`,
standardization, and a train-fitted map to `[0,pi]`. Every transform is fitted
on `train` only.

The quantum pool is the same 60 exact-statevector fidelity kernels: four maps,
five dimensions, and angle scales `{0.5,1,2}`. The P1-prime winner is selected
only by validation balanced accuracy, with lexicographic tie breaking.

The classical witness pool is all 115 prespecified v4 candidates. The
`customary_30` linear/RBF tier is secondary. SVC uses candidate-specific
five-fold train-CV over `C={0.01,0.1,1,10,100}`; GPC uses the frozen Laplace
implementation. Decision thresholds are defaults. The target labels never
select a candidate, hyperparameter, threshold, or family.

## 5. Physical prediction-label separation

The prospective pipeline has two fail-closed stages.

1. `lock-predictions` reads target covariates, writes the target labels to a
   separate sealed archive, computes every fixed prediction, and writes a
   manifest containing task, seed, model, selected quantum configuration,
   classical candidate identities, row-order digest, and prediction hashes.
   It must not compute or print any target performance metric, disagreement
   result, class prevalence, or acquisition curve.
2. `audit-labels` may run only after every technically available unit has a
   complete prediction-lock manifest and the aggregate lock manifest has been
   hashed. It verifies all hashes before opening the sealed labels once.

Partial availability is recorded before model execution. No result from a
completed task may influence handling of another task.

## 6. Frozen acquisition policies

For both classical tiers and all three thresholds, report:

1. the zero-label sharp region;
2. fixed `adaptive_bottleneck_cover`, using the v0.9 SHA-256 tie rule;
3. `random_active_disagreement`, over 200 frozen SHA-256 orders;
4. `nonadaptive_initial_coverage`, over 200 frozen SHA-256 orders;
5. the exact undeployable retrospective label oracle;
6. `hash_all` only as a weak diagnostic.

The two strong comparator roots append task, seed, model, tier, policy, and
draw to the protocol root. No order or threshold may be changed after labels
are opened. Balanced-accuracy MILP results at realized prevalence are a
retrospective metric sensitivity, not the primary label-free claim.

## 7. Frozen summaries and outcome categories

The primary empirical units are the six fixed task-classifier cells, each
summarized by the median label count over its five seeds against `full_115` at
`tau=0.010`. Also report all 30 seed-level cells without exclusions.

The prospective outcome is classified exactly once:

- **Strong prospective transfer:** all six task-classifier medians are at most
  50 labels, the overall seed-level median is at most 25, and no seed-level
  cell needs more than 100 labels under the fixed adaptive rule.
- **Partial prospective transfer:** at least five of six task-classifier
  medians are at most 50, the overall median is at most 50, and no
  task-classifier median exceeds 100.
- **Failure to transfer:** fewer than five task-classifier medians are at most
  50, or any task-classifier median exceeds 100.
- **Technically incomplete:** fewer than two tasks pass acquisition and
  integrity gates. With exactly two available tasks, apply the same rules to
  four task-classifier cells and require all four for partial transfer; the
  label remains technically limited.

The cut at 50 labels represents 10% of the fixed target batch. The old
security-task range of 0--33 is not reused as a success threshold.

No superiority claim for adaptive coverage is planned. Its paired differences
from random-active and initial-coverage medians are reported descriptively. A
strong transfer finding concerns the certificate and actionable-disagreement
principle, not domination by an acquisition heuristic.

## 8. Reporting constraints

Regardless of outcome, the manuscript must state that the task identifiers
were inherited from an earlier fallback list, while the Gate-2 hypotheses,
policies, and outcome categories were frozen prospectively for this use.

Prohibited claims include population generalization from three tasks,
classical simulability, complexity-theoretic separation, acquisition-policy
superiority without the prespecified controls, hardware validation, and
balanced-accuracy certification without a prespecified prevalence constraint.

All code, prediction locks, sealed-label hashes, audit curves, failed units,
and final manifests are included in the release snapshot.

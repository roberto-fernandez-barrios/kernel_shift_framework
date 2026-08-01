# v0.9 target-domain partial-identification specification

**Status: FROZEN on 2026-07-31 before generating any v0.9 prediction or
outcome artifact.**

**Post-freeze execution-integrity amendment (2026-07-31, before any v0.9
certificate existed):** the first reconstruction attempt stopped during the
classical export because individual SVC boundary predictions changed with the
active BLAS thread count. Diagnostic reruns found that two threads reproduced
both encountered frozen v4 metrics exactly, whereas one or four/eight threads
changed one of 500 predictions in one of the two configurations. No
prediction artifact or partial-identification outcome was produced by the
failed attempts. All v0.9 reconstruction is therefore executed with a fixed
threadpool limit of two and records this constraint. Exact metric
agreement remains mandatory; no numerical tolerance was relaxed.

This analysis is exploratory relative to the v0.4 confirmatory endpoint and
the prospectively frozen v0.5 external validation.  It must be labelled
`post-hoc exploratory` wherever it is reported.  The purpose of the pilot is
to decide, using rules fixed below, whether target-domain partial
identification is informative enough to justify rebuilding the manuscript
around it.  No result may be removed, redefined, or assigned a new threshold
after it is observed.

## 1. Scientific question

For a deployed quantum classifier `Q`, a finite target batch `U`, and a
prespecified finite classical reference family `C_B`, define

```text
d_j(U) = mean_x 1[Q(x) != C_j(x)]
Delta_B(y) = Accuracy(Q; y) - max_j Accuracy(C_j; y).
```

The hard-prediction sharp envelope is

```text
inf_y Delta_B(y) = -max_j d_j(U)
sup_y Delta_B(y) =  min_j d_j(U).
```

The upper endpoint is the primary estimand.  It is the largest accuracy
advantage compatible with the observed target inputs and joint prediction
matrix when target labels are otherwise unrestricted.  The lower endpoint is
reported to complete the mathematical envelope, but is not an indispensability
criterion and is not used to strengthen the headline.

Claims are conditional on the exact target batch, trained models, finite
reference family, and resource/search definition.  They are not claims of
global classical simulability, computational dequantization, equivalence, or
population performance.

## 2. Frozen pilot units

The pilot reuses the eight q1000, master-seed-42, q-split-42, model-seed-42
security runs fixed before the v0.6 finite-shot analysis in
`scripts/experiments/run_shots_mc_v6.py::FIXED_RUNS`.

- target role: the existing 500-row `ood_test` batch;
- quantum model: the P1'-selected candidate chosen on `id_val`;
- primary learner: SVC with its v4 train-CV-selected `C`;
- secondary learner: the frozen Laplace GPC;
- primary classical family: all 115 v4 extended-classical candidates for the
  same learner, each trained only on the original training split;
- customary reference: linear plus the five RBF blocks (30 candidates);
- no model is fitted, regularized, thresholded, or selected using OOD labels.

The prediction exporter may read OOD labels only in a separate integrity step
that verifies exact agreement with the frozen aggregate metrics.  The
partial-identification analysis consumes an artifact that contains no target
labels.  Evaluation labels, when unlocked after the certificate is produced,
are used only for descriptive sharpness/slack checks.

## 3. Primary and secondary estimands

### 3.1 Primary

For SVC and the full 115-candidate classical family:

```text
U_accuracy = min_j d_j
L_accuracy = -max_j d_j
witness = lexicographically first argmin_j d_j.
```

Report all eight fixed cases individually.  Source-dataset-equal summaries
are descriptive only and cannot hide case heterogeneity.

Material thresholds are frozen at

```text
tau in {0.005, 0.010, 0.020} accuracy.
```

For a target size of 500, these correspond to at most 2, 5, and 10
disagreements after applying the exact empirical threshold rather than a
rounded display value.

### 3.2 Secondary

1. SVC customary 30-candidate reference.
2. GPC hard predictions for the customary and full reference families.
3. Exact balanced-accuracy envelope conditional on a fixed target class count:
   a MILP for binary finite-batch labels and the corresponding LP relaxation.
   Because the security batches were constructed with label information, this
   is called a **design-conditional prevalence analysis**, not a generally
   label-free deployment guarantee.
4. The realized accuracy/BAcc family advantage after labels are unlocked,
   solely to verify containment and measure bound slack.
5. Classical-to-classical and quantum-to-quantum disagreement controls.

## 4. Search/reference frontier

Candidate count is called a **classical search/reference budget**, not a
computational resource budget.  Computational-resource language is allowed
only if time, memory, kernel evaluations, and inference cost are explicitly
measured.

The frozen reference tiers are:

- `customary_30`: linear and all RBF blocks across five dimensions;
- `full_115`: every extended-classical v4 candidate.

Equal-count 30/60 sensitivities use whole five-dimension kernel blocks and
stable SHA-256-derived permutations rooted at `ksf-v9-frontier-20260731`.
They report the distribution of the sharp upper endpoint over 5,000 nested
block orderings.  They are budget-sensitivity distributions, not confidence
intervals.  A single arbitrary 60-candidate subset is not a primary result.

For each material threshold `tau`, report the smallest evaluated budget whose
upper endpoint is at most `tau`, or `not_reached`.  When the answer depends on
the block ordering, report its probability over the frozen orderings.

## 5. Balanced-accuracy identification

For each unique joint hard-prediction signature `z`, let `n_z` be its batch
count and `k_z` the unknown number of positives.  With fixed total positives
`n_pos`:

```text
0 <= k_z <= n_z,  k_z integer,
sum_z k_z = n_pos.
```

The upper endpoint maximizes `t` subject to

```text
t <= BAcc(Q; k) - BAcc(C_j; k)  for every C_j.
```

The lower endpoint is the minimum, over classical candidates, of its pairwise
minimum BAcc difference.  `scipy.optimize.milp` is the frozen exact solver;
the LP relaxation is reported as a distributional sensitivity.  Solver
status, objective tolerance, number of signatures, and the fixed prevalence
must be stored.  An unknown natural target prevalence is handled by a
prespecified prevalence curve or interval, never by silently using unlocked
target labels.

## 6. Population statement

For models fixed before an i.i.d. target sample of size `m`, the finite-family
uniform correction

```text
sqrt(log(2 M / delta) / (2 m))
```

may be reported as a theoretical corollary.  It is not a manuscript-facing
empirical guarantee for the constructed security batches.  At `M=115` and
`m=500` the 95% correction is approximately 0.092 and is not materially
informative for the frozen thresholds.

## 7. Structural and finite-shot extensions

These stages are run only after the primary pilot artifact has passed all
integrity gates, but their hypotheses are frozen now.

1. **Train-only insufficiency construction:** provide normalized joint PSD
   Gram matrices with identical `K_TT` and different `K_UT`, and report the
   induced prediction-operator difference.  This is an explanatory
   proposition, not a claim that normalized fidelity differences are
   unbounded.
2. **Finite-shot extension:** for the existing 128, 512, 2,048, and 8,192-shot
   replicates and all three PSD conditions, archive hard predictions and
   compute both
   `d(Q_shot, Q_exact)` and `min_C d(Q_shot, C)`.
3. A rise in classical disagreement is called noise-induced non-emulability
   only when accompanied by a reported exact-to-shot self-disagreement.  It
   is never interpreted as evidence of advantage without performance gain.

## 8. Go/no-go rule for manuscript reconstruction

Rebuild the manuscript around partial identification only if at least one of
the following is observed without concealing contrary fixed cases:

1. the full-family SVC upper endpoint is at most 0.010 in a stable majority of
   the eight fixed cases and remains informative for entangling-ZZ winners;
2. the search-budget frontier reaches a frozen material threshold early and
   stably across sources;
3. the BAcc MILP produces materially narrow design-conditional bounds beyond
   the accuracy result;
4. finite-shot sampling increases classical non-emulability without a
   commensurate performance improvement, after accounting for self-disagreement;
5. the target-aware certificate adds information not contained in effective
   rank, KTA, or the available geometric diagnostics.

If none holds, v0.9 remains an archived exploratory analysis and the current
v0.8 manuscript is not weakened by adding an uninformative certificate.

Any transfer/generalization claim requires a subsequently frozen evaluation
on tasks not inspected when this specification was written.  Existing
TableShift outcomes are retrospective for v0.9 even though their original
v0.5 protocol was prospective.

## 9. Post-unlock exploratory extension: partial target labels

This section was appended after the zero-label artifacts and their evaluation
labels had been unlocked.  It is therefore explicitly **exploratory** for the
eight q1000 cases and cannot be represented as part of the original frozen
pilot.  The mathematical result is exact; empirical label-efficiency claims
require later confirmation on tasks whose labels have not been inspected.

For an audited subset `L` of target labels, define for each classical witness
`j` the observed signed accuracy-difference count

```text
S_j(L) = sum_{i in L} [1(q_i = y_i) - 1(c_ij = y_i)]
```

and let `R_j(L)` be the number of still-unlabelled points on which `q` and
`c_j` disagree.  The sharp endpoints over all completions of the unobserved
labels are

```text
lower(L) = min_j [S_j(L) - R_j(L)] / n,
upper(L) = min_j [S_j(L) + R_j(L)] / n.
```

Equivalently in the frozen binary setting, let `D_j` be the total
quantum--witness disagreement count, `a_j(L)` the audited disagreements whose
label favors `q`, and `b_j(L)` those whose label favors witness `j`.  Then

```text
lower(L) = min_j [-D_j + 2 a_j(L)] / n,
upper(L) = min_j [ D_j - 2 b_j(L)] / n.
```

Thus a counterexample favoring a disagreeing classical witness reduces that
witness's upper endpoint by exactly `2/n`; a queried label favoring `q` cannot
increase it.  This identity is reported alongside raw query counts so label
efficiency is not confused with statistical confidence.

At `|L|=0` these reduce to the frozen zero-label endpoints; at `|L|=n` both
equal the realized advantage against the best classical model.  Labels on
points where every model agrees with `q` cannot change either endpoint.

The exploratory evidence frontier evaluates `customary_30` and `full_115`
under the same material thresholds `tau in {0.005, 0.010, 0.020}`.  It reports
the first queried-label count at which the sharp upper endpoint is at most
`tau`.  Every witness-specific upper endpoint, and therefore their minimum,
is non-increasing as labels are revealed, so a crossed threshold remains a
valid certificate even when the constraining witness changes.

The original v0.9 analysis fixed two label-acquisition policies before
computing these curves:

1. `hash_all`: a prediction-independent SHA-256 ordering rooted at
   `ksf-v9-label-audit-20260731`, reported over 200 stable hash draws;
2. `adaptive_bottleneck_cover`: after each revealed label, identify the
   smallest witness-specific upper endpoint that can still be reduced, then
   query an unlabelled point that disagrees with the largest number of tied
   witnesses at that endpoint, with SHA-256 tie breaking.

The adaptive policy may use labels already audited but never unaudited labels.
Candidate count remains a search/reference budget; queried target labels are
a separate supervision budget.

### 9.1 v0.9.1 comparator amendment

This amendment was declared on 2026-08-01 after the original v0.9 curves were
known, but before the comparator outcomes below were computed.  It does not
make the acquisition study prospective.  Its purpose is to distinguish gains
from avoiding irrelevant points from gains due to bottleneck coverage.

Three additional comparators are frozen:

1. `random_active_disagreement`: at each step, sample by a stable SHA-256
   order from all unaudited points that disagree with at least one currently
   reducible bottleneck witness.  This policy, like the proposed adaptive
   rule, never spends a query on a point unable to change a current actionable
   bottleneck, but it does not optimize witness coverage.
2. `nonadaptive_initial_coverage`: rank points once by the number of
   zero-label bottleneck witnesses with which they disagree, using stable
   SHA-256 tie breaking.  It uses the prediction matrix but no revealed-label
   feedback.
3. `retrospective_label_oracle`: an undeployable exact ex-post lower bound on
   the number of queries needed by any subset.  For witness `j`, threshold
   `tau` requires

   ```text
   r_j(tau) = max(0, ceil((D_j - n tau) / 2))
   ```

   audited disagreements whose realized label favors that witness.  The exact
   oracle minimum is the smallest feasible `r_j(tau)` across witnesses; it is
   `not_reached` if the realized full-label advantage exceeds `tau`.  This is
   exact because crossing the family upper endpoint requires any one
   witness-specific endpoint, not every witness, to cross.

The two tie-broken comparator distributions use 200 frozen SHA-256 orders.
The primary v0.9.1 comparison is the witness-directed rule against both strong
comparators and the oracle lower bound.  `hash_all` remains only a weak
prediction-independent reference and must not carry the efficiency claim.

# v0.8 reviewer-revision specification

**Status: FROZEN before execution of any v0.8 outcome analysis.**

This document was committed on 2026-07-29 before generating any of the
`summary_v8_*` files or any result under `results/v8/`. The reviewer had
already reported that two feature maps were separable and disclosed aggregate
winner counts. Those facts are therefore not treated as prospectively tested
outcomes. All new performance contrasts below are labelled reviewer-motivated
sensitivities rather than preregistered confirmatory endpoints.

The v0.4 primary endpoint remains unchanged. v0.8 corrects the circuit
description, tests whether the primary conclusion depends on including
separable maps, decomposes four evaluation axes within one experimental
generation, quantifies logical and sampling resources, and tests a frozen
port/protocol-field ablation.

## 1. Circuit-family audit

The four Qiskit 2.3.1 feature maps are classified from their actual Pauli
strings and decomposed circuits:

- `zz_r1_full`: `ZZFeatureMap`, one repetition, full connectivity;
- `zz_r2_full`: `ZZFeatureMap`, two repetitions, full connectivity;
- `pauli_xz_r1_full`: `PauliFeatureMap`, one repetition,
  `paulis=["X", "Z"]`;
- `zmap_r2`: `ZFeatureMap`, two repetitions.

The first two form the **entangling ZZ stratum**. The last two form the
**separable product-map stratum**. For the latter,
`U(x) = tensor_j U_j(x_j)`, so their fidelity kernels factorize as
`|<phi(x)|phi(x')>|^2 = product_j |<phi_j(x_j)|phi_j(x'_j)>|^2`.
This factorization, rather than the generic fact that weak baselines can be
optimistic, is the quantum-specific interpretive focus of the revision.
The string `full` in the stored identifier
for `pauli_xz_r1_full` records the constructor argument but must not be
described as physical entanglement: single-qubit Pauli strings do not invoke
an entangler map.

For dimensions 4, 6, 8, 10, and 12, an automated table will report feature-map
depth, one- and two-qubit operation counts, and the corresponding
compute--uncompute fidelity-circuit template after decomposition to the
logical basis `rz`, `sx`, `x`, `cx`. Counts are compiler- and basis-conditional,
with Qiskit version, optimization level, and absence of device routing stated
in the table.

## 2. Entangling-versus-separable sensitivity

This diagnostic reuses the complete v4 run-level summaries; it does not
recompute kernels.

- Classifier: SVC only (the declared primary learner).
- Tuning: per-configuration train-only CV from v4.
- Selection: P1' on `id_val`.
- Evaluation: OOD balanced accuracy.
- Quantum strata:
  - all four maps: 12 map/angle blocks x 5 dimensions = 60 candidates;
  - entangling ZZ only: 6 blocks x 5 dimensions = 30 candidates;
  - separable product maps only: 6 blocks x 5 dimensions = 30 candidates.
- Comparator: extended classical family.
- Equal-count comparator:
  - 60-candidate quantum stratum versus 12 of 23 complete classical blocks;
  - 30-candidate quantum strata versus 6 of 23 complete classical blocks.
- Sampling: 5,000 `kernel_blocked` draws per group and stratum, without
  replacement, using independent SHA-256-derived NumPy seeds rooted at
  `20260729`.

Results are reported for each of the eight fixed scenario-groups and as a
three-source-dataset-equal descriptive mean. The analysis asks whether the
v0.4 direction is an artefact of combining product and entangling circuits; it
does not establish a computational quantum advantage or disadvantage.

The already-disclosed winner-composition counts are reported descriptively for
P1' and the same-test oracle, by classifier. They are not assigned uncertainty
or treated as a new test.

## 3. Within-generation 2 x 2 x 2 x 2 factorial

The earlier S1--S10 display crosses legacy and v4 generations and therefore
cannot isolate regularization. It will be retained only as a historical
cross-generation sensitivity map. The new factorial uses only the v4 data
generation and the complete `q1000` security stratum.

### 3.1 Units

- all 360 `q1000` run directories;
- 24 settings spanning eight scenario-groups;
- five q-split seeds and three model seeds inside each setting;
- SVC only;
- the full 60-candidate quantum family.

### 3.2 Axes

1. selection: same-test OOD oracle versus disjoint `id_val` P1';
2. regularization: fixed `C=1` versus train-CV-selected
   `C in {0.01, 0.1, 1, 10, 100}`;
3. classical reference: customary linear/RBF (30 candidates) versus extended
   family (115 candidates);
4. candidate budget: native full pools versus equal-count block sampling.

For the customary reference, equal count is 30: all six classical blocks are
used and six of the twelve quantum map/angle blocks are sampled. For the
extended reference, equal count is 60: all twelve quantum blocks are used and
twelve of the twenty-three classical blocks are sampled. The larger pool is
sampled 5,000 times without replacement; selection is repeated inside every
draw. The same frozen draws are used across selection and regularization cells
to make axis contrasts paired.

The output comprises all 16 endpoints, group-level effects, source-dataset
means, and the three-source-dataset-equal descriptive mean. Axis contrasts and
interactions are descriptive paired differences among these endpoints. They
show how the estimate changes under this finite design; causal language such
as "the contribution of an axis" remains prohibited.

### 3.3 New computation

The only new model outcomes needed for the factorial are fixed-`C=1` SVC
predictions on the v4 `id_val`, `id_test`, and `ood_test` roles. They will be
generated for all 83 kernel blocks per dimension in the q1000 stratum and
stored as `summary_v8_fixedc.csv`. Preprocessing, rows, embedding, feature maps,
kernel definitions, and tie-breaking remain identical to v4.

## 4. Port/protocol-field ablation

This reviewer-motivated sensitivity uses all 270 q1000 network-flow runs and
does not include EMBER. Features are removed **before** fitting the shared
train-only embedding.

Frozen exclusion rule:

- UNSW-NB15: remove
  `ct_src_dport_ltm`, `ct_dst_sport_ltm`, and `is_sm_ips_ports`.
  Direct categorical `proto`, `service`, and `state` fields were already absent
  from this export.
- ToN-IoT: remove `src_port`, `dst_port`, every feature beginning `proto_`,
  the feature `service`, and every feature beginning `service_`.

This is a port/protocol/service-field ablation, not a claim that all possible
shortcuts have been removed.

For every remaining v4 candidate, the runner records both fixed-`C=1` and
train-CV SVC outcomes on `id_val`, `id_test`, and `ood_test` in
`summary_v8_shortcut.csv`. The manuscript-facing comparison uses train-CV,
P1', the full 60-candidate quantum family, and 5,000 block-matched
60-candidate classical draws. Original and ablated effects use paired frozen
draws and are shown for the six fixed network scenario-groups. No claim that
identical input features guarantee identical family effects is permitted.

## 5. Circuit and shot resources

The finite-shot analysis is explicitly an estimator sensitivity, not a
hardware experiment. For a q1000 run and one kernel configuration, one
measurement replicate samples:

- the upper off-diagonal training triangle;
- ID-validation/test-to-training rectangular entries (stored jointly in the
  original 500-row ID block);
- OOD-to-training rectangular entries;
- the upper off-diagonal OOD square used for geometry.

Diagonals are fixed and require no shot estimate. The exact number of distinct
fidelity estimates and `shots x estimates` totals will be generated
algebraically for 128, 512, 2,048, and 8,192 shots. Tables distinguish:

- logical circuit templates;
- circuit invocations per kernel configuration and measurement replicate;
- shots per replicate;
- projected totals for the ten-case, 30-replicate sensitivity;
- costs not modelled: routing, calibration, mitigation, queueing, and device
  noise.

## 6. Statistical language

- Every interval over the eight security groups is a **pointwise 95% conditional
  cluster-t interval**, `n=5` q-split clusters. It is not simultaneous.
- The five cluster means underlying each manuscript-facing security interval
  will be published.
- No count of groups will be classified by whether a pointwise interval
  includes or excludes zero.
- Interval inclusion of zero is not equivalence. Phrases such as "tied",
  "no advantage", "no robust advantage remains", and
  "classical-favoured or tied" are prohibited. The default wording is
  "no consistent evidence of a quantum advantage was observed in these fixed
  cases."
- No multiplicity adjustment was prespecified. Because many pointwise
  intervals and diagnostics are displayed, they are interpreted
  descriptively and not as a familywise error-controlled decision procedure.
- BCa results with five clusters are retained only as a fragile descriptive
  sensitivity and are not used to strengthen conclusions.

## 7. Geometry and reproducibility corrections

The Huang geometric-difference claim is removed from the current empirical
argument because it was not computed across the complete v4 estimand. The
legacy EMBER-only implementation remains archived and is labelled historical.
Current geometry claims are restricted to effective rank, KTA, and explicitly
post-hoc associations. OOD KTA is not interpreted as a label-independent
mechanism because it uses OOD labels.

`environment.lock.yml` is a Windows-specific archival snapshot, not an exact
cross-platform environment. The repository is described as a reproducible
framework with reduced end-to-end validation, not an exact capture of the
original full runtime. All gates are called automated author-provided audits;
no independent audit is claimed.

## 8. Shift taxonomy and reporting

Results and Methods must distinguish:

1. label-conditional constructed tail stress tests (EMBER m1/m2 and network
   m2-centroid);
2. held-out attack-campaign plus, for UNSW-NB15, capture-partition shift;
3. externally defined TableShift domain shifts.

The first category is not presented as a naturally occurring deployment
process. The common-feature kernel swap controls inputs but does not prove that
shortcut-prone variables affect all kernel families equally.

## 9. Decision rules

The entangling-only and shortcut sensitivities may strengthen a conclusion only
if their sign and scale are reported for every fixed group. Heterogeneity may
not be hidden by the dataset-equal mean.

The within-generation factorial replaces individual-axis attribution from the
cross-generation S1--S10 display. Regardless of its outcome, acceptable
language is "associated with" or "changes under"; "determines", "causes",
"locates the reversal", and "complete specification curve" are prohibited.

No v0.8 result may be called independent replication, hardware validation,
population inference, equivalence, or evidence of computational speedup.

A new target-label-free geometry selector with coefficients learned
leave-one-source-dataset-out is explicitly outside v0.8. It would introduce a
new algorithm, hyperparameters, and external validation estimand and is
reserved for a separate study.

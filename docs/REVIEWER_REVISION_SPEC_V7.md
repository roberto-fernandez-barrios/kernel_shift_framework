# Reviewer-driven v0.7 revision specification

**Status: FROZEN on 2026-07-29 before computing the v0.7 finite-shot
out-of-sample extension.**

This specification resolves the second strict npj Quantum Information
pre-submission review. It does not change the primary exact-statevector
estimand or its numerical results.

## 1. Confirmatory candidate budget

The only primary budget input is
`results/v4/budget_confirmatory/coverage.csv`. It contains 115 classical and
60 quantum candidates in every one of the eight scenario-groups and both
classifiers. The matched budget is therefore **60 candidates per family in
all 16 group-classifier cells**.

The earlier incomplete-coverage output under `results/v4/budget/` is
provisional history and must not ship in the v0.7 artifact. No main-text,
caption, table generator, reproduction command, or audit document may
describe a 35/20 pool, a budget of 20, or sampling four of seven blocks.

The primary `kernel_blocked` draw samples 12 of the 23 complete
kernel-shape/scale blocks, each crossed with five dimensions, for 60
classical candidates. Uniform and kernel-stratified sampling remain
sensitivities.

## 2. Specification-curve interpretation

The ten rows remain a frozen descriptive path over complete, previously
computed endpoints. S1--S4 use the legacy generation and fixed `C=1`;
S5--S10 use the corrected v4 generation and train-only SVC regularization.
Consequently, the S4-to-S5 transition does not identify an isolated causal
effect of regularization.

The curve may show how bundles of evaluation choices are associated with the
reported family effect. It must not claim to determine, identify, isolate, or
separate the causal contribution of every individual choice. Selection,
reference, and budget contrasts within v4 can be described as cleaner
within-generation contrasts, but the curve as a whole is descriptive.

## 3. Finite-shot conditions

The exact-statevector primary result is unchanged. The repeated finite-shot
experiment remains conditional on the same eight frozen configurations, four
shot counts, and 30 SHA-256-seeded measurement replicates.

Three conditions are reported:

1. `pre_psd`: sampled train and evaluation blocks without PSD correction.
2. `independent_square_psd`: the existing heuristic. The sampled train Gram
   and OOD square Gram are projected independently by negative-eigenvalue
   clipping, while rectangular evaluation-vs-train blocks remain sampled but
   otherwise unchanged. This is an algorithmic intervention, not a coherent
   out-of-sample kernel correction.
3. `nystrom_psd`: a coherent finite-dimensional extension derived from the
   positive eigenspace of the sampled train Gram.

For `nystrom_psd`, write the sampled train Gram as

`K_train = U diag(lambda) U.T`

and retain eigenvalues above

`max(1e-12, eps * n * max(1, max(abs(lambda))))`.

Let `U+` and `lambda+` denote the retained eigenvectors and eigenvalues. The
training correction and evaluation feature coordinates are

`K_train+ = U+ diag(lambda+) U+.T`

`Phi_eval = K_eval,train U+ diag(lambda+)^(-1/2)`.

The coherent rectangular block and evaluation square Gram are

`K_eval,train+ = Phi_eval (U+ diag(lambda+)^(1/2)).T`

`K_eval,eval+ = Phi_eval Phi_eval.T`.

For OOD KTA, `K_ood,ood+` is derived from the sampled OOD-vs-train block by
this formula; the independently sampled OOD square is used only in the first
two conditions. The selected SVC `C` is recomputed by the same train-only
cross-validation procedure in every condition.

The run-level output must record retained rank, eigenvalue tolerance, and
Frobenius changes from the sampled train, rectangular, and OOD-square blocks.
All three conditions receive identical measurement seeds and no result-based
row removal.

## 4. Small-cluster uncertainty

The cluster-t interval remains a conditional sensitivity over five q-split
cluster means, not an equivalence test and not inference over a population of
datasets. The main Methods must cite the few-cluster t-statistic literature.
Supplementary Information must show, for every primary group-classifier cell,
the cluster-t interval, 9,999-draw BCa interval, and leave-one-q-split range.

## 5. Positioning and reporting

- Paper title and claims use “shape” or “are associated with”, not causal
  “determine/identify/create” language.
- Prior QML OOD work on quantum dynamics and data-dependent quantum Fisher
  information is cited; novelty is stated narrowly for the controlled
  classical-data kernel-swap benchmark.
- The train-fitted map places training coordinates in `[0, pi]`; held-out
  coordinates may extrapolate outside that interval.
- OOD KTA is labelled post hoc and label-reusing, not a mechanistic endpoint.
- The cover letter emphasizes importance and Collection fit without repeating
  the abstract's numerical sequence.
- Release documentation must state the actual public GitHub and Zenodo
  identities and checksums.

## 6. Stop/go gates

The v0.7 release may be frozen only if:

1. every confirmatory coverage row has budget 60;
2. the provisional incomplete-coverage directory is absent;
3. the specification curve and title contain no per-axis causal claim;
4. all eight finite-shot run files contain
   `4 * 30 * 3 = 360` complete rows;
5. the Nyström unit tests establish train/cross/square feature-map coherence;
6. all analysis, reporting, artifact-validation, and test commands pass;
7. both PDFs are rebuilt and visually inspected page by page;
8. the tagged commit, GitHub release, Zenodo version, PDFs, and checksums are
   identical.

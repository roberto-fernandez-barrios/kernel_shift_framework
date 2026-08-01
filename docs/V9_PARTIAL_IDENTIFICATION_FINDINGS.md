# v0.9 finding: sharp evidence frontiers for quantum advantage

Status: manuscript-reconstruction decision, 2026-07-31.  The zero-label SVC
pilot was frozen before its prediction outcomes were inspected.  The
partial-label extension was declared after target labels were unlocked and is
therefore exploratory on these eight cases.  GPC repeats the same declared
mathematics and acquisition rule but is not a prospective external validation.

v0.9.1 amendment, 2026-08-01: after the original curves were known, the
acquisition claim was stress-tested against two prediction-aware controls and
an exact retrospective label oracle.  This does not alter the certificate or
make the acquisition study prospective.

## Decision

The v0.9 analysis passes the scientific go/no-go gate.  The manuscript should
be reconstructed around **sharp, target-domain partial identification of
quantum predictive advantage**, with the existing protocol factorial,
entangling/product split, geometry audit, external tasks, and shot study acting
as validation gates and boundary conditions.

The central claim is not that disagreement, active testing, partial
identification, or classical surrogates are individually new.  It is:

> A resource-indexed classical reference family and a limited target-label
> audit jointly define an exact finite-batch region for the accuracy advantage
> of a fixed quantum classifier over the best classical witness.  This region
> yields deterministic, assumption-free certificates of how much quantum
> advantage remains possible on the observed target batch.

## Exact result

For fixed target predictions `q` and classical family `C_B`,

```text
Delta_B(y) = Acc(q;y) - max_j Acc(c_j;y).
```

With no target labels, its sharp envelope is

```text
[-max_j d(q,c_j), min_j d(q,c_j)].
```

After auditing labels on `L`, let `D_j` be the full disagreement count,
`a_j(L)` the audited disagreements whose label favors `q`, and `b_j(L)` those
whose label favors `c_j`.  In binary classification the sharp envelope is

```text
[ min_j (-D_j + 2 a_j(L))/n,
  min_j ( D_j - 2 b_j(L))/n ].
```

The upper endpoint is monotone under arbitrary adaptive label acquisition.
Every audited classical-favoring counterexample reduces its witness-specific
upper endpoint by exactly `2/n`; a quantum-favoring label cannot increase it.
No i.i.d. target sampling, calibrated confidence, prevalence, covariate-shift,
or label-shift assumption is used for this batch statement.

## Headline empirical result

The fixed target batch has 500 examples in every case.  Against the complete
115-candidate classical family, the zero-label upper endpoints and the number
of labels required by the adaptive bottleneck-cover audit to reduce the sharp
upper endpoint to at most 0.010 are:

| Fixed case | SVC upper at 0 labels | SVC labels | GPC upper at 0 labels | GPC labels |
|---|---:|---:|---:|---:|
| EMBER m1 | 0.050 | 12 | 0.080 | 29 |
| EMBER m2 | 0.054 | 15 | 0.074 | 33 |
| ToN-IoT scanning, constructed | 0.002 | 0 | 0.016 | 3 |
| ToN-IoT scanning, campaign | 0.002 | 0 | 0.006 | 0 |
| UNSW DoS, constructed | 0.010 | 0 | 0.066 | 22 |
| UNSW DoS, campaign | 0.034 | 11 | 0.028 | 12 |
| UNSW reconnaissance, constructed | 0.046 | 14 | 0.088 | 22 |
| UNSW reconnaissance, campaign | 0.034 | 28 | 0.030 | 14 |

Across all 16 case--classifier audits, the adaptive requirement is 0--33 of
500 labels (median 13).  Across the 12 selected entangling-ZZ models it is
0--33 (median 14).  Prediction-independent hash ordering requires a median of
245.5 labels across the same 16 cells.  The comparison is descriptive over 200
stable hash permutations, not a confidence interval.

### v0.9.1 acquisition stress test

The weak hash comparator greatly overstates the distinctiveness of the
adaptive heuristic.  At the same full-family 0.010 endpoint:

| Policy | Median labels over 16 cells | Range of cell medians/counts |
|---|---:|---:|
| Exact retrospective label oracle | 6.5 | 0--20 |
| Fixed adaptive bottleneck coverage | 13.0 | 0--33 |
| Dynamic random active disagreement | 12.5 | 0--34 |
| Non-adaptive initial coverage | 12.5 | 0--286.5 |
| Prediction-independent hash all | 245.5 | 0--371.5 |

Against random-active sampling, adaptive coverage wins/ties/loses 5/5/6
cells; against non-adaptive coverage it wins/ties/loses 6/4/6.  Its median gap
to the exact oracle is 4.5 labels.  The non-adaptive policy has a severe 286.5
label outlier when its initial witness set becomes stale, but otherwise the
strong controls track the adaptive rule closely.

The defensible conclusion is therefore not that bottleneck coverage is a new
superior acquisition algorithm.  It is that the exact certificate exposes an
actionable disagreement set on which several simple prediction-aware audits
can contract the remaining advantage with very few labels.  This negative
algorithmic result is reported prominently in v0.9.1.

Fifteen of sixteen fully labelled full-family accuracy advantages are
non-positive.  The remaining GPC case is +0.002, which is compatible with and
below the 0.010 certificate.  The result therefore rules out a *material*
advantage above the chosen threshold; it does not force the estimated effect
to be non-positive.

## Why classical reference breadth is part of the estimand

For ToN-IoT scanning under the constructed shift, the selected GPC quantum
model has realized accuracy advantage +0.028 over the customary 30-member
linear/RBF reference.  No amount of labeling can falsify a 0.010 advantage
against that restricted family.  Against the prespecified full family, the
realized advantage is -0.014 and only three targeted labels are needed to
reduce the sharp upper endpoint to 0.010.  Equal-count benchmarking and
classical-indispensability auditing therefore answer different questions and
must both be reported.

## Balanced accuracy, transport, and finite shots

- The prevalence-conditional exact BAcc MILP closely tracks the zero-label
  accuracy upper endpoint and does not rescue a broad zero-label certificate.
- Two normalized joint PSD Gram matrices can have the same `K_TT` while one
  target is orthogonal to the training set and another duplicates a training
  feature.  Their KRR target operators differ by
  `1/(1+n*lambda)`.  Train-only geometry cannot characterize deployment
  behavior without `K_UT`.
- At 128 shots, the SVC predictor changes a median 7.2--7.4% of target
  decisions relative to exact statevectors, depending on PSD handling.  The
  across-case median change in the full-family zero-label upper endpoint is
  only +0.003 to +0.009 and the realized BAcc change is near zero or negative.
  Noise-induced distinctness is heterogeneous and is not evidence of useful
  quantum non-emulability.

## Literature boundary

The paper must explicitly distinguish the contribution from:

- classical surrogates for QML, which reproduce trained quantum input--output
  relations ([Schreiber et al.](https://arxiv.org/abs/2206.11740));
- geometric screening of possible quantum advantage
  ([Huang et al.](https://www.nature.com/articles/s41467-021-22539-9));
- active testing, which estimates test risk efficiently and corrects sampling
  bias ([Kossen et al.](https://proceedings.mlr.press/v139/kossen21a.html));
- discordant-pair evaluation, which labels disagreements relative to a known
  baseline and assumes baseline performance/prevalence for sensitivity and
  specificity
  ([Musgrove et al.](https://www.nature.com/articles/s41598-023-48017-4));
- partial-identification bounds for weak-supervision metrics using fitted
  label-model marginals
  ([Polo et al.](https://papers.nips.cc/paper_files/paper/2024/hash/f4c6bec746b0aeca8c2cd15096f1ad1f-Abstract-Conference.html));
- recent analysis of when quantum and classical minimum-norm learners disagree
  ([Thabet et al.](https://www.nature.com/articles/s41534-026-01217-y)).

The current search found no prior work combining a sharp best-of-family
relative-advantage envelope, a classical-reference frontier, partial target
labels, and a QML advantage-falsification audit.  This is not proof of priority;
the manuscript should use "we introduce" only after a final systematic search
and should avoid an unqualified "first" claim.

## Manuscript architecture

1. Protocol validity: train-only tuning, target-label-free selection, and
   equal-budget performance comparison.
2. Predictive indispensability: sharp zero/partial-label identification and
   the classical-search x target-supervision frontier.
3. Quantum relevance: entangling versus product kernels, train--target
   transport, circuit resources, and finite-shot sensitivity.

Passing all three gates identifies a candidate for deeper advantage analysis;
failing any gate falsifies a stronger interpretation.  It still does not prove
complexity-theoretic quantum advantage.

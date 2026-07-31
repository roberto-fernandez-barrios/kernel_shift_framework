# Paper 1: Q1 completion and submission strategy

Status: npj-first v0.8 reviewer revision in progress, 29 July 2026.

## Target and positioning

Submit first to **npj Quantum Information** as an Article for the Collection
*Quantum machine learning: understanding capabilities, limitations, and
perspectives for quantum advantage*. This is an ambitious but defensible
submission. The editorial case is not that quantum kernels lose a broad
benchmark; it is that the evaluation protocol changes the apparent robustness
claim under distribution shift.

The official Collection page was rechecked on 2026-07-29: submissions are
open and the listed deadline is 2026-12-31:
<https://www.nature.com/collections/iheeaggidj/about-the-collection>.

The central contribution is the controlled combination of:

1. train--test distribution shift;
2. target-label-free model selection;
3. train-only regularization;
4. structurally matched candidate budgets;
5. a within-v4 factorial plus a historical cross-generation sensitivity map;
6. a circuit-aware separation of entangling ZZ and product maps;
7. fixed-case, dependence-aware conditional uncertainty;
8. prospectively frozen corroboration on three external shifts.

If npj declines the paper after this fully controlled submission, **EPJ
Quantum Technology** remains the natural fallback. Journal ranking, APC, and
institutional-agreement status must be checked in the current JCR and
University of Deusto systems immediately before submission.

## Defensible claim

In these low-qubit simulated regimes, the apparent robustness ordering of
quantum and classical kernel families under distribution shift depends on
regularization, target-label access, reference-family strength, and search
budget. Under the deployment-compatible controls, no consistent evidence of a
family-level quantum advantage is observed in the fixed low-qubit cases.

This is a methodological result about what the benchmark estimates. It is not
a claim of computational speedup, hardware advantage, universal classical
dominance, or population inference from the fixed datasets.

## Frozen headline evidence

Primary equal-budget P1' source-dataset-equal effects:

| Classifier | Quantum minus extended classical | Leave-one-source-dataset-out range |
|---|---:|---:|
| SVC | -0.005649 | [-0.007024, -0.004502] |
| GPC | -0.000937 | [-0.001900, +0.000939] |

Contextual target-label-free comparison against linear+RBF:

| Classifier | Quantum minus linear+RBF |
|---|---:|
| SVC | -0.004073 |
| GPC | +0.005907 |

The scenario-group estimates are heterogeneous and their intervals are
pointwise rather than simultaneous. They are reported descriptively and are
not classified by whether an interval includes zero; inclusion of zero is not
evidence of equivalence.

The security aggregate gives equal weight to three source datasets: EMBER,
UNSW-NB15, and ToN-IoT. UNSW DoS and reconnaissance are scenario-groups within
one source dataset, not independent datasets.

## Reviewer-driven strengthening

- The historical S1--S10 map and headline endpoint use the same three-source
  estimand, but the map is explicitly cross-generation and associational.
- A 2x2x2x2 SVC factorial within v4 crosses selection, regularization,
  reference family, and candidate budget without using the legacy/v4 boundary
  to isolate an axis. Its fixed-C/oracle/customary/native corner is +0.01308
  and its train-CV/ID-validation/extended/equal-count corner is -0.00467
  (descriptive contraction 0.01775). Mean paired changes are -0.00838 for
  regularization, -0.00795 for reference strength, -0.00367 for selection,
  and -0.00019 for budget; a +0.01394 regularization-by-reference interaction
  prohibits treating them as additive contributions.
- The circuit audit distinguishes two entangling ZZ maps from two separable
  product maps. At matched 30-candidate budgets, the source-dataset-equal SVC
  effects are -0.00928 and -0.00547, respectively; excluding product maps does
  not expose a hidden positive quantum effect.
- Logical circuit depth/CX counts and exact finite-shot sampling-resource
  projections are reported.
- A frozen q1000 ablation removes port/protocol/service-derived fields before
  the shared embedding and reports all six network fixed cases. The
  two-source-dataset-equal controlled effect changes from -0.00306 to
  -0.02452; the source-dataset changes are -0.04468 for ToN-IoT and +0.00176
  for UNSW-NB15, so the result is reported as heterogeneous rather than as a
  universal shortcut conclusion.
- `kernel_blocked` is explicitly primary because it samples whole
  kernel-shape blocks crossed with all five dimensions. Uniform and
  kernel-stratified sensitivities change no family-level conclusion.
- Rank matching reports the predefined 1.25 caliper, 75.2% retention,
  with-replacement reuse, rank discrepancy, alternative calipers, and
  one-to-one matching. Its interpretation is observational.
- Finite-shot analysis uses 30 independent measurement seeds for each of eight
  fixed exact Gram matrices and four shot counts under three conditions:
  unprojected, independent-square PSD, and a coherent train-eigenspace
  Nyström extension (2,880 evaluations). It remains conditional and does not
  simulate hardware. The complete sensitivity projects 4.24 trillion shots,
  before device routing, noise, mitigation, and job overhead.
- The primary matched budget is 60 candidates per family in all eight
  scenario-groups and both classifiers; the obsolete incomplete-coverage
  output has been removed from the submission artifact.
- The cross-generation sensitivity map is explicitly descriptive. Its S4--S5 boundary is
  generation-confounded and is not used to attribute an isolated causal
  effect to regularization.
- External College Scorecard, diabetes readmission, and ACS income shifts
  corroborate protocol sensitivity under a prospectively frozen design.
- All reproducibility-critical Methods are in the main manuscript;
  Supplementary Information contains extended results and diagnostics only.

## Novelty relative to concurrent work

Large and recent quantum-kernel benchmarks already provide strong classical
baselines, nested validation, spectral analysis, and in some cases hardware
experiments. The manuscript therefore concentrates its novelty on the joint
estimand:

> distribution shift + target-label-free selection + train-only
> regularization + structurally matched search + within-generation factorial
> + entangling/product circuit stratification + prospective external
> corroboration.

The paper should never rely on “quantum kernels lose” as its novelty claim.
Its contribution is showing how the apparent advantage changes across
evaluation bundles and providing a reproducible audit design.

## Claims to avoid

- Quantum advantage, practical advantage, speedup, or hardware relevance.
- A population-level effect inferred from eight scenario-groups, three source
  datasets, or three external tasks.
- A causal claim that geometry rather than quantumness carries information.
- A universal ordering over all encodings, qubit regimes, datasets, or
  classical and quantum kernel families.
- Treating fifteen nested pipeline realizations as fifteen independent
  observations.
- Calling candidate-count equality an isomorphic search space.

## Current submission package

- Main Article source: 10-word title and an abstract within the 150-word
  Article limit.
- Supplementary Information contains no Supplementary Methods.
- Concise collection-specific cover letter with the exact v0.5.0 prior-artifact
  identifier and overlap statement.
- Sequential numeric `sn-nature` bibliography with the published EPJ Quantum
  Technology reference and concurrent 2026 benchmark.
- Public code, frozen manifests, complete aggregated results, tests, and
  deterministic reporting scripts.
- Data, code, competing-interest, author-contribution, funding, and
  generative-AI declarations.

## Reproduction and release gates

Run from the repository root in the declared environment:

```bash
python scripts/reproduce_v6.py --stage analysis
python scripts/reproduce_v6.py --stage report
python scripts/reproduce_v6.py --stage audit
python scripts/analysis/validate_v6_artifacts.py
python scripts/reproduce_v8.py --stage all
python -m pytest tests -q
```

Current status:

- the immutable v0.7 reproduction and artifact gates pass;
- the complete 360-run fixed-C and 270-run shortcut campaigns finished with
  no worker errors;
- all v0.8 analysis, circuit, resource, campaign-coverage, and manuscript
  format gates pass;
- final v0.8 full-suite, PDF, and visual gates pass; artifact identity remains
  open until the reserved version is published and independently resolved;
- authors approve the target, claims, and required releases.

The immutable v0.6.0 DOI `10.5281/zenodo.21672470` remains public and resolves
to the historical tagged artifact. Corrective release v0.7.0 is public at
<https://github.com/roberto-fernandez-barrios/kernel_shift_framework/releases/tag/v0.7.0>
and <https://doi.org/10.5281/zenodo.21676563>; the DOI resolves, is DataCite
`findable`, and identifies the tagged source plus the inspected main and
Supplementary PDFs.
The v0.8.0 draft is reserved under immutable DOI
<https://doi.org/10.5281/zenodo.21717074>.

## Submission stop/go

Submit only when:

1. the v0.8 reserved DOI resolves to the exact tagged source and submitted PDFs;
2. GitHub, Zenodo, citation metadata, manuscript, and Supplementary
   Information identify v0.8.0 consistently;
3. the Collection and Article type are selected in the portal;
4. the current APC route or institutional agreement is confirmed;
5. suggested reviewers and conflicts are checked by the authors;
6. the journal reporting and editorial-policy forms are complete.

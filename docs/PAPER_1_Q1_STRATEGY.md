# Paper 1: Q1 completion and submission strategy

Status: npj-first v0.7 reviewer revision in progress, 29 July 2026.

## Target and positioning

Submit first to **npj Quantum Information** as an Article for the Collection
*Quantum machine learning: understanding capabilities, limitations, and
perspectives for quantum advantage*. This is an ambitious but defensible
submission. The editorial case is not that quantum kernels lose a broad
benchmark; it is that the evaluation protocol changes the apparent robustness
claim under distribution shift.

The central contribution is the controlled combination of:

1. train--test distribution shift;
2. target-label-free model selection;
3. train-only regularization;
4. structurally matched candidate budgets;
5. a ten-specification curve;
6. fixed-case, dependence-aware conditional uncertainty;
7. prospectively frozen corroboration on three external shifts.

If npj declines the paper after this fully controlled submission, **EPJ
Quantum Technology** remains the natural fallback. Journal ranking, APC, and
institutional-agreement status must be checked in the current JCR and
University of Deusto systems immediately before submission.

## Defensible claim

In these low-qubit simulated regimes, the apparent robustness ordering of
quantum and classical kernel families under distribution shift depends on
regularization, target-label access, reference-family strength, and search
budget. Once the deployment-compatible controls are imposed, no robust
family-level quantum advantage remains.

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

The primary SVC estimate is classical-favoured or within its conditional
interval of zero in seven of eight scenario-groups. The only scenario with a
small positive point estimate under both equal-budget classifiers is
UNSW-NB15 reconnaissance under held-out campaign shift; both intervals include
zero.

The security aggregate gives equal weight to three source datasets: EMBER,
UNSW-NB15, and ToN-IoT. UNSW DoS and reconnaissance are scenario-groups within
one source dataset, not independent datasets.

## Reviewer-driven strengthening

- The complete specification curve and headline endpoint now use the same
  three-source estimand.
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
  simulate hardware.
- The primary matched budget is 60 candidates per family in all eight
  scenario-groups and both classifiers; the obsolete incomplete-coverage
  output has been removed from the submission artifact.
- The specification curve is explicitly descriptive. Its S4--S5 boundary is
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
> regularization + structurally matched search + specification curve +
> prospective external corroboration.

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
python -m pytest tests -q
```

Current status:

- the focused v0.7 unit and artifact gates pass;
- the full reproduction, full test suite, PDF compilation, and page-by-page
  visual inspection remain release gates;
- authors approve the target, claims, and required releases.

The immutable v0.6.0 DOI `10.5281/zenodo.21672470` is public and resolves to
the tagged v0.6.0 artifact. The reviewer correction requires a new immutable
v0.7.0 Zenodo version derived from that record; its DOI must be reserved and
inserted before the final PDF and release freeze.

## Submission stop/go

Submit only when:

1. the reserved DOI resolves to the exact tagged source and submitted PDFs;
2. GitHub, Zenodo, citation metadata, manuscript, and Supplementary
   Information identify v0.7.0 consistently;
3. the Collection and Article type are selected in the portal;
4. the current APC route or institutional agreement is confirmed;
5. suggested reviewers and conflicts are checked by the authors;
6. the journal reporting and editorial-policy forms are complete.

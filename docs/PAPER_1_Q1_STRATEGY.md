# Paper 1: Q1 completion and submission strategy

Status: npj-first editorial plan implemented locally, 28 July 2026. The main
manuscript and separate Supplementary Information compile and have passed
visual PDF inspection. Public release alignment and co-author approval remain.

## Recommended decision

Prepare the paper first for **npj Quantum Information**, explicitly for the
open Collection *Quantum machine learning: understanding capabilities,
limitations, and perspectives for quantum advantage*. The call closes on
31 December 2026:

https://www.nature.com/collections/iheeaggidj

This is a stretch submission, but the call is unusually well aligned with the
paper's strongest defensible contribution: it identifies when an apparent
quantum-kernel robustness advantage is created by evaluation choices and when
it disappears under deployable selection and matched candidate budgets.

The submission ladder should be:

1. **npj Quantum Information, QML capabilities/limitations Collection.**
   Highest-value fit. The journal reports a 2025 JIF of 9.0. It is fully open
   access; the current original-research APC is EUR 3,690 before VAT, subject
   to institutional agreements or waivers.
2. **EPJ Quantum Technology.** Realistic fallback with direct quantum
   information/computation scope. The journal reports a 2025 JIF of 4.5, a
   median first decision of five days, and a current APC of EUR 1,990 before
   VAT. Confirm its exact JCR quartile and category in the University of
   Deusto's current JCR before treating it as satisfying a strict Q1 target.

Quantum Science and Technology is not recommended between these two: it is
highly selective and explicitly requires a significant, lasting advance of
broad interest. The present low-qubit, simulation-only study has a clearer
editorial route through the npj Collection and a more realistic fallback in
EPJ Quantum Technology.

Machine Learning: Science and Technology is a strong scientific fit for the
benchmark and reproducibility angle, but its current JCR quartile should be
verified before using it in a strict Q1-only ladder.

## The paper we should submit

### One-sentence claim

Under distribution shift, an apparent robustness advantage of fidelity
quantum kernels over customary baselines is not stable to train-only
regularization tuning, no-OOD-label model selection, and equal candidate
budgets; kernel geometry is informative but shared by quantum and classical
families.

### Contribution hierarchy

1. **Primary contribution:** an audit of evaluation choices for quantum-kernel
   robustness under distribution shift.
2. **Primary evidence:** the equal-budget P1' comparison, with configuration
   selection based only on ID validation and regularization tuned on training
   data.
3. **Mechanistic evidence:** effective-rank matching and geometry--OOD
   associations show that geometry, rather than kernel provenance, organizes
   the observed behaviour.
4. **Boundary evidence:** finite-shot perturbation shows that exact
   statevector geometry does not transfer unchanged to finite measurement.
5. **Context, not an endpoint:** the +0.037 same-test oracle result explains
   how an optimistic conclusion can be produced; it must never be presented
   as confirmatory evidence.

### Claims to avoid

- Quantum advantage, practical advantage, speedup, or hardware relevance.
- A population-level effect inferred from the eight fixed scenario-groups.
- A universal causal law connecting effective rank or alignment to OOD
  accuracy.
- A claim that all quantum kernels, encodings, qubit regimes, or datasets have
  been refuted.
- Broad novelty claims such as "the first rigorous quantum-kernel benchmark."

### Novelty relative to the closest 2025--2026 work

Large benchmarks already show that strong classical baselines frequently erase
quantum-kernel gains. A 2026 study additionally combines nested cross-validation,
spectral analysis, and a small hardware validation on tabular i.i.d. tasks.
The novelty here must therefore be narrower and explicit:

- train--test distribution shift is the primary evaluation axis;
- target-domain labels are unavailable to the deployment selector;
- candidate-family budget is treated as a fairness variable;
- constructed and held-out attack-campaign shifts are compared;
- inference is framed as fixed-case, dependence-aware estimation;
- geometry is studied specifically as a correlate of behaviour under shift.

## Evidence frozen for the main claim

Primary equal-budget P1' dataset-equal means:

| Classifier | Quantum minus extended classical | Leave-one-dataset-out range |
|---|---:|---:|
| SVC | -0.0050 | [-0.0060, -0.0040] |
| GPC | -0.0019 | [-0.0028, +0.0003] |

Contextual no-OOD-label comparison against linear+RBF:

| Classifier | Quantum minus linear+RBF |
|---|---:|
| SVC | -0.0043 |
| GPC | +0.0043 |

The primary SVC result is classical-favoured or conditionally tied in seven of
eight scenario-groups. The only group with a small positive SVC and GPC point
estimate against the equal-budget extended family is UNSW-Recon under
held-out-campaign shift; both intervals include zero.

These results are sufficient for the paper's main claim. Do not launch a new
wide experimental grid unless peer review requests it. New experiments now
would increase researcher degrees of freedom and risk weakening the frozen
confirmatory framing.

## Manuscript surgery

Completed locally. The main manuscript now contains approximately 5,300 words
of prose (approximately 5,900 including headings and captions) and 31 pages in
the double-spaced, line-numbered review template. The separate Supplementary
Information is six pages.

### Main paper

- [x] Fold Related Work into the Introduction.
- [x] Lead with the distribution-shift selection problem, not cybersecurity
  background or a generic QML survey.
- [x] Place Results and Discussion before the concise Methods section.
- [x] Keep four main figures:
  1. controlled protocol;
  2. equal-budget no-OOD-label headline comparison;
  3. combined rank-matching and geometry result;
  4. finite-shot boundary analysis.
- [x] Keep two compact main tables. The per-group headline table remains
  with a large table unless the exact group estimates are essential in main
  text.
- [x] Condense Discussion to implications, comparison with closest work, and
  limitations. Merge the present Conservative Interpretation and Conclusion.

### Supplementary Information

Moved without deleting them from the publication package:

- [x] detailed dataset and split-construction algorithms;
- [x] complete candidate grids and preprocessing specifications;
- [x] same-test oracle table;
- [x] full geometry descriptor definitions and per-group table;
- [x] Gaussian-process uncertainty diagnostics;
- [x] finite-shot model details and artifact pointers;
- [x] additional result inventory and machine-readable result map.

## Traceability issue already corrected

The previous headline table and figure were labelled as budget matched while
reading the full-pool family-comparison CSV. The primary extended-classical
comparison now reads `results/v4/inference_confirmatory/hierarchical_effects.csv`
with variant `budget60` and stratum `all`. The full-pool result is explicitly
a secondary sensitivity.

This distinction must be protected by a regression test before release.

## Artifact and release blockers

The local manuscript postdates the public v0.4.0 artifact (version DOI
`10.5281/zenodo.21488509`; concept DOI `10.5281/zenodo.19147649`) and is not
presently aligned with it. Before submission:

1. Make the README describe the negative, protocol-dependent result and use
   `python scripts/reproduce_v4.py --stage all` as the canonical entry point.
2. Add a regression test that checks headline-table values against the frozen
   equal-budget inference CSV.
3. Run the complete reproduction pipeline and test suite in the declared
   environment.
4. Freeze a new release commit and version (`v0.4.1` is the recommended patch
   release); ensure `pyproject.toml`, `CITATION.cff`, README, manuscript title,
   tag, and release notes agree. Never move the existing v0.4.0 tag.
5. Archive exactly that commit and results bundle in Zenodo.
6. Verify that the DOI in the manuscript resolves to that exact final archive,
   not the earlier v0.4.0 record.
7. Only then update the public GitHub repository. Remote publication requires
   explicit author approval.

## Submission package

- concise main manuscript and separate Supplementary Information;
- cover letter naming the npj QML Collection and explaining the precise
  capabilities/limitations contribution;
- graphical or one-paragraph significance summary if requested;
- reporting checklist and data/code availability statements;
- frozen artifact DOI and public repository;
- suggested reviewers selected for distribution shift, quantum kernels, and
  benchmarking, with conflicts checked by the authors.

## Stop/go criteria

The paper is ready to submit when:

- every headline number traces to the equal-budget frozen result;
- no abstract, caption, or conclusion implies population inference or hardware
  relevance;
- the main paper is below approximately 8,000 words;
- the full pipeline and tests pass from the documented environment;
- GitHub, Zenodo, citation metadata, and manuscript tell the same story;
- all co-authors approve the claims, journal, APC route, and final files.

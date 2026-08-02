# Paper 1 v0.8 submission and artifact checklist

Target: *npj Quantum Information*, Collection "Quantum machine learning:
understanding capabilities, limitations, and perspectives for quantum
advantage".

This checklist supplements the immutable v0.7 record. Historical tags and
Zenodo versions must not be moved or replaced.

## Scientific closure

- [x] Freeze the v0.8 sensitivity estimands before generating new outcomes.
- [x] Correct the separable versus entangling circuit characterization.
- [x] Add logical circuit and finite-shot resource accounting.
- [x] Replace causal interpretation of the cross-generation S1--S10 display
  with a within-v4 factorial.
- [x] Replace equivalence and simultaneous-inference language with pointwise,
  conditional fixed-case interpretation.
- [x] Remove the incomplete Huang geometric-difference claim.
- [x] Correct environment and author-provided-audit descriptions.
- [x] Complete all 360 fixed-C and 270 shortcut-ablation run outputs.
- [x] Generate and inspect the entangling/product, factorial, shortcut, and
  five-cluster-value outputs.
- [x] Insert every new result into the main manuscript and Supplementary
  Information without hiding fixed-case heterogeneity.
- [ ] Final author review of scientific claims and tables.

## Reproduction and quality gates

Run from the repository root in the declared environment:

```bash
python scripts/reproduce_v8.py --stage all
python -m pytest tests -q
```

Then regenerate and inspect:

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error sn-article.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

- [x] v0.7 integrity gates still pass.
- [x] v0.8 campaign and derived-artifact gates pass.
- [x] Complete test suite passes (79/79).
- [x] No undefined citations/references or overfull boxes.
- [x] Main and Supplementary PDFs rendered to page images and inspected.
- [x] Submission directory contains one unambiguous main PDF and one
  Supplementary PDF.
- [x] Article title is at most 15 words, abstract at most 150 words, and the
  main manuscript contains no separate Conclusions or Limitations section.
- [x] All procedural descriptions remain in main-file Methods; Supplementary
  Information contains results and diagnostics only.
- [x] Main manuscript has no more than ten display items and every multi-panel
  figure uses lower-case panel labels.
- [ ] Authors verify the public-data ethics and consent statement literally.

## Metadata and immutable release

- [x] Reserve the v0.8.0 version DOI from the existing Zenodo concept record:
  `10.5281/zenodo.21717074`.
- [x] Set version `0.8.0` in `pyproject.toml` and `CITATION.cff`.
- [x] Insert the reserved immutable DOI in the manuscript, Supplementary
  Information, README, cover letter, and citation metadata.
- [x] Align titles, authors, ORCIDs, affiliation, licence, and dates.
- [x] Commit and push the exact tested source, aggregated results, and PDFs:
  `275add120a665f1b5f8d545fbd66613143dc46db`.
- [x] Tag the tested commit as `v0.8.0`; the annotated tag resolves to that
  exact commit.
- [x] Publish the GitHub v0.8.0 release:
  <https://github.com/roberto-fernandez-barrios/kernel_shift_framework/releases/tag/v0.8.0>.
- [x] Publish and inspect the Zenodo v0.8.0 record:
  <https://doi.org/10.5281/zenodo.21717074>.
- [x] Confirm DOI resolution, DataCite `findable` status, public file identity,
  and checksums. The concept DOI `10.5281/zenodo.19147649` resolves to v0.8.0.

Verified SHA-256 values:

- main PDF:
  `ce42579c246651b2cbde43b02acd6b3f2b9f29f32511d3219fe26a69ac1f4f94`;
- Supplementary PDF:
  `a0bcbc51dc411b0afd062313e39c9718d906a619356a404e77c23ec417c4b613`;
- tagged source archive:
  `11a1d953c0e8bc1b072a9ad4483efe8b82d2fdb26f93920b0f717e70667533bb`.

## Submission stop/go

Submit only if:

1. every headline number is regenerated from the archived commit;
2. no claim implies equivalence, hardware advantage, computational speedup,
   causal geometry, or inference to a population of datasets;
3. the circuit-aware analysis distinguishes product from entangling maps;
4. GitHub, Zenodo, citation metadata, and the submitted files identify one
   immutable v0.8.0 artifact;
5. all authors approve the final package.

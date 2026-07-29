# Paper 1 v0.8 submission and artifact checklist

Target: *npj Quantum Information*, Collection "Quantum machine learning:
understanding capabilities, limitations, and perspectives for quantum
advantage".

This checklist supplements the immutable v0.7 record. Historical tags and
Zenodo versions must not be moved or replaced.

## Scientific closure

- [x] Freeze the reviewer-revision estimands before generating new outcomes.
- [x] Correct the separable versus entangling circuit characterization.
- [x] Add logical circuit and finite-shot resource accounting.
- [x] Replace causal interpretation of the cross-generation S1--S10 display
  with a within-v4 factorial.
- [x] Replace equivalence and simultaneous-inference language with pointwise,
  conditional fixed-case interpretation.
- [x] Remove the incomplete Huang geometric-difference claim.
- [x] Correct environment and author-provided-audit descriptions.
- [ ] Complete all 360 fixed-C and 270 shortcut-ablation run outputs.
- [ ] Generate and inspect the entangling/product, factorial, shortcut, and
  five-cluster-value outputs.
- [ ] Insert every new result into the main manuscript and Supplementary
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

- [ ] v0.7 integrity gates still pass.
- [ ] v0.8 campaign and derived-artifact gates pass.
- [ ] Complete test suite passes.
- [ ] No undefined citations/references or overfull boxes.
- [ ] Main and Supplementary PDFs rendered to page images and inspected.
- [ ] Submission directory contains one unambiguous main PDF and one
  Supplementary PDF.
- [ ] Article title is at most 15 words, abstract at most 150 words, and the
  main manuscript contains no separate Conclusions or Limitations section.
- [ ] All procedural descriptions remain in main-file Methods; Supplementary
  Information contains results and diagnostics only.
- [ ] Main manuscript has no more than ten display items and every multi-panel
  figure uses lower-case panel labels.
- [ ] Authors verify the public-data ethics and consent statement literally.

## Metadata and immutable release

- [ ] Reserve the v0.8.0 version DOI from the existing Zenodo concept record.
- [ ] Set version `0.8.0` in `pyproject.toml` and `CITATION.cff`.
- [ ] Insert the reserved immutable DOI in the manuscript, Supplementary
  Information, README, cover letter, and citation metadata.
- [ ] Align titles, authors, ORCIDs, affiliation, licence, and dates.
- [ ] Commit and push the exact tested source, aggregated results, and PDFs.
- [ ] Tag the tested commit as `v0.8.0`.
- [ ] Publish the GitHub v0.8.0 release.
- [ ] Publish and inspect the Zenodo v0.8.0 record.
- [ ] Confirm DOI resolution, DataCite status, file identity, and checksums.

## Submission stop/go

Submit only if:

1. every headline number is regenerated from the archived commit;
2. no claim implies equivalence, hardware advantage, computational speedup,
   causal geometry, or inference to a population of datasets;
3. the circuit-aware analysis distinguishes product from entangling maps;
4. GitHub, Zenodo, citation metadata, and the submitted files identify one
   immutable v0.8.0 artifact;
5. all authors approve the final package.

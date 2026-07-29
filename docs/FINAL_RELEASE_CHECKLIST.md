# Paper 1 submission and artifact-release checklist

Last local validation: 29 July 2026.

Target: *npj Quantum Information*, Collection "Quantum machine learning:
understanding capabilities, limitations, and perspectives for quantum
advantage". Fallback: *EPJ Quantum Technology*.

## Completed locally

- [x] Main title, abstract, introduction, results, discussion, and conclusion
  use the protocol-dependent negative result.
- [x] The headline table and figure read the frozen equal-budget `budget60`
  estimand, not the full-pool sensitivity.
- [x] No same-test oracle result is presented as confirmatory evidence.
- [x] No pooled population significance test is attached to the eight fixed
  scenario-groups.
- [x] Main manuscript reduced below 8,000 counted words.
- [x] Every reproducibility-critical method is reported in the main manuscript;
  Supplementary Information contains extended results and diagnostics only.
- [x] Main and Supplementary PDFs compile without undefined citations,
  undefined references, or overfull boxes.
- [x] Both PDFs rendered page by page and visually checked.
- [x] Reporting regression tests cover the three-source estimand, budget
  schemes, rank matching, repeated finite-shot analysis, and frozen hashes.
- [x] README describes the current result and the canonical v6 reproduction
  command.
- [x] The npj Quantum Information cover letter identifies the Collection,
  prior public draft artifact, overlap, and substantive differences.
- [x] Title (10 words) and abstract (135 words) are within the Article limits.
- [x] References are sequentially numbered and unresolved bibliography
  placeholders have been removed.
- [x] Data, code, competing-interest, author-contribution, funding, and
  generative-AI statements are present.

## Validate before freezing a release

Run from the repository root in the declared environment:

```bash
python scripts/reproduce_v6.py --stage analysis
python scripts/reproduce_v6.py --stage report
python scripts/reproduce_v6.py --stage audit
python scripts/analysis/validate_v6_artifacts.py
python -m pytest tests -q
```

Then regenerate the submission PDFs:

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error sn-article.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

Check the logs:

- no undefined citations or references;
- no overfull boxes;
- headline three-source-dataset-equal values remain SVC `-0.005649` and GPC
  `-0.000937` against the equal-budget extended family;
- main and Supplementary Information refer to the same protocol and version.

## Metadata to align in the release commit

- [x] Version `0.6.0` is set consistently in `pyproject.toml` and
  `CITATION.cff`; the release will use the immutable `v0.6.0` tag.
- [x] Ensure the README title, manuscript title, release notes, and Zenodo
  description state the same protocol-dependent conclusion.
- [x] Confirm author names, accents, ORCIDs, affiliation, and corresponding
  email.
- [x] Confirm the code and artifact rights holder and approval of the
  BSD-3-Clause licence; see `docs/LEGAL_RELEASE_NOTE.md`.
- [x] Confirm third-party dataset redistribution boundaries.
- [ ] Insert the reserved immutable v0.6.0 DOI in `CITATION.cff`, README, the
  main manuscript, Supplementary Information, and cover letter.

## Archive and public repository

- [x] Commit and push the exact tested source and generated pre-DOI manuscript
  files to draft pull request 2.
- [ ] Tag the exact release commit; do not move or overwrite the historical
  tags.
- [ ] Create a GitHub release from that tag.
- [ ] Publish the manually prepared Zenodo v0.6.0 draft and inspect the
  deposited file list.
- [ ] Confirm that the manuscript DOI resolves to the exact submitted release,
  not an earlier artifact state.
- [ ] Record SHA-256 checksums for the tagged source archive and both submitted
  PDFs.

The corresponding author has explicitly approved pushing, tagging, and
creating the required release. Publication of the reserved Zenodo draft still
requires the final artifact-identity check.

## Journal submission

- [x] Main manuscript source and PDF.
- [x] Supplementary Information source and PDF.
- [x] Target-specific cover letter.
- [ ] Confirm the Collection selection in the submission portal.
- [ ] Confirm article type and APC route or institutional agreement.
- [ ] Supply the final artifact DOI and repository URL.
- [x] Declare the earlier public v0.5.0 draft artifact and explain the overlap
  and differences.
- [ ] Prepare suggested reviewers and have all authors check conflicts,
  current affiliations, and institutional email addresses.
- [x] Obtain explicit co-author approval of the final claims and files.
- [ ] Complete the journal reporting and editorial-policy forms.

## Scientific stop/go

Submit only if all of the following remain true:

1. the abstract, headline table, figures, discussion, and cover letter trace to
   the frozen equal-budget analysis;
2. no text implies hardware advantage, speedup, universal dominance, or a
   population effect;
3. the tested commit is the archived commit;
4. GitHub, Zenodo, citation metadata, and manuscript identify the same release;
5. every co-author has approved the final package.

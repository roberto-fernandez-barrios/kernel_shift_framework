# Paper 1 submission and artifact-release checklist

Last local validation: 28 July 2026.

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
- [x] Detailed constructions and secondary analyses moved to separate
  Supplementary Information.
- [x] Main and Supplementary PDFs compile without undefined citations,
  undefined references, or overfull boxes.
- [x] Both PDFs rendered page by page and visually checked.
- [x] Reporting regression test added for the equal-budget source and table
  schema.
- [x] README describes the current result and the canonical v4 reproduction
  command.
- [x] npj Quantum Information and EPJ Quantum Technology cover letters updated
  to match the current claims.

## Validate before freezing a release

Run from the repository root in the declared environment:

```bash
python scripts/reproduce_v4.py --stage all
pytest -q
python scripts/smoke/smoke_test_cli.py
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
- headline values remain SVC `-0.0050` and GPC `-0.0019` against the
  equal-budget extended family;
- main and Supplementary Information refer to the same protocol and version.

## Metadata to align in the release commit

- [ ] Use a new release version for this post-v0.4.0 revision (`v0.4.1` is the
  recommended patch version), and set it consistently in `pyproject.toml`,
  `CITATION.cff`, the Git tag, and the GitHub release. Do not reuse the
  existing `v0.4.0` tag.
- [ ] Ensure the README title, manuscript title, release notes, and Zenodo
  description state the same protocol-dependent conclusion.
- [ ] Confirm author names, accents, ORCIDs, affiliation, and corresponding
  email.
- [ ] Confirm the code and artifact rights holder and approval of the
  BSD-3-Clause licence; see `docs/LEGAL_RELEASE_NOTE.md`.
- [ ] Confirm third-party dataset redistribution boundaries.

## Archive and public repository

- [ ] Commit the exact tested source and generated manuscript files.
- [ ] Tag the exact release commit; do not move or overwrite the historical
  v0.3.0 or v0.4.0 tags.
- [ ] Create a GitHub release from that tag.
- [ ] Wait for Zenodo ingestion and inspect the deposited file list.
- [ ] Confirm that the manuscript DOI resolves to the exact submitted release,
  not an earlier artifact state.
- [ ] If Zenodo assigns a new version DOI, update `CITATION.cff`, README, and
  both manuscript files, rebuild, and recheck before submission.

Remote publication, pushing, tagging, and Zenodo release creation require the
corresponding author's explicit approval.

## Journal submission

- [x] Main manuscript source and PDF.
- [x] Supplementary Information source and PDF.
- [x] Target-specific cover letter.
- [ ] Confirm the Collection selection in the submission portal.
- [ ] Confirm article type and APC route or institutional agreement.
- [ ] Supply the final artifact DOI and repository URL.
- [ ] Declare the earlier preprint and update it to the submitted version when
  appropriate.
- [ ] Prepare suggested reviewers and have all authors check conflicts,
  current affiliations, and institutional email addresses.
- [ ] Obtain explicit co-author approval of the final claims and files.
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

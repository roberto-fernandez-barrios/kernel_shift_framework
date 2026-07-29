# Paper 1 submission and artifact-release checklist

Last local validation: 29 July 2026.

Target: *npj Quantum Information*, Collection "Quantum machine learning:
understanding capabilities, limitations, and perspectives for quantum
advantage". Fallback: *EPJ Quantum Technology*.

## Scientific corrections completed locally

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
- [x] Rebuild the main and Supplementary PDFs without undefined citations,
  undefined references, or overfull boxes.
- [x] Render and inspect both PDFs page by page after the v0.7 changes.
- [x] Reporting regression tests cover the three-source estimand, complete
  60-candidate budget coverage, rank matching, coherent finite-shot PSD
  extension, and frozen hashes.
- [x] README describes the current result and the canonical v0.7 reproduction
  command.
- [x] The npj Quantum Information cover letter identifies the Collection,
  prior public draft artifact, overlap, and substantive differences.
- [x] Title (10 words) and abstract are within the Article limits.
- [x] References are sequentially numbered and unresolved bibliography
  placeholders have been removed; the manuscript uses `sn-nature`.
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

## Metadata to align in the v0.7 release commit

- [x] Set version `0.7.0` consistently in `pyproject.toml` and
  `CITATION.cff`; use the immutable `v0.7.0` tag.
- [x] Ensure the README title, manuscript title, release notes, and Zenodo
  description state the same protocol-dependent conclusion.
- [x] Confirm author names, accents, ORCIDs, affiliation, and corresponding
  email.
- [x] Confirm the code and artifact rights holder and approval of the
  BSD-3-Clause licence; see `docs/LEGAL_RELEASE_NOTE.md`.
- [x] Confirm third-party dataset redistribution boundaries.
- [x] Insert the reserved immutable v0.7.0 DOI in `CITATION.cff`, README, the
  main manuscript, Supplementary Information, and cover letter.

## Archive and public repository

- [x] v0.6.0 is public on GitHub and Zenodo; DOI
  `10.5281/zenodo.21672470` resolves and the historical tag is immutable.
- [x] Reserve DOI `10.5281/zenodo.21676563` for a new Zenodo v0.7.0 version derived from
  the public v0.6.0 record.
- [x] Commit and push the exact tested v0.7 source and generated PDFs.
- [x] Tag merge commit `77ab11a1938b4fc4cd5021ec4d93d04cf603b8a9`
  as `v0.7.0`; do not move or overwrite historical
  tags.
- [x] Create the GitHub v0.7.0 release from that tag:
  <https://github.com/roberto-fernandez-barrios/kernel_shift_framework/releases/tag/v0.7.0>.
- [x] Publish and inspect the Zenodo v0.7.0 file list:
  <https://zenodo.org/records/21676563>.
- [x] Confirm that DOI `10.5281/zenodo.21676563` resolves, is DataCite
  `findable`, and identifies the exact submitted v0.7
  release.
- [x] Record SHA-256 checksums for the tagged source archive and both submitted
  PDFs.

Final SHA-256 identity:

- main PDF: `a855ab2bbd52af6115ef1bead84501887dd3048894e09cf84e0ce02ead67fc63`;
- Supplementary PDF:
  `74d9c94eb949dcea7c365bcbcbf19deaa4ac972fc3f585eb064a3e9ce1661521`;
- tagged source ZIP:
  `c02c4a9309dddef5b1095daf879be0c3d91251ff22781c4e1d3f1aaddd927023`.

The corresponding author explicitly approved pushing, tagging, creating, and
publishing the v0.7 release. The final artifact-identity check passed on
29 July 2026.

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

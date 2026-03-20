# Final Release Checklist

This repository is already functionally validated. The remaining work is release-quality closure rather than core debugging.

## A. Repository contents

- [ ] `git status` is clean
- [ ] only canonical result artifacts are versioned
- [ ] no local sanity artifacts remain tracked
- [ ] no processed EMBER arrays remain tracked
- [ ] no extracted/raw EMBER data remains tracked
- [ ] no memmaps or pipeline-validation internals remain tracked

## B. Documentation

- [ ] `README.md` matches the final repository layout
- [ ] `docs/VALIDATION_STATUS.md` reflects the completed validation state
- [ ] `docs/REPRODUCIBILITY_NOTES.md` reflects canonical versioned outputs and exclusions
- [ ] `docs/FINAL_RELEASE_CHECKLIST.md` reflects release closure rather than pending debugging
- [ ] `CITATION.cff` is correct
- [ ] `LICENSE` is correct

## C. Environment

- [ ] `requirements.txt` is final
- [ ] `requirements-compat.txt` is final
- [ ] `environment.yml` is final
- [ ] optional: `environment.lock.yml` exported from the validated machine
- [ ] optional: `requirements.lock.txt` exported from the validated machine

## D. Continuous integration

- [ ] GitHub Actions workflow exists
- [ ] smoke test runs in CI
- [ ] summary-table generation runs in CI
- [ ] CI passes on the default branch

## E. Result artifacts

- [ ] `results/aggregated/` contains the canonical aggregated outputs
- [ ] `results/tables/` contains the canonical final tables
- [ ] duplicated intermediate result trees are excluded from version control

## F. Release metadata

- [ ] final repository URL confirmed
- [ ] version tag `v0.1.0` created
- [ ] GitHub Release created
- [ ] optional DOI / Zenodo integration completed
- [ ] optional final article citation added after acceptance

## G. Final publication decision

The repository is ready for public release when:

1. the tracked artifact set is clean,
2. the docs reflect the real validated state,
3. CI passes,
4. metadata is final,
5. the release tag is created.

At that point, the remaining improvements are optional polish rather than blockers.

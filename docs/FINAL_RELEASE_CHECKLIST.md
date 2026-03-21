# Final release checklist

This checklist is intended for the **post-validation public release phase** of the repository. The core scientific pipeline, reduced end-to-end validation path, and public packaging baseline are already in place.

## 1. Licensing / ownership
Confirm that the shipped `BSD-3-Clause` license is acceptable for the actual rights holder.

If institutional ownership applies, update the copyright line in `LICENSE` and, if needed, align:

- `CITATION.cff`
- `pyproject.toml`
- repository hosting namespace / organization

See also `docs/LEGAL_RELEASE_NOTE.md`.

## 2. Dependency and environment hygiene
Confirm that the dependency files accurately reflect their intended roles:

- `environment.yml` — recommended Conda setup
- `requirements.txt` — pinned top-level pip environment
- `requirements-compat.txt` — looser compatibility envelope
- `environment.lock.yml` — more explicit Conda snapshot, if retained
- `requirements.lock.txt` — more explicit pip snapshot, if retained

If lockfiles are kept in the repository, they should be defensible as real archival or near-archival environment snapshots rather than placeholders.

## 3. Package and CI hygiene
Confirm that public CI remains green and that it validates:

- `pip install .`
- CLI smoke test
- reporting stage
- Python **3.11** and **3.12** support

If CI is green at release time, no further action is required here.

## 4. Public metadata
Confirm that the public-facing repository metadata is aligned:

- repository URL
- release tag
- archival DOI / Zenodo record
- `CITATION.cff`
- README citation block

If a newer public release supersedes the current archival DOI, update the citation metadata accordingly.

## 5. Sanity checks before publishing a new release
Run from repository root:

```bash
python scripts/smoke/smoke_test_cli.py
```

Inspect:

- `docs/smoke_test_report.json`

Optional but useful:

```bash
python scripts/reporting/make_summary_tables.py \
  --metrics results/aggregated/AGG_ROOT_mean_std_metrics__by_variant_seed_and_size.csv \
  --drops results/aggregated/AGG_ROOT_mean_std_drop__by_variant_seed_and_size.csv \
  --outdir results/tables_check \
  --topn 10 \
  --round 4
```

## 6. Release publication step
For a GitHub + Zenodo archival workflow:

1. commit final metadata / documentation changes,
2. push to the public repository,
3. create the GitHub release,
4. wait for Zenodo ingestion,
5. verify the DOI record,
6. update the default branch citation metadata if needed.

## 7. What is optional rather than blocking
The following are useful improvements, but they are **not** blockers for a valid public research release:

- stricter future lockfile regeneration,
- additional GitHub templates (`CONTRIBUTING.md`, issue templates, etc.),
- cosmetic README improvements,
- future deprecation-warning cleanup.

## Practical release conclusion
If licensing is confirmed, CI is green, citation metadata is aligned, and the release archive is publicly resolvable, the repository should be considered **ready for public release and citation**.

# Artifact map

This map distinguishes executable source, costly frozen inputs, derived outputs,
and publication files. Raw source datasets are not redistributed.

| Component | Location | Scientific function | Regenerable | Reproducer | Preserve frozen |
|---|---|---|---|---|---|
| Protocol and implementation | `src/`, `scripts/` | Constructs splits, kernels, analyses, tables, and figures | Yes, given source data and compute | `scripts/reproduce_v4.py` through `scripts/reproduce_v11.py` | Source is versioned |
| Security-study run records | `results/ember_shift/`, `results/netflow/`, `results/netflow_bandwidth_sweep/` | Costly per-run inputs for the controlled comparisons | Yes, but computationally expensive | `scripts/reproduce_v4.py`, `scripts/reproduce_v8.py` | Yes |
| Analysis generations | `results/v4/` through `results/v10/` | Confirmatory, external, finite-shot, circuit, certificate, and prospective outputs | Derived from frozen inputs or locked predictions | Matching `scripts/reproduce_v*.py` | Yes |
| Prospective evidence chain | `docs/GATE2_PROSPECTIVE_REPLICATION_SPEC_V10.md`, `results/v10/gate2_prospective/` | Specification, prediction locks, separated labels, opening record, failures, manifests, and audit outputs | No retrospective replacement | `scripts/reproduce_v10.py` | Yes, byte-for-byte |
| Frozen contracts | `docs/*SPEC*.md`, `docs/*FREEZE*.json` | Records prespecified estimands and validation contracts | No | Checked by validators and tests | Yes, byte-for-byte |
| Publication | `manuscript/` | Main text, Supplementary Information, bibliography, figures, and compiled PDFs | Yes from included sources | LaTeX build plus manuscript validators | Sources and release PDFs |
| Release metadata | `README.md`, `CITATION.cff`, `docs/RELEASE_NOTES_*.md` | Navigation, citation, version identity, and release history | Yes | Release workflow | Version-specific |

For ordinary verification, start with `python -m pytest -q`,
`python scripts/smoke/smoke_test_cli.py`, and
`python scripts/reproduce_v11.py --stage all`.

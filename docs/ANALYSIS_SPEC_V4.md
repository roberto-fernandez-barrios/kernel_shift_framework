# ANALYSIS_SPEC_V4 — Frozen confirmatory analysis specification (v0.4 revision)

**Status: FROZEN on 2026-07-17, before any decisive v4 analysis was run.**
Frozen at commit `4f6d79a` (branch `paper-v4-methodological-audit`, forked from the
v0.3.0 release state). Any analysis added after this date is EXPLORATORY and must be
labeled as such in every table, figure, and sentence that uses it.

This spec implements `PLAN.md` (external methodological review response, 21 sections)
as amended by the author's binding constraints of 2026-07-17 (Section 2 below).
The working rule: **no conclusion may be written stronger than what the confirmatory
analyses defined here support, and the headline framing is chosen only after the
confirmatory results exist** (Section 8).

---

## 1. Primary confirmatory endpoint

> The primary confirmatory comparison is the **budget-matched P1' SVC out-of-distribution
> balanced-accuracy difference** between the quantum and extended-classical kernel
> families:
>
> `delta = BAcc_OOD(quantum) - BAcc_OOD(classical_extended)`
>
> where each family's configuration is selected on **ID-validation** data (never OOD,
> never ID-test), each kernel configuration carries its own regularization C chosen by
> **internal cross-validation on the training split only**, and both families search
> **equal candidate budgets** of kernel geometries.

- `delta > 0` favors the quantum family; `delta < 0` favors the classical family.
- **SVC is the primary classifier.** GPC is a secondary robustness and probabilistic-
  uncertainty analysis (its hyperparameters are fixed and declared; no symmetric
  regularization sweep exists for it, so it never carries the family-level claim).
- Reported per scenario-group: mean, median, conditional interval (Section 5),
  and the budget-sensitivity distribution (Section 4).
- The number of settings "won" and any isolated p-value are NOT endpoints.

## 2. Binding constraints (author, 2026-07-17) — verbatim, non-negotiable

1. **No global population p-value of any kind.** The 8 scenario-groups are not 8
   independent units; the global sign test over them is removed. Presentation:
   fixed case studies, per-case effects, a descriptive dataset-equal-weighted
   summary, between-case heterogeneity, and leave-one-dataset-out.
2. **Intervals over the 5 q-split seeds are conditional pipeline-realization
   uncertainty** (conditional on the benchmark pools and the design), NOT inference
   about a population of future datasets or shifts. Ibragimov–Müller cluster-t is
   retained as a conditional sensitivity; no population claim rests on it.
3. **GATE 1 primary = repeated equal-budget subsampling + performance-vs-budget
   curves.** The fixed matched60 design is a secondary sensitivity (its composition
   could be considered post hoc). Also report: expected top-k, family mean/median
   (selection-free quality), and area under the budget curve (AUC-B).
4. **C is not a candidate.** C is tuned per kernel configuration by internal CV using
   only the training split; the kernel configuration is then selected on
   ID-validation and evaluated on ID-test / OOD-test. The 60-vs-60 budget counts
   kernel GEOMETRIES; C introduces no additional winner's curse.
5. **ID-validation/ID-test split: not by even/odd indices.** Deterministic,
   class-stratified, based on stable sample IDs/hashes, with a published balance and
   overlap audit.
6. **The finite-shot analysis is a "finite-shot fidelity-estimation perturbation
   model", not a hardware simulation.** Symmetrize, set diagonal = 1, clip to [0,1],
   report metrics before AND after PSD projection; rectangular test-vs-train blocks
   receive binomial sampling + clipping only (no PSD projection).
7. **The preliminary pass over the legacy `id_test` selection validates software
   only.** It cannot decide the headline. The final framing is determined exclusively
   after: the new ID-validation/ID-test protocol, internal C tuning, the complete
   sweep, and equal budgets.

## 3. Protocols (final names; used everywhere including code and paper)

- **P1' — ID-validation deployment selection** (primary): per run and family, select
  the kernel configuration (at its internally CV-chosen C) with the best
  ID-validation balanced accuracy; report ID-test and OOD-test performance.
  No OOD labels are used at any point of selection.
- **P2 — Cross-realization OOD-supervised selection** (secondary): select using
  labeled OOD data from the other four q-split realizations; evaluate on the held-out
  realization. This protocol DOES use OOD labels (from realizations disjoint from the
  evaluation split — overlap audit published). It must never be called "honest",
  "no-OOD-label", or "deployment".
- **P3 — Same-test OOD oracle** (upper bound only): select and evaluate on the same
  OOD target. Never used for any deployable claim.

## 4. GATE 1 — budget-matched selection (confirmatory design)

- Candidate unit: kernel geometry = (kernel shape, length/angle scale, dimension).
  Pools (audited, deduplicated): classical_extended = 115, quantum = 60 for the five
  full-coverage scenario-groups; 35 / 20 for the three netflow `m2_centroid` groups
  until the v4 recompute closes the coverage gap (rows carry an explicit `budget`
  column and are never pooled across budgets).
- Primary analysis: repeated subsampling of the classical pool to the quantum budget
  (60, or 20 in reduced-coverage groups), B = 5000 resamples per scheme, applying P1'
  within the subsampled pool. Schemes: `uniform`; `kernel_stratified` (per-shape
  strata); `kernel_blocked` (12-of-23 shapes × all 5 dims — structural mirror of the
  quantum 12 maps × 5 dims; primary scheme).
- Budget curves: B ∈ {5, 10, 20, 30, 40, 50, 60} for BOTH families (symmetric
  subsampling; classical additionally at 115 as a descriptive extension), B = 2000
  resamples per point. Derived scalars per family/group: expected top-k for each k,
  family mean and median (selection-free), and normalized AUC-B over B ≤ 60.
- The resampling distribution is **budget sensitivity**, never a confidence interval.
- Secondary sensitivity: fixed `matched60` design (4 scale-bearing classical shapes ×
  scales {x0.3, x1, x3} × 5 dims) — reported separately, never the headline basis.

## 5. GATE 2 — effect estimation (no population inference)

- The 8 scenario-groups (EMBER×{m1,m2}; {UNSW-DoS, UNSW-Recon, ToN-IoT-Scanning} ×
  {m2_centroid, held-out attack-campaign shift}) are **fixed case studies**.
- Resampling/aggregation cluster: **q-split seed** (5 levels; ≤1.2% OOD sample overlap,
  audited). Master seed and size remain fixed strata inside clusters (master seeds
  share the OOD pool on EMBER m1 — 100% identical, audited; sizes are design factors).
- Per-group effect: nested means (size → master seed → model seed) within each q-split
  cluster; the 5 cluster means summarize the group.
- Interval: t-interval on the 5 cluster means (Ibragimov–Müller), reported as
  **conditional pipeline-realization uncertainty** — the exact wording
  "conditional on the benchmark pools and experimental design; not inference about a
  population of datasets or shifts" appears in every assumptions field and caption.
  BCa bootstrap (B = 9999) secondary; leave-one-q-split-out jackknife as diagnostic.
- Cross-group reporting: per-case effects, descriptive dataset-equal-weighted mean,
  between-case heterogeneity (range and SD of group effects), leave-one-dataset-out
  range. **No p-value in any row.**
- Every inferential row documents: `unit_of_resampling`, `n_independent_clusters`,
  `n_lower_level_obs`, `method`, `assumptions`.

## 6. Frozen seeds and parameters

| Item | Value |
|---|---|
| Budget resampling seed | `20260715` (numpy default_rng, spawned per scheme×model×group-branch×budget) |
| Resamples (headline schemes) | B = 5000 |
| Resamples (budget curves) | B = 2000 per point |
| ID-val/ID-test split | 50/50, class-stratified, ordered by SHA-256 of `"ksf-v4-idsplit::" + str(global_row_index)`; within each class stratum, sorted by hash digest, alternating assignment starting with `id_val` |
| Internal C CV | k = 5 stratified folds on train only; fold seed `20260717`; grid C ∈ {0.01, 0.1, 1, 10, 100}; select by mean fold balanced accuracy; ties → smaller C |
| GPC (secondary) | Laplace approximation, fixed hyperparameters as in v0.3 (declared in methods); no C analogue |
| Finite-shot subset | all scenario-groups, q1000 size only, model seed 42, all 5 q-split seeds; shots ∈ {128, 512, 2048, 8192}; perturbation seed `20260718`, one draw per (run, config, shots) |
| BCa bootstrap | B = 9999, seed `20260719` |

## 7. Computational fallback (declared before running)

If the q4000 quantum recompute (~74 min/run × 360 runs) cannot complete, the
confirmatory analysis restricts to q1000 + q2000 sizes with q4000 reported as a
robustness stratum from whatever fraction completes. This fallback is declared here,
before results exist, and may not be invoked selectively per result.

## 8. Framing decision rule (chosen ONLY after confirmatory results)

Following PLAN.md §18, on the confirmatory (post-recompute) GATE 1 + GATE 2 outputs:

- **Framing 1 — no robust family advantage**: budget-matched deltas centered near zero
  or direction inconsistent across scenario-groups / not robust to scheme choice.
- **Framing 2 — classical parity/advantage**: negative direction persists under equal
  budget in the large majority of resamples, in more than one dataset, robust to C,
  preprocessing, and clustering, not driven by a single kernel.
- **Framing 3 — regime-specific quantum advantage**: positive direction persists under
  all controls in specific scenario-groups, not universal.

In all three cases the central contribution remains: a controlled audit of how kernel
geometry and evaluation protocol determine conclusions about quantum kernels under
distribution shift.

Prohibited in public artifacts regardless of framing: any global population p-value;
"single mechanism explains"; "fully fair"; "calibrated uncertainty" (without an actual
calibration procedure); "natural drift" (use "held-out attack-campaign and
capture-partition shift"); "hardware simulation" (use "finite-shot fidelity-estimation
perturbation model"); "reversal" unless Framing 2 is selected by this rule;
"15 independent repetitions" (use "15 pipeline realizations").

## 9. Provenance

- v0.3.0 release (tag, GitHub release, Zenodo record) is immutable input. All v4
  outputs live under `results/v4/`. No v4 table or figure may read legacy analysis
  outputs (`results/honest_selection/`, `results/tables_v3/`, etc.); they may read the
  frozen per-run summary CSVs, whose SHA-256 hashes are recorded by
  `scripts/analysis/audit_v4.py` in `results/v4/audit/file_hashes.csv`.
- `scripts/analysis/hierarchical_stats.py` is legacy; its sign-flip permutation output
  (including `p = 2e-4`) is withdrawn and must not appear in any v4 artifact.

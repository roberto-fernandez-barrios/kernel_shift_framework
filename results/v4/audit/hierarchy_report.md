# v4 GATE 0 audit report

Frozen 2026-07-17T06:58:26+00:00 at `4f6d79a1e` (branch `paper-v4-methodological-audit`, base tag `v0.3.0`).

Hashed 2430 frozen summary CSVs (163 MB) across 4 roots.

## Candidate pools (kernel geometries, deduplicated)

```
                      group  classical_ext_gpc  classical_ext_svc  quantum_gpc  quantum_svc  budget_class
                   ember_m1              115.0              115.0         60.0         60.0   full_115v60
                   ember_m2              115.0              115.0         60.0         60.0   full_115v60
toniot_scanning_m2_centroid               35.0               35.0         20.0         20.0 reduced_35v20
toniot_scanning_natural_cur              115.0              115.0         60.0         60.0   full_115v60
       unsw_dos_m2_centroid               35.0               35.0         20.0         20.0 reduced_35v20
       unsw_dos_natural_cur              115.0              115.0         60.0         60.0   full_115v60
     unsw_recon_m2_centroid               35.0               35.0         20.0         20.0 reduced_35v20
     unsw_recon_natural_cur              115.0              115.0         60.0         60.0   full_115v60
```

## Hierarchy

```
      root  n_runs  n_groups                                                                                                                                          groups  n_ms  n_sizes  n_qs  n_model_seeds  balanced
     ember     270         2                                                                                                                               ember_m1;ember_m2     3        3     5              3      True
  ember_bw     270         2                                                                                                                               ember_m1;ember_m2     3        3     5              3      True
   netflow     810         6 toniot_scanning_m2_centroid;toniot_scanning_natural_cur;unsw_dos_m2_centroid;unsw_dos_natural_cur;unsw_recon_m2_centroid;unsw_recon_natural_cur     3        3     5              3      True
netflow_bw     405         3                                                                         toniot_scanning_natural_cur;unsw_dos_natural_cur;unsw_recon_natural_cur     3        3     5              3      True
```

## Sample-overlap findings (fraction of smaller set, MAX over pairs)

| case | axis | train | id_test | ood_test |
|---|---|---|---|---|
| ember_m1 | qsplit_seed | 0.004 | 0.008 | 0.012 |
| ember_m1 | master_seed | 0.002 | 0.002 | 1.000 |
| ember_m1 | size | 0.017 | 0.030 | 0.022 |
| ember_m2 | qsplit_seed | 0.004 | 0.008 | 0.012 |
| ember_m2 | master_seed | 0.004 | 0.000 | 0.004 |
| ember_m2 | size | 0.017 | 0.030 | 0.022 |
| unsw_dos_natural | qsplit_seed | 0.024 | 0.016 | 0.078 |
| unsw_dos_natural | master_seed | 0.007 | 0.006 | 0.036 |
| unsw_dos_natural | size | 1.000 | 0.032 | 0.252 |

## Reading

- q-split seeds are near-disjoint everywhere (max shared fraction ~0.08 on
  netflow OOD, ~0.01 on EMBER) -> defensible resampling clusters for
  CONDITIONAL pipeline-realization uncertainty (spec section 5).
- EMBER m1 ood_test is ~identical across master seeds (deterministic score
  tails): master seeds are NOT independent replicates there.
- Sizes: EMBER sizes are independently resampled (overlap ~ chance), but
  NETFLOW trains are NESTED across sizes (train overlap = 1.0 for at least
  one pair; OOD up to ~0.25). Either way, size is a fixed design factor and
  never a replicate -- the netflow nesting makes this mandatory, not optional.
- Consequence (spec sections 2.1-2.2): fixed case studies, no global
  population p-value; intervals conditional on benchmark pools.

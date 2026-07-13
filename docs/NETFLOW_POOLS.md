# Provenance of the network-flow reference/current pools

The netflow scenarios consume `data/processed/netflow/<scenario>/X.npy`, exported
from reference/current CSV pools. The pools are built by the two scripts in
`scripts/data/` (copied verbatim from the Paper-2 artifact so this repository is
self-contained end to end):

- `prepare_paper2_unsw_nb15_smoke.py` — UNSW-NB15 (scenarios: dos, reconnaissance)
- `prepare_paper2_ton_iot_q1_gate.py` — ToN-IoT (scenario: scanning)

## Exact construction semantics

**UNSW-NB15** (raw: official `UNSW_NB15_training-set.csv` / `UNSW_NB15_testing-set.csv`):
- `unsw_ref_no_<attack>` = the official **training** partition with all rows of the
  target attack category removed (benign + all other attack categories).
- `unsw_cur_<attack>` = the official **testing** partition restricted to Normal +
  the target attack category.
- Features: numeric-only flow features (39), dropping id/label/attack_cat and the
  categorical proto/service/state columns. Binary label: Normal -> BENIGN, else ATTACK.

**ToN-IoT** (raw: processed network flows):
- Normal rows are split disjointly between the reference and current pools (no
  shared benign rows). Non-target attack types go to the reference pool; the
  target attack type (scanning) appears only in the current pool.

## What the "natural drift" mechanism therefore measures

Training and ID pools are drawn from the reference regime; the OOD pool from the
current regime. The shift combines (a) an **emerging attack campaign**: the OOD
positive class is an attack category absent from training, and (b) for UNSW-NB15,
a **capture-partition change**: reference and current come from the two different
official partitions of the testbed capture. Class priors are balanced separately
in both pools by the split construction, so prevalence shift is deliberately
excluded from the measurement.

The manuscript names this mechanism accordingly (attack-campaign regime shift)
and does not claim temporal drift in the timestamp sense.

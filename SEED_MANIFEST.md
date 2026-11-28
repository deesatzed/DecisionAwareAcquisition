# Deterministic Random-Seed & Constants Provenance Manifest (`DecisionAwareAcquisition` `v2.2.0`)

All stochastic evaluations, two-phase stratified pilot draws, disjoint `50/50` split-pilot confirmations, patient-grouped repeated development splits, synthetic operating-envelope sweeps, exact design-based permutation checks, and `10,000`-replicate patient-cluster bootstraps in `DecisionAwareAcquisition` (`v2.2.0`, immutable computational commit `9ac6de0`) use explicit `numpy.random.default_rng(seed)` streams with zero unseeded global randomness.

## 1. Provenance Separation: Upstream Commit `8c8c4c8` vs. Post-`v1` Methodological Evaluations

Only the shared architectural and routing constants imported from the companion paper (`collision-zone-ed`, public GitHub commit `8c8c4c8` dated `2026-11-22T18:16:58-05:00` at `https://github.com/deesatzed/collision-zone-ed`) were frozen prior to constructing any of the three `DecisionAwareAcquisition` clinical cohorts (`2026-11-24/25`). Stage-B parameters were fixed during synthetic `E5` design, and all `v2.0`–`v2.2.0` additions (`ChannelCostSchedule`, `Common-Panel Bagged VOI`, `0.5x–3.0x` cost sweeps, `50/50` split-pilot confirmation, directional VOI ledgers, 4-way prior-shift taxonomy, and the post-hoc `2026-11-25` reference `MCID = 1.0/1,000`) are explicitly documented as post-`v1` methodological evaluations:

| Parameter | Symbol | Fixed Value | Provenance / Source |
| :--- | :---: | :---: | :--- |
| Pilot sampling fraction | $\rho_{\text{pilot}}$ | `0.20` | `collision-zone-ed` `8c8c4c8` (`Sensitivity: {0.15, 0.20, 0.30}`) |
| Two-phase Neyman logit-risk strata | $B_{\text{score}}$ | `8` (`16` cells $\times Y$) | `collision-zone-ed` `8c8c4c8` (`Sensitivity: {6, 8, 12}`) |
| Schervish / Buja Beta-KL concentration | $\kappa$ | `24.0` | `collision-zone-ed` `8c8c4c8` |
| Weighted-Pilot baseline stratum floor | $\text{floor}$ | `0.05` | `collision-zone-ed` `8c8c4c8` |
| Stage-2 KL-gate maximum concentration | $\kappa_{\max}$ | `20.0` | `collision-zone-ed` `8c8c4c8` |
| Stage-2 minimum collision-zone ESS | $M_{\min}$ | `240.0` | `collision-zone-ed` `8c8c4c8` |
| Pre-purchase VOI base-risk bins & quantiles | $(B_{\text{voi}}, K_q)$ | `(12, 33)` | `collision-zone-ed` `8c8c4c8` |
| Pre-purchase 2D dispersion quantile | $q_{\text{disp}}$ | `0.75` | `DecisionAwareAcquisition` `v2` (`Sensitivity: {0.65, 0.75, 0.85}`) |
| Stage-B OOF policy-screening FWER | $\alpha_{\text{pol}}$ | `0.10` | `DecisionAwareAcquisition` `v2` (`Sensitivity: {0.05, 0.10, 0.20}`) |
| Encounter overhead share (`ChannelCostSchedule`) | $\omega_0$ | `0.25` | `v2.2.0` post-review cost-accounting audit (`Sweep: 0.5x..3.0x`) |
| Reference bundle weight (Stanford / BWH & NTUH) | $w_{\text{ref}}$ | `2.00 / 1.20` | `v2.2.0` post-review cost-accounting audit (`Sweep: 0.5x..3.0x`) |
| Reference clinical utility threshold | $\texttt{MCID}$ | `1.0 / 1,000` | Post-hoc reference (`2026-11-25`, never claimed prespecified) |

---

## 2. Deterministic Seed Ledger Across Scripts & Tables

| Script / Module | Experiment / Table | Deterministic RNG Seed Formula | Replicates / Splits |
| :--- | :--- | :--- | :--- |
| `scripts/build_audited_cohorts_v2.py` | Patient-grouped `train/cal/test` splits (`89,395` Stanford `>2h` [`51,736` discharged subset], `51,158` BWH, `16,771` NTUH = `157,324` primary + `818` `[1]ROSC`) | `seed = 20261124` | `1` canonical patient-disjoint split per cohort |
| `scripts/run_paper3_v2_benchmark.py` | Canonical clinical tasks & `1,000`-rep / `3,600`-run synthetic operating-envelope suite | `canonical_seed = 42..49`; `1000 + rep` (`rep = 0..999`); `50000 + cell_idx * 100 + rep` (`rep = 0..49`) | `1,000` E5 reps + `72 x 50 = 3,600` envelope runs; `B = 10,000` cluster bootstraps |
| `scripts/run_paper3_v2_1_upgrades.py` | **Primary Two-Phase Discovery Benchmark** (`Weighted-CZ`, `IPW-CZ`, `UPCA Offset-CZ`, parametric Tao-Zeng-Lin offset MLE vs. IPW, `500`-seed independence audit) | `seed = 700 + rep` (`rep = 0..499` at $\rho=0.20$; `0..199` at $\rho \in \{0.15, 0.30\}$); perm `9000 + s` | `500` pilots ($\rho=0.20$) + `400` sensitivity pilots + `500` parametric + `500` perm |
| `scripts/run_paper3_v2_2_remediation.py` (`Block 1`) | Beta-KL / SRS HT identities, `XOR` stump counterexample, `400`-rep `E5` split-pilot & scheduled `B=2.0`, and `200`-pilot Stanford ICU `50/50` split vs. `4`-fold CF (with 2 Gaussian nulls) | Identities `seed = 19, 13`; `E5` `rep = 0..199` ($\sigma \in \{3.2, 1.6\}$); Stanford `8200 + rep` (`rep = 0..199`), McNemar `191, 192` | `400` synthetic runs + `200` Stanford pilots |
| `scripts/run_paper3_v2_2_remediation.py` (`Block 2`) | **$R = 10$ `Common-Panel Bagged VOI`** under `ChannelCostSchedule`, `Full 100%` (`All-M` & `Locked`), `Boundary Uncertainty`, channel-billed `Covert-CMI`/`Direct-VOI` (per-split & majority-mask), & `0.5x..3.0x` fee sweeps | `seed_r = 2100 + 100 * (canonical_seed % 10) + r_i` (`r_i = 0..9`); `B = 10,000` patient-cluster bootstraps `seed = 500 + canonical_seed + offset` | `R = 10` patient-grouped `train/cal` splits $\times 6$ clinical tasks; `5` cost multipliers $\times 6$ tasks |
| `scripts/run_paper3_v2_2_remediation.py` (`Block 3`) | Stanford ICU 4-threshold missed-case, pre-fee/fee-saving, & `Escalation-Only VOI` ledger + 4-way prior-shift taxonomy (`Unadjusted`, `Saerens EM`, `Disjoint Cal`, `Oracle Test`) | `seed = round(tau * 100000) + 901`; prior-shift cluster bootstraps `seed = 1401..1439` | `4 x 8 = 32` exchange-ledger rows + `4 x 4 = 16` prior-shift arms (`B = 10,000`) |
| `scripts/run_paper3_v2_2_remediation.py` (`Block 4`) | Multi-comparator subgroup equity, calibration, & `36` Cochran's $Q$ heterogeneity audits (`race`, `sex`, `age`, `payer`/`IRS-AGI` vs. `Full 100%`, `Stage-1`, `Global-CMI`) | `seed = 3000 + task_idx * 100 + attr_idx * 10 + grp_idx` | `3` cohorts $\times 4$ attributes (`36` Cochran's $Q$ tests; `B = 10,000`) |
| `scripts/verify_paper3_v2_2_tables.py` | Deterministic `< 1s` algebraic checkpoint-to-table verifier (tested in `tests/test_paper3_v2_2_table_verifier.py`, test `#48`) | Deterministic algebraic verification (`0` randomness) | `279` exact invariants across all `4` blocks |

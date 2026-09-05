# Live dolphinRust issue intake — 2026-08-24

> **Superseded #53 boundary (2026-08-28):** The model, frozen synthetic,
> resource, provenance, and identity parts of DR-053-TEMPORAL remain scheduled
> in dolphinRust. Its held-out field-evidence and independent temporal-review
> parts are deferred to `eo` after #53 closes. EO-GNSS-INPUT through
> EO-GNSS-SERVE and EO-487-SCIENCE retain that downstream ownership, including
> GroundPulse enablement; any publication is also EO-owned. DR-054-SPATIAL and
> its independent spatial review are unchanged. The original intake below is
> retained as a historical snapshot.

**Source:** GitHub issues, pull requests, and Actions refreshed 2026-08-24; `origin/main`
refreshed to `834f9a9ed8a3479afb4da670759af360db53b184`.

**Today plan:** `md/plans/demo-today-fixed-cube-eo-gnss-2026-08-24.md`.

**Full implementation plan:** `md/plans/fixed-cube-eo-gnss-implementation-2026-08-24.md`.

The earlier 45-minute constraint is withdrawn. The today plan optimizes for one credible real-data
demo before end of day; the full plan retains the production and scientific completion work.

**Execution boundary:** planning only. No issue mutation, branch update, merge, release, `eo`
submodule pin, deployment, replay, or external-data acquisition is authorized by this intake.

The prior uncertainty plan remains the detailed analytic acceptance source for #52, #54, and #53.
The master plan fixes the producer/consumer boundary: dolphinRust supplies reproducible fixed-cube
facts; `eo` owns GNSS acquisition, station selection, projection, anchoring, held-out validation,
persistence, serving, UI, and Fresno backfill.

| ID | Source | Current requirement | Disposition |
|---|---|---|---|
| DR-052-MERGE | [dolphinRust #52](https://github.com/morton-analytics-llc/dolphinRust/issues/52), [PR #56](https://github.com/morton-analytics-llc/dolphinRust/pull/56) | Reconcile PR #56 with current `main`, clarify its review receipt, prove the combined tree, merge manually, and verify post-merge `main` plus issue closure. | **Scheduled — T45-01 through T45-04.** Partial, pending current-tree CI, manual merge, and post-merge CI. |
| DR-FC-AXIS | Fixed-cube producer contract | Reject duplicate/non-increasing dates and equal-count/different-date burst stacks; bind `velocity.tif` to one exact common epoch set and the existing estimator identity. | **Scheduled — FC-01 and FC-02.** FC-01 is the first clean coding slice after #52 and fits 45 minutes. |
| DR-FC-MASK | Fixed-cube producer contract | Emit an explicit velocity-validity mask and bind velocity nodata, common-epoch support policy, estimator, gauge, and mask identity. | **Scheduled — FC-03.** Separate 45-minute coding slice after FC-01. |
| DR-FC-LOS | Fixed-cube producer contract | Require sourced run-specific signed LOS geometry for the science bundle; emit masked ENU ground-to-sensor components and prohibit the scalar incidence fallback. | **Scheduled — FC-04.** Separate producer slice; no GNSS logic belongs here. |
| DR-FC-REF | Fixed-cube producer contract | Persist authoritative burst IDs, ordered inputs/dates, temporal gauge, spatial reference pixel/coordinates, anchor-burst choice, bounded re-reference, grid, estimator, mask, and artifact hashes. | **Scheduled — FC-05 and FC-06.** This closes the dolphinRust producer boundary. |
| DR-054-SPATIAL | [dolphinRust #54](https://github.com/morton-analytics-llc/dolphinRust/issues/54) | Propagate bounded target/reference covariance, validate overlap/distance error and resources, and obtain independent scientific review. | **Scheduled — existing T54-01 through T54-07 after #52 and FC-AXIS/FC-REF.** It is split into bounded coding/research slices; no contract-only PR is a fix. |
| DR-053-TEMPORAL | [dolphinRust #53](https://github.com/morton-analytics-llc/dolphinRust/issues/53) | Calibrate irregular, heteroskedastic temporal slope inference with covariance-parameter uncertainty, preregistered coverage, independent field evidence, and review. | **Scheduled — existing T53-01 through T53-07 after #54.** `eo`, not dolphinRust, owns Fresno GNSS acquisition and station-A/station-B execution. |
| DR-057-ORBIT | [dolphinRust #57](https://github.com/morton-analytics-llc/dolphinRust/issues/57) | Expose per-stack POEORB/RESORB provenance without invalidating otherwise readable geometry metadata. | **Deferred — next standalone short coding slot, T57-01 through T57-05.** The data gate is satisfied by the committed real-granule crop at `/metadata/orbit/orbit_type`; re-enter when Ryan applies `backlog-ready`. Do not combine its merge commitment with DR-052-MERGE's 45-minute window. |
| EO-GNSS-INPUT | EO GNSS anchor | Version and persist the GNSS catalog/solution, select station A without outcome leakage and a distinct held-out station B on the finite COG, and record temporal/separation eligibility. | **Scheduled — EO-01 through EO-03.** EO-only. |
| EO-GNSS-PROJ | EO GNSS anchor | Project full ENU rates and covariance into the run's sourced signed LOS at each station; point-sample the exact fixed-cube COG and corrected uncertainty product. | **Scheduled — EO-04 and EO-05.** No vertical-only or nominal-angle shortcut. |
| EO-GNSS-ANCHOR | EO GNSS anchor | Estimate the LOS datum offset at station A, validate only at held-out station B, and keep datum estimation distinct from field validation. | **Scheduled — EO-06 and EO-07.** Missing independent uncertainty is `not_evaluable`, not pass. |
| EO-GNSS-PERSIST | EO GNSS anchor | Persist an immutable receipt binding catalog, stations, projections, COG/manifest hashes, samples, uncertainty, selection method, offset, residual, threshold, generation, and outcome. | **Scheduled — MIG-01 through MIG-03 and EO-08.** Forward-only migration; historical ties are not reinterpreted. |
| EO-GNSS-SERVE | EO GNSS anchor | Apply the persisted offset only through the correlated run lineage; promote serving only after the current-generation station-B and uncertainty gates pass. | **Scheduled — EO-09 through EO-11.** Includes API/report/UI disclosure. |
| EO-GNSS-DIRECTION | EO asset-relative GNSS view | Project the GNSS station's horizontal EN motion onto the station-to-nearest-versioned-asset axis and label toward/away only when its interval excludes zero. | **Scheduled — EO-12 and EO-13 after full ENU persistence.** It describes ground motion at the station relative to the asset, not movement of the asset. It is not a serving-promotion oracle. |
| EO-FRESNO-BACKFILL | Fresno production acceptance | Run one exact fixed-cube Fresno replay, execute station-A/B validation, persist the receipt, publish correlated artifacts, and prove DB/COG/PMTiles/API/UI lineage. | **Scheduled — EO-14 through EO-16 after producer release/pin, migrations, deploy, and scientific gates.** No historical mutation before those gates. |
| EO-487-SCIENCE | [eo #487](https://github.com/morton-analytics-llc/eo/issues/487#issuecomment-5387534874) | Before historical replay, provide temporal/spatial uncertainty, exact comparable lineage, then same-artifact COG, PMTiles, database, serving, API, and UI evidence. | **Scheduled — DR-054-SPATIAL, DR-053-TEMPORAL, EO-04 through EO-16.** A dolphinRust source merge alone cannot satisfy it. |
| EO-483-PROVENANCE | [eo #483](https://github.com/morton-analytics-llc/eo/issues/483) | Record orbit class, DEM identity/datum, corrections/model, and expose build identity. | **Deferred — eo #483.** DR-057-ORBIT supplies only the orbit-class producer; `eo` schema consumption and customer disclosure remain separate. |
| EO-475-WIRING | [eo #475](https://github.com/morton-analytics-llc/eo/issues/475), [#478](https://github.com/morton-analytics-llc/eo/issues/478), [#479](https://github.com/morton-analytics-llc/eo/issues/479) | Decide and wire existing SET, seasonal-model, loop-QC, and mask capabilities, then measure a known-AOI delta and correct claims. | **Out of scope here.** These are primarily `eo` config/network/output changes; no new dolphinRust uncertainty fix closes them. |

## Evidence affecting disposition

- PR #56 is open, non-draft, `CLEAN`/mergeable at `8eb413b3d111a7d6d983f81226c90251d4268c1f`.
  Its exact-head CI is green, but that check predates current `main` by one intake-only commit.
- PR #56 is 33 files and 17,770 added lines. A new exhaustive review does not fit 45 minutes;
  the time box supports a targeted acceptance-gate audit against the existing independent review,
  focused contracts, combined-tree CI, and post-merge `main` CI.
- PR #56 remains opt-in and writes `calibration_status=uncalibrated` plus
  `downstream_inference_status=blocked_pending_issue_54_and_53`. It does not unblock `eo`.
- `validation/make_geomprov_fixture.py` copies `orbit_type` from a real OPERA CSLC-S1 v1.1
  granule into `oracle/fixtures/geomprov_ci_cslc.h5`; the committed dataset is scalar `POEORB`
  and documents POEORB as precise and RESORB as restituted.
- `eo/main` is `07e77ff34257de67ba8a88f0f03d5305f054662e` and still pins dolphinRust
  `v1.5.0` / `8a63fe202cf7dc82161a4431aed8fe6d1a428f1c`.

## Coverage audit

All four live dolphinRust issues, all four fixed-cube producer requirements, the EO GNSS input,
projection, anchor, holdout, persistence, serving, asset-relative display, and Fresno backfill
have explicit dispositions in the master plan. EO-483 remains the orbit-provenance consumer;
EO-475/478/479 remain explicitly out of this GNSS/fixed-cube path. dolphinRust has no UI. Every
new EO backend result has a paired API/report/UI disclosure task.

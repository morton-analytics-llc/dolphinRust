# Velocity validation, v1.5.0, and atmospheric follow-up intake

**Source:** Ryan's 2026-08-17 directive covering dolphinRust #47, #44, v1.5.0,
the 2018 troposphere cohort, dolphinRust #41, and the related eo issues.
**Plan:** `md/plans/velocity-validation-release-atmosphere-2026-08-17.md`

| ID | Canonical requirement | Disposition |
|---|---|---|
| DR-047 | Isolate Earthdata token tests from the real repository `.env` while preserving environment-token-first lookup. Prove the exact 32-test local suite passes with `.env` present. | **Scheduled — T01** |
| DR-044-CONTRACT | Add a red synthetic scorer contract proving `velocity_seasonal` can change the sampled `velocity.tif` estimate without changing the displacement-series polyfit. | **Scheduled — T02** |
| DR-044-PAYLOAD | Keep the existing common-GNSS-epoch polyfit, add direct station-pixel `velocity.tif` sampling, report optional seasonal amplitude/phase per station, and identify every velocity estimator in the payload. | **Scheduled — T02** |
| DR-044-AB | Rescore the saved 2018 linear and seasonal runs without fetching or rerunning the cohort. Prove the raster residual changes from about -11.51 to -5.74 mm/yr while the independent polyfit remains about -262.01 mm/yr and seasonal amplitudes are about 25 mm. | **Scheduled — T03** |
| DR-REL-150 | Prepare and publish annotated release `v1.5.0` after #47/#44 merge: release the changelog, mark the breaking output changes, set Rust package metadata to 1.5.0, update release instructions, run all release checks, and create the GitHub Release. | **Scheduled — T04** |
| EO-PIN-150 | On a fresh branch from `eo/origin/main`, advance `vendor/dolphinRust` to the exact `v1.5.0` target, regenerate the worker lockfile, merge through green combined-tree CI, and distinguish that from production verification. | **Scheduled — T05** |
| EO-359 | Close the stale pin issue after the tagged bump. | **Out of scope — already closed on 2026-08-18 UTC; no second action** |
| EO-316 | Close after the pin. | **Out of scope — already closed as not needed as a separate tracker; eo #419 retains the fresh-run regression-residual provenance and dependent-confidence checks, with #440 as its restart dependency** |
| DR-TROPO-2018 | Write a no-fetch strategy that determines whether the 52 OPERA L4 files can be reduced before local landing and defines one integer transfer-budget gate before the cohort fetch. | **Scheduled — T06** |
| EO-238-ARCH | Resolve whether ERA5 belongs in dolphinRust or eo before scoping code. | **Scheduled — T07; live evidence shows eo #238 is already closed and superseded by eo #188's wrapper/upstream-only boundary** |
| DR-041 | Do not scope `era5_troposphere`; close dolphinRust #41 if the eo boundary excludes a local implementation. | **Scheduled — T07** |
| EO-419 | Recover CSLC staging/processing before production verification of the new pin. | **Out of scope — explicitly excluded from this dolphinRust plan; eo #440 is now an additional restart dependency** |

## Coverage audit

Every source item has one disposition above. Scheduled items map to T01-T07 in the
implementation plan. The two non-scheduled live items have named destinations and re-entry
conditions; no request is silently dropped.

# Velocity uncertainty producer contract intake

Source: GroundPulse Fresno demo science review, 2026-08-23.

| ID | Canonical requirement | Disposition |
|---|---|---|
| DRU-001 | Treat interferogram edge count, incidence-matrix rank, nominal network residual DOF, and misclosure only as network-solvability/unwrap diagnostics. Redundant interferograms sharing acquisitions are not independent statistical observations and cannot create an empirical posterior scale. This change does not claim to emit per-pixel valid-IFG count or network rank. | **Scheduled — T01-T03.** |
| DRU-002 | Compute a per-pixel IID-conditional temporal-fit slope component from the final corrected and spatially referenced per-date displacement series. Exclude the structural acquisition-0 gauge, require positive valid-date regression DOF and residual scale, and use unit relative precision. Do not derive inferential velocity uncertainty from stitched ministack CRLB or a diagonal redundant-IFG normal matrix. | **Scheduled — T01-T03.** |
| DRU-003 | Retain valid-date count, regression rank/DOF, residual RMS, IID slope SE, raw residual correlation, correlation-pair count, diagnostic inflation/effective N, cadence eligibility, and the exact temporal-covariance method/status. | **Scheduled — T01-T03.** |
| DRU-004 | Preserve default/off numerical behavior, bounded memory, physical units, correction-before-reference ordering, final-reference handling, mask propagation, and explicit IID/approximation/unavailable states. Any temporally corrected inferential SE requires explicit covariance propagation and preregistered coverage validation. | **Scheduled — T01-T04.** |
| DRU-005 | Relabel or suppress existing `empirical posterior`, global posterior DOF, and projected-posterior velocity claims so the parameter-covariance diagonal under an independent-IFG error model cannot be mistaken for calibrated evidence. | **Scheduled — T01-T04.** |
| DRU-006 | Propagate the covariance of compressed ministack references and cross-date terms before the stitched CRLB can be treated as global per-date relative variance. | **Deferred — destination: dolphinRust #52; re-enters after a bounded covariance design and two-ministack analytic fixture pass.** |
| DRU-007 | Select and validate any temporal-covariance slope estimator against preregistered Monte Carlo coverage and held-out GNSS evidence before emitting a corrected inferential sigma. | **Deferred — destination: dolphinRust #53; re-enters after an explicit covariance model and validation protocol are approved.** |
| DRU-008 | Make the uncertainty flag's point-estimator change explicit in the API, artifact provenance, changelog, and tests. Require a field canary comparison before GroundPulse enables the post-gauge unit-weighted estimator. | **Scheduled — T01-T04; consumer canary remains GP-DRU-001.** |
| DRU-009 | Propagate target/reference spatial covariance before the spatially referenced network diagonal can be treated as inferential variance. | **Deferred — destination: dolphinRust #54; re-enters after a bounded covariance design and overlap-aware analytic fixtures pass.** |
| GP-DRU-001 | Configure, persist, serve, and verify the released producer evidence in GroundPulse. | **Deferred — destination: `eo` plan `FDS-T04` through `FDS-T09`; re-enters after the engine PR is merged and a reviewed release is available.** |

## Coverage audit

DRU-001 through DRU-005 and DRU-008 are scheduled by T01-T04. DRU-006, DRU-007, and DRU-009 are deferred to #52, #53, and #54 with named re-entry gates. GP-DRU-001 is deferred to the named GroundPulse plan and release gate. dolphinRust has no UI; the paired API/UI work is owned by GroundPulse rather than invented here.

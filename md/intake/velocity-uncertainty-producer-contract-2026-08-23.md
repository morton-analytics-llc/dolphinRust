# Velocity uncertainty producer contract intake

Source: GroundPulse Fresno demo science review, 2026-08-23.

| ID | Canonical requirement | Disposition |
|---|---|---|
| DRU-001 | Treat valid interferogram count, inversion rank, network residual DOF, and misclosure only as network-solvability/unwrap diagnostics. Redundant interferograms sharing acquisitions are not independent statistical observations and cannot create an empirical posterior scale. | **Scheduled — T01-T03.** |
| DRU-002 | Compute a per-pixel temporal-fit slope component from the final spatially referenced per-date displacement series, the same estimator/design, positive valid-date regression DOF, and explicit reference-aware relative variance. Do not derive inferential velocity uncertainty from a diagonal redundant-IFG normal matrix. | **Scheduled — T01-T03.** |
| DRU-003 | Retain valid-date count, regression rank/DOF, residual RMS, IID slope SE, raw residual correlation, correlation-pair count, diagnostic inflation/effective N, cadence eligibility, and the exact temporal-covariance method/status. | **Scheduled — T01-T03.** |
| DRU-004 | Preserve default/off numerical behavior, bounded memory, physical units, final-reference handling, mask propagation, and honest IID/approximation/unavailable states. Any temporally corrected inferential SE requires explicit covariance propagation and preregistered coverage validation. | **Scheduled — T01-T04.** |
| DRU-005 | Relabel or suppress existing `empirical posterior`, global posterior DOF, and projected-posterior velocity claims so the diagonal-IFG approximation cannot be mistaken for calibrated or independent evidence. | **Scheduled — T01-T04.** |
| GP-DRU-001 | Configure, persist, serve, and verify the released producer evidence in GroundPulse. | **Deferred — destination: `eo` plan `FDS-T04` through `FDS-T09`; re-enters after the engine PR is merged and a reviewed release is available.** |

## Coverage audit

DRU-001 through DRU-005 are scheduled by T01-T04. GP-DRU-001 is deferred to the named GroundPulse plan and release gate. dolphinRust has no UI; the paired API/UI work is owned by GroundPulse rather than invented here.

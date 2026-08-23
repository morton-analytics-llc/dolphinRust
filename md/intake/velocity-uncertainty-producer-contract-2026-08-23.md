# Velocity uncertainty producer contract intake

Source: GroundPulse Fresno demo science review, 2026-08-23.

| ID | Canonical requirement | Disposition |
|---|---|---|
| DRU-001 | Classify empirical posterior scale per pixel from that pixel's valid interferogram rank and residual degrees of freedom; a globally overdetermined network must not make a locally exact pixel empirical. | **Scheduled — T01-T03.** |
| DRU-002 | Derive velocity slope uncertainty from the same per-pixel posterior displacement covariance produced by L2 inversion, without materializing a full covariance cube or substituting date-level CRLB precision. | **Scheduled — T01-T03.** |
| DRU-003 | Retain the velocity fit's lag-1 residual autocorrelation, inflation factor, effective sample size, valid-date count, and uncertainty eligibility as output evidence. | **Scheduled — T01-T03.** |
| DRU-004 | Preserve default/off numerical behavior, bounded memory, physical units, spatial-reference handling, mask propagation, and honest `crlb_bound`/unavailable states. Do not turn bandwidth-3 evidence into a calibration claim. | **Scheduled — T01-T04.** |
| GP-DRU-001 | Configure, persist, serve, and verify the released producer evidence in GroundPulse. | **Deferred — destination: `eo` plan `FDS-T04` through `FDS-T09`; re-enters after the engine PR is merged and a reviewed release is available.** |

## Coverage audit

DRU-001 through DRU-004 are scheduled by T01-T04. GP-DRU-001 is deferred to the named GroundPulse plan and release gate. dolphinRust has no UI; the paired API/UI work is owned by GroundPulse rather than invented here.

# dolphin-timeseries — SBAS inversion (reference: `dolphin/timeseries.py`)

## Domain
- Incidence matrix `A (n_ifgs × n_dates−1)` of ±1; solve `A φ = Δφ_unwrapped`.
- **L2 weighted least squares first** (faer), block-parallel; optional coherence weighting
  and `correlation_threshold` censoring. Linear velocity = trend of the series.
- L1/ADMM deferred until L2 is validated.

## Scope note
In scope. GroundPulse is adopting the Python dolphin, so dolphinRust replaces *dolphin's*
timeseries here — GP's older `gp-displacement` SBAS (Berardino 2002) becomes legacy. Match
dolphin's L1/L2 inversion as the drop-in target (L2 first, then L1/ADMM).

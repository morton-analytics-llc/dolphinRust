# dolphin-timeseries — SBAS inversion (reference: `dolphin/timeseries.py`)

## Domain
- Incidence matrix `A (n_ifgs × n_dates−1)` of ±1; solve `A φ = Δφ_unwrapped`.
- **L2 weighted least squares first** (faer), block-parallel; optional coherence weighting
  and `correlation_threshold` censoring. Linear velocity = trend of the series.
- L1/ADMM deferred until L2 is validated.
- **Velocity time-function terms (`velocity_model.rs`, issue #22)** — an annual sinusoid
  and/or configured Heaviside steps fitted *jointly* with the rate, so the reported rate is
  the rate rather than the rate plus whatever the seasonal cycle and step contributed over
  the sampled window. Forward divergence from dolphin (linear-only), config-gated by
  `timeseries_options.velocity_seasonal` / `velocity_step_dates`, **off by default**.
  `estimate_velocity_with_model` **panics on a linear model** on purpose: the linear fit has
  exactly one implementation (`inversion.rs`), the parity-critical one, and this module must
  never become a second one that drifts. Step epochs are inputs, never detected.

## Scope note
In scope. GroundPulse is adopting the Python dolphin, so dolphinRust replaces *dolphin's*
timeseries here — GP's older `gp-displacement` SBAS (Berardino 2002) becomes legacy. Match
dolphin's L1/L2 inversion as the drop-in target (L2 first, then L1/ADMM).

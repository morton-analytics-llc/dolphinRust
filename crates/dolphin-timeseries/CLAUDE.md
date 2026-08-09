# dolphin-timeseries — SBAS inversion (reference: `dolphin/timeseries.py`)

## Domain
- Incidence matrix `A (n_ifgs × n_dates−1)` of ±1; solve `A φ = Δφ_unwrapped`.
- **L2 weighted least squares first** (faer), block-parallel; optional coherence weighting
  and `correlation_threshold` censoring. Linear velocity = trend of the series.
- L1/ADMM deferred until L2 is validated.
- **Post-unwrap loop-closure QC (`loop_closure.rs`, issue #24)** — closes every triangle in
  the *unwrapped* network and masks pixels whose loops miss closure by >½ cycle, before the
  solve. Distinct from `dolphin-phaselink`'s closure phase, which is computed on the
  coherence matrix and bounded by `.arg()` to `(−π, π]`: a 2π unwrap error wraps to zero
  there and cannot be seen. **No-op on a single-reference network** (no loops), so it needs
  `max_bandwidth`/`max_temporal_baseline` — the same over-determined network #36 needs for
  an empirically scaled posterior. Off by default. Never masks a pixel with no evaluable
  loop: positive evidence only.

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

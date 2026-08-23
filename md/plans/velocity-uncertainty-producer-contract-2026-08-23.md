# Velocity uncertainty producer contract plan

Date: 2026-08-23

Intake: `md/intake/velocity-uncertainty-producer-contract-2026-08-23.md`

## Objective

Make dolphinRust emit enough per-pixel evidence for GroundPulse to distinguish an IID temporal-fit component, a reviewed temporal-covariance result, a diagnostic approximation, and an unavailable value. This is not a total-uncertainty or calibration claim.

## Selected contract

- L2 inversion reports valid observation count, model rank, algebraic residual DOF, and misclosure only as network/unwrap diagnostics. Interferograms sharing an acquisition have correlated errors; diagonal IFG weights and `(A'WA)^-1` cannot establish an empirical posterior or inferential velocity sigma.
- Velocity evidence comes from one estimator over the final corrected and spatially referenced per-date displacement series. Per-pixel regression evidence includes valid-date count, model rank, `n_valid - rank` DOF, residual RMS, and the IID residual slope SE from `s^2 (X'WX)^-1`, where `s^2 = sum(w_i e_i^2) / dof` and weights are relative rather than an independent calibration claim.
- Reference-aware relative date variance includes the selected reference pixel under the stated independent-spatial-pixel approximation. The reference pixel remains identically zero and abstains.
- Raw lag-1 residual correlation, correlation-pair count, diagnostic inflation, and effective N remain visible, but the scalar effective-N shortcut is not an inferential slope SE. It is unavailable for fewer than four valid dates or irregular/noncontiguous cadence and clamps diagnostic effective N to `[1, n_valid]`.
- A temporally corrected inferential slope SE is eligible only if an explicit time-covariance model is propagated through the fitted design and passes preregistered synthetic coverage tests for cadence, heteroskedasticity, missing dates, reference noise, and residual correlation. Otherwise only the IID component and diagnostic fields are emitted.
- Existing diagonal-IFG posterior products are renamed as approximations or withheld from inferential metadata. Global network DOF never labels every pixel empirical.
- Existing default configurations and rate estimates remain numerically unchanged when uncertainty output is disabled.
- Bandwidth 3 is a deployment candidate, not a library default. Its field coverage remains under-calibrated and must be benchmarked and independently validated by the consumer.

## Test contracts

| Contract | Red proof | Green requirement |
|---|---|---|
| C01 | Adding redundant IFG edges currently reduces diagonal normal-matrix variance and creates nominal residual DOF. | The same referenced per-date series has identical temporal-fit evidence regardless of redundant edge count; edge count/rank/DOF remain diagnostic only. |
| C02 | The current velocity sigma mixes CRLB weighting with a residual floor and does not state regression DOF. | A closed-form IID fixture matches `s^2 (X'WX)^-1` without the `max(1)` floor and retains valid-date count, rank, DOF, and residual RMS. |
| C03 | Current corrected output retains only sigma/inflation and treats any three adjacent retained dates as a valid AR(1) series. | Raw rho, pair count, diagnostic inflation/effective N, valid N, cadence state, and status match analytic fixtures; fewer than four or irregular/noncontiguous dates fail closed; effective N stays in `[1,n]`; velocity is bit-identical. |
| C04 | Spatially referenced displacement is fit with target-only CRLB precision. | A two-pixel fixture combines target and reference relative date variance before fitting; the reference output abstains and the target matches the independent-spatial-pixel contract. |
| C05 | Missing bounds or one global network DOF can still produce inferential metadata. | Per-pixel temporal status is IID-only, reviewed-covariance, diagnostic-only, or unavailable; missing reference/date evidence fails closed while displacement remains available. |
| C06 | Uncertainty-disabled output is the current baseline. | Default/off displacement, velocity, masks, and serialized outputs remain bit-identical. |
| C07 | Current metadata calls diagonal-IFG covariance an empirical/full posterior. | Outputs and docs identify it as a correlated-IFG diagonal approximation or omit it; it never authorizes velocity uncertainty. |
| C08 | A temporal-correlation method can pass analytic algebra while missing nominal coverage. | Deterministic Monte Carlo fixtures across the preregistered scenarios meet the named coverage tolerance before a corrected SE becomes eligible. |

## Task manifest

### T01 — red analytic contracts

**Ownership:** focused tests in `crates/dolphin-timeseries` and `crates/dolphin-workflows`; no production edits.

Add C01-C08 with closed-form fixtures and record the intended red failures. Use analytic linear algebra for deterministic contracts and a seeded, bounded coverage harness only for C08.

### T02 — bounded temporal-fit evidence

**Ownership:** `crates/dolphin-timeseries/src/inversion.rs` and its tests.

Extend the velocity fit/output with valid-date count, regression rank/DOF, IID residual slope SE, raw correlation, pair count, cadence eligibility, diagnostic inflation/effective N, and explicit method/status. Keep network inversion diagnostics separate. If C08 selects a temporal covariance method, propagate that covariance through the exact fitted design without an `n_dates x n_dates x area` cube.

### T03 — workflow propagation and artifacts

**Ownership:** `crates/dolphin-workflows/src/displacement.rs` and focused workflow contracts.

Select the final target-local reference after deterministic corrections, build reference-aware relative date variance, then carry per-pixel status/evidence through whole, bounded, trimmed, masked, scaled, returned, and serialized paths. Raster metadata must describe component scope, cadence, and method and must not use one global network DOF to label every pixel. Preserve the independent-spatial-reference approximation explicitly.

### T04 — verification and release boundary

**Ownership:** `CHANGELOG.md`, `VALIDATION.md`, intake/plan receipts, and version metadata only after T01-T03 are green.

Run focused and workspace gates, the seeded coverage gate, and independent science review, then open one unmerged PR. A later release may use v1.6.0 because the output/API contract changes. Do not claim calibration, change the bandwidth default, bump GroundPulse, or run production from this engine PR.

## Verification

```text
cargo fmt --all -- --check
cargo test -p dolphin-timeseries
cargo test -p dolphin-workflows displacement
cargo test -p dolphin-workflows --test displacement_contract
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
python -m unittest discover -s validation/tests
git diff --check
```

## Completion boundary

Engine completion is a green, reviewed, unmerged PR with exact test evidence. It does not make GroundPulse scientifically valid by itself. GroundPulse must pin the release, choose and benchmark the overdetermined network, persist comparison/support identity, recompute the Fresno canary, and pass terminal/API/UI verification before backfill.

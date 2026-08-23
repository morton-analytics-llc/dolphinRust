# Velocity uncertainty producer contract plan

Date: 2026-08-23

Intake: `md/intake/velocity-uncertainty-producer-contract-2026-08-23.md`

## Objective

Make dolphinRust emit enough per-pixel evidence for GroundPulse to distinguish an empirical temporal-fit uncertainty component from a CRLB bound or unavailable value. This is not a total-uncertainty or calibration claim.

## Selected contract

- L2 inversion reports each pixel's valid observation count, model rank, residual DOF, and posterior scale class.
- Velocity and velocity sigma come from one linear estimator over the inverted displacement series and that pixel's posterior covariance. The implementation computes the needed slope variance while the full per-pixel covariance is in hand; it does not retain an `n_dates x n_dates x area` cube.
- Spatial referencing propagates the reference pixel's temporal covariance under the existing independent-pixel approximation before the velocity uncertainty is finalized. The reference pixel remains identically zero and has no movement claim.
- AR(1) correction records `rho`, inflation factor, effective sample size, and valid-date count. Negative `rho` retains the existing no-deflation rule.
- A pixel is `empirical` only with full-rank solve, positive residual DOF, finite positive posterior-derived slope sigma, and finite retained correlation evidence. Exact, rank-deficient, partial, or unbounded pixels fail closed.
- Existing default configurations and rate estimates remain numerically unchanged when uncertainty output is disabled.
- Bandwidth 3 is a deployment candidate, not a library default. Its field coverage remains under-calibrated and must be benchmarked and independently validated by the consumer.

## Test contracts

| Contract | Red proof | Green requirement |
|---|---|---|
| C01 | A globally overdetermined network with one pixel missing enough interferograms currently labels every output `empirical`. | Per-pixel DOF is zero for the reduced pixel and positive for the fully observed pixel; only the latter is empirical. |
| C02 | A small analytic covariance with off-diagonal terms currently cannot produce a workflow velocity sigma. | The reported slope variance equals the closed-form linear propagation from the full posterior covariance and differs from the diagonal/CRLB substitute. |
| C03 | Current corrected output retains only sigma/inflation. | `rho`, inflation, effective N, valid N, and status layers match an analytic AR(1) fixture; velocity is bit-identical with correction on/off. |
| C04 | Spatial re-referencing currently updates displacement variance diagonals only after velocity fitting. | A two-pixel fixture propagates the reference covariance before slope uncertainty; the reference output abstains and the target matches the independent-pixel sum. |
| C05 | Missing bounds or locally zero DOF can still inherit global empirical raster tags. | Per-pixel status is `crlb_bound` or unavailable, and all inferential sigma/effective-N fields are null/NaN while displacement remains available. |
| C06 | Uncertainty-disabled output is the current baseline. | Default/off displacement, velocity, masks, and serialized outputs remain bit-identical. |

## Task manifest

### T01 — red analytic contracts

**Ownership:** focused tests in `crates/dolphin-timeseries` and `crates/dolphin-workflows`; no production edits.

Add C01-C06 with closed-form fixtures and record the intended red failures. Avoid a large external oracle: the required answers are analytic linear algebra and deterministic status transitions.

### T02 — bounded inversion and velocity evidence

**Ownership:** `crates/dolphin-timeseries/src/inversion.rs` and its tests.

Extend the per-pixel L2 solution/output with valid count, rank/DOF, covariance-derived slope uncertainty, and AR(1) evidence. Compute the slope functional from the full covariance inside the existing per-pixel solve. Keep the workspace output bounded to 2-D evidence layers plus the existing covariance diagonal cube.

### T03 — workflow propagation and artifacts

**Ownership:** `crates/dolphin-workflows/src/displacement.rs` and focused workflow contracts.

Carry the per-pixel status/evidence through whole, bounded, referenced, trimmed, masked, scaled, returned, and serialized paths. Raster metadata must describe component scope and must not use one global DOF to label every pixel. Preserve the existing independent-spatial-reference approximation explicitly.

### T04 — verification and release boundary

**Ownership:** `CHANGELOG.md`, `VALIDATION.md`, intake/plan receipts, and version metadata only after T01-T03 are green.

Run focused and workspace gates, independently review the science diff, and open one unmerged PR. A later release may use v1.6.0 because the output/API contract changes. Do not claim calibration, change the bandwidth default, bump GroundPulse, or run production from this engine PR.

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

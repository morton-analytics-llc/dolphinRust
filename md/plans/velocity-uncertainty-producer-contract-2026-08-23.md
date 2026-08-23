# Velocity uncertainty producer contract plan

Date: 2026-08-23

Intake: `md/intake/velocity-uncertainty-producer-contract-2026-08-23.md`

## Objective

Make dolphinRust emit enough per-pixel evidence for GroundPulse to distinguish an IID-conditional temporal-fit component, non-inferential temporal diagnostics, a parameter-covariance diagonal under an independent-IFG error model, and an unavailable value. This is not a corrected temporal-covariance, total-uncertainty, or calibration claim.

## Selected contract

- L2 inversion retains misclosure and labels the configured network's nominal residual DOF only as network/unwrap diagnostics. Interferograms sharing an acquisition have correlated errors; diagonal IFG weights and `(A'WA)^-1` cannot establish an empirical posterior or inferential velocity sigma. Per-pixel valid-IFG count and network rank are not emitted by this change.
- Velocity evidence comes from one estimator over the final corrected and spatially referenced per-date displacement series. Per-pixel regression evidence includes valid-date count, model rank, `n_valid - rank` DOF, residual RMS, and the IID residual slope SE from `s^2 (X'WX)^-1`, where `s^2 = sum(w_i e_i^2) / dof` and weights are relative rather than an independent calibration claim.
- Deterministic corrections precede the final spatial reference. Acquisition 0 is a structural gauge and is excluded from the stochastic fit. The remaining finite dates use unit relative precision; the selected reference remains identically zero and abstains because its residual scale is zero.
- Enabling the uncertainty component changes the primary point estimator from the full reconstructed-series fit using stitched-CRLB relative precision with whole-pixel unit fallback to the post-gauge unit-weighted fit. Return and raster provenance name the precision policy. GroundPulse must compare both rate fields in a fresh Fresno canary before enabling the flag.
- Raw lag-1 residual correlation, correlation-pair count, diagnostic inflation, and effective N remain visible, but the scalar effective-N shortcut is not an inferential slope SE. It is unavailable for fewer than four valid dates or irregular/noncontiguous cadence and clamps diagnostic effective N to `[1, n_valid]`.
- The stitched CRLB is not global date variance because compressed ministack reference covariance and cross-date terms are omitted. #52 owns that producer contract. A temporally corrected inferential slope SE remains unavailable until #53 selects an explicit covariance model and passes preregistered synthetic and held-out validation. This change emits only the IID-conditional component and diagnostic fields.
- Existing posterior products are labeled as parameter-covariance diagonals under an independent-IFG error model or withheld from inferential metadata. Global network DOF never labels every pixel empirical.
- Spatial referencing adds target/reference marginal variances but omits their covariance and labels that omission. #54 owns bounded overlap-aware propagation; this change does not infer it.
- With deterministic corrections disabled, existing default/off configurations and rate estimates remain numerically unchanged. Correction-enabled runs intentionally change because corrections now precede spatial referencing.
- Bandwidth 3 is a deployment candidate, not a library default. It has not passed field-coverage validation and must be benchmarked and independently validated by the consumer.

## Test contracts

| Contract | Red proof | Green requirement |
|---|---|---|
| C01 | Adding redundant IFG edges currently reduces diagonal normal-matrix variance and creates nominal residual DOF. | The same referenced per-date series has identical temporal-fit evidence regardless of redundant edge count; edge count/rank/DOF remain diagnostic only. |
| C02 | The current velocity sigma mixes CRLB weighting with a residual floor and does not state regression DOF. | A closed-form IID fixture matches `s^2 (X'WX)^-1` without the `max(1)` floor and retains valid-date count, rank, DOF, and residual RMS. |
| C03 | Current corrected output retains only sigma/inflation and treats any three adjacent retained dates as a valid AR(1) series. | Raw rho, pair count, diagnostic inflation/effective N, valid N, cadence state, and status match analytic fixtures; fewer than four or irregular/noncontiguous dates fail closed; effective N stays in `[1,n]`; velocity is bit-identical. |
| C04 | Corrections can be applied after reference and the stitched CRLB can enter the velocity fit as if it were global date variance. | A two-pixel fixture proves deterministic correction-before-reference ordering, exact final-reference zero, acquisition-0 gauge exclusion, target invariance to CRLB values, and reference abstention. |
| C05 | Missing reference/date evidence or one global network DOF can still produce inferential metadata. | An uncertainty-enabled workflow rejects a missing or non-exact final reference. Missing post-gauge dates are excluded from the IID fit; they set cadence to `missing` and disable lag-1 diagnostics but do not automatically suppress a full-rank, positive-DOF IID component. Network DOF stays diagnostic. |
| C06 | Uncertainty-disabled output is the current baseline. | With corrections disabled, default/off displacement, velocity, masks, and serialized outputs remain bit-identical. Correction-enabled fixtures follow the correction-before-reference contract. |
| C07 | Current metadata calls the network covariance an empirical/full posterior. | Outputs and docs identify it as a parameter-covariance diagonal under an independent-IFG error model or omit it; it never authorizes velocity uncertainty. |
| C08 | A temporal-correlation method can pass analytic algebra while missing nominal coverage. | **Deferred to #53.** No corrected SE exists until deterministic Monte Carlo fixtures and held-out evidence meet a preregistered coverage gate. |
| C09 | Enabling uncertainty silently changes the served point rate and artifacts omit units/reference-covariance assumptions. | API and COG provenance name the estimator; an on/off fixture proves rates can differ; uncertainty COGs declare physical units and omitted target/reference covariance. Spatial propagation remains deferred to #54 and the consumer canary remains blocked under GP-DRU-001. |

## Task manifest

### T01 — red analytic contracts

**Ownership:** focused tests in `crates/dolphin-timeseries` and `crates/dolphin-workflows`; no production edits.

Add C01-C07 and C09 with closed-form fixtures and record the intended red failures. C08 remains a fail-closed acceptance contract in #53; this change must not add a corrected estimator.

### T02 — bounded temporal-fit evidence

**Ownership:** `crates/dolphin-timeseries/src/inversion.rs` and its tests.

Extend the velocity fit/output with valid-date count, regression rank/DOF, IID residual slope SE, raw correlation, pair count, cadence eligibility, diagnostic inflation/effective N, and explicit method/status. Keep network inversion diagnostics separate. Do not implement a temporal-covariance correction in this task.

### T03 — workflow propagation and artifacts

**Ownership:** `crates/dolphin-workflows/src/displacement.rs` and focused workflow contracts.

Apply deterministic corrections before selecting/applying the final target-local reference, exclude invalid reference candidates and the structural gauge, and fit finite post-gauge dates with unit relative precision. Carry point-estimator identity and per-pixel status/evidence through whole, bounded, trimmed, masked, scaled, returned, and serialized paths. Raster metadata must describe physical units, component scope, gauge, temporal-covariance status, spatial-reference covariance omission, and calibration status and must not use one global network DOF to label every pixel.

### T04 — verification and release boundary

**Ownership:** `CHANGELOG.md`, `VALIDATION.md`, intake/plan receipts, and version metadata only after T01-T03 are green.

Run focused and workspace gates plus independent science review, then open one unmerged PR. A later release may use v1.6.0 because the output/API contract changes. Do not claim calibration, change the bandwidth default, resolve #52/#53 implicitly, bump GroundPulse, or run production from this engine PR.

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

## Intake coverage

- DRU-001, DRU-002, DRU-003, DRU-004, DRU-005, and DRU-008 are scheduled by T01 through T04.
- DRU-006 is deferred to dolphinRust #52 and re-enters after its bounded covariance design and
  two-ministack analytic fixture pass.
- DRU-007 is deferred to dolphinRust #53 and re-enters after its covariance model and validation
  protocol are approved.
- DRU-009 is deferred to dolphinRust #54 and re-enters after its bounded spatial-covariance design
  and overlap-aware analytic fixtures pass.
- GP-DRU-001 is deferred to the GroundPulse FDS-T04 through FDS-T09 release, serving, and demo
  gates after a reviewed engine release exists.

## Completion boundary

Engine completion is a green, reviewed, unmerged PR with exact test evidence. It does not make GroundPulse scientifically valid by itself. GroundPulse must pin the release, preserve the uncalibrated/component-only status, persist comparison/support identity, recompute the Fresno canary, and pass terminal/API/UI verification before backfill. #52/#53 remain separate gates for any future corrected inferential claim.

# Changelog

All notable changes to dolphinRust are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **`PhaseAngleLinearization`'s EMI inverse is now unrepresentable instead of `expect`ed**
  (issue #98). The prepared branch state folds the EMI regularized-gamma inverse into the
  `Emi` variant that owns it, replacing a detached `branch: FixedEstimatorBranch` +
  `gamma_inverse: Option<Mat<f64>>` pair whose EMI-implies-`Some` invariant held only by
  convention. `apply` matches the variant directly; no `Option` to unwrap remains on the
  `prepare`/`apply` path. Behavior-preserving — `phase_angle_jvp` and the estimator JVP
  contract tests (error precedence for `ZeroMagnitudeBranch`, `ThresholdBoundary`,
  `EigenvalueTie`, `VanishingReference`) pass unchanged.

## [v1.6.0] — 2026-08-30

### Breaking changes
- **Velocity uncertainty is now an uncalibrated IID-conditional component.**
  `write_velocity_uncertainty` fits the finite post-gauge dates from the final corrected,
  spatially referenced displacement series with unit relative precision. `velocity_sigma`
  no longer consumes the stitched CRLB or applies the scalar AR(1) effective-N multiplier.
  `correct_velocity_temporal_correlation: true` is rejected; the field remains readable only
  for YAML compatibility. `write_velocity_uncertainty: true` currently supports only the
  linear model and rejects seasonal or step terms. The public `VelocityOutputNeff` type and
  `estimate_velocity_with_uncertainty_neff` function are replaced by
  `VelocityDiagnosticsOutput` and `estimate_velocity_with_diagnostics`. Existing consumers
  must treat the component as unavailable unless its per-pixel status is `iid_conditional`.
  Enabling the flag also changes the point velocity from the full reconstructed-series fit using
  stitched-CRLB relative precision with whole-pixel unit fallback to the unit-weighted post-gauge
  fit, so served rates can change.
  `DisplacementOutput.velocity_estimator` and `VELOCITY_ESTIMATOR` metadata expose the exact
  path. Consumers must compare both estimators in a field canary before enabling the flag.
- **Modeled but unsupported dolphin config values now fail before input I/O** (issue #50).
  The public config tree has an exhaustive `Consumed` / `Conditional` /
  `CompatibilityOnly` registry. Compatibility-only fields still deserialize and round-trip,
  but a non-default value returns a path-specific config error instead of being ignored.
  Unsupported ICU/PHASS/SPURT/Whirlwind unwrap methods, invalid SNAPHU/Tophu option strings,
  and unbounded `output_options.epsg` also fail instead of falling through or claiming an
  unimplemented CRS fallback. GroundPulse's checked-in real-dolphin YAML currently sets
  `worker_settings.threads_per_worker: 6` and must be normalized before this engine is pinned;
  its programmatic configs retain supported defaults.

### Added
- **Orbit ephemeris provenance.** Geometry provenance now reads the sourced
  `/metadata/orbit/orbit_type` field, normalizing `POEORB`/`RESORB` to `precise`/`restituted`
  while keeping missing, unknown, and mixed values explicit and non-fatal to other geometry
  metadata. The artifact schema is `dolphinrust-geometry-provenance/4`; prior `/2` and `/3`
  artifacts remain deserializable.
- **A bounded sequential source-covariance replay operator is available behind an opt-in flag**
  (issue #52). `phase_linking.write_covariance_operator` streams fixed-branch phase-link and
  compression replay blocks into `phase_covariance_operator.h5`; a SHA-256-bound JSON manifest is
  committed last. Same-pixel temporal covariance is contracted from verified primitive-source
  factors without allocating a date-by-date-by-area cube, and acquisition 0 remains an exact zero
  gauge row and column. The initial scope is full-batch CPU/f64, Rect support, `AlwaysFirst`, and
  output reference 0. The CLI capture has no source-factor model and records
  `source_model_unavailable`; the low-level API requires a matching external resolver. Artifacts
  bind exact raw-source and numeric-factor receipts, and replay preflight reserves sparse-tree,
  support-vector, and padded estimator workspace before source I/O. Artifacts
  remain `uncalibrated` and `blocked_pending_issue_54_and_53`, and GroundPulse, resumable, spatial-
  reference, and velocity consumers remain disconnected.
- **Per-pixel temporal-fit support and diagnostics.** Velocity output now retains valid-date
  count, regression rank/DOF, uncertainty status, cadence status, raw lag-1 residual
  correlation, pair count, diagnostic-only inflation, and diagnostic-only effective sample
  size. The same fields are written as co-registered rasters. Metadata states
  `TEMPORAL_GAUGE=acquisition_0_excluded`, `TEMPORAL_COVARIANCE=not_modeled`, and
  `CALIBRATION_STATUS=uncalibrated_component`. Exact-linear fits report the point estimate but
  abstain on sigma because the residual scale is zero.
- **Uncertainty COGs now carry physical-unit and spatial-reference metadata.**
  `velocity_sigma.tif` declares `m/yr` or `rad/yr`; `displacement_variance_NN.tif` declares
  `m^2` or `rad^2` and states that target/reference covariance is not modeled.
- **Per-burst layover/shadow masks now enter before covariance and phase linking** (issue
  #50). `layover_shadow_mask_files` accepts one single-band native-grid GTiff per active burst, maps
  OPERA masks by burst ID independent of list order, and rejects missing, duplicate, extra,
  unparseable, misaligned, partially covered, or changed incremental inputs. Zero, raster
  nodata, GDAL-invalid pixels, and non-finite values are invalid; every finite nonzero pixel
  is valid. A stride cell is invalid only when all of its native pixels are invalid, and that
  validity reaches every final output layer. Resumable identity binds every mask backing file
  reported by GDAL. The later unwrap `mask_file` remains independent.
  GroundPulse does not yet extract or populate these masks; caller wiring requires a separate
  released-engine integration.

### Fixed
- **Deterministic corrections now precede the final spatial reference.** Atmospheric and tide
  corrections are applied before subtracting the selected reference history. Automatic whole-
  frame and bounded selection excludes mask-invalid or non-finite displacement pixels, and the
  bounded path reselects a target-local reference and refits through the same velocity fitting
  path. The emitted CRLB is labeled as a changing-reference ministack diagnostic. Displacement
  variance is labeled as a parameter-covariance diagonal under an independent-IFG error model,
  which is uncalibrated because the interferograms share acquisitions. Full stitched-reference,
  corrected temporal, and spatial-reference covariance are not inferred from the current
  products. #52 and #54 landed in this release; #53 was closed as not planned, leaving the
  conditional-sigma boundary in place.

- **`ComparatorDiagnostics` reports the inner failure behind an adjusted-variance fallback**
  (issue #95). The optional `source_status` field preserves the exact inner error under the
  public `WeakParameterIdentification` status instead of erasing it; the field is omitted from
  successful comparator JSON, so existing consumers are unaffected.
- **Frozen-attempt spatial covariance validation is reproducible again** (issue #94).
  Per-attempt frozen DGP streams, the fixed-L2 `1e8` phase/date condition policy, and the
  `ill_conditioned` / `nondifferentiable_node` statuses were restored while keeping the
  prepared estimator/JVP cache and bounded parallel replay.
- **The temporal synthetic scorer can certify a candidate** (issue #96). `coverage_bias_interval_score/7`
  adds `oracle_calibration`, `candidate_evaluation`, and `forensic_v5` modes with familywise
  bias calibration, disjoint seed domains enforced as an interval-overlap check, and a
  hash-bound calibration receipt. `forensic_v5` cannot reach a passing verdict, so the frozen
  v5 no-go cannot be certified retroactively.

## [v1.5.0] — 2026-08-17

### Breaking changes
- **Residual output semantics changed** (issue #40 / PR #42).
  `timeseries_residual_rms.tif` now contains temporal motion-model fit residuals. Consumers
  needing the former SBAS network misclosure must read `network_misclosure_rms.tif`.
- **Unimplemented interpolation is rejected** (issue #25).
  `unwrap_options.run_interpolation: true` now fails config validation instead of silently
  widening the bounded read and doing no interpolation.

### Added
- **Post-unwrap loop-closure QC gate** (issue #24, whose re-entry gate was a design review;
  the review is in the issue and the module docs). `dolphin-timeseries::loop_closure` closes
  every triangle in the **unwrapped** interferogram network and blanks pixels whose loops
  miss closure by more than half a cycle, before the SBAS solve.
  **It is not a duplicate of the existing closure-phase layer.** That one computes
  `∠(C[k,k+1]·C[k+1,k+2]·conj(C[k,k+2]))` on the coherence matrix — wrapped phase, bounded
  to `(−π, π]` by the `.arg()` — and targets decorrelation-driven bias. An unwrapping error
  is an integer multiple of `2π`, which is exactly what that `.arg()` discards: a clean 2π
  error wraps to zero and is invisible to it. There is a test asserting precisely that.
  Connected-component labels turned out to supply the correction *granularity* and a free
  prefilter, but no cross-interferogram information, so they cannot supply the detection.
  Gated by `timeseries_options.mask_unwrap_loop_errors` (default `false`); forward
  divergence from dolphin. **A no-op on a single-reference network**, which has no loops —
  it warns rather than erroring, since that is a configuration mismatch, not bad data. Emits
  `loop_closure_bad_count.tif` and `loop_closure_worst_cycles.tif`. A pixel with no
  evaluable loop is never masked: the gate acts only on positive evidence.
- **The uncertainty layers now declare their own scale, and the product is decided** (issue
  #36). `crlb_sigma_NN.tif`, `displacement_variance_NN.tif`, and `velocity_sigma.tif` were
  emitted with nothing to distinguish a Cramér–Rao *lower bound* from a predictive sigma; a
  consumer picking the wrong one understates risk by roughly 2× at the 90% level.
  - **The documented uncertainty product is the posterior from an over-determined network**
    (`interferogram_network.max_bandwidth` set): `displacement_variance_NN.tif` per epoch,
    `velocity_sigma.tif` for the rate. CRLB is a lower bound and under-covers by
    construction. On a **single-reference** network `dof = 0`, the residual inflation is
    pinned to 1, and the posterior reduces to the CRLB bound through a geometry factor — the
    measured coverage columns are identical there (0.500 / 0.583 / 0.833), and separate only
    at `max_bandwidth: 3` (0.583 / 0.833 / 1.000). The `max_bandwidth` default stays `null`
    to match dolphin, so this is a deployment choice, not a library default.
  - **Rasters now say which case they are in.** `crlb_sigma_NN.tif` carries
    `UNCERTAINTY_SCALE=crlb_bound` plus a `DESCRIPTION` naming it a bound; the posterior
    layers carry `UNCERTAINTY_SCALE` (`empirical` / `crlb_bound`), `POSTERIOR_DOF`, and a
    matching `DESCRIPTION`. The tag is computed from the actual network
    (`n_interferograms − (n_dates − 1)`), so a single-reference run self-identifies as
    carrying no empirical scale.
  - **`use_coherence_weights` stays `true`.** The unweighted posterior's better coverage is
    an artifact, not a result: on a single-reference network weighted and unweighted
    displacement are bit-identical and the unweighted posterior collapses to a spatially
    constant, dimensionless `(AᵀA)⁻¹`. See VALIDATION.md.
- **Solid-earth-tide correction** (issue #21). `dolphin-corrections::solid_earth_tide`
  models the lunisolar solid earth tide (IERS 2010 §7.1.1 step-1 degree-2 in-phase, nominal
  `h₂ = 0.6078`, `l₂ = 0.0847`, low-precision analytic Sun/Moon ephemerides) and expresses
  it as an equivalent per-date range delay, so it sums with the ionospheric and tropospheric
  layers and goes through the same `subtract_delay` stage. Unlike those it needs **no
  external data file** — only the acquisition time from the granule name and per-pixel LOS
  geometry — so it is gated by `correction_options.solid_earth_tide` (default `false`)
  rather than a file list. Both inputs are **required, not defaulted**: the tide is
  semidiurnal, so a defaulted acquisition time could be wrong by half a cycle, and a 3-D
  displacement vector cannot be projected into line of sight from the scalar
  `incidence_angle_deg`. Emits `solid_earth_tide_NN.tif`. On the three-station GNSS frame
  the LOS tide runs −157 to +138 mm per date with a 206 mm peak differential and up to
  3.35 mm of spatial spread across the frame — the spread is why it is evaluated per pixel
  rather than sampled once at the centre like IONEX. Omitted: degree-3, out-of-phase, and
  latitude-dependent Love-number terms (each ≲ 2 mm), IERS step-2 frequency dependence, and
  **ocean tidal loading**, which is the larger remaining gap and needs an external
  loading-coefficient grid. See VALIDATION.md, including why the −195 mm/yr velocity leak
  measured on that 13-epoch window is not a general figure.
- **Opt-in seasonal and step terms in the velocity fit** (issue #22).
  `dolphin-timeseries::estimate_velocity_with_model` fits an annual sinusoid and/or
  configured Heaviside steps **jointly** with the linear rate in one weighted least-squares
  solve, so a real seasonal cycle (groundwater, thermal) or a known step (co-seismic,
  instrument change) is reported separately instead of leaking into the reported rate.
  Config-gated by `timeseries_options.velocity_seasonal` (default `false`) and
  `timeseries_options.velocity_step_dates` (default empty, `YYYY-MM-DD`, resolved against
  acquisition 0). **Forward divergence from dolphin** — dolphin's `velocity.py` is
  linear-only — following the `correct_phase_bias` / `correct_velocity_temporal_correlation`
  precedent. With both unset the model is linear and the fit stays on the untouched degree-1
  estimators; `estimate_velocity_with_model` asserts against a linear model rather than
  reimplementing the parity-critical path. Step epochs are an **input, never detected**: a
  fitted step time is a different, nonlinear estimator. New layers when enabled:
  `velocity_seasonal_amplitude.tif`, `velocity_seasonal_phase_days.tif` (`UNITTYPE=days`),
  and `velocity_step_NN.tif` in `velocity_step_dates` order. Measured on a noiseless
  −25 mm/yr series with an 8 mm annual cycle, the linear-only fit reports a rate wrong by
  2.4 mm/yr over a 360-day window and by ~34 mm/yr over a 180-day one — the error is a
  function of which part of the cycle the acquisitions sampled, not of the ground. The
  bounded/tiled path re-fits through the same front door, so the model cannot reach the
  whole-frame path and miss the tiled one. Post-seismic (exponential/logarithmic) relaxation
  is deliberately not included: it needs a relaxation time constant, which is another
  nonlinear parameter or another knob, and neither is justified before these terms have
  been used on real data.
- **SHP neighbor selection is now applied during phase linking** (issue #29).
  `phase_linking.shp_method` / `shp_alpha` were accepted in config but never reached the
  covariance kernel — `sequential.rs` passed `neighbors: None` unconditionally, so
  `dolphin-shp` had no caller and covariance always used the full rectangular window. Each
  ministack now derives the GLRT or KS mask from its **real** acquisitions (carried
  compressed SLCs are projections, not observations) and passes it through. `shp_method:
  rect` keeps the unmasked kernel and is bit-identical to the previous output; the
  dolphin-oracle fixtures were generated without `neighbor_arrays`, so their configs now
  declare `rect` explicitly rather than inheriting dolphin's GLRT default, and every oracle
  contract is unchanged. This also fixes a correctness gap: with `beta: 0.0` the estimator's
  `Γ = |C|` need not be PSD, and an indefinite `Γ` yields a NaN CRLB that
  `apply_validity_mask` propagates to every emitted layer. On the MMX1/ICMX frame the finite
  footprint goes 98.8613% → 99.3062% and the GNSS uncertainty-reliability scorer completes
  for the first time, emitting `uncertainty_reliability.{json,csv,svg}`. See VALIDATION.md.
- **The AR(1) N_eff velocity-uncertainty correction is now reachable from config** (issue
  #33). `timeseries_options.correct_velocity_temporal_correlation` (forward divergence from
  dolphin, **off by default**, following the `correct_phase_bias` precedent) routes
  `fit_velocity` through `estimate_velocity_with_uncertainty_neff`, on both the whole-frame
  and bounded/tiled paths. Velocity is bit-identical with the flag on — only σ changes. On the
  MMX1/ICMX frame the inflation factor is median 1.0623, p90 1.3007, max 2.8814, with 59.9% of
  pixels inflated and 38.3% left at 1.0 (non-positive lag-1 autocorrelation). Note it is a
  no-op at both GNSS stations, whose residual autocorrelation is negative, so this dataset
  cannot validate the correction against ground truth — see VALIDATION.md.
- **Opt-in AR(1) temporal-correlation (N_eff) velocity-uncertainty correction**
  (issue #20). `dolphin-timeseries::estimate_velocity_with_uncertainty_neff` estimates
  the lag-1 autocorrelation of the WLS velocity-fit residuals and reports
  `sigma_temporal_corrected = sigma * sqrt((1+rho)/(1-rho))` (Zhang et al. 1997 / Agram
  & Zebker 2015) alongside the existing uncorrected `sigma`, without changing it —
  matching `../eo`#206's own fix for the identical understated-uncertainty bug.
  `estimate_velocity_with_uncertainty`/`VelocityOutput` are untouched (bit-identical);
  the new `VelocityOutputNeff` is a separate opt-in output so a downstream risk-tier
  threshold never moves without a reviewed follow-up. Analytic AR(1) contract fixture
  validates the inflation factor against the closed-form value and confirms it
  converges to a no-op at zero correlation.
- **Identifier-free temporal input-coverage provenance.** Geometry provenance advances to
  `dolphinrust-geometry-provenance/3` and records the versioned tile policy, aggregate and
  per-burst tile counts using stable ordinals, and final valid-pixel fraction. The workflow
  also exposes an explicit output validity mask and emits aggregate INFO coverage telemetry;
  source paths, acquisition identifiers, object keys, and AOI geometry are excluded.
- **Reliability and uncertainty outputs** (issues #12–#16). Unwrapping now retains actual
  per-interferogram connected-component labels and writes `conncomp_NN.tif`; L2 SBAS uses
  per-observation CRLB precision by default and exposes an opt-in posterior covariance API,
  displacement variance, residual RMS, and velocity sigma without allocating a covariance
  cube. CRLB rasters retain radians and now declare `UNITTYPE=rad`. The MMX1/ICMX harness runs
  weighted/unweighted A/B configurations and regenerates JSON/CSV/SVG 68/90/95% reliability
  artifacts with GNSS, CRLB-only, posterior-only, and combined uncertainty kept separate.
- **Distinct phase-linking coherence output** (issues #7 and #9). Optional
  `phase_linking.calc_average_coh` computes dolphin v0.35.0's bounded internal per-date
  `mean_j |C_ij|` values inside the fused per-pixel pass, excludes carried compressed-SLC
  pseudo-dates, and emits their real-date-weighted mean as `phase_linking_coherence.tif`.
  `DisplacementOutput` and geometry provenance expose the layer separately from estimator-fit
  `temporal_coherence.tif`; disabled output is explicit rather than an alias. Analytic,
  pinned-oracle, fused/staged, tiled/whole, multiburst, NRT, raster, and provenance contracts
  cover the full path.
- **Per-pixel LOS geometry ingest from OPERA CSLC-S1-STATIC.** The atmospheric-correction
  stage no longer projects zenith→line-of-sight with a single scalar `incidence_angle_deg`:
  when `correction_options.geometry_files` are supplied (per-burst CSLC-S1-STATIC granules),
  `dolphin-io::read_los_layers` reads the ground→sensor LOS unit-vector components
  (`/data/los_east`, `/data/los_north`), and `dolphin-corrections::resolve_los_geometry`
  reprojects + mosaics them onto the frame grid (first-covered-burst wins), deriving
  `up = sqrt(max(0, 1−e²−n²))` and per-pixel incidence `acos(up)` — character-identical to
  dolphin `atmosphere/ionosphere.py`. The iono/tropo slant then uses per-pixel `1/up`; with
  no geometry files the scalar path is **exactly bit-identical** to before. The resolved
  `LosGeometry{east,north,up}` is exposed on `DisplacementOutput.los_geometry` as the front
  door for the MMX1-colocated GPS ground-truth harness. Coverage is **fail-loud**: a frame
  extending beyond the supplied STATIC footprint (or a NaN/nodata hole) is a hard error, never
  a silent 0°/nadir pixel. Design: `md/design/per-pixel-los-geometry.md`; follow-up harness:
  `md/research/gps-feasibility-spike.md` §3.
- **Geometry-provenance artifact for GroundPulse asc/desc gating** (issue #1). dolphinRust
  now emits a machine-readable provenance record alongside each displacement run
  (orbit_direction, incidence_angle_deg, heading_deg, native range/azimuth spacing, the
  phase-linking coherence artifact key, and a `geometry_provenance` block naming source
  metadata keys + method version), sourced from real CSLC/product metadata — no guessed
  defaults. Absent provenance is represented explicitly so a consumer keeps asc/desc
  decomposition disabled when either geometry side lacks incidence/heading. New
  `dolphin-io::cslc_metadata` reader + `dolphin-workflows::provenance`; fixture-backed
  contract test. Design: `md/design/geometry-provenance.md`.

### Changed
- **GroundPulse can select a reduced serialization policy without changing the scientific
  run** (PR #46). `DisplacementOutputPolicy::GroundPulse` preserves the same computed arrays
  and provenance in memory while writing only the phase-linking-coherence raster consumed by
  GroundPulse; the existing CLI and NISAR paths retain full output by default. The controlled
  GroundPulse Docker comparison measured high-water memory down 5.46% and S3 requests down
  from 38 to 32.
- **Phase-linking covariance — row-separable box-sum** (`dolphin-phaselink::covariance`).
  The unmasked rectangular-window path (`neighbors: None`, which was the entire
  production path until SHP selection was wired in — see issue #29)
  now reuses per-output-row vertical sums across the row's output columns and sums each
  window directly in fixed left-to-right order, instead of re-reading the full
  `win_h×win_w` window per pixel. Targets the overlapping-window redundancy
  (~`win_w/strides.x`, ≈3.8× at dolphin defaults). The SHP-masked path is unchanged
  (retained direct kernel). Accumulation order differs from the direct kernel, so the
  result matches to the crate's coherence tolerance (~1e-4), **not** bit-exactly; the
  new `covariance_sliding_contract` pins that. Both the staged and fused unmasked paths
  share the one kernel, and each window's numerator depends only on its own samples, so
  `fused==staged` and `tiled==whole` stay **bit-identical**. Wall-clock speedup asserted
  by design, not yet benched. Vertical cross-row incremental (~3.7× more) is a follow-up.

### Fixed
- **The GNSS scorer now reports the pipeline's velocity model instead of being blind to it**
  (issue #44). The existing common-GNSS-epoch displacement polyfit remains as an independent
  estimator, while `insar_velocity_raster_mm_yr` and `difference_raster_mm_yr` sample
  `velocity.tif` at the declared station pixels. Optional seasonal amplitude/phase rasters are
  reported per station and every velocity scalar names its estimator. On the saved 2018
  MMX1-ICMX A/B, the unchanged polyfit is -262.0050 mm/yr in both runs; the raster-derived
  residual moves from -11.5075 to -5.7378 mm/yr with the seasonal model, whose station
  amplitudes are 24.9781 and 25.6244 mm.
- **Validation token tests no longer read the developer's real `.env`** (issue #47).
  Earthdata token resolution accepts an explicit env-file path while retaining process-token
  first, env-file second, and netrc fallback last. The exact local suite passes 32 tests with
  a real repository `.env` present before #44 adds its 33rd contract.
- **A scored GNSS validation run now always leaves a receipt describing itself** (issue #43).
  `validation/run_gps_ground_truth.py`'s `--score` branch wrote `gps_ground_truth.json` and
  returned without ever writing `run_receipt.json` — the only path in `execute()` that skipped
  it. A run root that previously failed (e.g. on a coverage gate) and was then re-scored to a
  pass kept the earlier failure's receipt (`status: "not_evaluable"`) sitting next to a passing
  `gps_ground_truth.json`; a fresh run root that passed on the first try got no receipt at all.
  The receipt is the only artifact carrying `commit`/`recipe_sha256`/`fixture_manifest_sha256`
  provenance, so either way a reader of `run_receipt.json` saw the wrong run or none. The three
  terminal paths in `execute()` now share `write_run_receipt`/`finalize_score_run`, so every
  path writes exactly one receipt matching the outcome that just happened; the scored path's
  receipt status is `gps.score_common_frame`'s own `"pass"`/`"fail"`, not the unrelated
  `"complete"`/`"not_evaluable"`/`"error"` vocabulary of the non-scored paths. Contract:
  `test_scored_pass_overwrites_stale_failure_receipt` (a stale failure receipt is replaced by a
  pass) and `test_score_run_without_prior_receipt_still_writes_one` (a fresh run root gains a
  receipt) in `validation/tests/test_gps_runner.py`.
- **`timeseries_residual_rms.tif` now carries the temporal motion-model fit residual it was
  always named for, and the SBAS network-inversion misclosure it actually served has its own
  raster** (issue #40, cross-repo signal from `../eo`#316). `SpatialProducts::timeseries_residual_rad`
  was populated from `InversionProducts::residual_rms` — the L2 network-inversion misclosure
  RMS (`A·φ = Δφ`) — not from the per-pixel scatter of displacement around the fitted velocity
  model that `dolphin-timeseries::velocity_model` / `estimate_velocity_with_uncertainty` already
  computed and `fit_velocity` discarded. Two consequences on `main`: the raster was **absent
  entirely** on the default `TimeseriesMethod::L1` path and on plain L2 without
  `write_posterior_uncertainty` (the misclosure quantity is `None` there), and where present it
  answered "did the interferogram network close," not "does displacement fit the model" — the
  two can and do diverge (a redundant, perfectly-closing network can still phase-link a
  temporally inconsistent epoch). `timeseries_residual_rms.tif` / `DisplacementOutput::
  timeseries_residual_rms` now carries the temporal-fit residual on every velocity path that
  computes a model fit (weighted linear, seasonal, step); it stays absent only on the
  unweighted-linear fast path, which computes no fit statistics at all (matching `velocity_sigma`'s
  existing rule there). The network misclosure is unchanged in value and semantics but renamed
  to `network_misclosure_rms.tif` / `DisplacementOutput::network_misclosure_rms`, still `Some`
  only for `write_posterior_uncertainty` L2 runs. **Breaking output-schema change**: a consumer
  reading `timeseries_residual_rms.tif` for the old (network-misclosure) meaning must switch to
  `network_misclosure_rms.tif` — flagged for `../eo`'s `gp-dolphin::sample_output`, the direct
  consumer named in `../eo`#316. Analytic contract
  (`network_misclosure_and_temporal_fit_residual_are_decoupled`): an over-determined network with
  zero misclosure but a motion-model-inconsistent epoch shows the two residuals moving
  independently, proving they were never the same quantity.
- **`unwrap_options.run_interpolation` is now rejected instead of silently ignored** (issue
  #25). `PreprocessOptions` round-trips a dolphin YAML and `crop.rs` already reserves
  `max_radius` of AOI halo for the stage, but no pre-unwrap interpolation exists — so
  setting the flag widened every bounded read and changed nothing. It now fails config
  validation, following the `correct_velocity_temporal_correlation` precedent (#37).
  dolphin's own default is `false`, so nothing that round-trips a real dolphin config is
  affected.
- **An unconfigured `interferogram_network` no longer fails the run** (found while working
  issue #25). `build_network` on an all-`None` network returned zero pairs and
  `finish_displacement` aborted with "interferogram_network produced no pairs" — a bare
  config could not run at all, where both dolphin versions fall back to a network. The
  fallback is the pinned dolphin v0.35.0 one (`InterferogramNetwork._check_zero_parameters`):
  single-reference on date 0, which is also what every oracle fixture config already states,
  so no contract changes. dolphin v0.42.0 moved its fallback to nearest-3
  (`max_bandwidth = 3`) — an output-changing default that stays out pending the re-pin
  decision; see PLAYBOOK §Elevated questions for the full v0.35-vs-v0.42 default diff.
- **The STATIC identity check no longer rejects the along-track neighbour the LOS mosaic
  needs** (issue #39). `verify_static_consistency` required every
  `correction_options.geometry_files` granule's `burst_id` to be in the CSLC stack's burst
  set — incompatible with the multi-burst LOS mosaic the same code advertises, because a
  frame legitimately extends past the valid LOS of the bursts being processed. On the
  three-station GNSS fixture that clips burst `008704`'s north-west corner by 309,940 pixels
  (3.21%) and needs `008705`, which is by definition not in the stack. The failure was silent:
  LOS marked absent, the scalar `incidence_angle_deg` (37°) substituted, and with troposphere
  enabled the zenith→slant projection used `1/cos 37° = 1.252` instead of the true per-pixel
  `1/up ≈ 1.18` — a ~6% error in the applied correction. The rule is now same **track** and
  pass; a different track is still rejected (the same frame intersects T041, T078 and T143).
  Because a neighbour's LOS is now mosaicked in, `resolve_los_geometry` checks rather than
  trusts first-covered-burst-wins: overlapping granules must agree to a median 1° over their
  shared pixels, else the new `CorrectionError::GeometryOverlapMismatch`.
- **A pixel with no usable CRLB no longer loses its displacement** (issue #34). The NaN bound
  on a singular `Γ` is correct and matches dolphin v0.42, but it reached the observation
  precisions as a **zero weight**, making the normal equations singular and destroying the
  displacement, after which `apply_validity_mask` blanked every other layer. A missing weight
  is missing information, not evidence the data is bad: such a pixel now weights uniformly
  and keeps displacement, velocity, and temporal coherence, while `crlb_sigma`,
  `velocity_sigma`, and `displacement_variance` still read as absent there. For a
  single-reference network the SBAS system is exactly determined, so weights cancel and the
  fallback is identical to the weighted solution; it is only an estimator change for an
  over-determined network (`max_bandwidth` set). On the MMX1/ICMX frame displacement coverage
  goes 99.3062% → **100%**, with uncertainty layers NaN at exactly the 0.69% unbounded pixels.
- **Locally empty phase-linking tiles now remain nodata instead of aborting a usable AOI.**
  Tiled processing skips a tile when any acquisition has no finite complex support in its
  dependency window, initializes every output layer as nodata, and links the remaining tiles.
  It still fails when an acquisition is empty across the whole burst or when no tile has
  complete temporal support. Multi-burst mosaics now allow finite overlap to replace nodata
  without letting a later burst's nodata erase earlier valid pixels.
- **All-non-finite phase-linking input now fails loudly** (issue #8). A stack containing an
  acquisition with no finite complex samples returns an error before covariance/estimation,
  matching pinned dolphin v0.35.0's `PhaseLinkRuntimeError` instead of allowing a zero
  coherence matrix to masquerade as `temporal_coherence=1.0` / zero displacement. Partially
  valid inputs retain the existing dolphin-compatible masking behavior.
- **`auto_tile_opt_in_holds_oracle_parity` compared against the wrong estimator.** The gate
  loaded the fixture config without pinning `use_coherence_weights`, which became default-true
  in the reliability wave, so it fitted a CRLB-weighted velocity and compared it to the
  unweighted dolphin oracle (velocity error 0.446 against a 1e-2 bar). Displacement still
  matched because the single-reference SBAS system is exactly determined and therefore
  weight-invariant. The gate now pins the estimator like `displacement_contract.rs`, so
  auto-tiling is the only variable. This never reached CI: `oracle/fixtures/*` is gitignored,
  so the oracle-parity gates skip there and had been green-by-skip.
- **Flaky HDF5 unit tests in `dolphin-io`.** `hdf5-metno` links a non-thread-safe HDF5, so
  parallel test threads creating/opening HDF5 fixtures raced and corrupted global library
  state (intermittent `geo`/`nisar`/geometry failures). HDF5-touching unit tests now serialize
  through a shared test lock.

### Deferred
- iono **ground→ionospheric-shell (450 km) incidence mapping** (pre-existing divergence from
  dolphin, not introduced here; the per-pixel path reproduces today's *ground*-incidence
  behavior); `local_incidence_angle` ingest; nearest-vs-bilinear resample at burst seams
  (single-burst consumer unaffected). See the design doc's Deferred section.

### Changed
- **The GNSS harness supports frames wider than one burst.** `crop_real.py` crops every
  covering CSLC-S1-STATIC granule instead of exactly one, clamping a partially-overlapping
  neighbour to its intersection, and the runner passes them all as
  `correction_options.geometry_files` for `dolphin-corrections` to mosaic. Needed because a
  three-station frame clips the primary burst's valid LOS at a corner (3.21% of pixels), which
  the fail-loud coverage gate correctly rejects. Only same-track neighbours are admissible.

- **Tropospheric delay is now taken at each pixel's terrain elevation** (issue #38). The real
  OPERA L4 product resolves delay over 145 height levels and GDAL maps each to a band, so the
  reader's `rasterband(1)` used the −500 m level rather than the terrain — over-stating the
  epoch-relative differential delay by 1.9–7.5× at ~2250 m. `correction_options.dem_file` now
  drives per-pixel linear interpolation between the two bracketing levels, and a
  height-resolved granule supplied *without* a DEM is rejected rather than silently
  mis-corrected. On the MMX1/ICMX/MXTX frame the MMX1−MXTX relative differential delay comes
  out at 31.4 mm against 30.4 mm computed independently in Python, versus 68.1 mm from the old
  band-1 path.

### Known limitations
- **The MMX1/ICMX coverage numbers are indicative, not a calibration claim.** Twelve
  temporally correlated epochs quantize every 68/90/95 bin at 8.3%, from one station pair on
  one burst. The CRLB columns under-cover **by construction** — a Cramér–Rao *lower* bound
  used as a predicted σ must under-predict the spread. The unweighted posterior is **not** an
  alternative: on an exactly-determined single-reference network it collapses to a spatially
  constant `(AᵀA)⁻¹` geometry factor with no physical scale. On that network the residual-based
  inflation is inert (`dof = 0`, residual RMS ~3e-18), so **no layer carries empirical
  scale** — the posterior column is bit-for-bit the CRLB column there. An over-determined
  network restores it: scored A/B on the same GNSS truth, `max_bandwidth: 3` moves
  posterior-only coverage from 0.500/0.583/0.833 to **0.583/0.833/1.000** and velocity
  agreement from −3.319 to −2.243 mm/yr. Recommendation: carry the over-determined posterior
  as the uncertainty product; the library default stays `null` to match dolphin, so this is a
  deployment choice (issue #36). Truth-set limits in issue #35.

## [v1.4.0] — 2026-06-18

### Changed
- **Phase-linking covariance — ~2.4× faster** (Phase 3), no accuracy change. The per-pixel
  sample-coherence matrix (`dolphin-phaselink::covariance`, the #1 hot path) now reduces via a
  direct **Hermitian** product — summing only the upper triangle over contiguous sample rows and
  mirroring the lower — instead of ndarray's generic complex `dot` (which has no SIMD/BLAS path for
  `Complex<f64>`) plus a per-pixel conjugate-transpose allocation. Real-frame phase-linking is
  **2.38× faster** (host-controlled same-session A/B: 3.07 → 1.29 s; throughput 432 → 1028
  kpix·slc/s) and beats the committed pre-R1 baseline (2.01 s) absolutely. The coherence matrix is
  Hermitian by construction so the result is identical; `covariance_matches_oracle` (≤1e-4) and all
  analytic/quality/GPU/sign contracts stay green. Measurements + methodology in `bench/PERF.md`.

### Added
- **3D-unwrap-ready dispatch interface** (Phase 5), `dolphin-workflows::unwrap_backend`. The unwrap
  backend is now behind a **network-level** `UnwrapBackend` trait — it receives the linked phase +
  date pairs (not pre-formed independent 2D interferograms), so a future spurt-style 3D
  spatiotemporal solver can implement the trait and unwrap the whole stack jointly without any
  pipeline change. The shipped backends `SnaphuBackend` and `TophuBackend` implement it via the
  unchanged per-ifg loop; **output is bit-identical** (the end-to-end oracle contract still passes
  through the new dispatch; trait seam covered by `unwrap_backend_contract.rs`). No spurt port — the
  interface only. The `ref·conj(sec)` ifg sign convention is preserved (guarded by
  `sign_convention.rs`).
- **Phase-bias / non-closure correction** (Phase 4), `dolphin-phaselink::phasebias` — Michaelides
  et al. (RSE 2022). **Not in Python dolphin** (leads the oracle). The nearest-neighbour closure of
  the coherence matrix satisfies `Ξ_k = β_k + β_{k+1}`; a per-pixel first-order constant
  bias-velocity `β̄ = mean_k(Ξ_k)/2` is subtracted from the linked series (`θ_n ← θ_n·e^{−j n β̄}`)
  before the interferogram network. Opt-in via `phase_linking.correct_phase_bias` (**off by
  default** → default output and the oracle/sign contracts are unchanged; forces closure
  computation when on). Validated by an analytic fixture (constant bias recovered to <1e-9, zero
  residual) and a **measured non-closure reduction 0.800 → 0.095 rad (8.4×)** on a 100-date series;
  wired end-to-end (`run_displacement`). Numbers in VALIDATION.md.
- **NRT incremental displacement — end-to-end front door** (Phase 2b). `run_displacement_resumable`
  returns a `DisplacementState` (per-burst resumable phase-linking state + the files consumed);
  `update_displacement` folds newly-arrived acquisitions into the series — re-phase-linking only
  each burst's open trailing ministack + new ones via the carried compressed SLC, then recomputing
  the non-causal downstream (ifg network → SNAPHU unwrap → SBAS → velocity) from the updated phase
  history. The result is **bit-identical to a full `run_displacement`** of the extended stack
  (max|Δ| = 0 through unwrap + inversion; `nrt_displacement_contract.rs`). Exposed as a
  `dolphin stream --config <yaml> --initial <N>` CLI subcommand (process an initial window, then
  fold each later acquisition in, rewriting outputs). An update must extend every burst (a SAR pass
  yields one CSLC per burst) and the prior files must be a date-ordered prefix.
- **NRT incremental ministack updates** (Phase 2), in `dolphin-workflows::sequential`. Sequential
  phase-linking is feed-forward — a ministack reads only the compressed SLCs of prior ministacks
  and its own real SLCs — so a ministack that has filled to `ministack_size` ("sealed") never
  changes when later acquisitions arrive. `run_sequential_resumable` returns a `SequentialState`
  (sealed ministacks' products + the open trailing ministack's raw SLCs); `update_sequential`
  folds in newly-arrived acquisitions by re-phase-linking **only** the open ministack and any new
  ones, carrying the sealed compressed SLCs. The result is **bit-identical** to a full rerun of
  the extended stack — `cpx_phase`, compressed SLCs, stitched temporal coherence, CRLB, and
  closure all match with max|Δ| = 0 (`tests/nrt_incremental_contract.rs`: block update,
  one-at-a-time streaming, and the sealed-boundary edge case). `MiniStackPlanner::plan_with_offset`
  resumes the carry-forward batch accounting for the tail. The non-causal downstream (ifg network
  → unwrap → timeseries → velocity) recomputes from the updated phase history; the operational
  speedup is in skipping re-phase-linking the sealed history of a long stack.

## [v1.3.0] — 2026-06-17

### Added
- **Atmospheric corrections — ionospheric + tropospheric** (second half of v1.3.0), in the new
  `dolphin-corrections` crate. Both produce a per-acquisition range delay (meters) on the frame
  grid; the apply stage subtracts the per-date delay (relative to acquisition 0) from the
  inverted LOS-phase series **before velocity**. **Off by default** (opt-in via correction
  files, matching dolphin) — with none configured, `run_displacement` output is unchanged.
  - **Ionosphere (`dolphin-corrections::ionosphere`)** — IONEX GNSS TEC maps → L-band range
    delay via the closed-form `delay = TEC_LOS·K/f²` (`K = 40.31`; Yunjun et al. 2022 / Chen &
    Zebker 2012), **scaled to the configured carrier** (`1/f²`). The dominant L-band term:
    `(f_C/f_L)² ≈ 18×` C-band for the same TEC. Closed-form contract green; **validated on a
    real IGS final GIM from CDDIS** — 56.5 TECU → 14.4 m L-band delay (18.5× C-band).
  - **Troposphere (`dolphin-corrections::troposphere`)** — OPERA L4 (`OPERA_L4_TROPO-ZENITH_V1`)
    netCDF ingest via GDAL's `NETCDF:` driver, then a **reprojecting resample**: same-CRS grids
    take the bilinear path, cross-CRS grids (global EPSG:4326 product → UTM frame) take the new
    `warp_to_frame` (GDAL bilinear `reproject`), zenith→slant by `1/cos(inc)`. Synthesized-fixture
    and 4326→UTM warp contracts green (analytic delay recovered at known frame pixels `< 5e-3 m`,
    bare-warp + end-to-end through `build_troposphere`); the old CRS-mismatch `warn!` path is gone.
    **Real granule validated end-to-end on a real UTM frame** — the global EPSG:4326
    `OPERA_L4_TROPO-ZENITH_V1` granule warps onto the real Mexico City UTM 32614 384² frame:
    applied zenith mean **2.553 m** (slant@39° ≈ 3.285 m), physically consistent with the city's
    ~2.2 km altitude vs the 2.79 m sea-level centre. `DelayGrid` now carries the source CRS WKT;
    a CRS-less L4 grid spanning geographic-degree ranges is assigned EPSG:4326 (the plate-carrée
    product spec). See `VALIDATION.md`.
  - **RAiDER fallback (`dolphin-corrections::raider`)** — subprocess + GDAL ingest, **gated
    behind a `raider_available()` check like SNAPHU**; returns `RaiderUnavailable` rather than
    being stubbed when RAiDER is absent. The L4 path is primary.
  - `correction_options` config mirrors dolphin's `ionosphere_files` / `geometry_files` /
    `dem_file` (a dolphin YAML round-trips); `troposphere_files` (direct OPERA-L4 ingest),
    `incidence_angle_deg`, and `troposphere_variable` (default `"total"` = hydrostatic + wet)
    are **forward divergences** — dolphin derives troposphere from a DEM via RAiDER and has no
    `troposphere_files`. Layers surface on `DisplacementOutput.{ionosphere_delay,
    troposphere_delay}` and as `ionosphere_NN.tif` / `troposphere_NN.tif` COGs.
  - `dolphin-io::grid_centroid_lonlat` — frame-centre (lon, lat) via a CRS transform, to sample
    the coarse global IONEX grid at the frame.
- **NISAR / L-band geocoded-SLC ingest path** (first half of v1.3.0) — reads a NISAR L-band
  GSLC stack end-to-end into a displacement product.
  - `dolphin-io::nisar` — `read_nisar_rslc` / `read_nisar_stack` read the NISAR complex-`f32`
    `{r, i}` compound grid as `Cf32`; `read_nisar_geotransform` derives the affine transform
    from the NISAR `xCoordinates`/`yCoordinates` arrays and the `projection.epsg_code`
    attribute (GDAL returns identity for this layout). Contract test vs a synthesized
    NISAR-layout fixture (pixel values, grid shape, geotransform, EPSG).
  - **De-risk correction:** the prompt assumed NISAR was a *complex-int16* compound; the real
    `NISAR_L2_GSLC_BETA_V1` granule is **complex-`f32` `{r, i}`** (same layout as OPERA), so
    the only NISAR-specific code is the geocoding metadata reader. Validated end-to-end on a
    real 7.2 GB granule (reader + geotransform/EPSG) — see `VALIDATION.md`.
  - `input_options.input_type: InputType { opera_cslc (default) | nisar_gslc }` selects the
    reader. **Forward divergence** — dolphin v0.35.0 has no product-type field (it dispatches
    by workflow entrypoint); legacy YAML round-trips to `opera_cslc`.
  - L-band wavelength (≈0.2384 m) threads through `input_options.wavelength` to the `−λ/4π`
    velocity scaling (`velocity_uses_nisar_wavelength` proves the NISAR λ is used, not the S1
    default). No new solver — L-band is a parameter change.
  - End-to-end contract (`nisar_e2e_contract`): a multi-acquisition synthesized NISAR stack
    runs through `run_displacement` → typed output + COGs, grid/EPSG/geotransform correct.
  - **Limitation:** geometrically correct but **atmospherically uncorrected**. Ionospheric
    (~16× the C-band effect) + tropospheric corrections are a separate later v1.3.0 loop.

### Fixed
- **Interferogram sign convention — inverted LOS sign in v1.0.0–v1.2.0, now corrected.**
  `displacement.rs::unwrap_pair` formed the ifg as `sec·conj(ref)`; dolphin **production**
  (`interferogram.py`) forms `ref·conj(sec)`. The reversed order **globally inverted the LOS
  displacement *and* velocity sign of every release v1.0.0–v1.2.0** — subsidence read as uplift
  and vice-versa. It was invisible because the oracle generator (`oracle/gen_displacement.py`)
  carried the *same* inversion, so the sign-sensitive contracts proved Rust agreed with a
  flipped oracle, not with production. **Impact for eo:** the `velocity_mm_yr` sign (subsidence
  vs uplift) that drives GroundPulse risk tiers was inverted in v1.0–v1.2 and is now correct.
  Fixed in `e1db05a`; the oracle was corrected in lockstep (`2c85a79`). Backfilled this release
  with an **always-on analytic sign guard** (`sign_convention`, proven to go red if `unwrap_pair`
  is reverted) and a **gated real-data test** (`sign_real_data`, `SIGN_REF_PROD_IFG`) confirming
  dolphinRust matches a full production `dolphin run` on the F38502/Corcoran subsidence bowl —
  displacement correlation **−0.97 → +0.99** before/after the fix. See `VALIDATION.md`
  §"Interferogram sign convention".

## [Unreleased] — v1.2.0

### Added
- **CRLB uncertainty + sequential closure-phase quality layers** (`dolphin-phaselink`),
  validated against a **forward dolphin oracle v0.42.0** used *only* for these two layers
  (existing kernels stay validated at v0.35.0).
  - `crlb::estimate_crlb` — per-date Cramér–Rao σ from the Fisher information of the
    coherence model (`X = 2L·(Γ⊙Γ⁻¹−I)`, σ = `sqrt(diag(inv(ΘᵀXΘ+εI)))`), CPU `faer`/f64.
    Singular / fully-decorrelated Γ → `NaN` past the reference date (the v0.42 fix). This is
    the physical per-pixel uncertainty that feeds GroundPulse's `confidence_score`.
  - `closure::estimate_closure_phases` — nearest-neighbour triplet non-closure
    `∠(C[k,k+1]·C[k+1,k+2]·conj(C[k,k+2]))`; the prerequisite signal for phase-bias work.
  - Surfaced on `DisplacementOutput` (`crlb_sigma`, `closure_phase`, both `Option<Array3<f64>>`)
    and written as per-band COGs (`crlb_sigma_NN.tif`, `closure_phase_NN.tif`), sharing the
    grid CRS/geotransform; produced end-to-end by `run_displacement`.
  - Config flags match dolphin: `phase_linking.write_crlb` (default **on**),
    `phase_linking.write_closure_phase` (default **off**) — a real dolphin YAML round-trips.
  - Contracts: `quality_v042_contract` (CRLB σ + closure max |Δ| < 1e-4 vs v0.42.0;
    singular-Γ NaN matches; analytic consistency checks). GPU CRLB is a later follow-up.
- **tophu-style multi-scale unwrapping** (`dolphin-unwrap::unwrap_multiscale`) — OPERA's
  production multi-scale strategy driven over the existing SNAPHU wrapper: **coherence-weighted**
  coarse multilook (low-trust blocks masked + filled from trusted neighbours) → single SNAPHU
  unwrap → nearest upsample → overlapping tiled SNAPHU (rayon) → **overlap-based inter-tile
  cycle reconciliation** (maximum-reliability spanning forest over the coherent overlaps) →
  **feathered tile merge**. **Opt-in** via `unwrap_method: tophu`; **SNAPHU stays the default
  and the default build is behaviourally unchanged.**
  - Config: dolphin's `tophu_options` block (`ntiles`, `downsample_factor`, `init_method`,
    `cost`) is now modeled, so a real dolphin YAML round-trips it; new `UnwrapMethod::Tophu`
    routes the unwrap network through it (dolphin reserves its `multiscale_unwrap` for
    ICU/PHASS — we expose it driving the SNAPHU solver we ship).
  - Contracts: ramp recovery within the raw-SNAPHU envelope, coarse-pass round-trip, planted
    inter-tile 2π jump resolved, 2×2-grid loop-consistency, coherence-weighted-coarse-tracks-
    truth, fill, tile-cover, and up-sample unit tests.
  - **Measured win** (`bench/UNWRAP.md`): on the frozen large low-coherence scenes tophu now
    **beats** raw SNAPHU on all three metrics on both scenes — discontinuities −9 % on both,
    gross-cycle-error −10 % on the steep+decorr-ring scene, rms ≤ raw on both. The scenes,
    noise model, seeds and metrics are unchanged from the earlier honest-loss measurement;
    only the algorithm changed (coherence-weighted coarse + overlap-graph merge + feathered
    seams replacing the per-tile snap-to-coarse). Prefer tophu for large partly-decorrelated
    scenes; SNAPHU stays the simpler default for small/coherent scenes.
- **Per-ministack temporal-coherence stitching** (`dolphin-workflows::sequential`) — the
  cross-ministack temporal-coherence reduction is now dolphin's NaN-aware mean
  (`numpy.nanmean`, `_average_or_rename`) rather than a plain mean. Equal on all-finite
  layers (parity preserved), but a pixel masked/decorrelated in some ministacks now averages
  only the finite ones instead of being diluted toward zero — matching dolphin on
  many-ministack frames and closing the per-band CRLB/closure concatenation caveat. Contract
  `stitching_and_quality_match_oracle_multiministack` vs v0.42 oracle (`gen_stitch_v042.py`)
  on a 2-ministack stack: stitched temp_coh + concatenated CRLB + closure all < 1e-3.

## [Unreleased] — v1.1.0

### Added
- **GPU compute backend — first-class** (`wgpu`/Metal, f32; compiled into the **default
  build**). Runtime-selected via `worker_settings.compute_backend` (`auto` / `cpu` / `gpu`):
  `auto` uses the GPU at/above the ~128² crossover and the CPU below; **no GPU adapter,
  unsupported `nslc`, or a `no-gpu` build → automatic CPU fallback with a warning, never a
  panic.** The CPU (`faer`, f64) path stays the correctness reference. Covariance + EVD/EMI
  run in-shader (one thread per pixel); GPU covariance supports the SHP neighbor mask and the
  EMI β regularization. EMI uses an **all-pixel-accurate hybrid**: the kernel flags
  ill-conditioned / near-degenerate / borderline-PD pixels (bottom eigengap, Rayleigh
  wrong-mode guard, coherence floor, min Cholesky pivot) and the host recomputes that minority
  on f64 `faer`. Real Mexico 384² stack: **max Δφ 0.607 mm across every pixel, no π-rad tail**
  (EVD 0.176 mm). `MAX_NSLC` lifted 16→32 via deterministic threadgroup scratch (bit-identical
  run-to-run). Wired through `run_displacement` (`dolphin_phaselink::ComputeEngine`). Build
  CPU-only with `--no-default-features --features no-gpu`. Honest speed: end-to-end on an
  *integrated* M2 Pro the GPU is ~0.66× on the real stack (slower) and ~1.09× on synthetic
  stacks above ~192² — the value is correctness + portability to discrete NVIDIA/AMD (same WGSL,
  unchanged). See `bench/GPU.md` and `VALIDATION.md`.
- **Auto spatial reference-point selection** (dolphin v0.36 center-of-mass): the displacement
  series is referenced to a stable pixel — `timeseries_options.reference_point` if set, else
  the quality-weighted center of mass of the largest high-coherence region
  (`dolphin_timeseries::select_reference_point` / `reference_to_point`). The chosen point is
  exposed on `DisplacementOutput::reference_point`. The pinned v0.35.0 oracle uses `argmin`
  (no center-of-mass), so selection is contract-tested analytically.
- **Speed baseline** (`bench/`): reproducible dolphinRust-vs-dolphin benchmark with per-stage
  `tracing` timing in `run_displacement` (`RUST_LOG=info`). Real-frame phase-linking 3.6×,
  end-to-end 2.0× (unwrap-bound by an emulated snaphu binary). See `bench/README.md`.

### Validated
- **Velocity absolute scale on a real deforming scene** (B4): Mexico City burst
  T005-008704-IW1 — velocity TLS (orthogonal) slope ≈1.03 vs the oracle with matching
  magnitude, closing the documented real-data scale gap. See `VALIDATION.md`.

### Integration
- **GroundPulse (eo) adoption**: a `gp-dolphin` crate + standalone worker in `../eo`
  (branch `feature/gp-dolphin-rust`) calls `run_displacement` in-process via
  `spawn_blocking`, lands a velocity COG via `gp-storage`, and writes
  `displacement_aoi_summary` + `aoi_raster_products` rows in PostGIS. One real OPERA
  frame ran end-to-end. Isolated as its own Cargo workspace because dolphinRust's
  `hdf5-metno` (system HDF5 2.x) cannot share a binary graph with eo's static
  `hdf5-sys` (HDF5 1.x). Unpushed, pending review.

## [1.0.0] — 2026-06-16

First complete build: an end-to-end, library-first Rust rebuild of the OPERA / DISP-S1
displacement pipeline, validated against Python `dolphin` v0.35.0 as a reference oracle to
physically-meaningful tolerances.

### Added
- **End-to-end displacement pipeline** (`dolphin_workflows::run_displacement`): read CSLC
  stack → sequential phase linking (EVD/EMI) → interferogram network → SNAPHU unwrap →
  SBAS inversion → velocity. Synchronous and runtime-agnostic (no tokio) for `spawn_blocking`.
- **Typed public result** (`DisplacementOutput`): displacement cube, velocity (raster units),
  `velocity_mm_yr`, temporal coherence, acquisition days, EPSG, and geotransform — returned
  in memory and mirrored to disk.
- **L1/ADMM inversion** (dolphin's default least-absolute-deviations) alongside L2 weighted
  least squares; config-driven via `timeseries_options.method` (default L1). Matches the
  dolphin oracle to < 1.5e-6 on a redundant network.
- **Physical velocity** in mm/yr: acquisition dates are parsed from CSLC filenames
  (`input_options.cslc_date_fmt`) to derive real temporal baselines, and LOS phase is
  converted via `−λ/4π` (`input_options.wavelength`, else the Sentinel-1 default).
- **Temporal coherence** quality layer (ministack-averaged, dolphin's
  `temporal_coherence_average`), surfaced in the result and written as a raster.
- **Cloud-Optimized GeoTIFF outputs** (tiled, DEFLATE, overviews) for velocity, temporal
  coherence, and per-date displacement, sharing the CSLC grid's CRS + geotransform
  (`dolphin_io::read_geotransform` reads OPERA coordinate arrays + EPSG).
- **`dolphin` CLI** — a thin wrapper over `run_displacement` consuming a genuine dolphin
  `DisplacementWorkflow` YAML unchanged.
- **Real-data validation harness** (`validation/run.sh`, `compare.py`) and per-kernel oracle
  contract tests for every numerical crate.
- **Docs**: README quickstart (CLI + library), `docs/usage.md` integration guide (incl. the
  `spawn_blocking` pattern and output schema), and a runnable
  `crates/dolphin-workflows/examples/run_synthetic.rs`.
- `#![warn(missing_docs)]` on every crate; `cargo doc --no-deps` is clean.

### Validation
- Per-kernel contracts vs dolphin v0.35.0 `.npy` fixtures all pass (phase-link eigenvector
  overlap > 0.999, coherence < 1e-4, L1 < 1.5e-6).
- End-to-end synthetic single-burst equivalence: displacement corr 1.0000 / demeaned
  RMS ≤ 0.05 rad; velocity absolute scale a = 1.0000 (noise-free) → 0.9997 (realistic speckle).
- Real OPERA tier (4 bursts incl. Central Valley): config compatibility PASS; engine
  agreement PASS (displacement RMS residual ≤ 0.008 rad, matching velocity magnitude +
  temporal coherence). Reproducer: `validation/{fetch_real,crop_real,scan_coherence}.py`,
  `run_real.sh`.

### Changed
- **The GNSS harness supports frames wider than one burst.** `crop_real.py` crops every
  covering CSLC-S1-STATIC granule instead of exactly one, clamping a partially-overlapping
  neighbour to its intersection, and the runner passes them all as
  `correction_options.geometry_files` for `dolphin-corrections` to mosaic. Needed because a
  three-station frame clips the primary burst's valid LOS at a corner (3.21% of pixels), which
  the fail-loud coverage gate correctly rejects. Only same-track neighbours are admissible.

- **Tropospheric delay is now taken at each pixel's terrain elevation** (issue #38). The real
  OPERA L4 product resolves delay over 145 height levels and GDAL maps each to a band, so the
  reader's `rasterband(1)` used the −500 m level rather than the terrain — over-stating the
  epoch-relative differential delay by 1.9–7.5× at ~2250 m. `correction_options.dem_file` now
  drives per-pixel linear interpolation between the two bracketing levels, and a
  height-resolved granule supplied *without* a DEM is rejected rather than silently
  mis-corrected. On the MMX1/ICMX/MXTX frame the MMX1−MXTX relative differential delay comes
  out at 31.4 mm against 30.4 mm computed independently in Python, versus 68.1 mm from the old
  band-1 path. Superseded detail of the original guard (issue #38). GDAL exposes each of the real product's 145 height levels as a
  band, so reading `rasterband(1)` used the −500 m level rather than the terrain. Measured at
  the MMX1/ICMX/MXTX stations (~2250 m), that over-states the epoch-relative differential
  delay by 1.9–7.5× — enabling `troposphere_files` today would have applied a correction about
  twice too large and read as "correction doesn't help". Selecting the level needs a DEM;
  until then the reader fails loudly, as the LOS-coverage gate already does.

### Known limitations / deferred
- **Real-data velocity absolute scale under strong signal** not independently pinned (sampled
  coherent scenes were tectonically stable); scale confirmed on the synthetic tier.
- Multi-burst stitching is implemented but not yet exercised on a real multi-burst frame.
- CRLB / closure-phase rasters, complex-GeoTIFF (CFloat32) writer, NISAR custom geotransform,
  `EagerLoader` prefetch, and tophu/spurt/whirlwind unwrappers are deferred (see STATUS.md).

[Unreleased]: https://github.com/morton-analytics-llc/dolphinRust/compare/v1.6.0...HEAD
[v1.6.0]: https://github.com/morton-analytics-llc/dolphinRust/compare/v1.5.0...v1.6.0
[v1.5.0]: https://github.com/morton-analytics-llc/dolphinRust/compare/v1.4.0...v1.5.0
[1.0.0]: https://github.com/morton-analytics-llc/dolphinRust/releases/tag/v1.0.0

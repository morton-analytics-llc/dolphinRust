# Temporal covariance slope inference

This document freezes the pure timeseries boundary for issue #53. This
pure-kernel contract does not by itself authorize a corrected raster, serving,
release, or a change to the conditional `velocity_sigma.tif` path. #53 closes
only after the frozen synthetic and resource gates pass and the
preregistration, producer source set, binary, #52 replay, and reviewed #54
spatial-factor identities match. [EO #505](https://github.com/morton-analytics-llc/eo/issues/505)
owns held-out GNSS field acceptance and independent temporal scientific review
after closure.

## Estimand and input

The estimand is the linear slope of an already spatially differenced series,
anchored at acquisition zero:

```text
y_t = beta * (t - t_0) + e_t,  y_0 = 0
V(theta) = C54_delta + sigma2 * D * R_rho * D
R[i,j] = rho_12 ^ (abs(t_i - t_j) / 12 days)
D[i,i] = sqrt(C54_delta[i,i] / geometric_mean_positive_diag)
```

`C54_delta` is consumed directly from the same-frame #54 target-minus-reference
factor. The estimator never reconstructs it from two marginal rasters. Missing
dates select rows and columns without imputation. Acquisition zero is removed
from the stochastic solve after its exact gauge is checked.

The supported initial model is linear Sentinel-1 slope only. Seasonal and step
models remain `unsupported_model`. A non-finite/asymmetric factor, missing gauge,
nonpositive diagonal, non-positive-definite total covariance, boundary rho,
ill-conditioned design, or insufficient dates fails closed.

## Comparators

The pure kernel evaluates origin-anchored OLS, oracle GLS using the generating
parameters, constrained REML plug-in GLS, a covariance-parameter curvature
adjusted scalar comparator, a slope profile-likelihood comparator whose
unrestricted and fixed-slope ML objectives share one frozen nuisance parameter
space, and complete-refit parametric bootstrap. Every bootstrap replicate
resimulates from the fitted total covariance and refits mean and covariance
parameters. The public Rust result contains point estimates and validation
intervals but no corrected inferential standard-error field.

Raw adjacent residual correlation is retained unclamped with pair count and
minimum/median/maximum elapsed gap. Fewer than three pairs means absent
correlation, not zero.

## Frozen synthetic grid

`validation/temporal_covariance_synthetic_engine_preregistration.json` schema
`dolphinrust-temporal-covariance-preregistration/5` enumerates every supported
factor and immutable threshold. The first implementation runs only compact
contract fixtures; the release run uses 1,050 seeds for each of 24 supported
cells on both execution paths, for 50,400 frozen attempts.

- Dates: 12, 24, 48, and 96 retained dates.
- Correlation: iid 0, 0.3, 0.6; 0.85 only for 48/96-date cells.
- Cadence: regular 12-day, alternating 6/18-day, jitter up to 4 days, and two
  36-day gaps.
- Missingness: none, 10% MCAR, 25% MCAR, and one contiguous 20% block.
- Variance ratios: 1, 4, and 16, with alternating and contiguous arrangements.
- Reference contribution: 0, 0.5, and 2, with only block-PSD target/reference
  joint factors.
- Reference replay: overlap 0.75/0.5/0.1, distance 10/50/100 pixels, and
  sequential depths 1/2/4. The supported production batch is exact; compressed
  JVP is a frozen unsupported stratum until its provider-backed joint replay is
  available.
- Methods: OLS, oracle GLS, legacy intercept-plus-slope WLS marked
  non-comparable, lag-one scalar effective-N, plug-in GLS REML, the REML
  covariance-parameter adjusted scalar, ML slope-profile likelihood, and
  complete-refit bootstrap.

The release JSONL batch has two explicit paths. `fixed_factor` consumes the
direct #54 covariance matrix. `production_path` consumes raw-complex target and
reference series and a frozen proper-complex noise model. Rust constructs one
same-seed source DAG, invokes the #52 temporal replay and the #54 joint
target/reference replay, checks their target marginal identity, and passes only
the resulting direct difference covariance to the estimator. A production-path
record fails closed on missing inputs, raw-complex validity, replay failure, or
any seed mismatch. Both paths use the same origin-anchored estimator and never
write a corrected product.

The per-cell gates are absolute standardized slope bias <= 0.05 empirical SD,
coverage error <= 0.03/0.02/0.015 at 68/90/95%, >=99% successful supported
emission, at least 198 of 200 complete refits, and no unsupported/boundary cell
is promoted. Conditional and
unconditional coverage, proper interval score, width, failed fits, resources,
and preregistration/code hashes are retained separately. Attempted failures are
never topped up, and aggregate result hashes and measured wall/RSS receipts are
created only after the batch exits.

The release producer sends same-cell consecutive seeds in frames of at most 32.
Each frame generates one direct factor and one temporal fit per seed, evaluates
the target lanes through the factor-native batch path, and caps Rayon at 12
workers. Each cell is an independently committed shard, so interruption resumes
from complete cells without replaying or retaining dense attempt evidence.

## dolphinRust closure and EO promotion boundary

dolphinRust closes #53 only after the complete frozen synthetic matrix passes,
the preregistered resource limits pass, and the preregistration, producer source
set, binary, #52 replay, and reviewed #54 spatial-factor identities all match.
The receipt then records `engine_validation_eligible = true` and
`engine_validation_status = synthetic_validated_scope_match`. This establishes a
temporal candidate validated against synthetic evidence only. Field calibration
and GroundPulse use require the downstream EO gates.

EO owns the untouched GNSS station-pair cohort, independent scientific review of
the temporal field evidence, GroundPulse enablement, and publication. Those gates
run against the exact dolphinRust candidate and identities after #53 closes. A
failed or non-evaluable EO gate leaves GroundPulse on conditional output. The
frozen #53 result remains unchanged. #54's independent spatial review remains
part of the #54 contract and is unchanged by this boundary.

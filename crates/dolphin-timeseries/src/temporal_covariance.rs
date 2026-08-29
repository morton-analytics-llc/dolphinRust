//! Validation-only temporal covariance and slope inference kernel for issue #53.
//!
//! The kernel operates on an already spatially differenced, origin-anchored
//! series and a direct #54 difference covariance.  It is deliberately separate
//! from workflow output code: no raster writer or corrected uncertainty product
//! is exposed here.  The public result contains point estimates, diagnostics,
//! and validation intervals, but no corrected inferential standard error.

use faer::linalg::evd::{compute_hermitian_evd_req, ComputeVectors};
use faer::{get_global_parallelism, Mat, Side};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal, StudentsT};

#[cfg(test)]
thread_local! {
    static CHOLESKY_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static DENSE_PROFILE_PLUGIN_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static DENSE_ADJUSTED_SCALAR_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) const DAYS_PER_YEAR: f64 = 365.25;
pub(super) const SYMMETRY_TOLERANCE: f64 = 1e-10;

/// Frozen complete-refit bootstrap attempt count from the #53 preregistration.
pub const COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS: usize = 200;

/// Conservative peak allocation composition for one complete temporal fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalCovarianceWorkspaceComposition {
    /// Caller-owned observation vector and dense difference covariance.
    pub input_bytes: u64,
    /// Selected inputs plus retained oracle and plug-in fits.
    pub retained_fit_bytes: u64,
    /// Largest profile-optimizer allocation above the retained fit.
    pub profile_optimizer_peak_bytes: u64,
    /// Complete-refit bootstrap driver and nested profile-fit peak.
    pub bootstrap_peak_bytes: u64,
    /// Linked Faer matrix, eigenvalue output, and queried EVD scratch.
    pub faer_condition_peak_bytes: u64,
    /// Conservative simultaneous peak of the complete production call.
    pub total_bytes: u64,
}

/// Derive the complete temporal-fit allocation bound from the linked Faer EVD
/// implementation and the actual optimizer/bootstrap lifetimes.
#[must_use]
pub fn temporal_covariance_workspace_composition(
    acquisition_count: usize,
    bootstrap_replicates: usize,
) -> Option<TemporalCovarianceWorkspaceComposition> {
    if acquisition_count < 2 {
        return None;
    }
    let matrix = nested_f64_matrix_bytes(acquisition_count)?;
    let vector = f64_vector_bytes(acquisition_count)?;
    let input_bytes = checked_sum(&[matrix, vector])?;

    // Selected covariance/days/observations/diagonal/relative shape, oracle
    // covariance, and the retained plug-in covariance coexist through bootstrap.
    let retained_fit_bytes =
        checked_sum(&[matrix, matrix, matrix, vector, vector, vector, vector])?;

    // A retained best candidate coexists with an active candidate. During GLS,
    // the active candidate owns one Cholesky and its inverse; its three solve
    // vectors are the larger vector phase.
    let inverse_peak = checked_sum(&[matrix, matrix, vector, vector, vector])?;
    let transformed_peak = checked_sum(&[matrix, matrix, vector, vector, vector, vector])?;
    let profile_optimizer_peak_bytes =
        checked_sum(&[matrix, matrix, inverse_peak.max(transformed_peak)])?;

    let faer_condition = faer_condition_workspace_bytes(acquisition_count)?;
    // The previous best covariance remains live while the final candidate is
    // conditioned, so both matrices precede the linked Faer workspace.
    let faer_condition_peak_bytes = checked_sum(&[matrix, matrix, faer_condition])?;

    // Bootstrap retains its Cholesky, simulated vectors, and slope reservoir
    // while a nested profile fit executes.
    let bootstrap_driver = checked_sum(&[
        matrix,
        vector,
        vector,
        vector,
        u64::try_from(bootstrap_replicates)
            .ok()?
            .checked_mul(std::mem::size_of::<f64>() as u64)?,
    ])?;
    let bootstrap_peak_bytes = bootstrap_driver
        .checked_add(profile_optimizer_peak_bytes.max(faer_condition_peak_bytes))?;
    let active_peak = profile_optimizer_peak_bytes
        .max(bootstrap_peak_bytes)
        .max(faer_condition_peak_bytes);
    let total_bytes = input_bytes
        .checked_add(retained_fit_bytes)?
        .checked_add(active_peak)?;
    Some(TemporalCovarianceWorkspaceComposition {
        input_bytes,
        retained_fit_bytes,
        profile_optimizer_peak_bytes,
        bootstrap_peak_bytes,
        faer_condition_peak_bytes,
        total_bytes,
    })
}

fn nested_f64_matrix_bytes(dimension: usize) -> Option<u64> {
    let capacity = vector_capacity_upper_bound(dimension)?;
    let values = capacity
        .checked_mul(capacity)?
        .checked_mul(std::mem::size_of::<f64>() as u64)?;
    let row_headers = capacity.checked_mul(std::mem::size_of::<Vec<f64>>() as u64)?;
    values.checked_add(row_headers)
}

fn f64_vector_bytes(length: usize) -> Option<u64> {
    vector_capacity_upper_bound(length)?.checked_mul(std::mem::size_of::<f64>() as u64)
}

fn vector_capacity_upper_bound(length: usize) -> Option<u64> {
    if length == 0 {
        Some(0)
    } else {
        u64::try_from(length.checked_next_power_of_two()?).ok()
    }
}

fn checked_sum(values: &[u64]) -> Option<u64> {
    values
        .iter()
        .try_fold(0_u64, |total, value| total.checked_add(*value))
}

fn faer_condition_workspace_bytes(dimension: usize) -> Option<u64> {
    let mut probe = Mat::<f64>::new();
    probe.reserve_exact(1, 0);
    let alignment = u64::try_from(probe.row_capacity()).ok()?.max(1);
    let rows = u64::try_from(dimension).ok()?;
    let padded_rows = rows.checked_add(alignment - 1)? / alignment * alignment;
    let matrix = padded_rows
        .checked_mul(rows)?
        .checked_mul(std::mem::size_of::<f64>() as u64)?;
    let eigenvalues = f64_vector_bytes(dimension)?;
    let scratch = u64::try_from(
        compute_hermitian_evd_req::<f64>(
            dimension,
            ComputeVectors::No,
            get_global_parallelism(),
            Default::default(),
        )
        .ok()?
        .size_bytes(),
    )
    .ok()?;
    checked_sum(&[matrix, eigenvalues, scratch])
}
/// Frozen minimum successful bootstrap count from the #53 preregistration.
pub const COMPLETE_REFIT_BOOTSTRAP_MINIMUM_SUCCESSES: usize = 198;
/// Stable identity of the preregistered #53 estimate candidate.
pub const COMPLETE_REFIT_BOOTSTRAP_METHOD: &str = "complete_refit_bootstrap";
/// Stable schema/method version of the preregistered #53 estimate candidate.
pub const COMPLETE_REFIT_BOOTSTRAP_METHOD_VERSION: u16 = 1;
/// Production scalar selected after synthetic validation and release-resource qualification.
pub const REML_COVARIANCE_PARAMETER_ADJUSTED_SCALAR_METHOD: &str =
    "reml_covariance_parameter_adjusted_scalar";
/// Production adjusted-scalar method version.
pub const REML_COVARIANCE_PARAMETER_ADJUSTED_SCALAR_METHOD_VERSION: u16 = 2;

type SubsetSeries = (Vec<f64>, Vec<f64>, Vec<Vec<f64>>);

/// Stable failure/status codes for temporal covariance fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalInferenceStatus {
    /// All requested point-estimate and validation computations completed.
    Evaluated,
    /// Fewer than the configured minimum number of post-gauge observations remain.
    InsufficientDates,
    /// Input dates are not finite and strictly increasing.
    DatesNotStrictlyIncreasing,
    /// Acquisition zero is absent, non-finite, or not origin anchored.
    GaugeMissing,
    /// Acquisition zero is present but its observation is not exactly zero.
    GaugeNotZero,
    /// The origin-anchored slope design has no rank.
    DesignRankDeficient,
    /// A covariance or design matrix exceeds the configured condition bound.
    DesignIllConditioned,
    /// A supplied covariance contains a non-finite or asymmetric value.
    CovarianceNonfinite,
    /// The total covariance is not positive definite.
    TotalCovarianceNotPositiveDefinite,
    /// The fitted correlation or process variance is at a configured boundary.
    CovarianceParameterAtBoundary,
    /// The fitted correlation is at its lower bound.
    RhoLowerBoundary,
    /// The fitted correlation is at its upper bound.
    RhoUpperBoundary,
    /// The fitted process variance is at its lower bound.
    ProcessVarianceLowerBoundary,
    /// The fitted process variance is at its upper bound.
    ProcessVarianceUpperBoundary,
    /// Too few complete-refit bootstrap replicates succeeded.
    BootstrapInsufficientSuccess,
    /// The requested cadence is outside the preregistered supported predicate.
    UnsupportedCadence,
    /// The covariance optimizer did not converge within its frozen iterations.
    OptimizerNonconverged,
    /// The likelihood is too flat to identify covariance parameters.
    WeakParameterIdentification,
    /// Legacy intercept-plus-slope WLS is reported only as a non-comparable diagnostic.
    LegacyNonComparable,
    /// A validation-only comparator was intentionally not executed by the selected product path.
    DiagnosticNotComputed,
}

/// Continuous-time AR(1) and profile-fit options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalCovarianceOptions {
    /// Correlation is evaluated at this reference lag in days.
    pub reference_lag_days: f64,
    /// Lower fitted correlation bound.
    pub rho_min: f64,
    /// Exclusive upper fitted correlation bound.
    pub rho_max: f64,
    /// Known generating correlation for the oracle comparator.
    pub oracle_rho: f64,
    /// Known generating residual process variance for the oracle comparator.
    pub oracle_process_variance: f64,
    /// Maximum allowed covariance condition number.
    pub condition_limit: f64,
    /// Minimum post-gauge dates needed for a slope fit.
    pub minimum_dates: usize,
    /// Number of complete-refit bootstrap replicates.
    pub bootstrap_replicates: usize,
    /// Minimum successful bootstrap replicates required for a validation interval.
    pub bootstrap_minimum_successes: usize,
    /// Deterministic bootstrap seed.
    pub bootstrap_seed: u64,
    /// Frozen maximum profile interval expansions.
    pub profile_max_expansions: usize,
    /// Frozen maximum profile endpoint iterations.
    pub profile_max_iterations: usize,
    /// Frozen maximum covariance optimizer coordinate iterations.
    pub optimizer_max_iterations: usize,
    /// Frozen objective convergence tolerance.
    pub optimizer_tolerance: f64,
    /// Frozen lower process-variance bound relative to the residual scale.
    pub process_variance_min_ratio: f64,
    /// Frozen upper process-variance bound relative to the residual scale.
    pub process_variance_max_ratio: f64,
    /// Minimum finite-difference profile curvature required for identification.
    pub minimum_profile_curvature: f64,
    /// Minimum supported adjacent acquisition gap in days.
    pub minimum_gap_days: f64,
    /// Maximum supported adjacent acquisition gap in days.
    pub maximum_gap_days: f64,
}

impl Default for TemporalCovarianceOptions {
    fn default() -> Self {
        Self {
            reference_lag_days: 12.0,
            rho_min: 0.0,
            rho_max: 0.98,
            oracle_rho: 0.3,
            oracle_process_variance: 1.0,
            condition_limit: 1e12,
            minimum_dates: 12,
            bootstrap_replicates: 200,
            bootstrap_minimum_successes: 198,
            bootstrap_seed: 0x53_2026,
            profile_max_expansions: 12,
            profile_max_iterations: 48,
            optimizer_max_iterations: 12,
            optimizer_tolerance: 1e-4,
            process_variance_min_ratio: 1e-6,
            process_variance_max_ratio: 1e6,
            minimum_profile_curvature: 1e-6,
            minimum_gap_days: 4.0,
            maximum_gap_days: 36.0,
        }
    }
}

/// Raw adjacent-residual correlation diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawCorrelationDiagnostics {
    /// Unclamped residual correlation, absent when fewer than three pairs exist.
    pub rho: Option<f64>,
    /// Number of adjacent residual pairs used.
    pub pair_count: usize,
    /// Minimum elapsed time among adjacent residual pairs.
    pub minimum_gap_days: Option<f64>,
    /// Median elapsed time among adjacent residual pairs.
    pub median_gap_days: Option<f64>,
    /// Maximum elapsed time among adjacent residual pairs.
    pub maximum_gap_days: Option<f64>,
}

/// A validation interval, not a production uncertainty product.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ValidationInterval {
    /// Lower empirical interval endpoint.
    pub lower: f64,
    /// Upper empirical interval endpoint.
    pub upper: f64,
    /// Number of successful complete-refit replicates.
    pub successful_replicates: usize,
}

/// Point and interval diagnostics for one validation comparator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparatorDiagnostics {
    /// Comparator point estimate in units per year.
    pub point_estimate: Option<f64>,
    /// Diagnostic standard error, never a production raster field.
    pub standard_error_diagnostic: Option<f64>,
    /// Symmetric 68% validation interval.
    pub interval_68: Option<ValidationInterval>,
    /// Symmetric 90% validation interval.
    pub interval_90: Option<ValidationInterval>,
    /// Symmetric 95% validation interval.
    pub interval_95: Option<ValidationInterval>,
    /// Width of the 68% validation interval.
    pub width_68: Option<f64>,
    /// Width of the 90% validation interval.
    pub width_90: Option<f64>,
    /// Width of the 95% validation interval.
    pub width_95: Option<f64>,
    /// Stable comparator disposition.
    pub status: TemporalInferenceStatus,
    /// Number of attempted resamples for this comparator.
    pub attempted_replicates: usize,
    /// Number of successful resamples for this comparator.
    pub successful_replicates: usize,
}

/// Scalar temporal candidates eligible for a promotion-neutral resource probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalScalarCandidateMethod {
    /// Plug-in GLS after profiled REML covariance fitting.
    PluginGlsReml,
    /// REML GLS with covariance-parameter scalar variance adjustment.
    RemlCovarianceParameterAdjustedScalar,
    /// ML slope-profile interval reduced to its registered scalar diagnostic.
    SlopeProfileLikelihoodMl,
}

/// One non-promoting scalar-candidate evaluation without bootstrap execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalScalarCandidateProbe {
    /// Exact candidate evaluated by the probe.
    pub method: TemporalScalarCandidateMethod,
    /// Candidate point, scalar standard error, intervals, and status.
    pub comparator: ComparatorDiagnostics,
    /// Retained post-gauge observation count.
    pub valid_date_count: usize,
    /// Origin-anchored design rank.
    pub rank: usize,
    /// Residual degrees of freedom.
    pub degrees_of_freedom: usize,
    /// Fitted continuous-time correlation when covariance fitting succeeded.
    pub fitted_rho: Option<f64>,
    /// Fitted residual process variance when covariance fitting succeeded.
    pub fitted_process_variance: Option<f64>,
    /// Fitted total-covariance condition number when available.
    pub covariance_condition_number: Option<f64>,
    /// Bootstrap attempts, fixed to zero for this non-promoting probe.
    pub bootstrap_attempts: usize,
}

/// OLS and known-parameter oracle GLS comparators used by synthetic validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalValidationBaselineComparators {
    /// Origin-anchored OLS comparator.
    pub ols: ComparatorDiagnostics,
    /// GLS comparator using the frozen generating covariance parameters.
    pub oracle_gls: ComparatorDiagnostics,
}

/// Evaluate the synthetic-validation OLS and oracle GLS baselines without profiling or bootstrap.
///
/// # Errors
/// Returns the same fail-closed input, cadence, design, and covariance statuses as the full
/// validation fit before its profiled estimators execute.
pub fn temporal_validation_baseline_comparators(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
) -> Result<TemporalValidationBaselineComparators, TemporalInferenceStatus> {
    validate_dates(days)?;
    if days.first().copied() != Some(0.0) {
        return Err(TemporalInferenceStatus::GaugeMissing);
    }
    let (selected_days, selected_y, selected_c) =
        subset_origin_anchored_covariance(days, observations, difference_covariance)?;
    validate_supported_cadence(&days[1..], options)?;
    if selected_days.len() < options.minimum_dates {
        return Err(TemporalInferenceStatus::InsufficientDates);
    }
    let information = dot(&selected_days, &selected_days);
    if !information.is_finite() || information <= 0.0 {
        return Err(TemporalInferenceStatus::DesignRankDeficient);
    }
    let degrees_of_freedom = selected_days.len().saturating_sub(1);
    let ols_slope = ols_slope(&selected_days, &selected_y)?;
    let ols_residual_scale = selected_y
        .iter()
        .zip(&selected_days)
        .map(|(value, day)| (value - ols_slope * day).powi(2))
        .sum::<f64>()
        / degrees_of_freedom.max(1) as f64;
    let oracle_covariance = total_difference_covariance(
        &selected_c,
        &selected_days,
        options.oracle_process_variance,
        options.oracle_rho,
        options.reference_lag_days,
    )?;
    let oracle = gls_fit(
        &selected_days,
        &selected_y,
        &oracle_covariance,
        options.condition_limit,
        true,
    )?;
    Ok(TemporalValidationBaselineComparators {
        ols: normal_comparator(
            ols_slope,
            (ols_residual_scale / information).sqrt(),
            TemporalInferenceStatus::Evaluated,
        ),
        oracle_gls: normal_comparator(
            oracle.slope,
            oracle.information_variance.sqrt(),
            TemporalInferenceStatus::Evaluated,
        ),
    })
}

/// Point estimates and validation evidence for one origin-anchored series.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalCovarianceFit {
    /// Overall disposition of the fit.
    pub status: TemporalInferenceStatus,
    /// OLS origin-anchored slope in units per day.
    pub ols_slope: Option<f64>,
    /// Oracle GLS slope using the known generating covariance.
    pub oracle_gls_slope: Option<f64>,
    /// Plug-in profiled GLS slope.
    pub plugin_gls_slope: Option<f64>,
    /// Adjusted/profile comparator slope. It is separate from the bootstrap interval.
    pub adjusted_profile_slope: Option<f64>,
    /// Observed plug-in slope paired with complete-refit bootstrap intervals.
    pub bootstrap_slope: Option<f64>,
    /// Validation-only complete-refit interval.
    pub bootstrap_interval: Option<ValidationInterval>,
    /// Fitted continuous-time correlation parameter.
    pub fitted_rho: Option<f64>,
    /// Fitted residual process variance parameter.
    pub fitted_process_variance: Option<f64>,
    /// Active fitted nuisance boundary handled by constrained inference.
    pub fitted_parameter_active_set: Option<TemporalInferenceStatus>,
    /// Unclamped residual diagnostics.
    pub raw_correlation: RawCorrelationDiagnostics,
    /// Number of retained post-gauge observations.
    pub valid_date_count: usize,
    /// Rank of the origin-anchored slope design.
    pub rank: usize,
    /// Residual degrees of freedom for the one-parameter slope.
    pub degrees_of_freedom: usize,
    /// Estimated condition number of the fitted covariance.
    pub covariance_condition_number: Option<f64>,
    /// OLS point/interval diagnostics.
    pub ols: ComparatorDiagnostics,
    /// Oracle GLS point/interval diagnostics.
    pub oracle_gls: ComparatorDiagnostics,
    /// Diagonal conditional-WLS comparator diagnostics.
    pub conditional_wls: ComparatorDiagnostics,
    /// Scalar effective-N diagnostic comparator.
    pub scalar_effective_n: ComparatorDiagnostics,
    /// Plug-in profiled GLS point/interval diagnostics.
    pub plugin_gls: ComparatorDiagnostics,
    /// Scalar covariance-parameter adjustment using the REML nuisance curvature.
    pub adjusted_scalar: ComparatorDiagnostics,
    /// Profile-likelihood comparator diagnostics.
    pub adjusted_profile: ComparatorDiagnostics,
    /// Complete-refit bootstrap diagnostics.
    pub complete_refit_bootstrap: ComparatorDiagnostics,
    /// Number of bootstrap attempts.
    pub bootstrap_attempts: usize,
    /// Number of successful bootstrap refits.
    pub bootstrap_successes: usize,
}

/// Factor-native REML results reused by the dense validation diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalCovariancePrefit {
    /// Selected plug-in slope in units per day.
    pub plugin_slope_per_day: f64,
    /// Factor-native plug-in comparator.
    pub plugin_gls: ComparatorDiagnostics,
    /// Factor-native covariance-parameter adjusted comparator.
    pub adjusted_scalar: ComparatorDiagnostics,
    /// Fitted continuous-time correlation parameter.
    pub fitted_rho: f64,
    /// Fitted residual process variance parameter.
    pub fitted_process_variance: f64,
    /// Active fitted nuisance boundary handled by constrained inference.
    pub fitted_parameter_active_set: Option<TemporalInferenceStatus>,
    /// Exact condition number or certified upper bound from the factor path.
    pub covariance_condition_number: Option<f64>,
}

/// Fail-closed disposition of the preregistered complete-refit estimate candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompleteRefitBootstrapEstimateStatus {
    /// The frozen complete-refit candidate is numerically complete.
    Evaluated,
    /// The overall temporal covariance fit did not evaluate.
    FitNotEvaluated,
    /// The selected complete-refit comparator did not evaluate.
    ComparatorNotEvaluated,
    /// Runtime options do not match the frozen preregistration.
    FrozenConfigurationMismatch,
    /// Fit and comparator bootstrap counters are not the exact frozen accounting.
    BootstrapAccountingMismatch,
    /// Successful refits do not meet both frozen and 99% requirements.
    BootstrapInsufficientSuccess,
    /// The selected point estimate or standard error is absent, non-finite, or invalid.
    InvalidEstimate,
}

/// Supported-cadence disposition carried by the complete-refit candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompleteRefitBootstrapCadenceStatus {
    /// The fit reached the frozen supported-cadence predicate.
    Supported,
    /// The scheduled acquisition cadence was outside the frozen predicate.
    Unsupported,
    /// Cadence could not be classified before another input failure.
    Unavailable,
}

/// Promotion-neutral, persistable complete-refit estimate candidate.
///
/// Calibration and product-promotion evidence are intentionally absent. The
/// workflow evidence validator must add those claims after validating the
/// immutable #52/#54/#53 receipt bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompleteRefitBootstrapEstimate {
    /// Numeric selection disposition.
    pub status: CompleteRefitBootstrapEstimateStatus,
    /// Exact upstream fit disposition.
    pub fit_status: TemporalInferenceStatus,
    /// Selected origin-anchored slope in units per year.
    pub slope_per_year: Option<f64>,
    /// Complete-refit bootstrap standard error in units per year.
    pub standard_error_per_year: Option<f64>,
    /// Retained post-gauge date count.
    pub valid_date_count: usize,
    /// Origin-anchored design rank.
    pub rank: usize,
    /// Residual degrees of freedom.
    pub degrees_of_freedom: usize,
    /// Minimum, median, and maximum retained cadence in days.
    pub cadence_days: [Option<f64>; 3],
    /// Frozen supported-cadence disposition.
    pub cadence_status: CompleteRefitBootstrapCadenceStatus,
    /// Unclamped adjacent residual correlation.
    pub raw_rho: Option<f64>,
    /// Fitted continuous-time correlation.
    pub fitted_rho: Option<f64>,
    /// Fitted residual process variance.
    pub fitted_process_variance: Option<f64>,
    /// Active fitted nuisance boundary handled by constrained inference.
    pub fitted_parameter_active_set: Option<TemporalInferenceStatus>,
    /// Stable selected-method identity.
    pub method: String,
    /// Stable selected-method version.
    pub method_version: u16,
    /// Fitted total covariance condition number.
    pub condition_number: Option<f64>,
    /// Complete-refit bootstrap attempts.
    pub bootstrap_attempts: usize,
    /// Successful complete-refit bootstrap attempts.
    pub bootstrap_successes: usize,
}

/// Validated lower-case SHA-256 digest used by validation provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    /// Parse one exact lower-case hexadecimal SHA-256 digest.
    pub fn new(value: impl Into<String>) -> Result<Self, &'static str> {
        let value = value.into();
        if value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            Ok(Self(value))
        } else {
            Err("SHA-256 digest must be 64 lower-case hexadecimal characters")
        }
    }

    /// Borrow the canonical hexadecimal digest.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Frozen validation-only approximation identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalCovarianceApproximation {
    /// Exact source-DAG contraction.
    Exact,
    /// Production compressed-JVP source-DAG contraction.
    CompressedJvp,
}

/// Frozen validation scope; this type cannot represent product promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalValidationScope {
    /// Synthetic validation execution.
    SyntheticValidation,
    /// Held-out field-validation execution.
    FieldValidation,
}

/// Reference selection and replay geometry bound into provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalReferenceProvenance {
    /// Stable geometry identity.
    pub geometry_id: String,
    /// Stable window identity.
    pub window_id: String,
    /// Target/reference support overlap fraction.
    pub overlap_fraction: f64,
    /// Target/reference distance in pixels.
    pub distance_pixels: f64,
    /// Maximum sequential replay ancestry depth.
    pub sequential_depth: usize,
    /// Exact or compressed-JVP replay identity.
    pub approximation: TemporalCovarianceApproximation,
}

/// Validation-only F53-03 provenance sidecar fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalCovarianceProvenance {
    /// Stable sidecar schema.
    pub schema: String,
    /// Estimator identity.
    pub estimator: String,
    /// Estimator implementation version.
    pub estimator_version: String,
    /// Retained post-gauge date count.
    pub valid_date_count: usize,
    /// Origin-anchored design rank.
    pub rank: usize,
    /// Residual degrees of freedom.
    pub degrees_of_freedom: usize,
    /// Minimum/median/maximum retained cadence in days.
    pub cadence_days: [Option<f64>; 3],
    /// Unclamped adjacent residual correlation.
    pub raw_rho: Option<f64>,
    /// Fitted continuous-time correlation.
    pub fitted_rho: Option<f64>,
    /// Fitted process variance.
    pub fitted_process_variance: Option<f64>,
    /// Active fitted nuisance boundary handled by constrained inference.
    pub fitted_parameter_active_set: Option<TemporalInferenceStatus>,
    /// #52 replay/input receipt SHA-256.
    pub issue52_receipt_sha256: Sha256Digest,
    /// #54 direct difference-factor receipt SHA-256.
    pub issue54_receipt_sha256: Sha256Digest,
    /// Typed reference geometry and replay identity.
    pub reference: TemporalReferenceProvenance,
    /// Fitted total covariance condition number.
    pub condition_number: Option<f64>,
    /// Scope identity for the validation series.
    pub scope: TemporalValidationScope,
    /// Complete-refit bootstrap attempts.
    pub bootstrap_attempts: usize,
    /// Complete-refit bootstrap successes.
    pub bootstrap_successes: usize,
    /// Approximation identity, if any.
    /// Validation receipt SHA-256.
    pub validation_receipt_sha256: Sha256Digest,
    /// Hash of the exact estimator inputs.
    pub estimator_input_sha256: Sha256Digest,
    /// Required successful-bootstrap fraction.
    pub bootstrap_minimum_success_fraction: f64,
    /// Name of the selected validation comparator.
    pub selected_method: String,
}

/// Inputs that bind an F53-03 provenance sidecar to #52/#54 and reference geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalCovarianceProvenanceInputs {
    /// #52 replay/input receipt SHA-256.
    pub issue52_receipt_sha256: Sha256Digest,
    /// #54 direct difference-factor receipt SHA-256.
    pub issue54_receipt_sha256: Sha256Digest,
    /// Typed reference geometry and replay identity.
    pub reference: TemporalReferenceProvenance,
    /// Scope identity.
    pub scope: TemporalValidationScope,
    /// Validation receipt SHA-256.
    pub validation_receipt_sha256: Sha256Digest,
    /// Hash of the exact estimator inputs.
    pub estimator_input_sha256: Sha256Digest,
    /// Name of the selected validation comparator.
    pub selected_method: String,
}

/// Construct the preregistered continuous-time AR(1) correlation matrix.
///
/// The exponent uses elapsed time, not retained-date index, so missing dates
/// and irregular cadence do not silently become equally spaced observations.
pub fn continuous_time_ar1_correlation(
    days: &[f64],
    rho_at_reference_lag: f64,
    reference_lag_days: f64,
) -> Result<Vec<Vec<f64>>, TemporalInferenceStatus> {
    if !reference_lag_days.is_finite() || reference_lag_days <= 0.0 {
        return Err(TemporalInferenceStatus::UnsupportedCadence);
    }
    if !rho_at_reference_lag.is_finite() || !(0.0..1.0).contains(&rho_at_reference_lag) {
        return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
    }
    validate_dates(days)?;
    Ok((0..days.len())
        .map(|row| {
            (0..days.len())
                .map(|column| {
                    if row == column {
                        1.0
                    } else if rho_at_reference_lag == 0.0 {
                        0.0
                    } else {
                        rho_at_reference_lag
                            .powf((days[row] - days[column]).abs() / reference_lag_days)
                    }
                })
                .collect()
        })
        .collect())
}

/// Compute the relative standard-deviation shape `D` from positive covariance diagonals.
pub fn relative_standard_deviation_shape(
    diagonal: &[f64],
) -> Result<Vec<f64>, TemporalInferenceStatus> {
    if diagonal.is_empty()
        || diagonal
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let geometric_mean =
        diagonal.iter().map(|value| value.ln()).sum::<f64>() / diagonal.len() as f64;
    let scale = geometric_mean.exp();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    Ok(diagonal
        .iter()
        .map(|value| (value / scale).sqrt())
        .collect())
}

/// Select finite observations and the matching covariance rows/columns.
pub fn subset_origin_anchored_covariance(
    days: &[f64],
    observations: &[f64],
    covariance: &[Vec<f64>],
) -> Result<SubsetSeries, TemporalInferenceStatus> {
    validate_dates(days)?;
    if observations.len() != days.len()
        || covariance.len() != days.len()
        || covariance.iter().any(|row| row.len() != days.len())
    {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    if observations.first().is_none_or(|value| !value.is_finite()) {
        return Err(TemporalInferenceStatus::GaugeMissing);
    }
    if observations[0] != 0.0 {
        return Err(TemporalInferenceStatus::GaugeNotZero);
    }
    let selected: Vec<usize> = (1..days.len())
        .filter(|&index| observations[index].is_finite())
        .collect();
    if selected.is_empty() {
        return Err(TemporalInferenceStatus::InsufficientDates);
    }
    let selected_days = selected.iter().map(|&index| days[index]).collect();
    let selected_observations = selected.iter().map(|&index| observations[index]).collect();
    let selected_covariance = selected
        .iter()
        .map(|&row| {
            selected
                .iter()
                .map(|&column| covariance[row][column])
                .collect()
        })
        .collect();
    Ok((selected_days, selected_observations, selected_covariance))
}

/// Build `C54_delta + sigma² D R(rho) D` for a retained post-gauge series.
pub fn total_difference_covariance(
    difference_covariance: &[Vec<f64>],
    days: &[f64],
    process_variance: f64,
    rho_at_reference_lag: f64,
    reference_lag_days: f64,
) -> Result<Vec<Vec<f64>>, TemporalInferenceStatus> {
    validate_square_covariance(difference_covariance)?;
    if difference_covariance.len() != days.len()
        || !process_variance.is_finite()
        || process_variance < 0.0
    {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let diagonal: Vec<f64> = difference_covariance
        .iter()
        .enumerate()
        .map(|(index, row)| row[index])
        .collect();
    let shape = relative_standard_deviation_shape(&diagonal)?;
    let correlation =
        continuous_time_ar1_correlation(days, rho_at_reference_lag, reference_lag_days)?;
    Ok((0..days.len())
        .map(|row| {
            (0..days.len())
                .map(|column| {
                    difference_covariance[row][column]
                        + process_variance * shape[row] * correlation[row][column] * shape[column]
                })
                .collect()
        })
        .collect())
}

/// Classify fitted covariance parameters against frozen lower and upper bounds.
#[must_use]
pub fn temporal_parameter_boundary_status(
    rho: f64,
    process_variance: f64,
    rho_bounds: [f64; 2],
    process_variance_bounds: [f64; 2],
    tolerance_fraction: f64,
) -> Option<TemporalInferenceStatus> {
    nuisance_parameter_active_set(
        rho,
        process_variance,
        rho_bounds,
        process_variance_bounds,
        tolerance_fraction,
    )
    .0
}

pub(super) fn nuisance_parameter_active_set(
    rho: f64,
    process_variance: f64,
    rho_bounds: [f64; 2],
    process_variance_bounds: [f64; 2],
    tolerance_fraction: f64,
) -> (Option<TemporalInferenceStatus>, bool, bool) {
    let tolerance = tolerance_fraction.clamp(1e-8, 1e-4);
    let rho_tolerance = tolerance * (rho_bounds[1] - rho_bounds[0]);
    let log_min = process_variance_bounds[0].ln();
    let log_max = process_variance_bounds[1].ln();
    let log_tolerance = tolerance * (log_max - log_min);
    let rho_lower = rho <= rho_bounds[0] + rho_tolerance;
    let rho_upper = rho >= rho_bounds[1] - rho_tolerance;
    let variance_lower = process_variance.ln() <= log_min + log_tolerance;
    let variance_upper = process_variance.ln() >= log_max - log_tolerance;
    let status = if rho_lower {
        Some(TemporalInferenceStatus::RhoLowerBoundary)
    } else if rho_upper {
        Some(TemporalInferenceStatus::RhoUpperBoundary)
    } else if variance_lower {
        Some(TemporalInferenceStatus::ProcessVarianceLowerBoundary)
    } else if variance_upper {
        Some(TemporalInferenceStatus::ProcessVarianceUpperBoundary)
    } else {
        None
    };
    (
        status,
        rho_lower || rho_upper,
        variance_lower || variance_upper,
    )
}

/// Fit the OLS/oracle/plugin/profile/bootstrap comparator set.
///
/// `observations[0]` is the exact acquisition-zero gauge and must be finite;
/// missing post-gauge dates are represented by `NaN`. `difference_covariance`
/// is consumed directly and must already be a same-frame #54 difference factor.
pub fn fit_temporal_covariance(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
) -> TemporalCovarianceFit {
    fit_temporal_covariance_impl(
        days,
        observations,
        difference_covariance,
        options,
        None,
        None,
    )
}

/// Fit dense validation diagnostics while reusing selected factor-native REML results.
#[must_use]
pub fn fit_temporal_covariance_from_prefit(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
    prefit: &TemporalCovariancePrefit,
) -> TemporalCovarianceFit {
    fit_temporal_covariance_impl(
        days,
        observations,
        difference_covariance,
        options,
        Some(prefit),
        None,
    )
}

/// Fit dense validation diagnostics while reusing factor-native REML results and the retained
/// difference-covariance factor for the fixed-slope ML profile.
///
/// `persisted_factor` contains one zero gauge row followed by the retained finite post-gauge rows
/// in the same order used by `prefit`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fit_temporal_covariance_from_factor_prefit(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    persisted_factor: &[f64],
    persisted_maximum_rank: usize,
    realized_rank: usize,
    options: &TemporalCovarianceOptions,
    prefit: &TemporalCovariancePrefit,
) -> TemporalCovarianceFit {
    fit_temporal_covariance_impl(
        days,
        observations,
        difference_covariance,
        options,
        Some(prefit),
        Some(FactorProfileInput {
            persisted_factor,
            persisted_maximum_rank,
            realized_rank,
        }),
    )
}

#[derive(Clone, Copy)]
struct FactorProfileInput<'a> {
    persisted_factor: &'a [f64],
    persisted_maximum_rank: usize,
    realized_rank: usize,
}

struct PreparedFactorProfileInput {
    compact_factor: Vec<f64>,
    maximum_rank: usize,
    realized_rank: usize,
}

#[allow(clippy::too_many_lines)]
fn validate_temporal_covariance_prefit(
    days: &[f64],
    observations: &[f64],
    options: &TemporalCovarianceOptions,
    prefit: &TemporalCovariancePrefit,
) -> Result<NuisanceBounds, TemporalInferenceStatus> {
    let bounds = nuisance_bounds(days, observations, options)?;
    let condition = prefit
        .covariance_condition_number
        .ok_or(TemporalInferenceStatus::CovarianceNonfinite)?;
    if !prefit.plugin_slope_per_day.is_finite()
        || !prefit.fitted_rho.is_finite()
        || !prefit.fitted_process_variance.is_finite()
        || !condition.is_finite()
        || condition < 1.0
    {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    if condition > options.condition_limit {
        return Err(TemporalInferenceStatus::DesignIllConditioned);
    }
    if prefit.fitted_rho < bounds.rho_lower
        || prefit.fitted_rho > bounds.rho_upper
        || prefit.fitted_process_variance <= 0.0
        || prefit.fitted_process_variance.ln() < bounds.log_variance_lower
        || prefit.fitted_process_variance.ln() > bounds.log_variance_upper
    {
        return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
    }
    let expected_point = prefit.plugin_slope_per_day * DAYS_PER_YEAR;
    let evaluated_matches = |comparator: &ComparatorDiagnostics, multipliers: [f64; 3]| {
        let Some(point) = comparator.point_estimate else {
            return false;
        };
        let Some(standard_error) = comparator.standard_error_diagnostic else {
            return false;
        };
        let scale = point.abs().max(expected_point.abs()).max(1.0);
        let interval_widths = [
            (comparator.interval_68, comparator.width_68, multipliers[0]),
            (comparator.interval_90, comparator.width_90, multipliers[1]),
            (comparator.interval_95, comparator.width_95, multipliers[2]),
        ];
        comparator.status == TemporalInferenceStatus::Evaluated
            && point.is_finite()
            && (point - expected_point).abs() <= 1e-10 * scale
            && standard_error.is_finite()
            && standard_error > 0.0
            && comparator.attempted_replicates == 0
            && comparator.successful_replicates == 0
            && interval_widths
                .into_iter()
                .all(|(interval, width, multiplier)| {
                    interval.zip(width).is_some_and(|(interval, width)| {
                        let expected_lower = point - multiplier * standard_error;
                        let expected_upper = point + multiplier * standard_error;
                        let expected_width = 2.0 * multiplier * standard_error;
                        let interval_scale = expected_lower
                            .abs()
                            .max(expected_upper.abs())
                            .max(interval.lower.abs())
                            .max(interval.upper.abs())
                            .max(1.0);
                        let width_scale = expected_width.abs().max(width.abs()).max(1.0);
                        interval.lower.is_finite()
                            && interval.upper.is_finite()
                            && interval.successful_replicates == 0
                            && (interval.lower - expected_lower).abs() <= 1e-10 * interval_scale
                            && (interval.upper - expected_upper).abs() <= 1e-10 * interval_scale
                            && width.is_finite()
                            && width > 0.0
                            && (width - expected_width).abs() <= 1e-10 * width_scale
                    })
                })
    };
    let normal = Normal::new(0.0, 1.0).map_err(|_| TemporalInferenceStatus::CovarianceNonfinite)?;
    let normal_multipliers = [
        normal.inverse_cdf(0.84),
        normal.inverse_cdf(0.95),
        normal.inverse_cdf(0.975),
    ];
    let student = StudentsT::new(0.0, 1.0, days.len().saturating_sub(1) as f64)
        .map_err(|_| TemporalInferenceStatus::CovarianceNonfinite)?;
    let student_multipliers = [
        student.inverse_cdf(0.84),
        student.inverse_cdf(0.95),
        student.inverse_cdf(0.975),
    ];
    let adjusted_scalar_valid =
        if prefit.adjusted_scalar.status == TemporalInferenceStatus::Evaluated {
            evaluated_matches(&prefit.adjusted_scalar, student_multipliers)
        } else {
            prefit.adjusted_scalar.point_estimate.is_none()
                && prefit.adjusted_scalar.standard_error_diagnostic.is_none()
                && prefit.adjusted_scalar.interval_68.is_none()
                && prefit.adjusted_scalar.interval_90.is_none()
                && prefit.adjusted_scalar.interval_95.is_none()
                && prefit.adjusted_scalar.width_68.is_none()
                && prefit.adjusted_scalar.width_90.is_none()
                && prefit.adjusted_scalar.width_95.is_none()
                && prefit.adjusted_scalar.attempted_replicates == 0
                && prefit.adjusted_scalar.successful_replicates == 0
        };
    if !evaluated_matches(&prefit.plugin_gls, normal_multipliers) || !adjusted_scalar_valid {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let expected_active_set = temporal_parameter_boundary_status(
        prefit.fitted_rho,
        prefit.fitted_process_variance,
        [bounds.rho_lower, bounds.rho_upper],
        [
            bounds.log_variance_lower.exp(),
            bounds.log_variance_upper.exp(),
        ],
        options.optimizer_tolerance * 0.01,
    );
    if prefit.fitted_parameter_active_set != expected_active_set {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    Ok(bounds)
}

#[allow(clippy::too_many_lines)]
fn fit_temporal_covariance_impl(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
    prefit: Option<&TemporalCovariancePrefit>,
    factor_profile: Option<FactorProfileInput<'_>>,
) -> TemporalCovarianceFit {
    let initial_raw_correlation = raw_adjacent_correlation(days, observations);
    let empty = |status| TemporalCovarianceFit {
        status,
        ols_slope: None,
        oracle_gls_slope: None,
        plugin_gls_slope: None,
        adjusted_profile_slope: None,
        bootstrap_slope: None,
        bootstrap_interval: None,
        fitted_rho: None,
        fitted_process_variance: None,
        fitted_parameter_active_set: None,
        raw_correlation: initial_raw_correlation.clone(),
        valid_date_count: 0,
        rank: 0,
        degrees_of_freedom: 0,
        covariance_condition_number: None,
        ols: empty_comparator(status),
        oracle_gls: empty_comparator(status),
        conditional_wls: empty_comparator(status),
        scalar_effective_n: empty_comparator(status),
        plugin_gls: empty_comparator(status),
        adjusted_scalar: empty_comparator(status),
        adjusted_profile: empty_comparator(status),
        complete_refit_bootstrap: empty_comparator(status),
        bootstrap_attempts: 0,
        bootstrap_successes: 0,
    };
    if let Err(status) = validate_dates(days) {
        return empty(status);
    }
    if days.first().copied() != Some(0.0) {
        return empty(TemporalInferenceStatus::GaugeMissing);
    }
    let (selected_days, selected_y, selected_c) =
        match subset_origin_anchored_covariance(days, observations, difference_covariance) {
            Ok(value) => value,
            Err(status) => return empty(status),
        };
    if let Err(status) = validate_supported_cadence(&days[1..], options) {
        return empty(status);
    }
    let n = selected_days.len();
    if n < options.minimum_dates {
        let mut result = empty(TemporalInferenceStatus::InsufficientDates);
        result.valid_date_count = n;
        return result;
    }
    let rank = if selected_days.iter().map(|day| day * day).sum::<f64>() > 0.0 {
        1
    } else {
        0
    };
    if rank == 0 {
        let mut result = empty(TemporalInferenceStatus::DesignRankDeficient);
        result.valid_date_count = n;
        result.rank = rank;
        return result;
    }
    let degrees_of_freedom = n.saturating_sub(rank);
    let diagonal: Vec<f64> = selected_c
        .iter()
        .enumerate()
        .map(|(index, row)| row[index])
        .collect();
    let _shape = match relative_standard_deviation_shape(&diagonal) {
        Ok(shape) => shape,
        Err(status) => return empty(status),
    };
    let oracle_v = match total_difference_covariance(
        &selected_c,
        &selected_days,
        options.oracle_process_variance,
        options.oracle_rho,
        options.reference_lag_days,
    ) {
        Ok(value) => value,
        Err(status) => return empty(status),
    };
    let oracle_fit = match gls_fit(
        &selected_days,
        &selected_y,
        &oracle_v,
        options.condition_limit,
        true,
    ) {
        Ok(value) => value,
        Err(status) => return empty(status),
    };
    let ols = match ols_slope(&selected_days, &selected_y) {
        Ok(value) => value,
        Err(status) => return empty(status),
    };
    if prefit.is_some()
        && (options.bootstrap_replicates != 0 || options.bootstrap_minimum_successes != 0)
    {
        return empty(TemporalInferenceStatus::DiagnosticNotComputed);
    }
    let prepared_factor_profile = if let Some(factor_profile) = factor_profile {
        let Some(persisted_stride) = n
            .checked_add(1)
            .and_then(|count| count.checked_mul(factor_profile.persisted_maximum_rank))
        else {
            return empty(TemporalInferenceStatus::CovarianceNonfinite);
        };
        if factor_profile.persisted_maximum_rank == 0
            || factor_profile.realized_rank == 0
            || factor_profile.realized_rank > n
            || factor_profile.realized_rank > factor_profile.persisted_maximum_rank
            || factor_profile.persisted_factor.len() != persisted_stride
            || factor_profile.persisted_factor[..factor_profile.persisted_maximum_rank]
                .iter()
                .any(|value| !value.is_finite() || *value != 0.0)
        {
            return empty(TemporalInferenceStatus::CovarianceNonfinite);
        }
        let mut compact_factor = vec![0.0; n * factor_profile.persisted_maximum_rank];
        for date in 0..n {
            let source = (date + 1) * factor_profile.persisted_maximum_rank;
            let destination = date * factor_profile.persisted_maximum_rank;
            let source_values =
                &factor_profile.persisted_factor[source..source + factor_profile.realized_rank];
            if source_values.iter().any(|value| !value.is_finite()) {
                return empty(TemporalInferenceStatus::CovarianceNonfinite);
            }
            compact_factor[destination..destination + factor_profile.realized_rank]
                .copy_from_slice(source_values);
        }
        let factor_covariance = difference_covariance_from_factor(
            n,
            &compact_factor,
            factor_profile.persisted_maximum_rank,
            factor_profile.realized_rank,
        );
        let factor_matches_covariance = factor_covariance.len() == selected_c.len()
            && factor_covariance
                .iter()
                .zip(&selected_c)
                .all(|(factor_row, covariance_row)| {
                    factor_row.len() == covariance_row.len()
                        && factor_row.iter().zip(covariance_row).all(
                            |(factor_value, covariance_value)| {
                                let scale = factor_value.abs().max(covariance_value.abs()).max(1.0);
                                (factor_value - covariance_value).abs() <= 1e-10 * scale
                            },
                        )
                });
        if !factor_matches_covariance {
            return empty(TemporalInferenceStatus::CovarianceNonfinite);
        }
        Some(PreparedFactorProfileInput {
            compact_factor,
            maximum_rank: factor_profile.persisted_maximum_rank,
            realized_rank: factor_profile.realized_rank,
        })
    } else {
        None
    };
    if let Some(prefit) = prefit {
        if let Err(status) =
            validate_temporal_covariance_prefit(&selected_days, &selected_y, options, prefit)
        {
            return empty(status);
        }
    }
    let plugin = if prefit.is_none() {
        match profile_plugin(&selected_days, &selected_y, &selected_c, options) {
            Ok(value) => Some(value),
            Err(status) => return empty(status),
        }
    } else {
        None
    };
    let plugin_slope = if let Some(prefit) = prefit {
        prefit.plugin_slope_per_day
    } else {
        plugin.as_ref().expect("dense plug-in fit is present").slope
    };
    let fitted_parameter_active_set = if let Some(prefit) = prefit {
        prefit.fitted_parameter_active_set
    } else {
        let plugin = plugin.as_ref().expect("dense plug-in fit is present");
        let bounds = match nuisance_bounds(&selected_days, &selected_y, options) {
            Ok(value) => value,
            Err(status) => return empty(status),
        };
        temporal_parameter_boundary_status(
            plugin.rho,
            plugin.process_variance,
            [bounds.rho_lower, bounds.rho_upper],
            [
                bounds.log_variance_lower.exp(),
                bounds.log_variance_upper.exp(),
            ],
            options.optimizer_tolerance * 0.01,
        )
    };
    let bootstrap = plugin.as_ref().map_or(
        BootstrapSummary {
            interval_68: None,
            interval_90: None,
            interval_95: None,
            attempts: 0,
            successes: 0,
            variance: f64::NAN,
            minimum_successes: 0,
        },
        |plugin| bootstrap_refit(&selected_days, &selected_c, plugin, options),
    );
    let residuals: Vec<f64> = selected_y
        .iter()
        .zip(&selected_days)
        .map(|(value, day)| value - plugin_slope * day)
        .collect();
    let raw_correlation = raw_adjacent_correlation(&selected_days, &residuals);
    let ols_information = dot(&selected_days, &selected_days);
    let ols_residuals: Vec<f64> = selected_y
        .iter()
        .zip(&selected_days)
        .map(|(value, day)| value - ols * day)
        .collect();
    let ols_scale = dot(&ols_residuals, &ols_residuals) / degrees_of_freedom.max(1) as f64;
    let ols = normal_comparator(
        ols,
        (ols_scale / ols_information).sqrt(),
        TemporalInferenceStatus::Evaluated,
    );
    let oracle = normal_comparator(
        oracle_fit.slope,
        oracle_fit.information_variance.sqrt(),
        TemporalInferenceStatus::Evaluated,
    );
    let conditional_wls = legacy_intercept_wls(&selected_days, &selected_y, &diagonal).map_or_else(
        empty_comparator,
        |slope| ComparatorDiagnostics {
            point_estimate: Some(slope * DAYS_PER_YEAR),
            status: TemporalInferenceStatus::LegacyNonComparable,
            ..empty_comparator(TemporalInferenceStatus::LegacyNonComparable)
        },
    );
    let ols_raw = raw_adjacent_correlation(&selected_days, &ols_residuals);
    let lag_one_rho = ols_raw.rho.unwrap_or(0.0).clamp(-0.99, 0.99);
    let effective_n = (n as f64 * (1.0 - lag_one_rho) / (1.0 + lag_one_rho)).clamp(1.0, n as f64);
    let scalar_effective_n = normal_comparator(
        ols.point_estimate.unwrap_or(f64::NAN) / DAYS_PER_YEAR,
        (ols_scale / ols_information * n as f64 / effective_n).sqrt(),
        TemporalInferenceStatus::Evaluated,
    );
    let plugin_comparator = if let Some(prefit) = prefit {
        prefit.plugin_gls.clone()
    } else {
        let plugin = plugin.as_ref().expect("dense plug-in fit is present");
        gls_fit(
            &selected_days,
            &selected_y,
            &plugin.covariance,
            options.condition_limit,
            true,
        )
        .map_or_else(empty_comparator, |fit| {
            normal_comparator(
                fit.slope,
                fit.information_variance.sqrt(),
                TemporalInferenceStatus::Evaluated,
            )
        })
    };
    let adjusted_profile_result = prepared_factor_profile.as_ref().map_or_else(
        || profile_comparator(&selected_days, &selected_y, &selected_c, options),
        |factor_profile| {
            factor_profile_comparator(
                &selected_days,
                &selected_y,
                &selected_c,
                factor_profile,
                options,
            )
        },
    );
    let profile_status = adjusted_profile_result.as_ref().err().copied();
    let adjusted_profile = adjusted_profile_result.unwrap_or_else(empty_comparator);
    let adjusted_scalar = if let Some(prefit) = prefit {
        prefit.adjusted_scalar.clone()
    } else {
        reml_adjusted_scalar_comparator(
            &selected_days,
            &selected_y,
            &selected_c,
            plugin.as_ref().expect("dense plug-in fit is present"),
            degrees_of_freedom,
            options,
        )
        .unwrap_or_else(empty_comparator)
    };
    let bootstrap_comparator = bootstrap_comparator(&bootstrap, plugin_slope);
    let minimum_bootstrap_successes = required_bootstrap_successes(options.bootstrap_replicates)
        .max(options.bootstrap_minimum_successes);
    let status = if let Some(status) = profile_status {
        status
    } else if bootstrap.successes < minimum_bootstrap_successes {
        TemporalInferenceStatus::BootstrapInsufficientSuccess
    } else if adjusted_scalar.status != TemporalInferenceStatus::Evaluated {
        adjusted_scalar.status
    } else {
        TemporalInferenceStatus::Evaluated
    };
    TemporalCovarianceFit {
        status,
        ols_slope: ols.point_estimate,
        oracle_gls_slope: oracle.point_estimate,
        plugin_gls_slope: Some(plugin_slope * DAYS_PER_YEAR),
        adjusted_profile_slope: adjusted_profile.point_estimate,
        bootstrap_slope: bootstrap_comparator.point_estimate,
        bootstrap_interval: bootstrap_comparator.interval_95,
        fitted_rho: Some(prefit.map_or_else(
            || plugin.as_ref().expect("dense plug-in fit is present").rho,
            |prefit| prefit.fitted_rho,
        )),
        fitted_process_variance: Some(prefit.map_or_else(
            || {
                plugin
                    .as_ref()
                    .expect("dense plug-in fit is present")
                    .process_variance
            },
            |prefit| prefit.fitted_process_variance,
        )),
        fitted_parameter_active_set,
        raw_correlation,
        valid_date_count: n,
        rank,
        degrees_of_freedom,
        covariance_condition_number: prefit.map_or_else(
            || {
                Some(
                    plugin
                        .as_ref()
                        .expect("dense plug-in fit is present")
                        .condition_number,
                )
            },
            |prefit| prefit.covariance_condition_number,
        ),
        ols,
        oracle_gls: oracle,
        conditional_wls,
        scalar_effective_n,
        plugin_gls: plugin_comparator,
        adjusted_scalar,
        adjusted_profile,
        complete_refit_bootstrap: bootstrap_comparator,
        bootstrap_attempts: bootstrap.attempts,
        bootstrap_successes: bootstrap.successes,
    }
}

/// Evaluate one scalar temporal candidate without running the frozen bootstrap.
///
/// This probe exists only for pre-outcome resource selection. It does not return
/// a production estimate identity and cannot authorize a temporal product.
#[must_use]
pub fn probe_temporal_scalar_candidate(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
    method: TemporalScalarCandidateMethod,
) -> TemporalScalarCandidateProbe {
    let empty = |status| TemporalScalarCandidateProbe {
        method,
        comparator: empty_comparator(status),
        valid_date_count: 0,
        rank: 0,
        degrees_of_freedom: 0,
        fitted_rho: None,
        fitted_process_variance: None,
        covariance_condition_number: None,
        bootstrap_attempts: 0,
    };
    if let Err(status) = validate_dates(days) {
        return empty(status);
    }
    if days.first().copied() != Some(0.0) {
        return empty(TemporalInferenceStatus::GaugeMissing);
    }
    let (selected_days, selected_y, selected_c) =
        match subset_origin_anchored_covariance(days, observations, difference_covariance) {
            Ok(value) => value,
            Err(status) => return empty(status),
        };
    if let Err(status) = validate_supported_cadence(&days[1..], options) {
        return empty(status);
    }
    let valid_date_count = selected_days.len();
    if valid_date_count < options.minimum_dates {
        let mut result = empty(TemporalInferenceStatus::InsufficientDates);
        result.valid_date_count = valid_date_count;
        return result;
    }
    let rank = usize::from(selected_days.iter().map(|day| day * day).sum::<f64>() > 0.0);
    if rank == 0 {
        let mut result = empty(TemporalInferenceStatus::DesignRankDeficient);
        result.valid_date_count = valid_date_count;
        return result;
    }
    let degrees_of_freedom = valid_date_count.saturating_sub(rank);
    let diagonal = selected_c
        .iter()
        .enumerate()
        .map(|(index, row)| row[index])
        .collect::<Vec<_>>();
    if let Err(status) = relative_standard_deviation_shape(&diagonal) {
        return empty(status);
    }
    let plugin = match profile_plugin(&selected_days, &selected_y, &selected_c, options) {
        Ok(value) => value,
        Err(status) => return empty(status),
    };
    let comparator = match method {
        TemporalScalarCandidateMethod::PluginGlsReml => gls_fit(
            &selected_days,
            &selected_y,
            &plugin.covariance,
            options.condition_limit,
            true,
        )
        .map(|fit| {
            normal_comparator(
                fit.slope,
                fit.information_variance.sqrt(),
                TemporalInferenceStatus::Evaluated,
            )
        }),
        TemporalScalarCandidateMethod::RemlCovarianceParameterAdjustedScalar => {
            reml_adjusted_scalar_comparator(
                &selected_days,
                &selected_y,
                &selected_c,
                &plugin,
                degrees_of_freedom,
                options,
            )
        }
        TemporalScalarCandidateMethod::SlopeProfileLikelihoodMl => {
            profile_comparator(&selected_days, &selected_y, &selected_c, options)
        }
    }
    .unwrap_or_else(empty_comparator);
    TemporalScalarCandidateProbe {
        method,
        comparator,
        valid_date_count,
        rank,
        degrees_of_freedom,
        fitted_rho: Some(plugin.rho),
        fitted_process_variance: Some(plugin.process_variance),
        covariance_condition_number: Some(plugin.condition_number),
        bootstrap_attempts: 0,
    }
}

/// Select the frozen complete-refit bootstrap estimate as a promotion-neutral candidate.
///
/// The returned point estimate and standard error remain absent unless the
/// overall fit and selected comparator evaluated, the frozen preregistration
/// is unchanged, bootstrap accounting is exact, and all selected values are
/// finite. This function does not validate calibration or promotion evidence.
#[must_use]
pub fn complete_refit_bootstrap_estimate(
    fit: &TemporalCovarianceFit,
    options: &TemporalCovarianceOptions,
) -> CompleteRefitBootstrapEstimate {
    let cadence_status = match fit.status {
        TemporalInferenceStatus::Evaluated => CompleteRefitBootstrapCadenceStatus::Supported,
        TemporalInferenceStatus::UnsupportedCadence => {
            CompleteRefitBootstrapCadenceStatus::Unsupported
        }
        _ => CompleteRefitBootstrapCadenceStatus::Unavailable,
    };
    let result = |status, slope_per_year, standard_error_per_year| CompleteRefitBootstrapEstimate {
        status,
        fit_status: fit.status,
        slope_per_year,
        standard_error_per_year,
        valid_date_count: fit.valid_date_count,
        rank: fit.rank,
        degrees_of_freedom: fit.degrees_of_freedom,
        cadence_days: [
            fit.raw_correlation.minimum_gap_days,
            fit.raw_correlation.median_gap_days,
            fit.raw_correlation.maximum_gap_days,
        ],
        cadence_status,
        raw_rho: fit.raw_correlation.rho,
        fitted_rho: fit.fitted_rho,
        fitted_process_variance: fit.fitted_process_variance,
        fitted_parameter_active_set: fit.fitted_parameter_active_set,
        method: COMPLETE_REFIT_BOOTSTRAP_METHOD.to_owned(),
        method_version: COMPLETE_REFIT_BOOTSTRAP_METHOD_VERSION,
        condition_number: fit.covariance_condition_number,
        bootstrap_attempts: fit.bootstrap_attempts,
        bootstrap_successes: fit.bootstrap_successes,
    };
    let abstain = |status| result(status, None, None);
    if options.bootstrap_replicates != COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS
        || options.bootstrap_minimum_successes != COMPLETE_REFIT_BOOTSTRAP_MINIMUM_SUCCESSES
    {
        return abstain(CompleteRefitBootstrapEstimateStatus::FrozenConfigurationMismatch);
    }
    if fit.status != TemporalInferenceStatus::Evaluated {
        return abstain(CompleteRefitBootstrapEstimateStatus::FitNotEvaluated);
    }
    let selected = &fit.complete_refit_bootstrap;
    if selected.status != TemporalInferenceStatus::Evaluated {
        return abstain(CompleteRefitBootstrapEstimateStatus::ComparatorNotEvaluated);
    }
    if fit.bootstrap_attempts != COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS
        || selected.attempted_replicates != COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS
        || fit.bootstrap_successes != selected.successful_replicates
    {
        return abstain(CompleteRefitBootstrapEstimateStatus::BootstrapAccountingMismatch);
    }
    let minimum_successes = required_bootstrap_successes(COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS)
        .max(COMPLETE_REFIT_BOOTSTRAP_MINIMUM_SUCCESSES);
    if fit.bootstrap_successes < minimum_successes {
        return abstain(CompleteRefitBootstrapEstimateStatus::BootstrapInsufficientSuccess);
    }
    let (Some(fit_slope), Some(selected_slope), Some(standard_error)) = (
        fit.bootstrap_slope,
        selected.point_estimate,
        selected.standard_error_diagnostic,
    ) else {
        return abstain(CompleteRefitBootstrapEstimateStatus::InvalidEstimate);
    };
    if !fit_slope.is_finite()
        || !selected_slope.is_finite()
        || !standard_error.is_finite()
        || standard_error <= 0.0
        || fit_slope != selected_slope
    {
        return abstain(CompleteRefitBootstrapEstimateStatus::InvalidEstimate);
    }
    result(
        CompleteRefitBootstrapEstimateStatus::Evaluated,
        Some(selected_slope),
        Some(standard_error),
    )
}

/// Build validation-only F53-03 provenance without enabling any product writer.
#[must_use]
pub fn temporal_covariance_provenance(
    fit: &TemporalCovarianceFit,
    inputs: TemporalCovarianceProvenanceInputs,
) -> Option<TemporalCovarianceProvenance> {
    if fit.status != TemporalInferenceStatus::Evaluated {
        return None;
    }
    Some(TemporalCovarianceProvenance {
        schema: "dolphinrust-temporal-covariance-provenance/2".to_owned(),
        estimator: "origin_anchored_temporal_covariance_slope".to_owned(),
        estimator_version: env!("CARGO_PKG_VERSION").to_owned(),
        valid_date_count: fit.valid_date_count,
        rank: fit.rank,
        degrees_of_freedom: fit.degrees_of_freedom,
        cadence_days: [
            fit.raw_correlation.minimum_gap_days,
            fit.raw_correlation.median_gap_days,
            fit.raw_correlation.maximum_gap_days,
        ],
        raw_rho: fit.raw_correlation.rho,
        fitted_rho: fit.fitted_rho,
        fitted_process_variance: fit.fitted_process_variance,
        fitted_parameter_active_set: fit.fitted_parameter_active_set,
        issue52_receipt_sha256: inputs.issue52_receipt_sha256,
        issue54_receipt_sha256: inputs.issue54_receipt_sha256,
        reference: inputs.reference,
        condition_number: fit.covariance_condition_number,
        scope: inputs.scope,
        bootstrap_attempts: fit.bootstrap_attempts,
        bootstrap_successes: fit.bootstrap_successes,
        validation_receipt_sha256: inputs.validation_receipt_sha256,
        estimator_input_sha256: inputs.estimator_input_sha256,
        bootstrap_minimum_success_fraction: 0.99,
        selected_method: inputs.selected_method,
    })
}

/// Compute an unclamped adjacent residual correlation and elapsed-gap summary.
pub fn raw_adjacent_correlation(days: &[f64], observations: &[f64]) -> RawCorrelationDiagnostics {
    let pairs: Vec<(f64, f64, f64)> = days
        .windows(2)
        .zip(observations.windows(2))
        .filter_map(|(x, y)| {
            (x[0].is_finite() && x[1].is_finite() && y[0].is_finite() && y[1].is_finite())
                .then_some((y[0], y[1], x[1] - x[0]))
        })
        .collect();
    let mut gaps: Vec<f64> = pairs
        .iter()
        .map(|(_, _, gap)| *gap)
        .filter(|gap| gap.is_finite() && *gap > 0.0)
        .collect();
    gaps.sort_by(f64::total_cmp);
    let rho = if pairs.len() < 3 {
        None
    } else {
        let left_mean = pairs.iter().map(|(left, _, _)| *left).sum::<f64>() / pairs.len() as f64;
        let right_mean = pairs.iter().map(|(_, right, _)| *right).sum::<f64>() / pairs.len() as f64;
        let (numerator, left_sum, right_sum) = pairs.iter().fold(
            (0.0, 0.0, 0.0),
            |(numerator, left_sum, right_sum), (left, right, _)| {
                let lc = *left - left_mean;
                let rc = *right - right_mean;
                (numerator + lc * rc, left_sum + lc * lc, right_sum + rc * rc)
            },
        );
        let denominator = (left_sum * right_sum).sqrt();
        (denominator > 0.0 && denominator.is_finite()).then_some(numerator / denominator)
    };
    RawCorrelationDiagnostics {
        rho,
        pair_count: pairs.len(),
        minimum_gap_days: gaps.first().copied(),
        median_gap_days: match gaps.len() {
            0 => None,
            length if length % 2 == 1 => gaps.get(length / 2).copied(),
            length => Some((gaps[length / 2 - 1] + gaps[length / 2]) / 2.0),
        },
        maximum_gap_days: gaps.last().copied(),
    }
}

#[derive(Debug, Clone)]
struct PluginFit {
    slope: f64,
    rho: f64,
    process_variance: f64,
    covariance: Vec<Vec<f64>>,
    condition_number: f64,
    information_variance: f64,
}

#[derive(Clone, Copy)]
pub(super) struct NuisanceBounds {
    pub(super) rho_lower: f64,
    pub(super) rho_upper: f64,
    pub(super) log_variance_lower: f64,
    pub(super) log_variance_upper: f64,
    pub(super) initial_log_variance: f64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct FactorObjectiveEvaluation {
    pub(super) score: f64,
    pub(super) slope: f64,
    pub(super) x_v_x: f64,
    #[cfg(test)]
    pub(super) x_v_y: f64,
    #[cfg(test)]
    pub(super) y_v_y: f64,
    #[cfg(test)]
    pub(super) log_determinant: f64,
    #[cfg(test)]
    pub(super) quadratic: f64,
    pub(super) dense_fallback_used: bool,
}

pub(super) struct PreparedFactorObjective {
    pub(super) design: Vec<f64>,
    pub(super) gap_exponents: Vec<f64>,
    pub(super) reference_lag_days: f64,
}

impl PreparedFactorObjective {
    pub(super) fn new(
        days: &[f64],
        reference_lag_days: f64,
    ) -> Result<Self, TemporalInferenceStatus> {
        if days.is_empty()
            || days.iter().any(|day| !day.is_finite())
            || days.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err(TemporalInferenceStatus::DatesNotStrictlyIncreasing);
        }
        if !reference_lag_days.is_finite() || reference_lag_days <= 0.0 {
            return Err(TemporalInferenceStatus::UnsupportedCadence);
        }
        Ok(Self {
            design: days.to_vec(),
            gap_exponents: days
                .windows(2)
                .map(|pair| (pair[1] - pair[0]) / reference_lag_days)
                .collect(),
            reference_lag_days,
        })
    }
}

pub(super) struct FactorObjectiveScratch {
    date_capacity: usize,
    rank_capacity: usize,
    inverse_shape: Vec<f64>,
    transition: Vec<f64>,
    inverse_innovation_scale: Vec<f64>,
    scaled_factor: Vec<f64>,
    whitened_factor: Vec<f64>,
    z_x: Vec<f64>,
    z_y: Vec<f64>,
    whitened_x: Vec<f64>,
    whitened_y: Vec<f64>,
    #[cfg(test)]
    small_lower: Vec<f64>,
    #[cfg(test)]
    h_x: Vec<f64>,
    #[cfg(test)]
    h_y: Vec<f64>,
    #[cfg(test)]
    solve_x: Vec<f64>,
    #[cfg(test)]
    solve_y: Vec<f64>,
}

impl FactorObjectiveScratch {
    pub(super) fn new(date_capacity: usize, rank_capacity: usize) -> Self {
        Self {
            date_capacity,
            rank_capacity,
            inverse_shape: vec![0.0; date_capacity],
            transition: vec![0.0; date_capacity.saturating_sub(1)],
            inverse_innovation_scale: vec![0.0; date_capacity.saturating_sub(1)],
            scaled_factor: vec![0.0; date_capacity.saturating_mul(rank_capacity)],
            whitened_factor: vec![0.0; date_capacity.saturating_mul(rank_capacity)],
            z_x: vec![0.0; date_capacity],
            z_y: vec![0.0; date_capacity],
            whitened_x: vec![0.0; date_capacity],
            whitened_y: vec![0.0; date_capacity],
            #[cfg(test)]
            small_lower: vec![0.0; rank_capacity.saturating_mul(rank_capacity)],
            #[cfg(test)]
            h_x: vec![0.0; rank_capacity],
            #[cfg(test)]
            h_y: vec![0.0; rank_capacity],
            #[cfg(test)]
            solve_x: vec![0.0; rank_capacity],
            #[cfg(test)]
            solve_y: vec![0.0; rank_capacity],
        }
    }

    fn supports(&self, dates: usize, rank: usize) -> bool {
        dates <= self.date_capacity && rank <= self.rank_capacity
    }
}

#[cfg(test)]
fn solve_flat_cholesky(lower: &[f64], dimension: usize, rhs: &[f64], solution: &mut [f64]) {
    for row in 0..dimension {
        let correction = (0..row)
            .map(|column| lower[row * dimension + column] * solution[column])
            .sum::<f64>();
        solution[row] = (rhs[row] - correction) / lower[row * dimension + row];
    }
    for row in (0..dimension).rev() {
        let correction = ((row + 1)..dimension)
            .map(|column| lower[column * dimension + row] * solution[column])
            .sum::<f64>();
        solution[row] = (solution[row] - correction) / lower[row * dimension + row];
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dense_factor_objective_fallback(
    prepared: &PreparedFactorObjective,
    observations: &[f64],
    factor: &[f64],
    maximum_rank: usize,
    realized_rank: usize,
    rho: f64,
    process_variance: f64,
    restricted: bool,
) -> Result<FactorObjectiveEvaluation, TemporalInferenceStatus> {
    let date_count = prepared.design.len();
    let difference_covariance = (0..date_count)
        .map(|left| {
            (0..date_count)
                .map(|right| {
                    (0..realized_rank)
                        .map(|component| {
                            factor[left * maximum_rank + component]
                                * factor[right * maximum_rank + component]
                        })
                        .sum()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let covariance = total_difference_covariance(
        &difference_covariance,
        &prepared.design,
        process_variance,
        rho,
        prepared.reference_lag_days,
    )?;
    let fit = gls_fit(
        &prepared.design,
        observations,
        &covariance,
        f64::INFINITY,
        false,
    )?;
    let inverse_x = mat_vec(&fit.inverse, &prepared.design);
    #[cfg(test)]
    let inverse_y = mat_vec(&fit.inverse, observations);
    let x_v_x = dot(&prepared.design, &inverse_x);
    #[cfg(test)]
    let x_v_y = dot(&prepared.design, &inverse_y);
    #[cfg(test)]
    let y_v_y = dot(observations, &inverse_y);
    Ok(FactorObjectiveEvaluation {
        score: fit.log_determinant + fit.quadratic_form + if restricted { x_v_x.ln() } else { 0.0 },
        slope: fit.slope,
        x_v_x,
        #[cfg(test)]
        x_v_y,
        #[cfg(test)]
        y_v_y,
        #[cfg(test)]
        log_determinant: fit.log_determinant,
        #[cfg(test)]
        quadratic: fit.quadratic_form,
        dense_fallback_used: true,
    })
}

#[cfg(test)]
#[allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
pub(super) fn factor_native_objective(
    prepared: &PreparedFactorObjective,
    observations: &[f64],
    factor: &[f64],
    maximum_rank: usize,
    realized_rank: usize,
    rho: f64,
    log_process_variance: f64,
    restricted: bool,
    scratch: &mut FactorObjectiveScratch,
) -> Result<FactorObjectiveEvaluation, TemporalInferenceStatus> {
    let date_count = prepared.design.len();
    if observations.len() != date_count
        || maximum_rank == 0
        || realized_rank == 0
        || realized_rank > maximum_rank
        || factor.len() != date_count.saturating_mul(maximum_rank)
        || factor.iter().any(|value| !value.is_finite())
        || observations.iter().any(|value| !value.is_finite())
        || !scratch.supports(date_count, realized_rank)
    {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    if !rho.is_finite() || !(0.0..1.0).contains(&rho) {
        return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
    }
    let process_variance = log_process_variance.exp();
    if !process_variance.is_finite() || process_variance <= 0.0 {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let mut mean_log_diagonal = 0.0;
    for row in 0..date_count {
        let diagonal = (0..realized_rank)
            .map(|component| factor[row * maximum_rank + component].powi(2))
            .sum::<f64>();
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return Err(TemporalInferenceStatus::CovarianceNonfinite);
        }
        scratch.inverse_shape[row] = diagonal;
        mean_log_diagonal += diagonal.ln();
    }
    mean_log_diagonal /= date_count as f64;
    let geometric_mean = mean_log_diagonal.exp();
    if !geometric_mean.is_finite() || geometric_mean <= 0.0 {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let mut log_shape_sum = 0.0;
    let mut maximum_shape = 0.0_f64;
    for row in 0..date_count {
        let shape = (scratch.inverse_shape[row] / geometric_mean).sqrt();
        if !shape.is_finite() || shape <= 0.0 {
            return Err(TemporalInferenceStatus::CovarianceNonfinite);
        }
        log_shape_sum += shape.ln();
        maximum_shape = maximum_shape.max(shape);
        scratch.inverse_shape[row] = 1.0 / shape;
        for component in 0..realized_rank {
            scratch.scaled_factor[row * realized_rank + component] =
                factor[row * maximum_rank + component] * scratch.inverse_shape[row];
        }
    }
    if process_variance * maximum_shape.powi(2) * f64::EPSILON * 8.0 > SYMMETRY_TOLERANCE
        || process_variance <= geometric_mean * 1e-5
    {
        return dense_factor_objective_fallback(
            prepared,
            observations,
            factor,
            maximum_rank,
            realized_rank,
            rho,
            process_variance,
            restricted,
        );
    }

    let mut log_determinant_r = 0.0;
    if rho == 0.0 {
        scratch.transition[..date_count.saturating_sub(1)].fill(0.0);
        scratch.inverse_innovation_scale[..date_count.saturating_sub(1)].fill(1.0);
    } else {
        for index in 0..date_count - 1 {
            let log_phi = rho.ln() * prepared.gap_exponents[index];
            let phi = log_phi.exp();
            let innovation = -(2.0 * log_phi).exp_m1();
            if !phi.is_finite() || !innovation.is_finite() || innovation <= 0.0 {
                return Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite);
            }
            scratch.transition[index] = phi;
            scratch.inverse_innovation_scale[index] = 1.0 / innovation.sqrt();
            log_determinant_r += innovation.ln();
        }
    }

    for row in 0..date_count {
        scratch.z_x[row] = prepared.design[row] * scratch.inverse_shape[row];
        scratch.z_y[row] = observations[row] * scratch.inverse_shape[row];
    }
    scratch.whitened_x[0] = scratch.z_x[0];
    scratch.whitened_y[0] = scratch.z_y[0];
    for component in 0..realized_rank {
        scratch.whitened_factor[component] = scratch.scaled_factor[component];
    }
    for row in 1..date_count {
        let transition = scratch.transition[row - 1];
        let inverse_scale = scratch.inverse_innovation_scale[row - 1];
        scratch.whitened_x[row] =
            (scratch.z_x[row] - transition * scratch.z_x[row - 1]) * inverse_scale;
        scratch.whitened_y[row] =
            (scratch.z_y[row] - transition * scratch.z_y[row - 1]) * inverse_scale;
        for component in 0..realized_rank {
            scratch.whitened_factor[row * realized_rank + component] = (scratch.scaled_factor
                [row * realized_rank + component]
                - transition * scratch.scaled_factor[(row - 1) * realized_rank + component])
                * inverse_scale;
        }
    }

    scratch.small_lower[..realized_rank * realized_rank].fill(0.0);
    for row in 0..realized_rank {
        for column in 0..=row {
            let gram = (0..date_count)
                .map(|date| {
                    scratch.whitened_factor[date * realized_rank + row]
                        * scratch.whitened_factor[date * realized_rank + column]
                })
                .sum::<f64>();
            let value = gram + process_variance * f64::from(row == column);
            let correction = (0..column)
                .map(|index| {
                    scratch.small_lower[row * realized_rank + index]
                        * scratch.small_lower[column * realized_rank + index]
                })
                .sum::<f64>();
            if row == column {
                let diagonal = value - correction;
                if !diagonal.is_finite() || diagonal <= 0.0 {
                    return Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite);
                }
                scratch.small_lower[row * realized_rank + column] = diagonal.sqrt();
            } else {
                scratch.small_lower[row * realized_rank + column] =
                    (value - correction) / scratch.small_lower[column * realized_rank + column];
            }
        }
    }

    for component in 0..realized_rank {
        scratch.h_x[component] = (0..date_count)
            .map(|row| {
                scratch.whitened_factor[row * realized_rank + component] * scratch.whitened_x[row]
            })
            .sum::<f64>();
        scratch.h_y[component] = (0..date_count)
            .map(|row| {
                scratch.whitened_factor[row * realized_rank + component] * scratch.whitened_y[row]
            })
            .sum::<f64>();
    }
    solve_flat_cholesky(
        &scratch.small_lower[..realized_rank * realized_rank],
        realized_rank,
        &scratch.h_x[..realized_rank],
        &mut scratch.solve_x[..realized_rank],
    );
    solve_flat_cholesky(
        &scratch.small_lower[..realized_rank * realized_rank],
        realized_rank,
        &scratch.h_y[..realized_rank],
        &mut scratch.solve_y[..realized_rank],
    );
    let process_standard_deviation = process_variance.sqrt();
    let (mut x_v_x, mut x_v_y, mut y_v_y) = (0.0, 0.0, 0.0);
    for row in 0..date_count {
        let fitted_x = (0..realized_rank)
            .map(|component| {
                scratch.whitened_factor[row * realized_rank + component]
                    * scratch.solve_x[component]
            })
            .sum::<f64>();
        let fitted_y = (0..realized_rank)
            .map(|component| {
                scratch.whitened_factor[row * realized_rank + component]
                    * scratch.solve_y[component]
            })
            .sum::<f64>();
        let residual_x = (scratch.whitened_x[row] - fitted_x) / process_standard_deviation;
        let residual_y = (scratch.whitened_y[row] - fitted_y) / process_standard_deviation;
        x_v_x += residual_x * residual_x;
        x_v_y += residual_x * residual_y;
        y_v_y += residual_y * residual_y;
    }
    x_v_x += dot(
        &scratch.solve_x[..realized_rank],
        &scratch.solve_x[..realized_rank],
    );
    x_v_y += dot(
        &scratch.solve_x[..realized_rank],
        &scratch.solve_y[..realized_rank],
    );
    y_v_y += dot(
        &scratch.solve_y[..realized_rank],
        &scratch.solve_y[..realized_rank],
    );
    if !x_v_x.is_finite() || x_v_x <= 0.0 || !x_v_y.is_finite() || !y_v_y.is_finite() {
        return Err(TemporalInferenceStatus::DesignRankDeficient);
    }
    let slope = x_v_y / x_v_x;
    let mut quadratic = 0.0;
    for row in 0..date_count {
        let fitted_x = (0..realized_rank)
            .map(|component| {
                scratch.whitened_factor[row * realized_rank + component]
                    * scratch.solve_x[component]
            })
            .sum::<f64>();
        let fitted_y = (0..realized_rank)
            .map(|component| {
                scratch.whitened_factor[row * realized_rank + component]
                    * scratch.solve_y[component]
            })
            .sum::<f64>();
        let residual =
            (scratch.whitened_y[row] - fitted_y - slope * (scratch.whitened_x[row] - fitted_x))
                / process_standard_deviation;
        quadratic += residual * residual;
    }
    quadratic += (0..realized_rank)
        .map(|component| (scratch.solve_y[component] - slope * scratch.solve_x[component]).powi(2))
        .sum::<f64>();
    let log_determinant_k = 2.0
        * (0..realized_rank)
            .map(|index| scratch.small_lower[index * realized_rank + index].ln())
            .sum::<f64>();
    let log_determinant = (date_count - realized_rank) as f64 * log_process_variance
        + 2.0 * log_shape_sum
        + log_determinant_r
        + log_determinant_k;
    let score = log_determinant + quadratic + if restricted { x_v_x.ln() } else { 0.0 };
    if !score.is_finite()
        || !slope.is_finite()
        || !quadratic.is_finite()
        || !log_determinant.is_finite()
    {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    Ok(FactorObjectiveEvaluation {
        score,
        slope,
        x_v_x,
        x_v_y,
        y_v_y,
        log_determinant,
        quadratic,
        dense_fallback_used: false,
    })
}

#[cfg(test)]
struct FactorObjectiveWorker {
    observations: Vec<f64>,
    scratch: FactorObjectiveScratch,
}

#[cfg(test)]
impl FactorObjectiveWorker {
    fn new(date_count: usize, maximum_rank: usize) -> Self {
        Self {
            observations: vec![0.0; date_count],
            scratch: FactorObjectiveScratch::new(date_count, maximum_rank),
        }
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn factor_native_objective_microbatch(
    prepared: &PreparedFactorObjective,
    observations_soa: &[f64],
    factors: &[f64],
    maximum_rank: usize,
    realized_ranks: &[usize],
    rhos: &[f64],
    log_process_variances: &[f64],
    restricted: bool,
    lane_width: usize,
) -> Result<Vec<Result<FactorObjectiveEvaluation, TemporalInferenceStatus>>, TemporalInferenceStatus>
{
    let target_count = realized_ranks.len();
    let date_count = prepared.design.len();
    let factor_stride = date_count
        .checked_mul(maximum_rank)
        .ok_or(TemporalInferenceStatus::CovarianceNonfinite)?;
    if target_count == 0
        || lane_width == 0
        || maximum_rank == 0
        || rhos.len() != target_count
        || log_process_variances.len() != target_count
        || observations_soa.len() != date_count.saturating_mul(target_count)
        || factors.len() != factor_stride.saturating_mul(target_count)
    {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let mut output = (0..target_count).map(|_| None).collect::<Vec<_>>();
    output.par_chunks_mut(lane_width).enumerate().for_each_init(
        || FactorObjectiveWorker::new(date_count, maximum_rank),
        |worker, (chunk_index, output_chunk)| {
            let first_target = chunk_index * lane_width;
            for (lane, destination) in output_chunk.iter_mut().enumerate() {
                let target = first_target + lane;
                for date in 0..date_count {
                    worker.observations[date] = observations_soa[date * target_count + target];
                }
                let factor_offset = target * factor_stride;
                *destination = Some(factor_native_objective(
                    prepared,
                    &worker.observations,
                    &factors[factor_offset..factor_offset + factor_stride],
                    maximum_rank,
                    realized_ranks[target],
                    rhos[target],
                    log_process_variances[target],
                    restricted,
                    &mut worker.scratch,
                ));
            }
        },
    );
    output
        .into_iter()
        .map(|value| value.ok_or(TemporalInferenceStatus::CovarianceNonfinite))
        .collect()
}

#[derive(Debug, Clone)]
pub(super) struct FactorProfileFit {
    #[cfg(test)]
    pub(super) score: f64,
    pub(super) slope: f64,
    pub(super) rho: f64,
    pub(super) process_variance: f64,
    pub(super) information_variance: f64,
    pub(super) condition_number: f64,
    #[cfg(test)]
    pub(super) primary_factorization_count: usize,
    #[cfg(test)]
    pub(super) profile_rho_curvature: f64,
    pub(super) dense_fallback_count: usize,
}

#[derive(Clone, Copy)]
struct SpectralTargetScale {
    log_shape_sum: f64,
}

#[derive(Clone)]
struct SpectralProjection {
    rho: f64,
    eigenvalues: Vec<f64>,
    c_x: Vec<f64>,
    c_y: Vec<f64>,
    a_x_x: f64,
    a_x_y: f64,
    a_y_y: f64,
    log_shape_sum: f64,
    log_determinant_r: f64,
    date_count: usize,
    realized_rank: usize,
}

#[derive(Debug, Clone, Copy)]
struct SpectralQEvaluation {
    score: f64,
    slope: f64,
    information_variance: f64,
    process_variance: f64,
    score_gradient_log_q: f64,
    score_curvature_log_q: f64,
}

#[derive(Clone)]
struct SpectralProfileAtRho {
    projection: SpectralProjection,
    fit: SpectralQEvaluation,
}

#[allow(clippy::too_many_arguments)]
fn prepare_spectral_target(
    prepared: &PreparedFactorObjective,
    observations: &[f64],
    factor: &[f64],
    maximum_rank: usize,
    realized_rank: usize,
    scratch: &mut FactorObjectiveScratch,
) -> Result<SpectralTargetScale, TemporalInferenceStatus> {
    let date_count = prepared.design.len();
    if observations.len() != date_count
        || maximum_rank == 0
        || realized_rank == 0
        || realized_rank > maximum_rank
        || factor.len() != date_count.saturating_mul(maximum_rank)
        || factor.iter().any(|value| !value.is_finite())
        || observations.iter().any(|value| !value.is_finite())
        || !scratch.supports(date_count, realized_rank)
    {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let mut mean_log_diagonal = 0.0;
    for row in 0..date_count {
        let diagonal = (0..realized_rank)
            .map(|component| factor[row * maximum_rank + component].powi(2))
            .sum::<f64>();
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return Err(TemporalInferenceStatus::CovarianceNonfinite);
        }
        scratch.inverse_shape[row] = diagonal;
        mean_log_diagonal += diagonal.ln();
    }
    mean_log_diagonal /= date_count as f64;
    let geometric_mean = mean_log_diagonal.exp();
    if !geometric_mean.is_finite() || geometric_mean <= 0.0 {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let mut log_shape_sum = 0.0;
    for row in 0..date_count {
        let shape = (scratch.inverse_shape[row] / geometric_mean).sqrt();
        if !shape.is_finite() || shape <= 0.0 {
            return Err(TemporalInferenceStatus::CovarianceNonfinite);
        }
        log_shape_sum += shape.ln();
        scratch.inverse_shape[row] = 1.0 / shape;
        scratch.z_x[row] = prepared.design[row] / shape;
        scratch.z_y[row] = observations[row] / shape;
        for component in 0..realized_rank {
            scratch.scaled_factor[row * realized_rank + component] =
                factor[row * maximum_rank + component] / shape;
        }
    }
    Ok(SpectralTargetScale { log_shape_sum })
}

#[allow(clippy::too_many_lines)]
fn spectral_projection(
    prepared: &PreparedFactorObjective,
    rho: f64,
    realized_rank: usize,
    scale: SpectralTargetScale,
    scratch: &mut FactorObjectiveScratch,
) -> Result<SpectralProjection, TemporalInferenceStatus> {
    if !rho.is_finite() || !(0.0..1.0).contains(&rho) {
        return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
    }
    let date_count = prepared.design.len();
    let mut log_determinant_r = 0.0;
    if rho == 0.0 {
        scratch.transition[..date_count.saturating_sub(1)].fill(0.0);
        scratch.inverse_innovation_scale[..date_count.saturating_sub(1)].fill(1.0);
    } else {
        for index in 0..date_count - 1 {
            let log_phi = rho.ln() * prepared.gap_exponents[index];
            let phi = log_phi.exp();
            let innovation = -(2.0 * log_phi).exp_m1();
            if !phi.is_finite() || !innovation.is_finite() || innovation <= 0.0 {
                return Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite);
            }
            scratch.transition[index] = phi;
            scratch.inverse_innovation_scale[index] = 1.0 / innovation.sqrt();
            log_determinant_r += innovation.ln();
        }
    }
    scratch.whitened_x[0] = scratch.z_x[0];
    scratch.whitened_y[0] = scratch.z_y[0];
    for component in 0..realized_rank {
        scratch.whitened_factor[component] = scratch.scaled_factor[component];
    }
    for row in 1..date_count {
        let transition = scratch.transition[row - 1];
        let inverse_scale = scratch.inverse_innovation_scale[row - 1];
        scratch.whitened_x[row] =
            (scratch.z_x[row] - transition * scratch.z_x[row - 1]) * inverse_scale;
        scratch.whitened_y[row] =
            (scratch.z_y[row] - transition * scratch.z_y[row - 1]) * inverse_scale;
        for component in 0..realized_rank {
            scratch.whitened_factor[row * realized_rank + component] = (scratch.scaled_factor
                [row * realized_rank + component]
                - transition * scratch.scaled_factor[(row - 1) * realized_rank + component])
                * inverse_scale;
        }
    }
    let a_x_x = dot(
        &scratch.whitened_x[..date_count],
        &scratch.whitened_x[..date_count],
    );
    let a_x_y = dot(
        &scratch.whitened_x[..date_count],
        &scratch.whitened_y[..date_count],
    );
    let a_y_y = dot(
        &scratch.whitened_y[..date_count],
        &scratch.whitened_y[..date_count],
    );
    let gram = Mat::from_fn(realized_rank, realized_rank, |row, column| {
        (0..date_count)
            .map(|date| {
                scratch.whitened_factor[date * realized_rank + row]
                    * scratch.whitened_factor[date * realized_rank + column]
            })
            .sum::<f64>()
    });
    let eigen = gram.selfadjoint_eigendecomposition(Side::Lower);
    let raw_eigenvalues = (0..realized_rank)
        .map(|index| eigen.s().column_vector()[index])
        .collect::<Vec<_>>();
    let eigen_scale = raw_eigenvalues
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    let tolerance = eigen_scale.max(1.0) * 1e-10;
    if raw_eigenvalues
        .iter()
        .any(|value| !value.is_finite() || *value < -tolerance)
    {
        return Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite);
    }
    let eigenvalues = raw_eigenvalues
        .into_iter()
        .map(|value| value.max(0.0))
        .collect::<Vec<_>>();
    let b_x = (0..realized_rank)
        .map(|component| {
            (0..date_count)
                .map(|row| {
                    scratch.whitened_factor[row * realized_rank + component]
                        * scratch.whitened_x[row]
                })
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let b_y = (0..realized_rank)
        .map(|component| {
            (0..date_count)
                .map(|row| {
                    scratch.whitened_factor[row * realized_rank + component]
                        * scratch.whitened_y[row]
                })
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let vectors = eigen.u();
    let c_x = (0..realized_rank)
        .map(|column| {
            (0..realized_rank)
                .map(|row| vectors[(row, column)] * b_x[row])
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let c_y = (0..realized_rank)
        .map(|column| {
            (0..realized_rank)
                .map(|row| vectors[(row, column)] * b_y[row])
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    Ok(SpectralProjection {
        rho,
        eigenvalues,
        c_x,
        c_y,
        a_x_x,
        a_x_y,
        a_y_y,
        log_shape_sum: scale.log_shape_sum,
        log_determinant_r,
        date_count,
        realized_rank,
    })
}

fn spectral_bilinear_derivatives(
    base: f64,
    left: &[f64],
    right: &[f64],
    eigenvalues: &[f64],
    process_variance: f64,
) -> (f64, f64, f64) {
    let (mut inverse_one, mut inverse_two, mut inverse_three) = (0.0, 0.0, 0.0);
    for ((left, right), eigenvalue) in left.iter().zip(right).zip(eigenvalues) {
        let product = left * right;
        let denominator = process_variance + eigenvalue;
        inverse_one += product / denominator;
        inverse_two += product / denominator.powi(2);
        inverse_three += product / denominator.powi(3);
    }
    let numerator = base - inverse_one;
    let first_numerator = inverse_two;
    let second_numerator = -2.0 * inverse_three;
    let value = numerator / process_variance;
    let first = first_numerator / process_variance - numerator / process_variance.powi(2);
    let second = second_numerator / process_variance
        - 2.0 * first_numerator / process_variance.powi(2)
        + 2.0 * numerator / process_variance.powi(3);
    (value, first, second)
}

fn spectral_q_evaluation(
    projection: &SpectralProjection,
    log_process_variance: f64,
) -> Result<SpectralQEvaluation, TemporalInferenceStatus> {
    let process_variance = log_process_variance.exp();
    if !process_variance.is_finite() || process_variance <= 0.0 {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let (x_v_x, x_v_x_first, x_v_x_second) = spectral_bilinear_derivatives(
        projection.a_x_x,
        &projection.c_x,
        &projection.c_x,
        &projection.eigenvalues,
        process_variance,
    );
    let (x_v_y, x_v_y_first, x_v_y_second) = spectral_bilinear_derivatives(
        projection.a_x_y,
        &projection.c_x,
        &projection.c_y,
        &projection.eigenvalues,
        process_variance,
    );
    let (y_v_y, y_v_y_first, y_v_y_second) = spectral_bilinear_derivatives(
        projection.a_y_y,
        &projection.c_y,
        &projection.c_y,
        &projection.eigenvalues,
        process_variance,
    );
    if !x_v_x.is_finite() || x_v_x <= 0.0 || !x_v_y.is_finite() || !y_v_y.is_finite() {
        return Err(TemporalInferenceStatus::DesignRankDeficient);
    }
    let slope = x_v_y / x_v_x;
    let quadratic = y_v_y - x_v_y.powi(2) / x_v_x;
    let quadratic_first = y_v_y_first - 2.0 * x_v_y * x_v_y_first / x_v_x
        + x_v_y.powi(2) * x_v_x_first / x_v_x.powi(2);
    let quadratic_second = y_v_y_second
        - 2.0 * (x_v_y_first.powi(2) + x_v_y * x_v_y_second) / x_v_x
        + 4.0 * x_v_y * x_v_y_first * x_v_x_first / x_v_x.powi(2)
        + x_v_y.powi(2) * x_v_x_second / x_v_x.powi(2)
        - 2.0 * x_v_y.powi(2) * x_v_x_first.powi(2) / x_v_x.powi(3);
    let log_determinant = 2.0 * projection.log_shape_sum
        + (projection.date_count - projection.realized_rank) as f64 * log_process_variance
        + projection.log_determinant_r
        + projection
            .eigenvalues
            .iter()
            .map(|eigenvalue| (process_variance + eigenvalue).ln())
            .sum::<f64>();
    let log_determinant_first = (projection.date_count - projection.realized_rank) as f64
        / process_variance
        + projection
            .eigenvalues
            .iter()
            .map(|eigenvalue| 1.0 / (process_variance + eigenvalue))
            .sum::<f64>();
    let log_determinant_second = -((projection.date_count - projection.realized_rank) as f64)
        / process_variance.powi(2)
        - projection
            .eigenvalues
            .iter()
            .map(|eigenvalue| 1.0 / (process_variance + eigenvalue).powi(2))
            .sum::<f64>();
    let score = log_determinant + quadratic + x_v_x.ln();
    let score_first = log_determinant_first + quadratic_first + x_v_x_first / x_v_x;
    let score_second = log_determinant_second + quadratic_second + x_v_x_second / x_v_x
        - (x_v_x_first / x_v_x).powi(2);
    let score_gradient_log_q = process_variance * score_first;
    let score_curvature_log_q =
        process_variance * score_first + process_variance.powi(2) * score_second;
    if !score.is_finite()
        || !slope.is_finite()
        || !quadratic.is_finite()
        || !score_gradient_log_q.is_finite()
        || !score_curvature_log_q.is_finite()
    {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    Ok(SpectralQEvaluation {
        score,
        slope,
        information_variance: 1.0 / x_v_x,
        process_variance,
        score_gradient_log_q,
        score_curvature_log_q,
    })
}

fn spectral_fixed_slope_q_score(
    projection: &SpectralProjection,
    log_process_variance: f64,
    slope: f64,
) -> Result<f64, TemporalInferenceStatus> {
    let process_variance = log_process_variance.exp();
    if !process_variance.is_finite() || process_variance <= 0.0 || !slope.is_finite() {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let base = projection.a_y_y - 2.0 * slope * projection.a_x_y + slope.powi(2) * projection.a_x_x;
    let mut projected = 0.0;
    let mut quadratic = 0.0;
    for ((c_x, c_y), eigenvalue) in projection
        .c_x
        .iter()
        .zip(&projection.c_y)
        .zip(&projection.eigenvalues)
    {
        let coefficient = c_y - slope * c_x;
        if *eigenvalue == 0.0 {
            if coefficient.abs() > 1e-10 * base.abs().sqrt().max(1.0) {
                return Err(TemporalInferenceStatus::CovarianceNonfinite);
            }
            continue;
        }
        let projected_component = coefficient.powi(2) / eigenvalue;
        projected += projected_component;
        quadratic += projected_component / (process_variance + eigenvalue);
    }
    let null_component = if projection.realized_rank == projection.date_count {
        0.0
    } else {
        let residual = base - projected;
        let tolerance = 1e-10 * base.abs().max(projected.abs()).max(1.0);
        if residual < -tolerance {
            return Err(TemporalInferenceStatus::CovarianceNonfinite);
        }
        residual.max(0.0)
    };
    quadratic += null_component / process_variance;
    let log_determinant = 2.0 * projection.log_shape_sum
        + (projection.date_count - projection.realized_rank) as f64 * log_process_variance
        + projection.log_determinant_r
        + projection
            .eigenvalues
            .iter()
            .map(|eigenvalue| (process_variance + eigenvalue).ln())
            .sum::<f64>();
    let score = log_determinant + quadratic;
    if !score.is_finite() || !quadratic.is_finite() {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    Ok(score)
}

fn profile_spectral_process_variance(
    projection: &SpectralProjection,
    bounds: NuisanceBounds,
    options: &TemporalCovarianceOptions,
) -> Result<SpectralQEvaluation, TemporalInferenceStatus> {
    let lower = spectral_q_evaluation(projection, bounds.log_variance_lower)?;
    let upper = spectral_q_evaluation(projection, bounds.log_variance_upper)?;
    if lower.score_gradient_log_q >= 0.0 {
        return Ok(lower);
    }
    if upper.score_gradient_log_q <= 0.0 {
        return Ok(upper);
    }
    let mut bracket = [bounds.log_variance_lower, bounds.log_variance_upper];
    let mut current = bounds.initial_log_variance.clamp(bracket[0], bracket[1]);
    let mut best = spectral_q_evaluation(projection, current)?;
    for _ in 0..12 {
        let evaluated = spectral_q_evaluation(projection, current)?;
        if evaluated.score < best.score {
            best = evaluated;
        }
        let objective_tolerance = options.optimizer_tolerance * (1.0 + evaluated.score.abs());
        let newton = current - evaluated.score_gradient_log_q / evaluated.score_curvature_log_q;
        if evaluated.score_curvature_log_q > 0.0
            && evaluated.score_gradient_log_q.powi(2) / (2.0 * evaluated.score_curvature_log_q)
                <= objective_tolerance
            && (newton - current).abs() <= options.optimizer_tolerance * (1.0 + current.abs())
        {
            return Ok(best);
        }
        if evaluated.score_gradient_log_q > 0.0 {
            bracket[1] = current;
        } else {
            bracket[0] = current;
        }
        current = if evaluated.score_curvature_log_q > 0.0
            && newton.is_finite()
            && newton > bracket[0]
            && newton < bracket[1]
        {
            newton
        } else {
            (bracket[0] + bracket[1]) / 2.0
        };
        if (bracket[1] - bracket[0]).abs() <= options.optimizer_tolerance * (1.0 + current.abs()) {
            let evaluated = spectral_q_evaluation(projection, current)?;
            return Ok(if evaluated.score < best.score {
                evaluated
            } else {
                best
            });
        }
    }
    let evaluated = spectral_q_evaluation(projection, current)?;
    if evaluated.score < best.score {
        best = evaluated;
    }
    Ok(best)
}

#[allow(clippy::too_many_arguments)]
fn spectral_profile_at_rho(
    prepared: &PreparedFactorObjective,
    rho: f64,
    realized_rank: usize,
    scale: SpectralTargetScale,
    bounds: NuisanceBounds,
    options: &TemporalCovarianceOptions,
    scratch: &mut FactorObjectiveScratch,
    factorization_count: &mut usize,
) -> Result<SpectralProfileAtRho, TemporalInferenceStatus> {
    let projection = spectral_projection(prepared, rho, realized_rank, scale, scratch)?;
    *factorization_count = factorization_count.saturating_add(1);
    let fit = profile_spectral_process_variance(&projection, bounds, options)?;
    Ok(SpectralProfileAtRho { projection, fit })
}

pub(super) fn total_covariance_from_factor(
    prepared: &PreparedFactorObjective,
    factor: &[f64],
    maximum_rank: usize,
    realized_rank: usize,
    rho: f64,
    process_variance: f64,
) -> Result<Vec<Vec<f64>>, TemporalInferenceStatus> {
    let difference_covariance = difference_covariance_from_factor(
        prepared.design.len(),
        factor,
        maximum_rank,
        realized_rank,
    );
    total_difference_covariance(
        &difference_covariance,
        &prepared.design,
        process_variance,
        rho,
        prepared.reference_lag_days,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FactorConditionMethod {
    ConservativeUpperBound,
    ExactEigenvalueFallback,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FactorConditionCertificate {
    pub(super) method: FactorConditionMethod,
    pub(super) reported_condition: f64,
    pub(super) conservative_upper_bound: f64,
    pub(super) exact_condition_number: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub(super) fn factor_condition_certificate(
    prepared: &PreparedFactorObjective,
    factor: &[f64],
    maximum_rank: usize,
    realized_rank: usize,
    rho: f64,
    process_variance: f64,
    condition_limit: f64,
) -> Result<FactorConditionCertificate, TemporalInferenceStatus> {
    let date_count = prepared.design.len();
    if date_count == 0
        || maximum_rank == 0
        || realized_rank == 0
        || realized_rank > maximum_rank
        || factor.len() != date_count.saturating_mul(maximum_rank)
        || factor.iter().any(|value| !value.is_finite())
        || !rho.is_finite()
        || !(0.0..1.0).contains(&rho)
        || !process_variance.is_finite()
        || process_variance <= 0.0
        || !condition_limit.is_finite()
        || condition_limit <= 0.0
    {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }

    let mut diagonal = Vec::with_capacity(date_count);
    let mut factor_frobenius_squared = 0.0;
    for row in 0..date_count {
        let row_squared = (0..realized_rank)
            .map(|component| factor[row * maximum_rank + component].powi(2))
            .sum::<f64>();
        if !row_squared.is_finite() || row_squared <= 0.0 {
            return Err(TemporalInferenceStatus::CovarianceNonfinite);
        }
        diagonal.push(row_squared);
        factor_frobenius_squared += row_squared;
    }
    let shape = relative_standard_deviation_shape(&diagonal)?;
    let minimum_shape = shape.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum_shape = shape.iter().copied().fold(0.0, f64::max);

    let mut transitions = Vec::with_capacity(date_count.saturating_sub(1));
    let mut innovations = Vec::with_capacity(date_count.saturating_sub(1));
    for &gap_exponent in &prepared.gap_exponents {
        let phi = if rho == 0.0 {
            0.0
        } else {
            (rho.ln() * gap_exponent).exp()
        };
        let innovation = if rho == 0.0 {
            1.0
        } else {
            -(2.0 * rho.ln() * gap_exponent).exp_m1()
        };
        if !phi.is_finite() || !innovation.is_finite() || innovation <= 0.0 {
            return Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite);
        }
        transitions.push(phi);
        innovations.push(innovation);
    }

    let mut left_correlation_sum = vec![0.0; date_count];
    for row in 1..date_count {
        left_correlation_sum[row] = transitions[row - 1] * (1.0 + left_correlation_sum[row - 1]);
    }
    let mut right_correlation_sum = vec![0.0; date_count];
    for row in (0..date_count.saturating_sub(1)).rev() {
        right_correlation_sum[row] = transitions[row] * (1.0 + right_correlation_sum[row + 1]);
    }
    let maximum_correlation_row_sum = (0..date_count)
        .map(|row| 1.0 + left_correlation_sum[row] + right_correlation_sum[row])
        .fold(0.0, f64::max);

    let maximum_precision_row_sum = if date_count == 1 {
        1.0
    } else {
        (0..date_count)
            .map(|row| {
                let diagonal = if row == 0 {
                    1.0 / innovations[0]
                } else if row + 1 == date_count {
                    1.0 / innovations[row - 1]
                } else {
                    1.0 / innovations[row - 1] + transitions[row].powi(2) / innovations[row]
                };
                diagonal
                    + if row > 0 {
                        transitions[row - 1] / innovations[row - 1]
                    } else {
                        0.0
                    }
                    + if row + 1 < date_count {
                        transitions[row] / innovations[row]
                    } else {
                        0.0
                    }
            })
            .fold(0.0, f64::max)
    };
    let upper_eigenvalue = factor_frobenius_squared
        + process_variance * maximum_shape.powi(2) * maximum_correlation_row_sum;
    let lower_eigenvalue = process_variance * minimum_shape.powi(2) / maximum_precision_row_sum;
    let conservative_upper_bound = upper_eigenvalue / lower_eigenvalue;
    if conservative_upper_bound.is_finite() && conservative_upper_bound <= condition_limit {
        return Ok(FactorConditionCertificate {
            method: FactorConditionMethod::ConservativeUpperBound,
            reported_condition: conservative_upper_bound,
            conservative_upper_bound,
            exact_condition_number: None,
        });
    }

    let covariance = total_covariance_from_factor(
        prepared,
        factor,
        maximum_rank,
        realized_rank,
        rho,
        process_variance,
    )?;
    let exact_condition_number = condition_number(&covariance);
    if !exact_condition_number.is_finite() || exact_condition_number > condition_limit {
        return Err(TemporalInferenceStatus::DesignIllConditioned);
    }
    Ok(FactorConditionCertificate {
        method: FactorConditionMethod::ExactEigenvalueFallback,
        reported_condition: exact_condition_number,
        conservative_upper_bound,
        exact_condition_number: Some(exact_condition_number),
    })
}

pub(super) fn difference_covariance_from_factor(
    date_count: usize,
    factor: &[f64],
    maximum_rank: usize,
    realized_rank: usize,
) -> Vec<Vec<f64>> {
    (0..date_count)
        .map(|left| {
            (0..date_count)
                .map(|right| {
                    (0..realized_rank)
                        .map(|component| {
                            factor[left * maximum_rank + component]
                                * factor[right * maximum_rank + component]
                        })
                        .sum()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn dense_factor_profile_fallback(
    prepared: &PreparedFactorObjective,
    observations: &[f64],
    factor: &[f64],
    maximum_rank: usize,
    realized_rank: usize,
    options: &TemporalCovarianceOptions,
    bounds: NuisanceBounds,
    _primary_factorization_count: usize,
    accept_boundary_solution: bool,
) -> Result<FactorProfileFit, TemporalInferenceStatus> {
    let difference_covariance = difference_covariance_from_factor(
        prepared.design.len(),
        factor,
        maximum_rank,
        realized_rank,
    );
    let fit = optimize_covariance(
        &prepared.design,
        observations,
        &difference_covariance,
        options,
        bounds,
        true,
        accept_boundary_solution,
    )?;
    #[cfg(test)]
    let score = covariance_objective(
        &prepared.design,
        observations,
        &difference_covariance,
        fit.rho,
        fit.process_variance.ln(),
        options,
        true,
    )?
    .0;
    Ok(FactorProfileFit {
        #[cfg(test)]
        score,
        slope: fit.slope,
        rho: fit.rho,
        process_variance: fit.process_variance,
        information_variance: fit.information_variance,
        condition_number: fit.condition_number,
        #[cfg(test)]
        primary_factorization_count: _primary_factorization_count,
        #[cfg(test)]
        profile_rho_curvature: f64::NAN,
        dense_fallback_count: 1,
    })
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn factor_native_profile_plugin(
    prepared: &PreparedFactorObjective,
    observations: &[f64],
    factor: &[f64],
    maximum_rank: usize,
    realized_rank: usize,
    options: &TemporalCovarianceOptions,
    scratch: &mut FactorObjectiveScratch,
    accept_boundary_solution: bool,
) -> Result<FactorProfileFit, TemporalInferenceStatus> {
    let bounds = nuisance_bounds(&prepared.design, observations, options)?;
    let scale = prepare_spectral_target(
        prepared,
        observations,
        factor,
        maximum_rank,
        realized_rank,
        scratch,
    )?;
    let mut factorization_count = 0_usize;
    let primary = (|| {
        let lower = spectral_profile_at_rho(
            prepared,
            bounds.rho_lower,
            realized_rank,
            scale,
            bounds,
            options,
            scratch,
            &mut factorization_count,
        )?;
        let upper = spectral_profile_at_rho(
            prepared,
            bounds.rho_upper,
            realized_rank,
            scale,
            bounds,
            options,
            scratch,
            &mut factorization_count,
        )?;
        let mut interval = [bounds.rho_lower, bounds.rho_upper];
        let mut x = (interval[0] + interval[1]) / 2.0;
        let mut current = spectral_profile_at_rho(
            prepared,
            x,
            realized_rank,
            scale,
            bounds,
            options,
            scratch,
            &mut factorization_count,
        )?;
        let mut best = [lower, upper]
            .into_iter()
            .min_by(|left, right| left.fit.score.total_cmp(&right.fit.score))
            .unwrap();
        if current.fit.score < best.fit.score {
            best = current.clone();
        }
        let (mut w, mut v) = (x, x);
        let (mut score_w, mut score_v) = (current.fit.score, current.fit.score);
        let (mut step, mut previous_step) = (0.0_f64, 0.0_f64);
        let mut converged = false;
        for _ in 0..14 {
            let midpoint = (interval[0] + interval[1]) / 2.0;
            let tolerance = options.optimizer_tolerance * (1.0 + x.abs());
            if (x - midpoint).abs() <= 2.0 * tolerance - (interval[1] - interval[0]) / 2.0 {
                converged = true;
                break;
            }
            let mut proposed = None;
            if previous_step.abs() > tolerance {
                let r = (x - w) * (current.fit.score - score_v);
                let mut q = (x - v) * (current.fit.score - score_w);
                let mut p = (x - v) * q - (x - w) * r;
                q = 2.0 * (q - r);
                if q > 0.0 {
                    p = -p;
                } else {
                    q = -q;
                }
                if q > 0.0
                    && p.abs() < (0.5 * q * previous_step).abs()
                    && p > q * (interval[0] - x)
                    && p < q * (interval[1] - x)
                {
                    proposed = Some(x + p / q);
                }
            }
            previous_step = step;
            let golden = 0.381_966_011_250_105_1;
            step = proposed.map_or_else(
                || {
                    if x < midpoint {
                        golden * (interval[1] - x)
                    } else {
                        golden * (interval[0] - x)
                    }
                },
                |candidate| candidate - x,
            );
            let candidate_rho = if step.abs() >= tolerance {
                x + step
            } else {
                x + tolerance.copysign(step)
            };
            let candidate = spectral_profile_at_rho(
                prepared,
                candidate_rho.clamp(interval[0], interval[1]),
                realized_rank,
                scale,
                bounds,
                options,
                scratch,
                &mut factorization_count,
            )?;
            if candidate.fit.score < best.fit.score {
                best = candidate.clone();
            }
            if candidate.fit.score <= current.fit.score {
                if candidate.projection.rho < x {
                    interval[1] = x;
                } else {
                    interval[0] = x;
                }
                v = w;
                score_v = score_w;
                w = x;
                score_w = current.fit.score;
                x = candidate.projection.rho;
                current = candidate;
            } else {
                if candidate.projection.rho < x {
                    interval[0] = candidate.projection.rho;
                } else {
                    interval[1] = candidate.projection.rho;
                }
                if candidate.fit.score <= score_w || w == x {
                    v = w;
                    score_v = score_w;
                    w = candidate.projection.rho;
                    score_w = candidate.fit.score;
                } else if candidate.fit.score <= score_v || v == x || v == w {
                    v = candidate.projection.rho;
                    score_v = candidate.fit.score;
                }
            }
        }
        let final_rho_interval_width = interval[1] - interval[0];
        let covariance = total_covariance_from_factor(
            prepared,
            factor,
            maximum_rank,
            realized_rank,
            best.projection.rho,
            best.fit.process_variance,
        )?;
        factorization_count = factorization_count.saturating_add(1);
        let condition = condition_number(&covariance);
        if !condition.is_finite() || condition > options.condition_limit {
            return Err(TemporalInferenceStatus::DesignIllConditioned);
        }
        if let Some(status) = temporal_parameter_boundary_status(
            best.projection.rho,
            best.fit.process_variance,
            [bounds.rho_lower, bounds.rho_upper],
            [
                bounds.log_variance_lower.exp(),
                bounds.log_variance_upper.exp(),
            ],
            options.optimizer_tolerance * 0.01,
        ) {
            if !accept_boundary_solution {
                return Err(status);
            }
            return Ok(FactorProfileFit {
                #[cfg(test)]
                score: best.fit.score,
                slope: best.fit.slope,
                rho: best.projection.rho,
                process_variance: best.fit.process_variance,
                information_variance: best.fit.information_variance,
                condition_number: condition,
                #[cfg(test)]
                primary_factorization_count: factorization_count,
                #[cfg(test)]
                profile_rho_curvature: f64::NAN,
                dense_fallback_count: 0,
            });
        }
        let rho_step = 1e-3_f64
            .min((best.projection.rho - bounds.rho_lower) / 2.0)
            .min((bounds.rho_upper - best.projection.rho) / 2.0);
        let log_step = 1e-2;
        if rho_step <= 0.0 {
            return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
        }
        let rho_plus = spectral_projection(
            prepared,
            best.projection.rho + rho_step,
            realized_rank,
            scale,
            scratch,
        )?;
        factorization_count = factorization_count.saturating_add(1);
        let rho_minus = spectral_projection(
            prepared,
            best.projection.rho - rho_step,
            realized_rank,
            scale,
            scratch,
        )?;
        factorization_count = factorization_count.saturating_add(1);
        let rho_curvature = (spectral_q_evaluation(&rho_plus, best.fit.process_variance.ln())?
            .score
            + spectral_q_evaluation(&rho_minus, best.fit.process_variance.ln())?.score
            - 2.0 * best.fit.score)
            / rho_step.powi(2);
        let variance_curvature =
            (spectral_q_evaluation(&best.projection, best.fit.process_variance.ln() + log_step)?
                .score
                + spectral_q_evaluation(
                    &best.projection,
                    best.fit.process_variance.ln() - log_step,
                )?
                .score
                - 2.0 * best.fit.score)
                / log_step.powi(2);
        if !rho_curvature.is_finite()
            || !variance_curvature.is_finite()
            || rho_curvature <= options.minimum_profile_curvature
            || variance_curvature <= options.minimum_profile_curvature
        {
            return Err(TemporalInferenceStatus::WeakParameterIdentification);
        }
        let curvature_rho_tolerance =
            (2.0 * options.optimizer_tolerance * (1.0 + best.fit.score.abs()) / rho_curvature)
                .sqrt()
                .min(5e-3);
        if !converged && final_rho_interval_width > curvature_rho_tolerance {
            return Err(TemporalInferenceStatus::OptimizerNonconverged);
        }
        Ok(FactorProfileFit {
            #[cfg(test)]
            score: best.fit.score,
            slope: best.fit.slope,
            rho: best.projection.rho,
            process_variance: best.fit.process_variance,
            information_variance: best.fit.information_variance,
            condition_number: condition,
            #[cfg(test)]
            primary_factorization_count: factorization_count,
            #[cfg(test)]
            profile_rho_curvature: rho_curvature,
            dense_fallback_count: 0,
        })
    })();
    match primary {
        Ok(fit) => Ok(fit),
        Err(TemporalInferenceStatus::OptimizerNonconverged)
        | Err(TemporalInferenceStatus::CovarianceNonfinite)
        | Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite)
        | Err(TemporalInferenceStatus::DesignRankDeficient)
        | Err(TemporalInferenceStatus::WeakParameterIdentification) => {
            dense_factor_profile_fallback(
                prepared,
                observations,
                factor,
                maximum_rank,
                realized_rank,
                options,
                bounds,
                factorization_count,
                accept_boundary_solution,
            )
        }
        Err(status) => Err(status),
    }
}

pub(super) fn empty_comparator(status: TemporalInferenceStatus) -> ComparatorDiagnostics {
    ComparatorDiagnostics {
        point_estimate: None,
        standard_error_diagnostic: None,
        interval_68: None,
        interval_90: None,
        interval_95: None,
        width_68: None,
        width_90: None,
        width_95: None,
        status,
        attempted_replicates: 0,
        successful_replicates: 0,
    }
}

struct BootstrapSummary {
    interval_68: Option<ValidationInterval>,
    interval_90: Option<ValidationInterval>,
    interval_95: Option<ValidationInterval>,
    attempts: usize,
    successes: usize,
    variance: f64,
    minimum_successes: usize,
}

fn required_bootstrap_successes(attempts: usize) -> usize {
    attempts.saturating_mul(99).saturating_add(99) / 100
}

pub(super) fn normal_comparator(
    slope_per_day: f64,
    standard_error_per_day: f64,
    status: TemporalInferenceStatus,
) -> ComparatorDiagnostics {
    let normal = Normal::new(0.0, 1.0).expect("standard normal parameters are valid");
    let z_68 = normal.inverse_cdf(0.84);
    let z_90 = normal.inverse_cdf(0.95);
    let z_95 = normal.inverse_cdf(0.975);
    let point = slope_per_day * DAYS_PER_YEAR;
    let standard_error = standard_error_per_day * DAYS_PER_YEAR;
    ComparatorDiagnostics {
        point_estimate: point.is_finite().then_some(point),
        standard_error_diagnostic: standard_error.is_finite().then_some(standard_error),
        interval_68: interval(point, standard_error, z_68, 0, 0),
        interval_90: interval(point, standard_error, z_90, 0, 0),
        interval_95: interval(point, standard_error, z_95, 0, 0),
        width_68: Some(2.0 * z_68 * standard_error),
        width_90: Some(2.0 * z_90 * standard_error),
        width_95: Some(2.0 * z_95 * standard_error),
        status,
        attempted_replicates: 0,
        successful_replicates: 0,
    }
}

#[allow(clippy::too_many_lines)]
pub(super) fn reml_covariance_parameter_adjusted_variance(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    fitted_rho: f64,
    fitted_process_variance: f64,
    information_variance: f64,
    options: &TemporalCovarianceOptions,
) -> Result<f64, TemporalInferenceStatus> {
    let bounds = nuisance_bounds(days, observations, options)?;
    let (active_set, rho_active, variance_active) = nuisance_parameter_active_set(
        fitted_rho,
        fitted_process_variance,
        [bounds.rho_lower, bounds.rho_upper],
        [
            bounds.log_variance_lower.exp(),
            bounds.log_variance_upper.exp(),
        ],
        options.optimizer_tolerance * 0.01,
    );
    let rho_step = 1e-3_f64
        .min((fitted_rho - bounds.rho_lower) / 2.0)
        .min((bounds.rho_upper - fitted_rho) / 2.0);
    let log_step = 1e-2_f64
        .min((fitted_process_variance.ln() - bounds.log_variance_lower) / 2.0)
        .min((bounds.log_variance_upper - fitted_process_variance.ln()) / 2.0);
    let theta = [fitted_rho, fitted_process_variance.ln()];
    let objective = |rho: f64, log_variance: f64| {
        covariance_objective(
            days,
            observations,
            difference_covariance,
            rho,
            log_variance,
            options,
            true,
        )
    };
    let (base, _) = objective(theta[0], theta[1])?;
    let covariance_scale = 2.0;
    let nuisance_variance = if active_set
        == Some(TemporalInferenceStatus::ProcessVarianceLowerBoundary)
    {
        0.0
    } else {
        match (rho_active, variance_active) {
            (true, true) => 0.0,
            (true, false) => {
                if log_step <= 0.0 {
                    return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
                }
                let (log_plus, log_plus_fit) = objective(theta[0], theta[1] + log_step)?;
                let (log_minus, log_minus_fit) = objective(theta[0], theta[1] - log_step)?;
                let curvature = (log_plus + log_minus - 2.0 * base) / log_step.powi(2);
                let slope_gradient = (log_plus_fit.slope - log_minus_fit.slope) / (2.0 * log_step);
                if !curvature.is_finite() || curvature <= options.minimum_profile_curvature {
                    return Err(TemporalInferenceStatus::WeakParameterIdentification);
                }
                covariance_scale * slope_gradient.powi(2) / curvature
            }
            (false, true) => {
                if rho_step <= 0.0 {
                    return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
                }
                let (rho_plus, rho_plus_fit) = objective(theta[0] + rho_step, theta[1])?;
                let (rho_minus, rho_minus_fit) = objective(theta[0] - rho_step, theta[1])?;
                let curvature = (rho_plus + rho_minus - 2.0 * base) / rho_step.powi(2);
                let slope_gradient = (rho_plus_fit.slope - rho_minus_fit.slope) / (2.0 * rho_step);
                if !curvature.is_finite() || curvature <= options.minimum_profile_curvature {
                    return Err(TemporalInferenceStatus::WeakParameterIdentification);
                }
                covariance_scale * slope_gradient.powi(2) / curvature
            }
            (false, false) => {
                if rho_step <= 0.0 || log_step <= 0.0 {
                    return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
                }
                let (rho_plus, rho_plus_fit) = objective(theta[0] + rho_step, theta[1])?;
                let (rho_minus, rho_minus_fit) = objective(theta[0] - rho_step, theta[1])?;
                let (log_plus, log_plus_fit) = objective(theta[0], theta[1] + log_step)?;
                let (log_minus, log_minus_fit) = objective(theta[0], theta[1] - log_step)?;
                let cross_pp = objective(theta[0] + rho_step, theta[1] + log_step)?.0;
                let cross_pm = objective(theta[0] + rho_step, theta[1] - log_step)?.0;
                let cross_mp = objective(theta[0] - rho_step, theta[1] + log_step)?.0;
                let cross_mm = objective(theta[0] - rho_step, theta[1] - log_step)?.0;
                let h00 = (rho_plus + rho_minus - 2.0 * base) / rho_step.powi(2);
                let h11 = (log_plus + log_minus - 2.0 * base) / log_step.powi(2);
                let h01 = (cross_pp - cross_pm - cross_mp + cross_mm) / (4.0 * rho_step * log_step);
                let determinant = h00 * h11 - h01 * h01;
                if !h00.is_finite()
                    || !h11.is_finite()
                    || !h01.is_finite()
                    || h00 <= options.minimum_profile_curvature
                    || h11 <= options.minimum_profile_curvature
                    || determinant <= 0.0
                {
                    return Err(TemporalInferenceStatus::WeakParameterIdentification);
                }
                let slope_gradient = [
                    (rho_plus_fit.slope - rho_minus_fit.slope) / (2.0 * rho_step),
                    (log_plus_fit.slope - log_minus_fit.slope) / (2.0 * log_step),
                ];
                covariance_scale
                    * (h11 * slope_gradient[0].powi(2)
                        - 2.0 * h01 * slope_gradient[0] * slope_gradient[1]
                        + h00 * slope_gradient[1].powi(2))
                    / determinant
            }
        }
    };
    let adjusted_variance = information_variance + nuisance_variance.max(0.0);
    if !adjusted_variance.is_finite() || adjusted_variance <= 0.0 {
        return Err(TemporalInferenceStatus::WeakParameterIdentification);
    }
    Ok(adjusted_variance)
}

fn reml_adjusted_scalar_comparator(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    plugin: &PluginFit,
    residual_dof: usize,
    options: &TemporalCovarianceOptions,
) -> Result<ComparatorDiagnostics, TemporalInferenceStatus> {
    #[cfg(test)]
    DENSE_ADJUSTED_SCALAR_CALLS.with(|calls| calls.set(calls.get() + 1));
    if residual_dof == 0 {
        return Err(TemporalInferenceStatus::WeakParameterIdentification);
    }
    let adjusted_variance = reml_covariance_parameter_adjusted_variance(
        days,
        observations,
        difference_covariance,
        plugin.rho,
        plugin.process_variance,
        plugin.information_variance,
        options,
    )?;
    let degrees_of_freedom = residual_dof as f64;
    let distribution = StudentsT::new(0.0, 1.0, degrees_of_freedom)
        .map_err(|_| TemporalInferenceStatus::WeakParameterIdentification)?;
    let point = plugin.slope * DAYS_PER_YEAR;
    let standard_error = adjusted_variance.sqrt() * DAYS_PER_YEAR;
    let multipliers = [
        distribution.inverse_cdf(0.84),
        distribution.inverse_cdf(0.95),
        distribution.inverse_cdf(0.975),
    ];
    let intervals = multipliers.map(|multiplier| {
        interval(point, standard_error, multiplier, 0, 0)
            .ok_or(TemporalInferenceStatus::WeakParameterIdentification)
    });
    let [interval_68, interval_90, interval_95] = intervals;
    let interval_68 = interval_68?;
    let interval_90 = interval_90?;
    let interval_95 = interval_95?;
    Ok(ComparatorDiagnostics {
        point_estimate: Some(point),
        standard_error_diagnostic: Some(standard_error),
        interval_68: Some(interval_68),
        interval_90: Some(interval_90),
        interval_95: Some(interval_95),
        width_68: Some(interval_68.upper - interval_68.lower),
        width_90: Some(interval_90.upper - interval_90.lower),
        width_95: Some(interval_95.upper - interval_95.lower),
        status: TemporalInferenceStatus::Evaluated,
        attempted_replicates: 0,
        successful_replicates: 0,
    })
}

struct PreparedExactProfile<'a> {
    days: &'a [f64],
    observations: &'a [f64],
    difference_covariance: &'a [Vec<f64>],
    shape: Vec<f64>,
    gap_exponents: Vec<Vec<f64>>,
}

impl<'a> PreparedExactProfile<'a> {
    fn new(
        days: &'a [f64],
        observations: &'a [f64],
        difference_covariance: &'a [Vec<f64>],
        reference_lag_days: f64,
    ) -> Result<Self, TemporalInferenceStatus> {
        validate_square_covariance(difference_covariance)?;
        validate_dates(days)?;
        if difference_covariance.len() != days.len()
            || observations.len() != days.len()
            || !reference_lag_days.is_finite()
            || reference_lag_days <= 0.0
        {
            return Err(TemporalInferenceStatus::CovarianceNonfinite);
        }
        let diagonal = difference_covariance
            .iter()
            .enumerate()
            .map(|(index, row)| row[index])
            .collect::<Vec<_>>();
        let shape = relative_standard_deviation_shape(&diagonal)?;
        let gap_exponents = (0..days.len())
            .map(|row| {
                (0..days.len())
                    .map(|column| (days[row] - days[column]).abs() / reference_lag_days)
                    .collect()
            })
            .collect();
        Ok(Self {
            days,
            observations,
            difference_covariance,
            shape,
            gap_exponents,
        })
    }
}

struct ExactFixedSlopeWorkspace<'profile, 'data> {
    prepared: &'profile PreparedExactProfile<'data>,
    residuals: Vec<f64>,
    correlation: Vec<Vec<f64>>,
    covariance: Vec<Vec<f64>>,
    lower: Vec<Vec<f64>>,
    forward: Vec<f64>,
    solution: Vec<f64>,
}

impl<'profile, 'data> ExactFixedSlopeWorkspace<'profile, 'data> {
    fn new(prepared: &'profile PreparedExactProfile<'data>, slope: f64) -> Self {
        let dimension = prepared.days.len();
        Self {
            prepared,
            residuals: prepared
                .observations
                .iter()
                .zip(prepared.days)
                .map(|(value, day)| value - slope * day)
                .collect(),
            correlation: vec![vec![0.0; dimension]; dimension],
            covariance: vec![vec![0.0; dimension]; dimension],
            lower: vec![vec![0.0; dimension]; dimension],
            forward: vec![0.0; dimension],
            solution: vec![0.0; dimension],
        }
    }

    fn prepare_rho(&mut self, rho: f64) -> Result<(), TemporalInferenceStatus> {
        if !rho.is_finite() || !(0.0..1.0).contains(&rho) {
            return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
        }
        for row in 0..self.correlation.len() {
            for column in 0..self.correlation.len() {
                self.correlation[row][column] = if row == column {
                    1.0
                } else if rho == 0.0 {
                    0.0
                } else {
                    rho.powf(self.prepared.gap_exponents[row][column])
                };
            }
        }
        Ok(())
    }

    fn score(&mut self, log_process_variance: f64) -> Result<f64, TemporalInferenceStatus> {
        let process_variance = log_process_variance.exp();
        if !process_variance.is_finite() || process_variance < 0.0 {
            return Err(TemporalInferenceStatus::CovarianceNonfinite);
        }
        let dimension = self.covariance.len();
        for row in 0..dimension {
            for column in 0..dimension {
                self.covariance[row][column] = self.prepared.difference_covariance[row][column]
                    + process_variance
                        * self.prepared.shape[row]
                        * self.correlation[row][column]
                        * self.prepared.shape[column];
            }
        }
        if !cholesky_into(&self.covariance, &mut self.lower) {
            return Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite);
        }
        solve_cholesky_into(
            &self.lower,
            &self.residuals,
            &mut self.forward,
            &mut self.solution,
        );
        let quadratic = dot(&self.residuals, &self.solution);
        solve_cholesky_into(
            &self.lower,
            self.prepared.days,
            &mut self.forward,
            &mut self.solution,
        );
        let information = dot(self.prepared.days, &self.solution);
        if !information.is_finite() || information <= 0.0 {
            return Err(TemporalInferenceStatus::DesignRankDeficient);
        }
        let log_determinant = 2.0
            * self
                .lower
                .iter()
                .enumerate()
                .map(|(index, row)| row[index].ln())
                .sum::<f64>();
        Ok(log_determinant + quadratic)
    }
}

struct FactorProfileProblem<'problem, 'data> {
    prepared: &'problem PreparedFactorObjective,
    observations: &'data [f64],
    factor: &'problem [f64],
    maximum_rank: usize,
    realized_rank: usize,
    dense_prepared: &'problem PreparedExactProfile<'data>,
}

#[allow(clippy::too_many_lines)]
fn factor_profile_comparator(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    factor_profile: &PreparedFactorProfileInput,
    options: &TemporalCovarianceOptions,
) -> Result<ComparatorDiagnostics, TemporalInferenceStatus> {
    if options.profile_max_iterations == 0 || options.profile_max_expansions == 0 {
        return Err(TemporalInferenceStatus::OptimizerNonconverged);
    }
    let bounds = nuisance_bounds(days, observations, options)?;
    let ml_fit = optimize_covariance(
        days,
        observations,
        difference_covariance,
        options,
        bounds,
        false,
        true,
    )?;
    let point = ml_fit.slope * DAYS_PER_YEAR;
    let unrestricted_objective = covariance_objective(
        days,
        observations,
        difference_covariance,
        ml_fit.rho,
        ml_fit.process_variance.ln(),
        options,
        false,
    )?
    .0;
    let prepared = PreparedFactorObjective::new(days, options.reference_lag_days)?;
    let dense_prepared = PreparedExactProfile::new(
        days,
        observations,
        difference_covariance,
        options.reference_lag_days,
    )?;
    let problem = FactorProfileProblem {
        prepared: &prepared,
        observations,
        factor: &factor_profile.compact_factor,
        maximum_rank: factor_profile.maximum_rank,
        realized_rank: factor_profile.realized_rank,
        dense_prepared: &dense_prepared,
    };
    let levels = [0.68, 0.90, 0.95];
    let interval_results = levels
        .into_par_iter()
        .map(|level| {
            let chi_square = ChiSquared::new(1.0)
                .map_err(|_| TemporalInferenceStatus::WeakParameterIdentification)?
                .inverse_cdf(level);
            let target = unrestricted_objective + chi_square;
            let scale = ml_fit.information_variance.sqrt().max(1e-10);
            let (lower, upper) = factor_profile_endpoint_pair(
                &problem,
                ml_fit.slope,
                scale,
                target,
                bounds,
                options,
            )?;
            Ok(ValidationInterval {
                lower: lower * DAYS_PER_YEAR,
                upper: upper * DAYS_PER_YEAR,
                successful_replicates: 0,
            })
        })
        .collect::<Vec<Result<_, TemporalInferenceStatus>>>()
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let z_95 = Normal::new(0.0, 1.0)
        .expect("standard normal parameters are valid")
        .inverse_cdf(0.975);
    let standard_error = (interval_results[2].upper - interval_results[2].lower) / (2.0 * z_95);
    Ok(ComparatorDiagnostics {
        point_estimate: Some(point),
        standard_error_diagnostic: standard_error.is_finite().then_some(standard_error),
        interval_68: Some(interval_results[0]),
        interval_90: Some(interval_results[1]),
        interval_95: Some(interval_results[2]),
        width_68: Some(interval_results[0].upper - interval_results[0].lower),
        width_90: Some(interval_results[1].upper - interval_results[1].lower),
        width_95: Some(interval_results[2].upper - interval_results[2].lower),
        status: TemporalInferenceStatus::Evaluated,
        attempted_replicates: 0,
        successful_replicates: 0,
    })
}

#[allow(clippy::too_many_arguments)]
fn factor_profile_endpoint_pair(
    problem: &FactorProfileProblem<'_, '_>,
    point: f64,
    initial_scale: f64,
    target: f64,
    bounds: NuisanceBounds,
    options: &TemporalCovarianceOptions,
) -> Result<(f64, f64), TemporalInferenceStatus> {
    let objective = |slope: f64| profile_fixed_slope_factor(problem, slope, bounds, options);
    let mut lower = point - initial_scale;
    let mut upper = point + initial_scale;
    let mut bounded_values = None;
    for _ in 0..options.profile_max_expansions {
        let (lower_value, upper_value) = rayon::join(|| objective(lower), || objective(upper));
        let values = (lower_value?, upper_value?);
        if values.0 > target && values.1 > target {
            bounded_values = Some(values);
            break;
        }
        let span = (upper - lower) * 2.0;
        lower = point - span;
        upper = point + span;
    }
    let (lower_value, upper_value) = if let Some(values) = bounded_values {
        values
    } else {
        let (lower_value, upper_value) = rayon::join(|| objective(lower), || objective(upper));
        (lower_value?, upper_value?)
    };
    if lower_value <= target || upper_value <= target {
        return Err(TemporalInferenceStatus::OptimizerNonconverged);
    }
    let (lower, upper) = rayon::join(
        || solve_profile_endpoint(&objective, point, lower, target, options),
        || solve_profile_endpoint(&objective, point, upper, target, options),
    );
    Ok((lower?, upper?))
}

fn profile_fixed_slope_factor(
    problem: &FactorProfileProblem<'_, '_>,
    slope: f64,
    bounds: NuisanceBounds,
    options: &TemporalCovarianceOptions,
) -> Result<f64, TemporalInferenceStatus> {
    let primary = profile_fixed_slope_factor_primary(problem, slope, bounds, options);
    if let Ok((factor_score, rho, log_process_variance)) = primary {
        let mut dense = ExactFixedSlopeWorkspace::new(problem.dense_prepared, slope);
        if dense.prepare_rho(rho).is_ok() {
            if let Ok(dense_score) = dense.score(log_process_variance) {
                let scale = factor_score.abs().max(dense_score.abs()).max(1.0);
                if (factor_score - dense_score).abs() <= 1e-10 * scale {
                    return Ok(dense_score);
                }
            }
        }
    }
    profile_fixed_slope(problem.dense_prepared, slope, bounds, options)
}

fn profile_fixed_slope_factor_primary(
    problem: &FactorProfileProblem<'_, '_>,
    slope: f64,
    bounds: NuisanceBounds,
    options: &TemporalCovarianceOptions,
) -> Result<(f64, f64, f64), TemporalInferenceStatus> {
    if options.optimizer_max_iterations == 0 || options.optimizer_tolerance <= 0.0 {
        return Err(TemporalInferenceStatus::OptimizerNonconverged);
    }
    let nested_tolerance = options.optimizer_tolerance * 0.01;
    let mut scratch =
        FactorObjectiveScratch::new(problem.prepared.design.len(), problem.realized_rank);
    let scale = prepare_spectral_target(
        problem.prepared,
        problem.observations,
        problem.factor,
        problem.maximum_rank,
        problem.realized_rank,
        &mut scratch,
    )?;
    let mut profiled_score = |rho: f64| {
        let projection = match spectral_projection(
            problem.prepared,
            rho,
            problem.realized_rank,
            scale,
            &mut scratch,
        ) {
            Ok(projection) => projection,
            Err(_) => return f64::INFINITY,
        };
        let log_variance = adaptive_golden_section_minimum(
            bounds.log_variance_lower,
            bounds.log_variance_upper,
            nested_tolerance,
            |candidate| {
                spectral_fixed_slope_q_score(&projection, candidate, slope).unwrap_or(f64::INFINITY)
            },
        );
        spectral_fixed_slope_q_score(&projection, log_variance, slope).unwrap_or(f64::INFINITY)
    };
    let rho = adaptive_golden_section_minimum(
        bounds.rho_lower,
        bounds.rho_upper,
        nested_tolerance,
        &mut profiled_score,
    );
    let projection = spectral_projection(
        problem.prepared,
        rho,
        problem.realized_rank,
        scale,
        &mut scratch,
    )?;
    let log_variance = adaptive_golden_section_minimum(
        bounds.log_variance_lower,
        bounds.log_variance_upper,
        nested_tolerance,
        |candidate| {
            spectral_fixed_slope_q_score(&projection, candidate, slope).unwrap_or(f64::INFINITY)
        },
    );
    Ok((
        spectral_fixed_slope_q_score(&projection, log_variance, slope)?,
        rho,
        log_variance,
    ))
}

fn profile_comparator(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
) -> Result<ComparatorDiagnostics, TemporalInferenceStatus> {
    if options.profile_max_iterations == 0 || options.profile_max_expansions == 0 {
        return Err(TemporalInferenceStatus::OptimizerNonconverged);
    }
    let bounds = nuisance_bounds(days, observations, options)?;
    let ml_fit = optimize_covariance(
        days,
        observations,
        difference_covariance,
        options,
        bounds,
        false,
        true,
    )?;
    let point = ml_fit.slope * DAYS_PER_YEAR;
    let unrestricted_objective = covariance_objective(
        days,
        observations,
        difference_covariance,
        ml_fit.rho,
        ml_fit.process_variance.ln(),
        options,
        false,
    )?
    .0;
    let prepared = PreparedExactProfile::new(
        days,
        observations,
        difference_covariance,
        options.reference_lag_days,
    )?;
    let levels = [0.68, 0.90, 0.95];
    let interval_results = levels
        .into_par_iter()
        .map(|level| {
            let chi_square = ChiSquared::new(1.0)
                .map_err(|_| TemporalInferenceStatus::WeakParameterIdentification)?
                .inverse_cdf(level);
            let target = unrestricted_objective + chi_square;
            let scale = ml_fit.information_variance.sqrt().max(1e-10);
            let (lower, upper) =
                profile_endpoint_pair(&prepared, ml_fit.slope, scale, target, bounds, options)?;
            Ok(ValidationInterval {
                lower: lower * DAYS_PER_YEAR,
                upper: upper * DAYS_PER_YEAR,
                successful_replicates: 0,
            })
        })
        .collect::<Vec<Result<_, TemporalInferenceStatus>>>();
    let intervals = interval_results
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let z_95 = Normal::new(0.0, 1.0)
        .expect("standard normal parameters are valid")
        .inverse_cdf(0.975);
    let standard_error = intervals[2].upper - intervals[2].lower;
    let standard_error = standard_error / (2.0 * z_95);
    Ok(ComparatorDiagnostics {
        point_estimate: Some(point),
        standard_error_diagnostic: standard_error.is_finite().then_some(standard_error),
        interval_68: Some(intervals[0]),
        interval_90: Some(intervals[1]),
        interval_95: Some(intervals[2]),
        width_68: Some(intervals[0].upper - intervals[0].lower),
        width_90: Some(intervals[1].upper - intervals[1].lower),
        width_95: Some(intervals[2].upper - intervals[2].lower),
        status: TemporalInferenceStatus::Evaluated,
        attempted_replicates: 0,
        successful_replicates: 0,
    })
}

#[allow(clippy::too_many_arguments)]
fn profile_endpoint_pair(
    prepared: &PreparedExactProfile<'_>,
    point: f64,
    initial_scale: f64,
    target: f64,
    bounds: NuisanceBounds,
    options: &TemporalCovarianceOptions,
) -> Result<(f64, f64), TemporalInferenceStatus> {
    let objective = |slope: f64| profile_fixed_slope(prepared, slope, bounds, options);
    let mut lower = point - initial_scale;
    let mut upper = point + initial_scale;
    let mut bounded_values = None;
    for _ in 0..options.profile_max_expansions {
        let (lower_value, upper_value) = rayon::join(|| objective(lower), || objective(upper));
        let values = (lower_value?, upper_value?);
        if values.0 > target && values.1 > target {
            bounded_values = Some(values);
            break;
        }
        let span = (upper - lower) * 2.0;
        lower = point - span;
        upper = point + span;
    }
    let (lower_value, upper_value) = if let Some(values) = bounded_values {
        values
    } else {
        let (lower_value, upper_value) = rayon::join(|| objective(lower), || objective(upper));
        (lower_value?, upper_value?)
    };
    if lower_value <= target || upper_value <= target {
        return Err(TemporalInferenceStatus::OptimizerNonconverged);
    }
    let (lower, upper) = rayon::join(
        || solve_profile_endpoint(&objective, point, lower, target, options),
        || solve_profile_endpoint(&objective, point, upper, target, options),
    );
    let lower = lower?;
    let upper = upper?;
    Ok((lower, upper))
}

fn solve_profile_endpoint<F: Fn(f64) -> Result<f64, TemporalInferenceStatus>>(
    objective: &F,
    point: f64,
    boundary: f64,
    target: f64,
    options: &TemporalCovarianceOptions,
) -> Result<f64, TemporalInferenceStatus> {
    let mut inside = point;
    let mut outside = boundary;
    for _ in 0..options.profile_max_iterations {
        let middle = (inside + outside) / 2.0;
        if objective(middle)? <= target {
            inside = middle;
        } else {
            outside = middle;
        }
        if (outside - inside).abs() <= options.optimizer_tolerance * (1.0 + middle.abs()) {
            return Ok((inside + outside) / 2.0);
        }
    }
    let endpoint = (inside + outside) / 2.0;
    if (outside - inside).abs() > options.optimizer_tolerance * (1.0 + endpoint.abs()) {
        return Err(TemporalInferenceStatus::OptimizerNonconverged);
    }
    Ok(endpoint)
}

fn bootstrap_comparator(summary: &BootstrapSummary, observed_slope: f64) -> ComparatorDiagnostics {
    let se = if summary.successes > 1 {
        Some(summary.variance.sqrt() * DAYS_PER_YEAR)
    } else {
        None
    };
    ComparatorDiagnostics {
        point_estimate: observed_slope
            .is_finite()
            .then_some(observed_slope * DAYS_PER_YEAR),
        standard_error_diagnostic: se,
        interval_68: scale_interval(summary.interval_68),
        interval_90: scale_interval(summary.interval_90),
        interval_95: scale_interval(summary.interval_95),
        width_68: interval_width(summary.interval_68),
        width_90: interval_width(summary.interval_90),
        width_95: interval_width(summary.interval_95),
        status: if summary.attempts == 0 {
            TemporalInferenceStatus::DiagnosticNotComputed
        } else if summary.successes >= summary.minimum_successes {
            TemporalInferenceStatus::Evaluated
        } else {
            TemporalInferenceStatus::BootstrapInsufficientSuccess
        },
        attempted_replicates: summary.attempts,
        successful_replicates: summary.successes,
    }
}

fn scale_interval(interval: Option<ValidationInterval>) -> Option<ValidationInterval> {
    interval.map(|value| ValidationInterval {
        lower: value.lower * DAYS_PER_YEAR,
        upper: value.upper * DAYS_PER_YEAR,
        successful_replicates: value.successful_replicates,
    })
}

fn interval_width(interval: Option<ValidationInterval>) -> Option<f64> {
    interval.map(|value| (value.upper - value.lower) * DAYS_PER_YEAR)
}

pub(super) fn interval(
    point: f64,
    standard_error: f64,
    multiplier: f64,
    attempts: usize,
    successes: usize,
) -> Option<ValidationInterval> {
    (point.is_finite() && standard_error.is_finite() && standard_error >= 0.0).then_some(
        ValidationInterval {
            lower: point - multiplier * standard_error,
            upper: point + multiplier * standard_error,
            successful_replicates: successes.max(attempts),
        },
    )
}

#[allow(clippy::too_many_lines)]
fn profile_plugin(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
) -> Result<PluginFit, TemporalInferenceStatus> {
    #[cfg(test)]
    DENSE_PROFILE_PLUGIN_CALLS.with(|calls| calls.set(calls.get() + 1));
    let bounds = nuisance_bounds(days, observations, options)?;
    optimize_covariance(
        days,
        observations,
        difference_covariance,
        options,
        bounds,
        true,
        true,
    )
}

pub(super) fn nuisance_bounds(
    days: &[f64],
    observations: &[f64],
    options: &TemporalCovarianceOptions,
) -> Result<NuisanceBounds, TemporalInferenceStatus> {
    if !options.rho_min.is_finite()
        || !options.rho_max.is_finite()
        || options.rho_min < 0.0
        || options.rho_min >= options.rho_max
        || options.rho_max >= 1.0
    {
        return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
    }
    let initial = ols_slope(days, observations)?;
    let scale = observations
        .iter()
        .zip(days)
        .map(|(y, x)| (y - initial * x).powi(2))
        .sum::<f64>()
        / observations.len() as f64;
    if !scale.is_finite() || scale <= 1e-12 {
        return Err(TemporalInferenceStatus::WeakParameterIdentification);
    }
    if options.optimizer_max_iterations == 0 || options.optimizer_tolerance <= 0.0 {
        return Err(TemporalInferenceStatus::OptimizerNonconverged);
    }
    let (log_min, log_max) = process_log_bounds(scale, options)?;
    let rho_upper = options.rho_max - 1e-8;
    if rho_upper <= options.rho_min {
        return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
    }
    Ok(NuisanceBounds {
        rho_lower: options.rho_min,
        rho_upper,
        log_variance_lower: log_min,
        log_variance_upper: log_max,
        initial_log_variance: scale.max(1e-12).ln(),
    })
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn optimize_covariance(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
    bounds: NuisanceBounds,
    restricted: bool,
    accept_boundary_solution: bool,
) -> Result<PluginFit, TemporalInferenceStatus> {
    let primary = covariance_coordinate_search(
        days,
        observations,
        difference_covariance,
        options,
        bounds,
        restricted,
    );
    let search = if primary.converged {
        primary
    } else {
        nested_covariance_search(
            days,
            observations,
            difference_covariance,
            options,
            bounds,
            restricted,
        )
    };
    finish_covariance_search(
        days,
        observations,
        difference_covariance,
        options,
        bounds,
        restricted,
        search.candidate,
        search.converged,
        accept_boundary_solution,
    )
}

struct CovarianceCoordinateSearch {
    candidate: Option<(f64, PluginFit)>,
    converged: bool,
}

#[allow(clippy::too_many_arguments)]
fn covariance_coordinate_search(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
    bounds: NuisanceBounds,
    restricted: bool,
) -> CovarianceCoordinateSearch {
    let mut rho;
    let mut log_variance = bounds.initial_log_variance;
    let mut previous_score = f64::INFINITY;
    let mut previous_rho = None;
    let mut previous_log_variance = None;
    let mut last_candidate = None;
    let mut converged = false;
    for _ in 0..options.optimizer_max_iterations {
        rho = golden_section_minimum(bounds.rho_lower, bounds.rho_upper, |candidate| {
            covariance_objective(
                days,
                observations,
                difference_covariance,
                candidate,
                log_variance,
                options,
                restricted,
            )
            .map_or(f64::INFINITY, |(score, _)| score)
        });
        log_variance = golden_section_minimum(
            bounds.log_variance_lower,
            bounds.log_variance_upper,
            |candidate| {
                covariance_objective(
                    days,
                    observations,
                    difference_covariance,
                    rho,
                    candidate,
                    options,
                    restricted,
                )
                .map_or(f64::INFINITY, |(score, _)| score)
            },
        );
        let candidate = covariance_objective(
            days,
            observations,
            difference_covariance,
            rho,
            log_variance,
            options,
            restricted,
        );
        let score = candidate.as_ref().map_or(f64::INFINITY, |value| value.0);
        last_candidate = candidate.ok();
        let parameter_converged = previous_rho.is_some_and(|previous: f64| {
            (previous - rho).abs() <= options.optimizer_tolerance * (1.0 + rho.abs())
        }) && previous_log_variance.is_some_and(|previous: f64| {
            (previous - log_variance).abs()
                <= options.optimizer_tolerance * (1.0 + log_variance.abs())
        });
        if score.is_finite()
            && ((previous_score - score).abs() <= options.optimizer_tolerance * (1.0 + score.abs())
                || parameter_converged)
        {
            converged = true;
            break;
        }
        previous_score = score;
        previous_rho = Some(rho);
        previous_log_variance = Some(log_variance);
    }
    CovarianceCoordinateSearch {
        candidate: last_candidate,
        converged,
    }
}

#[allow(clippy::too_many_arguments)]
fn nested_covariance_search(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
    bounds: NuisanceBounds,
    restricted: bool,
) -> CovarianceCoordinateSearch {
    let nested_tolerance = options.optimizer_tolerance * 0.01;
    let profiled_score = |rho: f64| {
        let log_variance = adaptive_golden_section_minimum(
            bounds.log_variance_lower,
            bounds.log_variance_upper,
            nested_tolerance,
            |candidate| {
                covariance_objective(
                    days,
                    observations,
                    difference_covariance,
                    rho,
                    candidate,
                    options,
                    restricted,
                )
                .map_or(f64::INFINITY, |(score, _)| score)
            },
        );
        covariance_objective(
            days,
            observations,
            difference_covariance,
            rho,
            log_variance,
            options,
            restricted,
        )
        .map_or(f64::INFINITY, |(score, _)| score)
    };
    let rho = adaptive_golden_section_minimum(
        bounds.rho_lower,
        bounds.rho_upper,
        nested_tolerance,
        profiled_score,
    );
    let log_variance = adaptive_golden_section_minimum(
        bounds.log_variance_lower,
        bounds.log_variance_upper,
        nested_tolerance,
        |candidate| {
            covariance_objective(
                days,
                observations,
                difference_covariance,
                rho,
                candidate,
                options,
                restricted,
            )
            .map_or(f64::INFINITY, |(score, _)| score)
        },
    );
    let candidate = covariance_objective(
        days,
        observations,
        difference_covariance,
        rho,
        log_variance,
        options,
        restricted,
    )
    .ok();
    CovarianceCoordinateSearch {
        converged: candidate.is_some(),
        candidate,
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_covariance_search(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
    bounds: NuisanceBounds,
    restricted: bool,
    best: Option<(f64, PluginFit)>,
    any_converged: bool,
    accept_boundary_solution: bool,
) -> Result<PluginFit, TemporalInferenceStatus> {
    if !any_converged {
        return Err(TemporalInferenceStatus::OptimizerNonconverged);
    }
    let (_, candidate) = best.ok_or(TemporalInferenceStatus::CovarianceParameterAtBoundary)?;
    let (base_objective, mut fit) = covariance_objective(
        days,
        observations,
        difference_covariance,
        candidate.rho,
        candidate.process_variance.ln(),
        options,
        restricted,
    )?;
    fit.condition_number = condition_number(&fit.covariance);
    if !fit.condition_number.is_finite() || fit.condition_number > options.condition_limit {
        return Err(TemporalInferenceStatus::DesignIllConditioned);
    }
    if !accept_boundary_solution {
        if let Some(status) = temporal_parameter_boundary_status(
            fit.rho,
            fit.process_variance,
            [bounds.rho_lower, bounds.rho_upper],
            [
                bounds.log_variance_lower.exp(),
                bounds.log_variance_upper.exp(),
            ],
            options.optimizer_tolerance * 0.01,
        ) {
            return Err(status);
        }
    }
    if !accept_boundary_solution {
        validate_profile_curvature(
            days,
            observations,
            difference_covariance,
            &fit,
            base_objective,
            bounds,
            restricted,
            options,
        )?;
    }
    Ok(fit)
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CovarianceStartTrace {
    converged: bool,
    score_bits: Option<u64>,
    rho_bits: Option<u64>,
    process_variance_bits: Option<u64>,
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn optimize_covariance_repeated(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
    bounds: NuisanceBounds,
    restricted: bool,
    repetitions: usize,
) -> (
    Result<PluginFit, TemporalInferenceStatus>,
    Vec<CovarianceStartTrace>,
) {
    let mut best = None;
    let mut any_converged = false;
    let mut trace = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let search = covariance_coordinate_search(
            days,
            observations,
            difference_covariance,
            options,
            bounds,
            restricted,
        );
        let candidate = search.candidate.as_ref();
        trace.push(CovarianceStartTrace {
            converged: search.converged,
            score_bits: candidate.map(|(score, _)| score.to_bits()),
            rho_bits: candidate.map(|(_, fit)| fit.rho.to_bits()),
            process_variance_bits: candidate.map(|(_, fit)| fit.process_variance.to_bits()),
        });
        any_converged |= search.converged;
        if let Some(candidate) = search.candidate {
            if best
                .as_ref()
                .is_none_or(|(score, _): &(f64, PluginFit)| candidate.0 < *score)
            {
                best = Some(candidate);
            }
        }
    }
    (
        finish_covariance_search(
            days,
            observations,
            difference_covariance,
            options,
            bounds,
            restricted,
            best,
            any_converged,
            false,
        ),
        trace,
    )
}

fn covariance_objective(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    rho: f64,
    log_process_variance: f64,
    options: &TemporalCovarianceOptions,
    restricted: bool,
) -> Result<(f64, PluginFit), TemporalInferenceStatus> {
    let process_variance = log_process_variance.exp();
    let covariance = total_difference_covariance(
        difference_covariance,
        days,
        process_variance,
        rho,
        options.reference_lag_days,
    )?;
    let fit = gls_fit(
        days,
        observations,
        &covariance,
        options.condition_limit,
        false,
    )?;
    let reml_correction = if restricted {
        (1.0 / fit.information_variance).ln()
    } else {
        0.0
    };
    let objective = fit.log_determinant + fit.quadratic_form + reml_correction;
    Ok((
        objective,
        PluginFit {
            slope: fit.slope,
            rho,
            process_variance,
            covariance,
            condition_number: fit.condition_number,
            information_variance: fit.information_variance,
        },
    ))
}

fn process_log_bounds(
    scale: f64,
    options: &TemporalCovarianceOptions,
) -> Result<(f64, f64), TemporalInferenceStatus> {
    if !options.process_variance_min_ratio.is_finite()
        || !options.process_variance_max_ratio.is_finite()
        || options.process_variance_min_ratio <= 0.0
        || options.process_variance_max_ratio <= options.process_variance_min_ratio
    {
        return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
    }
    Ok((
        (scale * options.process_variance_min_ratio).max(1e-12).ln(),
        (scale * options.process_variance_max_ratio).max(1e-12).ln(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn validate_profile_curvature(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    fit: &PluginFit,
    base: f64,
    bounds: NuisanceBounds,
    restricted: bool,
    options: &TemporalCovarianceOptions,
) -> Result<(), TemporalInferenceStatus> {
    if !options.minimum_profile_curvature.is_finite()
        || options.minimum_profile_curvature <= 0.0
        || !fit.information_variance.is_finite()
        || fit.information_variance <= 0.0
    {
        return Err(TemporalInferenceStatus::WeakParameterIdentification);
    }
    let rho_step = 1e-3_f64
        .min((fit.rho - bounds.rho_lower) / 2.0)
        .min((bounds.rho_upper - fit.rho) / 2.0);
    let log_step = 1e-2;
    if rho_step <= 0.0 {
        return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
    }
    let rho_curvature = (covariance_objective(
        days,
        observations,
        difference_covariance,
        fit.rho + rho_step,
        fit.process_variance.ln(),
        options,
        restricted,
    )?
    .0 + covariance_objective(
        days,
        observations,
        difference_covariance,
        fit.rho - rho_step,
        fit.process_variance.ln(),
        options,
        restricted,
    )?
    .0 - 2.0 * base)
        / rho_step.powi(2);
    let variance_curvature = (covariance_objective(
        days,
        observations,
        difference_covariance,
        fit.rho,
        fit.process_variance.ln() + log_step,
        options,
        restricted,
    )?
    .0 + covariance_objective(
        days,
        observations,
        difference_covariance,
        fit.rho,
        fit.process_variance.ln() - log_step,
        options,
        restricted,
    )?
    .0 - 2.0 * base)
        / log_step.powi(2);
    if !rho_curvature.is_finite()
        || !variance_curvature.is_finite()
        || rho_curvature <= options.minimum_profile_curvature
        || variance_curvature <= options.minimum_profile_curvature
    {
        return Err(TemporalInferenceStatus::WeakParameterIdentification);
    }
    Ok(())
}

fn golden_section_minimum<F>(mut lower: f64, mut upper: f64, mut objective: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    let lower_boundary = lower;
    let upper_boundary = upper;
    let lower_boundary_value = objective(lower_boundary);
    let upper_boundary_value = objective(upper_boundary);
    let ratio = 0.618_033_988_749_894_9;
    let mut left = upper - ratio * (upper - lower);
    let mut right = lower + ratio * (upper - lower);
    let mut left_value = objective(left);
    let mut right_value = objective(right);
    for _ in 0..16 {
        if left_value < right_value {
            upper = right;
            right = left;
            right_value = left_value;
            left = upper - ratio * (upper - lower);
            left_value = objective(left);
        } else {
            lower = left;
            left = right;
            left_value = right_value;
            right = lower + ratio * (upper - lower);
            right_value = objective(right);
        }
    }
    let interior = (lower + upper) / 2.0;
    let interior_value = objective(interior);
    [
        (lower_boundary, lower_boundary_value),
        (interior, interior_value),
        (upper_boundary, upper_boundary_value),
    ]
    .into_iter()
    .min_by(|left, right| left.1.total_cmp(&right.1))
    .map_or(interior, |(parameter, _)| parameter)
}

fn adaptive_golden_section_minimum<F>(
    mut lower: f64,
    mut upper: f64,
    tolerance: f64,
    mut objective: F,
) -> f64
where
    F: FnMut(f64) -> f64,
{
    let lower_boundary = lower;
    let upper_boundary = upper;
    let lower_boundary_value = objective(lower_boundary);
    let upper_boundary_value = objective(upper_boundary);
    let ratio = 0.618_033_988_749_894_9;
    let mut left = upper - ratio * (upper - lower);
    let mut right = lower + ratio * (upper - lower);
    let mut left_value = objective(left);
    let mut right_value = objective(right);
    for _ in 0..32 {
        let middle = (lower + upper) / 2.0;
        if upper - lower <= tolerance * 0.25 * (1.0 + middle.abs()) {
            break;
        }
        if left_value < right_value {
            upper = right;
            right = left;
            right_value = left_value;
            left = upper - ratio * (upper - lower);
            left_value = objective(left);
        } else {
            lower = left;
            left = right;
            left_value = right_value;
            right = lower + ratio * (upper - lower);
            right_value = objective(right);
        }
    }
    let interior = (lower + upper) / 2.0;
    let interior_value = objective(interior);
    [
        (lower_boundary, lower_boundary_value),
        (interior, interior_value),
        (upper_boundary, upper_boundary_value),
    ]
    .into_iter()
    .min_by(|left, right| left.1.total_cmp(&right.1))
    .map_or(interior, |(parameter, _)| parameter)
}

fn profile_fixed_slope(
    prepared: &PreparedExactProfile<'_>,
    slope: f64,
    bounds: NuisanceBounds,
    options: &TemporalCovarianceOptions,
) -> Result<f64, TemporalInferenceStatus> {
    if options.optimizer_max_iterations == 0 || options.optimizer_tolerance <= 0.0 {
        return Err(TemporalInferenceStatus::OptimizerNonconverged);
    }
    let nested_tolerance = options.optimizer_tolerance * 0.01;
    let mut workspace = ExactFixedSlopeWorkspace::new(prepared, slope);
    let mut profiled_score = |rho: f64| {
        if workspace.prepare_rho(rho).is_err() {
            return f64::INFINITY;
        }
        let log_variance = adaptive_golden_section_minimum(
            bounds.log_variance_lower,
            bounds.log_variance_upper,
            nested_tolerance,
            |candidate| workspace.score(candidate).unwrap_or(f64::INFINITY),
        );
        workspace.score(log_variance).unwrap_or(f64::INFINITY)
    };
    let rho = adaptive_golden_section_minimum(
        bounds.rho_lower,
        bounds.rho_upper,
        nested_tolerance,
        &mut profiled_score,
    );
    workspace.prepare_rho(rho)?;
    let log_variance = adaptive_golden_section_minimum(
        bounds.log_variance_lower,
        bounds.log_variance_upper,
        nested_tolerance,
        |candidate| workspace.score(candidate).unwrap_or(f64::INFINITY),
    );
    workspace.score(log_variance)
}

#[cfg(test)]
fn profile_fixed_objective(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    slope: f64,
    rho: f64,
    log_process_variance: f64,
    options: &TemporalCovarianceOptions,
) -> Result<(f64, PluginFit), TemporalInferenceStatus> {
    let process_variance = log_process_variance.exp();
    let covariance = total_difference_covariance(
        difference_covariance,
        days,
        process_variance,
        rho,
        options.reference_lag_days,
    )?;
    let lower =
        cholesky(&covariance).ok_or(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite)?;
    let residuals = observations
        .iter()
        .zip(days)
        .map(|(value, day)| value - slope * day)
        .collect::<Vec<_>>();
    let solved_residuals = solve_cholesky(&lower, &residuals);
    let solved_days = solve_cholesky(&lower, days);
    let information = dot(days, &solved_days);
    if !information.is_finite() || information <= 0.0 {
        return Err(TemporalInferenceStatus::DesignRankDeficient);
    }
    let log_determinant = 2.0
        * lower
            .iter()
            .enumerate()
            .map(|(index, row)| row[index].ln())
            .sum::<f64>();
    let objective = log_determinant + dot(&residuals, &solved_residuals);
    Ok((
        objective,
        PluginFit {
            slope,
            rho,
            process_variance,
            covariance,
            condition_number: f64::NAN,
            information_variance: 1.0 / information,
        },
    ))
}

struct GlsFit {
    slope: f64,
    quadratic_form: f64,
    log_determinant: f64,
    information_variance: f64,
    condition_number: f64,
    inverse: Vec<Vec<f64>>,
}

fn gls_fit(
    days: &[f64],
    observations: &[f64],
    covariance: &[Vec<f64>],
    condition_limit: f64,
    compute_condition_number: bool,
) -> Result<GlsFit, TemporalInferenceStatus> {
    let lower =
        cholesky(covariance).ok_or(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite)?;
    validate_square_covariance(covariance)?;
    let condition_number = if compute_condition_number {
        let condition = condition_number(covariance);
        if !condition.is_finite() || condition > condition_limit {
            return Err(TemporalInferenceStatus::DesignIllConditioned);
        }
        condition
    } else {
        f64::NAN
    };
    let inverse = invert_from_cholesky(&lower);
    let transformed_x = mat_vec(&inverse, days);
    let transformed_y = mat_vec(&inverse, observations);
    let information = dot(days, &transformed_x);
    if !information.is_finite() || information <= 0.0 {
        return Err(TemporalInferenceStatus::DesignRankDeficient);
    }
    let slope = dot(days, &transformed_y) / information;
    let residuals: Vec<f64> = observations
        .iter()
        .zip(days)
        .map(|(y, x)| y - slope * x)
        .collect();
    let quadratic_form = dot(&residuals, &mat_vec(&inverse, &residuals));
    let log_determinant = 2.0
        * lower
            .iter()
            .enumerate()
            .map(|(index, row)| row[index].ln())
            .sum::<f64>();
    Ok(GlsFit {
        slope,
        quadratic_form,
        log_determinant,
        information_variance: 1.0 / information,
        condition_number,
        inverse,
    })
}

fn bootstrap_refit(
    days: &[f64],
    difference_covariance: &[Vec<f64>],
    plugin: &PluginFit,
    options: &TemporalCovarianceOptions,
) -> BootstrapSummary {
    let minimum_successes = required_bootstrap_successes(options.bootstrap_replicates)
        .max(options.bootstrap_minimum_successes);
    if options.bootstrap_replicates == 0 {
        return BootstrapSummary {
            interval_68: None,
            interval_90: None,
            interval_95: None,
            attempts: 0,
            successes: 0,
            variance: f64::NAN,
            minimum_successes,
        };
    }
    let Some(cholesky) = cholesky(&plugin.covariance) else {
        return BootstrapSummary {
            interval_68: None,
            interval_90: None,
            interval_95: None,
            attempts: options.bootstrap_replicates,
            successes: 0,
            variance: f64::NAN,
            minimum_successes,
        };
    };
    let replicate_slopes = (0..options.bootstrap_replicates)
        .into_par_iter()
        .map(|replicate| {
            let mut state = splitmix64(options.bootstrap_seed ^ replicate as u64);
            let normal = (0..days.len())
                .map(|_| standard_normal(&mut state))
                .collect::<Vec<_>>();
            let residual = lower_mat_vec(&cholesky, &normal);
            let simulated = days
                .iter()
                .zip(residual)
                .map(|(day, noise)| plugin.slope * day + noise)
                .collect::<Vec<_>>();
            let bootstrap_options = TemporalCovarianceOptions {
                bootstrap_replicates: 0,
                bootstrap_minimum_successes: 0,
                bootstrap_seed: state,
                ..options.clone()
            };
            let bounds = nuisance_bounds(days, &simulated, &bootstrap_options)?;
            optimize_covariance(
                days,
                &simulated,
                difference_covariance,
                &bootstrap_options,
                bounds,
                true,
                true,
            )
            .map(|refit| refit.slope)
        })
        .collect::<Vec<_>>();
    let mut slopes = replicate_slopes
        .into_iter()
        .filter_map(Result::ok)
        .collect::<Vec<_>>();
    slopes.sort_by(f64::total_cmp);
    let successes = slopes.len();
    let mean = if successes == 0 {
        f64::NAN
    } else {
        slopes.iter().sum::<f64>() / successes as f64
    };
    let quantile = |fraction: f64| {
        let position = fraction * (successes.saturating_sub(1)) as f64;
        slopes[position.round() as usize]
    };
    let variance = if successes > 1 {
        slopes
            .iter()
            .map(|slope| (slope - mean).powi(2))
            .sum::<f64>()
            / (successes - 1) as f64
    } else {
        f64::NAN
    };
    let validation_interval = |fraction: f64| {
        (successes > 0).then_some(ValidationInterval {
            lower: quantile((1.0 - fraction) / 2.0),
            upper: quantile(1.0 - (1.0 - fraction) / 2.0),
            successful_replicates: successes,
        })
    };
    BootstrapSummary {
        interval_68: validation_interval(0.68),
        interval_90: validation_interval(0.90),
        interval_95: validation_interval(0.95),
        attempts: options.bootstrap_replicates,
        successes,
        variance,
        minimum_successes,
    }
}

fn validate_dates(days: &[f64]) -> Result<(), TemporalInferenceStatus> {
    if days.len() < 2
        || days
            .windows(2)
            .any(|pair| !pair[0].is_finite() || !pair[1].is_finite() || pair[1] <= pair[0])
    {
        return Err(TemporalInferenceStatus::DatesNotStrictlyIncreasing);
    }
    Ok(())
}

fn validate_supported_cadence(
    days: &[f64],
    options: &TemporalCovarianceOptions,
) -> Result<(), TemporalInferenceStatus> {
    if days.len() < 2
        || !options.minimum_gap_days.is_finite()
        || !options.maximum_gap_days.is_finite()
        || options.minimum_gap_days <= 0.0
        || options.maximum_gap_days < options.minimum_gap_days
    {
        return Err(TemporalInferenceStatus::UnsupportedCadence);
    }
    let gaps: Vec<f64> = days.windows(2).map(|pair| pair[1] - pair[0]).collect();
    if days[0] < options.minimum_gap_days
        || days[0] > options.maximum_gap_days
        || gaps.iter().any(|gap| {
            !gap.is_finite() || *gap < options.minimum_gap_days || *gap > options.maximum_gap_days
        })
    {
        return Err(TemporalInferenceStatus::UnsupportedCadence);
    }
    Ok(())
}

fn validate_square_covariance(matrix: &[Vec<f64>]) -> Result<(), TemporalInferenceStatus> {
    if matrix.is_empty() || matrix.iter().any(|row| row.len() != matrix.len()) {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    if (0..matrix.len()).any(|row| {
        (0..matrix.len())
            .any(|column| (matrix[row][column] - matrix[column][row]).abs() > SYMMETRY_TOLERANCE)
    }) {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    Ok(())
}

fn invert_from_cholesky(lower: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = lower.len();
    let mut inverse = vec![vec![0.0; n]; n];
    for column in 0..n {
        let mut unit = vec![0.0; n];
        unit[column] = 1.0;
        let solution = solve_cholesky(lower, &unit);
        for row in 0..n {
            inverse[row][column] = solution[row];
        }
    }
    inverse
}

pub(super) fn cholesky(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let mut lower = vec![vec![0.0; matrix.len()]; matrix.len()];
    cholesky_into(matrix, &mut lower).then_some(lower)
}

fn cholesky_into(matrix: &[Vec<f64>], lower: &mut [Vec<f64>]) -> bool {
    #[cfg(test)]
    CHOLESKY_CALLS.with(|calls| calls.set(calls.get() + 1));
    let n = matrix.len();
    for row in 0..n {
        for column in 0..=row {
            let sum = (0..column)
                .map(|index| lower[row][index] * lower[column][index])
                .sum::<f64>();
            if row == column {
                let diagonal = matrix[row][row] - sum;
                if !diagonal.is_finite() || diagonal <= 0.0 {
                    return false;
                }
                lower[row][column] = diagonal.sqrt();
            } else if lower[column][column] > 0.0 {
                lower[row][column] = (matrix[row][column] - sum) / lower[column][column];
            } else {
                return false;
            }
        }
    }
    true
}

fn solve_cholesky(lower: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    let n = rhs.len();
    let mut forward = vec![0.0; n];
    let mut solution = vec![0.0; n];
    solve_cholesky_into(lower, rhs, &mut forward, &mut solution);
    solution
}

fn solve_cholesky_into(lower: &[Vec<f64>], rhs: &[f64], forward: &mut [f64], solution: &mut [f64]) {
    let n = rhs.len();
    for row in 0..n {
        forward[row] = (rhs[row]
            - (0..row)
                .map(|column| lower[row][column] * forward[column])
                .sum::<f64>())
            / lower[row][row];
    }
    for row in (0..n).rev() {
        solution[row] = (forward[row]
            - ((row + 1)..n)
                .map(|column| lower[column][row] * solution[column])
                .sum::<f64>())
            / lower[row][row];
    }
}

fn condition_number(matrix: &[Vec<f64>]) -> f64 {
    let dimension = matrix.len();
    let mut symmetric = Mat::<f64>::zeros(dimension, dimension);
    for (row, values) in matrix.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            symmetric.write(row, column, *value);
        }
    }
    let eigenvalues = symmetric.selfadjoint_eigenvalues(Side::Lower);
    let largest = eigenvalues.iter().copied().fold(0.0, f64::max);
    let smallest = eigenvalues
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    largest / smallest
}

fn mat_vec(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect()
}

pub(super) fn lower_mat_vec(lower: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    lower
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn ols_slope(days: &[f64], observations: &[f64]) -> Result<f64, TemporalInferenceStatus> {
    let denominator = dot(days, days);
    if denominator <= 0.0 || !denominator.is_finite() {
        return Err(TemporalInferenceStatus::DesignRankDeficient);
    }
    Ok(dot(days, observations) / denominator)
}

fn legacy_intercept_wls(
    days: &[f64],
    observations: &[f64],
    variances: &[f64],
) -> Result<f64, TemporalInferenceStatus> {
    if days.len() != observations.len()
        || days.len() != variances.len()
        || variances
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let (sw, swx, swxx, swy, swxy) = days.iter().zip(observations).zip(variances).fold(
        (0.0, 0.0, 0.0, 0.0, 0.0),
        |(sw, swx, swxx, swy, swxy), ((day, value), variance)| {
            let weight = 1.0 / variance;
            (
                sw + weight,
                swx + weight * day,
                swxx + weight * day * day,
                swy + weight * value,
                swxy + weight * day * value,
            )
        },
    );
    let determinant = sw * swxx - swx * swx;
    if !determinant.is_finite() || determinant <= 0.0 {
        return Err(TemporalInferenceStatus::DesignRankDeficient);
    }
    Ok((sw * swxy - swx * swy) / determinant)
}

pub(super) fn standard_normal(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let u1 = ((*state >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let u2 = (*state >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

pub(super) fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_invert_positive_definite(
        matrix: &[Vec<f64>],
        condition_limit: f64,
        check_condition: bool,
    ) -> Result<Vec<Vec<f64>>, TemporalInferenceStatus> {
        validate_square_covariance(matrix)?;
        let Some(lower) = cholesky(matrix) else {
            return Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite);
        };
        if check_condition {
            let condition = condition_number(matrix);
            if !condition.is_finite() || condition > condition_limit {
                return Err(TemporalInferenceStatus::DesignIllConditioned);
            }
        }
        let n = matrix.len();
        let mut inverse = vec![vec![0.0; n]; n];
        for column in 0..n {
            let mut unit = vec![0.0; n];
            unit[column] = 1.0;
            let solution = solve_cholesky(&lower, &unit);
            for row in 0..n {
                inverse[row][column] = solution[row];
            }
        }
        Ok(inverse)
    }

    fn legacy_gls_fit(
        days: &[f64],
        observations: &[f64],
        covariance: &[Vec<f64>],
        condition_limit: f64,
        compute_condition_number: bool,
    ) -> Result<GlsFit, TemporalInferenceStatus> {
        let lower = cholesky(covariance)
            .ok_or(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite)?;
        let inverse =
            legacy_invert_positive_definite(covariance, condition_limit, compute_condition_number)?;
        let transformed_x = mat_vec(&inverse, days);
        let transformed_y = mat_vec(&inverse, observations);
        let information = dot(days, &transformed_x);
        if !information.is_finite() || information <= 0.0 {
            return Err(TemporalInferenceStatus::DesignRankDeficient);
        }
        let slope = dot(days, &transformed_y) / information;
        let residuals: Vec<f64> = observations
            .iter()
            .zip(days)
            .map(|(y, x)| y - slope * x)
            .collect();
        let quadratic_form = dot(&residuals, &mat_vec(&inverse, &residuals));
        let log_determinant = 2.0
            * lower
                .iter()
                .enumerate()
                .map(|(index, row)| row[index].ln())
                .sum::<f64>();
        Ok(GlsFit {
            slope,
            quadratic_form,
            log_determinant,
            information_variance: 1.0 / information,
            condition_number: if compute_condition_number {
                condition_number(covariance)
            } else {
                f64::NAN
            },
            inverse,
        })
    }

    fn legacy_profile_fixed_objective(
        days: &[f64],
        observations: &[f64],
        difference_covariance: &[Vec<f64>],
        slope: f64,
        rho: f64,
        log_process_variance: f64,
        options: &TemporalCovarianceOptions,
    ) -> Result<(f64, PluginFit), TemporalInferenceStatus> {
        let process_variance = log_process_variance.exp();
        let covariance = total_difference_covariance(
            difference_covariance,
            days,
            process_variance,
            rho,
            options.reference_lag_days,
        )?;
        let fit = legacy_gls_fit(
            days,
            observations,
            &covariance,
            options.condition_limit,
            false,
        )?;
        let residuals: Vec<f64> = observations
            .iter()
            .zip(days)
            .map(|(value, day)| value - slope * day)
            .collect();
        let inverse = legacy_invert_positive_definite(&covariance, options.condition_limit, false)?;
        let objective = fit.log_determinant + dot(&residuals, &mat_vec(&inverse, &residuals));
        Ok((
            objective,
            PluginFit {
                slope,
                rho,
                process_variance,
                covariance,
                condition_number: fit.condition_number,
                information_variance: fit.information_variance,
            },
        ))
    }

    fn assert_f64_bits(left: f64, right: f64) {
        assert_eq!(left.to_bits(), right.to_bits());
    }

    fn assert_matrix_bits(left: &[Vec<f64>], right: &[Vec<f64>]) {
        assert_eq!(left.len(), right.len());
        for (left_row, right_row) in left.iter().zip(right) {
            assert_eq!(left_row.len(), right_row.len());
            for (left_value, right_value) in left_row.iter().zip(right_row) {
                assert_f64_bits(*left_value, *right_value);
            }
        }
    }

    fn assert_gls_bits(left: &GlsFit, right: &GlsFit) {
        assert_f64_bits(left.slope, right.slope);
        assert_f64_bits(left.quadratic_form, right.quadratic_form);
        assert_f64_bits(left.log_determinant, right.log_determinant);
        assert_f64_bits(left.information_variance, right.information_variance);
        assert_f64_bits(left.condition_number, right.condition_number);
        assert_matrix_bits(&left.inverse, &right.inverse);
    }

    fn assert_plugin_bits(left: &PluginFit, right: &PluginFit) {
        assert_f64_bits(left.slope, right.slope);
        assert_f64_bits(left.rho, right.rho);
        assert_f64_bits(left.process_variance, right.process_variance);
        assert_matrix_bits(&left.covariance, &right.covariance);
        assert_f64_bits(left.condition_number, right.condition_number);
        assert_f64_bits(left.information_variance, right.information_variance);
    }

    #[derive(Debug)]
    struct DenseFactorObjective {
        score: f64,
        slope: f64,
        x_v_x: f64,
        x_v_y: f64,
        y_v_y: f64,
        log_determinant: f64,
        quadratic: f64,
    }

    #[allow(clippy::too_many_arguments)]
    fn dense_factor_objective(
        days: &[f64],
        observations: &[f64],
        factor: &[f64],
        maximum_rank: usize,
        realized_rank: usize,
        rho: f64,
        log_process_variance: f64,
        options: &TemporalCovarianceOptions,
    ) -> Result<DenseFactorObjective, TemporalInferenceStatus> {
        let covariance = (0..days.len())
            .map(|left| {
                (0..days.len())
                    .map(|right| {
                        (0..realized_rank)
                            .map(|component| {
                                factor[left * maximum_rank + component]
                                    * factor[right * maximum_rank + component]
                            })
                            .sum()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let total = total_difference_covariance(
            &covariance,
            days,
            log_process_variance.exp(),
            rho,
            options.reference_lag_days,
        )?;
        let fit = gls_fit(days, observations, &total, options.condition_limit, false)?;
        let inverse_x = mat_vec(&fit.inverse, days);
        let inverse_y = mat_vec(&fit.inverse, observations);
        let x_v_x = dot(days, &inverse_x);
        let x_v_y = dot(days, &inverse_y);
        let y_v_y = dot(observations, &inverse_y);
        Ok(DenseFactorObjective {
            score: fit.log_determinant + fit.quadratic_form + x_v_x.ln(),
            slope: fit.slope,
            x_v_x,
            x_v_y,
            y_v_y,
            log_determinant: fit.log_determinant,
            quadratic: fit.quadratic_form,
        })
    }

    fn release_factor_fixture(
        date_count: usize,
        maximum_rank: usize,
        realized_rank: usize,
        near_rank: bool,
    ) -> Vec<f64> {
        let mut factor = vec![0.0; date_count * maximum_rank];
        for row in 0..date_count {
            for component in 0..realized_rank {
                let diagonal = usize::from(component == row % realized_rank);
                let scale = if near_rank && component > 0 {
                    1e-5
                } else {
                    1.0
                };
                factor[row * maximum_rank + component] = scale
                    * (diagonal as f64 * (0.7 + row as f64 * 0.002)
                        + 0.03 * ((row + 1) as f64 * (component + 2) as f64 * 0.17).sin());
            }
        }
        factor
    }

    fn assert_relative(left: f64, right: f64, label: &str) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() <= 1e-10 * scale,
            "{label}: left={left:.17e}, right={right:.17e}, relative={:.3e}",
            (left - right).abs() / scale
        );
    }

    fn streamed_augmented_basis(
        prepared: &PreparedFactorObjective,
        observations: &[f64],
        factor: &[f64],
        maximum_rank: usize,
        realized_rank: usize,
        rho: f64,
    ) -> Vec<f64> {
        let date_count = prepared.design.len();
        let dimension = realized_rank + 2;
        let diagonal = (0..date_count)
            .map(|date| {
                (0..realized_rank)
                    .map(|component| factor[date * maximum_rank + component].powi(2))
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let geometric_mean =
            (diagonal.iter().map(|value| value.ln()).sum::<f64>() / date_count as f64).exp();
        let mut rows = vec![0.0; date_count * dimension];
        for date in 0..date_count {
            let inverse_shape = (diagonal[date] / geometric_mean).sqrt().recip();
            for component in 0..realized_rank {
                rows[date * dimension + component] =
                    factor[date * maximum_rank + component] * inverse_shape;
            }
            rows[date * dimension + realized_rank] = prepared.design[date] * inverse_shape;
            rows[date * dimension + realized_rank + 1] = observations[date] * inverse_shape;
        }
        let mut whitened = rows.clone();
        for date in 1..date_count {
            let phi = if rho == 0.0 {
                0.0
            } else {
                (rho.ln() * prepared.gap_exponents[date - 1]).exp()
            };
            let innovation = 1.0 - phi * phi;
            for variable in 0..dimension {
                whitened[date * dimension + variable] = (rows[date * dimension + variable]
                    - phi * rows[(date - 1) * dimension + variable])
                    / innovation.sqrt();
            }
        }
        let mut gram = vec![0.0; dimension * dimension];
        for left in 0..dimension {
            for right in 0..dimension {
                gram[left * dimension + right] = (0..date_count)
                    .map(|date| {
                        whitened[date * dimension + left] * whitened[date * dimension + right]
                    })
                    .sum();
            }
        }
        gram
    }

    fn five_point_first(values: [f64; 4], step: f64) -> f64 {
        (values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (12.0 * step)
    }

    fn five_point_second(values: [f64; 5], step: f64) -> f64 {
        (-values[0] + 16.0 * values[1] - 30.0 * values[2] + 16.0 * values[3] - values[4])
            / (12.0 * step.powi(2))
    }

    fn profiled_log_variance_score(
        prepared: &PreparedFactorObjective,
        observations: &[f64],
        factor: &[f64],
        maximum_rank: usize,
        realized_rank: usize,
        rho: f64,
        bounds: NuisanceBounds,
    ) -> (f64, f64) {
        let mut lower = bounds.log_variance_lower;
        let mut upper = bounds.log_variance_upper;
        let golden = 0.381_966_011_250_105_1;
        let mut left = lower + golden * (upper - lower);
        let mut right = upper - golden * (upper - lower);
        let score = |log_variance: f64| {
            let mut scratch = FactorObjectiveScratch::new(prepared.design.len(), realized_rank);
            factor_native_objective(
                prepared,
                observations,
                factor,
                maximum_rank,
                realized_rank,
                rho,
                log_variance,
                true,
                &mut scratch,
            )
            .unwrap()
            .score
        };
        let mut left_score = score(left);
        let mut right_score = score(right);
        for _ in 0..80 {
            if left_score <= right_score {
                upper = right;
                right = left;
                right_score = left_score;
                left = lower + golden * (upper - lower);
                left_score = score(left);
            } else {
                lower = left;
                left = right;
                left_score = right_score;
                right = upper - golden * (upper - lower);
                right_score = score(right);
            }
        }
        if left_score <= right_score {
            (left, left_score)
        } else {
            (right, right_score)
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    fn augmented_basis_reml_rho_derivatives_match_frozen_objective() {
        let options = TemporalCovarianceOptions::default();
        for &(date_count, maximum_rank, realized_rank, irregular) in
            &[(12, 12, 12, false), (48, 7, 7, true)]
        {
            let mut elapsed = 0.0;
            let days = (0..date_count)
                .map(|date| {
                    elapsed += if irregular {
                        [4.0, 7.0, 12.0, 19.0, 25.0, 31.0][date % 6]
                    } else {
                        12.0
                    };
                    elapsed
                })
                .collect::<Vec<_>>();
            let prepared = PreparedFactorObjective::new(&days, 12.0).unwrap();
            let factor = release_factor_fixture(date_count, maximum_rank, realized_rank, false);
            let observations = days
                .iter()
                .enumerate()
                .map(|(date, day)| {
                    0.012 * day
                        + (date as f64 * 0.43).sin() * 0.7
                        + (date as f64 * 0.19).cos() * 0.2
                })
                .collect::<Vec<_>>();
            let bounds = nuisance_bounds(&prepared.design, &observations, &options).unwrap();
            for &rho in &[0.05_f64, 0.3, 0.95] {
                for &log_variance in &[0.2_f64.ln(), 1.3_f64.ln()] {
                    let derivatives =
                        crate::temporal_covariance_batch::augmented_basis_reml_rho_derivatives(
                            &prepared,
                            &observations,
                            &factor,
                            maximum_rank,
                            realized_rank,
                            rho,
                            log_variance,
                        )
                        .unwrap();
                    let streamed = streamed_augmented_basis(
                        &prepared,
                        &observations,
                        &factor,
                        maximum_rank,
                        realized_rank,
                        rho,
                    );
                    for (actual, expected) in derivatives.augmented_basis.iter().zip(&streamed) {
                        assert_relative(*actual, *expected, "augmented basis/streamed");
                    }
                    let eta = rho.ln();
                    let eta_step = 1e-4;
                    let basis_at = |eta_value: f64| {
                        streamed_augmented_basis(
                            &prepared,
                            &observations,
                            &factor,
                            maximum_rank,
                            realized_rank,
                            eta_value.exp(),
                        )
                    };
                    let basis_m2 = basis_at(eta - 2.0 * eta_step);
                    let basis_m1 = basis_at(eta - eta_step);
                    let basis_p1 = basis_at(eta + eta_step);
                    let basis_p2 = basis_at(eta + 2.0 * eta_step);
                    for entry in 0..streamed.len() {
                        let first = five_point_first(
                            [
                                basis_m2[entry],
                                basis_m1[entry],
                                basis_p1[entry],
                                basis_p2[entry],
                            ],
                            eta_step,
                        );
                        let second = five_point_second(
                            [
                                basis_m2[entry],
                                basis_m1[entry],
                                streamed[entry],
                                basis_p1[entry],
                                basis_p2[entry],
                            ],
                            eta_step,
                        );
                        let scale = first.abs().max(1.0);
                        assert!(
                            (derivatives.augmented_basis_eta[entry] - first).abs() <= 2e-7 * scale
                        );
                        let scale = second.abs().max(1.0);
                        assert!(
                            (derivatives.augmented_basis_eta_eta[entry] - second).abs()
                                <= 2e-4 * scale
                        );
                    }
                    let objective_at = |eta_value: f64, q_value: f64| {
                        let mut scratch = FactorObjectiveScratch::new(date_count, realized_rank);
                        factor_native_objective(
                            &prepared,
                            &observations,
                            &factor,
                            maximum_rank,
                            realized_rank,
                            eta_value.exp(),
                            q_value,
                            true,
                            &mut scratch,
                        )
                        .unwrap()
                        .score
                    };
                    assert_relative(
                        derivatives.evaluation.score,
                        objective_at(eta, log_variance),
                        "derivative/frozen objective",
                    );
                    let q_step = 1e-4;
                    let eta_scores = [
                        objective_at(eta - 2.0 * eta_step, log_variance),
                        objective_at(eta - eta_step, log_variance),
                        objective_at(eta + eta_step, log_variance),
                        objective_at(eta + 2.0 * eta_step, log_variance),
                    ];
                    let q_scores = [
                        objective_at(eta, log_variance - 2.0 * q_step),
                        objective_at(eta, log_variance - q_step),
                        objective_at(eta, log_variance + q_step),
                        objective_at(eta, log_variance + 2.0 * q_step),
                    ];
                    let center = derivatives.evaluation.score;
                    let eta_second = five_point_second(
                        [
                            eta_scores[0],
                            eta_scores[1],
                            center,
                            eta_scores[2],
                            eta_scores[3],
                        ],
                        eta_step,
                    );
                    let q_second = five_point_second(
                        [q_scores[0], q_scores[1], center, q_scores[2], q_scores[3]],
                        q_step,
                    );
                    let cross = (objective_at(eta + eta_step, log_variance + q_step)
                        - objective_at(eta + eta_step, log_variance - q_step)
                        - objective_at(eta - eta_step, log_variance + q_step)
                        + objective_at(eta - eta_step, log_variance - q_step))
                        / (4.0 * eta_step * q_step);
                    for (actual, expected, tolerance) in [
                        (
                            derivatives.score_eta,
                            five_point_first(eta_scores, eta_step),
                            2e-6,
                        ),
                        (
                            derivatives.score_log_q,
                            five_point_first(q_scores, q_step),
                            2e-6,
                        ),
                        (derivatives.score_eta_eta, eta_second, 2e-3),
                        (derivatives.score_eta_log_q, cross, 2e-3),
                        (derivatives.score_log_q_log_q, q_second, 2e-3),
                    ] {
                        assert!((actual - expected).abs() <= tolerance * expected.abs().max(1.0));
                    }
                }
                let (profiled_q, profiled_score) = profiled_log_variance_score(
                    &prepared,
                    &observations,
                    &factor,
                    maximum_rank,
                    realized_rank,
                    rho,
                    bounds,
                );
                let profiled =
                    crate::temporal_covariance_batch::augmented_basis_reml_rho_derivatives(
                        &prepared,
                        &observations,
                        &factor,
                        maximum_rank,
                        realized_rank,
                        rho,
                        profiled_q,
                    )
                    .unwrap();
                let eta = rho.ln();
                let step = 5e-4;
                let left = profiled_log_variance_score(
                    &prepared,
                    &observations,
                    &factor,
                    maximum_rank,
                    realized_rank,
                    (eta - step).exp(),
                    bounds,
                )
                .1;
                let right = profiled_log_variance_score(
                    &prepared,
                    &observations,
                    &factor,
                    maximum_rank,
                    realized_rank,
                    (eta + step).exp(),
                    bounds,
                )
                .1;
                let independent_curvature = (left - 2.0 * profiled_score + right) / step.powi(2);
                assert!(
                    (profiled.profiled_eta_curvature - independent_curvature).abs()
                        <= 3e-3 * independent_curvature.abs().max(1.0),
                    "profile curvature n={date_count} r={realized_rank} rho={rho}: analytic={}, independent={independent_curvature}, q={profiled_q}, Lq={}, Lqq={}, Leq={}",
                    profiled.profiled_eta_curvature,
                    profiled.score_log_q,
                    profiled.score_log_q_log_q,
                    profiled.score_eta_log_q,
                );
            }
            assert_eq!(
                crate::temporal_covariance_batch::augmented_basis_reml_rho_derivatives(
                    &prepared,
                    &observations,
                    &factor,
                    maximum_rank,
                    realized_rank,
                    1.0,
                    0.0,
                )
                .unwrap_err(),
                TemporalInferenceStatus::CovarianceParameterAtBoundary
            );
            let mut endpoint_scratch = FactorObjectiveScratch::new(date_count, realized_rank);
            let exact_zero_endpoint = factor_native_objective(
                &prepared,
                &observations,
                &factor,
                maximum_rank,
                realized_rank,
                0.0,
                0.0,
                true,
                &mut endpoint_scratch,
            )
            .unwrap();
            assert!(exact_zero_endpoint.score.is_finite());
            assert!(!exact_zero_endpoint.dense_fallback_used);
            assert_eq!(
                crate::temporal_covariance_batch::augmented_basis_reml_rho_derivatives(
                    &prepared,
                    &observations,
                    &factor,
                    maximum_rank,
                    realized_rank,
                    0.0,
                    0.0,
                )
                .unwrap_err(),
                TemporalInferenceStatus::CovarianceParameterAtBoundary
            );
            let zero_factor = vec![0.0; factor.len()];
            assert_eq!(
                crate::temporal_covariance_batch::augmented_basis_reml_rho_derivatives(
                    &prepared,
                    &observations,
                    &zero_factor,
                    maximum_rank,
                    realized_rank,
                    0.3,
                    0.0,
                )
                .unwrap_err(),
                TemporalInferenceStatus::CovarianceNonfinite
            );
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn simd_augmented_basis_reml_derivatives_match_scalar_reference() {
        for &(date_count, maximum_rank, realized_rank, irregular) in &[
            (12, 12, 12, false),
            (12, 12, 3, false),
            (48, 7, 7, true),
            (48, 30, 30, false),
            (96, 30, 30, true),
        ] {
            let target_count = 9;
            let mut elapsed = 0.0;
            let days = (0..date_count)
                .map(|date| {
                    elapsed += if irregular {
                        [4.0, 7.0, 12.0, 19.0, 25.0, 31.0][date % 6]
                    } else {
                        12.0
                    };
                    elapsed
                })
                .collect::<Vec<_>>();
            let prepared = PreparedFactorObjective::new(&days, 12.0).unwrap();
            let one_factor = release_factor_fixture(date_count, maximum_rank, realized_rank, false);
            let mut factors = vec![0.0; target_count * one_factor.len()];
            let mut observations_soa = vec![0.0; date_count * target_count];
            for target in 0..target_count {
                let offset = target * one_factor.len();
                factors[offset..offset + one_factor.len()].copy_from_slice(&one_factor);
                for date in 0..date_count {
                    observations_soa[date * target_count + target] = 0.012 * days[date]
                        + (date as f64 * 0.43 + target as f64 * 0.017).sin() * 0.7
                        + (date as f64 * 0.19).cos() * 0.2;
                }
            }
            let ranks = vec![realized_rank; target_count];
            let rhos = [0.05, 0.3, 0.95, 0.22, 0.71, 0.44, 0.83, 0.12, 0.57];
            let log_variances = [
                0.2_f64.ln(),
                1.3_f64.ln(),
                0.7_f64.ln(),
                0.4_f64.ln(),
                1.1_f64.ln(),
                0.9_f64.ln(),
                0.3_f64.ln(),
                1.6_f64.ln(),
                0.6_f64.ln(),
            ];
            let run = |threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| {
                        crate::temporal_covariance_batch::factor_native_reml_derivative_batch(
                            &prepared,
                            &observations_soa,
                            &factors,
                            maximum_rank,
                            &ranks,
                            &rhos,
                            &log_variances,
                        )
                        .unwrap()
                    })
            };
            let one = run(1);
            let parallel = run(12);
            for target in 0..target_count {
                let observations = (0..date_count)
                    .map(|date| observations_soa[date * target_count + target])
                    .collect::<Vec<_>>();
                let offset = target * one_factor.len();
                let scalar =
                    crate::temporal_covariance_batch::augmented_basis_reml_rho_derivatives(
                        &prepared,
                        &observations,
                        &factors[offset..offset + one_factor.len()],
                        maximum_rank,
                        realized_rank,
                        rhos[target],
                        log_variances[target],
                    )
                    .unwrap();
                let actual = one[target].as_ref().unwrap();
                let threaded = parallel[target].as_ref().unwrap();
                for (name, actual, expected) in [
                    ("score", actual.evaluation.score, scalar.evaluation.score),
                    ("score_eta", actual.score_eta, scalar.score_eta),
                    ("score_log_q", actual.score_log_q, scalar.score_log_q),
                    ("score_eta_eta", actual.score_eta_eta, scalar.score_eta_eta),
                    (
                        "score_eta_log_q",
                        actual.score_eta_log_q,
                        scalar.score_eta_log_q,
                    ),
                    (
                        "score_log_q_log_q",
                        actual.score_log_q_log_q,
                        scalar.score_log_q_log_q,
                    ),
                    ("slope_eta", actual.slope_eta, scalar.slope_eta),
                    ("slope_log_q", actual.slope_log_q, scalar.slope_log_q),
                ] {
                    assert!(
                        (actual - expected).abs() <= 1e-10 * expected.abs().max(1.0),
                        "{name} n={date_count} target={target}: {actual} != {expected}"
                    );
                }
                assert_eq!(
                    actual.evaluation.score.to_bits(),
                    threaded.evaluation.score.to_bits()
                );
                assert_eq!(actual.score_eta.to_bits(), threaded.score_eta.to_bits());
                assert_eq!(actual.score_log_q.to_bits(), threaded.score_log_q.to_bits());
                assert_eq!(
                    actual.score_eta_eta.to_bits(),
                    threaded.score_eta_eta.to_bits()
                );
                assert_eq!(
                    actual.score_eta_log_q.to_bits(),
                    threaded.score_eta_log_q.to_bits()
                );
                assert_eq!(
                    actual.score_log_q_log_q.to_bits(),
                    threaded.score_log_q_log_q.to_bits()
                );
                assert_eq!(actual.slope_eta.to_bits(), threaded.slope_eta.to_bits());
                assert_eq!(actual.slope_log_q.to_bits(), threaded.slope_log_q.to_bits());
            }
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn analytic_adjusted_scalar_matches_high_accuracy_finite_difference_and_student_t_intervals() {
        let date_count = 12;
        let maximum_rank = 12;
        let days = (1..=date_count)
            .map(|date| date as f64 * 12.0)
            .collect::<Vec<_>>();
        let prepared = PreparedFactorObjective::new(&days, 12.0).unwrap();
        let factor = release_factor_fixture(date_count, maximum_rank, maximum_rank, false);
        let observations = days
            .iter()
            .enumerate()
            .map(|(date, day)| {
                0.013 * day + (date as f64 * 0.61).sin() * 0.8 + (date as f64 * 0.23).cos() * 0.15
            })
            .collect::<Vec<_>>();
        let options = TemporalCovarianceOptions::default();
        let mut execution = crate::temporal_covariance_batch::TemporalBatchExecution::new(
            &prepared,
            &observations,
            &factor,
            maximum_rank,
            &[maximum_rank],
        )
        .unwrap();
        let report = execution.profile_reml(&options, true, false).unwrap();
        assert_eq!(report.metrics.exact_optimizer_fallback_targets, 0);
        let actual = report.outcomes[0].as_ref().unwrap();
        let actual_adjusted_variance = actual
            .covariance_parameter_adjusted_variance
            .expect("primary augmented-jet fit provides adjusted variance");

        let evaluate = |eta: f64, log_q: f64| {
            let mut scratch = FactorObjectiveScratch::new(date_count, maximum_rank);
            let evaluation = factor_native_objective(
                &prepared,
                &observations,
                &factor,
                maximum_rank,
                maximum_rank,
                eta.exp(),
                log_q,
                true,
                &mut scratch,
            )
            .unwrap();
            (evaluation.score, evaluation.slope)
        };
        let eta = actual.rho.ln();
        let log_q = actual.process_variance.ln();
        let eta_step = 2e-4;
        let log_q_step = 2e-4;
        let (base_score, _) = evaluate(eta, log_q);
        let second_derivative = |coordinate: usize, step: f64| {
            let score = |offset: f64| {
                if coordinate == 0 {
                    evaluate(eta + offset * step, log_q).0
                } else {
                    evaluate(eta, log_q + offset * step).0
                }
            };
            (-score(2.0) + 16.0 * score(1.0) - 30.0 * base_score + 16.0 * score(-1.0) - score(-2.0))
                / (12.0 * step * step)
        };
        let derivative_weights = [(1.0, -2.0), (-8.0, -1.0), (8.0, 1.0), (-1.0, 2.0)];
        let slope_derivative = |coordinate: usize, step: f64| {
            derivative_weights
                .iter()
                .map(|&(weight, offset)| {
                    let slope = if coordinate == 0 {
                        evaluate(eta + offset * step, log_q).1
                    } else {
                        evaluate(eta, log_q + offset * step).1
                    };
                    weight * slope
                })
                .sum::<f64>()
                / (12.0 * step)
        };
        let h_eta_eta = second_derivative(0, eta_step);
        let h_log_q_log_q = second_derivative(1, log_q_step);
        let h_eta_log_q = derivative_weights
            .iter()
            .flat_map(|&(eta_weight, eta_offset)| {
                derivative_weights.iter().map(move |&(q_weight, q_offset)| {
                    eta_weight
                        * q_weight
                        * evaluate(eta + eta_offset * eta_step, log_q + q_offset * log_q_step).0
                })
            })
            .sum::<f64>()
            / (144.0 * eta_step * log_q_step);
        let slope_eta = slope_derivative(0, eta_step);
        let slope_log_q = slope_derivative(1, log_q_step);
        let determinant = h_eta_eta * h_log_q_log_q - h_eta_log_q.powi(2);
        let finite_difference_nuisance_variance = 2.0
            * (h_log_q_log_q * slope_eta.powi(2) - 2.0 * h_eta_log_q * slope_eta * slope_log_q
                + h_eta_eta * slope_log_q.powi(2))
            / determinant;
        let finite_difference_adjusted_variance =
            actual.information_variance + finite_difference_nuisance_variance.max(0.0);
        eprintln!(
            "adjusted_scalar analytic={actual_adjusted_variance:.17e} finite_difference={finite_difference_adjusted_variance:.17e}"
        );
        assert!(
            (actual_adjusted_variance - finite_difference_adjusted_variance).abs()
                <= 1e-7 * finite_difference_adjusted_variance.abs().max(1e-12),
            "analytic={actual_adjusted_variance:.17e}, finite_difference={finite_difference_adjusted_variance:.17e}"
        );

        let residual_degrees_of_freedom = date_count - 1;
        let distribution = StudentsT::new(0.0, 1.0, residual_degrees_of_freedom as f64).unwrap();
        for probability in [0.68, 0.90, 0.95] {
            let multiplier = distribution.inverse_cdf(0.5 + probability / 2.0);
            let analytic_half_width = multiplier * actual_adjusted_variance.sqrt();
            let finite_difference_half_width =
                multiplier * finite_difference_adjusted_variance.sqrt();
            for (analytic, expected) in [
                (
                    actual.slope - analytic_half_width,
                    actual.slope - finite_difference_half_width,
                ),
                (
                    actual.slope + analytic_half_width,
                    actual.slope + finite_difference_half_width,
                ),
            ] {
                assert!(
                    (analytic - expected).abs() <= 1e-7 * expected.abs().max(1e-12),
                    "p={probability}: analytic={analytic:.17e}, finite_difference={expected:.17e}"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn persisted_factor_batch_returns_plugin_and_adjusted_scalar_from_one_profile() {
        let date_count = 12;
        let maximum_rank = 12;
        let target_count = 2;
        let days = (1..=date_count)
            .map(|date| date as f64 * 12.0)
            .collect::<Vec<_>>();
        let one_factor = release_factor_fixture(date_count, maximum_rank, maximum_rank, false);
        let mut persisted_factors =
            Vec::with_capacity(target_count * (date_count + 1) * maximum_rank);
        for target in 0..target_count {
            persisted_factors.extend(std::iter::repeat_n(0.0, maximum_rank));
            persisted_factors.extend(
                one_factor
                    .iter()
                    .map(|value| value * (1.0 + target as f64 * 0.02)),
            );
        }
        let observations_soa = (0..date_count)
            .flat_map(|date| {
                let day = days[date];
                (0..target_count).map(move |target| {
                    0.013 * day
                        + (date as f64 * 0.61 + target as f64 * 0.17).sin() * 0.8
                        + (date as f64 * 0.23).cos() * 0.15
                })
            })
            .collect::<Vec<_>>();
        let report = crate::fit_temporal_factor_scalar_batch(
            &days,
            &observations_soa,
            &persisted_factors,
            maximum_rank,
            &vec![maximum_rank; target_count],
            &TemporalCovarianceOptions::default(),
        )
        .unwrap();
        assert_eq!(report.outcomes.len(), target_count);
        assert_eq!(report.metrics.profile_fit_count, target_count);
        assert_eq!(report.metrics.bootstrap_attempts, 0);
        for outcome in &report.outcomes {
            assert_eq!(
                outcome.plugin_gls_reml.status,
                TemporalInferenceStatus::Evaluated
            );
            assert_eq!(
                outcome.reml_covariance_parameter_adjusted_scalar.status,
                TemporalInferenceStatus::Evaluated
            );
            assert_eq!(
                outcome.plugin_gls_reml.point_estimate,
                outcome
                    .reml_covariance_parameter_adjusted_scalar
                    .point_estimate
            );
            assert!(
                outcome
                    .reml_covariance_parameter_adjusted_scalar
                    .standard_error_diagnostic
                    .unwrap()
                    >= outcome.plugin_gls_reml.standard_error_diagnostic.unwrap()
            );
        }

        let plugin_only = crate::fit_temporal_factor_plugin_batch(
            &days,
            &observations_soa,
            &persisted_factors,
            maximum_rank,
            &vec![maximum_rank; target_count],
            &TemporalCovarianceOptions::default(),
        )
        .unwrap();
        assert_eq!(plugin_only.metrics.profile_fit_count, target_count);
        assert_eq!(plugin_only.metrics.bootstrap_attempts, 0);
        assert_eq!(plugin_only.metrics.covariance_parameter_adjustment_count, 0);
        assert_eq!(
            plugin_only
                .metrics
                .covariance_parameter_derivative_lane_evaluations,
            0
        );
        assert_eq!(
            report.metrics.covariance_parameter_adjustment_count,
            target_count
        );
        assert!(
            report
                .metrics
                .covariance_parameter_derivative_lane_evaluations
                > 0
        );
        assert_eq!(
            plugin_only.metrics.optimizer_rho_lane_evaluations,
            report.metrics.optimizer_rho_lane_evaluations
        );
        assert_eq!(
            plugin_only.metrics.optimizer_q_objective_evaluations,
            report.metrics.optimizer_q_objective_evaluations
        );
        assert_eq!(
            plugin_only.metrics.optimizer_primary_rho_pass_histogram,
            report.metrics.optimizer_primary_rho_pass_histogram
        );
        for (plugin, adjusted) in plugin_only.outcomes.iter().zip(&report.outcomes) {
            assert_eq!(plugin.plugin_gls_reml, adjusted.plugin_gls_reml);
            assert_eq!(
                plugin.reml_covariance_parameter_adjusted_scalar.status,
                TemporalInferenceStatus::DiagnosticNotComputed
            );
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn factor_native_objective_matches_dense_oracle_for_release_shapes() {
        let options = TemporalCovarianceOptions::default();
        let mut dense_elapsed = std::time::Duration::ZERO;
        let mut factor_elapsed = std::time::Duration::ZERO;
        let mut evaluated = 0_usize;
        let mut dense_fallbacks = 0_usize;
        for (date_count, maximum_rank) in [(12, 12), (48, 30), (96, 30)] {
            assert_eq!(maximum_rank, date_count.min(2 * (date_count + 1).min(15)));
            for irregular in [false, true] {
                let mut elapsed = 0.0;
                let days = (0..date_count)
                    .map(|index| {
                        elapsed += if irregular {
                            [4.0, 7.0, 12.0, 19.0, 25.0, 31.0][index % 6]
                        } else {
                            12.0
                        };
                        elapsed
                    })
                    .collect::<Vec<_>>();
                let observations = days
                    .iter()
                    .enumerate()
                    .map(|(index, day)| {
                        0.018 * day
                            + (index as f64 * 0.43).sin() * 0.7
                            + (index as f64 * 0.17).cos() * 0.2
                    })
                    .collect::<Vec<_>>();
                let bounds = nuisance_bounds(&days, &observations, &options).unwrap();
                let rhos = [
                    bounds.rho_lower,
                    (bounds.rho_lower + bounds.rho_upper) / 2.0,
                    bounds.rho_upper,
                ];
                let log_variances = [
                    bounds.log_variance_lower,
                    (bounds.log_variance_lower + bounds.log_variance_upper) / 2.0,
                    bounds.log_variance_upper,
                ];
                let prepared =
                    PreparedFactorObjective::new(&days, options.reference_lag_days).unwrap();
                let mut scratch = FactorObjectiveScratch::new(date_count, maximum_rank);
                for (realized_rank, near_rank) in [
                    (maximum_rank, false),
                    (maximum_rank.min(3), false),
                    (maximum_rank, true),
                ] {
                    let factor =
                        release_factor_fixture(date_count, maximum_rank, realized_rank, near_rank);
                    for rho in rhos {
                        for log_process_variance in log_variances {
                            let started = std::time::Instant::now();
                            let dense = dense_factor_objective(
                                &days,
                                &observations,
                                &factor,
                                maximum_rank,
                                realized_rank,
                                rho,
                                log_process_variance,
                                &options,
                            );
                            dense_elapsed += started.elapsed();
                            let started = std::time::Instant::now();
                            let native = factor_native_objective(
                                &prepared,
                                &observations,
                                &factor,
                                maximum_rank,
                                realized_rank,
                                rho,
                                log_process_variance,
                                true,
                                &mut scratch,
                            );
                            factor_elapsed += started.elapsed();
                            dense_fallbacks += native
                                .as_ref()
                                .map_or(0, |value| usize::from(value.dense_fallback_used));
                            let case = format!(
                                "n={date_count},r={realized_rank},irregular={irregular},near={near_rank},rho={rho},log_sigma={log_process_variance}"
                            );
                            match (dense, native) {
                                (Err(dense), Err(native)) => assert_eq!(dense, native),
                                (Ok(dense), Ok(native)) => {
                                    assert_relative(
                                        dense.score,
                                        native.score,
                                        &format!("{case} REML objective"),
                                    );
                                    assert_relative(
                                        dense.slope,
                                        native.slope,
                                        &format!("{case} slope"),
                                    );
                                    assert_relative(
                                        dense.x_v_x,
                                        native.x_v_x,
                                        &format!("{case} xVx"),
                                    );
                                    assert_relative(
                                        dense.x_v_y,
                                        native.x_v_y,
                                        &format!("{case} xVy"),
                                    );
                                    assert_relative(
                                        dense.y_v_y,
                                        native.y_v_y,
                                        &format!("{case} yVy"),
                                    );
                                    assert_relative(
                                        dense.log_determinant,
                                        native.log_determinant,
                                        &format!("{case} log determinant"),
                                    );
                                    assert_relative(
                                        dense.quadratic,
                                        native.quadratic,
                                        &format!("{case} quadratic"),
                                    );
                                }
                                (dense, native) => {
                                    panic!(
                                        "{case} factor/dense status mismatch: {dense:?} != {native:?}"
                                    )
                                }
                            }
                            evaluated += 1;
                        }
                    }
                }
            }
        }
        eprintln!(
            "factor objective parity: cases={evaluated}, dense_fallbacks={dense_fallbacks}, dense_us={}, factor_us={}",
            dense_elapsed.as_micros(),
            factor_elapsed.as_micros()
        );
    }

    #[test]
    fn single_reml_start_matches_current_three_start_trace() {
        let days = (1..=12)
            .map(|index| index as f64 * 12.0)
            .collect::<Vec<_>>();
        let observations = days
            .iter()
            .enumerate()
            .map(|(index, day)| 0.012 * day + (index as f64 * 0.61).sin())
            .collect::<Vec<_>>();
        let factor = release_factor_fixture(days.len(), days.len(), days.len(), false);
        let covariance = (0..days.len())
            .map(|left| {
                (0..days.len())
                    .map(|right| {
                        (0..days.len())
                            .map(|component| {
                                factor[left * days.len() + component]
                                    * factor[right * days.len() + component]
                            })
                            .sum()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let options = TemporalCovarianceOptions::default();
        let bounds = nuisance_bounds(&days, &observations, &options).unwrap();
        let (three, trace) = optimize_covariance_repeated(
            &days,
            &observations,
            &covariance,
            &options,
            bounds,
            true,
            3,
        );
        assert_eq!(trace.len(), 3);
        assert!(trace.windows(2).all(|pair| pair[0] == pair[1]));
        let (single, single_trace) = optimize_covariance_repeated(
            &days,
            &observations,
            &covariance,
            &options,
            bounds,
            true,
            1,
        );
        assert_eq!(single_trace.as_slice(), &trace[..1]);
        match (three, single) {
            (Ok(three), Ok(single)) => assert_plugin_bits(&three, &single),
            (Err(three), Err(single)) => assert_eq!(three, single),
            (three, single) => {
                panic!("single/three-start status mismatch: {three:?} != {single:?}")
            }
        }
    }

    fn assert_factor_objective_bits(
        left: &Result<FactorObjectiveEvaluation, TemporalInferenceStatus>,
        right: &Result<FactorObjectiveEvaluation, TemporalInferenceStatus>,
    ) {
        match (left, right) {
            (Err(left), Err(right)) => assert_eq!(left, right),
            (Ok(left), Ok(right)) => {
                for (name, left, right) in [
                    ("x_v_x", left.x_v_x, right.x_v_x),
                    ("x_v_y", left.x_v_y, right.x_v_y),
                    ("y_v_y", left.y_v_y, right.y_v_y),
                    ("slope", left.slope, right.slope),
                    (
                        "log_determinant",
                        left.log_determinant,
                        right.log_determinant,
                    ),
                    ("quadratic", left.quadratic, right.quadratic),
                    ("score", left.score, right.score),
                ] {
                    assert_eq!(left.to_bits(), right.to_bits(), "{name}: {left} != {right}");
                }
                assert_eq!(left.dense_fallback_used, right.dense_fallback_used);
            }
            (left, right) => panic!("factor objective status mismatch: {left:?} != {right:?}"),
        }
    }

    fn assert_factor_objective_relative(
        left: &Result<FactorObjectiveEvaluation, TemporalInferenceStatus>,
        right: &Result<FactorObjectiveEvaluation, TemporalInferenceStatus>,
    ) {
        match (left, right) {
            (Err(left), Err(right)) => assert_eq!(left, right),
            (Ok(left), Ok(right)) => {
                for (name, left, right) in [
                    ("x_v_x", left.x_v_x, right.x_v_x),
                    ("x_v_y", left.x_v_y, right.x_v_y),
                    ("y_v_y", left.y_v_y, right.y_v_y),
                    ("slope", left.slope, right.slope),
                    (
                        "log_determinant",
                        left.log_determinant,
                        right.log_determinant,
                    ),
                    ("quadratic", left.quadratic, right.quadratic),
                    ("score", left.score, right.score),
                ] {
                    assert_relative(left, right, name);
                }
                assert_eq!(left.dense_fallback_used, right.dense_fallback_used);
            }
            (left, right) => panic!("factor objective status mismatch: {left:?} != {right:?}"),
        }
    }

    #[test]
    fn factor_microbatch_w1_w8_and_rayon_threads_are_bit_exact() {
        let date_count = 48;
        let maximum_rank = 30;
        let target_count = 17;
        let options = TemporalCovarianceOptions::default();
        let mut elapsed = 0.0;
        let days = (0..date_count)
            .map(|index| {
                elapsed += [4.0, 7.0, 12.0, 19.0, 25.0, 31.0][index % 6];
                elapsed
            })
            .collect::<Vec<_>>();
        let prepared = PreparedFactorObjective::new(&days, options.reference_lag_days).unwrap();
        let realized_ranks = (0..target_count)
            .map(|target| if target % 3 == 0 { 3 } else { maximum_rank })
            .collect::<Vec<_>>();
        let mut factors = vec![0.0; target_count * date_count * maximum_rank];
        let mut observations_soa = vec![0.0; date_count * target_count];
        for target in 0..target_count {
            let factor = release_factor_fixture(
                date_count,
                maximum_rank,
                realized_ranks[target],
                target % 5 == 0,
            );
            let offset = target * date_count * maximum_rank;
            factors[offset..offset + factor.len()].copy_from_slice(&factor);
            for date in 0..date_count {
                observations_soa[date * target_count + target] =
                    0.014 * days[date] + (date as f64 * 0.41 + target as f64 * 0.07).sin() * 0.6;
            }
        }
        let rhos = (0..target_count)
            .map(|target| {
                if target == 15 {
                    1.0
                } else {
                    0.15 + 0.03 * (target % 9) as f64
                }
            })
            .collect::<Vec<_>>();
        let log_process_variances = (0..target_count)
            .map(|target| (0.3 + target as f64 * 0.02).ln())
            .collect::<Vec<_>>();
        factors[16 * date_count * maximum_rank..16 * date_count * maximum_rank + maximum_rank]
            .fill(0.0);

        let run = |threads, lane_width| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    factor_native_objective_microbatch(
                        &prepared,
                        &observations_soa,
                        &factors,
                        maximum_rank,
                        &realized_ranks,
                        &rhos,
                        &log_process_variances,
                        true,
                        lane_width,
                    )
                    .unwrap()
                })
        };
        let scalar = run(1, 1);
        let lane_eight = run(1, 8);
        let parallel = run(12, 8);
        assert_eq!(scalar.len(), target_count);
        assert_eq!(lane_eight.len(), target_count);
        assert_eq!(parallel.len(), target_count);
        for target in 0..target_count {
            assert_factor_objective_bits(&scalar[target], &lane_eight[target]);
            assert_factor_objective_bits(&scalar[target], &parallel[target]);
        }
        assert!(matches!(
            scalar[15],
            Err(TemporalInferenceStatus::CovarianceParameterAtBoundary)
        ));
        assert!(matches!(
            scalar[16],
            Err(TemporalInferenceStatus::CovarianceNonfinite)
        ));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn simd_factor_batch_matches_scalar_factor_kernel() {
        let options = TemporalCovarianceOptions::default();
        for (date_count, irregular, target_count) in [(12_usize, false, 7_usize), (48, true, 9)] {
            let maximum_rank = date_count.min(30);
            let mut elapsed = 0.0;
            let days = (0..date_count)
                .map(|date| {
                    elapsed += if irregular {
                        [4.0, 7.0, 12.0, 19.0, 25.0, 31.0][date % 6]
                    } else {
                        12.0
                    };
                    elapsed
                })
                .collect::<Vec<_>>();
            let prepared = PreparedFactorObjective::new(&days, options.reference_lag_days).unwrap();
            let realized_ranks = (0..target_count)
                .map(|target| match target % 3 {
                    0 => 3,
                    1 => maximum_rank.min(7),
                    _ => maximum_rank,
                })
                .collect::<Vec<_>>();
            let mut factors = vec![0.0; target_count * date_count * maximum_rank];
            let mut observations_soa = vec![0.0; date_count * target_count];
            for target in 0..target_count {
                let factor = release_factor_fixture(
                    date_count,
                    maximum_rank,
                    realized_ranks[target],
                    target % 4 == 0,
                );
                let offset = target * date_count * maximum_rank;
                factors[offset..offset + factor.len()].copy_from_slice(&factor);
                for date in 0..date_count {
                    observations_soa[date * target_count + target] = 0.014 * days[date]
                        + (date as f64 * 0.41 + target as f64 * 0.07).sin() * 0.6;
                }
            }
            let rhos = (0..target_count)
                .map(|target| {
                    if target + 2 == target_count {
                        1.0
                    } else {
                        0.15 + 0.03 * (target % 9) as f64
                    }
                })
                .collect::<Vec<_>>();
            let log_process_variances = (0..target_count)
                .map(|target| {
                    if target == 0 {
                        -20.0
                    } else {
                        (0.3 + target as f64 * 0.02).ln()
                    }
                })
                .collect::<Vec<_>>();
            let invalid_target = target_count - 1;
            let invalid_offset = invalid_target * date_count * maximum_rank;
            factors[invalid_offset..invalid_offset + maximum_rank].fill(0.0);

            let scalar = (0..target_count)
                .map(|target| {
                    let observations = (0..date_count)
                        .map(|date| observations_soa[date * target_count + target])
                        .collect::<Vec<_>>();
                    let factor_offset = target * date_count * maximum_rank;
                    let mut scratch = FactorObjectiveScratch::new(date_count, maximum_rank);
                    factor_native_objective(
                        &prepared,
                        &observations,
                        &factors[factor_offset..factor_offset + date_count * maximum_rank],
                        maximum_rank,
                        realized_ranks[target],
                        rhos[target],
                        log_process_variances[target],
                        true,
                        &mut scratch,
                    )
                })
                .collect::<Vec<_>>();
            let run = |threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| {
                        crate::temporal_covariance_batch::factor_native_objective_batch(
                            &prepared,
                            &observations_soa,
                            &factors,
                            maximum_rank,
                            &realized_ranks,
                            &rhos,
                            &log_process_variances,
                            true,
                        )
                        .unwrap()
                    })
            };
            let one_thread = run(1);
            let twelve_threads = run(12);
            for target in 0..target_count {
                assert_factor_objective_bits(&scalar[target], &one_thread[target]);
                assert_factor_objective_bits(&scalar[target], &twelve_threads[target]);
            }
            assert!(scalar[0]
                .as_ref()
                .is_ok_and(|evaluation| evaluation.dense_fallback_used));
            assert!(matches!(
                scalar[target_count - 2],
                Err(TemporalInferenceStatus::CovarianceParameterAtBoundary)
            ));
            assert!(matches!(
                scalar[target_count - 1],
                Err(TemporalInferenceStatus::CovarianceNonfinite)
            ));
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn cadence_basis_matches_streamed_factor_kernel_and_is_deterministic() {
        use crate::temporal_covariance_batch::{
            TemporalBasisDisposition, TemporalBasisMode, TemporalBatchExecution,
        };

        let options = TemporalCovarianceOptions::default();
        for (date_count, maximum_rank, irregular) in
            [(12_usize, 12_usize, false), (48, 30, true), (96, 30, false)]
        {
            let target_count = 9;
            let mut elapsed = 0.0;
            let days = (0..date_count)
                .map(|date| {
                    elapsed += if irregular {
                        [4.0, 7.0, 12.0, 19.0, 25.0, 31.0][date % 6]
                    } else {
                        12.0
                    };
                    elapsed
                })
                .collect::<Vec<_>>();
            let prepared = PreparedFactorObjective::new(&days, options.reference_lag_days).unwrap();
            let ranks = (0..target_count)
                .map(|target| [3, maximum_rank.min(7), maximum_rank][target % 3])
                .collect::<Vec<_>>();
            let mut factors = vec![0.0; target_count * date_count * maximum_rank];
            let mut observations_soa = vec![0.0; date_count * target_count];
            for target in 0..target_count {
                let factor = release_factor_fixture(
                    date_count,
                    maximum_rank,
                    ranks[target],
                    target % 4 == 0,
                );
                let offset = target * date_count * maximum_rank;
                factors[offset..offset + factor.len()].copy_from_slice(&factor);
                for date in 0..date_count {
                    observations_soa[date * target_count + target] = 0.014 * days[date]
                        + (date as f64 * 0.41 + target as f64 * 0.07).sin() * 0.6;
                }
            }
            let rhos = (0..target_count)
                .map(|target| [0.0, 0.31, 0.95][target % 3])
                .collect::<Vec<_>>();
            let log_process_variances = (0..target_count)
                .map(|target| [0.08_f64, 0.5, 2.0][target % 3].ln())
                .collect::<Vec<_>>();
            let run = |threads, mode| {
                let mut execution = TemporalBatchExecution::new_with_basis_mode(
                    &prepared,
                    &observations_soa,
                    &factors,
                    maximum_rank,
                    &ranks,
                    mode,
                )
                .unwrap();
                let disposition = execution.metrics().basis_disposition;
                let outcomes = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| {
                        execution
                            .evaluate(&rhos, &log_process_variances, true)
                            .unwrap()
                            .to_vec()
                    });
                (disposition, outcomes)
            };
            let (basis_disposition, basis_one) = run(1, TemporalBasisMode::Auto);
            let (_, basis_parallel) = run(12, TemporalBasisMode::Auto);
            let (streamed_disposition, streamed) = run(1, TemporalBasisMode::Streamed);
            assert_eq!(basis_disposition, TemporalBasisDisposition::Prepared);
            assert_eq!(
                streamed_disposition,
                TemporalBasisDisposition::StreamedForced
            );
            for target in 0..target_count {
                assert_factor_objective_relative(&basis_one[target], &streamed[target]);
                assert_factor_objective_bits(&basis_one[target], &basis_parallel[target]);
            }
        }

        let days = (1..=12)
            .scan(0.0, |elapsed, gap| {
                *elapsed += gap as f64;
                Some(*elapsed)
            })
            .collect::<Vec<_>>();
        let prepared = PreparedFactorObjective::new(&days, options.reference_lag_days).unwrap();
        let factor = release_factor_fixture(12, 12, 12, false);
        let observations = days
            .iter()
            .map(|day| 0.014 * day + day.sin())
            .collect::<Vec<_>>();
        let ranks = [12];
        let mut execution = TemporalBatchExecution::new_with_basis_mode(
            &prepared,
            &observations,
            &factor,
            12,
            &ranks,
            TemporalBasisMode::Auto,
        )
        .unwrap();
        assert_eq!(
            execution.metrics().basis_disposition,
            TemporalBasisDisposition::StreamedTooManyGapClasses
        );
        let outcome = execution.evaluate(&[0.3], &[0.5_f64.ln()], true).unwrap();
        assert!(outcome[0].is_ok());
    }

    #[test]
    fn cheap_factor_condition_certificate_uses_exact_fallback_without_false_rejection() {
        for (date_count, maximum_rank) in [(12_usize, 12_usize), (48, 30), (96, 30)] {
            let days = (1..=date_count)
                .map(|date| date as f64 * 12.0)
                .collect::<Vec<_>>();
            let prepared = PreparedFactorObjective::new(&days, 12.0).unwrap();
            let factor = release_factor_fixture(date_count, maximum_rank, maximum_rank, false);
            let rho = 0.3;
            let process_variance = 0.5;
            let covariance = total_covariance_from_factor(
                &prepared,
                &factor,
                maximum_rank,
                maximum_rank,
                rho,
                process_variance,
            )
            .unwrap();
            let exact = condition_number(&covariance);
            let cheap = factor_condition_certificate(
                &prepared,
                &factor,
                maximum_rank,
                maximum_rank,
                rho,
                process_variance,
                1e12,
            )
            .unwrap();
            assert_eq!(cheap.method, FactorConditionMethod::ConservativeUpperBound);
            assert!(cheap.conservative_upper_bound >= exact * (1.0 - 1e-12));
            assert!(cheap.exact_condition_number.is_none());

            let fallback_limit = (cheap.conservative_upper_bound + exact) / 2.0;
            let fallback = factor_condition_certificate(
                &prepared,
                &factor,
                maximum_rank,
                maximum_rank,
                rho,
                process_variance,
                fallback_limit,
            )
            .unwrap();
            assert_eq!(
                fallback.method,
                FactorConditionMethod::ExactEigenvalueFallback
            );
            assert_eq!(
                fallback.exact_condition_number.unwrap().to_bits(),
                exact.to_bits()
            );
            assert_eq!(fallback.reported_condition.to_bits(), exact.to_bits());

            assert_eq!(
                factor_condition_certificate(
                    &prepared,
                    &factor,
                    maximum_rank,
                    maximum_rank,
                    rho,
                    process_variance,
                    exact * 0.99,
                ),
                Err(TemporalInferenceStatus::DesignIllConditioned)
            );
        }
    }

    #[test]
    fn basis_projection_omits_streamed_whitening_and_edge_expansion() {
        let days = (1..=12).map(|date| date as f64 * 12.0).collect::<Vec<_>>();
        let prepared = PreparedFactorObjective::new(&days, 12.0).unwrap();
        let factor = release_factor_fixture(12, 12, 12, false);
        let observations = days
            .iter()
            .enumerate()
            .map(|(date, day)| 0.014 * day + (date as f64 * 0.41).sin())
            .collect::<Vec<_>>();
        let mut execution = crate::temporal_covariance_batch::TemporalBatchExecution::new(
            &prepared,
            &observations,
            &factor,
            12,
            &[12],
        )
        .unwrap();
        let outcome = execution.evaluate(&[0.3], &[0.5_f64.ln()], true).unwrap();
        assert!(outcome[0].is_ok());
        let metrics = execution.metrics();
        assert_eq!(metrics.basis_streamed_whitening_elements, 0);
        assert_eq!(metrics.basis_edge_transition_elements, 0);
    }

    #[test]
    fn temporal_batch_execution_retains_only_worker_bounded_solver_scratch() {
        let date_count = 48;
        let maximum_rank = 30;
        let target_count = 1025;
        let days = (1..=date_count)
            .map(|date| date as f64 * 12.0)
            .collect::<Vec<_>>();
        let prepared = PreparedFactorObjective::new(&days, 12.0).unwrap();
        let one_factor = release_factor_fixture(date_count, maximum_rank, maximum_rank, false);
        let mut factors = vec![0.0; target_count * date_count * maximum_rank];
        for target in 0..target_count {
            let offset = target * date_count * maximum_rank;
            factors[offset..offset + one_factor.len()].copy_from_slice(&one_factor);
        }
        let observations = (0..date_count * target_count)
            .map(|index| (index as f64 * 0.017).sin())
            .collect::<Vec<_>>();
        let ranks = vec![maximum_rank; target_count];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let metrics = pool.install(|| {
            crate::temporal_covariance_batch::TemporalBatchExecution::new(
                &prepared,
                &observations,
                &factors,
                maximum_rank,
                &ranks,
            )
            .unwrap()
            .metrics()
        });
        assert_eq!(metrics.worker_count, 4);
        assert_eq!(metrics.retained_prepared_chunk_count, 0);
        assert!(metrics.total_retained_solver_scratch_bytes <= 4 * 8 * 1024 * 1024);
    }

    #[test]
    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    fn batched_profile_matches_scalar_spectral_order_and_compaction() {
        let date_count = 12;
        let maximum_rank = 12;
        let target_count = 17;
        let days = (1..=date_count)
            .map(|date| date as f64 * 12.0)
            .collect::<Vec<_>>();
        let prepared = PreparedFactorObjective::new(&days, 12.0).unwrap();
        let ranks = (0..target_count)
            .map(|target| [3, 7, maximum_rank][target % 3])
            .collect::<Vec<_>>();
        let mut factors = vec![0.0; target_count * date_count * maximum_rank];
        let mut observations_soa = vec![0.0; date_count * target_count];
        for target in 0..target_count {
            let factor =
                release_factor_fixture(date_count, maximum_rank, ranks[target], target % 5 == 0);
            let offset = target * date_count * maximum_rank;
            factors[offset..offset + factor.len()].copy_from_slice(&factor);
            for date in 0..date_count {
                observations_soa[date * target_count + target] = 0.013 * days[date]
                    + (date as f64 * 0.61 + target as f64 * 0.031).sin() * 0.8
                    + (date as f64 * 0.23).cos() * 0.15;
            }
        }
        for target in [4_usize, 15] {
            let offset = target * date_count * maximum_rank;
            factors[offset..offset + date_count * maximum_rank].fill(0.0);
        }
        let options = TemporalCovarianceOptions::default();
        let scalar = (0..target_count)
            .map(|target| {
                let observations = (0..date_count)
                    .map(|date| observations_soa[date * target_count + target])
                    .collect::<Vec<_>>();
                let offset = target * date_count * maximum_rank;
                let mut scratch = FactorObjectiveScratch::new(date_count, maximum_rank);
                factor_native_profile_plugin(
                    &prepared,
                    &observations,
                    &factors[offset..offset + date_count * maximum_rank],
                    maximum_rank,
                    ranks[target],
                    &options,
                    &mut scratch,
                    false,
                )
            })
            .collect::<Vec<_>>();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut execution =
                        crate::temporal_covariance_batch::TemporalBatchExecution::new(
                            &prepared,
                            &observations_soa,
                            &factors,
                            maximum_rank,
                            &ranks,
                        )
                        .unwrap();
                    let report = execution.profile_reml(&options, true, false).unwrap();
                    (report.outcomes.to_vec(), report.metrics.clone())
                })
        };
        let (one, one_metrics) = run(1);
        let (parallel, parallel_metrics) = run(12);
        assert_eq!(one.len(), target_count);
        assert_eq!(
            one_metrics.primary_rho_pass_histogram,
            parallel_metrics.primary_rho_pass_histogram
        );
        assert_eq!(
            one_metrics.q_objective_evaluations,
            parallel_metrics.q_objective_evaluations
        );
        assert_eq!(
            one_metrics.maximum_primary_rho_passes,
            parallel_metrics.maximum_primary_rho_passes
        );
        assert_eq!(
            one_metrics.exact_optimizer_fallback_targets,
            parallel_metrics.exact_optimizer_fallback_targets
        );
        assert_eq!(
            one_metrics.fixed_theta_dense_fallback_evaluations,
            parallel_metrics.fixed_theta_dense_fallback_evaluations
        );
        assert_eq!(
            one_metrics.condition_upper_bound_accepts,
            parallel_metrics.condition_upper_bound_accepts
        );
        assert_eq!(
            one_metrics.condition_exact_fallbacks,
            parallel_metrics.condition_exact_fallbacks
        );
        assert_eq!(
            one_metrics.compacted_lane_count,
            parallel_metrics.compacted_lane_count
        );
        assert_eq!(
            one_metrics.completed_lane_revisits,
            parallel_metrics.completed_lane_revisits
        );
        assert_eq!(
            one_metrics.rho_lane_evaluations,
            parallel_metrics.rho_lane_evaluations
        );
        assert_eq!(
            one_metrics.covariance_parameter_derivative_lane_evaluations,
            parallel_metrics.covariance_parameter_derivative_lane_evaluations
        );
        assert_eq!(
            one_metrics.per_target_rho_passes,
            parallel_metrics.per_target_rho_passes
        );
        assert!(one_metrics.maximum_worker_scratch_bytes <= 8 * 1024 * 1024);
        assert!(parallel_metrics.maximum_worker_scratch_bytes <= 8 * 1024 * 1024);
        assert_eq!(one_metrics.completed_lane_revisits, 0);
        assert_eq!(
            one_metrics.rho_lane_evaluations,
            one_metrics.per_target_rho_passes.iter().sum::<usize>()
        );
        assert!(one_metrics.maximum_primary_rho_passes <= 20);
        assert_eq!(
            one_metrics.primary_rho_pass_histogram.iter().sum::<u64>(),
            target_count as u64
        );
        for target in 0..target_count {
            match (&scalar[target], &one[target], &parallel[target]) {
                (Err(expected), Err(actual), Err(parallel)) => {
                    assert_eq!(expected, actual);
                    assert_eq!(actual, parallel);
                    if [4, 15].contains(&target) {
                        assert_eq!(one_metrics.per_target_rho_passes[target], 0);
                    }
                }
                (Ok(expected), Ok(actual), Ok(parallel)) => {
                    assert!(actual.score <= expected.score + 1e-8 * (1.0 + expected.score.abs()));
                    let curvature_tolerance = if actual.profile_rho_curvature.is_finite()
                        && actual.profile_rho_curvature > 0.0
                    {
                        (2.0 * options.optimizer_tolerance * (1.0 + actual.score.abs())
                            / actual.profile_rho_curvature)
                            .sqrt()
                            .min(5e-3)
                    } else {
                        0.0
                    };
                    let rho_tolerance = 5e-4_f64.max(curvature_tolerance);
                    assert!(
                        (actual.rho - expected.rho).abs() <= rho_tolerance,
                        "target {target} batch/scalar rho: expected={} score={} factorizations={}, actual={} score={}, curvature={}, tolerance={rho_tolerance}",
                        expected.rho,
                        expected.score,
                        expected.primary_factorization_count,
                        actual.rho,
                        actual.score,
                        actual.profile_rho_curvature,
                    );
                    assert!(
                        (actual.process_variance.ln() - expected.process_variance.ln()).abs()
                            <= 1.5e-2
                    );
                    let slope_tolerance = 2.0
                        * options.optimizer_tolerance
                        * actual.slope.abs().max(expected.slope.abs()).max(1e-12);
                    assert!(
                        (actual.slope - expected.slope).abs() <= slope_tolerance,
                        "target {target} batch/scalar slope: expected={}, actual={}, tolerance={slope_tolerance}",
                        expected.slope,
                        actual.slope,
                    );
                    assert!(
                        (actual.information_variance - expected.information_variance).abs()
                            <= 1e-8
                                * actual
                                    .information_variance
                                    .abs()
                                    .max(expected.information_variance.abs())
                                    .max(1.0)
                    );
                    assert_eq!(actual.score.to_bits(), parallel.score.to_bits());
                    assert_eq!(actual.rho.to_bits(), parallel.rho.to_bits());
                    assert_eq!(
                        actual.process_variance.to_bits(),
                        parallel.process_variance.to_bits()
                    );
                    assert_eq!(
                        actual
                            .covariance_parameter_adjusted_variance
                            .map(f64::to_bits),
                        parallel
                            .covariance_parameter_adjusted_variance
                            .map(f64::to_bits)
                    );
                    assert!(one_metrics.per_target_rho_passes[target] > 0);
                }
                (expected, actual, parallel) => panic!(
                    "target {target} profile mismatch: {expected:?} != {actual:?} != {parallel:?}"
                ),
            }
        }
    }

    #[test]
    fn batched_newton_profile_converges_release_fixture_without_exact_fallback() {
        let date_count = 12;
        let maximum_rank = 12;
        let target_count = 9;
        let days = (1..=date_count)
            .map(|date| date as f64 * 12.0)
            .collect::<Vec<_>>();
        let prepared = PreparedFactorObjective::new(&days, 12.0).unwrap();
        let one_factor = release_factor_fixture(date_count, maximum_rank, maximum_rank, false);
        let mut factors = vec![0.0; target_count * one_factor.len()];
        let mut observations_soa = vec![0.0; date_count * target_count];
        for target in 0..target_count {
            let offset = target * one_factor.len();
            factors[offset..offset + one_factor.len()].copy_from_slice(&one_factor);
            for date in 0..date_count {
                observations_soa[date * target_count + target] = 0.013 * days[date]
                    + (date as f64 * 0.61 + target as f64 * 0.031).sin() * 0.8
                    + (date as f64 * 0.23).cos() * 0.15;
            }
        }
        let ranks = vec![maximum_rank; target_count];
        let options = TemporalCovarianceOptions::default();
        let mut execution = crate::temporal_covariance_batch::TemporalBatchExecution::new(
            &prepared,
            &observations_soa,
            &factors,
            maximum_rank,
            &ranks,
        )
        .unwrap();
        let report = execution.profile_reml(&options, true, false).unwrap();
        assert!(report.outcomes.iter().all(Result::is_ok));
        assert_eq!(report.metrics.exact_optimizer_fallback_targets, 0);
        assert!(report.metrics.maximum_primary_rho_passes <= 7);
        assert!(report.metrics.q_objective_evaluations <= 7 * target_count);
    }

    #[test]
    #[ignore = "bounded release-resource timing probe"]
    fn temporal_batch_profile_full_tile_release_probe() {
        let date_count = 12;
        let maximum_rank = 12;
        let target_count = 65_536;
        let days = (1..=date_count)
            .map(|date| date as f64 * 12.0)
            .collect::<Vec<_>>();
        let prepared = PreparedFactorObjective::new(&days, 12.0).unwrap();
        let one_factor = release_factor_fixture(date_count, maximum_rank, maximum_rank, false);
        let mut factors = vec![0.0; target_count * date_count * maximum_rank];
        for (target, factor) in factors
            .chunks_exact_mut(date_count * maximum_rank)
            .enumerate()
        {
            factor.copy_from_slice(&one_factor);
            factor[0] *= 1.0 + (target % 4093) as f64 * 1e-7;
            factor[date_count * maximum_rank - 1] += (target % 2039) as f64 * 1e-7;
        }
        let mut observations = vec![0.0; date_count * target_count];
        for date in 0..date_count {
            for target in 0..target_count {
                observations[date * target_count + target] = 0.013 * days[date]
                    + (date as f64 * 0.61 + target as f64 * 0.031).sin() * 0.8
                    + (date as f64 * 0.23 + target as f64 * 0.000_17).cos() * 0.15;
            }
        }
        let fingerprints = (0..target_count)
            .map(|target| {
                (
                    factors[target * date_count * maximum_rank].to_bits(),
                    observations[target].to_bits(),
                )
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert!(fingerprints.len() >= 257);
        let ranks = vec![maximum_rank; target_count];
        let options = TemporalCovarianceOptions::default();
        let started = std::time::Instant::now();
        let construction_started = std::time::Instant::now();
        let mut execution = crate::temporal_covariance_batch::TemporalBatchExecution::new(
            &prepared,
            &observations,
            &factors,
            maximum_rank,
            &ranks,
        )
        .unwrap();
        let construction_elapsed = construction_started.elapsed();
        let profile_started = std::time::Instant::now();
        let report = execution.profile_reml(&options, true, false).unwrap();
        let profile_elapsed = profile_started.elapsed();
        let evaluated = report
            .outcomes
            .iter()
            .filter(|outcome| outcome.is_ok())
            .count();
        let metrics = report.metrics.clone();
        let elapsed = started.elapsed();
        eprintln!(
            "full_optimizer_release_probe={{\"targets\":{target_count},\"distinct_fingerprints\":{},\"execution_construction_us\":{},\"profile_us\":{},\"elapsed_us\":{},\"evaluated\":{evaluated},\"microblocks_prepared\":{},\"rho_lane_evaluations\":{},\"q_objective_evaluations\":{},\"optimizer_fallbacks\":{},\"condition_fallbacks\":{},\"compactions\":{},\"maximum_worker_scratch_bytes\":{},\"rho_pass_histogram\":{:?}}}",
            fingerprints.len(),
            construction_elapsed.as_micros(),
            profile_elapsed.as_micros(),
            elapsed.as_micros(),
            metrics.microblocks_prepared,
            metrics.rho_lane_evaluations,
            metrics.q_objective_evaluations,
            metrics.exact_optimizer_fallback_targets,
            metrics.condition_exact_fallbacks,
            metrics.compaction_events,
            metrics.maximum_worker_scratch_bytes,
            metrics.primary_rho_pass_histogram,
        );
        assert_eq!(evaluated, target_count);
        assert!(
            elapsed.as_micros() <= 17_830,
            "full optimizer exceeded the frozen 12-date candidate ceiling: {} us",
            elapsed.as_micros()
        );
        assert_eq!(metrics.exact_optimizer_fallback_targets, 0);
    }

    #[test]
    fn temporal_batch_execution_reuses_prepared_soa_and_bounded_scratch() {
        let date_count = 12;
        let maximum_rank = 12;
        let target_count = 513;
        let options = TemporalCovarianceOptions::default();
        let days = (0..date_count)
            .map(|date| (date + 1) as f64 * 12.0)
            .collect::<Vec<_>>();
        let prepared = PreparedFactorObjective::new(&days, options.reference_lag_days).unwrap();
        let mut factors = vec![0.0; target_count * date_count * maximum_rank];
        let mut observations_soa = vec![0.0; date_count * target_count];
        for target in 0..target_count {
            let factor = release_factor_fixture(date_count, maximum_rank, maximum_rank, false);
            let offset = target * date_count * maximum_rank;
            factors[offset..offset + factor.len()].copy_from_slice(&factor);
            for date in 0..date_count {
                observations_soa[date * target_count + target] =
                    0.014 * days[date] + (date as f64 * 0.41 + target as f64 * 0.007).sin();
            }
        }
        let ranks = vec![maximum_rank; target_count];
        let rhos = vec![0.3; target_count];
        let log_process_variances = vec![0.5_f64.ln(); target_count];
        let mut execution = crate::temporal_covariance_batch::TemporalBatchExecution::new(
            &prepared,
            &observations_soa,
            &factors,
            maximum_rank,
            &ranks,
        )
        .unwrap();
        let before = execution.allocation_signature();
        let metrics = execution.metrics();
        assert_eq!(metrics.theta_independent_preparations, 1);
        assert!((1..=256).contains(&metrics.maximum_chunk_targets));
        assert!(metrics.maximum_worker_scratch_bytes <= 8 * 1024 * 1024);
        let first = execution
            .evaluate(&rhos, &log_process_variances, true)
            .unwrap()
            .to_vec();
        let middle = execution.allocation_signature();
        let second = execution
            .evaluate(&rhos, &log_process_variances, true)
            .unwrap()
            .to_vec();
        let after = execution.allocation_signature();
        assert_eq!(before, middle);
        assert_eq!(middle, after);
        assert_eq!(execution.metrics().objective_evaluations, 2);
        for target in 0..target_count {
            assert_factor_objective_bits(&first[target], &second[target]);
        }
    }

    #[test]
    #[ignore = "release-only temporal objective throughput probe"]
    fn simd_factor_batch_release_throughput() {
        let options = TemporalCovarianceOptions::default();
        let target_count = 256 * 256;
        for (date_count, maximum_rank) in [(12_usize, 12_usize), (48, 30), (96, 30)] {
            let days = (0..date_count)
                .map(|date| (date + 1) as f64 * 12.0)
                .collect::<Vec<_>>();
            let prepared = PreparedFactorObjective::new(&days, options.reference_lag_days).unwrap();
            let mut factors = vec![0.0; target_count * date_count * maximum_rank];
            let mut observations_soa = vec![0.0; date_count * target_count];
            let factor = release_factor_fixture(date_count, maximum_rank, maximum_rank, false);
            for target in 0..target_count {
                let offset = target * date_count * maximum_rank;
                factors[offset..offset + factor.len()].copy_from_slice(&factor);
                for date in 0..date_count {
                    observations_soa[date * target_count + target] = 0.014 * days[date]
                        + (date as f64 * 0.41 + target as f64 * 0.007).sin() * 0.6;
                }
            }
            let ranks = vec![maximum_rank; target_count];
            let rhos = vec![0.3; target_count];
            let log_process_variances = vec![0.5_f64.ln(); target_count];
            let mut execution = crate::temporal_covariance_batch::TemporalBatchExecution::new(
                &prepared,
                &observations_soa,
                &factors,
                maximum_rank,
                &ranks,
            )
            .unwrap();
            let metrics = execution.metrics();
            assert_eq!(metrics.theta_independent_preparations, 1);
            assert!(metrics.maximum_worker_scratch_bytes <= 8 * 1024 * 1024);
            for threads in [1_usize, 12] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                pool.install(|| {
                    let warm = execution
                        .evaluate(&rhos, &log_process_variances, true)
                        .unwrap();
                    assert!(warm.iter().all(|result| result
                        .as_ref()
                        .is_ok_and(|evaluation| !evaluation.dense_fallback_used)));
                });
                let started = std::time::Instant::now();
                pool.install(|| {
                    let outcomes = execution
                        .evaluate(&rhos, &log_process_variances, true)
                        .unwrap();
                    assert!(outcomes.iter().all(Result::is_ok));
                });
                let elapsed = started.elapsed().as_secs_f64();
                let targets_per_second = target_count as f64 / elapsed;
                eprintln!(
                    "simd_factor_objective dates={date_count} rank={maximum_rank} threads={threads} chunk_targets={} worker_scratch_bytes={} elapsed_seconds={elapsed:.6} targets_per_second={targets_per_second:.3}",
                    metrics.maximum_chunk_targets,
                    metrics.maximum_worker_scratch_bytes,
                );
            }
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn spectral_profile_matches_single_start_reml_and_factorization_cap() {
        let options = TemporalCovarianceOptions::default();
        let mut evaluated = 0_usize;
        for irregular in [false, true] {
            let date_count = 12;
            let maximum_rank = 12;
            let mut elapsed = 0.0;
            let days = (0..date_count)
                .map(|index| {
                    elapsed += if irregular {
                        [4.0, 7.0, 12.0, 19.0, 25.0, 31.0][index % 6]
                    } else {
                        12.0
                    };
                    elapsed
                })
                .collect::<Vec<_>>();
            let observations = days
                .iter()
                .enumerate()
                .map(|(index, day)| {
                    0.013 * day
                        + (index as f64 * 0.61).sin() * 0.8
                        + (index as f64 * 0.23).cos() * 0.15
                })
                .collect::<Vec<_>>();
            for realized_rank in [3, maximum_rank] {
                let factor = release_factor_fixture(date_count, maximum_rank, realized_rank, false);
                let covariance = (0..date_count)
                    .map(|left| {
                        (0..date_count)
                            .map(|right| {
                                (0..realized_rank)
                                    .map(|component| {
                                        factor[left * maximum_rank + component]
                                            * factor[right * maximum_rank + component]
                                    })
                                    .sum()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let bounds = nuisance_bounds(&days, &observations, &options).unwrap();
                let dense = optimize_covariance(
                    &days,
                    &observations,
                    &covariance,
                    &options,
                    bounds,
                    true,
                    false,
                );
                let prepared =
                    PreparedFactorObjective::new(&days, options.reference_lag_days).unwrap();
                let mut scratch = FactorObjectiveScratch::new(date_count, maximum_rank);
                let spectral = factor_native_profile_plugin(
                    &prepared,
                    &observations,
                    &factor,
                    maximum_rank,
                    realized_rank,
                    &options,
                    &mut scratch,
                    false,
                );
                match (dense, spectral) {
                    (Err(dense), Err(spectral)) => assert_eq!(dense, spectral),
                    (Ok(dense), Ok(spectral)) => {
                        let dense_score = covariance_objective(
                            &days,
                            &observations,
                            &covariance,
                            dense.rho,
                            dense.process_variance.ln(),
                            &options,
                            true,
                        )
                        .unwrap()
                        .0;
                        let score_tolerance = 1e-8 * (1.0 + dense_score.abs());
                        assert!(
                            spectral.score <= dense_score + score_tolerance,
                            "spectral objective {} is worse than dense {}",
                            spectral.score,
                            dense_score
                        );
                        if (spectral.score - dense_score).abs() <= score_tolerance {
                            let curvature_tolerance = (2.0 * score_tolerance
                                / spectral.profile_rho_curvature)
                                .sqrt()
                                .min(5e-3);
                            let rho_tolerance = 5e-4_f64.max(curvature_tolerance);
                            assert!(
                                (spectral.rho - dense.rho).abs() <= rho_tolerance,
                                "irregular={irregular} rank={realized_rank}: spectral rho={} score={}, dense rho={} score={dense_score}, curvature={}, tolerance={rho_tolerance}",
                                spectral.rho,
                                spectral.score,
                                dense.rho,
                                spectral.profile_rho_curvature,
                            );
                            assert!(
                                (spectral.process_variance.ln() - dense.process_variance.ln())
                                    .abs()
                                    <= 1.5e-2
                            );
                            assert_relative(spectral.slope, dense.slope, "spectral slope");
                            assert!(
                                (spectral.information_variance - dense.information_variance).abs()
                                    <= 1e-8
                                        * spectral
                                            .information_variance
                                            .abs()
                                            .max(dense.information_variance.abs())
                                            .max(1.0)
                            );
                        }
                        assert!(spectral.primary_factorization_count <= 20);
                        assert_eq!(spectral.dense_fallback_count, 0);
                        evaluated += 1;
                    }
                    (Err(TemporalInferenceStatus::OptimizerNonconverged), Ok(spectral)) => {
                        assert!(spectral.primary_factorization_count <= 20);
                        assert_eq!(spectral.dense_fallback_count, 0);
                        evaluated += 1;
                    }
                    (dense, spectral) => {
                        panic!("spectral/dense status mismatch: {dense:?} != {spectral:?}")
                    }
                }
            }
        }
        assert!(evaluated > 0);

        let days = (1..=12)
            .map(|index| index as f64 * 12.0)
            .collect::<Vec<_>>();
        let observations = days
            .iter()
            .enumerate()
            .map(|(index, day)| 0.01 * day + (index as f64 * 0.7).sin())
            .collect::<Vec<_>>();
        let factor = vec![0.0; days.len() * days.len()];
        let prepared = PreparedFactorObjective::new(&days, 12.0).unwrap();
        let mut scratch = FactorObjectiveScratch::new(days.len(), days.len());
        assert_eq!(
            factor_native_profile_plugin(
                &prepared,
                &observations,
                &factor,
                days.len(),
                days.len(),
                &options,
                &mut scratch,
                false,
            )
            .unwrap_err(),
            TemporalInferenceStatus::CovarianceNonfinite
        );
        let nonconverging = TemporalCovarianceOptions {
            optimizer_max_iterations: 0,
            ..options
        };
        let factor = release_factor_fixture(days.len(), days.len(), days.len(), false);
        assert_eq!(
            factor_native_profile_plugin(
                &prepared,
                &observations,
                &factor,
                days.len(),
                days.len(),
                &nonconverging,
                &mut scratch,
                false,
            )
            .unwrap_err(),
            TemporalInferenceStatus::OptimizerNonconverged
        );
    }

    #[test]
    fn hot_path_dedup_matches_legacy_output_bits_and_failures() {
        let days = [12.0, 24.0, 36.0];
        let observations = [0.7, 1.9, 2.4];
        let covariance = vec![
            vec![2.0, 0.2, 0.1],
            vec![0.2, 1.5, 0.3],
            vec![0.1, 0.3, 1.2],
        ];
        for compute_condition in [false, true] {
            let actual =
                gls_fit(&days, &observations, &covariance, 1e12, compute_condition).unwrap();
            let legacy =
                legacy_gls_fit(&days, &observations, &covariance, 1e12, compute_condition).unwrap();
            assert_gls_bits(&actual, &legacy);
        }

        for (matrix, condition_limit) in [
            (vec![vec![1.0, 2.0], vec![2.0, 1.0]], 1e12),
            (vec![vec![2.0, 0.0], vec![0.5, 2.0]], 1e12),
            (vec![vec![1e-12, 0.0], vec![0.0, 1.0]], 10.0),
        ] {
            assert_eq!(
                gls_fit(&[1.0, 2.0], &[0.5, 1.5], &matrix, condition_limit, true)
                    .err()
                    .unwrap(),
                legacy_gls_fit(&[1.0, 2.0], &[0.5, 1.5], &matrix, condition_limit, true,)
                    .err()
                    .unwrap()
            );
        }

        let difference_covariance = vec![
            vec![0.4, 0.03, 0.01],
            vec![0.03, 0.5, 0.02],
            vec![0.01, 0.02, 0.6],
        ];
        let options = TemporalCovarianceOptions::default();
        let reference = profile_fixed_objective(
            &days,
            &observations,
            &difference_covariance,
            0.08,
            0.35,
            0.7_f64.ln(),
            &options,
        )
        .unwrap();
        let prepared = PreparedExactProfile::new(
            &days,
            &observations,
            &difference_covariance,
            options.reference_lag_days,
        )
        .unwrap();
        let mut workspace = ExactFixedSlopeWorkspace::new(&prepared, 0.08);
        workspace.prepare_rho(0.35).unwrap();
        let actual = workspace.score(0.7_f64.ln()).unwrap();
        let legacy = legacy_profile_fixed_objective(
            &days,
            &observations,
            &difference_covariance,
            0.08,
            0.35,
            0.7_f64.ln(),
            &options,
        )
        .unwrap();
        assert_f64_bits(actual, reference.0);
        assert_f64_bits(reference.0, legacy.0);
        assert_plugin_bits(&reference.1, &legacy.1);
    }

    #[test]
    fn gls_and_fixed_profile_factorize_each_covariance_once() {
        let days = [12.0, 24.0, 36.0];
        let observations = [0.7, 1.9, 2.4];
        let covariance = vec![
            vec![2.0, 0.2, 0.1],
            vec![0.2, 1.5, 0.3],
            vec![0.1, 0.3, 1.2],
        ];
        CHOLESKY_CALLS.with(|calls| calls.set(0));
        gls_fit(&days, &observations, &covariance, 1e12, true).unwrap();
        CHOLESKY_CALLS.with(|calls| assert_eq!(calls.get(), 1));

        let difference_covariance = vec![
            vec![0.4, 0.03, 0.01],
            vec![0.03, 0.5, 0.02],
            vec![0.01, 0.02, 0.6],
        ];
        CHOLESKY_CALLS.with(|calls| calls.set(0));
        profile_fixed_objective(
            &days,
            &observations,
            &difference_covariance,
            0.08,
            0.35,
            0.7_f64.ln(),
            &TemporalCovarianceOptions::default(),
        )
        .unwrap();
        CHOLESKY_CALLS.with(|calls| assert_eq!(calls.get(), 1));
    }

    #[test]
    fn prepared_exact_profile_reuses_rho_and_solve_buffers() {
        let days = [12.0, 24.0, 48.0];
        let observations = [0.7, 1.9, 2.4];
        let difference_covariance = vec![
            vec![0.4, 0.03, 0.01],
            vec![0.03, 0.5, 0.02],
            vec![0.01, 0.02, 0.6],
        ];
        let prepared =
            PreparedExactProfile::new(&days, &observations, &difference_covariance, 12.0).unwrap();
        let mut workspace = ExactFixedSlopeWorkspace::new(&prepared, 0.08);
        workspace.prepare_rho(0.35).unwrap();
        let correlation = workspace.correlation.clone();
        let pointers = (
            workspace.correlation.as_ptr(),
            workspace.covariance.as_ptr(),
            workspace.lower.as_ptr(),
            workspace.forward.as_ptr(),
            workspace.solution.as_ptr(),
        );
        CHOLESKY_CALLS.with(|calls| calls.set(0));
        for process_variance in [0.2_f64, 0.7, 1.4] {
            workspace.score(process_variance.ln()).unwrap();
        }
        CHOLESKY_CALLS.with(|calls| assert_eq!(calls.get(), 3));
        assert_eq!(workspace.correlation, correlation);
        assert_eq!(
            pointers,
            (
                workspace.correlation.as_ptr(),
                workspace.covariance.as_ptr(),
                workspace.lower.as_ptr(),
                workspace.forward.as_ptr(),
                workspace.solution.as_ptr(),
            )
        );
    }

    #[test]
    fn spectral_fixed_slope_score_matches_dense_oracle_for_release_shapes() {
        let options = TemporalCovarianceOptions::default();
        for date_count in [12_usize, 48, 96] {
            let maximum_rank = date_count;
            let mut elapsed = 0.0;
            let days = (0..date_count)
                .map(|index| {
                    elapsed += [6.0, 18.0, 12.0, 12.0][index % 4];
                    elapsed
                })
                .collect::<Vec<_>>();
            let observations = days
                .iter()
                .enumerate()
                .map(|(index, day)| {
                    0.013 * day
                        + (index as f64 * 0.61).sin() * 0.8
                        + (index as f64 * 0.23).cos() * 0.15
                })
                .collect::<Vec<_>>();
            for realized_rank in [3, maximum_rank] {
                let factor = release_factor_fixture(date_count, maximum_rank, realized_rank, false);
                let covariance = difference_covariance_from_factor(
                    date_count,
                    &factor,
                    maximum_rank,
                    realized_rank,
                );
                let prepared =
                    PreparedFactorObjective::new(&days, options.reference_lag_days).unwrap();
                let mut scratch = FactorObjectiveScratch::new(date_count, realized_rank);
                let scale = prepare_spectral_target(
                    &prepared,
                    &observations,
                    &factor,
                    maximum_rank,
                    realized_rank,
                    &mut scratch,
                )
                .unwrap();
                for rho in [0.05, 0.35, 0.85] {
                    let projection =
                        spectral_projection(&prepared, rho, realized_rank, scale, &mut scratch)
                            .unwrap();
                    for process_variance in [0.1_f64, 0.7, 4.0] {
                        for slope in [0.009_f64, 0.013, 0.018] {
                            let spectral = spectral_fixed_slope_q_score(
                                &projection,
                                process_variance.ln(),
                                slope,
                            )
                            .unwrap();
                            let dense = profile_fixed_objective(
                                &days,
                                &observations,
                                &covariance,
                                slope,
                                rho,
                                process_variance.ln(),
                                &options,
                            )
                            .unwrap()
                            .0;
                            assert_relative(spectral, dense, "fixed-slope ML score");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn factor_fixed_slope_profile_uses_one_dense_validation() {
        let options = TemporalCovarianceOptions::default();
        for date_count in [12_usize, 48, 96] {
            let maximum_rank = date_count;
            let days = (0..date_count)
                .map(|index| (index + 1) as f64 * 12.0)
                .collect::<Vec<_>>();
            let observations = days
                .iter()
                .enumerate()
                .map(|(index, day)| 0.013 * day + (index as f64 * 0.61).sin() * 0.8)
                .collect::<Vec<_>>();
            let factor = release_factor_fixture(date_count, maximum_rank, maximum_rank, false);
            let covariance =
                difference_covariance_from_factor(date_count, &factor, maximum_rank, maximum_rank);
            let prepared = PreparedFactorObjective::new(&days, options.reference_lag_days).unwrap();
            let dense_prepared = PreparedExactProfile::new(
                &days,
                &observations,
                &covariance,
                options.reference_lag_days,
            )
            .unwrap();
            let problem = FactorProfileProblem {
                prepared: &prepared,
                observations: &observations,
                factor: &factor,
                maximum_rank,
                realized_rank: maximum_rank,
                dense_prepared: &dense_prepared,
            };
            let bounds = nuisance_bounds(&days, &observations, &options).unwrap();
            CHOLESKY_CALLS.with(|calls| calls.set(0));
            profile_fixed_slope_factor(&problem, 0.013, bounds, &options).unwrap();
            CHOLESKY_CALLS.with(|calls| assert_eq!(calls.get(), 1, "date_count={date_count}"));
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    fn factor_prefit_skips_redundant_dense_reml_and_adjustment() {
        let days = (0..13).map(|index| index as f64 * 12.0).collect::<Vec<_>>();
        let mut observations = days
            .iter()
            .enumerate()
            .map(|(index, day)| 0.01 * day + (index as f64 * 0.7).sin() * 2.0)
            .collect::<Vec<_>>();
        observations[0] = 0.0;
        let mut covariance = vec![vec![0.0; days.len()]; days.len()];
        for (index, row) in covariance.iter_mut().enumerate().skip(1) {
            row[index] = 1.0;
        }
        let options = TemporalCovarianceOptions {
            bootstrap_replicates: 0,
            bootstrap_minimum_successes: 0,
            ..TemporalCovarianceOptions::default()
        };
        let full = fit_temporal_covariance(&days, &observations, &covariance, &options);
        let prefit = TemporalCovariancePrefit {
            plugin_slope_per_day: full.plugin_gls_slope.unwrap() / DAYS_PER_YEAR,
            plugin_gls: full.plugin_gls.clone(),
            adjusted_scalar: full.adjusted_scalar.clone(),
            fitted_rho: full.fitted_rho.unwrap(),
            fitted_process_variance: full.fitted_process_variance.unwrap(),
            fitted_parameter_active_set: full.fitted_parameter_active_set,
            covariance_condition_number: full.covariance_condition_number,
        };
        DENSE_PROFILE_PLUGIN_CALLS.with(|calls| calls.set(0));
        DENSE_ADJUSTED_SCALAR_CALLS.with(|calls| calls.set(0));
        let reused = fit_temporal_covariance_from_prefit(
            &days,
            &observations,
            &covariance,
            &options,
            &prefit,
        );
        DENSE_PROFILE_PLUGIN_CALLS.with(|calls| assert_eq!(calls.get(), 0));
        DENSE_ADJUSTED_SCALAR_CALLS.with(|calls| assert_eq!(calls.get(), 0));
        assert_eq!(reused.plugin_gls, full.plugin_gls);
        assert_eq!(reused.adjusted_scalar, full.adjusted_scalar);
        assert_eq!(reused.adjusted_profile, full.adjusted_profile);
        assert_eq!(reused.ols, full.ols);
        assert_eq!(reused.oracle_gls, full.oracle_gls);
        assert_eq!(reused.conditional_wls, full.conditional_wls);
        assert_eq!(reused.scalar_effective_n, full.scalar_effective_n);
        assert_eq!(reused.fitted_rho, full.fitted_rho);
        assert_eq!(reused.fitted_process_variance, full.fitted_process_variance);
        let mut persisted_factor = vec![0.0; days.len() * (days.len() - 1)];
        for date in 1..days.len() {
            persisted_factor[date * (days.len() - 1) + date - 1] = 1.0;
        }
        let factor_reused = fit_temporal_covariance_from_factor_prefit(
            &days,
            &observations,
            &covariance,
            &persisted_factor,
            days.len() - 1,
            days.len() - 1,
            &options,
            &prefit,
        );
        assert_eq!(
            factor_reused.adjusted_profile.status,
            full.adjusted_profile.status
        );
        assert_eq!(
            factor_reused.adjusted_profile.point_estimate,
            full.adjusted_profile.point_estimate
        );
        for (factor_interval, dense_interval) in [
            (
                factor_reused.adjusted_profile.interval_68,
                full.adjusted_profile.interval_68,
            ),
            (
                factor_reused.adjusted_profile.interval_90,
                full.adjusted_profile.interval_90,
            ),
            (
                factor_reused.adjusted_profile.interval_95,
                full.adjusted_profile.interval_95,
            ),
        ] {
            let factor_interval = factor_interval.unwrap();
            let dense_interval = dense_interval.unwrap();
            for (factor_endpoint, dense_endpoint) in [
                (factor_interval.lower, dense_interval.lower),
                (factor_interval.upper, dense_interval.upper),
            ] {
                let tolerance = 2.0
                    * options.optimizer_tolerance
                    * (1.0 + (dense_endpoint / DAYS_PER_YEAR).abs())
                    * DAYS_PER_YEAR;
                assert!((factor_endpoint - dense_endpoint).abs() <= tolerance);
            }
        }
        let mut mismatched_factor = persisted_factor.clone();
        mismatched_factor[days.len() - 1] *= 1.1;
        let mut mismatched_covariance = covariance.clone();
        mismatched_covariance[1][1] *= 1.21;
        let mismatched_full =
            fit_temporal_covariance(&days, &observations, &mismatched_covariance, &options);
        let mismatched_prefit = TemporalCovariancePrefit {
            plugin_slope_per_day: mismatched_full.plugin_gls_slope.unwrap() / DAYS_PER_YEAR,
            plugin_gls: mismatched_full.plugin_gls,
            adjusted_scalar: mismatched_full.adjusted_scalar,
            fitted_rho: mismatched_full.fitted_rho.unwrap(),
            fitted_process_variance: mismatched_full.fitted_process_variance.unwrap(),
            fitted_parameter_active_set: mismatched_full.fitted_parameter_active_set,
            covariance_condition_number: mismatched_full.covariance_condition_number,
        };
        let mismatch = fit_temporal_covariance_from_factor_prefit(
            &days,
            &observations,
            &covariance,
            &mismatched_factor,
            days.len() - 1,
            days.len() - 1,
            &options,
            &mismatched_prefit,
        );
        assert_eq!(
            mismatch.status,
            TemporalInferenceStatus::CovarianceNonfinite
        );
        assert_eq!(
            mismatch.plugin_gls.status,
            TemporalInferenceStatus::CovarianceNonfinite
        );
        assert_eq!(mismatch.plugin_gls.point_estimate, None);
        let (selected_days, selected_observations, _) =
            subset_origin_anchored_covariance(&days, &observations, &covariance).unwrap();
        let residuals = selected_observations
            .iter()
            .zip(&selected_days)
            .map(|(value, day)| value - prefit.plugin_slope_per_day * day)
            .collect::<Vec<_>>();
        assert_eq!(
            reused.raw_correlation,
            raw_adjacent_correlation(&selected_days, &residuals)
        );

        let mut invalid_prefit = prefit.clone();
        invalid_prefit.fitted_process_variance = 0.0;
        assert_eq!(
            fit_temporal_covariance_from_prefit(
                &days,
                &observations,
                &covariance,
                &options,
                &invalid_prefit,
            )
            .status,
            TemporalInferenceStatus::CovarianceParameterAtBoundary
        );
        let mut invalid_prefit = prefit.clone();
        invalid_prefit.fitted_rho = options.rho_max;
        assert_eq!(
            fit_temporal_covariance_from_prefit(
                &days,
                &observations,
                &covariance,
                &options,
                &invalid_prefit,
            )
            .status,
            TemporalInferenceStatus::CovarianceParameterAtBoundary
        );
        let mut invalid_prefit = prefit.clone();
        invalid_prefit.plugin_gls.point_estimate = invalid_prefit
            .plugin_gls
            .point_estimate
            .map(|point| point + 1.0);
        assert_eq!(
            fit_temporal_covariance_from_prefit(
                &days,
                &observations,
                &covariance,
                &options,
                &invalid_prefit,
            )
            .status,
            TemporalInferenceStatus::CovarianceNonfinite
        );
        let mut invalid_prefit = prefit.clone();
        invalid_prefit.covariance_condition_number = None;
        assert_eq!(
            fit_temporal_covariance_from_prefit(
                &days,
                &observations,
                &covariance,
                &options,
                &invalid_prefit,
            )
            .status,
            TemporalInferenceStatus::CovarianceNonfinite
        );
        let mut invalid_prefit = prefit.clone();
        invalid_prefit.covariance_condition_number = Some(options.condition_limit * 2.0);
        assert_eq!(
            fit_temporal_covariance_from_prefit(
                &days,
                &observations,
                &covariance,
                &options,
                &invalid_prefit,
            )
            .status,
            TemporalInferenceStatus::DesignIllConditioned
        );
        let mut invalid_prefit = prefit.clone();
        if let Some(interval) = invalid_prefit.plugin_gls.interval_68.as_mut() {
            interval.lower += 1.0;
            interval.upper += 1.0;
        }
        assert_eq!(
            fit_temporal_covariance_from_prefit(
                &days,
                &observations,
                &covariance,
                &options,
                &invalid_prefit,
            )
            .status,
            TemporalInferenceStatus::CovarianceNonfinite
        );
        let mut invalid_prefit = prefit;
        invalid_prefit.fitted_parameter_active_set =
            if invalid_prefit.fitted_parameter_active_set.is_some() {
                None
            } else {
                Some(TemporalInferenceStatus::DiagnosticNotComputed)
            };
        assert_eq!(
            fit_temporal_covariance_from_prefit(
                &days,
                &observations,
                &covariance,
                &options,
                &invalid_prefit,
            )
            .status,
            TemporalInferenceStatus::CovarianceNonfinite
        );
    }

    #[test]
    fn compact_factor_prefit_matches_dense_profile_with_missing_observation() {
        let days = (0..14).map(|index| index as f64 * 12.0).collect::<Vec<_>>();
        let mut observations = days
            .iter()
            .enumerate()
            .map(|(index, day)| 0.01 * day + (index as f64 * 0.7).sin() * 2.0)
            .collect::<Vec<_>>();
        observations[0] = 0.0;
        observations[4] = f64::NAN;
        let mut covariance = vec![vec![0.0; days.len()]; days.len()];
        for (index, row) in covariance.iter_mut().enumerate().skip(1) {
            row[index] = 1.0;
        }
        let options = TemporalCovarianceOptions {
            bootstrap_replicates: 0,
            bootstrap_minimum_successes: 0,
            ..TemporalCovarianceOptions::default()
        };
        let full = fit_temporal_covariance(&days, &observations, &covariance, &options);
        let prefit = TemporalCovariancePrefit {
            plugin_slope_per_day: full.plugin_gls_slope.unwrap() / DAYS_PER_YEAR,
            plugin_gls: full.plugin_gls.clone(),
            adjusted_scalar: full.adjusted_scalar.clone(),
            fitted_rho: full.fitted_rho.unwrap(),
            fitted_process_variance: full.fitted_process_variance.unwrap(),
            fitted_parameter_active_set: full.fitted_parameter_active_set,
            covariance_condition_number: full.covariance_condition_number,
        };
        let retained_count = days.len() - 2;
        let mut compact_persisted_factor = vec![0.0; (retained_count + 1) * retained_count];
        for retained_date in 0..retained_count {
            compact_persisted_factor[(retained_date + 1) * retained_count + retained_date] = 1.0;
        }
        let factor_reused = fit_temporal_covariance_from_factor_prefit(
            &days,
            &observations,
            &covariance,
            &compact_persisted_factor,
            retained_count,
            retained_count,
            &options,
            &prefit,
        );
        assert_eq!(factor_reused.valid_date_count, retained_count);
        assert_eq!(factor_reused.plugin_gls, full.plugin_gls);
        assert_eq!(factor_reused.adjusted_scalar, full.adjusted_scalar);
        assert_eq!(
            factor_reused.adjusted_profile.status,
            full.adjusted_profile.status
        );
        assert_eq!(
            factor_reused.adjusted_profile.point_estimate,
            full.adjusted_profile.point_estimate
        );
        for (factor_interval, dense_interval) in [
            (
                factor_reused.adjusted_profile.interval_68,
                full.adjusted_profile.interval_68,
            ),
            (
                factor_reused.adjusted_profile.interval_90,
                full.adjusted_profile.interval_90,
            ),
            (
                factor_reused.adjusted_profile.interval_95,
                full.adjusted_profile.interval_95,
            ),
        ] {
            let factor_interval = factor_interval.unwrap();
            let dense_interval = dense_interval.unwrap();
            for (factor_endpoint, dense_endpoint) in [
                (factor_interval.lower, dense_interval.lower),
                (factor_interval.upper, dense_interval.upper),
            ] {
                let tolerance = 2.0
                    * options.optimizer_tolerance
                    * (1.0 + (dense_endpoint / DAYS_PER_YEAR).abs())
                    * DAYS_PER_YEAR;
                assert!((factor_endpoint - dense_endpoint).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn scalar_candidate_probe_matches_full_fit_comparators_without_bootstrap() {
        let days = (0..13).map(|index| index as f64 * 12.0).collect::<Vec<_>>();
        let mut observations = days
            .iter()
            .enumerate()
            .map(|(index, day)| 0.01 * day + (index as f64 * 0.7).sin() * 2.0)
            .collect::<Vec<_>>();
        observations[0] = 0.0;
        let mut covariance = vec![vec![0.0; days.len()]; days.len()];
        for (index, row) in covariance.iter_mut().enumerate().skip(1) {
            row[index] = 1.0;
        }
        let options = TemporalCovarianceOptions {
            bootstrap_replicates: 0,
            bootstrap_minimum_successes: 0,
            ..TemporalCovarianceOptions::default()
        };
        let full = fit_temporal_covariance(&days, &observations, &covariance, &options);
        for (method, expected) in [
            (
                TemporalScalarCandidateMethod::PluginGlsReml,
                &full.plugin_gls,
            ),
            (
                TemporalScalarCandidateMethod::RemlCovarianceParameterAdjustedScalar,
                &full.adjusted_scalar,
            ),
            (
                TemporalScalarCandidateMethod::SlopeProfileLikelihoodMl,
                &full.adjusted_profile,
            ),
        ] {
            let probe = probe_temporal_scalar_candidate(
                &days,
                &observations,
                &covariance,
                &options,
                method,
            );
            assert_eq!(probe.comparator, *expected);
            assert_eq!(probe.valid_date_count, full.valid_date_count);
            assert_eq!(probe.rank, full.rank);
            assert_eq!(probe.degrees_of_freedom, full.degrees_of_freedom);
            assert_eq!(probe.bootstrap_attempts, 0);
        }
    }
}

//! Validation-only temporal covariance and slope inference kernel for issue #53.
//!
//! The kernel operates on an already spatially differenced, origin-anchored
//! series and a direct #54 difference covariance.  It is deliberately separate
//! from workflow output code: no raster writer or corrected uncertainty product
//! is exposed here.  The public result contains point estimates, diagnostics,
//! and validation intervals, but no corrected inferential standard error.

use faer::{Mat, Side};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal, StudentsT};

const DAYS_PER_YEAR: f64 = 365.25;
const SYMMETRY_TOLERANCE: f64 = 1e-10;

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
    let tolerance = tolerance_fraction.clamp(1e-8, 1e-4);
    let rho_tolerance = tolerance * (rho_bounds[1] - rho_bounds[0]);
    let log_min = process_variance_bounds[0].ln();
    let log_max = process_variance_bounds[1].ln();
    let log_tolerance = tolerance * (log_max - log_min);
    if rho <= rho_bounds[0] + rho_tolerance {
        Some(TemporalInferenceStatus::RhoLowerBoundary)
    } else if rho >= rho_bounds[1] - rho_tolerance {
        Some(TemporalInferenceStatus::RhoUpperBoundary)
    } else if process_variance.ln() <= log_min + log_tolerance {
        Some(TemporalInferenceStatus::ProcessVarianceLowerBoundary)
    } else if process_variance.ln() >= log_max - log_tolerance {
        Some(TemporalInferenceStatus::ProcessVarianceUpperBoundary)
    } else {
        None
    }
}

/// Fit the OLS/oracle/plugin/profile/bootstrap comparator set.
///
/// `observations[0]` is the exact acquisition-zero gauge and must be finite;
/// missing post-gauge dates are represented by `NaN`. `difference_covariance`
/// is consumed directly and must already be a same-frame #54 difference factor.
#[allow(clippy::too_many_lines)]
pub fn fit_temporal_covariance(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
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
    let plugin = match profile_plugin(&selected_days, &selected_y, &selected_c, options) {
        Ok(value) => value,
        Err(status) => return empty(status),
    };
    let bootstrap = bootstrap_refit(&selected_days, &selected_c, &plugin, options);
    let residuals: Vec<f64> = selected_y
        .iter()
        .zip(&selected_days)
        .map(|(value, day)| value - plugin.slope * day)
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
    let plugin_fit = gls_fit(
        &selected_days,
        &selected_y,
        &plugin.covariance,
        options.condition_limit,
        true,
    );
    let plugin_comparator = plugin_fit.map_or_else(empty_comparator, |fit| {
        normal_comparator(
            fit.slope,
            fit.information_variance.sqrt(),
            TemporalInferenceStatus::Evaluated,
        )
    });
    let adjusted_profile_result =
        profile_comparator(&selected_days, &selected_y, &selected_c, &plugin, options);
    let profile_status = adjusted_profile_result.as_ref().err().copied();
    let adjusted_profile = adjusted_profile_result.unwrap_or_else(empty_comparator);
    let adjusted_scalar = reml_adjusted_scalar_comparator(
        &selected_days,
        &selected_y,
        &selected_c,
        &plugin,
        degrees_of_freedom,
        options,
    )
    .unwrap_or_else(empty_comparator);
    let bootstrap_comparator = bootstrap_comparator(&bootstrap, plugin.slope);
    let minimum_bootstrap_successes = required_bootstrap_successes(options.bootstrap_replicates)
        .max(options.bootstrap_minimum_successes);
    let status = if let Some(status) = profile_status {
        status
    } else if bootstrap.successes < minimum_bootstrap_successes {
        TemporalInferenceStatus::BootstrapInsufficientSuccess
    } else {
        TemporalInferenceStatus::Evaluated
    };
    TemporalCovarianceFit {
        status,
        ols_slope: ols.point_estimate,
        oracle_gls_slope: oracle.point_estimate,
        plugin_gls_slope: Some(plugin.slope * DAYS_PER_YEAR),
        adjusted_profile_slope: adjusted_profile.point_estimate,
        bootstrap_slope: bootstrap_comparator.point_estimate,
        bootstrap_interval: bootstrap_comparator.interval_95,
        fitted_rho: Some(plugin.rho),
        fitted_process_variance: Some(plugin.process_variance),
        raw_correlation,
        valid_date_count: n,
        rank,
        degrees_of_freedom,
        covariance_condition_number: Some(plugin.condition_number),
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
        schema: "dolphinrust-temporal-covariance-provenance/1".to_owned(),
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

struct PluginFit {
    slope: f64,
    rho: f64,
    process_variance: f64,
    covariance: Vec<Vec<f64>>,
    condition_number: f64,
    information_variance: f64,
}

#[derive(Clone, Copy)]
struct NuisanceBounds {
    rho_lower: f64,
    rho_upper: f64,
    log_variance_lower: f64,
    log_variance_upper: f64,
    initial_log_variance: f64,
}

fn empty_comparator(status: TemporalInferenceStatus) -> ComparatorDiagnostics {
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

fn normal_comparator(
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

fn reml_adjusted_scalar_comparator(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    plugin: &PluginFit,
    residual_dof: usize,
    options: &TemporalCovarianceOptions,
) -> Result<ComparatorDiagnostics, TemporalInferenceStatus> {
    let bounds = nuisance_bounds(days, observations, options)?;
    let rho_step = 1e-3_f64
        .min((plugin.rho - bounds.rho_lower) / 2.0)
        .min((bounds.rho_upper - plugin.rho) / 2.0);
    let log_step = 1e-2_f64
        .min((plugin.process_variance.ln() - bounds.log_variance_lower) / 2.0)
        .min((bounds.log_variance_upper - plugin.process_variance.ln()) / 2.0);
    if rho_step <= 0.0 || log_step <= 0.0 {
        return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
    }
    let theta = [plugin.rho, plugin.process_variance.ln()];
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
    let covariance_scale = 2.0;
    let nuisance_variance = covariance_scale
        * (h11 * slope_gradient[0].powi(2) - 2.0 * h01 * slope_gradient[0] * slope_gradient[1]
            + h00 * slope_gradient[1].powi(2))
        / determinant;
    let adjusted_variance = plugin.information_variance + nuisance_variance.max(0.0);
    if !adjusted_variance.is_finite() || adjusted_variance <= 0.0 || residual_dof == 0 {
        return Err(TemporalInferenceStatus::WeakParameterIdentification);
    }
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

fn profile_comparator(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    _plugin: &PluginFit,
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
    let levels = [0.68, 0.90, 0.95];
    let mut intervals = Vec::with_capacity(levels.len());
    for level in levels {
        let chi_square = ChiSquared::new(1.0)
            .map_err(|_| TemporalInferenceStatus::WeakParameterIdentification)?
            .inverse_cdf(level);
        let target = unrestricted_objective + chi_square;
        let scale = ml_fit.information_variance.sqrt().max(1e-10);
        let (lower, upper) = profile_endpoint_pair(
            days,
            observations,
            difference_covariance,
            ml_fit.slope,
            scale,
            target,
            bounds,
            options,
        )?;
        intervals.push(ValidationInterval {
            lower: lower * DAYS_PER_YEAR,
            upper: upper * DAYS_PER_YEAR,
            successful_replicates: 0,
        });
    }
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
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    point: f64,
    initial_scale: f64,
    target: f64,
    bounds: NuisanceBounds,
    options: &TemporalCovarianceOptions,
) -> Result<(f64, f64), TemporalInferenceStatus> {
    let objective = |slope: f64| {
        profile_fixed_slope(
            days,
            observations,
            difference_covariance,
            slope,
            bounds,
            options,
        )
        .map(|(value, _)| value)
    };
    let mut lower = point - initial_scale;
    let mut upper = point + initial_scale;
    for _ in 0..options.profile_max_expansions {
        if objective(lower)? > target && objective(upper)? > target {
            break;
        }
        let span = (upper - lower) * 2.0;
        lower = point - span;
        upper = point + span;
    }
    if objective(lower)? <= target || objective(upper)? <= target {
        return Err(TemporalInferenceStatus::OptimizerNonconverged);
    }
    let lower = solve_profile_endpoint(&objective, point, lower, target, options)?;
    let upper = solve_profile_endpoint(&objective, point, upper, target, options)?;
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
        status: if summary.successes >= summary.minimum_successes {
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

fn interval(
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
    let bounds = nuisance_bounds(days, observations, options)?;
    optimize_covariance(
        days,
        observations,
        difference_covariance,
        options,
        bounds,
        true,
    )
}

fn nuisance_bounds(
    days: &[f64],
    observations: &[f64],
    options: &TemporalCovarianceOptions,
) -> Result<NuisanceBounds, TemporalInferenceStatus> {
    if !(0.0..1.0).contains(&options.rho_max) || options.rho_min < 0.0 {
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
    let rho_upper = (options.rho_max - 1e-8).max(options.rho_min + 1e-8);
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
) -> Result<PluginFit, TemporalInferenceStatus> {
    let mut best: Option<(f64, PluginFit)> = None;
    let mut any_converged = false;
    for initial_rho in [
        bounds.rho_lower,
        (bounds.rho_lower + bounds.rho_upper) / 2.0,
        bounds.rho_upper,
    ] {
        let mut rho = initial_rho;
        let mut log_variance = bounds.initial_log_variance;
        let mut previous_score = f64::INFINITY;
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
            let score = covariance_objective(
                days,
                observations,
                difference_covariance,
                rho,
                log_variance,
                options,
                restricted,
            )
            .map_or(f64::INFINITY, |value| value.0);
            if score.is_finite()
                && (previous_score - score).abs()
                    <= options.optimizer_tolerance * (1.0 + score.abs())
            {
                any_converged = true;
                break;
            }
            previous_score = score;
        }
        if let Ok(candidate) = covariance_objective(
            days,
            observations,
            difference_covariance,
            rho,
            log_variance,
            options,
            restricted,
        ) {
            if best.as_ref().is_none_or(|(score, _)| candidate.0 < *score) {
                best = Some(candidate);
            }
        }
    }
    if !any_converged {
        return Err(TemporalInferenceStatus::OptimizerNonconverged);
    }
    let (_, candidate) = best.ok_or(TemporalInferenceStatus::CovarianceParameterAtBoundary)?;
    let (_, mut fit) = covariance_objective(
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
    validate_profile_curvature(
        days,
        observations,
        difference_covariance,
        &fit,
        bounds,
        restricted,
        options,
    )?;
    Ok(fit)
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

fn validate_profile_curvature(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    fit: &PluginFit,
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
    let base = covariance_objective(
        days,
        observations,
        difference_covariance,
        fit.rho,
        fit.process_variance.ln(),
        options,
        restricted,
    )?
    .0;
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
    (lower + upper) / 2.0
}

fn profile_fixed_slope(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    slope: f64,
    bounds: NuisanceBounds,
    options: &TemporalCovarianceOptions,
) -> Result<(f64, PluginFit), TemporalInferenceStatus> {
    if options.optimizer_max_iterations == 0 || options.optimizer_tolerance <= 0.0 {
        return Err(TemporalInferenceStatus::OptimizerNonconverged);
    }
    let mut best: Option<(f64, PluginFit)> = None;
    let mut rho = (bounds.rho_lower + bounds.rho_upper) / 2.0;
    let mut log_variance = bounds.initial_log_variance;
    let mut converged = false;
    let mut previous_score = f64::INFINITY;
    for _ in 0..options.optimizer_max_iterations {
        rho = golden_section_minimum(bounds.rho_lower, bounds.rho_upper, |candidate| {
            profile_fixed_objective(
                days,
                observations,
                difference_covariance,
                slope,
                candidate,
                log_variance,
                options,
            )
            .map_or(f64::INFINITY, |(score, _)| score)
        });
        log_variance = golden_section_minimum(
            bounds.log_variance_lower,
            bounds.log_variance_upper,
            |candidate| {
                profile_fixed_objective(
                    days,
                    observations,
                    difference_covariance,
                    slope,
                    rho,
                    candidate,
                    options,
                )
                .map_or(f64::INFINITY, |(score, _)| score)
            },
        );
        let score = profile_fixed_objective(
            days,
            observations,
            difference_covariance,
            slope,
            rho,
            log_variance,
            options,
        )
        .map_or(f64::INFINITY, |value| value.0);
        if score.is_finite()
            && (previous_score - score).abs() <= options.optimizer_tolerance * (1.0 + score.abs())
        {
            converged = true;
            break;
        }
        previous_score = score;
    }
    if !converged {
        return Err(TemporalInferenceStatus::OptimizerNonconverged);
    }
    if let Ok(candidate) = profile_fixed_objective(
        days,
        observations,
        difference_covariance,
        slope,
        rho,
        log_variance,
        options,
    ) {
        best = Some(candidate);
    }
    let (_, candidate) = best.ok_or(TemporalInferenceStatus::OptimizerNonconverged)?;
    let candidate = profile_fixed_objective(
        days,
        observations,
        difference_covariance,
        slope,
        candidate.rho,
        candidate.process_variance.ln(),
        options,
    )?;
    if let Some(status) = temporal_parameter_boundary_status(
        candidate.1.rho,
        candidate.1.process_variance,
        [bounds.rho_lower, bounds.rho_upper],
        [
            bounds.log_variance_lower.exp(),
            bounds.log_variance_upper.exp(),
        ],
        options.optimizer_tolerance * 0.01,
    ) {
        return Err(status);
    }
    Ok(candidate)
}

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
    let fit = gls_fit(
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
    let inverse = invert_positive_definite(&covariance, options.condition_limit, false)?;
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

struct GlsFit {
    slope: f64,
    quadratic_form: f64,
    log_determinant: f64,
    information_variance: f64,
    condition_number: f64,
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
    let inverse = invert_positive_definite(covariance, condition_limit, compute_condition_number)?;
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
    let mut slopes = Vec::with_capacity(options.bootstrap_replicates);
    for replicate in 0..options.bootstrap_replicates {
        let mut state = splitmix64(options.bootstrap_seed ^ replicate as u64);
        let normal = (0..days.len())
            .map(|_| standard_normal(&mut state))
            .collect::<Vec<_>>();
        let residual = lower_mat_vec(&cholesky, &normal);
        let simulated: Vec<f64> = days
            .iter()
            .zip(residual)
            .map(|(day, noise)| plugin.slope * day + noise)
            .collect();
        if let Ok(refit) = profile_plugin(
            days,
            &simulated,
            difference_covariance,
            &TemporalCovarianceOptions {
                bootstrap_replicates: 0,
                bootstrap_minimum_successes: 0,
                bootstrap_seed: state,
                ..options.clone()
            },
        ) {
            slopes.push(refit.slope);
        }
    }
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

fn invert_positive_definite(
    matrix: &[Vec<f64>],
    condition_limit: f64,
    check_condition: bool,
) -> Result<Vec<Vec<f64>>, TemporalInferenceStatus> {
    validate_square_covariance(matrix)?;
    let Some(cholesky) = cholesky(matrix) else {
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
        let solution = solve_cholesky(&cholesky, &unit);
        for row in 0..n {
            inverse[row][column] = solution[row];
        }
    }
    Ok(inverse)
}

fn cholesky(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut lower = vec![vec![0.0; n]; n];
    for row in 0..n {
        for column in 0..=row {
            let sum = (0..column)
                .map(|index| lower[row][index] * lower[column][index])
                .sum::<f64>();
            if row == column {
                let diagonal = matrix[row][row] - sum;
                if !diagonal.is_finite() || diagonal <= 0.0 {
                    return None;
                }
                lower[row][column] = diagonal.sqrt();
            } else if lower[column][column] > 0.0 {
                lower[row][column] = (matrix[row][column] - sum) / lower[column][column];
            } else {
                return None;
            }
        }
    }
    Some(lower)
}

fn solve_cholesky(lower: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    let n = rhs.len();
    let mut forward = vec![0.0; n];
    for row in 0..n {
        forward[row] = (rhs[row]
            - (0..row)
                .map(|column| lower[row][column] * forward[column])
                .sum::<f64>())
            / lower[row][row];
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        solution[row] = (forward[row]
            - ((row + 1)..n)
                .map(|column| lower[column][row] * solution[column])
                .sum::<f64>())
            / lower[row][row];
    }
    solution
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

fn lower_mat_vec(lower: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
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

fn standard_normal(state: &mut u64) -> f64 {
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

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

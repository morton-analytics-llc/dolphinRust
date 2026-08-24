//! Validation-only temporal covariance and slope inference kernel for issue #53.
//!
//! The kernel operates on an already spatially differenced, origin-anchored
//! series and a direct #54 difference covariance.  It is deliberately separate
//! from workflow output code: no raster writer or corrected uncertainty product
//! is exposed here.  The public result contains point estimates, diagnostics,
//! and validation intervals, but no corrected inferential standard error.

use serde::{Deserialize, Serialize};

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
    /// Too few complete-refit bootstrap replicates succeeded.
    BootstrapInsufficientSuccess,
    /// The requested cadence is outside the preregistered supported predicate.
    UnsupportedCadence,
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
            bootstrap_minimum_successes: 180,
            bootstrap_seed: 0x53_2026,
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
    /// Mean complete-refit bootstrap slope, when enough replicates succeeded.
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
    /// Plug-in profiled GLS point/interval diagnostics.
    pub plugin_gls: ComparatorDiagnostics,
    /// Profile-likelihood comparator diagnostics.
    pub adjusted_profile: ComparatorDiagnostics,
    /// Complete-refit bootstrap diagnostics.
    pub complete_refit_bootstrap: ComparatorDiagnostics,
    /// Number of bootstrap attempts.
    pub bootstrap_attempts: usize,
    /// Number of successful bootstrap refits.
    pub bootstrap_successes: usize,
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
        plugin_gls: empty_comparator(status),
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
    let plugin_fit = gls_fit(
        &selected_days,
        &selected_y,
        &plugin.covariance,
        options.condition_limit,
    );
    let plugin_comparator = plugin_fit.map_or_else(empty_comparator, |fit| {
        normal_comparator(
            fit.slope,
            fit.information_variance.sqrt(),
            TemporalInferenceStatus::Evaluated,
        )
    });
    let adjusted_profile = profile_comparator(&plugin, degrees_of_freedom);
    let bootstrap_comparator = bootstrap_comparator(&bootstrap);
    let status = if bootstrap.successes < options.bootstrap_minimum_successes {
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
        plugin_gls: plugin_comparator,
        adjusted_profile,
        complete_refit_bootstrap: bootstrap_comparator,
        bootstrap_attempts: bootstrap.attempts,
        bootstrap_successes: bootstrap.successes,
    }
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
        median_gap_days: gaps.get(gaps.len() / 2).copied(),
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
    mean: f64,
    interval_68: Option<ValidationInterval>,
    interval_90: Option<ValidationInterval>,
    interval_95: Option<ValidationInterval>,
    attempts: usize,
    successes: usize,
    variance: f64,
    minimum_successes: usize,
}

fn normal_comparator(
    slope_per_day: f64,
    standard_error_per_day: f64,
    status: TemporalInferenceStatus,
) -> ComparatorDiagnostics {
    let point = slope_per_day * DAYS_PER_YEAR;
    let standard_error = standard_error_per_day * DAYS_PER_YEAR;
    ComparatorDiagnostics {
        point_estimate: point.is_finite().then_some(point),
        standard_error_diagnostic: standard_error.is_finite().then_some(standard_error),
        interval_68: interval(point, standard_error, 0.9944579, 0, 0),
        interval_90: interval(point, standard_error, 1.6448536, 0, 0),
        interval_95: interval(point, standard_error, 1.959964, 0, 0),
        width_68: Some(2.0 * 0.9944579 * standard_error),
        width_90: Some(2.0 * 1.6448536 * standard_error),
        width_95: Some(2.0 * 1.959964 * standard_error),
        status,
        attempted_replicates: 0,
        successful_replicates: 0,
    }
}

fn profile_comparator(plugin: &PluginFit, degrees_of_freedom: usize) -> ComparatorDiagnostics {
    let standard_error = plugin.information_variance.sqrt() * DAYS_PER_YEAR;
    let point = plugin.slope * DAYS_PER_YEAR;
    let (z68, z90, z95) = t_multipliers(degrees_of_freedom);
    ComparatorDiagnostics {
        point_estimate: Some(point),
        standard_error_diagnostic: Some(standard_error),
        interval_68: interval(point, standard_error, z68, 0, 0),
        interval_90: interval(point, standard_error, z90, 0, 0),
        interval_95: interval(point, standard_error, z95, 0, 0),
        width_68: Some(2.0 * z68 * standard_error),
        width_90: Some(2.0 * z90 * standard_error),
        width_95: Some(2.0 * z95 * standard_error),
        status: TemporalInferenceStatus::Evaluated,
        attempted_replicates: 0,
        successful_replicates: 0,
    }
}

fn bootstrap_comparator(summary: &BootstrapSummary) -> ComparatorDiagnostics {
    let se = if summary.successes > 1 {
        Some(summary.variance.sqrt() * DAYS_PER_YEAR)
    } else {
        None
    };
    ComparatorDiagnostics {
        point_estimate: (summary.successes > 0).then_some(summary.mean * DAYS_PER_YEAR),
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

fn t_multipliers(degrees_of_freedom: usize) -> (f64, f64, f64) {
    if degrees_of_freedom < 30 {
        (1.0, 1.833, 2.262)
    } else {
        (0.9944579, 1.6448536, 1.959964)
    }
}

fn profile_plugin(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
    options: &TemporalCovarianceOptions,
) -> Result<PluginFit, TemporalInferenceStatus> {
    if !(0.0..1.0).contains(&options.rho_max) || options.rho_min < 0.0 {
        return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
    }
    let initial = ols_slope(days, observations)?.abs().max(1e-6);
    let scale = observations
        .iter()
        .zip(days)
        .map(|(y, x)| (y - initial * x).powi(2))
        .sum::<f64>()
        / observations.len() as f64;
    let log_min = (scale * 1e-6).max(1e-12).ln();
    let log_max = (scale * 1e6).max(1e-12).ln();
    let rho_upper = (options.rho_max - 1e-8).max(options.rho_min + 1e-8);
    let mut best: Option<(f64, PluginFit)> = None;
    for initial_rho in [
        options.rho_min,
        (options.rho_min + rho_upper) / 2.0,
        rho_upper,
    ] {
        let mut rho = initial_rho;
        let mut log_variance = scale.max(1e-12).ln();
        for _ in 0..3 {
            rho = golden_section_minimum(options.rho_min, rho_upper, |candidate| {
                profile_objective(
                    days,
                    observations,
                    difference_covariance,
                    candidate,
                    log_variance,
                    options,
                )
                .map_or(f64::INFINITY, |(score, _)| score)
            });
            log_variance = golden_section_minimum(log_min, log_max, |candidate| {
                profile_objective(
                    days,
                    observations,
                    difference_covariance,
                    rho,
                    candidate,
                    options,
                )
                .map_or(f64::INFINITY, |(score, _)| score)
            });
        }
        if let Ok(candidate) = profile_objective(
            days,
            observations,
            difference_covariance,
            rho,
            log_variance,
            options,
        ) {
            if best.as_ref().is_none_or(|(score, _)| candidate.0 < *score) {
                best = Some(candidate);
            }
        }
    }
    let (_, fit) = best.ok_or(TemporalInferenceStatus::CovarianceParameterAtBoundary)?;
    if fit.rho >= rho_upper - 1e-6 || fit.process_variance <= log_min.exp() * 1.000001 {
        return Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
    }
    Ok(fit)
}

fn profile_objective(
    days: &[f64],
    observations: &[f64],
    difference_covariance: &[Vec<f64>],
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
    let fit = gls_fit(days, observations, &covariance, options.condition_limit)?;
    let objective = fit.log_determinant + fit.quadratic_form + fit.design_information.ln();
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

struct GlsFit {
    slope: f64,
    quadratic_form: f64,
    log_determinant: f64,
    design_information: f64,
    information_variance: f64,
    condition_number: f64,
}

fn gls_fit(
    days: &[f64],
    observations: &[f64],
    covariance: &[Vec<f64>],
    condition_limit: f64,
) -> Result<GlsFit, TemporalInferenceStatus> {
    let lower =
        cholesky(covariance).ok_or(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite)?;
    let inverse = invert_positive_definite(covariance, condition_limit)?;
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
        design_information: information,
        information_variance: 1.0 / information,
        condition_number: condition_number(covariance),
    })
}

fn bootstrap_refit(
    days: &[f64],
    difference_covariance: &[Vec<f64>],
    plugin: &PluginFit,
    options: &TemporalCovarianceOptions,
) -> BootstrapSummary {
    if options.bootstrap_replicates == 0 {
        return BootstrapSummary {
            mean: f64::NAN,
            interval_68: None,
            interval_90: None,
            interval_95: None,
            attempts: 0,
            successes: 0,
            variance: f64::NAN,
            minimum_successes: options.bootstrap_minimum_successes,
        };
    }
    let Some(cholesky) = cholesky(&plugin.covariance) else {
        return BootstrapSummary {
            mean: f64::NAN,
            interval_68: None,
            interval_90: None,
            interval_95: None,
            attempts: options.bootstrap_replicates,
            successes: 0,
            variance: f64::NAN,
            minimum_successes: options.bootstrap_minimum_successes,
        };
    };
    let mut state = options.bootstrap_seed;
    let mut slopes = Vec::with_capacity(options.bootstrap_replicates);
    for _ in 0..options.bootstrap_replicates {
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
        mean,
        interval_68: validation_interval(0.68),
        interval_90: validation_interval(0.90),
        interval_95: validation_interval(0.95),
        attempts: options.bootstrap_replicates,
        successes,
        variance,
        minimum_successes: options.bootstrap_minimum_successes,
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
) -> Result<Vec<Vec<f64>>, TemporalInferenceStatus> {
    validate_square_covariance(matrix)?;
    let Some(cholesky) = cholesky(matrix) else {
        return Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite);
    };
    let condition = condition_number(matrix);
    if !condition.is_finite() || condition > condition_limit {
        return Err(TemporalInferenceStatus::DesignIllConditioned);
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
    let eigenvalues = symmetric_eigenvalues(matrix);
    let largest = eigenvalues.iter().copied().fold(0.0, f64::max);
    let smallest = eigenvalues
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    largest / smallest
}

#[allow(clippy::needless_range_loop)]
fn symmetric_eigenvalues(matrix: &[Vec<f64>]) -> Vec<f64> {
    let n = matrix.len();
    let mut work = matrix.to_vec();
    for _ in 0..(n * n * 5).max(20) {
        let mut pivot = (0, 0);
        let mut largest = 0.0;
        for row in 0..n {
            for column in (row + 1)..n {
                if work[row][column].abs() > largest {
                    largest = work[row][column].abs();
                    pivot = (row, column);
                }
            }
        }
        if largest < 1e-14 {
            break;
        }
        let (row, column) = pivot;
        let angle = 0.5 * (2.0 * work[row][column]).atan2(work[row][row] - work[column][column]);
        let cosine = angle.cos();
        let sine = angle.sin();
        for index in 0..n {
            let left = work[index][row];
            let right = work[index][column];
            work[index][row] = cosine * left - sine * right;
            work[index][column] = sine * left + cosine * right;
        }
        for index in 0..n {
            let top = work[row][index];
            let bottom = work[column][index];
            work[row][index] = cosine * top - sine * bottom;
            work[column][index] = sine * top + cosine * bottom;
        }
    }
    (0..n).map(|index| work[index][index]).collect()
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

//! Validation-only temporal covariance and slope inference kernel for issue #53.
//!
//! The kernel operates on an already spatially differenced, origin-anchored
//! series and a direct #54 difference covariance.  It is deliberately separate
//! from workflow output code: no raster writer or corrected uncertainty product
//! is exposed here.  The public result contains point estimates, diagnostics,
//! and validation intervals, but no corrected inferential standard error.

use serde::{Deserialize, Serialize};

const DAYS_PER_YEAR: f64 = 365.25;
const RHO_STEP: f64 = 0.05;
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
            minimum_dates: 3,
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
    let oracle = match gls_slope(
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
    let status = if bootstrap.successes < options.bootstrap_minimum_successes {
        TemporalInferenceStatus::BootstrapInsufficientSuccess
    } else {
        TemporalInferenceStatus::Evaluated
    };
    TemporalCovarianceFit {
        status,
        ols_slope: Some(ols * DAYS_PER_YEAR),
        oracle_gls_slope: Some(oracle * DAYS_PER_YEAR),
        plugin_gls_slope: Some(plugin.slope * DAYS_PER_YEAR),
        adjusted_profile_slope: Some(plugin.slope * DAYS_PER_YEAR),
        bootstrap_slope: (bootstrap.successes > 0).then_some(bootstrap.mean * DAYS_PER_YEAR),
        bootstrap_interval: (bootstrap.successes > 0
            && bootstrap.successes >= options.bootstrap_minimum_successes)
            .then_some(ValidationInterval {
                lower: bootstrap.lower * DAYS_PER_YEAR,
                upper: bootstrap.upper * DAYS_PER_YEAR,
                successful_replicates: bootstrap.successes,
            }),
        fitted_rho: Some(plugin.rho),
        fitted_process_variance: Some(plugin.process_variance),
        raw_correlation,
        valid_date_count: n,
        rank,
        degrees_of_freedom,
        covariance_condition_number: Some(plugin.condition_number),
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
}

struct BootstrapSummary {
    mean: f64,
    lower: f64,
    upper: f64,
    successes: usize,
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
    let variance_candidates =
        [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0].map(|factor| (scale * factor).max(1e-12));
    let mut best: Option<(f64, PluginFit)> = None;
    let mut rho = options.rho_min;
    while rho < options.rho_max {
        for process_variance in variance_candidates {
            let covariance = total_difference_covariance(
                difference_covariance,
                days,
                process_variance,
                rho,
                options.reference_lag_days,
            )?;
            let fit = gls_fit(days, observations, &covariance, options.condition_limit)?;
            let objective = fit.log_determinant + fit.quadratic_form + fit.design_information.ln();
            let candidate = PluginFit {
                slope: fit.slope,
                rho,
                process_variance,
                covariance,
                condition_number: fit.condition_number,
            };
            if best.as_ref().is_none_or(|(score, _)| objective < *score) {
                best = Some((objective, candidate));
            }
        }
        rho += RHO_STEP;
    }
    best.map(|(_, fit)| fit)
        .ok_or(TemporalInferenceStatus::CovarianceParameterAtBoundary)
}

struct GlsFit {
    slope: f64,
    quadratic_form: f64,
    log_determinant: f64,
    design_information: f64,
    condition_number: f64,
}

fn gls_slope(
    days: &[f64],
    observations: &[f64],
    covariance: &[Vec<f64>],
    condition_limit: f64,
) -> Result<f64, TemporalInferenceStatus> {
    Ok(gls_fit(days, observations, covariance, condition_limit)?.slope)
}

fn gls_fit(
    days: &[f64],
    observations: &[f64],
    covariance: &[Vec<f64>],
    condition_limit: f64,
) -> Result<GlsFit, TemporalInferenceStatus> {
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
    let log_determinant = covariance
        .iter()
        .enumerate()
        .map(|(index, row)| row[index].abs().ln())
        .sum();
    Ok(GlsFit {
        slope,
        quadratic_form,
        log_determinant,
        design_information: information,
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
            lower: f64::NAN,
            upper: f64::NAN,
            successes: 0,
        };
    }
    let Some(cholesky) = cholesky(&plugin.covariance) else {
        return BootstrapSummary {
            mean: f64::NAN,
            lower: f64::NAN,
            upper: f64::NAN,
            successes: 0,
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
    BootstrapSummary {
        mean,
        lower: if successes > 0 {
            quantile(0.025)
        } else {
            f64::NAN
        },
        upper: if successes > 0 {
            quantile(0.975)
        } else {
            f64::NAN
        },
        successes,
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
    let diagonal = matrix
        .iter()
        .enumerate()
        .map(|(index, row)| row[index].abs())
        .collect::<Vec<_>>();
    diagonal.iter().copied().fold(0.0, f64::max)
        / diagonal
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .fold(f64::INFINITY, f64::min)
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

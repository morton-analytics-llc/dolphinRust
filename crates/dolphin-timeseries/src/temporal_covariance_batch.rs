use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use pulp::{Arch, Simd, WithSimd};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, StudentsT};

#[cfg(test)]
use crate::temporal_covariance::factor_native_objective;
use crate::temporal_covariance::{
    cholesky, dense_factor_objective_fallback, difference_covariance_from_factor, empty_comparator,
    factor_condition_certificate, factor_native_profile_plugin, interval, lower_mat_vec,
    normal_comparator, nuisance_bounds, nuisance_parameter_active_set,
    reml_covariance_parameter_adjusted_variance, splitmix64, standard_normal,
    temporal_parameter_boundary_status, total_covariance_from_factor, ComparatorDiagnostics,
    FactorConditionMethod, FactorObjectiveEvaluation, FactorObjectiveScratch,
    PreparedFactorObjective, TemporalCovarianceOptions, TemporalInferenceStatus,
    ValidationInterval, DAYS_PER_YEAR, SYMMETRY_TOLERANCE,
};

/// Maximum retained factor-native solver scratch for one Rayon worker.
pub const TEMPORAL_FACTOR_SCALAR_MAX_WORKER_SCRATCH_BYTES: usize = 8 * 1024 * 1024;
const MAX_BASIS_GAP_CLASSES: usize = 8;

type ObjectiveResult = Result<FactorObjectiveEvaluation, TemporalInferenceStatus>;
type ProfileResult = Result<TemporalBatchProfileEvaluation, TemporalInferenceStatus>;

/// One factor-native REML fit exposed as the comparable plug-in and adjusted scalars.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalFactorScalarPair {
    /// Plug-in GLS slope in units per day before comparator scaling.
    pub plugin_slope_per_day: Option<f64>,
    /// Plug-in GLS using the profiled REML covariance fit.
    pub plugin_gls_reml: ComparatorDiagnostics,
    /// Analytic covariance-parameter-adjusted scalar from the same REML fit.
    pub reml_covariance_parameter_adjusted_scalar: ComparatorDiagnostics,
    /// Fitted continuous-time correlation.
    pub fitted_rho: Option<f64>,
    /// Fitted residual process variance.
    pub fitted_process_variance: Option<f64>,
    /// Active fitted nuisance boundary handled by constrained inference.
    pub fitted_parameter_active_set: Option<TemporalInferenceStatus>,
    /// Conservative condition-number upper bound recorded by the primary path.
    pub condition_upper_bound: Option<f64>,
    /// Exact condition number when the conservative certificate was inconclusive.
    pub exact_condition_number: Option<f64>,
}

/// Bounded execution evidence for one paired factor-native batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalFactorScalarBatchMetrics {
    /// Target profiles attempted exactly once.
    pub profile_fit_count: usize,
    /// Complete-refit bootstrap attempts represented by this report.
    pub bootstrap_attempts: usize,
    /// Analytic covariance-parameter adjustments materialized by this method path.
    pub covariance_parameter_adjustment_count: usize,
    /// Theta lanes that materialized adjustment-only slope derivatives.
    pub covariance_parameter_derivative_lane_evaluations: usize,
    /// Shared REML rho-lane evaluations.
    pub optimizer_rho_lane_evaluations: usize,
    /// Shared REML process-variance objective evaluations.
    pub optimizer_q_objective_evaluations: usize,
    /// Shared REML primary-rho pass histogram, with the last bin saturating at 20.
    pub optimizer_primary_rho_pass_histogram: [u64; 21],
    /// Number of worker-local microblocks prepared.
    pub microblocks_prepared: usize,
    /// Maximum primary rho passes for any target.
    pub maximum_primary_rho_passes: usize,
    /// Targets sent to the exact optimizer fallback.
    pub exact_optimizer_fallback_targets: usize,
    /// Fixed-theta dense objective fallback evaluations.
    pub fixed_theta_dense_fallback_evaluations: usize,
    /// Exact condition-number fallback evaluations.
    pub condition_exact_fallbacks: usize,
    /// Maximum retained scratch for one worker.
    pub maximum_worker_scratch_bytes: usize,
    /// Persisted plus compact factor bytes retained during this batch.
    pub retained_factor_bytes: usize,
    /// Rayon workers represented by the bounded arena pool.
    pub worker_count: usize,
}

/// Paired scalar outcomes and bounded execution evidence for one persisted factor block.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalFactorScalarBatchReport {
    /// Results in persisted factor target order.
    pub outcomes: Vec<TemporalFactorScalarPair>,
    /// Batch execution counts and scratch high-water evidence.
    pub metrics: TemporalFactorScalarBatchMetrics,
}

/// Complete-refit bootstrap diagnostics from one factor-native replica batch.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalFactorBootstrapReport {
    /// Observed plug-in slope paired with the replica interval distribution.
    pub complete_refit_bootstrap: ComparatorDiagnostics,
    /// Batched profile execution evidence for the replica lanes.
    pub metrics: TemporalFactorScalarBatchMetrics,
}

/// Run every complete-refit bootstrap replica through one shared factor-native batch.
///
/// The fitted total covariance is factored once for simulation. Replica random streams and
/// summary order are identical to the dense reference bootstrap; only the refits are batched.
///
/// # Errors
/// Returns a fail-closed status when the observed fit, persisted factor, or replica batch is
/// malformed.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn fit_temporal_factor_complete_refit_bootstrap(
    post_gauge_days: &[f64],
    persisted_factor: &[f64],
    persisted_maximum_rank: usize,
    realized_rank: usize,
    observed_slope_per_year: f64,
    fitted_rho: f64,
    fitted_process_variance: f64,
    options: &TemporalCovarianceOptions,
) -> Result<TemporalFactorBootstrapReport, TemporalInferenceStatus> {
    let date_count = post_gauge_days.len();
    let acquisition_count = date_count
        .checked_add(1)
        .ok_or(TemporalInferenceStatus::CovarianceNonfinite)?;
    let persisted_stride = acquisition_count
        .checked_mul(persisted_maximum_rank)
        .ok_or(TemporalInferenceStatus::CovarianceNonfinite)?;
    let attempts = options.bootstrap_replicates;
    if attempts == 0
        || date_count < options.minimum_dates
        || realized_rank == 0
        || realized_rank > persisted_maximum_rank
        || realized_rank > date_count
        || persisted_factor.len() != persisted_stride
        || persisted_factor[..persisted_maximum_rank]
            .iter()
            .any(|value| !value.is_finite() || *value != 0.0)
        || !observed_slope_per_year.is_finite()
        || !fitted_rho.is_finite()
        || !fitted_process_variance.is_finite()
        || fitted_process_variance <= 0.0
    {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let mut compact_factor = vec![0.0; date_count * persisted_maximum_rank];
    for date in 0..date_count {
        let source = (date + 1) * persisted_maximum_rank;
        let destination = date * persisted_maximum_rank;
        compact_factor[destination..destination + realized_rank]
            .copy_from_slice(&persisted_factor[source..source + realized_rank]);
    }
    let prepared = PreparedFactorObjective::new(post_gauge_days, options.reference_lag_days)?;
    let total_covariance = total_covariance_from_factor(
        &prepared,
        &compact_factor,
        persisted_maximum_rank,
        realized_rank,
        fitted_rho,
        fitted_process_variance,
    )?;
    let lower = cholesky(&total_covariance)
        .ok_or(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite)?;
    let observed_slope_per_day = observed_slope_per_year / DAYS_PER_YEAR;
    let mut observations_soa = vec![0.0; date_count * attempts];
    for replicate in 0..attempts {
        let mut state = splitmix64(options.bootstrap_seed ^ replicate as u64);
        let normal = (0..date_count)
            .map(|_| standard_normal(&mut state))
            .collect::<Vec<_>>();
        let residual = lower_mat_vec(&lower, &normal);
        for date in 0..date_count {
            observations_soa[date * attempts + replicate] =
                observed_slope_per_day * post_gauge_days[date] + residual[date];
        }
    }
    let realized_ranks = vec![realized_rank; attempts];
    let bootstrap_options = TemporalCovarianceOptions {
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..options.clone()
    };
    let mut batch = fit_temporal_factor_scalar_batch_internal(
        post_gauge_days,
        &observations_soa,
        persisted_factor,
        persisted_maximum_rank,
        &realized_ranks,
        &bootstrap_options,
        false,
        true,
    )?;
    let mut slopes = batch
        .outcomes
        .iter()
        .filter_map(|outcome| {
            (outcome.plugin_gls_reml.status == TemporalInferenceStatus::Evaluated)
                .then_some(outcome.plugin_gls_reml.point_estimate)
                .flatten()
        })
        .collect::<Vec<_>>();
    slopes.sort_by(f64::total_cmp);
    let successes = slopes.len();
    let mean = if successes == 0 {
        f64::NAN
    } else {
        slopes.iter().sum::<f64>() / successes as f64
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
    let interval_68 = bootstrap_interval(&slopes, 0.68);
    let interval_90 = bootstrap_interval(&slopes, 0.90);
    let interval_95 = bootstrap_interval(&slopes, 0.95);
    let minimum_successes = attempts
        .saturating_mul(99)
        .saturating_add(99)
        .div_euclid(100)
        .max(options.bootstrap_minimum_successes);
    batch.metrics.bootstrap_attempts = attempts;
    Ok(TemporalFactorBootstrapReport {
        complete_refit_bootstrap: ComparatorDiagnostics {
            point_estimate: Some(observed_slope_per_year),
            standard_error_diagnostic: (successes > 1).then_some(variance.sqrt()),
            interval_68,
            interval_90,
            interval_95,
            width_68: interval_68.map(|value| value.upper - value.lower),
            width_90: interval_90.map(|value| value.upper - value.lower),
            width_95: interval_95.map(|value| value.upper - value.lower),
            status: if successes >= minimum_successes {
                TemporalInferenceStatus::Evaluated
            } else {
                TemporalInferenceStatus::BootstrapInsufficientSuccess
            },
            attempted_replicates: attempts,
            successful_replicates: successes,
        },
        metrics: batch.metrics,
    })
}

fn bootstrap_interval(slopes: &[f64], level: f64) -> Option<ValidationInterval> {
    (!slopes.is_empty()).then(|| {
        let quantile = |fraction: f64| {
            let position = fraction * slopes.len().saturating_sub(1) as f64;
            slopes[position.round() as usize]
        };
        ValidationInterval {
            lower: quantile((1.0 - level) / 2.0),
            upper: quantile(1.0 - (1.0 - level) / 2.0),
            successful_replicates: slopes.len(),
        }
    })
}

/// Fit plug-in GLS and its analytic covariance-parameter adjustment from one persisted factor.
///
/// `observations_soa` is date-major over post-gauge dates. `persisted_factors` contains either one
/// factor shared by equal-rank targets or one target-major factor per target, with the persisted
/// gauge row followed by post-gauge rows and `persisted_maximum_rank` components per row. Both
/// scalar comparators reuse one REML profile per target; this path never executes the
/// complete-refit bootstrap.
///
/// # Errors
/// Returns a fail-closed temporal status when dimensions, dates, ranks, or the batch profile are
/// invalid.
pub fn fit_temporal_factor_scalar_batch(
    post_gauge_days: &[f64],
    observations_soa: &[f64],
    persisted_factors: &[f64],
    persisted_maximum_rank: usize,
    realized_ranks: &[usize],
    options: &TemporalCovarianceOptions,
) -> Result<TemporalFactorScalarBatchReport, TemporalInferenceStatus> {
    fit_temporal_factor_scalar_batch_internal(
        post_gauge_days,
        observations_soa,
        persisted_factors,
        persisted_maximum_rank,
        realized_ranks,
        options,
        true,
        true,
    )
}

/// Fit only plug-in GLS from the shared factor-native REML profile.
///
/// This baseline executes the same factor preparation, optimizer, condition checks, and ordered
/// output path as [`fit_temporal_factor_scalar_batch`], but does not materialize the analytic
/// covariance-parameter adjustment.
///
/// # Errors
/// Returns the same fail-closed temporal statuses as the adjusted batch path.
pub fn fit_temporal_factor_plugin_batch(
    post_gauge_days: &[f64],
    observations_soa: &[f64],
    persisted_factors: &[f64],
    persisted_maximum_rank: usize,
    realized_ranks: &[usize],
    options: &TemporalCovarianceOptions,
) -> Result<TemporalFactorScalarBatchReport, TemporalInferenceStatus> {
    fit_temporal_factor_scalar_batch_internal(
        post_gauge_days,
        observations_soa,
        persisted_factors,
        persisted_maximum_rank,
        realized_ranks,
        options,
        false,
        false,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn fit_temporal_factor_scalar_batch_internal(
    post_gauge_days: &[f64],
    observations_soa: &[f64],
    persisted_factors: &[f64],
    persisted_maximum_rank: usize,
    realized_ranks: &[usize],
    options: &TemporalCovarianceOptions,
    materialize_adjustment: bool,
    accept_boundary_solution: bool,
) -> Result<TemporalFactorScalarBatchReport, TemporalInferenceStatus> {
    let target_count = realized_ranks.len();
    let date_count = post_gauge_days.len();
    let acquisition_count = date_count
        .checked_add(1)
        .ok_or(TemporalInferenceStatus::CovarianceNonfinite)?;
    let persisted_target_stride = acquisition_count
        .checked_mul(persisted_maximum_rank)
        .ok_or(TemporalInferenceStatus::CovarianceNonfinite)?;
    let repeated_factor_elements = target_count.saturating_mul(persisted_target_stride);
    let persisted_factor_count = if persisted_factors.len() == persisted_target_stride {
        1
    } else if persisted_factors.len() == repeated_factor_elements {
        target_count
    } else {
        0
    };
    let shared_factor = target_count > 1 && persisted_factor_count == 1;
    if target_count == 0
        || date_count < options.minimum_dates
        || persisted_maximum_rank == 0
        || observations_soa.len() != date_count.saturating_mul(target_count)
        || persisted_factor_count == 0
        || (shared_factor && realized_ranks.iter().any(|rank| *rank != realized_ranks[0]))
        || realized_ranks
            .iter()
            .any(|rank| *rank > date_count || *rank > persisted_maximum_rank)
    {
        return Err(if date_count < options.minimum_dates {
            TemporalInferenceStatus::InsufficientDates
        } else {
            TemporalInferenceStatus::CovarianceNonfinite
        });
    }
    let maximum_rank = realized_ranks.iter().copied().max().unwrap_or(0);
    if maximum_rank == 0 {
        let status = TemporalInferenceStatus::CovarianceNonfinite;
        return Ok(TemporalFactorScalarBatchReport {
            outcomes: realized_ranks
                .iter()
                .map(|_| failed_scalar_pair(status))
                .collect(),
            metrics: TemporalFactorScalarBatchMetrics {
                profile_fit_count: 0,
                bootstrap_attempts: 0,
                covariance_parameter_adjustment_count: 0,
                covariance_parameter_derivative_lane_evaluations: 0,
                optimizer_rho_lane_evaluations: 0,
                optimizer_q_objective_evaluations: 0,
                optimizer_primary_rho_pass_histogram: [0; 21],
                microblocks_prepared: 0,
                maximum_primary_rho_passes: 0,
                exact_optimizer_fallback_targets: 0,
                fixed_theta_dense_fallback_evaluations: 0,
                condition_exact_fallbacks: 0,
                maximum_worker_scratch_bytes: 0,
                retained_factor_bytes: persisted_factors
                    .len()
                    .saturating_mul(std::mem::size_of::<f64>()),
                worker_count: rayon::current_num_threads().max(1),
            },
        });
    }
    let compact_target_stride = date_count
        .checked_mul(maximum_rank)
        .ok_or(TemporalInferenceStatus::CovarianceNonfinite)?;
    let mut factors = vec![0.0; persisted_factor_count.saturating_mul(compact_target_stride)];
    for factor_index in 0..persisted_factor_count {
        let rank_index = if shared_factor { 0 } else { factor_index };
        let rank = realized_ranks[rank_index];
        if rank == 0 {
            continue;
        }
        let persisted_target = factor_index * persisted_target_stride;
        if persisted_factors[persisted_target..persisted_target + persisted_maximum_rank]
            .iter()
            .any(|value| !value.is_finite() || *value != 0.0)
        {
            return Err(TemporalInferenceStatus::GaugeMissing);
        }
        let compact_target = factor_index * compact_target_stride;
        for date in 0..date_count {
            let source = persisted_target + (date + 1) * persisted_maximum_rank;
            let destination = compact_target + date * maximum_rank;
            factors[destination..destination + rank]
                .copy_from_slice(&persisted_factors[source..source + rank]);
        }
    }
    let prepared = PreparedFactorObjective::new(post_gauge_days, options.reference_lag_days)?;
    let mut execution = TemporalBatchExecution::new(
        &prepared,
        observations_soa,
        &factors,
        maximum_rank,
        realized_ranks,
    )?;
    let (profile_outcomes, profile_metrics) = {
        let report =
            execution.profile_reml(options, materialize_adjustment, accept_boundary_solution)?;
        (report.outcomes.to_vec(), report.metrics)
    };
    let execution_metrics = execution.metrics();
    let residual_degrees_of_freedom = date_count.saturating_sub(1);
    let outcomes = profile_outcomes
        .into_iter()
        .map(|outcome| match outcome {
            Ok(evaluation) => scalar_pair(
                evaluation,
                residual_degrees_of_freedom,
                materialize_adjustment,
            ),
            Err(status) => failed_scalar_pair(status),
        })
        .collect::<Vec<_>>();
    let covariance_parameter_adjustment_count = outcomes
        .iter()
        .filter(|outcome| {
            outcome.reml_covariance_parameter_adjusted_scalar.status
                == TemporalInferenceStatus::Evaluated
        })
        .count();
    Ok(TemporalFactorScalarBatchReport {
        outcomes,
        metrics: TemporalFactorScalarBatchMetrics {
            profile_fit_count: realized_ranks.iter().filter(|rank| **rank > 0).count(),
            bootstrap_attempts: 0,
            covariance_parameter_adjustment_count,
            covariance_parameter_derivative_lane_evaluations: profile_metrics
                .covariance_parameter_derivative_lane_evaluations,
            optimizer_rho_lane_evaluations: profile_metrics.rho_lane_evaluations,
            optimizer_q_objective_evaluations: profile_metrics.q_objective_evaluations,
            optimizer_primary_rho_pass_histogram: profile_metrics.primary_rho_pass_histogram,
            microblocks_prepared: profile_metrics.microblocks_prepared,
            maximum_primary_rho_passes: profile_metrics.maximum_primary_rho_passes,
            exact_optimizer_fallback_targets: profile_metrics.exact_optimizer_fallback_targets,
            fixed_theta_dense_fallback_evaluations: profile_metrics
                .fixed_theta_dense_fallback_evaluations,
            condition_exact_fallbacks: profile_metrics.condition_exact_fallbacks,
            maximum_worker_scratch_bytes: profile_metrics
                .maximum_worker_scratch_bytes
                .max(execution_metrics.maximum_worker_scratch_bytes),
            retained_factor_bytes: persisted_factors
                .len()
                .saturating_add(factors.len())
                .saturating_mul(std::mem::size_of::<f64>()),
            worker_count: execution_metrics.worker_count,
        },
    })
}

fn failed_scalar_pair(status: TemporalInferenceStatus) -> TemporalFactorScalarPair {
    TemporalFactorScalarPair {
        plugin_slope_per_day: None,
        plugin_gls_reml: empty_comparator(status),
        reml_covariance_parameter_adjusted_scalar: empty_comparator(status),
        fitted_rho: None,
        fitted_process_variance: None,
        fitted_parameter_active_set: None,
        condition_upper_bound: None,
        exact_condition_number: None,
    }
}

fn scalar_pair(
    evaluation: TemporalBatchProfileEvaluation,
    residual_degrees_of_freedom: usize,
    materialize_adjustment: bool,
) -> TemporalFactorScalarPair {
    let plugin_gls_reml = normal_comparator(
        evaluation.slope,
        evaluation.information_variance.sqrt(),
        TemporalInferenceStatus::Evaluated,
    );
    let reml_covariance_parameter_adjusted_scalar = if materialize_adjustment {
        evaluation
            .covariance_parameter_adjusted_variance
            .filter(|variance| variance.is_finite() && *variance > 0.0)
            .and_then(|variance| {
                let distribution =
                    StudentsT::new(0.0, 1.0, residual_degrees_of_freedom as f64).ok()?;
                let point = evaluation.slope * DAYS_PER_YEAR;
                let standard_error = variance.sqrt() * DAYS_PER_YEAR;
                let interval_68 =
                    interval(point, standard_error, distribution.inverse_cdf(0.84), 0, 0)?;
                let interval_90 =
                    interval(point, standard_error, distribution.inverse_cdf(0.95), 0, 0)?;
                let interval_95 =
                    interval(point, standard_error, distribution.inverse_cdf(0.975), 0, 0)?;
                Some(ComparatorDiagnostics {
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
            })
            .unwrap_or_else(|| {
                empty_comparator(TemporalInferenceStatus::WeakParameterIdentification)
            })
    } else {
        empty_comparator(TemporalInferenceStatus::DiagnosticNotComputed)
    };
    TemporalFactorScalarPair {
        plugin_slope_per_day: Some(evaluation.slope),
        plugin_gls_reml,
        reml_covariance_parameter_adjusted_scalar,
        fitted_rho: Some(evaluation.rho),
        fitted_process_variance: Some(evaluation.process_variance),
        fitted_parameter_active_set: evaluation.fitted_parameter_active_set,
        condition_upper_bound: Some(evaluation.condition_upper_bound),
        exact_condition_number: evaluation.exact_condition_number,
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TemporalBatchProfileEvaluation {
    #[cfg(test)]
    pub(super) score: f64,
    pub(super) slope: f64,
    pub(super) rho: f64,
    pub(super) process_variance: f64,
    pub(super) fitted_parameter_active_set: Option<TemporalInferenceStatus>,
    pub(super) information_variance: f64,
    pub(super) covariance_parameter_adjusted_variance: Option<f64>,
    #[cfg(test)]
    pub(super) profile_rho_curvature: f64,
    pub(super) condition_upper_bound: f64,
    pub(super) exact_condition_number: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TemporalBatchProfileMetrics {
    pub(super) microblocks_prepared: usize,
    pub(super) primary_rho_pass_histogram: [u64; 21],
    pub(super) q_objective_evaluations: usize,
    pub(super) maximum_primary_rho_passes: usize,
    pub(super) exact_optimizer_fallback_targets: usize,
    pub(super) fixed_theta_dense_fallback_evaluations: usize,
    pub(super) condition_upper_bound_accepts: usize,
    pub(super) condition_exact_fallbacks: usize,
    pub(super) compaction_events: usize,
    pub(super) compacted_lane_count: usize,
    pub(super) completed_lane_revisits: usize,
    pub(super) rho_lane_evaluations: usize,
    pub(super) covariance_parameter_derivative_lane_evaluations: usize,
    pub(super) per_target_rho_passes: Vec<usize>,
    pub(super) maximum_worker_scratch_bytes: usize,
}

pub(super) struct TemporalBatchProfileReport<'a> {
    pub(super) outcomes: &'a [ProfileResult],
    pub(super) metrics: TemporalBatchProfileMetrics,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(super) struct AugmentedBasisRemlDerivatives {
    pub(super) augmented_basis: Vec<f64>,
    pub(super) augmented_basis_eta: Vec<f64>,
    pub(super) augmented_basis_eta_eta: Vec<f64>,
    pub(super) evaluation: FactorObjectiveEvaluation,
    pub(super) score_eta: f64,
    pub(super) score_log_q: f64,
    pub(super) score_eta_eta: f64,
    pub(super) score_eta_log_q: f64,
    pub(super) score_log_q_log_q: f64,
    pub(super) slope_eta: f64,
    pub(super) slope_log_q: f64,
    pub(super) profiled_eta_curvature: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(super) struct BatchRemlDerivativeEvaluation {
    pub(super) evaluation: FactorObjectiveEvaluation,
    pub(super) score_eta: f64,
    pub(super) score_log_q: f64,
    pub(super) score_eta_eta: f64,
    pub(super) score_eta_log_q: f64,
    pub(super) score_log_q_log_q: f64,
    pub(super) slope_eta: f64,
    pub(super) slope_log_q: f64,
}

#[cfg(test)]
type DerivativeResult = Result<BatchRemlDerivativeEvaluation, TemporalInferenceStatus>;

#[derive(Clone, Copy)]
struct SecondOrderJet {
    value: f64,
    eta: f64,
    log_q: f64,
    eta_eta: f64,
    eta_log_q: f64,
    log_q_log_q: f64,
}

impl SecondOrderJet {
    const fn constant(value: f64) -> Self {
        Self {
            value,
            eta: 0.0,
            log_q: 0.0,
            eta_eta: 0.0,
            eta_log_q: 0.0,
            log_q_log_q: 0.0,
        }
    }

    fn add(self, right: Self) -> Self {
        Self {
            value: self.value + right.value,
            eta: self.eta + right.eta,
            log_q: self.log_q + right.log_q,
            eta_eta: self.eta_eta + right.eta_eta,
            eta_log_q: self.eta_log_q + right.eta_log_q,
            log_q_log_q: self.log_q_log_q + right.log_q_log_q,
        }
    }

    fn subtract(self, right: Self) -> Self {
        Self {
            value: self.value - right.value,
            eta: self.eta - right.eta,
            log_q: self.log_q - right.log_q,
            eta_eta: self.eta_eta - right.eta_eta,
            eta_log_q: self.eta_log_q - right.eta_log_q,
            log_q_log_q: self.log_q_log_q - right.log_q_log_q,
        }
    }

    fn multiply(self, right: Self) -> Self {
        Self {
            value: self.value * right.value,
            eta: self.eta * right.value + self.value * right.eta,
            log_q: self.log_q * right.value + self.value * right.log_q,
            eta_eta: self.eta_eta * right.value
                + 2.0 * self.eta * right.eta
                + self.value * right.eta_eta,
            eta_log_q: self.eta_log_q * right.value
                + self.eta * right.log_q
                + self.log_q * right.eta
                + self.value * right.eta_log_q,
            log_q_log_q: self.log_q_log_q * right.value
                + 2.0 * self.log_q * right.log_q
                + self.value * right.log_q_log_q,
        }
    }

    fn scale(self, scale: f64) -> Self {
        Self {
            value: self.value * scale,
            eta: self.eta * scale,
            log_q: self.log_q * scale,
            eta_eta: self.eta_eta * scale,
            eta_log_q: self.eta_log_q * scale,
            log_q_log_q: self.log_q_log_q * scale,
        }
    }

    fn reciprocal(self) -> Self {
        let squared = self.value.powi(2);
        let cubed = self.value.powi(3);
        Self {
            value: self.value.recip(),
            eta: -self.eta / squared,
            log_q: -self.log_q / squared,
            eta_eta: 2.0 * self.eta.powi(2) / cubed - self.eta_eta / squared,
            eta_log_q: 2.0 * self.eta * self.log_q / cubed - self.eta_log_q / squared,
            log_q_log_q: 2.0 * self.log_q.powi(2) / cubed - self.log_q_log_q / squared,
        }
    }

    fn divide(self, right: Self) -> Self {
        self.multiply(right.reciprocal())
    }

    fn natural_log(self) -> Self {
        let squared = self.value.powi(2);
        Self {
            value: self.value.ln(),
            eta: self.eta / self.value,
            log_q: self.log_q / self.value,
            eta_eta: self.eta_eta / self.value - self.eta.powi(2) / squared,
            eta_log_q: self.eta_log_q / self.value - self.eta * self.log_q / squared,
            log_q_log_q: self.log_q_log_q / self.value - self.log_q.powi(2) / squared,
        }
    }

    #[cfg(test)]
    fn exponential(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            eta: value * self.eta,
            log_q: value * self.log_q,
            eta_eta: value * (self.eta_eta + self.eta.powi(2)),
            eta_log_q: value * (self.eta_log_q + self.eta * self.log_q),
            log_q_log_q: value * (self.log_q_log_q + self.log_q.powi(2)),
        }
    }

    #[cfg(test)]
    fn square_root(self) -> Self {
        self.natural_log().scale(0.5).exponential()
    }
}

#[cfg(test)]
fn solve_second_order_cholesky(
    lower: &[SecondOrderJet],
    dimension: usize,
    rhs: &[SecondOrderJet],
) -> Vec<SecondOrderJet> {
    let mut solution = vec![SecondOrderJet::constant(0.0); dimension];
    for row in 0..dimension {
        let mut value = rhs[row];
        for column in 0..row {
            value = value.subtract(lower[row * dimension + column].multiply(solution[column]));
        }
        solution[row] = value.divide(lower[row * dimension + row]);
    }
    for row in (0..dimension).rev() {
        let mut value = solution[row];
        for column in row + 1..dimension {
            value = value.subtract(lower[column * dimension + row].multiply(solution[column]));
        }
        solution[row] = value.divide(lower[row * dimension + row]);
    }
    solution
}

#[cfg(test)]
fn dot_second_order(left: &[SecondOrderJet], right: &[SecondOrderJet]) -> SecondOrderJet {
    left.iter()
        .zip(right)
        .fold(SecondOrderJet::constant(0.0), |total, (&left, &right)| {
            total.add(left.multiply(right))
        })
}

#[cfg(test)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn augmented_basis_reml_rho_derivatives(
    prepared: &PreparedFactorObjective,
    observations: &[f64],
    factor: &[f64],
    maximum_rank: usize,
    realized_rank: usize,
    rho: f64,
    log_process_variance: f64,
) -> Result<AugmentedBasisRemlDerivatives, TemporalInferenceStatus> {
    let date_count = prepared.design.len();
    if date_count == 0
        || observations.len() != date_count
        || maximum_rank == 0
        || realized_rank == 0
        || realized_rank > maximum_rank
        || factor.len() != date_count.saturating_mul(maximum_rank)
        || observations.iter().any(|value| !value.is_finite())
        || factor.iter().any(|value| !value.is_finite())
        || !rho.is_finite()
        || rho <= 0.0
        || rho >= 1.0
        || !log_process_variance.is_finite()
    {
        return Err(if !rho.is_finite() || rho <= 0.0 || rho >= 1.0 {
            TemporalInferenceStatus::CovarianceParameterAtBoundary
        } else {
            TemporalInferenceStatus::CovarianceNonfinite
        });
    }
    let process_variance = log_process_variance.exp();
    if !process_variance.is_finite() || process_variance <= 0.0 {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let mut diagonal = vec![0.0; date_count];
    let mut mean_log_diagonal = 0.0;
    for date in 0..date_count {
        diagonal[date] = (0..realized_rank)
            .map(|component| factor[date * maximum_rank + component].powi(2))
            .sum();
        if !diagonal[date].is_finite() || diagonal[date] <= 0.0 {
            return Err(TemporalInferenceStatus::CovarianceNonfinite);
        }
        mean_log_diagonal += diagonal[date].ln();
    }
    let geometric_mean = (mean_log_diagonal / date_count as f64).exp();
    let dimension = realized_rank + 2;
    let mut rows = vec![0.0; date_count * dimension];
    let mut log_shape_sum = 0.0;
    for date in 0..date_count {
        let shape = (diagonal[date] / geometric_mean).sqrt();
        let inverse_shape = shape.recip();
        log_shape_sum += shape.ln();
        for component in 0..realized_rank {
            rows[date * dimension + component] =
                factor[date * maximum_rank + component] * inverse_shape;
        }
        rows[date * dimension + realized_rank] = prepared.design[date] * inverse_shape;
        rows[date * dimension + realized_rank + 1] = observations[date] * inverse_shape;
    }
    let mut basis = vec![0.0; dimension * dimension];
    let mut basis_eta = vec![0.0; dimension * dimension];
    let mut basis_eta_eta = vec![0.0; dimension * dimension];
    for left in 0..dimension {
        for right in 0..dimension {
            basis[left * dimension + right] = rows[left] * rows[right];
        }
    }
    let mut log_determinant_r = 0.0;
    let mut log_determinant_r_eta = 0.0;
    let mut log_determinant_r_eta_eta = 0.0;
    for edge in 0..date_count.saturating_sub(1) {
        let gamma = prepared.gap_exponents[edge];
        let phi = if rho == 0.0 {
            0.0
        } else {
            (rho.ln() * gamma).exp()
        };
        let u = phi * phi;
        let delta = 1.0 - u;
        if !delta.is_finite() || delta <= 0.0 {
            return Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite);
        }
        let a = delta.recip();
        let b = -phi / delta;
        let c = a - 1.0;
        let a_eta = 2.0 * gamma * u / delta.powi(2);
        let a_eta_eta = 4.0 * gamma.powi(2) * u * (1.0 + u) / delta.powi(3);
        let b_eta = -gamma * phi * (1.0 + u) / delta.powi(2);
        let b_eta_eta = -gamma.powi(2) * phi * (1.0 + 6.0 * u + u * u) / delta.powi(3);
        let current = edge + 1;
        let previous = edge;
        for left in 0..dimension {
            for right in 0..dimension {
                let entry = left * dimension + right;
                let current_outer =
                    rows[current * dimension + left] * rows[current * dimension + right];
                let cross = rows[current * dimension + left] * rows[previous * dimension + right]
                    + rows[previous * dimension + left] * rows[current * dimension + right];
                let previous_outer =
                    rows[previous * dimension + left] * rows[previous * dimension + right];
                basis[entry] += a * current_outer + b * cross + c * previous_outer;
                basis_eta[entry] += a_eta * current_outer + b_eta * cross + a_eta * previous_outer;
                basis_eta_eta[entry] +=
                    a_eta_eta * current_outer + b_eta_eta * cross + a_eta_eta * previous_outer;
            }
        }
        log_determinant_r += delta.ln();
        log_determinant_r_eta += -2.0 * gamma * u / delta;
        log_determinant_r_eta_eta += -4.0 * gamma.powi(2) * u / delta.powi(2);
    }
    let jets = basis
        .iter()
        .zip(&basis_eta)
        .zip(&basis_eta_eta)
        .map(|((&value, &eta), &eta_eta)| SecondOrderJet {
            value,
            eta,
            log_q: 0.0,
            eta_eta,
            eta_log_q: 0.0,
            log_q_log_q: 0.0,
        })
        .collect::<Vec<_>>();
    let q = SecondOrderJet {
        value: log_process_variance,
        eta: 0.0,
        log_q: 1.0,
        eta_eta: 0.0,
        eta_log_q: 0.0,
        log_q_log_q: 0.0,
    };
    let process_variance_jet = q.exponential();
    let mut covariance = vec![SecondOrderJet::constant(0.0); realized_rank * realized_rank];
    for row in 0..realized_rank {
        for column in 0..realized_rank {
            covariance[row * realized_rank + column] = jets[row * dimension + column];
            if row == column {
                covariance[row * realized_rank + column] =
                    covariance[row * realized_rank + column].add(process_variance_jet);
            }
        }
    }
    let mut lower = vec![SecondOrderJet::constant(0.0); realized_rank * realized_rank];
    for row in 0..realized_rank {
        for column in 0..=row {
            let mut value = covariance[row * realized_rank + column];
            for inner in 0..column {
                value = value.subtract(
                    lower[row * realized_rank + inner]
                        .multiply(lower[column * realized_rank + inner]),
                );
            }
            if row == column {
                if !value.value.is_finite() || value.value <= 0.0 {
                    return Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite);
                }
                lower[row * realized_rank + column] = value.square_root();
            } else {
                lower[row * realized_rank + column] =
                    value.divide(lower[column * realized_rank + column]);
            }
        }
    }
    let x = realized_rank;
    let y = realized_rank + 1;
    let u_x = (0..realized_rank)
        .map(|component| jets[component * dimension + x])
        .collect::<Vec<_>>();
    let u_y = (0..realized_rank)
        .map(|component| jets[component * dimension + y])
        .collect::<Vec<_>>();
    let solve_x = solve_second_order_cholesky(&lower, realized_rank, &u_x);
    let solve_y = solve_second_order_cholesky(&lower, realized_rank, &u_y);
    let x_v_x = jets[x * dimension + x]
        .subtract(dot_second_order(&u_x, &solve_x))
        .divide(process_variance_jet);
    let x_v_y = jets[x * dimension + y]
        .subtract(dot_second_order(&u_x, &solve_y))
        .divide(process_variance_jet);
    let y_v_y = jets[y * dimension + y]
        .subtract(dot_second_order(&u_y, &solve_y))
        .divide(process_variance_jet);
    if !x_v_x.value.is_finite() || x_v_x.value <= 0.0 {
        return Err(TemporalInferenceStatus::DesignRankDeficient);
    }
    let slope = x_v_y.divide(x_v_x);
    let quadratic = y_v_y.subtract(x_v_y.multiply(x_v_y).divide(x_v_x));
    let mut log_determinant_k = SecondOrderJet::constant(0.0);
    for component in 0..realized_rank {
        log_determinant_k = log_determinant_k.add(
            lower[component * realized_rank + component]
                .natural_log()
                .scale(2.0),
        );
    }
    let log_determinant_r_jet = SecondOrderJet {
        value: log_determinant_r,
        eta: log_determinant_r_eta,
        log_q: 0.0,
        eta_eta: log_determinant_r_eta_eta,
        eta_log_q: 0.0,
        log_q_log_q: 0.0,
    };
    let log_determinant = q
        .scale((date_count - realized_rank) as f64)
        .add(SecondOrderJet::constant(2.0 * log_shape_sum))
        .add(log_determinant_r_jet)
        .add(log_determinant_k);
    let score = log_determinant.add(quadratic).add(x_v_x.natural_log());
    if !score.value.is_finite()
        || !score.eta.is_finite()
        || !score.log_q.is_finite()
        || !score.eta_eta.is_finite()
        || !score.eta_log_q.is_finite()
        || !score.log_q_log_q.is_finite()
        || !slope.eta.is_finite()
        || !slope.log_q.is_finite()
    {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    let mut reference_scratch = FactorObjectiveScratch::new(date_count, realized_rank);
    let evaluation = factor_native_objective(
        prepared,
        observations,
        factor,
        maximum_rank,
        realized_rank,
        rho,
        log_process_variance,
        true,
        &mut reference_scratch,
    )?;
    Ok(AugmentedBasisRemlDerivatives {
        augmented_basis: basis,
        augmented_basis_eta: basis_eta,
        augmented_basis_eta_eta: basis_eta_eta,
        evaluation,
        score_eta: score.eta,
        score_log_q: score.log_q,
        score_eta_eta: score.eta_eta,
        score_eta_log_q: score.eta_log_q,
        score_log_q_log_q: score.log_q_log_q,
        slope_eta: slope.eta,
        slope_log_q: slope.log_q,
        profiled_eta_curvature: score.eta_eta - score.eta_log_q.powi(2) / score.log_q_log_q,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TemporalBasisMode {
    Auto,
    #[cfg(test)]
    Streamed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TemporalBasisDisposition {
    Prepared,
    #[cfg(test)]
    StreamedForced,
    StreamedTooManyGapClasses,
}

struct GapBasisDefinition {
    exponents: Vec<f64>,
    counts: Vec<usize>,
    edge_classes: Vec<usize>,
}

struct PreparedChunk {
    targets: Vec<usize>,
    scratch: BatchScratch,
    outcomes: Vec<ObjectiveResult>,
    arch: Arch,
}

struct RankBucket {
    realized_rank: usize,
    targets_per_chunk: usize,
    targets: Vec<usize>,
}

struct WorkerArena {
    chunk: PreparedChunk,
    configured_lane_count: usize,
    #[cfg(test)]
    collected: Vec<(usize, ObjectiveResult)>,
    #[cfg(test)]
    derivative_collected: Vec<(usize, DerivativeResult)>,
    profile_collected: Vec<(usize, ProfileResult, usize)>,
    lane_states: Vec<ProfileLaneState>,
    profile_active: Vec<bool>,
    lane_rhos: Vec<f64>,
    lane_log_variances: Vec<f64>,
    retained_lane_indices: Vec<usize>,
    reference_observations: Vec<f64>,
    reference_scratch: FactorObjectiveScratch,
    profile_metrics: ArenaProfileMetrics,
}

#[derive(Clone, Copy)]
struct ProfileLaneState {
    bounds: Option<crate::temporal_covariance::NuisanceBounds>,
    x: f64,
    w: f64,
    score_x: f64,
    best_rho: f64,
    best_log_variance: f64,
    best_q_gradient: f64,
    best_q_curvature: f64,
    best_eta_gradient: f64,
    best_eta_curvature: f64,
    best_eta_log_q_curvature: f64,
    best_slope_eta: f64,
    best_slope_log_q: f64,
    best: Option<FactorObjectiveEvaluation>,
    rho_passes: usize,
    optimizer_complete: bool,
    completion_round: usize,
    finalized: bool,
    failed: Option<TemporalInferenceStatus>,
}

#[derive(Default)]
struct ArenaProfileMetrics {
    microblocks_prepared: usize,
    q_objective_evaluations: usize,
    exact_optimizer_fallback_targets: usize,
    fixed_theta_dense_fallback_evaluations: usize,
    condition_upper_bound_accepts: usize,
    condition_exact_fallbacks: usize,
    compaction_events: usize,
    compacted_lane_count: usize,
    completed_lane_revisits: usize,
    rho_lane_evaluations: usize,
    covariance_parameter_derivative_lane_evaluations: usize,
}

impl ProfileLaneState {
    fn failed(status: TemporalInferenceStatus) -> Self {
        Self {
            bounds: None,
            x: 0.0,
            w: 0.0,
            score_x: f64::INFINITY,
            best_rho: 0.0,
            best_log_variance: 0.0,
            best_q_gradient: f64::NAN,
            best_q_curvature: f64::NAN,
            best_eta_gradient: f64::NAN,
            best_eta_curvature: f64::NAN,
            best_eta_log_q_curvature: f64::NAN,
            best_slope_eta: f64::NAN,
            best_slope_log_q: f64::NAN,
            best: None,
            rho_passes: 0,
            optimizer_complete: false,
            completion_round: 0,
            finalized: false,
            failed: Some(status),
        }
    }

    fn new(bounds: crate::temporal_covariance::NuisanceBounds) -> Self {
        let midpoint = (bounds.rho_lower + bounds.rho_upper) / 2.0;
        Self {
            bounds: Some(bounds),
            x: midpoint,
            w: midpoint,
            score_x: f64::INFINITY,
            best_rho: midpoint,
            best_log_variance: bounds.initial_log_variance,
            best_q_gradient: f64::NAN,
            best_q_curvature: f64::NAN,
            best_eta_gradient: f64::NAN,
            best_eta_curvature: f64::NAN,
            best_eta_log_q_curvature: f64::NAN,
            best_slope_eta: f64::NAN,
            best_slope_log_q: f64::NAN,
            best: None,
            rho_passes: 0,
            optimizer_complete: false,
            completion_round: 0,
            finalized: false,
            failed: None,
        }
    }
}

struct StaticInputs<'a> {
    prepared: &'a PreparedFactorObjective,
    observations_soa: &'a [f64],
    factors: &'a [f64],
    maximum_rank: usize,
    target_count: usize,
}

fn factor_for_target(factors: &[f64], factor_stride: usize, target: usize) -> &[f64] {
    let factor_index = if factors.len() == factor_stride {
        0
    } else {
        target
    };
    let offset = factor_index * factor_stride;
    &factors[offset..offset + factor_stride]
}

#[cfg(test)]
struct EvaluationInputs<'a> {
    prepared: &'a PreparedFactorObjective,
    factors: &'a [f64],
    maximum_rank: usize,
    rhos: &'a [f64],
    log_process_variances: &'a [f64],
    restricted: bool,
}

impl GapBasisDefinition {
    fn new(gap_exponents: &[f64]) -> Self {
        let mut exponents = Vec::<f64>::new();
        let mut counts = Vec::<usize>::new();
        let mut edge_classes = Vec::with_capacity(gap_exponents.len());
        for &exponent in gap_exponents {
            let class = exponents
                .iter()
                .position(|candidate| candidate.to_bits() == exponent.to_bits())
                .unwrap_or_else(|| {
                    exponents.push(exponent);
                    counts.push(0);
                    exponents.len() - 1
                });
            counts[class] += 1;
            edge_classes.push(class);
        }
        Self {
            exponents,
            counts,
            edge_classes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TemporalBatchExecutionMetrics {
    pub(super) theta_independent_preparations: usize,
    pub(super) objective_evaluations: usize,
    pub(super) maximum_chunk_targets: usize,
    pub(super) maximum_worker_scratch_bytes: usize,
    pub(super) basis_disposition: TemporalBasisDisposition,
    pub(super) basis_streamed_whitening_elements: usize,
    pub(super) basis_edge_transition_elements: usize,
    pub(super) worker_count: usize,
    pub(super) retained_prepared_chunk_count: usize,
    pub(super) total_retained_solver_scratch_bytes: usize,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TemporalBatchAllocationSignature {
    result: (usize, usize),
    chunks: Vec<(usize, usize, usize, usize, usize, usize)>,
}

pub(super) struct TemporalBatchExecution<'a> {
    prepared: &'a PreparedFactorObjective,
    observations_soa: &'a [f64],
    factors: &'a [f64],
    maximum_rank: usize,
    target_count: usize,
    realized_ranks: Vec<usize>,
    rank_buckets: Vec<RankBucket>,
    worker_arenas: Vec<WorkerArena>,
    gap_basis: Option<GapBasisDefinition>,
    #[cfg(test)]
    results: Vec<ObjectiveResult>,
    profile_results: Vec<ProfileResult>,
    objective_evaluations: usize,
    basis_disposition: TemporalBasisDisposition,
}

impl<'a> TemporalBatchExecution<'a> {
    pub(super) fn new(
        prepared: &'a PreparedFactorObjective,
        observations_soa: &'a [f64],
        factors: &'a [f64],
        maximum_rank: usize,
        realized_ranks: &[usize],
    ) -> Result<Self, TemporalInferenceStatus> {
        Self::new_with_basis_mode(
            prepared,
            observations_soa,
            factors,
            maximum_rank,
            realized_ranks,
            TemporalBasisMode::Auto,
        )
    }

    pub(super) fn new_with_basis_mode(
        prepared: &'a PreparedFactorObjective,
        observations_soa: &'a [f64],
        factors: &'a [f64],
        maximum_rank: usize,
        realized_ranks: &[usize],
        basis_mode: TemporalBasisMode,
    ) -> Result<Self, TemporalInferenceStatus> {
        let target_count = realized_ranks.len();
        let date_count = prepared.design.len();
        let factor_stride = date_count
            .checked_mul(maximum_rank)
            .ok_or(TemporalInferenceStatus::CovarianceNonfinite)?;
        let shared_factor = factors.len() == factor_stride;
        if target_count == 0
            || maximum_rank == 0
            || observations_soa.len() != date_count.saturating_mul(target_count)
            || (!shared_factor && factors.len() != target_count.saturating_mul(factor_stride))
            || (target_count > 1
                && shared_factor
                && realized_ranks.iter().any(|rank| *rank != realized_ranks[0]))
        {
            return Err(TemporalInferenceStatus::CovarianceNonfinite);
        }
        let gap_definition = GapBasisDefinition::new(&prepared.gap_exponents);
        let (basis_disposition, gap_basis) = match basis_mode {
            #[cfg(test)]
            TemporalBasisMode::Streamed => (TemporalBasisDisposition::StreamedForced, None),
            TemporalBasisMode::Auto if gap_definition.exponents.len() <= MAX_BASIS_GAP_CLASSES => {
                (TemporalBasisDisposition::Prepared, Some(gap_definition))
            }
            TemporalBasisMode::Auto => (TemporalBasisDisposition::StreamedTooManyGapClasses, None),
        };
        let mut targets_by_rank = BTreeMap::<usize, Vec<usize>>::new();
        for (target, &rank) in realized_ranks.iter().enumerate() {
            if rank > 0 && rank <= maximum_rank {
                targets_by_rank.entry(rank).or_default().push(target);
            }
        }
        let available_workers = rayon::current_num_threads().max(1);
        let worker_count = if rayon::current_thread_index().is_some() {
            available_workers.min(4)
        } else if target_count >= available_workers.saturating_mul(2) {
            available_workers
        } else {
            available_workers.min(target_count).max(1)
        };
        let mut rank_buckets = Vec::new();
        for (rank, targets) in targets_by_rank {
            let admitted_targets = admitted_targets_per_chunk(
                date_count,
                rank,
                gap_basis.as_ref().map_or(0, |basis| basis.exponents.len()),
            )?;
            let targets_per_chunk = admitted_targets
                .min(targets.len().div_ceil(worker_count))
                .max(1);
            rank_buckets.push(RankBucket {
                realized_rank: rank,
                targets_per_chunk,
                targets,
            });
        }
        let initial_bucket = rank_buckets
            .iter()
            .max_by_key(|bucket| bucket.realized_rank)
            .ok_or(TemporalInferenceStatus::CovarianceNonfinite)?;
        let worker_arenas = (0..worker_count)
            .map(|_| {
                WorkerArena::new(
                    date_count,
                    initial_bucket.realized_rank,
                    initial_bucket.targets_per_chunk,
                    gap_basis.as_ref(),
                )
            })
            .collect();
        Ok(Self {
            prepared,
            observations_soa,
            factors,
            maximum_rank,
            target_count,
            realized_ranks: realized_ranks.to_vec(),
            rank_buckets,
            worker_arenas,
            gap_basis,
            #[cfg(test)]
            results: vec![Err(TemporalInferenceStatus::CovarianceNonfinite); target_count],
            profile_results: vec![Err(TemporalInferenceStatus::CovarianceNonfinite); target_count],
            objective_evaluations: 0,
            basis_disposition,
        })
    }

    #[cfg(test)]
    pub(super) fn evaluate(
        &mut self,
        rhos: &[f64],
        log_process_variances: &[f64],
        restricted: bool,
    ) -> Result<&[ObjectiveResult], TemporalInferenceStatus> {
        if rhos.len() != self.target_count || log_process_variances.len() != self.target_count {
            return Err(TemporalInferenceStatus::CovarianceNonfinite);
        }
        let inputs = EvaluationInputs {
            prepared: self.prepared,
            factors: self.factors,
            maximum_rank: self.maximum_rank,
            rhos,
            log_process_variances,
            restricted,
        };
        let static_inputs = StaticInputs {
            prepared: self.prepared,
            observations_soa: self.observations_soa,
            factors: self.factors,
            maximum_rank: self.maximum_rank,
            target_count: self.target_count,
        };
        self.results
            .fill(Err(TemporalInferenceStatus::CovarianceNonfinite));
        for bucket in &self.rank_buckets {
            let chunk_count = bucket.targets.len().div_ceil(bucket.targets_per_chunk);
            let next_chunk = AtomicUsize::new(0);
            self.worker_arenas.par_iter_mut().for_each(|arena| {
                arena.collected.clear();
                arena.ensure_configuration(
                    self.prepared.design.len(),
                    bucket.realized_rank,
                    bucket.targets_per_chunk,
                    self.gap_basis.as_ref(),
                );
                loop {
                    let chunk_index = next_chunk.fetch_add(1, Ordering::Relaxed);
                    if chunk_index >= chunk_count {
                        break;
                    }
                    let first = chunk_index * bucket.targets_per_chunk;
                    let last = (first + bucket.targets_per_chunk).min(bucket.targets.len());
                    let targets = &bucket.targets[first..last];
                    arena.prepare(&static_inputs, targets);
                    process_chunk(&inputs, &mut arena.chunk);
                    arena.collected.extend(
                        arena
                            .chunk
                            .targets
                            .iter()
                            .copied()
                            .zip(arena.chunk.outcomes.iter().copied()),
                    );
                }
            });
            for arena in &self.worker_arenas {
                for &(target, outcome) in &arena.collected {
                    self.results[target] = outcome;
                }
            }
        }
        self.objective_evaluations += 1;
        Ok(&self.results)
    }

    #[cfg(test)]
    fn evaluate_reml_derivatives(
        &mut self,
        rhos: &[f64],
        log_process_variances: &[f64],
    ) -> Result<Vec<DerivativeResult>, TemporalInferenceStatus> {
        if rhos.len() != self.target_count || log_process_variances.len() != self.target_count {
            return Err(TemporalInferenceStatus::CovarianceNonfinite);
        }
        let inputs = EvaluationInputs {
            prepared: self.prepared,
            factors: self.factors,
            maximum_rank: self.maximum_rank,
            rhos,
            log_process_variances,
            restricted: true,
        };
        let static_inputs = StaticInputs {
            prepared: self.prepared,
            observations_soa: self.observations_soa,
            factors: self.factors,
            maximum_rank: self.maximum_rank,
            target_count: self.target_count,
        };
        let mut results =
            vec![Err(TemporalInferenceStatus::CovarianceNonfinite); self.target_count];
        for bucket in &self.rank_buckets {
            let chunk_count = bucket.targets.len().div_ceil(bucket.targets_per_chunk);
            let next_chunk = AtomicUsize::new(0);
            self.worker_arenas.par_iter_mut().for_each(|arena| {
                arena.derivative_collected.clear();
                arena.ensure_configuration(
                    self.prepared.design.len(),
                    bucket.realized_rank,
                    bucket.targets_per_chunk,
                    self.gap_basis.as_ref(),
                );
                loop {
                    let chunk_index = next_chunk.fetch_add(1, Ordering::Relaxed);
                    if chunk_index >= chunk_count {
                        break;
                    }
                    let first = chunk_index * bucket.targets_per_chunk;
                    let last = (first + bucket.targets_per_chunk).min(bucket.targets.len());
                    let targets = &bucket.targets[first..last];
                    arena.prepare(&static_inputs, targets);
                    process_chunk_internal(&inputs, &mut arena.chunk, true);
                    for (lane, &target) in arena.chunk.targets.iter().enumerate() {
                        let result = match arena.chunk.outcomes[lane] {
                            Err(status) => Err(status),
                            Ok(_) if !(0.0..1.0).contains(&rhos[target]) => {
                                Err(TemporalInferenceStatus::CovarianceParameterAtBoundary)
                            }
                            Ok(evaluation) => {
                                let scratch = &arena.chunk.scratch;
                                let derivative = BatchRemlDerivativeEvaluation {
                                    evaluation,
                                    score_eta: scratch.score_gradient_eta[lane],
                                    score_log_q: scratch.score_gradient_log_q[lane],
                                    score_eta_eta: scratch.score_curvature_eta[lane],
                                    score_eta_log_q: scratch.score_curvature_eta_log_q[lane],
                                    score_log_q_log_q: scratch.score_curvature_log_q[lane],
                                    slope_eta: scratch.slope_gradient_eta[lane],
                                    slope_log_q: scratch.slope_gradient_log_q[lane],
                                };
                                if derivative.score_eta.is_finite()
                                    && derivative.score_log_q.is_finite()
                                    && derivative.score_eta_eta.is_finite()
                                    && derivative.score_eta_log_q.is_finite()
                                    && derivative.score_log_q_log_q.is_finite()
                                    && derivative.slope_eta.is_finite()
                                    && derivative.slope_log_q.is_finite()
                                {
                                    Ok(derivative)
                                } else {
                                    Err(TemporalInferenceStatus::CovarianceNonfinite)
                                }
                            }
                        };
                        arena.derivative_collected.push((target, result));
                    }
                }
            });
            for arena in &self.worker_arenas {
                for &(target, outcome) in &arena.derivative_collected {
                    results[target] = outcome;
                }
            }
        }
        Ok(results)
    }

    pub(super) fn profile_reml(
        &mut self,
        options: &TemporalCovarianceOptions,
        materialize_adjustment: bool,
        accept_boundary_solution: bool,
    ) -> Result<TemporalBatchProfileReport<'_>, TemporalInferenceStatus> {
        let static_inputs = StaticInputs {
            prepared: self.prepared,
            observations_soa: self.observations_soa,
            factors: self.factors,
            maximum_rank: self.maximum_rank,
            target_count: self.target_count,
        };
        self.profile_results
            .fill(Err(TemporalInferenceStatus::CovarianceNonfinite));
        let mut metrics = TemporalBatchProfileMetrics {
            microblocks_prepared: 0,
            primary_rho_pass_histogram: [0; 21],
            q_objective_evaluations: 0,
            maximum_primary_rho_passes: 0,
            exact_optimizer_fallback_targets: 0,
            fixed_theta_dense_fallback_evaluations: 0,
            condition_upper_bound_accepts: 0,
            condition_exact_fallbacks: 0,
            compaction_events: 0,
            compacted_lane_count: 0,
            completed_lane_revisits: 0,
            rho_lane_evaluations: 0,
            covariance_parameter_derivative_lane_evaluations: 0,
            per_target_rho_passes: vec![0; self.target_count],
            maximum_worker_scratch_bytes: 0,
        };
        for bucket in &self.rank_buckets {
            let chunk_count = bucket.targets.len().div_ceil(bucket.targets_per_chunk);
            let next_chunk = AtomicUsize::new(0);
            self.worker_arenas.par_iter_mut().for_each(|arena| {
                arena.profile_collected.clear();
                arena.profile_metrics = ArenaProfileMetrics::default();
                arena.ensure_configuration(
                    self.prepared.design.len(),
                    bucket.realized_rank,
                    bucket.targets_per_chunk,
                    self.gap_basis.as_ref(),
                );
                loop {
                    let chunk_index = next_chunk.fetch_add(1, Ordering::Relaxed);
                    if chunk_index >= chunk_count {
                        break;
                    }
                    let first = chunk_index * bucket.targets_per_chunk;
                    let last = (first + bucket.targets_per_chunk).min(bucket.targets.len());
                    let targets = &bucket.targets[first..last];
                    arena.prepare(&static_inputs, targets);
                    profile_microblock(
                        self.prepared,
                        self.factors,
                        self.maximum_rank,
                        options,
                        materialize_adjustment,
                        accept_boundary_solution,
                        arena,
                    );
                }
            });
            for arena in &self.worker_arenas {
                metrics.microblocks_prepared += arena.profile_metrics.microblocks_prepared;
                metrics.q_objective_evaluations += arena.profile_metrics.q_objective_evaluations;
                metrics.exact_optimizer_fallback_targets +=
                    arena.profile_metrics.exact_optimizer_fallback_targets;
                metrics.fixed_theta_dense_fallback_evaluations +=
                    arena.profile_metrics.fixed_theta_dense_fallback_evaluations;
                metrics.condition_upper_bound_accepts +=
                    arena.profile_metrics.condition_upper_bound_accepts;
                metrics.condition_exact_fallbacks +=
                    arena.profile_metrics.condition_exact_fallbacks;
                metrics.compaction_events += arena.profile_metrics.compaction_events;
                metrics.compacted_lane_count += arena.profile_metrics.compacted_lane_count;
                metrics.completed_lane_revisits += arena.profile_metrics.completed_lane_revisits;
                metrics.rho_lane_evaluations += arena.profile_metrics.rho_lane_evaluations;
                metrics.covariance_parameter_derivative_lane_evaluations += arena
                    .profile_metrics
                    .covariance_parameter_derivative_lane_evaluations;
                metrics.maximum_worker_scratch_bytes = metrics
                    .maximum_worker_scratch_bytes
                    .max(arena.chunk.scratch.allocated_bytes());
                for &(target, outcome, passes) in &arena.profile_collected {
                    self.profile_results[target] = outcome;
                    metrics.per_target_rho_passes[target] = passes;
                }
            }
        }
        for (&rank, &passes) in self
            .realized_ranks
            .iter()
            .zip(&metrics.per_target_rho_passes)
        {
            if rank == 0 {
                continue;
            }
            metrics.primary_rho_pass_histogram[passes.min(20)] += 1;
            metrics.maximum_primary_rho_passes = metrics.maximum_primary_rho_passes.max(passes);
        }
        Ok(TemporalBatchProfileReport {
            outcomes: &self.profile_results,
            metrics,
        })
    }

    pub(super) fn metrics(&self) -> TemporalBatchExecutionMetrics {
        TemporalBatchExecutionMetrics {
            theta_independent_preparations: 1,
            objective_evaluations: self.objective_evaluations,
            maximum_chunk_targets: self
                .rank_buckets
                .iter()
                .map(|bucket| bucket.targets_per_chunk.min(bucket.targets.len()))
                .max()
                .unwrap_or(0),
            maximum_worker_scratch_bytes: self
                .worker_arenas
                .iter()
                .map(|arena| arena.chunk.scratch.allocated_bytes())
                .max()
                .unwrap_or(0),
            basis_disposition: self.basis_disposition,
            basis_streamed_whitening_elements: self
                .worker_arenas
                .iter()
                .filter(|arena| arena.chunk.scratch.basis_enabled)
                .map(|arena| {
                    arena.chunk.scratch.whitened_factor.len()
                        + arena.chunk.scratch.whitened_x.len()
                        + arena.chunk.scratch.whitened_y.len()
                })
                .sum(),
            basis_edge_transition_elements: self
                .worker_arenas
                .iter()
                .filter(|arena| arena.chunk.scratch.basis_enabled)
                .map(|arena| {
                    arena.chunk.scratch.transition.len()
                        + arena.chunk.scratch.inverse_innovation_scale.len()
                })
                .sum(),
            worker_count: self.worker_arenas.len(),
            retained_prepared_chunk_count: 0,
            total_retained_solver_scratch_bytes: self
                .worker_arenas
                .iter()
                .map(|arena| arena.chunk.scratch.allocated_bytes())
                .sum(),
        }
    }

    #[cfg(test)]
    pub(super) fn allocation_signature(&self) -> TemporalBatchAllocationSignature {
        TemporalBatchAllocationSignature {
            result: (self.results.as_ptr() as usize, self.results.capacity()),
            chunks: self
                .worker_arenas
                .iter()
                .map(|arena| {
                    (
                        arena.chunk.targets.as_ptr() as usize,
                        arena.chunk.targets.capacity(),
                        arena.chunk.scratch.scaled_factor.as_ptr() as usize,
                        arena.chunk.scratch.scaled_factor.capacity(),
                        arena.chunk.outcomes.as_ptr() as usize,
                        arena.chunk.outcomes.capacity(),
                    )
                })
                .collect(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn factor_native_objective_batch(
    prepared: &PreparedFactorObjective,
    observations_soa: &[f64],
    factors: &[f64],
    maximum_rank: usize,
    realized_ranks: &[usize],
    rhos: &[f64],
    log_process_variances: &[f64],
    restricted: bool,
) -> Result<Vec<ObjectiveResult>, TemporalInferenceStatus> {
    let mut execution = TemporalBatchExecution::new_with_basis_mode(
        prepared,
        observations_soa,
        factors,
        maximum_rank,
        realized_ranks,
        TemporalBasisMode::Streamed,
    )?;
    Ok(execution
        .evaluate(rhos, log_process_variances, restricted)?
        .to_vec())
}

#[cfg(test)]
pub(super) fn factor_native_reml_derivative_batch(
    prepared: &PreparedFactorObjective,
    observations_soa: &[f64],
    factors: &[f64],
    maximum_rank: usize,
    realized_ranks: &[usize],
    rhos: &[f64],
    log_process_variances: &[f64],
) -> Result<Vec<DerivativeResult>, TemporalInferenceStatus> {
    let mut execution = TemporalBatchExecution::new(
        prepared,
        observations_soa,
        factors,
        maximum_rank,
        realized_ranks,
    )?;
    execution.evaluate_reml_derivatives(rhos, log_process_variances)
}

fn admitted_targets_per_chunk(
    date_count: usize,
    realized_rank: usize,
    basis_gap_classes: usize,
) -> Result<usize, TemporalInferenceStatus> {
    let triangular = realized_rank
        .checked_mul(realized_rank + 1)
        .and_then(|value| value.checked_div(2))
        .ok_or(TemporalInferenceStatus::CovarianceNonfinite)?;
    let basis_dimension = realized_rank.saturating_add(2);
    let basis_triangular = basis_dimension.saturating_mul(basis_dimension + 1) / 2;
    let basis_enabled = basis_gap_classes > 0;
    let values_per_target = if basis_enabled {
        date_count
            .checked_mul(realized_rank)
            .and_then(|value| value.checked_add(date_count.saturating_mul(4)))
            .and_then(|value| value.checked_add(triangular))
            .and_then(|value| value.checked_add(realized_rank.saturating_mul(4)))
            .and_then(|value| value.checked_add(basis_gap_classes.saturating_mul(7)))
            .and_then(|value| {
                value.checked_add(
                    basis_triangular.saturating_mul(10 + basis_gap_classes.saturating_mul(3)),
                )
            })
            .and_then(|value| value.checked_add(18))
            .and_then(|value| value.checked_add(23))
    } else {
        date_count
            .checked_mul(realized_rank)
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_add(date_count.saturating_mul(6)))
            .and_then(|value| value.checked_add(date_count.saturating_sub(1).saturating_mul(2)))
            .and_then(|value| value.checked_add(triangular))
            .and_then(|value| value.checked_add(realized_rank.saturating_mul(4)))
            .and_then(|value| value.checked_add(basis_triangular.saturating_mul(3)))
            .and_then(|value| value.checked_add(23))
    }
    .ok_or(TemporalInferenceStatus::CovarianceNonfinite)?;
    let bytes_per_target = values_per_target
        .checked_mul(std::mem::size_of::<f64>())
        .and_then(|value| value.checked_add(3 * std::mem::size_of::<bool>()))
        .ok_or(TemporalInferenceStatus::CovarianceNonfinite)?;
    let fixed_bytes = date_count
        .checked_add(basis_gap_classes)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
        .and_then(|value| {
            value.checked_add(
                basis_gap_classes
                    .saturating_add(date_count.saturating_sub(1))
                    .saturating_mul(std::mem::size_of::<usize>()),
            )
        })
        .ok_or(TemporalInferenceStatus::CovarianceNonfinite)?;
    let design_width = if date_count <= 12 { 256 } else { 128 };
    let admitted = (TEMPORAL_FACTOR_SCALAR_MAX_WORKER_SCRATCH_BYTES.saturating_sub(fixed_bytes)
        / bytes_per_target)
        .min(design_width);
    if admitted == 0 {
        return Err(TemporalInferenceStatus::CovarianceNonfinite);
    }
    Ok(admitted)
}

impl PreparedChunk {
    fn empty(
        date_count: usize,
        realized_rank: usize,
        lane_count: usize,
        gap_basis: Option<&GapBasisDefinition>,
    ) -> Self {
        Self {
            targets: Vec::with_capacity(lane_count),
            scratch: BatchScratch::new(date_count, realized_rank, lane_count, gap_basis),
            outcomes: Vec::with_capacity(lane_count),
            arch: Arch::new(),
        }
    }
}

impl WorkerArena {
    fn new(
        date_count: usize,
        realized_rank: usize,
        lane_count: usize,
        gap_basis: Option<&GapBasisDefinition>,
    ) -> Self {
        Self {
            chunk: PreparedChunk::empty(date_count, realized_rank, lane_count, gap_basis),
            configured_lane_count: lane_count,
            #[cfg(test)]
            collected: Vec::new(),
            #[cfg(test)]
            derivative_collected: Vec::new(),
            profile_collected: Vec::new(),
            lane_states: Vec::with_capacity(lane_count),
            profile_active: vec![false; lane_count],
            lane_rhos: vec![0.0; lane_count],
            lane_log_variances: vec![0.0; lane_count],
            retained_lane_indices: Vec::with_capacity(lane_count),
            reference_observations: vec![0.0; date_count],
            reference_scratch: FactorObjectiveScratch::new(date_count, realized_rank),
            profile_metrics: ArenaProfileMetrics::default(),
        }
    }

    fn ensure_configuration(
        &mut self,
        date_count: usize,
        realized_rank: usize,
        lane_count: usize,
        gap_basis: Option<&GapBasisDefinition>,
    ) {
        let basis_enabled = gap_basis.is_some();
        if self.chunk.scratch.date_count != date_count
            || self.chunk.scratch.realized_rank != realized_rank
            || self.configured_lane_count != lane_count
            || self.chunk.scratch.basis_enabled != basis_enabled
        {
            self.chunk = PreparedChunk::empty(date_count, realized_rank, lane_count, gap_basis);
            self.lane_states = Vec::with_capacity(lane_count);
            self.profile_active.resize(lane_count, false);
            self.lane_rhos.resize(lane_count, 0.0);
            self.lane_log_variances.resize(lane_count, 0.0);
            self.reference_observations.resize(date_count, 0.0);
            self.reference_scratch = FactorObjectiveScratch::new(date_count, realized_rank);
        }
        self.configured_lane_count = lane_count;
    }

    fn prepare(&mut self, inputs: &StaticInputs<'_>, targets: &[usize]) {
        self.chunk
            .scratch
            .restore_lane_count(self.configured_lane_count);
        self.chunk.targets.clear();
        self.chunk.targets.extend_from_slice(targets);
        self.chunk.outcomes.clear();
        self.chunk.outcomes.resize(
            targets.len(),
            Err(TemporalInferenceStatus::CovarianceNonfinite),
        );
        self.chunk.scratch.reset_static();
        for (lane, &target) in targets.iter().enumerate() {
            prepare_static_lane(inputs, target, lane, &mut self.chunk.scratch);
        }
        self.chunk.scratch.prepare_basis();
    }

    fn initialize_profile_states(
        &mut self,
        prepared: &PreparedFactorObjective,
        options: &TemporalCovarianceOptions,
    ) {
        let lane_count = self.chunk.targets.len();
        self.lane_states.clear();
        self.profile_active.fill(false);
        for lane in 0..lane_count {
            if !self.chunk.scratch.static_valid[lane] {
                self.lane_states.push(ProfileLaneState::failed(
                    TemporalInferenceStatus::CovarianceNonfinite,
                ));
                continue;
            }
            for date in 0..self.chunk.scratch.date_count {
                self.reference_observations[date] =
                    self.chunk.scratch.observations[date * self.chunk.scratch.lane_count + lane];
            }
            match nuisance_bounds(&prepared.design, &self.reference_observations, options) {
                Ok(bounds) => {
                    self.lane_states.push(ProfileLaneState::new(bounds));
                    self.profile_active[lane] = true;
                }
                Err(status) => self.lane_states.push(ProfileLaneState::failed(status)),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn exact_profile_reference(
    prepared: &PreparedFactorObjective,
    factors: &[f64],
    maximum_rank: usize,
    options: &TemporalCovarianceOptions,
    materialize_adjustment: bool,
    accept_boundary_solution: bool,
    arena: &mut WorkerArena,
    lane: usize,
) -> ProfileResult {
    let target = arena.chunk.targets[lane];
    let factor_stride = prepared.design.len() * maximum_rank;
    for date in 0..prepared.design.len() {
        arena.reference_observations[date] =
            arena.chunk.scratch.observations[date * arena.chunk.scratch.lane_count + lane];
    }
    let bounds = nuisance_bounds(&prepared.design, &arena.reference_observations, options)?;
    let factor = factor_for_target(factors, factor_stride, target);
    let realized_rank = arena.chunk.scratch.realized_rank;
    arena.profile_metrics.exact_optimizer_fallback_targets += 1;
    let fit = factor_native_profile_plugin(
        prepared,
        &arena.reference_observations,
        factor,
        maximum_rank,
        realized_rank,
        options,
        &mut arena.reference_scratch,
        accept_boundary_solution,
    )?;
    arena.profile_metrics.fixed_theta_dense_fallback_evaluations += fit.dense_fallback_count;
    let covariance_parameter_adjusted_variance = if materialize_adjustment {
        let difference_covariance = difference_covariance_from_factor(
            prepared.design.len(),
            factor,
            maximum_rank,
            realized_rank,
        );
        reml_covariance_parameter_adjusted_variance(
            &prepared.design,
            &arena.reference_observations,
            &difference_covariance,
            fit.rho,
            fit.process_variance,
            fit.information_variance,
            options,
        )
        .ok()
    } else {
        None
    };
    Ok(TemporalBatchProfileEvaluation {
        #[cfg(test)]
        score: fit.score,
        slope: fit.slope,
        rho: fit.rho,
        process_variance: fit.process_variance,
        fitted_parameter_active_set: temporal_parameter_boundary_status(
            fit.rho,
            fit.process_variance,
            [bounds.rho_lower, bounds.rho_upper],
            [
                bounds.log_variance_lower.exp(),
                bounds.log_variance_upper.exp(),
            ],
            options.optimizer_tolerance * 0.01,
        ),
        information_variance: fit.information_variance,
        covariance_parameter_adjusted_variance,
        #[cfg(test)]
        profile_rho_curvature: f64::NAN,
        condition_upper_bound: f64::INFINITY,
        exact_condition_number: Some(fit.condition_number),
    })
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn finalize_profile_lane(
    prepared: &PreparedFactorObjective,
    factors: &[f64],
    maximum_rank: usize,
    options: &TemporalCovarianceOptions,
    materialize_adjustment: bool,
    accept_boundary_solution: bool,
    arena: &mut WorkerArena,
    lane: usize,
) -> ProfileResult {
    let state = arena.lane_states[lane];
    let boundary_best = state.bounds.is_some_and(|bounds| {
        state.best.is_some()
            && temporal_parameter_boundary_status(
                state.best_rho,
                state.best_log_variance.exp(),
                [bounds.rho_lower, bounds.rho_upper],
                [
                    bounds.log_variance_lower.exp(),
                    bounds.log_variance_upper.exp(),
                ],
                options.optimizer_tolerance * 0.01,
            )
            .is_some()
    });
    let boundary_failure = matches!(
        state.failed,
        Some(TemporalInferenceStatus::CovarianceParameterAtBoundary)
            | Some(TemporalInferenceStatus::RhoLowerBoundary)
            | Some(TemporalInferenceStatus::RhoUpperBoundary)
            | Some(TemporalInferenceStatus::ProcessVarianceLowerBoundary)
            | Some(TemporalInferenceStatus::ProcessVarianceUpperBoundary)
    );
    let accept_bounded_best =
        accept_boundary_solution && boundary_best && (state.failed.is_none() || boundary_failure);
    let exact_fallback_status = matches!(
        state.failed,
        Some(TemporalInferenceStatus::OptimizerNonconverged)
            | Some(TemporalInferenceStatus::CovarianceNonfinite)
            | Some(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite)
            | Some(TemporalInferenceStatus::DesignRankDeficient)
            | Some(TemporalInferenceStatus::CovarianceParameterAtBoundary)
            | Some(TemporalInferenceStatus::RhoLowerBoundary)
            | Some(TemporalInferenceStatus::RhoUpperBoundary)
            | Some(TemporalInferenceStatus::ProcessVarianceLowerBoundary)
            | Some(TemporalInferenceStatus::ProcessVarianceUpperBoundary)
            | Some(TemporalInferenceStatus::WeakParameterIdentification)
    );
    let weak_identification_fallback = matches!(
        state.failed,
        Some(TemporalInferenceStatus::WeakParameterIdentification)
    );
    if ((state.rho_passes > 0 && exact_fallback_status) || weak_identification_fallback)
        && !accept_bounded_best
    {
        return exact_profile_reference(
            prepared,
            factors,
            maximum_rank,
            options,
            materialize_adjustment,
            accept_boundary_solution,
            arena,
            lane,
        );
    }
    if let Some(status) = state.failed.filter(|_| !accept_bounded_best) {
        return Err(status);
    }
    let best = state
        .best
        .ok_or(TemporalInferenceStatus::OptimizerNonconverged)?;
    let bounds = state
        .bounds
        .ok_or(TemporalInferenceStatus::OptimizerNonconverged)?;
    let target = arena.chunk.targets[lane];
    let factor_stride = prepared.design.len() * maximum_rank;
    let factor = factor_for_target(factors, factor_stride, target);
    let certificate = match factor_condition_certificate(
        prepared,
        factor,
        maximum_rank,
        arena.chunk.scratch.realized_rank,
        state.best_rho,
        state.best_log_variance.exp(),
        options.condition_limit,
    ) {
        Ok(certificate) => certificate,
        Err(status) => {
            arena.profile_metrics.condition_exact_fallbacks += 1;
            return Err(status);
        }
    };
    match certificate.method {
        FactorConditionMethod::ConservativeUpperBound => {
            arena.profile_metrics.condition_upper_bound_accepts += 1;
        }
        FactorConditionMethod::ExactEigenvalueFallback => {
            arena.profile_metrics.condition_exact_fallbacks += 1;
        }
    }
    if !accept_boundary_solution
        && temporal_parameter_boundary_status(
            state.best_rho,
            state.best_log_variance.exp(),
            [bounds.rho_lower, bounds.rho_upper],
            [
                bounds.log_variance_lower.exp(),
                bounds.log_variance_upper.exp(),
            ],
            options.optimizer_tolerance * 0.01,
        )
        .is_some()
    {
        return exact_profile_reference(
            prepared,
            factors,
            maximum_rank,
            options,
            materialize_adjustment,
            accept_boundary_solution,
            arena,
            lane,
        );
    }
    let information_variance = 1.0 / best.x_v_x;
    let (fitted_parameter_active_set, rho_active, variance_active) = nuisance_parameter_active_set(
        state.best_rho,
        state.best_log_variance.exp(),
        [bounds.rho_lower, bounds.rho_upper],
        [
            bounds.log_variance_lower.exp(),
            bounds.log_variance_upper.exp(),
        ],
        options.optimizer_tolerance * 0.01,
    );
    if !materialize_adjustment {
        return Ok(TemporalBatchProfileEvaluation {
            #[cfg(test)]
            score: best.score,
            slope: best.slope,
            rho: state.best_rho,
            process_variance: state.best_log_variance.exp(),
            fitted_parameter_active_set,
            information_variance,
            covariance_parameter_adjusted_variance: None,
            #[cfg(test)]
            profile_rho_curvature: f64::NAN,
            condition_upper_bound: certificate.conservative_upper_bound,
            exact_condition_number: certificate.exact_condition_number,
        });
    }
    let (_rho_curvature, nuisance_variance) = if fitted_parameter_active_set
        == Some(TemporalInferenceStatus::ProcessVarianceLowerBoundary)
    {
        (f64::NAN, 0.0)
    } else {
        match (rho_active, variance_active) {
            (true, true) => (f64::NAN, 0.0),
            (true, false) => {
                if !state.best_q_curvature.is_finite()
                    || state.best_q_curvature <= options.minimum_profile_curvature
                {
                    return Err(TemporalInferenceStatus::WeakParameterIdentification);
                }
                (
                    f64::NAN,
                    2.0 * state.best_slope_log_q.powi(2) / state.best_q_curvature,
                )
            }
            (false, true) => {
                if !state.best_eta_curvature.is_finite()
                    || state.best_eta_curvature <= options.minimum_profile_curvature
                {
                    return Err(TemporalInferenceStatus::WeakParameterIdentification);
                }
                (
                    (state.best_eta_curvature - state.best_eta_gradient) / state.best_rho.powi(2),
                    2.0 * state.best_slope_eta.powi(2) / state.best_eta_curvature,
                )
            }
            (false, false) => {
                let profile_eta_curvature = state.best_eta_curvature
                    - state.best_eta_log_q_curvature.powi(2) / state.best_q_curvature;
                let rho_curvature =
                    (profile_eta_curvature - state.best_eta_gradient) / state.best_rho.powi(2);
                let determinant = state.best_eta_curvature * state.best_q_curvature
                    - state.best_eta_log_q_curvature.powi(2);
                if !rho_curvature.is_finite()
                    || !state.best_q_curvature.is_finite()
                    || !determinant.is_finite()
                    || rho_curvature <= options.minimum_profile_curvature
                    || state.best_q_curvature <= options.minimum_profile_curvature
                    || determinant <= 0.0
                {
                    return Err(TemporalInferenceStatus::WeakParameterIdentification);
                }
                (
                    rho_curvature,
                    2.0 * (state.best_q_curvature * state.best_slope_eta.powi(2)
                        - 2.0
                            * state.best_eta_log_q_curvature
                            * state.best_slope_eta
                            * state.best_slope_log_q
                        + state.best_eta_curvature * state.best_slope_log_q.powi(2))
                        / determinant,
                )
            }
        }
    };
    let adjusted_variance = information_variance + nuisance_variance.max(0.0);
    if !nuisance_variance.is_finite() || !adjusted_variance.is_finite() || adjusted_variance <= 0.0
    {
        return Err(TemporalInferenceStatus::WeakParameterIdentification);
    }
    Ok(TemporalBatchProfileEvaluation {
        #[cfg(test)]
        score: best.score,
        slope: best.slope,
        rho: state.best_rho,
        process_variance: state.best_log_variance.exp(),
        fitted_parameter_active_set,
        information_variance,
        covariance_parameter_adjusted_variance: Some(adjusted_variance),
        #[cfg(test)]
        profile_rho_curvature: _rho_curvature,
        condition_upper_bound: certificate.conservative_upper_bound,
        exact_condition_number: certificate.exact_condition_number,
    })
}

fn finalize_terminal_lanes(
    prepared: &PreparedFactorObjective,
    factors: &[f64],
    maximum_rank: usize,
    options: &TemporalCovarianceOptions,
    materialize_adjustment: bool,
    accept_boundary_solution: bool,
    arena: &mut WorkerArena,
) {
    for lane in 0..arena.chunk.targets.len() {
        if arena.lane_states[lane].finalized
            || (arena.lane_states[lane].failed.is_none()
                && !arena.lane_states[lane].optimizer_complete)
        {
            continue;
        }
        let target = arena.chunk.targets[lane];
        let passes = arena.lane_states[lane].rho_passes;
        let outcome = finalize_profile_lane(
            prepared,
            factors,
            maximum_rank,
            options,
            materialize_adjustment,
            accept_boundary_solution,
            arena,
            lane,
        );
        arena.profile_collected.push((target, outcome, passes));
        arena.lane_states[lane].finalized = true;
    }
}

#[allow(clippy::too_many_lines)]
fn profile_microblock(
    prepared: &PreparedFactorObjective,
    factors: &[f64],
    maximum_rank: usize,
    options: &TemporalCovarianceOptions,
    materialize_adjustment: bool,
    accept_boundary_solution: bool,
    arena: &mut WorkerArena,
) {
    arena.profile_metrics.microblocks_prepared += 1;
    arena.initialize_profile_states(prepared, options);
    finalize_terminal_lanes(
        prepared,
        factors,
        maximum_rank,
        options,
        materialize_adjustment,
        accept_boundary_solution,
        arena,
    );
    arena.compact_unfinished_lanes();

    for lane in 0..arena.chunk.targets.len() {
        let state = &mut arena.lane_states[lane];
        let Some(bounds) = state.bounds else {
            continue;
        };
        let rho = ((bounds.rho_lower + bounds.rho_upper) / 2.0)
            .max(f64::EPSILON.sqrt())
            .min(bounds.rho_upper - f64::EPSILON.sqrt());
        state.x = rho.ln();
        state.w = bounds.initial_log_variance;
        state.score_x = f64::INFINITY;
        arena.lane_rhos[lane] = rho;
        arena.lane_log_variances[lane] = bounds.initial_log_variance;
    }

    for pass in 1..=13 {
        let lane_count = arena.chunk.targets.len();
        let mut active_count = 0;
        for lane in 0..lane_count {
            let state = &mut arena.lane_states[lane];
            let active = !state.finalized && state.failed.is_none() && !state.optimizer_complete;
            arena.profile_active[lane] = active;
            if active {
                active_count += 1;
                state.rho_passes += 1;
                arena.profile_metrics.rho_lane_evaluations += 1;
            }
        }
        if active_count == 0 {
            break;
        }
        process_profile_chunk(
            prepared,
            factors,
            maximum_rank,
            &arena.lane_rhos[..lane_count],
            &arena.lane_log_variances[..lane_count],
            &arena.profile_active[..lane_count],
            materialize_adjustment,
            &mut arena.chunk,
        );
        for lane in 0..lane_count {
            if !arena.profile_active[lane] {
                continue;
            }
            arena.profile_metrics.q_objective_evaluations += 1;
            let evaluation = match arena.chunk.outcomes[lane] {
                Ok(evaluation) if !evaluation.dense_fallback_used => {
                    if materialize_adjustment {
                        arena
                            .profile_metrics
                            .covariance_parameter_derivative_lane_evaluations += 1;
                    }
                    evaluation
                }
                Ok(_) => {
                    arena.profile_metrics.fixed_theta_dense_fallback_evaluations += 1;
                    arena.lane_states[lane].failed =
                        Some(TemporalInferenceStatus::OptimizerNonconverged);
                    continue;
                }
                Err(_) => {
                    arena.lane_states[lane].failed =
                        Some(TemporalInferenceStatus::OptimizerNonconverged);
                    continue;
                }
            };
            let gradient_eta = arena.chunk.scratch.score_gradient_eta[lane];
            let gradient_q = arena.chunk.scratch.score_gradient_log_q[lane];
            let curvature_eta = arena.chunk.scratch.score_curvature_eta[lane];
            let curvature_eta_q = arena.chunk.scratch.score_curvature_eta_log_q[lane];
            let curvature_q = arena.chunk.scratch.score_curvature_log_q[lane];
            let slope_derivatives = materialize_adjustment.then(|| {
                (
                    arena.chunk.scratch.slope_gradient_eta[lane],
                    arena.chunk.scratch.slope_gradient_log_q[lane],
                )
            });
            if !gradient_eta.is_finite()
                || !gradient_q.is_finite()
                || !curvature_eta.is_finite()
                || !curvature_eta_q.is_finite()
                || !curvature_q.is_finite()
                || slope_derivatives.is_some_and(|(slope_eta, slope_log_q)| {
                    !slope_eta.is_finite() || !slope_log_q.is_finite()
                })
            {
                arena.lane_states[lane].failed =
                    Some(TemporalInferenceStatus::OptimizerNonconverged);
                continue;
            }
            let current_eta = arena.lane_rhos[lane].ln();
            let current_q = arena.lane_log_variances[lane];
            let state = &mut arena.lane_states[lane];
            let score_tolerance = options.optimizer_tolerance * (1.0 + evaluation.score.abs());
            if state.score_x.is_finite() && evaluation.score > state.score_x + score_tolerance {
                arena.lane_rhos[lane] = ((current_eta + state.x) / 2.0).exp();
                arena.lane_log_variances[lane] = (current_q + state.w) / 2.0;
                continue;
            }

            state.score_x = evaluation.score;
            state.x = current_eta;
            state.w = current_q;
            if state.best.is_none_or(|best| evaluation.score < best.score) {
                state.best = Some(evaluation);
                state.best_rho = arena.lane_rhos[lane];
                state.best_log_variance = current_q;
                state.best_q_gradient = gradient_q;
                state.best_q_curvature = curvature_q;
                state.best_eta_gradient = gradient_eta;
                state.best_eta_curvature = curvature_eta;
                state.best_eta_log_q_curvature = curvature_eta_q;
                if let Some((slope_eta, slope_log_q)) = slope_derivatives {
                    state.best_slope_eta = slope_eta;
                    state.best_slope_log_q = slope_log_q;
                }
            }

            let profile_eta_curvature = curvature_eta - curvature_eta_q.powi(2) / curvature_q;
            let determinant = curvature_eta * curvature_q - curvature_eta_q.powi(2);
            let positive_profile = curvature_q > 0.0
                && profile_eta_curvature > 0.0
                && determinant.is_finite()
                && determinant > 0.0;
            let (newton_eta_step, newton_q_step) = if positive_profile {
                (
                    (-curvature_q * gradient_eta + curvature_eta_q * gradient_q) / determinant,
                    (curvature_eta_q * gradient_eta - curvature_eta * gradient_q) / determinant,
                )
            } else {
                (f64::INFINITY, f64::INFINITY)
            };
            let joint_newton_decrement =
                -0.5 * (gradient_eta * newton_eta_step + gradient_q * newton_q_step);
            let eta_tolerance = options.optimizer_tolerance * (1.0 + current_eta.abs());
            let q_tolerance = options.optimizer_tolerance * (1.0 + current_q.abs());
            let converged = positive_profile
                && joint_newton_decrement.is_finite()
                && joint_newton_decrement >= 0.0
                && joint_newton_decrement <= score_tolerance
                && newton_eta_step.abs() <= eta_tolerance
                && newton_q_step.abs() <= q_tolerance;
            if converged {
                state.optimizer_complete = true;
                state.completion_round = pass;
                continue;
            }

            let (mut eta_step, mut q_step) = if positive_profile {
                (newton_eta_step, newton_q_step)
            } else {
                (
                    -0.5 * gradient_eta.signum(),
                    if curvature_q > 0.0 {
                        -gradient_q / curvature_q
                    } else {
                        -0.5 * gradient_q.signum()
                    },
                )
            };
            eta_step = eta_step.clamp(-1.0, 1.0);
            q_step = q_step.clamp(-2.0, 2.0);
            let bounds = state.bounds.expect("active Newton lane has bounds");
            let rho_lower = bounds.rho_lower.max(f64::EPSILON.sqrt());
            let rho_upper = bounds.rho_upper - f64::EPSILON.sqrt();
            let next_eta = (current_eta + eta_step).clamp(rho_lower.ln(), rho_upper.ln());
            let next_q =
                (current_q + q_step).clamp(bounds.log_variance_lower, bounds.log_variance_upper);
            if !next_eta.is_finite() || !next_q.is_finite() {
                state.failed = Some(TemporalInferenceStatus::OptimizerNonconverged);
                continue;
            }
            arena.lane_rhos[lane] = next_eta.exp();
            arena.lane_log_variances[lane] = next_q;
        }
        finalize_terminal_lanes(
            prepared,
            factors,
            maximum_rank,
            options,
            materialize_adjustment,
            accept_boundary_solution,
            arena,
        );
        arena.compact_unfinished_lanes();
    }

    for state in &mut arena.lane_states {
        if !state.finalized && state.failed.is_none() && !state.optimizer_complete {
            state.failed = Some(TemporalInferenceStatus::OptimizerNonconverged);
        }
    }
    finalize_terminal_lanes(
        prepared,
        factors,
        maximum_rank,
        options,
        materialize_adjustment,
        accept_boundary_solution,
        arena,
    );
}

fn compact_lane_major<T: Copy>(
    values: &mut [T],
    item_count: usize,
    old_lane_count: usize,
    retained_lanes: &[usize],
) {
    let new_lane_count = retained_lanes.len();
    for item in 0..item_count {
        for (destination_lane, &source_lane) in retained_lanes.iter().enumerate() {
            values[item * new_lane_count + destination_lane] =
                values[item * old_lane_count + source_lane];
        }
    }
}

#[allow(clippy::too_many_lines)]
fn compact_batch_scratch_lanes(scratch: &mut BatchScratch, retained_lanes: &[usize]) {
    let old_lanes = scratch.lane_count;
    let date_count = scratch.date_count;
    let rank = scratch.realized_rank;
    let triangular = rank * (rank + 1) / 2;
    let basis_triangular = scratch.basis_dimension * (scratch.basis_dimension + 1) / 2;
    let basis_coefficients = if scratch.basis_enabled {
        1 + 3 * scratch.gap_class_exponents.len()
    } else {
        0
    };
    compact_lane_major(
        &mut scratch.class_transition,
        scratch.gap_class_exponents.len(),
        old_lanes,
        retained_lanes,
    );
    compact_lane_major(
        &mut scratch.class_inverse_innovation,
        scratch.gap_class_exponents.len(),
        old_lanes,
        retained_lanes,
    );
    compact_lane_major(
        &mut scratch.class_log_innovation,
        scratch.gap_class_exponents.len(),
        old_lanes,
        retained_lanes,
    );
    for values in [
        &mut scratch.class_a_eta,
        &mut scratch.class_a_eta_eta,
        &mut scratch.class_b_eta,
        &mut scratch.class_b_eta_eta,
    ] {
        compact_lane_major(
            values,
            scratch.gap_class_exponents.len(),
            old_lanes,
            retained_lanes,
        );
    }
    compact_lane_major(
        &mut scratch.basis,
        basis_coefficients * basis_triangular,
        old_lanes,
        retained_lanes,
    );
    compact_lane_major(
        &mut scratch.projected_basis,
        usize::from(scratch.basis_enabled) * basis_triangular,
        old_lanes,
        retained_lanes,
    );
    for values in [
        &mut scratch.projected_basis_eta,
        &mut scratch.projected_basis_eta_eta,
    ] {
        compact_lane_major(
            values,
            usize::from(scratch.basis_enabled) * basis_triangular,
            old_lanes,
            retained_lanes,
        );
    }
    for values in [
        &mut scratch.observations,
        &mut scratch.inverse_shape,
        &mut scratch.z_x,
        &mut scratch.z_y,
    ] {
        compact_lane_major(values, date_count, old_lanes, retained_lanes);
    }
    for values in [&mut scratch.whitened_x, &mut scratch.whitened_y] {
        compact_lane_major(
            values,
            usize::from(!scratch.basis_enabled) * date_count,
            old_lanes,
            retained_lanes,
        );
    }
    compact_lane_major(
        &mut scratch.scaled_factor,
        date_count * rank,
        old_lanes,
        retained_lanes,
    );
    compact_lane_major(
        &mut scratch.whitened_factor,
        usize::from(!scratch.basis_enabled) * date_count * rank,
        old_lanes,
        retained_lanes,
    );
    for values in [
        &mut scratch.transition,
        &mut scratch.inverse_innovation_scale,
    ] {
        compact_lane_major(
            values,
            usize::from(!scratch.basis_enabled) * date_count.saturating_sub(1),
            old_lanes,
            retained_lanes,
        );
    }
    compact_lane_major(&mut scratch.lower, triangular, old_lanes, retained_lanes);
    for values in [
        &mut scratch.h_x,
        &mut scratch.h_y,
        &mut scratch.solve_x,
        &mut scratch.solve_y,
    ] {
        compact_lane_major(values, rank, old_lanes, retained_lanes);
    }
    compact_lane_major(
        &mut scratch.bilinear_jets,
        usize::from(scratch.basis_enabled) * 18,
        old_lanes,
        retained_lanes,
    );
    compact_lane_major(
        &mut scratch.augmented_lower_jets,
        usize::from(scratch.basis_enabled) * 6 * basis_triangular,
        old_lanes,
        retained_lanes,
    );
    for values in [
        &mut scratch.process_variance,
        &mut scratch.process_standard_deviation,
        &mut scratch.log_process_variance,
        &mut scratch.geometric_mean,
        &mut scratch.maximum_shape,
        &mut scratch.log_shape_sum,
        &mut scratch.log_determinant_r,
        &mut scratch.log_determinant_r_eta,
        &mut scratch.log_determinant_r_eta_eta,
        &mut scratch.x_v_x,
        &mut scratch.x_v_y,
        &mut scratch.y_v_y,
        &mut scratch.slope,
        &mut scratch.quadratic,
        &mut scratch.score_gradient_log_q,
        &mut scratch.score_curvature_log_q,
        &mut scratch.score_gradient_eta,
        &mut scratch.score_curvature_eta,
        &mut scratch.score_curvature_eta_log_q,
        &mut scratch.work_a,
        &mut scratch.work_b,
        &mut scratch.work_c,
        &mut scratch.work_d,
    ] {
        compact_lane_major(values, 1, old_lanes, retained_lanes);
    }
    compact_lane_major(&mut scratch.static_valid, 1, old_lanes, retained_lanes);
    compact_lane_major(&mut scratch.active, 1, old_lanes, retained_lanes);
    compact_lane_major(&mut scratch.positive_definite, 1, old_lanes, retained_lanes);
    for values in [
        &mut scratch.process_variance,
        &mut scratch.process_standard_deviation,
        &mut scratch.log_process_variance,
        &mut scratch.geometric_mean,
        &mut scratch.maximum_shape,
        &mut scratch.log_shape_sum,
        &mut scratch.log_determinant_r,
        &mut scratch.log_determinant_r_eta,
        &mut scratch.log_determinant_r_eta_eta,
        &mut scratch.x_v_x,
        &mut scratch.x_v_y,
        &mut scratch.y_v_y,
        &mut scratch.slope,
        &mut scratch.quadratic,
        &mut scratch.score_gradient_log_q,
        &mut scratch.score_curvature_log_q,
        &mut scratch.score_gradient_eta,
        &mut scratch.score_curvature_eta,
        &mut scratch.score_curvature_eta_log_q,
        &mut scratch.work_a,
        &mut scratch.work_b,
        &mut scratch.work_c,
        &mut scratch.work_d,
    ] {
        values.truncate(retained_lanes.len());
    }
    scratch.static_valid.truncate(retained_lanes.len());
    scratch.active.truncate(retained_lanes.len());
    scratch.positive_definite.truncate(retained_lanes.len());
    scratch.lane_count = retained_lanes.len();
}

fn compact_values<T: Copy>(values: &mut [T], retained_lanes: &[usize]) {
    for (destination, &source) in retained_lanes.iter().enumerate() {
        values[destination] = values[source];
    }
}

impl WorkerArena {
    fn compact_unfinished_lanes(&mut self) {
        let old_lane_count = self.chunk.targets.len();
        self.retained_lane_indices.clear();
        self.retained_lane_indices.extend(
            self.lane_states
                .iter()
                .enumerate()
                .filter_map(|(lane, state)| (!state.finalized).then_some(lane)),
        );
        let new_lane_count = self.retained_lane_indices.len();
        if new_lane_count * 2 >= old_lane_count {
            return;
        }
        compact_batch_scratch_lanes(&mut self.chunk.scratch, &self.retained_lane_indices);
        for (destination, &source) in self.retained_lane_indices.iter().enumerate() {
            self.chunk.targets[destination] = self.chunk.targets[source];
            self.chunk.outcomes[destination] = self.chunk.outcomes[source];
            self.lane_states[destination] = self.lane_states[source];
        }
        compact_values(&mut self.profile_active, &self.retained_lane_indices);
        compact_values(&mut self.lane_rhos, &self.retained_lane_indices);
        compact_values(&mut self.lane_log_variances, &self.retained_lane_indices);
        self.chunk.targets.truncate(new_lane_count);
        self.chunk.outcomes.truncate(new_lane_count);
        self.lane_states.truncate(new_lane_count);
        self.profile_metrics.compaction_events += 1;
        self.profile_metrics.compacted_lane_count += old_lane_count - new_lane_count;
    }
}

#[cfg(test)]
fn process_chunk(inputs: &EvaluationInputs<'_>, chunk: &mut PreparedChunk) {
    process_chunk_internal(inputs, chunk, false);
}

#[cfg(test)]
fn process_chunk_internal(
    inputs: &EvaluationInputs<'_>,
    chunk: &mut PreparedChunk,
    compute_q_derivatives: bool,
) {
    chunk.scratch.reset_dynamic();
    chunk
        .outcomes
        .fill(Err(TemporalInferenceStatus::CovarianceNonfinite));
    for (lane, &target) in chunk.targets.iter().enumerate() {
        prepare_theta_lane(
            inputs.prepared,
            inputs.factors,
            inputs.maximum_rank,
            target,
            inputs.rhos[target],
            inputs.log_process_variances[target],
            inputs.restricted,
            lane,
            &mut chunk.scratch,
            &mut chunk.outcomes[lane],
        );
    }

    execute_prepared_theta_chunk(
        inputs.restricted,
        chunk,
        compute_q_derivatives,
        compute_q_derivatives,
    );
}

#[allow(clippy::too_many_arguments)]
fn process_profile_chunk(
    prepared: &PreparedFactorObjective,
    factors: &[f64],
    maximum_rank: usize,
    rhos: &[f64],
    log_process_variances: &[f64],
    active: &[bool],
    materialize_adjustment: bool,
    chunk: &mut PreparedChunk,
) {
    chunk.scratch.reset_dynamic();
    chunk
        .outcomes
        .fill(Err(TemporalInferenceStatus::CovarianceNonfinite));
    for (lane, &target) in chunk.targets.iter().enumerate() {
        if !active[lane] {
            continue;
        }
        prepare_theta_lane(
            prepared,
            factors,
            maximum_rank,
            target,
            rhos[lane],
            log_process_variances[lane],
            true,
            lane,
            &mut chunk.scratch,
            &mut chunk.outcomes[lane],
        );
    }

    execute_prepared_theta_chunk(true, chunk, true, materialize_adjustment);
}

fn execute_prepared_theta_chunk(
    restricted: bool,
    chunk: &mut PreparedChunk,
    compute_q_derivatives: bool,
    materialize_adjustment: bool,
) {
    let lane_count = chunk.targets.len();

    if chunk.scratch.basis_enabled && compute_q_derivatives {
        chunk.arch.dispatch(AugmentedJetMainKernel {
            scratch: &mut chunk.scratch,
            materialize_adjustment,
        });
    } else if chunk.scratch.basis_enabled {
        chunk.arch.dispatch(BasisMainKernel {
            scratch: &mut chunk.scratch,
        });
    } else {
        chunk.arch.dispatch(MainKernel {
            scratch: &mut chunk.scratch,
        });
    }
    for lane in 0..lane_count {
        if !chunk.scratch.active[lane] {
            continue;
        }
        if !chunk.scratch.positive_definite[lane] {
            chunk.scratch.active[lane] = false;
            chunk.outcomes[lane] = Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite);
            continue;
        }
        let x_v_x = chunk.scratch.x_v_x[lane];
        let x_v_y = chunk.scratch.x_v_y[lane];
        let y_v_y = chunk.scratch.y_v_y[lane];
        if !x_v_x.is_finite() || x_v_x <= 0.0 || !x_v_y.is_finite() || !y_v_y.is_finite() {
            chunk.scratch.active[lane] = false;
            chunk.outcomes[lane] = Err(TemporalInferenceStatus::DesignRankDeficient);
            continue;
        }
        chunk.scratch.slope[lane] = x_v_y / x_v_x;
    }
    if chunk.scratch.basis_enabled {
        chunk.arch.dispatch(ProjectedQuadraticKernel {
            scratch: &mut chunk.scratch,
        });
    } else {
        chunk.arch.dispatch(QuadraticKernel {
            scratch: &mut chunk.scratch,
        });
    }
    for lane in 0..lane_count {
        if !chunk.scratch.active[lane] {
            continue;
        }
        chunk.outcomes[lane] = finish_lane(restricted, lane, &chunk.scratch);
    }
}

fn prepare_static_lane(
    inputs: &StaticInputs<'_>,
    target: usize,
    lane: usize,
    scratch: &mut BatchScratch,
) {
    let date_count = inputs.prepared.design.len();
    let realized_rank = scratch.realized_rank;
    let factor_stride = date_count * inputs.maximum_rank;
    let factor = factor_for_target(inputs.factors, factor_stride, target);
    if factor.iter().any(|value| !value.is_finite())
        || (0..date_count)
            .any(|date| !inputs.observations_soa[date * inputs.target_count + target].is_finite())
    {
        return;
    }
    let mut mean_log_diagonal = 0.0;
    for row in 0..date_count {
        let diagonal = (0..realized_rank)
            .map(|component| factor[row * inputs.maximum_rank + component].powi(2))
            .sum::<f64>();
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return;
        }
        scratch.inverse_shape[row * scratch.lane_count + lane] = diagonal;
        mean_log_diagonal += diagonal.ln();
    }
    mean_log_diagonal /= date_count as f64;
    let geometric_mean = mean_log_diagonal.exp();
    if !geometric_mean.is_finite() || geometric_mean <= 0.0 {
        return;
    }
    let mut log_shape_sum = 0.0;
    let mut maximum_shape = 0.0_f64;
    for row in 0..date_count {
        let shape =
            (scratch.inverse_shape[row * scratch.lane_count + lane] / geometric_mean).sqrt();
        if !shape.is_finite() || shape <= 0.0 {
            return;
        }
        log_shape_sum += shape.ln();
        maximum_shape = maximum_shape.max(shape);
        let inverse_shape = 1.0 / shape;
        scratch.inverse_shape[row * scratch.lane_count + lane] = inverse_shape;
        let observation = inputs.observations_soa[row * inputs.target_count + target];
        scratch.observations[row * scratch.lane_count + lane] = observation;
        scratch.z_x[row * scratch.lane_count + lane] = inputs.prepared.design[row] * inverse_shape;
        scratch.z_y[row * scratch.lane_count + lane] = observation * inverse_shape;
        for component in 0..realized_rank {
            let value = factor[row * inputs.maximum_rank + component];
            scratch.scaled_factor[(row * realized_rank + component) * scratch.lane_count + lane] =
                value * inverse_shape;
        }
    }
    scratch.geometric_mean[lane] = geometric_mean;
    scratch.maximum_shape[lane] = maximum_shape;
    scratch.log_shape_sum[lane] = log_shape_sum;
    scratch.static_valid[lane] = true;
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn prepare_theta_lane(
    prepared: &PreparedFactorObjective,
    factors: &[f64],
    maximum_rank: usize,
    target: usize,
    rho: f64,
    log_process_variance: f64,
    restricted: bool,
    lane: usize,
    scratch: &mut BatchScratch,
    outcome: &mut ObjectiveResult,
) {
    if !scratch.static_valid[lane] {
        return;
    }
    if !rho.is_finite() || !(0.0..1.0).contains(&rho) {
        *outcome = Err(TemporalInferenceStatus::CovarianceParameterAtBoundary);
        return;
    }
    let process_variance = log_process_variance.exp();
    if !process_variance.is_finite() || process_variance <= 0.0 {
        return;
    }
    let date_count = prepared.design.len();
    let factor_stride = date_count * maximum_rank;
    let factor = factor_for_target(factors, factor_stride, target);
    let geometric_mean = scratch.geometric_mean[lane];
    let maximum_shape = scratch.maximum_shape[lane];
    if process_variance * maximum_shape.powi(2) * f64::EPSILON * 8.0 > SYMMETRY_TOLERANCE
        || process_variance <= geometric_mean * 1e-5
    {
        for date in 0..date_count {
            scratch.fallback_observations[date] =
                scratch.observations[date * scratch.lane_count + lane];
        }
        *outcome = dense_factor_objective_fallback(
            prepared,
            &scratch.fallback_observations,
            factor,
            maximum_rank,
            scratch.realized_rank,
            rho,
            process_variance,
            restricted,
        );
        return;
    }

    let mut log_determinant_r = 0.0;
    let mut log_determinant_r_eta = f64::NAN;
    let mut log_determinant_r_eta_eta = f64::NAN;
    if scratch.basis_enabled {
        if rho == 0.0 {
            for class in 0..scratch.gap_class_exponents.len() {
                let index = class * scratch.lane_count + lane;
                scratch.class_transition[index] = 0.0;
                scratch.class_inverse_innovation[index] = 1.0;
                scratch.class_log_innovation[index] = 0.0;
                scratch.class_a_eta[index] = f64::NAN;
                scratch.class_a_eta_eta[index] = f64::NAN;
                scratch.class_b_eta[index] = f64::NAN;
                scratch.class_b_eta_eta[index] = f64::NAN;
            }
        } else {
            log_determinant_r_eta = 0.0;
            log_determinant_r_eta_eta = 0.0;
            for class in 0..scratch.gap_class_exponents.len() {
                let gamma = scratch.gap_class_exponents[class];
                let log_phi = rho.ln() * gamma;
                let phi = log_phi.exp();
                let innovation = -(2.0 * log_phi).exp_m1();
                if !phi.is_finite() || !innovation.is_finite() || innovation <= 0.0 {
                    *outcome = Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite);
                    return;
                }
                let index = class * scratch.lane_count + lane;
                scratch.class_transition[index] = phi;
                scratch.class_inverse_innovation[index] = 1.0 / innovation;
                scratch.class_log_innovation[index] = innovation.ln();
                let u = phi * phi;
                let innovation_squared = innovation * innovation;
                let innovation_cubed = innovation_squared * innovation;
                scratch.class_a_eta[index] = 2.0 * gamma * u / innovation_squared;
                scratch.class_a_eta_eta[index] =
                    4.0 * gamma.powi(2) * u * (1.0 + u) / innovation_cubed;
                scratch.class_b_eta[index] = -gamma * phi * (1.0 + u) / innovation_squared;
                scratch.class_b_eta_eta[index] =
                    -gamma.powi(2) * phi * (1.0 + 6.0 * u + u * u) / innovation_cubed;
                let count = scratch.gap_class_counts[class] as f64;
                log_determinant_r_eta += -2.0 * count * gamma * u / innovation;
                log_determinant_r_eta_eta += -4.0 * count * gamma.powi(2) * u / innovation_squared;
            }
        }
        for class in 0..scratch.gap_class_exponents.len() {
            let class_index = class * scratch.lane_count + lane;
            log_determinant_r +=
                scratch.gap_class_counts[class] as f64 * scratch.class_log_innovation[class_index];
        }
    } else if rho == 0.0 {
        for edge in 0..date_count.saturating_sub(1) {
            scratch.transition[edge * scratch.lane_count + lane] = 0.0;
            scratch.inverse_innovation_scale[edge * scratch.lane_count + lane] = 1.0;
        }
    } else {
        for edge in 0..date_count - 1 {
            let log_phi = rho.ln() * prepared.gap_exponents[edge];
            let phi = log_phi.exp();
            let innovation = -(2.0 * log_phi).exp_m1();
            if !phi.is_finite() || !innovation.is_finite() || innovation <= 0.0 {
                *outcome = Err(TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite);
                return;
            }
            scratch.transition[edge * scratch.lane_count + lane] = phi;
            scratch.inverse_innovation_scale[edge * scratch.lane_count + lane] =
                1.0 / innovation.sqrt();
            log_determinant_r += innovation.ln();
        }
    }
    scratch.process_variance[lane] = process_variance;
    scratch.process_standard_deviation[lane] = process_variance.sqrt();
    scratch.log_process_variance[lane] = log_process_variance;
    scratch.log_determinant_r[lane] = log_determinant_r;
    scratch.log_determinant_r_eta[lane] = log_determinant_r_eta;
    scratch.log_determinant_r_eta_eta[lane] = log_determinant_r_eta_eta;
    scratch.active[lane] = true;
}

fn finish_lane(restricted: bool, lane: usize, scratch: &BatchScratch) -> ObjectiveResult {
    let mut log_determinant_k = 0.0;
    for component in 0..scratch.realized_rank {
        log_determinant_k += scratch.lower
            [(triangular_index(component, component) * scratch.lane_count) + lane]
            .ln();
    }
    log_determinant_k *= 2.0;
    let log_determinant = (scratch.date_count - scratch.realized_rank) as f64
        * scratch.log_process_variance[lane]
        + 2.0 * scratch.log_shape_sum[lane]
        + scratch.log_determinant_r[lane]
        + log_determinant_k;
    let x_v_x = scratch.x_v_x[lane];
    #[cfg(test)]
    let x_v_y = scratch.x_v_y[lane];
    #[cfg(test)]
    let y_v_y = scratch.y_v_y[lane];
    let slope = scratch.slope[lane];
    let quadratic = scratch.quadratic[lane];
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
        #[cfg(test)]
        x_v_y,
        #[cfg(test)]
        y_v_y,
        #[cfg(test)]
        log_determinant,
        #[cfg(test)]
        quadratic,
        dense_fallback_used: false,
    })
}

struct BatchScratch {
    date_count: usize,
    realized_rank: usize,
    lane_count: usize,
    basis_enabled: bool,
    basis_dimension: usize,
    gap_class_exponents: Vec<f64>,
    gap_class_counts: Vec<usize>,
    edge_gap_classes: Vec<usize>,
    class_transition: Vec<f64>,
    class_inverse_innovation: Vec<f64>,
    class_log_innovation: Vec<f64>,
    class_a_eta: Vec<f64>,
    class_a_eta_eta: Vec<f64>,
    class_b_eta: Vec<f64>,
    class_b_eta_eta: Vec<f64>,
    basis: Vec<f64>,
    projected_basis: Vec<f64>,
    projected_basis_eta: Vec<f64>,
    projected_basis_eta_eta: Vec<f64>,
    observations: Vec<f64>,
    inverse_shape: Vec<f64>,
    transition: Vec<f64>,
    inverse_innovation_scale: Vec<f64>,
    scaled_factor: Vec<f64>,
    whitened_factor: Vec<f64>,
    z_x: Vec<f64>,
    z_y: Vec<f64>,
    whitened_x: Vec<f64>,
    whitened_y: Vec<f64>,
    lower: Vec<f64>,
    h_x: Vec<f64>,
    h_y: Vec<f64>,
    solve_x: Vec<f64>,
    solve_y: Vec<f64>,
    bilinear_jets: Vec<f64>,
    augmented_lower_jets: Vec<f64>,
    process_variance: Vec<f64>,
    process_standard_deviation: Vec<f64>,
    log_process_variance: Vec<f64>,
    geometric_mean: Vec<f64>,
    maximum_shape: Vec<f64>,
    log_shape_sum: Vec<f64>,
    log_determinant_r: Vec<f64>,
    log_determinant_r_eta: Vec<f64>,
    log_determinant_r_eta_eta: Vec<f64>,
    x_v_x: Vec<f64>,
    x_v_y: Vec<f64>,
    y_v_y: Vec<f64>,
    slope: Vec<f64>,
    quadratic: Vec<f64>,
    score_gradient_log_q: Vec<f64>,
    score_curvature_log_q: Vec<f64>,
    score_gradient_eta: Vec<f64>,
    score_curvature_eta: Vec<f64>,
    score_curvature_eta_log_q: Vec<f64>,
    slope_gradient_eta: Vec<f64>,
    slope_gradient_log_q: Vec<f64>,
    work_a: Vec<f64>,
    work_b: Vec<f64>,
    work_c: Vec<f64>,
    work_d: Vec<f64>,
    fallback_observations: Vec<f64>,
    static_valid: Vec<bool>,
    active: Vec<bool>,
    positive_definite: Vec<bool>,
}

impl BatchScratch {
    #[allow(clippy::too_many_lines)]
    fn new(
        date_count: usize,
        realized_rank: usize,
        lane_count: usize,
        gap_basis: Option<&GapBasisDefinition>,
    ) -> Self {
        let date_lanes = date_count * lane_count;
        let factor_lanes = date_count * realized_rank * lane_count;
        let rank_lanes = realized_rank * lane_count;
        let triangular_lanes = realized_rank * (realized_rank + 1) / 2 * lane_count;
        let basis_enabled = gap_basis.is_some();
        let basis_dimension = realized_rank + 2;
        let basis_triangular = basis_dimension * (basis_dimension + 1) / 2;
        let gap_class_count = gap_basis.map_or(0, |basis| basis.exponents.len());
        let basis_coefficients = if basis_enabled {
            1 + 3 * gap_class_count
        } else {
            0
        };
        let mut scratch = Self {
            date_count,
            realized_rank,
            lane_count,
            basis_enabled,
            basis_dimension,
            gap_class_exponents: gap_basis.map_or_else(Vec::new, |basis| basis.exponents.clone()),
            gap_class_counts: gap_basis.map_or_else(Vec::new, |basis| basis.counts.clone()),
            edge_gap_classes: gap_basis.map_or_else(Vec::new, |basis| basis.edge_classes.clone()),
            class_transition: vec![0.0; gap_class_count * lane_count],
            class_inverse_innovation: vec![1.0; gap_class_count * lane_count],
            class_log_innovation: vec![0.0; gap_class_count * lane_count],
            class_a_eta: vec![0.0; gap_class_count * lane_count],
            class_a_eta_eta: vec![0.0; gap_class_count * lane_count],
            class_b_eta: vec![0.0; gap_class_count * lane_count],
            class_b_eta_eta: vec![0.0; gap_class_count * lane_count],
            basis: vec![0.0; basis_coefficients * basis_triangular * lane_count],
            projected_basis: vec![0.0; basis_triangular * lane_count],
            projected_basis_eta: vec![0.0; basis_triangular * lane_count],
            projected_basis_eta_eta: vec![0.0; basis_triangular * lane_count],
            observations: vec![0.0; date_lanes],
            inverse_shape: vec![1.0; date_lanes],
            transition: vec![
                0.0;
                if basis_enabled {
                    0
                } else {
                    date_count.saturating_sub(1) * lane_count
                }
            ],
            inverse_innovation_scale: vec![
                1.0;
                if basis_enabled {
                    0
                } else {
                    date_count.saturating_sub(1) * lane_count
                }
            ],
            scaled_factor: vec![0.0; factor_lanes],
            whitened_factor: vec![0.0; if basis_enabled { 0 } else { factor_lanes }],
            z_x: vec![0.0; date_lanes],
            z_y: vec![0.0; date_lanes],
            whitened_x: vec![0.0; if basis_enabled { 0 } else { date_lanes }],
            whitened_y: vec![0.0; if basis_enabled { 0 } else { date_lanes }],
            lower: vec![0.0; triangular_lanes],
            h_x: vec![0.0; rank_lanes],
            h_y: vec![0.0; rank_lanes],
            solve_x: vec![0.0; rank_lanes],
            solve_y: vec![0.0; rank_lanes],
            bilinear_jets: vec![0.0; usize::from(basis_enabled) * 18 * lane_count],
            augmented_lower_jets: vec![
                0.0;
                usize::from(basis_enabled)
                    * 6
                    * basis_triangular
                    * lane_count
            ],
            process_variance: vec![1.0; lane_count],
            process_standard_deviation: vec![1.0; lane_count],
            log_process_variance: vec![0.0; lane_count],
            geometric_mean: vec![1.0; lane_count],
            maximum_shape: vec![1.0; lane_count],
            log_shape_sum: vec![0.0; lane_count],
            log_determinant_r: vec![0.0; lane_count],
            log_determinant_r_eta: vec![0.0; lane_count],
            log_determinant_r_eta_eta: vec![0.0; lane_count],
            x_v_x: vec![0.0; lane_count],
            x_v_y: vec![0.0; lane_count],
            y_v_y: vec![0.0; lane_count],
            slope: vec![0.0; lane_count],
            quadratic: vec![0.0; lane_count],
            score_gradient_log_q: vec![f64::NAN; lane_count],
            score_curvature_log_q: vec![f64::NAN; lane_count],
            score_gradient_eta: vec![f64::NAN; lane_count],
            score_curvature_eta: vec![f64::NAN; lane_count],
            score_curvature_eta_log_q: vec![f64::NAN; lane_count],
            slope_gradient_eta: vec![f64::NAN; lane_count],
            slope_gradient_log_q: vec![f64::NAN; lane_count],
            work_a: vec![0.0; lane_count],
            work_b: vec![0.0; lane_count],
            work_c: vec![0.0; lane_count],
            work_d: vec![0.0; lane_count],
            fallback_observations: vec![0.0; date_count],
            static_valid: vec![false; lane_count],
            active: vec![false; lane_count],
            positive_definite: vec![true; lane_count],
        };
        for lane in 0..lane_count {
            scratch.initialize_safe_lane(lane);
        }
        scratch
    }

    fn prepare_basis(&mut self) {
        if !self.basis_enabled {
            return;
        }
        let triangular = self.basis_dimension * (self.basis_dimension + 1) / 2;
        for lane in 0..self.lane_count {
            for left in 0..self.basis_dimension {
                for right in 0..=left {
                    let entry = triangular_index(left, right);
                    let base = self.static_basis_value(0, left, lane)
                        * self.static_basis_value(0, right, lane);
                    self.basis[entry * self.lane_count + lane] = base;
                    for edge in 0..self.date_count.saturating_sub(1) {
                        let class = self.edge_gap_classes[edge];
                        let current = edge + 1;
                        let previous = edge;
                        let current_left = self.static_basis_value(current, left, lane);
                        let current_right = self.static_basis_value(current, right, lane);
                        let previous_left = self.static_basis_value(previous, left, lane);
                        let previous_right = self.static_basis_value(previous, right, lane);
                        for (coefficient, value) in [
                            (0, current_left * current_right),
                            (
                                1,
                                current_left * previous_right + previous_left * current_right,
                            ),
                            (2, previous_left * previous_right),
                        ] {
                            let coefficient = 1 + 3 * class + coefficient;
                            let index = (coefficient * triangular + entry) * self.lane_count + lane;
                            self.basis[index] += value;
                        }
                    }
                }
            }
        }
    }

    fn restore_lane_count(&mut self, lane_count: usize) {
        self.lane_count = lane_count;
        for values in [
            &mut self.process_variance,
            &mut self.process_standard_deviation,
            &mut self.log_process_variance,
            &mut self.geometric_mean,
            &mut self.maximum_shape,
            &mut self.log_shape_sum,
            &mut self.log_determinant_r,
            &mut self.log_determinant_r_eta,
            &mut self.log_determinant_r_eta_eta,
            &mut self.x_v_x,
            &mut self.x_v_y,
            &mut self.y_v_y,
            &mut self.slope,
            &mut self.quadratic,
            &mut self.score_gradient_log_q,
            &mut self.score_curvature_log_q,
            &mut self.score_gradient_eta,
            &mut self.score_curvature_eta,
            &mut self.score_curvature_eta_log_q,
            &mut self.slope_gradient_eta,
            &mut self.slope_gradient_log_q,
            &mut self.work_a,
            &mut self.work_b,
            &mut self.work_c,
            &mut self.work_d,
        ] {
            values.resize(lane_count, 0.0);
        }
        self.static_valid.resize(lane_count, false);
        self.active.resize(lane_count, false);
        self.positive_definite.resize(lane_count, true);
    }

    fn static_basis_value(&self, date: usize, variable: usize, lane: usize) -> f64 {
        if variable < self.realized_rank {
            self.scaled_factor[(date * self.realized_rank + variable) * self.lane_count + lane]
        } else if variable == self.realized_rank {
            self.z_x[date * self.lane_count + lane]
        } else {
            self.z_y[date * self.lane_count + lane]
        }
    }

    fn initialize_safe_lane(&mut self, lane: usize) {
        self.active[lane] = false;
        self.process_variance[lane] = 1.0;
        self.process_standard_deviation[lane] = 1.0;
        self.log_process_variance[lane] = 0.0;
        self.geometric_mean[lane] = 1.0;
        self.maximum_shape[lane] = 1.0;
        self.log_shape_sum[lane] = 0.0;
        self.log_determinant_r[lane] = 0.0;
        for row in 0..self.date_count {
            self.observations[row * self.lane_count + lane] = 0.0;
            self.inverse_shape[row * self.lane_count + lane] = 1.0;
            self.z_x[row * self.lane_count + lane] = 0.0;
            self.z_y[row * self.lane_count + lane] = 0.0;
            for component in 0..self.realized_rank {
                let value = f64::from(component == row % self.realized_rank);
                let index = (row * self.realized_rank + component) * self.lane_count + lane;
                self.scaled_factor[index] = value;
            }
        }
        if !self.basis_enabled {
            for edge in 0..self.date_count.saturating_sub(1) {
                self.transition[edge * self.lane_count + lane] = 0.0;
                self.inverse_innovation_scale[edge * self.lane_count + lane] = 1.0;
            }
        }
    }

    fn reset_static(&mut self) {
        self.static_valid.fill(false);
        self.basis.fill(0.0);
        self.projected_basis.fill(0.0);
        self.projected_basis_eta.fill(0.0);
        self.projected_basis_eta_eta.fill(0.0);
        for lane in 0..self.lane_count {
            self.initialize_safe_lane(lane);
        }
    }

    fn reset_dynamic(&mut self) {
        self.active.fill(false);
        self.positive_definite.fill(true);
        self.transition.fill(0.0);
        self.inverse_innovation_scale.fill(1.0);
        self.process_variance.fill(1.0);
        self.process_standard_deviation.fill(1.0);
        self.log_process_variance.fill(0.0);
        self.log_determinant_r.fill(0.0);
        self.log_determinant_r_eta.fill(0.0);
        self.log_determinant_r_eta_eta.fill(0.0);
        self.x_v_x.fill(0.0);
        self.x_v_y.fill(0.0);
        self.y_v_y.fill(0.0);
        self.slope.fill(0.0);
        self.quadratic.fill(0.0);
        self.score_gradient_log_q.fill(f64::NAN);
        self.score_curvature_log_q.fill(f64::NAN);
        self.score_gradient_eta.fill(f64::NAN);
        self.score_curvature_eta.fill(f64::NAN);
        self.score_curvature_eta_log_q.fill(f64::NAN);
        self.slope_gradient_eta.fill(f64::NAN);
        self.slope_gradient_log_q.fill(f64::NAN);
    }

    fn allocated_bytes(&self) -> usize {
        let f64_capacity = self.gap_class_exponents.capacity()
            + self.class_transition.capacity()
            + self.class_inverse_innovation.capacity()
            + self.class_log_innovation.capacity()
            + self.class_a_eta.capacity()
            + self.class_a_eta_eta.capacity()
            + self.class_b_eta.capacity()
            + self.class_b_eta_eta.capacity()
            + self.basis.capacity()
            + self.projected_basis.capacity()
            + self.projected_basis_eta.capacity()
            + self.projected_basis_eta_eta.capacity()
            + self.observations.capacity()
            + self.inverse_shape.capacity()
            + self.transition.capacity()
            + self.inverse_innovation_scale.capacity()
            + self.scaled_factor.capacity()
            + self.whitened_factor.capacity()
            + self.z_x.capacity()
            + self.z_y.capacity()
            + self.whitened_x.capacity()
            + self.whitened_y.capacity()
            + self.lower.capacity()
            + self.h_x.capacity()
            + self.h_y.capacity()
            + self.solve_x.capacity()
            + self.solve_y.capacity()
            + self.bilinear_jets.capacity()
            + self.augmented_lower_jets.capacity()
            + self.process_variance.capacity()
            + self.process_standard_deviation.capacity()
            + self.log_process_variance.capacity()
            + self.geometric_mean.capacity()
            + self.maximum_shape.capacity()
            + self.log_shape_sum.capacity()
            + self.log_determinant_r.capacity()
            + self.log_determinant_r_eta.capacity()
            + self.log_determinant_r_eta_eta.capacity()
            + self.x_v_x.capacity()
            + self.x_v_y.capacity()
            + self.y_v_y.capacity()
            + self.slope.capacity()
            + self.quadratic.capacity()
            + self.score_gradient_log_q.capacity()
            + self.score_curvature_log_q.capacity()
            + self.score_gradient_eta.capacity()
            + self.score_curvature_eta.capacity()
            + self.score_curvature_eta_log_q.capacity()
            + self.slope_gradient_eta.capacity()
            + self.slope_gradient_log_q.capacity()
            + self.work_a.capacity()
            + self.work_b.capacity()
            + self.work_c.capacity()
            + self.work_d.capacity()
            + self.fallback_observations.capacity();
        f64_capacity * std::mem::size_of::<f64>()
            + self.gap_class_counts.capacity() * std::mem::size_of::<usize>()
            + self.edge_gap_classes.capacity() * std::mem::size_of::<usize>()
            + self.static_valid.capacity() * std::mem::size_of::<bool>()
            + self.active.capacity() * std::mem::size_of::<bool>()
            + self.positive_definite.capacity() * std::mem::size_of::<bool>()
    }
}

struct MainKernel<'a> {
    scratch: &'a mut BatchScratch,
}

struct BasisMainKernel<'a> {
    scratch: &'a mut BatchScratch,
}

struct AugmentedJetMainKernel<'a> {
    scratch: &'a mut BatchScratch,
    materialize_adjustment: bool,
}

impl WithSimd for AugmentedJetMainKernel<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        combine_projected_basis(simd, self.scratch);
        factorize_augmented_basis_jets(simd, self.scratch);
        finish_profile_derivatives(self.scratch, self.materialize_adjustment);
    }
}

impl WithSimd for BasisMainKernel<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        combine_projected_basis(simd, self.scratch);
        factorize_projected_basis(simd, self.scratch);
        form_projected_rhs(self.scratch);
        solve_cholesky(simd, self.scratch);
        form_projected_bilinears(simd, self.scratch);
    }
}

impl WithSimd for MainKernel<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        whiten(simd, self.scratch);
        factorize(simd, self.scratch);
        form_rhs(simd, self.scratch);
        solve_cholesky(simd, self.scratch);
        form_bilinears(simd, self.scratch);
    }
}

#[inline(always)]
fn combine_projected_basis<S: Simd>(simd: S, scratch: &mut BatchScratch) {
    let lanes = scratch.lane_count;
    let triangular = scratch.basis_dimension * (scratch.basis_dimension + 1) / 2;
    for entry in 0..triangular {
        let destination = entry * lanes;
        scratch.projected_basis[destination..destination + lanes]
            .copy_from_slice(&scratch.basis[destination..destination + lanes]);
        scratch.projected_basis_eta[destination..destination + lanes].fill(0.0);
        scratch.projected_basis_eta_eta[destination..destination + lanes].fill(0.0);
        for class in 0..scratch.gap_class_exponents.len() {
            let current = ((1 + 3 * class) * triangular + entry) * lanes;
            let cross = current + triangular * lanes;
            let previous = cross + triangular * lanes;
            let theta = class * lanes;
            basis_accumulate(
                simd,
                &mut scratch.projected_basis[destination..destination + lanes],
                &scratch.basis[current..current + lanes],
                &scratch.basis[cross..cross + lanes],
                &scratch.basis[previous..previous + lanes],
                &scratch.class_transition[theta..theta + lanes],
                &scratch.class_inverse_innovation[theta..theta + lanes],
            );
            basis_derivative_accumulate(
                simd,
                &mut scratch.projected_basis_eta[destination..destination + lanes],
                &scratch.basis[current..current + lanes],
                &scratch.basis[cross..cross + lanes],
                &scratch.basis[previous..previous + lanes],
                &scratch.class_a_eta[theta..theta + lanes],
                &scratch.class_b_eta[theta..theta + lanes],
            );
            basis_derivative_accumulate(
                simd,
                &mut scratch.projected_basis_eta_eta[destination..destination + lanes],
                &scratch.basis[current..current + lanes],
                &scratch.basis[cross..cross + lanes],
                &scratch.basis[previous..previous + lanes],
                &scratch.class_a_eta_eta[theta..theta + lanes],
                &scratch.class_b_eta_eta[theta..theta + lanes],
            );
        }
    }
}

#[inline(always)]
fn factorize_projected_basis<S: Simd>(simd: S, scratch: &mut BatchScratch) {
    let lanes = scratch.lane_count;
    scratch.lower.fill(0.0);
    for row in 0..scratch.realized_rank {
        for column in 0..=row {
            let source = triangular_index(row, column) * lanes;
            scratch
                .work_a
                .copy_from_slice(&scratch.projected_basis[source..source + lanes]);
            if row == column {
                add_in_place(simd, &mut scratch.work_a, &scratch.process_variance);
            }
            scratch.work_b.fill(0.0);
            for inner in 0..column {
                let left = triangular_index(row, inner) * lanes;
                let right = triangular_index(column, inner) * lanes;
                multiply_accumulate(
                    simd,
                    &mut scratch.work_b,
                    &scratch.lower[left..left + lanes],
                    &scratch.lower[right..right + lanes],
                );
            }
            let destination = triangular_index(row, column) * lanes;
            if row == column {
                difference(
                    simd,
                    &mut scratch.lower[destination..destination + lanes],
                    &scratch.work_a,
                    &scratch.work_b,
                );
                for lane in 0..lanes {
                    let diagonal = scratch.lower[destination + lane];
                    if scratch.active[lane] && (!diagonal.is_finite() || diagonal <= 0.0) {
                        scratch.positive_definite[lane] = false;
                        scratch.lower[destination + lane] = 1.0;
                    } else {
                        scratch.lower[destination + lane] = diagonal.sqrt();
                    }
                }
            } else {
                let denominator = triangular_index(column, column) * lanes;
                scratch
                    .work_c
                    .copy_from_slice(&scratch.lower[denominator..denominator + lanes]);
                difference_divide(
                    simd,
                    &mut scratch.lower[destination..destination + lanes],
                    &scratch.work_a,
                    &scratch.work_b,
                    &scratch.work_c,
                );
            }
        }
    }
}

fn form_projected_rhs(scratch: &mut BatchScratch) {
    let lanes = scratch.lane_count;
    let x = scratch.realized_rank;
    let y = scratch.realized_rank + 1;
    for component in 0..scratch.realized_rank {
        let destination = component * lanes;
        let x_source = triangular_index(x, component) * lanes;
        let y_source = triangular_index(y, component) * lanes;
        scratch.h_x[destination..destination + lanes]
            .copy_from_slice(&scratch.projected_basis[x_source..x_source + lanes]);
        scratch.h_y[destination..destination + lanes]
            .copy_from_slice(&scratch.projected_basis[y_source..y_source + lanes]);
    }
}

#[inline(always)]
fn form_projected_bilinears<S: Simd>(simd: S, scratch: &mut BatchScratch) {
    let lanes = scratch.lane_count;
    let x = scratch.realized_rank;
    let y = scratch.realized_rank + 1;
    scratch.work_a.fill(0.0);
    scratch.work_b.fill(0.0);
    scratch.work_c.fill(0.0);
    for component in 0..scratch.realized_rank {
        let start = component * lanes;
        multiply_accumulate(
            simd,
            &mut scratch.work_a,
            &scratch.h_x[start..start + lanes],
            &scratch.solve_x[start..start + lanes],
        );
        multiply_accumulate(
            simd,
            &mut scratch.work_b,
            &scratch.h_x[start..start + lanes],
            &scratch.solve_y[start..start + lanes],
        );
        multiply_accumulate(
            simd,
            &mut scratch.work_c,
            &scratch.h_y[start..start + lanes],
            &scratch.solve_y[start..start + lanes],
        );
    }
    let x_x = triangular_index(x, x) * lanes;
    let y_x = triangular_index(y, x) * lanes;
    let y_y = triangular_index(y, y) * lanes;
    difference_divide(
        simd,
        &mut scratch.x_v_x,
        &scratch.projected_basis[x_x..x_x + lanes],
        &scratch.work_a,
        &scratch.process_variance,
    );
    difference_divide(
        simd,
        &mut scratch.x_v_y,
        &scratch.projected_basis[y_x..y_x + lanes],
        &scratch.work_b,
        &scratch.process_variance,
    );
    difference_divide(
        simd,
        &mut scratch.y_v_y,
        &scratch.projected_basis[y_y..y_y + lanes],
        &scratch.work_c,
        &scratch.process_variance,
    );
}

#[inline(always)]
fn factorize_augmented_basis_jets<S: Simd>(simd: S, scratch: &mut BatchScratch) {
    let lanes = scratch.lane_count;
    let rank = scratch.realized_rank;
    let dimension = scratch.basis_dimension;
    scratch.augmented_lower_jets.fill(0.0);
    scratch.lower.fill(0.0);
    for column in 0..rank {
        for row in column..dimension {
            let source = triangular_index(row, column);
            copy_augmented_basis_jet(scratch, source, row == column && row < rank);
            for inner in 0..column {
                let left = triangular_index(row, inner) * 6 * lanes;
                let right = triangular_index(column, inner) * 6 * lanes;
                subtract_jet_product(
                    simd,
                    &mut scratch.bilinear_jets[..6 * lanes],
                    &scratch.augmented_lower_jets[left..left + 6 * lanes],
                    &scratch.augmented_lower_jets[right..right + 6 * lanes],
                    lanes,
                );
            }
            let destination = source * 6 * lanes;
            if row == column {
                square_root_jet(
                    &mut scratch.augmented_lower_jets[destination..destination + 6 * lanes],
                    &scratch.bilinear_jets[..6 * lanes],
                    &scratch.active,
                    &mut scratch.positive_definite,
                    lanes,
                );
                let packed = triangular_index(row, column) * lanes;
                scratch.lower[packed..packed + lanes].copy_from_slice(
                    &scratch.augmented_lower_jets[destination..destination + lanes],
                );
            } else {
                let denominator = triangular_index(column, column) * 6 * lanes;
                let (before_destination, destination_and_after) =
                    scratch.augmented_lower_jets.split_at_mut(destination);
                divide_jet(
                    simd,
                    &mut destination_and_after[..6 * lanes],
                    &scratch.bilinear_jets[..6 * lanes],
                    &before_destination[denominator..denominator + 6 * lanes],
                    lanes,
                );
            }
        }
    }
    for (pair, row, column) in [
        (0, rank, rank),
        (1, rank + 1, rank),
        (2, rank + 1, rank + 1),
    ] {
        let source = triangular_index(row, column);
        let destination = pair * 6 * lanes;
        copy_augmented_basis_jet_to(
            &scratch.projected_basis,
            &scratch.projected_basis_eta,
            &scratch.projected_basis_eta_eta,
            source,
            lanes,
            &mut scratch.bilinear_jets[destination..destination + 6 * lanes],
        );
        for inner in 0..rank {
            let left = triangular_index(row, inner) * 6 * lanes;
            let right = triangular_index(column, inner) * 6 * lanes;
            subtract_jet_product(
                simd,
                &mut scratch.bilinear_jets[destination..destination + 6 * lanes],
                &scratch.augmented_lower_jets[left..left + 6 * lanes],
                &scratch.augmented_lower_jets[right..right + 6 * lanes],
                lanes,
            );
        }
    }
}

fn copy_augmented_basis_jet(scratch: &mut BatchScratch, entry: usize, add_process: bool) {
    let lanes = scratch.lane_count;
    copy_augmented_basis_jet_to(
        &scratch.projected_basis,
        &scratch.projected_basis_eta,
        &scratch.projected_basis_eta_eta,
        entry,
        lanes,
        &mut scratch.bilinear_jets[..6 * lanes],
    );
    if add_process {
        for lane in 0..lanes {
            let process_variance = scratch.process_variance[lane];
            scratch.bilinear_jets[lane] += process_variance;
            scratch.bilinear_jets[2 * lanes + lane] += process_variance;
            scratch.bilinear_jets[5 * lanes + lane] += process_variance;
        }
    }
}

fn copy_augmented_basis_jet_to(
    basis: &[f64],
    basis_eta: &[f64],
    basis_eta_eta: &[f64],
    entry: usize,
    lanes: usize,
    output: &mut [f64],
) {
    output[..lanes].copy_from_slice(&basis[entry * lanes..(entry + 1) * lanes]);
    output[lanes..2 * lanes].copy_from_slice(&basis_eta[entry * lanes..(entry + 1) * lanes]);
    output[2 * lanes..3 * lanes].fill(0.0);
    output[3 * lanes..4 * lanes]
        .copy_from_slice(&basis_eta_eta[entry * lanes..(entry + 1) * lanes]);
    output[4 * lanes..6 * lanes].fill(0.0);
}

fn bilinear_jet(scratch: &BatchScratch, pair: usize, lane: usize) -> SecondOrderJet {
    let lanes = scratch.lane_count;
    let start = pair * 6 * lanes;
    SecondOrderJet {
        value: scratch.bilinear_jets[start + lane],
        eta: scratch.bilinear_jets[start + lanes + lane],
        log_q: scratch.bilinear_jets[start + 2 * lanes + lane],
        eta_eta: scratch.bilinear_jets[start + 3 * lanes + lane],
        eta_log_q: scratch.bilinear_jets[start + 4 * lanes + lane],
        log_q_log_q: scratch.bilinear_jets[start + 5 * lanes + lane],
    }
    .divide(SecondOrderJet {
        value: scratch.process_variance[lane],
        eta: 0.0,
        log_q: scratch.process_variance[lane],
        eta_eta: 0.0,
        eta_log_q: 0.0,
        log_q_log_q: scratch.process_variance[lane],
    })
}

fn finish_profile_derivatives(scratch: &mut BatchScratch, materialize_adjustment: bool) {
    let rank = scratch.realized_rank;
    let lanes = scratch.lane_count;
    for lane in 0..lanes {
        if !scratch.active[lane] || !scratch.positive_definite[lane] {
            continue;
        }
        let x_v_x = bilinear_jet(scratch, 0, lane);
        let x_v_y = bilinear_jet(scratch, 1, lane);
        let y_v_y = bilinear_jet(scratch, 2, lane);
        scratch.x_v_x[lane] = x_v_x.value;
        scratch.x_v_y[lane] = x_v_y.value;
        scratch.y_v_y[lane] = y_v_y.value;
        if materialize_adjustment {
            let slope = x_v_y.divide(x_v_x);
            scratch.slope_gradient_eta[lane] = slope.eta;
            scratch.slope_gradient_log_q[lane] = slope.log_q;
        }
        let quadratic = y_v_y.subtract(x_v_y.multiply(x_v_y).divide(x_v_x));
        let mut log_determinant_k = SecondOrderJet::constant(0.0);
        for component in 0..rank {
            let start = triangular_index(component, component) * 6 * lanes;
            let diagonal = SecondOrderJet {
                value: scratch.augmented_lower_jets[start + lane],
                eta: scratch.augmented_lower_jets[start + lanes + lane],
                log_q: scratch.augmented_lower_jets[start + 2 * lanes + lane],
                eta_eta: scratch.augmented_lower_jets[start + 3 * lanes + lane],
                eta_log_q: scratch.augmented_lower_jets[start + 4 * lanes + lane],
                log_q_log_q: scratch.augmented_lower_jets[start + 5 * lanes + lane],
            };
            log_determinant_k = log_determinant_k.add(diagonal.natural_log().scale(2.0));
        }
        let q = SecondOrderJet {
            value: scratch.log_process_variance[lane],
            eta: 0.0,
            log_q: 1.0,
            eta_eta: 0.0,
            eta_log_q: 0.0,
            log_q_log_q: 0.0,
        };
        let log_determinant_r = SecondOrderJet {
            value: scratch.log_determinant_r[lane],
            eta: scratch.log_determinant_r_eta[lane],
            log_q: 0.0,
            eta_eta: scratch.log_determinant_r_eta_eta[lane],
            eta_log_q: 0.0,
            log_q_log_q: 0.0,
        };
        let log_determinant = q
            .scale((scratch.date_count - rank) as f64)
            .add(SecondOrderJet::constant(2.0 * scratch.log_shape_sum[lane]))
            .add(log_determinant_r)
            .add(log_determinant_k);
        let score = log_determinant.add(quadratic).add(x_v_x.natural_log());
        scratch.score_gradient_eta[lane] = score.eta;
        scratch.score_gradient_log_q[lane] = score.log_q;
        scratch.score_curvature_eta[lane] = score.eta_eta;
        scratch.score_curvature_eta_log_q[lane] = score.eta_log_q;
        scratch.score_curvature_log_q[lane] = score.log_q_log_q;
    }
}

struct QuadraticKernel<'a> {
    scratch: &'a mut BatchScratch,
}

struct ProjectedQuadraticKernel<'a> {
    scratch: &'a mut BatchScratch,
}

impl WithSimd for ProjectedQuadraticKernel<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        quadratic_from_bilinears(simd, self.scratch);
    }
}

impl WithSimd for QuadraticKernel<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        form_quadratic(simd, self.scratch);
    }
}

#[inline(always)]
fn whiten<S: Simd>(simd: S, scratch: &mut BatchScratch) {
    let lanes = scratch.lane_count;
    scratch.whitened_x[..lanes].copy_from_slice(&scratch.z_x[..lanes]);
    scratch.whitened_y[..lanes].copy_from_slice(&scratch.z_y[..lanes]);
    scratch.whitened_factor[..scratch.realized_rank * lanes]
        .copy_from_slice(&scratch.scaled_factor[..scratch.realized_rank * lanes]);
    for row in 1..scratch.date_count {
        let transition = &scratch.transition[(row - 1) * lanes..row * lanes];
        let scale = &scratch.inverse_innovation_scale[(row - 1) * lanes..row * lanes];
        innovation(
            simd,
            &mut scratch.whitened_x[row * lanes..(row + 1) * lanes],
            &scratch.z_x[row * lanes..(row + 1) * lanes],
            &scratch.z_x[(row - 1) * lanes..row * lanes],
            transition,
            scale,
        );
        innovation(
            simd,
            &mut scratch.whitened_y[row * lanes..(row + 1) * lanes],
            &scratch.z_y[row * lanes..(row + 1) * lanes],
            &scratch.z_y[(row - 1) * lanes..row * lanes],
            transition,
            scale,
        );
        for component in 0..scratch.realized_rank {
            let current = (row * scratch.realized_rank + component) * lanes;
            let previous = ((row - 1) * scratch.realized_rank + component) * lanes;
            innovation(
                simd,
                &mut scratch.whitened_factor[current..current + lanes],
                &scratch.scaled_factor[current..current + lanes],
                &scratch.scaled_factor[previous..previous + lanes],
                transition,
                scale,
            );
        }
    }
}

#[inline(always)]
fn factorize<S: Simd>(simd: S, scratch: &mut BatchScratch) {
    let lanes = scratch.lane_count;
    scratch.lower.fill(0.0);
    for row in 0..scratch.realized_rank {
        for column in 0..=row {
            scratch.work_a.fill(0.0);
            for date in 0..scratch.date_count {
                let left = (date * scratch.realized_rank + row) * lanes;
                let right = (date * scratch.realized_rank + column) * lanes;
                multiply_accumulate(
                    simd,
                    &mut scratch.work_a,
                    &scratch.whitened_factor[left..left + lanes],
                    &scratch.whitened_factor[right..right + lanes],
                );
            }
            if row == column {
                add_in_place(simd, &mut scratch.work_a, &scratch.process_variance);
            }
            scratch.work_b.fill(0.0);
            for inner in 0..column {
                let left = triangular_index(row, inner) * lanes;
                let right = triangular_index(column, inner) * lanes;
                multiply_accumulate(
                    simd,
                    &mut scratch.work_b,
                    &scratch.lower[left..left + lanes],
                    &scratch.lower[right..right + lanes],
                );
            }
            let destination = triangular_index(row, column) * lanes;
            if row == column {
                difference(
                    simd,
                    &mut scratch.lower[destination..destination + lanes],
                    &scratch.work_a,
                    &scratch.work_b,
                );
                for lane in 0..lanes {
                    let diagonal = scratch.lower[destination + lane];
                    if scratch.active[lane] && (!diagonal.is_finite() || diagonal <= 0.0) {
                        scratch.positive_definite[lane] = false;
                        scratch.lower[destination + lane] = 1.0;
                    } else {
                        scratch.lower[destination + lane] = diagonal.sqrt();
                    }
                }
            } else {
                let denominator = triangular_index(column, column) * lanes;
                scratch
                    .work_c
                    .copy_from_slice(&scratch.lower[denominator..denominator + lanes]);
                difference_divide(
                    simd,
                    &mut scratch.lower[destination..destination + lanes],
                    &scratch.work_a,
                    &scratch.work_b,
                    &scratch.work_c,
                );
            }
        }
    }
}

#[inline(always)]
fn form_rhs<S: Simd>(simd: S, scratch: &mut BatchScratch) {
    let lanes = scratch.lane_count;
    scratch.h_x.fill(0.0);
    scratch.h_y.fill(0.0);
    for component in 0..scratch.realized_rank {
        let destination = component * lanes;
        for row in 0..scratch.date_count {
            let factor = (row * scratch.realized_rank + component) * lanes;
            multiply_accumulate(
                simd,
                &mut scratch.h_x[destination..destination + lanes],
                &scratch.whitened_factor[factor..factor + lanes],
                &scratch.whitened_x[row * lanes..(row + 1) * lanes],
            );
            multiply_accumulate(
                simd,
                &mut scratch.h_y[destination..destination + lanes],
                &scratch.whitened_factor[factor..factor + lanes],
                &scratch.whitened_y[row * lanes..(row + 1) * lanes],
            );
        }
    }
}

#[inline(always)]
fn solve_cholesky<S: Simd>(simd: S, scratch: &mut BatchScratch) {
    solve_one(simd, scratch, true);
    solve_one(simd, scratch, false);
}

#[inline(always)]
fn solve_one<S: Simd>(simd: S, scratch: &mut BatchScratch, x: bool) {
    let lanes = scratch.lane_count;
    let rhs = if x { &scratch.h_x } else { &scratch.h_y };
    let solution = if x {
        &mut scratch.solve_x
    } else {
        &mut scratch.solve_y
    };
    for row in 0..scratch.realized_rank {
        scratch.work_a.fill(0.0);
        for column in 0..row {
            let lower = triangular_index(row, column) * lanes;
            multiply_accumulate(
                simd,
                &mut scratch.work_a,
                &scratch.lower[lower..lower + lanes],
                &solution[column * lanes..(column + 1) * lanes],
            );
        }
        let diagonal = triangular_index(row, row) * lanes;
        difference_divide(
            simd,
            &mut solution[row * lanes..(row + 1) * lanes],
            &rhs[row * lanes..(row + 1) * lanes],
            &scratch.work_a,
            &scratch.lower[diagonal..diagonal + lanes],
        );
    }
    for row in (0..scratch.realized_rank).rev() {
        scratch.work_a.fill(0.0);
        for column in row + 1..scratch.realized_rank {
            let lower = triangular_index(column, row) * lanes;
            multiply_accumulate(
                simd,
                &mut scratch.work_a,
                &scratch.lower[lower..lower + lanes],
                &solution[column * lanes..(column + 1) * lanes],
            );
        }
        scratch
            .work_b
            .copy_from_slice(&solution[row * lanes..(row + 1) * lanes]);
        let diagonal = triangular_index(row, row) * lanes;
        difference_divide(
            simd,
            &mut solution[row * lanes..(row + 1) * lanes],
            &scratch.work_b,
            &scratch.work_a,
            &scratch.lower[diagonal..diagonal + lanes],
        );
    }
}

#[inline(always)]
fn subtract_product_accumulate<S: Simd>(simd: S, output: &mut [f64], left: &[f64], right: &[f64]) {
    let (output_vectors, output_tail) = S::f64s_as_mut_simd(output);
    let (left_vectors, left_tail) = S::f64s_as_simd(left);
    let (right_vectors, right_tail) = S::f64s_as_simd(right);
    for index in 0..output_vectors.len() {
        let product = simd.f64s_mul(left_vectors[index], right_vectors[index]);
        output_vectors[index] = simd.f64s_sub(output_vectors[index], product);
    }
    for index in 0..output_tail.len() {
        output_tail[index] -= left_tail[index] * right_tail[index];
    }
}

#[inline(always)]
fn subtract_jet_product<S: Simd>(
    simd: S,
    output: &mut [f64],
    left: &[f64],
    right: &[f64],
    lanes: usize,
) {
    subtract_product_accumulate(simd, &mut output[..lanes], &left[..lanes], &right[..lanes]);
    subtract_product_accumulate(
        simd,
        &mut output[lanes..2 * lanes],
        &left[lanes..2 * lanes],
        &right[..lanes],
    );
    subtract_product_accumulate(
        simd,
        &mut output[lanes..2 * lanes],
        &left[..lanes],
        &right[lanes..2 * lanes],
    );
    subtract_product_accumulate(
        simd,
        &mut output[2 * lanes..3 * lanes],
        &left[2 * lanes..3 * lanes],
        &right[..lanes],
    );
    subtract_product_accumulate(
        simd,
        &mut output[2 * lanes..3 * lanes],
        &left[..lanes],
        &right[2 * lanes..3 * lanes],
    );
    subtract_product_accumulate(
        simd,
        &mut output[3 * lanes..4 * lanes],
        &left[3 * lanes..4 * lanes],
        &right[..lanes],
    );
    subtract_scaled_product_accumulate(
        simd,
        &mut output[3 * lanes..4 * lanes],
        &left[lanes..2 * lanes],
        &right[lanes..2 * lanes],
        2.0,
    );
    subtract_product_accumulate(
        simd,
        &mut output[3 * lanes..4 * lanes],
        &left[..lanes],
        &right[3 * lanes..4 * lanes],
    );
    subtract_product_accumulate(
        simd,
        &mut output[4 * lanes..5 * lanes],
        &left[4 * lanes..5 * lanes],
        &right[..lanes],
    );
    subtract_product_accumulate(
        simd,
        &mut output[4 * lanes..5 * lanes],
        &left[lanes..2 * lanes],
        &right[2 * lanes..3 * lanes],
    );
    subtract_product_accumulate(
        simd,
        &mut output[4 * lanes..5 * lanes],
        &left[2 * lanes..3 * lanes],
        &right[lanes..2 * lanes],
    );
    subtract_product_accumulate(
        simd,
        &mut output[4 * lanes..5 * lanes],
        &left[..lanes],
        &right[4 * lanes..5 * lanes],
    );
    subtract_product_accumulate(
        simd,
        &mut output[5 * lanes..6 * lanes],
        &left[5 * lanes..6 * lanes],
        &right[..lanes],
    );
    subtract_scaled_product_accumulate(
        simd,
        &mut output[5 * lanes..6 * lanes],
        &left[2 * lanes..3 * lanes],
        &right[2 * lanes..3 * lanes],
        2.0,
    );
    subtract_product_accumulate(
        simd,
        &mut output[5 * lanes..6 * lanes],
        &left[..lanes],
        &right[5 * lanes..6 * lanes],
    );
}

fn square_root_jet(
    output: &mut [f64],
    input: &[f64],
    active: &[bool],
    positive_definite: &mut [bool],
    lanes: usize,
) {
    for lane in 0..lanes {
        let value = input[lane];
        if active[lane] && (!value.is_finite() || value <= 0.0) {
            positive_definite[lane] = false;
            output[lane] = 1.0;
            for derivative in 1..6 {
                output[derivative * lanes + lane] = 0.0;
            }
            continue;
        }
        let root = value.sqrt();
        let eta = input[lanes + lane] / (2.0 * root);
        let log_q = input[2 * lanes + lane] / (2.0 * root);
        output[lane] = root;
        output[lanes + lane] = eta;
        output[2 * lanes + lane] = log_q;
        output[3 * lanes + lane] = (input[3 * lanes + lane] - 2.0 * eta * eta) / (2.0 * root);
        output[4 * lanes + lane] = (input[4 * lanes + lane] - 2.0 * eta * log_q) / (2.0 * root);
        output[5 * lanes + lane] = (input[5 * lanes + lane] - 2.0 * log_q * log_q) / (2.0 * root);
    }
}

#[inline(always)]
fn divide_jet<S: Simd>(
    simd: S,
    output: &mut [f64],
    numerator: &[f64],
    denominator: &[f64],
    lanes: usize,
) {
    let (value, output) = output.split_at_mut(lanes);
    let (eta, output) = output.split_at_mut(lanes);
    let (log_q, output) = output.split_at_mut(lanes);
    let (eta_eta, output) = output.split_at_mut(lanes);
    let (eta_log_q, log_q_log_q) = output.split_at_mut(lanes);
    divide(simd, value, &numerator[..lanes], &denominator[..lanes]);
    eta.copy_from_slice(&numerator[lanes..2 * lanes]);
    subtract_product_accumulate(simd, eta, &denominator[lanes..2 * lanes], value);
    divide_in_place(simd, eta, &denominator[..lanes]);
    log_q.copy_from_slice(&numerator[2 * lanes..3 * lanes]);
    subtract_product_accumulate(simd, log_q, &denominator[2 * lanes..3 * lanes], value);
    divide_in_place(simd, log_q, &denominator[..lanes]);
    eta_eta.copy_from_slice(&numerator[3 * lanes..4 * lanes]);
    subtract_product_accumulate(simd, eta_eta, &denominator[3 * lanes..4 * lanes], value);
    subtract_scaled_product_accumulate(simd, eta_eta, &denominator[lanes..2 * lanes], eta, 2.0);
    divide_in_place(simd, eta_eta, &denominator[..lanes]);
    eta_log_q.copy_from_slice(&numerator[4 * lanes..5 * lanes]);
    subtract_product_accumulate(simd, eta_log_q, &denominator[4 * lanes..5 * lanes], value);
    subtract_product_accumulate(simd, eta_log_q, &denominator[lanes..2 * lanes], log_q);
    subtract_product_accumulate(simd, eta_log_q, &denominator[2 * lanes..3 * lanes], eta);
    divide_in_place(simd, eta_log_q, &denominator[..lanes]);
    log_q_log_q.copy_from_slice(&numerator[5 * lanes..6 * lanes]);
    subtract_product_accumulate(simd, log_q_log_q, &denominator[5 * lanes..6 * lanes], value);
    subtract_scaled_product_accumulate(
        simd,
        log_q_log_q,
        &denominator[2 * lanes..3 * lanes],
        log_q,
        2.0,
    );
    divide_in_place(simd, log_q_log_q, &denominator[..lanes]);
}

#[inline(always)]
fn subtract_scaled_product_accumulate<S: Simd>(
    simd: S,
    output: &mut [f64],
    left: &[f64],
    right: &[f64],
    scale: f64,
) {
    let (output_vectors, output_tail) = S::f64s_as_mut_simd(output);
    let (left_vectors, left_tail) = S::f64s_as_simd(left);
    let (right_vectors, right_tail) = S::f64s_as_simd(right);
    let scale_vector = simd.f64s_splat(scale);
    for index in 0..output_vectors.len() {
        let product = simd.f64s_mul(left_vectors[index], right_vectors[index]);
        output_vectors[index] =
            simd.f64s_sub(output_vectors[index], simd.f64s_mul(scale_vector, product));
    }
    for index in 0..output_tail.len() {
        output_tail[index] -= scale * left_tail[index] * right_tail[index];
    }
}

#[inline(always)]
fn form_bilinears<S: Simd>(simd: S, scratch: &mut BatchScratch) {
    let lanes = scratch.lane_count;
    scratch.x_v_x.fill(0.0);
    scratch.x_v_y.fill(0.0);
    scratch.y_v_y.fill(0.0);
    for row in 0..scratch.date_count {
        fitted_values(simd, scratch, row);
        difference_divide(
            simd,
            &mut scratch.work_c,
            &scratch.whitened_x[row * lanes..(row + 1) * lanes],
            &scratch.work_a,
            &scratch.process_standard_deviation,
        );
        difference_divide(
            simd,
            &mut scratch.work_d,
            &scratch.whitened_y[row * lanes..(row + 1) * lanes],
            &scratch.work_b,
            &scratch.process_standard_deviation,
        );
        multiply_accumulate(simd, &mut scratch.x_v_x, &scratch.work_c, &scratch.work_c);
        multiply_accumulate(simd, &mut scratch.x_v_y, &scratch.work_c, &scratch.work_d);
        multiply_accumulate(simd, &mut scratch.y_v_y, &scratch.work_d, &scratch.work_d);
    }
    scratch.work_a.fill(0.0);
    scratch.work_b.fill(0.0);
    scratch.work_c.fill(0.0);
    for component in 0..scratch.realized_rank {
        let start = component * lanes;
        multiply_accumulate(
            simd,
            &mut scratch.work_a,
            &scratch.solve_x[start..start + lanes],
            &scratch.solve_x[start..start + lanes],
        );
        multiply_accumulate(
            simd,
            &mut scratch.work_b,
            &scratch.solve_x[start..start + lanes],
            &scratch.solve_y[start..start + lanes],
        );
        multiply_accumulate(
            simd,
            &mut scratch.work_c,
            &scratch.solve_y[start..start + lanes],
            &scratch.solve_y[start..start + lanes],
        );
    }
    add_in_place(simd, &mut scratch.x_v_x, &scratch.work_a);
    add_in_place(simd, &mut scratch.x_v_y, &scratch.work_b);
    add_in_place(simd, &mut scratch.y_v_y, &scratch.work_c);
}

#[inline(always)]
fn fitted_values<S: Simd>(simd: S, scratch: &mut BatchScratch, row: usize) {
    let lanes = scratch.lane_count;
    scratch.work_a.fill(0.0);
    scratch.work_b.fill(0.0);
    for component in 0..scratch.realized_rank {
        let factor = (row * scratch.realized_rank + component) * lanes;
        let solve = component * lanes;
        multiply_accumulate(
            simd,
            &mut scratch.work_a,
            &scratch.whitened_factor[factor..factor + lanes],
            &scratch.solve_x[solve..solve + lanes],
        );
        multiply_accumulate(
            simd,
            &mut scratch.work_b,
            &scratch.whitened_factor[factor..factor + lanes],
            &scratch.solve_y[solve..solve + lanes],
        );
    }
}

#[inline(always)]
fn form_quadratic<S: Simd>(simd: S, scratch: &mut BatchScratch) {
    let lanes = scratch.lane_count;
    scratch.quadratic.fill(0.0);
    for row in 0..scratch.date_count {
        fitted_values(simd, scratch, row);
        difference(
            simd,
            &mut scratch.work_c,
            &scratch.whitened_x[row * lanes..(row + 1) * lanes],
            &scratch.work_a,
        );
        difference(
            simd,
            &mut scratch.work_d,
            &scratch.whitened_y[row * lanes..(row + 1) * lanes],
            &scratch.work_b,
        );
        subtract_product_divide(
            simd,
            &mut scratch.work_a,
            &scratch.work_d,
            &scratch.slope,
            &scratch.work_c,
            &scratch.process_standard_deviation,
        );
        multiply_accumulate(
            simd,
            &mut scratch.quadratic,
            &scratch.work_a,
            &scratch.work_a,
        );
    }
    scratch.work_b.fill(0.0);
    for component in 0..scratch.realized_rank {
        let start = component * lanes;
        subtract_product(
            simd,
            &mut scratch.work_a,
            &scratch.solve_y[start..start + lanes],
            &scratch.slope,
            &scratch.solve_x[start..start + lanes],
        );
        for lane in 0..lanes {
            scratch.work_b[lane] += scratch.work_a[lane].powi(2);
        }
    }
    add_in_place(simd, &mut scratch.quadratic, &scratch.work_b);
}

#[inline(always)]
fn quadratic_from_bilinears<S: Simd>(simd: S, scratch: &mut BatchScratch) {
    subtract_product(
        simd,
        &mut scratch.work_a,
        &scratch.y_v_y,
        &scratch.slope,
        &scratch.x_v_y,
    );
    subtract_product(
        simd,
        &mut scratch.work_b,
        &scratch.x_v_y,
        &scratch.slope,
        &scratch.x_v_x,
    );
    subtract_product(
        simd,
        &mut scratch.quadratic,
        &scratch.work_a,
        &scratch.slope,
        &scratch.work_b,
    );
}

const fn triangular_index(row: usize, column: usize) -> usize {
    row * (row + 1) / 2 + column
}

#[inline(always)]
fn innovation<S: Simd>(
    simd: S,
    output: &mut [f64],
    current: &[f64],
    previous: &[f64],
    transition: &[f64],
    scale: &[f64],
) {
    let (output_vectors, output_tail) = S::f64s_as_mut_simd(output);
    let (current_vectors, current_tail) = S::f64s_as_simd(current);
    let (previous_vectors, previous_tail) = S::f64s_as_simd(previous);
    let (transition_vectors, transition_tail) = S::f64s_as_simd(transition);
    let (scale_vectors, scale_tail) = S::f64s_as_simd(scale);
    for index in 0..output_vectors.len() {
        let predicted = simd.f64s_mul(transition_vectors[index], previous_vectors[index]);
        let difference = simd.f64s_sub(current_vectors[index], predicted);
        output_vectors[index] = simd.f64s_mul(difference, scale_vectors[index]);
    }
    for index in 0..output_tail.len() {
        output_tail[index] = (current_tail[index] - transition_tail[index] * previous_tail[index])
            * scale_tail[index];
    }
}

#[inline(always)]
fn multiply_accumulate<S: Simd>(simd: S, output: &mut [f64], left: &[f64], right: &[f64]) {
    let (output_vectors, output_tail) = S::f64s_as_mut_simd(output);
    let (left_vectors, left_tail) = S::f64s_as_simd(left);
    let (right_vectors, right_tail) = S::f64s_as_simd(right);
    for index in 0..output_vectors.len() {
        let product = simd.f64s_mul(left_vectors[index], right_vectors[index]);
        output_vectors[index] = simd.f64s_add(output_vectors[index], product);
    }
    for index in 0..output_tail.len() {
        output_tail[index] += left_tail[index] * right_tail[index];
    }
}

#[inline(always)]
fn basis_accumulate<S: Simd>(
    simd: S,
    output: &mut [f64],
    current: &[f64],
    cross: &[f64],
    previous: &[f64],
    transition: &[f64],
    inverse_innovation: &[f64],
) {
    let (output_vectors, output_tail) = S::f64s_as_mut_simd(output);
    let (current_vectors, current_tail) = S::f64s_as_simd(current);
    let (cross_vectors, cross_tail) = S::f64s_as_simd(cross);
    let (previous_vectors, previous_tail) = S::f64s_as_simd(previous);
    let (transition_vectors, transition_tail) = S::f64s_as_simd(transition);
    let (inverse_vectors, inverse_tail) = S::f64s_as_simd(inverse_innovation);
    for index in 0..output_vectors.len() {
        let transition_cross = simd.f64s_mul(transition_vectors[index], cross_vectors[index]);
        let transition_squared =
            simd.f64s_mul(transition_vectors[index], transition_vectors[index]);
        let previous_term = simd.f64s_mul(transition_squared, previous_vectors[index]);
        let difference = simd.f64s_sub(current_vectors[index], transition_cross);
        let numerator = simd.f64s_add(difference, previous_term);
        let weighted = simd.f64s_mul(numerator, inverse_vectors[index]);
        output_vectors[index] = simd.f64s_add(output_vectors[index], weighted);
    }
    for index in 0..output_tail.len() {
        let transition = transition_tail[index];
        let numerator = current_tail[index] - transition * cross_tail[index]
            + transition * transition * previous_tail[index];
        output_tail[index] += numerator * inverse_tail[index];
    }
}

#[inline(always)]
fn basis_derivative_accumulate<S: Simd>(
    simd: S,
    output: &mut [f64],
    current: &[f64],
    cross: &[f64],
    previous: &[f64],
    diagonal_weight: &[f64],
    cross_weight: &[f64],
) {
    let (output_vectors, output_tail) = S::f64s_as_mut_simd(output);
    let (current_vectors, current_tail) = S::f64s_as_simd(current);
    let (cross_vectors, cross_tail) = S::f64s_as_simd(cross);
    let (previous_vectors, previous_tail) = S::f64s_as_simd(previous);
    let (diagonal_vectors, diagonal_tail) = S::f64s_as_simd(diagonal_weight);
    let (cross_weight_vectors, cross_weight_tail) = S::f64s_as_simd(cross_weight);
    for index in 0..output_vectors.len() {
        let current_term = simd.f64s_mul(diagonal_vectors[index], current_vectors[index]);
        let cross_term = simd.f64s_mul(cross_weight_vectors[index], cross_vectors[index]);
        let previous_term = simd.f64s_mul(diagonal_vectors[index], previous_vectors[index]);
        let edge = simd.f64s_add(simd.f64s_add(current_term, cross_term), previous_term);
        output_vectors[index] = simd.f64s_add(output_vectors[index], edge);
    }
    for index in 0..output_tail.len() {
        output_tail[index] += diagonal_tail[index] * current_tail[index]
            + cross_weight_tail[index] * cross_tail[index]
            + diagonal_tail[index] * previous_tail[index];
    }
}

#[inline(always)]
fn add_in_place<S: Simd>(simd: S, output: &mut [f64], right: &[f64]) {
    let (output_vectors, output_tail) = S::f64s_as_mut_simd(output);
    let (right_vectors, right_tail) = S::f64s_as_simd(right);
    for index in 0..output_vectors.len() {
        output_vectors[index] = simd.f64s_add(output_vectors[index], right_vectors[index]);
    }
    for index in 0..output_tail.len() {
        output_tail[index] += right_tail[index];
    }
}

#[inline(always)]
fn difference<S: Simd>(simd: S, output: &mut [f64], left: &[f64], right: &[f64]) {
    let (output_vectors, output_tail) = S::f64s_as_mut_simd(output);
    let (left_vectors, left_tail) = S::f64s_as_simd(left);
    let (right_vectors, right_tail) = S::f64s_as_simd(right);
    for index in 0..output_vectors.len() {
        output_vectors[index] = simd.f64s_sub(left_vectors[index], right_vectors[index]);
    }
    for index in 0..output_tail.len() {
        output_tail[index] = left_tail[index] - right_tail[index];
    }
}

#[inline(always)]
fn divide<S: Simd>(simd: S, output: &mut [f64], numerator: &[f64], denominator: &[f64]) {
    let (output_vectors, output_tail) = S::f64s_as_mut_simd(output);
    let (numerator_vectors, numerator_tail) = S::f64s_as_simd(numerator);
    let (denominator_vectors, denominator_tail) = S::f64s_as_simd(denominator);
    for index in 0..output_vectors.len() {
        output_vectors[index] = simd.f64s_div(numerator_vectors[index], denominator_vectors[index]);
    }
    for index in 0..output_tail.len() {
        output_tail[index] = numerator_tail[index] / denominator_tail[index];
    }
}

#[inline(always)]
fn divide_in_place<S: Simd>(simd: S, output: &mut [f64], denominator: &[f64]) {
    let (output_vectors, output_tail) = S::f64s_as_mut_simd(output);
    let (denominator_vectors, denominator_tail) = S::f64s_as_simd(denominator);
    for index in 0..output_vectors.len() {
        output_vectors[index] = simd.f64s_div(output_vectors[index], denominator_vectors[index]);
    }
    for index in 0..output_tail.len() {
        output_tail[index] /= denominator_tail[index];
    }
}

#[inline(always)]
fn difference_divide<S: Simd>(
    simd: S,
    output: &mut [f64],
    left: &[f64],
    right: &[f64],
    denominator: &[f64],
) {
    let (output_vectors, output_tail) = S::f64s_as_mut_simd(output);
    let (left_vectors, left_tail) = S::f64s_as_simd(left);
    let (right_vectors, right_tail) = S::f64s_as_simd(right);
    let (denominator_vectors, denominator_tail) = S::f64s_as_simd(denominator);
    for index in 0..output_vectors.len() {
        let difference = simd.f64s_sub(left_vectors[index], right_vectors[index]);
        output_vectors[index] = simd.f64s_div(difference, denominator_vectors[index]);
    }
    for index in 0..output_tail.len() {
        output_tail[index] = (left_tail[index] - right_tail[index]) / denominator_tail[index];
    }
}

#[inline(always)]
fn subtract_product<S: Simd>(
    simd: S,
    output: &mut [f64],
    base: &[f64],
    left: &[f64],
    right: &[f64],
) {
    let (output_vectors, output_tail) = S::f64s_as_mut_simd(output);
    let (base_vectors, base_tail) = S::f64s_as_simd(base);
    let (left_vectors, left_tail) = S::f64s_as_simd(left);
    let (right_vectors, right_tail) = S::f64s_as_simd(right);
    for index in 0..output_vectors.len() {
        let product = simd.f64s_mul(left_vectors[index], right_vectors[index]);
        output_vectors[index] = simd.f64s_sub(base_vectors[index], product);
    }
    for index in 0..output_tail.len() {
        output_tail[index] = base_tail[index] - left_tail[index] * right_tail[index];
    }
}

#[inline(always)]
fn subtract_product_divide<S: Simd>(
    simd: S,
    output: &mut [f64],
    base: &[f64],
    left: &[f64],
    right: &[f64],
    denominator: &[f64],
) {
    let (output_vectors, output_tail) = S::f64s_as_mut_simd(output);
    let (base_vectors, base_tail) = S::f64s_as_simd(base);
    let (left_vectors, left_tail) = S::f64s_as_simd(left);
    let (right_vectors, right_tail) = S::f64s_as_simd(right);
    let (denominator_vectors, denominator_tail) = S::f64s_as_simd(denominator);
    for index in 0..output_vectors.len() {
        let product = simd.f64s_mul(left_vectors[index], right_vectors[index]);
        let difference = simd.f64s_sub(base_vectors[index], product);
        output_vectors[index] = simd.f64s_div(difference, denominator_vectors[index]);
    }
    for index in 0..output_tail.len() {
        output_tail[index] =
            (base_tail[index] - left_tail[index] * right_tail[index]) / denominator_tail[index];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_covariance::NuisanceBounds;

    #[test]
    fn bootstrap_interval_is_absent_when_every_refit_fails() {
        assert_eq!(bootstrap_interval(&[], 0.95), None);
    }

    #[test]
    fn nested_factor_batch_bounds_worker_arenas() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(12)
            .build()
            .unwrap();
        pool.install(|| {
            let days = [6.0, 12.0];
            let prepared = PreparedFactorObjective::new(&days, 12.0).unwrap();
            let target_count = 200;
            let observations = vec![0.0; days.len() * target_count];
            let factors = vec![1.0; days.len() * target_count];
            let ranks = vec![1; target_count];
            let execution =
                TemporalBatchExecution::new(&prepared, &observations, &factors, 1, &ranks).unwrap();
            assert_eq!(execution.metrics().worker_count, 4);
        });
    }

    #[test]
    fn shared_persisted_factor_batch_matches_repeated_factor_batch() {
        let post_gauge_days = (1..=12).map(|date| date as f64 * 12.0).collect::<Vec<_>>();
        let target_count = 17;
        let maximum_rank = post_gauge_days.len();
        let mut persisted_factor = vec![0.0; (post_gauge_days.len() + 1) * maximum_rank];
        for date in 0..post_gauge_days.len() {
            persisted_factor[(date + 1) * maximum_rank + date] = (0.5 + date as f64 * 0.01).sqrt();
        }
        let observations = (0..post_gauge_days.len() * target_count)
            .map(|index| {
                let date = index / target_count;
                let target = index % target_count;
                0.01 * post_gauge_days[date]
                    + (date as f64 * 0.37 + target as f64 * 0.11).sin() * 0.2
            })
            .collect::<Vec<_>>();
        let ranks = vec![maximum_rank; target_count];
        let options = TemporalCovarianceOptions {
            bootstrap_replicates: 0,
            bootstrap_minimum_successes: 0,
            ..Default::default()
        };
        let shared = fit_temporal_factor_scalar_batch_internal(
            &post_gauge_days,
            &observations,
            &persisted_factor,
            maximum_rank,
            &ranks,
            &options,
            false,
            true,
        )
        .unwrap();
        let repeated = fit_temporal_factor_scalar_batch_internal(
            &post_gauge_days,
            &observations,
            &persisted_factor.repeat(target_count),
            maximum_rank,
            &ranks,
            &options,
            false,
            true,
        )
        .unwrap();
        assert_eq!(shared.outcomes, repeated.outcomes);
        assert_eq!(
            shared.metrics.retained_factor_bytes * target_count,
            repeated.metrics.retained_factor_bytes
        );
        let mut shared_metrics = shared.metrics;
        shared_metrics.retained_factor_bytes = repeated.metrics.retained_factor_bytes;
        assert_eq!(shared_metrics, repeated.metrics);
    }

    #[test]
    fn batch_profile_compacts_without_revisiting_converged_lanes() {
        let mut arena = WorkerArena::new(2, 1, 8, None);
        arena.chunk.targets.extend(0..8);
        arena
            .chunk
            .outcomes
            .resize(8, Err(TemporalInferenceStatus::CovarianceNonfinite));
        let bounds = NuisanceBounds {
            rho_lower: 0.0,
            rho_upper: 0.98,
            log_variance_lower: -10.0,
            log_variance_upper: 10.0,
            initial_log_variance: 0.0,
        };
        arena.lane_states = (0..8).map(|_| ProfileLaneState::new(bounds)).collect();
        for date in 0..2 {
            for target in 0..8 {
                arena.chunk.scratch.observations[date * 8 + target] =
                    date as f64 * 100.0 + target as f64;
            }
        }
        let completion_round = [1_usize, 1, 1, 1, 1, 2, 2, 12];
        let mut visits = [0_usize; 8];
        let mut rho_evaluations = 3 * completion_round.len();
        for round in 1..=12 {
            for lane in 0..arena.chunk.targets.len() {
                if arena.lane_states[lane].finalized {
                    continue;
                }
                let target = arena.chunk.targets[lane];
                visits[target] += 1;
                rho_evaluations += 1;
                if completion_round[target] == round {
                    arena.lane_states[lane].finalized = true;
                }
            }
            arena.compact_unfinished_lanes();
            for (lane, &target) in arena.chunk.targets.iter().enumerate() {
                if arena.lane_states[lane].finalized {
                    continue;
                }
                for date in 0..2 {
                    assert_eq!(
                        arena.chunk.scratch.observations
                            [date * arena.chunk.scratch.lane_count + lane]
                            .to_bits(),
                        (date as f64 * 100.0 + target as f64).to_bits()
                    );
                }
            }
        }
        assert_eq!(visits, completion_round);
        assert_eq!(
            rho_evaluations,
            3 * completion_round.len() + completion_round.iter().sum::<usize>()
        );
        assert!(arena.chunk.targets.is_empty());
        assert_eq!(arena.chunk.scratch.lane_count, 0);
        assert_eq!(arena.profile_metrics.compaction_events, 3);
        assert_eq!(arena.profile_metrics.compacted_lane_count, 8);
    }
}

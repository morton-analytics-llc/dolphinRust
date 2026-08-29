//! Red/green analytic contracts for the pure temporal #53 kernel.

use dolphin_timeseries::{
    complete_refit_bootstrap_estimate, continuous_time_ar1_correlation, fit_temporal_covariance,
    fit_temporal_factor_scalar_batch, relative_standard_deviation_shape,
    subset_origin_anchored_covariance, temporal_covariance_provenance,
    temporal_covariance_workspace_composition, temporal_parameter_boundary_status,
    total_difference_covariance, CompleteRefitBootstrapCadenceStatus,
    CompleteRefitBootstrapEstimateStatus, Sha256Digest, TemporalCovarianceApproximation,
    TemporalCovarianceOptions, TemporalCovarianceProvenanceInputs, TemporalInferenceStatus,
    TemporalReferenceProvenance, TemporalValidationScope, COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS,
    COMPLETE_REFIT_BOOTSTRAP_MINIMUM_SUCCESSES,
};
use statrs::function::erf::erf;

#[test]
fn temporal_workspace_retains_relative_shape_through_bootstrap() {
    let acquisition_count: usize = 13;
    let capacity = acquisition_count.next_power_of_two();
    let matrix_bytes = (capacity * capacity * std::mem::size_of::<f64>()
        + capacity * std::mem::size_of::<Vec<f64>>()) as u64;
    let vector_bytes = (capacity * std::mem::size_of::<f64>()) as u64;
    let composition = temporal_covariance_workspace_composition(
        acquisition_count,
        COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS,
    )
    .unwrap();
    assert_eq!(
        composition.retained_fit_bytes,
        3 * matrix_bytes + 4 * vector_bytes
    );
}

#[test]
fn temporal_workspace_uses_vector_capacity_upper_bounds() {
    let thirteen =
        temporal_covariance_workspace_composition(13, COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS).unwrap();
    let sixteen =
        temporal_covariance_workspace_composition(16, COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS).unwrap();
    assert_eq!(thirteen.input_bytes, sixteen.input_bytes);
    assert_eq!(thirteen.retained_fit_bytes, sixteen.retained_fit_bytes);
}

fn twelve_date_fixture() -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    direct_factor_fixture(1.0)
}

fn direct_factor_fixture(diagonal: f64) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let days: Vec<f64> = (0..13).map(|index| index as f64 * 12.0).collect();
    let mut observations: Vec<f64> = days
        .iter()
        .enumerate()
        .map(|(index, day)| 0.01 * day + (index as f64 * 0.7).sin() * 2.0)
        .collect();
    observations[0] = 0.0;
    let mut covariance = vec![vec![0.0; days.len()]; days.len()];
    for (index, row) in covariance.iter_mut().enumerate().skip(1) {
        row[index] = diagonal;
    }
    (days, observations, covariance)
}

fn issue54_difference_variance(name: &str) -> f64 {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../dolphin-workflows/tests/fixtures/spatial_reference_covariance_cases.json"
    ))
    .unwrap();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == name)
        .unwrap();
    let target = case["expected_target_variance"].as_f64().unwrap();
    let reference = case["expected_reference_variance"].as_f64().unwrap();
    let cross = case["expected_cross_covariance"].as_f64().unwrap();
    let difference = case["expected_difference_variance"].as_f64().unwrap();
    assert_eq!(difference, target + reference - 2.0 * cross);
    difference
}

fn frozen_bootstrap_candidate_fixture() -> (
    dolphin_timeseries::TemporalCovarianceFit,
    TemporalCovarianceOptions,
) {
    let (days, observations, covariance) = twelve_date_fixture();
    let fit_options = TemporalCovarianceOptions {
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    let mut fit = fit_temporal_covariance(&days, &observations, &covariance, &fit_options);
    assert_eq!(fit.status, TemporalInferenceStatus::Evaluated);
    fit.bootstrap_slope = Some(3.5);
    fit.bootstrap_attempts = COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS;
    fit.bootstrap_successes = COMPLETE_REFIT_BOOTSTRAP_MINIMUM_SUCCESSES;
    fit.complete_refit_bootstrap.status = TemporalInferenceStatus::Evaluated;
    fit.complete_refit_bootstrap.point_estimate = Some(3.5);
    fit.complete_refit_bootstrap.standard_error_diagnostic = Some(0.4);
    fit.complete_refit_bootstrap.attempted_replicates = COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS;
    fit.complete_refit_bootstrap.successful_replicates = COMPLETE_REFIT_BOOTSTRAP_MINIMUM_SUCCESSES;
    (fit, TemporalCovarianceOptions::default())
}

#[test]
fn complete_refit_bootstrap_candidate_requires_frozen_evaluated_evidence() {
    let (fit, options) = frozen_bootstrap_candidate_fixture();
    let selected = complete_refit_bootstrap_estimate(&fit, &options);

    assert_eq!(
        selected.status,
        CompleteRefitBootstrapEstimateStatus::Evaluated
    );
    assert_eq!(selected.slope_per_year, Some(3.5));
    assert_eq!(selected.standard_error_per_year, Some(0.4));
    assert_eq!(selected.valid_date_count, fit.valid_date_count);
    assert_eq!(selected.rank, fit.rank);
    assert_eq!(selected.degrees_of_freedom, fit.degrees_of_freedom);
    assert_eq!(selected.raw_rho, fit.raw_correlation.rho);
    assert_eq!(selected.fitted_rho, fit.fitted_rho);
    assert_eq!(
        selected.fitted_process_variance,
        fit.fitted_process_variance
    );
    assert_eq!(
        selected.fitted_parameter_active_set,
        fit.fitted_parameter_active_set
    );
    assert_eq!(selected.condition_number, fit.covariance_condition_number);
    assert_eq!(
        selected.cadence_status,
        CompleteRefitBootstrapCadenceStatus::Supported
    );
    assert_eq!(selected.method, "complete_refit_bootstrap");
    assert_eq!(selected.method_version, 1);
    assert_eq!(
        selected.bootstrap_attempts,
        COMPLETE_REFIT_BOOTSTRAP_ATTEMPTS
    );
    assert_eq!(
        selected.bootstrap_successes,
        COMPLETE_REFIT_BOOTSTRAP_MINIMUM_SUCCESSES
    );
    assert_eq!(
        serde_json::to_value(selected).unwrap()["status"],
        "evaluated"
    );
    assert_eq!(
        serde_json::to_value(TemporalInferenceStatus::OptimizerNonconverged).unwrap(),
        "OptimizerNonconverged"
    );
}

#[test]
fn complete_refit_bootstrap_candidate_abstains_on_fit_or_comparator_failure() {
    let (mut fit, options) = frozen_bootstrap_candidate_fixture();
    fit.status = TemporalInferenceStatus::OptimizerNonconverged;
    let selected = complete_refit_bootstrap_estimate(&fit, &options);
    assert_eq!(
        selected.status,
        CompleteRefitBootstrapEstimateStatus::FitNotEvaluated
    );
    assert!(selected.slope_per_year.is_none());
    assert!(selected.standard_error_per_year.is_none());

    fit.status = TemporalInferenceStatus::UnsupportedCadence;
    let selected = complete_refit_bootstrap_estimate(&fit, &options);
    assert_eq!(
        selected.cadence_status,
        CompleteRefitBootstrapCadenceStatus::Unsupported
    );

    let (mut fit, options) = frozen_bootstrap_candidate_fixture();
    fit.complete_refit_bootstrap.status = TemporalInferenceStatus::BootstrapInsufficientSuccess;
    assert_eq!(
        complete_refit_bootstrap_estimate(&fit, &options).status,
        CompleteRefitBootstrapEstimateStatus::ComparatorNotEvaluated
    );
}

#[test]
fn complete_refit_bootstrap_candidate_rejects_nonfinite_or_inconsistent_values() {
    let (fit, options) = frozen_bootstrap_candidate_fixture();
    for (bootstrap_slope, point_estimate, standard_error) in [
        (Some(f64::NAN), Some(3.5), Some(0.4)),
        (Some(3.5), Some(f64::INFINITY), Some(0.4)),
        (Some(3.5), Some(3.5), Some(f64::NAN)),
        (Some(3.5), Some(3.5), Some(0.0)),
        (Some(3.5), Some(3.5), Some(-0.1)),
        (Some(3.6), Some(3.5), Some(0.4)),
    ] {
        let mut invalid = fit.clone();
        invalid.bootstrap_slope = bootstrap_slope;
        invalid.complete_refit_bootstrap.point_estimate = point_estimate;
        invalid.complete_refit_bootstrap.standard_error_diagnostic = standard_error;
        let selected = complete_refit_bootstrap_estimate(&invalid, &options);
        assert_eq!(
            selected.status,
            CompleteRefitBootstrapEstimateStatus::InvalidEstimate
        );
        assert!(selected.slope_per_year.is_none());
        assert!(selected.standard_error_per_year.is_none());
    }
}

#[test]
fn complete_refit_bootstrap_candidate_requires_exact_frozen_accounting() {
    let (fit, mut options) = frozen_bootstrap_candidate_fixture();
    options.bootstrap_replicates -= 1;
    assert_eq!(
        complete_refit_bootstrap_estimate(&fit, &options).status,
        CompleteRefitBootstrapEstimateStatus::FrozenConfigurationMismatch
    );

    let (mut fit, options) = frozen_bootstrap_candidate_fixture();
    fit.bootstrap_attempts -= 1;
    assert_eq!(
        complete_refit_bootstrap_estimate(&fit, &options).status,
        CompleteRefitBootstrapEstimateStatus::BootstrapAccountingMismatch
    );

    let (mut fit, options) = frozen_bootstrap_candidate_fixture();
    fit.complete_refit_bootstrap.successful_replicates -= 1;
    assert_eq!(
        complete_refit_bootstrap_estimate(&fit, &options).status,
        CompleteRefitBootstrapEstimateStatus::BootstrapAccountingMismatch
    );

    let (mut fit, options) = frozen_bootstrap_candidate_fixture();
    fit.bootstrap_successes = COMPLETE_REFIT_BOOTSTRAP_MINIMUM_SUCCESSES - 1;
    fit.complete_refit_bootstrap.successful_replicates = fit.bootstrap_successes;
    assert_eq!(
        complete_refit_bootstrap_estimate(&fit, &options).status,
        CompleteRefitBootstrapEstimateStatus::BootstrapInsufficientSuccess
    );
}

#[test]
fn continuous_time_covariance_uses_elapsed_days() {
    let correlation = continuous_time_ar1_correlation(&[0.0, 6.0, 24.0], 0.5, 12.0).unwrap();
    assert!((correlation[0][1] - 0.5_f64.powf(0.5)).abs() < 1e-12);
    assert!((correlation[0][2] - 0.5_f64.powf(2.0)).abs() < 1e-12);
    assert_eq!(correlation[1][2], correlation[2][1]);
}

#[test]
fn d_scaling_uses_geometric_mean_of_positive_diagonal() {
    let shape = relative_standard_deviation_shape(&[1.0, 4.0, 16.0]).unwrap();
    assert!((shape[0] - 0.5).abs() < 1e-12);
    assert!((shape[1] - 1.0).abs() < 1e-12);
    assert!((shape[2] - 2.0).abs() < 1e-12);
}

#[test]
fn missing_dates_subset_rows_columns_but_keeps_origin_gauge_out() {
    let covariance = vec![
        vec![0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.2, 0.1],
        vec![0.0, 0.2, 2.0, 0.3],
        vec![0.0, 0.1, 0.3, 3.0],
    ];
    let (days, values, selected) = subset_origin_anchored_covariance(
        &[0.0, 6.0, 12.0, 24.0],
        &[0.0, 1.0, f64::NAN, 4.0],
        &covariance,
    )
    .unwrap();
    assert_eq!(days, vec![6.0, 24.0]);
    assert_eq!(values, vec![1.0, 4.0]);
    assert_eq!(selected, vec![vec![1.0, 0.1], vec![0.1, 3.0]]);
}

#[test]
fn acquisition_zero_gauge_must_be_exactly_zero() {
    let covariance = vec![vec![0.0, 0.0], vec![0.0, 1.0]];
    assert_eq!(
        subset_origin_anchored_covariance(&[0.0, 12.0], &[f64::EPSILON, 1.0], &covariance),
        Err(TemporalInferenceStatus::GaugeNotZero)
    );
}

#[test]
fn even_length_gap_median_averages_the_two_center_gaps() {
    let diagnostics = dolphin_timeseries::raw_adjacent_correlation(
        &[0.0, 6.0, 18.0, 36.0, 60.0],
        &[0.0, 1.0, 0.5, 1.5, 1.0],
    );
    assert_eq!(diagnostics.median_gap_days, Some(15.0));
}

#[test]
fn direct_difference_factor_is_used_without_two_marginal_reconstruction() {
    let difference = vec![vec![1.0, -0.25], vec![-0.25, 4.0]];
    let total = total_difference_covariance(&difference, &[6.0, 24.0], 2.0, 0.0, 12.0).unwrap();
    assert!((total[0][0] - 2.0).abs() < 1e-12);
    assert_eq!(total[0][1], difference[0][1]);
    assert_eq!(total[1][0], difference[1][0]);
}

#[test]
fn direct_issue54_cross_covariance_controls_oracle_slope_standard_error() {
    let options = TemporalCovarianceOptions {
        oracle_rho: 0.3,
        oracle_process_variance: 1.0,
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    let standard_error = |difference_diagonal| {
        let (days, observations, covariance) = direct_factor_fixture(difference_diagonal);
        let fit = fit_temporal_covariance(&days, &observations, &covariance, &options);
        assert_eq!(fit.status, TemporalInferenceStatus::Evaluated);
        fit.oracle_gls.standard_error_diagnostic.unwrap()
    };

    let positive_cross = standard_error(issue54_difference_variance("positive"));
    let independent = standard_error(issue54_difference_variance("independent"));
    let negative_cross = standard_error(issue54_difference_variance("negative"));
    assert!(positive_cross < independent);
    assert!(independent < negative_cross);
}

#[test]
fn coincident_and_invalid_direct_issue54_factors_abstain() {
    let options = TemporalCovarianceOptions {
        minimum_dates: 2,
        oracle_process_variance: 0.0,
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    let days = [0.0, 12.0, 24.0];
    let observations = [0.0, 1.0, 2.0];
    let coincident = vec![vec![issue54_difference_variance("coincident"); 3]; 3];
    assert_eq!(
        fit_temporal_covariance(&days, &observations, &coincident, &options).status,
        TemporalInferenceStatus::CovarianceNonfinite
    );

    let asymmetric = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.5],
        vec![0.0, 0.25, 1.0],
    ];
    assert_eq!(
        fit_temporal_covariance(&days, &observations, &asymmetric, &options).status,
        TemporalInferenceStatus::CovarianceNonfinite
    );

    let nonfinite = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, f64::NAN],
        vec![0.0, f64::NAN, 1.0],
    ];
    assert_eq!(
        fit_temporal_covariance(&days, &observations, &nonfinite, &options).status,
        TemporalInferenceStatus::CovarianceNonfinite
    );

    let indefinite = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 2.0],
        vec![0.0, 2.0, 1.0],
    ];
    assert_eq!(
        fit_temporal_covariance(&days, &observations, &indefinite, &options).status,
        TemporalInferenceStatus::TotalCovarianceNotPositiveDefinite
    );
}

#[test]
fn short_irregular_covariance_uses_the_fitted_boundary_active_set() {
    let days = [6.0, 12.0, 30.0, 48.0];
    let observations = [0.1, 0.8, 1.4, 2.6];
    let difference = [
        vec![0.001, 0.0002, 0.0, 0.0],
        vec![0.0002, 0.001, 0.0004, 0.0],
        vec![0.0, 0.0004, 0.001, 0.0003],
        vec![0.0, 0.0, 0.0003, 0.001],
    ];
    let options = TemporalCovarianceOptions {
        oracle_rho: 0.6,
        oracle_process_variance: 1.0,
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        minimum_dates: 3,
        ..Default::default()
    };
    let fit = fit_temporal_covariance(
        &[0.0, days[0], days[1], days[2], days[3]],
        &[
            0.0,
            observations[0],
            observations[1],
            observations[2],
            observations[3],
        ],
        &{
            let mut matrix = vec![vec![0.0; 5]; 5];
            matrix[0][0] = 0.0;
            for (index, row) in difference.iter().enumerate() {
                for (column, value) in row.iter().enumerate() {
                    matrix[index + 1][column + 1] = *value;
                }
            }
            matrix
        },
        &options,
    );
    assert_eq!(fit.status, TemporalInferenceStatus::Evaluated);
    assert_eq!(
        fit.fitted_parameter_active_set,
        Some(TemporalInferenceStatus::RhoLowerBoundary)
    );
    assert!(fit.adjusted_scalar.interval_95.is_some());
}

#[test]
fn scalar_effective_n_has_closed_form_variance_and_coverage_counterexample() {
    let x = [1.0_f64, 3.0];
    let covariance = [[1.0_f64, 0.8], [0.8, 1.0]];
    let determinant = covariance[0][0] * covariance[1][1] - covariance[0][1].powi(2);
    let inverse = [
        [
            covariance[1][1] / determinant,
            -covariance[0][1] / determinant,
        ],
        [
            -covariance[1][0] / determinant,
            covariance[0][0] / determinant,
        ],
    ];
    let information = x
        .iter()
        .enumerate()
        .map(|(row, value)| {
            value
                * x.iter()
                    .enumerate()
                    .map(|(column, other)| inverse[row][column] * other)
                    .sum::<f64>()
        })
        .sum::<f64>();
    let true_variance = 1.0 / information;
    let rho = covariance[0][1];
    let effective_n = (2.0 * (1.0 - rho) / (1.0 + rho)).clamp(1.0, 2.0);
    let scalar_variance = 2.0 / effective_n / x.iter().map(|value| value * value).sum::<f64>();
    assert!((true_variance - scalar_variance).abs() > 0.05);
    let nominal_68_coverage = erf(1.0 / 2.0_f64.sqrt());
    assert!((nominal_68_coverage - 0.6827).abs() < 0.01);
    let wrong_scale = (scalar_variance / true_variance).sqrt();
    let wrong_coverage = erf(wrong_scale / 2.0_f64.sqrt());
    assert!((wrong_coverage - 0.6827).abs() > 0.05);
}

#[test]
fn invalid_dates_and_missing_gauge_fail_closed() {
    let options = TemporalCovarianceOptions {
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    let covariance = vec![vec![0.0, 0.0], vec![0.0, 1.0]];
    let invalid_dates = fit_temporal_covariance(&[0.0, 0.0], &[0.0, 1.0], &covariance, &options);
    assert_eq!(
        invalid_dates.status,
        TemporalInferenceStatus::DatesNotStrictlyIncreasing
    );
    let missing_gauge =
        fit_temporal_covariance(&[0.0, 1.0], &[f64::NAN, 1.0], &covariance, &options);
    assert_eq!(missing_gauge.status, TemporalInferenceStatus::GaugeMissing);
}

#[test]
fn default_minimum_dates_and_ml_profile_are_explicit() {
    assert_eq!(TemporalCovarianceOptions::default().minimum_dates, 12);
    assert_eq!(
        TemporalCovarianceOptions::default().bootstrap_replicates,
        200
    );
    assert_eq!(
        TemporalCovarianceOptions::default().bootstrap_minimum_successes,
        198
    );
    let (days, observations, covariance) = twelve_date_fixture();
    let options = TemporalCovarianceOptions {
        oracle_rho: 0.6,
        oracle_process_variance: 1.0,
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    let fit = fit_temporal_covariance(&days, &observations, &covariance, &options);
    assert_eq!(fit.status, TemporalInferenceStatus::Evaluated);
    assert!(fit.fitted_rho.unwrap().is_finite());
    assert!(fit.fitted_process_variance.unwrap().is_finite());
    assert!(fit.covariance_condition_number.unwrap() >= 1.0);
    assert_eq!(
        fit.conditional_wls.status,
        TemporalInferenceStatus::LegacyNonComparable
    );
    assert!(fit.conditional_wls.interval_95.is_none());
    assert!(fit.scalar_effective_n.interval_95.is_some());
    assert!(
        fit.adjusted_profile.interval_95.unwrap().upper
            > fit.adjusted_profile.interval_95.unwrap().lower
    );
    assert_ne!(
        fit.adjusted_profile.interval_95, fit.plugin_gls.interval_95,
        "profile likelihood must not alias plugin normal intervals"
    );
}

#[test]
fn bootstrap_emits_all_validation_levels_with_attempt_accounting() {
    let (days, observations, covariance) = twelve_date_fixture();
    let options = TemporalCovarianceOptions {
        bootstrap_replicates: 20,
        bootstrap_minimum_successes: 10,
        ..Default::default()
    };
    let fit = fit_temporal_covariance(&days, &observations, &covariance, &options);
    assert_eq!(fit.bootstrap_attempts, 20);
    assert!(fit.bootstrap_successes >= 10);
    assert!(fit.complete_refit_bootstrap.interval_68.is_some());
    assert!(fit.complete_refit_bootstrap.interval_90.is_some());
    assert!(fit.complete_refit_bootstrap.interval_95.is_some());
    assert_eq!(
        fit.complete_refit_bootstrap.point_estimate,
        fit.plugin_gls_slope
    );
    assert!(fit.ols.standard_error_diagnostic.is_some());
    assert!(fit.oracle_gls.standard_error_diagnostic.is_some());
    assert!(fit.plugin_gls.standard_error_diagnostic.is_some());
    assert!(fit.adjusted_scalar.standard_error_diagnostic.is_some());
    assert!(
        fit.adjusted_scalar.standard_error_diagnostic.unwrap()
            >= fit.plugin_gls.standard_error_diagnostic.unwrap()
    );
    assert!(fit.ols.width_68.unwrap() < fit.ols.width_95.unwrap());
    assert!(fit.oracle_gls.width_68.unwrap() < fit.oracle_gls.width_95.unwrap());
    assert!(fit.plugin_gls.width_68.unwrap() < fit.plugin_gls.width_95.unwrap());
    assert!(fit.adjusted_profile.width_68.unwrap() < fit.adjusted_profile.width_95.unwrap());
    assert!(
        fit.complete_refit_bootstrap.width_68.unwrap()
            < fit.complete_refit_bootstrap.width_95.unwrap()
    );
}

#[test]
fn invalid_profile_boundary_and_condition_fail_closed() {
    let (days, observations, covariance) = twelve_date_fixture();
    let boundary = TemporalCovarianceOptions {
        rho_max: 1.0,
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    assert_eq!(
        fit_temporal_covariance(&days, &observations, &covariance, &boundary).status,
        TemporalInferenceStatus::CovarianceParameterAtBoundary
    );
    let ill_conditioned: Vec<Vec<f64>> = (0..13)
        .map(|index| {
            (0..13)
                .map(|column| {
                    if index == column {
                        if index == 1 {
                            1e-8
                        } else {
                            1.0
                        }
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let options = TemporalCovarianceOptions {
        condition_limit: 10.0,
        oracle_process_variance: 0.0,
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    assert_eq!(
        fit_temporal_covariance(&days, &observations, &ill_conditioned, &options).status,
        TemporalInferenceStatus::DesignIllConditioned
    );
}

#[test]
fn crossed_or_nonfinite_rho_bounds_fail_before_optimization() {
    let (days, observations, covariance) = twelve_date_fixture();
    for (rho_min, rho_max) in [(0.5, 0.5), (0.8, 0.2), (f64::NAN, 0.98)] {
        let options = TemporalCovarianceOptions {
            rho_min,
            rho_max,
            bootstrap_replicates: 0,
            bootstrap_minimum_successes: 0,
            ..Default::default()
        };
        assert_eq!(
            fit_temporal_covariance(&days, &observations, &covariance, &options).status,
            TemporalInferenceStatus::CovarianceParameterAtBoundary
        );
    }
}

#[test]
fn optimizer_and_weak_identification_statuses_are_distinct() {
    let (days, observations, covariance) = twelve_date_fixture();
    let nonconverged = TemporalCovarianceOptions {
        optimizer_max_iterations: 0,
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    assert_eq!(
        fit_temporal_covariance(&days, &observations, &covariance, &nonconverged).status,
        TemporalInferenceStatus::OptimizerNonconverged
    );
    let exact: Vec<f64> = days.iter().map(|day| day * 0.01).collect();
    let weak = TemporalCovarianceOptions {
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    assert_eq!(
        fit_temporal_covariance(&days, &exact, &covariance, &weak).status,
        TemporalInferenceStatus::WeakParameterIdentification
    );
}

#[test]
fn fitted_parameter_boundaries_have_distinct_statuses() {
    assert_eq!(
        temporal_parameter_boundary_status(0.0, 1.0, [0.0, 0.98], [0.1, 10.0], 1e-6),
        Some(TemporalInferenceStatus::RhoLowerBoundary)
    );
    assert_eq!(
        temporal_parameter_boundary_status(0.98, 1.0, [0.0, 0.98], [0.1, 10.0], 1e-6),
        Some(TemporalInferenceStatus::RhoUpperBoundary)
    );
    assert_eq!(
        temporal_parameter_boundary_status(0.5, 0.1, [0.0, 0.98], [0.1, 10.0], 1e-6),
        Some(TemporalInferenceStatus::ProcessVarianceLowerBoundary)
    );
    assert_eq!(
        temporal_parameter_boundary_status(0.5, 10.0, [0.0, 0.98], [0.1, 10.0], 1e-6),
        Some(TemporalInferenceStatus::ProcessVarianceUpperBoundary)
    );
}

#[test]
fn fitted_rho_endpoint_uses_lower_boundary_active_set() {
    let (days, _, covariance) = direct_factor_fixture(0.01);
    let observations = days
        .iter()
        .enumerate()
        .map(|(index, day)| {
            if index == 0 {
                0.0
            } else {
                0.01 * day + if index % 2 == 0 { -0.5 } else { 0.5 }
            }
        })
        .collect::<Vec<_>>();
    let options = TemporalCovarianceOptions {
        bootstrap_replicates: 32,
        bootstrap_minimum_successes: 32,
        ..Default::default()
    };
    let fit = fit_temporal_covariance(&days, &observations, &covariance, &options);
    assert_eq!(fit.status, TemporalInferenceStatus::Evaluated);
    assert_eq!(
        fit.fitted_parameter_active_set,
        Some(TemporalInferenceStatus::RhoLowerBoundary)
    );
    assert!(fit.plugin_gls_slope.is_some());
    assert_eq!(
        fit.adjusted_scalar.status,
        TemporalInferenceStatus::Evaluated
    );
    assert_eq!(fit.bootstrap_attempts, 32);
    assert_eq!(fit.bootstrap_successes, 32);
    assert_eq!(
        fit.complete_refit_bootstrap.status,
        TemporalInferenceStatus::Evaluated
    );
}

#[test]
fn fitted_rho_endpoint_uses_upper_boundary_active_set() {
    let days: Vec<f64> = (0..13).map(|index| index as f64 * 12.0).collect();
    let observations = days
        .iter()
        .enumerate()
        .map(|(index, day)| 0.01 * day + (index as f64 * 0.07).sin() * 0.2)
        .collect::<Vec<_>>();
    let mut covariance = vec![vec![0.0; days.len()]; days.len()];
    for (index, row) in covariance.iter_mut().enumerate().skip(1) {
        row[index] = 1e-8;
    }
    let options = TemporalCovarianceOptions {
        rho_max: 0.9,
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    let fit = fit_temporal_covariance(&days, &observations, &covariance, &options);
    assert_eq!(fit.status, TemporalInferenceStatus::Evaluated);
    assert_eq!(
        fit.fitted_parameter_active_set,
        Some(TemporalInferenceStatus::RhoUpperBoundary),
        "{fit:?}"
    );
    assert!(fit.adjusted_profile.interval_95.is_some());
}

#[test]
fn fitted_process_variance_endpoints_use_active_set_inference() {
    let (days, observations, covariance) = twelve_date_fixture();
    for (minimum_ratio, maximum_ratio, expected) in [
        (
            2.0,
            10.0,
            TemporalInferenceStatus::ProcessVarianceLowerBoundary,
        ),
        (
            0.01,
            0.1,
            TemporalInferenceStatus::ProcessVarianceUpperBoundary,
        ),
    ] {
        let options = TemporalCovarianceOptions {
            process_variance_min_ratio: minimum_ratio,
            process_variance_max_ratio: maximum_ratio,
            bootstrap_replicates: 0,
            bootstrap_minimum_successes: 0,
            ..Default::default()
        };
        let fit = fit_temporal_covariance(&days, &observations, &covariance, &options);
        assert_eq!(fit.status, TemporalInferenceStatus::Evaluated, "{fit:?}");
        assert_eq!(fit.fitted_parameter_active_set, Some(expected), "{fit:?}");
        assert_eq!(
            fit.adjusted_scalar.status,
            TemporalInferenceStatus::Evaluated,
            "{fit:?}"
        );
        if expected == TemporalInferenceStatus::ProcessVarianceLowerBoundary {
            assert_eq!(
                fit.adjusted_scalar
                    .standard_error_diagnostic
                    .map(f64::to_bits),
                fit.plugin_gls.standard_error_diagnostic.map(f64::to_bits),
                "rho is unidentified and adds no nuisance adjustment when q is fixed at zero"
            );
        }
        assert!(fit.adjusted_profile.interval_95.is_some(), "{fit:?}");
    }
}

#[test]
fn factor_native_upper_boundary_active_set_matches_dense_fit() {
    let days: Vec<f64> = (0..13).map(|index| index as f64 * 12.0).collect();
    let observations = days
        .iter()
        .enumerate()
        .map(|(index, day)| 0.01 * day + (index as f64 * 0.07).sin() * 0.2)
        .collect::<Vec<_>>();
    let mut covariance = vec![vec![0.0; days.len()]; days.len()];
    let maximum_rank = days.len() - 1;
    let mut persisted_factor = vec![0.0; days.len() * maximum_rank];
    for index in 1..days.len() {
        covariance[index][index] = 1e-8;
        persisted_factor[index * maximum_rank + index - 1] = 1e-4;
    }
    let options = TemporalCovarianceOptions {
        rho_max: 0.9,
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    let dense = fit_temporal_covariance(&days, &observations, &covariance, &options);
    let factor = fit_temporal_factor_scalar_batch(
        &days[1..],
        &observations[1..],
        &persisted_factor,
        maximum_rank,
        &[maximum_rank],
        &options,
    )
    .unwrap();
    let factor = &factor.outcomes[0];
    assert_eq!(
        factor.fitted_parameter_active_set,
        dense.fitted_parameter_active_set
    );
    assert_eq!(
        factor.fitted_parameter_active_set,
        Some(TemporalInferenceStatus::RhoUpperBoundary)
    );
    assert_eq!(
        factor.reml_covariance_parameter_adjusted_scalar.status,
        TemporalInferenceStatus::Evaluated
    );
}

#[test]
fn factor_native_process_variance_active_sets_match_dense_fit() {
    let (days, observations, covariance) = twelve_date_fixture();
    let maximum_rank = days.len() - 1;
    let mut persisted_factor = vec![0.0; days.len() * maximum_rank];
    for index in 1..days.len() {
        persisted_factor[index * maximum_rank + index - 1] = 1.0;
    }
    for (minimum_ratio, maximum_ratio, expected) in [
        (
            2.0,
            10.0,
            TemporalInferenceStatus::ProcessVarianceLowerBoundary,
        ),
        (
            0.01,
            0.1,
            TemporalInferenceStatus::ProcessVarianceUpperBoundary,
        ),
    ] {
        let options = TemporalCovarianceOptions {
            process_variance_min_ratio: minimum_ratio,
            process_variance_max_ratio: maximum_ratio,
            bootstrap_replicates: 0,
            bootstrap_minimum_successes: 0,
            ..Default::default()
        };
        let dense = fit_temporal_covariance(&days, &observations, &covariance, &options);
        let factor = fit_temporal_factor_scalar_batch(
            &days[1..],
            &observations[1..],
            &persisted_factor,
            maximum_rank,
            &[maximum_rank],
            &options,
        )
        .unwrap();
        let factor = &factor.outcomes[0];
        assert_eq!(
            factor.fitted_parameter_active_set,
            dense.fitted_parameter_active_set
        );
        assert_eq!(factor.fitted_parameter_active_set, Some(expected));
        assert_eq!(
            factor.reml_covariance_parameter_adjusted_scalar.status,
            TemporalInferenceStatus::Evaluated
        );
        if expected == TemporalInferenceStatus::ProcessVarianceLowerBoundary {
            assert_eq!(
                factor
                    .reml_covariance_parameter_adjusted_scalar
                    .standard_error_diagnostic
                    .map(f64::to_bits),
                factor
                    .plugin_gls_reml
                    .standard_error_diagnostic
                    .map(f64::to_bits),
                "rho is unidentified and adds no nuisance adjustment when q is fixed at zero"
            );
        }
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn factor_adjusted_variance_fallback_preserves_inner_status() {
    let post_gauge_days = [
        12.0, 72.0, 84.0, 96.0, 108.0, 120.0, 132.0, 144.0, 156.0, 168.0, 180.0, 192.0,
    ];
    let observations = [
        0.5790543185703872,
        0.3454714938706878,
        0.3793593550031802,
        0.45341801167979534,
        0.9097372225076334,
        1.004745552203953,
        0.6122253422608358,
        0.4667671964469262,
        0.49160061690431744,
        0.9518507797137118,
        1.4268121135628533,
        1.467862764876594,
    ];
    let persisted_factor = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.28911381767013933,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.17728546347159616,
        0.8823102847503889,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.17737669428256214,
        0.8334923445403002,
        0.28648559821625336,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.17730353533533655,
        0.8233901138495123,
        0.1359851928376667,
        0.26046847072633245,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.17532438862817062,
        0.4077598744113832,
        0.07971804376168785,
        0.05794875942811115,
        0.7733682255932232,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1752103750726381,
        0.41226677588771593,
        0.08027975909350774,
        0.058375081274033444,
        0.7244211782761996,
        0.3074093790770856,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.17563340432718105,
        0.405289539120125,
        0.07896011899883917,
        0.05560571872196813,
        0.6995425096533624,
        0.11193590975763161,
        0.288563752825119,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.17170601640337552,
        0.4024270986947315,
        0.08017147236427316,
        0.059174813399150254,
        0.23397779081301695,
        0.026946912530560984,
        0.05779256282393859,
        0.7175618270066104,
        0.0,
        0.0,
        0.0,
        0.0,
        0.17193785889699023,
        0.40273990269734694,
        0.08030107373132016,
        0.05919466296978043,
        0.23391222173717713,
        0.02554751267426533,
        0.058880793589639525,
        0.6534431838557471,
        0.2801833635848498,
        0.0,
        0.0,
        0.0,
        0.1716158693730082,
        0.4022364204780767,
        0.08033840267743063,
        0.059353181146565306,
        0.2369438119029612,
        0.027070863021488034,
        0.060151260559963775,
        0.6712548772320652,
        0.13873125039970008,
        0.2582311579887401,
        0.0,
        0.0,
        0.16838924607065323,
        0.3948025961266679,
        0.07883271907859116,
        0.05854850727517735,
        0.23235002642592234,
        0.02514285961423379,
        0.05721570320281127,
        0.14661599739312395,
        0.026237502585798104,
        0.00704859568215559,
        0.7001466933767322,
        0.0,
        0.16759285164192297,
        0.39300385146376854,
        0.07849718395917518,
        0.05858879820771345,
        0.23177847694218023,
        0.024727006162657113,
        0.05676710184144539,
        0.15548875397673032,
        0.02651608110373496,
        0.011070718723818722,
        0.6474764528528059,
        0.3134316146162293,
    ];
    let options = TemporalCovarianceOptions {
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        bootstrap_seed: 1_651_721_217_991_215_729,
        minimum_profile_curvature: f64::MAX,
        ..Default::default()
    };
    let report = fit_temporal_factor_scalar_batch(
        &post_gauge_days,
        &observations,
        &persisted_factor,
        12,
        &[12],
        &options,
    )
    .unwrap();
    assert_eq!(report.metrics.exact_optimizer_fallback_targets, 1);
    let outcome = &report.outcomes[0];
    assert_eq!(
        outcome.reml_covariance_parameter_adjusted_scalar.status,
        TemporalInferenceStatus::WeakParameterIdentification
    );
    assert_eq!(
        serde_json::to_value(&outcome.reml_covariance_parameter_adjusted_scalar).unwrap()
            ["source_status"],
        serde_json::to_value(TemporalInferenceStatus::WeakParameterIdentification).unwrap()
    );
    let successful = serde_json::to_value(&outcome.plugin_gls_reml).unwrap();
    assert!(!successful
        .as_object()
        .unwrap()
        .contains_key("source_status"));
}

#[test]
fn unsupported_cadence_and_profile_failure_do_not_emit_selected_intervals() {
    let (mut days, observations, covariance) = twelve_date_fixture();
    days[5] = days[4] + 48.0;
    for index in 6..days.len() {
        days[index] = days[index - 1] + 12.0;
    }
    let options = TemporalCovarianceOptions {
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    assert_eq!(
        fit_temporal_covariance(&days, &observations, &covariance, &options).status,
        TemporalInferenceStatus::UnsupportedCadence
    );

    let (days, observations, covariance) = twelve_date_fixture();
    let nonconverged = TemporalCovarianceOptions {
        profile_max_iterations: 0,
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    let fit = fit_temporal_covariance(&days, &observations, &covariance, &nonconverged);
    assert_eq!(fit.status, TemporalInferenceStatus::OptimizerNonconverged);
    assert!(fit.adjusted_profile.interval_68.is_none());
    assert!(fit.adjusted_profile.interval_90.is_none());
    assert!(fit.adjusted_profile.interval_95.is_none());
}

#[test]
fn provenance_contract_binds_issue_52_issue_54_reference_and_receipt() {
    let (days, observations, covariance) = twelve_date_fixture();
    let options = TemporalCovarianceOptions {
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..Default::default()
    };
    let fit = fit_temporal_covariance(&days, &observations, &covariance, &options);
    let provenance = temporal_covariance_provenance(
        &fit,
        TemporalCovarianceProvenanceInputs {
            issue52_receipt_sha256: Sha256Digest::new("52".repeat(32)).unwrap(),
            issue54_receipt_sha256: Sha256Digest::new("54".repeat(32)).unwrap(),
            reference: TemporalReferenceProvenance {
                geometry_id: "burst/reference-window".to_owned(),
                window_id: "window-0".to_owned(),
                overlap_fraction: 0.95,
                distance_pixels: 12.0,
                sequential_depth: 2,
                approximation: TemporalCovarianceApproximation::Exact,
            },
            scope: TemporalValidationScope::SyntheticValidation,
            validation_receipt_sha256: Sha256Digest::new("53".repeat(32)).unwrap(),
            estimator_input_sha256: Sha256Digest::new("51".repeat(32)).unwrap(),
            selected_method: "complete_refit_bootstrap".to_owned(),
        },
    )
    .unwrap();
    assert_eq!(provenance.valid_date_count, 12);
    assert_eq!(provenance.rank, 1);
    assert_eq!(provenance.issue52_receipt_sha256.as_str().len(), 64);
    assert_eq!(provenance.issue54_receipt_sha256.as_str().len(), 64);
    assert_eq!(provenance.reference.geometry_id, "burst/reference-window");
    assert_eq!(
        provenance.fitted_parameter_active_set,
        fit.fitted_parameter_active_set
    );
    assert_eq!(provenance.bootstrap_attempts, 0);
    assert_eq!(
        provenance.validation_receipt_sha256.as_str(),
        "53".repeat(32)
    );
}

#[test]
fn failed_fit_never_emits_promotion_provenance() {
    let options = TemporalCovarianceOptions::default();
    let fit = fit_temporal_covariance(
        &[0.0, 12.0],
        &[1.0, 2.0],
        &[vec![0.0, 0.0], vec![0.0, 1.0]],
        &options,
    );
    assert_eq!(fit.status, TemporalInferenceStatus::GaugeNotZero);
    let digest = Sha256Digest::new("00".repeat(32)).unwrap();
    let inputs = TemporalCovarianceProvenanceInputs {
        issue52_receipt_sha256: digest.clone(),
        issue54_receipt_sha256: digest.clone(),
        reference: TemporalReferenceProvenance {
            geometry_id: "reference".to_owned(),
            window_id: "window".to_owned(),
            overlap_fraction: 1.0,
            distance_pixels: 0.0,
            sequential_depth: 1,
            approximation: TemporalCovarianceApproximation::Exact,
        },
        scope: TemporalValidationScope::SyntheticValidation,
        validation_receipt_sha256: digest.clone(),
        estimator_input_sha256: digest,
        selected_method: "complete_refit_bootstrap".to_owned(),
    };
    assert!(temporal_covariance_provenance(&fit, inputs).is_none());
}

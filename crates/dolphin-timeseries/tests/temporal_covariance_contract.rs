//! Red/green analytic contracts for the pure temporal #53 kernel.

use dolphin_timeseries::{
    continuous_time_ar1_correlation, fit_temporal_covariance, relative_standard_deviation_shape,
    subset_origin_anchored_covariance, temporal_covariance_provenance,
    temporal_parameter_boundary_status, total_difference_covariance, TemporalCovarianceOptions,
    TemporalCovarianceProvenanceInputs, TemporalInferenceStatus,
};
use statrs::function::erf::erf;

fn twelve_date_fixture() -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let days: Vec<f64> = (0..13).map(|index| index as f64 * 12.0).collect();
    let observations: Vec<f64> = days
        .iter()
        .enumerate()
        .map(|(index, day)| 0.01 * day + (index as f64 * 0.07).sin() * 0.2)
        .collect();
    let covariance: Vec<Vec<f64>> = (0..13).map(|_| (0..13).map(|_| 0.0).collect()).collect();
    let covariance = covariance
        .into_iter()
        .enumerate()
        .map(|(row, mut values)| {
            if row > 0 {
                values[row] = 1e-8;
            }
            values
        })
        .collect();
    (days, observations, covariance)
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
fn direct_difference_factor_is_used_without_two_marginal_reconstruction() {
    let difference = vec![vec![1.0, -0.25], vec![-0.25, 4.0]];
    let total = total_difference_covariance(&difference, &[6.0, 24.0], 2.0, 0.0, 12.0).unwrap();
    assert!((total[0][0] - 2.0).abs() < 1e-12);
    assert_eq!(total[0][1], difference[0][1]);
    assert_eq!(total[1][0], difference[1][0]);
}

#[test]
fn short_irregular_covariance_fails_weak_identification() {
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
    assert_eq!(
        fit.status,
        TemporalInferenceStatus::WeakParameterIdentification
    );
    assert!(fit.adjusted_profile.interval_95.is_none());
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
            issue52_receipt_sha256: "52".repeat(32),
            issue54_receipt_sha256: "54".repeat(32),
            reference_geometry: "burst/reference-window".to_owned(),
            reference_window: "window-0".to_owned(),
            overlap_fraction: 0.95,
            distance_pixels: 12.0,
            scope: "synthetic-validation".to_owned(),
            approximation: None,
            validation_receipt_sha256: "53".repeat(32),
        },
    );
    assert_eq!(provenance.valid_date_count, 12);
    assert_eq!(provenance.rank, 1);
    assert_eq!(provenance.issue52_receipt_sha256.len(), 64);
    assert_eq!(provenance.issue54_receipt_sha256.len(), 64);
    assert_eq!(provenance.reference_geometry, "burst/reference-window");
    assert_eq!(provenance.bootstrap_attempts, 0);
    assert_eq!(provenance.validation_receipt_sha256, "53".repeat(32));
}

//! Red/green analytic contracts for the pure temporal #53 kernel.

use dolphin_timeseries::{
    continuous_time_ar1_correlation, fit_temporal_covariance, relative_standard_deviation_shape,
    subset_origin_anchored_covariance, total_difference_covariance, TemporalCovarianceOptions,
    TemporalInferenceStatus,
};

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
fn scalar_effective_n_is_not_the_oracle_for_irregular_covariance() {
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
    assert!(fit.ols_slope.is_some());
    assert!(fit.oracle_gls_slope.is_some());
    assert!(fit.plugin_gls_slope.is_some());
    assert!((fit.ols_slope.unwrap() - fit.oracle_gls_slope.unwrap()).abs() > 1e-6);
    assert!(fit.bootstrap_interval.is_none());
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
fn default_minimum_dates_and_reml_fit_are_explicit() {
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
    assert!(
        fit.adjusted_profile.interval_95.unwrap().upper
            > fit.adjusted_profile.interval_95.unwrap().lower
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

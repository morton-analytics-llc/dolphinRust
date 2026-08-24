//! Contract tests for the fixed-valid-observation L2 E/H covariance map.

use dolphin_timeseries::{
    convert_covariance_units, date_contrast, solve_fixed_l2_spatial_covariance,
    solve_fixed_l2_spatial_covariance_from_factor, spatial_l2_branch_status, SpatialL2Branch,
    SpatialL2Status, FIXED_L2_SPATIAL_COVARIANCE_METHOD,
};
use ndarray::array;

#[test]
fn fixed_l2_replays_joint_observation_covariance_through_e_h() {
    let design = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let observations = array![1.0, 2.0, 3.0];
    let observation_covariance = array![[2.0, 0.5, 0.0], [0.5, 2.0, 0.25], [0.0, 0.25, 1.0]];
    let result = solve_fixed_l2_spatial_covariance(
        design.view(),
        observations.view(),
        observation_covariance.view(),
        array![1.0, 0.0].view(),
        array![0.0, 1.0].view(),
    )
    .expect("full-rank fixed L2 map");

    assert_eq!(result.status, SpatialL2Status::Valid);
    assert_eq!(result.method, FIXED_L2_SPATIAL_COVARIANCE_METHOD);
    assert_eq!(result.design_rank, 2);
    assert_eq!(result.observation_rank, 3);
    assert!(result.parameters.iter().all(|value| value.is_finite()));
    let identity = result.h_map.dot(&design);
    for (index, value) in identity.indexed_iter() {
        assert!((*value - f64::from(index.0 == index.1)).abs() < 1.0e-10);
    }
    assert!(
        (result.target_reference_covariance[(0, 1)] - result.target_reference_covariance[(1, 0)])
            .abs()
            < 1.0e-12
    );
    assert!(result.difference_covariance.is_finite());
    assert!(result.difference_covariance > 0.0);
    assert!(result.observation_log_pseudodeterminant.is_finite());
    assert!(result.normal_log_pseudodeterminant.is_finite());
    assert_eq!(result.covariance_diagonal().len(), 2);
    assert_eq!(result.covariance_block(&[1]).unwrap().dim(), (1, 1));
    assert_eq!(result.covariance_block(&[0, 1]).unwrap().dim(), (2, 2));
}

#[test]
fn gauge_and_unit_conversion_are_exact_and_branches_fail_closed() {
    assert_eq!(date_contrast(0, 4).unwrap(), array![0.0, 0.0, 0.0]);
    assert_eq!(date_contrast(2, 4).unwrap(), array![0.0, 1.0, 0.0]);
    assert_eq!(
        spatial_l2_branch_status(SpatialL2Branch::FixedL2),
        SpatialL2Status::Valid
    );
    assert_eq!(
        spatial_l2_branch_status(SpatialL2Branch::L1),
        SpatialL2Status::UnsupportedL1
    );
    assert_eq!(
        spatial_l2_branch_status(SpatialL2Branch::ChangedBranch),
        SpatialL2Status::UnsupportedChangedBranch
    );

    let covariance = array![[1.0, 0.25], [0.25, 2.0]];
    let converted = convert_covariance_units(covariance.view(), 1000.0).unwrap();
    assert_eq!(converted[(0, 0)], 1_000_000.0);
    assert_eq!(converted[(0, 1)], 250_000.0);
}

#[test]
fn rank_deficient_and_invalid_inputs_do_not_promote() {
    let error = solve_fixed_l2_spatial_covariance(
        array![[1.0, 1.0], [2.0, 2.0]].view(),
        array![1.0, 2.0].view(),
        array![[1.0, 0.0], [0.0, 1.0]].view(),
        array![1.0, 0.0].view(),
        array![0.0, 1.0].view(),
    )
    .unwrap_err();
    assert_eq!(error.status, SpatialL2Status::RankDeficient);
    let error = solve_fixed_l2_spatial_covariance(
        array![[1.0, 0.0]].view(),
        array![f64::NAN].view(),
        array![[1.0]].view(),
        array![1.0, 0.0].view(),
        array![0.0, 1.0].view(),
    )
    .unwrap_err();
    assert_eq!(error.status, SpatialL2Status::NonFinite);
}

#[test]
fn persisted_source_factor_congruence_matches_direct_h_c_h_transpose() {
    let design = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let observations = array![1.0, 2.0, 3.0];
    let source_factor = array![[1.0, 0.0], [0.2, 1.0], [0.1, 0.3]];
    let target = date_contrast(2, 3).unwrap();
    let reference = date_contrast(0, 3).unwrap();
    let result = solve_fixed_l2_spatial_covariance_from_factor(
        design.view(),
        observations.view(),
        source_factor.view(),
        target.view(),
        reference.view(),
    )
    .expect("bounded source-factor replay");
    let observation_covariance = source_factor.dot(&source_factor.t());
    let direct = solve_fixed_l2_spatial_covariance(
        design.view(),
        observations.view(),
        observation_covariance.view(),
        target.view(),
        reference.view(),
    )
    .expect("direct fixed-valid covariance");
    for ((index, factor_value), direct_value) in result
        .parameter_covariance
        .indexed_iter()
        .zip(direct.parameter_covariance.iter())
    {
        assert!((factor_value - direct_value).abs() < 1.0e-10, "{index:?}");
    }
    assert!((result.difference_covariance - direct.difference_covariance).abs() < 1.0e-10);
    assert_eq!(result.target_reference_factor.nrows(), 2);
    assert_eq!(result.difference_factor.nrows(), 1);
    let selected = result.covariance_block(&[0, 1]).unwrap();
    for (left, right) in selected.iter().zip(result.parameter_covariance.iter()) {
        assert!((left - right).abs() < 1.0e-12);
    }
    assert_eq!(
        result.covariance_diagonal().len(),
        result.parameter_factor.nrows()
    );
    assert!(result.e_map.row(1).iter().all(|value| *value == 0.0));
}

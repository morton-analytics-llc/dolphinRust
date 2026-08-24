use dolphin_core::Cf64;
use dolphin_phaselink::{
    estimate_empirical_proper_complex_factor, EmpiricalProperComplexConfig,
    EmpiricalSourceModelError, SourceId, EMPIRICAL_PROPER_COMPLEX_METHOD,
    EMPIRICAL_PROPER_COMPLEX_VERSION,
};
use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3};

fn stack() -> Array3<Cf64> {
    Array3::from_shape_fn((2, 4, 5), |(date, row, column)| {
        let x = (row * 5 + column) as f64;
        match date {
            0 => Cf64::new(1.0 + x, 0.5 * x),
            1 => Cf64::new(2.0 - 0.25 * x, 1.0 + x),
            _ => unreachable!(),
        }
    })
}

fn config(alpha: f64) -> EmpiricalProperComplexConfig {
    EmpiricalProperComplexConfig::new(1, 1, alpha, 1.0e-12, [11; 32]).unwrap()
}

fn estimate<'a>(
    values: ArrayView3<'a, Cf64>,
    valid: ArrayView2<'a, bool>,
    grid_origin: (usize, usize),
    source_pixel: (usize, usize),
    config: &EmpiricalProperComplexConfig,
) -> dolphin_phaselink::EmpiricalProperComplexEstimate {
    estimate_empirical_proper_complex_factor(
        SourceId::new(41),
        &[20240101, 20240113],
        values,
        valid,
        grid_origin,
        (0, 0),
        (4, 5),
        source_pixel,
        [17; 32],
        config,
    )
    .unwrap()
}

fn covariance_from_factor(lower: &Array2<Cf64>) -> Array2<Cf64> {
    Array2::from_shape_fn(lower.dim(), |(row, column)| {
        (0..lower.ncols())
            .map(|k| lower[(row, k)] * lower[(column, k)].conj())
            .sum()
    })
}

#[test]
fn reconstructs_zero_mean_shrunk_analytic_covariance() {
    let values = stack();
    let valid = Array2::from_elem((4, 5), true);
    let alpha = 0.25;
    let estimate = estimate(values.view(), valid.view(), (0, 0), (2, 2), &config(alpha));

    let mut samples = Vec::new();
    for row in 1..=3 {
        for column in 1..=3 {
            samples.push([values[(0, row, column)], values[(1, row, column)]]);
        }
    }
    let empirical = Array2::from_shape_fn((2, 2), |(row, column)| {
        samples
            .iter()
            .map(|sample| sample[row] * sample[column].conj())
            .sum::<Cf64>()
            / samples.len() as f64
    });
    let expected = Array2::from_shape_fn((2, 2), |(row, column)| {
        if row == column {
            empirical[(row, column)]
        } else {
            empirical[(row, column)] * (1.0 - alpha)
        }
    });

    let reconstructed = covariance_from_factor(estimate.factor().lower());
    for (actual, expected) in reconstructed.iter().zip(expected.iter()) {
        assert!((*actual - *expected).norm() < 1.0e-10);
    }
    assert_eq!(estimate.factor().component_ids(), &[20240101, 20240113]);
    assert_eq!(
        estimate.receipt().method(),
        "source_centered_empirical_proper_complex_v1"
    );
    assert_eq!(estimate.receipt().method(), EMPIRICAL_PROPER_COMPLEX_METHOD);
    assert_eq!(estimate.receipt().version(), 1);
    assert_eq!(
        estimate.receipt().version(),
        EMPIRICAL_PROPER_COMPLEX_VERSION
    );
    assert_eq!(estimate.receipt().sample_count(), 9);
}

#[test]
fn factor_is_target_consumer_independent_and_deterministic() {
    let values = stack();
    let valid = Array2::from_elem((4, 5), true);
    let config = config(0.2);

    let first = estimate(values.view(), valid.view(), (0, 0), (2, 2), &config);
    let second = estimate(values.view(), valid.view(), (0, 0), (2, 2), &config);

    assert_eq!(first.factor().source(), second.factor().source());
    assert_eq!(
        first.factor().component_ids(),
        second.factor().component_ids()
    );
    assert_eq!(first.factor().model_hash(), second.factor().model_hash());
    assert_eq!(first.factor().lower(), second.factor().lower());
    assert_eq!(first.receipt(), second.receipt());
}

#[test]
fn inward_clamps_fixed_support_at_native_border() {
    let values = stack();
    let valid = Array2::from_elem((4, 5), true);
    let estimate = estimate(values.view(), valid.view(), (0, 0), (0, 4), &config(0.3));

    assert_eq!(estimate.receipt().window_origin(), (0, 2));
    assert_eq!(estimate.receipt().window_shape(), (3, 3));
    assert_eq!(estimate.receipt().sample_count(), 9);
}

#[test]
fn identical_global_source_window_is_tile_equivalent() {
    let values = stack();
    let valid = Array2::from_elem((4, 5), true);
    let config = config(0.4);
    let whole = estimate(values.view(), valid.view(), (0, 0), (2, 2), &config);
    let tile = estimate(
        values.slice(s![.., 1..4, 1..4]),
        valid.slice(s![1..4, 1..4]),
        (1, 1),
        (2, 2),
        &config,
    );

    assert_eq!(whole.factor().model_hash(), tile.factor().model_hash());
    assert_eq!(whole.factor().lower(), tile.factor().lower());
    assert_eq!(whole.receipt(), tile.receipt());
}

#[test]
fn tile_without_canonical_native_halo_fails_instead_of_reclamping() {
    let values = stack();
    let valid = Array2::from_elem((4, 5), true);

    assert_eq!(
        estimate_empirical_proper_complex_factor(
            SourceId::new(41),
            &[20240101, 20240113],
            values.slice(s![.., 2..4, 2..5]),
            valid.slice(s![2..4, 2..5]),
            (2, 2),
            (0, 0),
            (4, 5),
            (2, 2),
            [17; 32],
            &config(0.2),
        )
        .unwrap_err(),
        EmpiricalSourceModelError::MissingSupport
    );
}

#[test]
fn invalid_nonfinite_and_missing_support_fail_closed() {
    assert_eq!(
        EmpiricalProperComplexConfig::new(1, 1, 0.0, 1.0e-12, [11; 32]),
        Err(EmpiricalSourceModelError::InvalidShrinkage)
    );
    assert_eq!(
        EmpiricalProperComplexConfig::new(1, 1, 0.2, 0.0, [11; 32]),
        Err(EmpiricalSourceModelError::InvalidRelativeDiagonalFloor)
    );
    assert_eq!(
        EmpiricalProperComplexConfig::new(1, 1, 0.2, 1.0e-12, [0; 32]),
        Err(EmpiricalSourceModelError::MissingModelIdentity)
    );

    let mut values = stack();
    let valid = Array2::from_elem((4, 5), true);
    values[(0, 2, 2)] = Cf64::new(f64::NAN, 0.0);
    assert_eq!(
        estimate_empirical_proper_complex_factor(
            SourceId::new(41),
            &[1, 2],
            values.view(),
            valid.view(),
            (0, 0),
            (0, 0),
            (4, 5),
            (2, 2),
            [17; 32],
            &config(0.2),
        )
        .unwrap_err(),
        EmpiricalSourceModelError::NonFiniteSample
    );

    let values = stack();
    let missing = Array2::from_elem((4, 5), false);
    assert_eq!(
        estimate_empirical_proper_complex_factor(
            SourceId::new(41),
            &[1, 2],
            values.view(),
            missing.view(),
            (0, 0),
            (0, 0),
            (4, 5),
            (2, 2),
            [17; 32],
            &config(0.2),
        )
        .unwrap_err(),
        EmpiricalSourceModelError::MissingSupport
    );

    let too_small = Array2::from_elem((2, 2), true);
    assert_eq!(
        estimate_empirical_proper_complex_factor(
            SourceId::new(41),
            &[1, 2],
            values.slice(s![.., ..2, ..2]),
            too_small.view(),
            (0, 0),
            (0, 0),
            (4, 5),
            (0, 0),
            [17; 32],
            &config(0.2),
        )
        .unwrap_err(),
        EmpiricalSourceModelError::MissingSupport
    );
}

#[test]
fn relative_diagonal_floor_rejects_underpowered_component() {
    let values = Array3::from_shape_fn((2, 3, 3), |(date, _, _)| match date {
        0 => Cf64::new(1.0, 0.0),
        1 => Cf64::new(1.0e-7, 0.0),
        _ => unreachable!(),
    });
    let valid = Array2::from_elem((3, 3), true);

    assert_eq!(
        estimate_empirical_proper_complex_factor(
            SourceId::new(41),
            &[1, 2],
            values.view(),
            valid.view(),
            (0, 0),
            (0, 0),
            (3, 3),
            (1, 1),
            [17; 32],
            &config(0.2),
        )
        .unwrap_err(),
        EmpiricalSourceModelError::DiagonalBelowRelativeFloor(1)
    );
}

#[test]
fn relative_rank_floor_rejects_nearly_collinear_components() {
    let values = Array3::from_elem((2, 3, 3), Cf64::new(1.0, 0.0));
    let valid = Array2::from_elem((3, 3), true);

    assert_eq!(
        estimate_empirical_proper_complex_factor(
            SourceId::new(41),
            &[1, 2],
            values.view(),
            valid.view(),
            (0, 0),
            (0, 0),
            (3, 3),
            (1, 1),
            [17; 32],
            &config(1.0e-15),
        )
        .unwrap_err(),
        EmpiricalSourceModelError::RankBelowRelativeFloor(1)
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn receipt_digest_binds_content_config_and_date_order() {
    let values = stack();
    let valid = Array2::from_elem((4, 5), true);
    let baseline = estimate(values.view(), valid.view(), (0, 0), (2, 2), &config(0.2));

    let mut changed_values = values.clone();
    changed_values[(0, 2, 2)] += Cf64::new(0.125, -0.25);
    let content = estimate(
        changed_values.view(),
        valid.view(),
        (0, 0),
        (2, 2),
        &config(0.2),
    );
    let changed_config = estimate(values.view(), valid.view(), (0, 0), (2, 2), &config(0.3));
    let changed_floor = estimate(
        values.view(),
        valid.view(),
        (0, 0),
        (2, 2),
        &EmpiricalProperComplexConfig::new(1, 1, 0.2, 1.0e-10, [11; 32]).unwrap(),
    );
    let reversed = estimate_empirical_proper_complex_factor(
        SourceId::new(41),
        &[20240113, 20240101],
        values.slice(s![..;-1, .., ..]),
        valid.view(),
        (0, 0),
        (0, 0),
        (4, 5),
        (2, 2),
        [17; 32],
        &config(0.2),
    )
    .unwrap();
    let changed_data_identity = estimate_empirical_proper_complex_factor(
        SourceId::new(41),
        &[20240101, 20240113],
        values.view(),
        valid.view(),
        (0, 0),
        (0, 0),
        (4, 5),
        (2, 2),
        [18; 32],
        &config(0.2),
    )
    .unwrap();
    let changed_model_identity = estimate(
        values.view(),
        valid.view(),
        (0, 0),
        (2, 2),
        &EmpiricalProperComplexConfig::new(1, 1, 0.2, 1.0e-12, [12; 32]).unwrap(),
    );
    let changed_source = estimate_empirical_proper_complex_factor(
        SourceId::new(42),
        &[20240101, 20240113],
        values.view(),
        valid.view(),
        (0, 0),
        (0, 0),
        (4, 5),
        (2, 2),
        [17; 32],
        &config(0.2),
    )
    .unwrap();

    assert_ne!(baseline.receipt().digest(), content.receipt().digest());
    assert_ne!(
        baseline.receipt().digest(),
        changed_config.receipt().digest()
    );
    assert_ne!(baseline.receipt().digest(), reversed.receipt().digest());
    assert_ne!(
        baseline.receipt().digest(),
        changed_floor.receipt().digest()
    );
    assert_ne!(
        baseline.receipt().digest(),
        changed_data_identity.receipt().digest()
    );
    assert_eq!(
        baseline.factor().model_hash(),
        content.factor().model_hash()
    );
    assert_ne!(
        baseline.factor().model_hash(),
        changed_config.factor().model_hash()
    );
    assert_eq!(
        baseline.factor().model_hash(),
        reversed.factor().model_hash()
    );
    assert_eq!(
        baseline.factor().model_hash(),
        changed_data_identity.factor().model_hash()
    );
    assert_ne!(
        baseline.factor().model_hash(),
        changed_model_identity.factor().model_hash()
    );
    assert_eq!(
        baseline.factor().model_hash(),
        changed_source.factor().model_hash()
    );
    assert_ne!(
        baseline.factor().numeric_receipt_digest(),
        content.factor().numeric_receipt_digest()
    );
    assert_ne!(
        baseline.factor().numeric_receipt_digest(),
        reversed.factor().numeric_receipt_digest()
    );
    assert_eq!(
        baseline.factor().numeric_receipt_digest(),
        changed_data_identity.factor().numeric_receipt_digest()
    );
    assert_ne!(
        baseline.factor().numeric_receipt_digest(),
        changed_source.factor().numeric_receipt_digest()
    );
}

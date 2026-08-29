//! Issue #54 local shared-source influence contracts.

use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_phaselink::covariance::replay_rect_pixel_covariance;
use dolphin_phaselink::estimator::{
    phase_angle_jvp, process_coherence_matrix, FixedEstimatorBranch, PhaseAngleLinearization,
};
use dolphin_phaselink::{
    contract_source_factors, reference_specific_influence_v1, NativeSourcePixel,
    ProperComplexFactor, SpatialInfluenceError,
};
use ndarray::{array, Array1, Array2, Array3};
use std::collections::BTreeMap;

const EPS: f64 = 1e-6;
const TOLERANCE: f64 = 3e-5;

fn stack() -> Array3<Cf64> {
    Array3::from_shape_fn((4, 5, 5), |(date, row, column)| {
        let amplitude = 0.9 + 0.11 * date as f64 + 0.03 * row as f64 + 0.02 * column as f64;
        let phase = 0.21 * date as f64 + 0.13 * row as f64 - 0.07 * column as f64;
        Cf64::from_polar(amplitude, phase)
    })
}

fn source_factors() -> BTreeMap<NativeSourcePixel, ProperComplexFactor> {
    let lower = Array2::from_diag(&Array1::from_elem(4, Cf64::new(1.0, 0.0)));
    (0..5)
        .flat_map(|row| (0..5).map(move |column| (row, column)))
        .map(|(row, column)| {
            let pixel = NativeSourcePixel::new(row, column);
            let source = (row * 5 + column) as u64;
            let factor = ProperComplexFactor::new(
                dolphin_phaselink::SourceId::new(source),
                (0..4).map(|component| 100 + component).collect(),
                [source as u8 + 1; 32],
                lower.clone(),
            )
            .unwrap();
            (pixel, factor)
        })
        .collect()
}

fn canonical_direction(component: usize, imaginary: bool) -> Array1<Cf64> {
    Array1::from_shape_fn(4, |date| {
        if date != component {
            return Cf64::new(0.0, 0.0);
        }
        let scale = std::f64::consts::FRAC_1_SQRT_2;
        if imaginary {
            Cf64::new(0.0, scale)
        } else {
            Cf64::new(scale, 0.0)
        }
    })
}

fn perturb_source(
    stack: &Array3<Cf64>,
    source: NativeSourcePixel,
    direction: &Array1<Cf64>,
    scale: f64,
) -> Array3<Cf64> {
    let mut values = stack.clone();
    for (date, delta) in direction.iter().enumerate() {
        values[(date, source.row, source.column)] += scale * *delta;
    }
    values
}

fn phase(
    stack: &Array3<Cf64>,
    output: (usize, usize),
    half_window: HalfWindow,
    strides: Strides,
    validity: &Array2<bool>,
    branch: FixedEstimatorBranch,
) -> Array1<Cf64> {
    let replay =
        replay_rect_pixel_covariance(stack.view(), output, half_window, strides, validity.view())
            .unwrap();
    let estimate = match branch {
        FixedEstimatorBranch::Evd => {
            process_coherence_matrix(replay.coherence.view(), true, 0.0, 0.0, 0)
        }
        FixedEstimatorBranch::Emi {
            beta,
            zero_correlation_threshold,
        } => process_coherence_matrix(
            replay.coherence.view(),
            false,
            beta,
            zero_correlation_threshold,
            0,
        ),
    };
    estimate.phase
}

fn angle_difference(plus: Cf64, minus: Cf64) -> f64 {
    (plus * minus.conj()).arg() / (2.0 * EPS)
}

#[test]
fn analytic_source_factor_fixtures_preserve_cross_covariance() {
    let cases = [
        (array![[1.0, 0.0]], array![[0.0, 1.0]], 2.0, 0.0),
        (array![[1.0, 0.0]], array![[0.5, 0.5]], 0.5, 0.5),
        (array![[1.0, 0.0]], array![[-1.0, 0.0]], 4.0, -1.0),
        (array![[1.0, 0.0]], array![[1.0, 0.0]], 0.0, 1.0),
    ];
    for (target, reference, expected_difference, expected_cross) in cases {
        let result = contract_source_factors(target.view(), reference.view()).unwrap();
        assert!((result.difference_covariance[(0, 0)] - expected_difference).abs() < 1e-12);
        assert!((result.target_reference_covariance[(0, 0)] - expected_cross).abs() < 1e-12);
    }
    assert!(contract_source_factors(array![[1.0]].view(), array![[1.0, 2.0]].view()).is_err());
}

#[test]
fn spatial_covariance_is_symmetric_psd_and_not_the_issue_52_marginal_sum() {
    let target = array![[1.0, 0.0], [0.0, 1.0]];
    let reference = array![[0.5, 0.5], [0.0, 0.0]];
    let result = contract_source_factors(target.view(), reference.view()).unwrap();
    for covariance in [
        &result.target_covariance,
        &result.reference_covariance,
        &result.difference_covariance,
    ] {
        for row in 0..covariance.nrows() {
            for column in 0..covariance.ncols() {
                assert!((covariance[(row, column)] - covariance[(column, row)]).abs() < 1e-12);
            }
        }
        for vector in [array![1.0, -0.5], array![-0.25, 2.0]] {
            let quadratic = vector.dot(&covariance.dot(&vector));
            assert!(quadratic >= -1e-12, "quadratic form {quadratic}");
        }
    }
    let marginal_sum = &result.target_covariance + &result.reference_covariance;
    assert!((result.difference_covariance[(0, 0)] - marginal_sum[(0, 0)]).abs() > 1e-6);
    assert!(result.target_reference_covariance[(0, 0)].abs() > 1e-6);
}

#[test]
fn rect_border_stride_mask_and_shared_source_keys_are_explicit() {
    let stack = stack();
    let mut validity = Array2::from_elem((5, 5), true);
    validity[(2, 2)] = false;
    let factors = source_factors();
    let result = reference_specific_influence_v1(
        stack.view(),
        (0, 0),
        (1, 1),
        HalfWindow { y: 1, x: 1 },
        Strides { y: 2, x: 2 },
        validity.view(),
        &factors,
        FixedEstimatorBranch::Evd,
        0,
        1e-9,
        1.0,
    )
    .unwrap();
    assert!(result
        .source_pixels
        .iter()
        .all(|pixel| validity[(pixel.row, pixel.column)]));
    assert_eq!(
        result.target_factor.ncols(),
        *result.source_factor_offsets.last().unwrap()
    );
    assert_eq!(result.reference_factor.dim(), result.target_factor.dim());
    assert!(result.target_factor.iter().any(|value| value.abs() > 1e-12));
    let target_only = result
        .source_pixels
        .iter()
        .position(|pixel| pixel.row == 0 && pixel.column == 0)
        .unwrap();
    let target_start = result.source_factor_offsets[target_only];
    let target_end = result.source_factor_offsets[target_only + 1];
    assert!(result
        .target_factor
        .slice(ndarray::s![.., target_start..target_end])
        .iter()
        .any(|value| value.abs() > 1e-12));
    assert!(result
        .reference_factor
        .slice(ndarray::s![.., target_start..target_end])
        .iter()
        .all(|value| value.abs() <= 1e-12));
    assert!(result
        .difference_covariance
        .row(0)
        .iter()
        .all(|value| *value == 0.0));
    assert!(result
        .difference_covariance
        .column(0)
        .iter()
        .all(|value| *value == 0.0));
    assert!(result
        .difference_covariance
        .diag()
        .iter()
        .all(|value| *value >= -1e-12));
}

#[test]
fn evd_and_emi_source_jvps_match_finite_difference() {
    let stack = stack();
    let validity = Array2::from_elem((5, 5), true);
    let source = NativeSourcePixel::new(1, 1);
    let factors = source_factors();
    for branch in [
        FixedEstimatorBranch::Evd,
        FixedEstimatorBranch::Emi {
            beta: 0.2,
            zero_correlation_threshold: 1e-8,
        },
    ] {
        let result = reference_specific_influence_v1(
            stack.view(),
            (1, 1),
            (1, 1),
            HalfWindow { y: 1, x: 1 },
            Strides { y: 1, x: 1 },
            validity.view(),
            &factors,
            branch,
            0,
            1e-9,
            1.0,
        )
        .unwrap();
        let source_column = result
            .source_pixels
            .iter()
            .position(|pixel| *pixel == source)
            .unwrap();
        for component in 0..4 {
            for imaginary in [false, true] {
                let direction = canonical_direction(component, imaginary);
                let plus = perturb_source(&stack, source, &direction, EPS);
                let minus = perturb_source(&stack, source, &direction, -EPS);
                let plus_phase = phase(
                    &plus,
                    (1, 1),
                    HalfWindow { y: 1, x: 1 },
                    Strides { y: 1, x: 1 },
                    &validity,
                    branch,
                );
                let minus_phase = phase(
                    &minus,
                    (1, 1),
                    HalfWindow { y: 1, x: 1 },
                    Strides { y: 1, x: 1 },
                    &validity,
                    branch,
                );
                let offset = result.source_factor_offsets[source_column]
                    + component
                    + usize::from(imaginary) * 4;
                for date in 0..4 {
                    assert!(
                        (angle_difference(plus_phase[date], minus_phase[date])
                            - result.target_factor[(date, offset)])
                            .abs()
                            < TOLERANCE,
                        "branch={branch:?} date={date} component={component} imaginary={imaginary}"
                    );
                }
            }
        }
    }
}

#[test]
fn prepared_phase_linearization_matches_repeated_jvps_exactly() {
    let stack = stack();
    let validity = Array2::from_elem((5, 5), true);
    let replay = replay_rect_pixel_covariance(
        stack.view(),
        (1, 1),
        HalfWindow { y: 1, x: 1 },
        Strides { y: 1, x: 1 },
        validity.view(),
    )
    .unwrap();
    let dimensions = replay.coherence.dim();
    let directions = [
        Array2::from_shape_fn(dimensions, |(row, column)| {
            Cf64::new(
                (row + column + 1) as f64 * 1e-3,
                (row as f64 - column as f64) * 2e-4,
            )
        }),
        Array2::from_shape_fn(dimensions, |(row, column)| {
            Cf64::new(
                (row * column + 1) as f64 * -3e-4,
                (column as f64 - row as f64) * 1e-4,
            )
        }),
    ];
    for branch in [
        FixedEstimatorBranch::Evd,
        FixedEstimatorBranch::Emi {
            beta: 0.2,
            zero_correlation_threshold: 1e-8,
        },
    ] {
        let prepared =
            PhaseAngleLinearization::prepare(replay.coherence.view(), branch, 0, 1e-9).unwrap();
        for direction in &directions {
            let expected =
                phase_angle_jvp(replay.coherence.view(), direction.view(), branch, 0, 1e-9)
                    .unwrap();
            assert_eq!(prepared.apply(direction.view()).unwrap(), expected);
        }
    }
}

#[test]
fn gauge_change_is_a_congruence_of_the_same_source_factor() {
    let stack = stack();
    let validity = Array2::from_elem((5, 5), true);
    let factors = source_factors();
    let zero = reference_specific_influence_v1(
        stack.view(),
        (1, 1),
        (1, 2),
        HalfWindow { y: 1, x: 1 },
        Strides { y: 1, x: 1 },
        validity.view(),
        &factors,
        FixedEstimatorBranch::Evd,
        0,
        1e-9,
        1.0,
    )
    .unwrap();
    let one = reference_specific_influence_v1(
        stack.view(),
        (1, 1),
        (1, 2),
        HalfWindow { y: 1, x: 1 },
        Strides { y: 1, x: 1 },
        validity.view(),
        &factors,
        FixedEstimatorBranch::Evd,
        1,
        1e-9,
        1.0,
    )
    .unwrap();
    assert_eq!(zero.target_factor.ncols(), one.target_factor.ncols());
    for row in 0..zero.target_factor.nrows() {
        for column in 0..zero.target_factor.ncols() {
            assert!(
                (one.target_factor[(row, column)]
                    - (zero.target_factor[(row, column)] - zero.target_factor[(1, column)]))
                    .abs()
                    < 1e-10
            );
        }
    }
    for row in 0..one.difference_covariance.nrows() {
        for column in 0..one.difference_covariance.ncols() {
            let expected = zero.difference_covariance[(row, column)]
                - zero.difference_covariance[(1, column)]
                - zero.difference_covariance[(row, 1)]
                + zero.difference_covariance[(1, 1)];
            assert!((one.difference_covariance[(row, column)] - expected).abs() < 1e-9);
        }
    }
}

#[test]
fn invalid_reference_and_nonfinite_branch_fail_closed() {
    let stack = stack();
    let validity = Array2::from_elem((5, 5), true);
    let factors = source_factors();
    let error = reference_specific_influence_v1(
        stack.view(),
        (9, 9),
        (1, 1),
        HalfWindow { y: 1, x: 1 },
        Strides { y: 1, x: 1 },
        validity.view(),
        &factors,
        FixedEstimatorBranch::Evd,
        0,
        1e-9,
        1.0,
    )
    .unwrap_err();
    assert_eq!(error, SpatialInfluenceError::InvalidReference);
    assert_eq!(error.status().as_str(), "invalid_reference");

    let mut missing_factor = factors;
    missing_factor.remove(&NativeSourcePixel::new(0, 0));
    let error = reference_specific_influence_v1(
        stack.view(),
        (1, 1),
        (1, 2),
        HalfWindow { y: 1, x: 1 },
        Strides { y: 1, x: 1 },
        validity.view(),
        &missing_factor,
        FixedEstimatorBranch::Evd,
        0,
        1e-9,
        1.0,
    )
    .unwrap_err();
    assert_eq!(error.status().as_str(), "invalid_source_factor");

    let error = reference_specific_influence_v1(
        stack.view(),
        (1, 1),
        (1, 2),
        HalfWindow { y: 1, x: 1 },
        Strides { y: 1, x: 1 },
        Array2::from_elem((5, 5), false).view(),
        &source_factors(),
        FixedEstimatorBranch::Evd,
        0,
        1e-9,
        1.0,
    )
    .unwrap_err();
    assert_eq!(error.status().as_str(), "replay_failure");

    let error = reference_specific_influence_v1(
        stack.view(),
        (1, 1),
        (1, 2),
        HalfWindow { y: 1, x: 1 },
        Strides { y: 1, x: 1 },
        Array2::from_elem((5, 5), true).view(),
        &source_factors(),
        FixedEstimatorBranch::Evd,
        0,
        0.5,
        1.0,
    )
    .unwrap_err();
    assert_eq!(error.status().as_str(), "unsupported_branch");
}

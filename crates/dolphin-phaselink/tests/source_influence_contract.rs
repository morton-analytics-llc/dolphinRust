//! Local analytic contracts for the issue #52 source-influence replay kernels.

use dolphin_core::config::ComputeBackend;
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_phaselink::covariance::{
    normalize_numerator_jvp, rect_pixel_source_coherence_jvp, replay_rect_pixel_covariance,
    replay_rect_source_values, CovarianceReplayError, NativeSourcePixel,
};
use dolphin_phaselink::estimator::{phase_angle_jvp, EstimatorJvpError, FixedEstimatorBranch};
use dolphin_phaselink::quality::{
    compress_pixel_jvp, compress_with_replay, CompressionJvpError, CompressionReplayStatus,
};
use dolphin_phaselink::{
    ComputeEngine, FixedBranchStatus, FusedParams, InfluenceDag, InfluenceError, InfluenceNode,
    NodeId, ParentEdge, ProperComplexFactor, SourceDefinition, SourceEdge, SourceId,
    TemporalCoordinate,
};
use ndarray::{array, Array1, Array2, Array3};

const EPS: f64 = 1e-6;
const JVP_TOL: f64 = 2e-6;

fn stack() -> Array3<Cf64> {
    Array3::from_shape_fn((4, 3, 3), |(t, r, c)| {
        let amplitude = 0.8 + 0.17 * t as f64 + 0.04 * r as f64 + 0.03 * c as f64;
        let phase =
            0.31 * t as f64 + 0.13 * r as f64 - 0.09 * c as f64 + 0.027 * (t * r + c) as f64;
        Cf64::from_polar(amplitude, phase)
    })
}

fn perturb_source(
    stack: &Array3<Cf64>,
    source: NativeSourcePixel,
    direction: &Array1<Cf64>,
    scale: f64,
) -> Array3<Cf64> {
    let mut perturbed = stack.clone();
    for (t, delta) in direction.iter().enumerate() {
        perturbed[(t, source.row, source.column)] += scale * delta;
    }
    perturbed
}

fn angle_difference(plus: Cf64, minus: Cf64) -> f64 {
    (plus * minus.conj()).arg() / (2.0 * EPS)
}

#[test]
fn proper_complex_factor_has_canonical_real_embedding() {
    let lower = array![
        [Cf64::new(2.0, 0.0), Cf64::new(0.0, 0.0)],
        [Cf64::new(0.5, -0.25), Cf64::new(1.5, 0.0)]
    ];
    let factor =
        ProperComplexFactor::new(SourceId::new(7), vec![101, 102], [9; 32], lower.clone()).unwrap();
    let got = factor.real_embedding();
    let scale = std::f64::consts::FRAC_1_SQRT_2;
    let expected = array![
        [2.0, 0.0, -0.0, -0.0],
        [0.5, 1.5, 0.25, -0.0],
        [0.0, 0.0, 2.0, 0.0],
        [-0.25, 0.0, 0.5, 1.5]
    ] * scale;
    assert_eq!(got.dim(), (4, 4));
    assert!(got
        .iter()
        .zip(expected.iter())
        .all(|(actual, expected)| (actual - expected).abs() < 1e-15));

    let not_lower = array![
        [Cf64::new(1.0, 0.0), Cf64::new(0.1, 0.0)],
        [Cf64::new(0.0, 0.0), Cf64::new(1.0, 0.0)]
    ];
    assert!(ProperComplexFactor::new(SourceId::new(8), vec![1, 2], [8; 32], not_lower).is_err());
}

#[test]
fn influence_graph_validation_fails_closed() {
    let mut dag = InfluenceDag::new();
    assert_eq!(
        dag.add_source(SourceDefinition::new(SourceId::new(1), 1, [0; 32])),
        Err(InfluenceError::MissingModelHash)
    );
    dag.add_source(SourceDefinition::new(SourceId::new(1), 1, [1; 32]))
        .unwrap();
    assert_eq!(
        dag.add_node(
            InfluenceNode::new(NodeId::new(1), 1)
                .with_source(SourceEdge::new(SourceId::new(1), array![[1.0, 2.0]])),
        ),
        Err(InfluenceError::ShapeMismatch)
    );
    assert_eq!(
        dag.add_node(
            InfluenceNode::new(NodeId::new(2), 1)
                .with_parent(ParentEdge::new(NodeId::new(99), array![[1.0]])),
        ),
        Err(InfluenceError::UnknownNode(NodeId::new(99)))
    );
    dag.add_node(
        InfluenceNode::new(NodeId::new(3), 1)
            .with_source(SourceEdge::new(SourceId::new(1), array![[1.0]])),
    )
    .unwrap();
    assert_eq!(
        dag.temporal_covariance(&[TemporalCoordinate::node(NodeId::new(3), 1)]),
        Err(InfluenceError::ComponentOutOfBounds {
            node: NodeId::new(3),
            component: 1,
            dimension: 1,
        })
    );
}

#[test]
fn factor_bound_source_contracts_through_real_embedding() {
    let factor = ProperComplexFactor::new(
        SourceId::new(40),
        vec![400],
        [40; 32],
        array![[Cf64::new(2.0, 0.0)]],
    )
    .unwrap();
    let mut dag = InfluenceDag::new();
    dag.add_source(factor.source_definition().unwrap()).unwrap();
    let node = NodeId::new(41);
    dag.add_node(
        InfluenceNode::new(node, 1).with_source(
            factor
                .bind_real_jacobian(array![[1.0, 2.0]].view())
                .unwrap(),
        ),
    )
    .unwrap();
    let covariance = dag
        .temporal_covariance(&[TemporalCoordinate::node(node, 0)])
        .unwrap();
    assert!((covariance[(0, 0)] - 10.0).abs() < 1e-12);
}

#[test]
fn influence_graph_rejects_nonfinite_contraction() {
    let mut dag = InfluenceDag::new();
    let source = SourceId::new(50);
    dag.add_source(SourceDefinition::new(source, 1, [50; 32]))
        .unwrap();
    let parent = NodeId::new(51);
    dag.add_node(
        InfluenceNode::new(parent, 1).with_source(SourceEdge::new(source, array![[1e308]])),
    )
    .unwrap();
    let child = NodeId::new(52);
    dag.add_node(
        InfluenceNode::new(child, 1).with_parent(ParentEdge::new(parent, array![[1e308]])),
    )
    .unwrap();
    assert_eq!(
        dag.temporal_covariance(&[TemporalCoordinate::node(child, 0)]),
        Err(InfluenceError::NonFiniteContraction)
    );
}

#[test]
fn rect_replay_exposes_exact_support_and_source_jvp() {
    let stack = stack();
    let valid = Array2::from_elem((3, 3), true);
    let half = HalfWindow { y: 1, x: 1 };
    let strides = Strides { y: 1, x: 1 };
    let replay =
        replay_rect_pixel_covariance(stack.view(), (1, 1), half, strides, valid.view()).unwrap();
    assert_eq!(replay.source_pixels.len(), 9);
    assert_eq!(replay.source_pixels[0], NativeSourcePixel::new(0, 0));
    assert_eq!(replay.source_pixels[8], NativeSourcePixel::new(2, 2));
    let source_values = Array2::from_shape_fn(
        (stack.dim().0, replay.source_pixels.len()),
        |(date, index)| {
            let source = replay.source_pixels[index];
            stack[(date, source.row, source.column)]
        },
    );
    let bounded = replay_rect_source_values(
        replay.descriptor,
        replay.output,
        &replay.source_pixels,
        source_values.view(),
    )
    .unwrap();
    assert_eq!(bounded.numerator, replay.numerator);
    assert_eq!(bounded.coherence, replay.coherence);
    assert_eq!(
        replay
            .descriptor
            .nearest_output(NativeSourcePixel::new(2, 2))
            .unwrap(),
        (2, 2)
    );

    let source = NativeSourcePixel::new(1, 2);
    let direction = array![
        Cf64::new(0.11, -0.07),
        Cf64::new(-0.04, 0.09),
        Cf64::new(0.08, 0.03),
        Cf64::new(-0.05, -0.06),
    ];
    let analytic =
        rect_pixel_source_coherence_jvp(stack.view(), &replay, source, direction.view(), 1e-10)
            .unwrap();
    let plus = perturb_source(&stack, source, &direction, EPS);
    let minus = perturb_source(&stack, source, &direction, -EPS);
    let plus =
        replay_rect_pixel_covariance(plus.view(), (1, 1), half, strides, valid.view()).unwrap();
    let minus =
        replay_rect_pixel_covariance(minus.view(), (1, 1), half, strides, valid.view()).unwrap();
    let numeric = (&plus.coherence - &minus.coherence) / (2.0 * EPS);
    let error = analytic
        .iter()
        .zip(numeric.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    assert!(error < JVP_TOL, "coherence source JVP error {error:e}");
}

fn check_phase_jvp(branch: FixedEstimatorBranch) {
    let stack = stack();
    let valid = Array2::from_elem((3, 3), true);
    let half = HalfWindow { y: 1, x: 1 };
    let strides = Strides { y: 1, x: 1 };
    let replay =
        replay_rect_pixel_covariance(stack.view(), (1, 1), half, strides, valid.view()).unwrap();
    let source = NativeSourcePixel::new(0, 1);
    let direction = array![
        Cf64::new(0.03, 0.07),
        Cf64::new(-0.06, 0.02),
        Cf64::new(0.08, -0.04),
        Cf64::new(-0.02, -0.05),
    ];
    let dc =
        rect_pixel_source_coherence_jvp(stack.view(), &replay, source, direction.view(), 1e-10)
            .unwrap();
    let analytic = phase_angle_jvp(replay.coherence.view(), dc.view(), branch, 0, 1e-9).unwrap();

    let estimate = |scale: f64| {
        let perturbed = perturb_source(&stack, source, &direction, scale);
        let replay =
            replay_rect_pixel_covariance(perturbed.view(), (1, 1), half, strides, valid.view())
                .unwrap();
        let (use_evd, beta, cut) = match branch {
            FixedEstimatorBranch::Evd => (true, 0.0, 0.0),
            FixedEstimatorBranch::Emi {
                beta,
                zero_correlation_threshold,
            } => (false, beta, zero_correlation_threshold),
        };
        dolphin_phaselink::process_coherence_matrix(replay.coherence.view(), use_evd, beta, cut, 0)
    };
    let plus = estimate(EPS);
    let minus = estimate(-EPS);
    assert_eq!(plus.estimator, minus.estimator);
    for i in 0..analytic.len() {
        let numeric = angle_difference(plus.phase[i], minus.phase[i]);
        assert!(
            (analytic[i] - numeric).abs() < JVP_TOL,
            "phase[{i}] JVP {} != {numeric} for {branch:?}",
            analytic[i]
        );
    }
}

#[test]
fn evd_phase_source_jvp_matches_raw_complex_difference() {
    check_phase_jvp(FixedEstimatorBranch::Evd);
}

#[test]
fn emi_phase_source_jvp_matches_raw_complex_difference() {
    check_phase_jvp(FixedEstimatorBranch::Emi {
        beta: 0.2,
        zero_correlation_threshold: 0.0,
    });
}

#[test]
fn estimator_jvp_rejects_emi_fallback_and_selected_tie() {
    let zero = Array2::from_elem((3, 3), Cf64::new(0.0, 0.0));
    let singular = Array2::from_elem((3, 3), Cf64::new(1.0, 0.0));
    assert_eq!(
        phase_angle_jvp(
            singular.view(),
            zero.view(),
            FixedEstimatorBranch::Emi {
                beta: 0.0,
                zero_correlation_threshold: 0.0,
            },
            0,
            1e-10,
        ),
        Err(EstimatorJvpError::EmiFallback)
    );

    let tied = Array2::from_shape_fn((3, 3), |(row, column)| match row == column {
        true => Cf64::new(1.0, 0.0),
        false => Cf64::new(-0.4, 0.0),
    });
    assert_eq!(
        phase_angle_jvp(
            tied.view(),
            zero.view(),
            FixedEstimatorBranch::Evd,
            0,
            1e-10,
        ),
        Err(EstimatorJvpError::EigenvalueTie)
    );
}

#[test]
fn covariance_normalization_rejects_boundary_and_nonfinite_state() {
    let boundary = array![[Cf64::new(1e-6, 0.0)]];
    let zero = array![[Cf64::new(0.0, 0.0)]];
    assert_eq!(
        normalize_numerator_jvp(boundary.view(), zero.view(), 0.0),
        Err(CovarianceReplayError::AmplitudeFloorBoundary)
    );
    let nonfinite = array![[Cf64::new(f64::INFINITY, 0.0)]];
    assert_eq!(
        normalize_numerator_jvp(nonfinite.view(), zero.view(), 1e-10),
        Err(CovarianceReplayError::NonFiniteState)
    );
    assert_eq!(
        normalize_numerator_jvp(boundary.view(), nonfinite.view(), 0.0),
        Err(CovarianceReplayError::NonFiniteState)
    );
}

#[test]
fn compression_jvp_carries_phase_and_amplitude() {
    let samples = array![
        Cf64::from_polar(1.2, 0.7),
        Cf64::from_polar(2.1, 1.1),
        Cf64::from_polar(0.9, -0.2),
    ];
    let phase_angles = array![0.1, 0.4, -0.5];
    let phases = phase_angles.mapv(|angle| Cf64::from_polar(1.0, angle));
    let sample_direction = array![
        Cf64::new(0.2, -0.1),
        Cf64::new(-0.05, 0.17),
        Cf64::new(0.11, 0.08),
    ];
    let phase_direction = array![0.03, -0.08, 0.06];
    let got = compress_pixel_jvp(
        samples.view(),
        phases.view(),
        sample_direction.view(),
        phase_direction.view(),
        1e-10,
    )
    .unwrap();

    let evaluate = |scale: f64| {
        let z = &samples + &sample_direction.mapv(|value| value * scale);
        let theta = Array1::from_shape_fn(phases.len(), |i| {
            phases[i] * Cf64::from_polar(1.0, scale * phase_direction[i])
        });
        let z = Array3::from_shape_vec((3, 1, 1), z.to_vec()).unwrap();
        let theta = Array3::from_shape_vec((3, 1, 1), theta.to_vec()).unwrap();
        dolphin_phaselink::compress(z.view(), theta.view(), 0, None)[(0, 0)]
    };
    assert!((got.value - evaluate(0.0)).norm() < 1e-12);
    let numeric = (evaluate(EPS) - evaluate(-EPS)) / (2.0 * EPS);
    assert!(
        (got.direction - numeric).norm() < JVP_TOL,
        "compression JVP {} != {numeric}",
        got.direction
    );
    assert!(got.mean_amplitude_direction.abs() > 1e-6);
}

#[test]
fn compression_jvp_rejects_zero_projection_and_nodata_branch() {
    let phases = array![Cf64::new(1.0, 0.0), Cf64::new(1.0, 0.0)];
    let zero_complex = array![Cf64::new(0.0, 0.0), Cf64::new(0.0, 0.0)];
    let zero_real = array![0.0, 0.0];
    let zero_projection = array![Cf64::new(1.0, 0.0), Cf64::new(-1.0, 0.0)];
    assert!(matches!(
        compress_pixel_jvp(
            zero_projection.view(),
            phases.view(),
            zero_complex.view(),
            zero_real.view(),
            1e-10,
        ),
        Err(CompressionJvpError::ZeroProjection)
    ));
    let positive_projection = array![Cf64::new(1.0, 0.0), Cf64::new(2.0, 0.0)];
    assert!(matches!(
        compress_pixel_jvp(
            positive_projection.view(),
            phases.view(),
            zero_complex.view(),
            zero_real.view(),
            1e-10,
        ),
        Err(CompressionJvpError::NodataBranch)
    ));
}

fn fused_params() -> FusedParams {
    FusedParams {
        use_evd: true,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        reference_idx: 0,
        compute_crlb: false,
        crlb_reference_idx: 0,
        num_looks: 1.0,
        compute_closure: false,
        compute_average_coherence: false,
        average_coherence_start_idx: 0,
    }
}

#[test]
fn opt_in_fused_receipt_preserves_legacy_output() {
    let stack = stack();
    let half = HalfWindow { y: 1, x: 1 };
    let strides = Strides { y: 1, x: 1 };
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let valid = Array2::from_elem((3, 3), true);
    let legacy = engine
        .link(stack.view(), half, strides, None, fused_params())
        .unwrap();
    let mut replay = engine
        .link_with_source_replay(
            stack.view(),
            half,
            strides,
            None,
            fused_params(),
            valid.view(),
            1e-10,
        )
        .unwrap();
    assert_eq!(replay.estimate.cpx_phase, legacy.cpx_phase);
    assert_eq!(
        replay.estimate.temporal_coherence,
        legacy.temporal_coherence
    );
    assert!(replay
        .phase
        .branch_status
        .iter()
        .all(|status| *status == FixedBranchStatus::Evd));
    assert!(replay
        .phase
        .selected_eigengap
        .iter()
        .all(|gap| *gap > 1e-10));

    let pixel =
        replay_rect_pixel_covariance(stack.view(), (1, 1), half, strides, valid.view()).unwrap();
    let estimate =
        dolphin_phaselink::process_coherence_matrix(pixel.coherence.view(), true, 0.0, 0.0, 0);
    for date in 0..stack.dim().0 {
        let phase = Cf64::from_polar(1.0, estimate.phase[date].arg());
        assert_eq!(phase, replay.estimate.cpx_phase[(date, 1, 1)]);
    }

    let mut output_validity = Array2::from_elem((3, 3), true);
    output_validity[(1, 1)] = false;
    replay
        .phase
        .apply_output_validity(output_validity.view())
        .unwrap();
    assert_eq!(
        replay.phase.branch_status[(1, 1)],
        FixedBranchStatus::Masked
    );
    assert_eq!(replay.phase.branch_status[(0, 0)], FixedBranchStatus::Evd);
}

#[test]
fn source_replay_classifies_fixed_support_before_numeric_erasure() {
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let half = HalfWindow { y: 1, x: 1 };
    let strides = Strides { y: 1, x: 1 };
    let valid = Array2::from_elem((3, 3), true);
    let mut partial = stack();
    partial[(2, 1, 1)] = Cf64::new(f64::NAN, 0.0);
    let replay = engine
        .link_with_source_replay(
            partial.view(),
            half,
            strides,
            None,
            fused_params(),
            valid.view(),
            1e-10,
        )
        .unwrap();
    assert!(replay
        .phase
        .branch_status
        .iter()
        .all(|status| *status == FixedBranchStatus::NonFiniteState));

    let empty = Array2::from_elem((3, 3), false);
    let replay = engine
        .link_with_source_replay(
            stack().view(),
            half,
            strides,
            None,
            fused_params(),
            empty.view(),
            1e-10,
        )
        .unwrap();
    assert!(replay
        .phase
        .branch_status
        .iter()
        .all(|status| *status == FixedBranchStatus::Masked));
}

#[test]
fn source_replay_detects_amplitude_floor_and_zero_stride() {
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let stack = Array3::from_elem((2, 1, 1), Cf64::new(1e-3, 0.0));
    let valid = Array2::from_elem((1, 1), true);
    let replay = engine
        .link_with_source_replay(
            stack.view(),
            HalfWindow { y: 0, x: 0 },
            Strides { y: 1, x: 1 },
            None,
            fused_params(),
            valid.view(),
            0.0,
        )
        .unwrap();
    assert_eq!(
        replay.phase.branch_status[(0, 0)],
        FixedBranchStatus::AmplitudeFloorBoundary
    );

    let masked_stack = Array3::from_shape_fn((2, 1, 3), |(_, _, column)| match column {
        1 => Cf64::new(1e-3, 0.0),
        _ => Cf64::new(1.0, 0.0),
    });
    let masked_validity = array![[false, true, false]];
    let masked_replay = engine
        .link_with_source_replay(
            masked_stack.view(),
            HalfWindow { y: 0, x: 1 },
            Strides { y: 1, x: 1 },
            None,
            fused_params(),
            masked_validity.view(),
            0.0,
        )
        .unwrap();
    assert!(masked_replay
        .phase
        .branch_status
        .iter()
        .all(|status| *status == FixedBranchStatus::AmplitudeFloorBoundary));

    assert!(engine
        .link_with_source_replay(
            stack.view(),
            HalfWindow { y: 0, x: 0 },
            Strides { y: 0, x: 1 },
            None,
            fused_params(),
            valid.view(),
            1e-10,
        )
        .is_err());
}

#[test]
fn compression_replay_records_mixed_validity_without_dropping_valid_pixels() {
    let angles = [0.1, 0.4, -0.2];
    let offsets = [0.6, -0.7, 0.2];
    let slc = Array3::from_shape_fn((3, 1, 3), |(date, _, column)| {
        if column == 2 {
            Cf64::new(0.0, 0.0)
        } else {
            Cf64::from_polar(1.0 + 0.2 * date as f64, angles[date] + offsets[column])
        }
    });
    let phase = Array3::from_shape_fn((3, 1, 3), |(date, _, _)| {
        Cf64::from_polar(1.0, angles[date])
    });
    let validity = array![[true, false, true]];
    let legacy = dolphin_phaselink::compress(slc.view(), phase.view(), 0, None);
    let replay =
        compress_with_replay(slc.view(), phase.view(), 0, None, validity.view(), 1e-10).unwrap();
    assert_eq!(replay.status[(0, 0)], CompressionReplayStatus::Valid);
    assert_eq!(replay.status[(0, 1)], CompressionReplayStatus::Masked);
    assert_eq!(
        replay.status[(0, 2)],
        CompressionReplayStatus::ZeroIncludedAmplitude
    );
    assert_eq!(replay.compressed[(0, 0)], legacy[(0, 0)]);
    assert!(replay.compressed[(0, 1)].re.is_nan());
    assert!(replay.compressed[(0, 1)].im.is_nan());
    assert!(replay.projection[(0, 0)].is_finite());
    assert!(replay.mean_amplitude[(0, 0)].is_finite());
}

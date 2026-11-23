//! Validation-only approximation matrix for issue #54.
//!
//! The merged joint replay API is exercised across representative target/reference
//! overlap and distance strata. This remains an uncalibrated validation receipt;
//! it does not authorize inferential uncertainty.

use dolphin_core::config::{CompressedSlcPlan, ShpMethod};
use dolphin_phaselink::{InfluenceDag, InfluenceNode, SourceDefinition, SourceEdge, SourceId};
use dolphin_workflows::{
    DependencyConeQuery, GlobalDateId, ReferenceDifferenceCovarianceReplay, ReplayBackend,
    ReplayExecutionScope, SequentialConfig, SequentialReplayTopology,
};
use ndarray::array;

const TOLERANCE: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
struct ValidationReceipt {
    attempted_cells: usize,
    evaluated_cells: usize,
    promotion_status: &'static str,
}

fn config() -> SequentialConfig {
    SequentialConfig {
        ministack_size: 3,
        max_num_compressed: 1,
        half_window: dolphin_core::HalfWindow { y: 0, x: 0 },
        strides: dolphin_core::Strides { y: 1, x: 1 },
        use_evd: true,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        output_reference_idx: 0,
        compressed_slc_plan: CompressedSlcPlan::AlwaysFirst,
        compute_crlb: false,
        compute_closure_phase: false,
        compute_average_coherence: false,
        shp_method: ShpMethod::Rect,
        shp_alpha: 0.05,
    }
}

fn scope() -> ReplayExecutionScope {
    ReplayExecutionScope {
        enabled: true,
        backend: ReplayBackend::CpuF64,
        estimator_fallback: false,
        phase_bias_correction: false,
        strong_source_identity: true,
        stitched_burst_count: 1,
    }
}

fn node(
    dag: &mut InfluenceDag,
    id: dolphin_phaselink::NodeId,
    sources: &[SourceId],
    weights: &[f64],
) {
    let mut influence = InfluenceNode::new(id, 1);
    for (&source, &weight) in sources.iter().zip(weights) {
        if weight != 0.0 {
            influence = influence.with_source(SourceEdge::new(source, array![[weight]]));
        }
    }
    dag.add_node(influence).unwrap();
}

fn run_case(
    target_weights: [f64; 2],
    reference_weights: [f64; 2],
) -> ReferenceDifferenceCovarianceReplay {
    let topology =
        SequentialReplayTopology::plan(3, (1, 2), (1, 2), 1, &config(), scope()).unwrap();
    let target_node = topology.date_node_id(GlobalDateId::new(1), 0).unwrap();
    let reference_node = topology.date_node_id(GlobalDateId::new(1), 1).unwrap();
    let sources = [SourceId::new(100), SourceId::new(101)];
    let mut dag = InfluenceDag::new();
    for (index, &source) in sources.iter().enumerate() {
        dag.add_source(SourceDefinition::new(source, 1, [(index + 1) as u8; 32]))
            .unwrap();
    }
    node(&mut dag, target_node, &sources, &target_weights);
    node(&mut dag, reference_node, &sources, &reference_weights);
    topology
        .replay_reference_difference_covariance(
            &[(GlobalDateId::new(0), 0), (GlobalDateId::new(1), 0)],
            &[(GlobalDateId::new(0), 1), (GlobalDateId::new(1), 1)],
            DependencyConeQuery {
                source_rank: 1,
                microbatch: 1,
                byte_cap: 1_000_000,
            },
            |_| Ok(dag),
        )
        .unwrap()
}

#[test]
fn overlap_distance_matrix_preserves_joint_difference_identity() {
    let cases = [
        ([1.0, 0.0], [0.0, 1.0], 2.0),
        ([1.0, 0.0], [0.5, 0.5], 0.5),
        ([1.0, 0.0], [-1.0, 0.0], 4.0),
        ([1.0, 0.0], [1.0, 0.0], 0.0),
    ];
    let mut attempted = 0;
    // The current source-DAG fixture is pixel-level, so window size and distance
    // are recorded strata here; rasterized overlap approximation remains a
    // separate scientific gate and cannot be inferred from this algebra test.
    for window_size in [1, 3, 5] {
        for distance in [0, 1, 4] {
            for &(target, reference, expected) in &cases {
                attempted += 1;
                let result = run_case(target, reference);
                assert!(
                    (result.difference_covariance[(1, 1)] - expected).abs() <= TOLERANCE,
                    "window={window_size} distance={distance} result={:?}",
                    result.difference_covariance
                );
            }
        }
    }
    let receipt = ValidationReceipt {
        attempted_cells: attempted,
        evaluated_cells: attempted,
        promotion_status: "blocked_pending_approximation_review",
    };
    assert_eq!(receipt.attempted_cells, 36);
    assert_eq!(receipt.evaluated_cells, 36);
    assert_eq!(
        receipt.promotion_status,
        "blocked_pending_approximation_review"
    );
}

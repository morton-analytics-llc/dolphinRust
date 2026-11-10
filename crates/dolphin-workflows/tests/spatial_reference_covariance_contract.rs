//! Issue #54 red contract for one joint target/reference source-DAG query.
//!
//! #52 intentionally accepts one output pixel per temporal replay. The #54 API
//! must add a separate reference-specific query that contracts target and
//! reference paths at shared primitive sources. It must not subtract two #52
//! marginal results. Version 1 remains batch-only and single-burst.

use dolphin_core::config::{CompressedSlcPlan, ShpMethod};
use dolphin_phaselink::{InfluenceDag, InfluenceNode, SourceDefinition, SourceEdge, SourceId};
use dolphin_workflows::{
    DependencyConeQuery, GlobalDateId, ReferenceSpecificExecutionMode,
    ReferenceSpecificReplayScope, ReplayBackend, ReplayExecutionScope, SequentialConfig,
    SequentialReplayTopology, SpatialCovarianceStatus,
};
use ndarray::array;
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/spatial_reference_covariance_cases.json");
const TOLERANCE: f64 = 1e-12;

#[derive(Deserialize)]
struct Fixture {
    schema: String,
    cases: Vec<CovarianceCase>,
}

#[derive(Deserialize)]
struct CovarianceCase {
    name: String,
    reference_output: usize,
    target_weights: Vec<f64>,
    reference_weights: Vec<f64>,
    expected_target_variance: f64,
    expected_reference_variance: f64,
    expected_cross_covariance: f64,
    expected_difference_variance: f64,
}

fn fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("committed issue #54 fixture must parse")
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

fn source_dag_scope() -> ReplayExecutionScope {
    ReplayExecutionScope {
        enabled: true,
        backend: ReplayBackend::CpuF64,
        estimator_fallback: false,
        phase_bias_correction: false,
        strong_source_identity: true,
        stitched_burst_count: 1,
    }
}

fn add_weighted_node(
    dag: &mut InfluenceDag,
    node: dolphin_phaselink::NodeId,
    sources: &[SourceId],
    weights: &[f64],
) {
    assert_eq!(sources.len(), weights.len());
    let mut influence = InfluenceNode::new(node, 1);
    for (&source, &weight) in sources.iter().zip(weights) {
        if weight != 0.0 {
            influence = influence.with_source(SourceEdge::new(source, array![[weight]]));
        }
    }
    dag.add_node(influence).unwrap();
}

fn assert_close(actual: f64, expected: f64, case: &str, quantity: &str) {
    assert!(
        (actual - expected).abs() <= TOLERANCE,
        "{case} {quantity}: got {actual:.16e}, expected {expected:.16e}"
    );
}

#[test]
fn joint_source_replay_contracts_independent_positive_negative_and_coincident_cases() {
    let fixture = fixture();
    assert_eq!(
        fixture.schema,
        "dolphinrust-spatial-reference-covariance-contract/1"
    );

    for case in fixture.cases {
        let topology =
            SequentialReplayTopology::plan(3, (1, 2), (1, 2), 1, &config(), source_dag_scope())
                .expect("supported source-DAG topology");
        let target_node = topology.date_node_id(GlobalDateId::new(1), 0).unwrap();
        let reference_node = topology
            .date_node_id(GlobalDateId::new(1), case.reference_output)
            .unwrap();
        let sources = [SourceId::new(100), SourceId::new(101), SourceId::new(102)];
        let mut dag = InfluenceDag::new();
        for (index, &source) in sources.iter().enumerate() {
            dag.add_source(SourceDefinition::new(source, 1, [(index + 1) as u8; 32]))
                .unwrap();
        }
        add_weighted_node(&mut dag, target_node, &sources, &case.target_weights);
        if reference_node != target_node {
            add_weighted_node(&mut dag, reference_node, &sources, &case.reference_weights);
        }

        let result = topology
            .replay_reference_difference_covariance(
                &[(GlobalDateId::new(0), 0), (GlobalDateId::new(1), 0)],
                &[
                    (GlobalDateId::new(0), case.reference_output),
                    (GlobalDateId::new(1), case.reference_output),
                ],
                DependencyConeQuery {
                    source_rank: 1,
                    microbatch: 1,
                    byte_cap: 1_000_000,
                },
                |_| Ok(dag),
            )
            .expect("supported reference-specific replay");

        assert_close(
            result.target_covariance[(1, 1)],
            case.expected_target_variance,
            &case.name,
            "target variance",
        );
        assert_close(
            result.reference_covariance[(1, 1)],
            case.expected_reference_variance,
            &case.name,
            "reference variance",
        );
        assert_close(
            result.target_reference_covariance[(1, 1)],
            case.expected_cross_covariance,
            &case.name,
            "cross covariance",
        );
        assert_close(
            result.difference_covariance[(1, 1)],
            case.expected_difference_variance,
            &case.name,
            "difference variance",
        );
        assert!(
            result
                .difference_covariance
                .row(0)
                .iter()
                .all(|value| *value == 0.0),
            "{} must preserve the exact acquisition-0 gauge",
            case.name,
        );
        assert!(
            result
                .difference_covariance
                .column(0)
                .iter()
                .all(|value| *value == 0.0),
            "{} must preserve the exact acquisition-0 gauge",
            case.name,
        );
    }
}

#[test]
fn version_one_fails_closed_for_nrt_and_stitched_multiburst_scope() {
    assert_eq!(
        ReferenceSpecificReplayScope::new(ReferenceSpecificExecutionMode::Nrt, 1).disposition(),
        SpatialCovarianceStatus::UnsupportedNrtReplay,
    );
    assert_eq!(
        ReferenceSpecificReplayScope::new(ReferenceSpecificExecutionMode::Batch, 2).disposition(),
        SpatialCovarianceStatus::UnsupportedMultiburstReference,
    );
}

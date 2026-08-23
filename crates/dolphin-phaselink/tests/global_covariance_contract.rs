//! Issue #52 analytic contract for the replayable source-influence DAG.
//!
//! The fixture is committed and loaded unconditionally. It covers the exact
//! acquisition-0 gauge, two-ministack temporal propagation, and a strided
//! shared-source graph where two native paths must meet at one primitive root.

use dolphin_phaselink::{
    InfluenceDag, InfluenceNode, NodeId, ParentEdge, SourceDefinition, SourceEdge, SourceId,
    TemporalCoordinate,
};
use ndarray::{array, Array2};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/two_ministack_source_dag.json");
const TOLERANCE: f64 = 1e-12;

#[derive(Deserialize)]
struct Fixture {
    second_block_cholesky: Vec<Vec<f64>>,
    expected_history_covariance: Vec<Vec<f64>>,
    expected_stride_covariance: Vec<Vec<f64>>,
    expected_stride_history: Vec<Vec<f64>>,
}

fn fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("committed issue #52 fixture must parse")
}

fn matrix(rows: &[Vec<f64>]) -> Array2<f64> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    assert!(rows.iter().all(|row| row.len() == ncols));
    Array2::from_shape_vec(
        (nrows, ncols),
        rows.iter().flatten().copied().collect(),
    )
    .expect("rectangular fixture matrix")
}

fn assert_close(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim());
    for ((row, col), &value) in actual.indexed_iter() {
        let want = expected[(row, col)];
        assert!(
            (value - want).abs() <= TOLERANCE,
            "matrix mismatch at ({row},{col}): got {value:.16e}, want {want:.16e}"
        );
    }
}

#[test]
fn two_ministacks_reconstruct_global_covariance_with_exact_gauge() {
    let f = fixture();
    let mut dag = InfluenceDag::new();
    let eta1 = SourceId::new(1);
    let eta2 = SourceId::new(2);
    dag.add_source(SourceDefinition::new(eta1, 1, [1; 32]))
        .unwrap();
    dag.add_source(SourceDefinition::new(eta2, 2, [2; 32]))
        .unwrap();

    let x1 = NodeId::new(10);
    dag.add_node(
        InfluenceNode::new(x1, 1)
            .with_source(SourceEdge::new(eta1, array![[2.0]])),
    )
    .unwrap();
    let compressed = NodeId::new(11);
    dag.add_node(
        InfluenceNode::new(compressed, 1)
            .with_parent(ParentEdge::new(x1, array![[0.5]])),
    )
    .unwrap();
    let x23 = NodeId::new(20);
    dag.add_node(
        InfluenceNode::new(x23, 2)
            .with_parent(ParentEdge::new(compressed, array![[1.0], [1.0]]))
            .with_source(SourceEdge::new(
                eta2,
                matrix(&f.second_block_cholesky),
            )),
    )
    .unwrap();

    let covariance = dag
        .temporal_covariance(&[
            TemporalCoordinate::Gauge,
            TemporalCoordinate::node(x1, 0),
            TemporalCoordinate::node(x23, 0),
            TemporalCoordinate::node(x23, 1),
        ])
        .unwrap();
    assert_close(&covariance, &matrix(&f.expected_history_covariance));
    assert!(covariance.row(0).iter().all(|value| *value == 0.0));
    assert!(covariance.column(0).iter().all(|value| *value == 0.0));
}

#[test]
fn strided_native_paths_contract_at_shared_source_roots() {
    let f = fixture();
    let mut dag = InfluenceDag::new();
    let c0_source = SourceId::new(100);
    let c1_source = SourceId::new(101);
    dag.add_source(SourceDefinition::new(c0_source, 1, [10; 32]))
        .unwrap();
    dag.add_source(SourceDefinition::new(c1_source, 1, [11; 32]))
        .unwrap();

    let native_nodes = [
        (NodeId::new(1000), c0_source),
        (NodeId::new(1001), c0_source),
        (NodeId::new(1002), c1_source),
        (NodeId::new(1003), c1_source),
    ];
    for (node, source) in native_nodes {
        dag.add_node(
            InfluenceNode::new(node, 1)
                .with_source(SourceEdge::new(source, array![[1.0]])),
        )
        .unwrap();
    }

    let downstream = NodeId::new(2000);
    dag.add_node(
        InfluenceNode::new(downstream, 2)
            .with_parent(ParentEdge::new(NodeId::new(1000), array![[1.0 / 3.0], [0.0]]))
            .with_parent(ParentEdge::new(
                NodeId::new(1001),
                array![[1.0 / 3.0], [1.0 / 3.0]],
            ))
            .with_parent(ParentEdge::new(
                NodeId::new(1002),
                array![[1.0 / 3.0], [1.0 / 3.0]],
            ))
            .with_parent(ParentEdge::new(NodeId::new(1003), array![[0.0], [1.0 / 3.0]])),
    )
    .unwrap();

    let stride_covariance = dag
        .temporal_covariance(&[
            TemporalCoordinate::node(downstream, 0),
            TemporalCoordinate::node(downstream, 1),
        ])
        .unwrap();
    assert_close(
        &stride_covariance,
        &matrix(&f.expected_stride_covariance),
    );

    let history = dag
        .temporal_covariance(&[
            TemporalCoordinate::Gauge,
            TemporalCoordinate::node(NodeId::new(1000), 0),
            TemporalCoordinate::node(downstream, 0),
        ])
        .unwrap();
    assert_close(&history, &matrix(&f.expected_stride_history));
}

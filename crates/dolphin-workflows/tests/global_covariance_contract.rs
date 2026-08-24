//! Issue #52 unconditional workflow contract for `sequential_source_dag_v1`.

use std::collections::BTreeMap;
use std::sync::Mutex;

use dolphin_core::config::{CompressedSlcPlan, ComputeBackend, ShpMethod};
use dolphin_core::Cf64;
use dolphin_io::{
    read_covariance_operator_block_with_receipt, CovarianceOperatorBlock, CovarianceOperatorGrid,
    CovarianceOperatorMetadata, CovarianceOperatorStatus, CovarianceOperatorWriter,
    CovariancePhaseComponentKind, CovarianceReplayStatus, DownstreamInferenceStatus,
    SourceReplayIdentity, StitchedCovarianceStatus,
};
use dolphin_phaselink::{
    phase_angle_jvp_workspace_bytes, ComputeEngine, FixedEstimatorBranch, InfluenceDag,
    InfluenceNode, NodeId, ParentEdge, ProperComplexFactor, SourceDefinition, SourceEdge, SourceId,
    TemporalCoordinate,
};
use dolphin_workflows::{
    admit_covariance_artifact_disk_with_identity_index, finalize_covariance_artifact,
    run_sequential, run_sequential_with_covariance_capture, sequential_replay_kernel_digest,
    sequential_source_model_identity_digest, CovarianceArtifactReplayProvider,
    CovarianceArtifactTransaction, DependencyConeQuery, GlobalBlockId, GlobalDateId, ReplayBackend,
    ReplayExecutionScope, ReplayIdNamespace, ReplayStatus, ResolvedCompressionReplay,
    ResolvedPhaseReplay, ResolvedPrimitiveSource, SequentialConfig,
    SequentialCovarianceCaptureRequest, SequentialPrimitiveSourceResolver, SequentialReplayBlock,
    SequentialReplayBuildIdentity, SequentialReplayError, SequentialReplayTopology,
    SequentialSourceProviderIdentity, SequentialSourceReplayProvider, COVARIANCE_OPERATOR_FILENAME,
};
use ndarray::{array, Array1, Array2, Array3};
use sha2::{Digest, Sha256};

const SOURCE_PROVIDER: &str = "captured-provider";
const SOURCE_PROVIDER_VERSION: &str = "1";
const SOURCE_MODEL: &str = "proper-complex";
const SOURCE_MODEL_VERSION: &str = "1";
static HDF5_LOCK: Mutex<()> = Mutex::new(());

fn config() -> SequentialConfig {
    SequentialConfig {
        ministack_size: 3,
        max_num_compressed: 2,
        half_window: dolphin_core::HalfWindow { y: 1, x: 1 },
        strides: dolphin_core::Strides { y: 2, x: 2 },
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

fn source_model_identity_digest() -> [u8; 32] {
    sequential_source_model_identity_digest(
        SOURCE_PROVIDER,
        SOURCE_PROVIDER_VERSION,
        SOURCE_MODEL,
        SOURCE_MODEL_VERSION,
    )
}

fn bind_test_factor_receipts(blocks: &mut [CovarianceOperatorBlock]) {
    for block in blocks {
        block.source_factor_digests.clear();
        for &source_id in &block.source_ids {
            let factor = ProperComplexFactor::new(
                SourceId::new(source_id),
                block
                    .source_date_indices
                    .iter()
                    .copied()
                    .map(u64::from)
                    .collect(),
                [9; 32],
                Array2::from_diag_elem(block.source_date_indices.len(), Cf64::new(0.02, 0.0)),
            )
            .unwrap();
            block
                .source_factor_digests
                .extend_from_slice(&factor.numeric_receipt_digest());
        }
    }
}

struct CapturedProvider {
    identity: SequentialSourceProviderIdentity,
    blocks: BTreeMap<GlobalBlockId, CovarianceOperatorBlock>,
    stack: Array3<Cf64>,
    source_reads: usize,
    fail_source_model: bool,
    dishonest_samples: bool,
}

impl SequentialSourceReplayProvider for CapturedProvider {
    fn identity(&self) -> &SequentialSourceProviderIdentity {
        &self.identity
    }

    fn maximum_resident_bytes(&self) -> u64 {
        256
    }

    fn resolve_source(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError> {
        self.source_reads += 1;
        if self.fail_source_model {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceModelUnavailable,
                "test source model is unavailable",
            ));
        }
        let stored = self.blocks.get(&block.id).unwrap();
        let columns = self.stack.dim().2;
        let row = native_index / columns;
        let column = native_index % columns;
        let mut samples = Array1::from_iter(
            stored
                .source_date_indices
                .iter()
                .map(|&date| self.stack[(date as usize, row, column)]),
        );
        let mut content = Sha256::new();
        for sample in &samples {
            content.update(sample.re.to_le_bytes());
            content.update(sample.im.to_le_bytes());
        }
        let content_digest = content.finalize().into();
        if self.dishonest_samples {
            samples[0].re += 1.0;
        }
        let id = SourceId::new(stored.source_ids[native_index]);
        let factor = ProperComplexFactor::new(
            id,
            stored
                .source_date_indices
                .iter()
                .map(|&date| u64::from(date))
                .collect(),
            self.identity.source_model_hash,
            Array2::from_diag_elem(samples.len(), Cf64::new(0.02, 0.0)),
        )
        .unwrap();
        Ok(ResolvedPrimitiveSource {
            id,
            samples,
            factor,
            content_digest,
        })
    }

    fn resolve_phase(
        &mut self,
        block: &SequentialReplayBlock,
        output_index: usize,
    ) -> Result<ResolvedPhaseReplay, SequentialReplayError> {
        let stored = self.blocks.get(&block.id).unwrap();
        let width = stored.phase_components.len();
        Ok(ResolvedPhaseReplay {
            id: NodeId::new(stored.phase_node_ids[output_index]),
            linked_phase: Array1::from_iter(
                stored.phase_angles[output_index * width..(output_index + 1) * width]
                    .iter()
                    .map(|&angle| Cf64::from_polar(1.0, angle)),
            ),
            selected_eigenvalue: stored.selected_eigenvalue[output_index],
            selected_eigengap: stored.eigen_gap[output_index],
            status: stored.status[output_index],
            estimator_branch: stored.estimator_branch,
            branch_tolerance: stored.branch_tolerance,
        })
    }

    fn resolve_compression(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedCompressionReplay, SequentialReplayError> {
        let stored = self.blocks.get(&block.id).unwrap();
        Ok(ResolvedCompressionReplay {
            id: NodeId::new(stored.compressed_node_ids[native_index]),
            value: stored.compressed_raster[native_index],
            projection: stored.projection_accumulator[native_index],
            mean_amplitude: stored.mean_amplitude[native_index],
            status: stored.compressed_status[native_index],
        })
    }
}

impl SequentialPrimitiveSourceResolver for CapturedProvider {
    fn identity(&self) -> &SequentialSourceProviderIdentity {
        &self.identity
    }

    fn maximum_resident_bytes(&self) -> u64 {
        256
    }

    fn resolve_source(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError> {
        SequentialSourceReplayProvider::resolve_source(self, block, native_index)
    }
}

struct ChangedFactorResolver(CapturedProvider);

impl SequentialPrimitiveSourceResolver for ChangedFactorResolver {
    fn identity(&self) -> &SequentialSourceProviderIdentity {
        &self.0.identity
    }

    fn maximum_resident_bytes(&self) -> u64 {
        256
    }

    fn resolve_source(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError> {
        let mut source =
            SequentialSourceReplayProvider::resolve_source(&mut self.0, block, native_index)?;
        source.factor = ProperComplexFactor::new(
            source.id,
            source.factor.component_ids().to_vec(),
            *source.factor.model_hash(),
            Array2::from_diag_elem(source.samples.len(), Cf64::new(0.04, 0.0)),
        )
        .unwrap();
        Ok(source)
    }
}

#[test]
fn topology_has_global_block_date_and_source_ids_with_cap_eviction_ancestry() {
    let topology = SequentialReplayTopology::plan(12, (4, 4), (2, 2), 9, &config(), scope())
        .expect("supported topology");
    assert_eq!(topology.status(), ReplayStatus::Valid);
    assert_eq!(topology.blocks().len(), 4);
    assert_eq!(topology.blocks()[3].id, GlobalBlockId::new(3));
    assert_eq!(
        topology.blocks()[3].carried_parent_ids,
        vec![GlobalBlockId::new(1), GlobalBlockId::new(2)],
        "the carry cap evicts block 0 as a direct parent",
    );
    assert_eq!(
        topology.reverse_frontier(&[GlobalBlockId::new(3)]).unwrap(),
        vec![
            GlobalBlockId::new(3),
            GlobalBlockId::new(2),
            GlobalBlockId::new(1),
            GlobalBlockId::new(0),
        ],
        "evicted direct parents remain in transitive ancestry",
    );
    assert_eq!(topology.blocks()[2].real_date_start, GlobalDateId::new(6));
    assert_ne!(
        topology.source_id(GlobalBlockId::new(1), 0).unwrap(),
        topology.source_id(GlobalBlockId::new(2), 0).unwrap(),
    );
}

#[test]
fn version_one_rejects_multi_output_covariance_queries() {
    let topology = SequentialReplayTopology::plan(8, (4, 4), (2, 2), 9, &config(), scope())
        .expect("supported topology");
    let error = topology
        .estimate_dependency_cone(
            &[(GlobalDateId::new(1), 0), (GlobalDateId::new(2), 1)],
            6,
            2,
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("one output pixel"), "{error}");
}

#[test]
fn replay_uses_shared_graph_algebra_and_inserts_an_exact_acquisition_zero_gauge() {
    let topology = SequentialReplayTopology::plan(6, (3, 3), (1, 1), 9, &config(), scope())
        .expect("supported topology");
    let source0 = topology.source_id(GlobalBlockId::new(0), 0).unwrap();
    let source1 = topology.source_id(GlobalBlockId::new(1), 0).unwrap();
    let date1 = topology.date_node_id(GlobalDateId::new(1), 0).unwrap();
    let compressed0 = topology
        .compressed_node_id(GlobalBlockId::new(0), 0)
        .unwrap();
    let date3 = topology.date_node_id(GlobalDateId::new(3), 0).unwrap();

    let mut dag = InfluenceDag::new();
    dag.add_source(SourceDefinition::new(source0, 1, [1; 32]))
        .unwrap();
    dag.add_source(SourceDefinition::new(source1, 1, [2; 32]))
        .unwrap();
    dag.add_node(InfluenceNode::new(date1, 1).with_source(SourceEdge::new(source0, array![[2.0]])))
        .unwrap();
    dag.add_node(
        InfluenceNode::new(compressed0, 1).with_parent(ParentEdge::new(date1, array![[0.5]])),
    )
    .unwrap();
    dag.add_node(
        InfluenceNode::new(date3, 1)
            .with_parent(ParentEdge::new(compressed0, array![[1.0]]))
            .with_source(SourceEdge::new(source1, array![[3.0]])),
    )
    .unwrap();

    assert_eq!(
        topology
            .temporal_coordinate(GlobalDateId::new(0), 0)
            .unwrap(),
        TemporalCoordinate::Gauge,
    );
    let result = topology
        .replay_temporal_covariance(
            &[
                (GlobalDateId::new(0), 0),
                (GlobalDateId::new(1), 0),
                (GlobalDateId::new(3), 0),
            ],
            DependencyConeQuery {
                source_rank: 1,
                microbatch: 1,
                byte_cap: 1_000_000,
            },
            |_| Ok(dag),
        )
        .unwrap();
    assert_eq!(
        result.covariance,
        array![[0.0, 0.0, 0.0], [0.0, 4.0, 2.0], [0.0, 2.0, 10.0]]
    );
    assert!(result.covariance.row(0).iter().all(|value| *value == 0.0));
    assert!(result
        .covariance
        .column(0)
        .iter()
        .all(|value| *value == 0.0));
}

#[test]
fn dependency_cone_preflight_rejects_one_byte_below_the_exact_bound() {
    let topology = SequentialReplayTopology::plan(6, (4, 4), (2, 2), 9, &config(), scope())
        .expect("supported topology");
    let selection = [
        (GlobalDateId::new(0), 0),
        (GlobalDateId::new(1), 0),
        (GlobalDateId::new(3), 0),
    ];
    let estimate = topology.estimate_dependency_cone(&selection, 2, 1).unwrap();
    assert_eq!(
        (
            estimate.frontier_bytes,
            estimate.source_window_bytes,
            estimate.operator_bytes,
            estimate.support_bytes,
            estimate.covariance_bytes,
            estimate.provider_bytes,
        ),
        (1_944, 47_384, 47_960, 6, 72, 0)
    );
    let estimator_workspace =
        phase_angle_jvp_workspace_bytes(4, FixedEstimatorBranch::Evd).unwrap();
    assert_eq!(estimate.baseline_bytes, 2_280 + estimator_workspace);
    assert_eq!(
        estimate.total_bytes,
        estimate.frontier_bytes
            + estimate.source_window_bytes
            + estimate.operator_bytes
            + estimate.baseline_bytes
            + estimate.support_bytes
            + estimate.covariance_bytes
            + estimate.provider_bytes
    );

    let error = topology
        .preflight_dependency_cone(&selection, 2, 1, estimate.total_bytes - 1)
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::DependencyConeExceedsBudget);
    assert_eq!(error.estimate().unwrap(), &estimate);
    topology
        .preflight_dependency_cone(&selection, 2, 1, estimate.total_bytes)
        .expect("the exact bound is admitted");
}

#[test]
fn evd_and_emi_max_support_workspaces_are_bound_before_replay() {
    let selection = [(GlobalDateId::new(1), 0)];
    let evd_topology = SequentialReplayTopology::plan(6, (4, 4), (2, 2), 9, &config(), scope())
        .expect("supported EVD topology");
    let evd = evd_topology
        .estimate_dependency_cone(&selection, 6, 1)
        .unwrap();

    let mut emi_config = config();
    emi_config.use_evd = false;
    emi_config.beta = 0.1;
    let emi_topology = SequentialReplayTopology::plan(6, (4, 4), (2, 2), 9, &emi_config, scope())
        .expect("supported EMI topology");
    let emi = emi_topology
        .estimate_dependency_cone(&selection, 6, 1)
        .unwrap();

    assert!(emi.baseline_bytes > evd.baseline_bytes);
    for (topology, estimate) in [(&evd_topology, evd), (&emi_topology, emi)] {
        let error = topology
            .preflight_dependency_cone(&selection, 6, 1, estimate.total_bytes - 1)
            .unwrap_err();
        assert_eq!(error.status(), ReplayStatus::DependencyConeExceedsBudget);
        assert_eq!(
            topology
                .preflight_dependency_cone(&selection, 6, 1, estimate.total_bytes)
                .unwrap(),
            estimate
        );
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn large_support_replay_enforces_exact_cap_and_source_payload_receipt() {
    let stack = Array3::from_shape_fn((6, 15, 29), |(date, row, col)| {
        let amplitude = 1.0 + 0.03 * date as f64 + 0.001 * (row + col) as f64;
        let phase = 0.2 + 0.09 * date as f64 + 0.003 * row as f64 - 0.002 * col as f64;
        Cf64::from_polar(amplitude, phase)
    });
    let mut cfg = config();
    cfg.half_window = dolphin_core::HalfWindow { y: 7, x: 14 };
    cfg.strides = dolphin_core::Strides { y: 15, x: 29 };
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "large-support".to_owned(),
        source_manifest_digest: [3; 32],
        source_model_version_digest: source_model_identity_digest(),
        native_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 15,
            cols: 29,
            stride_y: 1,
            stride_x: 1,
        },
        output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: 15,
            stride_x: 29,
        },
        owned_output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: 15,
            stride_x: 29,
        },
        branch_tolerance: 1e-10,
    };
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let mut captured = Vec::new();
    run_sequential_with_covariance_capture(stack.view(), &cfg, &engine, &request, |block| {
        captured.push(block);
        Ok(())
    })
    .unwrap();
    bind_test_factor_receipts(&mut captured);
    let topology = SequentialReplayTopology::plan_identified(
        6,
        (15, 29),
        (1, 1),
        435,
        Array2::from_elem((15, 29), true).view(),
        &cfg,
        scope(),
        ReplayIdNamespace {
            burst_id: request.burst_id.clone(),
            source_manifest_digest: request.source_manifest_digest,
            source_model_version_digest: request.source_model_version_digest,
            native_origin: (0, 0),
            output_origin: (0, 0),
            owned_output_origin: (0, 0),
            owned_output_shape: (1, 1),
        },
    )
    .unwrap();
    let identity = SequentialSourceProviderIdentity {
        source_manifest_digest: request.source_manifest_digest,
        provider: SOURCE_PROVIDER.to_owned(),
        provider_version: SOURCE_PROVIDER_VERSION.to_owned(),
        model: SOURCE_MODEL.to_owned(),
        model_version: SOURCE_MODEL_VERSION.to_owned(),
        source_model_version_digest: request.source_model_version_digest,
        source_model_hash: [9; 32],
    };
    let blocks = captured
        .into_iter()
        .map(|block| (GlobalBlockId::new(block.block_id), block))
        .collect();
    let mut provider = CapturedProvider {
        identity,
        blocks,
        stack,
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let selection = [
        (GlobalDateId::new(0), 0),
        (GlobalDateId::new(1), 0),
        (GlobalDateId::new(3), 0),
    ];
    let estimate = topology.estimate_dependency_cone(&selection, 6, 1).unwrap();
    let provider_bytes = SequentialSourceReplayProvider::maximum_resident_bytes(&provider);
    let error = topology
        .replay_temporal_covariance_from_provider(
            &selection,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: estimate.total_bytes + provider_bytes - 1,
            },
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::DependencyConeExceedsBudget);
    assert_eq!(
        provider.source_reads, 0,
        "cap rejection precedes source I/O"
    );

    let replay = topology
        .replay_temporal_covariance_from_provider(
            &selection,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: estimate.total_bytes + provider_bytes,
            },
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap();
    let expected_source_payload = 435 * (3 * 16 + 3 * 3 * 16 + 3 * 8 + 32);
    assert_eq!(replay.source_cache_peak_bytes, expected_source_payload);
    assert!(replay.source_cache_peak_bytes <= replay.dependency_cone.source_window_bytes);
    assert!(replay.covariance.iter().all(|value| value.is_finite()));
}

#[test]
fn fixed_local_query_bound_does_not_scale_with_frame_area_before_ancestry_saturates() {
    let small = SequentialReplayTopology::plan(6, (32, 32), (16, 16), 9, &config(), scope())
        .expect("small supported topology");
    let large = SequentialReplayTopology::plan(6, (64, 64), (32, 32), 9, &config(), scope())
        .expect("large supported topology");
    let small_output = 4 * 16 + 4;
    let large_output = 4 * 32 + 4;
    let small_estimate = small
        .estimate_dependency_cone(&[(GlobalDateId::new(3), small_output)], 2, 1)
        .unwrap();
    let large_estimate = large
        .estimate_dependency_cone(&[(GlobalDateId::new(3), large_output)], 2, 1)
        .unwrap();
    assert_eq!(small_estimate.total_bytes, large_estimate.total_bytes);
    assert!(small_estimate.total_bytes < 160 * 1024);
}

#[test]
fn production_sequential_path_streams_replay_blocks_without_changing_legacy_output() {
    let stack = Array3::from_shape_fn((6, 4, 4), |(date, row, col)| {
        let amplitude = 1.0 + 0.07 * date as f64 + 0.01 * (row + col) as f64;
        let phase = 0.11 * date as f64 + 0.017 * row as f64 - 0.013 * col as f64;
        Cf64::from_polar(amplitude, phase)
    });
    let cfg = config();
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let legacy = run_sequential(stack.view(), &cfg, &engine).unwrap();
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "burst-a".to_owned(),
        source_manifest_digest: [3; 32],
        source_model_version_digest: source_model_identity_digest(),
        native_grid: CovarianceOperatorGrid {
            row_start: 10,
            col_start: 20,
            rows: 4,
            cols: 4,
            stride_y: 1,
            stride_x: 1,
        },
        output_grid: CovarianceOperatorGrid {
            row_start: 5,
            col_start: 10,
            rows: 2,
            cols: 2,
            stride_y: 2,
            stride_x: 2,
        },
        owned_output_grid: CovarianceOperatorGrid {
            row_start: 5,
            col_start: 10,
            rows: 2,
            cols: 2,
            stride_y: 2,
            stride_x: 2,
        },
        branch_tolerance: 1e-10,
    };
    let mut blocks = Vec::new();
    let captured =
        run_sequential_with_covariance_capture(stack.view(), &cfg, &engine, &request, |block| {
            blocks.push(block);
            Ok(())
        })
        .unwrap();
    assert_eq!(captured.cpx_phase, legacy.cpx_phase);
    assert!(captured
        .compressed_slcs
        .iter()
        .flatten()
        .zip(legacy.compressed_slcs.iter().flatten())
        .all(|(left, right)| left.re.to_bits() == right.re.to_bits()
            && left.im.to_bits() == right.im.to_bits()));
    assert_eq!(blocks.len(), 2);
    assert!(blocks.iter().all(|block| block
        .phase_angles
        .chunks_exact(block.phase_components.len())
        .all(|phases| phases[0] == 0.0)));
    assert_eq!(blocks[0].generation, 0);
    assert!(blocks[0].carry_parent_ids.is_empty());
    assert_eq!(blocks[1].generation, 1);
    assert_eq!(blocks[1].carry_parent_ids, vec![blocks[0].block_id]);
    assert!(blocks[0].block_id < blocks[1].block_id);
    assert_eq!(blocks[1].source_date_indices, vec![3, 4, 5]);
    assert_eq!(blocks[1].phase_components.len(), 4);
    assert_eq!(
        blocks[1].phase_components[0].kind,
        CovariancePhaseComponentKind::CompressedParent
    );
    assert_eq!(blocks[1].support_bits_per_output, 9);
    assert_eq!(blocks[1].support_bits.len(), 8);
    assert_eq!(blocks[1].owned_output_grid, blocks[1].output_grid);
}

#[test]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
fn production_replay_preflights_streams_jvps_and_bounds_two_parent_block_reads() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let stack = Array3::from_shape_fn((8, 4, 4), |(date, row, col)| {
        let amplitude = 1.0 + 0.07 * date as f64 + 0.01 * (row + col) as f64;
        let phase = 0.4 + 0.11 * date as f64 + 0.017 * row as f64 - 0.013 * col as f64;
        Cf64::from_polar(amplitude, phase)
    });
    let cfg = config();
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "burst-a".to_owned(),
        source_manifest_digest: [3; 32],
        source_model_version_digest: source_model_identity_digest(),
        native_grid: CovarianceOperatorGrid {
            row_start: 10,
            col_start: 20,
            rows: 4,
            cols: 4,
            stride_y: 1,
            stride_x: 1,
        },
        output_grid: CovarianceOperatorGrid {
            row_start: 5,
            col_start: 10,
            rows: 2,
            cols: 2,
            stride_y: 2,
            stride_x: 2,
        },
        owned_output_grid: CovarianceOperatorGrid {
            row_start: 5,
            col_start: 10,
            rows: 2,
            cols: 2,
            stride_y: 2,
            stride_x: 2,
        },
        branch_tolerance: 1e-10,
    };
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let mut captured = Vec::new();
    run_sequential_with_covariance_capture(stack.view(), &cfg, &engine, &request, |block| {
        captured.push(block);
        Ok(())
    })
    .unwrap();
    bind_test_factor_receipts(&mut captured);
    let artifact_blocks = captured.clone();
    let validity = Array2::from_elem((4, 4), true);
    let topology = SequentialReplayTopology::plan_identified(
        8,
        (4, 4),
        (2, 2),
        9,
        validity.view(),
        &cfg,
        scope(),
        ReplayIdNamespace {
            burst_id: request.burst_id.clone(),
            source_manifest_digest: request.source_manifest_digest,
            source_model_version_digest: request.source_model_version_digest,
            native_origin: (10, 20),
            output_origin: (5, 10),
            owned_output_origin: (5, 10),
            owned_output_shape: (2, 2),
        },
    )
    .unwrap();
    let identity = SequentialSourceProviderIdentity {
        source_manifest_digest: request.source_manifest_digest,
        provider: SOURCE_PROVIDER.to_owned(),
        provider_version: SOURCE_PROVIDER_VERSION.to_owned(),
        model: SOURCE_MODEL.to_owned(),
        model_version: SOURCE_MODEL_VERSION.to_owned(),
        source_model_version_digest: request.source_model_version_digest,
        source_model_hash: [9; 32],
    };
    let blocks = captured
        .into_iter()
        .map(|block| (GlobalBlockId::new(block.block_id), block))
        .collect();
    let mut provider = CapturedProvider {
        identity,
        blocks,
        stack,
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let selection = [
        (GlobalDateId::new(0), 0),
        (GlobalDateId::new(1), 0),
        (GlobalDateId::new(3), 0),
    ];
    let internal = topology.estimate_dependency_cone(&selection, 6, 1).unwrap();
    let error = topology
        .replay_temporal_covariance_from_provider(
            &selection,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: internal.total_bytes
                    + SequentialSourceReplayProvider::maximum_resident_bytes(&provider)
                    - 1,
            },
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::DependencyConeExceedsBudget);
    assert_eq!(provider.source_reads, 0, "preflight precedes source I/O");

    let result = topology
        .replay_temporal_covariance_from_provider(
            &selection,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: internal.total_bytes
                    + SequentialSourceReplayProvider::maximum_resident_bytes(&provider),
            },
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap();
    assert_eq!(result.dependency_cone.provider_bytes, 256);
    assert!(result.covariance.iter().all(|value| value.is_finite()));
    assert!(result.covariance.row(0).iter().all(|value| *value == 0.0));
    assert!(result
        .covariance
        .column(0)
        .iter()
        .all(|value| *value == 0.0));
    assert!(result.covariance[(1, 1)] > 0.0);

    provider.dishonest_samples = true;
    let error = topology
        .replay_temporal_covariance_from_provider(
            &selection,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: u64::MAX,
            },
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::SourceIdentityMismatch);
    provider.dishonest_samples = false;

    provider.fail_source_model = true;
    let source_reads_before_gauge = provider.source_reads;
    let gauge = topology
        .replay_temporal_covariance_from_provider(
            &[(GlobalDateId::new(0), 0)],
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: u64::MAX,
            },
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap();
    assert_eq!(gauge.covariance, Array2::<f64>::zeros((1, 1)));
    assert_eq!(gauge.dependency_cone.provider_bytes, 0);
    assert_eq!(provider.source_reads, source_reads_before_gauge);

    let error = topology
        .replay_temporal_covariance_from_provider(
            &selection,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: u64::MAX,
            },
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::SourceModelUnavailable);

    let encode = |digest: [u8; 32]| {
        format!(
            "sha256:{}",
            digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        )
    };
    let build_identity = SequentialReplayBuildIdentity {
        normalized_config_digest: topology.normalized_config_digest(),
        kernel_digest: sequential_replay_kernel_digest(),
        branch_tolerance: request.branch_tolerance,
    };
    let metadata = CovarianceOperatorMetadata {
        normalized_config_digest: encode(build_identity.normalized_config_digest),
        kernel_digest: encode(build_identity.kernel_digest),
        source: SourceReplayIdentity {
            manifest_digest: Some(encode(request.source_manifest_digest)),
            provider: Some(SOURCE_PROVIDER.to_owned()),
            provider_version: Some(SOURCE_PROVIDER_VERSION.to_owned()),
            model: Some(SOURCE_MODEL.to_owned()),
            model_version: Some(SOURCE_MODEL_VERSION.to_owned()),
            model_version_digest: Some(encode(request.source_model_version_digest)),
            model_receipt_digest: Some(encode([9; 32])),
        },
        replay_status: CovarianceReplayStatus::Replayable,
        stitched_status: StitchedCovarianceStatus::NotStitched,
        downstream_inference_status: DownstreamInferenceStatus::BlockedPendingIssue54And53,
        ..CovarianceOperatorMetadata::default()
    };
    let directory = std::env::temp_dir().join(format!(
        "dolphin-workflow-capped-replay-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = CovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join("phase_covariance_operator.h5.scratch");
    let plan = topology
        .covariance_operator_plan(&request.burst_id)
        .unwrap();
    let missing_factor_scratch = directory.join("missing-factor-receipt.h5");
    let mut missing_factor = artifact_blocks[0].clone();
    missing_factor.source_factor_digests.fill(0);
    let mut missing_factor_writer =
        CovarianceOperatorWriter::create(&missing_factor_scratch, &metadata, &plan).unwrap();
    let error = missing_factor_writer
        .write_block(&missing_factor)
        .unwrap_err()
        .to_string();
    assert!(error.contains("numeric factor receipt"), "{error}");
    drop(missing_factor_writer);
    std::fs::remove_file(missing_factor_scratch).unwrap();
    let mut writer = CovarianceOperatorWriter::create(&scratch, &metadata, &plan).unwrap();
    for block in &artifact_blocks {
        writer.write_block(block).unwrap();
    }
    let write_receipt = writer.finish().unwrap();
    drop(transaction);
    let uncommitted_hdf5 = directory.join(COVARIANCE_OPERATOR_FILENAME);
    std::fs::copy(&scratch, &uncommitted_hdf5).unwrap();
    let uncommitted_raw = CapturedProvider {
        identity: provider.identity.clone(),
        blocks: provider.blocks.clone(),
        stack: provider.stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let error = CovarianceArtifactReplayProvider::open(
        &directory,
        1_000_000,
        &topology,
        build_identity,
        uncommitted_raw,
    )
    .err()
    .expect("an HDF5 file without its commit marker must be rejected");
    assert_eq!(error.status(), ReplayStatus::SourceUnavailable);
    std::fs::remove_file(&uncommitted_hdf5).unwrap();
    let disk = admit_covariance_artifact_disk_with_identity_index(
        10 * 1024 * 1024,
        write_receipt.peak_identity_index_disk_bytes,
        u64::MAX,
    )
    .unwrap();
    let transaction = CovarianceArtifactTransaction::acquire(&directory).unwrap();
    finalize_covariance_artifact(&transaction, &scratch, &metadata, disk, &write_receipt).unwrap();
    drop(transaction);
    let root_payload = read_covariance_operator_block_with_receipt(
        directory.join(COVARIANCE_OPERATOR_FILENAME),
        topology.blocks()[0].id.get(),
        u64::MAX,
    )
    .unwrap()
    .logical_payload_bytes;

    let descriptor_directory = std::env::temp_dir().join(format!(
        "dolphin-workflow-source-model-unavailable-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&descriptor_directory);
    std::fs::create_dir_all(&descriptor_directory).unwrap();
    let descriptor_transaction =
        CovarianceArtifactTransaction::acquire(&descriptor_directory).unwrap();
    let descriptor_scratch = descriptor_directory.join("phase_covariance_operator.h5.scratch");
    let mut descriptor_metadata = metadata.clone();
    descriptor_metadata.replay_status = CovarianceReplayStatus::SourceModelUnavailable;
    descriptor_metadata.source.model_receipt_digest = None;
    let mut descriptor_writer =
        CovarianceOperatorWriter::create(&descriptor_scratch, &descriptor_metadata, &plan).unwrap();
    for block in &artifact_blocks {
        descriptor_writer.write_block(block).unwrap();
    }
    let descriptor_write = descriptor_writer.finish().unwrap();
    let descriptor_disk = admit_covariance_artifact_disk_with_identity_index(
        10 * 1024 * 1024,
        descriptor_write.peak_identity_index_disk_bytes,
        u64::MAX,
    )
    .unwrap();
    finalize_covariance_artifact(
        &descriptor_transaction,
        &descriptor_scratch,
        &descriptor_metadata,
        descriptor_disk,
        &descriptor_write,
    )
    .unwrap();
    drop(descriptor_transaction);
    let descriptor_raw = CapturedProvider {
        identity: provider.identity.clone(),
        blocks: provider.blocks.clone(),
        stack: provider.stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let error = CovarianceArtifactReplayProvider::open(
        &descriptor_directory,
        1_000_000,
        &topology,
        build_identity,
        descriptor_raw,
    )
    .err()
    .expect("a descriptor-only CLI artifact must preserve its source-model status");
    assert_eq!(error.status(), ReplayStatus::SourceModelUnavailable);
    std::fs::remove_dir_all(descriptor_directory).unwrap();

    let mut stale_kernel = build_identity;
    stale_kernel.kernel_digest[0] ^= 0xff;
    let stale_raw = CapturedProvider {
        identity: provider.identity.clone(),
        blocks: provider.blocks.clone(),
        stack: provider.stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let error = CovarianceArtifactReplayProvider::open(
        &directory,
        1_000_000,
        &topology,
        stale_kernel,
        stale_raw,
    )
    .err()
    .expect("a stale kernel digest must fail admission");
    assert_eq!(error.status(), ReplayStatus::ReplayStateMismatch);

    let mut stale_cfg = cfg;
    stale_cfg.beta = 0.25;
    let stale_topology = SequentialReplayTopology::plan_identified(
        8,
        (4, 4),
        (2, 2),
        9,
        validity.view(),
        &stale_cfg,
        scope(),
        ReplayIdNamespace {
            burst_id: request.burst_id.clone(),
            source_manifest_digest: request.source_manifest_digest,
            source_model_version_digest: request.source_model_version_digest,
            native_origin: (10, 20),
            output_origin: (5, 10),
            owned_output_origin: (5, 10),
            owned_output_shape: (2, 2),
        },
    )
    .unwrap();
    let stale_build = SequentialReplayBuildIdentity {
        normalized_config_digest: stale_topology.normalized_config_digest(),
        ..build_identity
    };
    let stale_raw = CapturedProvider {
        identity: provider.identity.clone(),
        blocks: provider.blocks.clone(),
        stack: provider.stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let error = CovarianceArtifactReplayProvider::open(
        &directory,
        1_000_000,
        &stale_topology,
        stale_build,
        stale_raw,
    )
    .err()
    .expect("a stale normalized configuration must fail admission");
    assert_eq!(error.status(), ReplayStatus::ReplayStateMismatch);

    let mut stale_provider_identity = provider.identity.clone();
    stale_provider_identity.provider_version = "2".to_owned();
    let stale_raw = CapturedProvider {
        identity: stale_provider_identity,
        blocks: provider.blocks.clone(),
        stack: provider.stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let error = CovarianceArtifactReplayProvider::open(
        &directory,
        1_000_000,
        &topology,
        build_identity,
        stale_raw,
    )
    .err()
    .expect("a stale provider version must fail admission");
    assert_eq!(error.status(), ReplayStatus::SourceIdentityMismatch);

    let mut changed_stack = provider.stack.clone();
    changed_stack[(0, 0, 0)] += Cf64::new(1e-3, 0.0);
    let changed_raw = CapturedProvider {
        identity: provider.identity.clone(),
        blocks: provider.blocks.clone(),
        stack: changed_stack,
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let mut changed_artifact = CovarianceArtifactReplayProvider::open(
        &directory,
        1_000_000,
        &topology,
        build_identity,
        changed_raw,
    )
    .unwrap();
    let changed_selection = [(GlobalDateId::new(1), 0)];
    let changed_estimate = topology
        .estimate_dependency_cone(&changed_selection, 6, 1)
        .unwrap();
    let error = topology
        .replay_temporal_covariance_from_provider(
            &changed_selection,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: changed_estimate.total_bytes + changed_artifact.maximum_resident_bytes(),
            },
            request.branch_tolerance,
            &mut changed_artifact,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::SourceIdentityMismatch);

    let raw = CapturedProvider {
        identity: provider.identity.clone(),
        blocks: provider.blocks.clone(),
        stack: provider.stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let mut artifact = CovarianceArtifactReplayProvider::open(
        &directory,
        1_000_000,
        &topology,
        build_identity,
        raw,
    )
    .unwrap();
    let error = CovarianceArtifactTransaction::acquire(&directory)
        .err()
        .expect("an artifact reader must exclude a concurrent commit");
    assert!(error.to_string().contains("active replay reader"));
    let metrics = artifact.metrics();
    assert_eq!(metrics.operator_block_reads, 0);
    assert_eq!(metrics.operator_block_cache_hits, 0);
    assert_eq!(metrics.cached_block_id, None);
    assert_eq!(metrics.block_reservation_bytes, 1_000_000);
    assert_eq!(metrics.logical_block_bytes_read, Some(0));
    assert_eq!(metrics.current_cached_payload_bytes, None);
    assert_eq!(metrics.peak_cached_payload_bytes, None);
    let artifact_selection = [(GlobalDateId::new(0), 0), (GlobalDateId::new(1), 0)];
    let artifact_internal = topology
        .estimate_dependency_cone(&artifact_selection, 6, 1)
        .unwrap();
    let artifact_result = topology
        .replay_temporal_covariance_from_provider(
            &artifact_selection,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: artifact_internal.total_bytes + artifact.maximum_resident_bytes(),
            },
            request.branch_tolerance,
            &mut artifact,
        )
        .unwrap();
    assert_eq!(artifact_result.dependency_cone.provider_bytes, 1_000_256);
    assert!(artifact_result.source_cache_peak_bytes > 0);
    assert!(
        artifact_result.source_cache_peak_bytes
            <= artifact_result.dependency_cone.source_window_bytes
    );
    assert_eq!(artifact_result.covariance[(0, 0)], 0.0);
    let metrics = artifact.metrics();
    assert_eq!(metrics.operator_block_reads, 1);
    assert!(metrics.operator_block_cache_hits > 0);
    assert_eq!(metrics.cached_block_id, Some(topology.blocks()[0].id.get()));
    assert_eq!(metrics.logical_block_bytes_read, Some(root_payload));
    assert_eq!(metrics.current_cached_payload_bytes, Some(root_payload));
    assert_eq!(metrics.peak_cached_payload_bytes, Some(root_payload));

    let two_parent_selection = [(GlobalDateId::new(6), 0)];
    let two_parent_internal = topology
        .estimate_dependency_cone(&two_parent_selection, 6, 1)
        .unwrap();
    let reads_before = artifact.metrics().operator_block_reads;
    let error = topology
        .replay_temporal_covariance_from_provider(
            &two_parent_selection,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: two_parent_internal.total_bytes + artifact.maximum_resident_bytes() - 1,
            },
            request.branch_tolerance,
            &mut artifact,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::DependencyConeExceedsBudget);
    assert_eq!(artifact.metrics().operator_block_reads, reads_before);
    topology
        .replay_temporal_covariance_from_provider(
            &two_parent_selection,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: two_parent_internal.total_bytes + artifact.maximum_resident_bytes(),
            },
            request.branch_tolerance,
            &mut artifact,
        )
        .unwrap();
    let two_parent_reads = artifact.metrics().operator_block_reads - reads_before;
    assert!(
        two_parent_reads <= 16,
        "two carried parents must be read parent-major, not once per support pixel"
    );
    drop(artifact);
    drop(changed_artifact);
    drop(CovarianceArtifactTransaction::acquire(&directory).unwrap());

    let changed_factor_raw = ChangedFactorResolver(CapturedProvider {
        identity: provider.identity.clone(),
        blocks: provider.blocks.clone(),
        stack: provider.stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    });
    let mut changed_factor_artifact = CovarianceArtifactReplayProvider::open(
        &directory,
        1_000_000,
        &topology,
        build_identity,
        changed_factor_raw,
    )
    .unwrap();
    let error = topology
        .replay_temporal_covariance_from_provider(
            &artifact_selection,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: artifact_internal.total_bytes
                    + changed_factor_artifact.maximum_resident_bytes(),
            },
            request.branch_tolerance,
            &mut changed_factor_artifact,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::SourceIdentityMismatch);
    drop(changed_factor_artifact);

    let raw = CapturedProvider {
        identity: provider.identity.clone(),
        blocks: provider.blocks.clone(),
        stack: provider.stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let error =
        CovarianceArtifactReplayProvider::open(&directory, 1, &topology, build_identity, raw)
            .err()
            .expect("metadata admission must enforce the provider cap");
    assert_eq!(error.status(), ReplayStatus::SourceUnavailable);

    use std::io::Write as _;
    std::fs::OpenOptions::new()
        .append(true)
        .open(directory.join(COVARIANCE_OPERATOR_FILENAME))
        .unwrap()
        .write_all(&[0])
        .unwrap();
    let tampered_raw = CapturedProvider {
        identity: provider.identity,
        blocks: provider.blocks,
        stack: provider.stack,
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let error = CovarianceArtifactReplayProvider::open(
        &directory,
        1_000_000,
        &topology,
        build_identity,
        tampered_raw,
    )
    .err()
    .expect("post-commit HDF5 mutation must invalidate the manifest");
    assert_eq!(error.status(), ReplayStatus::SourceUnavailable);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
#[allow(clippy::too_many_lines)]
fn production_two_ministack_covariance_matches_central_difference_jjt() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let cfg = SequentialConfig {
        ministack_size: 2,
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
    };
    let amplitudes = [1.0, 1.2, 0.9, 1.1];
    let phases = [0.4, 0.9, 1.25, 1.8];
    let stack = Array3::from_shape_fn((4, 1, 1), |(date, _, _)| {
        Cf64::from_polar(amplitudes[date], phases[date])
    });
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "finite-difference-burst".to_owned(),
        source_manifest_digest: [31; 32],
        source_model_version_digest: source_model_identity_digest(),
        native_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: 1,
            stride_x: 1,
        },
        output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: 1,
            stride_x: 1,
        },
        owned_output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: 1,
            stride_x: 1,
        },
        branch_tolerance: 1e-10,
    };
    let topology = SequentialReplayTopology::plan_identified(
        4,
        (1, 1),
        (1, 1),
        1,
        Array2::from_elem((1, 1), true).view(),
        &cfg,
        scope(),
        ReplayIdNamespace {
            burst_id: request.burst_id.clone(),
            source_manifest_digest: request.source_manifest_digest,
            source_model_version_digest: request.source_model_version_digest,
            native_origin: (0, 0),
            output_origin: (0, 0),
            owned_output_origin: (0, 0),
            owned_output_shape: (1, 1),
        },
    )
    .unwrap();
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let mut blocks = Vec::new();
    run_sequential_with_covariance_capture(stack.view(), &cfg, &engine, &request, |block| {
        blocks.push(block);
        Ok(())
    })
    .unwrap();
    bind_test_factor_receipts(&mut blocks);
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[1].carry_parent_ids, vec![blocks[0].block_id]);
    assert!(blocks.iter().all(|block| {
        block
            .status
            .iter()
            .all(|status| *status == CovarianceOperatorStatus::Valid)
            && block
                .compressed_status
                .iter()
                .all(|status| *status == CovarianceOperatorStatus::Valid)
            && block
                .eigen_gap
                .iter()
                .all(|gap| *gap > request.branch_tolerance)
    }));

    let encode = |digest: [u8; 32]| {
        format!(
            "sha256:{}",
            digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        )
    };
    let build_identity = SequentialReplayBuildIdentity {
        normalized_config_digest: topology.normalized_config_digest(),
        kernel_digest: sequential_replay_kernel_digest(),
        branch_tolerance: request.branch_tolerance,
    };
    let metadata = CovarianceOperatorMetadata {
        normalized_config_digest: encode(build_identity.normalized_config_digest),
        kernel_digest: encode(build_identity.kernel_digest),
        source: SourceReplayIdentity {
            manifest_digest: Some(encode(request.source_manifest_digest)),
            provider: Some(SOURCE_PROVIDER.to_owned()),
            provider_version: Some(SOURCE_PROVIDER_VERSION.to_owned()),
            model: Some(SOURCE_MODEL.to_owned()),
            model_version: Some(SOURCE_MODEL_VERSION.to_owned()),
            model_version_digest: Some(encode(request.source_model_version_digest)),
            model_receipt_digest: Some(encode([9; 32])),
        },
        replay_status: CovarianceReplayStatus::Replayable,
        stitched_status: StitchedCovarianceStatus::NotStitched,
        downstream_inference_status: DownstreamInferenceStatus::BlockedPendingIssue54And53,
        ..CovarianceOperatorMetadata::default()
    };
    let directory = std::env::temp_dir().join(format!(
        "dolphin-workflow-finite-difference-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = CovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join("phase_covariance_operator.h5.scratch");
    let plan = topology
        .covariance_operator_plan(&request.burst_id)
        .unwrap();
    let mut writer = CovarianceOperatorWriter::create(&scratch, &metadata, &plan).unwrap();
    for block in &blocks {
        writer.write_block(block).unwrap();
    }
    let write_receipt = writer.finish().unwrap();
    let disk = admit_covariance_artifact_disk_with_identity_index(
        4 * 1024 * 1024,
        write_receipt.peak_identity_index_disk_bytes,
        u64::MAX,
    )
    .unwrap();
    finalize_covariance_artifact(&transaction, &scratch, &metadata, disk, &write_receipt).unwrap();
    drop(transaction);

    let identity = SequentialSourceProviderIdentity {
        source_manifest_digest: request.source_manifest_digest,
        provider: SOURCE_PROVIDER.to_owned(),
        provider_version: SOURCE_PROVIDER_VERSION.to_owned(),
        model: SOURCE_MODEL.to_owned(),
        model_version: SOURCE_MODEL_VERSION.to_owned(),
        source_model_version_digest: request.source_model_version_digest,
        source_model_hash: [9; 32],
    };
    let stored_blocks = blocks
        .into_iter()
        .map(|block| (GlobalBlockId::new(block.block_id), block))
        .collect();
    let raw = CapturedProvider {
        identity,
        blocks: stored_blocks,
        stack: stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let mut provider = CovarianceArtifactReplayProvider::open(
        &directory,
        1_000_000,
        &topology,
        build_identity,
        raw,
    )
    .unwrap();
    let selection = (0..4)
        .map(|date| (GlobalDateId::new(date), 0))
        .collect::<Vec<_>>();
    let source_rank = 4;
    let estimate = topology
        .estimate_dependency_cone(&selection, source_rank, 1)
        .unwrap();
    let replay = topology
        .replay_temporal_covariance_from_provider(
            &selection,
            DependencyConeQuery {
                source_rank,
                microbatch: 1,
                byte_cap: estimate.total_bytes + provider.maximum_resident_bytes(),
            },
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap();

    let epsilon = 1e-4;
    let sigma = 0.02 * std::f64::consts::FRAC_1_SQRT_2;
    let mut jacobian = Array2::<f64>::zeros((4, 8));
    for block in 0..2 {
        for basis in 0..4 {
            let date = block * 2 + basis % 2;
            let perturbation = match basis < 2 {
                true => Cf64::new(epsilon * sigma, 0.0),
                false => Cf64::new(0.0, epsilon * sigma),
            };
            let mut plus_stack = stack.clone();
            let mut minus_stack = stack.clone();
            plus_stack[(date, 0, 0)] += perturbation;
            minus_stack[(date, 0, 0)] -= perturbation;
            let plus = run_sequential(plus_stack.view(), &cfg, &engine).unwrap();
            let minus = run_sequential(minus_stack.view(), &cfg, &engine).unwrap();
            for output_date in 0..4 {
                jacobian[(output_date, block * 4 + basis)] = (plus.cpx_phase[(output_date, 0, 0)]
                    * minus.cpx_phase[(output_date, 0, 0)].conj())
                .arg()
                    / (2.0 * epsilon);
            }
        }
    }
    let oracle = jacobian.dot(&jacobian.t());
    assert!(jacobian
        .slice(ndarray::s![2, ..4])
        .iter()
        .any(|v| v.abs() > 1e-8));
    assert!(jacobian
        .slice(ndarray::s![2, 4..])
        .iter()
        .any(|v| v.abs() > 1e-8));
    assert!(oracle[(1, 2)].abs() > 1e-10);
    for ((row, col), expected) in oracle.indexed_iter() {
        let actual = replay.covariance[(row, col)];
        let tolerance = 5e-9 + 5e-5 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "covariance[{row},{col}] {actual} != central-difference {expected}"
        );
    }
    assert!(replay.covariance.row(0).iter().all(|value| *value == 0.0));
    assert!(replay
        .covariance
        .column(0)
        .iter()
        .all(|value| *value == 0.0));
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn overlapping_tile_records_share_source_ids_but_not_record_ids() {
    let cfg = config();
    let identity = |native_col, output_col, owned_col| ReplayIdNamespace {
        burst_id: "burst-a".to_owned(),
        source_manifest_digest: [7; 32],
        source_model_version_digest: [8; 32],
        native_origin: (0, native_col),
        output_origin: (0, output_col),
        owned_output_origin: (0, owned_col),
        owned_output_shape: (2, 1),
    };
    let validity = ndarray::Array2::from_elem((4, 4), true);
    let left = SequentialReplayTopology::plan_identified(
        6,
        (4, 4),
        (2, 2),
        9,
        validity.view(),
        &cfg,
        scope(),
        identity(0, 0, 0),
    )
    .unwrap();
    let right = SequentialReplayTopology::plan_identified(
        6,
        (4, 4),
        (2, 2),
        9,
        validity.view(),
        &cfg,
        scope(),
        identity(2, 1, 1),
    )
    .unwrap();
    assert_ne!(left.blocks()[0].id, right.blocks()[0].id);
    assert_eq!(
        left.source_id(left.blocks()[0].id, 2).unwrap(),
        right.source_id(right.blocks()[0].id, 0).unwrap(),
        "global native column 2 is one consumer-independent source",
    );
}

#[test]
fn unsupported_scope_returns_stable_status_before_topology_allocation() {
    let mut cfg = config();
    cfg.compressed_slc_plan = CompressedSlcPlan::FirstPerMinistack;
    assert_eq!(
        SequentialReplayTopology::plan(6, (4, 4), (2, 2), 9, &cfg, scope())
            .unwrap_err()
            .status(),
        ReplayStatus::UnsupportedReferencePlan,
    );

    let mut cfg = config();
    cfg.output_reference_idx = 1;
    assert_eq!(
        SequentialReplayTopology::plan(6, (4, 4), (2, 2), 9, &cfg, scope())
            .unwrap_err()
            .status(),
        ReplayStatus::UnsupportedOutputReference,
    );

    let mut unsupported = scope();
    unsupported.backend = ReplayBackend::Gpu;
    assert_eq!(
        SequentialReplayTopology::plan(6, (4, 4), (2, 2), 9, &config(), unsupported)
            .unwrap_err()
            .status(),
        ReplayStatus::UnsupportedBackend,
    );

    let mut cfg = config();
    cfg.shp_method = ShpMethod::Glrt;
    assert_eq!(
        SequentialReplayTopology::plan(6, (4, 4), (2, 2), 9, &cfg, scope())
            .unwrap_err()
            .status(),
        ReplayStatus::UnsupportedShpMethod,
    );
}

#[test]
fn workflow_rejects_invalid_references_zero_carry_and_nonfinite_acquisitions() {
    let stack = Array3::from_shape_fn((6, 3, 3), |(date, row, column)| {
        Cf64::from_polar(1.0 + 0.01 * (row + column) as f64, 0.3 + 0.1 * date as f64)
    });
    let engine = ComputeEngine::new(ComputeBackend::Cpu);

    let mut invalid_reference = config();
    invalid_reference.output_reference_idx = 6;
    assert!(run_sequential(stack.view(), &invalid_reference, &engine).is_err());
    invalid_reference.output_reference_idx = usize::MAX;
    assert!(run_sequential(stack.view(), &invalid_reference, &engine).is_err());

    let mut zero_carry = config();
    zero_carry.max_num_compressed = 0;
    assert!(run_sequential(stack.view(), &zero_carry, &engine).is_ok());
    let zero_carry_request = SequentialCovarianceCaptureRequest {
        burst_id: "zero-carry".to_owned(),
        source_manifest_digest: [13; 32],
        source_model_version_digest: [14; 32],
        native_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 3,
            cols: 3,
            stride_y: 1,
            stride_x: 1,
        },
        output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: 2,
            stride_x: 2,
        },
        owned_output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: 2,
            stride_x: 2,
        },
        branch_tolerance: 1e-10,
    };
    assert!(run_sequential_with_covariance_capture(
        stack.view(),
        &zero_carry,
        &engine,
        &zero_carry_request,
        |_| Ok(())
    )
    .is_err());

    let mut nonfinite = stack;
    nonfinite
        .index_axis_mut(ndarray::Axis(0), 2)
        .fill(Cf64::new(f64::NAN, f64::NAN));
    assert!(run_sequential(nonfinite.view(), &config(), &engine).is_err());

    let invalid_tolerance = SequentialCovarianceCaptureRequest {
        burst_id: "invalid-tolerance".to_owned(),
        source_manifest_digest: [15; 32],
        source_model_version_digest: [16; 32],
        native_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 3,
            cols: 3,
            stride_y: 1,
            stride_x: 1,
        },
        output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 2,
            cols: 2,
            stride_y: 2,
            stride_x: 2,
        },
        owned_output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 2,
            cols: 2,
            stride_y: 2,
            stride_x: 2,
        },
        branch_tolerance: 0.0,
    };
    let error = run_sequential_with_covariance_capture(
        nonfinite.view(),
        &config(),
        &engine,
        &invalid_tolerance,
        |_| Ok(()),
    )
    .err()
    .unwrap();
    assert_eq!(error.status(), ReplayStatus::InvalidTopology);
}

#[test]
fn singular_fixed_estimator_nodes_are_persisted_as_explicit_statuses() {
    let stack = Array3::from_shape_fn((3, 3, 3), |(date, row, column)| {
        let sample = row * 3 + column;
        if sample > 1 {
            return Cf64::new(0.0, 0.0);
        }
        let signed = match sample {
            0 => 1.0,
            _ => -1.0,
        };
        let phase = match date {
            0 => 0.3,
            1 => 0.3 + signed * 2.0 * std::f64::consts::PI / 3.0,
            _ => 0.3 - signed * 2.0 * std::f64::consts::PI / 3.0,
        };
        Cf64::from_polar(1.0, phase)
    });
    let mut cfg = config();
    cfg.ministack_size = 3;
    cfg.strides = dolphin_core::Strides { y: 3, x: 3 };
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "singular".to_owned(),
        source_manifest_digest: [12; 32],
        source_model_version_digest: [13; 32],
        native_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 3,
            cols: 3,
            stride_y: 1,
            stride_x: 1,
        },
        output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: 3,
            stride_x: 3,
        },
        owned_output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: 3,
            stride_x: 3,
        },
        branch_tolerance: 1e-8,
    };
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let mut blocks = Vec::new();
    run_sequential_with_covariance_capture(stack.view(), &cfg, &engine, &request, |block| {
        blocks.push(block);
        Ok(())
    })
    .unwrap();
    assert_eq!(
        blocks[0].status,
        vec![CovarianceOperatorStatus::SingularLocalInformation]
    );
}

//! Issue #52 unconditional workflow contract for `sequential_source_dag_v1`.

use std::collections::BTreeMap;

use dolphin_core::config::{CompressedSlcPlan, ComputeBackend, ShpMethod};
use dolphin_core::Cf64;
use dolphin_io::{
    CovarianceOperatorBlock, CovarianceOperatorGrid, CovarianceOperatorMetadata,
    CovarianceOperatorStatus, CovarianceOperatorWriter, CovariancePhaseComponentKind,
    CovarianceReplayStatus, DownstreamInferenceStatus, SourceReplayIdentity,
    StitchedCovarianceStatus,
};
use dolphin_phaselink::{
    ComputeEngine, InfluenceDag, InfluenceNode, NodeId, ParentEdge, ProperComplexFactor,
    SourceDefinition, SourceEdge, SourceId, TemporalCoordinate,
};
use dolphin_workflows::{
    run_sequential, run_sequential_with_covariance_capture, CovarianceArtifactReplayProvider,
    DependencyConeQuery, GlobalBlockId, GlobalDateId, ReplayBackend, ReplayExecutionScope,
    ReplayIdNamespace, ReplayStatus, ResolvedCompressionReplay, ResolvedPhaseReplay,
    ResolvedPrimitiveSource, SequentialConfig, SequentialCovarianceCaptureRequest,
    SequentialPrimitiveSourceResolver, SequentialReplayBlock, SequentialReplayError,
    SequentialReplayTopology, SequentialSourceProviderIdentity, SequentialSourceReplayProvider,
};
use ndarray::{array, Array1, Array2, Array3};

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

struct CapturedProvider {
    identity: SequentialSourceProviderIdentity,
    blocks: BTreeMap<GlobalBlockId, CovarianceOperatorBlock>,
    stack: Array3<Cf64>,
    source_reads: usize,
    fail_source_model: bool,
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
        let samples = Array1::from_iter(
            stored
                .source_date_indices
                .iter()
                .map(|&date| self.stack[(date as usize, row, column)]),
        );
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
            content_digest: [11; 32],
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
    assert_eq!(estimate.frontier_bytes, 4_344);
    assert_eq!(estimate.source_window_bytes, 296);
    assert_eq!(estimate.operator_bytes, 6_392);
    assert_eq!(estimate.baseline_bytes, 2_368);
    assert_eq!(estimate.support_bytes, 6);
    assert_eq!(estimate.covariance_bytes, 72);
    assert_eq!(estimate.provider_bytes, 0);
    assert_eq!(estimate.total_bytes, 13_478);

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
    assert!(small_estimate.total_bytes < 32 * 32 * 32);
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
        source_model_version_digest: [4; 32],
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
#[allow(clippy::too_many_lines)]
fn production_replay_preflights_streams_jvps_and_bounds_two_parent_block_reads() {
    let stack = Array3::from_shape_fn((9, 4, 4), |(date, row, col)| {
        let amplitude = 1.0 + 0.07 * date as f64 + 0.01 * (row + col) as f64;
        let phase = 0.4 + 0.11 * date as f64 + 0.017 * row as f64 - 0.013 * col as f64;
        Cf64::from_polar(amplitude, phase)
    });
    let cfg = config();
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "burst-a".to_owned(),
        source_manifest_digest: [3; 32],
        source_model_version_digest: [4; 32],
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
    let artifact_blocks = captured.clone();
    let validity = Array2::from_elem((4, 4), true);
    let topology = SequentialReplayTopology::plan_identified(
        9,
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

    provider.fail_source_model = true;
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
    let metadata = CovarianceOperatorMetadata {
        normalized_config_digest: encode([10; 32]),
        kernel_digest: encode([11; 32]),
        source: SourceReplayIdentity {
            manifest_digest: Some(encode(request.source_manifest_digest)),
            provider: Some("captured-provider".to_owned()),
            provider_version: Some("1".to_owned()),
            model: Some("proper-complex".to_owned()),
            model_version: Some(encode(request.source_model_version_digest)),
            model_receipt_digest: Some(encode([9; 32])),
        },
        replay_status: CovarianceReplayStatus::Replayable,
        stitched_status: StitchedCovarianceStatus::NotStitched,
        downstream_inference_status: DownstreamInferenceStatus::BlockedPendingIssue54And53,
        ..CovarianceOperatorMetadata::default()
    };
    let path = std::env::temp_dir().join(format!(
        "dolphin-workflow-capped-replay-{}.h5",
        std::process::id()
    ));
    let mut writer = CovarianceOperatorWriter::create(&path, &metadata).unwrap();
    for block in &artifact_blocks {
        writer.write_block(block).unwrap();
    }
    writer.finish().unwrap();

    let raw = CapturedProvider {
        identity: provider.identity.clone(),
        blocks: provider.blocks.clone(),
        stack: provider.stack.clone(),
        source_reads: 0,
        fail_source_model: false,
    };
    let mut artifact = CovarianceArtifactReplayProvider::open(&path, 1_000_000, raw).unwrap();
    let metrics = artifact.metrics();
    assert_eq!(metrics.operator_block_reads, 0);
    assert_eq!(metrics.operator_block_cache_hits, 0);
    assert_eq!(metrics.cached_block_id, None);
    assert_eq!(metrics.block_reservation_bytes, 1_000_000);
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
    assert_eq!(artifact_result.covariance[(0, 0)], 0.0);
    let metrics = artifact.metrics();
    assert_eq!(metrics.operator_block_reads, 1);
    assert!(metrics.operator_block_cache_hits > 0);
    assert_eq!(metrics.cached_block_id, Some(topology.blocks()[0].id.get()));
    assert_eq!(metrics.logical_block_bytes_read, None);
    assert_eq!(metrics.current_cached_payload_bytes, None);
    assert_eq!(metrics.peak_cached_payload_bytes, None);

    let two_parent_selection = [(GlobalDateId::new(6), 0)];
    let two_parent_internal = topology
        .estimate_dependency_cone(&two_parent_selection, 6, 1)
        .unwrap();
    let reads_before = artifact.metrics().operator_block_reads;
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

    let raw = CapturedProvider {
        identity: provider.identity,
        blocks: provider.blocks,
        stack: provider.stack,
        source_reads: 0,
        fail_source_model: false,
    };
    let error = CovarianceArtifactReplayProvider::open(&path, 1, raw)
        .err()
        .expect("metadata admission must enforce the provider cap");
    assert_eq!(error.status(), ReplayStatus::SourceUnavailable);
    std::fs::remove_file(path).unwrap();
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
    assert!(run_sequential(stack.view(), &zero_carry, &engine).is_err());

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

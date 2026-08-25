//! Issue #52 unconditional workflow contract for `sequential_source_dag_v1`.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::rc::Rc;
use std::sync::Mutex;

use dolphin_core::config::{
    CompressedSlcPlan, ComputeBackend, EmpiricalSourceFactorOptions, InputType, ShpMethod,
};
use dolphin_core::{BlockIndices, Cf32, Cf64};
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
use dolphin_shp::{estimate_neighbors_glrt, estimate_neighbors_ks};
use dolphin_workflows::{
    admit_covariance_artifact_disk_with_identity_index, empirical_source_factor_receipt_digest,
    finalize_covariance_artifact,
    replay_global_reference_difference_covariance_from_provider_bundle, run_sequential,
    run_sequential_masked_with_covariance_capture_and_source_factors,
    run_sequential_with_covariance_capture,
    run_sequential_with_covariance_capture_and_source_factors, sequential_replay_kernel_digest,
    sequential_source_model_identity_digest, CovarianceArtifactReplayProvider,
    CovarianceArtifactTransaction, CslcCovarianceManifest, DependencyConeQuery, GlobalBlockId,
    GlobalDateId, GlobalReferenceCovarianceQuery, ReplayBackend, ReplayExecutionScope,
    ReplayIdNamespace, ReplayStatus, ResolvedCompressionReplay, ResolvedPhaseReplay,
    ResolvedPrimitiveSource, SequentialConfig, SequentialCovarianceCaptureRequest,
    SequentialPrimitiveSourceResolver, SequentialReplayBlock, SequentialReplayBuildIdentity,
    SequentialReplayError, SequentialReplayTopology, SequentialSourceProviderIdentity,
    SequentialSourceReplayProvider, SequentialTileReplayProvider, COVARIANCE_OPERATOR_FILENAME,
    CSLC_COVARIANCE_SOURCE_MODEL, CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
    CSLC_COVARIANCE_SOURCE_PROVIDER, CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
};
use ndarray::{array, Array1, Array2, Array3, Axis};
use sha2::{Digest, Sha256};

const SOURCE_PROVIDER: &str = "captured-provider";
const SOURCE_PROVIDER_VERSION: &str = "1";
const SOURCE_MODEL: &str = "proper-complex";
const SOURCE_MODEL_VERSION: &str = "1";
static HDF5_LOCK: Mutex<()> = Mutex::new(());

fn write_cslc_source_member(path: &Path, date: usize, changed: bool) {
    write_cslc_source_member_shape(path, date, changed, (5, 5));
}

fn write_cslc_source_member_shape(path: &Path, date: usize, changed: bool, shape: (usize, usize)) {
    let _ = std::fs::remove_file(path);
    let values = Array2::from_shape_fn(shape, |(row, col)| {
        let bump = if changed && row == shape.0 / 2 && col == shape.1 / 2 {
            3.0
        } else {
            0.0
        };
        Cf32::new(
            1.0 + date as f32 * 0.2 + row as f32 * 0.03 + bump,
            0.5 + col as f32 * 0.04 - date as f32 * 0.01,
        )
    });
    let file = hdf5::File::create(path).unwrap();
    file.new_dataset_builder()
        .with_data(&values)
        .create("data")
        .unwrap();
}

fn write_constant_cslc_source_member(path: &Path, date: usize) {
    let _ = std::fs::remove_file(path);
    let value = Cf32::new(1.0 + date as f32 * 0.2, 0.5 - date as f32 * 0.01);
    let values = Array2::from_elem((5, 5), value);
    let file = hdf5::File::create(path).unwrap();
    file.new_dataset_builder()
        .with_data(&values)
        .create("data")
        .unwrap();
}

struct FixedValidity(Array2<bool>);

impl dolphin_workflows::CslcCovarianceValidityReader for FixedValidity {
    fn read_validity(&self, block: BlockIndices) -> Result<Array2<bool>, SequentialReplayError> {
        Ok(self
            .0
            .slice(ndarray::s![block.rows(), block.cols()])
            .to_owned())
    }
}

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

fn encode_digest(digest: [u8; 32]) -> String {
    format!(
        "sha256:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

struct CountingPrimitiveResolver<R> {
    inner: R,
    calls: Rc<Cell<u64>>,
}

impl<R> SequentialPrimitiveSourceResolver for CountingPrimitiveResolver<R>
where
    R: SequentialPrimitiveSourceResolver,
{
    fn identity(&self) -> &SequentialSourceProviderIdentity {
        self.inner.identity()
    }

    fn maximum_resident_bytes(&self) -> u64 {
        self.inner.maximum_resident_bytes()
    }

    fn factor_receipt_digest(
        &self,
        source: &ResolvedPrimitiveSource,
    ) -> Result<[u8; 32], SequentialReplayError> {
        self.inner.factor_receipt_digest(source)
    }

    fn resolve_source(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError> {
        self.calls.set(self.calls.get() + 1);
        self.inner.resolve_source(block, native_index)
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn cslc_member_bytes_and_order_define_shared_tile_edge_factor_identity() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    assert_eq!(
        CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
        dolphin_phaselink::EMPIRICAL_PROPER_COMPLEX_VERSION.to_string()
    );
    let root = std::env::temp_dir().join(format!(
        "dolphin_cslc_covariance_source_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    assert!(CslcCovarianceManifest::capture(
        InputType::OperaCslc,
        "/data",
        &[root.join("missing.h5")],
    )
    .is_err());
    let paths = (0..3)
        .map(|date| root.join(format!("source_{date}.h5")))
        .collect::<Vec<_>>();
    for (date, path) in paths.iter().enumerate() {
        write_cslc_source_member(path, date, false);
    }
    let manifest = CslcCovarianceManifest::capture(InputType::OperaCslc, "/data", &paths).unwrap();
    let reversed = CslcCovarianceManifest::capture(
        InputType::OperaCslc,
        "/data",
        &paths.iter().cloned().rev().collect::<Vec<_>>(),
    )
    .unwrap();
    assert_ne!(manifest.digest(), reversed.digest());

    let model_version = sequential_source_model_identity_digest(
        CSLC_COVARIANCE_SOURCE_PROVIDER,
        CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
        CSLC_COVARIANCE_SOURCE_MODEL,
        CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
    );
    let block = SequentialReplayBlock {
        id: GlobalBlockId::new(9),
        generation: 0,
        real_date_start: GlobalDateId::new(0),
        num_real_dates: 3,
        carried_parent_ids: Vec::new(),
        phase_dimension: 2,
    };
    let options = EmpiricalSourceFactorOptions {
        half_window: dolphin_core::HalfWindow { y: 1, x: 1 },
        shrinkage_alpha: 0.2,
        relative_diagonal_floor: 1e-8,
    };
    let mut left = manifest
        .resolver(
            &[0, 1, 2],
            "burst",
            (0, 0),
            (5, 5),
            CovarianceOperatorGrid {
                row_start: 0,
                col_start: 0,
                rows: 5,
                cols: 5,
                stride_y: 1,
                stride_x: 1,
            },
            &options,
            model_version,
            None,
        )
        .unwrap();
    let mut right = manifest
        .resolver(
            &[0, 1, 2],
            "burst",
            (1, 1),
            (3, 3),
            CovarianceOperatorGrid {
                row_start: 1,
                col_start: 1,
                rows: 3,
                cols: 3,
                stride_y: 1,
                stride_x: 1,
            },
            &options,
            model_version,
            None,
        )
        .unwrap();
    let from_left = left.resolve_source(&block, 12).unwrap();
    let left_receipt = left.factor_receipt_digest(&from_left).unwrap();
    let from_right = right.resolve_source(&block, 4).unwrap();
    let right_receipt = right.factor_receipt_digest(&from_right).unwrap();
    assert_eq!(from_left.id, from_right.id);
    assert_eq!(from_left.content_digest, from_right.content_digest);
    assert_eq!(
        from_left.factor.numeric_receipt_digest(),
        from_right.factor.numeric_receipt_digest()
    );
    assert_eq!(left_receipt, right_receipt);
    assert!(from_left
        .factor
        .numeric_receipt_digest()
        .iter()
        .any(|byte| *byte != 0));
    let mut invalid_dates = block.clone();
    invalid_dates.real_date_start = GlobalDateId::new(2);
    invalid_dates.num_real_dates = 2;
    assert!(matches!(
        left.resolve_source(&invalid_dates, 12),
        Err(SequentialReplayError::Provider(
            ReplayStatus::SourceIdentityMismatch,
            _
        ))
    ));

    let mut mutation_probe = manifest
        .resolver(
            &[0, 1, 2],
            "burst",
            (0, 0),
            (5, 5),
            CovarianceOperatorGrid {
                row_start: 0,
                col_start: 0,
                rows: 5,
                cols: 5,
                stride_y: 1,
                stride_x: 1,
            },
            &options,
            model_version,
            None,
        )
        .unwrap();

    write_cslc_source_member(&paths[1], 1, true);
    assert!(matches!(
        mutation_probe.resolve_source(&block, 12),
        Err(SequentialReplayError::Provider(
            ReplayStatus::SourceIdentityMismatch,
            _
        ))
    ));
    assert!(manifest.verify_unchanged().is_err());
    let changed = CslcCovarianceManifest::capture(InputType::OperaCslc, "/data", &paths).unwrap();
    assert_ne!(manifest.digest(), changed.digest());
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_dir(root);
}

#[test]
fn exact_factor_receipt_binds_validity_even_when_numeric_factor_is_unchanged() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let root = std::env::temp_dir().join(format!(
        "dolphin_cslc_factor_receipt_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let paths = (0..3)
        .map(|date| root.join(format!("source_{date}.h5")))
        .collect::<Vec<_>>();
    for (date, path) in paths.iter().enumerate() {
        write_constant_cslc_source_member(path, date);
    }
    let manifest = CslcCovarianceManifest::capture(InputType::OperaCslc, "/data", &paths).unwrap();
    let model_version = sequential_source_model_identity_digest(
        CSLC_COVARIANCE_SOURCE_PROVIDER,
        CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
        CSLC_COVARIANCE_SOURCE_MODEL,
        CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
    );
    let options = EmpiricalSourceFactorOptions {
        half_window: dolphin_core::HalfWindow { y: 1, x: 1 },
        shrinkage_alpha: 0.2,
        relative_diagonal_floor: 1e-8,
    };
    let block = SequentialReplayBlock {
        id: GlobalBlockId::new(9),
        generation: 0,
        real_date_start: GlobalDateId::new(0),
        num_real_dates: 3,
        carried_parent_ids: Vec::new(),
        phase_dimension: 2,
    };
    let all = FixedValidity(Array2::from_elem((5, 5), true));
    let mut changed_bits = Array2::from_elem((5, 5), true);
    changed_bits[(1, 1)] = false;
    let changed = FixedValidity(changed_bits);
    let grid = CovarianceOperatorGrid {
        row_start: 0,
        col_start: 0,
        rows: 5,
        cols: 5,
        stride_y: 1,
        stride_x: 1,
    };
    let mut first = manifest
        .resolver(
            &[0, 1, 2],
            "burst",
            (0, 0),
            (5, 5),
            grid,
            &options,
            model_version,
            Some(&all),
        )
        .unwrap();
    let mut second = manifest
        .resolver(
            &[0, 1, 2],
            "burst",
            (0, 0),
            (5, 5),
            grid,
            &options,
            model_version,
            Some(&changed),
        )
        .unwrap();
    let first_source = first.resolve_source(&block, 12).unwrap();
    let first_receipt = first.factor_receipt_digest(&first_source).unwrap();
    let second_source = second.resolve_source(&block, 12).unwrap();
    let second_receipt = second.factor_receipt_digest(&second_source).unwrap();
    assert_ne!(first_receipt, second_receipt);
    assert_eq!(
        first_source.factor.numeric_receipt_digest(),
        second_source.factor.numeric_receipt_digest()
    );
    assert_ne!(
        empirical_source_factor_receipt_digest(
            first_receipt,
            first_source.factor.numeric_receipt_digest(),
        ),
        empirical_source_factor_receipt_digest(
            second_receipt,
            second_source.factor.numeric_receipt_digest(),
        )
    );
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_dir(root);
}

#[test]
fn production_source_resolution_caches_one_expanded_tile_read_per_member() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let root =
        std::env::temp_dir().join(format!("dolphin_cslc_factor_cache_{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let paths = (0..3)
        .map(|date| root.join(format!("source_{date}.h5")))
        .collect::<Vec<_>>();
    let production_tile_shape = (129, 129);
    for (date, path) in paths.iter().enumerate() {
        write_cslc_source_member_shape(path, date, false, production_tile_shape);
    }
    let manifest = CslcCovarianceManifest::capture(InputType::OperaCslc, "/data", &paths).unwrap();
    assert_eq!(
        manifest.resource_estimate().decoded_content_bytes,
        9 * 129 * 129 * std::mem::size_of::<Cf32>() as u64
    );
    assert_eq!(manifest.resource_estimate().identity_window_reads, 9);
    assert_eq!(
        manifest.resource_estimate().maximum_resident_bytes,
        129 * 129 * std::mem::size_of::<Cf32>() as u64
    );
    let model_version = sequential_source_model_identity_digest(
        CSLC_COVARIANCE_SOURCE_PROVIDER,
        CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
        CSLC_COVARIANCE_SOURCE_MODEL,
        CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
    );
    let options = EmpiricalSourceFactorOptions {
        half_window: dolphin_core::HalfWindow { y: 1, x: 1 },
        shrinkage_alpha: 0.2,
        relative_diagonal_floor: 1e-8,
    };
    let mut resolver = manifest
        .resolver(
            &[0, 1, 2],
            "burst",
            (0, 0),
            production_tile_shape,
            CovarianceOperatorGrid {
                row_start: 0,
                col_start: 0,
                rows: production_tile_shape.0 as u32,
                cols: production_tile_shape.1 as u32,
                stride_y: 1,
                stride_x: 1,
            },
            &options,
            model_version,
            None,
        )
        .unwrap();
    let block = SequentialReplayBlock {
        id: GlobalBlockId::new(9),
        generation: 0,
        real_date_start: GlobalDateId::new(0),
        num_real_dates: 3,
        carried_parent_ids: Vec::new(),
        phase_dimension: 2,
    };
    for native in 0..production_tile_shape.0 * production_tile_shape.1 {
        resolver.resolve_source(&block, native).unwrap();
    }
    let metrics = resolver.metrics();
    assert_eq!(metrics.member_window_reads, 3);
    assert_eq!(metrics.tile_cache_loads, 1);
    assert_eq!(
        metrics.source_resolutions,
        (production_tile_shape.0 * production_tile_shape.1) as u64
    );
    assert!(resolver.maximum_resident_bytes() >= metrics.peak_cached_bytes);
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_dir(root);
}

#[test]
#[allow(clippy::too_many_lines)]
fn production_masked_capture_persists_and_validates_masked_source_receipts() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let root =
        std::env::temp_dir().join(format!("dolphin_cslc_masked_factor_{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let paths = (0..3)
        .map(|date| root.join(format!("source_{date}.h5")))
        .collect::<Vec<_>>();
    for (date, path) in paths.iter().enumerate() {
        write_cslc_source_member(path, date, false);
    }
    let manifest = CslcCovarianceManifest::capture(InputType::OperaCslc, "/data", &paths).unwrap();
    let model_version = sequential_source_model_identity_digest(
        CSLC_COVARIANCE_SOURCE_PROVIDER,
        CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
        CSLC_COVARIANCE_SOURCE_MODEL,
        CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
    );
    let options = EmpiricalSourceFactorOptions {
        half_window: dolphin_core::HalfWindow { y: 1, x: 1 },
        shrinkage_alpha: 0.2,
        relative_diagonal_floor: 1e-8,
    };
    let mut validity = Array2::from_elem((5, 5), true);
    validity[(0, 0)] = false;
    let reader = FixedValidity(validity.clone());
    let native_grid = CovarianceOperatorGrid {
        row_start: 0,
        col_start: 0,
        rows: 5,
        cols: 5,
        stride_y: 1,
        stride_x: 1,
    };
    let mut resolver = manifest
        .resolver(
            &[0, 1, 2],
            "burst",
            (0, 0),
            (5, 5),
            native_grid,
            &options,
            model_version,
            Some(&reader),
        )
        .unwrap();
    let stack = Array3::from_shape_fn((3, 5, 5), |(date, row, col)| {
        Cf64::new(
            f64::from(1.0_f32 + date as f32 * 0.2 + row as f32 * 0.03),
            f64::from(0.5_f32 + col as f32 * 0.04 - date as f32 * 0.01),
        )
    });
    let cfg = config();
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "burst".to_owned(),
        source_manifest_digest: manifest.digest(),
        source_model_version_digest: model_version,
        native_grid,
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
        branch_tolerance: 1e-10,
    };
    let mut blocks = Vec::new();
    run_sequential_masked_with_covariance_capture_and_source_factors(
        stack.view(),
        validity.view(),
        &cfg,
        &ComputeEngine::new(ComputeBackend::Cpu),
        &request,
        &mut resolver,
        |block| {
            blocks.push(block);
            Ok(())
        },
    )
    .unwrap();
    assert!(!blocks.is_empty());
    for block in &blocks {
        let masked = &block.source_factor_digests[..32];
        let valid = &block.source_factor_digests[32..64];
        assert!(masked.iter().any(|byte| *byte != 0));
        assert_ne!(masked, valid);
        assert_eq!(block.native_validity_bits[0] & 1, 0);
    }
    assert_eq!(
        resolver.metrics().source_resolutions,
        24 * blocks.len() as u64
    );

    let topology = SequentialReplayTopology::plan_identified(
        3,
        (5, 5),
        (2, 2),
        9,
        validity.view(),
        &cfg,
        scope(),
        ReplayIdNamespace {
            burst_id: request.burst_id.clone(),
            source_manifest_digest: request.source_manifest_digest,
            source_model_version_digest: request.source_model_version_digest,
            native_origin: (request.native_grid.row_start, request.native_grid.col_start),
            output_origin: (request.output_grid.row_start, request.output_grid.col_start),
            owned_output_origin: (
                request.owned_output_grid.row_start,
                request.owned_output_grid.col_start,
            ),
            owned_output_shape: (
                request.owned_output_grid.rows as usize,
                request.owned_output_grid.cols as usize,
            ),
        },
    )
    .unwrap();
    let source_identity = resolver.source_identity().clone();
    let build_identity = SequentialReplayBuildIdentity {
        normalized_config_digest: topology.normalized_config_digest(),
        kernel_digest: sequential_replay_kernel_digest(),
        branch_tolerance: request.branch_tolerance,
    };
    let metadata = CovarianceOperatorMetadata {
        normalized_config_digest: encode_digest(build_identity.normalized_config_digest),
        kernel_digest: encode_digest(build_identity.kernel_digest),
        source: SourceReplayIdentity {
            manifest_digest: Some(encode_digest(source_identity.source_manifest_digest)),
            provider: Some(source_identity.provider.clone()),
            provider_version: Some(source_identity.provider_version.clone()),
            model: Some(source_identity.model.clone()),
            model_version: Some(source_identity.model_version.clone()),
            model_version_digest: Some(encode_digest(source_identity.source_model_version_digest)),
            model_receipt_digest: Some(encode_digest(source_identity.source_model_hash)),
        },
        replay_status: CovarianceReplayStatus::Replayable,
        stitched_status: StitchedCovarianceStatus::NotStitched,
        downstream_inference_status: DownstreamInferenceStatus::BlockedPendingIssue54And53,
        ..CovarianceOperatorMetadata::default()
    };
    let plan = topology.covariance_operator_plan("burst").unwrap();
    let write_artifact = |directory: &Path, artifact_blocks: &[CovarianceOperatorBlock]| {
        let _ = std::fs::remove_dir_all(directory);
        std::fs::create_dir_all(directory).unwrap();
        let transaction = CovarianceArtifactTransaction::acquire(directory).unwrap();
        let scratch = directory.join("phase_covariance_operator.h5.scratch");
        let mut writer = CovarianceOperatorWriter::create(&scratch, &metadata, &plan).unwrap();
        for block in artifact_blocks {
            writer.write_block(block).unwrap();
        }
        let receipt = writer.finish().unwrap();
        let disk = admit_covariance_artifact_disk_with_identity_index(
            10 * 1024 * 1024,
            receipt.peak_identity_index_disk_bytes,
            u64::MAX,
        )
        .unwrap();
        finalize_covariance_artifact(&transaction, &scratch, &metadata, disk, &receipt).unwrap();
    };
    let artifact_directory = root.join("artifact");
    write_artifact(&artifact_directory, &blocks);
    let replay_resolver = manifest
        .resolver(
            &[0, 1, 2],
            "burst",
            (0, 0),
            (5, 5),
            native_grid,
            &options,
            model_version,
            Some(&reader),
        )
        .unwrap();
    let calls = Rc::new(Cell::new(0));
    let mut provider = CovarianceArtifactReplayProvider::open(
        &artifact_directory,
        10 * 1024 * 1024,
        &topology,
        build_identity,
        CountingPrimitiveResolver {
            inner: replay_resolver,
            calls: Rc::clone(&calls),
        },
    )
    .unwrap();
    let error =
        SequentialSourceReplayProvider::resolve_source(&mut provider, &topology.blocks()[0], 0)
            .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::MaskedNode);
    assert_eq!(
        calls.get(),
        0,
        "masked replay must not invoke the factor resolver"
    );

    let mut tampered = blocks.clone();
    tampered[0].source_factor_digests[0] ^= 1;
    let tampered_directory = root.join("tampered-artifact");
    write_artifact(&tampered_directory, &tampered);
    let replay_resolver = manifest
        .resolver(
            &[0, 1, 2],
            "burst",
            (0, 0),
            (5, 5),
            native_grid,
            &options,
            model_version,
            Some(&reader),
        )
        .unwrap();
    let mut provider = CovarianceArtifactReplayProvider::open(
        &tampered_directory,
        10 * 1024 * 1024,
        &topology,
        build_identity,
        replay_resolver,
    )
    .unwrap();
    let error =
        SequentialSourceReplayProvider::resolve_source(&mut provider, &topology.blocks()[0], 1)
            .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::SourceIdentityMismatch);
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_dir(root);
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
                .extend_from_slice(&empirical_source_factor_receipt_digest(
                    test_exact_factor_receipt(source_id),
                    factor.numeric_receipt_digest(),
                ));
        }
    }
}

fn test_exact_factor_receipt(source_id: u64) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:test_exact_factor_receipt:v1");
    digest.update(source_id.to_le_bytes());
    digest.finalize().into()
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
            realized_support: {
                let bits = stored.support_bits_per_output as usize;
                let bytes = bits.div_ceil(8);
                let packed = &stored.support_bits[output_index * bytes..(output_index + 1) * bytes];
                (0..bits)
                    .map(|slot| packed[slot / 8] & (1 << (slot % 8)) != 0)
                    .collect()
            },
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

    fn factor_receipt_digest(
        &self,
        source: &ResolvedPrimitiveSource,
    ) -> Result<[u8; 32], SequentialReplayError> {
        Ok(test_exact_factor_receipt(source.id.get()))
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

    fn factor_receipt_digest(
        &self,
        source: &ResolvedPrimitiveSource,
    ) -> Result<[u8; 32], SequentialReplayError> {
        Ok(test_exact_factor_receipt(source.id.get()))
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
        (1_944, 47_608, 47_960, 6, 72, 0)
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
#[allow(clippy::too_many_lines)]
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

    let mut resolver = CapturedProvider {
        identity: SequentialSourceProviderIdentity {
            source_manifest_digest: request.source_manifest_digest,
            provider: SOURCE_PROVIDER.to_owned(),
            provider_version: SOURCE_PROVIDER_VERSION.to_owned(),
            model: SOURCE_MODEL.to_owned(),
            model_version: SOURCE_MODEL_VERSION.to_owned(),
            source_model_version_digest: request.source_model_version_digest,
            source_model_hash: [9; 32],
        },
        blocks: blocks
            .iter()
            .cloned()
            .map(|block| (GlobalBlockId::new(block.block_id), block))
            .collect(),
        stack: stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let mut factors = Vec::new();
    run_sequential_with_covariance_capture_and_source_factors(
        stack.view(),
        &cfg,
        &engine,
        &request,
        &mut resolver,
        |block| {
            factors.push(block);
            Ok(())
        },
    )
    .unwrap();
    assert!(factors.iter().all(|block| {
        block.source_factor_digests.len() == block.source_content_digests.len()
            && block
                .source_factor_digests
                .chunks_exact(32)
                .all(|digest| digest.iter().any(|byte| *byte != 0))
    }));
}

#[test]
#[allow(clippy::too_many_lines)]
fn adaptive_capture_matches_glrt_and_ks_production_and_persists_realized_support() {
    let stack = Array3::from_shape_fn((6, 4, 4), |(date, row, col)| {
        let amplitude = 1.0 + 0.04 * date as f64 + 0.002 * (row + col) as f64;
        let phase = 0.13 * date as f64 + 0.021 * row as f64 - 0.016 * col as f64;
        Cf64::from_polar(amplitude, phase)
    });
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "adaptive-burst".to_owned(),
        source_manifest_digest: [43; 32],
        source_model_version_digest: source_model_identity_digest(),
        native_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 4,
            cols: 4,
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
        branch_tolerance: 1e-10,
    };
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    for method in [ShpMethod::Glrt, ShpMethod::Ks] {
        let mut cfg = config();
        cfg.shp_method = method;
        let production = run_sequential(stack.view(), &cfg, &engine).unwrap();
        let mut blocks = Vec::new();
        let captured = run_sequential_with_covariance_capture(
            stack.view(),
            &cfg,
            &engine,
            &request,
            |block| {
                blocks.push(block);
                Ok(())
            },
        )
        .unwrap();
        assert!(captured
            .cpx_phase
            .iter()
            .zip(production.cpx_phase.iter())
            .all(|(left, right)| left.re.to_bits() == right.re.to_bits()
                && left.im.to_bits() == right.im.to_bits()));
        assert!(captured
            .compressed_slcs
            .iter()
            .flatten()
            .zip(production.compressed_slcs.iter().flatten())
            .all(|(left, right)| left.re.to_bits() == right.re.to_bits()
                && left.im.to_bits() == right.im.to_bits()));
        for block in &blocks {
            let dates = block
                .source_date_indices
                .iter()
                .map(|&date| date as usize)
                .collect::<Vec<_>>();
            let real = stack.select(Axis(0), &dates);
            let amplitude = real.mapv(|value| value.norm());
            let expected = match method {
                ShpMethod::Glrt => estimate_neighbors_glrt(
                    amplitude.mean_axis(Axis(0)).unwrap().view(),
                    amplitude.var_axis(Axis(0), 0.0).view(),
                    cfg.half_window,
                    real.dim().0,
                    cfg.strides,
                    cfg.shp_alpha,
                ),
                ShpMethod::Ks => estimate_neighbors_ks(
                    amplitude.view(),
                    cfg.half_window,
                    cfg.strides,
                    cfg.shp_alpha,
                    false,
                ),
                ShpMethod::Rect => unreachable!(),
            };
            assert_eq!(block.support_bits_per_output, 9);
            for output in 0..4 {
                let packed = &block.support_bits[output * 2..output * 2 + 2];
                let output_row = output / 2;
                let output_col = output % 2;
                for slot in 0..9 {
                    assert_eq!(
                        packed[slot / 8] & (1 << (slot % 8)) != 0,
                        expected[(output_row, output_col, slot / 3, slot % 3)],
                    );
                }
                assert_eq!(packed[1] & !1, 0);
            }
        }
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn adaptive_replay_uses_persisted_support_and_rejects_a_changed_mask() {
    let stack = Array3::from_shape_fn((3, 4, 4), |(date, row, col)| {
        let amplitude = 0.8 + 0.12 * date as f64 + 0.03 * (row * 4 + col) as f64;
        let phase = 0.17 * date as f64 + 0.09 * row as f64 - 0.07 * col as f64;
        Cf64::from_polar(amplitude, phase)
    });
    let mut cfg = config();
    cfg.shp_method = ShpMethod::Glrt;
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "adaptive-replay-burst".to_owned(),
        source_manifest_digest: [47; 32],
        source_model_version_digest: source_model_identity_digest(),
        native_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 4,
            cols: 4,
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
        branch_tolerance: 1e-10,
    };
    let mut captured = Vec::new();
    run_sequential_with_covariance_capture(
        stack.view(),
        &cfg,
        &ComputeEngine::new(ComputeBackend::Cpu),
        &request,
        |block| {
            captured.push(block);
            Ok(())
        },
    )
    .unwrap();
    bind_test_factor_receipts(&mut captured);
    let topology = SequentialReplayTopology::plan_identified(
        3,
        (4, 4),
        (2, 2),
        9,
        Array2::from_elem((4, 4), true).view(),
        &cfg,
        scope(),
        ReplayIdNamespace {
            burst_id: request.burst_id.clone(),
            source_manifest_digest: request.source_manifest_digest,
            source_model_version_digest: request.source_model_version_digest,
            native_origin: (0, 0),
            output_origin: (0, 0),
            owned_output_origin: (0, 0),
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
    let provider_for = |blocks: Vec<CovarianceOperatorBlock>| CapturedProvider {
        identity: identity.clone(),
        blocks: blocks
            .into_iter()
            .map(|block| (GlobalBlockId::new(block.block_id), block))
            .collect(),
        stack: stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let selection = [(GlobalDateId::new(0), 0), (GlobalDateId::new(1), 0)];
    let query = DependencyConeQuery {
        source_rank: 6,
        microbatch: 1,
        byte_cap: u64::MAX,
    };
    let mut provider = provider_for(captured.clone());
    let replay = topology
        .replay_temporal_covariance_from_provider(
            &selection,
            query,
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap();
    assert!(replay.covariance.iter().all(|value| value.is_finite()));

    let reference_selection = [(GlobalDateId::new(0), 1), (GlobalDateId::new(1), 1)];
    let mut provider = provider_for(captured.clone());
    let pair = topology
        .replay_reference_difference_covariance_from_provider(
            &selection,
            &reference_selection,
            query,
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap();
    let mut expected_union = BTreeSet::new();
    for (output, column_start) in [(0, 0_u64), (1, 1_u64)] {
        let packed = &captured[0].support_bits[output * 2..output * 2 + 2];
        for slot in 0..9 {
            if packed[slot / 8] & (1 << (slot % 8)) != 0 {
                expected_union.insert(((slot / 3) as u64, column_start + (slot % 3) as u64));
            }
        }
    }
    assert_eq!(
        pair.effective_looks.as_ref().unwrap().support_union_count,
        expected_union.len()
    );

    let mut changed = captured;
    changed[0].support_bits[0] = 1;
    changed[0].support_bits[1] = 0;
    let mut provider = provider_for(changed);
    let error = topology
        .replay_temporal_covariance_from_provider(
            &selection,
            query,
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::ReplayStateMismatch);
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

    let joint_selection = [
        (GlobalDateId::new(0), 0),
        (GlobalDateId::new(1), 0),
        (GlobalDateId::new(3), 0),
        (GlobalDateId::new(6), 0),
    ];
    let shared_reference = [
        (GlobalDateId::new(0), 1),
        (GlobalDateId::new(1), 1),
        (GlobalDateId::new(3), 1),
        (GlobalDateId::new(6), 1),
    ];
    let reads_before_joint = provider.source_reads;
    let shared = topology
        .replay_reference_difference_covariance_from_provider(
            &joint_selection,
            &shared_reference,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: u64::MAX,
            },
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap();
    assert!(provider.source_reads > reads_before_joint);
    assert!(shared
        .difference_covariance
        .iter()
        .all(|value| value.is_finite()));
    assert!(shared.difference_covariance[(3, 3)] > 0.0);
    assert!(shared.source_cache_peak_bytes <= shared.dependency_cone.source_window_bytes);
    assert_eq!(shared.dependency_cone.provider_bytes, 256);
    assert_ne!(shared.reference_signature, [0; 32]);

    let provider_for_same_topology = || CapturedProvider {
        identity: provider.identity.clone(),
        blocks: provider.blocks.clone(),
        stack: provider.stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let mut target_provider = provider_for_same_topology();
    let mut reference_provider = provider_for_same_topology();
    let cross_api_same_topology = topology
        .replay_cross_topology_reference_difference_covariance_from_providers(
            &joint_selection,
            &mut target_provider,
            &topology,
            &shared_reference,
            &mut reference_provider,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: u64::MAX,
            },
            request.branch_tolerance,
        )
        .unwrap();
    assert_eq!(
        cross_api_same_topology.target_covariance,
        shared.target_covariance
    );
    assert_eq!(
        cross_api_same_topology.reference_covariance,
        shared.reference_covariance
    );
    assert_eq!(
        cross_api_same_topology.target_reference_covariance,
        shared.target_reference_covariance
    );
    assert_eq!(
        cross_api_same_topology.difference_covariance,
        shared.difference_covariance
    );
    assert_eq!(
        cross_api_same_topology.reference_signature,
        shared.reference_signature
    );
    assert_eq!(
        cross_api_same_topology.effective_looks,
        shared.effective_looks
    );

    let mut target_provider = provider_for_same_topology();
    let mut tampered_reference_provider = provider_for_same_topology();
    tampered_reference_provider.dishonest_samples = true;
    let error = topology
        .replay_cross_topology_reference_difference_covariance_from_providers(
            &joint_selection,
            &mut target_provider,
            &topology,
            &shared_reference,
            &mut tampered_reference_provider,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: u64::MAX,
            },
            request.branch_tolerance,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::SourceIdentityMismatch);
    assert!(tampered_reference_provider.source_reads > 0);

    let mut target_provider = provider_for_same_topology();
    let mut tampered_reference_provider = provider_for_same_topology();
    for block in tampered_reference_provider.blocks.values_mut() {
        block.support_bits[0] ^= 1;
    }
    let error = topology
        .replay_cross_topology_reference_difference_covariance_from_providers(
            &joint_selection,
            &mut target_provider,
            &topology,
            &shared_reference,
            &mut tampered_reference_provider,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: u64::MAX,
            },
            request.branch_tolerance,
        )
        .unwrap_err();
    assert!(matches!(
        error.status(),
        ReplayStatus::ReplayStateMismatch | ReplayStatus::SourceIdentityMismatch
    ));

    let mut disjoint_cfg = cfg;
    disjoint_cfg.half_window = dolphin_core::HalfWindow { y: 0, x: 0 };
    let mut disjoint_blocks = Vec::new();
    run_sequential_with_covariance_capture(
        provider.stack.view(),
        &disjoint_cfg,
        &engine,
        &request,
        |block| {
            disjoint_blocks.push(block);
            Ok(())
        },
    )
    .unwrap();
    bind_test_factor_receipts(&mut disjoint_blocks);
    let disjoint_topology = SequentialReplayTopology::plan_identified(
        8,
        (4, 4),
        (2, 2),
        1,
        Array2::from_elem((4, 4), true).view(),
        &disjoint_cfg,
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
    let disjoint_blocks = disjoint_blocks
        .into_iter()
        .map(|block| (GlobalBlockId::new(block.block_id), block))
        .collect();
    let disjoint_identity = provider.identity.clone();
    let mut disjoint_provider = CapturedProvider {
        identity: disjoint_identity,
        blocks: disjoint_blocks,
        stack: provider.stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let disjoint_reference = [
        (GlobalDateId::new(0), 3),
        (GlobalDateId::new(1), 3),
        (GlobalDateId::new(3), 3),
        (GlobalDateId::new(6), 3),
    ];
    let disjoint = disjoint_topology
        .replay_reference_difference_covariance_from_provider(
            &joint_selection,
            &disjoint_reference,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: u64::MAX,
            },
            request.branch_tolerance,
            &mut disjoint_provider,
        )
        .unwrap();
    assert!(disjoint
        .target_reference_covariance
        .iter()
        .all(|value| value.abs() < 1.0e-12));
    assert_ne!(disjoint.reference_signature, shared.reference_signature);

    let coincident = topology
        .replay_reference_difference_covariance_from_provider(
            &joint_selection,
            &joint_selection,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: u64::MAX,
            },
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap();
    assert_eq!(
        coincident.difference_covariance,
        Array2::<f64>::zeros((4, 4))
    );
    assert_ne!(coincident.reference_signature, shared.reference_signature);

    let reads_before_cap = provider.source_reads;
    let error = topology
        .replay_reference_difference_covariance_from_provider(
            &joint_selection,
            &shared_reference,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: 0,
            },
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::DependencyConeExceedsBudget);
    assert_eq!(provider.source_reads, reads_before_cap);

    let error = topology
        .replay_reference_difference_covariance_from_provider(
            &joint_selection,
            &[
                (GlobalDateId::new(0), 1),
                (GlobalDateId::new(1), 1),
                (GlobalDateId::new(4), 1),
                (GlobalDateId::new(6), 1),
            ],
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: u64::MAX,
            },
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::InvalidReference);
    assert_eq!(provider.source_reads, reads_before_cap);

    let target_marginal = topology
        .replay_temporal_covariance_from_provider(
            &joint_selection,
            DependencyConeQuery {
                source_rank: 6,
                microbatch: 1,
                byte_cap: u64::MAX,
            },
            request.branch_tolerance,
            &mut provider,
        )
        .unwrap();
    let effective_fraction = shared.effective_looks.as_ref().unwrap().fraction;
    for row in 0..4 {
        for column in 0..4 {
            assert!(
                (shared.target_covariance[(row, column)]
                    - target_marginal.covariance[(row, column)] / effective_fraction)
                    .abs()
                    < 1.0e-10
            );
        }
    }

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
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
fn cross_tile_joint_replay_matches_dense_shared_source_oracle_and_fails_closed() {
    let cfg = config();
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let full_stack = Array3::from_shape_fn((3, 4, 8), |(date, row, column)| {
        let amplitude = 0.9 + 0.08 * date as f64 + 0.01 * (row + column) as f64;
        let phase = 0.2 + 0.17 * date as f64 + 0.023 * row as f64 - 0.019 * column as f64;
        Cf64::from_polar(amplitude, phase)
    });
    let left_stack = full_stack.slice(ndarray::s![.., .., 0..6]).to_owned();
    let right_stack = full_stack.slice(ndarray::s![.., .., 2..8]).to_owned();
    let identity = SequentialSourceProviderIdentity {
        source_manifest_digest: [61; 32],
        provider: SOURCE_PROVIDER.to_owned(),
        provider_version: SOURCE_PROVIDER_VERSION.to_owned(),
        model: SOURCE_MODEL.to_owned(),
        model_version: SOURCE_MODEL_VERSION.to_owned(),
        source_model_version_digest: source_model_identity_digest(),
        source_model_hash: [9; 32],
    };
    let capture_tile = |stack: &Array3<Cf64>, native_col: u64, output_col: u64| {
        let owned_output_cols = if native_col == 0 { 1 } else { 3 };
        let request = SequentialCovarianceCaptureRequest {
            burst_id: "cross-tile-burst".to_owned(),
            source_manifest_digest: identity.source_manifest_digest,
            source_model_version_digest: identity.source_model_version_digest,
            native_grid: CovarianceOperatorGrid {
                row_start: 0,
                col_start: native_col,
                rows: 4,
                cols: 6,
                stride_y: 1,
                stride_x: 1,
            },
            output_grid: CovarianceOperatorGrid {
                row_start: 0,
                col_start: output_col,
                rows: 2,
                cols: 3,
                stride_y: 2,
                stride_x: 2,
            },
            owned_output_grid: CovarianceOperatorGrid {
                row_start: 0,
                col_start: output_col,
                rows: 2,
                cols: owned_output_cols,
                stride_y: 2,
                stride_x: 2,
            },
            branch_tolerance: 1e-10,
        };
        let mut blocks = Vec::new();
        run_sequential_with_covariance_capture(stack.view(), &cfg, &engine, &request, |block| {
            blocks.push(block);
            Ok(())
        })
        .unwrap();
        bind_test_factor_receipts(&mut blocks);
        let topology = SequentialReplayTopology::plan_identified(
            3,
            (4, 6),
            (2, 3),
            9,
            Array2::from_elem((4, 6), true).view(),
            &cfg,
            scope(),
            ReplayIdNamespace {
                burst_id: request.burst_id.clone(),
                source_manifest_digest: request.source_manifest_digest,
                source_model_version_digest: request.source_model_version_digest,
                native_origin: (0, native_col),
                output_origin: (0, output_col),
                owned_output_origin: (0, output_col),
                owned_output_shape: (2, owned_output_cols as usize),
            },
        )
        .unwrap();
        let provider = CapturedProvider {
            identity: identity.clone(),
            blocks: blocks
                .into_iter()
                .map(|block| (GlobalBlockId::new(block.block_id), block))
                .collect(),
            stack: stack.clone(),
            source_reads: 0,
            fail_source_model: false,
            dishonest_samples: false,
        };
        (topology, provider, request.branch_tolerance)
    };
    let (left, mut left_provider, branch_tolerance) = capture_tile(&left_stack, 0, 0);
    let (right, mut right_provider, _) = capture_tile(&right_stack, 2, 1);
    let dates = (0..3)
        .map(|date| GlobalDateId::new(date as u32))
        .collect::<Vec<_>>();
    let target = dates.iter().map(|&date| (date, 0)).collect::<Vec<_>>();
    let reference = dates.iter().map(|&date| (date, 0)).collect::<Vec<_>>();
    let query = DependencyConeQuery {
        source_rank: 6,
        microbatch: 1,
        byte_cap: u64::MAX,
    };
    let replay = left
        .replay_cross_topology_reference_difference_covariance_from_providers(
            &target,
            &mut left_provider,
            &right,
            &reference,
            &mut right_provider,
            query,
            branch_tolerance,
        )
        .unwrap();
    assert!(replay.target_reference_covariance[(2, 2)].abs() > 1e-12);
    let naive_difference = &replay.target_covariance + &replay.reference_covariance;
    assert!(
        (naive_difference[(2, 2)] - replay.difference_covariance[(2, 2)]).abs() > 1e-12,
        "independent marginal addition must not replace shared-source contraction",
    );
    assert!(replay.source_cache_peak_bytes <= replay.dependency_cone.source_window_bytes);
    assert_eq!(replay.dependency_cone.provider_bytes, 512);
    let expected_support = (0_u64..3)
        .flat_map(|row| (0_u64..5).map(move |column| (row, column)))
        .collect::<Vec<_>>();
    let expected_denominator = expected_support
        .iter()
        .flat_map(|left| {
            expected_support.iter().map(move |right| {
                let row = left.0.abs_diff(right.0) as f64;
                let column = left.1.abs_diff(right.1) as f64;
                (-(row.hypot(column)) / 1.5).exp()
            })
        })
        .sum::<f64>();
    let expected_fraction = expected_support.len() as f64 / expected_denominator;
    let effective_looks = replay.effective_looks.as_ref().unwrap();
    assert_eq!(effective_looks.model, "source_factor_declared_v1");
    assert_eq!(effective_looks.distance_scale_pixels, 1.5);
    assert_eq!(effective_looks.support_union_count, expected_support.len());
    assert!((effective_looks.fraction - expected_fraction).abs() < 1e-15);
    assert_ne!(effective_looks.receipt, [0; 32]);
    let mut expected_receipt = Sha256::new();
    expected_receipt.update(b"dolphinrust:effective-looks-realization:v1");
    expected_receipt.update(b"source_factor_declared_v1");
    expected_receipt.update(1.5_f64.to_bits().to_le_bytes());
    expected_receipt.update((expected_support.len() as u64).to_le_bytes());
    for &(row, column) in &expected_support {
        expected_receipt.update(row.to_le_bytes());
        expected_receipt.update(column.to_le_bytes());
    }
    expected_receipt.update(expected_fraction.to_bits().to_le_bytes());
    expected_receipt.update(replay.source_factor_receipt);
    expected_receipt.update(replay.support_receipt);
    assert_eq!(effective_looks.receipt, expected_receipt.finalize()[..]);
    let config_only_receipt = Sha256::digest(b"source_factor_declared_v1:1.5");
    assert_ne!(effective_looks.receipt, config_only_receipt[..]);

    let epsilon = 1e-4;
    let sigma = 0.02 * std::f64::consts::FRAC_1_SQRT_2;
    let mut jacobian = Array2::<f64>::zeros((6, 3 * 4 * 8 * 2));
    let output_values = |result: &dolphin_workflows::SequentialOutput| {
        let target = (0..3).map(|date| result.cpx_phase[(date, 0, 0)]);
        let reference = (0..3).map(|date| result.cpx_phase[(date, 0, 1)]);
        target.chain(reference).collect::<Vec<_>>()
    };
    let mut column = 0;
    for date in 0..3 {
        for row in 0..4 {
            for col in 0..8 {
                for imaginary in [false, true] {
                    let perturbation = match imaginary {
                        false => Cf64::new(epsilon * sigma, 0.0),
                        true => Cf64::new(0.0, epsilon * sigma),
                    };
                    let mut plus = full_stack.clone();
                    let mut minus = full_stack.clone();
                    plus[(date, row, col)] += perturbation;
                    minus[(date, row, col)] -= perturbation;
                    let plus = output_values(&run_sequential(plus.view(), &cfg, &engine).unwrap());
                    let minus =
                        output_values(&run_sequential(minus.view(), &cfg, &engine).unwrap());
                    for output in 0..6 {
                        jacobian[(output, column)] =
                            (plus[output] * minus[output].conj()).arg() / (2.0 * epsilon);
                    }
                    column += 1;
                }
            }
        }
    }
    let oracle = jacobian.dot(&jacobian.t());
    let mut actual = Array2::<f64>::zeros((6, 6));
    actual
        .slice_mut(ndarray::s![..3, ..3])
        .assign(&replay.target_covariance);
    actual
        .slice_mut(ndarray::s![3.., 3..])
        .assign(&replay.reference_covariance);
    actual
        .slice_mut(ndarray::s![..3, 3..])
        .assign(&replay.target_reference_covariance);
    actual
        .slice_mut(ndarray::s![3.., ..3])
        .assign(&replay.target_reference_covariance.t());
    for ((row, col), unscaled) in oracle.indexed_iter() {
        let expected = unscaled / expected_fraction;
        let tolerance = 5e-9 + 5e-5 * expected.abs();
        assert!(
            (actual[(row, col)] - expected).abs() <= tolerance,
            "joint covariance[{row},{col}] {} != dense oracle {expected}",
            actual[(row, col)]
        );
    }

    let provider_for = |provider: &CapturedProvider| CapturedProvider {
        identity: provider.identity.clone(),
        blocks: provider.blocks.clone(),
        stack: provider.stack.clone(),
        source_reads: 0,
        fail_source_model: false,
        dishonest_samples: false,
    };
    let mut global_left = provider_for(&left_provider);
    let mut global_right = provider_for(&right_provider);
    let mut bundle = [
        SequentialTileReplayProvider::new(&left, &mut global_left),
        SequentialTileReplayProvider::new(&right, &mut global_right),
    ];
    let global = replay_global_reference_difference_covariance_from_provider_bundle(
        &mut bundle,
        GlobalReferenceCovarianceQuery {
            burst_id: "cross-tile-burst",
            target: (0, 0),
            reference: (0, 1),
            ordered_dates: &dates,
            source_rank: 6,
            byte_cap: u64::MAX,
            branch_tolerance,
        },
    )
    .unwrap();
    assert_eq!(global.joint_phase_covariance, actual);
    assert_eq!(
        global.replay.difference_covariance,
        replay.difference_covariance
    );
    assert_eq!(
        global.replay.source_factor_receipt,
        replay.source_factor_receipt
    );
    assert_eq!(global.replay.support_receipt, replay.support_receipt);
    assert_eq!(global.replay.effective_looks, replay.effective_looks);
    assert_ne!(global.replay.source_factor_receipt, [0; 32]);
    assert_ne!(global.replay.support_receipt, [0; 32]);
    assert_eq!(global.replay.target_disposition, ReplayStatus::Valid);
    assert_eq!(global.replay.reference_disposition, ReplayStatus::Valid);
    assert!(global.resource_high_water_bytes >= global.replay.dependency_cone.total_bytes);

    let mut reverse_left = provider_for(&left_provider);
    let mut reverse_right = provider_for(&right_provider);
    let mut reverse_bundle = [
        SequentialTileReplayProvider::new(&right, &mut reverse_right),
        SequentialTileReplayProvider::new(&left, &mut reverse_left),
    ];
    let reversed = replay_global_reference_difference_covariance_from_provider_bundle(
        &mut reverse_bundle,
        GlobalReferenceCovarianceQuery {
            burst_id: "cross-tile-burst",
            target: (0, 0),
            reference: (0, 1),
            ordered_dates: &dates,
            source_rank: 6,
            byte_cap: u64::MAX,
            branch_tolerance,
        },
    )
    .unwrap();
    assert_eq!(
        reversed.joint_phase_covariance,
        global.joint_phase_covariance
    );
    assert_eq!(
        reversed.replay.reference_signature,
        global.replay.reference_signature
    );
    assert_eq!(
        reversed.replay.source_factor_receipt,
        global.replay.source_factor_receipt
    );
    assert_eq!(
        reversed.replay.support_receipt,
        global.replay.support_receipt
    );
    assert_eq!(
        reversed.replay.effective_looks,
        global.replay.effective_looks
    );

    let mut bounded_left = provider_for(&left_provider);
    let mut bounded_right = provider_for(&right_provider);
    let mut bounded_bundle = [
        SequentialTileReplayProvider::new(&left, &mut bounded_left),
        SequentialTileReplayProvider::new(&right, &mut bounded_right),
    ];
    let error = replay_global_reference_difference_covariance_from_provider_bundle(
        &mut bounded_bundle,
        GlobalReferenceCovarianceQuery {
            burst_id: "cross-tile-burst",
            target: (0, 0),
            reference: (0, 1),
            ordered_dates: &dates,
            source_rank: 6,
            byte_cap: global.resource_high_water_bytes - 1,
            branch_tolerance,
        },
    )
    .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::DependencyConeExceedsBudget);
    assert_eq!(bounded_left.source_reads, 0);
    assert_eq!(bounded_right.source_reads, 0);

    let mut coincident_left = provider_for(&left_provider);
    let mut coincident_right = provider_for(&right_provider);
    let mut coincident_bundle = [
        SequentialTileReplayProvider::new(&left, &mut coincident_left),
        SequentialTileReplayProvider::new(&right, &mut coincident_right),
    ];
    let coincident = replay_global_reference_difference_covariance_from_provider_bundle(
        &mut coincident_bundle,
        GlobalReferenceCovarianceQuery {
            burst_id: "cross-tile-burst",
            target: (0, 1),
            reference: (0, 1),
            ordered_dates: &dates,
            source_rank: 6,
            byte_cap: u64::MAX,
            branch_tolerance,
        },
    )
    .unwrap()
    .replay;
    assert_eq!(
        coincident.difference_covariance,
        Array2::<f64>::zeros((3, 3))
    );
    assert_eq!(
        coincident.target_covariance,
        coincident.reference_covariance
    );
    assert_eq!(
        coincident.target_covariance,
        coincident.target_reference_covariance
    );

    let (far, mut far_provider, _) = capture_tile(&right_stack, 20, 10);
    let mut disjoint_left = provider_for(&left_provider);
    let disjoint = left
        .replay_cross_topology_reference_difference_covariance_from_providers(
            &target,
            &mut disjoint_left,
            &far,
            &reference,
            &mut far_provider,
            query,
            branch_tolerance,
        )
        .unwrap();
    assert!(disjoint
        .target_reference_covariance
        .iter()
        .all(|value| *value == 0.0));

    let mut cap_left = provider_for(&left_provider);
    let mut cap_right = provider_for(&right_provider);
    let cap_reads = (cap_left.source_reads, cap_right.source_reads);
    let error = left
        .replay_cross_topology_reference_difference_covariance_from_providers(
            &target,
            &mut cap_left,
            &right,
            &reference,
            &mut cap_right,
            DependencyConeQuery {
                byte_cap: 0,
                ..query
            },
            branch_tolerance,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::DependencyConeExceedsBudget);
    assert_eq!((cap_left.source_reads, cap_right.source_reads), cap_reads);

    let mut changed_identity = provider_for(&right_provider);
    changed_identity.identity.source_model_hash = [99; 32];
    let mut identity_left = provider_for(&left_provider);
    let error = left
        .replay_cross_topology_reference_difference_covariance_from_providers(
            &target,
            &mut identity_left,
            &right,
            &reference,
            &mut changed_identity,
            query,
            branch_tolerance,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::SourceIdentityMismatch);

    let mut changed_validity = Array2::from_elem((4, 6), true);
    changed_validity[(0, 0)] = false;
    let mismatched_mask = SequentialReplayTopology::plan_identified(
        3,
        (4, 6),
        (2, 3),
        9,
        changed_validity.view(),
        &cfg,
        scope(),
        ReplayIdNamespace {
            burst_id: "cross-tile-burst".to_owned(),
            source_manifest_digest: identity.source_manifest_digest,
            source_model_version_digest: identity.source_model_version_digest,
            native_origin: (0, 0),
            output_origin: (0, 0),
            owned_output_origin: (0, 0),
            owned_output_shape: (2, 3),
        },
    )
    .unwrap();
    let mut mask_left = provider_for(&left_provider);
    let mut mask_right = provider_for(&left_provider);
    let error = left
        .replay_cross_topology_reference_difference_covariance_from_providers(
            &target,
            &mut mask_left,
            &mismatched_mask,
            &reference,
            &mut mask_right,
            query,
            branch_tolerance,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::ReplayStateMismatch);

    let mut changed_cfg = cfg;
    changed_cfg.half_window = dolphin_core::HalfWindow { y: 0, x: 0 };
    let mismatched_support = SequentialReplayTopology::plan_identified(
        3,
        (4, 6),
        (2, 3),
        1,
        Array2::from_elem((4, 6), true).view(),
        &changed_cfg,
        scope(),
        ReplayIdNamespace {
            burst_id: "cross-tile-burst".to_owned(),
            source_manifest_digest: identity.source_manifest_digest,
            source_model_version_digest: identity.source_model_version_digest,
            native_origin: (0, 2),
            output_origin: (0, 1),
            owned_output_origin: (0, 1),
            owned_output_shape: (2, 3),
        },
    )
    .unwrap();
    let mut support_left = provider_for(&left_provider);
    let mut support_right = provider_for(&right_provider);
    let error = left
        .replay_cross_topology_reference_difference_covariance_from_providers(
            &target,
            &mut support_left,
            &mismatched_support,
            &reference,
            &mut support_right,
            query,
            branch_tolerance,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::ReplayStateMismatch);
    assert_eq!(support_left.source_reads, 0);
    assert_eq!(support_right.source_reads, 0);

    let mut invalid_left = provider_for(&left_provider);
    let mut invalid_right = provider_for(&right_provider);
    let invalid_dates = [(GlobalDateId::new(1), 0), (GlobalDateId::new(2), 0)];
    let error = left
        .replay_cross_topology_reference_difference_covariance_from_providers(
            &invalid_dates,
            &mut invalid_left,
            &right,
            &invalid_dates,
            &mut invalid_right,
            query,
            branch_tolerance,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::InvalidReference);

    let mut order_left = provider_for(&left_provider);
    let mut order_right = provider_for(&right_provider);
    let wrong_reference_order = [
        (GlobalDateId::new(0), 0),
        (GlobalDateId::new(2), 0),
        (GlobalDateId::new(1), 0),
    ];
    let error = left
        .replay_cross_topology_reference_difference_covariance_from_providers(
            &target,
            &mut order_left,
            &right,
            &wrong_reference_order,
            &mut order_right,
            query,
            branch_tolerance,
        )
        .unwrap_err();
    assert_eq!(error.status(), ReplayStatus::InvalidReference);
    assert_eq!(order_left.source_reads, 0);
    assert_eq!(order_right.source_reads, 0);
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

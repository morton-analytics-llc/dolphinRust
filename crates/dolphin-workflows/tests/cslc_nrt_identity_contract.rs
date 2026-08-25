use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use dolphin_core::config::{
    CompressedSlcPlan, ComputeBackend, EmpiricalSourceFactorOptions, InputType, ShpMethod,
};
use dolphin_core::{Cf32, Cf64, HalfWindow, Strides};
use dolphin_io::covariance::{
    CovarianceGenerationBlockIdentity, CovarianceGenerationIdentity, CovarianceGenerationRegistry,
    COVARIANCE_GENERATION_REGISTRY_SCHEMA_VERSION,
};
use dolphin_io::{
    CovarianceCalibrationStatus, CovarianceOperatorGrid, CovarianceOperatorMetadata,
    CovarianceOperatorWriter, CovarianceReplayStatus, DownstreamInferenceStatus,
    SourceReplayIdentity, StitchedCovarianceStatus,
};
use dolphin_phaselink::ComputeEngine;
use dolphin_workflows::{
    empirical_factor_config, plan_sequential_covariance_capture,
    run_sequential_resumable_with_covariance_capture_and_source_factors,
    sequential_replay_config_digest, sequential_replay_kernel_digest,
    sequential_source_model_identity_digest,
    update_sequential_with_covariance_capture_and_source_factors, CslcCovarianceManifest,
    GlobalBlockId, GlobalDateId, ReplayStatus, SequentialConfig,
    SequentialCovarianceCaptureRequest, SequentialCovarianceRevision,
    SequentialPrimitiveSourceResolver, SequentialReplayBlock, SequentialReplayError,
    CSLC_COVARIANCE_SOURCE_MODEL, CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
    CSLC_COVARIANCE_SOURCE_PROVIDER, CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
};
use ndarray::{Array2, Array3};

static HDF5_LOCK: Mutex<()> = Mutex::new(());

fn write_member(path: &Path, date: usize, changed: bool) {
    let _ = std::fs::remove_file(path);
    let values = Array2::from_shape_fn((3, 4), |(row, col)| {
        let mutation = if changed && (row, col) == (1, 2) {
            5.0
        } else {
            0.0
        };
        Cf32::new(
            1.0 + date as f32 * 0.2 + row as f32 * 0.01 + mutation,
            0.5 + col as f32 * 0.03,
        )
    });
    let file = hdf5::File::create(path).unwrap();
    file.new_dataset_builder()
        .with_data(&values)
        .create("data")
        .unwrap();
}

fn paths(root: &Path, count: usize) -> Vec<PathBuf> {
    (0..count)
        .map(|date| root.join(format!("source_{date}.h5")))
        .collect()
}

fn encode(digest: [u8; 32]) -> String {
    format!(
        "sha256:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

#[test]
#[allow(clippy::too_many_lines)]
fn generation_bundle_routes_exact_empirical_receipts_and_rejects_mixed_identity() {
    let _hdf5 = HDF5_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let root = std::env::temp_dir().join(format!("dolphin-cslc-nrt-bundle-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let member_paths = paths(&root, 3);
    for (date, path) in member_paths.iter().enumerate() {
        write_member(path, date, false);
    }
    let manifest =
        CslcCovarianceManifest::capture(InputType::OperaCslc, "/data", &member_paths).unwrap();
    let options = EmpiricalSourceFactorOptions {
        half_window: HalfWindow { y: 1, x: 1 },
        shrinkage_alpha: 0.2,
        relative_diagonal_floor: 1e-8,
    };
    let model_version = sequential_source_model_identity_digest(
        CSLC_COVARIANCE_SOURCE_PROVIDER,
        CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
        CSLC_COVARIANCE_SOURCE_MODEL,
        CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
    );
    let model_receipt = *empirical_factor_config(&options).unwrap().config_digest();
    let generation = |generation: u32, dates: Vec<u32>| CovarianceGenerationIdentity {
        burst_id: "burst".to_owned(),
        generation,
        source_member_manifest_digest: manifest
            .generation_member_manifest_digest(
                &dates.iter().map(|date| *date as usize).collect::<Vec<_>>(),
                "burst",
                generation,
            )
            .unwrap(),
        source_date_indices: dates,
        source_model_version_digest: model_version,
        source_model_receipt_digest: model_receipt,
        normalized_config_digest: [1; 32],
        kernel_digest: [2; 32],
        mask_digest: [3; 32],
        blocks: vec![CovarianceGenerationBlockIdentity {
            block_id: u64::from(generation) + 1,
            block_sha256: [4; 32],
        }],
        sealed: generation == 0,
    };
    let registry = CovarianceGenerationRegistry {
        schema_version: COVARIANCE_GENERATION_REGISTRY_SCHEMA_VERSION,
        full_source_manifest_digest: manifest.digest(),
        generations: vec![generation(0, vec![0, 1]), generation(1, vec![2])],
    };
    let grid = CovarianceOperatorGrid {
        row_start: 0,
        col_start: 0,
        rows: 3,
        cols: 4,
        stride_y: 1,
        stride_x: 1,
    };
    let mut bundle = manifest
        .resolver_bundle(
            &[0, 1, 2],
            &registry,
            "burst",
            (0, 0),
            (3, 4),
            grid,
            &options,
            model_version,
            None,
        )
        .unwrap();
    assert_eq!(bundle.identity().source_manifest_digest, manifest.digest());
    let block_0 = SequentialReplayBlock {
        id: GlobalBlockId::new(1),
        generation: 0,
        real_date_start: GlobalDateId::new(0),
        num_real_dates: 2,
        carried_parent_ids: Vec::new(),
        phase_dimension: 1,
    };
    let block_1 = SequentialReplayBlock {
        id: GlobalBlockId::new(2),
        generation: 1,
        real_date_start: GlobalDateId::new(2),
        num_real_dates: 1,
        carried_parent_ids: vec![block_0.id],
        phase_dimension: 1,
    };
    assert_eq!(
        bundle
            .identity_for_block(&block_0)
            .unwrap()
            .source_manifest_digest,
        registry.generations[0].source_member_manifest_digest
    );
    let source_0 = bundle.resolve_source(&block_0, 0).unwrap();
    let receipt_0 = bundle.factor_receipt_digest(&source_0).unwrap();
    assert_ne!(receipt_0, [0; 32]);
    let source_1 = bundle.resolve_source(&block_1, 0).unwrap();
    let receipt_1 = bundle.factor_receipt_digest(&source_1).unwrap();
    assert_ne!(receipt_1, [0; 32]);
    assert_ne!(receipt_0, receipt_1);
    assert!(matches!(
        bundle.factor_receipt_digest(&source_0),
        Err(SequentialReplayError::Provider(
            ReplayStatus::SourceIdentityMismatch,
            _
        ))
    ));
    assert_eq!(bundle.active_generation(), Some(1));
    assert!(bundle.metrics().peak_cached_bytes > 0);

    let mut stale = registry.clone();
    stale.generations[1].source_member_manifest_digest[0] ^= 0xff;
    let error = match manifest.resolver_bundle(
        &[0, 1, 2],
        &stale,
        "burst",
        (0, 0),
        (3, 4),
        grid,
        &options,
        model_version,
        None,
    ) {
        Ok(_) => panic!("stale generation manifest must fail"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("generation source manifest"),
        "{error}"
    );
    for path in member_paths {
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_dir(root);
}

#[test]
#[allow(clippy::too_many_lines)]
fn resumable_capture_persists_empirical_receipts_and_leaves_only_tail_open() {
    let _hdf5 = HDF5_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let root =
        std::env::temp_dir().join(format!("dolphin-cslc-nrt-capture-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let member_paths = paths(&root, 3);
    for (date, path) in member_paths.iter().enumerate() {
        write_member(path, date, false);
    }
    let manifest =
        CslcCovarianceManifest::capture(InputType::OperaCslc, "/data", &member_paths).unwrap();
    let options = EmpiricalSourceFactorOptions {
        half_window: HalfWindow { y: 1, x: 1 },
        shrinkage_alpha: 0.2,
        relative_diagonal_floor: 1e-8,
    };
    let model_version = sequential_source_model_identity_digest(
        CSLC_COVARIANCE_SOURCE_PROVIDER,
        CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
        CSLC_COVARIANCE_SOURCE_MODEL,
        CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
    );
    let model_receipt = *empirical_factor_config(&options).unwrap().config_digest();
    let grid = CovarianceOperatorGrid {
        row_start: 0,
        col_start: 0,
        rows: 3,
        cols: 4,
        stride_y: 1,
        stride_x: 1,
    };
    let cfg = SequentialConfig {
        ministack_size: 2,
        max_num_compressed: 4,
        half_window: HalfWindow { y: 1, x: 1 },
        strides: Strides { y: 1, x: 1 },
        use_evd: true,
        beta: 0.0,
        zero_correlation_threshold: 0.0,
        output_reference_idx: 0,
        compressed_slc_plan: CompressedSlcPlan::AlwaysFirst,
        compute_crlb: false,
        compute_closure_phase: false,
        compute_average_coherence: false,
        shp_method: ShpMethod::Rect,
        shp_alpha: 0.001,
    };
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "burst".to_owned(),
        source_manifest_digest: manifest.digest(),
        source_model_version_digest: model_version,
        native_grid: grid,
        output_grid: grid,
        owned_output_grid: grid,
        branch_tolerance: 1e-10,
    };
    let revision = SequentialCovarianceRevision {
        full_source_manifest_digest: manifest.digest(),
        prior_full_source_manifest_digest: None,
        generation_source_manifest_digests: vec![
            manifest
                .generation_member_manifest_digest(&[0, 1], "burst", 0)
                .unwrap(),
            manifest
                .generation_member_manifest_digest(&[2], "burst", 1)
                .unwrap(),
        ],
        source_model_receipt_digest: model_receipt,
    };
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let plan =
        plan_sequential_covariance_capture(3, (3, 4), &cfg, &engine, &request, &revision).unwrap();
    let mut bundle = manifest
        .resolver_bundle(
            &[0, 1, 2],
            plan.generation_registry(),
            "burst",
            (0, 0),
            (3, 4),
            grid,
            &options,
            model_version,
            None,
        )
        .unwrap();
    let stack = Array3::from_shape_fn((3, 3, 4), |(date, row, col)| {
        let value = Cf32::new(
            1.0 + date as f32 * 0.2 + row as f32 * 0.01,
            0.5 + col as f32 * 0.03,
        );
        Cf64::new(f64::from(value.re), f64::from(value.im))
    });
    let mut blocks = Vec::new();
    let (_, state) = run_sequential_resumable_with_covariance_capture_and_source_factors(
        stack.view(),
        &cfg,
        &engine,
        &request,
        &revision,
        &mut bundle,
        |block| {
            blocks.push(block);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(blocks.len(), 2);
    assert!(blocks.iter().all(|block| block
        .source_factor_digests
        .chunks_exact(32)
        .all(|receipt| receipt.iter().any(|byte| *byte != 0))));
    assert!(state.generation_registry().generations[0].sealed);
    assert!(!state.generation_registry().generations[1].sealed);
    assert_eq!(state.open_real_slcs().dim().0, 1);
    assert_eq!(bundle.active_generation(), Some(1));

    let parent_path = root.join("parent_operator.h5");
    let metadata = CovarianceOperatorMetadata {
        normalized_config_digest: encode(sequential_replay_config_digest(&cfg)),
        kernel_digest: encode(sequential_replay_kernel_digest()),
        source: SourceReplayIdentity {
            manifest_digest: Some(encode(manifest.digest())),
            provider: Some(CSLC_COVARIANCE_SOURCE_PROVIDER.to_owned()),
            provider_version: Some(CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION.to_owned()),
            model: Some(CSLC_COVARIANCE_SOURCE_MODEL.to_owned()),
            model_version: Some(CSLC_COVARIANCE_SOURCE_MODEL_VERSION.to_owned()),
            model_version_digest: Some(encode(model_version)),
            model_receipt_digest: Some(encode(model_receipt)),
        },
        replay_status: CovarianceReplayStatus::Replayable,
        stitched_status: StitchedCovarianceStatus::NotStitched,
        calibration_status: CovarianceCalibrationStatus::Uncalibrated,
        downstream_inference_status: DownstreamInferenceStatus::BlockedPendingIssue54And53,
        ..CovarianceOperatorMetadata::default()
    };
    let mut writer = CovarianceOperatorWriter::create_with_generation_registry(
        &parent_path,
        &metadata,
        state.operator_plan(),
        state.generation_registry(),
    )
    .unwrap();
    for block in &blocks {
        writer.write_block(block).unwrap();
    }
    state.seal_writer_generations(&mut writer).unwrap();
    writer.finish().unwrap();

    let fourth = root.join("source_3.h5");
    write_member(&fourth, 3, false);
    let mut extended_paths = member_paths.clone();
    extended_paths.push(fourth);
    let extended = manifest.verify_prefix_and_extend(&extended_paths).unwrap();
    let extended_request = SequentialCovarianceCaptureRequest {
        source_manifest_digest: extended.digest(),
        ..request.clone()
    };
    let extended_revision = SequentialCovarianceRevision {
        full_source_manifest_digest: extended.digest(),
        prior_full_source_manifest_digest: Some(manifest.digest()),
        generation_source_manifest_digests: vec![
            extended
                .generation_member_manifest_digest(&[0, 1], "burst", 0)
                .unwrap(),
            extended
                .generation_member_manifest_digest(&[2, 3], "burst", 1)
                .unwrap(),
        ],
        source_model_receipt_digest: model_receipt,
    };
    let extended_plan = plan_sequential_covariance_capture(
        4,
        (3, 4),
        &cfg,
        &engine,
        &extended_request,
        &extended_revision,
    )
    .unwrap();
    let mut extended_bundle = extended
        .resolver_bundle(
            &[0, 1, 2, 3],
            extended_plan.generation_registry(),
            "burst",
            (0, 0),
            (3, 4),
            grid,
            &options,
            model_version,
            None,
        )
        .unwrap();
    let parent_reader =
        dolphin_io::CovarianceOperatorBlockReader::open(&parent_path, u64::MAX).unwrap();
    let copy_cap = parent_reader
        .validate_sealed_blocks(u64::MAX)
        .unwrap()
        .maximum_block_read_bytes;
    let fourth_stack = Array3::from_shape_fn((1, 3, 4), |(_, row, col)| {
        let value = Cf32::new(1.0 + 3.0 * 0.2 + row as f32 * 0.01, 0.5 + col as f32 * 0.03);
        Cf64::new(f64::from(value.re), f64::from(value.im))
    });
    let incremental_blocks = RefCell::new(Vec::new());
    let (_, extended_state) = update_sequential_with_covariance_capture_and_source_factors(
        &state,
        fourth_stack.view(),
        &cfg,
        &engine,
        &extended_request,
        &extended_revision,
        &parent_reader,
        copy_cap,
        &mut extended_bundle,
        |block_id| {
            incremental_blocks.borrow_mut().push(
                parent_reader
                    .read_block_with_receipt(block_id, copy_cap)?
                    .block,
            );
            Ok(())
        },
        |block| {
            incremental_blocks.borrow_mut().push(block);
            Ok(())
        },
    )
    .unwrap();
    let mut incremental_blocks = incremental_blocks.into_inner();
    incremental_blocks.sort_by_key(|block| block.generation);
    assert_eq!(incremental_blocks.len(), 2);
    assert_eq!(incremental_blocks[0], blocks[0]);
    assert!(incremental_blocks[1]
        .source_factor_digests
        .chunks_exact(32)
        .all(|receipt| receipt.iter().any(|byte| *byte != 0)));
    assert!(extended_state
        .generation_registry()
        .generations
        .iter()
        .all(|identity| identity.sealed));
    assert_eq!(extended_state.open_real_slcs().dim().0, 0);
    assert_eq!(extended_bundle.active_generation(), Some(1));
    drop(parent_reader);
    let _ = std::fs::remove_file(parent_path);
    for path in extended_paths {
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_dir(root);
}

#[test]
#[allow(clippy::too_many_lines)]
fn verified_prefix_extension_preserves_generation_receipt_and_binds_resolver() {
    let _hdf5 = HDF5_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let root = std::env::temp_dir().join(format!("dolphin-cslc-nrt-prefix-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let all_paths = paths(&root, 3);
    for (date, path) in all_paths.iter().enumerate() {
        write_member(path, date, false);
    }
    let parent =
        CslcCovarianceManifest::capture(InputType::OperaCslc, "/data", &all_paths[..2]).unwrap();
    let sealed_receipt = parent
        .generation_member_manifest_digest(&[0, 1], "burst", 0)
        .unwrap();
    let extended = parent.verify_prefix_and_extend(&all_paths).unwrap();
    assert_ne!(extended.digest(), parent.digest());
    assert_eq!(
        extended
            .generation_member_manifest_digest(&[0, 1], "burst", 0)
            .unwrap(),
        sealed_receipt
    );

    let model_version = sequential_source_model_identity_digest(
        CSLC_COVARIANCE_SOURCE_PROVIDER,
        CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
        CSLC_COVARIANCE_SOURCE_MODEL,
        CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
    );
    let resolver = extended
        .resolver_for_generation(
            &[0, 1],
            &[0, 1],
            "burst",
            0,
            (0, 0),
            (3, 4),
            CovarianceOperatorGrid {
                row_start: 0,
                col_start: 0,
                rows: 3,
                cols: 4,
                stride_y: 1,
                stride_x: 1,
            },
            &EmpiricalSourceFactorOptions {
                half_window: HalfWindow { y: 1, x: 1 },
                shrinkage_alpha: 0.2,
                relative_diagonal_floor: 1e-8,
            },
            model_version,
            None,
        )
        .unwrap();
    assert_eq!(resolver.generation_manifest_digest(), sealed_receipt);
    assert_eq!(resolver.full_revision_manifest_digest(), extended.digest());
    let tail_receipt = extended
        .generation_member_manifest_digest(&[2], "burst", 1)
        .unwrap();
    let mut tail = extended
        .resolver_for_generation(
            &[0, 1, 2],
            &[2],
            "burst",
            1,
            (0, 0),
            (3, 4),
            CovarianceOperatorGrid {
                row_start: 0,
                col_start: 0,
                rows: 3,
                cols: 4,
                stride_y: 1,
                stride_x: 1,
            },
            &EmpiricalSourceFactorOptions {
                half_window: HalfWindow { y: 1, x: 1 },
                shrinkage_alpha: 0.2,
                relative_diagonal_floor: 1e-8,
            },
            model_version,
            None,
        )
        .unwrap();
    assert_eq!(tail.generation_manifest_digest(), tail_receipt);
    assert_eq!(tail.full_revision_manifest_digest(), extended.digest());
    let wrong_generation = SequentialReplayBlock {
        id: GlobalBlockId::new(1),
        generation: 0,
        real_date_start: GlobalDateId::new(0),
        num_real_dates: 2,
        carried_parent_ids: Vec::new(),
        phase_dimension: 1,
    };
    assert!(matches!(
        tail.resolve_source(&wrong_generation, 0),
        Err(SequentialReplayError::Provider(
            ReplayStatus::SourceIdentityMismatch,
            _
        ))
    ));
    let wrong_dates = SequentialReplayBlock {
        generation: 1,
        ..wrong_generation.clone()
    };
    assert!(matches!(
        tail.resolve_source(&wrong_dates, 0),
        Err(SequentialReplayError::Provider(
            ReplayStatus::SourceIdentityMismatch,
            _
        ))
    ));
    assert_eq!(tail.metrics().member_window_reads, 0);
    let matching_generation = SequentialReplayBlock {
        id: GlobalBlockId::new(2),
        generation: 1,
        real_date_start: GlobalDateId::new(2),
        num_real_dates: 1,
        carried_parent_ids: Vec::new(),
        phase_dimension: 0,
    };
    tail.resolve_source(&matching_generation, 0).unwrap();
    assert!(tail.metrics().member_window_reads > 0);
    for path in all_paths {
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_dir(root);
}

#[test]
fn prefix_drift_fails_before_missing_extension_is_read() {
    let _hdf5 = HDF5_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let root = std::env::temp_dir().join(format!("dolphin-cslc-nrt-drift-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let prior_paths = paths(&root, 2);
    for (date, path) in prior_paths.iter().enumerate() {
        write_member(path, date, false);
    }
    let parent =
        CslcCovarianceManifest::capture(InputType::OperaCslc, "/data", &prior_paths).unwrap();
    let missing_extension = root.join("missing_extension.h5");
    let mut extended_paths = prior_paths.clone();
    extended_paths.push(missing_extension);
    write_member(&prior_paths[0], 0, true);
    let error = parent
        .verify_prefix_and_extend(&extended_paths)
        .unwrap_err()
        .to_string();
    assert!(error.contains("CSLC member"), "{error}");
    assert!(!error.contains("missing_extension"), "{error}");

    let reordered = vec![prior_paths[1].clone(), prior_paths[0].clone()];
    let error = parent
        .verify_prefix_and_extend(&reordered)
        .unwrap_err()
        .to_string();
    assert!(error.contains("ordered prefix"), "{error}");
    let error = parent
        .verify_prefix_and_extend(&prior_paths[..1])
        .unwrap_err()
        .to_string();
    assert!(error.contains("ordered prefix"), "{error}");
    let substituted = vec![root.join("substituted.h5"), prior_paths[1].clone()];
    let error = parent
        .verify_prefix_and_extend(&substituted)
        .unwrap_err()
        .to_string();
    assert!(error.contains("ordered prefix"), "{error}");
    for path in prior_paths {
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_dir(root);
}

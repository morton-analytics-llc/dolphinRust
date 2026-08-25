use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use dolphin_core::config::{CompressedSlcPlan, ComputeBackend, ShpMethod};
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_io::{
    covariance_source_model_identity_digest, CovarianceCalibrationStatus, CovarianceOperatorBlock,
    CovarianceOperatorBlockReader, CovarianceOperatorGrid, CovarianceOperatorMetadata,
    CovarianceOperatorWriter, CovarianceReplayStatus, DownstreamInferenceStatus,
    SourceReplayIdentity, StitchedCovarianceStatus,
};
use dolphin_phaselink::ComputeEngine;
use dolphin_workflows::{
    plan_sequential_covariance_capture, run_sequential_resumable_masked_with_covariance_capture,
    run_sequential_resumable_with_covariance_capture, sequential_replay_config_digest,
    sequential_replay_kernel_digest, update_sequential_masked_with_covariance_capture,
    update_sequential_with_covariance_capture, SequentialConfig, SequentialCovarianceRevision,
    SequentialCovarianceState, SequentialReplayError,
};
use ndarray::{s, Array2, Array3};
use sha2::{Digest, Sha256};

static HDF5_LOCK: Mutex<()> = Mutex::new(());

fn config() -> SequentialConfig {
    SequentialConfig {
        ministack_size: 5,
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
    }
}

fn stack(dates: usize) -> Array3<Cf64> {
    Array3::from_shape_fn((dates, 4, 4), |(date, row, col)| {
        Cf64::from_polar(
            1.0 + 0.03 * (row + col) as f64,
            0.17 * date as f64 + 0.02 * row as f64 - 0.01 * col as f64,
        )
    })
}

fn request(full_digest: [u8; 32]) -> dolphin_workflows::SequentialCovarianceCaptureRequest {
    dolphin_workflows::SequentialCovarianceCaptureRequest {
        burst_id: "nrt-burst".to_owned(),
        source_manifest_digest: full_digest,
        source_model_version_digest: covariance_source_model_identity_digest(
            "nrt-fixture",
            "1",
            "fixture-proper-complex",
            "1",
        ),
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
            rows: 4,
            cols: 4,
            stride_y: 1,
            stride_x: 1,
        },
        owned_output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 4,
            cols: 4,
            stride_y: 1,
            stride_x: 1,
        },
        branch_tolerance: 1e-10,
    }
}

fn generation_digest(generation: u32, start: usize, stop: usize) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"nrt-covariance-generation-fixture-v1");
    digest.update(generation.to_le_bytes());
    digest.update((start as u64).to_le_bytes());
    digest.update((stop as u64).to_le_bytes());
    digest.finalize().into()
}

fn revision_9() -> SequentialCovarianceRevision {
    revision(9, None)
}

fn revision_13(prior: Option<[u8; 32]>) -> SequentialCovarianceRevision {
    revision(13, prior)
}

fn revision(count: usize, prior: Option<[u8; 32]>) -> SequentialCovarianceRevision {
    SequentialCovarianceRevision {
        full_source_manifest_digest: [count as u8; 32],
        prior_full_source_manifest_digest: prior,
        generation_source_manifest_digests: (0..count.div_ceil(5))
            .map(|generation| {
                let start = generation * 5;
                generation_digest(generation as u32, start, (start + 5).min(count))
            })
            .collect(),
        source_model_receipt_digest: [0x41; 32],
    }
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

fn metadata(
    cfg: &SequentialConfig,
    capture: &dolphin_workflows::SequentialCovarianceCaptureRequest,
    revision: &SequentialCovarianceRevision,
) -> CovarianceOperatorMetadata {
    CovarianceOperatorMetadata {
        normalized_config_digest: encode(sequential_replay_config_digest(cfg)),
        kernel_digest: encode(sequential_replay_kernel_digest()),
        source: SourceReplayIdentity {
            manifest_digest: Some(encode(revision.full_source_manifest_digest)),
            provider: Some("nrt-fixture".to_owned()),
            provider_version: Some("1".to_owned()),
            model: Some("fixture-proper-complex".to_owned()),
            model_version: Some("1".to_owned()),
            model_version_digest: Some(encode(capture.source_model_version_digest)),
            model_receipt_digest: Some(encode(revision.source_model_receipt_digest)),
        },
        replay_status: CovarianceReplayStatus::SourceModelUnavailable,
        stitched_status: StitchedCovarianceStatus::NotStitched,
        calibration_status: CovarianceCalibrationStatus::Uncalibrated,
        downstream_inference_status: DownstreamInferenceStatus::BlockedPendingIssue54And53,
        ..CovarianceOperatorMetadata::default()
    }
}

fn artifact_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "dolphin-nrt-covariance-{label}-{}.h5",
        std::process::id()
    ))
}

fn write_artifact(
    path: &Path,
    cfg: &SequentialConfig,
    request: &dolphin_workflows::SequentialCovarianceCaptureRequest,
    revision: &SequentialCovarianceRevision,
    state: &SequentialCovarianceState,
    blocks: &[CovarianceOperatorBlock],
) {
    let _ = std::fs::remove_file(path);
    let mut writer = CovarianceOperatorWriter::create_with_generation_registry(
        path,
        &metadata(cfg, request, revision),
        state.operator_plan(),
        state.generation_registry(),
    )
    .unwrap();
    for block in blocks {
        writer.write_block(block).unwrap();
    }
    writer.finish().unwrap();
}

#[test]
#[allow(clippy::too_many_lines)]
fn nrt_capture_reuses_only_sealed_generations_and_matches_fresh_bytes() {
    let _guard = HDF5_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let cfg = config();
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let values = stack(13);
    let initial_revision = revision_9();
    let initial_request = request(initial_revision.full_source_manifest_digest);
    let initial_plan = plan_sequential_covariance_capture(
        9,
        (4, 4),
        &cfg,
        &engine,
        &initial_request,
        &initial_revision,
    )
    .unwrap();
    assert!(initial_plan
        .generation_registry()
        .generations
        .iter()
        .all(|identity| !identity.sealed
            && identity
                .blocks
                .iter()
                .all(|block| block.block_sha256 == [0; 32])));
    let parent_path = artifact_path("parent");
    let _ = std::fs::remove_file(&parent_path);
    let mut parent_writer = CovarianceOperatorWriter::create_with_generation_registry(
        &parent_path,
        &metadata(&cfg, &initial_request, &initial_revision),
        initial_plan.operator_plan(),
        initial_plan.generation_registry(),
    )
    .unwrap();
    let mut initial_blocks = Vec::new();
    let (_, state_9) = run_sequential_resumable_with_covariance_capture(
        values.slice(s![..9, .., ..]),
        &cfg,
        &engine,
        &initial_request,
        &initial_revision,
        |block| {
            parent_writer.write_block(&block)?;
            initial_blocks.push(block);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(state_9.open_real_slcs().dim().0, 4);
    assert!(state_9.generation_registry().generations[0].sealed);
    assert!(!state_9.generation_registry().generations[1].sealed);
    assert_ne!(
        initial_blocks[0].source_manifest_digest,
        initial_blocks[1].source_manifest_digest
    );
    state_9.seal_writer_generations(&mut parent_writer).unwrap();
    parent_writer.finish().unwrap();
    let parent = CovarianceOperatorBlockReader::open(&parent_path, u64::MAX).unwrap();
    let exact_cap = parent
        .validate_sealed_blocks(u64::MAX)
        .unwrap()
        .maximum_block_read_bytes;
    let extended_revision = revision_13(Some(initial_revision.full_source_manifest_digest));
    let extended_request = request(extended_revision.full_source_manifest_digest);
    let mut copies = 0;
    let mut captures = 0;
    let error = match update_sequential_with_covariance_capture(
        &state_9,
        values.slice(s![9.., .., ..]),
        &cfg,
        &engine,
        &extended_request,
        &extended_revision,
        &parent,
        exact_cap - 1,
        |_| {
            copies += 1;
            Ok(())
        },
        |_| {
            captures += 1;
            Ok(())
        },
    ) {
        Err(error) => error,
        Ok(_) => panic!("one-byte-below sealed copy cap must fail"),
    };
    assert!(error.to_string().contains("byte cap"), "{error}");
    assert_eq!((copies, captures), (0, 0));

    let incremental_blocks = RefCell::new(Vec::new());
    let (incremental_output, state_13) = update_sequential_with_covariance_capture(
        &state_9,
        values.slice(s![9.., .., ..]),
        &cfg,
        &engine,
        &extended_request,
        &extended_revision,
        &parent,
        exact_cap,
        |block_id| {
            incremental_blocks
                .borrow_mut()
                .push(parent.read_block_with_receipt(block_id, exact_cap)?.block);
            Ok(())
        },
        |block| {
            incremental_blocks.borrow_mut().push(block);
            Ok(())
        },
    )
    .unwrap();
    let mut fresh_blocks = Vec::new();
    let fresh_revision = revision_13(None);
    let (fresh_output, fresh_state) = run_sequential_resumable_with_covariance_capture(
        values.view(),
        &cfg,
        &engine,
        &extended_request,
        &fresh_revision,
        |block| {
            fresh_blocks.push(block);
            Ok(())
        },
    )
    .unwrap();
    let mut incremental_blocks = incremental_blocks.into_inner();
    incremental_blocks.sort_by_key(|block| block.generation);
    fresh_blocks.sort_by_key(|block| block.generation);
    assert_eq!(incremental_blocks.len(), fresh_blocks.len());
    for (incremental, fresh) in incremental_blocks.iter().zip(&fresh_blocks) {
        assert_eq!(incremental.generation, fresh.generation);
        assert_eq!(
            dolphin_io::covariance::covariance_operator_block_sha256(incremental),
            dolphin_io::covariance::covariance_operator_block_sha256(fresh),
            "generation {} logical bytes",
            incremental.generation
        );
    }
    assert_eq!(
        state_13.generation_registry(),
        fresh_state.generation_registry()
    );
    assert_eq!(incremental_output.cpx_phase, fresh_output.cpx_phase);
    assert!(incremental_output
        .compressed_slcs
        .iter()
        .zip(&fresh_output.compressed_slcs)
        .flat_map(|(left, right)| left.iter().zip(right))
        .all(|(left, right)| left.re.to_bits() == right.re.to_bits()
            && left.im.to_bits() == right.im.to_bits()));
    let _ = std::fs::remove_file(parent_path);
}

#[test]
fn one_at_a_time_extensions_complete_the_same_registry_as_fresh_capture() {
    let _guard = HDF5_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let cfg = config();
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let values = stack(13);
    let initial_revision = revision(9, None);
    let initial_request = request(initial_revision.full_source_manifest_digest);
    let mut blocks = Vec::new();
    let (_, mut state) = run_sequential_resumable_with_covariance_capture(
        values.slice(s![..9, .., ..]),
        &cfg,
        &engine,
        &initial_request,
        &initial_revision,
        |block| {
            blocks.push(block);
            Ok(())
        },
    )
    .unwrap();
    let mut prior_revision = initial_revision;
    let mut parent_path = artifact_path("stream-9");
    write_artifact(
        &parent_path,
        &cfg,
        &initial_request,
        &prior_revision,
        &state,
        &blocks,
    );
    for count in 10..=13 {
        let parent = CovarianceOperatorBlockReader::open(&parent_path, u64::MAX).unwrap();
        let cap = parent
            .validate_sealed_blocks(u64::MAX)
            .unwrap()
            .maximum_block_read_bytes;
        let next_revision = revision(count, Some(prior_revision.full_source_manifest_digest));
        let next_request = request(next_revision.full_source_manifest_digest);
        let next_blocks = RefCell::new(Vec::new());
        let (_, next_state) = update_sequential_with_covariance_capture(
            &state,
            values.slice(s![count - 1..count, .., ..]),
            &cfg,
            &engine,
            &next_request,
            &next_revision,
            &parent,
            cap,
            |block_id| {
                next_blocks
                    .borrow_mut()
                    .push(parent.read_block_with_receipt(block_id, cap)?.block);
                Ok(())
            },
            |block| {
                next_blocks.borrow_mut().push(block);
                Ok(())
            },
        )
        .unwrap();
        drop(parent);
        let next_path = artifact_path(&format!("stream-{count}"));
        write_artifact(
            &next_path,
            &cfg,
            &next_request,
            &next_revision,
            &next_state,
            &next_blocks.into_inner(),
        );
        let _ = std::fs::remove_file(parent_path);
        parent_path = next_path;
        prior_revision = next_revision;
        state = next_state;
    }
    let fresh_revision = revision(13, None);
    let fresh_request = request(fresh_revision.full_source_manifest_digest);
    let (_, fresh_state) = run_sequential_resumable_with_covariance_capture(
        values.view(),
        &cfg,
        &engine,
        &fresh_request,
        &fresh_revision,
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(
        state.generation_registry(),
        fresh_state.generation_registry()
    );
    let _ = std::fs::remove_file(parent_path);
}

#[test]
#[allow(clippy::too_many_lines)]
fn update_identity_tamper_fails_before_copy_or_capture() {
    let _guard = HDF5_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let cfg = config();
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let values = stack(13);
    let initial_revision = revision_9();
    let initial_request = request(initial_revision.full_source_manifest_digest);
    let mut blocks = Vec::new();
    let (_, state) = run_sequential_resumable_with_covariance_capture(
        values.slice(s![..9, .., ..]),
        &cfg,
        &engine,
        &initial_request,
        &initial_revision,
        |block| {
            blocks.push(block);
            Ok(())
        },
    )
    .unwrap();
    let parent_path = artifact_path("tamper-parent");
    write_artifact(
        &parent_path,
        &cfg,
        &initial_request,
        &initial_revision,
        &state,
        &blocks,
    );
    let parent = CovarianceOperatorBlockReader::open(&parent_path, u64::MAX).unwrap();
    let request_13 = request([13; 32]);
    let base = revision_13(Some([9; 32]));
    for label in ["prefix", "model", "generation"] {
        let mut changed = base.clone();
        match label {
            "prefix" => changed.prior_full_source_manifest_digest = Some([8; 32]),
            "model" => changed.source_model_receipt_digest = [8; 32],
            "generation" => changed.generation_source_manifest_digests[0] = [8; 32],
            _ => unreachable!(),
        }
        let touched = Cell::new(0);
        assert!(update_sequential_with_covariance_capture(
            &state,
            values.slice(s![9.., .., ..]),
            &cfg,
            &engine,
            &request_13,
            &changed,
            &parent,
            u64::MAX,
            |_| {
                touched.set(touched.get() + 1);
                Ok(())
            },
            |_| {
                touched.set(touched.get() + 1);
                Ok(())
            },
        )
        .is_err());
        assert_eq!(touched.get(), 0, "{label}");
    }

    let mut changed_cfg = cfg;
    changed_cfg.beta = 0.25;
    let touched = Cell::new(0);
    assert!(update_sequential_with_covariance_capture(
        &state,
        values.slice(s![9.., .., ..]),
        &changed_cfg,
        &engine,
        &request_13,
        &base,
        &parent,
        u64::MAX,
        |_| {
            touched.set(touched.get() + 1);
            Ok(())
        },
        |_| {
            touched.set(touched.get() + 1);
            Ok(())
        },
    )
    .is_err());
    assert_eq!(touched.get(), 0);

    let valid = Array2::from_elem((4, 4), true);
    let mut masked_blocks = Vec::new();
    let (_, masked_state) = run_sequential_resumable_masked_with_covariance_capture(
        values.slice(s![..9, .., ..]),
        valid.view(),
        &cfg,
        &engine,
        &initial_request,
        &initial_revision,
        |block| {
            masked_blocks.push(block);
            Ok(())
        },
    )
    .unwrap();
    let masked_path = artifact_path("masked-parent");
    write_artifact(
        &masked_path,
        &cfg,
        &initial_request,
        &initial_revision,
        &masked_state,
        &masked_blocks,
    );
    let masked_parent = CovarianceOperatorBlockReader::open(&masked_path, u64::MAX).unwrap();
    let touched = Cell::new(0);
    assert!(update_sequential_with_covariance_capture(
        &state,
        values.slice(s![9.., .., ..]),
        &cfg,
        &engine,
        &request_13,
        &base,
        &masked_parent,
        u64::MAX,
        |_| {
            touched.set(touched.get() + 1);
            Ok(())
        },
        |_| {
            touched.set(touched.get() + 1);
            Ok(())
        },
    )
    .is_err());
    assert_eq!(touched.get(), 0, "mismatched parent registry");
    let mut changed_mask = valid;
    changed_mask[(0, 0)] = false;
    let touched = Cell::new(0);
    assert!(update_sequential_masked_with_covariance_capture(
        &masked_state,
        values.slice(s![9.., .., ..]),
        changed_mask.view(),
        &cfg,
        &engine,
        &request_13,
        &base,
        &masked_parent,
        u64::MAX,
        |_| {
            touched.set(touched.get() + 1);
            Ok(())
        },
        |_| {
            touched.set(touched.get() + 1);
            Ok(())
        },
    )
    .is_err());
    assert_eq!(touched.get(), 0);
    let _ = std::fs::remove_file(parent_path);
    let _ = std::fs::remove_file(masked_path);
}

#[test]
fn callback_errors_remain_fail_closed() {
    let cfg = config();
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let values = stack(9);
    let revision = revision_9();
    let request = request(revision.full_source_manifest_digest);
    let error = match run_sequential_resumable_with_covariance_capture(
        values.view(),
        &cfg,
        &engine,
        &request,
        &revision,
        |_| {
            Err(SequentialReplayError::Execution(
                "fixture sink rejected block",
            ))
        },
    ) {
        Err(error) => error,
        Ok(_) => panic!("rejected sink must fail capture"),
    };
    assert!(error.to_string().contains("fixture sink"));
}

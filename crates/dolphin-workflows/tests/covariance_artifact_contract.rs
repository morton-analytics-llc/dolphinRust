use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use dolphin_core::Cf64;
use dolphin_io::{
    covariance_source_model_identity_digest, read_covariance_operator_metadata_with_byte_cap,
    CovarianceBurstPlan, CovarianceCalibrationStatus, CovarianceEstimatorBranch,
    CovarianceOperatorBlock, CovarianceOperatorGrid, CovarianceOperatorMetadata,
    CovarianceOperatorPlan, CovarianceOperatorStatus, CovarianceOperatorWriteReceipt,
    CovarianceOperatorWriter, CovariancePhaseComponent, CovariancePhaseComponentKind,
    CovarianceRectSupport, CovarianceReplayStatus, CovarianceSupportOrdering, CovarianceTilePlan,
    DownstreamInferenceStatus, SourceReplayIdentity, StitchedCovarianceStatus,
    COVARIANCE_OPERATOR_METHOD, COVARIANCE_OPERATOR_METHOD_VERSION,
    COVARIANCE_OPERATOR_SCHEMA_VERSION,
};
use dolphin_workflows::{
    admit_covariance_artifact_disk, admit_covariance_artifact_disk_with_identity_index,
    covariance_artifact_disk_bytes, finalize_covariance_artifact,
    read_covariance_artifact_manifest, read_covariance_artifact_manifest_with_byte_cap,
    CovarianceArtifactManifest, CovarianceArtifactTransaction, COVARIANCE_OPERATOR_FILENAME,
    COVARIANCE_OPERATOR_MANIFEST_FILENAME,
};

static NEXT_PATH: AtomicU64 = AtomicU64::new(0);
static HDF5_LOCK: Mutex<()> = Mutex::new(());

fn temporary_directory() -> PathBuf {
    let sequence = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "dolphin-covariance-artifact-contract-{}-{sequence}",
        std::process::id()
    ))
}

fn metadata() -> CovarianceOperatorMetadata {
    let model_version_digest = covariance_source_model_identity_digest(
        "fixture-provider",
        "1",
        "proper-complex-tangent",
        "1",
    );
    CovarianceOperatorMetadata {
        schema_version: COVARIANCE_OPERATOR_SCHEMA_VERSION,
        method: COVARIANCE_OPERATOR_METHOD.to_owned(),
        method_version: COVARIANCE_OPERATOR_METHOD_VERSION,
        crate_version: env!("CARGO_PKG_VERSION").to_owned(),
        producer_commit: Some("da9116c".to_owned()),
        gauge_date_index: 0,
        normalized_config_digest: format!("sha256:{}", "11".repeat(32)),
        kernel_digest: format!("sha256:{}", "22".repeat(32)),
        source: SourceReplayIdentity {
            manifest_digest: Some(format!("sha256:{}", "33".repeat(32))),
            provider: Some("fixture-provider".to_owned()),
            provider_version: Some("1".to_owned()),
            model: Some("proper-complex-tangent".to_owned()),
            model_version: Some("1".to_owned()),
            model_version_digest: Some(format!(
                "sha256:{}",
                model_version_digest
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            )),
            model_receipt_digest: Some(format!("sha256:{}", "44".repeat(32))),
        },
        replay_status: CovarianceReplayStatus::SourceModelUnavailable,
        stitched_status: StitchedCovarianceStatus::NotStitched,
        calibration_status: CovarianceCalibrationStatus::Uncalibrated,
        downstream_inference_status: DownstreamInferenceStatus::BlockedPendingIssue54And53,
    }
}

fn block() -> CovarianceOperatorBlock {
    CovarianceOperatorBlock {
        burst_id: "fixture-burst".to_owned(),
        source_manifest_digest: [0x33; 32],
        source_model_version_digest: covariance_source_model_identity_digest(
            "fixture-provider",
            "1",
            "proper-complex-tangent",
            "1",
        ),
        block_id: 0,
        generation: 0,
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
        rect_support: CovarianceRectSupport {
            half_window_rows: 0,
            half_window_cols: 0,
            ordering: CovarianceSupportOrdering::RowMajorInwardClampV1,
        },
        branch_tolerance: 1e-6,
        reference_date_index: 0,
        source_date_indices: vec![0, 1],
        ordered_date_indices: vec![0, 1],
        source_ids: vec![100],
        source_content_digests: vec![7; 32],
        source_factor_digests: vec![8; 32],
        phase_node_ids: vec![200],
        compressed_node_ids: vec![300],
        carry_parent_ids: Vec::new(),
        nearest_output_map: vec![0],
        phase_components: vec![
            CovariancePhaseComponent {
                kind: CovariancePhaseComponentKind::GaugeDate,
                id: 0,
            },
            CovariancePhaseComponent {
                kind: CovariancePhaseComponentKind::RetainedDate,
                id: 1,
            },
        ],
        phase_angles: vec![0.0, 0.2],
        compressed_raster: vec![Cf64::new(1.0, 0.2)],
        compressed_status: vec![CovarianceOperatorStatus::Valid],
        projection_accumulator: vec![Cf64::new(2.0, 0.3)],
        mean_amplitude: vec![1.0],
        support_bits_per_output: 1,
        support_bits: vec![1],
        native_validity_bits: vec![1],
        estimator_branch: CovarianceEstimatorBranch::Evd,
        selected_eigenvalue: vec![1.0],
        eigen_gap: vec![0.5],
        status: vec![CovarianceOperatorStatus::Valid],
    }
}

fn high_stride_block() -> CovarianceOperatorBlock {
    let native_side = 256_usize;
    let native_area = native_side * native_side;
    let mut source_content_digests = vec![0_u8; native_area * 32];
    for digest in source_content_digests.chunks_exact_mut(32) {
        digest[0] = 7;
    }
    CovarianceOperatorBlock {
        burst_id: "high-stride-burst".to_owned(),
        source_manifest_digest: [0x33; 32],
        source_model_version_digest: covariance_source_model_identity_digest(
            "fixture-provider",
            "1",
            "proper-complex-tangent",
            "1",
        ),
        block_id: 0,
        generation: 0,
        native_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: native_side as u32,
            cols: native_side as u32,
            stride_y: 1,
            stride_x: 1,
        },
        output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: native_side as u32,
            stride_x: native_side as u32,
        },
        owned_output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: native_side as u32,
            stride_x: native_side as u32,
        },
        rect_support: CovarianceRectSupport {
            half_window_rows: 0,
            half_window_cols: 0,
            ordering: CovarianceSupportOrdering::RowMajorInwardClampV1,
        },
        branch_tolerance: 1e-6,
        reference_date_index: 0,
        source_date_indices: vec![0, 1],
        ordered_date_indices: vec![0, 1],
        source_ids: (1..=native_area as u64).collect(),
        source_content_digests,
        source_factor_digests: vec![8; native_area * 32],
        phase_node_ids: vec![1_000_000],
        compressed_node_ids: (2_000_000..2_000_000 + native_area as u64).collect(),
        carry_parent_ids: Vec::new(),
        nearest_output_map: vec![0; native_area],
        phase_components: vec![
            CovariancePhaseComponent {
                kind: CovariancePhaseComponentKind::GaugeDate,
                id: 0,
            },
            CovariancePhaseComponent {
                kind: CovariancePhaseComponentKind::RetainedDate,
                id: 1,
            },
        ],
        phase_angles: vec![0.0, 0.2],
        compressed_raster: vec![Cf64::new(1.0, 0.2); native_area],
        compressed_status: vec![CovarianceOperatorStatus::Valid; native_area],
        projection_accumulator: vec![Cf64::new(2.0, 0.3); native_area],
        mean_amplitude: vec![1.0; native_area],
        support_bits_per_output: 1,
        support_bits: vec![1],
        native_validity_bits: vec![0xff; native_area / 8],
        estimator_branch: CovarianceEstimatorBranch::Evd,
        selected_eigenvalue: vec![1.0],
        eigen_gap: vec![0.5],
        status: vec![CovarianceOperatorStatus::Valid],
    }
}

fn plan(block: &CovarianceOperatorBlock) -> CovarianceOperatorPlan {
    CovarianceOperatorPlan {
        source_manifest_digest: block.source_manifest_digest,
        source_model_version_digest: block.source_model_version_digest,
        bursts: vec![CovarianceBurstPlan {
            burst_id: block.burst_id.clone(),
            source_dates_by_generation: vec![block.source_date_indices.clone()],
            tiles: vec![CovarianceTilePlan {
                native_grid: block.native_grid,
                output_grid: block.output_grid,
                owned_output_grid: block.owned_output_grid,
            }],
        }],
    }
}

fn write_complete_operator(
    path: &std::path::Path,
    operator_metadata: &CovarianceOperatorMetadata,
) -> CovarianceOperatorWriteReceipt {
    let block = block();
    let mut writer =
        CovarianceOperatorWriter::create(path, operator_metadata, &plan(&block)).unwrap();
    writer.write_block(&block).unwrap();
    writer.finish().unwrap()
}

#[test]
fn c52_19_manifest_is_the_last_commit_marker_and_binds_hdf5_bytes() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let directory = temporary_directory();
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = CovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join("phase_covariance_operator.h5.scratch");

    let expected_metadata = metadata();
    let write_receipt = write_complete_operator(&scratch, &expected_metadata);
    let disk = admit_covariance_artifact_disk_with_identity_index(
        10 * 1024 * 1024,
        write_receipt.peak_identity_index_disk_bytes,
        u64::MAX,
    )
    .unwrap();
    let receipt = finalize_covariance_artifact(
        &transaction,
        &scratch,
        &expected_metadata,
        disk,
        &write_receipt,
    )
    .unwrap();
    drop(transaction);

    assert_eq!(receipt.hdf5_file, COVARIANCE_OPERATOR_FILENAME);
    assert!(receipt.hdf5_bytes > 0);
    assert_eq!(receipt.hdf5_sha256.len(), 64);
    assert_eq!(receipt.method, COVARIANCE_OPERATOR_METHOD);
    assert_eq!(receipt.calibration_status, "uncalibrated");
    assert_eq!(
        receipt.downstream_inference_status,
        "blocked_pending_issue_54_and_53"
    );
    assert!(!scratch.exists());
    assert!(directory.join(COVARIANCE_OPERATOR_FILENAME).exists());
    assert!(directory
        .join(COVARIANCE_OPERATOR_MANIFEST_FILENAME)
        .exists());

    let parsed = read_covariance_artifact_manifest(&directory).unwrap();
    assert_eq!(parsed, receipt);
    let encoded = std::fs::read(directory.join(COVARIANCE_OPERATOR_MANIFEST_FILENAME)).unwrap();
    let direct: CovarianceArtifactManifest = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(direct, receipt);

    let mut tampered = direct;
    tampered.normalized_config_digest = format!("sha256:{}", "55".repeat(32));
    std::fs::write(
        directory.join(COVARIANCE_OPERATOR_MANIFEST_FILENAME),
        serde_json::to_vec_pretty(&tampered).unwrap(),
    )
    .unwrap();
    let error = read_covariance_artifact_manifest(&directory)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("does not match the committed HDF5"),
        "{error}"
    );

    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn finalization_rejects_metadata_that_differs_from_the_hdf5_header() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let directory = temporary_directory();
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = CovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join("phase_covariance_operator.h5.scratch");
    let write_receipt = write_complete_operator(&scratch, &metadata());

    let mut mismatched = metadata();
    mismatched.kernel_digest = format!("sha256:{}", "66".repeat(32));
    let disk = admit_covariance_artifact_disk_with_identity_index(
        10 * 1024 * 1024,
        write_receipt.peak_identity_index_disk_bytes,
        u64::MAX,
    )
    .unwrap();
    let error =
        finalize_covariance_artifact(&transaction, &scratch, &mismatched, disk, &write_receipt)
            .unwrap_err()
            .to_string();
    assert!(
        error.contains("does not match finalization metadata"),
        "{error}"
    );
    assert!(scratch.exists());
    assert!(!directory
        .join(COVARIANCE_OPERATOR_MANIFEST_FILENAME)
        .exists());

    drop(transaction);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn finalization_rejects_hdf5_changed_after_the_writer_sealed_it() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let directory = temporary_directory();
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = CovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join("phase_covariance_operator.h5.scratch");
    let expected_metadata = metadata();
    let write_receipt = write_complete_operator(&scratch, &expected_metadata);

    let file = hdf5::File::open_rw(&scratch).unwrap();
    file.dataset("blocks/00000000000000000000/selected_eigenvalue")
        .unwrap()
        .write_raw(&[2.0_f64])
        .unwrap();
    file.flush().unwrap();
    drop(file);

    let disk = admit_covariance_artifact_disk_with_identity_index(
        10 * 1024 * 1024,
        write_receipt.peak_identity_index_disk_bytes,
        u64::MAX,
    )
    .unwrap();
    let error = finalize_covariance_artifact(
        &transaction,
        &scratch,
        &expected_metadata,
        disk,
        &write_receipt,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("changed after the writer sealed"), "{error}");
    assert!(!directory
        .join(COVARIANCE_OPERATOR_MANIFEST_FILENAME)
        .exists());
    drop(transaction);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn c52_18_disk_preflight_counts_final_scratch_and_twenty_five_percent_margin() {
    assert_eq!(covariance_artifact_disk_bytes(1_000).unwrap(), 2_500);
    assert_eq!(covariance_artifact_disk_bytes(1).unwrap(), 3);
    assert!(covariance_artifact_disk_bytes(u64::MAX).is_err());
    assert_eq!(
        admit_covariance_artifact_disk(1_000, 2_500)
            .unwrap()
            .required_free_bytes,
        2_500
    );
    assert!(admit_covariance_artifact_disk(1_000, 2_499).is_err());
}

#[test]
fn topology_validation_cap_is_independent_of_admitted_hdf5_bytes() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let directory = temporary_directory();
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = CovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join("phase_covariance_operator.h5.scratch");
    let expected_metadata = metadata();
    let block = high_stride_block();
    let mut writer =
        CovarianceOperatorWriter::create(&scratch, &expected_metadata, &plan(&block)).unwrap();
    writer.write_block(&block).unwrap();
    let write_receipt = writer.finish().unwrap();
    let hdf5_bytes = std::fs::metadata(&scratch).unwrap().len();
    assert!(write_receipt.metadata_validation_bytes < hdf5_bytes);
    assert!(read_covariance_operator_metadata_with_byte_cap(&scratch, hdf5_bytes).is_err());

    let disk = admit_covariance_artifact_disk_with_identity_index(
        hdf5_bytes,
        write_receipt.peak_identity_index_disk_bytes,
        u64::MAX,
    )
    .unwrap();
    finalize_covariance_artifact(
        &transaction,
        &scratch,
        &expected_metadata,
        disk,
        &write_receipt,
    )
    .unwrap();
    drop(transaction);
    read_covariance_artifact_manifest_with_byte_cap(&directory, 64 * 1024).unwrap();

    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn manifest_read_is_capped_before_json_allocation() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let directory = temporary_directory();
    std::fs::create_dir_all(&directory).unwrap();
    let manifest_path = directory.join(COVARIANCE_OPERATOR_MANIFEST_FILENAME);
    std::fs::write(&manifest_path, vec![b' '; 4096]).unwrap();
    let error = read_covariance_artifact_manifest_with_byte_cap(&directory, 1024)
        .unwrap_err()
        .to_string();
    assert!(error.contains("exceeds byte cap 1024"), "{error}");
    std::fs::remove_dir_all(directory).unwrap();
}

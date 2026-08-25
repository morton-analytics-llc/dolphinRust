use dolphin_io::{
    spatial_reference_calibration_scope_digest, write_spatial_reference_covariance,
    CovarianceOperatorGrid, SpatialReferenceCalibrationScope, SpatialReferenceCovarianceBlock,
    SpatialReferenceCovarianceMetadata, SpatialReferenceCovarianceStatus,
    SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE, SPATIAL_REFERENCE_COVARIANCE_METHOD,
    SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
};
use dolphin_workflows::spatial_covariance_artifact::{
    spatial_reference_covariance_code_digest, spatial_reference_covariance_design_digest,
    spatial_reference_covariance_preregistration_digest,
    SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RECEIPT_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RESULT_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_DESIGN_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME, SPATIAL_REFERENCE_COVARIANCE_LOCK_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_MANIFEST_SCRATCH_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_METHOD_MANIFEST_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_PREREGISTRATION_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_PRODUCER_BINARY_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_RESOURCE_RECEIPT_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_REVIEW_RECEIPT_FILENAME,
};
use dolphin_workflows::{
    finalize_spatial_reference_covariance_artifact,
    read_spatial_reference_covariance_artifact_manifest,
    SpatialReferenceCovarianceArtifactTransaction, SPATIAL_REFERENCE_COVARIANCE_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME,
};
use sha2::{Digest, Sha256};

fn digest(byte: u8) -> String {
    format!("sha256:{}", format!("{byte:02x}").repeat(32))
}

fn metadata() -> SpatialReferenceCovarianceMetadata {
    SpatialReferenceCovarianceMetadata {
        schema_version: SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
        method: SPATIAL_REFERENCE_COVARIANCE_METHOD.to_owned(),
        method_version: 1,
        crate_version: env!("CARGO_PKG_VERSION").to_owned(),
        producer_commit: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
        burst_id: "T078-165482-IW1".to_owned(),
        crs: "EPSG:32611".to_owned(),
        units: "radians".to_owned(),
        geotransform: [500_000.0, 30.0, 0.0, 4_200_000.0, 0.0, -30.0],
        full_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: 1,
            stride_x: 1,
        },
        reference_row: 0,
        reference_col: 0,
        gauge_date_index: 0,
        ordered_date_indices: vec![0, 1],
        acquisition_days: vec![0.0, 12.0],
        mask_digest: digest(0x11),
        source_replay_digest: digest(0x22),
        l2_map_digest: digest(0x33),
        reference_signature_digest: digest(0x44),
        approximation_receipt_digest: digest(0x55),
        resource_receipt_digest: digest(0x66),
        review_receipt_digest: String::new(),
        method_manifest_digest: String::new(),
        calibration_scope_digest: String::new(),
        source_model_digest: digest(0x67),
        effective_looks_digest: digest(0x68),
        support_method: "rect".to_owned(),
        support_digest: digest(0x69),
        correction_order_digest: digest(0x6a),
        unwrap_branch_digest: digest(0x6b),
        burst_ownership_digest: digest(0x6c),
        source_burst_ids: vec!["T078-165482-IW1".to_owned()],
        reference_source_burst_index: 0,
        calibration_scope: SpatialReferenceCalibrationScope::Uncalibrated,
        maximum_block_bytes: 1024,
    }
}

fn block() -> SpatialReferenceCovarianceBlock {
    SpatialReferenceCovarianceBlock {
        block_id: 1,
        target_grid: metadata().full_grid,
        maximum_rank: 1,
        rank_by_target: vec![1],
        status: vec![SpatialReferenceCovarianceStatus::Valid],
        source_burst_index_by_target: vec![0],
        difference_factor: vec![0.0, 1.0],
        approximation_error_bound: vec![SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE],
        effective_looks_fraction: vec![0.75],
        support_union_count: vec![9],
        effective_looks_receipt: vec![0x71; 32],
        resource_high_water_bytes: vec![2048],
        source_factor_digest: digest(0x77),
    }
}

fn calibrated_block() -> SpatialReferenceCovarianceBlock {
    let mut value = block();
    value.approximation_error_bound = vec![0.01];
    value
}

fn calibrated_metadata() -> SpatialReferenceCovarianceMetadata {
    let mut value = metadata();
    value.calibration_scope = SpatialReferenceCalibrationScope::CalibratedScopeMatch;
    value.review_receipt_digest = digest(0x81);
    value.method_manifest_digest = digest(0x82);
    value.calibration_scope_digest = spatial_reference_calibration_scope_digest(&value);
    value
}

fn content_digest(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

#[allow(clippy::too_many_lines)]
fn write_promotion_evidence(
    directory: &std::path::Path,
    value: &mut SpatialReferenceCovarianceMetadata,
) {
    value.calibration_scope = SpatialReferenceCalibrationScope::CalibratedScopeMatch;
    value.review_receipt_digest.clear();
    value.method_manifest_digest.clear();
    std::fs::write(
        directory.join(SPATIAL_REFERENCE_COVARIANCE_PREREGISTRATION_FILENAME),
        include_bytes!("../../../validation/spatial_covariance_preregistration.json"),
    )
    .unwrap();
    std::fs::write(
        directory.join(SPATIAL_REFERENCE_COVARIANCE_DESIGN_FILENAME),
        include_bytes!("../../../md/design/spatial-reference-covariance.md"),
    )
    .unwrap();
    let binary_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_PRODUCER_BINARY_FILENAME);
    let current_exe = std::env::current_exe().unwrap();
    if std::fs::hard_link(&current_exe, &binary_path).is_err() {
        std::fs::copy(&current_exe, &binary_path).unwrap();
    }
    let producer_binary_sha256 = content_digest(&std::fs::read(&binary_path).unwrap());
    let code_sha256 = spatial_reference_covariance_code_digest();
    let preregistration_sha256 = spatial_reference_covariance_preregistration_digest();
    let design_sha256 = spatial_reference_covariance_design_digest();
    let result = serde_json::json!({
        "schema_version": 1,
        "method": value.method,
        "method_version": value.method_version,
        "crate_version": value.crate_version,
        "producer_commit": value.producer_commit.as_deref().unwrap(),
        "status": "passed",
        "scope": {
            "burst_id": value.burst_id,
            "crs": value.crs,
            "units": value.units,
            "geotransform": value.geotransform,
            "acquisition_days": value.acquisition_days,
            "grid_row_start": value.full_grid.row_start,
            "grid_col_start": value.full_grid.col_start,
            "grid_rows": value.full_grid.rows,
            "grid_cols": value.full_grid.cols,
            "grid_stride_y": value.full_grid.stride_y,
            "grid_stride_x": value.full_grid.stride_x,
            "reference_row": value.reference_row,
            "reference_col": value.reference_col,
            "gauge_date_index": value.gauge_date_index,
            "ordered_date_indices": value.ordered_date_indices,
            "mask_digest": value.mask_digest,
            "source_replay_digest": value.source_replay_digest,
            "l2_map_digest": value.l2_map_digest,
            "reference_signature_digest": value.reference_signature_digest,
            "source_model_digest": value.source_model_digest,
            "effective_looks_digest": value.effective_looks_digest,
            "support_method": value.support_method,
            "support_digest": value.support_digest,
            "correction_order_digest": value.correction_order_digest,
            "unwrap_branch_digest": value.unwrap_branch_digest,
            "burst_ownership_digest": value.burst_ownership_digest,
            "source_burst_ids": value.source_burst_ids,
            "reference_source_burst_index": value.reference_source_burst_index,
            "maximum_block_bytes": value.maximum_block_bytes,
        },
        "evaluated_cases": 5000,
        "maximum_absolute_error": 1.0e-12,
        "tolerance": 1.0e-10,
    });
    let result_bytes = serde_json::to_vec_pretty(&result).unwrap();
    std::fs::write(
        directory.join(SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RESULT_FILENAME),
        &result_bytes,
    )
    .unwrap();
    let result_sha256 = content_digest(&result_bytes);
    let approximation = serde_json::json!({
        "schema_version": 1,
        "method": value.method,
        "method_version": value.method_version,
        "crate_version": value.crate_version,
        "producer_commit": value.producer_commit.as_deref().unwrap(),
        "status": "passed",
        "code_sha256": code_sha256,
        "producer_binary_file": SPATIAL_REFERENCE_COVARIANCE_PRODUCER_BINARY_FILENAME,
        "producer_binary_sha256": producer_binary_sha256,
        "preregistration_file": SPATIAL_REFERENCE_COVARIANCE_PREREGISTRATION_FILENAME,
        "preregistration_sha256": preregistration_sha256,
        "design_file": SPATIAL_REFERENCE_COVARIANCE_DESIGN_FILENAME,
        "design_sha256": design_sha256,
        "result_file": SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RESULT_FILENAME,
        "result_sha256": result_sha256,
    });
    let approximation_bytes = serde_json::to_vec_pretty(&approximation).unwrap();
    std::fs::write(
        directory.join(SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RECEIPT_FILENAME),
        &approximation_bytes,
    )
    .unwrap();
    value.approximation_receipt_digest = content_digest(&approximation_bytes);
    let resource = serde_json::json!({
        "schema_version": 1,
        "method": value.method,
        "method_version": value.method_version,
        "crate_version": value.crate_version,
        "producer_commit": value.producer_commit.as_deref().unwrap(),
        "status": "passed",
        "code_sha256": code_sha256,
        "producer_binary_sha256": producer_binary_sha256,
        "preregistration_sha256": preregistration_sha256,
        "design_sha256": design_sha256,
        "result_sha256": result_sha256,
        "peak_resident_set_bytes": 1,
        "wall_micros": 1,
        "maximum_block_bytes": value.maximum_block_bytes,
    });
    let resource_bytes = serde_json::to_vec_pretty(&resource).unwrap();
    std::fs::write(
        directory.join(SPATIAL_REFERENCE_COVARIANCE_RESOURCE_RECEIPT_FILENAME),
        &resource_bytes,
    )
    .unwrap();
    value.resource_receipt_digest = content_digest(&resource_bytes);
    value.calibration_scope_digest = spatial_reference_calibration_scope_digest(value);
    let review = serde_json::json!({
        "schema_version": 1,
        "method": value.method,
        "method_version": value.method_version,
        "crate_version": value.crate_version,
        "producer_commit": value.producer_commit.as_deref().unwrap(),
        "reviewer": "independent-reviewer",
        "review_status": "approved_no_unresolved_findings",
        "unresolved_findings": 0,
        "code_sha256": code_sha256,
        "producer_binary_sha256": producer_binary_sha256,
        "preregistration_sha256": preregistration_sha256,
        "design_sha256": design_sha256,
        "result_sha256": result_sha256,
        "approximation_receipt_digest": value.approximation_receipt_digest,
        "resource_receipt_digest": value.resource_receipt_digest,
        "calibration_scope_digest": value.calibration_scope_digest,
    });
    let review_bytes = serde_json::to_vec_pretty(&review).unwrap();
    std::fs::write(
        directory.join(SPATIAL_REFERENCE_COVARIANCE_REVIEW_RECEIPT_FILENAME),
        &review_bytes,
    )
    .unwrap();
    value.review_receipt_digest = content_digest(&review_bytes);
    let method_manifest = serde_json::json!({
        "schema_version": 1,
        "method": value.method,
        "method_version": value.method_version,
        "crate_version": value.crate_version,
        "producer_commit": value.producer_commit.as_deref().unwrap(),
        "manifest_status": "reviewed_scope_match",
        "code_sha256": code_sha256,
        "producer_binary_sha256": producer_binary_sha256,
        "preregistration_sha256": preregistration_sha256,
        "design_sha256": design_sha256,
        "result_sha256": result_sha256,
        "approximation_receipt_digest": value.approximation_receipt_digest,
        "resource_receipt_digest": value.resource_receipt_digest,
        "review_receipt_digest": value.review_receipt_digest,
        "calibration_scope_digest": value.calibration_scope_digest,
    });
    let method_bytes = serde_json::to_vec_pretty(&method_manifest).unwrap();
    std::fs::write(
        directory.join(SPATIAL_REFERENCE_COVARIANCE_METHOD_MANIFEST_FILENAME),
        &method_bytes,
    )
    .unwrap();
    value.method_manifest_digest = content_digest(&method_bytes);
}

#[test]
fn product_boundary_uses_frozen_final_scratch_and_lock_names() {
    assert_eq!(
        SPATIAL_REFERENCE_COVARIANCE_FILENAME,
        "referenced_displacement_covariance_factor.h5"
    );
    assert_eq!(
        SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME,
        "referenced_displacement_covariance_provenance.json"
    );
    assert_eq!(
        SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME,
        "referenced_displacement_covariance_factor.h5.scratch"
    );
    assert_eq!(
        SPATIAL_REFERENCE_COVARIANCE_MANIFEST_SCRATCH_FILENAME,
        "referenced_displacement_covariance_provenance.json.scratch"
    );
    assert_eq!(
        SPATIAL_REFERENCE_COVARIANCE_LOCK_FILENAME,
        "referenced_displacement_covariance.capture.lock"
    );
    assert_eq!(
        SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RECEIPT_FILENAME,
        "referenced_displacement_covariance_approximation_receipt.json"
    );
    assert_eq!(
        SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RESULT_FILENAME,
        "referenced_displacement_covariance_approximation_result.json"
    );
    assert_eq!(
        SPATIAL_REFERENCE_COVARIANCE_RESOURCE_RECEIPT_FILENAME,
        "referenced_displacement_covariance_resource_receipt.json"
    );
}

#[test]
fn manifest_is_written_last_and_binds_hdf5_and_scope() {
    let directory = std::env::temp_dir().join(format!(
        "dolphin_spatial_covariance_artifact_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
    let receipt = write_spatial_reference_covariance(&scratch, &metadata(), &[block()]).unwrap();
    assert!(!directory
        .join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME)
        .exists());
    let manifest = finalize_spatial_reference_covariance_artifact(
        &transaction,
        &scratch,
        &metadata(),
        &receipt,
    )
    .unwrap();
    assert_eq!(manifest.hdf5_sha256, receipt.hdf5_sha256);
    assert_eq!(manifest.reference_signature_digest, digest(0x44));
    assert_eq!(manifest.schema_version, 3);
    assert_eq!(manifest.geotransform, metadata().geotransform);
    assert_eq!(manifest.acquisition_days, metadata().acquisition_days);
    assert_eq!(
        manifest.effective_looks_fraction_dataset,
        "blocks/{block_id:020}/effective_looks_fraction"
    );
    assert_eq!(
        manifest.support_union_count_dataset,
        "blocks/{block_id:020}/support_union_count"
    );
    assert_eq!(
        manifest.effective_looks_receipt_dataset,
        "blocks/{block_id:020}/effective_looks_receipt"
    );
    assert_eq!(
        manifest.resource_high_water_bytes_dataset,
        "blocks/{block_id:020}/resource_high_water_bytes"
    );
    assert_eq!(manifest.calibration_scope, "uncalibrated");
    assert!(directory
        .join(SPATIAL_REFERENCE_COVARIANCE_FILENAME)
        .exists());
    drop(transaction);
    assert_eq!(
        read_spatial_reference_covariance_artifact_manifest(&directory).unwrap(),
        manifest
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn tampered_hdf5_or_manifest_identity_fails_closed() {
    let directory = std::env::temp_dir().join(format!(
        "dolphin_spatial_covariance_tamper_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
    let receipt = write_spatial_reference_covariance(&scratch, &metadata(), &[block()]).unwrap();
    finalize_spatial_reference_covariance_artifact(&transaction, &scratch, &metadata(), &receipt)
        .unwrap();
    drop(transaction);

    let manifest_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME);
    let original = std::fs::read(&manifest_path).unwrap();
    let mut manifest: serde_json::Value = serde_json::from_slice(&original).unwrap();
    manifest["reference_signature_digest"] = serde_json::Value::String(digest(0x99));
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    assert!(read_spatial_reference_covariance_artifact_manifest(&directory).is_err());

    let mut manifest: serde_json::Value = serde_json::from_slice(&original).unwrap();
    manifest["geotransform"][0] = serde_json::json!(500_030.0);
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    assert!(read_spatial_reference_covariance_artifact_manifest(&directory).is_err());

    let mut manifest: serde_json::Value = serde_json::from_slice(&original).unwrap();
    manifest["acquisition_days"][1] = serde_json::json!(13.0);
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    assert!(read_spatial_reference_covariance_artifact_manifest(&directory).is_err());

    std::fs::write(&manifest_path, original).unwrap();
    let hdf5 = directory.join(SPATIAL_REFERENCE_COVARIANCE_FILENAME);
    let mut bytes = std::fs::read(&hdf5).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&hdf5, bytes).unwrap();
    assert!(read_spatial_reference_covariance_artifact_manifest(&directory).is_err());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn noncanonical_scratch_and_active_writer_are_unreadable() {
    let directory = std::env::temp_dir().join(format!(
        "dolphin_spatial_covariance_lock_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join("replacement.h5.scratch");
    let receipt = write_spatial_reference_covariance(&scratch, &metadata(), &[block()]).unwrap();
    let error = finalize_spatial_reference_covariance_artifact(
        &transaction,
        &scratch,
        &metadata(),
        &receipt,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("canonical transaction path"), "{error}");
    let error = read_spatial_reference_covariance_artifact_manifest(&directory)
        .unwrap_err()
        .to_string();
    assert!(error.contains("being replaced"), "{error}");
    drop(transaction);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn valid_pair_is_immutable_and_stale_scratch_is_recovered() {
    let directory = std::env::temp_dir().join(format!(
        "dolphin_spatial_covariance_immutable_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let first_transaction =
        SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
    let receipt = write_spatial_reference_covariance(&scratch, &metadata(), &[block()]).unwrap();
    let first = finalize_spatial_reference_covariance_artifact(
        &first_transaction,
        &scratch,
        &metadata(),
        &receipt,
    )
    .unwrap();
    drop(first_transaction);
    let final_hdf5 = directory.join(SPATIAL_REFERENCE_COVARIANCE_FILENAME);
    let final_manifest = directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME);
    let original_hdf5 = std::fs::read(&final_hdf5).unwrap();
    let original_manifest = std::fs::read(&final_manifest).unwrap();

    std::fs::write(
        directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME),
        b"stale",
    )
    .unwrap();
    std::fs::write(
        directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_SCRATCH_FILENAME),
        b"stale",
    )
    .unwrap();
    let replacement_transaction =
        SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    assert!(!directory
        .join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME)
        .exists());
    assert!(!directory
        .join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_SCRATCH_FILENAME)
        .exists());
    let replacement_receipt =
        write_spatial_reference_covariance(&scratch, &metadata(), &[block()]).unwrap();
    let error = finalize_spatial_reference_covariance_artifact(
        &replacement_transaction,
        &scratch,
        &metadata(),
        &replacement_receipt,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("already exists"), "{error}");
    assert_eq!(std::fs::read(&final_hdf5).unwrap(), original_hdf5);
    assert_eq!(std::fs::read(&final_manifest).unwrap(), original_manifest);
    drop(replacement_transaction);

    let recovery = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    assert!(!scratch.exists());
    drop(recovery);
    assert_eq!(
        read_spatial_reference_covariance_artifact_manifest(&directory).unwrap(),
        first
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn incomplete_and_corrupt_final_pairs_are_unreadable_then_recoverable() {
    let directory = std::env::temp_dir().join(format!(
        "dolphin_spatial_covariance_recovery_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let orphan = directory.join(SPATIAL_REFERENCE_COVARIANCE_FILENAME);
    write_spatial_reference_covariance(&orphan, &metadata(), &[block()]).unwrap();
    std::fs::write(
        directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_SCRATCH_FILENAME),
        b"interrupted manifest commit",
    )
    .unwrap();
    assert!(read_spatial_reference_covariance_artifact_manifest(&directory).is_err());
    let transaction = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    assert!(!orphan.exists());
    assert!(directory
        .join(format!(
            "{SPATIAL_REFERENCE_COVARIANCE_FILENAME}.quarantine.0"
        ))
        .exists());
    assert!(!directory
        .join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_SCRATCH_FILENAME)
        .exists());
    let scratch = directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
    let receipt = write_spatial_reference_covariance(&scratch, &metadata(), &[block()]).unwrap();
    finalize_spatial_reference_covariance_artifact(&transaction, &scratch, &metadata(), &receipt)
        .unwrap();
    drop(transaction);

    let final_manifest = directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME);
    std::fs::write(&final_manifest, b"corrupt").unwrap();
    assert!(read_spatial_reference_covariance_artifact_manifest(&directory).is_err());
    let recovery = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    assert!(!directory
        .join(SPATIAL_REFERENCE_COVARIANCE_FILENAME)
        .exists());
    assert!(!final_manifest.exists());
    assert!(directory
        .join(format!(
            "{SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME}.quarantine.1"
        ))
        .exists());
    assert!(directory
        .join(format!(
            "{SPATIAL_REFERENCE_COVARIANCE_FILENAME}.quarantine.1"
        ))
        .exists());
    drop(recovery);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn calibration_requires_complete_nonzero_evidence_and_never_follows_file_presence() {
    let directory = std::env::temp_dir().join(format!(
        "dolphin_spatial_covariance_calibration_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
    let uncalibrated = metadata();
    let receipt = write_spatial_reference_covariance(&scratch, &uncalibrated, &[block()]).unwrap();
    let manifest = finalize_spatial_reference_covariance_artifact(
        &transaction,
        &scratch,
        &uncalibrated,
        &receipt,
    )
    .unwrap();
    assert_eq!(manifest.calibration_scope, "uncalibrated");
    drop(transaction);
    std::fs::remove_dir_all(&directory).unwrap();
    std::fs::create_dir_all(&directory).unwrap();

    let transaction = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    let missing_evidence = calibrated_metadata();
    let receipt =
        write_spatial_reference_covariance(&scratch, &missing_evidence, &[calibrated_block()])
            .unwrap();
    let error = finalize_spatial_reference_covariance_artifact(
        &transaction,
        &scratch,
        &missing_evidence,
        &receipt,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("preregistration"), "{error}");
    assert!(!directory
        .join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME)
        .exists());
    drop(transaction);
    std::fs::remove_dir_all(&directory).unwrap();
    std::fs::create_dir_all(&directory).unwrap();

    let transaction = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    let mut calibrated = metadata();
    write_promotion_evidence(&directory, &mut calibrated);
    let receipt =
        write_spatial_reference_covariance(&scratch, &calibrated, &[calibrated_block()]).unwrap();
    let manifest = finalize_spatial_reference_covariance_artifact(
        &transaction,
        &scratch,
        &calibrated,
        &receipt,
    )
    .unwrap();
    assert_eq!(manifest.calibration_scope, "calibrated_scope_match");
    drop(transaction);
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn promotion_evidence_hashes_and_bindings_must_match_actual_files() {
    let directory = std::env::temp_dir().join(format!(
        "dolphin_spatial_covariance_evidence_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let scratch = directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
    let transaction = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    let mut stale = metadata();
    write_promotion_evidence(&directory, &mut stale);
    let method_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_METHOD_MANIFEST_FILENAME);
    let mut method: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&method_path).unwrap()).unwrap();
    method["resource_receipt_digest"] = serde_json::Value::String(digest(0xee));
    let method_bytes = serde_json::to_vec_pretty(&method).unwrap();
    std::fs::write(&method_path, &method_bytes).unwrap();
    stale.method_manifest_digest = content_digest(&method_bytes);
    let receipt =
        write_spatial_reference_covariance(&scratch, &stale, &[calibrated_block()]).unwrap();
    let error =
        finalize_spatial_reference_covariance_artifact(&transaction, &scratch, &stale, &receipt)
            .unwrap_err()
            .to_string();
    assert!(error.contains("resource receipt"), "{error}");
    drop(transaction);
    std::fs::remove_dir_all(&directory).unwrap();

    std::fs::create_dir_all(&directory).unwrap();
    let transaction = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    let mut calibrated = metadata();
    write_promotion_evidence(&directory, &mut calibrated);
    let receipt =
        write_spatial_reference_covariance(&scratch, &calibrated, &[calibrated_block()]).unwrap();
    finalize_spatial_reference_covariance_artifact(&transaction, &scratch, &calibrated, &receipt)
        .unwrap();
    drop(transaction);
    let review_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_REVIEW_RECEIPT_FILENAME);
    std::fs::write(&review_path, b"{}").unwrap();
    assert!(read_spatial_reference_covariance_artifact_manifest(&directory).is_err());
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn tampered_design_numeric_result_and_resource_receipt_fail_closed() {
    for (index, evidence_file) in [
        SPATIAL_REFERENCE_COVARIANCE_DESIGN_FILENAME,
        SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RESULT_FILENAME,
        SPATIAL_REFERENCE_COVARIANCE_RESOURCE_RECEIPT_FILENAME,
    ]
    .iter()
    .enumerate()
    {
        let directory = std::env::temp_dir().join(format!(
            "dolphin_spatial_covariance_bound_evidence_{}_{}",
            std::process::id(),
            index
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let transaction =
            SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
        let scratch = directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
        let mut calibrated = metadata();
        write_promotion_evidence(&directory, &mut calibrated);
        std::fs::write(directory.join(evidence_file), b"{}\n").unwrap();
        let receipt =
            write_spatial_reference_covariance(&scratch, &calibrated, &[calibrated_block()])
                .unwrap();
        let error = finalize_spatial_reference_covariance_artifact(
            &transaction,
            &scratch,
            &calibrated,
            &receipt,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("design")
                || error.contains("approximation result")
                || error.contains("resource receipt"),
            "{evidence_file}: {error}"
        );
        drop(transaction);
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn self_consistent_wrapper_hashes_cannot_replace_current_code_design_or_binary() {
    for (index, identity) in ["code_sha256", "design_sha256", "producer_binary_sha256"]
        .iter()
        .enumerate()
    {
        let directory = std::env::temp_dir().join(format!(
            "dolphin_spatial_covariance_arbitrary_identity_{}_{}",
            std::process::id(),
            index
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let transaction =
            SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
        let scratch = directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
        let mut calibrated = metadata();
        write_promotion_evidence(&directory, &mut calibrated);
        let mut arbitrary = digest(0xfa);
        if *identity == "design_sha256" {
            let replacement_design = b"arbitrary replacement design\n";
            std::fs::write(
                directory.join(SPATIAL_REFERENCE_COVARIANCE_DESIGN_FILENAME),
                replacement_design,
            )
            .unwrap();
            arbitrary = content_digest(replacement_design);
        } else if *identity == "producer_binary_sha256" {
            let replacement_binary = b"arbitrary replacement binary\n";
            std::fs::remove_file(
                directory.join(SPATIAL_REFERENCE_COVARIANCE_PRODUCER_BINARY_FILENAME),
            )
            .unwrap();
            std::fs::write(
                directory.join(SPATIAL_REFERENCE_COVARIANCE_PRODUCER_BINARY_FILENAME),
                replacement_binary,
            )
            .unwrap();
            arbitrary = content_digest(replacement_binary);
        }
        let approximation_path =
            directory.join(SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RECEIPT_FILENAME);
        let mut approximation: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&approximation_path).unwrap()).unwrap();
        approximation[*identity] = serde_json::Value::String(arbitrary.clone());
        let approximation_bytes = serde_json::to_vec_pretty(&approximation).unwrap();
        std::fs::write(&approximation_path, &approximation_bytes).unwrap();
        calibrated.approximation_receipt_digest = content_digest(&approximation_bytes);

        let resource_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_RESOURCE_RECEIPT_FILENAME);
        let mut resource: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&resource_path).unwrap()).unwrap();
        resource[*identity] = serde_json::Value::String(arbitrary.clone());
        let resource_bytes = serde_json::to_vec_pretty(&resource).unwrap();
        std::fs::write(&resource_path, &resource_bytes).unwrap();
        calibrated.resource_receipt_digest = content_digest(&resource_bytes);
        calibrated.calibration_scope_digest =
            spatial_reference_calibration_scope_digest(&calibrated);

        let review_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_REVIEW_RECEIPT_FILENAME);
        let mut review: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&review_path).unwrap()).unwrap();
        review[*identity] = serde_json::Value::String(arbitrary.clone());
        review["approximation_receipt_digest"] =
            serde_json::Value::String(calibrated.approximation_receipt_digest.clone());
        review["resource_receipt_digest"] =
            serde_json::Value::String(calibrated.resource_receipt_digest.clone());
        review["calibration_scope_digest"] =
            serde_json::Value::String(calibrated.calibration_scope_digest.clone());
        let review_bytes = serde_json::to_vec_pretty(&review).unwrap();
        std::fs::write(&review_path, &review_bytes).unwrap();
        calibrated.review_receipt_digest = content_digest(&review_bytes);

        let method_path = directory.join(SPATIAL_REFERENCE_COVARIANCE_METHOD_MANIFEST_FILENAME);
        let mut method: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&method_path).unwrap()).unwrap();
        method[*identity] = serde_json::Value::String(arbitrary);
        method["approximation_receipt_digest"] =
            serde_json::Value::String(calibrated.approximation_receipt_digest.clone());
        method["resource_receipt_digest"] =
            serde_json::Value::String(calibrated.resource_receipt_digest.clone());
        method["review_receipt_digest"] =
            serde_json::Value::String(calibrated.review_receipt_digest.clone());
        method["calibration_scope_digest"] =
            serde_json::Value::String(calibrated.calibration_scope_digest.clone());
        let method_bytes = serde_json::to_vec_pretty(&method).unwrap();
        std::fs::write(&method_path, &method_bytes).unwrap();
        calibrated.method_manifest_digest = content_digest(&method_bytes);

        let receipt =
            write_spatial_reference_covariance(&scratch, &calibrated, &[calibrated_block()])
                .unwrap();
        let error = finalize_spatial_reference_covariance_artifact(
            &transaction,
            &scratch,
            &calibrated,
            &receipt,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("current code")
                || error.contains("current design")
                || error.contains("current binary"),
            "{identity}: {error}"
        );
        drop(transaction);
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn missing_calibrated_evidence_is_quarantined_as_deterministically_incomplete() {
    let directory = std::env::temp_dir().join(format!(
        "dolphin_spatial_covariance_missing_evidence_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
    let mut calibrated = metadata();
    write_promotion_evidence(&directory, &mut calibrated);
    let receipt =
        write_spatial_reference_covariance(&scratch, &calibrated, &[calibrated_block()]).unwrap();
    finalize_spatial_reference_covariance_artifact(&transaction, &scratch, &calibrated, &receipt)
        .unwrap();
    drop(transaction);
    std::fs::remove_file(
        directory.join(SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RESULT_FILENAME),
    )
    .unwrap();

    let recovery = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    assert!(!directory
        .join(SPATIAL_REFERENCE_COVARIANCE_FILENAME)
        .exists());
    assert!(!directory
        .join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME)
        .exists());
    assert!(directory
        .join(format!(
            "{SPATIAL_REFERENCE_COVARIANCE_FILENAME}.quarantine.0"
        ))
        .exists());
    drop(recovery);
    std::fs::remove_dir_all(directory).unwrap();
}

#[cfg(unix)]
#[test]
fn permission_denied_calibrated_evidence_preserves_the_final_pair() {
    use std::os::unix::fs::PermissionsExt;

    let directory = std::env::temp_dir().join(format!(
        "dolphin_spatial_covariance_evidence_permission_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
    let mut calibrated = metadata();
    write_promotion_evidence(&directory, &mut calibrated);
    let receipt =
        write_spatial_reference_covariance(&scratch, &calibrated, &[calibrated_block()]).unwrap();
    finalize_spatial_reference_covariance_artifact(&transaction, &scratch, &calibrated, &receipt)
        .unwrap();
    drop(transaction);
    let result = directory.join(SPATIAL_REFERENCE_COVARIANCE_APPROXIMATION_RESULT_FILENAME);
    std::fs::set_permissions(&result, std::fs::Permissions::from_mode(0o000)).unwrap();
    let error = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory)
        .err()
        .expect("unverifiable calibrated evidence must block recovery")
        .to_string();
    assert!(error.contains("unverifiable"), "{error}");
    assert!(directory
        .join(SPATIAL_REFERENCE_COVARIANCE_FILENAME)
        .exists());
    assert!(directory
        .join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME)
        .exists());
    std::fs::set_permissions(&result, std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn calibrated_scope_requires_every_nonzero_scope_identity() {
    for (index, identity) in [
        "mask",
        "reference",
        "source_replay",
        "l2_map",
        "source_model",
        "effective_looks",
        "support",
        "correction",
        "unwrap",
        "burst_ownership",
        "approximation",
        "resource",
    ]
    .iter()
    .enumerate()
    {
        let directory = std::env::temp_dir().join(format!(
            "dolphin_spatial_covariance_zero_scope_{}_{}",
            std::process::id(),
            index
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let transaction =
            SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
        let scratch = directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
        let zero = format!("sha256:{}", "00".repeat(32));
        let mut value = metadata();
        match *identity {
            "mask" => value.mask_digest = zero.clone(),
            "reference" => value.reference_signature_digest = zero.clone(),
            "source_replay" => value.source_replay_digest = zero.clone(),
            "l2_map" => value.l2_map_digest = zero.clone(),
            "source_model" => value.source_model_digest = zero.clone(),
            "effective_looks" => value.effective_looks_digest = zero.clone(),
            "support" => value.support_digest = zero.clone(),
            "correction" => value.correction_order_digest = zero.clone(),
            "unwrap" => value.unwrap_branch_digest = zero.clone(),
            "burst_ownership" => value.burst_ownership_digest = zero.clone(),
            "approximation" | "resource" => {}
            _ => unreachable!(),
        }
        write_promotion_evidence(&directory, &mut value);
        match *identity {
            "approximation" => value.approximation_receipt_digest = zero,
            "resource" => value.resource_receipt_digest = zero,
            _ => {}
        }
        value.calibration_scope_digest = spatial_reference_calibration_scope_digest(&value);
        let error =
            match write_spatial_reference_covariance(&scratch, &value, &[calibrated_block()]) {
                Ok(receipt) => finalize_spatial_reference_covariance_artifact(
                    &transaction,
                    &scratch,
                    &value,
                    &receipt,
                )
                .unwrap_err()
                .to_string(),
                Err(error) => error.to_string(),
            };
        assert!(
            error.contains("nonzero scope identities")
                || error.contains("strong promotion receipts"),
            "{identity}: {error}"
        );
        drop(transaction);
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn calibrated_promotion_rejects_non_commit_producer_labels() {
    let directory = std::env::temp_dir().join(format!(
        "dolphin_spatial_covariance_producer_commit_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
    let mut calibrated = metadata();
    calibrated.producer_commit = Some("abc123".to_owned());
    write_promotion_evidence(&directory, &mut calibrated);
    let error = write_spatial_reference_covariance(&scratch, &calibrated, &[calibrated_block()])
        .unwrap_err()
        .to_string();
    assert!(error.contains("exact producer"), "{error}");
    drop(transaction);
    std::fs::remove_dir_all(directory).unwrap();
}

#[cfg(unix)]
#[test]
fn recovery_preserves_final_pair_when_verification_is_permission_denied() {
    use std::os::unix::fs::PermissionsExt;

    let directory = std::env::temp_dir().join(format!(
        "dolphin_spatial_covariance_permission_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let transaction = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory).unwrap();
    let scratch = directory.join(SPATIAL_REFERENCE_COVARIANCE_HDF5_SCRATCH_FILENAME);
    let receipt = write_spatial_reference_covariance(&scratch, &metadata(), &[block()]).unwrap();
    finalize_spatial_reference_covariance_artifact(&transaction, &scratch, &metadata(), &receipt)
        .unwrap();
    drop(transaction);
    let manifest = directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME);
    let hdf5 = directory.join(SPATIAL_REFERENCE_COVARIANCE_FILENAME);
    std::fs::set_permissions(&manifest, std::fs::Permissions::from_mode(0o000)).unwrap();
    let error = SpatialReferenceCovarianceArtifactTransaction::acquire(&directory)
        .err()
        .expect("unverifiable final pair must block recovery")
        .to_string();
    assert!(error.contains("unverifiable"), "{error}");
    assert!(manifest.exists());
    assert!(hdf5.exists());
    std::fs::set_permissions(&manifest, std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn oversized_manifest_is_rejected_from_metadata_before_bounded_read() {
    let directory = std::env::temp_dir().join(format!(
        "dolphin_spatial_covariance_manifest_cap_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME),
        vec![b' '; 1024 * 1024 + 1],
    )
    .unwrap();
    let error = read_spatial_reference_covariance_artifact_manifest(&directory)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("1048577 exceeds byte cap 1048576"),
        "{error}"
    );
    std::fs::remove_dir_all(directory).unwrap();
}

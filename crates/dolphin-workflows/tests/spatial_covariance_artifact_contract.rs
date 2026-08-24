use dolphin_io::{
    write_spatial_reference_covariance, CovarianceOperatorGrid, SpatialReferenceCalibrationScope,
    SpatialReferenceCovarianceBlock, SpatialReferenceCovarianceMetadata,
    SpatialReferenceCovarianceStatus, SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE,
    SPATIAL_REFERENCE_COVARIANCE_METHOD, SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
};
use dolphin_workflows::{
    finalize_spatial_reference_covariance_artifact,
    read_spatial_reference_covariance_artifact_manifest,
    SpatialReferenceCovarianceArtifactTransaction, SPATIAL_REFERENCE_COVARIANCE_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME,
};

fn digest(byte: u8) -> String {
    format!("sha256:{}", format!("{byte:02x}").repeat(32))
}

fn metadata() -> SpatialReferenceCovarianceMetadata {
    SpatialReferenceCovarianceMetadata {
        schema_version: SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
        method: SPATIAL_REFERENCE_COVARIANCE_METHOD.to_owned(),
        method_version: 1,
        crate_version: env!("CARGO_PKG_VERSION").to_owned(),
        producer_commit: Some("abc123".to_owned()),
        burst_id: "T078-165482-IW1".to_owned(),
        crs: "EPSG:32611".to_owned(),
        units: "radians".to_owned(),
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
        source_factor_digest: digest(0x77),
    }
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
    let scratch = directory.join("spatial-reference.scratch.h5");
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
    assert_eq!(manifest.calibration_scope, "uncalibrated");
    assert!(directory
        .join(SPATIAL_REFERENCE_COVARIANCE_FILENAME)
        .exists());
    assert_eq!(
        read_spatial_reference_covariance_artifact_manifest(&directory).unwrap(),
        manifest
    );
    drop(transaction);
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
    let scratch = directory.join("spatial-reference.scratch.h5");
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

    std::fs::write(&manifest_path, original).unwrap();
    let hdf5 = directory.join(SPATIAL_REFERENCE_COVARIANCE_FILENAME);
    let mut bytes = std::fs::read(&hdf5).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&hdf5, bytes).unwrap();
    assert!(read_spatial_reference_covariance_artifact_manifest(&directory).is_err());
    std::fs::remove_dir_all(directory).unwrap();
}

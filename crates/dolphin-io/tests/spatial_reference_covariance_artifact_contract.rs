use dolphin_io::{
    read_spatial_reference_covariance_block, read_spatial_reference_covariance_header,
    write_spatial_reference_covariance, CovarianceOperatorGrid, SpatialReferenceCalibrationScope,
    SpatialReferenceCovarianceBlock, SpatialReferenceCovarianceMetadata,
    SpatialReferenceCovarianceStatus, SPATIAL_REFERENCE_COVARIANCE_METHOD,
    SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
};

fn digest(byte: u8) -> String {
    format!("sha256:{}", format!("{byte:02x}").repeat(32))
}

fn grid(row_start: u64, rows: u32) -> CovarianceOperatorGrid {
    CovarianceOperatorGrid {
        row_start,
        col_start: 0,
        rows,
        cols: 1,
        stride_y: 1,
        stride_x: 1,
    }
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
        full_grid: grid(0, 2),
        reference_row: 1,
        reference_col: 0,
        gauge_date_index: 0,
        ordered_date_indices: vec![0, 1, 2],
        mask_digest: digest(0x11),
        source_replay_digest: digest(0x22),
        l2_map_digest: digest(0x33),
        reference_signature_digest: digest(0x44),
        approximation_receipt_digest: digest(0x55),
        resource_receipt_digest: digest(0x66),
        calibration_scope: SpatialReferenceCalibrationScope::Uncalibrated,
        maximum_block_bytes: 4096,
    }
}

fn block() -> SpatialReferenceCovarianceBlock {
    SpatialReferenceCovarianceBlock {
        block_id: 7,
        target_grid: grid(0, 2),
        maximum_rank: 2,
        rank_by_target: vec![2, 1],
        status: vec![
            SpatialReferenceCovarianceStatus::Valid,
            SpatialReferenceCovarianceStatus::Valid,
        ],
        difference_factor: vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, // target 0
            0.0, 0.0, 0.5, 0.0, 1.5, 0.0, // target 1
        ],
        approximation_error_bound: vec![0.01, 0.02],
        source_factor_digest: digest(0x77),
    }
}

#[test]
fn chunked_reference_factor_round_trips_under_a_byte_cap() {
    let path = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_factor_{}.h5",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let receipt = write_spatial_reference_covariance(&path, &metadata(), &[block()]).unwrap();
    assert!(receipt.hdf5_bytes > 0);
    assert_eq!(receipt.hdf5_sha256.len(), 64);
    assert_eq!(receipt.block_count, 1);

    let read_metadata = read_spatial_reference_covariance_header(&path, 4096).unwrap();
    assert_eq!(read_metadata, metadata());
    let read = read_spatial_reference_covariance_block(&path, 7, 4096).unwrap();
    assert_eq!(read.block, block());
    assert_eq!(read.logical_payload_bytes, 12 * 8 + 2 * 4 + 2 * 2 + 2 * 8);

    let error = read_spatial_reference_covariance_block(&path, 7, 32).unwrap_err();
    assert!(error.to_string().contains("byte cap"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn malformed_scope_gauge_hash_and_factor_fail_before_commit() {
    let base = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_invalid_{}",
        std::process::id()
    ));
    let mut invalid_metadata = metadata();
    invalid_metadata.gauge_date_index = 1;
    assert!(write_spatial_reference_covariance(
        base.with_extension("gauge.h5"),
        &invalid_metadata,
        &[block()]
    )
    .is_err());

    invalid_metadata = metadata();
    invalid_metadata.mask_digest = "weak".to_owned();
    assert!(write_spatial_reference_covariance(
        base.with_extension("hash.h5"),
        &invalid_metadata,
        &[block()]
    )
    .is_err());

    invalid_metadata = metadata();
    invalid_metadata.calibration_scope = SpatialReferenceCalibrationScope::CalibratedScopeMatch;
    invalid_metadata.approximation_receipt_digest.clear();
    assert!(write_spatial_reference_covariance(
        base.with_extension("scope.h5"),
        &invalid_metadata,
        &[block()]
    )
    .is_err());

    let mut invalid_block = block();
    invalid_block.rank_by_target[0] = 3;
    assert!(write_spatial_reference_covariance(
        base.with_extension("rank.h5"),
        &metadata(),
        &[invalid_block]
    )
    .is_err());
}

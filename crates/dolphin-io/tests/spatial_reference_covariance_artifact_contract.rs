use dolphin_io::{
    read_spatial_reference_covariance_block, read_spatial_reference_covariance_block_ids,
    read_spatial_reference_covariance_header, spatial_reference_calibration_scope_digest,
    spatial_reference_effective_looks_digest, spatial_reference_runtime_resource_receipt_digest,
    write_spatial_reference_covariance, CovarianceOperatorGrid, SpatialReferenceCalibrationScope,
    SpatialReferenceCovarianceBlock, SpatialReferenceCovarianceMetadata,
    SpatialReferenceCovarianceStatus, SpatialReferenceCovarianceWriter,
    SpatialReferenceRuntimeResourceReceipt, SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE,
    SPATIAL_REFERENCE_COVARIANCE_METHOD, SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
    SPATIAL_REFERENCE_COVARIANCE_STATUS_REGISTRY, SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE,
};
use std::sync::{Mutex, MutexGuard};

static HDF5_LOCK: Mutex<()> = Mutex::new(());

fn hdf5_guard() -> MutexGuard<'static, ()> {
    HDF5_LOCK.lock().unwrap()
}

const DETAILED_STATUSES: &[SpatialReferenceCovarianceStatus] = &[
    SpatialReferenceCovarianceStatus::Valid,
    SpatialReferenceCovarianceStatus::InvalidReference,
    SpatialReferenceCovarianceStatus::ReplayUnsupported,
    SpatialReferenceCovarianceStatus::L2RankDeficient,
    SpatialReferenceCovarianceStatus::ScopeMismatch,
    SpatialReferenceCovarianceStatus::UnsupportedMultiburstReference,
    SpatialReferenceCovarianceStatus::MaskedTarget,
    SpatialReferenceCovarianceStatus::TemporalFactorInvalid,
    SpatialReferenceCovarianceStatus::ReplayUnavailable,
    SpatialReferenceCovarianceStatus::ReplayMismatch,
    SpatialReferenceCovarianceStatus::InfluenceInvalid,
    SpatialReferenceCovarianceStatus::NondifferentiableEstimator,
    SpatialReferenceCovarianceStatus::UnstableAdaptiveSupport,
    SpatialReferenceCovarianceStatus::UnsupportedL1,
    SpatialReferenceCovarianceStatus::UnsupportedPhaseBias,
    SpatialReferenceCovarianceStatus::UnsupportedCorrectionOrder,
    SpatialReferenceCovarianceStatus::TiedEigenvalue,
    SpatialReferenceCovarianceStatus::EmptySupport,
    SpatialReferenceCovarianceStatus::NonfiniteSource,
    SpatialReferenceCovarianceStatus::UnsupportedModel,
    SpatialReferenceCovarianceStatus::IllConditioned,
    SpatialReferenceCovarianceStatus::SupportIdentityMismatch,
];

fn digest(byte: u8) -> String {
    format!("sha256:{}", format!("{byte:02x}").repeat(32))
}

#[test]
fn streaming_writer_keeps_incomplete_artifacts_unreadable_and_rejects_duplicate_blocks() {
    let _hdf5 = hdf5_guard();
    assert_eq!(SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION, 4);
    let path = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_streaming_{}.h5",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let mut writer = SpatialReferenceCovarianceWriter::create(&path, &metadata()).unwrap();
    assert!(read_spatial_reference_covariance_header(&path, 4096).is_err());
    writer.write_block(&block()).unwrap();
    assert!(writer.write_block(&block()).is_err());
    let receipt = writer.finish().unwrap();
    assert_eq!(receipt.block_count, 1);
    assert_eq!(
        read_spatial_reference_covariance_header(&path, 4096).unwrap(),
        metadata()
    );
    std::fs::remove_file(path).unwrap();

    let partial_path = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_partial_{}.h5",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&partial_path);
    let mut partial = block();
    partial.target_grid = grid(0, 1);
    partial.rank_by_target.truncate(1);
    partial.status.truncate(1);
    partial.source_burst_index_by_target.truncate(1);
    partial.difference_factor.truncate(6);
    partial.approximation_error_bound.truncate(1);
    partial
        .effective_looks_fraction
        .as_mut()
        .unwrap()
        .truncate(1);
    partial.support_union_count.as_mut().unwrap().truncate(1);
    partial
        .effective_looks_receipt
        .as_mut()
        .unwrap()
        .truncate(32);
    partial
        .resource_high_water_bytes
        .as_mut()
        .unwrap()
        .truncate(1);
    partial.condition_number.as_mut().unwrap().truncate(1);
    let mut writer = SpatialReferenceCovarianceWriter::create(&partial_path, &metadata()).unwrap();
    writer.write_block(&partial).unwrap();
    let mut overlap = partial.clone();
    overlap.block_id += 1;
    assert!(writer.write_block(&overlap).is_err());
    assert!(writer.finish().is_err());
    assert!(read_spatial_reference_covariance_header(&partial_path, 4096).is_err());
    std::fs::remove_file(partial_path).unwrap();
}

#[test]
fn runtime_resource_receipt_seals_observed_provider_peak_and_rejects_tamper() {
    let _hdf5 = hdf5_guard();
    let path = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_runtime_resource_{}.h5",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let initial = metadata();
    let mut writer = SpatialReferenceCovarianceWriter::create(&path, &initial).unwrap();
    writer.write_block(&block()).unwrap();
    let observed = SpatialReferenceRuntimeResourceReceipt {
        provider_peak_count: 1,
        provider_peak_bytes: 256,
        ..initial.runtime_resource_receipt.unwrap()
    };
    let sealed = writer.seal_runtime_resource_receipt(observed).unwrap();
    assert_eq!(sealed.runtime_resource_receipt, Some(observed));
    assert_eq!(
        sealed.runtime_resource_receipt_digest,
        spatial_reference_runtime_resource_receipt_digest(observed)
    );
    let realized_effective = spatial_reference_effective_looks_digest(&[block()]).unwrap();
    let sealed = writer.seal_effective_looks_digest().unwrap();
    assert_eq!(sealed.effective_looks_digest, realized_effective);
    writer.finish().unwrap();
    assert_eq!(
        read_spatial_reference_covariance_header(&path, 4096)
            .unwrap()
            .runtime_resource_receipt,
        Some(observed)
    );
    assert_eq!(
        read_spatial_reference_covariance_header(&path, 4096)
            .unwrap()
            .effective_looks_digest,
        realized_effective
    );
    let file = hdf5::File::open_rw(&path).unwrap();
    file.dataset("metadata/provider_peak_bytes")
        .unwrap()
        .write_raw(&[257_u64])
        .unwrap();
    file.flush().unwrap();
    file.close().unwrap();
    assert!(read_spatial_reference_covariance_header(&path, 4096).is_err());
    std::fs::remove_file(path).unwrap();
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
    let runtime_resource_receipt = SpatialReferenceRuntimeResourceReceipt {
        working_set_byte_cap: 16384,
        factor_block_high_water_bytes: 8192,
        serialization_high_water_bytes: 2048,
        fixed_l2_workspace_admission_bytes: 2048,
        fixed_l2_workspace_observed_high_water_bytes: 1024,
        replay_admission_high_water_bytes: 4096,
        replay_observed_high_water_bytes: 1536,
        provider_peak_count: 2,
        provider_peak_bytes: 1024,
        preflight_provider_open_count: 4,
        production_provider_open_count: 2,
        operator_block_reads: 2,
        operator_block_cache_hits: 3,
        source_member_window_reads: 4,
        source_tile_cache_loads: 2,
        source_resolutions: 5,
        working_set_admission_high_water_bytes: 16384,
        working_set_observed_high_water_bytes: 12800,
    };
    let mut metadata = SpatialReferenceCovarianceMetadata {
        schema_version: SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
        method: SPATIAL_REFERENCE_COVARIANCE_METHOD.to_owned(),
        method_version: 1,
        crate_version: env!("CARGO_PKG_VERSION").to_owned(),
        producer_commit: Some("abc123".to_owned()),
        burst_id: "T078-165482-IW1".to_owned(),
        crs: "EPSG:32611".to_owned(),
        units: "radians".to_owned(),
        geotransform: Some([500_000.0, 30.0, 0.0, 4_200_000.0, 0.0, -30.0]),
        full_grid: grid(0, 2),
        reference_row: 1,
        reference_col: 0,
        gauge_date_index: 0,
        ordered_date_indices: vec![0, 1, 2],
        acquisition_days: Some(vec![0.0, 12.0, 24.0]),
        mask_digest: digest(0x11),
        source_replay_digest: digest(0x22),
        l2_map_digest: digest(0x33),
        reference_signature_digest: digest(0x44),
        approximation_receipt_digest: digest(0x55),
        resource_receipt_digest: digest(0x66),
        runtime_resource_receipt_digest: spatial_reference_runtime_resource_receipt_digest(
            runtime_resource_receipt,
        ),
        runtime_resource_receipt: Some(runtime_resource_receipt),
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
        maximum_block_bytes: 16384,
    };
    metadata.effective_looks_digest = spatial_reference_effective_looks_digest(&[block()]).unwrap();
    metadata
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
        source_burst_index_by_target: vec![0, 0],
        difference_factor: vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 0.25, // target 0
            0.0, 0.0, 0.5, 0.0, 1.5, 0.0, // target 1
        ],
        approximation_error_bound: vec![SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE; 2],
        effective_looks_fraction: Some(vec![0.75, 0.5]),
        support_union_count: Some(vec![9, 12]),
        effective_looks_receipt: Some([vec![0x71; 32], vec![0x72; 32]].concat()),
        resource_high_water_bytes: Some(vec![1024, 1536]),
        condition_number: Some(vec![16.0, 1.0]),
        source_factor_digest: digest(0x77),
    }
}

#[test]
fn chunked_reference_factor_round_trips_under_a_byte_cap() {
    let _hdf5 = hdf5_guard();
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
    assert_eq!(
        read_metadata.geotransform,
        Some([500_000.0, 30.0, 0.0, 4_200_000.0, 0.0, -30.0])
    );
    let read = read_spatial_reference_covariance_block(&path, 7, 4096).unwrap();
    assert_eq!(read.block.status, block().status);
    assert_eq!(read.block.difference_factor, block().difference_factor);
    assert_eq!(read.block.effective_looks_fraction, Some(vec![0.75, 0.5]));
    assert_eq!(read.block.support_union_count, Some(vec![9, 12]));
    assert_eq!(read.block.resource_high_water_bytes, Some(vec![1024, 1536]));
    assert_eq!(
        read.block.effective_looks_receipt.as_ref().unwrap().len(),
        64
    );
    assert!(read
        .block
        .approximation_error_bound
        .iter()
        .all(|bound| bound.is_nan()));
    assert_eq!(
        read.logical_payload_bytes,
        12 * 8 + 2 * 4 + 2 * 2 + 2 * 4 + 2 * 8 + 2 * 8 + 2 * 8 + 2 * 32 + 2 * 8 + 2 * 8
    );
    let mut lower = 0_u64;
    let mut upper = 4096_u64;
    while lower < upper {
        let middle = lower + (upper - lower) / 2;
        if read_spatial_reference_covariance_header(&path, middle).is_ok() {
            upper = middle;
        } else {
            lower = middle + 1;
        }
    }
    let exact_combined_cap = lower + read.logical_payload_bytes;
    read_spatial_reference_covariance_block(&path, 7, exact_combined_cap).unwrap();
    let error =
        read_spatial_reference_covariance_block(&path, 7, exact_combined_cap - 1).unwrap_err();
    assert!(error.to_string().contains("byte cap"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn sealed_reference_factor_enumerates_block_ids_under_a_byte_cap() {
    let _hdf5 = hdf5_guard();
    let path = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_block_ids_{}.h5",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let mut metadata = metadata();
    let mut first = block();
    first.target_grid = grid(0, 1);
    first.rank_by_target.truncate(1);
    first.status.truncate(1);
    first.source_burst_index_by_target.truncate(1);
    first.difference_factor.truncate(6);
    first.approximation_error_bound.truncate(1);
    first.effective_looks_fraction.as_mut().unwrap().truncate(1);
    first.support_union_count.as_mut().unwrap().truncate(1);
    first.effective_looks_receipt.as_mut().unwrap().truncate(32);
    first
        .resource_high_water_bytes
        .as_mut()
        .unwrap()
        .truncate(1);
    first.condition_number.as_mut().unwrap().truncate(1);
    let mut second = block();
    second.block_id = 11;
    second.target_grid = grid(1, 1);
    second.rank_by_target = vec![second.rank_by_target[1]];
    second.status = vec![second.status[1]];
    second.source_burst_index_by_target = vec![second.source_burst_index_by_target[1]];
    second.difference_factor = second.difference_factor[6..].to_vec();
    second.approximation_error_bound = vec![second.approximation_error_bound[1]];
    second.effective_looks_fraction = Some(vec![0.5]);
    second.support_union_count = Some(vec![12]);
    second.effective_looks_receipt = Some(vec![0x72; 32]);
    second.resource_high_water_bytes = Some(vec![1536]);
    second.condition_number = Some(vec![1.0]);
    metadata.effective_looks_digest =
        spatial_reference_effective_looks_digest(&[first.clone(), second.clone()]).unwrap();
    write_spatial_reference_covariance(&path, &metadata, &[second, first]).unwrap();

    let mut exact_cap = 0;
    while read_spatial_reference_covariance_block_ids(&path, exact_cap).is_err() {
        exact_cap += 1;
    }
    assert_eq!(
        read_spatial_reference_covariance_block_ids(&path, exact_cap).unwrap(),
        vec![7, 11]
    );
    assert!(read_spatial_reference_covariance_block_ids(&path, exact_cap - 1).is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn detailed_status_registry_round_trips_and_unknown_codes_fail_closed() {
    let _hdf5 = hdf5_guard();
    let path = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_statuses_{}.h5",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let target_count = DETAILED_STATUSES.len();
    let mut metadata = metadata();
    metadata.full_grid.rows = u32::try_from(target_count).unwrap();
    metadata.reference_row = 0;
    let mut status_block = SpatialReferenceCovarianceBlock {
        block_id: 9,
        target_grid: metadata.full_grid,
        maximum_rank: 1,
        rank_by_target: vec![0; target_count],
        status: DETAILED_STATUSES.to_vec(),
        source_burst_index_by_target: vec![
            SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE;
            target_count
        ],
        difference_factor: vec![0.0; target_count * metadata.ordered_date_indices.len()],
        approximation_error_bound: vec![
            SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE;
            target_count
        ],
        effective_looks_fraction: Some(vec![f64::NAN; target_count]),
        support_union_count: Some(vec![0; target_count]),
        effective_looks_receipt: Some(vec![0; target_count * 32]),
        resource_high_water_bytes: Some(vec![0; target_count]),
        condition_number: Some(vec![f64::NAN; target_count]),
        source_factor_digest: digest(0x77),
    };
    status_block.rank_by_target[0] = 1;
    status_block.source_burst_index_by_target[0] = 0;
    status_block.difference_factor[1] = 1.0;
    status_block.effective_looks_fraction.as_mut().unwrap()[0] = 0.75;
    status_block.support_union_count.as_mut().unwrap()[0] = 9;
    status_block.effective_looks_receipt.as_mut().unwrap()[..32].fill(0x71);
    status_block.resource_high_water_bytes.as_mut().unwrap()[0] = 1024;
    status_block.condition_number.as_mut().unwrap()[0] = 1.0;
    status_block.resource_high_water_bytes.as_mut().unwrap()[8] = 1536;

    assert_eq!(
        SPATIAL_REFERENCE_COVARIANCE_STATUS_REGISTRY.len(),
        target_count
    );
    assert_eq!(
        SPATIAL_REFERENCE_COVARIANCE_STATUS_REGISTRY
            .iter()
            .map(|entry| entry.name)
            .collect::<Vec<_>>(),
        vec![
            "valid",
            "invalid_reference",
            "replay_unsupported",
            "l2_rank_deficient",
            "scope_mismatch",
            "unsupported_multiburst_reference",
            "masked_target",
            "temporal_factor_invalid",
            "replay_unavailable",
            "replay_mismatch",
            "influence_invalid",
            "nondifferentiable_estimator",
            "unstable_adaptive_support",
            "unsupported_l1",
            "unsupported_phase_bias",
            "unsupported_correction_order",
            "tied_eigenvalue",
            "empty_support",
            "nonfinite_source",
            "unsupported_model",
            "ill_conditioned",
            "support_identity_mismatch",
        ]
    );
    write_spatial_reference_covariance(&path, &metadata, &[status_block.clone()]).unwrap();
    let read = read_spatial_reference_covariance_block(&path, 9, 16_384).unwrap();
    assert_eq!(read.block.status, status_block.status);
    assert_eq!(
        read.block.resource_high_water_bytes,
        status_block.resource_high_water_bytes
    );

    let file = hdf5::File::open_rw(&path).unwrap();
    let dataset = file.dataset("blocks/00000000000000000009/status").unwrap();
    let mut codes = dataset.read_raw::<u16>().unwrap();
    codes[0] = u16::MAX;
    dataset.write_raw(&codes).unwrap();
    file.flush().unwrap();
    file.close().unwrap();
    let error = read_spatial_reference_covariance_block(&path, 9, 16_384).unwrap_err();
    assert!(error
        .to_string()
        .contains("unknown spatial reference covariance status"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn coincident_valid_target_round_trips_zero_rank_but_noncoincident_does_not() {
    let _hdf5 = hdf5_guard();
    let path = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_coincident_{}.h5",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let mut coincident = block();
    coincident.rank_by_target[0] = 0;
    coincident.difference_factor[..6].fill(0.0);
    coincident.condition_number.as_mut().unwrap()[0] = f64::NAN;
    let mut coincident_metadata = metadata();
    coincident_metadata.reference_row = 0;
    write_spatial_reference_covariance(&path, &coincident_metadata, &[coincident]).unwrap();
    let read = read_spatial_reference_covariance_block(&path, 7, 4096).unwrap();
    assert_eq!(
        read.block.status[0],
        SpatialReferenceCovarianceStatus::Valid
    );
    assert_eq!(read.block.rank_by_target[0], 0);
    assert!(read.block.difference_factor[..6]
        .iter()
        .all(|value| *value == 0.0));
    let mut noncoincident = block();
    noncoincident.rank_by_target[0] = 0;
    noncoincident.difference_factor[..6].fill(0.0);
    noncoincident.condition_number.as_mut().unwrap()[0] = f64::NAN;
    assert!(write_spatial_reference_covariance(
        path.with_extension("noncoincident.h5"),
        &metadata(),
        &[noncoincident]
    )
    .is_err());
    let mut zero_declared_column = block();
    for index in [1, 3, 5] {
        zero_declared_column.difference_factor[index] = 0.0;
    }
    assert!(write_spatial_reference_covariance(
        path.with_extension("zero-column.h5"),
        &metadata(),
        &[zero_declared_column]
    )
    .is_err());
    let mut collinear = block();
    collinear.difference_factor[1] = 0.0;
    collinear.difference_factor[3] = 0.5;
    collinear.difference_factor[5] = 0.0;
    assert!(write_spatial_reference_covariance(
        path.with_extension("collinear.h5"),
        &metadata(),
        &[collinear]
    )
    .is_err());
    let mut over_conditioned = block();
    over_conditioned.difference_factor[5] = 1.0e-5;
    over_conditioned.condition_number.as_mut().unwrap()[0] = 1.0e10;
    assert!(write_spatial_reference_covariance(
        path.with_extension("over-conditioned.h5"),
        &metadata(),
        &[over_conditioned]
    )
    .is_err());
}

#[test]
fn reader_rejects_effective_look_realization_tamper() {
    let _hdf5 = hdf5_guard();
    let path = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_effective_tamper_{}.h5",
        std::process::id()
    ));
    write_spatial_reference_covariance(&path, &metadata(), &[block()]).unwrap();
    let file = hdf5::File::open_rw(&path).unwrap();
    file.dataset("blocks/00000000000000000007/effective_looks_fraction")
        .unwrap()
        .write_raw(&[0.5_f64, 0.5])
        .unwrap();
    file.flush().unwrap();
    file.close().unwrap();
    assert!(read_spatial_reference_covariance_block(&path, 7, 4096).is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn approximation_bounds_are_absent_until_a_valid_scope_is_calibrated() {
    let _hdf5 = hdf5_guard();
    let base = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_bounds_{}",
        std::process::id()
    ));
    let mut false_uncalibrated_bound = block();
    false_uncalibrated_bound
        .approximation_error_bound
        .fill(0.01);
    assert!(write_spatial_reference_covariance(
        base.with_extension("uncalibrated.h5"),
        &metadata(),
        &[false_uncalibrated_bound]
    )
    .is_err());

    let mut calibrated_metadata = metadata();
    calibrated_metadata.producer_commit = Some("a".repeat(40));
    calibrated_metadata.calibration_scope = SpatialReferenceCalibrationScope::CalibratedScopeMatch;
    calibrated_metadata.review_receipt_digest = digest(0x81);
    calibrated_metadata.method_manifest_digest = digest(0x82);
    calibrated_metadata.calibration_scope_digest =
        spatial_reference_calibration_scope_digest(&calibrated_metadata);
    assert!(write_spatial_reference_covariance(
        base.with_extension("missing-bound.h5"),
        &calibrated_metadata,
        &[block()]
    )
    .is_err());

    let mut calibrated_block = block();
    calibrated_block.approximation_error_bound = vec![0.01, 0.02];
    write_spatial_reference_covariance(
        base.with_extension("calibrated.h5"),
        &calibrated_metadata,
        &[calibrated_block],
    )
    .unwrap();

    let mut unsupported = block();
    unsupported.rank_by_target[0] = 0;
    unsupported.status[0] = SpatialReferenceCovarianceStatus::MaskedTarget;
    unsupported.difference_factor[..6].fill(0.0);
    unsupported.effective_looks_fraction.as_mut().unwrap()[0] = f64::NAN;
    unsupported.support_union_count.as_mut().unwrap()[0] = 0;
    unsupported.effective_looks_receipt.as_mut().unwrap()[..32].fill(0);
    unsupported.resource_high_water_bytes.as_mut().unwrap()[0] = 0;
    unsupported.condition_number.as_mut().unwrap()[0] = f64::NAN;
    unsupported.approximation_error_bound[0] = 0.01;
    assert!(write_spatial_reference_covariance(
        base.with_extension("unsupported-bound.h5"),
        &calibrated_metadata,
        &[unsupported]
    )
    .is_err());

    let mut calibrated_masked = block();
    calibrated_masked.rank_by_target[0] = 0;
    calibrated_masked.status[0] = SpatialReferenceCovarianceStatus::MaskedTarget;
    calibrated_masked.source_burst_index_by_target[0] = SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE;
    calibrated_masked.difference_factor[..6].fill(0.0);
    calibrated_masked.effective_looks_fraction.as_mut().unwrap()[0] = f64::NAN;
    calibrated_masked.support_union_count.as_mut().unwrap()[0] = 0;
    calibrated_masked.effective_looks_receipt.as_mut().unwrap()[..32].fill(0);
    calibrated_masked
        .resource_high_water_bytes
        .as_mut()
        .unwrap()[0] = 0;
    calibrated_masked.condition_number.as_mut().unwrap()[0] = f64::NAN;
    calibrated_masked.approximation_error_bound[1] = 0.02;
    let mut calibrated_masked_metadata = calibrated_metadata.clone();
    calibrated_masked_metadata.effective_looks_digest =
        spatial_reference_effective_looks_digest(std::slice::from_ref(&calibrated_masked)).unwrap();
    calibrated_masked_metadata.calibration_scope_digest =
        spatial_reference_calibration_scope_digest(&calibrated_masked_metadata);
    write_spatial_reference_covariance(
        base.with_extension("calibrated-masked.h5"),
        &calibrated_masked_metadata,
        &[calibrated_masked],
    )
    .unwrap();
}

#[test]
fn calibrated_scope_requires_nonzero_exact_identity_receipts_and_rejects_tamper() {
    let _hdf5 = hdf5_guard();
    let path = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_scope_{}.h5",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let mut calibrated_metadata = metadata();
    calibrated_metadata.producer_commit = Some("a".repeat(40));
    calibrated_metadata.calibration_scope = SpatialReferenceCalibrationScope::CalibratedScopeMatch;
    calibrated_metadata.review_receipt_digest = digest(0x81);
    calibrated_metadata.method_manifest_digest = digest(0x82);
    calibrated_metadata.approximation_receipt_digest = digest(0x83);
    calibrated_metadata.resource_receipt_digest = digest(0x84);
    calibrated_metadata.calibration_scope_digest =
        spatial_reference_calibration_scope_digest(&calibrated_metadata);
    let mut calibrated_block = block();
    calibrated_block.approximation_error_bound = vec![0.01, 0.02];
    write_spatial_reference_covariance(&path, &calibrated_metadata, &[calibrated_block]).unwrap();

    let file = hdf5::File::open_rw(&path).unwrap();
    file.group("metadata")
        .unwrap()
        .attr("reference_row")
        .unwrap()
        .write_scalar(&0_u64)
        .unwrap();
    file.flush().unwrap();
    file.close().unwrap();
    assert!(read_spatial_reference_covariance_header(&path, 4096).is_err());

    let mut zero_identity = calibrated_metadata;
    zero_identity.source_replay_digest = digest(0x00);
    zero_identity.calibration_scope_digest =
        spatial_reference_calibration_scope_digest(&zero_identity);
    assert!(write_spatial_reference_covariance(
        path.with_extension("zero-source.h5"),
        &zero_identity,
        &[{
            let mut value = block();
            value.approximation_error_bound = vec![0.01, 0.02];
            value
        }]
    )
    .is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn calibrated_scope_requires_and_binds_an_exact_producer_code_identity() {
    let _hdf5 = hdf5_guard();
    let path = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_producer_{}.h5",
        std::process::id()
    ));
    let mut calibrated = metadata();
    calibrated.calibration_scope = SpatialReferenceCalibrationScope::CalibratedScopeMatch;
    calibrated.review_receipt_digest = digest(0x81);
    calibrated.method_manifest_digest = digest(0x82);
    calibrated.producer_commit = None;
    calibrated.calibration_scope_digest = spatial_reference_calibration_scope_digest(&calibrated);
    let mut calibrated_block = block();
    calibrated_block.approximation_error_bound = vec![0.01, 0.02];
    assert!(
        write_spatial_reference_covariance(&path, &calibrated, &[calibrated_block.clone()])
            .is_err()
    );

    calibrated.producer_commit = Some("a".repeat(40));
    let first = spatial_reference_calibration_scope_digest(&calibrated);
    calibrated.producer_commit = Some("b".repeat(40));
    let second = spatial_reference_calibration_scope_digest(&calibrated);
    assert_ne!(first, second);
    calibrated.producer_commit = Some("not-an-immutable-code-identity".to_owned());
    calibrated.calibration_scope_digest = spatial_reference_calibration_scope_digest(&calibrated);
    assert!(write_spatial_reference_covariance(&path, &calibrated, &[calibrated_block]).is_err());
}

#[test]
fn legacy_v2_uncalibrated_finite_bounds_are_readable_but_new_writes_require_v4() {
    let _hdf5 = hdf5_guard();
    let path = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_legacy_v2_{}.h5",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    write_spatial_reference_covariance(&path, &metadata(), &[block()]).unwrap();
    let file = hdf5::File::open_rw(&path).unwrap();
    file.attr("schema_version")
        .unwrap()
        .write_scalar(&2_u16)
        .unwrap();
    file.dataset("blocks/00000000000000000007/approximation_error_bound")
        .unwrap()
        .write_raw(&[0.01_f64, 0.02])
        .unwrap();
    let metadata_group = file.group("metadata").unwrap();
    metadata_group.unlink("geotransform").unwrap();
    metadata_group.unlink("acquisition_days").unwrap();
    metadata_group
        .unlink("runtime_resource_receipt_digest")
        .unwrap();
    for name in [
        "working_set_byte_cap",
        "factor_block_high_water_bytes",
        "serialization_high_water_bytes",
        "fixed_l2_workspace_admission_bytes",
        "fixed_l2_workspace_observed_high_water_bytes",
        "replay_admission_high_water_bytes",
        "replay_observed_high_water_bytes",
        "provider_peak_count",
        "provider_peak_bytes",
        "preflight_provider_open_count",
        "production_provider_open_count",
        "operator_block_reads",
        "operator_block_cache_hits",
        "source_member_window_reads",
        "source_tile_cache_loads",
        "source_resolutions",
        "working_set_admission_high_water_bytes",
        "working_set_observed_high_water_bytes",
    ] {
        metadata_group.unlink(name).unwrap();
    }
    let block_group = file.group("blocks/00000000000000000007").unwrap();
    for name in [
        "effective_looks_fraction",
        "support_union_count",
        "effective_looks_receipt",
        "resource_high_water_bytes",
        "condition_number",
    ] {
        block_group.unlink(name).unwrap();
    }
    file.flush().unwrap();
    file.close().unwrap();

    let legacy = read_spatial_reference_covariance_header(&path, 4096).unwrap();
    assert_eq!(legacy.schema_version, 2);
    assert_eq!(legacy.geotransform, None);
    assert_eq!(legacy.acquisition_days, None);
    assert_eq!(legacy.runtime_resource_receipt, None);
    let legacy_block = read_spatial_reference_covariance_block(&path, 7, 4096)
        .unwrap()
        .block;
    assert_eq!(legacy_block.approximation_error_bound, vec![0.01, 0.02]);
    assert_eq!(legacy_block.effective_looks_fraction, None);
    assert_eq!(legacy_block.support_union_count, None);
    assert_eq!(legacy_block.effective_looks_receipt, None);
    assert_eq!(legacy_block.resource_high_water_bytes, None);
    assert_eq!(legacy_block.condition_number, None);
    assert!(SpatialReferenceCovarianceWriter::create(&path, &legacy).is_err());
    let file = hdf5::File::open_rw(&path).unwrap();
    file.attr("calibration_scope")
        .unwrap()
        .write_scalar(&1_u16)
        .unwrap();
    file.flush().unwrap();
    file.close().unwrap();
    assert!(read_spatial_reference_covariance_header(&path, 4096).is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn previous_v3_layout_without_coordinates_or_runtime_receipts_remains_readable() {
    let _hdf5 = hdf5_guard();
    let path = std::env::temp_dir().join(format!(
        "dolphin_spatial_reference_previous_v3_{}.h5",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    write_spatial_reference_covariance(&path, &metadata(), &[block()]).unwrap();
    let file = hdf5::File::open_rw(&path).unwrap();
    file.attr("schema_version")
        .unwrap()
        .write_scalar(&3_u16)
        .unwrap();
    let metadata_group = file.group("metadata").unwrap();
    for name in [
        "geotransform",
        "acquisition_days",
        "runtime_resource_receipt_digest",
        "working_set_byte_cap",
        "factor_block_high_water_bytes",
        "serialization_high_water_bytes",
        "fixed_l2_workspace_admission_bytes",
        "fixed_l2_workspace_observed_high_water_bytes",
        "replay_admission_high_water_bytes",
        "replay_observed_high_water_bytes",
        "provider_peak_count",
        "provider_peak_bytes",
        "preflight_provider_open_count",
        "production_provider_open_count",
        "operator_block_reads",
        "operator_block_cache_hits",
        "source_member_window_reads",
        "source_tile_cache_loads",
        "source_resolutions",
        "working_set_admission_high_water_bytes",
        "working_set_observed_high_water_bytes",
    ] {
        metadata_group.unlink(name).unwrap();
    }
    let block_group = file.group("blocks/00000000000000000007").unwrap();
    for name in [
        "effective_looks_fraction",
        "support_union_count",
        "effective_looks_receipt",
        "resource_high_water_bytes",
        "condition_number",
    ] {
        block_group.unlink(name).unwrap();
    }
    file.flush().unwrap();
    file.close().unwrap();

    let mut expected = metadata();
    expected.schema_version = 3;
    expected.geotransform = None;
    expected.acquisition_days = None;
    expected.runtime_resource_receipt_digest.clear();
    expected.runtime_resource_receipt = None;
    assert_eq!(
        read_spatial_reference_covariance_header(&path, 4096).unwrap(),
        expected
    );
    let previous = read_spatial_reference_covariance_block(&path, 7, 4096)
        .unwrap()
        .block;
    assert_eq!(previous.effective_looks_fraction, None);
    assert_eq!(previous.support_union_count, None);
    assert_eq!(previous.effective_looks_receipt, None);
    assert_eq!(previous.resource_high_water_bytes, None);
    assert_eq!(previous.condition_number, None);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn malformed_scope_gauge_hash_and_factor_fail_before_commit() {
    let _hdf5 = hdf5_guard();
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
    invalid_metadata.geotransform = None;
    assert!(write_spatial_reference_covariance(
        base.with_extension("missing-geotransform.h5"),
        &invalid_metadata,
        &[block()]
    )
    .is_err());

    let mut missing_realization = block();
    missing_realization.effective_looks_receipt = None;
    assert!(write_spatial_reference_covariance(
        base.with_extension("missing-realization.h5"),
        &metadata(),
        &[missing_realization]
    )
    .is_err());

    invalid_metadata = metadata();
    invalid_metadata.producer_commit = Some("a".repeat(40));
    invalid_metadata.calibration_scope = SpatialReferenceCalibrationScope::CalibratedScopeMatch;
    invalid_metadata.review_receipt_digest = digest(0x81);
    invalid_metadata.method_manifest_digest = digest(0x82);
    invalid_metadata.calibration_scope_digest =
        spatial_reference_calibration_scope_digest(&invalid_metadata);
    invalid_metadata.source_model_digest = digest(0x83);
    assert!(write_spatial_reference_covariance(
        base.with_extension("scope.h5"),
        &invalid_metadata,
        &[block()]
    )
    .is_err());

    invalid_metadata.calibration_scope_digest =
        spatial_reference_calibration_scope_digest(&invalid_metadata);
    let mut calibrated_block = block();
    calibrated_block.approximation_error_bound = vec![0.01, 0.02];
    write_spatial_reference_covariance(
        base.with_extension("calibrated.h5"),
        &invalid_metadata,
        &[calibrated_block],
    )
    .unwrap();

    let mut cross_burst = block();
    cross_burst.source_burst_index_by_target[0] = 1;
    assert!(write_spatial_reference_covariance(
        base.with_extension("ownership.h5"),
        &metadata(),
        &[cross_burst]
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

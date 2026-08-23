use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use dolphin_io::{
    read_covariance_operator, CovarianceEstimatorBranch, CovarianceOperatorBlock,
    CovarianceOperatorGrid, CovarianceOperatorMetadata, CovarianceOperatorStatus,
    CovarianceOperatorWriter, CovariancePhaseComponent, CovariancePhaseComponentKind,
    CovarianceReplayStatus, DownstreamInferenceStatus, SourceReplayIdentity,
    StitchedCovarianceStatus, COVARIANCE_OPERATOR_METHOD, COVARIANCE_OPERATOR_METHOD_VERSION,
    COVARIANCE_OPERATOR_SCHEMA_VERSION,
};
use num_complex::Complex64;

static NEXT_PATH: AtomicU64 = AtomicU64::new(0);
static HDF5_LOCK: Mutex<()> = Mutex::new(());

fn temporary_hdf5_path() -> PathBuf {
    let sequence = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "dolphin-covariance-contract-{}-{sequence}.h5",
        std::process::id()
    ))
}

fn metadata() -> CovarianceOperatorMetadata {
    CovarianceOperatorMetadata {
        schema_version: COVARIANCE_OPERATOR_SCHEMA_VERSION,
        method: COVARIANCE_OPERATOR_METHOD.to_owned(),
        method_version: COVARIANCE_OPERATOR_METHOD_VERSION,
        crate_version: env!("CARGO_PKG_VERSION").to_owned(),
        producer_commit: None,
        gauge_date_index: 0,
        normalized_config_digest: "sha256:config".to_owned(),
        kernel_digest: "sha256:kernel".to_owned(),
        source: SourceReplayIdentity {
            manifest_digest: Some("sha256:manifest".to_owned()),
            provider: Some("fixture-provider".to_owned()),
            provider_version: Some("1".to_owned()),
            model: Some("proper-complex-tangent".to_owned()),
            model_version: Some("1".to_owned()),
            model_receipt_digest: Some("sha256:model".to_owned()),
        },
        replay_status: CovarianceReplayStatus::Replayable,
        stitched_status: StitchedCovarianceStatus::NotStitched,
        downstream_inference_status: DownstreamInferenceStatus::BlockedPendingIssue54And53,
    }
}

fn block() -> CovarianceOperatorBlock {
    CovarianceOperatorBlock {
        burst_id: "t087_185678_iw2".to_owned(),
        block_id: 2,
        generation: 2,
        native_grid: CovarianceOperatorGrid {
            row_start: 10,
            col_start: 20,
            rows: 2,
            cols: 2,
            stride_y: 1,
            stride_x: 1,
        },
        output_grid: CovarianceOperatorGrid {
            row_start: 5,
            col_start: 10,
            rows: 1,
            cols: 2,
            stride_y: 2,
            stride_x: 2,
        },
        owned_output_grid: CovarianceOperatorGrid {
            row_start: 5,
            col_start: 11,
            rows: 1,
            cols: 1,
            stride_y: 2,
            stride_x: 2,
        },
        reference_date_index: 0,
        source_date_indices: vec![3, 4],
        ordered_date_indices: vec![3, 4],
        source_ids: vec![100, 101, 102, 103],
        phase_node_ids: vec![200, 201],
        compressed_node_ids: vec![300, 301, 302, 303],
        carry_parent_ids: vec![0, 1],
        nearest_output_map: vec![0, 0, 1, 1],
        phase_components: vec![
            CovariancePhaseComponent {
                kind: CovariancePhaseComponentKind::CompressedParent,
                id: 0,
            },
            CovariancePhaseComponent {
                kind: CovariancePhaseComponentKind::CompressedParent,
                id: 1,
            },
            CovariancePhaseComponent {
                kind: CovariancePhaseComponentKind::RetainedDate,
                id: 3,
            },
            CovariancePhaseComponent {
                kind: CovariancePhaseComponentKind::RetainedDate,
                id: 4,
            },
        ],
        phase_angles: vec![0.0, 0.1, 0.2, 0.3, 0.0, 0.4, 0.5, 0.6],
        compressed_raster: vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.9, 0.1),
            Complex64::new(0.8, 0.2),
            Complex64::new(0.7, 0.3),
        ],
        compressed_status: vec![
            CovarianceOperatorStatus::Valid,
            CovarianceOperatorStatus::Masked,
            CovarianceOperatorStatus::InvalidCompression,
            CovarianceOperatorStatus::Nondifferentiable,
        ],
        projection_accumulator: vec![
            Complex64::new(3.0, 0.2),
            Complex64::new(2.9, 0.3),
            Complex64::new(2.8, 0.4),
            Complex64::new(2.7, 0.5),
        ],
        mean_amplitude: vec![1.0, 1.1, 1.2, 1.3],
        support_bits_per_output: 5,
        support_bits: vec![0b0001_1111, 0b0001_0101],
        native_validity_bits: vec![0b0000_1111],
        estimator_branch: CovarianceEstimatorBranch::Emi,
        selected_eigenvalue: vec![3.5, 3.25],
        eigen_gap: vec![1.5, 1.25],
        status: vec![
            CovarianceOperatorStatus::Valid,
            CovarianceOperatorStatus::Masked,
        ],
    }
}

#[test]
fn c52_17_block_operator_hdf5_round_trip_preserves_replay_state() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let path = temporary_hdf5_path();
    let expected_metadata = metadata();
    let expected_block = block();

    let mut writer = CovarianceOperatorWriter::create(&path, &expected_metadata).unwrap();
    writer.write_block(&expected_block).unwrap();
    writer.finish().unwrap();

    let artifact = read_covariance_operator(&path).unwrap();
    assert_eq!(artifact.metadata, expected_metadata);
    assert_eq!(artifact.blocks, vec![expected_block]);

    let file = hdf5::File::open(&path).unwrap();
    let block_group = file.group("blocks/00000000000000000002").unwrap();
    for dataset_name in block_group.member_names().unwrap() {
        assert!(
            !dataset_name.contains("incidence") && !dataset_name.contains("ancestor"),
            "expanded numeric operator leaked into {dataset_name}"
        );
    }
    let phase_angles = block_group.dataset("phase_angles").unwrap();
    assert_eq!(phase_angles.shape(), vec![2, 4]);
    assert!(phase_angles.chunk().is_some());
    assert_eq!(
        block_group
            .group("owned_output_grid")
            .unwrap()
            .attr("col_start")
            .unwrap()
            .read_scalar::<u64>()
            .unwrap(),
        11
    );

    std::fs::remove_file(path).unwrap();
}

#[test]
fn phase_component_map_rejects_a_carried_parent_mislabeled_as_a_date() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let path = temporary_hdf5_path();
    let mut invalid_block = block();
    invalid_block.phase_components[1] = CovariancePhaseComponent {
        kind: CovariancePhaseComponentKind::RetainedDate,
        id: 1,
    };

    let mut writer = CovarianceOperatorWriter::create(&path, &metadata()).unwrap();
    let error = writer.write_block(&invalid_block).unwrap_err().to_string();
    assert!(error.contains("phase component map"), "{error}");
    drop(writer);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn covariance_operator_writer_rejects_unknown_schema_and_method_versions() {
    let path = temporary_hdf5_path();
    let mut wrong_schema = metadata();
    wrong_schema.schema_version += 1;
    let error = CovarianceOperatorWriter::create(&path, &wrong_schema)
        .unwrap_err()
        .to_string();
    assert!(error.contains("schema version"), "{error}");

    let mut wrong_method = metadata();
    wrong_method.method = "rejected_temporal_factor_v0".to_owned();
    let error = CovarianceOperatorWriter::create(&path, &wrong_method)
        .unwrap_err()
        .to_string();
    assert!(error.contains("method"), "{error}");

    wrong_method = metadata();
    wrong_method.method_version += 1;
    let error = CovarianceOperatorWriter::create(&path, &wrong_method)
        .unwrap_err()
        .to_string();
    assert!(error.contains("method version"), "{error}");
}

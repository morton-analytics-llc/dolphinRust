use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use dolphin_io::{
    covariance_source_model_identity_digest, read_covariance_operator,
    read_covariance_operator_block, read_covariance_operator_block_with_receipt,
    read_covariance_operator_header_with_byte_cap, read_covariance_operator_metadata,
    read_covariance_operator_metadata_with_byte_cap, read_covariance_operator_with_byte_cap,
    CovarianceBurstPlan, CovarianceCalibrationStatus, CovarianceEstimatorBranch,
    CovarianceOperatorBlock, CovarianceOperatorGrid, CovarianceOperatorMetadata,
    CovarianceOperatorPlan, CovarianceOperatorStatus, CovarianceOperatorWriter,
    CovariancePhaseComponent, CovariancePhaseComponentKind, CovarianceRectSupport,
    CovarianceReplayStatus, CovarianceSupportOrdering, CovarianceTilePlan,
    DownstreamInferenceStatus, SourceReplayIdentity, StitchedCovarianceStatus,
    COVARIANCE_OPERATOR_METHOD, COVARIANCE_OPERATOR_METHOD_VERSION,
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

fn digest(byte: u8) -> String {
    format!("sha256:{}", format!("{byte:02x}").repeat(32))
}

fn metadata() -> CovarianceOperatorMetadata {
    CovarianceOperatorMetadata {
        schema_version: COVARIANCE_OPERATOR_SCHEMA_VERSION,
        method: COVARIANCE_OPERATOR_METHOD.to_owned(),
        method_version: COVARIANCE_OPERATOR_METHOD_VERSION,
        crate_version: env!("CARGO_PKG_VERSION").to_owned(),
        producer_commit: None,
        gauge_date_index: 0,
        normalized_config_digest: digest(1),
        kernel_digest: digest(2),
        source: SourceReplayIdentity {
            manifest_digest: Some(digest(3)),
            provider: Some("fixture-provider".to_owned()),
            provider_version: Some("1".to_owned()),
            model: Some("proper-complex-tangent".to_owned()),
            model_version: Some("1".to_owned()),
            model_version_digest: Some(format!(
                "sha256:{}",
                covariance_source_model_identity_digest(
                    "fixture-provider",
                    "1",
                    "proper-complex-tangent",
                    "1",
                )
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
            )),
            model_receipt_digest: Some(digest(4)),
        },
        replay_status: CovarianceReplayStatus::SourceModelUnavailable,
        stitched_status: StitchedCovarianceStatus::NotStitched,
        calibration_status: CovarianceCalibrationStatus::Uncalibrated,
        downstream_inference_status: DownstreamInferenceStatus::BlockedPendingIssue54And53,
    }
}

fn block() -> CovarianceOperatorBlock {
    CovarianceOperatorBlock {
        burst_id: "t087_185678_iw2".to_owned(),
        source_manifest_digest: [3; 32],
        source_model_version_digest: covariance_source_model_identity_digest(
            "fixture-provider",
            "1",
            "proper-complex-tangent",
            "1",
        ),
        block_id: 2,
        generation: 2,
        native_grid: CovarianceOperatorGrid {
            row_start: 10,
            col_start: 20,
            rows: 1,
            cols: 4,
            stride_y: 1,
            stride_x: 1,
        },
        output_grid: CovarianceOperatorGrid {
            row_start: 10,
            col_start: 10,
            rows: 1,
            cols: 2,
            stride_y: 1,
            stride_x: 2,
        },
        owned_output_grid: CovarianceOperatorGrid {
            row_start: 10,
            col_start: 11,
            rows: 1,
            cols: 1,
            stride_y: 1,
            stride_x: 2,
        },
        rect_support: CovarianceRectSupport {
            half_window_rows: 0,
            half_window_cols: 1,
            ordering: CovarianceSupportOrdering::RowMajorInwardClampV1,
        },
        branch_tolerance: 1e-6,
        reference_date_index: 0,
        source_date_indices: vec![3, 4],
        ordered_date_indices: vec![3, 4],
        source_ids: vec![100, 101, 102, 103],
        source_content_digests: vec![7; 4 * 32],
        source_factor_digests: vec![8; 4 * 32],
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
        support_bits_per_output: 3,
        support_bits: vec![0b0000_0101, 0b0000_0110],
        native_validity_bits: vec![0b0000_1101],
        estimator_branch: CovarianceEstimatorBranch::Emi,
        selected_eigenvalue: vec![3.5, 3.25],
        eigen_gap: vec![1.5, 1.25],
        status: vec![
            CovarianceOperatorStatus::Valid,
            CovarianceOperatorStatus::Masked,
        ],
    }
}

fn block_chain() -> Vec<CovarianceOperatorBlock> {
    let child = block();
    let mut root = child.clone();
    root.block_id = 0;
    root.generation = 0;
    root.source_date_indices = vec![0, 1];
    root.ordered_date_indices = vec![0, 1];
    root.carry_parent_ids.clear();
    root.phase_components = vec![
        CovariancePhaseComponent {
            kind: CovariancePhaseComponentKind::GaugeDate,
            id: 0,
        },
        CovariancePhaseComponent {
            kind: CovariancePhaseComponentKind::RetainedDate,
            id: 1,
        },
    ];
    root.phase_angles = vec![0.0, 0.1, 0.0, 0.2];
    for id in &mut root.source_ids {
        *id += 1_000;
    }
    for id in &mut root.phase_node_ids {
        *id += 1_000;
    }
    for id in &mut root.compressed_node_ids {
        *id += 1_000;
    }

    let mut middle = child.clone();
    middle.block_id = 1;
    middle.generation = 1;
    middle.source_date_indices = vec![2];
    middle.ordered_date_indices = vec![2];
    middle.carry_parent_ids = vec![0];
    middle.phase_components = vec![
        CovariancePhaseComponent {
            kind: CovariancePhaseComponentKind::CompressedParent,
            id: 0,
        },
        CovariancePhaseComponent {
            kind: CovariancePhaseComponentKind::RetainedDate,
            id: 2,
        },
    ];
    middle.phase_angles = vec![0.0, 0.15, 0.0, 0.25];
    for id in &mut middle.source_ids {
        *id += 2_000;
    }
    for id in &mut middle.phase_node_ids {
        *id += 2_000;
    }
    for id in &mut middle.compressed_node_ids {
        *id += 2_000;
    }
    vec![root, middle, child]
}

fn plan_for_blocks(blocks: &[CovarianceOperatorBlock]) -> CovarianceOperatorPlan {
    let mut bursts = BTreeMap::<String, BTreeMap<u32, Vec<u32>>>::new();
    for block in blocks {
        bursts
            .entry(block.burst_id.clone())
            .or_default()
            .entry(block.generation)
            .or_insert_with(|| block.source_date_indices.clone());
    }
    CovarianceOperatorPlan {
        source_manifest_digest: [3; 32],
        source_model_version_digest: covariance_source_model_identity_digest(
            "fixture-provider",
            "1",
            "proper-complex-tangent",
            "1",
        ),
        bursts: bursts
            .into_iter()
            .map(|(burst_id, generations)| CovarianceBurstPlan {
                tiles: blocks
                    .iter()
                    .filter(|block| block.burst_id == burst_id && block.generation == 0)
                    .map(|block| CovarianceTilePlan {
                        native_grid: block.native_grid,
                        output_grid: block.output_grid,
                        owned_output_grid: block.owned_output_grid,
                    })
                    .collect(),
                burst_id,
                source_dates_by_generation: generations.into_values().collect(),
            })
            .collect(),
    }
}

fn default_plan() -> CovarianceOperatorPlan {
    plan_for_blocks(&block_chain())
}

fn overlapping_tile_root(tile_row: u64, tile_col: u64, block_id: u64) -> CovarianceOperatorBlock {
    let side = 4_usize;
    let area = side * side;
    let row_start = tile_row * 2;
    let col_start = tile_col * 2;
    let source_ids = (0..area)
        .map(|index| {
            let row = row_start + (index / side) as u64;
            let col = col_start + (index % side) as u64;
            10_000 + row * 1_000 + col
        })
        .collect();
    let mut source_content_digests = vec![0_u8; area * 32];
    for digest in source_content_digests.chunks_exact_mut(32) {
        digest[0] = 7;
    }
    CovarianceOperatorBlock {
        burst_id: "row-major-overlap".to_owned(),
        source_manifest_digest: [3; 32],
        source_model_version_digest: covariance_source_model_identity_digest(
            "fixture-provider",
            "1",
            "proper-complex-tangent",
            "1",
        ),
        block_id,
        generation: 0,
        native_grid: CovarianceOperatorGrid {
            row_start,
            col_start,
            rows: side as u32,
            cols: side as u32,
            stride_y: 1,
            stride_x: 1,
        },
        output_grid: CovarianceOperatorGrid {
            row_start,
            col_start,
            rows: side as u32,
            cols: side as u32,
            stride_y: 1,
            stride_x: 1,
        },
        owned_output_grid: CovarianceOperatorGrid {
            row_start,
            col_start,
            rows: 2,
            cols: 2,
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
        source_ids,
        source_content_digests,
        source_factor_digests: vec![8; area * 32],
        phase_node_ids: (0..area as u64)
            .map(|index| 1_000_000 + block_id * 100 + index)
            .collect(),
        compressed_node_ids: (0..area as u64)
            .map(|index| 2_000_000 + block_id * 100 + index)
            .collect(),
        carry_parent_ids: Vec::new(),
        nearest_output_map: (0..area as u32).collect(),
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
        phase_angles: (0..area).flat_map(|_| [0.0, 0.2]).collect(),
        compressed_raster: vec![Complex64::new(1.0, 0.2); area],
        compressed_status: vec![CovarianceOperatorStatus::Valid; area],
        projection_accumulator: vec![Complex64::new(2.0, 0.3); area],
        mean_amplitude: vec![1.0; area],
        support_bits_per_output: 1,
        support_bits: vec![1; area],
        native_validity_bits: vec![0xff; area / 8],
        estimator_branch: CovarianceEstimatorBranch::Evd,
        selected_eigenvalue: vec![1.0; area],
        eigen_gap: vec![0.5; area],
        status: vec![CovarianceOperatorStatus::Valid; area],
    }
}

fn write_artifact(path: &PathBuf, blocks: &[CovarianceOperatorBlock]) {
    let mut writer =
        CovarianceOperatorWriter::create(path, &metadata(), &plan_for_blocks(blocks)).unwrap();
    for block in blocks {
        writer.write_block(block).unwrap();
    }
    writer.finish().unwrap();
}

#[test]
fn c52_17_block_operator_hdf5_round_trip_preserves_replay_state() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let path = temporary_hdf5_path();
    let expected_metadata = metadata();
    let expected_blocks = block_chain();
    write_artifact(&path, &expected_blocks);

    let artifact = read_covariance_operator(&path).unwrap();
    assert_eq!(artifact.metadata, expected_metadata);
    assert_eq!(artifact.blocks, expected_blocks);
    let receipt = read_covariance_operator_block_with_receipt(&path, 2, u64::MAX).unwrap();
    assert_eq!(receipt.block, expected_blocks[2]);
    assert_eq!(receipt.logical_payload_bytes, 774);
    assert_eq!(
        read_covariance_operator_block(&path, 2, u64::MAX).unwrap(),
        expected_blocks[2]
    );

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

    let _ = std::fs::remove_file(path);
}

#[test]
fn replayable_writer_rejects_noncanonical_source_and_node_ids() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let path = temporary_hdf5_path();
    let blocks = block_chain();
    let mut replayable = metadata();
    replayable.replay_status = CovarianceReplayStatus::Replayable;
    let mut writer =
        CovarianceOperatorWriter::create(&path, &replayable, &plan_for_blocks(&blocks)).unwrap();
    let error = writer.write_block(&blocks[0]).unwrap_err().to_string();
    assert!(error.contains("canonically derived"), "{error}");
    drop(writer);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn phase_component_map_rejects_a_carried_parent_mislabeled_as_a_date() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let path = temporary_hdf5_path();
    let mut blocks = block_chain();
    let mut invalid_block = blocks.pop().unwrap();
    invalid_block.phase_components[1] = CovariancePhaseComponent {
        kind: CovariancePhaseComponentKind::RetainedDate,
        id: 1,
    };

    let mut writer = CovarianceOperatorWriter::create(&path, &metadata(), &default_plan()).unwrap();
    for ancestor in blocks {
        writer.write_block(&ancestor).unwrap();
    }
    let error = writer.write_block(&invalid_block).unwrap_err().to_string();
    assert!(error.contains("phase component map"), "{error}");
    drop(writer);
    let _ = std::fs::remove_file(path);
}

#[test]
fn metadata_reader_checks_completion_and_registries_without_loading_blocks() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let incomplete_path = temporary_hdf5_path();
    let writer =
        CovarianceOperatorWriter::create(&incomplete_path, &metadata(), &default_plan()).unwrap();
    drop(writer);
    let error = read_covariance_operator_metadata(&incomplete_path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("incomplete"), "{error}");
    std::fs::remove_file(incomplete_path).unwrap();

    let corrupt_path = temporary_hdf5_path();
    write_artifact(&corrupt_path, &block_chain());
    let file = hdf5::File::open_rw(&corrupt_path).unwrap();
    file.dataset("registries/method_codes")
        .unwrap()
        .write_raw(&[99_u16])
        .unwrap();
    file.flush().unwrap();
    drop(file);
    let error = read_covariance_operator_metadata(&corrupt_path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("registry mismatch"), "{error}");
    std::fs::remove_file(corrupt_path).unwrap();
}

#[test]
fn finalization_and_metadata_reader_require_a_nonempty_valid_topology() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let empty_path = temporary_hdf5_path();
    let writer =
        CovarianceOperatorWriter::create(&empty_path, &metadata(), &default_plan()).unwrap();
    let error = writer.finish().unwrap_err().to_string();
    assert!(error.contains("at least one block"), "{error}");
    std::fs::remove_file(empty_path).unwrap();

    let orphan_path = temporary_hdf5_path();
    write_artifact(&orphan_path, &block_chain());
    let file = hdf5::File::open_rw(&orphan_path).unwrap();
    file.dataset("blocks/00000000000000000002/carry_parent_ids")
        .unwrap()
        .write_raw(&[0_u64, 9_u64])
        .unwrap();
    file.flush().unwrap();
    drop(file);
    let error = read_covariance_operator_metadata(&orphan_path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("parent"), "{error}");
    std::fs::remove_file(orphan_path).unwrap();
}

#[test]
fn masked_outputs_may_retain_realized_support_for_replay() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let path = temporary_hdf5_path();
    let blocks = block_chain();
    write_artifact(&path, &blocks);
    assert_eq!(read_covariance_operator(&path).unwrap().blocks, blocks);
    std::fs::remove_file(path).unwrap();
}

fn write_first_block_error(mut block: CovarianceOperatorBlock) -> String {
    block.block_id = 0;
    block.generation = 0;
    block.source_date_indices = vec![0, 1];
    block.ordered_date_indices = vec![0, 1];
    block.carry_parent_ids.clear();
    block.phase_components = vec![
        CovariancePhaseComponent {
            kind: CovariancePhaseComponentKind::GaugeDate,
            id: 0,
        },
        CovariancePhaseComponent {
            kind: CovariancePhaseComponentKind::RetainedDate,
            id: 1,
        },
    ];
    block.phase_angles = vec![0.0, 0.1, 0.0, 0.2];
    let path = temporary_hdf5_path();
    let mut writer = CovarianceOperatorWriter::create(
        &path,
        &metadata(),
        &plan_for_blocks(std::slice::from_ref(&block)),
    )
    .unwrap();
    let error = writer.write_block(&block).unwrap_err().to_string();
    drop(writer);
    std::fs::remove_file(path).unwrap();
    error
}

#[test]
fn rect_geometry_support_and_nearest_map_are_exact_contracts() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let mut cases: Vec<(&str, CovarianceOperatorBlock)> = Vec::new();

    let mut native_stride = block();
    native_stride.native_grid.stride_x = 2;
    cases.push(("native grid stride", native_stride));

    let mut output_shape = block();
    output_shape.output_grid.stride_x = 1;
    output_shape.owned_output_grid.stride_x = 1;
    cases.push(("output grid shape", output_shape));

    let mut origin = block();
    origin.output_grid.col_start = 11;
    origin.owned_output_grid.col_start = 11;
    cases.push(("grid origins", origin));

    let mut oversized_window = block();
    oversized_window.rect_support.half_window_cols = 2;
    oversized_window.support_bits_per_output = 5;
    oversized_window.support_bits = vec![0b0001_1111, 0b0001_1111];
    cases.push(("Rect window", oversized_window));

    let mut support = block();
    support.support_bits[0] ^= 0b10;
    cases.push(("support bits", support));

    let mut nearest = block();
    nearest.nearest_output_map = vec![0, 1, 0, 1];
    cases.push(("nearest-output map", nearest));

    for (label, invalid) in cases {
        let error = write_first_block_error(invalid);
        assert!(error.contains(label), "{label}: {error}");
    }
}

fn offset_ids(block: &mut CovarianceOperatorBlock, offset: u64) {
    for id in &mut block.phase_node_ids {
        *id += offset;
    }
    for id in &mut block.compressed_node_ids {
        *id += offset;
    }
}

#[test]
fn topology_rejects_repeated_or_skipped_dates_across_generations() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    for date in [1, 3] {
        let path = temporary_hdf5_path();
        let mut blocks = block_chain();
        let root = blocks.remove(0);
        let mut child = blocks.remove(0);
        child.source_date_indices = vec![date];
        child.ordered_date_indices = vec![date];
        child.phase_components[1].id = u64::from(date);
        let mut writer =
            CovarianceOperatorWriter::create(&path, &metadata(), &default_plan()).unwrap();
        writer.write_block(&root).unwrap();
        let error = writer.write_block(&child).unwrap_err().to_string();
        assert!(
            error.contains("date chronology") || error.contains("date map differ"),
            "{date}: {error}"
        );
        drop(writer);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn topology_rejects_cross_tile_identity_ownership_and_burst_conflicts() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let root = block_chain().remove(0);

    let path = temporary_hdf5_path();
    let mut conflicting_source = root.clone();
    conflicting_source.block_id = 10;
    conflicting_source.owned_output_grid.col_start = 11;
    conflicting_source.source_ids[0] = 999;
    offset_ids(&mut conflicting_source, 1_000);
    let mut first = root.clone();
    first.owned_output_grid.col_start = 10;
    let planned = [first.clone(), conflicting_source.clone()];
    let mut writer =
        CovarianceOperatorWriter::create(&path, &metadata(), &plan_for_blocks(&planned)).unwrap();
    writer.write_block(&first).unwrap();
    let error = writer
        .write_block(&conflicting_source)
        .unwrap_err()
        .to_string();
    assert!(error.contains("shared native source"), "{error}");
    drop(writer);
    std::fs::remove_file(path).unwrap();

    let path = temporary_hdf5_path();
    let mut conflicting_digest = root.clone();
    conflicting_digest.block_id = 10;
    conflicting_digest.owned_output_grid.col_start = 11;
    conflicting_digest.source_content_digests[0] ^= 0xff;
    offset_ids(&mut conflicting_digest, 1_000);
    let mut first = root.clone();
    first.owned_output_grid.col_start = 10;
    let planned = [first.clone(), conflicting_digest.clone()];
    let mut writer =
        CovarianceOperatorWriter::create(&path, &metadata(), &plan_for_blocks(&planned)).unwrap();
    writer.write_block(&first).unwrap();
    let error = writer
        .write_block(&conflicting_digest)
        .unwrap_err()
        .to_string();
    assert!(error.contains("content digest"), "{error}");
    drop(writer);
    std::fs::remove_file(path).unwrap();

    let path = temporary_hdf5_path();
    let mut conflicting_factor = root.clone();
    conflicting_factor.block_id = 10;
    conflicting_factor.owned_output_grid.col_start = 11;
    conflicting_factor.source_factor_digests[0] ^= 0xff;
    offset_ids(&mut conflicting_factor, 1_000);
    let mut first = root.clone();
    first.owned_output_grid.col_start = 10;
    let planned = [first.clone(), conflicting_factor.clone()];
    let mut writer =
        CovarianceOperatorWriter::create(&path, &metadata(), &plan_for_blocks(&planned)).unwrap();
    writer.write_block(&first).unwrap();
    let error = writer
        .write_block(&conflicting_factor)
        .unwrap_err()
        .to_string();
    assert!(error.contains("numeric factor receipt"), "{error}");
    drop(writer);
    std::fs::remove_file(path).unwrap();

    let path = temporary_hdf5_path();
    let mut conflicting_validity = root.clone();
    conflicting_validity.block_id = 10;
    conflicting_validity.owned_output_grid.col_start = 11;
    conflicting_validity.native_validity_bits = vec![0b0000_1100];
    conflicting_validity.support_bits[0] = 0b0000_0100;
    conflicting_validity.compressed_status[0] = CovarianceOperatorStatus::Masked;
    offset_ids(&mut conflicting_validity, 1_000);
    let mut first = root.clone();
    first.owned_output_grid.col_start = 10;
    let planned = [first.clone(), conflicting_validity.clone()];
    let mut writer =
        CovarianceOperatorWriter::create(&path, &metadata(), &plan_for_blocks(&planned)).unwrap();
    writer.write_block(&first).unwrap();
    let error = writer
        .write_block(&conflicting_validity)
        .unwrap_err()
        .to_string();
    assert!(error.contains("different validity"), "{error}");
    drop(writer);
    std::fs::remove_file(path).unwrap();

    let path = temporary_hdf5_path();
    let mut reused_id = root.clone();
    reused_id.block_id = 10;
    reused_id.native_grid.col_start = 24;
    reused_id.output_grid.col_start = 12;
    reused_id.owned_output_grid.col_start = 12;
    offset_ids(&mut reused_id, 1_000);
    let planned = [root.clone(), reused_id.clone()];
    let mut writer =
        CovarianceOperatorWriter::create(&path, &metadata(), &plan_for_blocks(&planned)).unwrap();
    writer.write_block(&root).unwrap();
    let error = writer.write_block(&reused_id).unwrap_err().to_string();
    assert!(error.contains("source ID"), "{error}");
    drop(writer);
    std::fs::remove_file(path).unwrap();

    let path = temporary_hdf5_path();
    let mut overlapping_owner = root.clone();
    overlapping_owner.block_id = 10;
    offset_ids(&mut overlapping_owner, 1_000);
    let planned = [root.clone(), overlapping_owner];
    let error = CovarianceOperatorWriter::create(&path, &metadata(), &plan_for_blocks(&planned))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("owned output grids overlap") || error.contains("strict row-major"),
        "{error}"
    );
    let _ = std::fs::remove_file(path);

    let path = temporary_hdf5_path();
    let mut second_burst = root.clone();
    second_burst.block_id = 10;
    second_burst.burst_id = "second-burst".to_owned();
    second_burst.native_grid.col_start = 24;
    second_burst.output_grid.col_start = 12;
    second_burst.owned_output_grid.col_start = 12;
    for id in &mut second_burst.source_ids {
        *id += 2_000;
    }
    offset_ids(&mut second_burst, 2_000);
    let planned = [root.clone(), second_burst];
    let error = CovarianceOperatorWriter::create(&path, &metadata(), &plan_for_blocks(&planned))
        .unwrap_err()
        .to_string();
    assert!(error.contains("stitched status"), "{error}");
    let _ = std::fs::remove_file(path);

    let path = temporary_hdf5_path();
    let mut inconsistent_dates = root.clone();
    inconsistent_dates.block_id = 10;
    inconsistent_dates.native_grid.col_start = 24;
    inconsistent_dates.output_grid.col_start = 12;
    inconsistent_dates.owned_output_grid.col_start = 12;
    inconsistent_dates.source_date_indices = vec![0, 2];
    inconsistent_dates.ordered_date_indices = vec![0, 2];
    inconsistent_dates.phase_components[1].id = 2;
    for id in &mut inconsistent_dates.source_ids {
        *id += 3_000;
    }
    offset_ids(&mut inconsistent_dates, 3_000);
    let planned = [root.clone(), inconsistent_dates.clone()];
    let mut writer =
        CovarianceOperatorWriter::create(&path, &metadata(), &plan_for_blocks(&planned)).unwrap();
    writer.write_block(&root).unwrap();
    let error = writer
        .write_block(&inconsistent_dates)
        .unwrap_err()
        .to_string();
    assert!(error.contains("contiguous ordered range"), "{error}");
    drop(writer);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn topology_rejects_cross_tile_geometry_drift() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let root = block_chain().remove(0);
    let mut mismatched_geometry = root.clone();
    mismatched_geometry.block_id = 10;
    mismatched_geometry.native_grid.col_start = 24;
    mismatched_geometry.native_grid.cols = 2;
    mismatched_geometry.output_grid.col_start = 24;
    mismatched_geometry.output_grid.stride_x = 1;
    mismatched_geometry.owned_output_grid = mismatched_geometry.output_grid;
    mismatched_geometry.rect_support.half_window_cols = 0;
    mismatched_geometry.source_ids = vec![5_100, 5_101];
    mismatched_geometry.source_content_digests = vec![7; 2 * 32];
    mismatched_geometry.source_factor_digests = vec![8; 2 * 32];
    mismatched_geometry.compressed_node_ids = vec![5_300, 5_301];
    mismatched_geometry.nearest_output_map = vec![0, 1];
    mismatched_geometry.compressed_raster = vec![Complex64::new(1.0, 0.0); 2];
    mismatched_geometry.compressed_status = vec![CovarianceOperatorStatus::Valid; 2];
    mismatched_geometry.projection_accumulator = vec![Complex64::new(3.0, 0.2); 2];
    mismatched_geometry.mean_amplitude = vec![1.0; 2];
    mismatched_geometry.support_bits_per_output = 1;
    mismatched_geometry.support_bits = vec![1, 1];
    mismatched_geometry.native_validity_bits = vec![0b11];
    mismatched_geometry.status = vec![CovarianceOperatorStatus::Valid; 2];
    offset_ids(&mut mismatched_geometry, 5_000);
    let path = temporary_hdf5_path();
    let planned = [root.clone(), mismatched_geometry.clone()];
    let mut writer =
        CovarianceOperatorWriter::create(&path, &metadata(), &plan_for_blocks(&planned)).unwrap();
    writer.write_block(&root).unwrap();
    let error = writer
        .write_block(&mismatched_geometry)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("geometry, estimator branch, or tolerance"),
        "{error}"
    );
    drop(writer);
    std::fs::remove_file(path).unwrap();

    for mutate in ["branch", "tolerance"] {
        let mut drift = root.clone();
        drift.block_id = 10;
        drift.native_grid.col_start = 24;
        drift.output_grid.col_start = 12;
        drift.owned_output_grid.col_start = 12;
        for id in &mut drift.source_ids {
            *id += 10_000;
        }
        offset_ids(&mut drift, 10_000);
        match mutate {
            "branch" => drift.estimator_branch = CovarianceEstimatorBranch::Evd,
            "tolerance" => drift.branch_tolerance = 1e-3,
            _ => unreachable!(),
        }
        let path = temporary_hdf5_path();
        let planned = [root.clone(), drift.clone()];
        let mut writer =
            CovarianceOperatorWriter::create(&path, &metadata(), &plan_for_blocks(&planned))
                .unwrap();
        writer.write_block(&root).unwrap();
        let error = writer.write_block(&drift).unwrap_err().to_string();
        assert!(error.contains("estimator branch, or tolerance"), "{error}");
        drop(writer);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn valid_compression_rejects_the_zero_phase_nodata_branch() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    for phase in [0.0, 1e-6] {
        let mut invalid = block();
        invalid.projection_accumulator[0] = Complex64::from_polar(3.0, phase);
        let error = write_first_block_error(invalid);
        assert!(error.contains("projection phase"), "{phase}: {error}");
    }
}

#[test]
fn writer_and_reader_reject_orphaned_or_non_immediate_parent_topology() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let orphan_path = temporary_hdf5_path();
    let mut writer =
        CovarianceOperatorWriter::create(&orphan_path, &metadata(), &default_plan()).unwrap();
    let error = writer.write_block(&block()).unwrap_err().to_string();
    assert!(error.contains("parent"), "{error}");
    drop(writer);
    std::fs::remove_file(orphan_path).unwrap();

    let corrupt_path = temporary_hdf5_path();
    write_artifact(&corrupt_path, &block_chain());
    let file = hdf5::File::open_rw(&corrupt_path).unwrap();
    file.dataset("blocks/00000000000000000002/carry_parent_ids")
        .unwrap()
        .write_raw(&[0_u64, 9_u64])
        .unwrap();
    file.flush().unwrap();
    drop(file);
    let error = read_covariance_operator(&corrupt_path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("parent"), "{error}");
    std::fs::remove_file(corrupt_path).unwrap();
}

#[test]
fn valid_nodes_reject_nan_nonzero_gauge_and_eigen_tie() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    for (label, mutate) in [
        (
            "nonfinite",
            (|block: &mut CovarianceOperatorBlock| block.phase_angles[1] = f64::NAN)
                as fn(&mut CovarianceOperatorBlock),
        ),
        (
            "reference phase",
            (|block: &mut CovarianceOperatorBlock| block.phase_angles[0] = 0.01)
                as fn(&mut CovarianceOperatorBlock),
        ),
        (
            "eigen gap",
            (|block: &mut CovarianceOperatorBlock| block.eigen_gap[0] = block.branch_tolerance)
                as fn(&mut CovarianceOperatorBlock),
        ),
    ] {
        let path = temporary_hdf5_path();
        let mut blocks = block_chain();
        let mut invalid_block = blocks.pop().unwrap();
        mutate(&mut invalid_block);
        let mut writer =
            CovarianceOperatorWriter::create(&path, &metadata(), &default_plan()).unwrap();
        for ancestor in blocks {
            writer.write_block(&ancestor).unwrap();
        }
        let error = writer.write_block(&invalid_block).unwrap_err().to_string();
        assert!(error.contains(label), "{label}: {error}");
        drop(writer);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn capped_reader_rejects_transposed_shapes_and_budget_before_loading_values() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let path = temporary_hdf5_path();
    write_artifact(&path, &block_chain());

    let error = read_covariance_operator_block(&path, 2, 1)
        .unwrap_err()
        .to_string();
    assert!(error.contains("byte cap"), "{error}");
    let error = read_covariance_operator_with_byte_cap(&path, 1)
        .unwrap_err()
        .to_string();
    assert!(error.contains("byte cap"), "{error}");

    let file = hdf5::File::open_rw(&path).unwrap();
    let group = file.group("blocks/00000000000000000002").unwrap();
    let values = group
        .dataset("phase_angles")
        .unwrap()
        .read_raw::<f64>()
        .unwrap();
    group.unlink("phase_angles").unwrap();
    group
        .new_dataset::<f64>()
        .shape((4, 2))
        .create("phase_angles")
        .unwrap()
        .write_raw(&values)
        .unwrap();
    file.flush().unwrap();
    drop(file);
    let error = read_covariance_operator_metadata(&path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("phase_angles shape"), "{error}");
    let error = read_covariance_operator_block(&path, 2, u64::MAX)
        .unwrap_err()
        .to_string();
    assert!(error.contains("phase_angles shape"), "{error}");
    std::fs::remove_file(path).unwrap();
}

fn replace_u8_dataset(file: &hdf5::File, group_name: &str, name: &str, values: &[u8]) {
    let group = file.group(group_name).unwrap();
    group.unlink(name).unwrap();
    group
        .new_dataset::<u8>()
        .shape(values.len())
        .create(name)
        .unwrap()
        .write_raw(values)
        .unwrap();
}

#[test]
fn metadata_reader_rejects_missing_or_unexpected_recursive_schema_members() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();

    let missing_path = temporary_hdf5_path();
    write_artifact(&missing_path, &block_chain());
    let file = hdf5::File::open_rw(&missing_path).unwrap();
    file.group("blocks/00000000000000000002")
        .unwrap()
        .unlink("phase_angles")
        .unwrap();
    file.flush().unwrap();
    drop(file);
    let error = read_covariance_operator_metadata(&missing_path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("missing or unexpected"), "{error}");
    std::fs::remove_file(missing_path).unwrap();

    let root_extra_path = temporary_hdf5_path();
    write_artifact(&root_extra_path, &block_chain());
    let file = hdf5::File::open_rw(&root_extra_path).unwrap();
    file.new_dataset::<f64>()
        .shape((2, 2, 2, 2))
        .create("incidence")
        .unwrap();
    file.flush().unwrap();
    drop(file);
    let error = read_covariance_operator_metadata(&root_extra_path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("root schema"), "{error}");
    std::fs::remove_file(root_extra_path).unwrap();

    let soft_link_path = temporary_hdf5_path();
    write_artifact(&soft_link_path, &block_chain());
    let file = hdf5::File::open_rw(&soft_link_path).unwrap();
    file.unlink("method").unwrap();
    file.link_soft("crate_version", "method").unwrap();
    file.flush().unwrap();
    drop(file);
    let error = read_covariance_operator_metadata(&soft_link_path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("root schema"), "{error}");
    std::fs::remove_file(soft_link_path).unwrap();

    let nested_extra_path = temporary_hdf5_path();
    write_artifact(&nested_extra_path, &block_chain());
    let file = hdf5::File::open_rw(&nested_extra_path).unwrap();
    file.group("blocks/00000000000000000002/native_grid")
        .unwrap()
        .new_dataset::<f64>()
        .shape((2, 2, 2, 2))
        .create("ancestor_coefficients")
        .unwrap();
    file.flush().unwrap();
    drop(file);
    let error = read_covariance_operator_metadata(&nested_extra_path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("grid schema"), "{error}");
    std::fs::remove_file(nested_extra_path).unwrap();
}

#[test]
fn full_metadata_reader_budgets_global_topology_but_block_reads_are_local() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();

    let metadata_path = temporary_hdf5_path();
    write_artifact(&metadata_path, &block_chain());
    let file = hdf5::File::open_rw(&metadata_path).unwrap();
    replace_u8_dataset(&file, "/", "method", &vec![b'x'; 8_192]);
    file.flush().unwrap();
    drop(file);
    let error = read_covariance_operator_metadata_with_byte_cap(&metadata_path, 4_096)
        .unwrap_err()
        .to_string();
    assert!(error.contains("byte cap"), "{error}");
    std::fs::remove_file(metadata_path).unwrap();

    let names_path = temporary_hdf5_path();
    write_artifact(&names_path, &block_chain());
    let file = hdf5::File::open_rw(&names_path).unwrap();
    file.group("blocks")
        .unwrap()
        .create_group(&"9".repeat(8_192))
        .unwrap();
    file.flush().unwrap();
    drop(file);
    let error = read_covariance_operator_metadata_with_byte_cap(&names_path, 4_096)
        .unwrap_err()
        .to_string();
    assert!(error.contains("byte cap"), "{error}");
    std::fs::remove_file(names_path).unwrap();

    let topology_path = temporary_hdf5_path();
    write_artifact(&topology_path, &block_chain());
    let file = hdf5::File::open_rw(&topology_path).unwrap();
    replace_u8_dataset(
        &file,
        "blocks/00000000000000000000",
        "burst_id",
        &vec![b'b'; 8_192],
    );
    file.flush().unwrap();
    drop(file);
    let selected = read_covariance_operator_block(&topology_path, 2, 16 * 1024).unwrap();
    assert_eq!(selected.block_id, 2);
    std::fs::remove_file(topology_path).unwrap();
}

#[test]
fn writer_retains_only_the_current_spatial_tile_chain() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let path = temporary_hdf5_path();
    let mut blocks = Vec::new();
    for tile in 0..16_u64 {
        let block_offset = tile * 10;
        let id_offset = tile * 100_000;
        let col_offset = tile * 100;
        for mut block in block_chain() {
            block.block_id += block_offset;
            for parent in &mut block.carry_parent_ids {
                *parent += block_offset;
            }
            for component in &mut block.phase_components {
                if component.kind == CovariancePhaseComponentKind::CompressedParent {
                    component.id += block_offset;
                }
            }
            block.native_grid.col_start += col_offset * 2;
            block.output_grid.col_start += col_offset;
            block.owned_output_grid.col_start += col_offset;
            for id in &mut block.source_ids {
                *id += id_offset;
            }
            offset_ids(&mut block, id_offset);
            blocks.push(block);
        }
    }
    let mut writer =
        CovarianceOperatorWriter::create(&path, &metadata(), &plan_for_blocks(&blocks)).unwrap();
    let mut maximum_retained = 0;
    for block in &blocks {
        writer.write_block(block).unwrap();
        maximum_retained = maximum_retained.max(writer.retained_topology_block_count());
    }
    assert_eq!(maximum_retained, 3);
    assert_eq!(writer.retained_topology_block_count(), 3);
    writer.finish().unwrap();
    std::fs::remove_file(path).unwrap();
}

#[test]
fn writer_disk_checks_vertical_overlap_older_than_the_previous_tile() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let path = temporary_hdf5_path();
    let horizontal = (0..3)
        .map(|col| overlapping_tile_root(0, col, col + 1))
        .collect::<Vec<_>>();
    let mut vertical = overlapping_tile_root(1, 0, 10);
    vertical.source_content_digests[0] ^= 0xff;
    let mut planned = horizontal.clone();
    planned.push(vertical.clone());
    let mut writer =
        CovarianceOperatorWriter::create(&path, &metadata(), &plan_for_blocks(&planned)).unwrap();
    for block in &horizontal {
        writer.write_block(block).unwrap();
    }
    let error = writer.write_block(&vertical).unwrap_err().to_string();
    assert!(error.contains("content digest"), "{error}");
    drop(writer);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn header_and_selected_block_reject_soft_linked_payloads() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let path = temporary_hdf5_path();
    write_artifact(&path, &block_chain());
    let file = hdf5::File::open_rw(&path).unwrap();
    let selected = "00000000000000000002";
    let blocks = file.group("blocks").unwrap();
    blocks.unlink(selected).unwrap();
    blocks.link_soft("00000000000000000001", selected).unwrap();
    file.flush().unwrap();
    drop(file);

    let header_error = read_covariance_operator_header_with_byte_cap(&path, 64 * 1024)
        .unwrap_err()
        .to_string();
    assert!(header_error.contains("hard link"), "{header_error}");
    let block_error = read_covariance_operator_block(&path, 2, 64 * 1024)
        .unwrap_err()
        .to_string();
    assert!(block_error.contains("hard link"), "{block_error}");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn metadata_and_registry_string_datasets_require_canonical_shapes_and_types() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let path = temporary_hdf5_path();
    write_artifact(&path, &block_chain());
    let file = hdf5::File::open_rw(&path).unwrap();
    let method = file.dataset("method").unwrap().read_raw::<u8>().unwrap();
    file.unlink("method").unwrap();
    file.new_dataset::<u8>()
        .shape((1, method.len()))
        .create("method")
        .unwrap()
        .write_raw(&method)
        .unwrap();
    file.flush().unwrap();
    drop(file);
    let error = read_covariance_operator_metadata(&path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("method shape"), "{error}");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn covariance_operator_writer_rejects_unknown_schema_and_method_versions() {
    let _hdf5 = HDF5_LOCK.lock().unwrap();
    let path = temporary_hdf5_path();
    let mut wrong_schema = metadata();
    wrong_schema.schema_version += 1;
    let error = CovarianceOperatorWriter::create(&path, &wrong_schema, &default_plan())
        .unwrap_err()
        .to_string();
    assert!(error.contains("schema version"), "{error}");

    let mut wrong_method = metadata();
    wrong_method.method = "rejected_temporal_factor_v0".to_owned();
    let error = CovarianceOperatorWriter::create(&path, &wrong_method, &default_plan())
        .unwrap_err()
        .to_string();
    assert!(error.contains("method"), "{error}");

    wrong_method = metadata();
    wrong_method.method_version += 1;
    let error = CovarianceOperatorWriter::create(&path, &wrong_method, &default_plan())
        .unwrap_err()
        .to_string();
    assert!(error.contains("method version"), "{error}");

    let mut weak_digest = metadata();
    weak_digest.source.manifest_digest = Some("sha256:manifest".to_owned());
    let error = CovarianceOperatorWriter::create(&path, &weak_digest, &default_plan())
        .unwrap_err()
        .to_string();
    assert!(error.contains("SHA-256"), "{error}");
}

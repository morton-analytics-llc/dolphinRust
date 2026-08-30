//! Bounded release-resource benchmark for temporal covariance products.

use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{ensure, Context, Result};
use dolphin_core::{config::TemporalUncertaintyOptions, BlockIndices};
use dolphin_io::{
    spatial_reference_runtime_resource_receipt_digest, BoundedCogWriter, CovarianceOperatorGrid,
    SpatialReferenceCalibrationScope, SpatialReferenceCovarianceBlock,
    SpatialReferenceCovarianceMetadata, SpatialReferenceCovarianceStatus,
    SpatialReferenceCovarianceWriter, SpatialReferenceRuntimeResourceReceipt,
    SPATIAL_REFERENCE_COVARIANCE_METHOD, SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
};
use dolphin_timeseries::TemporalScalarCandidateMethod;
use dolphin_workflows::temporal_covariance_product::{
    run_temporal_scalar_candidate_resource_probe, temporal_inference_resource_receipt,
    TemporalInferenceBinaryIdentity, TemporalInferenceHostIdentity,
    TemporalInferenceResourceMeasurement, TemporalInferenceScalarMeasurement,
    TEMPORAL_DIRECT_FACTOR_RECEIPT_FILENAME, TEMPORAL_RESOURCE_TILE_COLUMNS,
    TEMPORAL_RESOURCE_TILE_ROWS,
};
use dolphin_workflows::SPATIAL_REFERENCE_COVARIANCE_FILENAME;
use ndarray::Array2;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const BINARY_CAP_BYTES: u64 = 256 * 1024 * 1024;
const RECEIPT_CAP_BYTES: usize = 4 * 1024 * 1024;
const CASES: [u64; 3] = [12, 48, 96];
const FINGERPRINT_PERIOD: usize = 257;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DirectFactorReceipt {
    schema: String,
    acquisition_count: usize,
    post_gauge_date_count: usize,
    source_manifest_sha256: String,
    issue52_operator_receipt_sha256: String,
    issue52_source_factor_receipt_sha256: String,
    issue54_fixed_l2_map_receipt_sha256: String,
    issue54_replay_source_factor_receipt_sha256: String,
    issue54_replay_support_receipt_sha256: String,
    reference_signature_sha256: String,
    factor_sha256: String,
    realized_rank: usize,
    covariance_condition_number: f64,
    positive_off_diagonal_energy: f64,
    replay_resource_high_water_bytes: u64,
    effective_looks_fraction: f64,
    source_correlation_support_union_count: usize,
    source_correlation_receipt_sha256: String,
    difference_covariance_sha256: String,
    difference_factor: Vec<Vec<f64>>,
    factor_generation_count: usize,
}

fn main() -> Result<()> {
    let arguments = std::env::args_os().collect::<Vec<_>>();
    if arguments
        .get(1)
        .is_some_and(|value| value == "--scalar-probe")
    {
        ensure!(arguments.len() == 5, "invalid scalar-probe arguments");
        let dates = arguments[2]
            .to_string_lossy()
            .parse::<u64>()
            .context("parsing scalar-probe date count")?;
        let method = parse_method(&arguments[4].to_string_lossy())?;
        let measurement =
            run_temporal_scalar_candidate_resource_probe(Path::new(&arguments[3]), dates, method)?;
        serde_json::to_writer(std::io::stdout().lock(), &measurement)?;
        return Ok(());
    }
    ensure!(
        matches!(arguments.len(), 1 | 3)
            && (arguments.len() == 1
                || arguments
                    .get(1)
                    .is_some_and(|value| value == "--pre-outcome-selection-receipt")),
        "temporal benchmark accepts only --pre-outcome-selection-receipt PATH"
    );

    let executable = std::env::current_exe().context("locating benchmark binary")?;
    let batch_binary = executable
        .parent()
        .context("benchmark binary has no parent")?
        .join(format!(
            "temporal_covariance_batch{}",
            std::env::consts::EXE_SUFFIX
        ));
    let selection_sha256 = arguments
        .get(2)
        .map(|path| read_bounded(Path::new(path), RECEIPT_CAP_BYTES as u64))
        .transpose()?
        .map(|bytes| sha256(&bytes));
    let benchmark_identity = observed_binary_identity(&executable)?;
    let batch_identity = observed_binary_identity(&batch_binary)?;
    let scratch = BenchmarkScratch::create()?;
    let worker_count = std::thread::available_parallelism()?.get();
    let mut measurements = Vec::with_capacity(CASES.len());
    for dates in CASES {
        measurements.push(run_case(
            &executable,
            &batch_binary,
            scratch.path(),
            dates,
            worker_count,
        )?);
    }
    let receipt = temporal_inference_resource_receipt(
        batch_identity,
        benchmark_identity,
        TemporalInferenceHostIdentity {
            operating_system: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            logical_processor_count: u64::try_from(worker_count)?,
            rayon_thread_count: u64::try_from(worker_count)?,
            omp_thread_count: 1,
            openblas_thread_count: 1,
            mkl_thread_count: 1,
            veclib_thread_count: 1,
        },
        selection_sha256,
        measurements,
    )?;
    let bytes = serde_json::to_vec(&receipt)?;
    ensure!(bytes.len() <= RECEIPT_CAP_BYTES, "receipt exceeds byte cap");
    let mut output = std::io::stdout().lock();
    output.write_all(&bytes)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn parse_method(value: &str) -> Result<TemporalScalarCandidateMethod> {
    match value {
        "plugin_gls_reml" => Ok(TemporalScalarCandidateMethod::PluginGlsReml),
        "reml_covariance_parameter_adjusted_scalar" => {
            Ok(TemporalScalarCandidateMethod::RemlCovarianceParameterAdjustedScalar)
        }
        _ => anyhow::bail!("unsupported temporal scalar method"),
    }
}

fn run_case(
    executable: &Path,
    batch_binary: &Path,
    scratch_root: &Path,
    dates: u64,
    worker_count: usize,
) -> Result<TemporalInferenceResourceMeasurement> {
    let case_directory = scratch_root.join(format!("post-gauge-{dates}"));
    let fixture = case_directory.join("fixture");
    std::fs::create_dir_all(&fixture)?;
    let varied_target_fingerprint_count = prepare_fixture(batch_binary, &fixture, dates)?;
    let orders = [
        TemporalScalarCandidateMethod::PluginGlsReml,
        TemporalScalarCandidateMethod::RemlCovarianceParameterAdjustedScalar,
        TemporalScalarCandidateMethod::RemlCovarianceParameterAdjustedScalar,
        TemporalScalarCandidateMethod::PluginGlsReml,
    ];
    let mut plugin_trials = Vec::with_capacity(2);
    let mut adjusted_trials = Vec::with_capacity(2);
    for (trial, method) in orders.into_iter().enumerate() {
        let trial_directory = case_directory.join(format!("trial-{trial}"));
        link_fixture(&fixture, &trial_directory, dates)?;
        let measurement =
            run_scalar_trial(executable, &trial_directory, dates, method, worker_count)?;
        match method {
            TemporalScalarCandidateMethod::PluginGlsReml => plugin_trials.push(measurement),
            TemporalScalarCandidateMethod::RemlCovarianceParameterAdjustedScalar => {
                adjusted_trials.push(measurement);
            }
            TemporalScalarCandidateMethod::SlopeProfileLikelihoodMl => unreachable!(),
        }
    }
    let plugin = aggregate_trials(plugin_trials)?;
    let adjusted = aggregate_trials(adjusted_trials)?;
    ensure!(
        plugin.factor_sha256 == adjusted.factor_sha256
            && plugin.direct_factor_receipt_sha256 == adjusted.direct_factor_receipt_sha256
            && plugin.factor_block_reads == adjusted.factor_block_reads
            && plugin.optimizer_rho_lane_evaluations == adjusted.optimizer_rho_lane_evaluations
            && plugin.optimizer_q_objective_evaluations
                == adjusted.optimizer_q_objective_evaluations
            && plugin.optimizer_primary_rho_pass_histogram
                == adjusted.optimizer_primary_rho_pass_histogram,
        "scalar arms changed shared factor or optimizer work"
    );
    Ok(TemporalInferenceResourceMeasurement {
        post_gauge_date_count: dates,
        acquisition_count: dates + 1,
        target_count: TEMPORAL_RESOURCE_TILE_ROWS * TEMPORAL_RESOURCE_TILE_COLUMNS,
        varied_target_fingerprint_count,
        adjusted_to_plugin_wall_ratio: adjusted.wall_micros as f64 / plugin.wall_micros as f64,
        adjusted_to_plugin_full_product_wall_ratio: adjusted.full_product_wall_micros as f64
            / plugin.full_product_wall_micros as f64,
        plugin_gls_reml: plugin,
        reml_covariance_parameter_adjusted_scalar: adjusted,
    })
}

fn run_scalar_trial(
    executable: &Path,
    fixture: &Path,
    dates: u64,
    method: TemporalScalarCandidateMethod,
    worker_count: usize,
) -> Result<TemporalInferenceScalarMeasurement> {
    let method_name = match method {
        TemporalScalarCandidateMethod::PluginGlsReml => "plugin_gls_reml",
        TemporalScalarCandidateMethod::RemlCovarianceParameterAdjustedScalar => {
            "reml_covariance_parameter_adjusted_scalar"
        }
        TemporalScalarCandidateMethod::SlopeProfileLikelihoodMl => unreachable!(),
    };
    let output = Command::new(executable)
        .args(["--scalar-probe", &dates.to_string()])
        .arg(fixture)
        .arg(method_name)
        .env("RAYON_NUM_THREADS", worker_count.to_string())
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("VECLIB_MAXIMUM_THREADS", "1")
        .output()?;
    ensure!(
        output.status.success(),
        "scalar trial failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    ensure!(
        output.stdout.len() <= RECEIPT_CAP_BYTES,
        "trial receipt exceeds cap"
    );
    serde_json::from_slice(&output.stdout).context("parsing scalar trial")
}

fn aggregate_trials(
    mut trials: Vec<TemporalInferenceScalarMeasurement>,
) -> Result<TemporalInferenceScalarMeasurement> {
    ensure!(trials.len() == 2, "method requires two trials");
    let second = trials.pop().expect("two trials have a second element");
    let mut first = trials.pop().expect("two trials have a first element");
    ensure!(
        first.method == second.method
            && first.factor_sha256 == second.factor_sha256
            && first.direct_factor_receipt_sha256 == second.direct_factor_receipt_sha256
            && first.factor_block_reads == second.factor_block_reads
            && first.nonreference_realized_rank == second.nonreference_realized_rank
            && first.processed_pixels == second.processed_pixels
            && first.evaluated_pixels == second.evaluated_pixels
            && first.profile_fit_count == second.profile_fit_count
            && first.bootstrap_attempts == second.bootstrap_attempts
            && first.optimizer_rho_lane_evaluations == second.optimizer_rho_lane_evaluations
            && first.optimizer_q_objective_evaluations == second.optimizer_q_objective_evaluations
            && first.optimizer_primary_rho_pass_histogram
                == second.optimizer_primary_rho_pass_histogram
            && first.covariance_parameter_derivative_lane_evaluations
                == second.covariance_parameter_derivative_lane_evaluations
            && first.covariance_parameter_adjustment_count
                == second.covariance_parameter_adjustment_count
            && first.rayon_worker_count == second.rayon_worker_count
            && first.maximum_worker_scratch_bytes == second.maximum_worker_scratch_bytes
            && first.exact_optimizer_fallback_targets == second.exact_optimizer_fallback_targets
            && first.condition_exact_fallbacks == second.condition_exact_fallbacks
            && first.checksum == second.checksum,
        "counterbalanced repetitions changed non-timing evidence"
    );
    let walls = vec![first.wall_micros, second.wall_micros];
    let full_walls = vec![
        first.full_product_wall_micros,
        second.full_product_wall_micros,
    ];
    first.wall_micros = *walls.iter().max().expect("two wall trials");
    first.wall_micros_trials = walls;
    first.full_product_wall_micros = *full_walls.iter().max().expect("two product trials");
    first.full_product_wall_micros_trials = full_walls;
    first.peak_resident_set_bytes = first
        .peak_resident_set_bytes
        .max(second.peak_resident_set_bytes);
    Ok(first)
}

fn prepare_fixture(batch_binary: &Path, directory: &Path, dates: u64) -> Result<u64> {
    let acquisition_count = usize::try_from(dates + 1)?;
    let output = Command::new(batch_binary)
        .args(["--resource-factor", &acquisition_count.to_string()])
        .output()?;
    ensure!(
        output.status.success(),
        "direct factor producer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    ensure!(
        !output.stdout.is_empty() && output.stdout.len() <= RECEIPT_CAP_BYTES,
        "direct factor receipt exceeds cap"
    );
    let direct: DirectFactorReceipt = serde_json::from_slice(&output.stdout)?;
    validate_direct_factor(&direct, acquisition_count)?;
    std::fs::write(
        directory.join(TEMPORAL_DIRECT_FACTOR_RECEIPT_FILENAME),
        &output.stdout,
    )?;

    let rows = usize::try_from(TEMPORAL_RESOURCE_TILE_ROWS)?;
    let columns = usize::try_from(TEMPORAL_RESOURCE_TILE_COLUMNS)?;
    let geotransform = [500_000.0, 30.0, 0.0, 4_200_000.0, 0.0, -30.0];
    let window = BlockIndices {
        row_start: 0,
        row_stop: rows,
        col_start: 0,
        col_stop: columns,
    };
    let mut fingerprints = HashSet::with_capacity(FINGERPRINT_PERIOD + 1);
    for date in 1..acquisition_count {
        let day = date as f64 * 12.0;
        let values = Array2::from_shape_fn((rows, columns), |(row, column)| {
            let target = row * columns + column;
            let fingerprint = target % FINGERPRINT_PERIOD;
            let phase = fingerprint as f64 * 0.017;
            let value = 0.013 * day
                + 0.8 * (date as f64 * 0.61 + phase).sin()
                + 0.15 * (date as f64 * 0.23 + phase * 0.5).cos()
                + fingerprint as f64 * 0.001;
            if date == 1 {
                fingerprints.insert((
                    usize::from(target != 0) * direct.realized_rank,
                    (value as f32).to_bits(),
                ));
            }
            value as f32
        });
        write_cog(
            &directory.join(format!("displacement_{:03}.tif", date - 1)),
            values.view(),
            geotransform,
            window,
        )?;
    }
    write_mask_cog(
        &directory.join("velocity_validity_mask.tif"),
        Array2::from_elem((rows, columns), 1_u8).view(),
        geotransform,
        window,
    )?;
    write_factor_artifact(directory, &direct, geotransform, rows, columns)?;
    ensure!(
        fingerprints.len() >= FINGERPRINT_PERIOD,
        "insufficient target variation"
    );
    Ok(u64::try_from(fingerprints.len())?)
}

fn validate_direct_factor(direct: &DirectFactorReceipt, acquisition_count: usize) -> Result<()> {
    let canonical = |value: &str| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    };
    ensure!(
        direct.schema == "dolphinrust-temporal-direct-factor/1"
            && direct.acquisition_count == acquisition_count
            && direct.post_gauge_date_count == acquisition_count - 1
            && direct.realized_rank == acquisition_count - 1
            && direct.factor_generation_count == 1
            && direct.covariance_condition_number.is_finite()
            && direct.covariance_condition_number >= 1.0
            && direct.positive_off_diagonal_energy > 0.0
            && direct.replay_resource_high_water_bytes > 0
            && direct.effective_looks_fraction > 0.0
            && direct.effective_looks_fraction <= 1.0
            && direct.source_correlation_support_union_count > 0
            && [
                &direct.source_manifest_sha256,
                &direct.issue52_operator_receipt_sha256,
                &direct.issue52_source_factor_receipt_sha256,
                &direct.issue54_fixed_l2_map_receipt_sha256,
                &direct.issue54_replay_source_factor_receipt_sha256,
                &direct.issue54_replay_support_receipt_sha256,
                &direct.reference_signature_sha256,
                &direct.factor_sha256,
                &direct.source_correlation_receipt_sha256,
                &direct.difference_covariance_sha256,
            ]
            .into_iter()
            .all(|value| canonical(value))
            && direct.difference_factor.len() == acquisition_count
            && direct
                .difference_factor
                .iter()
                .all(|row| row.len() == direct.realized_rank)
            && direct.difference_factor[0]
                .iter()
                .all(|value| *value == 0.0),
        "direct factor receipt differs from release contract"
    );
    let observed_factor_sha256 = sha256(&serde_json::to_vec(&direct.difference_factor)?);
    ensure!(
        observed_factor_sha256 == direct.factor_sha256,
        "direct factor bytes disagree with receipt: declared={}, observed={}",
        direct.factor_sha256,
        observed_factor_sha256,
    );
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn write_factor_artifact(
    directory: &Path,
    direct: &DirectFactorReceipt,
    geotransform: [f64; 6],
    rows: usize,
    columns: usize,
) -> Result<()> {
    let config = TemporalUncertaintyOptions::default();
    let acquisitions = direct.acquisition_count;
    let per_target_bytes = u64::try_from(acquisitions)?
        .checked_mul(u64::try_from(acquisitions)?)
        .and_then(|value| value.checked_mul(8))
        .and_then(|value| value.checked_add(82))
        .context("factor target bytes overflow")?;
    let factor_budget = config
        .factor_block_read_cap_bytes
        .checked_sub(direct.replay_resource_high_water_bytes)
        .and_then(|value| value.checked_sub(1))
        .context("no factor serialization budget")?
        / 2;
    let rows_per_block = (usize::try_from(factor_budget / per_target_bytes)? / columns)
        .max(1)
        .min(rows);
    let maximum_targets = rows_per_block * columns;
    let factor_high_water = u64::try_from(maximum_targets)?
        .checked_mul(per_target_bytes)
        .context("factor high-water overflow")?;
    ensure!(
        factor_high_water <= factor_budget,
        "factor row exceeds writer budget"
    );
    let admission = factor_high_water
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_add(direct.replay_resource_high_water_bytes))
        .context("admission receipt overflow")?;
    let observed = admission - 1;
    let runtime = SpatialReferenceRuntimeResourceReceipt {
        working_set_byte_cap: config.factor_block_read_cap_bytes,
        factor_block_high_water_bytes: factor_high_water,
        serialization_high_water_bytes: factor_high_water,
        fixed_l2_workspace_admission_bytes: 1,
        fixed_l2_workspace_observed_high_water_bytes: 0,
        replay_admission_high_water_bytes: direct.replay_resource_high_water_bytes,
        replay_observed_high_water_bytes: direct.replay_resource_high_water_bytes,
        provider_peak_count: 0,
        provider_peak_bytes: 0,
        preflight_provider_open_count: 0,
        production_provider_open_count: 0,
        operator_block_reads: 0,
        operator_block_cache_hits: 0,
        source_member_window_reads: 0,
        source_tile_cache_loads: 0,
        source_resolutions: 0,
        working_set_admission_high_water_bytes: admission,
        working_set_observed_high_water_bytes: observed,
    };
    let receipt_bytes = read_bounded(
        &directory.join(TEMPORAL_DIRECT_FACTOR_RECEIPT_FILENAME),
        RECEIPT_CAP_BYTES as u64,
    )?;
    let derived = |label: &[u8]| {
        let mut digest = Sha256::new();
        digest.update(label);
        digest.update(&receipt_bytes);
        format!("sha256:{:x}", digest.finalize())
    };
    let mut metadata = SpatialReferenceCovarianceMetadata {
        schema_version: SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
        method: SPATIAL_REFERENCE_COVARIANCE_METHOD.to_owned(),
        method_version: 1,
        crate_version: env!("CARGO_PKG_VERSION").to_owned(),
        producer_commit: None,
        burst_id: "temporal-inference-resource-direct-factor".to_owned(),
        crs: "EPSG:32611".to_owned(),
        units: "radians".to_owned(),
        geotransform: Some(geotransform),
        full_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: u32::try_from(rows)?,
            cols: u32::try_from(columns)?,
            stride_y: 1,
            stride_x: 1,
        },
        reference_row: 0,
        reference_col: 0,
        gauge_date_index: 0,
        ordered_date_indices: (0..u32::try_from(acquisitions)?).collect(),
        acquisition_days: Some((0..acquisitions).map(|date| date as f64 * 12.0).collect()),
        mask_digest: derived(b"resource-fixture-mask-v1"),
        source_replay_digest: strong_digest(&direct.issue52_operator_receipt_sha256)?,
        l2_map_digest: strong_digest(&direct.issue54_fixed_l2_map_receipt_sha256)?,
        reference_signature_digest: strong_digest(&direct.reference_signature_sha256)?,
        approximation_receipt_digest: derived(b"resource-fixture-uncalibrated-v1"),
        resource_receipt_digest: format!("sha256:{}", sha256(&receipt_bytes)),
        runtime_resource_receipt_digest: spatial_reference_runtime_resource_receipt_digest(runtime),
        runtime_resource_receipt: Some(runtime),
        review_receipt_digest: String::new(),
        method_manifest_digest: String::new(),
        calibration_scope_digest: String::new(),
        source_model_digest: strong_digest(&direct.issue52_source_factor_receipt_sha256)?,
        effective_looks_digest: derived(b"resource-fixture-effective-placeholder-v1"),
        support_method: "rect".to_owned(),
        support_digest: strong_digest(&direct.source_correlation_receipt_sha256)?,
        correction_order_digest: derived(b"resource-fixture-correction-order-v1"),
        unwrap_branch_digest: derived(b"resource-fixture-unwrap-branch-v1"),
        burst_ownership_digest: derived(b"resource-fixture-burst-ownership-v1"),
        source_burst_ids: vec!["temporal-inference-resource-direct-factor".to_owned()],
        reference_source_burst_index: 0,
        calibration_scope: SpatialReferenceCalibrationScope::Uncalibrated,
        maximum_block_bytes: config.factor_block_read_cap_bytes,
    };
    let path = directory.join(SPATIAL_REFERENCE_COVARIANCE_FILENAME);
    let mut writer = SpatialReferenceCovarianceWriter::create(&path, &metadata)?;
    let look_receipt = parse_sha256(&direct.source_correlation_receipt_sha256)?;
    let mut padded = vec![0.0_f64; acquisitions * acquisitions];
    for date in 0..acquisitions {
        padded[date * acquisitions..date * acquisitions + direct.realized_rank]
            .copy_from_slice(&direct.difference_factor[date]);
    }
    let mut block_id = 1_u64;
    for row_start in (0..rows).step_by(rows_per_block) {
        let block_rows = rows_per_block.min(rows - row_start);
        let targets = block_rows * columns;
        let mut factor = Vec::with_capacity(targets * padded.len());
        let mut ranks = Vec::with_capacity(targets);
        let mut conditions = Vec::with_capacity(targets);
        for target in 0..targets {
            if row_start * columns + target == 0 {
                factor.extend(std::iter::repeat_n(0.0, padded.len()));
                ranks.push(0);
                conditions.push(f64::NAN);
            } else {
                factor.extend_from_slice(&padded);
                ranks.push(u32::try_from(direct.realized_rank)?);
                conditions.push(direct.covariance_condition_number);
            }
        }
        writer.write_block(&SpatialReferenceCovarianceBlock {
            block_id,
            target_grid: CovarianceOperatorGrid {
                row_start: u64::try_from(row_start)?,
                col_start: 0,
                rows: u32::try_from(block_rows)?,
                cols: u32::try_from(columns)?,
                stride_y: 1,
                stride_x: 1,
            },
            maximum_rank: u32::try_from(acquisitions)?,
            rank_by_target: ranks,
            status: vec![SpatialReferenceCovarianceStatus::Valid; targets],
            source_burst_index_by_target: vec![0; targets],
            difference_factor: factor,
            approximation_error_bound: vec![f64::NAN; targets],
            effective_looks_fraction: Some(vec![direct.effective_looks_fraction; targets]),
            support_union_count: Some(vec![
                u64::try_from(
                    direct.source_correlation_support_union_count
                )?;
                targets
            ]),
            effective_looks_receipt: Some(
                (0..targets).flat_map(|_| look_receipt).collect::<Vec<_>>(),
            ),
            resource_high_water_bytes: Some(vec![direct.replay_resource_high_water_bytes; targets]),
            condition_number: Some(conditions),
            source_factor_digest: strong_digest(
                &direct.issue54_replay_source_factor_receipt_sha256,
            )?,
        })?;
        block_id += 1;
    }
    metadata = writer.seal_effective_looks_digest()?;
    ensure!(
        metadata.runtime_resource_receipt == Some(runtime),
        "writer changed runtime receipt"
    );
    writer.finish()?;
    Ok(())
}

fn strong_digest(value: &str) -> Result<String> {
    parse_sha256(value)?;
    Ok(format!("sha256:{value}"))
}

fn parse_sha256(value: &str) -> Result<[u8; 32]> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "noncanonical SHA-256 digest"
    );
    let mut output = [0_u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)?;
    }
    Ok(output)
}

fn write_cog(
    path: &Path,
    values: ndarray::ArrayView2<'_, f32>,
    geotransform: [f64; 6],
    window: BlockIndices,
) -> Result<()> {
    let scratch = path.with_extension("scratch.tif");
    let mut writer = BoundedCogWriter::<f32>::create(
        &scratch,
        values.dim(),
        geotransform,
        Some(32611),
        Some(f64::NAN),
        &[("UNITTYPE", "radians")],
    )?;
    writer.write_window(window, values)?;
    writer.finalize(path)?;
    Ok(())
}

fn write_mask_cog(
    path: &Path,
    values: ndarray::ArrayView2<'_, u8>,
    geotransform: [f64; 6],
    window: BlockIndices,
) -> Result<()> {
    let scratch = path.with_extension("scratch.tif");
    let mut writer = BoundedCogWriter::<u8>::create(
        &scratch,
        values.dim(),
        geotransform,
        Some(32611),
        Some(0.0),
        &[("MASK_ROLE", "velocity_support")],
    )?;
    writer.write_window(window, values)?;
    writer.finalize(path)?;
    Ok(())
}

fn link_fixture(source: &Path, destination: &Path, dates: u64) -> Result<()> {
    std::fs::create_dir(destination)?;
    let mut names = vec![
        SPATIAL_REFERENCE_COVARIANCE_FILENAME.to_owned(),
        TEMPORAL_DIRECT_FACTOR_RECEIPT_FILENAME.to_owned(),
        "velocity_validity_mask.tif".to_owned(),
    ];
    names.extend((0..dates).map(|date| format!("displacement_{date:03}.tif")));
    for name in names {
        std::fs::hard_link(source.join(&name), destination.join(name))?;
    }
    Ok(())
}

fn observed_binary_identity(path: &Path) -> Result<TemporalInferenceBinaryIdentity> {
    let bytes = read_bounded(path, BINARY_CAP_BYTES)?;
    ensure!(!bytes.is_empty(), "observed binary is empty");
    Ok(TemporalInferenceBinaryIdentity {
        sha256: sha256(&bytes),
        bytes: u64::try_from(bytes.len())?,
    })
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_bounded(path: &Path, cap: u64) -> Result<Vec<u8>> {
    let metadata = std::fs::metadata(path)?;
    ensure!(
        metadata.is_file() && metadata.len() <= cap,
        "bounded read rejected file"
    );
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len())?);
    File::open(path)?.take(cap + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= cap, "bounded read exceeded cap");
    Ok(bytes)
}

struct BenchmarkScratch {
    path: PathBuf,
}

impl BenchmarkScratch {
    fn create() -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "dolphin-temporal-inference-bench-{}",
            std::process::id()
        ));
        ensure!(!path.exists(), "benchmark scratch already exists");
        std::fs::create_dir(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for BenchmarkScratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

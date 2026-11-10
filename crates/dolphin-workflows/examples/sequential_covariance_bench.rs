//! Release-mode resource receipt for sequential_source_dag_v1.
//!
//! Single-spatial-block temporal receipt (13/26/52 dates, 256x256 output,
//! M=13, configured K=10):
//!
//!     cargo run --release -p dolphin-workflows --example sequential_covariance_bench \
//!         --no-default-features --features no-gpu
//!
//! Practical end-to-end smoke receipt (26 dates, 8x8 output, two blocks):
//!
//!     cargo run --release -p dolphin-workflows --example sequential_covariance_bench \
//!         --no-default-features --features no-gpu -- --smoke
//!
//! Every case runs in a fresh child process. The child streams production
//! capture blocks into the HDF5 operator writer, finalizes and verifies the
//! artifact, then replays the complete same-pixel temporal covariance through
//! the capped artifact provider and dependency-cone query. No runtime-ratio
//! acceptance gate is applied. Inputs and source factors are deterministic
//! synthetic fixtures; the receipt measures resource behavior, not scientific
//! accuracy on production data. The 52-date case observes three carried blocks;
//! it does not saturate K=10 or measure multi-tile scaling. Receipts are emitted
//! as JSON on stdout; successful scratch artifacts are deleted.

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use std::{fs, path::PathBuf};

use anyhow::{bail, ensure, Context, Result};
use dolphin_core::config::{CompressedSlcPlan, ComputeBackend, ShpMethod};
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_io::{
    covariance_identity_index_peak_bytes, CovarianceCalibrationStatus, CovarianceOperatorBlock,
    CovarianceOperatorGrid, CovarianceOperatorMetadata, CovarianceOperatorWriter,
    CovarianceReplayStatus, DownstreamInferenceStatus, SourceReplayIdentity,
    StitchedCovarianceStatus, COVARIANCE_OPERATOR_METHOD,
};
use dolphin_phaselink::{ComputeEngine, ProperComplexFactor};
use dolphin_workflows::{
    finalize_covariance_artifact, preflight_covariance_artifact_disk_with_identity_index,
    read_covariance_artifact_manifest, run_sequential_with_covariance_capture,
    sequential_replay_kernel_digest, sequential_source_model_identity_digest,
    CovarianceArtifactReplayProvider, CovarianceArtifactTransaction, DependencyConeQuery,
    GlobalDateId, ReplayBackend, ReplayExecutionScope, ReplayIdNamespace, ReplayStatus,
    ResolvedPrimitiveSource, SequentialConfig, SequentialCovarianceCaptureRequest,
    SequentialPrimitiveSourceResolver, SequentialReplayBlock, SequentialReplayBuildIdentity,
    SequentialReplayError, SequentialReplayTopology, SequentialSourceProviderIdentity,
    SequentialSourceReplayProvider,
};
use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MINISTACK_SIZE: usize = 13;
const MAX_CARRIED: usize = 10;
const FULL_OUTPUT_SIZE: usize = 256;
const FULL_HALF_WINDOW: HalfWindow = HalfWindow { y: 7, x: 14 };
const SMOKE_OUTPUT_SIZE: usize = 8;
const SMOKE_HALF_WINDOW: HalfWindow = HalfWindow { y: 1, x: 1 };
const BRANCH_TOLERANCE: f64 = 1e-10;
const SOURCE_PROVIDER: &str = "synthetic-formula";
const SOURCE_PROVIDER_VERSION: &str = "1";
const SOURCE_MODEL: &str = "diagonal-proper-complex";
const SOURCE_MODEL_VERSION: &str = "1";
const HDF5_OVERHEAD_BYTES_PER_BLOCK: u64 = 8 * 1024 * 1024;
const METADATA_READ_ALLOWANCE_BYTES: u64 = 8 * 1024 * 1024;
const BLOCK_LINK_NAME_AND_OVERHEAD_BYTES: u64 = 20 + 64;
const BENCHMARK_BURST_ID: &str = "sequential-covariance-bench";

#[derive(Debug, Clone, Copy)]
struct CaseConfig {
    dates: usize,
    output_size: usize,
    half_window: HalfWindow,
    mode: &'static str,
}

#[derive(Debug, Serialize, Deserialize)]
struct CaseReceipt {
    method: String,
    mode: String,
    dates: usize,
    output_rows: usize,
    output_columns: usize,
    ministack_size: usize,
    configured_maximum_carried: usize,
    observed_maximum_carried: usize,
    support_slots: usize,
    blocks: usize,
    hdf5_bytes: u64,
    hdf5_sha256: String,
    dataset_payload_bytes: u64,
    hdf5_to_payload_ratio: f64,
    projected_hdf5_bytes: u64,
    disk_required_free_bytes: u64,
    disk_available_free_bytes: u64,
    dependency_cone_bytes: u64,
    dependency_cone_provider_bytes: u64,
    dependency_cone_blocks: usize,
    artifact_block_read_cap_bytes: u64,
    artifact_block_reads: u64,
    artifact_block_cache_hits: u64,
    artifact_retained_block_payload_bytes_loaded: u64,
    artifact_block_read_seconds: f64,
    artifact_retained_payload_rate_bytes_per_second: f64,
    artifact_cache_current_payload_bytes: u64,
    artifact_cache_peak_payload_bytes: u64,
    artifact_cache_reservation_bytes: u64,
    artifact_cached_block_id: Option<u64>,
    artifact_verification_seconds: f64,
    artifact_open_seconds: f64,
    capture_overlap_topology_reads: u64,
    capture_overlap_topology_bytes: u64,
    capture_peak_retained_topology_blocks: usize,
    capture_peak_frontier_chains: usize,
    identity_index_bytes_read: u64,
    identity_index_bytes_written: u64,
    identity_index_peak_disk_bytes: u64,
    identity_index_peak_block_records: usize,
    identity_index_merges: u64,
    identity_index_disk_cap_bytes: u64,
    source_provider: String,
    source_manifest_digest: String,
    source_model_version_digest: String,
    source_model_receipt_digest: String,
    source_resolve_calls: u64,
    raw_source_resolved_bytes: u64,
    source_factor_resolved_bytes: u64,
    source_resolved_bytes: u64,
    source_cache_bytes: u64,
    source_resolution_throughput_bytes_per_second: f64,
    capture_wall_seconds: f64,
    query_wall_seconds: f64,
    wall_seconds: f64,
    peak_rss_bytes: Option<u64>,
    rayon_threads: usize,
    covariance_dimension: usize,
    covariance_checksum: f64,
}

#[derive(Debug, Serialize)]
struct StorageRatioReceipt {
    numerator_dates: usize,
    denominator_dates: usize,
    hdf5_bytes_ratio: f64,
    dataset_payload_bytes_ratio: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkReceipt {
    cases: Vec<CaseReceipt>,
    storage_ratios: Vec<StorageRatioReceipt>,
}

#[derive(Default)]
struct ResolverMetrics {
    calls: AtomicU64,
    raw_bytes: AtomicU64,
    factor_bytes: AtomicU64,
    nanoseconds: AtomicU64,
}

struct SyntheticSourceResolver {
    identity: SequentialSourceProviderIdentity,
    topology: Arc<SequentialReplayTopology>,
    columns: usize,
    metrics: Arc<ResolverMetrics>,
}

impl SequentialPrimitiveSourceResolver for SyntheticSourceResolver {
    fn identity(&self) -> &SequentialSourceProviderIdentity {
        &self.identity
    }

    fn maximum_resident_bytes(&self) -> u64 {
        0
    }

    fn resolve_source(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError> {
        let started = Instant::now();
        let row = native_index / self.columns;
        let column = native_index % self.columns;
        let samples = Array1::from_iter((0..block.num_real_dates).map(|offset| {
            synthetic_sample(block.real_date_start.get() as usize + offset, row, column)
        }));
        let mut digest = Sha256::new();
        for sample in &samples {
            digest.update(sample.re.to_le_bytes());
            digest.update(sample.im.to_le_bytes());
        }
        let content_digest = digest.finalize().into();
        let id =
            self.topology
                .source_id_for_content_digest(block.id, native_index, &content_digest)?;
        let component_ids = (block.real_date_start.get()
            ..block.real_date_start.get() + block.num_real_dates as u32)
            .map(u64::from)
            .collect();
        let factor = ProperComplexFactor::new(
            id,
            component_ids,
            source_model_receipt_digest(),
            Array2::from_diag_elem(block.num_real_dates, Cf64::new(0.02, 0.0)),
        )
        .map_err(|_| {
            SequentialReplayError::Provider(
                ReplayStatus::SourceModelUnavailable,
                "benchmark proper-complex factor is invalid",
            )
        })?;
        let raw_bytes = (samples.len() as u64).saturating_mul(16);
        let factor_bytes = (block.num_real_dates as u64)
            .saturating_mul(block.num_real_dates as u64)
            .saturating_mul(16)
            .saturating_add((block.num_real_dates as u64).saturating_mul(8))
            .saturating_add(32);
        self.metrics.calls.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .raw_bytes
            .fetch_add(raw_bytes, Ordering::Relaxed);
        self.metrics
            .factor_bytes
            .fetch_add(factor_bytes, Ordering::Relaxed);
        self.metrics.nanoseconds.fetch_add(
            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        Ok(ResolvedPrimitiveSource {
            id,
            samples,
            factor,
            content_digest,
        })
    }
}

fn main() -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "sequential covariance resource receipts require --release"
    );
    let mut arguments = std::env::args().skip(1);
    match arguments.next().as_deref() {
        Some("--child") => run_child(arguments),
        Some("--smoke") => run_parent(&[CaseConfig {
            dates: 26,
            output_size: SMOKE_OUTPUT_SIZE,
            half_window: SMOKE_HALF_WINDOW,
            mode: "smoke",
        }]),
        None => run_parent(&[
            CaseConfig {
                dates: 13,
                output_size: FULL_OUTPUT_SIZE,
                half_window: FULL_HALF_WINDOW,
                mode: "full",
            },
            CaseConfig {
                dates: 26,
                output_size: FULL_OUTPUT_SIZE,
                half_window: FULL_HALF_WINDOW,
                mode: "full",
            },
            CaseConfig {
                dates: 52,
                output_size: FULL_OUTPUT_SIZE,
                half_window: FULL_HALF_WINDOW,
                mode: "full",
            },
        ]),
        Some(argument) => bail!("unknown argument {argument}; expected --smoke"),
    }
}

fn run_parent(cases: &[CaseConfig]) -> Result<()> {
    let executable = std::env::current_exe().context("resolving benchmark executable")?;
    let mut receipts = Vec::with_capacity(cases.len());
    for &case in cases {
        eprintln!(
            "running {}-date {}x{} {} receipt in a fresh child",
            case.dates, case.output_size, case.output_size, case.mode
        );
        receipts.push(run_case_child(&executable, case)?);
    }
    let storage_ratios = [(26, 13), (52, 26)]
        .into_iter()
        .filter_map(|(numerator_dates, denominator_dates)| {
            let numerator = receipts
                .iter()
                .find(|receipt| receipt.dates == numerator_dates)?;
            let denominator = receipts
                .iter()
                .find(|receipt| receipt.dates == denominator_dates)?;
            Some(StorageRatioReceipt {
                numerator_dates,
                denominator_dates,
                hdf5_bytes_ratio: numerator.hdf5_bytes as f64 / denominator.hdf5_bytes as f64,
                dataset_payload_bytes_ratio: numerator.dataset_payload_bytes as f64
                    / denominator.dataset_payload_bytes as f64,
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&BenchmarkReceipt {
            cases: receipts,
            storage_ratios,
        })?
    );
    Ok(())
}

fn run_case_child(executable: &std::path::Path, case: CaseConfig) -> Result<CaseReceipt> {
    let dates = case.dates.to_string();
    let output_size = case.output_size.to_string();
    let half_window_y = case.half_window.y.to_string();
    let half_window_x = case.half_window.x.to_string();
    let output = Command::new(executable)
        .args([
            "--child",
            &dates,
            &output_size,
            &half_window_y,
            &half_window_x,
            case.mode,
        ])
        .output()
        .context("launching fresh covariance benchmark child")?;
    ensure!(
        output.status.success(),
        "covariance benchmark child failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    serde_json::from_slice(&output.stdout).context("parsing covariance benchmark child receipt")
}

fn run_child(mut arguments: impl Iterator<Item = String>) -> Result<()> {
    let dates = parse_argument(&mut arguments, "dates")?;
    let output_size = parse_argument(&mut arguments, "output size")?;
    let half_window_y = parse_argument(&mut arguments, "half-window rows")?;
    let half_window_x = parse_argument(&mut arguments, "half-window columns")?;
    let mode = arguments.next().context("missing benchmark mode")?;
    ensure!(arguments.next().is_none(), "too many child arguments");
    let mode = match mode.as_str() {
        "full" => "full",
        "smoke" => "smoke",
        _ => bail!("invalid benchmark mode"),
    };
    let receipt = run_case(CaseConfig {
        dates,
        output_size,
        half_window: HalfWindow {
            y: half_window_y,
            x: half_window_x,
        },
        mode,
    })?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}

fn parse_argument(
    arguments: &mut impl Iterator<Item = String>,
    name: &'static str,
) -> Result<usize> {
    arguments
        .next()
        .with_context(|| format!("missing {name}"))?
        .parse()
        .with_context(|| format!("invalid {name}"))
}

#[allow(clippy::too_many_lines)]
fn run_case(case: CaseConfig) -> Result<CaseReceipt> {
    let started = Instant::now();
    ensure!(case.dates > 0, "benchmark date count must be positive");
    ensure!(
        u32::try_from(case.dates).is_ok(),
        "benchmark date count exceeds u32"
    );
    let window_rows = case
        .half_window
        .y
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .context("benchmark window row count overflow")?;
    let window_columns = case
        .half_window
        .x
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .context("benchmark window column count overflow")?;
    ensure!(
        case.output_size >= window_rows && case.output_size >= window_columns,
        "benchmark window exceeds its output block"
    );
    let shape = (case.output_size, case.output_size);
    let support_slots = window_rows
        .checked_mul(window_columns)
        .context("benchmark support count overflow")?;
    let cfg = benchmark_config(case.half_window);
    let output_shape = cfg.strides.out_shape(shape);
    ensure!(output_shape == shape, "benchmark output shape changed");
    let validity = Array2::from_elem(shape, true);
    let namespace = benchmark_namespace(case);
    let topology = Arc::new(SequentialReplayTopology::plan_identified(
        case.dates,
        shape,
        output_shape,
        support_slots,
        validity.view(),
        &cfg,
        benchmark_scope(),
        namespace.clone(),
    )?);
    drop(validity);
    let expected_payload = estimated_dataset_payload_bytes(case, support_slots)?;
    let projected_hdf5_bytes = expected_payload
        .checked_mul(2)
        .and_then(|bytes| {
            bytes.checked_add(
                (topology.blocks().len() as u64 + 1).checked_mul(HDF5_OVERHEAD_BYTES_PER_BLOCK)?,
            )
        })
        .context("projected HDF5 byte count overflow")?;
    let directory = temporary_directory(case);
    fs::create_dir(&directory).with_context(|| {
        format!(
            "creating covariance benchmark directory {}",
            directory.display()
        )
    })?;
    let result = run_case_in_directory(
        case,
        &cfg,
        topology,
        namespace,
        support_slots,
        expected_payload,
        projected_hdf5_bytes,
        &directory,
        started,
    )
    .with_context(|| {
        format!(
            "covariance benchmark failed; artifacts retained at {}",
            directory.display()
        )
    });
    if result.is_ok() {
        fs::remove_dir_all(&directory).with_context(|| {
            format!(
                "removing covariance benchmark directory {}",
                directory.display()
            )
        })?;
    }
    result
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_case_in_directory(
    case: CaseConfig,
    cfg: &SequentialConfig,
    topology: Arc<SequentialReplayTopology>,
    namespace: ReplayIdNamespace,
    support_slots: usize,
    expected_payload: u64,
    projected_hdf5_bytes: u64,
    directory: &std::path::Path,
    started: Instant,
) -> Result<CaseReceipt> {
    let transaction = CovarianceArtifactTransaction::acquire(directory)?;
    let identity_records = u64::try_from(topology.blocks().len())?
        .checked_mul(u64::try_from(
            case.output_size
                .checked_mul(case.output_size)
                .context("benchmark output area overflow")?,
        )?)
        .and_then(|records| records.checked_mul(3))
        .context("benchmark identity record count overflow")?;
    let identity_index_peak_bytes = covariance_identity_index_peak_bytes(identity_records)?;
    let disk_admission = preflight_covariance_artifact_disk_with_identity_index(
        directory,
        projected_hdf5_bytes,
        identity_index_peak_bytes,
    )?;
    let build_identity = SequentialReplayBuildIdentity {
        normalized_config_digest: topology.normalized_config_digest(),
        kernel_digest: sequential_replay_kernel_digest(),
        branch_tolerance: BRANCH_TOLERANCE,
    };
    let metadata = benchmark_metadata(&topology, namespace.source_manifest_digest);
    let scratch = directory.join("phase_covariance_operator.h5.scratch");
    let writer_plan = topology.covariance_operator_plan(&namespace.burst_id)?;
    let mut writer = CovarianceOperatorWriter::create_with_identity_index_disk_cap(
        &scratch,
        &metadata,
        &writer_plan,
        identity_index_peak_bytes,
    )
    .context("creating covariance operator scratch writer")?;
    let stack = Array3::from_shape_fn(
        (case.dates, case.output_size, case.output_size),
        |(date, row, column)| synthetic_sample(date, row, column),
    );
    let grid_size = u32::try_from(case.output_size).context("benchmark grid exceeds u32")?;
    let request = SequentialCovarianceCaptureRequest {
        burst_id: namespace.burst_id.clone(),
        source_manifest_digest: namespace.source_manifest_digest,
        source_model_version_digest: namespace.source_model_version_digest,
        native_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: grid_size,
            cols: grid_size,
            stride_y: 1,
            stride_x: 1,
        },
        output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: grid_size,
            cols: grid_size,
            stride_y: 1,
            stride_x: 1,
        },
        owned_output_grid: CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: grid_size,
            cols: grid_size,
            stride_y: 1,
            stride_x: 1,
        },
        branch_tolerance: BRANCH_TOLERANCE,
    };
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let mut dataset_payload_bytes = 0_u64;
    let mut maximum_block_payload_bytes = 0_u64;
    let mut maximum_block_topology_bytes = 0_u64;
    let mut block_count = 0_usize;
    let capture_started = Instant::now();
    let output = run_sequential_with_covariance_capture(
        stack.view(),
        cfg,
        &engine,
        &request,
        |mut block| {
            bind_benchmark_factor_receipts(&mut block)
                .map_err(|_| "binding benchmark numeric factor receipts")?;
            let payload = block_dataset_payload_bytes(&block)
                .ok_or("benchmark covariance payload byte count overflow")?;
            dataset_payload_bytes = dataset_payload_bytes
                .checked_add(payload)
                .ok_or("benchmark covariance payload byte count overflow")?;
            maximum_block_payload_bytes = maximum_block_payload_bytes.max(payload);
            let topology_bytes = block_topology_payload_bytes(&block)
                .ok_or("benchmark covariance topology byte count overflow")?;
            maximum_block_topology_bytes = maximum_block_topology_bytes.max(topology_bytes);
            block_count += 1;
            writer
                .write_block(&block)
                .map_err(|_| "writing covariance benchmark block")
        },
    )?;
    drop(output);
    drop(stack);
    let write_receipt = writer
        .finish()
        .context("finishing covariance HDF5 writer")?;
    let capture_wall_seconds = capture_started.elapsed().as_secs_f64();
    ensure!(
        block_count == topology.blocks().len(),
        "captured block count differs from topology"
    );
    ensure!(
        dataset_payload_bytes == expected_payload,
        "captured dataset payload differs from its schema-derived projection"
    );
    let manifest = finalize_covariance_artifact(
        &transaction,
        &scratch,
        &metadata,
        disk_admission,
        &write_receipt,
    )?;
    drop(transaction);
    let verification_started = Instant::now();
    ensure!(
        read_covariance_artifact_manifest(directory)? == manifest,
        "committed covariance manifest verification changed its receipt"
    );
    let artifact_verification_seconds = verification_started.elapsed().as_secs_f64();

    let metrics = Arc::new(ResolverMetrics::default());
    let identity = SequentialSourceProviderIdentity {
        source_manifest_digest: namespace.source_manifest_digest,
        provider: SOURCE_PROVIDER.to_owned(),
        provider_version: SOURCE_PROVIDER_VERSION.to_owned(),
        model: SOURCE_MODEL.to_owned(),
        model_version: SOURCE_MODEL_VERSION.to_owned(),
        source_model_version_digest: source_model_identity_digest(),
        source_model_hash: source_model_receipt_digest(),
    };
    let resolver = SyntheticSourceResolver {
        identity,
        topology: Arc::clone(&topology),
        columns: case.output_size,
        metrics: Arc::clone(&metrics),
    };
    let artifact_block_read_cap_bytes = maximum_block_payload_bytes
        .checked_add(
            maximum_block_topology_bytes
                .checked_mul(8)
                .context("artifact topology workspace cap overflow")?,
        )
        .and_then(|bytes| bytes.checked_add(BLOCK_LINK_NAME_AND_OVERHEAD_BYTES))
        .and_then(|bytes| bytes.checked_add(METADATA_READ_ALLOWANCE_BYTES))
        .context("artifact block read cap overflow")?;
    let open_started = Instant::now();
    let mut provider = CovarianceArtifactReplayProvider::open(
        directory,
        artifact_block_read_cap_bytes,
        &topology,
        build_identity,
        resolver,
    )?;
    let artifact_open_seconds = open_started.elapsed().as_secs_f64();
    let output_index = (case.output_size / 2) * case.output_size + case.output_size / 2;
    let selection = (0..case.dates)
        .map(|date| (GlobalDateId::new(date as u32), output_index))
        .collect::<Vec<_>>();
    let source_rank = 2 * MINISTACK_SIZE;
    let internal = topology.estimate_dependency_cone(&selection, source_rank, 1)?;
    let byte_cap = internal
        .total_bytes
        .checked_add(provider.maximum_resident_bytes())
        .context("dependency-cone byte cap overflow")?;
    let query_started = Instant::now();
    let replay = topology.replay_temporal_covariance_from_provider(
        &selection,
        DependencyConeQuery {
            source_rank,
            microbatch: 1,
            byte_cap,
        },
        BRANCH_TOLERANCE,
        &mut provider,
    )?;
    let query_wall_seconds = query_started.elapsed().as_secs_f64();
    let artifact_metrics = provider.metrics();
    ensure!(
        replay.covariance.dim() == (case.dates, case.dates)
            && replay.covariance.iter().all(|value| value.is_finite())
            && replay.covariance.row(0).iter().all(|value| *value == 0.0)
            && replay
                .covariance
                .column(0)
                .iter()
                .all(|value| *value == 0.0),
        "streaming replay returned an invalid acquisition-0-gauge covariance"
    );
    let source_nanoseconds = metrics.nanoseconds.load(Ordering::Relaxed);
    let raw_source_resolved_bytes = metrics.raw_bytes.load(Ordering::Relaxed);
    let source_factor_resolved_bytes = metrics.factor_bytes.load(Ordering::Relaxed);
    let source_resolved_bytes = raw_source_resolved_bytes
        .checked_add(source_factor_resolved_bytes)
        .context("source resolved byte receipt overflow")?;
    let source_resolution_throughput_bytes_per_second = match source_nanoseconds {
        0 => 0.0,
        value => source_resolved_bytes as f64 / (value as f64 * 1e-9),
    };
    let covariance_checksum = replay.covariance.iter().sum();
    let artifact_retained_block_payload_bytes_loaded = artifact_metrics
        .logical_block_bytes_read
        .context("artifact provider omitted its retained-payload byte receipt")?;
    let artifact_block_read_seconds = artifact_metrics.operator_block_read_elapsed.as_secs_f64();
    let artifact_retained_payload_rate_bytes_per_second = match artifact_block_read_seconds > 0.0 {
        true => artifact_retained_block_payload_bytes_loaded as f64 / artifact_block_read_seconds,
        false => 0.0,
    };
    let artifact_cache_current_payload_bytes = artifact_metrics
        .current_cached_payload_bytes
        .context("artifact provider omitted its current cache payload receipt")?;
    let artifact_cache_peak_payload_bytes = artifact_metrics
        .peak_cached_payload_bytes
        .context("artifact provider omitted its peak cache payload receipt")?;
    let wall_seconds = started.elapsed().as_secs_f64();
    Ok(CaseReceipt {
        method: COVARIANCE_OPERATOR_METHOD.to_owned(),
        mode: case.mode.to_owned(),
        dates: case.dates,
        output_rows: case.output_size,
        output_columns: case.output_size,
        ministack_size: MINISTACK_SIZE,
        configured_maximum_carried: MAX_CARRIED,
        observed_maximum_carried: topology
            .blocks()
            .iter()
            .map(|block| block.carried_parent_ids.len())
            .max()
            .unwrap_or(0),
        support_slots,
        blocks: block_count,
        hdf5_bytes: manifest.hdf5_bytes,
        hdf5_sha256: manifest.hdf5_sha256,
        dataset_payload_bytes,
        hdf5_to_payload_ratio: manifest.hdf5_bytes as f64 / dataset_payload_bytes as f64,
        projected_hdf5_bytes,
        disk_required_free_bytes: disk_admission.required_free_bytes,
        disk_available_free_bytes: disk_admission.available_free_bytes,
        dependency_cone_bytes: replay.dependency_cone.total_bytes,
        dependency_cone_provider_bytes: replay.dependency_cone.provider_bytes,
        dependency_cone_blocks: replay.dependency_cone.block_ids.len(),
        artifact_block_read_cap_bytes,
        artifact_block_reads: artifact_metrics.operator_block_reads,
        artifact_block_cache_hits: artifact_metrics.operator_block_cache_hits,
        artifact_retained_block_payload_bytes_loaded,
        artifact_block_read_seconds,
        artifact_retained_payload_rate_bytes_per_second,
        artifact_cache_current_payload_bytes,
        artifact_cache_peak_payload_bytes,
        artifact_cache_reservation_bytes: artifact_metrics.block_reservation_bytes,
        artifact_cached_block_id: artifact_metrics.cached_block_id,
        artifact_verification_seconds,
        artifact_open_seconds,
        capture_overlap_topology_reads: write_receipt.overlap_topology_reads,
        capture_overlap_topology_bytes: write_receipt.overlap_topology_bytes,
        capture_peak_retained_topology_blocks: write_receipt.peak_retained_topology_blocks,
        capture_peak_frontier_chains: write_receipt.peak_frontier_chains,
        identity_index_bytes_read: write_receipt.identity_index_bytes_read,
        identity_index_bytes_written: write_receipt.identity_index_bytes_written,
        identity_index_peak_disk_bytes: write_receipt.peak_identity_index_disk_bytes,
        identity_index_peak_block_records: write_receipt.peak_identity_block_records,
        identity_index_merges: write_receipt.identity_index_merges,
        identity_index_disk_cap_bytes: write_receipt.identity_index_disk_cap_bytes,
        source_provider: "synthetic-formula-v1".to_owned(),
        source_manifest_digest: strong_digest(namespace.source_manifest_digest),
        source_model_version_digest: strong_digest(namespace.source_model_version_digest),
        source_model_receipt_digest: strong_digest(source_model_receipt_digest()),
        source_resolve_calls: metrics.calls.load(Ordering::Relaxed),
        raw_source_resolved_bytes,
        source_factor_resolved_bytes,
        source_resolved_bytes,
        source_cache_bytes: replay.source_cache_peak_bytes,
        source_resolution_throughput_bytes_per_second,
        capture_wall_seconds,
        query_wall_seconds,
        wall_seconds,
        peak_rss_bytes: peak_rss_bytes(),
        rayon_threads: rayon::current_num_threads(),
        covariance_dimension: replay.covariance.nrows(),
        covariance_checksum,
    })
}

fn benchmark_config(half_window: HalfWindow) -> SequentialConfig {
    SequentialConfig {
        ministack_size: MINISTACK_SIZE,
        max_num_compressed: MAX_CARRIED,
        half_window,
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
        shp_alpha: 0.05,
    }
}

fn benchmark_scope() -> ReplayExecutionScope {
    ReplayExecutionScope {
        enabled: true,
        backend: ReplayBackend::CpuF64,
        estimator_fallback: false,
        phase_bias_correction: false,
        strong_source_identity: true,
        stitched_burst_count: 1,
    }
}

fn benchmark_namespace(case: CaseConfig) -> ReplayIdNamespace {
    ReplayIdNamespace {
        burst_id: BENCHMARK_BURST_ID.to_owned(),
        source_manifest_digest: source_manifest_digest(case),
        source_model_version_digest: source_model_identity_digest(),
        native_origin: (0, 0),
        output_origin: (0, 0),
        owned_output_origin: (0, 0),
        owned_output_shape: (case.output_size, case.output_size),
    }
}

fn source_manifest_digest(case: CaseConfig) -> [u8; 32] {
    Sha256::digest(
        format!(
            "synthetic-formula-v1;dates={};native-shape={}x{}",
            case.dates, case.output_size, case.output_size
        )
        .as_bytes(),
    )
    .into()
}

fn benchmark_metadata(
    topology: &SequentialReplayTopology,
    source_manifest_digest: [u8; 32],
) -> CovarianceOperatorMetadata {
    CovarianceOperatorMetadata {
        normalized_config_digest: strong_digest(topology.normalized_config_digest()),
        kernel_digest: strong_digest(sequential_replay_kernel_digest()),
        source: SourceReplayIdentity {
            manifest_digest: Some(strong_digest(source_manifest_digest)),
            provider: Some(SOURCE_PROVIDER.to_owned()),
            provider_version: Some(SOURCE_PROVIDER_VERSION.to_owned()),
            model: Some(SOURCE_MODEL.to_owned()),
            model_version: Some(SOURCE_MODEL_VERSION.to_owned()),
            model_version_digest: Some(strong_digest(source_model_identity_digest())),
            model_receipt_digest: Some(strong_digest(source_model_receipt_digest())),
        },
        replay_status: CovarianceReplayStatus::Replayable,
        stitched_status: StitchedCovarianceStatus::NotStitched,
        calibration_status: CovarianceCalibrationStatus::Uncalibrated,
        downstream_inference_status: DownstreamInferenceStatus::BlockedPendingIssue54And53,
        ..CovarianceOperatorMetadata::default()
    }
}

fn source_model_identity_digest() -> [u8; 32] {
    sequential_source_model_identity_digest(
        SOURCE_PROVIDER,
        SOURCE_PROVIDER_VERSION,
        SOURCE_MODEL,
        SOURCE_MODEL_VERSION,
    )
}

fn source_model_receipt_digest() -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:synthetic_source_model:v1");
    digest.update(SOURCE_MODEL.as_bytes());
    digest.update(SOURCE_MODEL_VERSION.as_bytes());
    digest.update(0.02_f64.to_le_bytes());
    digest.update(b"proper-complex-real-embedding");
    digest.finalize().into()
}

fn bind_benchmark_factor_receipts(block: &mut CovarianceOperatorBlock) -> Result<()> {
    block.source_factor_digests.clear();
    for &source_id in &block.source_ids {
        let factor = ProperComplexFactor::new(
            dolphin_phaselink::SourceId::new(source_id),
            block
                .source_date_indices
                .iter()
                .copied()
                .map(u64::from)
                .collect(),
            source_model_receipt_digest(),
            Array2::from_diag_elem(block.source_date_indices.len(), Cf64::new(0.02, 0.0)),
        )
        .map_err(|error| anyhow::anyhow!(error))?;
        block
            .source_factor_digests
            .extend_from_slice(&factor.numeric_receipt_digest());
    }
    Ok(())
}

fn synthetic_sample(date: usize, row: usize, column: usize) -> Cf64 {
    let amplitude = 1.0 + 0.07 * date as f64 + 0.01 * (row + column) as f64;
    let phase = 0.4 + 0.11 * date as f64 + 0.017 * row as f64 - 0.013 * column as f64;
    Cf64::from_polar(amplitude, phase)
}

fn estimated_dataset_payload_bytes(case: CaseConfig, support_slots: usize) -> Result<u64> {
    let area = case
        .output_size
        .checked_mul(case.output_size)
        .context("benchmark area overflow")?;
    let blocks = case.dates.div_ceil(MINISTACK_SIZE);
    (0..blocks).try_fold(0_u64, |total, generation| {
        let start = generation * MINISTACK_SIZE;
        let real_dates = (case.dates - start).min(MINISTACK_SIZE);
        let carried = generation.min(MAX_CARRIED);
        let block = estimated_block_payload_bytes(area, support_slots, real_dates, carried)?;
        total
            .checked_add(block)
            .context("payload estimate overflow")
    })
}

fn estimated_block_payload_bytes(
    area: usize,
    support_slots: usize,
    real_dates: usize,
    carried: usize,
) -> Result<u64> {
    let components = real_dates
        .checked_add(carried)
        .context("phase component count overflow")?;
    let bytes = (BENCHMARK_BURST_ID.len() as u128)
        + 64
        + (real_dates as u128) * 8
        + (area as u128) * 8
        + (area as u128) * 32
        + (area as u128) * 32
        + (area as u128) * 8
        + (area as u128) * 8
        + (carried as u128) * 8
        + (components as u128) * 10
        + (area as u128) * 4
        + (area as u128) * (components as u128) * 8
        + (area as u128) * 16
        + (area as u128) * 2
        + (area as u128) * 16
        + (area as u128) * 8
        + (area as u128) * (support_slots.div_ceil(8) as u128)
        + (area.div_ceil(8) as u128)
        + (area as u128) * 8
        + (area as u128) * 8
        + (area as u128) * 2;
    u64::try_from(bytes).context("block payload estimate exceeds u64")
}

fn block_dataset_payload_bytes(block: &CovarianceOperatorBlock) -> Option<u64> {
    let entries = [
        (block.burst_id.len(), 1_usize),
        (block.source_manifest_digest.len(), 1),
        (block.source_model_version_digest.len(), 1),
        (block.source_date_indices.len(), 4_usize),
        (block.ordered_date_indices.len(), 4),
        (block.source_ids.len(), 8),
        (block.source_content_digests.len(), 1),
        (block.source_factor_digests.len(), 1),
        (block.phase_node_ids.len(), 8),
        (block.compressed_node_ids.len(), 8),
        (block.carry_parent_ids.len(), 8),
        (block.phase_components.len(), 2 + 8),
        (block.nearest_output_map.len(), 4),
        (block.phase_angles.len(), 8),
        (block.compressed_raster.len(), 16),
        (block.compressed_status.len(), 2),
        (block.projection_accumulator.len(), 16),
        (block.mean_amplitude.len(), 8),
        (block.support_bits.len(), 1),
        (block.native_validity_bits.len(), 1),
        (block.selected_eigenvalue.len(), 8),
        (block.eigen_gap.len(), 8),
        (block.status.len(), 2),
    ];
    entries.into_iter().try_fold(0_u64, |total, (len, width)| {
        let bytes = len
            .checked_mul(width)
            .and_then(|value| u64::try_from(value).ok())?;
        total.checked_add(bytes)
    })
}

fn block_topology_payload_bytes(block: &CovarianceOperatorBlock) -> Option<u64> {
    let entries = [
        (block.burst_id.len(), 1_usize),
        (block.source_manifest_digest.len(), 1),
        (block.source_model_version_digest.len(), 1),
        (block.source_date_indices.len(), 4),
        (block.ordered_date_indices.len(), 4),
        (block.source_ids.len(), 8),
        (block.source_content_digests.len(), 1),
        (block.source_factor_digests.len(), 1),
        (block.phase_node_ids.len(), 8),
        (block.compressed_node_ids.len(), 8),
        (block.carry_parent_ids.len(), 8),
        (block.phase_components.len(), 2 + 8),
    ];
    entries.into_iter().try_fold(0_u64, |total, (len, width)| {
        let bytes = len
            .checked_mul(width)
            .and_then(|value| u64::try_from(value).ok())?;
        total.checked_add(bytes)
    })
}

fn temporary_directory(case: CaseConfig) -> PathBuf {
    std::env::temp_dir().join(format!(
        "dolphin-sequential-covariance-bench-{}-{}-{}",
        std::process::id(),
        case.dates,
        case.output_size
    ))
}

fn strong_digest(digest: [u8; 32]) -> String {
    let hexadecimal: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    format!("sha256:{hexadecimal}")
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn peak_rss_bytes() -> Option<u64> {
    // SAFETY: usage is initialized for getrusage, which writes a complete
    // rusage value for the current live process.
    let usage = unsafe {
        let mut usage = std::mem::zeroed::<libc::rusage>();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            return None;
        }
        usage
    };
    let rss = u64::try_from(usage.ru_maxrss).ok()?;
    #[cfg(target_os = "macos")]
    return Some(rss);
    #[cfg(target_os = "linux")]
    return rss.checked_mul(1024);
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn peak_rss_bytes() -> Option<u64> {
    None
}

//! Release JSONL runner for fixed-factor and same-seed #52/#54 validation paths.

use std::collections::BTreeMap;

use dolphin_core::config::{CompressedSlcPlan, ComputeBackend, ShpMethod};
use dolphin_core::Cf64;
use dolphin_io::{
    covariance::covariance_operator_block_sha256, CovarianceOperatorBlock, CovarianceOperatorGrid,
};
use dolphin_phaselink::{
    estimate_empirical_proper_complex_factor, ComputeEngine, EmpiricalProperComplexConfig, NodeId,
    SourceId,
};
use dolphin_timeseries::inversion::{fixed_l2_pixel_map, PixelL2ObservationMap};
use dolphin_timeseries::spatial_covariance::{
    propagate_fixed_l2_difference_covariance, SpatialL2Branch, FIXED_L2_SPATIAL_COVARIANCE_METHOD,
};
use dolphin_timeseries::{fit_temporal_covariance, fit_temporal_covariance_from_factor_prefit};
use dolphin_timeseries::{
    fit_temporal_factor_complete_refit_bootstrap, fit_temporal_factor_scalar_batch,
    temporal_covariance_provenance, Sha256Digest, TemporalCovarianceFit, TemporalCovarianceOptions,
    TemporalCovariancePrefit, TemporalCovarianceProvenance, TemporalCovarianceProvenanceInputs,
    TemporalInferenceStatus, TemporalReferenceProvenance, TemporalValidationScope,
};
use dolphin_workflows::{
    estimate_global_reference_difference_covariance_from_provider_bundle,
    replay_global_reference_difference_covariance_from_provider_bundle, run_sequential,
    run_sequential_with_covariance_capture_and_source_factors, sequential_replay_kernel_digest,
    sequential_source_model_identity_digest, GlobalBlockId, GlobalDateId,
    GlobalReferenceCovarianceQuery, ReplayBackend, ReplayExecutionScope, ReplayIdNamespace,
    ReplayStatus, ResolvedCompressionReplay, ResolvedPhaseReplay, ResolvedPrimitiveSource,
    SequentialConfig, SequentialCovarianceCaptureRequest, SequentialPrimitiveSourceResolver,
    SequentialReplayBlock, SequentialReplayError, SequentialReplayTopology,
    SequentialSourceProviderIdentity, SequentialSourceReplayProvider, SequentialTileReplayProvider,
    SourceCorrelationModel,
};
use faer::{Mat, Side};
use ndarray::{s, Array1, Array2, Array3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{self, BufRead, Write};
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ExecutionPath {
    FixedFactor,
    ProductionPath,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    cell_id: String,
    cell_index: usize,
    outer_seed_index: usize,
    seed_sha256: String,
    seed: u64,
    days: Vec<f64>,
    options: TemporalCovarianceOptions,
    production_path: ProductionPathInput,
    #[serde(default)]
    retain_dense_evidence: bool,
    #[serde(default)]
    conditional_oracle_replicates: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProductionPathInput {
    source_seed: u64,
    native_shape: [usize; 2],
    target: [usize; 2],
    reference_pixel: [usize; 2],
    raw_complex_stack: Vec<Vec<[f64; 2]>>,
    carrier_stack: Vec<Vec<[f64; 2]>>,
    intended_difference_variance: Vec<f64>,
    latent_ar_path: Vec<f64>,
    measurement_normal_path: Vec<f64>,
    truth_slope_per_day: f64,
    source_correlation_model: String,
    source_correlation_distance_scale_pixels: f64,
    outer_coverage_dgp: String,
    conditional_covariance_oracle: String,
    validity: Vec<bool>,
    reference: TemporalReferenceProvenance,
    scope: TemporalValidationScope,
    capture_scope_sha256: Sha256Digest,
    validation_receipt_sha256: Sha256Digest,
    selected_method: String,
}

#[derive(Debug, Serialize)]
struct ProductionReceipts {
    capture_scope_sha256: Sha256Digest,
    source_manifest_sha256: Sha256Digest,
    source_model_sha256: Sha256Digest,
    evd_operator_sha256: Sha256Digest,
    evd_source_factor_sha256: Sha256Digest,
    fixed_l2_map_sha256: Sha256Digest,
    issue52_receipt_sha256: Sha256Digest,
    issue54_receipt_sha256: Sha256Digest,
    numeric_evidence_sha256: Sha256Digest,
    temporal_dgp_receipt_sha256: Sha256Digest,
    fixed_l2_difference_factor_sha256: Sha256Digest,
    fixed_l2_realized_rank: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    fixed_l2_difference_factor: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fixed_l2_difference_covariance: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fixed_l2_difference_variance: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    carrier_difference_history: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    linked_difference_history: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_carrier_difference_history: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_linked_difference_history: Option<Vec<f64>>,
    source_correlation_model: &'static str,
    source_correlation_distance_scale_pixels: f64,
    source_correlation_support_union_count: usize,
    effective_looks_fraction: f64,
    source_correlation_receipt_sha256: Sha256Digest,
    outer_coverage_dgp: &'static str,
    conditional_covariance_oracle: &'static str,
    conditional_oracle_replicates: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    conditional_oracle_covariance: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    conditional_oracle_receipt_sha256: Option<Sha256Digest>,
    temporal_profile_fit_count: usize,
    temporal_bootstrap_attempts: usize,
}

#[derive(Debug, Serialize)]
struct Response {
    schema: &'static str,
    execution_path: ExecutionPath,
    cell_id: String,
    cell_index: usize,
    outer_seed_index: usize,
    seed_sha256: String,
    seed: u64,
    factor_sha256: Option<Sha256Digest>,
    realized_factor_rank: Option<usize>,
    fixed_factor_status: Option<TemporalInferenceStatus>,
    production_path_status: Option<&'static str>,
    comparator_methods: [&'static str; 8],
    attempted: bool,
    emitted: bool,
    failed: bool,
    fit: Option<TemporalCovarianceFit>,
    provenance: Option<TemporalCovarianceProvenance>,
    production_receipts: Option<ProductionReceipts>,
    record_sha256: Sha256Digest,
}

#[derive(Debug, Serialize)]
struct FrameResourceReceipt {
    schema: &'static str,
    request_count: usize,
    record_count: usize,
    factor_generation_count: usize,
    temporal_fit_count: usize,
    profile_fit_count: usize,
    bootstrap_attempts: usize,
    attempt_record_count: usize,
    rayon_worker_count: usize,
    wall_micros: u128,
    resident_set_bytes_before: u64,
    resident_set_bytes_after: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrameRequest {
    schema: String,
    requests: Vec<Request>,
}

#[derive(Debug, Serialize)]
struct FrameResponse {
    schema: &'static str,
    records: Vec<Response>,
    resource: FrameResourceReceipt,
}

#[derive(Debug, Serialize)]
struct DirectFactorReceipt {
    schema: &'static str,
    acquisition_count: usize,
    post_gauge_date_count: usize,
    source_manifest_sha256: Sha256Digest,
    issue52_operator_receipt_sha256: Sha256Digest,
    issue52_source_factor_receipt_sha256: Sha256Digest,
    issue54_fixed_l2_map_receipt_sha256: Sha256Digest,
    issue54_replay_source_factor_receipt_sha256: Sha256Digest,
    issue54_replay_support_receipt_sha256: Sha256Digest,
    reference_signature_sha256: Sha256Digest,
    factor_sha256: Sha256Digest,
    realized_rank: usize,
    covariance_condition_number: f64,
    positive_off_diagonal_energy: f64,
    replay_resource_high_water_bytes: u64,
    effective_looks_fraction: f64,
    source_correlation_support_union_count: usize,
    source_correlation_receipt_sha256: Sha256Digest,
    difference_covariance_sha256: Sha256Digest,
    difference_factor: Vec<Vec<f64>>,
    factor_generation_count: usize,
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    if arguments.get(1).map(String::as_str) == Some("--resource-factor") {
        if arguments.len() != 3 {
            return Err("resource factor mode requires one acquisition count".into());
        }
        let acquisition_count = arguments[2].parse::<usize>()?;
        if ![13, 49, 97].contains(&acquisition_count) {
            return Err("resource factor acquisition count must be 13, 49, or 97".into());
        }
        let branch = direct_production_branch(acquisition_count)?;
        if branch.realized_rank != acquisition_count - 1 {
            return Err("resource factor did not realize exact full post-gauge rank".into());
        }
        let positive_off_diagonal_energy = branch
            .difference_covariance
            .iter()
            .enumerate()
            .flat_map(|(row, values)| {
                values
                    .iter()
                    .enumerate()
                    .filter(move |(column, _)| *column != row)
                    .map(|(_, value)| value.abs())
            })
            .sum::<f64>();
        if !positive_off_diagonal_energy.is_finite() || positive_off_diagonal_energy <= 0.0 {
            return Err("resource factor covariance has no positive off-diagonal energy".into());
        }
        let receipt = DirectFactorReceipt {
            schema: "dolphinrust-temporal-direct-factor/1",
            acquisition_count,
            post_gauge_date_count: acquisition_count - 1,
            source_manifest_sha256: digest_bytes([0x42; 32]),
            issue52_operator_receipt_sha256: branch.operator_receipt,
            issue52_source_factor_receipt_sha256: branch.source_factor_receipt,
            issue54_fixed_l2_map_receipt_sha256: branch.fixed_l2_map_receipt,
            issue54_replay_source_factor_receipt_sha256: branch.replay_source_factor_receipt,
            issue54_replay_support_receipt_sha256: branch.replay_support_receipt,
            reference_signature_sha256: branch.reference_signature,
            factor_sha256: branch.difference_factor_sha256,
            realized_rank: branch.realized_rank,
            covariance_condition_number: branch.covariance_condition_number,
            positive_off_diagonal_energy,
            replay_resource_high_water_bytes: branch.replay_resource_high_water_bytes,
            effective_looks_fraction: branch.effective_looks_fraction,
            source_correlation_support_union_count: branch.source_correlation_support_union_count,
            source_correlation_receipt_sha256: branch.source_correlation_receipt,
            difference_covariance_sha256: digest_json(&branch.difference_covariance),
            difference_factor: branch.difference_factor,
            factor_generation_count: 1,
        };
        serde_json::to_writer(io::stdout().lock(), &receipt)?;
        return Ok(());
    }
    if arguments.len() != 1 {
        return Err("temporal batch accepts JSON frames or --resource-factor".into());
    }
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    let mut line = Vec::new();
    while read_bounded_line(&mut input, &mut line, MAX_REQUEST_LINE_BYTES)? {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let frame = decode_frame_request(&line)?;
        let request_count = frame.requests.len();
        let rss_before = resident_set_bytes();
        let started = Instant::now();
        let records = evaluate_frame_records(frame.requests)?;
        let rss_after = resident_set_bytes();
        let profile_fit_count = records
            .iter()
            .filter_map(|record| record.production_receipts.as_ref())
            .map(|receipt| receipt.temporal_profile_fit_count)
            .sum();
        let bootstrap_attempts = records
            .iter()
            .filter_map(|record| record.production_receipts.as_ref())
            .map(|receipt| receipt.temporal_bootstrap_attempts)
            .sum();
        let response = FrameResponse {
            schema: TEMPORAL_BATCH_SCHEMA,
            records,
            resource: FrameResourceReceipt {
                schema: TEMPORAL_BATCH_RESOURCE_SCHEMA,
                request_count,
                record_count: request_count * 2,
                factor_generation_count: request_count,
                temporal_fit_count: request_count,
                profile_fit_count,
                bootstrap_attempts,
                attempt_record_count: request_count * 2,
                rayon_worker_count: rayon::current_num_threads(),
                wall_micros: started.elapsed().as_micros(),
                resident_set_bytes_before: rss_before,
                resident_set_bytes_after: rss_after,
            },
        };
        let encoded = serde_json::to_vec(&response)?;
        if encoded.len() > MAX_RESPONSE_LINE_BYTES {
            return Err("temporal covariance response exceeds its line cap".into());
        }
        output.write_all(&encoded)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}

fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    line: &mut Vec<u8>,
    cap: usize,
) -> io::Result<bool> {
    line.clear();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(!line.is_empty());
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let take = newline.unwrap_or(available.len());
        if line.len().saturating_add(take) > cap {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "temporal covariance request exceeds its line cap",
            ));
        }
        line.extend_from_slice(&available[..take]);
        reader.consume(take + usize::from(newline.is_some()));
        if newline.is_some() {
            return Ok(true);
        }
    }
}

fn decode_frame_request(bytes: &[u8]) -> io::Result<FrameRequest> {
    if bytes.len() > MAX_REQUEST_LINE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "temporal covariance frame exceeds its byte cap",
        ));
    }
    let frame: FrameRequest = serde_json::from_slice(bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("temporal covariance frame is malformed: {error}"),
        )
    })?;
    if frame.schema != TEMPORAL_BATCH_SCHEMA || !(1..=32).contains(&frame.requests.len()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "temporal covariance frame schema or request count is invalid",
        ));
    }
    let first = &frame.requests[0];
    let seeds_are_one_cell_and_consecutive =
        frame.requests.iter().enumerate().all(|(offset, request)| {
            request.cell_id == first.cell_id
                && request.cell_index == first.cell_index
                && first
                    .outer_seed_index
                    .checked_add(offset)
                    .is_some_and(|expected| request.outer_seed_index == expected)
        });
    if !seeds_are_one_cell_and_consecutive {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "temporal covariance frame must contain consecutive seeds for one cell",
        ));
    }
    Ok(frame)
}

fn evaluate_frame_records(requests: Vec<Request>) -> Result<Vec<Response>, &'static str> {
    if !(1..=32).contains(&requests.len()) {
        return Err("temporal covariance frame request count is invalid");
    }
    let pairs = if requests.len() == 1 {
        vec![evaluate_seed_request(
            requests
                .into_iter()
                .next()
                .expect("one-request frame is nonempty"),
        )]
    } else {
        requests
            .into_par_iter()
            .map(evaluate_seed_request)
            .collect::<Vec<_>>()
    };
    let mut records = Vec::with_capacity(pairs.len() * 2);
    for pair in pairs {
        records.extend(pair);
    }
    Ok(records)
}

fn evaluate_seed_request(request: Request) -> [Response; 2] {
    let evaluation = match prepare_production(&request) {
        Ok(prepared) => evaluate_prepared_production(&request, prepared),
        Err(status) => (None, Some(status), None, None, None),
    };
    let (_, production_path_status, fit, provenance, production_receipts) = evaluation;
    let fixed_factor_status = fit.as_ref().map(|fit| fit.status);
    let factor_sha256 = production_receipts
        .as_ref()
        .map(|receipt| receipt.fixed_l2_difference_factor_sha256.clone());
    let realized_factor_rank = production_receipts
        .as_ref()
        .map(|receipt| receipt.fixed_l2_realized_rank);
    let fixed = response_record(
        &request,
        ExecutionPath::FixedFactor,
        fixed_factor_status,
        None,
        fit.clone(),
        None,
        None,
        factor_sha256.clone(),
        realized_factor_rank,
    );
    let production = response_record(
        &request,
        ExecutionPath::ProductionPath,
        None,
        production_path_status,
        fit,
        provenance,
        production_receipts,
        factor_sha256,
        realized_factor_rank,
    );
    [fixed, production]
}

#[allow(clippy::too_many_arguments)]
fn response_record(
    request: &Request,
    execution_path: ExecutionPath,
    fixed_factor_status: Option<TemporalInferenceStatus>,
    production_path_status: Option<&'static str>,
    fit: Option<TemporalCovarianceFit>,
    provenance: Option<TemporalCovarianceProvenance>,
    production_receipts: Option<ProductionReceipts>,
    factor_sha256: Option<Sha256Digest>,
    realized_factor_rank: Option<usize>,
) -> Response {
    let emitted = fit
        .as_ref()
        .is_some_and(|value| value.status == TemporalInferenceStatus::Evaluated);
    let comparator_methods = [
        "ols",
        "oracle_gls",
        "legacy_intercept_slope_wls_non_comparable",
        "lag_one_scalar_effective_n",
        "plugin_gls_reml",
        "reml_covariance_parameter_adjusted_scalar",
        "slope_profile_likelihood_ml",
        "complete_refit_bootstrap",
    ];
    let record_sha256 = semantic_digest_json(&(
        TEMPORAL_BATCH_SCHEMA,
        execution_path,
        request.cell_id.as_str(),
        request.cell_index,
        request.outer_seed_index,
        request.seed_sha256.as_str(),
        request.seed,
        &factor_sha256,
        realized_factor_rank,
        fixed_factor_status,
        production_path_status,
        comparator_methods,
        &fit,
        &provenance,
        &production_receipts,
    ));
    Response {
        schema: TEMPORAL_BATCH_SCHEMA,
        execution_path,
        cell_id: request.cell_id.clone(),
        cell_index: request.cell_index,
        outer_seed_index: request.outer_seed_index,
        seed_sha256: request.seed_sha256.clone(),
        seed: request.seed,
        factor_sha256,
        realized_factor_rank,
        fixed_factor_status,
        production_path_status,
        comparator_methods,
        attempted: true,
        emitted,
        failed: !emitted,
        fit,
        provenance,
        production_receipts,
        record_sha256,
    }
}

#[cfg(test)]
fn canonical_record_bytes(response: &Response) -> Vec<u8> {
    serde_json::to_vec(response).expect("temporal covariance response record must serialize")
}

type Evaluation = (
    Option<TemporalInferenceStatus>,
    Option<&'static str>,
    Option<TemporalCovarianceFit>,
    Option<TemporalCovarianceProvenance>,
    Option<ProductionReceipts>,
);

struct PreparedProduction {
    branch: ProductionBranch,
    carrier_difference_history: Vec<f64>,
    source_carrier_difference_history: Vec<f64>,
    source_linked_difference_history: Vec<f64>,
    source_manifest: [u8; 32],
    source_model_digest: [u8; 32],
    capture_scope: Sha256Digest,
}

#[allow(clippy::too_many_lines)]
fn prepare_production(request: &Request) -> Result<PreparedProduction, &'static str> {
    let input = &request.production_path;
    if input.source_seed != request.seed {
        return Err("source_seed_mismatch");
    }
    if input.scope != TemporalValidationScope::SyntheticValidation
        || input.selected_method != SELECTED_METHOD
        || input.source_correlation_model != SOURCE_CORRELATION_MODEL
        || input.source_correlation_distance_scale_pixels
            != SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS
        || input.outer_coverage_dgp != OUTER_COVERAGE_DGP
        || input.conditional_covariance_oracle != CONDITIONAL_COVARIANCE_ORACLE
        || request.conditional_oracle_replicates == 1
        || request.conditional_oracle_replicates > MAX_CONDITIONAL_ORACLE_REPLICATES
    {
        return Err("production_contract_mismatch");
    }
    let capture_scope = capture_scope_digest(request, input);
    if capture_scope != input.capture_scope_sha256 {
        return Err("capture_scope_mismatch");
    }
    let dates = request.days.len();
    let native_area = input.native_shape[0].saturating_mul(input.native_shape[1]);
    if dates < 2
        || input.native_shape != [1, 7]
        || input.target[0] >= input.native_shape[0]
        || input.target[1] >= input.native_shape[1]
        || input.reference_pixel[0] >= input.native_shape[0]
        || input.reference_pixel[1] >= input.native_shape[1]
        || input.raw_complex_stack.len() != dates
        || input.carrier_stack.len() != dates
        || input.intended_difference_variance.len() != dates
        || input.latent_ar_path.len() != dates
        || input.measurement_normal_path.len() != dates
        || input
            .raw_complex_stack
            .iter()
            .any(|row| row.len() != native_area)
        || input
            .carrier_stack
            .iter()
            .any(|row| row.len() != native_area)
        || input
            .intended_difference_variance
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || input.intended_difference_variance[0] != 0.0
        || input.latent_ar_path.iter().any(|value| !value.is_finite())
        || input
            .measurement_normal_path
            .iter()
            .any(|value| !value.is_finite())
        || input.latent_ar_path[0] != 0.0
        || !input.truth_slope_per_day.is_finite()
        || input.validity.len() != dates
        || !input.validity.first().copied().unwrap_or(false)
    {
        return Err("raw_complex_invalid");
    }
    let stack =
        complex_stack(&input.raw_complex_stack, input.native_shape).ok_or("raw_complex_invalid")?;
    let carrier =
        complex_stack(&input.carrier_stack, input.native_shape).ok_or("raw_complex_invalid")?;
    let source_model = source_model_config().map_err(|_| "source_model_invalid")?;
    let source_carrier_difference_history = difference_history(
        &carrier,
        (input.target[0], input.target[1]),
        (input.reference_pixel[0], input.reference_pixel[1]),
    )?;
    let source_manifest = source_manifest_digest(request, input, &stack);
    let mut branch = run_production_branch(
        &stack,
        source_manifest,
        &source_model,
        (input.target[0] as u64, input.target[1] as u64),
        (
            input.reference_pixel[0] as u64,
            input.reference_pixel[1] as u64,
        ),
        &input.reference,
        (request.conditional_oracle_replicates > 0)
            .then_some((request.conditional_oracle_replicates, request.seed)),
    )?;
    if branch.source_correlation_model != input.source_correlation_model
        || branch.source_correlation_distance_scale_pixels
            != input.source_correlation_distance_scale_pixels
    {
        return Err("source_correlation_mismatch");
    }
    let source_linked_difference_history = branch.difference_history.clone();
    let (carrier_difference_history, linked_difference_history) = conforming_temporal_histories(
        &request.days,
        &input.validity,
        &branch.difference_factor,
        branch.realized_rank,
        &input.latent_ar_path,
        &input.measurement_normal_path,
        input.truth_slope_per_day,
        request.options.oracle_process_variance,
    )?;
    branch.difference_history = linked_difference_history;
    Ok(PreparedProduction {
        branch,
        carrier_difference_history,
        source_carrier_difference_history,
        source_linked_difference_history,
        source_manifest,
        source_model_digest: *source_model.config_digest(),
        capture_scope,
    })
}

#[allow(clippy::too_many_lines)]
fn evaluate_prepared_production(request: &Request, prepared: PreparedProduction) -> Evaluation {
    let input = &request.production_path;
    let source_manifest = prepared.source_manifest;
    let source_model_digest = prepared.source_model_digest;
    let carrier_difference_history = prepared.carrier_difference_history;
    let source_carrier_difference_history = prepared.source_carrier_difference_history;
    let source_linked_difference_history = prepared.source_linked_difference_history;
    let capture_scope = prepared.capture_scope;
    let evd = prepared.branch;
    let linked_difference_history = evd.difference_history.clone();
    let mut observations = linked_difference_history.clone();
    observations
        .iter_mut()
        .zip(&input.validity)
        .for_each(|(value, valid)| {
            if !valid {
                *value = f64::NAN;
            }
        });
    let issue52_receipt = digest_json(&(
        source_manifest,
        source_model_digest,
        evd.operator_receipt.as_str(),
        evd.source_factor_receipt.as_str(),
    ));
    let issue54_receipt = digest_json(&(
        FIXED_L2_SPATIAL_COVARIANCE_METHOD,
        evd.fixed_l2_map_receipt.as_str(),
        evd.replay_source_factor_receipt.as_str(),
        evd.replay_support_receipt.as_str(),
        evd.reference_signature.as_str(),
        input.source_correlation_model.as_str(),
        input.source_correlation_distance_scale_pixels,
        evd.source_correlation_support_union_count,
        evd.source_correlation_receipt.as_str(),
        PHASELINK_SOURCE_JVP_METHOD,
        PHASELINK_SOURCE_JVP_CONTRACT,
        PHASELINK_SPATIAL_JVP_CONTRACT,
        sequential_replay_kernel_digest(),
        &evd.difference_covariance,
    ));
    let retained = (1..request.days.len())
        .filter(|&index| observations[index].is_finite())
        .collect::<Vec<_>>();
    let retained_days = retained
        .iter()
        .map(|&index| request.days[index])
        .collect::<Vec<_>>();
    let retained_observations = retained
        .iter()
        .map(|&index| observations[index])
        .collect::<Vec<_>>();
    let (persisted_factor, persisted_maximum_rank, realized_rank) =
        if retained.len() == request.days.len() - 1 {
            let persisted_maximum_rank = request.days.len();
            let mut persisted_factor = vec![0.0; request.days.len() * persisted_maximum_rank];
            for (date, row) in evd.difference_factor.iter().enumerate() {
                persisted_factor[date * persisted_maximum_rank
                    ..date * persisted_maximum_rank + evd.realized_rank]
                    .copy_from_slice(row);
            }
            (persisted_factor, persisted_maximum_rank, evd.realized_rank)
        } else {
            let retained_count = retained.len();
            let mut covariance = vec![0.0; retained_count * retained_count];
            for (row, &source_row) in retained.iter().enumerate() {
                for (column, &source_column) in retained.iter().enumerate() {
                    covariance[row * retained_count + column] = (0..evd.realized_rank)
                        .map(|component| {
                            evd.difference_factor[source_row][component]
                                * evd.difference_factor[source_column][component]
                        })
                        .sum();
                }
            }
            let mut lower = vec![0.0; retained_count * retained_count];
            let mut factor_valid = retained_count > 0;
            for row in 0..retained_count {
                for column in 0..=row {
                    let mut value = covariance[row * retained_count + column];
                    for inner in 0..column {
                        value -= lower[row * retained_count + inner]
                            * lower[column * retained_count + inner];
                    }
                    if row == column {
                        if !value.is_finite() || value <= 0.0 {
                            factor_valid = false;
                            break;
                        }
                        lower[row * retained_count + column] = value.sqrt();
                    } else {
                        lower[row * retained_count + column] =
                            value / lower[column * retained_count + column];
                    }
                }
                if !factor_valid {
                    break;
                }
            }
            if factor_valid {
                let mut persisted_factor = vec![0.0; (retained_count + 1) * retained_count];
                persisted_factor[retained_count..].copy_from_slice(&lower);
                (persisted_factor, retained_count, retained_count)
            } else {
                (Vec::new(), 0, 0)
            }
        };
    let scalar_report = (realized_rank > 0).then(|| {
        fit_temporal_factor_scalar_batch(
            &retained_days,
            &retained_observations,
            &persisted_factor,
            persisted_maximum_rank,
            &[realized_rank],
            &request.options,
        )
    });
    let scalar_report = scalar_report.and_then(Result::ok);
    let scalar_pair = scalar_report
        .as_ref()
        .and_then(|report| report.outcomes.first());
    let dense_options = TemporalCovarianceOptions {
        bootstrap_replicates: 0,
        bootstrap_minimum_successes: 0,
        ..request.options.clone()
    };
    let bootstrap_report = scalar_pair.and_then(|pair| {
        fit_temporal_factor_complete_refit_bootstrap(
            &retained_days,
            &persisted_factor,
            persisted_maximum_rank,
            realized_rank,
            pair.plugin_gls_reml.point_estimate?,
            pair.fitted_rho?,
            pair.fitted_process_variance?,
            &request.options,
        )
        .ok()
    });
    let prefit = scalar_pair.and_then(|pair| {
        Some(TemporalCovariancePrefit {
            plugin_slope_per_day: pair.plugin_slope_per_day?,
            plugin_gls: pair.plugin_gls_reml.clone(),
            adjusted_scalar: pair.reml_covariance_parameter_adjusted_scalar.clone(),
            fitted_rho: pair.fitted_rho?,
            fitted_process_variance: pair.fitted_process_variance?,
            fitted_parameter_active_set: pair.fitted_parameter_active_set,
            covariance_condition_number: pair.exact_condition_number.or(pair.condition_upper_bound),
        })
    });
    let mut fit = prefit.as_ref().map_or_else(
        || {
            fit_temporal_covariance(
                &request.days,
                &observations,
                &evd.difference_covariance,
                &dense_options,
            )
        },
        |prefit| {
            fit_temporal_covariance_from_factor_prefit(
                &request.days,
                &observations,
                &evd.difference_covariance,
                &persisted_factor,
                persisted_maximum_rank,
                realized_rank,
                &dense_options,
                prefit,
            )
        },
    );
    if let Some(pair) = scalar_pair {
        fit.plugin_gls_slope = pair.plugin_gls_reml.point_estimate;
        fit.plugin_gls = pair.plugin_gls_reml.clone();
        fit.adjusted_scalar = pair.reml_covariance_parameter_adjusted_scalar.clone();
        fit.fitted_rho = pair.fitted_rho;
        fit.fitted_process_variance = pair.fitted_process_variance;
        fit.fitted_parameter_active_set = pair.fitted_parameter_active_set;
        fit.covariance_condition_number =
            pair.exact_condition_number.or(pair.condition_upper_bound);
    }
    if let Some(report) = &bootstrap_report {
        fit.bootstrap_slope = report.complete_refit_bootstrap.point_estimate;
        fit.bootstrap_interval = report.complete_refit_bootstrap.interval_95;
        fit.bootstrap_attempts = report.complete_refit_bootstrap.attempted_replicates;
        fit.bootstrap_successes = report.complete_refit_bootstrap.successful_replicates;
        fit.complete_refit_bootstrap = report.complete_refit_bootstrap.clone();
    }
    fit.status = if fit.adjusted_profile.status != TemporalInferenceStatus::Evaluated {
        fit.adjusted_profile.status
    } else if fit.complete_refit_bootstrap.status != TemporalInferenceStatus::Evaluated {
        fit.complete_refit_bootstrap.status
    } else {
        scalar_pair.map_or(fit.status, |pair| {
            pair.reml_covariance_parameter_adjusted_scalar.status
        })
    };
    let temporal_profile_fit_count = scalar_report
        .as_ref()
        .map_or(0, |report| report.metrics.profile_fit_count);
    let temporal_bootstrap_attempts = bootstrap_report
        .as_ref()
        .map_or(0, |report| report.metrics.bootstrap_attempts);
    let fixed_l2_difference_covariance = evd.difference_covariance.clone();
    let fixed_l2_difference_variance = fixed_l2_difference_covariance
        .iter()
        .enumerate()
        .map(|(index, row)| row[index])
        .collect();
    let numeric_evidence_sha256 = digest_json(&(
        &evd.difference_factor,
        evd.realized_rank,
        &fixed_l2_difference_covariance,
        &fixed_l2_difference_variance,
        &carrier_difference_history,
        &linked_difference_history,
        &source_carrier_difference_history,
        &source_linked_difference_history,
        &evd.conditional_oracle_covariance,
    ));
    let temporal_dgp_receipt_sha256 = digest_json(&(
        OUTER_COVERAGE_DGP,
        source_manifest,
        evd.difference_factor_sha256.as_str(),
        evd.realized_rank,
        &request.days,
        &input.validity,
        input.truth_slope_per_day,
        request.options.oracle_process_variance,
        request.options.oracle_rho,
        request.options.reference_lag_days,
        &input.latent_ar_path,
        &input.measurement_normal_path,
        &carrier_difference_history,
        &linked_difference_history,
        &source_carrier_difference_history,
        &source_linked_difference_history,
    ));
    let retain_dense = request.retain_dense_evidence;
    let provenance = temporal_covariance_provenance(
        &fit,
        TemporalCovarianceProvenanceInputs {
            issue52_receipt_sha256: issue52_receipt.clone(),
            issue54_receipt_sha256: issue54_receipt.clone(),
            reference: evd.reference,
            scope: input.scope,
            validation_receipt_sha256: input.validation_receipt_sha256.clone(),
            estimator_input_sha256: digest_json(&(
                input,
                &request.days,
                &request.options,
                &observations,
                source_manifest,
                evd.operator_receipt.as_str(),
                evd.replay_source_factor_receipt.as_str(),
                evd.difference_factor_sha256.as_str(),
                evd.realized_rank,
                temporal_dgp_receipt_sha256.as_str(),
            )),
            selected_method: input.selected_method.clone(),
        },
    );
    let conditional_oracle_receipt_sha256 =
        evd.conditional_oracle_covariance
            .as_ref()
            .map(|covariance| {
                digest_json(&(
                    CONDITIONAL_COVARIANCE_ORACLE,
                    request.conditional_oracle_replicates,
                    source_model_digest,
                    evd.operator_receipt.as_str(),
                    evd.source_factor_receipt.as_str(),
                    evd.replay_source_factor_receipt.as_str(),
                    evd.fixed_l2_map_receipt.as_str(),
                    evd.source_correlation_receipt.as_str(),
                    covariance,
                ))
            });
    let receipts = ProductionReceipts {
        capture_scope_sha256: capture_scope,
        source_manifest_sha256: digest_bytes(source_manifest),
        source_model_sha256: digest_bytes(source_model_digest),
        evd_operator_sha256: evd.operator_receipt,
        evd_source_factor_sha256: evd.source_factor_receipt,
        fixed_l2_map_sha256: evd.fixed_l2_map_receipt,
        issue52_receipt_sha256: issue52_receipt,
        issue54_receipt_sha256: issue54_receipt,
        numeric_evidence_sha256,
        temporal_dgp_receipt_sha256,
        fixed_l2_difference_factor_sha256: evd.difference_factor_sha256.clone(),
        fixed_l2_realized_rank: evd.realized_rank,
        fixed_l2_difference_factor: retain_dense.then_some(evd.difference_factor),
        fixed_l2_difference_covariance: retain_dense.then_some(fixed_l2_difference_covariance),
        fixed_l2_difference_variance: retain_dense.then_some(fixed_l2_difference_variance),
        carrier_difference_history: retain_dense.then_some(carrier_difference_history),
        linked_difference_history: retain_dense.then_some(linked_difference_history),
        source_carrier_difference_history: retain_dense
            .then_some(source_carrier_difference_history),
        source_linked_difference_history: retain_dense.then_some(source_linked_difference_history),
        source_correlation_model: evd.source_correlation_model,
        source_correlation_distance_scale_pixels: evd.source_correlation_distance_scale_pixels,
        source_correlation_support_union_count: evd.source_correlation_support_union_count,
        effective_looks_fraction: evd.effective_looks_fraction,
        source_correlation_receipt_sha256: evd.source_correlation_receipt,
        outer_coverage_dgp: OUTER_COVERAGE_DGP,
        conditional_covariance_oracle: CONDITIONAL_COVARIANCE_ORACLE,
        conditional_oracle_replicates: request.conditional_oracle_replicates,
        conditional_oracle_covariance: evd.conditional_oracle_covariance,
        conditional_oracle_receipt_sha256,
        temporal_profile_fit_count,
        temporal_bootstrap_attempts,
    };
    let status = if fit.status == TemporalInferenceStatus::Evaluated {
        "evaluated"
    } else {
        "estimator_failed"
    };
    (None, Some(status), Some(fit), provenance, Some(receipts))
}

fn sequential_config() -> SequentialConfig {
    SequentialConfig {
        ministack_size: 3,
        max_num_compressed: 1,
        half_window: dolphin_core::HalfWindow { y: 0, x: 1 },
        strides: dolphin_core::Strides { y: 1, x: 1 },
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

fn replay_scope() -> ReplayExecutionScope {
    ReplayExecutionScope {
        enabled: true,
        backend: ReplayBackend::CpuF64,
        estimator_fallback: false,
        phase_bias_correction: false,
        strong_source_identity: true,
        stitched_burst_count: 1,
    }
}

const SOURCE_PROVIDER: &str = "temporal-covariance-validation-memory";
const SOURCE_PROVIDER_VERSION: &str = "1";
const SOURCE_MODEL: &str = "source_centered_empirical_proper_complex_v1";
const SOURCE_MODEL_VERSION: &str = "1";
const SOURCE_MODEL_SHRINKAGE_ALPHA: f64 = 0.1;
const BRANCH_TOLERANCE: f64 = 1e-10;
const REPLAY_BYTE_CAP: u64 = 1 << 30;
const SOURCE_CORRELATION_MODEL: &str = "exponential_euclidean_v1";
const SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS: f64 = 1.5;
const OUTER_COVERAGE_DGP: &str = "actual_c54_gaussian_measurement_post_link_ar_v1";
const CONDITIONAL_COVARIANCE_ORACLE: &str = "fixed_capture_common_factor_monte_carlo_v1";
const SELECTED_METHOD: &str = "reml_covariance_parameter_adjusted_scalar";
const MAX_CONDITIONAL_ORACLE_REPLICATES: usize = 16_384;
const MAX_REQUEST_LINE_BYTES: usize = 4 * 1024 * 1024;
const MAX_RESPONSE_LINE_BYTES: usize = 4 * 1024 * 1024;
const TEMPORAL_BATCH_SCHEMA: &str = "dolphinrust-temporal-covariance-batch/7";
const TEMPORAL_BATCH_RESOURCE_SCHEMA: &str =
    "dolphinrust-temporal-covariance-batch-frame-resource/1";
const PHASELINK_SOURCE_JVP_METHOD: &str = "raw_complex_to_phase_source_jvp_v1";
const PHASELINK_SOURCE_JVP_CONTRACT: &str =
    "dolphin_phaselink::source_influence_contract::evd_phase_source_jvp_matches_raw_complex_difference";
const PHASELINK_SPATIAL_JVP_CONTRACT: &str =
    "dolphin_phaselink::spatial_reference_covariance_contract::evd_and_emi_source_jvps_match_finite_difference";

struct ProductionBranch {
    difference_history: Vec<f64>,
    difference_factor: Vec<Vec<f64>>,
    difference_factor_sha256: Sha256Digest,
    realized_rank: usize,
    covariance_condition_number: f64,
    difference_covariance: Vec<Vec<f64>>,
    operator_receipt: Sha256Digest,
    source_factor_receipt: Sha256Digest,
    fixed_l2_map_receipt: Sha256Digest,
    replay_source_factor_receipt: Sha256Digest,
    replay_support_receipt: Sha256Digest,
    reference_signature: Sha256Digest,
    source_correlation_model: &'static str,
    source_correlation_distance_scale_pixels: f64,
    source_correlation_support_union_count: usize,
    effective_looks_fraction: f64,
    source_correlation_receipt: Sha256Digest,
    replay_resource_high_water_bytes: u64,
    conditional_oracle_covariance: Option<Vec<Vec<f64>>>,
    reference: TemporalReferenceProvenance,
}

struct InMemoryProvider<'a> {
    identity: SequentialSourceProviderIdentity,
    topology: SequentialReplayTopology,
    blocks: BTreeMap<GlobalBlockId, CovarianceOperatorBlock>,
    stack: ndarray::ArrayView3<'a, Cf64>,
    validity: ndarray::ArrayView2<'a, bool>,
    source_model: &'a EmpiricalProperComplexConfig,
    data_identity: [u8; 32],
    factor_receipts: BTreeMap<SourceId, [u8; 32]>,
}

impl InMemoryProvider<'_> {
    fn resolve(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError> {
        let columns = self.stack.dim().2;
        let row = native_index / columns;
        let column = native_index % columns;
        let date_start = block.real_date_start.get() as usize;
        let date_stop = date_start + block.num_real_dates;
        let samples =
            Array1::from_iter((date_start..date_stop).map(|date| self.stack[(date, row, column)]));
        let mut content = Sha256::new();
        for sample in &samples {
            content.update(sample.re.to_bits().to_le_bytes());
            content.update(sample.im.to_bits().to_le_bytes());
        }
        let content_digest = content.finalize().into();
        let source =
            self.topology
                .source_id_for_content_digest(block.id, native_index, &content_digest)?;
        let component_ids = (date_start..date_stop)
            .map(|date| date as u64)
            .collect::<Vec<_>>();
        let estimate = estimate_empirical_proper_complex_factor(
            source,
            &component_ids,
            self.stack.slice(s![date_start..date_stop, .., ..]),
            self.validity,
            (0, 0),
            (0, 0),
            (self.stack.dim().1, self.stack.dim().2),
            (row, column),
            self.data_identity,
            self.source_model,
        )
        .map_err(|_| {
            SequentialReplayError::Provider(
                ReplayStatus::SourceModelUnavailable,
                "temporal validation empirical source factor is unavailable",
            )
        })?;
        let (factor, receipt) = estimate.into_parts();
        self.factor_receipts.insert(source, *receipt.digest());
        Ok(ResolvedPrimitiveSource {
            id: source,
            samples,
            factor,
            content_digest,
        })
    }
}

impl SequentialSourceReplayProvider for InMemoryProvider<'_> {
    fn identity(&self) -> &SequentialSourceProviderIdentity {
        &self.identity
    }

    fn maximum_resident_bytes(&self) -> u64 {
        (self.stack.len() * std::mem::size_of::<Cf64>() + self.validity.len()) as u64
    }

    fn resolve_source(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError> {
        self.resolve(block, native_index)
    }

    fn resolve_phase(
        &mut self,
        block: &SequentialReplayBlock,
        output_index: usize,
    ) -> Result<ResolvedPhaseReplay, SequentialReplayError> {
        let stored = self
            .blocks
            .get(&block.id)
            .ok_or(SequentialReplayError::Provider(
                ReplayStatus::SourceUnavailable,
                "temporal validation operator block is absent",
            ))?;
        let width = stored.phase_components.len();
        let bits = stored.support_bits_per_output as usize;
        let bytes = bits.div_ceil(8);
        let packed = &stored.support_bits[output_index * bytes..(output_index + 1) * bytes];
        Ok(ResolvedPhaseReplay {
            id: NodeId::new(stored.phase_node_ids[output_index]),
            linked_phase: Array1::from_iter(
                stored.phase_angles[output_index * width..(output_index + 1) * width]
                    .iter()
                    .map(|&angle| Cf64::from_polar(1.0, angle)),
            ),
            selected_eigenvalue: stored.selected_eigenvalue[output_index],
            selected_eigengap: stored.eigen_gap[output_index],
            realized_support: (0..bits)
                .map(|slot| packed[slot / 8] & (1 << (slot % 8)) != 0)
                .collect(),
            status: stored.status[output_index],
            estimator_branch: stored.estimator_branch,
            branch_tolerance: stored.branch_tolerance,
        })
    }

    fn resolve_compression(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedCompressionReplay, SequentialReplayError> {
        let stored = self
            .blocks
            .get(&block.id)
            .ok_or(SequentialReplayError::Provider(
                ReplayStatus::SourceUnavailable,
                "temporal validation operator block is absent",
            ))?;
        Ok(ResolvedCompressionReplay {
            id: NodeId::new(stored.compressed_node_ids[native_index]),
            value: stored.compressed_raster[native_index],
            projection: stored.projection_accumulator[native_index],
            mean_amplitude: stored.mean_amplitude[native_index],
            status: stored.compressed_status[native_index],
        })
    }
}

impl SequentialPrimitiveSourceResolver for InMemoryProvider<'_> {
    fn identity(&self) -> &SequentialSourceProviderIdentity {
        &self.identity
    }

    fn maximum_resident_bytes(&self) -> u64 {
        SequentialSourceReplayProvider::maximum_resident_bytes(self)
    }

    fn factor_receipt_digest(
        &self,
        source: &ResolvedPrimitiveSource,
    ) -> Result<[u8; 32], SequentialReplayError> {
        self.factor_receipts
            .get(&source.id)
            .copied()
            .ok_or(SequentialReplayError::Provider(
                ReplayStatus::SourceIdentityMismatch,
                "temporal validation factor receipt is absent",
            ))
    }

    fn resolve_source(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError> {
        self.resolve(block, native_index)
    }
}

#[allow(clippy::too_many_lines)]
fn run_production_branch(
    stack: &Array3<Cf64>,
    source_manifest_digest: [u8; 32],
    source_model: &EmpiricalProperComplexConfig,
    target: (u64, u64),
    reference: (u64, u64),
    claimed_reference: &TemporalReferenceProvenance,
    conditional_oracle: Option<(usize, u64)>,
) -> Result<ProductionBranch, &'static str> {
    let mut config = sequential_config();
    config.use_evd = true;
    let (dates, rows, columns) = stack.dim();
    let grid = CovarianceOperatorGrid {
        row_start: 0,
        col_start: 0,
        rows: rows as u32,
        cols: columns as u32,
        stride_y: 1,
        stride_x: 1,
    };
    let source_model_version_digest = sequential_source_model_identity_digest(
        SOURCE_PROVIDER,
        SOURCE_PROVIDER_VERSION,
        SOURCE_MODEL,
        SOURCE_MODEL_VERSION,
    );
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "temporal-validation-burst".to_owned(),
        source_manifest_digest,
        source_model_version_digest,
        native_grid: grid,
        output_grid: grid,
        owned_output_grid: grid,
        branch_tolerance: BRANCH_TOLERANCE,
    };
    let validity = Array2::from_elem((rows, columns), true);
    let topology = SequentialReplayTopology::plan_identified(
        dates,
        (rows, columns),
        (rows, columns),
        3,
        validity.view(),
        &config,
        replay_scope(),
        ReplayIdNamespace {
            burst_id: request.burst_id.clone(),
            source_manifest_digest,
            source_model_version_digest,
            native_origin: (0, 0),
            output_origin: (0, 0),
            owned_output_origin: (0, 0),
            owned_output_shape: (rows, columns),
        },
    )
    .map_err(|_| "topology planning failed")?;
    let identity = SequentialSourceProviderIdentity {
        source_manifest_digest,
        provider: SOURCE_PROVIDER.to_owned(),
        provider_version: SOURCE_PROVIDER_VERSION.to_owned(),
        model: SOURCE_MODEL.to_owned(),
        model_version: SOURCE_MODEL_VERSION.to_owned(),
        source_model_version_digest,
        source_model_hash: *source_model.config_digest(),
    };
    let mut provider = InMemoryProvider {
        identity,
        topology: topology.clone(),
        blocks: BTreeMap::new(),
        stack: stack.view(),
        validity: validity.view(),
        source_model,
        data_identity: source_manifest_digest,
        factor_receipts: BTreeMap::new(),
    };
    let mut blocks = Vec::new();
    let output = run_sequential_with_covariance_capture_and_source_factors(
        stack.view(),
        &config,
        &ComputeEngine::new(ComputeBackend::Cpu),
        &request,
        &mut provider,
        |block| {
            blocks.push(block);
            Ok(())
        },
    )
    .map_err(|_| "production sequential capture failed")?;
    let operator_receipt = operator_receipt(&blocks, true);
    let source_factor_receipt = block_source_factor_receipt(&blocks, true);
    provider.blocks = blocks
        .into_iter()
        .map(|block| (GlobalBlockId::new(block.block_id), block))
        .collect();
    let ordered_dates = (0..dates)
        .map(|date| GlobalDateId::new(date as u32))
        .collect::<Vec<_>>();
    let source_rank = topology
        .blocks()
        .iter()
        .map(|block| block.num_real_dates * 2)
        .max()
        .ok_or("production topology has no blocks")?;
    let preflight_query = GlobalReferenceCovarianceQuery {
        burst_id: &request.burst_id,
        target,
        reference,
        ordered_dates: &ordered_dates,
        source_rank,
        source_correlation: SourceCorrelationModel::ExponentialEuclidean {
            distance_scale_pixels: SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS,
        },
        byte_cap: REPLAY_BYTE_CAP,
        branch_tolerance: BRANCH_TOLERANCE,
    };
    let estimate = {
        let mut tiles = [SequentialTileReplayProvider::new(&topology, &mut provider)];
        estimate_global_reference_difference_covariance_from_provider_bundle(
            &mut tiles,
            preflight_query,
        )
        .map_err(|_| "production replay preflight failed")?
    };
    let exact_query = GlobalReferenceCovarianceQuery {
        byte_cap: estimate.total_bytes,
        ..preflight_query
    };
    let replay = {
        let mut tiles = [SequentialTileReplayProvider::new(&topology, &mut provider)];
        replay_global_reference_difference_covariance_from_provider_bundle(&mut tiles, exact_query)
            .map_err(|_| "production replay failed")?
    };
    if replay.resource_high_water_bytes != estimate.total_bytes {
        return Err("production replay exceeded its exact preflight receipt");
    }
    let effective_looks = replay
        .replay
        .effective_looks
        .as_ref()
        .ok_or("production replay omitted effective looks")?;
    let source_correlation_model = effective_looks.model;
    let source_correlation_distance_scale_pixels = effective_looks.distance_scale_pixels;
    let source_correlation_support_union_count = effective_looks.support_union_count;
    let effective_looks_fraction = effective_looks.fraction;
    let source_correlation_receipt = digest_bytes(effective_looks.receipt);
    let target_history = phase_history(&output.cpx_phase, (target.0 as usize, target.1 as usize));
    let reference_history = phase_history(
        &output.cpx_phase,
        (reference.0 as usize, reference.1 as usize),
    );
    let incidence = Array2::eye(dates - 1);
    let mut observations = Array3::zeros((dates - 1, 1, 2));
    for date in 1..dates {
        observations[(date - 1, 0, 0)] = target_history[date];
        observations[(date - 1, 0, 1)] = reference_history[date];
    }
    let target_map = fixed_l2_pixel_map(
        incidence.view(),
        observations.view(),
        None,
        (0, 0),
        dates - 1,
    )
    .map_err(|_| "target fixed-L2 map failed")?;
    let reference_map = fixed_l2_pixel_map(
        incidence.view(),
        observations.view(),
        None,
        (0, 1),
        dates - 1,
    )
    .map_err(|_| "reference fixed-L2 map failed")?;
    let propagated = propagate_fixed_l2_difference_covariance(
        &target_map,
        &reference_map,
        replay.joint_phase_covariance.view(),
        SpatialL2Branch::FixedL2,
    )
    .map_err(|_| "fixed-L2 covariance propagation failed")?;
    let conditional_oracle_covariance = conditional_oracle
        .map(|(replicates, seed)| {
            conditional_common_factor_covariance(
                replay.joint_phase_covariance.view(),
                propagated.propagation_map.view(),
                replicates,
                seed,
            )
        })
        .transpose()?;
    let fixed_l2_map_receipt = fixed_l2_map_receipt(target, &target_map, reference, &reference_map);
    let realized_reference = realized_reference_provenance(
        columns,
        target.1 as usize,
        reference.1 as usize,
        topology
            .blocks()
            .iter()
            .map(|block| block.generation)
            .max()
            .unwrap_or(0) as usize,
    );
    if &realized_reference != claimed_reference {
        return Err("reference_context_mismatch");
    }
    let mut difference_history = target_history
        .iter()
        .zip(&reference_history)
        .map(|(target, reference)| target - reference)
        .collect::<Vec<_>>();
    unwrap_phases(&mut difference_history);
    difference_history[0] = 0.0;
    let difference_factor = matrix_rows(&propagated.date_factor);
    let difference_factor_sha256 = digest_json(&difference_factor);
    let realized_rank = propagated.date_factor.ncols();
    let difference_covariance =
        covariance_from_factor(&difference_factor).map_err(|_| "fixed-L2 factor is malformed")?;
    if difference_covariance
        .iter()
        .flatten()
        .zip(propagated.date_covariance.iter())
        .any(|(factor, propagated)| {
            (factor - propagated).abs() > 1e-12 * factor.abs().max(propagated.abs()).max(1.0)
        })
    {
        return Err("fixed-L2 factor covariance disagrees with propagation");
    }
    Ok(ProductionBranch {
        difference_history,
        difference_factor,
        difference_factor_sha256,
        realized_rank,
        covariance_condition_number: propagated.covariance_condition_number,
        difference_covariance,
        operator_receipt,
        source_factor_receipt,
        fixed_l2_map_receipt,
        replay_source_factor_receipt: digest_bytes(replay.replay.source_factor_receipt),
        replay_support_receipt: digest_bytes(replay.replay.support_receipt),
        reference_signature: digest_bytes(replay.replay.reference_signature),
        source_correlation_model,
        source_correlation_distance_scale_pixels,
        source_correlation_support_union_count,
        effective_looks_fraction,
        source_correlation_receipt,
        replay_resource_high_water_bytes: replay.resource_high_water_bytes,
        conditional_oracle_covariance,
        reference: realized_reference,
    })
}

fn conditional_common_factor_covariance(
    covariance: ndarray::ArrayView2<'_, f64>,
    propagation: ndarray::ArrayView2<'_, f64>,
    replicates: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>, &'static str> {
    if !(2..=MAX_CONDITIONAL_ORACLE_REPLICATES).contains(&replicates)
        || covariance.nrows() != covariance.ncols()
        || covariance.ncols() != propagation.ncols()
        || covariance.iter().any(|value| !value.is_finite())
        || propagation.iter().any(|value| !value.is_finite())
    {
        return Err("conditional common-factor oracle input is invalid");
    }
    let size = covariance.nrows();
    let symmetric = Mat::from_fn(size, size, |row, column| {
        0.5 * (covariance[(row, column)] + covariance[(column, row)])
    });
    let eigen = symmetric.selfadjoint_eigendecomposition(Side::Lower);
    let values = (0..size)
        .map(|index| eigen.s().column_vector()[index])
        .collect::<Vec<_>>();
    let scale = values.iter().copied().map(f64::abs).fold(0.0, f64::max);
    let tolerance = scale * 1.0e-10;
    if !scale.is_finite() || scale == 0.0 || values.iter().any(|value| *value < -tolerance) {
        return Err("conditional common-factor covariance is not positive semidefinite");
    }
    let vectors = eigen.u();
    let factor = Array2::from_shape_fn((size, size), |(row, column)| {
        vectors[(row, column)] * values[column].max(0.0).sqrt()
    });
    let output = propagation.nrows();
    let mut mean = Array1::<f64>::zeros(output);
    let mut sum_products = Array2::<f64>::zeros((output, output));
    let mut state = seed ^ 0x434f_4d4d_4f4e_4654;
    for replicate in 0..replicates {
        let normal = Array1::from_shape_fn(size, |_| {
            state = splitmix64(state);
            let uniform_one = ((state >> 11) as f64 / (1_u64 << 53) as f64).max(1.0e-15);
            state = splitmix64(state);
            let uniform_two = (state >> 11) as f64 / (1_u64 << 53) as f64;
            (-2.0 * uniform_one.ln()).sqrt() * (std::f64::consts::TAU * uniform_two).cos()
        });
        let value = propagation.dot(&factor.dot(&normal));
        let count = (replicate + 1) as f64;
        let delta = &value - &mean;
        mean.scaled_add(count.recip(), &delta);
        let updated = &value - &mean;
        for row in 0..output {
            for column in 0..output {
                sum_products[(row, column)] += delta[row] * updated[column];
            }
        }
    }
    let empirical = sum_products.mapv(|value| value / (replicates - 1) as f64);
    Ok(matrix_rows(&empirical))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn matrix_rows(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    matrix.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn covariance_from_factor(factor: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, &'static str> {
    let date_count = factor.len();
    let realized_rank = factor.first().map_or(0, Vec::len);
    if date_count < 2
        || realized_rank == 0
        || realized_rank > date_count
        || factor
            .iter()
            .any(|row| row.len() != realized_rank || row.iter().any(|value| !value.is_finite()))
    {
        return Err("fixed factor is malformed");
    }
    Ok((0..date_count)
        .map(|left| {
            (0..date_count)
                .map(|right| {
                    (0..realized_rank)
                        .map(|component| factor[left][component] * factor[right][component])
                        .sum()
                })
                .collect()
        })
        .collect())
}

#[cfg(test)]
struct FixedFactorEvaluation {
    fit: TemporalCovarianceFit,
    factor_sha256: Sha256Digest,
    difference_covariance: Vec<Vec<f64>>,
}

#[cfg(test)]
fn fit_fixed_factor_from_production_branch(
    days: &[f64],
    observations: &[f64],
    branch: &ProductionBranch,
    options: &TemporalCovarianceOptions,
) -> Result<FixedFactorEvaluation, &'static str> {
    let difference_covariance = covariance_from_factor(&branch.difference_factor)?;
    if difference_covariance != branch.difference_covariance {
        return Err("production factor covariance disagrees with propagated covariance");
    }
    Ok(FixedFactorEvaluation {
        fit: fit_temporal_covariance(days, observations, &difference_covariance, options),
        factor_sha256: digest_json(&branch.difference_factor),
        difference_covariance,
    })
}

fn complex_stack(values: &[Vec<[f64; 2]>], native_shape: [usize; 2]) -> Option<Array3<Cf64>> {
    let complex_values = values
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| Cf64::new(value[0], value[1]))
        .collect::<Vec<_>>();
    if complex_values
        .iter()
        .any(|value| !value.is_finite() || value.norm_sqr() == 0.0)
    {
        return None;
    }
    Array3::from_shape_vec(
        (values.len(), native_shape[0], native_shape[1]),
        complex_values,
    )
    .ok()
}

fn difference_history(
    carrier: &Array3<Cf64>,
    target: (usize, usize),
    reference: (usize, usize),
) -> Result<Vec<f64>, &'static str> {
    let config = sequential_config();
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let output = run_sequential(carrier.view(), &config, &engine)
        .map_err(|_| "carrier_sequential_failed")?;
    let target_history = phase_history(&output.cpx_phase, target);
    let reference_history = phase_history(&output.cpx_phase, reference);
    let mut difference = target_history
        .iter()
        .zip(reference_history)
        .map(|(target, reference)| target - reference)
        .collect::<Vec<_>>();
    unwrap_phases(&mut difference);
    difference[0] = 0.0;
    difference
        .iter()
        .all(|value| value.is_finite())
        .then_some(difference)
        .ok_or("carrier_sequential_failed")
}

#[allow(clippy::too_many_arguments)]
fn conforming_temporal_histories(
    days: &[f64],
    validity: &[bool],
    difference_factor: &[Vec<f64>],
    realized_rank: usize,
    latent_ar_path: &[f64],
    measurement_normal_path: &[f64],
    truth_slope_per_day: f64,
    process_variance: f64,
) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
    let dates = days.len();
    if dates < 2
        || validity.len() != dates
        || difference_factor.len() != dates
        || latent_ar_path.len() != dates
        || measurement_normal_path.len() != dates
        || realized_rank == 0
        || realized_rank > measurement_normal_path.len()
        || latent_ar_path.iter().any(|value| !value.is_finite())
        || measurement_normal_path
            .iter()
            .any(|value| !value.is_finite())
        || !truth_slope_per_day.is_finite()
        || !process_variance.is_finite()
        || process_variance < 0.0
        || !validity[0]
        || latent_ar_path[0] != 0.0
    {
        return Err("temporal_dgp_invalid");
    }
    let diagonal = difference_factor
        .iter()
        .map(|row| {
            if row.len() < realized_rank {
                return None;
            }
            let value = row[..realized_rank]
                .iter()
                .map(|entry| entry * entry)
                .sum::<f64>();
            value.is_finite().then_some(value)
        })
        .collect::<Option<Vec<_>>>()
        .ok_or("temporal_dgp_invalid")?;
    if difference_factor[0][..realized_rank]
        .iter()
        .any(|value| *value != 0.0)
    {
        return Err("temporal_dgp_invalid");
    }
    if diagonal[1..].iter().any(|value| *value <= 0.0) {
        return Err("temporal_dgp_invalid");
    }
    let retained = (1..dates)
        .filter(|index| validity[*index])
        .collect::<Vec<_>>();
    if retained.is_empty() {
        return Err("temporal_dgp_invalid");
    }
    let scale = (retained
        .iter()
        .map(|index| diagonal[*index].ln())
        .sum::<f64>()
        / retained.len() as f64)
        .exp();
    if !scale.is_finite() || scale <= 0.0 {
        return Err("temporal_dgp_invalid");
    }
    let process_scale = process_variance.sqrt();
    let mut carrier = Vec::with_capacity(dates);
    let mut linked = Vec::with_capacity(dates);
    for index in 0..dates {
        let signal = if index == 0 {
            0.0
        } else {
            truth_slope_per_day * days[index]
                + process_scale * (diagonal[index] / scale).sqrt() * latent_ar_path[index]
        };
        let measurement_error = if index == 0 {
            0.0
        } else {
            difference_factor[index][..realized_rank]
                .iter()
                .zip(measurement_normal_path)
                .map(|(factor, normal)| factor * normal)
                .sum::<f64>()
        };
        let observation = signal + measurement_error;
        if !signal.is_finite() || !measurement_error.is_finite() || !observation.is_finite() {
            return Err("temporal_dgp_invalid");
        }
        carrier.push(signal);
        linked.push(observation);
    }
    carrier[0] = 0.0;
    linked[0] = 0.0;
    Ok((carrier, linked))
}

fn capture_scope_digest(request: &Request, input: &ProductionPathInput) -> Sha256Digest {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:temporal-capture-scope:v2");
    update_digest_string(&mut digest, &request.cell_id);
    digest.update((request.cell_index as u64).to_le_bytes());
    digest.update((request.days.len() as u64).to_le_bytes());
    for value in &request.days {
        digest.update(value.to_bits().to_le_bytes());
    }
    for value in input.native_shape {
        digest.update((value as u64).to_le_bytes());
    }
    update_digest_string(&mut digest, &input.reference.geometry_id);
    update_digest_string(&mut digest, &input.reference.window_id);
    digest.update(input.reference.overlap_fraction.to_bits().to_le_bytes());
    digest.update(input.reference.distance_pixels.to_bits().to_le_bytes());
    digest.update((input.reference.sequential_depth as u64).to_le_bytes());
    update_digest_string(
        &mut digest,
        match input.reference.approximation {
            dolphin_timeseries::TemporalCovarianceApproximation::Exact => "exact",
            dolphin_timeseries::TemporalCovarianceApproximation::CompressedJvp => "compressed_jvp",
        },
    );
    for value in input.reference_pixel {
        digest.update((value as u64).to_le_bytes());
    }
    update_digest_string(
        &mut digest,
        match input.scope {
            TemporalValidationScope::SyntheticValidation => "synthetic_validation",
            TemporalValidationScope::FieldValidation => "field_validation",
        },
    );
    digest.update(input.source_seed.to_le_bytes());
    for value in input.target {
        digest.update((value as u64).to_le_bytes());
    }
    update_digest_string(&mut digest, &input.source_correlation_model);
    digest.update(
        input
            .source_correlation_distance_scale_pixels
            .to_bits()
            .to_le_bytes(),
    );
    update_digest_string(&mut digest, &input.outer_coverage_dgp);
    update_digest_string(&mut digest, &input.conditional_covariance_oracle);
    digest_bytes(digest.finalize().into())
}

fn update_digest_string(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value.as_bytes());
}

fn source_manifest_digest(
    request: &Request,
    input: &ProductionPathInput,
    stack: &Array3<Cf64>,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:temporal-validation-raw-source-manifest:v1");
    digest.update(input.source_seed.to_le_bytes());
    digest.update(input.source_correlation_model.as_bytes());
    digest.update(input.outer_coverage_dgp.as_bytes());
    digest.update(
        input
            .source_correlation_distance_scale_pixels
            .to_bits()
            .to_le_bytes(),
    );
    digest.update((request.days.len() as u64).to_le_bytes());
    for &day in &request.days {
        digest.update(day.to_bits().to_le_bytes());
    }
    for dimension in [stack.dim().0, stack.dim().1, stack.dim().2] {
        digest.update((dimension as u64).to_le_bytes());
    }
    for value in stack {
        digest.update(value.re.to_bits().to_le_bytes());
        digest.update(value.im.to_bits().to_le_bytes());
    }
    digest.finalize().into()
}

fn source_model_config(
) -> Result<EmpiricalProperComplexConfig, dolphin_phaselink::EmpiricalSourceModelError> {
    let model_identity = Sha256::digest(b"dolphinrust:temporal-validation-source-model:v1").into();
    EmpiricalProperComplexConfig::new(0, 1, SOURCE_MODEL_SHRINKAGE_ALPHA, 1e-8, model_identity)
}

fn operator_receipt(blocks: &[CovarianceOperatorBlock], use_evd: bool) -> Sha256Digest {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:temporal-validation-captured-operator:v1");
    digest.update([u8::from(use_evd)]);
    digest.update(sequential_replay_kernel_digest());
    digest.update((blocks.len() as u64).to_le_bytes());
    for block in blocks {
        digest.update(covariance_operator_block_sha256(block));
    }
    digest_bytes(digest.finalize().into())
}

fn block_source_factor_receipt(blocks: &[CovarianceOperatorBlock], use_evd: bool) -> Sha256Digest {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:temporal-validation-captured-source-factors:v1");
    digest.update([u8::from(use_evd)]);
    for block in blocks {
        digest.update(block.block_id.to_le_bytes());
        digest.update((block.source_factor_digests.len() as u64).to_le_bytes());
        digest.update(&block.source_factor_digests);
    }
    digest_bytes(digest.finalize().into())
}

fn fixed_l2_map_receipt(
    target_coordinates: (u64, u64),
    target: &PixelL2ObservationMap,
    reference_coordinates: (u64, u64),
    reference: &PixelL2ObservationMap,
) -> Sha256Digest {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:temporal-validation-fixed-l2-map:v1");
    digest.update(FIXED_L2_SPATIAL_COVARIANCE_METHOD.as_bytes());
    for (role, coordinates, map) in [
        (b"target".as_slice(), target_coordinates, target),
        (b"reference".as_slice(), reference_coordinates, reference),
    ] {
        digest.update((role.len() as u64).to_le_bytes());
        digest.update(role);
        digest.update(coordinates.0.to_le_bytes());
        digest.update(coordinates.1.to_le_bytes());
        digest.update((map.valid_observation_indices().len() as u64).to_le_bytes());
        for &index in map.valid_observation_indices() {
            digest.update((index as u64).to_le_bytes());
        }
        for &precision in map.precisions() {
            digest.update(precision.to_bits().to_le_bytes());
        }
        for &value in map.observation_phase_map() {
            digest.update(value.to_bits().to_le_bytes());
        }
        for &value in map.h_map() {
            digest.update(value.to_bits().to_le_bytes());
        }
        digest.update(map.condition_number().to_bits().to_le_bytes());
    }
    digest_bytes(digest.finalize().into())
}

fn realized_reference_provenance(
    columns: usize,
    target_column: usize,
    reference_column: usize,
    sequential_depth: usize,
) -> TemporalReferenceProvenance {
    let support = |column: usize| {
        let start = column.saturating_sub(1).min(columns - 3);
        (start..start + 3).collect::<std::collections::BTreeSet<_>>()
    };
    let target = support(target_column);
    let reference = support(reference_column);
    let overlap = target.intersection(&reference).count();
    let union = target.union(&reference).count();
    TemporalReferenceProvenance {
        geometry_id: "synthetic_same_frame_reference".to_owned(),
        window_id: match (target_column, reference_column) {
            (1, 2) => "near_exact".to_owned(),
            (1, 3) => "mid_exact".to_owned(),
            (1, 5) => "far_exact".to_owned(),
            _ => format!("target_col_{target_column}_reference_col_{reference_column}"),
        },
        overlap_fraction: overlap as f64 / union as f64,
        distance_pixels: target_column.abs_diff(reference_column) as f64,
        sequential_depth,
        approximation: dolphin_timeseries::TemporalCovarianceApproximation::Exact,
    }
}

fn phase_history(phases: &Array3<Cf64>, output: (usize, usize)) -> Vec<f64> {
    let mut history = phases
        .slice(s![.., output.0, output.1])
        .iter()
        .map(|value| value.arg())
        .collect::<Vec<_>>();
    unwrap_phases(&mut history);
    let gauge = history[0];
    history.iter_mut().for_each(|value| *value -= gauge);
    history[0] = 0.0;
    history
}

fn unwrap_phases(phases: &mut [f64]) {
    for index in 1..phases.len() {
        while phases[index] - phases[index - 1] > std::f64::consts::PI {
            phases[index] -= std::f64::consts::TAU;
        }
        while phases[index] - phases[index - 1] < -std::f64::consts::PI {
            phases[index] += std::f64::consts::TAU;
        }
    }
}

fn digest_bytes(value: [u8; 32]) -> Sha256Digest {
    let text = value
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Sha256Digest::new(text).expect("SHA-256 formatter is canonical")
}

fn digest_json<T: Serialize>(value: &T) -> Sha256Digest {
    let bytes = serde_json::to_vec(value).expect("validation receipt must serialize");
    let digest = Sha256::digest(bytes);
    let text = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Sha256Digest::new(text).expect("SHA-256 formatter is canonical")
}

fn semantic_digest_json<T: Serialize>(value: &T) -> Sha256Digest {
    let mut value = serde_json::to_value(value).expect("validation record must serialize");
    normalize_semantic_numbers(&mut value);
    digest_json(&value)
}

fn normalize_semantic_numbers(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Number(number) if number.is_f64() => {
            let bits = number
                .as_f64()
                .expect("JSON float must remain representable")
                .to_bits();
            *value = serde_json::Value::String(format!("f64:{bits:016x}"));
        }
        serde_json::Value::Array(values) => {
            values.iter_mut().for_each(normalize_semantic_numbers);
        }
        serde_json::Value::Object(fields) => {
            fields.values_mut().for_each(normalize_semantic_numbers);
        }
        _ => {}
    }
}

fn resident_set_bytes() -> u64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        if let Some(kibibytes) = status.lines().find_map(|line| {
            line.strip_prefix("VmHWM:")
                .and_then(|value| value.split_whitespace().next())
                .and_then(|value| value.parse::<u64>().ok())
        }) {
            return kibibytes.saturating_mul(1024);
        }
    }
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: `getrusage` initializes the supplied `rusage` on success.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0 {
            // SAFETY: the successful call above initialized `usage`.
            let maximum = unsafe { usage.assume_init() }.ru_maxrss;
            #[cfg(target_os = "macos")]
            let bytes = u64::try_from(maximum).unwrap_or(0);
            #[cfg(not(target_os = "macos"))]
            let bytes = u64::try_from(maximum).unwrap_or(0).saturating_mul(1024);
            return bytes;
        }
    }
    Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map_or(0, |kibibytes| kibibytes.saturating_mul(1024))
}

fn deterministic_stack(date_count: usize) -> Array3<Cf64> {
    Array3::from_shape_fn((date_count, 1, 7), |(date, _, column)| {
        let amplitude = 1.0 + 0.05 * ((date + 1) as f64 * (column + 2) as f64 * 0.17).sin().abs();
        let phase = 0.013 * date as f64 * (column + 1) as f64
            + 0.31 * (date as f64 * 0.29 + column as f64 * 0.47).sin();
        Cf64::from_polar(amplitude, phase)
    })
}

fn direct_production_branch(date_count: usize) -> Result<ProductionBranch, &'static str> {
    let stack = deterministic_stack(date_count);
    let source_model = source_model_config().map_err(|_| "source model configuration failed")?;
    run_production_branch(
        &stack,
        [0x42; 32],
        &source_model,
        (0, 1),
        (0, 5),
        &realized_reference_provenance(7, 1, 5, (date_count - 1) / 3),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conforming_dgp_uses_retained_actual_c54_shape_and_factor_draw() {
        let days = [0.0, 1.0, 2.0, 3.0];
        let validity = [true, true, false, true];
        let factor = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 4.0],
            vec![3.0, 4.0],
        ];
        let latent = [0.0, 2.0, -1.0, 0.5];
        let measurement_normal = [0.2, -0.3, 0.4, 0.7];
        let (carrier, linked) = conforming_temporal_histories(
            &days,
            &validity,
            &factor,
            2,
            &latent,
            &measurement_normal,
            0.1,
            0.04,
        )
        .unwrap();
        let expected = [
            0.0,
            0.1 + 0.4 / 5.0_f64.sqrt(),
            0.2 - 0.2 * (16.0_f64 / 5.0).sqrt(),
            0.3 + 0.1 * 5.0_f64.sqrt(),
        ];
        let expected_measurement_error = [0.0, 0.2, -1.2, -0.6];
        for index in 0..days.len() {
            assert!((carrier[index] - expected[index]).abs() < 1e-15);
            assert!(
                ((linked[index] - carrier[index]) - expected_measurement_error[index]).abs()
                    < 1e-15
            );
        }
        let mut invalid_factor = factor;
        invalid_factor[0][0] = f64::EPSILON;
        assert_eq!(
            conforming_temporal_histories(
                &days,
                &validity,
                &invalid_factor,
                2,
                &latent,
                &measurement_normal,
                0.1,
                0.04,
            ),
            Err("temporal_dgp_invalid")
        );
    }

    #[test]
    fn resident_set_receipt_is_nonzero() {
        assert!(resident_set_bytes() > 0);
    }
    use std::io::Cursor;

    fn deterministic_seed_request_at(date_count: usize, outer_seed_index: usize) -> Request {
        let stack = deterministic_stack(date_count);
        let values = (0..date_count)
            .map(|date| {
                (0..7)
                    .map(|column| {
                        let value = stack[(date, 0, column)];
                        [value.re, value.im]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let seed = 42_u64.checked_add(outer_seed_index as u64).unwrap();
        let seed_sha256 = Sha256::digest(seed.to_le_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let mut request = Request {
            cell_id: "frame-v7-fixture".to_owned(),
            cell_index: 0,
            outer_seed_index,
            seed_sha256,
            seed,
            days: (0..date_count).map(|date| date as f64 * 12.0).collect(),
            options: TemporalCovarianceOptions::default(),
            production_path: ProductionPathInput {
                source_seed: seed,
                native_shape: [1, 7],
                target: [0, 1],
                reference_pixel: [0, 5],
                raw_complex_stack: values.clone(),
                carrier_stack: values,
                intended_difference_variance: vec![0.0; date_count],
                latent_ar_path: vec![0.0; date_count],
                measurement_normal_path: vec![0.0; date_count],
                truth_slope_per_day: 0.01,
                source_correlation_model: SOURCE_CORRELATION_MODEL.to_owned(),
                source_correlation_distance_scale_pixels: SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS,
                outer_coverage_dgp: OUTER_COVERAGE_DGP.to_owned(),
                conditional_covariance_oracle: CONDITIONAL_COVARIANCE_ORACLE.to_owned(),
                validity: vec![true; date_count],
                reference: realized_reference_provenance(7, 1, 5, (date_count - 1) / 3),
                scope: TemporalValidationScope::SyntheticValidation,
                capture_scope_sha256: Sha256Digest::new("00".repeat(32)).unwrap(),
                validation_receipt_sha256: Sha256Digest::new("53".repeat(32)).unwrap(),
                selected_method: SELECTED_METHOD.to_owned(),
            },
            retain_dense_evidence: false,
            conditional_oracle_replicates: 0,
        };
        let capture_scope = capture_scope_digest(&request, &request.production_path);
        request.production_path.capture_scope_sha256 = capture_scope;
        request
    }

    fn deterministic_seed_request(date_count: usize) -> Request {
        deterministic_seed_request_at(date_count, 0)
    }

    #[test]
    fn bounded_line_accepts_exact_cap_and_rejects_one_byte_over() {
        let mut exact = Cursor::new(b"abcd\n".to_vec());
        let mut line = Vec::new();
        assert!(read_bounded_line(&mut exact, &mut line, 4).unwrap());
        assert_eq!(line, b"abcd");

        let mut over = Cursor::new(b"abcde\n".to_vec());
        let error = read_bounded_line(&mut over, &mut line, 4).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn production_factor_bytes_and_covariance_are_the_fixed_factor_input() {
        let date_count = 13;
        let branch = direct_production_branch(date_count).unwrap();
        assert_eq!(branch.realized_rank, date_count - 1);
        assert_eq!(
            branch.difference_factor_sha256,
            digest_json(&branch.difference_factor)
        );
        let serialized_factor = serde_json::to_vec(&branch.difference_factor).unwrap();
        let decoded_factor = serde_json::from_slice::<Vec<Vec<f64>>>(&serialized_factor).unwrap();
        assert_eq!(
            branch.difference_factor_sha256,
            digest_json(&decoded_factor)
        );
        let reconstructed = covariance_from_factor(&branch.difference_factor).unwrap();
        assert_eq!(reconstructed, branch.difference_covariance);
        let off_diagonal_energy = reconstructed
            .iter()
            .enumerate()
            .flat_map(|(row, values)| {
                values
                    .iter()
                    .enumerate()
                    .filter(move |(column, _)| *column != row)
                    .map(|(_, value)| value.powi(2))
            })
            .sum::<f64>();
        assert!(off_diagonal_energy > 0.0);

        let observations = branch.difference_history.clone();
        let options = TemporalCovarianceOptions::default();
        let fixed = fit_fixed_factor_from_production_branch(
            &(0..date_count)
                .map(|date| date as f64 * 12.0)
                .collect::<Vec<_>>(),
            &observations,
            &branch,
            &options,
        )
        .unwrap();
        let direct = fit_temporal_covariance(
            &(0..date_count)
                .map(|date| date as f64 * 12.0)
                .collect::<Vec<_>>(),
            &observations,
            &branch.difference_covariance,
            &options,
        );
        assert_eq!(fixed.factor_sha256, branch.difference_factor_sha256);
        assert_eq!(fixed.difference_covariance, branch.difference_covariance);
        assert_eq!(fixed.fit, direct);

        let mut supplied_covariance = serde_json::to_value(deterministic_seed_request(4)).unwrap();
        supplied_covariance["difference_covariance"] =
            serde_json::to_value(branch.difference_covariance).unwrap();
        assert!(serde_json::from_value::<Request>(supplied_covariance).is_err());
    }

    #[test]
    fn production_branch_observed_full_rank_49_and_97_acquisitions_is_admitted() {
        for date_count in [49_usize, 97] {
            let branch = direct_production_branch(date_count).unwrap();
            assert_eq!(branch.realized_rank, date_count - 1);
            assert_eq!(branch.difference_factor.len(), date_count);
            assert!(branch
                .difference_factor
                .iter()
                .all(|row| row.len() == branch.realized_rank));
            assert_eq!(
                covariance_from_factor(&branch.difference_factor).unwrap(),
                branch.difference_covariance
            );
        }
    }

    #[test]
    fn frame_executes_every_frozen_method_and_complete_refit_bootstrap() {
        let request = deterministic_seed_request(13);
        let mut prepared = prepare_production(&request).unwrap();
        let mut residual = 0.0;
        prepared.branch.difference_history = request
            .days
            .iter()
            .enumerate()
            .map(|(date, day)| {
                if date == 0 {
                    0.0
                } else {
                    let word = splitmix64(date as u64 ^ 0x53_b005_7a11);
                    let noise = (word >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
                    residual = 0.5 * residual + 3.0_f64.sqrt() * noise;
                    0.013 * day + residual
                }
            })
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(12)
            .build()
            .unwrap();
        let (_, _, fit, _, receipts) =
            pool.install(|| evaluate_prepared_production(&request, prepared));
        let production = fit.as_ref().unwrap();
        assert_eq!(production.bootstrap_attempts, 200);
        assert!(
            production.bootstrap_successes >= 198,
            "bootstrap successes: {}, fit: {production:?}",
            production.bootstrap_successes
        );
        assert_eq!(
            production.complete_refit_bootstrap.attempted_replicates,
            200
        );
        assert_eq!(
            production.complete_refit_bootstrap.successful_replicates,
            production.bootstrap_successes
        );
        assert_eq!(production.status, TemporalInferenceStatus::Evaluated);
        assert_eq!(production.ols.status, TemporalInferenceStatus::Evaluated);
        assert_eq!(
            production.oracle_gls.status,
            TemporalInferenceStatus::Evaluated
        );
        assert_eq!(
            production.plugin_gls.status,
            TemporalInferenceStatus::Evaluated
        );
        assert_eq!(
            production.adjusted_scalar.status,
            TemporalInferenceStatus::Evaluated
        );
        assert_eq!(
            production.conditional_wls.status,
            TemporalInferenceStatus::LegacyNonComparable
        );
        for diagnostic in [
            &production.scalar_effective_n,
            &production.adjusted_profile,
            &production.complete_refit_bootstrap,
        ] {
            assert_eq!(diagnostic.status, TemporalInferenceStatus::Evaluated);
        }
        assert!(production.adjusted_profile_slope.is_some());
        let receipts = receipts.as_ref().unwrap();
        assert_eq!(receipts.temporal_profile_fit_count, 1);
        assert_eq!(receipts.temporal_bootstrap_attempts, 200);
    }

    #[test]
    fn production_missingness_compacts_factor_and_keeps_factor_prefit_evaluated() {
        let mut request = deterministic_seed_request(14);
        request.production_path.validity[4] = false;
        request.production_path.capture_scope_sha256 =
            capture_scope_digest(&request, &request.production_path);
        let mut prepared = prepare_production(&request).unwrap();
        let mut residual = 0.0;
        prepared.branch.difference_history = request
            .days
            .iter()
            .enumerate()
            .map(|(date, day)| {
                if date == 0 {
                    0.0
                } else {
                    let word = splitmix64(date as u64 ^ 0x53_b005_7a11);
                    let noise = (word >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
                    residual = 0.5 * residual + 3.0_f64.sqrt() * noise;
                    0.013 * day + residual
                }
            })
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(12)
            .build()
            .unwrap();
        let (_, _, fit, _, receipts) =
            pool.install(|| evaluate_prepared_production(&request, prepared));
        let fit = fit.unwrap();
        assert_eq!(fit.valid_date_count, 12);
        assert_eq!(fit.status, TemporalInferenceStatus::Evaluated);
        assert_eq!(fit.plugin_gls.status, TemporalInferenceStatus::Evaluated);
        assert_eq!(
            fit.adjusted_scalar.status,
            TemporalInferenceStatus::Evaluated
        );
        assert_eq!(
            fit.adjusted_profile.status,
            TemporalInferenceStatus::Evaluated
        );
        let receipts = receipts.unwrap();
        assert_eq!(receipts.temporal_profile_fit_count, 1);
        assert_eq!(receipts.temporal_bootstrap_attempts, 200);
    }

    #[test]
    fn largest_frozen_temporal_shape_runs_all_methods() {
        let request = deterministic_seed_request(97);
        let mut prepared = prepare_production(&request).unwrap();
        let mut residual = 0.0;
        prepared.branch.difference_history = request
            .days
            .iter()
            .enumerate()
            .map(|(date, day)| {
                if date == 0 {
                    0.0
                } else {
                    let word = splitmix64(date as u64 ^ 0x53_b005_7a11);
                    let noise = (word >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
                    residual = 0.5 * residual + 3.0_f64.sqrt() * noise;
                    0.013 * day + residual
                }
            })
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(12)
            .build()
            .unwrap();
        let (_, _, fit, _, receipts) =
            pool.install(|| evaluate_prepared_production(&request, prepared));
        let fit = fit.unwrap();
        assert_eq!(fit.valid_date_count, 96, "fit: {fit:?}");
        assert_eq!(fit.bootstrap_attempts, 200);
        assert!(fit.bootstrap_successes >= 198);
        assert_eq!(
            fit.adjusted_profile.status,
            TemporalInferenceStatus::Evaluated
        );
        assert_eq!(
            fit.complete_refit_bootstrap.status,
            TemporalInferenceStatus::Evaluated
        );
        assert_eq!(receipts.unwrap().temporal_bootstrap_attempts, 200);
    }

    #[test]
    fn frame_v7_caps_orders_and_preserves_one_vs_32_record_bytes() {
        let requests = (0..32)
            .map(|outer_seed_index| deterministic_seed_request_at(4, outer_seed_index))
            .collect::<Vec<_>>();
        let frame_bytes = serde_json::to_vec(&serde_json::json!({
            "schema": TEMPORAL_BATCH_SCHEMA,
            "requests": requests,
        }))
        .unwrap();
        assert!(frame_bytes.len() <= MAX_REQUEST_LINE_BYTES);
        let decoded = decode_frame_request(&frame_bytes).unwrap();

        let one_thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let twelve_thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(12)
            .build()
            .unwrap();
        let one_thread = one_thread_pool
            .install(|| evaluate_frame_records(decoded.requests.clone()))
            .unwrap();
        let twelve_thread = twelve_thread_pool
            .install(|| evaluate_frame_records(decoded.requests.clone()))
            .unwrap();
        assert_eq!(one_thread.len(), 64);
        assert_eq!(twelve_thread.len(), 64);
        assert_eq!(
            one_thread
                .iter()
                .map(canonical_record_bytes)
                .collect::<Vec<_>>(),
            twelve_thread
                .iter()
                .map(canonical_record_bytes)
                .collect::<Vec<_>>()
        );

        for (seed_index, (request, pair)) in decoded
            .requests
            .iter()
            .zip(twelve_thread.chunks_exact(2))
            .enumerate()
        {
            let one = evaluate_frame_records(vec![request.clone()]).unwrap();
            assert_eq!(pair[0].execution_path, ExecutionPath::FixedFactor);
            assert_eq!(pair[1].execution_path, ExecutionPath::ProductionPath);
            assert_eq!(pair[0].outer_seed_index, seed_index);
            assert_eq!(pair[0].fit, pair[1].fit);
            assert_eq!(pair[0].factor_sha256, pair[1].factor_sha256);
            assert_eq!(pair[0].realized_factor_rank, pair[1].realized_factor_rank);
            assert_ne!(
                canonical_record_bytes(&pair[0]),
                canonical_record_bytes(&pair[1])
            );
            assert_eq!(
                canonical_record_bytes(&pair[0]),
                canonical_record_bytes(&one[0])
            );
            assert_eq!(
                canonical_record_bytes(&pair[1]),
                canonical_record_bytes(&one[1])
            );
            for record in pair {
                let object = serde_json::to_value(record)
                    .unwrap()
                    .as_object()
                    .unwrap()
                    .clone();
                assert!(!object.contains_key("resource"));
                assert!(!object.contains_key("wall_micros"));
                assert!(!object.contains_key("resident_set_bytes_before"));
                assert!(!object.contains_key("resident_set_bytes_after"));
            }
        }

        let profile_fit_count = twelve_thread
            .iter()
            .filter_map(|record| record.production_receipts.as_ref())
            .map(|receipt| receipt.temporal_profile_fit_count)
            .sum();
        let bootstrap_attempts = twelve_thread
            .iter()
            .filter_map(|record| record.production_receipts.as_ref())
            .map(|receipt| receipt.temporal_bootstrap_attempts)
            .sum();
        let response = FrameResponse {
            schema: TEMPORAL_BATCH_SCHEMA,
            records: twelve_thread,
            resource: FrameResourceReceipt {
                schema: TEMPORAL_BATCH_RESOURCE_SCHEMA,
                request_count: 32,
                record_count: 64,
                factor_generation_count: 32,
                temporal_fit_count: 32,
                profile_fit_count,
                bootstrap_attempts,
                attempt_record_count: 64,
                rayon_worker_count: 12,
                wall_micros: 0,
                resident_set_bytes_before: 0,
                resident_set_bytes_after: 0,
            },
        };
        let response_bytes = serde_json::to_vec(&response).unwrap();
        assert!(response_bytes.len() <= MAX_RESPONSE_LINE_BYTES);
        assert_eq!(response.resource.factor_generation_count, 32);
        assert_eq!(response.resource.temporal_fit_count, 32);
        assert_eq!(response.resource.profile_fit_count, profile_fit_count);
        assert_eq!(response.resource.bootstrap_attempts, bootstrap_attempts);
        assert_eq!(response.resource.attempt_record_count, 64);

        assert!(decode_frame_request(&vec![b' '; MAX_REQUEST_LINE_BYTES + 1]).is_err());
        let zero = serde_json::to_vec(&serde_json::json!({
            "schema": TEMPORAL_BATCH_SCHEMA,
            "requests": [],
        }))
        .unwrap();
        assert!(decode_frame_request(&zero).is_err());
        let too_many = serde_json::to_vec(&serde_json::json!({
            "schema": TEMPORAL_BATCH_SCHEMA,
            "requests": (0..33)
                .map(|outer_seed_index| deterministic_seed_request_at(4, outer_seed_index))
                .collect::<Vec<_>>(),
        }))
        .unwrap();
        assert!(decode_frame_request(&too_many).is_err());
    }

    #[test]
    fn frame_v7_rejects_cross_cell_and_nonconsecutive_seed_indices() {
        let first = deterministic_seed_request(4);
        let mut second = first.clone();
        second.outer_seed_index = 1;
        second.seed = second.seed.checked_add(1).unwrap();
        second.production_path.source_seed = second.seed;

        let mut cross_cell = second.clone();
        cross_cell.cell_id.push_str("-other");
        let cross_cell = serde_json::to_vec(&serde_json::json!({
            "schema": TEMPORAL_BATCH_SCHEMA,
            "requests": [first.clone(), cross_cell],
        }))
        .unwrap();
        assert!(decode_frame_request(&cross_cell).is_err());

        let mut nonconsecutive = second;
        nonconsecutive.outer_seed_index = 2;
        let nonconsecutive = serde_json::to_vec(&serde_json::json!({
            "schema": TEMPORAL_BATCH_SCHEMA,
            "requests": [first, nonconsecutive],
        }))
        .unwrap();
        assert!(decode_frame_request(&nonconsecutive).is_err());
    }

    #[test]
    fn compact_default_receipts_bound_max_n_frame_and_retained_shard() {
        let pair = evaluate_seed_request(deterministic_seed_request(13));
        let fixed = serde_json::to_value(&pair[0]).unwrap();
        let mut production = serde_json::to_value(&pair[1]).unwrap();
        let production_receipts = production["production_receipts"].as_object_mut().unwrap();
        assert!(!production_receipts.contains_key("fixed_l2_difference_factor"));
        production["realized_factor_rank"] = serde_json::json!(96);
        production["production_receipts"]["fixed_l2_realized_rank"] = serde_json::json!(96);

        let records = (0..32)
            .flat_map(|_| [fixed.clone(), production.clone()])
            .collect::<Vec<_>>();
        let frame = serde_json::json!({
            "schema": TEMPORAL_BATCH_SCHEMA,
            "records": records,
            "resource": {
                "schema": TEMPORAL_BATCH_RESOURCE_SCHEMA,
                "request_count": 32,
                "record_count": 64,
                "factor_generation_count": 32,
                "temporal_fit_count": 32,
                "profile_fit_count": 32,
                "bootstrap_attempts": 6_400,
                "attempt_record_count": 64,
                "rayon_worker_count": 12,
                "wall_micros": 1,
                "resident_set_bytes_before": 1,
                "resident_set_bytes_after": 1,
            },
        });
        assert!(serde_json::to_vec(&frame).unwrap().len() <= MAX_RESPONSE_LINE_BYTES);

        let retained_pair_bytes = serde_json::to_vec(&fixed).unwrap().len()
            + serde_json::to_vec(&production).unwrap().len()
            + 2;
        assert!(retained_pair_bytes * 1_050 <= 16 * 1024 * 1024);
    }
}

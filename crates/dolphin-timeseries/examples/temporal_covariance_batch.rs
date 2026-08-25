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
use dolphin_timeseries::{
    fit_temporal_covariance, temporal_covariance_provenance, Sha256Digest, TemporalCovarianceFit,
    TemporalCovarianceOptions, TemporalCovarianceProvenance, TemporalCovarianceProvenanceInputs,
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
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{self, BufRead, Write};
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ExecutionPath {
    FixedFactor,
    ProductionPath,
}

#[derive(Debug, Deserialize)]
struct Request {
    execution_path: ExecutionPath,
    cell_id: String,
    cell_index: usize,
    outer_seed_index: usize,
    seed_sha256: String,
    seed: u64,
    days: Vec<f64>,
    options: TemporalCovarianceOptions,
    fixed_factor: Option<FixedFactorInput>,
    production_path: Option<ProductionPathInput>,
    #[serde(default)]
    retain_dense_evidence: bool,
    #[serde(default)]
    conditional_oracle_replicates: usize,
}

#[derive(Debug, Deserialize)]
struct FixedFactorInput {
    observations: Vec<Option<f64>>,
    difference_covariance: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ProductionPathInput {
    source_seed: u64,
    native_shape: [usize; 2],
    target: [usize; 2],
    reference_pixel: [usize; 2],
    raw_complex_stack: Vec<Vec<[f64; 2]>>,
    carrier_stack: Vec<Vec<[f64; 2]>>,
    intended_difference_variance: Vec<f64>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    fixed_l2_difference_covariance: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fixed_l2_difference_variance: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    carrier_difference_history: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    linked_difference_history: Option<Vec<f64>>,
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
    fixed_factor_status: Option<TemporalInferenceStatus>,
    production_path_status: Option<&'static str>,
    comparator_methods: [&'static str; 8],
    attempted: bool,
    emitted: bool,
    failed: bool,
    fit: Option<TemporalCovarianceFit>,
    provenance: Option<TemporalCovarianceProvenance>,
    production_receipts: Option<ProductionReceipts>,
    resource: ResourceReceipt,
}

#[derive(Debug, Serialize)]
struct ResourceReceipt {
    wall_micros: u128,
    resident_set_bytes_before: u64,
    resident_set_bytes_after: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    let mut line = Vec::new();
    while read_bounded_line(&mut input, &mut line, MAX_REQUEST_LINE_BYTES)? {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let request: Request = serde_json::from_slice(&line)?;
        let rss_before = resident_set_bytes();
        let started = Instant::now();
        let (fixed_factor_status, production_path_status, fit, provenance, production_receipts) =
            evaluate(&request);
        let rss_after = resident_set_bytes();
        let emitted = fit
            .as_ref()
            .is_some_and(|value| value.status == TemporalInferenceStatus::Evaluated);
        let response = Response {
            schema: "dolphinrust-temporal-covariance-batch/6",
            execution_path: request.execution_path,
            cell_id: request.cell_id,
            cell_index: request.cell_index,
            outer_seed_index: request.outer_seed_index,
            seed_sha256: request.seed_sha256,
            seed: request.seed,
            fixed_factor_status,
            production_path_status,
            comparator_methods: [
                "ols",
                "oracle_gls",
                "legacy_intercept_slope_wls_non_comparable",
                "lag_one_scalar_effective_n",
                "plugin_gls_reml",
                "reml_covariance_parameter_adjusted_scalar",
                "slope_profile_likelihood_ml",
                "complete_refit_bootstrap",
            ],
            attempted: true,
            emitted,
            failed: !emitted,
            fit,
            provenance,
            production_receipts,
            resource: ResourceReceipt {
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

type Evaluation = (
    Option<TemporalInferenceStatus>,
    Option<&'static str>,
    Option<TemporalCovarianceFit>,
    Option<TemporalCovarianceProvenance>,
    Option<ProductionReceipts>,
);

fn evaluate(request: &Request) -> Evaluation {
    match request.execution_path {
        ExecutionPath::FixedFactor => {
            let Some(input) = &request.fixed_factor else {
                return (None, None, None, None, None);
            };
            let observations = input
                .observations
                .iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            let fit = fit_temporal_covariance(
                &request.days,
                &observations,
                &input.difference_covariance,
                &request.options,
            );
            (Some(fit.status), None, Some(fit), None, None)
        }
        ExecutionPath::ProductionPath => evaluate_production(request),
    }
}

#[allow(clippy::too_many_lines)]
fn evaluate_production(request: &Request) -> Evaluation {
    let Some(input) = &request.production_path else {
        return (None, Some("production_inputs_missing"), None, None, None);
    };
    if input.source_seed != request.seed {
        return (None, Some("source_seed_mismatch"), None, None, None);
    }
    if input.scope != TemporalValidationScope::SyntheticValidation
        || input.selected_method != "complete_refit_bootstrap"
        || input.source_correlation_model != SOURCE_CORRELATION_MODEL
        || input.source_correlation_distance_scale_pixels
            != SOURCE_CORRELATION_DISTANCE_SCALE_PIXELS
        || input.outer_coverage_dgp != OUTER_COVERAGE_DGP
        || input.conditional_covariance_oracle != CONDITIONAL_COVARIANCE_ORACLE
        || request.conditional_oracle_replicates == 1
        || request.conditional_oracle_replicates > MAX_CONDITIONAL_ORACLE_REPLICATES
    {
        return (None, Some("production_contract_mismatch"), None, None, None);
    }
    let capture_scope = capture_scope_digest(request, input);
    if capture_scope != input.capture_scope_sha256 {
        return (None, Some("capture_scope_mismatch"), None, None, None);
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
        || input.validity.len() != dates
        || !input.validity.first().copied().unwrap_or(false)
    {
        return (None, Some("raw_complex_invalid"), None, None, None);
    }
    let Some(stack) = complex_stack(&input.raw_complex_stack, input.native_shape) else {
        return (None, Some("raw_complex_invalid"), None, None, None);
    };
    let Some(carrier) = complex_stack(&input.carrier_stack, input.native_shape) else {
        return (None, Some("raw_complex_invalid"), None, None, None);
    };
    let source_model = match source_model_config() {
        Ok(value) => value,
        Err(_) => return (None, Some("source_model_invalid"), None, None, None),
    };
    let carrier_difference_history = match difference_history(
        &carrier,
        (input.target[0], input.target[1]),
        (input.reference_pixel[0], input.reference_pixel[1]),
    ) {
        Ok(value) => value,
        Err(status) => return (None, Some(status), None, None, None),
    };
    let source_manifest = source_manifest_digest(request, input, &stack);
    let evd = match run_production_branch(
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
    ) {
        Ok(value) => value,
        Err(status) => return (None, Some(status), None, None, None),
    };
    if evd.source_correlation_model != input.source_correlation_model
        || evd.source_correlation_distance_scale_pixels
            != input.source_correlation_distance_scale_pixels
    {
        return (None, Some("source_correlation_mismatch"), None, None, None);
    }
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
        source_model.config_digest(),
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
    let fit = fit_temporal_covariance(
        &request.days,
        &observations,
        &evd.difference_covariance,
        &request.options,
    );
    let fixed_l2_difference_covariance = evd.difference_covariance.clone();
    let fixed_l2_difference_variance = fixed_l2_difference_covariance
        .iter()
        .enumerate()
        .map(|(index, row)| row[index])
        .collect();
    let numeric_evidence_sha256 = digest_json(&(
        &fixed_l2_difference_covariance,
        &fixed_l2_difference_variance,
        &carrier_difference_history,
        &linked_difference_history,
        &evd.conditional_oracle_covariance,
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
                source_manifest,
                evd.operator_receipt.as_str(),
                evd.replay_source_factor_receipt.as_str(),
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
                    source_model.config_digest(),
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
        source_model_sha256: digest_bytes(*source_model.config_digest()),
        evd_operator_sha256: evd.operator_receipt,
        evd_source_factor_sha256: evd.source_factor_receipt,
        fixed_l2_map_sha256: evd.fixed_l2_map_receipt,
        issue52_receipt_sha256: issue52_receipt,
        issue54_receipt_sha256: issue54_receipt,
        numeric_evidence_sha256,
        fixed_l2_difference_covariance: retain_dense.then_some(fixed_l2_difference_covariance),
        fixed_l2_difference_variance: retain_dense.then_some(fixed_l2_difference_variance),
        carrier_difference_history: retain_dense.then_some(carrier_difference_history),
        linked_difference_history: retain_dense.then_some(linked_difference_history),
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
const OUTER_COVERAGE_DGP: &str = "physical_raw_space_v1";
const CONDITIONAL_COVARIANCE_ORACLE: &str = "fixed_capture_common_factor_monte_carlo_v1";
const MAX_CONDITIONAL_ORACLE_REPLICATES: usize = 16_384;
const MAX_REQUEST_LINE_BYTES: usize = 4 * 1024 * 1024;
const MAX_RESPONSE_LINE_BYTES: usize = 4 * 1024 * 1024;
const PHASELINK_SOURCE_JVP_METHOD: &str = "raw_complex_to_phase_source_jvp_v1";
const PHASELINK_SOURCE_JVP_CONTRACT: &str =
    "dolphin_phaselink::source_influence_contract::evd_phase_source_jvp_matches_raw_complex_difference";
const PHASELINK_SPATIAL_JVP_CONTRACT: &str =
    "dolphin_phaselink::spatial_reference_covariance_contract::evd_and_emi_source_jvps_match_finite_difference";

struct ProductionBranch {
    difference_history: Vec<f64>,
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
    Ok(ProductionBranch {
        difference_history,
        difference_covariance: matrix_rows(&propagated.date_covariance),
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

fn capture_scope_digest(request: &Request, input: &ProductionPathInput) -> Sha256Digest {
    digest_json(&serde_json::json!({
        "cell_id": request.cell_id,
        "cell_index": request.cell_index,
        "days": request.days,
        "native_shape": input.native_shape,
        "reference": input.reference,
        "reference_pixel": input.reference_pixel,
        "scope": input.scope,
        "source_seed": input.source_seed,
        "target": input.target,
        "source_correlation_model": input.source_correlation_model,
        "source_correlation_distance_scale_pixels": input.source_correlation_distance_scale_pixels,
        "outer_coverage_dgp": input.outer_coverage_dgp,
        "conditional_covariance_oracle": input.conditional_covariance_oracle,
    }))
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
    Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map_or(0, |kibibytes| kibibytes.saturating_mul(1024))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

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
}

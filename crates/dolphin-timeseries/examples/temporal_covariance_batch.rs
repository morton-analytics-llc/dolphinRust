//! Release JSONL runner for fixed-factor and same-seed #52/#54 validation paths.

use dolphin_core::config::{CompressedSlcPlan, ShpMethod};
use dolphin_phaselink::{InfluenceDag, InfluenceNode, SourceDefinition, SourceEdge, SourceId};
use dolphin_timeseries::{
    fit_temporal_covariance, temporal_covariance_provenance, Sha256Digest, TemporalCovarianceFit,
    TemporalCovarianceOptions, TemporalCovarianceProvenance, TemporalCovarianceProvenanceInputs,
    TemporalInferenceStatus, TemporalReferenceProvenance, TemporalValidationScope,
};
use dolphin_workflows::{
    DependencyConeQuery, GlobalDateId, ReplayBackend, ReplayExecutionScope, SequentialConfig,
    SequentialReplayTopology,
};
use ndarray::Array2;
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
    seed: u64,
    days: Vec<f64>,
    options: TemporalCovarianceOptions,
    fixed_factor: Option<FixedFactorInput>,
    production_path: Option<ProductionPathInput>,
}

#[derive(Debug, Deserialize)]
struct FixedFactorInput {
    observations: Vec<Option<f64>>,
    difference_covariance: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ProductionPathInput {
    raw_complex_seed: u64,
    issue52_seed: u64,
    issue54_seed: u64,
    target_raw_complex: Vec<[f64; 2]>,
    reference_raw_complex: Vec<[f64; 2]>,
    complex_noise_standard_deviation: Vec<f64>,
    validity: Vec<bool>,
    reference: TemporalReferenceProvenance,
    scope: TemporalValidationScope,
    validation_receipt_sha256: Sha256Digest,
    selected_method: String,
}

#[derive(Debug, Serialize)]
struct Response {
    schema: &'static str,
    execution_path: ExecutionPath,
    cell_id: String,
    cell_index: usize,
    seed: u64,
    fixed_factor_status: Option<TemporalInferenceStatus>,
    production_path_status: Option<&'static str>,
    comparator_methods: [&'static str; 8],
    attempted: bool,
    emitted: bool,
    failed: bool,
    fit: Option<TemporalCovarianceFit>,
    provenance: Option<TemporalCovarianceProvenance>,
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
    let mut output = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Request = serde_json::from_str(&line)?;
        let rss_before = resident_set_bytes();
        let started = Instant::now();
        let (fixed_factor_status, production_path_status, fit, provenance) = evaluate(&request);
        let rss_after = resident_set_bytes();
        let emitted = fit
            .as_ref()
            .is_some_and(|value| value.status == TemporalInferenceStatus::Evaluated);
        let response = Response {
            schema: "dolphinrust-temporal-covariance-batch/3",
            execution_path: request.execution_path,
            cell_id: request.cell_id,
            cell_index: request.cell_index,
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
            resource: ResourceReceipt {
                wall_micros: started.elapsed().as_micros(),
                resident_set_bytes_before: rss_before,
                resident_set_bytes_after: rss_after,
            },
        };
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
    }
    Ok(())
}

type Evaluation = (
    Option<TemporalInferenceStatus>,
    Option<&'static str>,
    Option<TemporalCovarianceFit>,
    Option<TemporalCovarianceProvenance>,
);
type SourceCovariancePair = (Vec<Vec<f64>>, Vec<Vec<f64>>);

fn evaluate(request: &Request) -> Evaluation {
    match request.execution_path {
        ExecutionPath::FixedFactor => {
            let Some(input) = &request.fixed_factor else {
                return (None, None, None, None);
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
            (Some(fit.status), None, Some(fit), None)
        }
        ExecutionPath::ProductionPath => evaluate_production(request),
    }
}

fn evaluate_production(request: &Request) -> Evaluation {
    let Some(input) = &request.production_path else {
        return (None, Some("production_inputs_missing"), None, None);
    };
    if [
        input.raw_complex_seed,
        input.issue52_seed,
        input.issue54_seed,
    ]
    .iter()
    .any(|seed| *seed != request.seed)
    {
        return (None, Some("source_seed_mismatch"), None, None);
    }
    let n = request.days.len();
    if input.target_raw_complex.len() != n
        || input.reference_raw_complex.len() != n
        || input.complex_noise_standard_deviation.len() != n
        || input.validity.len() != n
        || !input.validity.first().copied().unwrap_or(false)
    {
        return (None, Some("raw_complex_invalid"), None, None);
    }
    let Some(mut observations) =
        raw_complex_difference(&input.target_raw_complex, &input.reference_raw_complex, n)
    else {
        return (None, Some("raw_complex_invalid"), None, None);
    };
    if observations[0] != 0.0 {
        return (None, Some("raw_complex_gauge_not_zero"), None, None);
    }
    observations
        .iter_mut()
        .zip(&input.validity)
        .for_each(|(value, valid)| {
            if !valid {
                *value = f64::NAN;
            }
        });
    let Ok((issue52_covariance, difference_covariance)) = replay_source_covariance(input, n) else {
        return (None, Some("source_replay_failed"), None, None);
    };
    let fit = fit_temporal_covariance(
        &request.days,
        &observations,
        &difference_covariance,
        &request.options,
    );
    let provenance = temporal_covariance_provenance(
        &fit,
        TemporalCovarianceProvenanceInputs {
            issue52_receipt_sha256: digest_json(&(input.issue52_seed, &issue52_covariance)),
            issue54_receipt_sha256: digest_json(&(input.issue54_seed, &difference_covariance)),
            reference: input.reference.clone(),
            scope: input.scope,
            validation_receipt_sha256: input.validation_receipt_sha256.clone(),
            estimator_input_sha256: digest_json(input),
            selected_method: input.selected_method.clone(),
        },
    );
    let status = if fit.status == TemporalInferenceStatus::Evaluated {
        "evaluated"
    } else {
        "estimator_failed"
    };
    (None, Some(status), Some(fit), provenance)
}

fn sequential_config() -> SequentialConfig {
    SequentialConfig {
        ministack_size: 3,
        max_num_compressed: 1,
        half_window: dolphin_core::HalfWindow { y: 0, x: 0 },
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

fn replay_source_covariance(
    input: &ProductionPathInput,
    dates: usize,
) -> Result<SourceCovariancePair, ()> {
    let topology = SequentialReplayTopology::plan(
        dates,
        (1, 2),
        (1, 2),
        1,
        &sequential_config(),
        replay_scope(),
    )
    .map_err(|_| ())?;
    let target = (0..dates)
        .map(|date| (GlobalDateId::new(date as u32), 0))
        .collect::<Vec<_>>();
    let reference = (0..dates)
        .map(|date| (GlobalDateId::new(date as u32), 1))
        .collect::<Vec<_>>();
    let query = DependencyConeQuery {
        source_rank: 4,
        microbatch: 1,
        byte_cap: 1 << 30,
    };
    let issue52 = topology
        .replay_temporal_covariance(&target, query, |_| build_raw_dag(&topology, input))
        .map_err(|_| ())?;
    let issue54 = topology
        .replay_reference_difference_covariance(&target, &reference, query, |_| {
            build_raw_dag(&topology, input)
        })
        .map_err(|_| ())?;
    if issue52.covariance != issue54.target_covariance {
        return Err(());
    }
    Ok((
        matrix_rows(&issue52.covariance),
        matrix_rows(&issue54.difference_covariance),
    ))
}

fn build_raw_dag(
    topology: &SequentialReplayTopology,
    input: &ProductionPathInput,
) -> Result<InfluenceDag, dolphin_phaselink::InfluenceError> {
    let mut dag = InfluenceDag::new();
    let overlap = input.reference.overlap_fraction;
    if !overlap.is_finite() || !(0.0..1.0).contains(&overlap) {
        return Err(dolphin_phaselink::InfluenceError::NonFiniteOperator);
    }
    let independent = (1.0 - overlap * overlap).sqrt();
    for date in 1..input.target_raw_complex.len() {
        let source = SourceId::new(10_000 + date as u64);
        let mut digest = [0_u8; 32];
        digest[..8].copy_from_slice(&input.raw_complex_seed.to_le_bytes());
        digest[8..16].copy_from_slice(&(date as u64).to_le_bytes());
        dag.add_source(SourceDefinition::new(source, 4, digest))?;
        let sigma = input.complex_noise_standard_deviation[date];
        let target_gradient = phase_gradient(input.target_raw_complex[date], sigma)
            .ok_or(dolphin_phaselink::InfluenceError::NonFiniteOperator)?;
        let reference_gradient = phase_gradient(input.reference_raw_complex[date], sigma)
            .ok_or(dolphin_phaselink::InfluenceError::NonFiniteOperator)?;
        let target_edge = Array2::from_shape_vec(
            (1, 4),
            vec![target_gradient[0], target_gradient[1], 0.0, 0.0],
        )
        .expect("fixed edge shape");
        let reference_edge = Array2::from_shape_vec(
            (1, 4),
            vec![
                overlap * reference_gradient[0],
                overlap * reference_gradient[1],
                independent * reference_gradient[0],
                independent * reference_gradient[1],
            ],
        )
        .expect("fixed edge shape");
        let target_node = topology
            .date_node_id(GlobalDateId::new(date as u32), 0)
            .map_err(|_| dolphin_phaselink::InfluenceError::NonFiniteOperator)?;
        let reference_node = topology
            .date_node_id(GlobalDateId::new(date as u32), 1)
            .map_err(|_| dolphin_phaselink::InfluenceError::NonFiniteOperator)?;
        dag.add_node(
            InfluenceNode::new(target_node, 1).with_source(SourceEdge::new(source, target_edge)),
        )?;
        dag.add_node(
            InfluenceNode::new(reference_node, 1)
                .with_source(SourceEdge::new(source, reference_edge)),
        )?;
    }
    Ok(dag)
}

fn phase_gradient(value: [f64; 2], sigma: f64) -> Option<[f64; 2]> {
    let radius_squared = value[0] * value[0] + value[1] * value[1];
    (value.iter().all(|item| item.is_finite())
        && sigma.is_finite()
        && sigma > 0.0
        && radius_squared.is_finite()
        && radius_squared > 0.0)
        .then_some([
            -sigma * value[1] / radius_squared,
            sigma * value[0] / radius_squared,
        ])
}

fn matrix_rows(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    matrix.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn raw_complex_difference(
    target: &[[f64; 2]],
    reference: &[[f64; 2]],
    dimension: usize,
) -> Option<Vec<f64>> {
    if target.len() != dimension || reference.len() != dimension || dimension < 2 {
        return None;
    }
    let mut phases = Vec::with_capacity(dimension);
    for (target, reference) in target.iter().zip(reference) {
        if target
            .iter()
            .chain(reference)
            .any(|value| !value.is_finite())
            || target[0].hypot(target[1]) == 0.0
            || reference[0].hypot(reference[1]) == 0.0
        {
            return None;
        }
        phases.push(target[1].atan2(target[0]) - reference[1].atan2(reference[0]));
    }
    for index in 1..phases.len() {
        while phases[index] - phases[index - 1] > std::f64::consts::PI {
            phases[index] -= std::f64::consts::TAU;
        }
        while phases[index] - phases[index - 1] < -std::f64::consts::PI {
            phases[index] += std::f64::consts::TAU;
        }
    }
    Some(phases)
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

//! Release-mode JSONL runner for fixed-factor and same-seed #52/#54 validation paths.

use dolphin_timeseries::{
    fit_temporal_covariance, temporal_covariance_provenance, TemporalCovarianceFit,
    TemporalCovarianceOptions, TemporalCovarianceProvenance, TemporalCovarianceProvenanceInputs,
    TemporalInferenceStatus,
};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
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

#[derive(Debug, Deserialize)]
struct ProductionPathInput {
    raw_complex_seed: u64,
    issue52_seed: u64,
    issue54_seed: u64,
    target_raw_complex: Vec<[f64; 2]>,
    reference_raw_complex: Vec<[f64; 2]>,
    validity: Vec<bool>,
    issue52_target_factor: Vec<Vec<f64>>,
    issue52_reference_factor: Vec<Vec<f64>>,
    issue54_difference_factor: Vec<Vec<f64>>,
    provenance: TemporalCovarianceProvenanceInputs,
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
    comparator_methods: [&'static str; 7],
    attempted: bool,
    emitted: bool,
    failed: bool,
    fit: Option<TemporalCovarianceFit>,
    provenance: Option<TemporalCovarianceProvenance>,
    resource: ResourceReceipt,
}

#[derive(Debug, Serialize)]
struct ResourceReceipt {
    wall_millis: u128,
    peak_rss_bytes: Option<u64>,
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
        let started = Instant::now();
        let (fixed_factor_status, production_path_status, fit, provenance) = evaluate(&request);
        let emitted = fit
            .as_ref()
            .is_some_and(|value| value.status == TemporalInferenceStatus::Evaluated);
        let response = Response {
            schema: "dolphinrust-temporal-covariance-batch/2",
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
                "plugin_gls_ml",
                "slope_profile_likelihood",
                "complete_refit_bootstrap",
            ],
            attempted: true,
            emitted,
            failed: !emitted,
            fit,
            provenance,
            resource: ResourceReceipt {
                wall_millis: started.elapsed().as_millis(),
                peak_rss_bytes: None,
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

fn evaluate(request: &Request) -> Evaluation {
    match request.execution_path {
        ExecutionPath::FixedFactor => {
            let Some(input) = &request.fixed_factor else {
                return (None, None, None, None);
            };
            let observations = optional_observations(&input.observations);
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
    if !valid_factor(&input.issue52_target_factor, n)
        || !valid_factor(&input.issue52_reference_factor, n)
        || !valid_factor(&input.issue54_difference_factor, n)
    {
        return (None, Some("source_factor_invalid"), None, None);
    }
    let Some(mut observations) =
        raw_complex_difference(&input.target_raw_complex, &input.reference_raw_complex, n)
    else {
        return (None, Some("raw_complex_invalid"), None, None);
    };
    if input.validity.len() != n || !input.validity.first().copied().unwrap_or(false) {
        return (None, Some("raw_complex_invalid"), None, None);
    }
    observations
        .iter_mut()
        .zip(&input.validity)
        .for_each(|(value, valid)| {
            if !valid {
                *value = f64::NAN;
            }
        });
    let difference_covariance = factor_covariance(&input.issue54_difference_factor);
    let fit = fit_temporal_covariance(
        &request.days,
        &observations,
        &difference_covariance,
        &request.options,
    );
    let provenance = temporal_covariance_provenance(&fit, input.provenance.clone());
    let status = if fit.status == TemporalInferenceStatus::Evaluated {
        "evaluated"
    } else {
        "estimator_failed"
    };
    (None, Some(status), Some(fit), Some(provenance))
}

fn optional_observations(values: &[Option<f64>]) -> Vec<f64> {
    values
        .iter()
        .map(|value| value.unwrap_or(f64::NAN))
        .collect()
}

fn valid_factor(factor: &[Vec<f64>], dimension: usize) -> bool {
    factor.len() == dimension
        && factor
            .iter()
            .all(|row| row.len() == dimension && row.iter().all(|value| value.is_finite()))
}

fn factor_covariance(factor: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..factor.len())
        .map(|row| {
            (0..factor.len())
                .map(|column| {
                    factor[row]
                        .iter()
                        .zip(&factor[column])
                        .map(|(left, right)| left * right)
                        .sum()
                })
                .collect()
        })
        .collect()
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
    let gauge = phases[0];
    phases.iter_mut().for_each(|value| *value -= gauge);
    Some(phases)
}

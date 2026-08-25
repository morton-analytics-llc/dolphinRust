//! Execute and emit the production spatial-covariance resource preflight.

use std::collections::BTreeMap;

use anyhow::{bail, ensure, Context, Result};
use dolphin_workflows::spatial_covariance_validation::run_benchmark_preflight;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Serialize)]
struct AllocationComponent {
    name: &'static str,
    bytes: u64,
    source: &'static str,
    component_sha256: String,
}

#[derive(Serialize)]
struct BenchmarkReceipt {
    block_count: u64,
    maximum_sources_per_block: u64,
    maximum_dependency_depth: u64,
    reference_cone_sources: u64,
    persisted_block_bytes: u64,
    scratch_bytes: u64,
    final_bytes: u64,
    allocation_components: Vec<AllocationComponent>,
    maximum_simultaneously_retained_bytes: u64,
    dependency_cone_bytes: u64,
    replay_reservation_bytes: u64,
    source_influence_bytes: u64,
    source_correlation_workspace_bytes: u64,
    source_correlation_model: String,
    source_cache_peak_bytes: u64,
    admitted_block_targets: u64,
    tile_pixels: u64,
    date_count: u64,
    runtime_resource_receipt: Value,
}

fn parse_positive(value: std::ffi::OsString, name: &str) -> Result<u64> {
    let value = value
        .into_string()
        .map_err(|_| anyhow::anyhow!("{name} is not UTF-8"))?
        .parse::<u64>()
        .with_context(|| format!("parsing {name}"))?;
    ensure!(value > 0, "{name} must be positive");
    Ok(value)
}

fn parse_args() -> Result<(u64, u64)> {
    let mut arguments = std::env::args_os().skip(1);
    let mut tile_pixels = None;
    let mut dates = None;
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--tile-pixels") => {
                ensure!(
                    tile_pixels.is_none(),
                    "--tile-pixels may be supplied only once"
                );
                tile_pixels = Some(parse_positive(
                    arguments.next().context("--tile-pixels requires a value")?,
                    "tile pixels",
                )?);
            }
            Some("--dates") => {
                ensure!(dates.is_none(), "--dates may be supplied only once");
                dates = Some(parse_positive(
                    arguments.next().context("--dates requires a value")?,
                    "dates",
                )?);
            }
            _ => bail!("unknown argument {}", argument.to_string_lossy()),
        }
    }
    Ok((
        tile_pixels.context("--tile-pixels is required")?,
        dates.context("--dates is required")?,
    ))
}

fn component(name: &'static str, bytes: u64, source: &'static str) -> AllocationComponent {
    let unsigned = serde_json::json!({
        "name": name,
        "bytes": bytes,
        "source": source,
    });
    let component_sha256 = sha256_canonical_json(&unsigned);
    AllocationComponent {
        name,
        bytes,
        source,
        component_sha256,
    }
}

fn sha256_canonical_json(value: &Value) -> String {
    let mut encoded = Vec::new();
    write_canonical_json(value, &mut encoded);
    format!("{:x}", Sha256::digest(encoded))
}

fn write_canonical_json(value: &Value, output: &mut Vec<u8>) {
    match value {
        Value::Object(values) => {
            output.push(b'{');
            let ordered: BTreeMap<_, _> = values.iter().collect();
            for (index, (key, item)) in ordered.into_iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                output.extend(serde_json::to_vec(key).expect("JSON object key is serializable"));
                output.push(b':');
                write_canonical_json(item, output);
            }
            output.push(b'}');
        }
        Value::Array(values) => {
            output.push(b'[');
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                write_canonical_json(item, output);
            }
            output.push(b']');
        }
        _ => output.extend(serde_json::to_vec(value).expect("JSON scalar is serializable")),
    }
}

fn main() -> Result<()> {
    let (tile_pixels, dates) = parse_args()?;
    let evidence = run_benchmark_preflight(
        usize::try_from(tile_pixels).context("tile pixels exceed usize")?,
        usize::try_from(dates).context("dates exceed usize")?,
    )?;
    let runtime = evidence.runtime_resource_receipt;
    let components = vec![
        component(
            "factor_block",
            runtime.factor_block_high_water_bytes,
            "production_resource_admission.factor_block_high_water_bytes",
        ),
        component(
            "serialization",
            runtime.serialization_high_water_bytes,
            "production_resource_admission.serialization_high_water_bytes",
        ),
        component(
            "fixed_l2_workspace",
            runtime.fixed_l2_workspace_admission_bytes,
            "fixed_l2_difference_workspace_composition.total_bytes",
        ),
        component(
            "replay_reservation",
            runtime.replay_admission_high_water_bytes,
            "production replay dependency-cone estimate",
        ),
    ];
    let receipt = BenchmarkReceipt {
        block_count: evidence.block_count,
        maximum_sources_per_block: evidence.reference_cone_sources,
        maximum_dependency_depth: evidence.maximum_dependency_depth,
        reference_cone_sources: evidence.reference_cone_sources,
        persisted_block_bytes: runtime.factor_block_high_water_bytes,
        scratch_bytes: runtime.serialization_high_water_bytes,
        final_bytes: evidence.covariance_result_bytes,
        allocation_components: components,
        maximum_simultaneously_retained_bytes: runtime.working_set_admission_high_water_bytes,
        dependency_cone_bytes: evidence.dependency_cone_bytes,
        replay_reservation_bytes: evidence.replay_reservation_bytes,
        source_influence_bytes: evidence.source_influence_bytes,
        source_correlation_workspace_bytes: evidence.source_correlation_workspace_bytes,
        source_correlation_model: evidence.source_correlation_model,
        source_cache_peak_bytes: evidence.source_cache_peak_bytes,
        admitted_block_targets: evidence.admitted_block_targets,
        tile_pixels,
        date_count: dates,
        runtime_resource_receipt: serde_json::to_value(runtime)?,
    };
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}

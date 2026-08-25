//! JSONL driver for the bounded production spatial-covariance validation runner.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use dolphin_workflows::spatial_covariance_validation::{
    run_frozen_attempt, run_validation_case, write_validation_fixture, FrozenAttemptRequest,
    PortableDgpTables, ValidationCoupling,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const RESPONSE_SCHEMA: &str = "dolphinrust.spatial-covariance.production-parity-fixture-evidence/4";
const MAX_RECORD_BYTES: usize = 262_144;
const MAX_PREREGISTRATION_BYTES: u64 = 4 << 20;
const MAX_PORTABLE_TABLE_BYTES: u64 = 8 << 20;

struct Arguments {
    preregistration: Option<PathBuf>,
    artifact_directory: Option<PathBuf>,
    cell_id: Option<String>,
    parity_fixture: bool,
}

fn coupling(request: &FrozenAttemptRequest) -> Result<ValidationCoupling> {
    if request.position == "masked" || request.eigen_stress == "tied_eigenvalue" {
        return Ok(ValidationCoupling::Invalid);
    }
    if request.pair_geometry == "coincident" {
        return Ok(ValidationCoupling::Coincident);
    }
    if request.pair_geometry.starts_with("disjoint") {
        return Ok(ValidationCoupling::Independent);
    }
    if request.pair_geometry.ends_with("_positive") {
        return Ok(ValidationCoupling::Positive);
    }
    if request.pair_geometry.ends_with("_negative") {
        return Ok(ValidationCoupling::Negative);
    }
    bail!("unsupported pair geometry {}", request.pair_geometry)
}

fn parse_args() -> Result<Arguments> {
    let mut arguments = std::env::args_os().skip(1);
    let mut artifact_directory = None;
    let mut preregistration = None;
    let mut cell_id = None;
    let mut parity_fixture = false;
    while let Some(argument) = arguments.next() {
        match argument.to_string_lossy().as_ref() {
            "--parity-fixture" => {
                ensure!(
                    !parity_fixture,
                    "--parity-fixture may be supplied only once"
                );
                parity_fixture = true;
            }
            "--ephemeral-evidence-stdout" => {}
            "--artifact-directory" => {
                ensure!(
                    artifact_directory.is_none(),
                    "--artifact-directory may be supplied only once"
                );
                artifact_directory = Some(PathBuf::from(
                    arguments
                        .next()
                        .context("--artifact-directory requires a path")?,
                ));
            }
            "--preregistration" => {
                ensure!(
                    preregistration.is_none(),
                    "--preregistration may be supplied only once"
                );
                preregistration = Some(PathBuf::from(
                    arguments
                        .next()
                        .context("--preregistration requires a path")?,
                ));
            }
            "--cell-id" => {
                ensure!(cell_id.is_none(), "--cell-id may be supplied only once");
                cell_id = Some(
                    arguments
                        .next()
                        .context("--cell-id requires a value")?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            _ => bail!("unknown argument {}", argument.to_string_lossy()),
        }
    }
    ensure!(
        parity_fixture || preregistration.is_some(),
        "full-cell mode requires --preregistration"
    );
    Ok(Arguments {
        preregistration,
        artifact_directory,
        cell_id,
        parity_fixture,
    })
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    let arguments = parse_args()?;
    let preregistration = arguments
        .preregistration
        .as_ref()
        .map(|path| read_json(path, MAX_PREREGISTRATION_BYTES))
        .transpose()?;
    let tables = match (arguments.preregistration.as_ref(), preregistration.as_ref()) {
        (Some(path), Some(preregistration)) if !arguments.parity_fixture => {
            let asset = preregistration
                .get("portable_dgp_asset")
                .context("preregistration omits portable DGP asset receipt")?;
            let relative = asset
                .get("path")
                .and_then(Value::as_str)
                .context("portable DGP asset omits path")?;
            let parent = path
                .parent()
                .context("preregistration has no parent directory")?;
            let canonical_parent = parent.canonicalize()?;
            let asset_path = parent.join(relative).canonicalize()?;
            ensure!(
                asset_path.starts_with(&canonical_parent),
                "portable DGP asset escapes preregistration directory"
            );
            let (table_value, table_bytes) =
                read_json_bytes(&asset_path, MAX_PORTABLE_TABLE_BYTES)?;
            ensure!(
                asset.get("byte_count").and_then(Value::as_u64) == Some(table_bytes.len() as u64),
                "portable DGP asset byte count differs"
            );
            let table_sha256 = format!("{:x}", Sha256::digest(&table_bytes));
            ensure!(
                asset.get("sha256").and_then(Value::as_str) == Some(table_sha256.as_str()),
                "portable DGP asset digest differs"
            );
            Some(PortableDgpTables::from_documents(
                preregistration,
                &table_value,
            )?)
        }
        _ => None,
    };
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let mut record = Vec::new();
    let mut record_index = 0_u64;

    loop {
        record.clear();
        let bytes = input
            .read_until(b'\n', &mut record)
            .context("reading validation request JSONL")?;
        if bytes == 0 {
            break;
        }
        ensure!(
            bytes <= MAX_RECORD_BYTES && record.ends_with(b"\n"),
            "validation request record exceeds the byte cap or lacks newline framing"
        );
        let request: FrozenAttemptRequest =
            serde_json::from_slice(&record).context("parsing validation request JSONL")?;
        if let Some(cell_id) = arguments.cell_id.as_ref() {
            ensure!(
                &request.cell_id == cell_id,
                "request differs from --cell-id"
            );
        }
        let result = if arguments.parity_fixture {
            let coupling = coupling(&request)?;
            let parity = if record_index == 0 {
                if let Some(directory) = arguments.artifact_directory.as_ref() {
                    std::fs::create_dir_all(directory).with_context(|| {
                        format!("creating artifact directory {}", directory.display())
                    })?;
                    write_validation_fixture(directory, coupling, request.seed_index)?
                } else {
                    run_validation_case(coupling, request.seed_index)?
                }
            } else {
                run_validation_case(coupling, request.seed_index)?
            };
            let mut value = serde_json::to_value(parity)?;
            value["schema"] = Value::String(RESPONSE_SCHEMA.to_owned());
            value["cell_id"] = Value::String(request.cell_id.clone());
            value["cell_ordinal"] = Value::from(request.cell_ordinal);
            value["seed_sha256"] = Value::String(request.seed_sha256.clone());
            for (name, field) in [
                ("half_window", &request.half_window),
                ("stride", &request.stride),
                ("support", &request.support),
                ("position", &request.position),
                ("pair_geometry", &request.pair_geometry),
                ("block_topology", &request.block_topology),
                ("estimator", &request.estimator),
                ("eigen_stress", &request.eigen_stress),
                ("source_process", &request.source_process),
            ] {
                value[name] = Value::String(field.clone());
            }
            value
        } else {
            run_frozen_attempt(
                preregistration
                    .as_ref()
                    .context("full-cell preregistration is absent")?,
                tables.as_ref().context("portable DGP tables are absent")?,
                &request,
            )?
        };
        let encoded = serde_json::to_vec(&result).context("encoding validation response")?;
        ensure!(
            encoded.len() < MAX_RECORD_BYTES,
            "validation response exceeds the byte cap"
        );
        output.write_all(&encoded)?;
        output.write_all(b"\n")?;
        record_index += 1;
    }
    output.flush()?;
    Ok(())
}

fn read_json(path: &Path, byte_cap: u64) -> Result<Value> {
    Ok(read_json_bytes(path, byte_cap)?.0)
}

fn read_json_bytes(path: &std::path::Path, byte_cap: u64) -> Result<(Value, Vec<u8>)> {
    let metadata = std::fs::metadata(path)?;
    ensure!(
        metadata.is_file() && metadata.len() <= byte_cap,
        "JSON input exceeds its byte cap"
    );
    let bytes = std::fs::read(path)?;
    ensure!(
        bytes.len() as u64 == metadata.len(),
        "JSON input changed during read"
    );
    Ok((serde_json::from_slice(&bytes)?, bytes))
}

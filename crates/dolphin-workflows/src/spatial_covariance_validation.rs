//! Validation runner for the production reference-specific covariance path.
//!
//! It constructs a deterministic influence graph, executes the production
//! joint replay, applies the production fixed-L2 map, builds the production
//! factor block, and optionally round-trips that block through the production
//! HDF5 writer and capped reader.

use std::collections::BTreeMap;
use std::mem::size_of;
use std::path::Path;

use anyhow::{Context, Result};
use dolphin_core::config::{
    CompressedSlcPlan, ComputeBackend, DisplacementWorkflow, EmpiricalSourceFactorOptions,
    InputType, ShpMethod,
};
use dolphin_core::{BlockIndices, Cf32, Cf64};
use dolphin_io::{
    read_spatial_reference_covariance_block, read_spatial_reference_covariance_header,
    CovarianceOperatorBlock, CovarianceOperatorGrid, CovarianceOperatorMetadata,
    CovarianceOperatorStatus, CovarianceOperatorWriter, CovarianceReplayStatus,
    DownstreamInferenceStatus, SourceReplayIdentity, SpatialReferenceCovarianceStatus,
    SpatialReferenceRuntimeResourceReceipt, StitchedCovarianceStatus,
    SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION, SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE,
};
use dolphin_phaselink::source_model::{
    estimate_empirical_proper_complex_factor, EmpiricalProperComplexConfig,
};
use dolphin_phaselink::{
    ComputeEngine, InfluenceDag, InfluenceNode, NodeId, SourceDefinition, SourceEdge, SourceId,
};
use dolphin_timeseries::spatial_covariance::fixed_l2_difference_workspace_composition;
use ndarray::{array, s, Array1, Array2, Array3, ArrayView2, ArrayView3, Axis};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::covariance_artifact::{
    admit_covariance_artifact_disk_with_identity_index, finalize_covariance_artifact,
    CovarianceArtifactTransaction,
};
use crate::cslc_covariance_source::{
    CslcCovarianceManifest, CSLC_COVARIANCE_SOURCE_MODEL, CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
    CSLC_COVARIANCE_SOURCE_PROVIDER, CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
};
use crate::sequential::{
    run_sequential_masked_with_covariance_capture_and_source_factors,
    run_sequential_with_covariance_capture,
    run_sequential_with_covariance_capture_and_source_factors, SequentialConfig,
};
use crate::sequential_covariance::{
    estimate_global_reference_difference_covariance_from_provider_bundle,
    replay_global_reference_difference_covariance_from_provider_bundle,
    sequential_replay_config_digest, sequential_replay_kernel_digest,
    sequential_source_model_identity_digest, DependencyConeQuery, GlobalBlockId, GlobalDateId,
    GlobalReferenceCovarianceQuery, ReferenceDifferenceCovarianceReplay, ReplayBackend,
    ReplayExecutionScope, ReplayIdNamespace, ReplayStatus, ResolvedCompressionReplay,
    ResolvedPhaseReplay, ResolvedPrimitiveSource, SequentialCovarianceCaptureRequest,
    SequentialPrimitiveSourceResolver, SequentialReplayBlock, SequentialReplayError,
    SequentialReplayTopology, SequentialSourceProviderIdentity, SequentialSourceReplayProvider,
    SequentialTileReplayProvider, SourceCorrelationModel,
};
use crate::spatial_covariance_artifact::{
    read_spatial_reference_covariance_artifact_manifest, SPATIAL_REFERENCE_COVARIANCE_FILENAME,
    SPATIAL_REFERENCE_COVARIANCE_LOCK_FILENAME, SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME,
};
use crate::spatial_reference_covariance_output::{
    build_factor_block, correction_order_digest, factor_block_shape, production_resource_admission,
    unwrap_branch_digest, BurstOutputMapping, CapturedReplayTile, FixedL2WorkflowInputs,
    ProductionCovarianceReplayContext, ProductionCovarianceState, TargetFactor,
    PRODUCTION_REFERENCE_COVARIANCE_WORKING_SET_BYTE_CAP,
};

const VALIDATION_BYTE_CAP: u64 = 1 << 20;
const VALIDATION_DATES: usize = 2;
const FULL_CELL_BYTE_CAP: u64 = 1 << 30;

/// Parsed frozen tables for the cross-language portable validation DGP.
pub struct PortableDgpTables {
    dgp_generator_identity: [u8; 32],
    normal_quantiles: Vec<f64>,
    index_bits: u32,
    temporal_rho: f64,
    innovation_weight: f64,
    independent_local_weight: f64,
    independent_spatial_weight: f64,
    spatial_local_weight: f64,
    spatial_global_weight: f64,
    noise_scale: f64,
    amplitude_scale: BTreeMap<i64, BTreeMap<String, Vec<f64>>>,
    phasor: BTreeMap<i64, Vec<(f64, f64)>>,
    latent_phase: BTreeMap<i64, Vec<f64>>,
}

/// One exact request from the frozen v4 cell/seed iterator.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenAttemptRequest {
    /// Request schema identity.
    pub schema: String,
    /// Exact pipe-joined dimension identity.
    pub cell_id: String,
    /// Lexicographic frozen cell ordinal.
    pub cell_ordinal: u64,
    /// Fixed zero-based seed index.
    pub seed_index: u64,
    /// Frozen seed derivation receipt.
    pub seed_sha256: String,
    /// Half-window dimension identity.
    pub half_window: String,
    /// Stride dimension identity.
    pub stride: String,
    /// SHP support method identity.
    pub support: String,
    /// Spatial boundary position identity.
    pub position: String,
    /// Target/reference geometry identity.
    pub pair_geometry: String,
    /// Sequential block topology identity.
    pub block_topology: String,
    /// Fixed EMI or EVD branch.
    pub estimator: String,
    /// Eigenvalue stress identity.
    pub eigen_stress: String,
    /// Primitive source-process identity.
    pub source_process: String,
}

impl FrozenAttemptRequest {
    fn dimensions(&self) -> [(&'static str, &str); 9] {
        [
            ("half_window", &self.half_window),
            ("stride", &self.stride),
            ("support", &self.support),
            ("position", &self.position),
            ("pair_geometry", &self.pair_geometry),
            ("block_topology", &self.block_topology),
            ("estimator", &self.estimator),
            ("eigen_stress", &self.eigen_stress),
            ("source_process", &self.source_process),
        ]
    }

    fn validate(&self, preregistration: &Value) -> Result<u64> {
        anyhow::ensure!(
            self.schema == "dolphinrust.spatial-covariance.attempt/4",
            "unsupported frozen attempt request schema"
        );
        let expected_id = self
            .dimensions()
            .iter()
            .map(|(_, value)| *value)
            .collect::<Vec<_>>()
            .join("|");
        anyhow::ensure!(
            self.cell_id == expected_id,
            "frozen attempt dimensions disagree"
        );
        let cells = expected_cell_ids_from_preregistration(preregistration)?;
        let expected_cell = cells.get(self.cell_ordinal as usize);
        anyhow::ensure!(
            expected_cell == Some(&self.cell_id),
            "frozen attempt cell ordinal differs from the preregistered iterator: expected {expected_cell:?}"
        );
        let seed = preregistration
            .pointer("/seed_schedule/validation_seed")
            .and_then(Value::as_str)
            .context("preregistration omits validation seed")?;
        let expected_seed = format!(
            "{:x}",
            Sha256::digest(format!("{seed}||{}||{}", self.cell_id, self.seed_index))
        );
        anyhow::ensure!(
            self.seed_sha256 == expected_seed,
            "frozen attempt seed digest differs"
        );
        let dgp_id = if self.pair_geometry.ends_with("_negative") {
            let positive_id = self.cell_id.replace("_negative|", "_positive|");
            cells
                .iter()
                .position(|cell| cell == &positive_id)
                .map_or(self.cell_ordinal, |ordinal| ordinal as u64)
        } else {
            self.cell_ordinal
        };
        Ok(dgp_id)
    }
}

fn expected_cell_ids_from_preregistration(preregistration: &Value) -> Result<Vec<String>> {
    let matrix = preregistration
        .get("matrix_contract")
        .context("preregistration omits matrix contract")?;
    let order = matrix
        .get("dimension_order")
        .and_then(Value::as_array)
        .context("matrix contract omits dimension order")?;
    let dimensions = preregistration
        .get("dimensions")
        .context("preregistration omits dimensions")?;
    let mut levels = Vec::with_capacity(order.len());
    for name in order {
        let name = name.as_str().context("dimension name is not a string")?;
        let values = dimensions
            .get(name)
            .and_then(Value::as_array)
            .with_context(|| format!("dimension {name} is absent"))?
            .iter()
            .map(|entry| {
                entry
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .with_context(|| format!("dimension {name} has an invalid level"))
            })
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(!values.is_empty(), "frozen dimension has no levels");
        levels.push(values);
    }
    let defaults = levels
        .iter()
        .map(|values| values[0].clone())
        .collect::<Vec<_>>();
    let mut cells = std::collections::BTreeSet::new();
    for left in 0..levels.len() {
        for right in left + 1..levels.len() {
            for left_value in &levels[left] {
                for right_value in &levels[right] {
                    let mut labels = defaults.clone();
                    labels[left] = left_value.clone();
                    labels[right] = right_value.clone();
                    cells.insert(labels);
                }
            }
        }
    }
    for cell in matrix
        .get("risk_cells")
        .and_then(Value::as_array)
        .context("matrix contract omits risk cells")?
    {
        cells.insert(
            cell.as_str()
                .context("risk cell is not a string")?
                .split('|')
                .map(str::to_owned)
                .collect::<Vec<_>>(),
        );
    }
    let expected = matrix
        .get("expected_cell_count")
        .and_then(Value::as_u64)
        .context("matrix contract omits expected cell count")?;
    anyhow::ensure!(
        cells.len() as u64 == expected,
        "frozen cell iterator count differs"
    );
    Ok(cells.into_iter().map(|labels| labels.join("|")).collect())
}

impl PortableDgpTables {
    /// Parse and bind the frozen DGP tables from the accepted preregistration.
    ///
    /// # Errors
    /// Returns an error for missing, malformed, non-finite, or digest-inconsistent tables.
    pub fn from_preregistration(preregistration: &Value) -> Result<Self> {
        let tables = preregistration
            .get("portable_dgp_tables")
            .context("preregistration omits embedded portable DGP tables")?;
        Self::from_documents(preregistration, tables)
    }

    /// Parse a separately hash-bound portable table asset.
    ///
    /// # Errors
    /// Returns an error for asset hash/size drift or malformed numeric tables.
    #[allow(clippy::too_many_lines)]
    pub fn from_documents(preregistration: &Value, tables: &Value) -> Result<Self> {
        let identity_text = preregistration
            .pointer("/determinism/dgp_generator_identity_sha256")
            .and_then(Value::as_str)
            .context("preregistration omits the DGP generator identity")?;
        anyhow::ensure!(
            identity_text.len() == 64,
            "DGP generator identity is not a SHA-256 digest"
        );
        let mut dgp_generator_identity = [0_u8; 32];
        for (index, byte) in dgp_generator_identity.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&identity_text[index * 2..index * 2 + 2], 16)
                .context("DGP generator identity is not hexadecimal")?;
        }
        let normal = tables
            .get("normal_quantile")
            .context("portable DGP omits its normal table")?;
        let index_bits = u32::try_from(
            normal
                .get("index_bits")
                .and_then(Value::as_u64)
                .context("portable normal table omits index_bits")?,
        )?;
        anyhow::ensure!(
            index_bits > 0 && index_bits < 64,
            "portable normal index width is invalid"
        );
        let normal_quantiles = normal
            .get("entries")
            .and_then(Value::as_array)
            .context("portable normal table omits entries")?
            .iter()
            .map(parse_f64_bits_value)
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(
            normal_quantiles.len() == 1usize << index_bits,
            "portable normal table length differs from its index width"
        );
        let coefficients = tables
            .get("coefficients")
            .context("portable DGP omits coefficient tables")?;
        let scalar = |name: &str| -> Result<f64> {
            parse_f64_bits_value(
                coefficients
                    .get(name)
                    .with_context(|| format!("portable DGP omits {name}"))?,
            )
        };
        let amplitude_scale = parse_scalar_table(
            coefficients
                .get("amplitude_scale_bits")
                .context("portable DGP omits amplitude scale table")?,
        )?;
        let phasor = coefficients
            .get("phasor_bits")
            .and_then(Value::as_object)
            .context("portable DGP omits phasor table")?
            .iter()
            .map(|(key, values)| {
                let argument = key.parse::<i64>()?;
                let entries = values
                    .as_array()
                    .context("portable phasor entries are not an array")?
                    .iter()
                    .map(|pair| {
                        let pair = pair
                            .as_array()
                            .context("portable phasor entry is not a pair")?;
                        anyhow::ensure!(pair.len() == 2, "portable phasor entry is not a pair");
                        Ok((
                            parse_f64_bits_value(&pair[0])?,
                            parse_f64_bits_value(&pair[1])?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((argument, entries))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let latent_phase = coefficients
            .get("latent_phase_bits")
            .and_then(Value::as_object)
            .context("portable DGP omits latent phase table")?
            .iter()
            .map(|(key, values)| {
                Ok((
                    key.parse::<i64>()?,
                    values
                        .as_array()
                        .context("portable latent phase entries are not an array")?
                        .iter()
                        .map(parse_f64_bits_value)
                        .collect::<Result<Vec<_>>>()?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            dgp_generator_identity,
            normal_quantiles,
            index_bits,
            temporal_rho: scalar("temporal_rho_bits")?,
            innovation_weight: scalar("innovation_weight_bits")?,
            independent_local_weight: scalar("independent_local_weight_bits")?,
            independent_spatial_weight: scalar("independent_spatial_weight_bits")?,
            spatial_local_weight: scalar("spatial_local_weight_bits")?,
            spatial_global_weight: scalar("spatial_global_weight_bits")?,
            noise_scale: scalar("noise_scale_bits")?,
            amplitude_scale,
            phasor,
            latent_phase,
        })
    }

    fn normal(
        &self,
        cell_ordinal: u64,
        seed_index: u64,
        coordinate: (i64, i64),
        date: u32,
        stream: &str,
    ) -> Result<f64> {
        anyhow::ensure!(
            !stream.is_empty() && stream.len() < u16::MAX as usize && stream.is_ascii(),
            "portable DGP stream identity is invalid"
        );
        let mut digest = Sha256::new();
        digest.update(b"dolphinrust:spatial-covariance-dgp:v1\0");
        digest.update(self.dgp_generator_identity);
        digest.update(cell_ordinal.to_le_bytes());
        digest.update(seed_index.to_le_bytes());
        digest.update(coordinate.0.to_le_bytes());
        digest.update(coordinate.1.to_le_bytes());
        digest.update(date.to_le_bytes());
        digest.update((stream.len() as u16).to_le_bytes());
        digest.update(stream.as_bytes());
        digest.update(0_u64.to_le_bytes());
        let bytes = digest.finalize();
        let word = u64::from_le_bytes(
            bytes[..8]
                .try_into()
                .expect("SHA-256 prefix is eight bytes"),
        );
        let index = (word >> (64 - self.index_bits)) as usize;
        Ok(self.normal_quantiles[index])
    }

    /// Generate one exact complex64 source history using fixed scalar order.
    ///
    /// # Errors
    /// Returns an error when the frozen coefficient tables do not cover the coordinate/date cell.
    #[allow(clippy::too_many_arguments)]
    pub fn source_history(
        &self,
        cell_ordinal: u64,
        seed_index: u64,
        coordinate: (i64, i64),
        dates: usize,
        spatial: bool,
        eigen_stress: &str,
        global_loading: f64,
    ) -> Result<Vec<Cf64>> {
        let amplitude = self
            .amplitude_scale
            .get(&(coordinate.0 + 3 * coordinate.1))
            .and_then(|stress| stress.get(eigen_stress))
            .context("portable amplitude table does not cover the cell")?;
        let phasor = self
            .phasor
            .get(&(2 * coordinate.0 - coordinate.1))
            .context("portable phasor table does not cover the cell")?;
        anyhow::ensure!(
            dates <= amplitude.len() && dates <= phasor.len(),
            "portable DGP date count exceeds the frozen tables"
        );
        anyhow::ensure!(
            global_loading == 1.0 || global_loading == -1.0,
            "portable DGP signed loading must be exactly plus or minus one"
        );
        let (local_weight, spatial_weight) = if spatial {
            (self.spatial_local_weight, self.spatial_global_weight)
        } else {
            (
                self.independent_local_weight,
                self.independent_spatial_weight,
            )
        };
        let mut state_real = 0.0;
        let mut state_imaginary = 0.0;
        let mut values = Vec::with_capacity(dates);
        for date in 0..dates {
            let date_u32 = u32::try_from(date)?;
            let innovation = |local_stream: &str, global_stream: &str| -> Result<f64> {
                let local =
                    self.normal(cell_ordinal, seed_index, coordinate, date_u32, local_stream)?;
                let global =
                    self.normal(cell_ordinal, seed_index, (0, 0), date_u32, global_stream)?;
                let weighted_local = local_weight * local;
                let weighted_global = spatial_weight * global;
                Ok(weighted_local + weighted_global)
            };
            let innovation_real = innovation("local-signal-real", "global-signal-real")?;
            let innovation_imaginary = innovation("local-signal-imag", "global-signal-imag")?;
            if date == 0 {
                state_real = innovation_real;
                state_imaginary = innovation_imaginary;
            } else {
                let previous_real = self.temporal_rho * state_real;
                let weighted_real = self.innovation_weight * innovation_real;
                state_real = previous_real + weighted_real;
                let previous_imaginary = self.temporal_rho * state_imaginary;
                let weighted_imaginary = self.innovation_weight * innovation_imaginary;
                state_imaginary = previous_imaginary + weighted_imaginary;
            }
            let noise_real =
                self.noise_scale * innovation("local-noise-real", "global-noise-real")?;
            let noise_imaginary =
                self.noise_scale * innovation("local-noise-imag", "global-noise-imag")?;
            let base_real = amplitude[date] * state_real + noise_real;
            let unsigned_base_imaginary = amplitude[date] * state_imaginary + noise_imaginary;
            let base_imaginary = global_loading * unsigned_base_imaginary;
            let (cosine, sine) = phasor[date];
            let value_real = base_real * cosine - base_imaginary * sine;
            let value_imaginary = base_real * sine + base_imaginary * cosine;
            values.push(Cf64::new(
                f64::from(value_real as f32),
                f64::from(value_imaginary as f32),
            ));
        }
        Ok(values)
    }

    /// Exact latent phase history for a frozen coordinate.
    pub fn latent_history(&self, coordinate: (i64, i64), dates: usize) -> Result<Vec<f64>> {
        let values = self
            .latent_phase
            .get(&(2 * coordinate.0 - coordinate.1))
            .context("portable latent table does not cover the cell")?;
        anyhow::ensure!(
            dates <= values.len(),
            "latent date count exceeds the frozen table"
        );
        Ok(values[..dates].to_vec())
    }
}

fn parse_f64_bits_value(value: &Value) -> Result<f64> {
    let text = value
        .as_str()
        .context("portable DGP IEEE-754 entry is not a string")?;
    anyhow::ensure!(
        text.len() == 16
            && text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "portable DGP IEEE-754 entry is malformed"
    );
    let value = f64::from_bits(u64::from_str_radix(text, 16)?);
    anyhow::ensure!(
        value.is_finite(),
        "portable DGP IEEE-754 entry is non-finite"
    );
    Ok(value)
}

fn parse_scalar_table(value: &Value) -> Result<BTreeMap<i64, BTreeMap<String, Vec<f64>>>> {
    value
        .as_object()
        .context("portable scalar table is not an object")?
        .iter()
        .map(|(key, stress)| {
            let by_stress = stress
                .as_object()
                .context("portable scalar stress table is not an object")?
                .iter()
                .map(|(name, entries)| {
                    Ok((
                        name.clone(),
                        entries
                            .as_array()
                            .context("portable scalar entries are not an array")?
                            .iter()
                            .map(parse_f64_bits_value)
                            .collect::<Result<Vec<_>>>()?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?;
            Ok((key.parse::<i64>()?, by_stress))
        })
        .collect()
}

struct FrozenCellGeometry {
    target: (i64, i64),
    reference: (i64, i64),
    half_window: (usize, usize),
    stride: (usize, usize),
    native_tile_shape: (usize, usize),
    dates: usize,
    ministack_size: usize,
    max_num_compressed: usize,
    date_axis: Value,
    topology: Value,
}

impl FrozenCellGeometry {
    fn parse(preregistration: &Value, request: &FrozenAttemptRequest) -> Result<Self> {
        let key = format!("{}|{}", request.half_window, request.stride);
        let window = preregistration
            .pointer("/generator/coordinates/window_stride")
            .and_then(|value| value.get(&key))
            .context("frozen coordinate table does not cover the request")?;
        let target = pair_i64(
            window
                .get("target_by_position")
                .and_then(|value| value.get(&request.position))
                .context("frozen coordinate table omits target position")?,
        )?;
        let delta = pair_i64(
            window
                .get("reference_delta_by_pair_geometry")
                .and_then(|value| value.get(&request.pair_geometry))
                .context("frozen coordinate table omits pair geometry")?,
        )?;
        let reference = (target.0 + delta.0, target.1 + delta.1);
        let half_window_u64 = pair_u64(
            window
                .get("half_window")
                .context("frozen coordinate table omits half-window")?,
        )?;
        let stride_u64 = pair_u64(
            window
                .get("stride")
                .context("frozen coordinate table omits stride")?,
        )?;
        let native_u64 = pair_u64(
            preregistration
                .pointer("/generator/full_replay_dgp/native_tile_shape")
                .context("frozen generator omits native tile shape")?,
        )?;
        let topology = preregistration
            .pointer("/generator/acquisition/topologies")
            .and_then(|value| value.get(&request.block_topology))
            .cloned()
            .context("frozen acquisition topology is absent")?;
        let dates = usize::try_from(
            topology
                .get("acquisition_count")
                .and_then(Value::as_u64)
                .context("frozen topology omits acquisition count")?,
        )?;
        let ministack_size = usize::try_from(
            topology
                .get("ministack_size")
                .and_then(Value::as_u64)
                .context("frozen topology omits ministack size")?,
        )?;
        let max_num_compressed = usize::try_from(
            topology
                .get("max_num_compressed")
                .and_then(Value::as_u64)
                .context("frozen topology omits compressed cap")?,
        )?;
        let date_axis = topology
            .get("date_axis")
            .cloned()
            .context("frozen topology omits date axis")?;
        Ok(Self {
            target,
            reference,
            half_window: (
                usize::try_from(half_window_u64.0)?,
                usize::try_from(half_window_u64.1)?,
            ),
            stride: (
                usize::try_from(stride_u64.0)?,
                usize::try_from(stride_u64.1)?,
            ),
            native_tile_shape: (
                usize::try_from(native_u64.0)?,
                usize::try_from(native_u64.1)?,
            ),
            dates,
            ministack_size,
            max_num_compressed,
            date_axis,
            topology,
        })
    }
}

fn pair_i64(value: &Value) -> Result<(i64, i64)> {
    let values = value
        .as_array()
        .context("frozen coordinate is not a pair")?;
    anyhow::ensure!(values.len() == 2, "frozen coordinate is not a pair");
    Ok((
        values[0].as_i64().context("frozen row is not an integer")?,
        values[1]
            .as_i64()
            .context("frozen column is not an integer")?,
    ))
}

fn pair_u64(value: &Value) -> Result<(u64, u64)> {
    let values = value.as_array().context("frozen dimension is not a pair")?;
    anyhow::ensure!(values.len() == 2, "frozen dimension is not a pair");
    Ok((
        values[0]
            .as_u64()
            .context("frozen row dimension is not unsigned")?,
        values[1]
            .as_u64()
            .context("frozen column dimension is not unsigned")?,
    ))
}

fn native_center_to_output(center: i64, stride: usize) -> Result<usize> {
    anyhow::ensure!(stride > 0 && center >= 0, "invalid native center or stride");
    let center = usize::try_from(center)?;
    let offset = stride / 2;
    let relative = center
        .checked_sub(offset)
        .context("native center precedes the production stride offset")?;
    anyhow::ensure!(
        relative.is_multiple_of(stride),
        "native center is not congruent with the production output grid"
    );
    Ok(relative / stride)
}

fn output_to_native_center(output: usize, stride: usize) -> Result<i64> {
    anyhow::ensure!(stride > 0, "output stride must be positive");
    let center = output
        .checked_mul(stride)
        .and_then(|value| value.checked_add(stride / 2))
        .context("production native center overflows usize")?;
    Ok(i64::try_from(center)?)
}

fn inward_clamped_support(
    center: (i64, i64),
    half_window: (usize, usize),
    native_shape: (usize, usize),
) -> Result<Vec<(i64, i64)>> {
    anyhow::ensure!(
        center.0 >= 0 && center.1 >= 0,
        "frozen target/reference is negative"
    );
    let window = (2 * half_window.0 + 1, 2 * half_window.1 + 1);
    anyhow::ensure!(
        window.0 <= native_shape.0 && window.1 <= native_shape.1,
        "frozen support exceeds the native tile"
    );
    let row = usize::try_from(center.0)?;
    let column = usize::try_from(center.1)?;
    let row_start = row.saturating_sub(half_window.0);
    let column_start = column.saturating_sub(half_window.1);
    Ok((row_start..row_start + window.0)
        .flat_map(|row| {
            (column_start..column_start + window.1).map(move |column| (row as i64, column as i64))
        })
        .collect())
}

fn factor_halo(
    support: impl IntoIterator<Item = (i64, i64)>,
    geometry: &FrozenCellGeometry,
) -> Result<std::collections::BTreeSet<(i64, i64)>> {
    let mut halo = std::collections::BTreeSet::new();
    for coordinate in support {
        halo.extend(inward_clamped_support(
            coordinate,
            geometry.half_window,
            geometry.native_tile_shape,
        )?);
    }
    Ok(halo)
}

fn tied_probe_stack(preregistration: &Value) -> Result<Array3<Cf64>> {
    let values = preregistration
        .pointer("/generator/singular_local_information_probe/raw_complex_binary64_bits")
        .and_then(Value::as_array)
        .context("preregistration omits the singular probe samples")?;
    anyhow::ensure!(values.len() == 9, "singular probe source count differs");
    let mut stack = Array3::zeros((3, 3, 3));
    for (source, history) in values.iter().enumerate() {
        let history = history
            .as_array()
            .context("singular probe history is not an array")?;
        anyhow::ensure!(history.len() == 3, "singular probe date count differs");
        for (date, components) in history.iter().enumerate() {
            let components = components
                .as_array()
                .context("singular probe component is not an array")?;
            anyhow::ensure!(
                components.len() == 2,
                "singular probe component count differs"
            );
            let parse = |index: usize| -> Result<f64> {
                let bits = components[index]
                    .as_str()
                    .context("singular probe component bits are not a string")?;
                Ok(f64::from_bits(u64::from_str_radix(bits, 16)?))
            };
            stack[(date, source / 3, source % 3)] = Cf64::new(parse(0)?, parse(1)?);
        }
    }
    Ok(stack)
}

fn execute_tied_probe(preregistration: &Value) -> Result<CovarianceOperatorStatus> {
    let probe = preregistration
        .pointer("/generator/singular_local_information_probe")
        .context("preregistration omits the singular probe")?;
    let mut config = validation_config();
    config.ministack_size = 3;
    config.max_num_compressed = 1;
    config.half_window = dolphin_core::HalfWindow { y: 1, x: 1 };
    config.strides = dolphin_core::Strides { y: 3, x: 3 };
    config.use_evd = true;
    config.shp_method = ShpMethod::Rect;
    let grid = CovarianceOperatorGrid {
        row_start: 0,
        col_start: 0,
        rows: 1,
        cols: 1,
        stride_y: 3,
        stride_x: 3,
    };
    let native_grid = CovarianceOperatorGrid {
        row_start: 0,
        col_start: 0,
        rows: 3,
        cols: 3,
        stride_y: 1,
        stride_x: 1,
    };
    let branch_tolerance = probe
        .get("branch_tolerance")
        .and_then(Value::as_f64)
        .context("singular probe omits branch tolerance")?;
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "spatial-covariance-singular-probe".to_owned(),
        source_manifest_digest: Sha256::digest(canonical_json_bytes(probe)?).into(),
        source_model_version_digest: [0x53; 32],
        native_grid,
        output_grid: grid,
        owned_output_grid: grid,
        branch_tolerance,
    };
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let mut blocks = Vec::new();
    run_sequential_with_covariance_capture(
        tied_probe_stack(preregistration)?.view(),
        &config,
        &engine,
        &request,
        |block| {
            blocks.push(block);
            Ok(())
        },
    )?;
    anyhow::ensure!(
        blocks.len() == 1,
        "singular probe emitted the wrong block count"
    );
    let block = &blocks[0];
    anyhow::ensure!(
        block.status.len() == 1,
        "singular probe emitted the wrong node count"
    );
    anyhow::ensure!(
        block.eigen_gap[0] <= branch_tolerance,
        "singular probe eigen gap exceeds its branch tolerance"
    );
    Ok(block.status[0])
}

#[allow(clippy::too_many_lines)]
fn build_tied_probe_evidence(
    preregistration: &Value,
    request: &FrozenAttemptRequest,
) -> Result<Value> {
    anyhow::ensure!(
        execute_tied_probe(preregistration)? == CovarianceOperatorStatus::SingularLocalInformation,
        "singular probe did not reach the production singular status"
    );
    let probe = preregistration
        .pointer("/generator/singular_local_information_probe")
        .context("preregistration omits the singular probe")?;
    let stack = tied_probe_stack(preregistration)?;
    let support = (0..3_i64)
        .flat_map(|row| (0..3_i64).map(move |column| (row, column)))
        .collect::<Vec<_>>();
    let mut raw_digest = Sha256::new();
    raw_digest.update(b"singular-local-information-probe-v1");
    raw_digest.update((support.len() as u64).to_le_bytes());
    for &(row, column) in &support {
        raw_digest.update(row.to_le_bytes());
        raw_digest.update(column.to_le_bytes());
        raw_digest.update(3_u64.to_le_bytes());
        for date in 0..3 {
            let value = stack[(date, row as usize, column as usize)];
            raw_digest.update(value.re.to_bits().to_le_bytes());
            raw_digest.update(value.im.to_bits().to_le_bytes());
        }
    }
    let raw_input_sha256 = format!("{:x}", raw_digest.finalize());
    let source_receipt = serde_json::json!([{
        "block_id": 0,
        "sources": support.iter().map(|&(row, column)| [row, column]).collect::<Vec<_>>(),
    }]);
    let support_sha256 = sha256_json_value(&source_receipt)?;
    let ancestry = serde_json::json!({
        "probe_schema": probe.get("schema").context("singular probe omits schema")?,
        "native_shape": probe.get("native_shape").context("singular probe omits native shape")?,
        "date_axis": probe.get("date_axis").context("singular probe omits date axis")?,
        "half_window": probe.get("half_window").context("singular probe omits half window")?,
        "stride": probe.get("stride").context("singular probe omits stride")?,
        "ministack_size": probe.get("ministack_size").context("singular probe omits ministack size")?,
        "max_num_compressed": probe.get("max_num_compressed").context("singular probe omits compressed cap")?,
        "estimator": probe.get("estimator").context("singular probe omits estimator")?,
        "branch_tolerance": probe.get("branch_tolerance").context("singular probe omits branch tolerance")?,
    });
    let raw_identity = serde_json::json!({
        "cell_id": request.cell_id,
        "seed_index": request.seed_index,
        "dgp_generator_identity_sha256": preregistration.pointer("/determinism/dgp_generator_identity_sha256").context("preregistration omits DGP generator identity")?,
        "probe_sha256": sha256_json_value(probe)?,
        "raw_input_sha256": raw_input_sha256,
        "expected_production_status": "singular_local_information",
        "scientific_numeric_axes_executed": false,
    });
    let generator = preregistration
        .get("generator")
        .context("preregistration omits generator")?;
    let source_model = generator
        .get("source_centered_empirical")
        .context("generator omits source model")?;
    let zero = "0".repeat(64);
    let mut evidence = serde_json::json!({
        "schema": "dolphinrust.spatial-covariance.attempt-evidence/4",
        "cell_id": request.cell_id,
        "cell_ordinal": request.cell_ordinal,
        "seed_index": request.seed_index,
        "seed_sha256": request.seed_sha256,
        "status": "singular_local_information",
        "emitted": false,
        "factor_emitted": false,
        "raw_input_shape": [9, 3, 2],
        "raw_input_value_count": 54,
        "raw_input_sha256": raw_input_sha256,
        "target_raw_input_sha256": raw_input_sha256,
        "reference_raw_input_sha256": raw_input_sha256,
        "sequential_ancestry_sha256": sha256_json_value(&ancestry)?,
        "raw_dgp_identity_sha256": sha256_json_value(&raw_identity)?,
        "latent_history_sha256": zero,
        "estimate_sha256": zero,
        "predicted_covariance_sha256": zero,
        "date_axis_sha256": sha256_json_value(probe.get("date_axis").context("singular probe omits date axis")?)?,
        "generator_hash": sha256_json_value(generator)?,
        "config_hash": sha256_json_value(generator)?,
        "source_model_hash": sha256_json_value(source_model)?,
    })
    .as_object()
    .cloned()
    .expect("singular evidence identity is an object");
    let disposition = serde_json::json!({
        "target_coordinate": [0, 0],
        "reference_coordinate": [0, 0],
        "target_support_sha256": support_sha256,
        "reference_support_sha256": support_sha256,
        "target_source_count": 9,
        "reference_source_count": 9,
        "intersection_source_count": 9,
        "union_source_count": 9,
        "realized_overlap_jaccard": 1.0,
        "signed_cross_influence": Value::Null,
        "signed_influence_sign": pair_sign(&request.pair_geometry)?,
        "effective_looks_fraction": 1.0,
        "effective_looks_application": "source_influence_joint_contraction_v1",
        "source_correlation_model": if request.source_process == "independent_complex_looks" { "identity_v1" } else { "exponential_euclidean_v1" },
        "source_correlation_distance_scale_pixels": if request.source_process == "independent_complex_looks" { 0.0 } else { 1.5 },
        "estimator_branch": "evd",
        "target_estimate_history": Value::Null,
        "reference_estimate_history": Value::Null,
        "predicted_difference_covariance": Value::Null,
        "production_operator_matrix": Value::Null,
        "contrast_weights": Value::Null,
        "operator_sha256": zero,
    });
    evidence.extend(
        disposition
            .as_object()
            .expect("singular evidence disposition is an object")
            .clone(),
    );
    Ok(Value::Object(evidence))
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn build_empty_support_evidence(
    preregistration: &Value,
    tables: &PortableDgpTables,
    request: &FrozenAttemptRequest,
    dgp_ordinal: u64,
    geometry: &FrozenCellGeometry,
    raw: &BTreeMap<(i64, i64), Vec<Cf64>>,
    status: &str,
) -> Result<Value> {
    let blocks = geometry
        .topology
        .get("expected_blocks")
        .and_then(Value::as_array)
        .context("topology omits expected blocks")?;
    let empty_receipt = Value::Array(
        blocks
            .iter()
            .map(|block| {
                Ok(serde_json::json!({
                    "block_id": block.get("block_id").and_then(Value::as_u64).context("expected block omits id")?,
                    "sources": Vec::<[i64; 2]>::new(),
                }))
            })
            .collect::<Result<Vec<_>>>()?,
    );
    let support_sha256 = sha256_json_value(&empty_receipt)?;
    let raw_union = raw
        .keys()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let empty = std::iter::empty::<(i64, i64)>();
    let raw_input_sha256 = raw_source_digest("raw-input-v4", raw_union.iter().copied(), raw)?;
    let target_raw_input_sha256 = raw_source_digest("source-raw-input-v4", empty, raw)?;
    let reference_raw_input_sha256 =
        raw_source_digest("source-raw-input-v4", std::iter::empty::<(i64, i64)>(), raw)?;
    let ancestry = serde_json::json!({
        "date_axis": geometry.date_axis,
        "expected_blocks": blocks,
        "max_num_compressed": geometry.max_num_compressed,
        "partial_tail_count": geometry.topology.get("partial_tail_count").context("topology omits partial tail")?,
    });
    let raw_shape = serde_json::json!([raw_union.len(), geometry.dates, 2]);
    let raw_identity = serde_json::json!({
        "cell_id": request.cell_id,
        "dgp_cell_ordinal": dgp_ordinal,
        "seed_index": request.seed_index,
        "dgp_generator_identity_sha256": preregistration.pointer("/determinism/dgp_generator_identity_sha256").context("preregistration omits DGP generator identity")?,
        "shape": raw_shape,
        "target_coordinate": [geometry.target.0, geometry.target.1],
        "reference_coordinate": [geometry.reference.0, geometry.reference.1],
        "target_support_sha256": support_sha256,
        "reference_support_sha256": support_sha256,
        "target_factor_support_sha256": support_sha256,
        "reference_factor_support_sha256": support_sha256,
        "raw_input_sha256": raw_input_sha256,
        "target_raw_input_sha256": target_raw_input_sha256,
        "reference_raw_input_sha256": reference_raw_input_sha256,
        "sequential_ancestry_sha256": sha256_json_value(&ancestry)?,
        "estimator": request.estimator,
        "eigen_stress": request.eigen_stress,
        "source_process": request.source_process,
    });
    let latent_values = tables
        .latent_history(geometry.target, geometry.dates)?
        .into_iter()
        .chain(tables.latent_history(geometry.reference, geometry.dates)?)
        .collect::<Vec<_>>();
    let generator = preregistration
        .get("generator")
        .context("preregistration omits generator")?;
    let source_model = generator
        .get("source_centered_empirical")
        .context("generator omits source model")?;
    let zero = "0".repeat(64);
    let mut evidence = serde_json::json!({
        "schema": "dolphinrust.spatial-covariance.attempt-evidence/4",
        "cell_id": request.cell_id,
        "cell_ordinal": request.cell_ordinal,
        "seed_index": request.seed_index,
        "seed_sha256": request.seed_sha256,
        "status": status,
        "emitted": false,
        "factor_emitted": false,
        "raw_input_shape": raw_shape,
        "raw_input_value_count": 2 * raw_union.len() * geometry.dates,
        "raw_input_sha256": raw_input_sha256,
        "target_raw_input_sha256": target_raw_input_sha256,
        "reference_raw_input_sha256": reference_raw_input_sha256,
        "sequential_ancestry_sha256": sha256_json_value(&ancestry)?,
        "raw_dgp_identity_sha256": sha256_json_value(&raw_identity)?,
        "latent_history_sha256": numeric_digest("latent-phase-history-v4", &latent_values)?,
        "estimate_sha256": zero,
        "predicted_covariance_sha256": zero,
        "date_axis_sha256": sha256_json_value(&geometry.date_axis)?,
        "generator_hash": sha256_json_value(generator)?,
        "config_hash": sha256_json_value(generator)?,
        "source_model_hash": sha256_json_value(source_model)?,
    })
    .as_object()
    .cloned()
    .expect("empty-support identity is an object");
    let disposition = serde_json::json!({
        "target_coordinate": [geometry.target.0, geometry.target.1],
        "reference_coordinate": [geometry.reference.0, geometry.reference.1],
        "target_support_sha256": support_sha256,
        "reference_support_sha256": support_sha256,
        "target_source_count": 0,
        "reference_source_count": 0,
        "intersection_source_count": 0,
        "union_source_count": 0,
        "realized_overlap_jaccard": 0.0,
        "signed_cross_influence": Value::Null,
        "signed_influence_sign": pair_sign(&request.pair_geometry)?,
        "effective_looks_fraction": Value::Null,
        "effective_looks_application": "source_influence_joint_contraction_v1",
        "source_correlation_model": if request.source_process == "independent_complex_looks" { "identity_v1" } else { "exponential_euclidean_v1" },
        "source_correlation_distance_scale_pixels": if request.source_process == "independent_complex_looks" { 0.0 } else { 1.5 },
        "estimator_branch": request.estimator,
        "target_estimate_history": Value::Null,
        "reference_estimate_history": Value::Null,
        "predicted_difference_covariance": Value::Null,
        "production_operator_matrix": Value::Null,
        "contrast_weights": Value::Null,
        "operator_sha256": zero,
    });
    evidence.extend(
        disposition
            .as_object()
            .expect("empty-support disposition is an object")
            .clone(),
    );
    Ok(Value::Object(evidence))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_nondifferentiable_evidence(
    preregistration: &Value,
    tables: &PortableDgpTables,
    request: &FrozenAttemptRequest,
    dgp_ordinal: u64,
    geometry: &FrozenCellGeometry,
    raw: &BTreeMap<(i64, i64), Vec<Cf64>>,
) -> Result<Value> {
    anyhow::ensure!(
        request.support == "rect",
        "nondifferentiable support reconstruction is only defined for rectangular support"
    );
    let blocks = geometry
        .topology
        .get("expected_blocks")
        .and_then(Value::as_array)
        .context("topology omits expected blocks")?;
    let target_phase = inward_clamped_support(
        geometry.target,
        geometry.half_window,
        geometry.native_tile_shape,
    )?
    .into_iter()
    .collect::<std::collections::BTreeSet<_>>();
    let reference_phase = inward_clamped_support(
        geometry.reference,
        geometry.half_window,
        geometry.native_tile_shape,
    )?
    .into_iter()
    .collect::<std::collections::BTreeSet<_>>();
    let target_support_by_block = vec![target_phase.clone(); blocks.len()];
    let reference_support_by_block = vec![reference_phase.clone(); blocks.len()];
    let target_receipt = support_halo_receipt(blocks, &target_support_by_block)?;
    let reference_receipt = support_halo_receipt(blocks, &reference_support_by_block)?;
    let target_support_sha256 = sha256_json_value(&target_receipt)?;
    let reference_support_sha256 = sha256_json_value(&reference_receipt)?;
    let target_halo_by_block = target_support_by_block
        .iter()
        .map(|support| factor_halo(support.iter().copied(), geometry))
        .collect::<Result<Vec<_>>>()?;
    let reference_halo_by_block = reference_support_by_block
        .iter()
        .map(|support| factor_halo(support.iter().copied(), geometry))
        .collect::<Result<Vec<_>>>()?;
    let target_halo_receipt = support_halo_receipt(blocks, &target_halo_by_block)?;
    let reference_halo_receipt = support_halo_receipt(blocks, &reference_halo_by_block)?;
    let target_halo = target_halo_by_block
        .iter()
        .flatten()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let reference_halo = reference_halo_by_block
        .iter()
        .flatten()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let raw_union = raw
        .keys()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let intersection = target_phase.intersection(&reference_phase).count();
    let union = target_phase.union(&reference_phase).count();
    anyhow::ensure!(union > 0, "nondifferentiable production support is empty");
    let mut effective_support = target_phase
        .union(&reference_phase)
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    for _ in 1..blocks.len() {
        effective_support = factor_halo(effective_support.iter().copied(), geometry)?;
    }
    let effective_fraction = if request.source_process == "independent_complex_looks" {
        1.0
    } else {
        effective_looks_fraction(&effective_support)
    };
    let negative = request.pair_geometry.ends_with("_negative");
    let loading = |coordinate: (i64, i64)| {
        if !negative
            || squared_distance(coordinate, geometry.target)
                <= squared_distance(coordinate, geometry.reference)
        {
            1.0
        } else {
            -1.0
        }
    };
    let target_loading =
        target_phase.iter().copied().map(&loading).sum::<f64>() / target_phase.len() as f64;
    let reference_loading =
        reference_phase.iter().copied().map(&loading).sum::<f64>() / reference_phase.len() as f64;
    let sign = pair_sign(&request.pair_geometry)?;
    let signed_cross_influence = if sign == "positive" || sign == "negative" {
        target_loading * reference_loading
    } else {
        0.0
    };
    let latent_values = tables
        .latent_history(geometry.target, geometry.dates)?
        .into_iter()
        .chain(tables.latent_history(geometry.reference, geometry.dates)?)
        .collect::<Vec<_>>();
    let ancestry = serde_json::json!({
        "date_axis": geometry.date_axis,
        "expected_blocks": blocks,
        "max_num_compressed": geometry.max_num_compressed,
        "partial_tail_count": geometry.topology.get("partial_tail_count").context("topology omits partial tail")?,
    });
    let ancestry_sha256 = sha256_json_value(&ancestry)?;
    let raw_shape = serde_json::json!([raw_union.len(), geometry.dates, 2]);
    let raw_input_sha256 = raw_source_digest("raw-input-v4", raw_union.iter().copied(), raw)?;
    let target_raw_input_sha256 =
        raw_source_digest("source-raw-input-v4", target_halo.iter().copied(), raw)?;
    let reference_raw_input_sha256 =
        raw_source_digest("source-raw-input-v4", reference_halo.iter().copied(), raw)?;
    let raw_identity = serde_json::json!({
        "cell_id": request.cell_id,
        "dgp_cell_ordinal": dgp_ordinal,
        "seed_index": request.seed_index,
        "dgp_generator_identity_sha256": preregistration.pointer("/determinism/dgp_generator_identity_sha256").context("preregistration omits DGP generator identity")?,
        "shape": raw_shape,
        "target_coordinate": [geometry.target.0, geometry.target.1],
        "reference_coordinate": [geometry.reference.0, geometry.reference.1],
        "target_support_sha256": target_support_sha256,
        "reference_support_sha256": reference_support_sha256,
        "target_factor_support_sha256": sha256_json_value(&target_halo_receipt)?,
        "reference_factor_support_sha256": sha256_json_value(&reference_halo_receipt)?,
        "raw_input_sha256": raw_input_sha256,
        "target_raw_input_sha256": target_raw_input_sha256,
        "reference_raw_input_sha256": reference_raw_input_sha256,
        "sequential_ancestry_sha256": ancestry_sha256,
        "estimator": request.estimator,
        "eigen_stress": request.eigen_stress,
        "source_process": request.source_process,
    });
    let generator = preregistration
        .get("generator")
        .context("preregistration omits generator")?;
    let source_model = generator
        .get("source_centered_empirical")
        .context("generator omits source model")?;
    let zero = "0".repeat(64);
    let mut evidence = serde_json::json!({
        "schema": "dolphinrust.spatial-covariance.attempt-evidence/4",
        "cell_id": request.cell_id,
        "cell_ordinal": request.cell_ordinal,
        "seed_index": request.seed_index,
        "seed_sha256": request.seed_sha256,
        "status": "nondifferentiable_node",
        "emitted": false,
        "factor_emitted": false,
        "raw_input_shape": raw_shape,
        "raw_input_value_count": 2 * raw_union.len() * geometry.dates,
        "raw_input_sha256": raw_input_sha256,
        "target_raw_input_sha256": target_raw_input_sha256,
        "reference_raw_input_sha256": reference_raw_input_sha256,
        "sequential_ancestry_sha256": ancestry_sha256,
        "raw_dgp_identity_sha256": sha256_json_value(&raw_identity)?,
        "latent_history_sha256": numeric_digest("latent-phase-history-v4", &latent_values)?,
        "estimate_sha256": zero,
        "predicted_covariance_sha256": zero,
        "date_axis_sha256": sha256_json_value(&geometry.date_axis)?,
        "generator_hash": sha256_json_value(generator)?,
        "config_hash": sha256_json_value(generator)?,
        "source_model_hash": sha256_json_value(source_model)?,
    })
    .as_object()
    .cloned()
    .expect("nondifferentiable evidence identity is an object");
    let disposition = serde_json::json!({
        "target_coordinate": [geometry.target.0, geometry.target.1],
        "reference_coordinate": [geometry.reference.0, geometry.reference.1],
        "target_support_sha256": target_support_sha256,
        "reference_support_sha256": reference_support_sha256,
        "target_source_count": target_phase.len(),
        "reference_source_count": reference_phase.len(),
        "intersection_source_count": intersection,
        "union_source_count": union,
        "realized_overlap_jaccard": intersection as f64 / union as f64,
        "signed_cross_influence": signed_cross_influence,
        "signed_influence_sign": sign,
        "effective_looks_fraction": effective_fraction,
        "effective_looks_application": "source_influence_joint_contraction_v1",
        "source_correlation_model": if request.source_process == "independent_complex_looks" { "identity_v1" } else { "exponential_euclidean_v1" },
        "source_correlation_distance_scale_pixels": if request.source_process == "independent_complex_looks" { 0.0 } else { 1.5 },
        "estimator_branch": request.estimator,
        "target_estimate_history": Value::Null,
        "reference_estimate_history": Value::Null,
        "predicted_difference_covariance": Value::Null,
        "production_operator_matrix": Value::Null,
        "contrast_weights": Value::Null,
        "operator_sha256": zero,
    });
    evidence.extend(
        disposition
            .as_object()
            .expect("nondifferentiable disposition is an object")
            .clone(),
    );
    Ok(Value::Object(evidence))
}

/// Regenerate and execute one frozen v4 attempt through the actual production path.
///
/// # Errors
/// Returns an error for scope drift, unsupported coordinates, missing portable
/// table coverage, production fail-close, or evidence/hash inconsistency.
#[allow(clippy::too_many_lines)]
pub fn run_frozen_attempt(
    preregistration: &Value,
    tables: &PortableDgpTables,
    request: &FrozenAttemptRequest,
) -> Result<Value> {
    let dgp_ordinal = request.validate(preregistration)?;
    if request.eigen_stress == "tied_eigenvalue" && request.position != "masked" {
        return build_tied_probe_evidence(preregistration, request);
    }
    let geometry = FrozenCellGeometry::parse(preregistration, request)?;
    let target_candidates = inward_clamped_support(
        geometry.target,
        geometry.half_window,
        geometry.native_tile_shape,
    )?;
    let reference_candidates = inward_clamped_support(
        geometry.reference,
        geometry.half_window,
        geometry.native_tile_shape,
    )?;
    let possible_phase = target_candidates
        .iter()
        .chain(&reference_candidates)
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let block_count = geometry
        .topology
        .get("expected_blocks")
        .and_then(Value::as_array)
        .context("topology omits expected blocks")?
        .len();
    let mut possible_halo = possible_phase.clone();
    for _ in 0..block_count {
        possible_halo = factor_halo(possible_halo.iter().copied(), &geometry)?;
    }
    let first = possible_halo
        .first()
        .copied()
        .context("frozen production crop is empty")?;
    let last = possible_halo
        .last()
        .copied()
        .context("frozen production crop is empty")?;
    let crop_start = (
        (first.0 as usize / geometry.stride.0) * geometry.stride.0,
        (possible_halo.iter().map(|value| value.1).min().unwrap() as usize / geometry.stride.1)
            * geometry.stride.1,
    );
    let maximum_column = possible_halo.iter().map(|value| value.1).max().unwrap() as usize;
    let crop_stop = (
        ((last.0 as usize + 1).div_ceil(geometry.stride.0)) * geometry.stride.0,
        ((maximum_column + 1).div_ceil(geometry.stride.1)) * geometry.stride.1,
    );
    let crop_shape = (crop_stop.0 - crop_start.0, crop_stop.1 - crop_start.1);
    anyhow::ensure!(
        crop_shape.0.is_multiple_of(geometry.stride.0)
            && crop_shape.1.is_multiple_of(geometry.stride.1),
        "frozen production crop is not stride aligned"
    );
    let spatial = request.source_process == "spatial_correlation_stress"
        || request.pair_geometry.ends_with("_positive")
        || request.pair_geometry.ends_with("_negative");
    let negative = request.pair_geometry.ends_with("_negative");
    let loading = |coordinate: (i64, i64)| {
        if !negative
            || squared_distance(coordinate, geometry.target)
                <= squared_distance(coordinate, geometry.reference)
        {
            1.0
        } else {
            -1.0
        }
    };
    let mut raw = BTreeMap::new();
    for &coordinate in &possible_halo {
        raw.insert(
            coordinate,
            tables.source_history(
                dgp_ordinal,
                request.seed_index,
                coordinate,
                geometry.dates,
                spatial,
                &request.eigen_stress,
                loading(coordinate),
            )?,
        );
    }
    let mut stack = Array3::from_elem(
        (geometry.dates, crop_shape.0, crop_shape.1),
        Cf64::new(0.0, 0.0),
    );
    let mut validity = Array2::from_elem(crop_shape, false);
    for (&(row, column), values) in &raw {
        let local = (row as usize - crop_start.0, column as usize - crop_start.1);
        validity[local] = true;
        for (date, &value) in values.iter().enumerate() {
            stack[(date, local.0, local.1)] = value;
        }
    }
    let unmasked_validity = validity.clone();
    if request.position == "masked" {
        let target_output = (
            native_center_to_output(geometry.target.0, geometry.stride.0)?,
            native_center_to_output(geometry.target.1, geometry.stride.1)?,
        );
        let looked_start = (
            target_output
                .0
                .checked_mul(geometry.stride.0)
                .context("masked target row overflows")?,
            target_output
                .1
                .checked_mul(geometry.stride.1)
                .context("masked target column overflows")?,
        );
        for row in looked_start.0..looked_start.0 + geometry.stride.0 {
            for column in looked_start.1..looked_start.1 + geometry.stride.1 {
                if row >= crop_start.0
                    && row < crop_stop.0
                    && column >= crop_start.1
                    && column < crop_stop.1
                {
                    validity[(row - crop_start.0, column - crop_start.1)] = false;
                }
            }
        }
    }
    let source_model_value = preregistration
        .pointer("/generator/source_centered_empirical")
        .context("preregistration omits empirical source model")?;
    let source_model_identity: [u8; 32] =
        Sha256::digest(serde_json::to_vec(source_model_value)?).into();
    let source_model = EmpiricalProperComplexConfig::new(
        geometry.half_window.0,
        geometry.half_window.1,
        0.05,
        1e-12,
        source_model_identity,
    )?;
    let source_manifest_digest: [u8; 32] = Sha256::digest(raw_source_bytes(
        b"dolphinrust:spatial-covariance-production-crop:v1",
        possible_halo.iter().copied(),
        &raw,
    )?)
    .into();
    let mut config = validation_config();
    config.ministack_size = geometry.ministack_size;
    config.max_num_compressed = geometry.max_num_compressed;
    config.half_window = dolphin_core::HalfWindow {
        y: geometry.half_window.0,
        x: geometry.half_window.1,
    };
    config.strides = dolphin_core::Strides {
        y: geometry.stride.0,
        x: geometry.stride.1,
    };
    config.use_evd = request.estimator == "evd";
    config.shp_method = match request.support.as_str() {
        "rect" => ShpMethod::Rect,
        "glrt_frozen" => ShpMethod::Glrt,
        "ks_frozen" => ShpMethod::Ks,
        _ => anyhow::bail!("unsupported frozen SHP method"),
    };
    config.shp_alpha = 0.001;
    let output_shape = (
        crop_shape.0 / geometry.stride.0,
        crop_shape.1 / geometry.stride.1,
    );
    let output_origin = (
        crop_start.0 / geometry.stride.0,
        crop_start.1 / geometry.stride.1,
    );
    let grid = CovarianceOperatorGrid {
        row_start: u64::try_from(output_origin.0)?,
        col_start: u64::try_from(output_origin.1)?,
        rows: u32::try_from(output_shape.0)?,
        cols: u32::try_from(output_shape.1)?,
        stride_y: u32::try_from(geometry.stride.0)?,
        stride_x: u32::try_from(geometry.stride.1)?,
    };
    let capture = SequentialCovarianceCaptureRequest {
        burst_id: "spatial-covariance-f54-07-v4".to_owned(),
        source_manifest_digest,
        source_model_version_digest: source_model_identity,
        native_grid: CovarianceOperatorGrid {
            row_start: u64::try_from(crop_start.0)?,
            col_start: u64::try_from(crop_start.1)?,
            rows: u32::try_from(crop_shape.0)?,
            cols: u32::try_from(crop_shape.1)?,
            stride_y: 1,
            stride_x: 1,
        },
        output_grid: grid,
        owned_output_grid: grid,
        branch_tolerance: 1e-10,
    };
    let target = (
        u64::try_from(native_center_to_output(
            geometry.target.0,
            geometry.stride.0,
        )?)?,
        u64::try_from(native_center_to_output(
            geometry.target.1,
            geometry.stride.1,
        )?)?,
    );
    let reference = (
        u64::try_from(native_center_to_output(
            geometry.reference.0,
            geometry.stride.0,
        )?)?,
        u64::try_from(native_center_to_output(
            geometry.reference.1,
            geometry.stride.1,
        )?)?,
    );
    let make_inputs = |attempt_validity: Array2<bool>| ProductionCellInputs {
        stack: stack.clone(),
        validity: attempt_validity,
        config,
        capture: capture.clone(),
        target,
        reference,
        source_model: source_model.clone(),
        source_correlation: if request.source_process == "independent_complex_looks" {
            SourceCorrelationModel::Identity
        } else {
            SourceCorrelationModel::ExponentialEuclidean {
                distance_scale_pixels: 1.5,
            }
        },
        data_identity: source_manifest_digest,
    };
    let production = match run_production_cell(make_inputs(unmasked_validity)) {
        Ok(production) => production,
        Err(error)
            if error
                .downcast_ref::<SequentialReplayError>()
                .is_some_and(|replay| replay.status() == ReplayStatus::MaskedNode) =>
        {
            if request.position == "masked" {
                let masked_error = run_production_cell(make_inputs(validity))
                    .expect_err("masked empty-support attempt unexpectedly emitted covariance");
                anyhow::ensure!(
                    masked_error
                        .downcast_ref::<SequentialReplayError>()
                        .is_some_and(|replay| replay.status() == ReplayStatus::MaskedNode),
                    "masked empty-support attempt did not preserve its production status"
                );
                return build_empty_support_evidence(
                    preregistration,
                    tables,
                    request,
                    dgp_ordinal,
                    &geometry,
                    &raw,
                    "masked_target",
                );
            }
            anyhow::ensure!(
                request.support == "ks_frozen",
                "undeclared production empty-support status for {}",
                request.cell_id
            );
            return build_empty_support_evidence(
                preregistration,
                tables,
                request,
                dgp_ordinal,
                &geometry,
                &raw,
                "empty_support",
            );
        }
        Err(error) => {
            if error
                .downcast_ref::<SequentialReplayError>()
                .is_some_and(|replay| replay.status() == ReplayStatus::NondifferentiableNode)
            {
                return build_nondifferentiable_evidence(
                    preregistration,
                    tables,
                    request,
                    dgp_ordinal,
                    &geometry,
                    &raw,
                );
            }
            return Err(
                if let Some(replay) = error.downcast_ref::<SequentialReplayError>() {
                    anyhow::anyhow!(
                        "frozen production cell {} failed with {}: {error}",
                        request.cell_id,
                        replay.status().as_str()
                    )
                } else {
                    error.context(format!("frozen production cell {} failed", request.cell_id))
                },
            );
        }
    };
    let mut evidence = build_frozen_attempt_evidence(
        preregistration,
        tables,
        request,
        dgp_ordinal,
        &geometry,
        &raw,
        loading,
        production,
    )?;
    if request.position == "masked" {
        let error = run_production_cell(make_inputs(validity))
            .expect_err("masked production attempt unexpectedly emitted covariance");
        let replay = error
            .downcast_ref::<SequentialReplayError>()
            .context("masked production attempt did not preserve its replay status")?;
        anyhow::ensure!(
            replay.status() == ReplayStatus::MaskedNode,
            "masked production attempt returned {}",
            replay.status().as_str()
        );
        let object = evidence
            .as_object_mut()
            .expect("frozen attempt evidence is an object");
        object.insert(
            "status".to_owned(),
            Value::String("masked_target".to_owned()),
        );
        object.insert("emitted".to_owned(), Value::Bool(false));
        object.insert("factor_emitted".to_owned(), Value::Bool(false));
        for name in [
            "signed_cross_influence",
            "target_estimate_history",
            "reference_estimate_history",
            "predicted_difference_covariance",
            "production_operator_matrix",
            "contrast_weights",
        ] {
            object.insert(name.to_owned(), Value::Null);
        }
        for name in [
            "estimate_sha256",
            "predicted_covariance_sha256",
            "operator_sha256",
        ] {
            object.insert(name.to_owned(), Value::String("0".repeat(64)));
        }
    }
    Ok(evidence)
}

fn squared_distance(left: (i64, i64), right: (i64, i64)) -> i128 {
    let row = i128::from(left.0 - right.0);
    let column = i128::from(left.1 - right.1);
    row * row + column * column
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_frozen_attempt_evidence<F>(
    preregistration: &Value,
    tables: &PortableDgpTables,
    request: &FrozenAttemptRequest,
    dgp_ordinal: u64,
    geometry: &FrozenCellGeometry,
    raw: &BTreeMap<(i64, i64), Vec<Cf64>>,
    loading: F,
    production: ProductionCellResult,
) -> Result<Value>
where
    F: Fn((i64, i64)) -> f64,
{
    let target_phase = production
        .target_support
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let reference_phase = production
        .reference_support
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let target_halo_by_block = production
        .target_support_by_block
        .iter()
        .map(|support| factor_halo(support.iter().copied(), geometry))
        .collect::<Result<Vec<_>>>()?;
    let reference_halo_by_block = production
        .reference_support_by_block
        .iter()
        .map(|support| factor_halo(support.iter().copied(), geometry))
        .collect::<Result<Vec<_>>>()?;
    let target_halo = target_halo_by_block
        .iter()
        .flatten()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let reference_halo = reference_halo_by_block
        .iter()
        .flatten()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let raw_union = raw
        .keys()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    anyhow::ensure!(
        raw_union
            .iter()
            .all(|coordinate| raw.contains_key(coordinate)),
        "production source-factor halo escaped the portable raw crop"
    );
    let intersection = target_phase.intersection(&reference_phase).count();
    let union = target_phase.union(&reference_phase).count();
    anyhow::ensure!(union > 0, "production phase support union is empty");
    let effective_support = production
        .effective_support
        .iter()
        .map(|&(row, column)| Ok((i64::try_from(row)?, i64::try_from(column)?)))
        .collect::<Result<std::collections::BTreeSet<_>>>()?;
    anyhow::ensure!(
        production.support_union_count == effective_support.len() as u64,
        "production effective-look support count {} differs from the transitive replay support {}",
        production.support_union_count,
        effective_support.len()
    );
    let (expected_correlation_model, expected_correlation_scale, expected_fraction) =
        if request.source_process == "independent_complex_looks" {
            ("identity_v1", 0.0, 1.0)
        } else {
            (
                "exponential_euclidean_v1",
                1.5,
                effective_looks_fraction(&effective_support),
            )
        };
    anyhow::ensure!(
        production.source_correlation_model == expected_correlation_model
            && production.source_correlation_distance_scale_pixels == expected_correlation_scale
            && (production.effective_looks_fraction - expected_fraction).abs() <= 1e-15,
        "production source-correlation realization differs from the frozen source process"
    );
    let target_loading =
        target_phase.iter().copied().map(&loading).sum::<f64>() / target_phase.len() as f64;
    let reference_loading =
        reference_phase.iter().copied().map(&loading).sum::<f64>() / reference_phase.len() as f64;
    let sign = pair_sign(&request.pair_geometry)?;
    let signed_cross_influence = if sign == "positive" || sign == "negative" {
        target_loading * reference_loading
    } else {
        0.0
    };
    let latent_target = tables.latent_history(geometry.target, geometry.dates)?;
    let latent_reference = if geometry.target == geometry.reference {
        latent_target.clone()
    } else {
        tables.latent_history(geometry.reference, geometry.dates)?
    };
    let latent_values = latent_target
        .iter()
        .chain(&latent_reference)
        .copied()
        .collect::<Vec<_>>();
    let generator = preregistration
        .get("generator")
        .context("preregistration omits generator")?;
    let source_model = generator
        .get("source_centered_empirical")
        .context("generator omits source model")?;
    let ancestry = serde_json::json!({
        "date_axis": geometry.date_axis,
        "expected_blocks": geometry.topology.get("expected_blocks").context("topology omits blocks")?,
        "max_num_compressed": geometry.max_num_compressed,
        "partial_tail_count": geometry.topology.get("partial_tail_count").context("topology omits partial tail")?,
    });
    let ancestry_sha256 = sha256_json_value(&ancestry)?;
    let target_halo_receipt = support_halo_receipt(
        geometry
            .topology
            .get("expected_blocks")
            .and_then(Value::as_array)
            .context("topology omits expected blocks")?,
        &target_halo_by_block,
    )?;
    let reference_halo_receipt = support_halo_receipt(
        geometry
            .topology
            .get("expected_blocks")
            .and_then(Value::as_array)
            .context("topology omits expected blocks")?,
        &reference_halo_by_block,
    )?;
    let raw_shape = serde_json::json!([raw_union.len(), geometry.dates, 2]);
    let raw_identity = serde_json::json!({
        "cell_id": request.cell_id,
        "dgp_cell_ordinal": dgp_ordinal,
        "seed_index": request.seed_index,
        "dgp_generator_identity_sha256": preregistration.pointer("/determinism/dgp_generator_identity_sha256").context("preregistration omits DGP generator identity")?,
        "shape": raw_shape,
        "target_coordinate": [geometry.target.0, geometry.target.1],
        "reference_coordinate": [geometry.reference.0, geometry.reference.1],
        "target_support_sha256": production.target_support_sha256,
        "reference_support_sha256": production.reference_support_sha256,
        "target_factor_support_sha256": sha256_json_value(&target_halo_receipt)?,
        "reference_factor_support_sha256": sha256_json_value(&reference_halo_receipt)?,
        "raw_input_sha256": raw_source_digest("raw-input-v4", raw_union.iter().copied(), raw)?,
        "target_raw_input_sha256": raw_source_digest("source-raw-input-v4", target_halo.iter().copied(), raw)?,
        "reference_raw_input_sha256": raw_source_digest("source-raw-input-v4", reference_halo.iter().copied(), raw)?,
        "sequential_ancestry_sha256": ancestry_sha256,
        "estimator": request.estimator,
        "eigen_stress": request.eigen_stress,
        "source_process": request.source_process,
    });
    let target_values = production.target_estimate_history.clone();
    let reference_values = production.reference_estimate_history.clone();
    let estimate_values = target_values
        .iter()
        .chain(&reference_values)
        .copied()
        .collect::<Vec<_>>();
    let covariance_values = production
        .predicted_difference_covariance
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let operator_values = production
        .production_operator_matrix
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let mut evidence = serde_json::json!({
        "schema": "dolphinrust.spatial-covariance.attempt-evidence/4",
        "cell_id": request.cell_id,
        "cell_ordinal": request.cell_ordinal,
        "seed_index": request.seed_index,
        "seed_sha256": request.seed_sha256,
        "status": production.status,
        "emitted": true,
        "factor_emitted": true,
        "raw_input_sha256": raw_source_digest("raw-input-v4", raw_union.iter().copied(), raw)?,
        "latent_history_sha256": numeric_digest("latent-phase-history-v4", &latent_values)?,
        "estimate_sha256": numeric_digest("estimate-history-v4", &estimate_values)?,
        "predicted_covariance_sha256": numeric_digest("predicted-difference-covariance-v4", &covariance_values)?,
        "date_axis_sha256": sha256_json_value(&geometry.date_axis)?,
        "generator_hash": sha256_json_value(generator)?,
        "config_hash": sha256_json_value(generator)?,
        "source_model_hash": sha256_json_value(source_model)?,
        "target_coordinate": [geometry.target.0, geometry.target.1],
        "reference_coordinate": [geometry.reference.0, geometry.reference.1],
        "target_support_sha256": production.target_support_sha256,
        "reference_support_sha256": production.reference_support_sha256,
        "target_source_count": target_phase.len(),
        "reference_source_count": reference_phase.len(),
        "intersection_source_count": intersection,
        "union_source_count": union,
        "realized_overlap_jaccard": intersection as f64 / union as f64,
        "signed_cross_influence": signed_cross_influence,
        "signed_influence_sign": sign,
        "effective_looks_fraction": production.effective_looks_fraction,
        "effective_looks_application": "source_influence_joint_contraction_v1",
        "source_correlation_model": production.source_correlation_model,
        "source_correlation_distance_scale_pixels": production.source_correlation_distance_scale_pixels,
    })
    .as_object()
    .cloned()
    .expect("attempt evidence literal is an object");
    let numeric = serde_json::json!({
        "estimator_branch": request.estimator,
        "target_estimate_history": target_values,
        "reference_estimate_history": reference_values,
        "predicted_difference_covariance": production.predicted_difference_covariance,
        "production_operator_matrix": production.production_operator_matrix,
        "contrast_weights": production.contrast_weights,
        "operator_sha256": numeric_digest("production-operator-v4", &operator_values)?,
        "raw_input_shape": raw_shape,
        "raw_input_value_count": 2 * raw_union.len() * geometry.dates,
        "target_raw_input_sha256": raw_source_digest("source-raw-input-v4", target_halo.iter().copied(), raw)?,
        "reference_raw_input_sha256": raw_source_digest("source-raw-input-v4", reference_halo.iter().copied(), raw)?,
        "sequential_ancestry_sha256": ancestry_sha256,
        "raw_dgp_identity_sha256": sha256_json_value(&raw_identity)?,
    });
    evidence.extend(
        numeric
            .as_object()
            .expect("attempt numeric evidence literal is an object")
            .clone(),
    );
    Ok(Value::Object(evidence))
}

fn support_halo_receipt(
    blocks: &[Value],
    supports: &[std::collections::BTreeSet<(i64, i64)>],
) -> Result<Value> {
    anyhow::ensure!(
        blocks.len() == supports.len(),
        "factor support receipt block count differs"
    );
    Ok(Value::Array(
        blocks
            .iter()
            .zip(supports)
            .map(|(block, support)| {
                Ok(serde_json::json!({
                    "block_id": block.get("block_id").and_then(Value::as_u64).context("expected block omits id")?,
                    "sources": support.iter().map(|&(row, column)| [row, column]).collect::<Vec<_>>(),
                }))
            })
            .collect::<Result<Vec<_>>>()?,
    ))
}

fn pair_sign(geometry: &str) -> Result<&'static str> {
    if geometry == "coincident" {
        Ok("zero")
    } else if geometry.ends_with("_positive") {
        Ok("positive")
    } else if geometry.ends_with("_negative") {
        Ok("negative")
    } else if geometry.starts_with("disjoint") {
        Ok("none")
    } else {
        anyhow::bail!("unsupported frozen pair geometry")
    }
}

fn effective_looks_fraction(support: &std::collections::BTreeSet<(i64, i64)>) -> f64 {
    let denominator = support
        .iter()
        .flat_map(|left| {
            support.iter().map(move |right| {
                let row = (left.0 - right.0) as f64;
                let column = (left.1 - right.1) as f64;
                (-(row.hypot(column)) / 1.5).exp()
            })
        })
        .sum::<f64>();
    support.len() as f64 / denominator
}

fn sha256_json_value(value: &Value) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(canonical_json_bytes(value)?)
    ))
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>> {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(values) => {
                let mut keys = values.keys().collect::<Vec<_>>();
                keys.sort_unstable();
                let mut output = serde_json::Map::new();
                for key in keys {
                    output.insert(key.clone(), sorted(&values[key]));
                }
                Value::Object(output)
            }
            Value::Array(values) => Value::Array(values.iter().map(sorted).collect()),
            _ => value.clone(),
        }
    }
    let encoded = serde_json::to_vec(&sorted(value))?;
    let mut output = Vec::with_capacity(encoded.len() + 8);
    let mut in_string = false;
    let mut escaped = false;
    let mut index = 0;
    while index < encoded.len() {
        let byte = encoded[index];
        if in_string {
            output.push(byte);
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if byte == b'"' {
            in_string = true;
        }
        if byte == b'e'
            && index + 3 < encoded.len()
            && matches!(encoded[index + 1], b'+' | b'-')
            && encoded[index + 2].is_ascii_digit()
            && !encoded[index + 3].is_ascii_digit()
        {
            output.extend_from_slice(&[byte, encoded[index + 1], b'0', encoded[index + 2]]);
            index += 3;
            continue;
        }
        output.push(byte);
        index += 1;
    }
    Ok(output)
}

fn numeric_digest(domain: &str, values: &[f64]) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(domain.as_bytes());
    digest.update((values.len() as u64).to_be_bytes());
    for &value in values {
        anyhow::ensure!(
            value.is_finite(),
            "numeric digest contains non-finite evidence"
        );
        let canonical = if value == 0.0 { 0.0 } else { value };
        digest.update(canonical.to_bits().to_be_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn raw_source_digest(
    domain: &str,
    support: impl IntoIterator<Item = (i64, i64)>,
    raw: &BTreeMap<(i64, i64), Vec<Cf64>>,
) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(raw_source_bytes(domain.as_bytes(), support, raw)?)
    ))
}

fn raw_source_bytes(
    domain: &[u8],
    support: impl IntoIterator<Item = (i64, i64)>,
    raw: &BTreeMap<(i64, i64), Vec<Cf64>>,
) -> Result<Vec<u8>> {
    let support = support.into_iter().collect::<Vec<_>>();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&(support.len() as u64).to_le_bytes());
    for coordinate in support {
        bytes.extend_from_slice(&coordinate.0.to_le_bytes());
        bytes.extend_from_slice(&coordinate.1.to_le_bytes());
        let values = raw
            .get(&coordinate)
            .context("raw source digest coordinate is absent")?;
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            anyhow::ensure!(
                value.is_finite(),
                "raw source digest contains non-finite evidence"
            );
            let real = if value.re == 0.0 {
                0.0
            } else {
                value.re as f32
            };
            let imaginary = if value.im == 0.0 {
                0.0
            } else {
                value.im as f32
            };
            bytes.extend_from_slice(&real.to_bits().to_le_bytes());
            bytes.extend_from_slice(&imaginary.to_bits().to_le_bytes());
        }
    }
    Ok(bytes)
}

/// Exact inputs for one bounded production-path validation attempt.
pub struct ProductionCellInputs {
    /// Portable raw complex stack for the bounded native crop.
    pub stack: Array3<Cf64>,
    /// Immutable validity mask for the bounded native crop.
    pub validity: Array2<bool>,
    /// Production sequential configuration selected by the frozen cell.
    pub config: SequentialConfig,
    /// Strong capture identity and global crop grids.
    pub capture: SequentialCovarianceCaptureRequest,
    /// Global looked-grid target coordinate.
    pub target: (u64, u64),
    /// Global looked-grid reference coordinate.
    pub reference: (u64, u64),
    /// Source-centered empirical factor configuration.
    pub source_model: EmpiricalProperComplexConfig,
    /// Exact primitive-source spatial correlation used by production replay.
    pub source_correlation: SourceCorrelationModel,
    /// Immutable raw DGP identity.
    pub data_identity: [u8; 32],
}

/// Actual sequential estimates and production covariance evidence for one attempt.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProductionCellResult {
    /// Stable production disposition.
    pub status: String,
    /// Actual EMI/EVD target history in acquisition order.
    pub target_estimate_history: Vec<f64>,
    /// Actual EMI/EVD reference history in acquisition order.
    pub reference_estimate_history: Vec<f64>,
    /// Actual production 2N phase-joint operator.
    pub production_operator_matrix: Vec<Vec<f64>>,
    /// Actual production fixed-L2 difference covariance.
    pub predicted_difference_covariance: Vec<Vec<f64>>,
    /// Exact last-date target-minus-reference contrast.
    pub contrast_weights: Vec<f64>,
    /// Actual production replay reservation.
    pub resource_high_water_bytes: u64,
    /// Exact provider-inclusive dependency-cone reservation.
    pub dependency_cone_bytes: u64,
    /// Retained per-source influence operator bytes.
    pub source_influence_bytes: u64,
    /// Query-local source-correlation factorization workspace bytes.
    pub source_correlation_workspace_bytes: u64,
    /// Peak provider source/factor payload retained during replay.
    pub source_cache_peak_bytes: u64,
    /// Exact source-factor aggregate receipt.
    pub source_factor_receipt: String,
    /// Exact effective-look fraction applied by production.
    pub effective_looks_fraction: f64,
    /// Exact primitive-source correlation model identity.
    pub source_correlation_model: String,
    /// Exact primitive-source correlation scale in native pixels.
    pub source_correlation_distance_scale_pixels: f64,
    /// Realized target/reference support union size.
    pub support_union_count: u64,
    /// Primitive source coordinates actually resolved by production replay.
    #[serde(skip_serializing)]
    pub effective_support: Vec<(u64, u64)>,
    /// Canonical production target-support receipt digest.
    pub target_support_sha256: String,
    /// Canonical production reference-support receipt digest.
    pub reference_support_sha256: String,
    /// Realized target support coordinates, retained only in process.
    #[serde(skip_serializing)]
    pub target_support: Vec<(i64, i64)>,
    /// Realized reference support coordinates, retained only in process.
    #[serde(skip_serializing)]
    pub reference_support: Vec<(i64, i64)>,
    /// Per-generation realized target support, retained only in process.
    #[serde(skip_serializing)]
    pub target_support_by_block: Vec<Vec<(i64, i64)>>,
    /// Per-generation realized reference support, retained only in process.
    #[serde(skip_serializing)]
    pub reference_support_by_block: Vec<Vec<(i64, i64)>>,
}

struct ValidationInMemoryProvider<'a> {
    identity: SequentialSourceProviderIdentity,
    topology: SequentialReplayTopology,
    blocks: BTreeMap<GlobalBlockId, CovarianceOperatorBlock>,
    stack: ArrayView3<'a, Cf64>,
    validity: ArrayView2<'a, bool>,
    supplied_origin: (usize, usize),
    native_origin: (usize, usize),
    native_shape: (usize, usize),
    source_model: EmpiricalProperComplexConfig,
    data_identity: [u8; 32],
    factor_receipts: BTreeMap<SourceId, [u8; 32]>,
    resolved_coordinates: std::collections::BTreeSet<(u64, u64)>,
}

impl ValidationInMemoryProvider<'_> {
    fn resolve(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError> {
        let columns = self.stack.dim().2;
        let row = native_index / columns;
        let column = native_index % columns;
        let global_row = self
            .supplied_origin
            .0
            .checked_add(row)
            .ok_or(SequentialReplayError::Invalid("native row overflows usize"))?;
        let global_column =
            self.supplied_origin
                .1
                .checked_add(column)
                .ok_or(SequentialReplayError::Invalid(
                    "native column overflows usize",
                ))?;
        self.resolved_coordinates.insert((
            u64::try_from(global_row)
                .map_err(|_| SequentialReplayError::Invalid("global source row exceeds u64"))?,
            u64::try_from(global_column)
                .map_err(|_| SequentialReplayError::Invalid("global source column exceeds u64"))?,
        ));
        let date_start = block.real_date_start.get() as usize;
        let date_stop = date_start + block.num_real_dates;
        let component_ids = (date_start..date_stop)
            .map(|date| date as u64)
            .collect::<Vec<_>>();
        let samples =
            Array1::from_iter((date_start..date_stop).map(|date| self.stack[(date, row, column)]));
        let mut content = Sha256::new();
        for sample in &samples {
            content.update(sample.re.to_bits().to_le_bytes());
            content.update(sample.im.to_bits().to_le_bytes());
        }
        let content_digest: [u8; 32] = content.finalize().into();
        let source =
            self.topology
                .source_id_for_content_digest(block.id, native_index, &content_digest)?;
        let selected = self.stack.slice(s![date_start..date_stop, .., ..]);
        let estimate = estimate_empirical_proper_complex_factor(
            source,
            &component_ids,
            selected,
            self.validity.view(),
            self.supplied_origin,
            self.native_origin,
            self.native_shape,
            (global_row, global_column),
            self.data_identity,
            &self.source_model,
        )
        .map_err(|_| {
            SequentialReplayError::Provider(
                ReplayStatus::SourceModelUnavailable,
                "validation empirical source factor is unavailable",
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

impl SequentialSourceReplayProvider for ValidationInMemoryProvider<'_> {
    fn identity(&self) -> &SequentialSourceProviderIdentity {
        &self.identity
    }

    fn maximum_resident_bytes(&self) -> u64 {
        let stack = self.stack.len() as u64 * std::mem::size_of::<Cf64>() as u64;
        let validity = self.validity.len() as u64;
        stack + validity
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
                "validation operator block is absent",
            ))?;
        let width = stored.phase_components.len();
        Ok(ResolvedPhaseReplay {
            id: NodeId::new(stored.phase_node_ids[output_index]),
            linked_phase: Array1::from_iter(
                stored.phase_angles[output_index * width..(output_index + 1) * width]
                    .iter()
                    .map(|&angle| Cf64::from_polar(1.0, angle)),
            ),
            selected_eigenvalue: stored.selected_eigenvalue[output_index],
            selected_eigengap: stored.eigen_gap[output_index],
            realized_support: {
                let bits = stored.support_bits_per_output as usize;
                let bytes = bits.div_ceil(8);
                let packed = &stored.support_bits[output_index * bytes..(output_index + 1) * bytes];
                (0..bits)
                    .map(|slot| packed[slot / 8] & (1 << (slot % 8)) != 0)
                    .collect()
            },
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
                "validation operator block is absent",
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

impl SequentialPrimitiveSourceResolver for ValidationInMemoryProvider<'_> {
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
                "validation source factor receipt is absent",
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

/// Signed target/reference coupling used by the frozen validation cohorts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationCoupling {
    /// Shared positive influence.
    Positive,
    /// Matched marginals with no shared influence.
    Independent,
    /// Shared negative influence.
    Negative,
    /// Exact target/reference identity.
    Coincident,
    /// Invalid reference fail-closed fixture.
    Invalid,
}

/// Numeric evidence emitted by the validation runner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialCovarianceValidationResult {
    /// Frozen evidence schema.
    pub schema: String,
    /// Cohort identity.
    pub coupling: ValidationCoupling,
    /// Deterministic seed identity.
    pub seed_index: u64,
    /// Production target status.
    pub status: String,
    /// Target marginal phase covariance, row-major.
    pub target_covariance: Vec<Vec<f64>>,
    /// Reference marginal phase covariance, row-major.
    pub reference_covariance: Vec<Vec<f64>>,
    /// Target/reference cross phase covariance, row-major.
    pub target_reference_covariance: Vec<Vec<f64>>,
    /// Production fixed-L2 target-minus-reference covariance, row-major.
    pub difference_covariance: Vec<Vec<f64>>,
    /// Trace of the production fixed-L2 difference covariance.
    pub difference_variance: f64,
    /// Digest of the in-memory production factor bytes.
    pub factor_digest: String,
    /// Digest of the factor re-read from the production HDF5 artifact.
    pub persisted_factor_digest: String,
    /// Production replay dependency-cone bytes.
    pub dependency_cone_bytes: u64,
    /// Retained per-source influence operator bytes.
    pub source_influence_bytes: u64,
    /// Query-local source-correlation factorization workspace bytes.
    pub source_correlation_workspace_bytes: u64,
    /// Exact primitive-source correlation model identity.
    pub source_correlation_model: String,
    /// Production source-cache peak bytes.
    pub source_cache_peak_bytes: u64,
    /// Production HDF5 bytes when a fixture was persisted.
    pub hdf5_bytes: u64,
    /// Production HDF5 digest when a fixture was persisted.
    pub hdf5_sha256: String,
    /// Run-root-relative production HDF5 path when persisted.
    pub hdf5_path: String,
    /// Production provenance sidecar digest when a fixture was persisted.
    pub sidecar_sha256: String,
    /// Run-root-relative production provenance sidecar path when persisted.
    pub sidecar_path: String,
    /// Bounded-run production HDF5 digest.
    pub bounded_hdf5_sha256: String,
    /// Run-root-relative bounded-run HDF5 path.
    pub bounded_hdf5_path: String,
    /// Bounded-run production sidecar digest.
    pub bounded_sidecar_sha256: String,
    /// Run-root-relative bounded-run sidecar path.
    pub bounded_sidecar_path: String,
    /// Whole-frame runtime resource receipt digest read from the artifact.
    pub runtime_resource_receipt_digest: String,
    /// Bounded-run runtime resource receipt digest read from the artifact.
    pub bounded_runtime_resource_receipt_digest: String,
    /// HDF5 schema version verified by the capped production reader.
    pub hdf5_schema_version: u16,
    /// Production artifact-manifest schema version.
    pub manifest_schema_version: u16,
}

/// Resource evidence produced by the actual replay and production admission code.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpatialCovarianceBenchmarkEvidence {
    /// Requested target count.
    pub tile_pixels: u64,
    /// Native pixels actually allocated and processed by production capture.
    pub processed_tile_pixels: u64,
    /// Exact native production-capture grid.
    pub capture_native_shape: [u64; 2],
    /// Requested acquisition count.
    pub date_count: u64,
    /// Actual sequential block count.
    pub block_count: u64,
    /// Maximum dependency depth in the planned block chain.
    pub maximum_dependency_depth: u64,
    /// Primitive sources resolved by the benchmark replay.
    pub reference_cone_sources: u64,
    /// Actual replay dependency-cone allocation.
    pub dependency_cone_bytes: u64,
    /// Wrapper-inclusive global replay reservation admitted by production.
    pub replay_reservation_bytes: u64,
    /// Retained per-source influence operator bytes.
    pub source_influence_bytes: u64,
    /// Query-local source-correlation factorization workspace bytes.
    pub source_correlation_workspace_bytes: u64,
    /// Exact primitive-source correlation model identity.
    pub source_correlation_model: String,
    /// Actual replay source-cache high water.
    pub source_cache_peak_bytes: u64,
    /// Actual dense covariance result bytes in the replay dependency cone.
    pub covariance_result_bytes: u64,
    /// Targets admitted in one production factor block.
    pub admitted_block_targets: u64,
    /// Exact production admission and observed-resource receipt.
    pub runtime_resource_receipt: BenchmarkRuntimeResourceReceipt,
}

/// Serializable mirror of the exact production runtime receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BenchmarkRuntimeResourceReceipt {
    /// Production aggregate working-set cap.
    pub working_set_byte_cap: u64,
    /// Largest admitted factor block.
    pub factor_block_high_water_bytes: u64,
    /// Largest serialization reservation.
    pub serialization_high_water_bytes: u64,
    /// Fixed-L2 workspace admitted by production.
    pub fixed_l2_workspace_admission_bytes: u64,
    /// Fixed-L2 workspace observed during numeric execution.
    pub fixed_l2_workspace_observed_high_water_bytes: u64,
    /// Replay reservation admitted by production.
    pub replay_admission_high_water_bytes: u64,
    /// Replay allocation observed during numeric execution.
    pub replay_observed_high_water_bytes: u64,
    /// Peak live provider count.
    pub provider_peak_count: u64,
    /// Peak live provider bytes.
    pub provider_peak_bytes: u64,
    /// Provider opens during preflight.
    pub preflight_provider_open_count: u64,
    /// Provider opens during numeric production replay.
    pub production_provider_open_count: u64,
    /// Operator block reads.
    pub operator_block_reads: u64,
    /// Operator block cache hits.
    pub operator_block_cache_hits: u64,
    /// Source-member window reads.
    pub source_member_window_reads: u64,
    /// Source tile-cache loads.
    pub source_tile_cache_loads: u64,
    /// Primitive source resolutions.
    pub source_resolutions: u64,
    /// Aggregate admitted high water.
    pub working_set_admission_high_water_bytes: u64,
    /// Aggregate observed high water.
    pub working_set_observed_high_water_bytes: u64,
}

impl From<SpatialReferenceRuntimeResourceReceipt> for BenchmarkRuntimeResourceReceipt {
    fn from(value: SpatialReferenceRuntimeResourceReceipt) -> Self {
        Self {
            working_set_byte_cap: value.working_set_byte_cap,
            factor_block_high_water_bytes: value.factor_block_high_water_bytes,
            serialization_high_water_bytes: value.serialization_high_water_bytes,
            fixed_l2_workspace_admission_bytes: value.fixed_l2_workspace_admission_bytes,
            fixed_l2_workspace_observed_high_water_bytes: value
                .fixed_l2_workspace_observed_high_water_bytes,
            replay_admission_high_water_bytes: value.replay_admission_high_water_bytes,
            replay_observed_high_water_bytes: value.replay_observed_high_water_bytes,
            provider_peak_count: value.provider_peak_count,
            provider_peak_bytes: value.provider_peak_bytes,
            preflight_provider_open_count: value.preflight_provider_open_count,
            production_provider_open_count: value.production_provider_open_count,
            operator_block_reads: value.operator_block_reads,
            operator_block_cache_hits: value.operator_block_cache_hits,
            source_member_window_reads: value.source_member_window_reads,
            source_tile_cache_loads: value.source_tile_cache_loads,
            source_resolutions: value.source_resolutions,
            working_set_admission_high_water_bytes: value.working_set_admission_high_water_bytes,
            working_set_observed_high_water_bytes: value.working_set_observed_high_water_bytes,
        }
    }
}

/// Execute one bounded validation attempt through actual sequential capture,
/// provider-bundle replay, and the production fixed-L2 propagation path.
///
/// # Errors
/// Returns a fail-closed production replay or fixed-L2 error when the cell is
/// unsupported, masked, tied, malformed, or above the byte cap.
#[allow(clippy::too_many_lines)]
pub fn run_production_cell(inputs: ProductionCellInputs) -> Result<ProductionCellResult> {
    let (dates, native_rows, native_columns) = inputs.stack.dim();
    anyhow::ensure!(
        dates > 1 && inputs.validity.dim() == (native_rows, native_columns),
        "production validation stack and validity dimensions disagree"
    );
    anyhow::ensure!(
        inputs.capture.native_grid.rows as usize == native_rows
            && inputs.capture.native_grid.cols as usize == native_columns,
        "production validation capture native grid differs from the raw crop"
    );
    let output_shape = (
        inputs.capture.output_grid.rows as usize,
        inputs.capture.output_grid.cols as usize,
    );
    let support_slots = inputs
        .config
        .half_window
        .y
        .checked_mul(2)
        .and_then(|rows| rows.checked_add(1))
        .and_then(|rows| {
            inputs
                .config
                .half_window
                .x
                .checked_mul(2)
                .and_then(|columns| columns.checked_add(1))
                .and_then(|columns| rows.checked_mul(columns))
        })
        .context("production validation support dimensions overflow")?;
    let namespace = ReplayIdNamespace {
        burst_id: inputs.capture.burst_id.clone(),
        source_manifest_digest: inputs.capture.source_manifest_digest,
        source_model_version_digest: inputs.capture.source_model_version_digest,
        native_origin: (
            inputs.capture.native_grid.row_start,
            inputs.capture.native_grid.col_start,
        ),
        output_origin: (
            inputs.capture.output_grid.row_start,
            inputs.capture.output_grid.col_start,
        ),
        owned_output_origin: (
            inputs.capture.owned_output_grid.row_start,
            inputs.capture.owned_output_grid.col_start,
        ),
        owned_output_shape: (
            inputs.capture.owned_output_grid.rows as usize,
            inputs.capture.owned_output_grid.cols as usize,
        ),
    };
    let topology = SequentialReplayTopology::plan_identified(
        dates,
        (native_rows, native_columns),
        output_shape,
        support_slots,
        inputs.validity.view(),
        &inputs.config,
        validation_scope(),
        namespace,
    )?;
    let identity = SequentialSourceProviderIdentity {
        source_manifest_digest: inputs.capture.source_manifest_digest,
        provider: "spatial-covariance-validation-memory".to_owned(),
        provider_version: "1".to_owned(),
        model: "source_centered_empirical_proper_complex_v1".to_owned(),
        model_version: "1".to_owned(),
        source_model_version_digest: inputs.capture.source_model_version_digest,
        source_model_hash: *inputs.source_model.config_digest(),
    };
    let supplied_origin = (
        usize::try_from(inputs.capture.native_grid.row_start)?,
        usize::try_from(inputs.capture.native_grid.col_start)?,
    );
    let stack = inputs.stack;
    let validity = inputs.validity;
    let mut provider = ValidationInMemoryProvider {
        identity,
        topology: topology.clone(),
        blocks: BTreeMap::new(),
        stack: stack.view(),
        validity: validity.view(),
        supplied_origin,
        native_origin: supplied_origin,
        native_shape: (native_rows, native_columns),
        source_model: inputs.source_model,
        data_identity: inputs.data_identity,
        factor_receipts: BTreeMap::new(),
        resolved_coordinates: std::collections::BTreeSet::new(),
    };
    let mut blocks = Vec::new();
    let engine = ComputeEngine::new(ComputeBackend::Cpu);
    let output = if validity.iter().all(|&valid| valid) {
        run_sequential_with_covariance_capture_and_source_factors(
            stack.view(),
            &inputs.config,
            &engine,
            &inputs.capture,
            &mut provider,
            |block| {
                blocks.push(block);
                Ok(())
            },
        )?
    } else {
        run_sequential_masked_with_covariance_capture_and_source_factors(
            stack.view(),
            validity.view(),
            &inputs.config,
            &engine,
            &inputs.capture,
            &mut provider,
            |block| {
                blocks.push(block);
                Ok(())
            },
        )?
    };
    let target_local = local_output_coordinate(inputs.target, &inputs.capture.output_grid)?;
    let reference_local = local_output_coordinate(inputs.reference, &inputs.capture.output_grid)?;
    let target_support_by_block = blocks
        .iter()
        .map(|block| realized_support(block, target_local, &inputs.config))
        .collect::<Result<Vec<_>>>()?;
    let reference_support_by_block = blocks
        .iter()
        .map(|block| realized_support(block, reference_local, &inputs.config))
        .collect::<Result<Vec<_>>>()?;
    let target_support = target_support_by_block
        .iter()
        .flatten()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let reference_support = reference_support_by_block
        .iter()
        .flatten()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let target_support_sha256 = support_receipt_digest(&blocks, &target_support_by_block)?;
    let reference_support_sha256 = support_receipt_digest(&blocks, &reference_support_by_block)?;
    provider.blocks = blocks
        .into_iter()
        .map(|block| (GlobalBlockId::new(block.block_id), block))
        .collect();
    provider.resolved_coordinates.clear();
    let ordered_dates = (0..dates)
        .map(|date| Ok(GlobalDateId::new(u32::try_from(date)?)))
        .collect::<Result<Vec<_>>>()?;
    let source_rank = topology
        .blocks()
        .iter()
        .map(|block| block.num_real_dates * 2)
        .max()
        .context("production validation topology has no blocks")?;
    let query = GlobalReferenceCovarianceQuery {
        burst_id: &inputs.capture.burst_id,
        target: inputs.target,
        reference: inputs.reference,
        ordered_dates: &ordered_dates,
        source_rank,
        source_correlation: inputs.source_correlation,
        byte_cap: FULL_CELL_BYTE_CAP,
        branch_tolerance: inputs.capture.branch_tolerance,
    };
    let estimate = {
        let mut tiles = [SequentialTileReplayProvider::new(&topology, &mut provider)];
        estimate_global_reference_difference_covariance_from_provider_bundle(&mut tiles, query)?
    };
    anyhow::ensure!(
        estimate.total_bytes <= FULL_CELL_BYTE_CAP,
        "production validation replay exceeds its byte cap"
    );
    let replay = {
        let mut tiles = [SequentialTileReplayProvider::new(&topology, &mut provider)];
        replay_global_reference_difference_covariance_from_provider_bundle(&mut tiles, query)?
    };
    anyhow::ensure!(
        replay.resource_high_water_bytes == estimate.total_bytes,
        "production validation replay differs from its preflight estimate"
    );
    let target_history = phase_history(&output.cpx_phase, target_local);
    let reference_history = phase_history(&output.cpx_phase, reference_local);
    let fixed_l2 = full_date_fixed_l2_inputs(&target_history, &reference_history)?;
    let propagated = fixed_l2
        .propagate_joint_phase_covariance((0, 0), (0, 1), replay.joint_phase_covariance.view())
        .map_err(anyhow::Error::new)?;
    let mut contrast_weights = vec![0.0; dates * 2];
    contrast_weights[dates - 1] = 1.0;
    contrast_weights[dates * 2 - 1] = -1.0;
    let effective_looks = replay
        .replay
        .effective_looks
        .as_ref()
        .context("production validation replay omitted effective-look evidence")?;
    anyhow::ensure!(
        provider.resolved_coordinates.len() == effective_looks.support_union_count,
        "production provider support count differs from its effective-look receipt"
    );
    Ok(ProductionCellResult {
        status: "valid".to_owned(),
        target_estimate_history: target_history,
        reference_estimate_history: reference_history,
        production_operator_matrix: matrix_rows(&replay.joint_phase_covariance),
        predicted_difference_covariance: matrix_rows(&propagated.date_covariance),
        contrast_weights,
        resource_high_water_bytes: replay.resource_high_water_bytes,
        dependency_cone_bytes: replay.replay.dependency_cone.total_bytes,
        source_influence_bytes: replay.replay.dependency_cone.source_influence_bytes,
        source_correlation_workspace_bytes: replay
            .replay
            .dependency_cone
            .source_correlation_workspace_bytes,
        source_cache_peak_bytes: replay.replay.source_cache_peak_bytes,
        source_factor_receipt: hex_bytes(&replay.replay.source_factor_receipt),
        effective_looks_fraction: effective_looks.fraction,
        source_correlation_model: effective_looks.model.to_owned(),
        source_correlation_distance_scale_pixels: effective_looks.distance_scale_pixels,
        support_union_count: u64::try_from(effective_looks.support_union_count)?,
        effective_support: provider.resolved_coordinates.into_iter().collect(),
        target_support_sha256,
        reference_support_sha256,
        target_support,
        reference_support,
        target_support_by_block,
        reference_support_by_block,
    })
}

fn realized_support(
    block: &CovarianceOperatorBlock,
    output: (usize, usize),
    config: &SequentialConfig,
) -> Result<Vec<(i64, i64)>> {
    let rows = block.native_grid.rows as usize;
    let columns = block.native_grid.cols as usize;
    let window_rows = 2 * config.half_window.y + 1;
    let window_columns = 2 * config.half_window.x + 1;
    anyhow::ensure!(
        rows >= window_rows && columns >= window_columns,
        "production validation crop is smaller than its phase support"
    );
    let output_global = (
        usize::try_from(block.output_grid.row_start)?
            .checked_add(output.0)
            .context("production validation global output row overflows")?,
        usize::try_from(block.output_grid.col_start)?
            .checked_add(output.1)
            .context("production validation global output column overflows")?,
    );
    let (row_start, column_start) = realized_support_window_origin(
        output_global,
        (
            usize::try_from(block.native_grid.row_start)?,
            usize::try_from(block.native_grid.col_start)?,
        ),
        (rows, columns),
        config,
    )?;
    let output_columns = block.output_grid.cols as usize;
    let output_index = output
        .0
        .checked_mul(output_columns)
        .and_then(|value| value.checked_add(output.1))
        .context("production validation output index overflows")?;
    let bits = block.support_bits_per_output as usize;
    let bytes = bits.div_ceil(8);
    let packed = block
        .support_bits
        .get(output_index * bytes..(output_index + 1) * bytes)
        .context("production validation support receipt is truncated")?;
    let native_row = i64::try_from(block.native_grid.row_start)?;
    let native_column = i64::try_from(block.native_grid.col_start)?;
    Ok((0..bits)
        .filter(|slot| packed[slot / 8] & (1 << (slot % 8)) != 0)
        .map(|slot| {
            (
                native_row + (row_start + slot / window_columns) as i64,
                native_column + (column_start + slot % window_columns) as i64,
            )
        })
        .collect())
}

fn realized_support_window_origin(
    output_global: (usize, usize),
    native_origin: (usize, usize),
    native_shape: (usize, usize),
    config: &SequentialConfig,
) -> Result<(usize, usize)> {
    let window = (2 * config.half_window.y + 1, 2 * config.half_window.x + 1);
    anyhow::ensure!(
        native_shape.0 >= window.0 && native_shape.1 >= window.1,
        "production validation crop is smaller than its phase support"
    );
    let center_global = (
        usize::try_from(output_to_native_center(output_global.0, config.strides.y)?)?,
        usize::try_from(output_to_native_center(output_global.1, config.strides.x)?)?,
    );
    let center_local = (
        center_global
            .0
            .checked_sub(native_origin.0)
            .context("production output center precedes native row origin")?,
        center_global
            .1
            .checked_sub(native_origin.1)
            .context("production output center precedes native column origin")?,
    );
    anyhow::ensure!(
        center_local.0 < native_shape.0 && center_local.1 < native_shape.1,
        "production output center is outside its native grid"
    );
    Ok((
        center_local
            .0
            .saturating_sub(config.half_window.y)
            .min(native_shape.0 - window.0),
        center_local
            .1
            .saturating_sub(config.half_window.x)
            .min(native_shape.1 - window.1),
    ))
}

fn support_receipt_digest(
    blocks: &[CovarianceOperatorBlock],
    supports: &[Vec<(i64, i64)>],
) -> Result<String> {
    let receipt = blocks
        .iter()
        .zip(supports)
        .map(|(block, support)| {
            serde_json::json!({
                "block_id": block.generation,
                "sources": support.iter().map(|&(row, column)| [row, column]).collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    let bytes = canonical_json_bytes(&serde_json::to_value(receipt)?)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn local_output_coordinate(
    global: (u64, u64),
    grid: &CovarianceOperatorGrid,
) -> Result<(usize, usize)> {
    let row = global
        .0
        .checked_sub(grid.row_start)
        .context("production validation output precedes row origin")?;
    let column = global
        .1
        .checked_sub(grid.col_start)
        .context("production validation output precedes column origin")?;
    anyhow::ensure!(
        row < u64::from(grid.rows) && column < u64::from(grid.cols),
        "production validation output is outside the bounded crop"
    );
    Ok((usize::try_from(row)?, usize::try_from(column)?))
}

fn phase_history(stack: &Array3<Cf64>, pixel: (usize, usize)) -> Vec<f64> {
    let mut history = stack
        .axis_iter(Axis(0))
        .map(|date| date[pixel].arg())
        .collect::<Vec<_>>();
    let acquisition_zero = history[0];
    for value in &mut history {
        *value -= acquisition_zero;
    }
    history[0] = 0.0;
    history
}

fn full_date_fixed_l2_inputs(target: &[f64], reference: &[f64]) -> Result<FixedL2WorkflowInputs> {
    anyhow::ensure!(
        target.len() == reference.len() && target.len() > 1,
        "production validation phase histories disagree"
    );
    let dates = target.len();
    let design = Array2::from_shape_fn((dates, dates - 1), |(date, increment)| {
        f64::from(increment < date)
    });
    let mut observations = Array3::zeros((dates, 1, 2));
    for date in 0..dates {
        observations[(date, 0, 0)] = target[date];
        observations[(date, 0, 1)] = reference[date];
    }
    FixedL2WorkflowInputs::new(
        design,
        observations,
        Some(Array3::from_elem((dates, 1, 2), 1.0)),
    )
}

/// Execute the production replay estimator, fixed-L2 workspace calculation,
/// block sizing, and aggregate resource admission for one benchmark point.
///
/// # Errors
/// Returns an error for invalid dimensions, replay failure, overflow, or a
/// production admission above the frozen working-set cap.
pub fn run_benchmark_preflight(
    tile_pixels: usize,
    dates: usize,
) -> Result<SpatialCovarianceBenchmarkEvidence> {
    anyhow::ensure!(
        tile_pixels > 0 && dates > 1,
        "benchmark dimensions must be positive with at least two dates"
    );
    let (replay, block_count, capture_shape) = benchmark_production_cell(tile_pixels, dates)?;
    let replay_reservation = replay.resource_high_water_bytes;
    let workspace = fixed_l2_difference_workspace_composition(dates).map_err(anyhow::Error::new)?;
    let block_shape = factor_block_shape(
        (1, tile_pixels),
        dates,
        PRODUCTION_REFERENCE_COVARIANCE_WORKING_SET_BYTE_CAP,
        workspace,
        replay_reservation,
    )?;
    let targets = block_shape
        .0
        .checked_mul(block_shape.1)
        .context("benchmark admitted block area overflows usize")?;
    let receipt = production_resource_admission(
        targets,
        dates,
        workspace,
        replay_reservation,
        PRODUCTION_REFERENCE_COVARIANCE_WORKING_SET_BYTE_CAP,
    )?;
    Ok(SpatialCovarianceBenchmarkEvidence {
        tile_pixels: u64::try_from(tile_pixels)?,
        processed_tile_pixels: u64::try_from(
            capture_shape
                .0
                .checked_mul(capture_shape.1)
                .context("benchmark production capture area overflows")?,
        )?,
        capture_native_shape: [
            u64::try_from(capture_shape.0)?,
            u64::try_from(capture_shape.1)?,
        ],
        date_count: u64::try_from(dates)?,
        block_count: u64::try_from(block_count)?,
        maximum_dependency_depth: u64::try_from(block_count.saturating_sub(1))?,
        reference_cone_sources: replay.support_union_count,
        dependency_cone_bytes: replay.dependency_cone_bytes,
        replay_reservation_bytes: replay.resource_high_water_bytes,
        source_influence_bytes: replay.source_influence_bytes,
        source_correlation_workspace_bytes: replay.source_correlation_workspace_bytes,
        source_correlation_model: replay.source_correlation_model,
        source_cache_peak_bytes: replay.source_cache_peak_bytes,
        covariance_result_bytes: u64::try_from(4 * dates * dates * size_of::<f64>())?,
        admitted_block_targets: u64::try_from(targets)?,
        runtime_resource_receipt: receipt.into(),
    })
}

/// Execute one deterministic cohort through replay, fixed-L2, and factor assembly.
///
/// # Errors
/// Returns an error when replay, fixed-L2 propagation, or production factor
/// validation fails.
pub fn run_validation_case(
    coupling: ValidationCoupling,
    seed_index: u64,
) -> Result<SpatialCovarianceValidationResult> {
    let replay = replay_case(coupling, seed_index)?;
    let joint = joint_covariance(&replay);
    let propagated = fixed_l2_inputs()?
        .propagate_joint_phase_covariance((0, 0), (0, 1), joint.view())
        .map_err(anyhow::Error::new)?;
    let status = validation_status(coupling);
    let block = validation_block(status, coupling, &replay, propagated.date_factor.clone())?;
    let factor_digest = factor_digest(&block.difference_factor);
    let difference = propagated.date_covariance;
    Ok(SpatialCovarianceValidationResult {
        schema: "dolphinrust.spatial-covariance.production-validation/4".to_owned(),
        coupling,
        seed_index,
        status: status_name(status).to_owned(),
        target_covariance: matrix_rows(&replay.target_covariance),
        reference_covariance: matrix_rows(&replay.reference_covariance),
        target_reference_covariance: matrix_rows(&replay.target_reference_covariance),
        difference_variance: difference.diag().sum(),
        difference_covariance: matrix_rows(&difference),
        factor_digest,
        persisted_factor_digest: String::new(),
        dependency_cone_bytes: replay.dependency_cone.total_bytes,
        source_influence_bytes: replay.dependency_cone.source_influence_bytes,
        source_correlation_workspace_bytes: replay
            .dependency_cone
            .source_correlation_workspace_bytes,
        source_correlation_model: replay
            .effective_looks
            .as_ref()
            .map_or_else(String::new, |effective| effective.model.to_owned()),
        source_cache_peak_bytes: replay.source_cache_peak_bytes,
        hdf5_bytes: 0,
        hdf5_sha256: String::new(),
        hdf5_path: String::new(),
        sidecar_sha256: String::new(),
        sidecar_path: String::new(),
        bounded_hdf5_sha256: String::new(),
        bounded_hdf5_path: String::new(),
        bounded_sidecar_sha256: String::new(),
        bounded_sidecar_path: String::new(),
        runtime_resource_receipt_digest: String::new(),
        bounded_runtime_resource_receipt_digest: String::new(),
        hdf5_schema_version: SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
        manifest_schema_version: 3,
    })
}

/// Persist one bounded production parity fixture and re-read its exact factor.
///
/// # Errors
/// Returns an error for invalid factor metadata, transactional commit failure,
/// or capped read failure.
pub fn write_validation_fixture(
    directory: &Path,
    coupling: ValidationCoupling,
    seed_index: u64,
) -> Result<SpatialCovarianceValidationResult> {
    let mut result = run_validation_case(coupling, seed_index)?;
    let whole =
        emit_actual_production_fixture(&directory.join("whole"), coupling, seed_index, None)?;
    let bounded = emit_actual_production_fixture(
        &directory.join("bounded"),
        coupling,
        seed_index,
        Some(BlockIndices {
            row_start: 1,
            row_stop: 3,
            col_start: 1,
            col_stop: 3,
        }),
    )?;
    result.factor_digest = whole.factor_digest.clone();
    result.persisted_factor_digest = whole.factor_digest;
    result.hdf5_bytes = whole.hdf5_bytes;
    result.hdf5_sha256 = whole.hdf5_sha256;
    result.hdf5_path = format!("whole/{SPATIAL_REFERENCE_COVARIANCE_FILENAME}");
    result.sidecar_sha256 = whole.sidecar_sha256;
    result.sidecar_path = format!("whole/{SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME}");
    result.bounded_hdf5_sha256 = bounded.hdf5_sha256;
    result.bounded_hdf5_path = format!("bounded/{SPATIAL_REFERENCE_COVARIANCE_FILENAME}");
    result.bounded_sidecar_sha256 = bounded.sidecar_sha256;
    result.bounded_sidecar_path =
        format!("bounded/{SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME}");
    result.runtime_resource_receipt_digest = whole.runtime_resource_receipt_digest;
    result.bounded_runtime_resource_receipt_digest = bounded.runtime_resource_receipt_digest;
    result.hdf5_schema_version = whole.hdf5_schema_version;
    result.manifest_schema_version = whole.manifest_schema_version;
    Ok(result)
}

struct EmittedProductionFixture {
    factor_digest: String,
    hdf5_bytes: u64,
    hdf5_sha256: String,
    sidecar_sha256: String,
    runtime_resource_receipt_digest: String,
    hdf5_schema_version: u16,
    manifest_schema_version: u16,
}

#[allow(clippy::too_many_lines)]
fn emit_actual_production_fixture(
    directory: &Path,
    coupling: ValidationCoupling,
    seed_index: u64,
    trim: Option<BlockIndices>,
) -> Result<EmittedProductionFixture> {
    std::fs::create_dir_all(directory)?;
    let shape = (3, 3);
    let paths = (0..VALIDATION_DATES)
        .map(|date| directory.join(format!("cslc-{date}.h5")))
        .collect::<Vec<_>>();
    let phase_sign = if coupling == ValidationCoupling::Negative {
        -1.0
    } else {
        1.0
    };
    let source_value = |date: usize, row: usize, column: usize| {
        let phase = 0.07 * date as f64
            + 0.013 * row as f64
            + phase_sign * 0.009 * column as f64
            + 0.0001 * seed_index as f64;
        let value = Cf64::from_polar(1.0 + 0.02 * row as f64, phase);
        Cf32::new(value.re as f32, value.im as f32)
    };
    for (date, path) in paths.iter().enumerate() {
        let values = Array2::from_shape_fn(shape, |(row, column)| source_value(date, row, column));
        let file = hdf5::File::create(path)?;
        file.new_dataset_builder()
            .with_data(&values)
            .create("data")?;
    }
    let source_manifest = CslcCovarianceManifest::capture(InputType::OperaCslc, "/data", &paths)?;
    let source_model_version_digest = sequential_source_model_identity_digest(
        CSLC_COVARIANCE_SOURCE_PROVIDER,
        CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
        CSLC_COVARIANCE_SOURCE_MODEL,
        CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
    );
    let source_options = EmpiricalSourceFactorOptions {
        half_window: dolphin_core::HalfWindow { y: 0, x: 0 },
        shrinkage_alpha: 0.2,
        relative_diagonal_floor: 1e-8,
    };
    let source_model = crate::cslc_covariance_source::empirical_factor_config(&source_options)?;
    let mut resolver = source_manifest.resolver(
        &[0, 1],
        "spatial-covariance-validation",
        (0, 0),
        shape,
        CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 3,
            cols: 3,
            stride_y: 1,
            stride_x: 1,
        },
        &source_options,
        source_model_version_digest,
        None,
    )?;
    let stack = Array3::from_shape_fn(
        (VALIDATION_DATES, shape.0, shape.1),
        |(date, row, column)| {
            let value = source_value(date, row, column);
            Cf64::new(f64::from(value.re), f64::from(value.im))
        },
    );
    let config = validation_config();
    let grid = CovarianceOperatorGrid {
        row_start: 0,
        col_start: 0,
        rows: 3,
        cols: 3,
        stride_y: 1,
        stride_x: 1,
    };
    let request = SequentialCovarianceCaptureRequest {
        burst_id: "spatial-covariance-validation".to_owned(),
        source_manifest_digest: source_manifest.digest(),
        source_model_version_digest,
        native_grid: grid,
        output_grid: grid,
        owned_output_grid: grid,
        branch_tolerance: 1e-10,
    };
    let mut blocks = Vec::new();
    run_sequential_with_covariance_capture_and_source_factors(
        stack.view(),
        &config,
        &ComputeEngine::new(ComputeBackend::Cpu),
        &request,
        &mut resolver,
        |block| {
            blocks.push(block);
            Ok(())
        },
    )?;
    let topology = SequentialReplayTopology::plan_identified(
        VALIDATION_DATES,
        shape,
        shape,
        1,
        Array2::from_elem(shape, true).view(),
        &config,
        validation_scope(),
        ReplayIdNamespace {
            burst_id: request.burst_id.clone(),
            source_manifest_digest: request.source_manifest_digest,
            source_model_version_digest: request.source_model_version_digest,
            native_origin: (0, 0),
            output_origin: (0, 0),
            owned_output_origin: (0, 0),
            owned_output_shape: shape,
        },
    )?;
    let metadata = CovarianceOperatorMetadata {
        normalized_config_digest: format!(
            "sha256:{}",
            hex_bytes(&sequential_replay_config_digest(&config))
        ),
        kernel_digest: format!("sha256:{}", hex_bytes(&sequential_replay_kernel_digest())),
        source: SourceReplayIdentity {
            manifest_digest: Some(format!("sha256:{}", hex_bytes(&source_manifest.digest()))),
            provider: Some(CSLC_COVARIANCE_SOURCE_PROVIDER.to_owned()),
            provider_version: Some(CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION.to_owned()),
            model: Some(CSLC_COVARIANCE_SOURCE_MODEL.to_owned()),
            model_version: Some(CSLC_COVARIANCE_SOURCE_MODEL_VERSION.to_owned()),
            model_version_digest: Some(format!(
                "sha256:{}",
                hex_bytes(&source_model_version_digest)
            )),
            model_receipt_digest: Some(format!(
                "sha256:{}",
                hex_bytes(source_model.config_digest())
            )),
        },
        replay_status: CovarianceReplayStatus::Replayable,
        stitched_status: StitchedCovarianceStatus::NotStitched,
        downstream_inference_status: DownstreamInferenceStatus::BlockedPendingIssue54And53,
        ..CovarianceOperatorMetadata::default()
    };
    let plan = topology.covariance_operator_plan(&request.burst_id)?;
    let transaction = CovarianceArtifactTransaction::acquire(directory)?;
    let scratch = directory.join("phase_covariance_operator.h5.scratch");
    let mut writer = CovarianceOperatorWriter::create(&scratch, &metadata, &plan)?;
    for block in &blocks {
        writer.write_block(block)?;
    }
    let receipt = writer.finish()?;
    let disk = admit_covariance_artifact_disk_with_identity_index(
        16 << 20,
        receipt.peak_identity_index_disk_bytes,
        u64::MAX,
    )?;
    let operator_manifest =
        finalize_covariance_artifact(&transaction, &scratch, &metadata, disk, &receipt)?;
    drop(transaction);
    let fixed_l2_inputs = FixedL2WorkflowInputs::new(
        array![[-1.0], [1.0]],
        Array3::from_shape_fn((2, shape.0, shape.1), |(date, row, column)| {
            0.07 * date as f64 + 0.013 * row as f64 + phase_sign * 0.009 * column as f64
        }),
        Some(Array3::from_elem((2, shape.0, shape.1), 1.0)),
    )?;
    let validity = Array2::from_elem(shape, true);
    let mut state = ProductionCovarianceState {
        replay_context: Some(ProductionCovarianceReplayContext {
            source_manifest,
            operator_manifest,
            tiles: vec![CapturedReplayTile {
                request: request.clone(),
                member_indices: vec![0, 1],
                processed_origin: (0, 0),
                processed_shape: shape,
                native_validity: validity.clone(),
                num_real_dates: VALIDATION_DATES,
            }],
            masks: BTreeMap::new(),
            operator_block_byte_cap: 16 << 20,
        }),
        fixed_l2_inputs: Some(fixed_l2_inputs),
        ownership: Array3::from_elem((VALIDATION_DATES, shape.0, shape.1), 0),
        seam_rotations: vec![(0, vec![Cf64::new(1.0, 0.0); VALIDATION_DATES])],
        source_burst_ids: vec![request.burst_id.clone()],
        burst_output_mappings: vec![BurstOutputMapping {
            owner: 0,
            frame_origin: (0, 0),
            output_origin: (0, 0),
            shape,
        }],
        analysis_origin: (0, 0),
        correction_order_digest: correction_order_digest(
            &dolphin_core::config::CorrectionOptions::default(),
            None,
            &crate::corrections::CorrectionLayers {
                ionosphere: None,
                troposphere: None,
                los_geometry: None,
                solid_earth_tide: None,
            },
        )?,
        unwrap_branch_digest: unwrap_branch_digest(
            dolphin_core::config::UnwrapMethod::Native,
            b"spatial-covariance-validation-native",
            &[(0, 1)],
            validity.view(),
            Array3::from_elem((1, shape.0, shape.1), 1).view(),
            false,
        ),
    };
    let mut emitted_validity = validity;
    let mut reference = if coupling == ValidationCoupling::Coincident {
        (1, 1)
    } else {
        (1, 0)
    };
    let mut geotransform = [100.0, 30.0, 0.0, 200.0, 0.0, -30.0];
    if let Some(target) = trim {
        state.trim(target);
        emitted_validity = emitted_validity
            .slice(s![target.rows(), target.cols()])
            .to_owned();
        reference = (0, 0);
        geotransform[0] += target.col_start as f64 * geotransform[1];
        geotransform[3] += target.row_start as f64 * geotransform[5];
    }
    let mut workflow = DisplacementWorkflow {
        work_directory: directory.to_owned(),
        cslc_file_list: paths.clone(),
        ..DisplacementWorkflow::default()
    };
    workflow.input_options.input_type = InputType::OperaCslc;
    workflow.input_options.subdataset = Some("/data".to_owned());
    workflow.phase_linking.shp_method = ShpMethod::Rect;
    workflow.phase_linking.empirical_source_factor = source_options;
    state.emit(
        &workflow,
        &config,
        emitted_validity.view(),
        reference,
        32611,
        geotransform,
        &[0.0, 12.0],
    )?;
    let manifest = read_spatial_reference_covariance_artifact_manifest(directory)?;
    let hdf5 = directory.join(SPATIAL_REFERENCE_COVARIANCE_FILENAME);
    let header = read_spatial_reference_covariance_header(&hdf5, VALIDATION_BYTE_CAP)?;
    let persisted = read_spatial_reference_covariance_block(&hdf5, 0, VALIDATION_BYTE_CAP)?;
    let emitted = EmittedProductionFixture {
        factor_digest: factor_digest(&persisted.block.difference_factor),
        hdf5_bytes: manifest.hdf5_bytes,
        hdf5_sha256: manifest.hdf5_sha256,
        sidecar_sha256: sha256_path(
            &directory.join(SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME),
        )?,
        runtime_resource_receipt_digest: manifest.runtime_resource_receipt_digest,
        hdf5_schema_version: header.schema_version,
        manifest_schema_version: manifest.schema_version,
    };
    for path in paths.into_iter().chain([
        directory.join(crate::covariance_artifact::COVARIANCE_OPERATOR_FILENAME),
        directory.join(crate::covariance_artifact::COVARIANCE_OPERATOR_MANIFEST_FILENAME),
        directory.join(crate::covariance_artifact::COVARIANCE_OPERATOR_LOCK_FILENAME),
        directory.join(SPATIAL_REFERENCE_COVARIANCE_LOCK_FILENAME),
    ]) {
        if path.exists() {
            std::fs::remove_file(path)?;
        }
    }
    Ok(emitted)
}

fn validation_block(
    status: SpatialReferenceCovarianceStatus,
    coupling: ValidationCoupling,
    replay: &ReferenceDifferenceCovarianceReplay,
    date_factor: Array2<f64>,
) -> Result<dolphin_io::SpatialReferenceCovarianceBlock> {
    build_factor_block(
        0,
        CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 2,
            stride_y: 1,
            stride_x: 1,
        },
        VALIDATION_DATES,
        1.0,
        &[
            target_factor(status, coupling, replay, date_factor),
            target_factor(
                if coupling == ValidationCoupling::Coincident {
                    SpatialReferenceCovarianceStatus::InvalidReference
                } else {
                    SpatialReferenceCovarianceStatus::Valid
                },
                ValidationCoupling::Coincident,
                replay,
                Array2::zeros((VALIDATION_DATES, 0)),
            ),
        ],
    )
}

fn replay_case(
    coupling: ValidationCoupling,
    seed_index: u64,
) -> Result<ReferenceDifferenceCovarianceReplay> {
    let topology = SequentialReplayTopology::plan(
        VALIDATION_DATES,
        (1, 2),
        (1, 2),
        1,
        &validation_config(),
        validation_scope(),
    )
    .context("planning deterministic validation topology")?;
    let target_node = topology
        .date_node_id(GlobalDateId::new(1), 0)
        .context("validation target node is absent")?;
    let reference_node = topology
        .date_node_id(GlobalDateId::new(1), 1)
        .context("validation reference node is absent")?;
    let sources = [
        SourceId::new(100 + seed_index * 2),
        SourceId::new(101 + seed_index * 2),
    ];
    let mut dag = InfluenceDag::new();
    for (index, source) in sources.iter().copied().enumerate() {
        dag.add_source(SourceDefinition::new(source, 1, [(index + 1) as u8; 32]))?;
    }
    add_node(&mut dag, target_node, &sources, [1.0, 0.0])?;
    add_node(
        &mut dag,
        reference_node,
        &sources,
        reference_weights(coupling),
    )?;
    topology
        .replay_reference_difference_covariance(
            &[(GlobalDateId::new(0), 0), (GlobalDateId::new(1), 0)],
            &[(GlobalDateId::new(0), 1), (GlobalDateId::new(1), 1)],
            DependencyConeQuery {
                source_rank: 1,
                microbatch: 1,
                byte_cap: VALIDATION_BYTE_CAP,
            },
            |_| Ok(dag),
        )
        .map_err(anyhow::Error::new)
}

fn benchmark_production_cell(
    tile_pixels: usize,
    dates: usize,
) -> Result<(ProductionCellResult, usize, (usize, usize))> {
    let side = (1..=tile_pixels)
        .take_while(|value| value.saturating_mul(*value) <= tile_pixels)
        .last()
        .context("benchmark tile area has no square root")?;
    anyhow::ensure!(
        side >= 3 && side.checked_mul(side) == Some(tile_pixels),
        "benchmark tile_pixels must be a square area with side at least three"
    );
    let mut config = benchmark_config();
    config.half_window = dolphin_core::HalfWindow { y: 1, x: 1 };
    let shape = (side, side);
    let topology =
        SequentialReplayTopology::plan(dates, shape, shape, 9, &config, validation_scope())
            .context("planning benchmark replay topology")?;
    let block_count = topology.blocks().len();
    let stack = Array3::from_shape_fn((dates, shape.0, shape.1), |(date, row, column)| {
        let key = (date as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (row as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
            ^ (column as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
        let unit = |stream: u64| {
            let mut value = key ^ stream;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            (value >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
        };
        let amplitude = 0.75 + 0.5 * unit(0x243f_6a88_85a3_08d3);
        let phase = std::f64::consts::TAU * unit(0x1319_8a2e_0370_7344);
        Cf64::from_polar(amplitude, phase)
    });
    let grid = CovarianceOperatorGrid {
        row_start: 0,
        col_start: 0,
        rows: u32::try_from(shape.0)?,
        cols: u32::try_from(shape.1)?,
        stride_y: 1,
        stride_x: 1,
    };
    let result = run_production_cell(ProductionCellInputs {
        stack,
        validity: Array2::from_elem(shape, true),
        config,
        capture: SequentialCovarianceCaptureRequest {
            burst_id: "spatial-covariance-benchmark".to_owned(),
            source_manifest_digest: [81; 32],
            source_model_version_digest: [82; 32],
            native_grid: grid,
            output_grid: grid,
            owned_output_grid: grid,
            branch_tolerance: 1e-10,
        },
        target: (u64::try_from(side / 2)?, u64::try_from(side / 2)?),
        reference: (u64::try_from(side / 2)?, u64::try_from(side / 2 + 1)?),
        source_model: EmpiricalProperComplexConfig::new(1, 1, 0.05, 1e-12, [83; 32])?,
        source_correlation: SourceCorrelationModel::ExponentialEuclidean {
            distance_scale_pixels: 1.5,
        },
        data_identity: [84; 32],
    })
    .context("executing benchmark production cell")?;
    Ok((result, block_count, shape))
}

fn add_node(
    dag: &mut InfluenceDag,
    id: dolphin_phaselink::NodeId,
    sources: &[SourceId; 2],
    weights: [f64; 2],
) -> Result<()> {
    let mut node = InfluenceNode::new(id, 1);
    for (source, weight) in sources.iter().copied().zip(weights) {
        if weight != 0.0 {
            node = node.with_source(SourceEdge::new(source, array![[weight]]));
        }
    }
    dag.add_node(node)?;
    Ok(())
}

fn reference_weights(coupling: ValidationCoupling) -> [f64; 2] {
    match coupling {
        ValidationCoupling::Positive => [0.5, 0.5],
        ValidationCoupling::Independent | ValidationCoupling::Invalid => [0.0, 1.0],
        ValidationCoupling::Negative => [-1.0, 0.0],
        ValidationCoupling::Coincident => [1.0, 0.0],
    }
}

fn joint_covariance(replay: &ReferenceDifferenceCovarianceReplay) -> Array2<f64> {
    let dates = replay.target_covariance.nrows();
    let mut joint = Array2::zeros((2 * dates, 2 * dates));
    joint
        .slice_mut(ndarray::s![..dates, ..dates])
        .assign(&replay.target_covariance);
    joint
        .slice_mut(ndarray::s![dates.., dates..])
        .assign(&replay.reference_covariance);
    joint
        .slice_mut(ndarray::s![..dates, dates..])
        .assign(&replay.target_reference_covariance);
    joint
        .slice_mut(ndarray::s![dates.., ..dates])
        .assign(&replay.target_reference_covariance.t());
    joint
}

fn fixed_l2_inputs() -> Result<FixedL2WorkflowInputs> {
    FixedL2WorkflowInputs::new(
        array![[-1.0], [1.0]],
        Array3::zeros((2, 1, 2)),
        Some(Array3::from_elem((2, 1, 2), 1.0)),
    )
}

fn target_factor(
    status: SpatialReferenceCovarianceStatus,
    coupling: ValidationCoupling,
    replay: &ReferenceDifferenceCovarianceReplay,
    date_factor: Array2<f64>,
) -> TargetFactor {
    let valid = status == SpatialReferenceCovarianceStatus::Valid;
    let condition_number = if valid && date_factor.ncols() > 0 {
        1.0
    } else {
        f64::NAN
    };
    TargetFactor {
        status,
        source_burst_index: if valid {
            0
        } else {
            SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE
        },
        date_factor: valid.then_some(date_factor),
        source_factor_receipt: if valid {
            replay.source_factor_receipt
        } else {
            [0; 32]
        },
        effective_looks_fraction: if valid { 1.0 } else { f64::NAN },
        support_union_count: u64::from(valid),
        effective_looks_receipt: if valid {
            digest_bytes(format!("{coupling:?}").as_bytes())
        } else {
            [0; 32]
        },
        resource_high_water_bytes: if valid {
            replay.dependency_cone.total_bytes.max(1)
        } else {
            0
        },
        condition_number,
    }
}

fn validation_status(coupling: ValidationCoupling) -> SpatialReferenceCovarianceStatus {
    if coupling == ValidationCoupling::Invalid {
        SpatialReferenceCovarianceStatus::InvalidReference
    } else {
        SpatialReferenceCovarianceStatus::Valid
    }
}

fn validation_config() -> SequentialConfig {
    SequentialConfig {
        ministack_size: 2,
        max_num_compressed: 1,
        half_window: dolphin_core::HalfWindow { y: 0, x: 0 },
        strides: dolphin_core::Strides { y: 1, x: 1 },
        use_evd: true,
        beta: 0.1,
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

fn benchmark_config() -> SequentialConfig {
    SequentialConfig {
        ministack_size: 5,
        max_num_compressed: 2,
        ..validation_config()
    }
}

fn validation_scope() -> ReplayExecutionScope {
    ReplayExecutionScope {
        enabled: true,
        backend: ReplayBackend::CpuF64,
        estimator_fallback: false,
        phase_bias_correction: false,
        strong_source_identity: true,
        stitched_burst_count: 1,
    }
}

fn factor_digest(values: &[f64]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:spatial-covariance-validation-factor:v1");
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn matrix_rows(values: &Array2<f64>) -> Vec<Vec<f64>> {
    values.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn digest_bytes(value: &[u8]) -> [u8; 32] {
    Sha256::digest(value).into()
}

fn sha256_path(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn status_name(status: SpatialReferenceCovarianceStatus) -> &'static str {
    match status {
        SpatialReferenceCovarianceStatus::Valid => "valid",
        SpatialReferenceCovarianceStatus::InvalidReference => "invalid_reference",
        _ => "not_evaluable",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn actual_cell_inputs(use_evd: bool) -> ProductionCellInputs {
        let dates = 5;
        let stack = Array3::from_shape_fn((dates, 7, 7), |(date, row, column)| {
            let mut state = 0x9e37_79b9_7f4a_7c15_u64
                ^ (date as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
                ^ (row as u64).wrapping_mul(0x94d0_49bb_1331_11eb)
                ^ column as u64;
            state ^= state >> 30;
            state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            let noise_real = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            state ^= state >> 27;
            state = state.wrapping_mul(0x94d0_49bb_1331_11eb);
            let noise_imag = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            let amplitude = 1.0 + 0.04 * date as f64 + 0.003 * (row + 2 * column) as f64;
            let phase = 0.09 * date as f64 + 0.013 * row as f64 - 0.007 * column as f64;
            Cf64::from_polar(amplitude, phase) + Cf64::new(0.2 * noise_real, 0.2 * noise_imag)
        });
        let mut config = validation_config();
        config.ministack_size = 5;
        config.max_num_compressed = 2;
        config.half_window = dolphin_core::HalfWindow { y: 1, x: 1 };
        config.use_evd = use_evd;
        let source_model = EmpiricalProperComplexConfig::new(1, 1, 0.05, 1e-12, [41; 32]).unwrap();
        ProductionCellInputs {
            stack,
            validity: Array2::from_elem((7, 7), true),
            config,
            capture: SequentialCovarianceCaptureRequest {
                burst_id: "validation-full-cell".to_owned(),
                source_manifest_digest: [42; 32],
                source_model_version_digest: [43; 32],
                native_grid: CovarianceOperatorGrid {
                    row_start: 0,
                    col_start: 0,
                    rows: 7,
                    cols: 7,
                    stride_y: 1,
                    stride_x: 1,
                },
                output_grid: CovarianceOperatorGrid {
                    row_start: 0,
                    col_start: 0,
                    rows: 7,
                    cols: 7,
                    stride_y: 1,
                    stride_x: 1,
                },
                owned_output_grid: CovarianceOperatorGrid {
                    row_start: 0,
                    col_start: 0,
                    rows: 7,
                    cols: 7,
                    stride_y: 1,
                    stride_x: 1,
                },
                branch_tolerance: 1e-10,
            },
            target: (3, 4),
            reference: (3, 5),
            source_model,
            source_correlation: SourceCorrelationModel::ExponentialEuclidean {
                distance_scale_pixels: 1.5,
            },
            data_identity: [44; 32],
        }
    }

    #[test]
    fn full_cell_runner_executes_actual_emi_and_evd_production_paths() {
        for use_evd in [false, true] {
            let result = run_production_cell(actual_cell_inputs(use_evd)).unwrap();
            assert_eq!(result.status, "valid");
            assert_eq!(result.target_estimate_history.len(), 5);
            assert_eq!(result.production_operator_matrix.len(), 10);
            assert_eq!(result.predicted_difference_covariance.len(), 5);
            assert!(result.resource_high_water_bytes > 0);
            assert!(result.effective_looks_fraction > 0.0);
            assert!(result.support_union_count > 0);
        }
    }

    #[test]
    fn native_centers_round_trip_through_the_strided_output_grid() {
        for (native, stride, output) in [(129, 2, 64), (130, 4, 32), (254, 4, 63)] {
            assert_eq!(native_center_to_output(native, stride).unwrap(), output);
            assert_eq!(output_to_native_center(output, stride).unwrap(), native);
        }
        assert!(native_center_to_output(128, 2).is_err());
        assert!(native_center_to_output(128, 4).is_err());
        assert!(output_to_native_center(usize::MAX, 4).is_err());
    }

    #[test]
    fn benchmark_capture_processes_the_exact_requested_area() {
        let evidence = run_benchmark_preflight(64, 5).unwrap();
        assert_eq!(evidence.tile_pixels, 64);
        assert_eq!(evidence.processed_tile_pixels, 64);
        assert_eq!(evidence.capture_native_shape, [8, 8]);
        assert!(evidence.block_count > 0);
        assert!(evidence.reference_cone_sources > 0);
        assert!(run_benchmark_preflight(63, 5).is_err());
    }

    #[test]
    fn stride_four_realized_support_uses_the_production_center_offset() {
        let mut config = validation_config();
        config.half_window = dolphin_core::HalfWindow { y: 1, x: 1 };
        config.strides = dolphin_core::Strides { y: 4, x: 4 };
        assert_eq!(
            realized_support_window_origin((32, 32), (120, 120), (20, 20), &config).unwrap(),
            (9, 9)
        );
    }

    #[test]
    fn masked_target_runs_the_actual_masked_path_and_fails_closed() {
        let mut inputs = actual_cell_inputs(false);
        inputs.validity[(3, 4)] = false;
        let error = run_production_cell(inputs).unwrap_err();
        let replay = error
            .downcast_ref::<SequentialReplayError>()
            .expect("masked production error retains its replay status");
        assert_eq!(replay.status(), ReplayStatus::MaskedNode);
    }

    #[test]
    fn frozen_tied_probe_reaches_the_actual_evd_singular_status() {
        let preregistration: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../validation/spatial_covariance_preregistration.json"
        )))
        .unwrap();
        assert_eq!(
            execute_tied_probe(&preregistration).unwrap(),
            dolphin_io::CovarianceOperatorStatus::SingularLocalInformation
        );
    }

    #[test]
    fn zero_beta_emi_threshold_boundary_fails_closed() {
        let stack = Array3::from_shape_vec(
            (2, 1, 3),
            vec![
                Cf64::new(1.0, 0.0),
                Cf64::new(1.0, 0.0),
                Cf64::new(0.0, 0.0),
                Cf64::new(1.0, 0.0),
                Cf64::new(-1.0, 0.0),
                Cf64::new(0.0, 0.0),
            ],
        )
        .unwrap();
        let mut config = validation_config();
        config.beta = 0.0;
        config.half_window = dolphin_core::HalfWindow { y: 0, x: 1 };
        config.strides = dolphin_core::Strides { y: 1, x: 3 };
        config.use_evd = false;
        let grid = CovarianceOperatorGrid {
            row_start: 0,
            col_start: 0,
            rows: 1,
            cols: 1,
            stride_y: 1,
            stride_x: 3,
        };
        let error = run_production_cell(ProductionCellInputs {
            stack,
            validity: Array2::from_elem((1, 3), true),
            config,
            capture: SequentialCovarianceCaptureRequest {
                burst_id: "zero-beta-boundary".to_owned(),
                source_manifest_digest: [71; 32],
                source_model_version_digest: [72; 32],
                native_grid: CovarianceOperatorGrid {
                    row_start: 0,
                    col_start: 0,
                    rows: 1,
                    cols: 3,
                    stride_y: 1,
                    stride_x: 1,
                },
                output_grid: grid,
                owned_output_grid: grid,
                branch_tolerance: 1e-10,
            },
            target: (0, 0),
            reference: (0, 0),
            source_model: EmpiricalProperComplexConfig::new(0, 1, 0.05, 1e-12, [73; 32]).unwrap(),
            source_correlation: SourceCorrelationModel::Identity,
            data_identity: [74; 32],
        })
        .unwrap_err();
        let replay = error.downcast_ref::<SequentialReplayError>().unwrap();
        assert_eq!(replay.status(), ReplayStatus::NondifferentiableNode);
    }

    #[test]
    fn four_block_attempt_retains_the_complete_transitive_replay_halo() {
        let preregistration: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../validation/spatial_covariance_preregistration.json"
        )))
        .unwrap();
        let asset: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../validation/spatial_covariance_portable_tables.json"
        )))
        .unwrap();
        let tables = PortableDgpTables::from_documents(&preregistration, &asset).unwrap();
        let cell_id = "hw_1x1|stride_1|glrt_frozen|interior|coincident|four_blocks|emi|well_separated|independent_complex_looks";
        let seed_sha256 = format!(
            "{:x}",
            Sha256::digest(format!("spatial-covariance-f54-07-v2||{cell_id}||0"))
        );
        let evidence = run_frozen_attempt(
            &preregistration,
            &tables,
            &FrozenAttemptRequest {
                schema: "dolphinrust.spatial-covariance.attempt/4".to_owned(),
                cell_id: cell_id.to_owned(),
                cell_ordinal: 2,
                seed_index: 0,
                seed_sha256,
                half_window: "hw_1x1".to_owned(),
                stride: "stride_1".to_owned(),
                support: "glrt_frozen".to_owned(),
                position: "interior".to_owned(),
                pair_geometry: "coincident".to_owned(),
                block_topology: "four_blocks".to_owned(),
                estimator: "emi".to_owned(),
                eigen_stress: "well_separated".to_owned(),
                source_process: "independent_complex_looks".to_owned(),
            },
        )
        .unwrap();
        assert_eq!(evidence["status"], "valid");
        assert_eq!(evidence["raw_input_shape"][0], 121);
    }

    #[test]
    fn stride_two_attempt_binds_the_congruent_native_center_and_realized_support() {
        let preregistration: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../validation/spatial_covariance_preregistration.json"
        )))
        .unwrap();
        let asset: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../validation/spatial_covariance_portable_tables.json"
        )))
        .unwrap();
        let tables = PortableDgpTables::from_documents(&preregistration, &asset).unwrap();
        let cell_id = "hw_1x1|stride_2|rect|interior|coincident|four_blocks|emi|well_separated|independent_complex_looks";
        let evidence = run_frozen_attempt(
            &preregistration,
            &tables,
            &FrozenAttemptRequest {
                schema: "dolphinrust.spatial-covariance.attempt/4".to_owned(),
                cell_id: cell_id.to_owned(),
                cell_ordinal: 246,
                seed_index: 0,
                seed_sha256: format!(
                    "{:x}",
                    Sha256::digest(format!("spatial-covariance-f54-07-v2||{cell_id}||0"))
                ),
                half_window: "hw_1x1".to_owned(),
                stride: "stride_2".to_owned(),
                support: "rect".to_owned(),
                position: "interior".to_owned(),
                pair_geometry: "coincident".to_owned(),
                block_topology: "four_blocks".to_owned(),
                estimator: "emi".to_owned(),
                eigen_stress: "well_separated".to_owned(),
                source_process: "independent_complex_looks".to_owned(),
            },
        )
        .unwrap();
        assert_eq!(evidence["status"], "valid");
        assert_eq!(evidence["target_coordinate"], serde_json::json!([129, 129]));
        assert_eq!(
            evidence["reference_coordinate"],
            serde_json::json!([129, 129])
        );
        assert_eq!(
            evidence["raw_input_sha256"],
            "88e93e0373beab04e191ca341d712b817e61a7b806ea2f337de9572748cbbedd"
        );
        assert_eq!(
            evidence["target_support_sha256"],
            "aab85c3a2f14eb18c03116971c1b58658132b1c0ab14c932e48f6fd8c9b10a79"
        );
        assert_eq!(
            evidence["target_support_sha256"],
            evidence["reference_support_sha256"]
        );
    }

    #[test]
    fn stochastic_nondifferentiable_attempt_is_retained_without_exempting_the_cell() {
        let preregistration: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../validation/spatial_covariance_preregistration.json"
        )))
        .unwrap();
        let asset: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../validation/spatial_covariance_portable_tables.json"
        )))
        .unwrap();
        let tables = PortableDgpTables::from_documents(&preregistration, &asset).unwrap();
        let cell_id = "hw_1x1|stride_1|rect|interior|shared_50_negative|four_blocks|emi|well_separated|independent_complex_looks";
        let request = |seed_index| FrozenAttemptRequest {
            schema: "dolphinrust.spatial-covariance.attempt/4".to_owned(),
            cell_id: cell_id.to_owned(),
            cell_ordinal: 168,
            seed_index,
            seed_sha256: format!(
                "{:x}",
                Sha256::digest(format!(
                    "spatial-covariance-f54-07-v2||{cell_id}||{seed_index}"
                ))
            ),
            half_window: "hw_1x1".to_owned(),
            stride: "stride_1".to_owned(),
            support: "rect".to_owned(),
            position: "interior".to_owned(),
            pair_geometry: "shared_50_negative".to_owned(),
            block_topology: "four_blocks".to_owned(),
            estimator: "emi".to_owned(),
            eigen_stress: "well_separated".to_owned(),
            source_process: "independent_complex_looks".to_owned(),
        };
        let failed = run_frozen_attempt(&preregistration, &tables, &request(0)).unwrap();
        assert_eq!(failed["status"], "nondifferentiable_node");
        assert_eq!(failed["emitted"], false);
        assert_eq!(failed["factor_emitted"], false);
        assert!(failed["target_estimate_history"].is_null());
        let emitted = run_frozen_attempt(&preregistration, &tables, &request(1)).unwrap();
        assert_eq!(emitted["status"], "valid");
        assert_eq!(emitted["emitted"], true);
        assert_eq!(failed["raw_input_shape"], emitted["raw_input_shape"]);
    }

    #[test]
    fn frozen_production_histories_use_the_exact_acquisition_zero_gauge() {
        let preregistration: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../validation/spatial_covariance_preregistration.json"
        )))
        .unwrap();
        let asset: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../validation/spatial_covariance_portable_tables.json"
        )))
        .unwrap();
        let tables = PortableDgpTables::from_documents(&preregistration, &asset).unwrap();
        let cell_id = "hw_1x1|stride_1|glrt_frozen|bounded_halo|coincident|one_block|emi|well_separated|independent_complex_looks";
        let evidence = run_frozen_attempt(
            &preregistration,
            &tables,
            &FrozenAttemptRequest {
                schema: "dolphinrust.spatial-covariance.attempt/4".to_owned(),
                cell_id: cell_id.to_owned(),
                cell_ordinal: 1,
                seed_index: 0,
                seed_sha256: format!(
                    "{:x}",
                    Sha256::digest(format!("spatial-covariance-f54-07-v2||{cell_id}||0"))
                ),
                half_window: "hw_1x1".to_owned(),
                stride: "stride_1".to_owned(),
                support: "glrt_frozen".to_owned(),
                position: "bounded_halo".to_owned(),
                pair_geometry: "coincident".to_owned(),
                block_topology: "one_block".to_owned(),
                estimator: "emi".to_owned(),
                eigen_stress: "well_separated".to_owned(),
                source_process: "independent_complex_looks".to_owned(),
            },
        )
        .unwrap();
        assert_eq!(evidence["target_estimate_history"][0], 0.0);
        assert_eq!(evidence["reference_estimate_history"][0], 0.0);
        let target: Vec<f64> =
            serde_json::from_value(evidence["target_estimate_history"].clone()).unwrap();
        let reference: Vec<f64> =
            serde_json::from_value(evidence["reference_estimate_history"].clone()).unwrap();
        assert!(target
            .iter()
            .zip(&reference)
            .all(|(target, reference)| (target - reference).abs() <= 1e-15));
        let covariance: Vec<Vec<f64>> =
            serde_json::from_value(evidence["predicted_difference_covariance"].clone()).unwrap();
        let covariance_values = covariance.iter().flatten().copied().collect::<Vec<_>>();
        assert!(covariance_values.iter().all(|&value| value == 0.0));
        assert_eq!(
            evidence["predicted_covariance_sha256"],
            numeric_digest("predicted-difference-covariance-v4", &covariance_values).unwrap()
        );
    }

    #[test]
    fn tile_edge_pair_spans_native_tiles_with_complete_support() {
        let preregistration: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../validation/spatial_covariance_preregistration.json"
        )))
        .unwrap();
        let asset: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../validation/spatial_covariance_portable_tables.json"
        )))
        .unwrap();
        let tables = PortableDgpTables::from_documents(&preregistration, &asset).unwrap();
        let cell_id = "hw_1x1|stride_1|rect|tile_edge|disjoint_after_depth_1|one_block|emi|well_separated|independent_complex_looks";
        let evidence = run_frozen_attempt(
            &preregistration,
            &tables,
            &FrozenAttemptRequest {
                schema: "dolphinrust.spatial-covariance.attempt/4".to_owned(),
                cell_id: cell_id.to_owned(),
                cell_ordinal: 232,
                seed_index: 0,
                seed_sha256: format!(
                    "{:x}",
                    Sha256::digest(format!("spatial-covariance-f54-07-v2||{cell_id}||0"))
                ),
                half_window: "hw_1x1".to_owned(),
                stride: "stride_1".to_owned(),
                support: "rect".to_owned(),
                position: "tile_edge".to_owned(),
                pair_geometry: "disjoint_after_depth_1".to_owned(),
                block_topology: "one_block".to_owned(),
                estimator: "emi".to_owned(),
                eigen_stress: "well_separated".to_owned(),
                source_process: "independent_complex_looks".to_owned(),
            },
        )
        .unwrap();
        assert_eq!(evidence["status"], "valid");
        assert_eq!(evidence["target_coordinate"], serde_json::json!([128, 254]));
        assert_eq!(
            evidence["reference_coordinate"],
            serde_json::json!([128, 259])
        );
        assert_eq!(evidence["target_source_count"], 9);
        assert_eq!(evidence["reference_source_count"], 9);
        assert_eq!(evidence["intersection_source_count"], 0);
    }

    #[test]
    fn matched_cohorts_use_production_replay_fixed_l2_and_factor_writer() {
        let independent = run_validation_case(ValidationCoupling::Independent, 7).unwrap();
        let positive = run_validation_case(ValidationCoupling::Positive, 7).unwrap();
        let negative = run_validation_case(ValidationCoupling::Negative, 7).unwrap();
        assert!(positive.difference_variance < independent.difference_variance);
        assert!(independent.difference_variance < negative.difference_variance);
    }

    #[test]
    fn deterministic_coincident_and_invalid_cases_fail_closed() {
        let coincident = run_validation_case(ValidationCoupling::Coincident, 3).unwrap();
        assert_eq!(coincident.difference_variance, 0.0);
        let invalid = run_validation_case(ValidationCoupling::Invalid, 3).unwrap();
        assert_eq!(invalid.status, "invalid_reference");
    }

    #[test]
    fn persisted_fixtures_round_trip_exact_production_factor_bytes() {
        for (index, coupling) in [
            ValidationCoupling::Independent,
            ValidationCoupling::Positive,
            ValidationCoupling::Negative,
            ValidationCoupling::Coincident,
            ValidationCoupling::Invalid,
        ]
        .into_iter()
        .enumerate()
        {
            let directory = std::env::temp_dir().join(format!(
                "dolphinrust-spatial-covariance-validation-{}-{index}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&directory);
            std::fs::create_dir_all(&directory).unwrap();
            let result = write_validation_fixture(&directory, coupling, 11).unwrap();
            assert_eq!(result.factor_digest, result.persisted_factor_digest);
            assert_eq!(result.hdf5_schema_version, 4);
            assert_eq!(result.manifest_schema_version, 3);
            std::fs::remove_dir_all(directory).unwrap();
        }
    }
}

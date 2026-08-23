//! Block-indexed HDF5 persistence for the sequential covariance replay operator.

use std::path::Path;

use hdf5::{Group, H5Type};
use ndarray::ArrayView2;
use num_complex::Complex64;

use crate::{IoError, Result};

/// HDF5 schema version for covariance replay operators.
pub const COVARIANCE_OPERATOR_SCHEMA_VERSION: u16 = 1;
/// Stable method name for the source-keyed sequential replay DAG.
pub const COVARIANCE_OPERATOR_METHOD: &str = "sequential_source_dag_v1";
/// Numeric version of [`COVARIANCE_OPERATOR_METHOD`].
pub const COVARIANCE_OPERATOR_METHOD_VERSION: u16 = 1;

/// One stable numeric-code/name pair persisted in an artifact registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CovarianceRegistryEntry {
    /// Stable numeric code.
    pub code: u16,
    /// Stable machine-readable name.
    pub name: &'static str,
}

/// Supported covariance operator methods.
pub const COVARIANCE_METHOD_REGISTRY: &[CovarianceRegistryEntry] = &[CovarianceRegistryEntry {
    code: 1,
    name: COVARIANCE_OPERATOR_METHOD,
}];

/// Stable phase-estimator branch selected for one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum CovarianceEstimatorBranch {
    /// Eigenvalue decomposition branch.
    Evd = 1,
    /// Eigenvector-based maximum-likelihood branch.
    Emi = 2,
}

impl CovarianceEstimatorBranch {
    const fn code(self) -> u16 {
        self as u16
    }

    fn from_code(code: u16) -> Result<Self> {
        match code {
            1 => Ok(Self::Evd),
            2 => Ok(Self::Emi),
            _ => Err(invalid(format!(
                "unknown covariance estimator branch {code}"
            ))),
        }
    }
}

/// Stable estimator-branch registry.
pub const COVARIANCE_ESTIMATOR_BRANCH_REGISTRY: &[CovarianceRegistryEntry] = &[
    CovarianceRegistryEntry {
        code: CovarianceEstimatorBranch::Evd as u16,
        name: "evd",
    },
    CovarianceRegistryEntry {
        code: CovarianceEstimatorBranch::Emi as u16,
        name: "emi",
    },
];

/// Per-output-node validity of the persisted replay state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum CovarianceOperatorStatus {
    /// The fixed branch is replayable for this output node.
    Valid = 0,
    /// The output node is masked.
    Masked = 1,
    /// No valid support contributed to the output node.
    NoContributor = 2,
    /// The local information state is singular.
    SingularLocalInformation = 3,
    /// The stored estimator state is non-finite.
    NonfiniteState = 4,
    /// A local Jacobian was non-finite.
    NonfiniteJacobian = 5,
    /// The fixed branch is at a nondifferentiable boundary.
    Nondifferentiable = 6,
    /// Compression could not be differentiated on its realized branch.
    InvalidCompression = 7,
}

impl CovarianceOperatorStatus {
    const fn code(self) -> u16 {
        self as u16
    }

    fn from_code(code: u16) -> Result<Self> {
        match code {
            0 => Ok(Self::Valid),
            1 => Ok(Self::Masked),
            2 => Ok(Self::NoContributor),
            3 => Ok(Self::SingularLocalInformation),
            4 => Ok(Self::NonfiniteState),
            5 => Ok(Self::NonfiniteJacobian),
            6 => Ok(Self::Nondifferentiable),
            7 => Ok(Self::InvalidCompression),
            _ => Err(invalid(format!(
                "unknown covariance operator status {code}"
            ))),
        }
    }
}

/// Stable per-node operator-status registry.
pub const COVARIANCE_OPERATOR_STATUS_REGISTRY: &[CovarianceRegistryEntry] = &[
    CovarianceRegistryEntry {
        code: CovarianceOperatorStatus::Valid as u16,
        name: "valid",
    },
    CovarianceRegistryEntry {
        code: CovarianceOperatorStatus::Masked as u16,
        name: "masked",
    },
    CovarianceRegistryEntry {
        code: CovarianceOperatorStatus::NoContributor as u16,
        name: "no_contributor",
    },
    CovarianceRegistryEntry {
        code: CovarianceOperatorStatus::SingularLocalInformation as u16,
        name: "singular_local_information",
    },
    CovarianceRegistryEntry {
        code: CovarianceOperatorStatus::NonfiniteState as u16,
        name: "nonfinite_state",
    },
    CovarianceRegistryEntry {
        code: CovarianceOperatorStatus::NonfiniteJacobian as u16,
        name: "nonfinite_jacobian",
    },
    CovarianceRegistryEntry {
        code: CovarianceOperatorStatus::Nondifferentiable as u16,
        name: "nondifferentiable",
    },
    CovarianceRegistryEntry {
        code: CovarianceOperatorStatus::InvalidCompression as u16,
        name: "invalid_compression",
    },
];

/// Artifact-level ability to resolve and replay the frozen source state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum CovarianceReplayStatus {
    /// All required replay identities are present.
    Replayable = 0,
    /// The ordered source manifest is absent.
    SourceManifestMissing = 1,
    /// The resolved source manifest does not match the artifact.
    SourceManifestMismatch = 2,
    /// The realized support was not frozen.
    SupportNotFrozen = 3,
    /// The estimator backend is outside the supported producer scope.
    UnsupportedBackend = 4,
    /// The immutable raw source bytes cannot be resolved.
    SourceUnavailable = 5,
    /// The caller-supplied source covariance model cannot be resolved.
    SourceModelUnavailable = 6,
}

impl CovarianceReplayStatus {
    const fn code(self) -> u16 {
        self as u16
    }

    fn from_code(code: u16) -> Result<Self> {
        match code {
            0 => Ok(Self::Replayable),
            1 => Ok(Self::SourceManifestMissing),
            2 => Ok(Self::SourceManifestMismatch),
            3 => Ok(Self::SupportNotFrozen),
            4 => Ok(Self::UnsupportedBackend),
            5 => Ok(Self::SourceUnavailable),
            6 => Ok(Self::SourceModelUnavailable),
            _ => Err(invalid(format!("unknown covariance replay status {code}"))),
        }
    }
}

/// Stable replay-status registry.
pub const COVARIANCE_REPLAY_STATUS_REGISTRY: &[CovarianceRegistryEntry] = &[
    CovarianceRegistryEntry {
        code: CovarianceReplayStatus::Replayable as u16,
        name: "replayable",
    },
    CovarianceRegistryEntry {
        code: CovarianceReplayStatus::SourceManifestMissing as u16,
        name: "source_manifest_missing",
    },
    CovarianceRegistryEntry {
        code: CovarianceReplayStatus::SourceManifestMismatch as u16,
        name: "source_manifest_mismatch",
    },
    CovarianceRegistryEntry {
        code: CovarianceReplayStatus::SupportNotFrozen as u16,
        name: "support_not_frozen",
    },
    CovarianceRegistryEntry {
        code: CovarianceReplayStatus::UnsupportedBackend as u16,
        name: "unsupported_backend",
    },
    CovarianceRegistryEntry {
        code: CovarianceReplayStatus::SourceUnavailable as u16,
        name: "source_unavailable",
    },
    CovarianceRegistryEntry {
        code: CovarianceReplayStatus::SourceModelUnavailable as u16,
        name: "source_model_unavailable",
    },
];

/// Status of covariance after any multiburst overlap leveling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum StitchedCovarianceStatus {
    /// The artifact is for one burst and was not stitched.
    NotStitched = 0,
    /// A stitched covariance is unavailable because seam uncertainty is not modeled.
    UnsupportedSeamCovariance = 1,
}

impl StitchedCovarianceStatus {
    const fn code(self) -> u16 {
        self as u16
    }

    fn from_code(code: u16) -> Result<Self> {
        match code {
            0 => Ok(Self::NotStitched),
            1 => Ok(Self::UnsupportedSeamCovariance),
            _ => Err(invalid(format!(
                "unknown stitched covariance status {code}"
            ))),
        }
    }
}

/// Stable stitched-covariance-status registry.
pub const STITCHED_COVARIANCE_STATUS_REGISTRY: &[CovarianceRegistryEntry] = &[
    CovarianceRegistryEntry {
        code: StitchedCovarianceStatus::NotStitched as u16,
        name: "not_stitched",
    },
    CovarianceRegistryEntry {
        code: StitchedCovarianceStatus::UnsupportedSeamCovariance as u16,
        name: "unsupported_seam_covariance",
    },
];

/// Downstream inference eligibility of a version-1 operator artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum DownstreamInferenceStatus {
    /// Spatial propagation and temporal coverage validation remain incomplete.
    BlockedPendingIssue54And53 = 0,
}

impl DownstreamInferenceStatus {
    const fn code(self) -> u16 {
        self as u16
    }

    fn from_code(code: u16) -> Result<Self> {
        match code {
            0 => Ok(Self::BlockedPendingIssue54And53),
            _ => Err(invalid(format!(
                "unknown downstream inference status {code}"
            ))),
        }
    }
}

/// Stable downstream-inference-status registry.
pub const DOWNSTREAM_INFERENCE_STATUS_REGISTRY: &[CovarianceRegistryEntry] =
    &[CovarianceRegistryEntry {
        code: DownstreamInferenceStatus::BlockedPendingIssue54And53 as u16,
        name: "blocked_pending_issue_54_and_53",
    }];

/// Immutable source/provider/model identity required for deterministic replay.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceReplayIdentity {
    /// Digest of the ordered source manifest.
    pub manifest_digest: Option<String>,
    /// Resolver/provider name.
    pub provider: Option<String>,
    /// Resolver/provider version.
    pub provider_version: Option<String>,
    /// Caller-supplied proper-complex source model name.
    pub model: Option<String>,
    /// Caller-supplied source model version.
    pub model_version: Option<String>,
    /// Digest of the source-model receipt and ordered component identity.
    pub model_receipt_digest: Option<String>,
}

impl SourceReplayIdentity {
    fn validate(&self, replay_status: CovarianceReplayStatus) -> Result<()> {
        for (name, value) in [
            ("source.manifest_digest", &self.manifest_digest),
            ("source.provider", &self.provider),
            ("source.provider_version", &self.provider_version),
            ("source.model", &self.model),
            ("source.model_version", &self.model_version),
            ("source.model_receipt_digest", &self.model_receipt_digest),
        ] {
            if value.as_ref().is_some_and(|text| text.is_empty()) {
                return Err(invalid(format!("{name} must be absent or nonempty")));
            }
        }
        if replay_status == CovarianceReplayStatus::Replayable
            && [
                self.manifest_digest.as_ref(),
                self.provider.as_ref(),
                self.provider_version.as_ref(),
                self.model.as_ref(),
                self.model_version.as_ref(),
                self.model_receipt_digest.as_ref(),
            ]
            .contains(&None)
        {
            return Err(invalid(
                "replayable covariance operator requires complete source/provider/model identity",
            ));
        }
        Ok(())
    }
}

/// Artifact-level covariance replay metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CovarianceOperatorMetadata {
    /// HDF5 schema version.
    pub schema_version: u16,
    /// Stable covariance method name.
    pub method: String,
    /// Numeric covariance method version.
    pub method_version: u16,
    /// Producing crate version.
    pub crate_version: String,
    /// Producing Git commit when supplied by the build.
    pub producer_commit: Option<String>,
    /// Exact gauge date index, currently zero.
    pub gauge_date_index: u32,
    /// Digest of normalized phase-linking configuration.
    pub normalized_config_digest: String,
    /// Digest identifying the replay kernel implementation.
    pub kernel_digest: String,
    /// Immutable external source and source-model identity.
    pub source: SourceReplayIdentity,
    /// Artifact replay status.
    pub replay_status: CovarianceReplayStatus,
    /// Covariance status after any multiburst stitching.
    pub stitched_status: StitchedCovarianceStatus,
    /// Downstream inference eligibility.
    pub downstream_inference_status: DownstreamInferenceStatus,
}

impl Default for CovarianceOperatorMetadata {
    fn default() -> Self {
        Self {
            schema_version: COVARIANCE_OPERATOR_SCHEMA_VERSION,
            method: COVARIANCE_OPERATOR_METHOD.to_owned(),
            method_version: COVARIANCE_OPERATOR_METHOD_VERSION,
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            producer_commit: None,
            gauge_date_index: 0,
            normalized_config_digest: String::new(),
            kernel_digest: String::new(),
            source: SourceReplayIdentity::default(),
            replay_status: CovarianceReplayStatus::SourceManifestMissing,
            stitched_status: StitchedCovarianceStatus::NotStitched,
            downstream_inference_status: DownstreamInferenceStatus::BlockedPendingIssue54And53,
        }
    }
}

impl CovarianceOperatorMetadata {
    fn validate(&self) -> Result<()> {
        if self.schema_version != COVARIANCE_OPERATOR_SCHEMA_VERSION {
            return Err(invalid(format!(
                "unsupported covariance operator schema version {}",
                self.schema_version
            )));
        }
        if self.method != COVARIANCE_OPERATOR_METHOD {
            return Err(invalid(format!(
                "unsupported covariance operator method {}",
                self.method
            )));
        }
        if self.method_version != COVARIANCE_OPERATOR_METHOD_VERSION {
            return Err(invalid(format!(
                "unsupported covariance operator method version {}",
                self.method_version
            )));
        }
        if self.gauge_date_index != 0 {
            return Err(invalid("covariance operator gauge date index must be zero"));
        }
        for (name, value) in [
            ("crate_version", self.crate_version.as_str()),
            (
                "normalized_config_digest",
                self.normalized_config_digest.as_str(),
            ),
            ("kernel_digest", self.kernel_digest.as_str()),
        ] {
            if value.is_empty() {
                return Err(invalid(format!("covariance operator {name} is empty")));
            }
        }
        if self
            .producer_commit
            .as_ref()
            .is_some_and(|commit| commit.is_empty())
        {
            return Err(invalid("producer_commit must be absent or nonempty"));
        }
        self.source.validate(self.replay_status)
    }
}

/// Global origin, shape, and sampling stride of one operator grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CovarianceOperatorGrid {
    /// Global starting row.
    pub row_start: u64,
    /// Global starting column.
    pub col_start: u64,
    /// Number of rows.
    pub rows: u32,
    /// Number of columns.
    pub cols: u32,
    /// Row stride relative to the native grid.
    pub stride_y: u32,
    /// Column stride relative to the native grid.
    pub stride_x: u32,
}

impl CovarianceOperatorGrid {
    fn area(self) -> Result<usize> {
        if self.rows == 0 || self.cols == 0 {
            return Err(invalid(
                "covariance operator grid dimensions must be positive",
            ));
        }
        if self.stride_y == 0 || self.stride_x == 0 {
            return Err(invalid("covariance operator grid strides must be positive"));
        }
        usize::try_from(u64::from(self.rows) * u64::from(self.cols))
            .map_err(|_| invalid("covariance operator grid area exceeds usize"))
    }
}

/// Persisted numeric state for one block of the implicit source-keyed replay DAG.
#[derive(Debug, Clone, PartialEq)]
pub struct CovarianceOperatorBlock {
    /// Burst identity owning the block.
    pub burst_id: String,
    /// Deterministic block identifier.
    pub block_id: u64,
    /// Sequential generation/ministack number.
    pub generation: u32,
    /// Native source/compressed-raster grid.
    pub native_grid: CovarianceOperatorGrid,
    /// Looked phase/output grid.
    pub output_grid: CovarianceOperatorGrid,
    /// Reference date index used by the block.
    pub reference_date_index: u32,
    /// Ordered global acquisition-date indices represented by phase angles.
    pub ordered_date_indices: Vec<u32>,
    /// Deterministic primitive-source IDs, one per native pixel.
    pub source_ids: Vec<u64>,
    /// Deterministic phase-node IDs, one per output pixel.
    pub phase_node_ids: Vec<u64>,
    /// Deterministic compressed-node IDs, one per native pixel.
    pub compressed_node_ids: Vec<u64>,
    /// Ordered carried compressed parents from earlier generations.
    pub carry_parent_ids: Vec<u64>,
    /// Native-pixel to nearest looked-output mapping.
    pub nearest_output_map: Vec<u32>,
    /// Linked phase angles in output-pixel-major, date-minor order.
    pub phase_angles: Vec<f64>,
    /// Complex compressed SLC raster.
    pub compressed_raster: Vec<Complex64>,
    /// Complex compression projection accumulator.
    pub projection_accumulator: Vec<Complex64>,
    /// Mean amplitude used by compression.
    pub mean_amplitude: Vec<f64>,
    /// Number of realized-support bits stored for each output pixel.
    pub support_bits_per_output: u32,
    /// Output-pixel-major bit-packed realized support.
    pub support_bits: Vec<u8>,
    /// Bit-packed native source validity.
    pub native_validity_bits: Vec<u8>,
    /// Fixed production estimator branch.
    pub estimator_branch: CovarianceEstimatorBranch,
    /// Selected estimator eigenvalue, one per output pixel.
    pub selected_eigenvalue: Vec<f64>,
    /// Selected eigenvalue gap, one per output pixel.
    pub eigen_gap: Vec<f64>,
    /// Stable operator status, one per output pixel.
    pub status: Vec<CovarianceOperatorStatus>,
}

impl CovarianceOperatorBlock {
    fn validate(&self, gauge_date_index: u32) -> Result<()> {
        if self.burst_id.is_empty() {
            return Err(invalid("covariance operator burst_id is empty"));
        }
        if self.ordered_date_indices.is_empty() {
            return Err(invalid("covariance operator block has no dates"));
        }
        if self.reference_date_index != gauge_date_index {
            return Err(invalid(
                "covariance operator block reference does not match gauge",
            ));
        }
        if self.ordered_date_indices.first() != Some(&gauge_date_index) {
            return Err(invalid(
                "covariance operator block dates do not start at the gauge",
            ));
        }
        if self
            .ordered_date_indices
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "covariance operator date indices are not strictly ordered",
            ));
        }

        let native_area = self.native_grid.area()?;
        let output_area = self.output_grid.area()?;
        check_len("source_ids", self.source_ids.len(), native_area)?;
        check_len(
            "compressed_node_ids",
            self.compressed_node_ids.len(),
            native_area,
        )?;
        check_len(
            "nearest_output_map",
            self.nearest_output_map.len(),
            native_area,
        )?;
        check_len(
            "compressed_raster",
            self.compressed_raster.len(),
            native_area,
        )?;
        check_len(
            "projection_accumulator",
            self.projection_accumulator.len(),
            native_area,
        )?;
        check_len("mean_amplitude", self.mean_amplitude.len(), native_area)?;
        check_len("phase_node_ids", self.phase_node_ids.len(), output_area)?;
        check_len(
            "phase_angles",
            self.phase_angles.len(),
            output_area
                .checked_mul(self.ordered_date_indices.len())
                .ok_or_else(|| invalid("phase angle dimensions overflow usize"))?,
        )?;
        check_len(
            "support_bits",
            self.support_bits.len(),
            output_area
                .checked_mul(bits_to_bytes(self.support_bits_per_output)?)
                .ok_or_else(|| invalid("support dimensions overflow usize"))?,
        )?;
        check_len(
            "native_validity_bits",
            self.native_validity_bits.len(),
            native_area.div_ceil(8),
        )?;
        check_len(
            "selected_eigenvalue",
            self.selected_eigenvalue.len(),
            output_area,
        )?;
        check_len("eigen_gap", self.eigen_gap.len(), output_area)?;
        check_len("status", self.status.len(), output_area)?;
        if self
            .nearest_output_map
            .iter()
            .any(|&index| usize::try_from(index).map_or(true, |i| i >= output_area))
        {
            return Err(invalid("nearest_output_map contains an out-of-range index"));
        }
        Ok(())
    }
}

/// A checked covariance replay artifact loaded from HDF5.
#[derive(Debug, Clone, PartialEq)]
pub struct CovarianceOperatorArtifact {
    /// Artifact metadata and external source identity.
    pub metadata: CovarianceOperatorMetadata,
    /// Blocks ordered by deterministic block ID.
    pub blocks: Vec<CovarianceOperatorBlock>,
}

/// Incremental HDF5 scratch writer for block-indexed covariance replay state.
#[derive(Debug)]
pub struct CovarianceOperatorWriter {
    file: hdf5::File,
    metadata: CovarianceOperatorMetadata,
}

impl CovarianceOperatorWriter {
    /// Create an incomplete scratch artifact and persist its checked registries.
    pub fn create(path: impl AsRef<Path>, metadata: &CovarianceOperatorMetadata) -> Result<Self> {
        metadata.validate()?;
        let file = hdf5::File::create(path)?;
        write_metadata(&file, metadata)?;
        write_registries(&file)?;
        file.create_group("blocks")?;
        file.new_attr::<u8>().create("complete")?.write_scalar(&0)?;
        file.flush()?;
        Ok(Self {
            file,
            metadata: metadata.clone(),
        })
    }

    /// Append one validated block without constructing expanded incidence tensors.
    pub fn write_block(&mut self, block: &CovarianceOperatorBlock) -> Result<()> {
        block.validate(self.metadata.gauge_date_index)?;
        let group_name = format!("blocks/{:020}", block.block_id);
        if self.file.link_exists(&group_name) {
            return Err(invalid(format!(
                "duplicate covariance operator block {}",
                block.block_id
            )));
        }
        let group = self.file.create_group(&group_name)?;
        write_block(&group, block)?;
        self.file.flush()?;
        Ok(())
    }

    /// Mark the artifact complete and flush all block data.
    pub fn finish(self) -> Result<()> {
        self.file.attr("complete")?.write_scalar(&1u8)?;
        self.file.flush()?;
        Ok(())
    }
}

/// Read and validate a complete covariance replay operator artifact.
pub fn read_covariance_operator(path: impl AsRef<Path>) -> Result<CovarianceOperatorArtifact> {
    let file = hdf5::File::open(path)?;
    if file.attr("complete")?.read_scalar::<u8>()? != 1 {
        return Err(invalid(
            "covariance operator scratch artifact is incomplete",
        ));
    }
    validate_registries(&file)?;
    let metadata = read_metadata(&file)?;
    metadata.validate()?;

    let blocks_group = file.group("blocks")?;
    let mut names = blocks_group.member_names()?;
    names.sort_unstable();
    let mut blocks = Vec::with_capacity(names.len());
    for name in names {
        let group = blocks_group.group(&name)?;
        let block = read_block(&group)?;
        if name != format!("{:020}", block.block_id) {
            return Err(invalid(format!(
                "covariance operator block group {name} does not match block ID {}",
                block.block_id
            )));
        }
        block.validate(metadata.gauge_date_index)?;
        blocks.push(block);
    }
    Ok(CovarianceOperatorArtifact { metadata, blocks })
}

fn write_metadata(file: &hdf5::File, metadata: &CovarianceOperatorMetadata) -> Result<()> {
    write_scalar_attr(file, "schema_version", metadata.schema_version)?;
    write_scalar_attr(file, "method_version", metadata.method_version)?;
    write_scalar_attr(file, "gauge_date_index", metadata.gauge_date_index)?;
    write_scalar_attr(file, "replay_status", metadata.replay_status.code())?;
    write_scalar_attr(file, "stitched_status", metadata.stitched_status.code())?;
    write_scalar_attr(
        file,
        "downstream_inference_status",
        metadata.downstream_inference_status.code(),
    )?;
    write_string(file, "method", &metadata.method)?;
    write_string(file, "crate_version", &metadata.crate_version)?;
    write_optional_string(file, "producer_commit", metadata.producer_commit.as_deref())?;
    write_string(
        file,
        "normalized_config_digest",
        &metadata.normalized_config_digest,
    )?;
    write_string(file, "kernel_digest", &metadata.kernel_digest)?;

    let source = file.create_group("source")?;
    write_optional_string(
        &source,
        "manifest_digest",
        metadata.source.manifest_digest.as_deref(),
    )?;
    write_optional_string(&source, "provider", metadata.source.provider.as_deref())?;
    write_optional_string(
        &source,
        "provider_version",
        metadata.source.provider_version.as_deref(),
    )?;
    write_optional_string(&source, "model", metadata.source.model.as_deref())?;
    write_optional_string(
        &source,
        "model_version",
        metadata.source.model_version.as_deref(),
    )?;
    write_optional_string(
        &source,
        "model_receipt_digest",
        metadata.source.model_receipt_digest.as_deref(),
    )
}

fn read_metadata(file: &hdf5::File) -> Result<CovarianceOperatorMetadata> {
    let source = file.group("source")?;
    Ok(CovarianceOperatorMetadata {
        schema_version: read_scalar_attr(file, "schema_version")?,
        method: read_string(file, "method")?,
        method_version: read_scalar_attr(file, "method_version")?,
        crate_version: read_string(file, "crate_version")?,
        producer_commit: read_optional_string(file, "producer_commit")?,
        gauge_date_index: read_scalar_attr(file, "gauge_date_index")?,
        normalized_config_digest: read_string(file, "normalized_config_digest")?,
        kernel_digest: read_string(file, "kernel_digest")?,
        source: SourceReplayIdentity {
            manifest_digest: read_optional_string(&source, "manifest_digest")?,
            provider: read_optional_string(&source, "provider")?,
            provider_version: read_optional_string(&source, "provider_version")?,
            model: read_optional_string(&source, "model")?,
            model_version: read_optional_string(&source, "model_version")?,
            model_receipt_digest: read_optional_string(&source, "model_receipt_digest")?,
        },
        replay_status: CovarianceReplayStatus::from_code(read_scalar_attr(file, "replay_status")?)?,
        stitched_status: StitchedCovarianceStatus::from_code(read_scalar_attr(
            file,
            "stitched_status",
        )?)?,
        downstream_inference_status: DownstreamInferenceStatus::from_code(read_scalar_attr(
            file,
            "downstream_inference_status",
        )?)?,
    })
}

fn write_registries(file: &hdf5::File) -> Result<()> {
    let group = file.create_group("registries")?;
    for (name, registry) in [
        ("method", COVARIANCE_METHOD_REGISTRY),
        ("estimator_branch", COVARIANCE_ESTIMATOR_BRANCH_REGISTRY),
        ("operator_status", COVARIANCE_OPERATOR_STATUS_REGISTRY),
        ("replay_status", COVARIANCE_REPLAY_STATUS_REGISTRY),
        ("stitched_status", STITCHED_COVARIANCE_STATUS_REGISTRY),
        (
            "downstream_inference_status",
            DOWNSTREAM_INFERENCE_STATUS_REGISTRY,
        ),
    ] {
        let codes: Vec<_> = registry.iter().map(|entry| entry.code).collect();
        let names = registry
            .iter()
            .map(|entry| entry.name)
            .collect::<Vec<_>>()
            .join("\n");
        write_chunked_1d(&group, &format!("{name}_codes"), &codes)?;
        write_string(&group, &format!("{name}_names"), &names)?;
    }
    Ok(())
}

fn validate_registries(file: &hdf5::File) -> Result<()> {
    let group = file.group("registries")?;
    for (name, registry) in [
        ("method", COVARIANCE_METHOD_REGISTRY),
        ("estimator_branch", COVARIANCE_ESTIMATOR_BRANCH_REGISTRY),
        ("operator_status", COVARIANCE_OPERATOR_STATUS_REGISTRY),
        ("replay_status", COVARIANCE_REPLAY_STATUS_REGISTRY),
        ("stitched_status", STITCHED_COVARIANCE_STATUS_REGISTRY),
        (
            "downstream_inference_status",
            DOWNSTREAM_INFERENCE_STATUS_REGISTRY,
        ),
    ] {
        let expected_codes: Vec<_> = registry.iter().map(|entry| entry.code).collect();
        let expected_names = registry
            .iter()
            .map(|entry| entry.name)
            .collect::<Vec<_>>()
            .join("\n");
        let actual_codes: Vec<u16> = group.dataset(&format!("{name}_codes"))?.read_raw()?;
        let actual_names = read_string(&group, &format!("{name}_names"))?;
        if actual_codes != expected_codes || actual_names != expected_names {
            return Err(invalid(format!("covariance {name} registry mismatch")));
        }
    }
    Ok(())
}

fn write_block(group: &Group, block: &CovarianceOperatorBlock) -> Result<()> {
    write_scalar_attr(group, "block_id", block.block_id)?;
    write_scalar_attr(group, "generation", block.generation)?;
    write_scalar_attr(group, "reference_date_index", block.reference_date_index)?;
    write_scalar_attr(
        group,
        "support_bits_per_output",
        block.support_bits_per_output,
    )?;
    write_scalar_attr(group, "estimator_branch", block.estimator_branch.code())?;
    write_string(group, "burst_id", &block.burst_id)?;
    write_grid(group, "native_grid", block.native_grid)?;
    write_grid(group, "output_grid", block.output_grid)?;

    write_chunked_1d(group, "ordered_date_indices", &block.ordered_date_indices)?;
    write_chunked_1d(group, "source_ids", &block.source_ids)?;
    write_chunked_1d(group, "phase_node_ids", &block.phase_node_ids)?;
    write_chunked_1d(group, "compressed_node_ids", &block.compressed_node_ids)?;
    write_chunked_1d(group, "carry_parent_ids", &block.carry_parent_ids)?;

    let native_shape = (
        usize::try_from(block.native_grid.rows).map_err(|_| invalid("native rows exceed usize"))?,
        usize::try_from(block.native_grid.cols)
            .map_err(|_| invalid("native columns exceed usize"))?,
    );
    let output_shape = (
        usize::try_from(block.output_grid.rows).map_err(|_| invalid("output rows exceed usize"))?,
        usize::try_from(block.output_grid.cols)
            .map_err(|_| invalid("output columns exceed usize"))?,
    );
    write_chunked_2d(
        group,
        "nearest_output_map",
        native_shape,
        &block.nearest_output_map,
    )?;
    write_chunked_2d(
        group,
        "phase_angles",
        (
            output_shape.0 * output_shape.1,
            block.ordered_date_indices.len(),
        ),
        &block.phase_angles,
    )?;
    write_chunked_2d(
        group,
        "compressed_raster",
        native_shape,
        &block.compressed_raster,
    )?;
    write_chunked_2d(
        group,
        "projection_accumulator",
        native_shape,
        &block.projection_accumulator,
    )?;
    write_chunked_2d(group, "mean_amplitude", native_shape, &block.mean_amplitude)?;
    write_chunked_2d(
        group,
        "support_bits",
        (
            output_shape.0 * output_shape.1,
            bits_to_bytes(block.support_bits_per_output)?,
        ),
        &block.support_bits,
    )?;
    write_chunked_1d(group, "native_validity_bits", &block.native_validity_bits)?;
    write_chunked_2d(
        group,
        "selected_eigenvalue",
        output_shape,
        &block.selected_eigenvalue,
    )?;
    write_chunked_2d(group, "eigen_gap", output_shape, &block.eigen_gap)?;
    let statuses: Vec<_> = block.status.iter().map(|status| status.code()).collect();
    write_chunked_2d(group, "status", output_shape, &statuses)
}

fn read_block(group: &Group) -> Result<CovarianceOperatorBlock> {
    let status_codes: Vec<u16> = group.dataset("status")?.read_raw()?;
    let status = status_codes
        .into_iter()
        .map(CovarianceOperatorStatus::from_code)
        .collect::<Result<_>>()?;
    Ok(CovarianceOperatorBlock {
        burst_id: read_string(group, "burst_id")?,
        block_id: read_scalar_attr(group, "block_id")?,
        generation: read_scalar_attr(group, "generation")?,
        native_grid: read_grid(group, "native_grid")?,
        output_grid: read_grid(group, "output_grid")?,
        reference_date_index: read_scalar_attr(group, "reference_date_index")?,
        ordered_date_indices: group.dataset("ordered_date_indices")?.read_raw()?,
        source_ids: group.dataset("source_ids")?.read_raw()?,
        phase_node_ids: group.dataset("phase_node_ids")?.read_raw()?,
        compressed_node_ids: group.dataset("compressed_node_ids")?.read_raw()?,
        carry_parent_ids: group.dataset("carry_parent_ids")?.read_raw()?,
        nearest_output_map: group.dataset("nearest_output_map")?.read_raw()?,
        phase_angles: group.dataset("phase_angles")?.read_raw()?,
        compressed_raster: group.dataset("compressed_raster")?.read_raw()?,
        projection_accumulator: group.dataset("projection_accumulator")?.read_raw()?,
        mean_amplitude: group.dataset("mean_amplitude")?.read_raw()?,
        support_bits_per_output: read_scalar_attr(group, "support_bits_per_output")?,
        support_bits: group.dataset("support_bits")?.read_raw()?,
        native_validity_bits: group.dataset("native_validity_bits")?.read_raw()?,
        estimator_branch: CovarianceEstimatorBranch::from_code(read_scalar_attr(
            group,
            "estimator_branch",
        )?)?,
        selected_eigenvalue: group.dataset("selected_eigenvalue")?.read_raw()?,
        eigen_gap: group.dataset("eigen_gap")?.read_raw()?,
        status,
    })
}

fn write_grid(group: &Group, name: &str, grid: CovarianceOperatorGrid) -> Result<()> {
    let grid_group = group.create_group(name)?;
    write_scalar_attr(&grid_group, "row_start", grid.row_start)?;
    write_scalar_attr(&grid_group, "col_start", grid.col_start)?;
    write_scalar_attr(&grid_group, "rows", grid.rows)?;
    write_scalar_attr(&grid_group, "cols", grid.cols)?;
    write_scalar_attr(&grid_group, "stride_y", grid.stride_y)?;
    write_scalar_attr(&grid_group, "stride_x", grid.stride_x)
}

fn read_grid(group: &Group, name: &str) -> Result<CovarianceOperatorGrid> {
    let grid_group = group.group(name)?;
    Ok(CovarianceOperatorGrid {
        row_start: read_scalar_attr(&grid_group, "row_start")?,
        col_start: read_scalar_attr(&grid_group, "col_start")?,
        rows: read_scalar_attr(&grid_group, "rows")?,
        cols: read_scalar_attr(&grid_group, "cols")?,
        stride_y: read_scalar_attr(&grid_group, "stride_y")?,
        stride_x: read_scalar_attr(&grid_group, "stride_x")?,
    })
}

fn write_scalar_attr<T: H5Type>(group: &Group, name: &str, value: T) -> Result<()> {
    group.new_attr::<T>().create(name)?.write_scalar(&value)?;
    Ok(())
}

fn read_scalar_attr<T: H5Type>(group: &Group, name: &str) -> Result<T> {
    Ok(group.attr(name)?.read_scalar()?)
}

fn write_string(group: &Group, name: &str, value: &str) -> Result<()> {
    group
        .new_dataset_builder()
        .with_data(value.as_bytes())
        .create(name)?;
    Ok(())
}

fn read_string(group: &Group, name: &str) -> Result<String> {
    String::from_utf8(group.dataset(name)?.read_raw()?)
        .map_err(|error| invalid(format!("{name} is not UTF-8: {error}")))
}

fn write_optional_string(group: &Group, name: &str, value: Option<&str>) -> Result<()> {
    write_string(group, name, value.unwrap_or_default())
}

fn read_optional_string(group: &Group, name: &str) -> Result<Option<String>> {
    let value = read_string(group, name)?;
    Ok((!value.is_empty()).then_some(value))
}

fn write_chunked_1d<T: H5Type>(group: &Group, name: &str, values: &[T]) -> Result<()> {
    let builder = group.new_dataset_builder().with_data(values);
    if values.is_empty() {
        builder.create(name)?;
    } else {
        builder.chunk((values.len().min(4096),)).create(name)?;
    }
    Ok(())
}

fn write_chunked_2d<T: H5Type>(
    group: &Group,
    name: &str,
    shape: (usize, usize),
    values: &[T],
) -> Result<()> {
    let view = ArrayView2::from_shape(shape, values)
        .map_err(|error| invalid(format!("{name} shape: {error}")))?;
    group
        .new_dataset_builder()
        .with_data(view)
        .chunk((shape.0.min(64), shape.1.min(256)))
        .create(name)?;
    Ok(())
}

fn bits_to_bytes(bits: u32) -> Result<usize> {
    if bits == 0 {
        return Err(invalid("support_bits_per_output must be positive"));
    }
    usize::try_from(bits.div_ceil(8)).map_err(|_| invalid("support bit count exceeds usize"))
}

fn check_len(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(invalid(format!(
            "covariance operator {name} length {actual} != {expected}"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> IoError {
    IoError::Shape(message.into())
}

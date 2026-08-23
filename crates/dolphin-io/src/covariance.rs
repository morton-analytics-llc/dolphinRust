//! Block-indexed HDF5 persistence for the sequential covariance replay operator.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use hdf5::{Group, H5Type, LinkType};
use ndarray::ArrayView2;
use num_complex::Complex64;

use crate::{IoError, Result};

/// HDF5 schema version for covariance replay operators.
pub const COVARIANCE_OPERATOR_SCHEMA_VERSION: u16 = 1;
/// Stable method name for the source-keyed sequential replay DAG.
pub const COVARIANCE_OPERATOR_METHOD: &str = "sequential_source_dag_v1";
/// Numeric version of [`COVARIANCE_OPERATOR_METHOD`].
pub const COVARIANCE_OPERATOR_METHOD_VERSION: u16 = 1;

const COVARIANCE_ROOT_MEMBERS: &[&str] = &[
    "method",
    "crate_version",
    "producer_commit",
    "normalized_config_digest",
    "kernel_digest",
    "source",
    "registries",
    "blocks",
];
const COVARIANCE_ROOT_ATTRIBUTES: &[&str] = &[
    "schema_version",
    "method_version",
    "gauge_date_index",
    "replay_status",
    "stitched_status",
    "calibration_status",
    "downstream_inference_status",
    "complete",
];
const COVARIANCE_SOURCE_MEMBERS: &[&str] = &[
    "manifest_digest",
    "provider",
    "provider_version",
    "model",
    "model_version",
    "model_receipt_digest",
];
const COVARIANCE_REGISTRY_MEMBERS: &[&str] = &[
    "method_codes",
    "method_names",
    "estimator_branch_codes",
    "estimator_branch_names",
    "phase_component_kind_codes",
    "phase_component_kind_names",
    "support_ordering_codes",
    "support_ordering_names",
    "operator_status_codes",
    "operator_status_names",
    "replay_status_codes",
    "replay_status_names",
    "stitched_status_codes",
    "stitched_status_names",
    "calibration_status_codes",
    "calibration_status_names",
    "downstream_inference_status_codes",
    "downstream_inference_status_names",
];
const COVARIANCE_BLOCK_ATTRIBUTES: &[&str] = &[
    "block_id",
    "generation",
    "reference_date_index",
    "support_bits_per_output",
    "estimator_branch",
    "branch_tolerance",
];
const COVARIANCE_GRID_ATTRIBUTES: &[&str] = &[
    "row_start",
    "col_start",
    "rows",
    "cols",
    "stride_y",
    "stride_x",
];
const COVARIANCE_RECT_SUPPORT_ATTRIBUTES: &[&str] =
    &["half_window_rows", "half_window_cols", "ordering"];
const BLOCK_NAME_BUDGET_BYTES: u64 = 64;
const TOPOLOGY_WORKSPACE_MULTIPLIER: u64 = 8;

const COVARIANCE_BLOCK_MEMBERS: &[&str] = &[
    "burst_id",
    "native_grid",
    "output_grid",
    "owned_output_grid",
    "rect_support",
    "source_date_indices",
    "ordered_date_indices",
    "source_ids",
    "phase_node_ids",
    "compressed_node_ids",
    "carry_parent_ids",
    "phase_component_kinds",
    "phase_component_ids",
    "nearest_output_map",
    "phase_angles",
    "compressed_raster",
    "compressed_status",
    "projection_accumulator",
    "mean_amplitude",
    "support_bits",
    "native_validity_bits",
    "selected_eigenvalue",
    "eigen_gap",
    "status",
];

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

/// Kind of one ordered component in a block's combined phase solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum CovariancePhaseComponentKind {
    /// The exact acquisition-zero gauge in the first block.
    GaugeDate = 0,
    /// A new real acquisition retained in the public date series.
    RetainedDate = 1,
    /// A compressed parent block prepended to a later ministack.
    CompressedParent = 2,
}

impl CovariancePhaseComponentKind {
    const fn code(self) -> u16 {
        self as u16
    }

    fn from_code(code: u16) -> Result<Self> {
        match code {
            0 => Ok(Self::GaugeDate),
            1 => Ok(Self::RetainedDate),
            2 => Ok(Self::CompressedParent),
            _ => Err(invalid(format!(
                "unknown covariance phase component kind {code}"
            ))),
        }
    }
}

/// Stable combined-phase-component-kind registry.
pub const COVARIANCE_PHASE_COMPONENT_KIND_REGISTRY: &[CovarianceRegistryEntry] = &[
    CovarianceRegistryEntry {
        code: CovariancePhaseComponentKind::GaugeDate as u16,
        name: "gauge_date",
    },
    CovarianceRegistryEntry {
        code: CovariancePhaseComponentKind::RetainedDate as u16,
        name: "retained_date",
    },
    CovarianceRegistryEntry {
        code: CovariancePhaseComponentKind::CompressedParent as u16,
        name: "compressed_parent",
    },
];

/// Stable ordering used to encode a clamped rectangular support window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum CovarianceSupportOrdering {
    /// Row-major nominal window positions with inward clamping at raster edges.
    RowMajorInwardClampV1 = 1,
}

impl CovarianceSupportOrdering {
    const fn code(self) -> u16 {
        self as u16
    }

    fn from_code(code: u16) -> Result<Self> {
        match code {
            1 => Ok(Self::RowMajorInwardClampV1),
            _ => Err(invalid(format!(
                "unknown covariance support ordering {code}"
            ))),
        }
    }
}

/// Stable rectangular-support-ordering registry.
pub const COVARIANCE_SUPPORT_ORDERING_REGISTRY: &[CovarianceRegistryEntry] =
    &[CovarianceRegistryEntry {
        code: CovarianceSupportOrdering::RowMajorInwardClampV1 as u16,
        name: "row_major_inward_clamp_v1",
    }];

/// Geometry and ordering of each bit-packed rectangular support window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CovarianceRectSupport {
    /// Nominal half-window size in native rows.
    pub half_window_rows: u32,
    /// Nominal half-window size in native columns.
    pub half_window_cols: u32,
    /// Stable support-position ordering and edge-clamp rule.
    pub ordering: CovarianceSupportOrdering,
}

impl CovarianceRectSupport {
    fn bit_count(self) -> Result<usize> {
        let rows = u64::from(self.half_window_rows)
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid("rect support row count overflow"))?;
        let cols = u64::from(self.half_window_cols)
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid("rect support column count overflow"))?;
        usize::try_from(
            rows.checked_mul(cols)
                .ok_or_else(|| invalid("rect support area overflow"))?,
        )
        .map_err(|_| invalid("rect support area exceeds usize"))
    }
}

/// One ordered component of the production combined phase solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CovariancePhaseComponent {
    /// Component kind, which determines the identity namespace.
    pub kind: CovariancePhaseComponentKind,
    /// Global date index or compact parent block ID according to [`Self::kind`].
    pub id: u64,
}

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

/// Calibration status of the source model bound to an operator artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum CovarianceCalibrationStatus {
    /// The caller-supplied source model has not passed issue #54 calibration.
    Uncalibrated = 0,
}

impl CovarianceCalibrationStatus {
    const fn code(self) -> u16 {
        self as u16
    }

    fn from_code(code: u16) -> Result<Self> {
        match code {
            0 => Ok(Self::Uncalibrated),
            _ => Err(invalid(format!(
                "unknown covariance calibration status {code}"
            ))),
        }
    }
}

/// Stable source-model-calibration-status registry.
pub const COVARIANCE_CALIBRATION_STATUS_REGISTRY: &[CovarianceRegistryEntry] =
    &[CovarianceRegistryEntry {
        code: CovarianceCalibrationStatus::Uncalibrated as u16,
        name: "uncalibrated",
    }];

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
        for (name, value) in [
            ("source manifest", self.manifest_digest.as_deref()),
            ("source model receipt", self.model_receipt_digest.as_deref()),
        ] {
            if let Some(value) = value {
                ensure_valid(
                    is_sha256_digest(value),
                    match name {
                        "source manifest" => {
                            "source manifest digest is not a strong SHA-256 digest"
                        }
                        _ => "source model receipt digest is not a strong SHA-256 digest",
                    },
                )?;
            }
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
    /// Source-model calibration status.
    pub calibration_status: CovarianceCalibrationStatus,
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
            calibration_status: CovarianceCalibrationStatus::Uncalibrated,
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
        for (name, value) in [
            (
                "normalized_config_digest",
                self.normalized_config_digest.as_str(),
            ),
            ("kernel_digest", self.kernel_digest.as_str()),
        ] {
            ensure_valid(
                is_sha256_digest(value),
                match name {
                    "normalized_config_digest" => {
                        "normalized_config_digest is not a strong SHA-256 digest"
                    }
                    _ => "kernel_digest is not a strong SHA-256 digest",
                },
            )?;
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

    fn contains(self, other: Self) -> bool {
        if self.stride_y != other.stride_y || self.stride_x != other.stride_x {
            return false;
        }
        let self_row_stop = self.row_start.checked_add(u64::from(self.rows));
        let self_col_stop = self.col_start.checked_add(u64::from(self.cols));
        let other_row_stop = other.row_start.checked_add(u64::from(other.rows));
        let other_col_stop = other.col_start.checked_add(u64::from(other.cols));
        match (self_row_stop, self_col_stop, other_row_stop, other_col_stop) {
            (Some(sr), Some(sc), Some(or), Some(oc)) => {
                other.row_start >= self.row_start
                    && other.col_start >= self.col_start
                    && or <= sr
                    && oc <= sc
            }
            _ => false,
        }
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
    /// Full looked phase replay grid, including any tile halo needed by owned outputs.
    pub output_grid: CovarianceOperatorGrid,
    /// Public output rectangle owned by this record; must be contained in
    /// [`Self::output_grid`]. Halo phase nodes are replay dependencies only.
    pub owned_output_grid: CovarianceOperatorGrid,
    /// Fixed rectangular support geometry used to encode [`Self::support_bits`].
    pub rect_support: CovarianceRectSupport,
    /// Positive tolerance separating a fixed estimator branch from a tie.
    pub branch_tolerance: f64,
    /// Reference date index used by the block.
    pub reference_date_index: u32,
    /// Ordered raw real-acquisition indices in each native source vector.
    pub source_date_indices: Vec<u32>,
    /// Ordered global acquisition-date indices represented by phase angles.
    pub ordered_date_indices: Vec<u32>,
    /// Deterministic primitive-source IDs, one per native pixel.
    pub source_ids: Vec<u64>,
    /// Deterministic phase-node IDs, one per output pixel.
    pub phase_node_ids: Vec<u64>,
    /// Deterministic compressed-node IDs, one per native pixel.
    pub compressed_node_ids: Vec<u64>,
    /// Ordered compact parent block IDs from earlier generations.
    pub carry_parent_ids: Vec<u64>,
    /// Native-pixel to nearest looked-output mapping.
    pub nearest_output_map: Vec<u32>,
    /// Ordered carried-plus-real component map for the linked solution.
    pub phase_components: Vec<CovariancePhaseComponent>,
    /// Linked phase angles in output-pixel-major, component-minor order.
    pub phase_angles: Vec<f64>,
    /// Complex compressed SLC raster.
    pub compressed_raster: Vec<Complex64>,
    /// Stable compression status, one per native pixel.
    pub compressed_status: Vec<CovarianceOperatorStatus>,
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
        ensure_valid(!self.burst_id.is_empty(), "empty covariance burst_id")?;
        ensure_valid(!self.source_date_indices.is_empty(), "no source dates")?;
        ensure_valid(!self.ordered_date_indices.is_empty(), "no output dates")?;
        ensure_valid(
            self.reference_date_index == gauge_date_index,
            "gauge mismatch",
        )?;
        ensure_valid(
            strictly_increasing(&self.source_date_indices)
                && strictly_increasing(&self.ordered_date_indices),
            "unordered covariance source/output dates",
        )?;
        ensure_valid(
            self.source_date_indices == self.ordered_date_indices,
            "source/output date maps differ",
        )?;
        ensure_valid(
            strictly_increasing(&self.carry_parent_ids)
                && self
                    .carry_parent_ids
                    .iter()
                    .all(|&parent| parent < self.block_id),
            "carried parent IDs are not ordered predecessors",
        )?;
        ensure_valid(
            !self.carry_parent_ids.is_empty()
                || self.source_date_indices.first() == Some(&gauge_date_index),
            "first block source omits gauge date",
        )?;
        ensure_valid(
            self.carry_parent_ids.is_empty()
                || !self.source_date_indices.contains(&gauge_date_index),
            "later block source repeats gauge date",
        )?;
        let parents = self
            .carry_parent_ids
            .iter()
            .map(|&id| (CovariancePhaseComponentKind::CompressedParent, id));
        let dates = self.source_date_indices.iter().map(|&date| {
            let kind = match date == gauge_date_index {
                true => CovariancePhaseComponentKind::GaugeDate,
                false => CovariancePhaseComponentKind::RetainedDate,
            };
            (kind, u64::from(date))
        });
        ensure_valid(
            self.phase_components
                .iter()
                .map(|component| (component.kind, component.id))
                .eq(parents.chain(dates)),
            "covariance operator phase component map does not match carried parents and source dates",
        )?;
        let (native_area, output_area) = self.validate_grids()?;
        for (name, actual) in [
            ("source_ids", self.source_ids.len()),
            ("compressed_node_ids", self.compressed_node_ids.len()),
            ("nearest_output_map", self.nearest_output_map.len()),
            ("compressed_raster", self.compressed_raster.len()),
            ("compressed_status", self.compressed_status.len()),
            ("projection_accumulator", self.projection_accumulator.len()),
            ("mean_amplitude", self.mean_amplitude.len()),
        ] {
            check_len(name, actual, native_area)?;
        }
        for (name, actual) in [
            ("phase_node_ids", self.phase_node_ids.len()),
            ("selected_eigenvalue", self.selected_eigenvalue.len()),
            ("eigen_gap", self.eigen_gap.len()),
            ("status", self.status.len()),
        ] {
            check_len(name, actual, output_area)?;
        }
        check_len(
            "phase_angles",
            self.phase_angles.len(),
            output_area
                .checked_mul(self.phase_components.len())
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
        ensure_valid(
            self.nearest_output_map
                .iter()
                .all(|&index| usize::try_from(index).is_ok_and(|i| i < output_area)),
            "nearest_output_map contains an out-of-range index",
        )?;
        self.validate_support(native_area, output_area)?;
        self.validate_numeric_state(native_area, output_area)?;
        Ok(())
    }

    fn validate_grids(&self) -> Result<(usize, usize)> {
        let native_area = self.native_grid.area()?;
        let output_area = self.output_grid.area()?;
        self.owned_output_grid.area()?;
        ensure_valid(
            self.native_grid.stride_y == 1 && self.native_grid.stride_x == 1,
            "native grid stride must be one",
        )?;
        ensure_valid(
            self.output_grid.rows == self.native_grid.rows / self.output_grid.stride_y
                && self.output_grid.cols == self.native_grid.cols / self.output_grid.stride_x,
            "output grid shape does not match native grid and strides",
        )?;
        ensure_valid(
            self.output_grid
                .row_start
                .checked_mul(u64::from(self.output_grid.stride_y))
                == Some(self.native_grid.row_start)
                && self
                    .output_grid
                    .col_start
                    .checked_mul(u64::from(self.output_grid.stride_x))
                    == Some(self.native_grid.col_start),
            "native and output grid origins are not stride-aligned",
        )?;
        ensure_valid(
            self.native_grid
                .row_start
                .checked_add(u64::from(self.native_grid.rows))
                .is_some()
                && self
                    .native_grid
                    .col_start
                    .checked_add(u64::from(self.native_grid.cols))
                    .is_some(),
            "native grid extent overflows global coordinates",
        )?;
        ensure_valid(
            self.output_grid.contains(self.owned_output_grid),
            "owned output grid is not contained in replay output grid",
        )?;
        Ok((native_area, output_area))
    }

    fn validate_support(&self, native_area: usize, _output_area: usize) -> Result<()> {
        let support_bit_count = self.rect_support.bit_count()?;
        let window_rows = self
            .rect_support
            .half_window_rows
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid("Rect window row count overflow"))?;
        let window_cols = self
            .rect_support
            .half_window_cols
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid("Rect window column count overflow"))?;
        ensure_valid(
            window_rows <= self.native_grid.rows && window_cols <= self.native_grid.cols,
            "Rect window exceeds the native grid",
        )?;
        ensure_valid(
            usize::try_from(self.support_bits_per_output) == Ok(support_bit_count),
            "support bit count does not match Rect geometry",
        )?;
        let bytes_per_output = bits_to_bytes(self.support_bits_per_output)?;
        let trailing_mask = trailing_bit_mask(support_bit_count);
        for (output_index, status) in self.status.iter().enumerate() {
            let start = output_index * bytes_per_output;
            let row = &self.support_bits[start..start + bytes_per_output];
            ensure_valid(
                row.last().is_some_and(|byte| byte & trailing_mask == 0),
                "support bits set positions outside Rect geometry",
            )?;
            let nonempty = row.iter().any(|&byte| byte != 0);
            let output_row = output_index / self.output_grid.cols as usize;
            let output_col = output_index % self.output_grid.cols as usize;
            let row_start = window_origin(
                output_row,
                self.rect_support.half_window_rows as usize,
                self.output_grid.stride_y as usize,
                self.native_grid.rows as usize,
            );
            let col_start = window_origin(
                output_col,
                self.rect_support.half_window_cols as usize,
                self.output_grid.stride_x as usize,
                self.native_grid.cols as usize,
            );
            for slot in 0..support_bit_count {
                let native_row = row_start + slot / window_cols as usize;
                let native_col = col_start + slot % window_cols as usize;
                let native_index = native_row * self.native_grid.cols as usize + native_col;
                ensure_valid(
                    packed_bit(row, slot) == packed_bit(&self.native_validity_bits, native_index),
                    "support bits do not match native validity and Rect clamp",
                )?;
            }
            match status {
                CovarianceOperatorStatus::Valid => {
                    ensure_valid(nonempty, "valid output has empty support")?
                }
                CovarianceOperatorStatus::NoContributor => {
                    ensure_valid(!nonempty, "no-contributor output has nonempty support")?
                }
                _ => {}
            }
        }
        let row_looks = (self.native_grid.rows as usize / self.output_grid.rows as usize).max(1);
        let col_looks = (self.native_grid.cols as usize / self.output_grid.cols as usize).max(1);
        for (native_index, &stored) in self.nearest_output_map.iter().enumerate() {
            let native_row = native_index / self.native_grid.cols as usize;
            let native_col = native_index % self.native_grid.cols as usize;
            let output_row = (native_row / row_looks).min(self.output_grid.rows as usize - 1);
            let output_col = (native_col / col_looks).min(self.output_grid.cols as usize - 1);
            let expected = output_row * self.output_grid.cols as usize + output_col;
            ensure_valid(
                usize::try_from(stored) == Ok(expected),
                "nearest-output map differs from production repeat/clamp mapping",
            )?;
        }
        ensure_valid(
            packed_trailing_bits_are_zero(&self.native_validity_bits, native_area),
            "native validity sets bits outside the native grid",
        )
    }

    fn validate_numeric_state(&self, native_area: usize, output_area: usize) -> Result<()> {
        ensure_valid(
            self.branch_tolerance.is_finite() && self.branch_tolerance > 0.0,
            "branch tolerance must be finite and positive",
        )?;
        let component_count = self.phase_components.len();
        for output_index in 0..output_area {
            if self.status[output_index] != CovarianceOperatorStatus::Valid {
                continue;
            }
            let start = output_index * component_count;
            let phases = &self.phase_angles[start..start + component_count];
            ensure_valid(
                phases.iter().all(|phase| phase.is_finite()),
                "valid output has nonfinite phase state",
            )?;
            ensure_valid(phases[0] == 0.0, "valid output reference phase is not zero")?;
            ensure_valid(
                self.selected_eigenvalue[output_index].is_finite()
                    && self.selected_eigenvalue[output_index] > 0.0,
                "valid output has nonpositive eigenvalue",
            )?;
            ensure_valid(
                self.eigen_gap[output_index].is_finite()
                    && self.eigen_gap[output_index] > self.branch_tolerance,
                "valid output eigen gap is at or below branch tolerance",
            )?;
        }
        for native_index in 0..native_area {
            let is_valid = packed_bit(&self.native_validity_bits, native_index);
            let status = self.compressed_status[native_index];
            ensure_valid(
                (status == CovarianceOperatorStatus::Masked) != is_valid,
                "native mask and compressed status disagree",
            )?;
            ensure_valid(
                !matches!(
                    status,
                    CovarianceOperatorStatus::NoContributor
                        | CovarianceOperatorStatus::SingularLocalInformation
                ),
                "compressed node has an unsupported status",
            )?;
            if status != CovarianceOperatorStatus::Valid {
                continue;
            }
            let compressed = self.compressed_raster[native_index];
            let projection = self.projection_accumulator[native_index];
            let amplitude = self.mean_amplitude[native_index];
            let finite = compressed.re.is_finite()
                && compressed.im.is_finite()
                && projection.re.is_finite()
                && projection.im.is_finite()
                && amplitude.is_finite();
            ensure_valid(finite, "valid compressed node has nonfinite numeric state")?;
            ensure_valid(
                amplitude > self.branch_tolerance
                    && compressed.norm_sqr().is_finite()
                    && compressed.norm_sqr() > self.branch_tolerance.powi(2),
                "valid compressed node has amplitude at or below branch tolerance",
            )?;
            ensure_valid(
                projection.norm_sqr().is_finite()
                    && projection.norm_sqr() > self.branch_tolerance.powi(2),
                "valid compressed node has nonpositive projection",
            )?;
            ensure_valid(
                projection.arg().abs() > self.branch_tolerance,
                "valid compressed node projection phase is on the nodata branch",
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CovarianceBlockTopology {
    block_id: u64,
    generation: u32,
    burst_id: String,
    native_grid: CovarianceOperatorGrid,
    output_grid: CovarianceOperatorGrid,
    owned_output_grid: CovarianceOperatorGrid,
    rect_support: CovarianceRectSupport,
    reference_date_index: u32,
    source_date_indices: Vec<u32>,
    ordered_date_indices: Vec<u32>,
    source_ids: Vec<u64>,
    phase_node_ids: Vec<u64>,
    compressed_node_ids: Vec<u64>,
    carry_parent_ids: Vec<u64>,
    phase_components: Vec<CovariancePhaseComponent>,
}

impl From<&CovarianceOperatorBlock> for CovarianceBlockTopology {
    fn from(block: &CovarianceOperatorBlock) -> Self {
        Self {
            block_id: block.block_id,
            generation: block.generation,
            burst_id: block.burst_id.clone(),
            native_grid: block.native_grid,
            output_grid: block.output_grid,
            owned_output_grid: block.owned_output_grid,
            rect_support: block.rect_support,
            reference_date_index: block.reference_date_index,
            source_date_indices: block.source_date_indices.clone(),
            ordered_date_indices: block.ordered_date_indices.clone(),
            source_ids: block.source_ids.clone(),
            phase_node_ids: block.phase_node_ids.clone(),
            compressed_node_ids: block.compressed_node_ids.clone(),
            carry_parent_ids: block.carry_parent_ids.clone(),
            phase_components: block.phase_components.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CovarianceSourceLocation {
    block_id: u64,
    native_index: usize,
}

#[derive(Debug, Default)]
struct CovarianceTopologyState {
    blocks: BTreeMap<u64, CovarianceBlockTopology>,
    source_locations: BTreeMap<u64, CovarianceSourceLocation>,
    node_ids: BTreeSet<u64>,
}

impl CovarianceTopologyState {
    fn validate(
        &self,
        block: &CovarianceBlockTopology,
        stitched_status: StitchedCovarianceStatus,
    ) -> Result<()> {
        validate_block_topology(block, &self.blocks)?;
        validate_cross_record_topology(block, self, stitched_status)
    }

    fn insert(&mut self, block: CovarianceBlockTopology) {
        for (native_index, &source_id) in block.source_ids.iter().enumerate() {
            self.source_locations
                .entry(source_id)
                .or_insert(CovarianceSourceLocation {
                    block_id: block.block_id,
                    native_index,
                });
        }
        self.node_ids.extend(block.phase_node_ids.iter().copied());
        self.node_ids
            .extend(block.compressed_node_ids.iter().copied());
        self.blocks.insert(block.block_id, block);
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
    topology: CovarianceTopologyState,
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
            topology: CovarianceTopologyState::default(),
        })
    }

    /// Append one validated block without constructing expanded incidence tensors.
    pub fn write_block(&mut self, block: &CovarianceOperatorBlock) -> Result<()> {
        block.validate(self.metadata.gauge_date_index)?;
        let topology = CovarianceBlockTopology::from(block);
        self.topology
            .validate(&topology, self.metadata.stitched_status)?;
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
        self.topology.insert(topology);
        Ok(())
    }

    /// Mark the artifact complete and flush all block data.
    pub fn finish(self) -> Result<()> {
        validate_root_schema(&self.file)?;
        inspect_metadata_layout(&self.file)?;
        validate_registries(&self.file)?;
        ensure_valid(
            read_metadata(&self.file)? == self.metadata,
            "covariance operator metadata changed before finalization",
        )?;
        let names = block_names(&self.file)?;
        ensure_valid(
            !names.is_empty(),
            "covariance operator requires at least one block",
        )?;
        let expected_names = self
            .topology
            .blocks
            .keys()
            .map(|block_id| format!("{block_id:020}"))
            .collect::<Vec<_>>();
        ensure_valid(
            names == expected_names,
            "covariance operator block index changed before finalization",
        )?;
        for name in &names {
            inspect_block_layout(&self.file.group(&format!("blocks/{name}"))?)?;
        }
        validate_topology_headers(&self.file, &names, &self.metadata)?;
        self.file.attr("complete")?.write_scalar(&1u8)?;
        self.file.flush()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct ReadBudget {
    cap: u64,
    used: u64,
}

impl ReadBudget {
    const fn new(cap: u64) -> Self {
        Self { cap, used: 0 }
    }

    fn charge(&mut self, bytes: u64) -> Result<()> {
        let used = self
            .used
            .checked_add(bytes)
            .ok_or_else(|| invalid("covariance read byte count overflow"))?;
        if used > self.cap {
            return Err(invalid(format!(
                "covariance read allocation {used} exceeds byte cap {}",
                self.cap
            )));
        }
        self.used = used;
        Ok(())
    }
}

/// Read and structurally validate a complete operator without an allocation cap.
///
/// This convenience wrapper does not load numeric block values. Use
/// [`read_covariance_operator_metadata_with_byte_cap`] for untrusted or large artifacts.
pub fn read_covariance_operator_metadata(
    path: impl AsRef<Path>,
) -> Result<CovarianceOperatorMetadata> {
    read_covariance_operator_metadata_with_byte_cap(path, u64::MAX)
}

/// Read and structurally validate a complete operator under an allocation cap.
///
/// Metadata, registries, block names, and the complete topology are shape-checked
/// and charged before their variable-length values are loaded. Numeric payload
/// datasets are type/rank/shape checked but not loaded.
pub fn read_covariance_operator_metadata_with_byte_cap(
    path: impl AsRef<Path>,
    byte_cap: u64,
) -> Result<CovarianceOperatorMetadata> {
    let file = hdf5::File::open(path)?;
    let mut budget = ReadBudget::new(byte_cap);
    let metadata = read_checked_metadata(&file, &mut budget)?;
    let names = nonempty_block_names_with_budget(&file, &mut budget)?;
    let mut topology_bytes = 0_u64;
    for name in &names {
        let group = file.group(&format!("blocks/{name}"))?;
        inspect_block_layout(&group)?;
        checked_add_bytes(&mut topology_bytes, inspect_topology_layout(&group)?)?;
    }
    budget.charge(topology_workspace_bytes(topology_bytes)?)?;
    validate_topology_headers(&file, &names, &metadata)?;
    Ok(metadata)
}

/// Read and validate one block after rejecting payloads above `byte_cap`.
///
/// Dataset types, ranks, shapes, and element counts are checked before any
/// dataset in the selected block is loaded.
pub fn read_covariance_operator_block(
    path: impl AsRef<Path>,
    block_id: u64,
    byte_cap: u64,
) -> Result<CovarianceOperatorBlock> {
    let file = hdf5::File::open(path)?;
    let mut budget = ReadBudget::new(byte_cap);
    let metadata = read_checked_metadata(&file, &mut budget)?;
    let names = nonempty_block_names_with_budget(&file, &mut budget)?;
    let name = format!("{block_id:020}");
    ensure_valid(
        names.binary_search(&name).is_ok(),
        "covariance block is missing",
    )?;
    let mut selected_payload_bytes = None;
    let mut topology_bytes = 0_u64;
    for candidate in &names {
        let candidate_group = file.group(&format!("blocks/{candidate}"))?;
        let payload_bytes = inspect_block_layout(&candidate_group)?;
        checked_add_bytes(
            &mut topology_bytes,
            inspect_topology_layout(&candidate_group)?,
        )?;
        if candidate == &name {
            selected_payload_bytes = Some(payload_bytes);
        }
    }
    budget.charge(selected_payload_bytes.ok_or_else(|| invalid("covariance block is missing"))?)?;
    budget.charge(topology_workspace_bytes(topology_bytes)?)?;
    validate_topology_headers(&file, &names, &metadata)?;
    let group = file.group(&format!("blocks/{name}"))?;
    let block = read_block(&group)?;
    block.validate(metadata.gauge_date_index)?;
    ensure_valid(
        block.block_id == block_id,
        "covariance block group ID mismatch",
    )?;
    Ok(block)
}

/// Read a complete artifact after rejecting its aggregate block payload above a cap.
pub fn read_covariance_operator_with_byte_cap(
    path: impl AsRef<Path>,
    byte_cap: u64,
) -> Result<CovarianceOperatorArtifact> {
    let file = hdf5::File::open(path)?;
    let mut budget = ReadBudget::new(byte_cap);
    let metadata = read_checked_metadata(&file, &mut budget)?;
    let names = nonempty_block_names_with_budget(&file, &mut budget)?;
    let mut payload_bytes = 0_u64;
    let mut topology_bytes = 0_u64;
    for name in &names {
        let group = file.group(&format!("blocks/{name}"))?;
        payload_bytes = payload_bytes
            .checked_add(inspect_block_layout(&group)?)
            .ok_or_else(|| invalid("covariance block payload byte count overflow"))?;
        checked_add_bytes(&mut topology_bytes, inspect_topology_layout(&group)?)?;
    }
    budget.charge(payload_bytes)?;
    budget.charge(topology_workspace_bytes(topology_bytes)?)?;
    validate_topology_headers(&file, &names, &metadata)?;

    let mut blocks = Vec::with_capacity(names.len());
    for name in names {
        let group = file.group(&format!("blocks/{name}"))?;
        let block = read_block(&group)?;
        ensure_valid(
            name == format!("{:020}", block.block_id),
            "covariance block group ID mismatch",
        )?;
        block.validate(metadata.gauge_date_index)?;
        blocks.push(block);
    }
    Ok(CovarianceOperatorArtifact { metadata, blocks })
}

/// Read and validate a complete covariance replay operator artifact without a cap.
///
/// This eager convenience wrapper loads every block. Production replay should use
/// a capped block reader.
pub fn read_covariance_operator(path: impl AsRef<Path>) -> Result<CovarianceOperatorArtifact> {
    read_covariance_operator_with_byte_cap(path, u64::MAX)
}

fn read_checked_metadata(
    file: &hdf5::File,
    budget: &mut ReadBudget,
) -> Result<CovarianceOperatorMetadata> {
    validate_root_schema(file)?;
    budget.charge(inspect_metadata_layout(file)?)?;
    if read_scalar_attr::<u8>(file, "complete")? != 1 {
        return Err(invalid(
            "covariance operator scratch artifact is incomplete",
        ));
    }
    validate_registries(file)?;
    let metadata = read_metadata(file)?;
    metadata.validate()?;
    Ok(metadata)
}

fn block_names(file: &hdf5::File) -> Result<Vec<String>> {
    block_names_with_budget(file, &mut ReadBudget::new(u64::MAX))
}

fn block_names_with_budget(file: &hdf5::File, budget: &mut ReadBudget) -> Result<Vec<String>> {
    let blocks_group = file.group("blocks")?;
    validate_exact_schema(&blocks_group, None, &[], "covariance blocks schema")?;

    struct BlockNameScan {
        budget: ReadBudget,
        names: Vec<String>,
        error: Option<IoError>,
    }

    let scan = blocks_group.iter_visit_default(
        BlockNameScan {
            budget: *budget,
            names: Vec::new(),
            error: None,
        },
        |_, name, info, scan| {
            if info.link_type != LinkType::Hard {
                scan.error = Some(invalid("covariance block entry is not a hard link"));
                return false;
            }
            let Some(bytes) = u64::try_from(name.len())
                .ok()
                .and_then(|length| length.checked_add(BLOCK_NAME_BUDGET_BYTES))
            else {
                scan.error = Some(invalid("covariance block-name byte count overflow"));
                return false;
            };
            if let Err(error) = scan.budget.charge(bytes) {
                scan.error = Some(error);
                return false;
            }
            let Ok(parsed) = name.parse::<u64>() else {
                scan.error = Some(invalid("invalid covariance block group name"));
                return false;
            };
            if name != format!("{parsed:020}") {
                scan.error = Some(invalid(
                    "covariance block group is not a canonical padded ID",
                ));
                return false;
            }
            scan.names.push(name.to_owned());
            true
        },
    )?;
    *budget = scan.budget;
    if let Some(error) = scan.error {
        return Err(error);
    }
    let mut names = scan.names;
    names.sort_unstable();
    Ok(names)
}

fn nonempty_block_names_with_budget(
    file: &hdf5::File,
    budget: &mut ReadBudget,
) -> Result<Vec<String>> {
    let names = block_names_with_budget(file, budget)?;
    ensure_valid(
        !names.is_empty(),
        "covariance operator requires at least one block",
    )?;
    Ok(names)
}

fn inspect_block_layout(group: &Group) -> Result<u64> {
    validate_exact_schema(
        group,
        Some(COVARIANCE_BLOCK_MEMBERS),
        COVARIANCE_BLOCK_ATTRIBUTES,
        "covariance block contains a missing or unexpected dataset",
    )?;

    let native = read_grid(group, "native_grid")?;
    let output = read_grid(group, "output_grid")?;
    read_grid(group, "owned_output_grid")?;
    let native_shape = [native.rows as usize, native.cols as usize];
    let output_shape = [output.rows as usize, output.cols as usize];
    let native_area = native.area()?;
    let output_area = output.area()?;
    let support = read_rect_support(group)?;
    let support_bit_count = support.bit_count()?;
    let stored_support_count = read_scalar_attr::<u32>(group, "support_bits_per_output")?;
    ensure_valid(
        usize::try_from(stored_support_count) == Ok(support_bit_count),
        "support bit count does not match Rect geometry",
    )?;
    read_scalar_attr::<f64>(group, "branch_tolerance")?;
    read_scalar_attr::<u16>(group, "estimator_branch")?;

    let mut bytes = 0_u64;
    let (burst_shape, burst_bytes) = inspect_dataset::<u8>(group, "burst_id", None)?;
    ensure_valid(
        burst_shape.len() == 1 && burst_shape[0] > 0,
        "burst_id shape is not a nonempty vector",
    )?;
    checked_add_bytes(&mut bytes, burst_bytes)?;

    let (date_shape, date_bytes) = inspect_dataset::<u32>(group, "source_date_indices", None)?;
    ensure_valid(
        date_shape.len() == 1 && date_shape[0] > 0,
        "source_date_indices shape is not a nonempty vector",
    )?;
    checked_add_bytes(&mut bytes, date_bytes)?;
    add_exact_dataset::<u32>(group, "ordered_date_indices", &date_shape, &mut bytes)?;
    add_exact_dataset::<u64>(group, "source_ids", &[native_area], &mut bytes)?;
    add_exact_dataset::<u64>(group, "phase_node_ids", &[output_area], &mut bytes)?;
    add_exact_dataset::<u64>(group, "compressed_node_ids", &[native_area], &mut bytes)?;

    let (parent_shape, parent_bytes) = inspect_dataset::<u64>(group, "carry_parent_ids", None)?;
    ensure_valid(parent_shape.len() == 1, "carry_parent_ids is not rank one")?;
    checked_add_bytes(&mut bytes, parent_bytes)?;
    let component_count = parent_shape[0]
        .checked_add(date_shape[0])
        .ok_or_else(|| invalid("phase component count overflow"))?;
    add_exact_dataset::<u16>(
        group,
        "phase_component_kinds",
        &[component_count],
        &mut bytes,
    )?;
    add_exact_dataset::<u64>(group, "phase_component_ids", &[component_count], &mut bytes)?;
    add_exact_dataset::<u32>(group, "nearest_output_map", &native_shape, &mut bytes)?;
    add_exact_dataset::<f64>(
        group,
        "phase_angles",
        &[output_area, component_count],
        &mut bytes,
    )?;
    add_exact_dataset::<Complex64>(group, "compressed_raster", &native_shape, &mut bytes)?;
    add_exact_dataset::<u16>(group, "compressed_status", &native_shape, &mut bytes)?;
    add_exact_dataset::<Complex64>(group, "projection_accumulator", &native_shape, &mut bytes)?;
    add_exact_dataset::<f64>(group, "mean_amplitude", &native_shape, &mut bytes)?;
    add_exact_dataset::<u8>(
        group,
        "support_bits",
        &[output_area, support_bit_count.div_ceil(8)],
        &mut bytes,
    )?;
    add_exact_dataset::<u8>(
        group,
        "native_validity_bits",
        &[native_area.div_ceil(8)],
        &mut bytes,
    )?;
    for name in ["selected_eigenvalue", "eigen_gap"] {
        add_exact_dataset::<f64>(group, name, &output_shape, &mut bytes)?;
    }
    add_exact_dataset::<u16>(group, "status", &output_shape, &mut bytes)?;
    Ok(bytes)
}

fn inspect_dataset<T: H5Type>(
    group: &Group,
    name: &str,
    expected_shape: Option<&[usize]>,
) -> Result<(Vec<usize>, u64)> {
    let dataset = group.dataset(name)?;
    ensure_valid(
        dataset.dtype()?.is::<T>(),
        "covariance dataset has an unexpected element type",
    )?;
    let shape = dataset.shape();
    if let Some(expected) = expected_shape {
        if shape != expected {
            return Err(invalid(format!("{name} shape {shape:?} != {expected:?}")));
        }
    }
    let bytes = dataset
        .size()
        .checked_mul(dataset.dtype()?.size())
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| invalid(format!("{name} payload byte count overflow")))?;
    Ok((shape, bytes))
}

fn inspect_topology_layout(group: &Group) -> Result<u64> {
    let mut bytes = 0_u64;
    for (name, element_size) in [
        ("burst_id", 1_u64),
        ("source_date_indices", 4),
        ("ordered_date_indices", 4),
        ("source_ids", 8),
        ("phase_node_ids", 8),
        ("compressed_node_ids", 8),
        ("carry_parent_ids", 8),
        ("phase_component_kinds", 2),
        ("phase_component_ids", 8),
    ] {
        let dataset = group.dataset(name)?;
        let dataset_bytes = u64::try_from(dataset.size())
            .ok()
            .and_then(|count| count.checked_mul(element_size))
            .ok_or_else(|| invalid("covariance topology byte count overflow"))?;
        checked_add_bytes(&mut bytes, dataset_bytes)?;
    }
    Ok(bytes)
}

fn add_exact_dataset<T: H5Type>(
    group: &Group,
    name: &str,
    expected_shape: &[usize],
    total: &mut u64,
) -> Result<()> {
    let (_, bytes) = inspect_dataset::<T>(group, name, Some(expected_shape))?;
    checked_add_bytes(total, bytes)
}

fn checked_add_bytes(total: &mut u64, bytes: u64) -> Result<()> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| invalid("covariance block payload byte count overflow"))?;
    Ok(())
}

fn topology_workspace_bytes(bytes: u64) -> Result<u64> {
    bytes
        .checked_mul(TOPOLOGY_WORKSPACE_MULTIPLIER)
        .ok_or_else(|| invalid("covariance topology workspace byte count overflow"))
}

fn validate_topology_headers(
    file: &hdf5::File,
    names: &[String],
    metadata: &CovarianceOperatorMetadata,
) -> Result<()> {
    let mut topology = CovarianceTopologyState::default();
    for name in names {
        let group = file.group(&format!("blocks/{name}"))?;
        let block_id = read_scalar_attr(&group, "block_id")?;
        ensure_valid(
            *name == format!("{block_id:020}"),
            "covariance block group ID mismatch",
        )?;
        let entry = CovarianceBlockTopology {
            block_id,
            generation: read_scalar_attr(&group, "generation")?,
            burst_id: read_string(&group, "burst_id")?,
            native_grid: read_grid(&group, "native_grid")?,
            output_grid: read_grid(&group, "output_grid")?,
            owned_output_grid: read_grid(&group, "owned_output_grid")?,
            rect_support: read_rect_support(&group)?,
            reference_date_index: read_scalar_attr(&group, "reference_date_index")?,
            source_date_indices: group.dataset("source_date_indices")?.read_raw()?,
            ordered_date_indices: group.dataset("ordered_date_indices")?.read_raw()?,
            source_ids: group.dataset("source_ids")?.read_raw()?,
            phase_node_ids: group.dataset("phase_node_ids")?.read_raw()?,
            compressed_node_ids: group.dataset("compressed_node_ids")?.read_raw()?,
            carry_parent_ids: group.dataset("carry_parent_ids")?.read_raw()?,
            phase_components: read_phase_components(&group)?,
        };
        validate_topology_header(&entry, metadata.gauge_date_index, names.len())?;
        topology.validate(&entry, metadata.stitched_status)?;
        topology.insert(entry);
    }
    Ok(())
}

fn validate_root_schema(file: &hdf5::File) -> Result<()> {
    validate_exact_schema(
        file,
        Some(COVARIANCE_ROOT_MEMBERS),
        COVARIANCE_ROOT_ATTRIBUTES,
        "covariance root schema contains missing or unexpected members",
    )?;
    let source = file.group("source")?;
    validate_exact_schema(
        &source,
        Some(COVARIANCE_SOURCE_MEMBERS),
        &[],
        "covariance source schema contains missing or unexpected members",
    )?;
    let registries = file.group("registries")?;
    validate_exact_schema(
        &registries,
        Some(COVARIANCE_REGISTRY_MEMBERS),
        &[],
        "covariance registry schema contains missing or unexpected members",
    )
}

fn validate_exact_schema(
    group: &Group,
    expected_members: Option<&[&str]>,
    expected_attributes: &[&str],
    message: &'static str,
) -> Result<()> {
    if let Some(expected_members) = expected_members {
        ensure_valid(group.len() == expected_members.len() as u64, message)?;
        let exact_members = group.iter_visit_default(true, |_, name, info, exact| {
            *exact &= info.link_type == LinkType::Hard && expected_members.contains(&name);
            *exact
        })?;
        ensure_valid(exact_members, message)?;
    }
    ensure_valid(
        group.loc_info()?.num_attrs == expected_attributes.len(),
        message,
    )?;
    ensure_valid(
        expected_attributes
            .iter()
            .all(|name| group.attr(name).is_ok()),
        message,
    )
}

fn inspect_metadata_layout(file: &hdf5::File) -> Result<u64> {
    let mut bytes = 0_u64;
    for name in [
        "method",
        "crate_version",
        "producer_commit",
        "normalized_config_digest",
        "kernel_digest",
    ] {
        add_string_dataset(file, name, &mut bytes)?;
    }
    let source = file.group("source")?;
    for name in COVARIANCE_SOURCE_MEMBERS {
        add_string_dataset(&source, name, &mut bytes)?;
    }
    let registries = file.group("registries")?;
    for (name, registry) in covariance_registries() {
        add_exact_dataset::<u16>(
            &registries,
            &format!("{name}_codes"),
            &[registry.len()],
            &mut bytes,
        )?;
        let expected_names = registry
            .iter()
            .map(|entry| entry.name)
            .collect::<Vec<_>>()
            .join("\n");
        add_exact_dataset::<u8>(
            &registries,
            &format!("{name}_names"),
            &[expected_names.len()],
            &mut bytes,
        )?;
    }
    Ok(bytes)
}

fn add_string_dataset(group: &Group, name: &str, total: &mut u64) -> Result<()> {
    let (shape, bytes) = inspect_dataset::<u8>(group, name, None)?;
    if shape.len() != 1 {
        return Err(invalid(format!("{name} shape {shape:?} is not rank one")));
    }
    checked_add_bytes(total, bytes)
}

fn covariance_registries() -> [(&'static str, &'static [CovarianceRegistryEntry]); 9] {
    [
        ("method", COVARIANCE_METHOD_REGISTRY),
        ("estimator_branch", COVARIANCE_ESTIMATOR_BRANCH_REGISTRY),
        (
            "phase_component_kind",
            COVARIANCE_PHASE_COMPONENT_KIND_REGISTRY,
        ),
        ("support_ordering", COVARIANCE_SUPPORT_ORDERING_REGISTRY),
        ("operator_status", COVARIANCE_OPERATOR_STATUS_REGISTRY),
        ("replay_status", COVARIANCE_REPLAY_STATUS_REGISTRY),
        ("stitched_status", STITCHED_COVARIANCE_STATUS_REGISTRY),
        ("calibration_status", COVARIANCE_CALIBRATION_STATUS_REGISTRY),
        (
            "downstream_inference_status",
            DOWNSTREAM_INFERENCE_STATUS_REGISTRY,
        ),
    ]
}

fn write_metadata(file: &hdf5::File, metadata: &CovarianceOperatorMetadata) -> Result<()> {
    write_scalar_attr(file, "schema_version", metadata.schema_version)?;
    write_scalar_attr(file, "method_version", metadata.method_version)?;
    write_scalar_attr(file, "gauge_date_index", metadata.gauge_date_index)?;
    write_scalar_attr(file, "replay_status", metadata.replay_status.code())?;
    write_scalar_attr(file, "stitched_status", metadata.stitched_status.code())?;
    write_scalar_attr(
        file,
        "calibration_status",
        metadata.calibration_status.code(),
    )?;
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
        calibration_status: CovarianceCalibrationStatus::from_code(read_scalar_attr(
            file,
            "calibration_status",
        )?)?,
        downstream_inference_status: DownstreamInferenceStatus::from_code(read_scalar_attr(
            file,
            "downstream_inference_status",
        )?)?,
    })
}

fn write_registries(file: &hdf5::File) -> Result<()> {
    let group = file.create_group("registries")?;
    for (name, registry) in covariance_registries() {
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
    for (name, registry) in covariance_registries() {
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
    write_scalar_attr(group, "branch_tolerance", block.branch_tolerance)?;
    write_string(group, "burst_id", &block.burst_id)?;
    write_grid(group, "native_grid", block.native_grid)?;
    write_grid(group, "output_grid", block.output_grid)?;
    write_grid(group, "owned_output_grid", block.owned_output_grid)?;
    write_rect_support(group, block.rect_support)?;
    write_chunked_1d(group, "source_date_indices", &block.source_date_indices)?;
    write_chunked_1d(group, "ordered_date_indices", &block.ordered_date_indices)?;
    write_chunked_1d(group, "source_ids", &block.source_ids)?;
    write_chunked_1d(group, "phase_node_ids", &block.phase_node_ids)?;
    write_chunked_1d(group, "compressed_node_ids", &block.compressed_node_ids)?;
    write_chunked_1d(group, "carry_parent_ids", &block.carry_parent_ids)?;
    let phase_component_kinds = block
        .phase_components
        .iter()
        .map(|component| component.kind.code())
        .collect::<Vec<_>>();
    let phase_component_ids = block
        .phase_components
        .iter()
        .map(|component| component.id)
        .collect::<Vec<_>>();
    write_chunked_1d(group, "phase_component_kinds", &phase_component_kinds)?;
    write_chunked_1d(group, "phase_component_ids", &phase_component_ids)?;
    let native_shape = (
        block.native_grid.rows as usize,
        block.native_grid.cols as usize,
    );
    let output_shape = (
        block.output_grid.rows as usize,
        block.output_grid.cols as usize,
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
            block.phase_components.len(),
        ),
        &block.phase_angles,
    )?;
    write_chunked_2d(
        group,
        "compressed_raster",
        native_shape,
        &block.compressed_raster,
    )?;
    let compressed_statuses = block
        .compressed_status
        .iter()
        .map(|status| status.code())
        .collect::<Vec<_>>();
    write_chunked_2d(
        group,
        "compressed_status",
        native_shape,
        &compressed_statuses,
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
    let compressed_status_codes: Vec<u16> = group.dataset("compressed_status")?.read_raw()?;
    let compressed_status = compressed_status_codes
        .into_iter()
        .map(CovarianceOperatorStatus::from_code)
        .collect::<Result<_>>()?;
    let phase_components = read_phase_components(group)?;
    Ok(CovarianceOperatorBlock {
        burst_id: read_string(group, "burst_id")?,
        block_id: read_scalar_attr(group, "block_id")?,
        generation: read_scalar_attr(group, "generation")?,
        native_grid: read_grid(group, "native_grid")?,
        output_grid: read_grid(group, "output_grid")?,
        owned_output_grid: read_grid(group, "owned_output_grid")?,
        rect_support: read_rect_support(group)?,
        branch_tolerance: read_scalar_attr(group, "branch_tolerance")?,
        reference_date_index: read_scalar_attr(group, "reference_date_index")?,
        source_date_indices: group.dataset("source_date_indices")?.read_raw()?,
        ordered_date_indices: group.dataset("ordered_date_indices")?.read_raw()?,
        source_ids: group.dataset("source_ids")?.read_raw()?,
        phase_node_ids: group.dataset("phase_node_ids")?.read_raw()?,
        compressed_node_ids: group.dataset("compressed_node_ids")?.read_raw()?,
        carry_parent_ids: group.dataset("carry_parent_ids")?.read_raw()?,
        nearest_output_map: group.dataset("nearest_output_map")?.read_raw()?,
        phase_components,
        phase_angles: group.dataset("phase_angles")?.read_raw()?,
        compressed_raster: group.dataset("compressed_raster")?.read_raw()?,
        compressed_status,
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

fn read_phase_components(group: &Group) -> Result<Vec<CovariancePhaseComponent>> {
    let phase_component_kinds: Vec<u16> = group.dataset("phase_component_kinds")?.read_raw()?;
    let phase_component_ids: Vec<u64> = group.dataset("phase_component_ids")?.read_raw()?;
    if phase_component_kinds.len() != phase_component_ids.len() {
        return Err(invalid(
            "covariance phase component kind and ID counts differ",
        ));
    }
    phase_component_kinds
        .into_iter()
        .zip(phase_component_ids)
        .map(|(kind, id)| {
            Ok(CovariancePhaseComponent {
                kind: CovariancePhaseComponentKind::from_code(kind)?,
                id,
            })
        })
        .collect()
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
    validate_exact_schema(
        &grid_group,
        Some(&[]),
        COVARIANCE_GRID_ATTRIBUTES,
        "covariance grid schema contains missing or unexpected members",
    )?;
    Ok(CovarianceOperatorGrid {
        row_start: read_scalar_attr(&grid_group, "row_start")?,
        col_start: read_scalar_attr(&grid_group, "col_start")?,
        rows: read_scalar_attr(&grid_group, "rows")?,
        cols: read_scalar_attr(&grid_group, "cols")?,
        stride_y: read_scalar_attr(&grid_group, "stride_y")?,
        stride_x: read_scalar_attr(&grid_group, "stride_x")?,
    })
}

fn write_rect_support(group: &Group, rect: CovarianceRectSupport) -> Result<()> {
    let support = group.create_group("rect_support")?;
    write_scalar_attr(&support, "half_window_rows", rect.half_window_rows)?;
    write_scalar_attr(&support, "half_window_cols", rect.half_window_cols)?;
    write_scalar_attr(&support, "ordering", rect.ordering.code())
}

fn read_rect_support(group: &Group) -> Result<CovarianceRectSupport> {
    let support = group.group("rect_support")?;
    validate_exact_schema(
        &support,
        Some(&[]),
        COVARIANCE_RECT_SUPPORT_ATTRIBUTES,
        "covariance Rect support schema contains missing or unexpected members",
    )?;
    Ok(CovarianceRectSupport {
        half_window_rows: read_scalar_attr(&support, "half_window_rows")?,
        half_window_cols: read_scalar_attr(&support, "half_window_cols")?,
        ordering: CovarianceSupportOrdering::from_code(read_scalar_attr(&support, "ordering")?)?,
    })
}

fn write_scalar_attr<T: H5Type>(group: &Group, name: &str, value: T) -> Result<()> {
    group.new_attr::<T>().create(name)?.write_scalar(&value)?;
    Ok(())
}

fn read_scalar_attr<T: H5Type>(group: &Group, name: &str) -> Result<T> {
    let attribute = group.attr(name)?;
    ensure_valid(
        attribute.dtype()?.is::<T>() && attribute.is_scalar(),
        "covariance scalar attribute has an unexpected type or shape",
    )?;
    Ok(attribute.read_scalar()?)
}

fn write_string(group: &Group, name: &str, value: &str) -> Result<()> {
    group
        .new_dataset_builder()
        .with_data(value.as_bytes())
        .create(name)?;
    Ok(())
}

fn read_string(group: &Group, name: &str) -> Result<String> {
    let (shape, _) = inspect_dataset::<u8>(group, name, None)?;
    if shape.len() != 1 {
        return Err(invalid(format!("{name} shape {shape:?} is not rank one")));
    }
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

fn trailing_bit_mask(bit_count: usize) -> u8 {
    let remainder = bit_count % 8;
    match remainder {
        0 => 0,
        _ => !((1_u8 << remainder) - 1),
    }
}

fn packed_bit(bits: &[u8], index: usize) -> bool {
    bits[index / 8] & (1 << (index % 8)) != 0
}

fn packed_trailing_bits_are_zero(bits: &[u8], bit_count: usize) -> bool {
    bits.last()
        .is_some_and(|byte| byte & trailing_bit_mask(bit_count) == 0)
}

fn window_origin(output: usize, half_window: usize, stride: usize, native_len: usize) -> usize {
    let window_len = 2 * half_window + 1;
    let center = stride / 2 + output * stride;
    center
        .saturating_sub(half_window)
        .min(native_len - window_len)
}

fn validate_topology_header(
    block: &CovarianceBlockTopology,
    gauge_date_index: u32,
    block_count: usize,
) -> Result<()> {
    ensure_valid(!block.burst_id.is_empty(), "empty covariance burst_id")?;
    ensure_valid(
        block.reference_date_index == gauge_date_index,
        "gauge mismatch",
    )?;
    ensure_valid(
        !block.source_date_indices.is_empty()
            && block.source_date_indices == block.ordered_date_indices
            && strictly_increasing(&block.source_date_indices),
        "invalid covariance topology date map",
    )?;
    ensure_valid(
        strictly_increasing(&block.carry_parent_ids)
            && block
                .carry_parent_ids
                .iter()
                .all(|&parent| parent < block.block_id)
            && block.carry_parent_ids.len() <= block_count,
        "invalid covariance topology parent map",
    )?;
    ensure_valid(
        !block.carry_parent_ids.is_empty()
            || block.source_date_indices.first() == Some(&gauge_date_index),
        "first block source omits gauge date",
    )?;
    ensure_valid(
        block.carry_parent_ids.is_empty() || !block.source_date_indices.contains(&gauge_date_index),
        "later block source repeats gauge date",
    )?;
    let parents = block
        .carry_parent_ids
        .iter()
        .map(|&id| (CovariancePhaseComponentKind::CompressedParent, id));
    let dates = block.source_date_indices.iter().map(|&date| {
        let kind = match date == gauge_date_index {
            true => CovariancePhaseComponentKind::GaugeDate,
            false => CovariancePhaseComponentKind::RetainedDate,
        };
        (kind, u64::from(date))
    });
    ensure_valid(
        block
            .phase_components
            .iter()
            .map(|component| (component.kind, component.id))
            .eq(parents.chain(dates)),
        "covariance topology phase component map is invalid",
    )?;
    let native_area = block.native_grid.area()?;
    let output_area = block.output_grid.area()?;
    block.owned_output_grid.area()?;
    grid_stop(block.native_grid.row_start, block.native_grid.rows)?;
    grid_stop(block.native_grid.col_start, block.native_grid.cols)?;
    ensure_valid(
        block.native_grid.stride_y == 1 && block.native_grid.stride_x == 1,
        "native grid stride must be one",
    )?;
    ensure_valid(
        block.output_grid.rows == block.native_grid.rows / block.output_grid.stride_y
            && block.output_grid.cols == block.native_grid.cols / block.output_grid.stride_x,
        "output grid shape does not match native grid and strides",
    )?;
    ensure_valid(
        block
            .output_grid
            .row_start
            .checked_mul(u64::from(block.output_grid.stride_y))
            == Some(block.native_grid.row_start)
            && block
                .output_grid
                .col_start
                .checked_mul(u64::from(block.output_grid.stride_x))
                == Some(block.native_grid.col_start),
        "native and output grid origins are not stride-aligned",
    )?;
    ensure_valid(
        block.output_grid.contains(block.owned_output_grid),
        "owned output grid is not contained in replay output grid",
    )?;
    let support_rows = u64::from(block.rect_support.half_window_rows) * 2 + 1;
    let support_cols = u64::from(block.rect_support.half_window_cols) * 2 + 1;
    ensure_valid(
        support_rows <= u64::from(block.native_grid.rows)
            && support_cols <= u64::from(block.native_grid.cols),
        "Rect window exceeds the native grid",
    )?;
    for (name, actual, expected) in [
        ("source_ids", block.source_ids.len(), native_area),
        (
            "compressed_node_ids",
            block.compressed_node_ids.len(),
            native_area,
        ),
        ("phase_node_ids", block.phase_node_ids.len(), output_area),
    ] {
        check_len(name, actual, expected)?;
    }
    Ok(())
}

fn validate_block_topology(
    block: &CovarianceBlockTopology,
    prior: &BTreeMap<u64, CovarianceBlockTopology>,
) -> Result<()> {
    ensure_valid(
        (block.generation == 0) == block.carry_parent_ids.is_empty(),
        "generation zero must be exactly the parentless root",
    )?;
    let mut parent_generations = Vec::with_capacity(block.carry_parent_ids.len());
    let mut immediate_prior = None;
    for parent_id in &block.carry_parent_ids {
        let parent = prior
            .get(parent_id)
            .ok_or_else(|| invalid(format!("covariance parent block {parent_id} is missing")))?;
        ensure_valid(
            parent.generation < block.generation,
            "covariance parent generation does not strictly precede child",
        )?;
        parent_generations.push(parent.generation);
        if parent.generation.checked_add(1) == Some(block.generation) {
            immediate_prior = Some(parent);
        }
        ensure_valid(
            parent.burst_id == block.burst_id
                && parent.native_grid == block.native_grid
                && parent.output_grid == block.output_grid
                && parent.owned_output_grid == block.owned_output_grid,
            "covariance parent burst or replay/owned/native grid differs from child",
        )?;
    }
    ensure_valid(
        block.generation == 0 || immediate_prior.is_some(),
        "covariance parents omit the immediate prior generation",
    )?;
    ensure_valid(
        parent_generations
            .windows(2)
            .all(|pair| pair[0].checked_add(1) == Some(pair[1])),
        "covariance parent generations are not a consecutive ordered suffix",
    )?;
    if let Some(parent) = immediate_prior {
        ensure_valid(
            parent
                .source_date_indices
                .last()
                .and_then(|date| date.checked_add(1))
                == block.source_date_indices.first().copied(),
            "covariance block date chronology is not contiguous",
        )?;
    }
    Ok(())
}

fn validate_cross_record_topology(
    block: &CovarianceBlockTopology,
    state: &CovarianceTopologyState,
    stitched_status: StitchedCovarianceStatus,
) -> Result<()> {
    let mut local_sources = BTreeMap::new();
    for (native_index, &source_id) in block.source_ids.iter().enumerate() {
        if let Some(prior_index) = local_sources.insert(source_id, native_index) {
            ensure_valid(
                source_key_matches(block, native_index, block, prior_index),
                "one covariance source ID identifies different primitive sources",
            )?;
        }
        if let Some(location) = state.source_locations.get(&source_id) {
            let prior = state.blocks.get(&location.block_id).ok_or_else(|| {
                invalid("covariance source index references a missing topology block")
            })?;
            ensure_valid(
                source_key_matches(block, native_index, prior, location.native_index),
                "one covariance source ID identifies different primitive sources",
            )?;
        }
    }

    let mut local_nodes = BTreeSet::new();
    for &node_id in block
        .phase_node_ids
        .iter()
        .chain(block.compressed_node_ids.iter())
    {
        ensure_valid(
            local_nodes.insert(node_id) && !state.node_ids.contains(&node_id),
            "covariance phase/compressed node ID is not globally unique",
        )?;
    }

    for prior in state.blocks.values() {
        if prior.burst_id != block.burst_id {
            ensure_valid(
                stitched_status == StitchedCovarianceStatus::UnsupportedSeamCovariance,
                "multiple covariance bursts require unsupported-seam stitched status",
            )?;
            continue;
        }
        if prior.generation != block.generation {
            continue;
        }
        ensure_valid(
            prior.source_date_indices == block.source_date_indices,
            "covariance tiles in one generation date map differ",
        )?;
        ensure_valid(
            prior.output_grid.stride_y == block.output_grid.stride_y
                && prior.output_grid.stride_x == block.output_grid.stride_x
                && prior.rect_support == block.rect_support,
            "covariance tiles in one generation geometry differ",
        )?;
        ensure_valid(
            !grids_overlap(prior.owned_output_grid, block.owned_output_grid),
            "covariance owned output grids overlap within one generation",
        )?;
        validate_shared_native_sources(prior, block)?;
    }
    Ok(())
}

fn source_key_matches(
    left: &CovarianceBlockTopology,
    left_index: usize,
    right: &CovarianceBlockTopology,
    right_index: usize,
) -> bool {
    left.burst_id == right.burst_id
        && left.generation == right.generation
        && left.source_date_indices == right.source_date_indices
        && native_coordinate(left.native_grid, left_index)
            == native_coordinate(right.native_grid, right_index)
}

fn native_coordinate(grid: CovarianceOperatorGrid, index: usize) -> (u64, u64) {
    let cols = grid.cols as usize;
    (
        grid.row_start + (index / cols) as u64,
        grid.col_start + (index % cols) as u64,
    )
}

fn validate_shared_native_sources(
    left: &CovarianceBlockTopology,
    right: &CovarianceBlockTopology,
) -> Result<()> {
    let row_start = left.native_grid.row_start.max(right.native_grid.row_start);
    let col_start = left.native_grid.col_start.max(right.native_grid.col_start);
    let row_stop = grid_stop(left.native_grid.row_start, left.native_grid.rows)?.min(grid_stop(
        right.native_grid.row_start,
        right.native_grid.rows,
    )?);
    let col_stop = grid_stop(left.native_grid.col_start, left.native_grid.cols)?.min(grid_stop(
        right.native_grid.col_start,
        right.native_grid.cols,
    )?);
    if row_start >= row_stop || col_start >= col_stop {
        return Ok(());
    }
    for row in row_start..row_stop {
        for col in col_start..col_stop {
            let left_index = ((row - left.native_grid.row_start) as usize)
                * left.native_grid.cols as usize
                + (col - left.native_grid.col_start) as usize;
            let right_index = ((row - right.native_grid.row_start) as usize)
                * right.native_grid.cols as usize
                + (col - right.native_grid.col_start) as usize;
            ensure_valid(
                left.source_ids[left_index] == right.source_ids[right_index],
                "shared native source has different consumer IDs across tiles",
            )?;
        }
    }
    Ok(())
}

fn grids_overlap(left: CovarianceOperatorGrid, right: CovarianceOperatorGrid) -> bool {
    if left.stride_y != right.stride_y || left.stride_x != right.stride_x {
        return false;
    }
    let left_row_stop = left.row_start.checked_add(u64::from(left.rows));
    let left_col_stop = left.col_start.checked_add(u64::from(left.cols));
    let right_row_stop = right.row_start.checked_add(u64::from(right.rows));
    let right_col_stop = right.col_start.checked_add(u64::from(right.cols));
    match (left_row_stop, left_col_stop, right_row_stop, right_col_stop) {
        (Some(lr), Some(lc), Some(rr), Some(rc)) => {
            left.row_start < rr
                && right.row_start < lr
                && left.col_start < rc
                && right.col_start < lc
        }
        _ => true,
    }
}

fn grid_stop(start: u64, len: u32) -> Result<u64> {
    start
        .checked_add(u64::from(len))
        .ok_or_else(|| invalid("covariance grid extent overflows global coordinates"))
}

fn check_len(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(invalid(format!(
            "covariance operator {name} length {actual} != {expected}"
        )));
    }
    Ok(())
}

fn ensure_valid(condition: bool, message: &'static str) -> Result<()> {
    if !condition {
        return Err(invalid(message));
    }
    Ok(())
}

fn strictly_increasing<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn is_sha256_digest(value: &str) -> bool {
    let hex = value.strip_prefix("sha256:").unwrap_or(value);
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn invalid(message: impl Into<String>) -> IoError {
    IoError::Shape(message.into())
}

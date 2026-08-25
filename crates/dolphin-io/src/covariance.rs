//! Block-indexed HDF5 persistence for the sequential covariance replay operator.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use hdf5::{Group, H5Type, LinkType};
use ndarray::{ArrayView2, ArrayView3};
use num_complex::Complex64;
use sha2::{Digest, Sha256};

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
    "model_version_digest",
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
const IDENTITY_RECORD_BYTES: u64 = 32;
const SOURCE_IDENTITY_KIND: u8 = 0;
const NODE_IDENTITY_KIND: u8 = 1;
static IDENTITY_RUN_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Conservative peak temporary disk for a sorted-run identity index.
///
/// The bound holds the current unique runs plus one same-sized merge output.
///
/// # Errors
/// Returns an error when the record projection exceeds `u64`.
pub fn covariance_identity_index_peak_bytes(identity_records: u64) -> Result<u64> {
    identity_records
        .checked_mul(IDENTITY_RECORD_BYTES)
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| invalid("covariance identity-index peak projection overflow"))
}

/// Digest the ordered resolver and source-model identity used by replay IDs.
#[must_use]
pub fn covariance_source_model_identity_digest(
    provider: &str,
    provider_version: &str,
    model: &str,
    model_version: &str,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:sequential_source_model_identity:v1");
    for value in [provider, provider_version, model, model_version] {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value.as_bytes());
    }
    digest.finalize().into()
}

const COVARIANCE_BLOCK_MEMBERS: &[&str] = &[
    "burst_id",
    "source_manifest_digest",
    "source_model_version_digest",
    "native_grid",
    "output_grid",
    "owned_output_grid",
    "rect_support",
    "source_date_indices",
    "ordered_date_indices",
    "source_ids",
    "source_content_digests",
    "source_factor_digests",
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
    /// Digest of the ordered provider/model names and versions used by replay IDs.
    pub model_version_digest: Option<String>,
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
            ("source.model_version_digest", &self.model_version_digest),
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
                self.model_version_digest.as_ref(),
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
            (
                "source model version identity",
                self.model_version_digest.as_deref(),
            ),
        ] {
            if let Some(value) = value {
                ensure_valid(
                    is_sha256_digest(value),
                    match name {
                        "source manifest" => {
                            "source manifest digest is not a strong SHA-256 digest"
                        }
                        "source model receipt" => {
                            "source model receipt digest is not a strong SHA-256 digest"
                        }
                        _ => "source model version digest is not a strong SHA-256 digest",
                    },
                )?;
            }
        }
        let identity = [
            self.provider.as_deref(),
            self.provider_version.as_deref(),
            self.model.as_deref(),
            self.model_version.as_deref(),
        ];
        let identity_count = identity.iter().filter(|value| value.is_some()).count();
        ensure_valid(
            identity_count == 0 || identity_count == identity.len(),
            "source provider/model identity must be entirely present or absent",
        )?;
        if let [Some(provider), Some(provider_version), Some(model), Some(model_version)] = identity
        {
            let expected = covariance_source_model_identity_digest(
                provider,
                provider_version,
                model,
                model_version,
            );
            ensure_valid(
                self.model_version_digest
                    .as_deref()
                    .and_then(sha256_digest_bytes)
                    == Some(expected),
                "source model version digest differs from provider/model identity",
            )?;
        } else {
            ensure_valid(
                self.model_version_digest.is_none(),
                "source model version digest requires provider/model identity",
            )?;
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

/// Expected source-date generations for one burst in a complete operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CovarianceBurstPlan {
    /// Burst identity used by every planned block chain.
    pub burst_id: String,
    /// Ordered source-date indices for generation 0 through the final generation.
    pub source_dates_by_generation: Vec<Vec<u32>>,
    /// Exact row-major tile grids expected once for this burst.
    pub tiles: Vec<CovarianceTilePlan>,
}

/// Exact native, replay-output, and owned-output grids for one tile chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CovarianceTilePlan {
    /// Full native read grid, including the sequential dependency halo.
    pub native_grid: CovarianceOperatorGrid,
    /// Full looked replay grid produced from the native read.
    pub output_grid: CovarianceOperatorGrid,
    /// Non-overlapping public output grid owned by this tile.
    pub owned_output_grid: CovarianceOperatorGrid,
}

/// Complete burst and generation plan supplied before operator capture starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CovarianceOperatorPlan {
    /// Ordered source-manifest digest used by every captured block namespace.
    pub source_manifest_digest: [u8; 32],
    /// Provider/model identity digest used by every captured block namespace.
    pub source_model_version_digest: [u8; 32],
    /// Expected burst plans. Each burst must be written in one contiguous run.
    pub bursts: Vec<CovarianceBurstPlan>,
}

impl CovarianceOperatorPlan {
    fn validate(&self, metadata: &CovarianceOperatorMetadata) -> Result<()> {
        ensure_valid(
            self.source_manifest_digest.iter().any(|byte| *byte != 0)
                && self
                    .source_model_version_digest
                    .iter()
                    .any(|byte| *byte != 0),
            "covariance operator plan requires strong source namespace digests",
        )?;
        ensure_valid(
            metadata
                .source
                .manifest_digest
                .as_deref()
                .and_then(sha256_digest_bytes)
                == Some(self.source_manifest_digest),
            "covariance capture plan source manifest differs from metadata",
        )?;
        ensure_valid(
            metadata
                .source
                .model_version_digest
                .as_deref()
                .and_then(sha256_digest_bytes)
                == Some(self.source_model_version_digest),
            "covariance capture plan source-model identity differs from metadata",
        )?;
        ensure_valid(
            !self.bursts.is_empty(),
            "covariance operator plan requires at least one burst",
        )?;
        ensure_valid(
            self.bursts.len() == 1
                || metadata.stitched_status == StitchedCovarianceStatus::UnsupportedSeamCovariance,
            "multiple covariance burst plans require unsupported-seam stitched status",
        )?;
        let mut burst_ids = BTreeSet::new();
        for burst in &self.bursts {
            ensure_valid(
                !burst.burst_id.is_empty() && burst_ids.insert(burst.burst_id.as_str()),
                "covariance operator plan has an empty or repeated burst ID",
            )?;
            ensure_valid(
                !burst.source_dates_by_generation.is_empty(),
                "covariance burst plan requires at least one generation",
            )?;
            ensure_valid(
                !burst.tiles.is_empty(),
                "covariance burst plan requires at least one tile",
            )?;
            let mut expected_date = 0_u32;
            for dates in &burst.source_dates_by_generation {
                ensure_valid(
                    consecutive(dates) && dates.first().copied() == Some(expected_date),
                    "covariance burst plan dates are not one contiguous acquisition sequence",
                )?;
                expected_date = dates
                    .last()
                    .and_then(|date| date.checked_add(1))
                    .ok_or_else(|| invalid("covariance burst plan date index overflow"))?;
            }
            let mut prior: Option<CovarianceTilePlan> = None;
            let mut ownership_frontier = Vec::<CovarianceOperatorGrid>::new();
            for tile in &burst.tiles {
                tile.native_grid.area()?;
                tile.output_grid.area()?;
                tile.owned_output_grid.area()?;
                ensure_valid(
                    tile.output_grid.contains(tile.owned_output_grid),
                    "covariance planned owned output grid is outside its replay grid",
                )?;
                if let Some(prior) = prior {
                    ensure_valid(
                        (
                            tile.owned_output_grid.row_start,
                            tile.owned_output_grid.col_start,
                        ) > (
                            prior.owned_output_grid.row_start,
                            prior.owned_output_grid.col_start,
                        ),
                        "covariance burst tiles are not in strict row-major order",
                    )?;
                }
                ownership_frontier.retain(|grid| {
                    grid_stop(grid.row_start, grid.rows)
                        .is_ok_and(|stop| stop > tile.owned_output_grid.row_start)
                });
                ensure_valid(
                    ownership_frontier
                        .iter()
                        .all(|grid| !grids_overlap(*grid, tile.owned_output_grid)),
                    "covariance planned owned output grids overlap",
                )?;
                ownership_frontier.push(tile.owned_output_grid);
                prior = Some(*tile);
            }
        }
        Ok(())
    }

    fn expected_generations(&self) -> Result<BTreeMap<String, BTreeMap<u32, Vec<u32>>>> {
        self.bursts
            .iter()
            .map(|burst| {
                let generations = burst
                    .source_dates_by_generation
                    .iter()
                    .enumerate()
                    .map(|(generation, dates)| {
                        u32::try_from(generation)
                            .map(|generation| (generation, dates.clone()))
                            .map_err(|_| invalid("covariance burst generation index exceeds u32"))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?;
                Ok((burst.burst_id.clone(), generations))
            })
            .collect()
    }

    fn expected_tiles(&self) -> BTreeMap<String, Vec<CovarianceTilePlan>> {
        self.bursts
            .iter()
            .map(|burst| (burst.burst_id.clone(), burst.tiles.clone()))
            .collect()
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

/// Derive the stable block ID for one identified burst/tile generation.
#[must_use]
pub fn covariance_record_block_id(
    burst_id: &str,
    source_manifest_digest: [u8; 32],
    source_model_version_digest: [u8; 32],
    generation: u32,
    native_grid: CovarianceOperatorGrid,
    output_grid: CovarianceOperatorGrid,
    owned_output_grid: CovarianceOperatorGrid,
) -> u64 {
    let mut digest = Sha256::new();
    digest.update(COVARIANCE_OPERATOR_METHOD.as_bytes());
    digest.update(b"record_block");
    digest.update(burst_id.as_bytes());
    digest.update(source_manifest_digest);
    digest.update(source_model_version_digest);
    for value in [
        native_grid.row_start,
        native_grid.col_start,
        u64::from(native_grid.rows),
        u64::from(native_grid.cols),
        output_grid.row_start,
        output_grid.col_start,
        u64::from(output_grid.rows),
        u64::from(output_grid.cols),
        owned_output_grid.row_start,
        owned_output_grid.col_start,
        u64::from(owned_output_grid.rows),
        u64::from(owned_output_grid.cols),
    ] {
        digest.update(value.to_le_bytes());
    }
    let tile_hash = u64::from_le_bytes(digest.finalize()[..8].try_into().expect("SHA-256 prefix"))
        & ((1_u64 << 48) - 1);
    (u64::from(generation) << 48) | tile_hash
}

/// Derive one stable source/node locator in the replay namespace.
///
/// # Errors
/// Returns an error when `local` is outside `grid` or its coordinate overflows.
#[allow(clippy::too_many_arguments)]
pub fn covariance_identified_id(
    kind: &[u8],
    burst_id: &str,
    source_manifest_digest: [u8; 32],
    source_model_version_digest: [u8; 32],
    major: u64,
    secondary: u64,
    grid: CovarianceOperatorGrid,
    local: usize,
) -> Result<u64> {
    let area = grid.area()?;
    ensure_valid(local < area, "covariance ID index is outside its grid")?;
    let columns =
        usize::try_from(grid.cols).map_err(|_| invalid("covariance grid columns exceed usize"))?;
    let local_row =
        u64::try_from(local / columns).map_err(|_| invalid("covariance ID row exceeds u64"))?;
    let local_column =
        u64::try_from(local % columns).map_err(|_| invalid("covariance ID column exceeds u64"))?;
    let row = grid
        .row_start
        .checked_add(local_row)
        .ok_or_else(|| invalid("covariance ID row overflows u64"))?;
    let column = grid
        .col_start
        .checked_add(local_column)
        .ok_or_else(|| invalid("covariance ID column overflows u64"))?;
    let mut digest = Sha256::new();
    digest.update(COVARIANCE_OPERATOR_METHOD.as_bytes());
    digest.update(kind);
    digest.update(burst_id.as_bytes());
    digest.update(source_manifest_digest);
    digest.update(source_model_version_digest);
    digest.update(major.to_le_bytes());
    digest.update(secondary.to_le_bytes());
    digest.update(row.to_le_bytes());
    digest.update(column.to_le_bytes());
    Ok(u64::from_le_bytes(
        digest.finalize()[..8].try_into().expect("SHA-256 prefix"),
    ))
}

/// Bind a primitive-source locator to its exact raw-sample digest.
///
/// # Errors
/// Returns an error for an all-zero content digest.
pub fn covariance_content_bound_source_id(
    locator_id: u64,
    content_digest: &[u8; 32],
) -> Result<u64> {
    ensure_valid(
        content_digest.iter().any(|byte| *byte != 0),
        "covariance source content digest is all zero",
    )?;
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:content_bound_source_id:v1");
    digest.update(locator_id.to_le_bytes());
    digest.update(content_digest);
    Ok(u64::from_le_bytes(
        digest.finalize()[..8]
            .try_into()
            .expect("SHA-256 prefix has eight bytes"),
    ))
}

/// Persisted numeric state for one block of the implicit source-keyed replay DAG.
#[derive(Debug, Clone, PartialEq)]
pub struct CovarianceOperatorBlock {
    /// Burst identity owning the block.
    pub burst_id: String,
    /// Full ordered source-manifest digest used by the capture namespace.
    pub source_manifest_digest: [u8; 32],
    /// Full provider/model identity digest used by the capture namespace.
    pub source_model_version_digest: [u8; 32],
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
    /// SHA-256 digest bytes for each ordered raw source vector, native-pixel
    /// major. Exactly 32 bytes are stored per primitive source.
    pub source_content_digests: Vec<u8>,
    /// SHA-256 digest bytes for each exact proper-complex numeric factor,
    /// native-pixel major. Replayable artifacts store 32 bytes per source.
    pub source_factor_digests: Vec<u8>,
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
    #[allow(clippy::too_many_lines)]
    fn validate(&self, gauge_date_index: u32) -> Result<()> {
        ensure_valid(!self.burst_id.is_empty(), "empty covariance burst_id")?;
        ensure_valid(
            self.source_manifest_digest.iter().any(|byte| *byte != 0)
                && self
                    .source_model_version_digest
                    .iter()
                    .any(|byte| *byte != 0),
            "covariance block source namespace digest is missing",
        )?;
        ensure_valid(!self.source_date_indices.is_empty(), "no source dates")?;
        ensure_valid(!self.ordered_date_indices.is_empty(), "no output dates")?;
        ensure_valid(
            self.reference_date_index == gauge_date_index,
            "gauge mismatch",
        )?;
        ensure_valid(
            consecutive(&self.source_date_indices)
                && self.ordered_date_indices == self.source_date_indices,
            "covariance source/output dates are not one contiguous ordered range",
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
        check_len(
            "source_content_digests",
            self.source_content_digests.len(),
            native_area
                .checked_mul(32)
                .ok_or_else(|| invalid("source digest dimensions overflow usize"))?,
        )?;
        check_len(
            "source_factor_digests",
            self.source_factor_digests.len(),
            native_area
                .checked_mul(32)
                .ok_or_else(|| invalid("source factor digest dimensions overflow usize"))?,
        )?;
        ensure_valid(
            self.source_content_digests
                .chunks_exact(32)
                .all(|digest| digest.iter().any(|byte| *byte != 0)),
            "primitive source has an all-zero content digest",
        )?;
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
                    !packed_bit(row, slot) || packed_bit(&self.native_validity_bits, native_index),
                    "support bits exceed native validity and Rect clamp",
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
    source_manifest_digest: [u8; 32],
    source_model_version_digest: [u8; 32],
    native_grid: CovarianceOperatorGrid,
    output_grid: CovarianceOperatorGrid,
    owned_output_grid: CovarianceOperatorGrid,
    rect_support: CovarianceRectSupport,
    reference_date_index: u32,
    source_date_indices: Vec<u32>,
    ordered_date_indices: Vec<u32>,
    source_ids: Vec<u64>,
    source_content_digests: Vec<u8>,
    source_factor_digests: Vec<u8>,
    native_validity_bits: Vec<u8>,
    estimator_branch: CovarianceEstimatorBranch,
    branch_tolerance_bits: u64,
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
            source_manifest_digest: block.source_manifest_digest,
            source_model_version_digest: block.source_model_version_digest,
            native_grid: block.native_grid,
            output_grid: block.output_grid,
            owned_output_grid: block.owned_output_grid,
            rect_support: block.rect_support,
            reference_date_index: block.reference_date_index,
            source_date_indices: block.source_date_indices.clone(),
            ordered_date_indices: block.ordered_date_indices.clone(),
            source_ids: block.source_ids.clone(),
            source_content_digests: block.source_content_digests.clone(),
            source_factor_digests: block.source_factor_digests.clone(),
            native_validity_bits: block.native_validity_bits.clone(),
            estimator_branch: block.estimator_branch,
            branch_tolerance_bits: block.branch_tolerance.to_bits(),
            phase_node_ids: block.phase_node_ids.clone(),
            compressed_node_ids: block.compressed_node_ids.clone(),
            carry_parent_ids: block.carry_parent_ids.clone(),
            phase_components: block.phase_components.clone(),
        }
    }
}

fn validate_replayable_ids(block: &CovarianceBlockTopology) -> Result<()> {
    ensure_valid(
        block
            .source_factor_digests
            .chunks_exact(32)
            .all(|digest| digest.iter().any(|byte| *byte != 0)),
        "replayable covariance source has no numeric factor receipt",
    )?;
    let expected_block_id = covariance_record_block_id(
        &block.burst_id,
        block.source_manifest_digest,
        block.source_model_version_digest,
        block.generation,
        block.native_grid,
        block.output_grid,
        block.owned_output_grid,
    );
    ensure_valid(
        block.block_id == expected_block_id,
        "replayable covariance block ID is not canonically derived",
    )?;
    let date_count = u64::try_from(block.source_date_indices.len())
        .map_err(|_| invalid("covariance source date count exceeds u64"))?;
    let first_date = block
        .source_date_indices
        .first()
        .copied()
        .ok_or_else(|| invalid("replayable covariance block has no source date"))?;
    let source_secondary = (u64::from(first_date) << 32) | date_count;
    for (native_index, (&actual, digest)) in block
        .source_ids
        .iter()
        .zip(block.source_content_digests.chunks_exact(32))
        .enumerate()
    {
        let digest: &[u8; 32] = digest
            .try_into()
            .map_err(|_| invalid("covariance source digest width changed"))?;
        let locator = covariance_identified_id(
            b"source",
            &block.burst_id,
            block.source_manifest_digest,
            block.source_model_version_digest,
            u64::from(block.generation),
            source_secondary,
            block.native_grid,
            native_index,
        )?;
        let expected = covariance_content_bound_source_id(locator, digest)?;
        ensure_valid(
            actual == expected,
            "replayable covariance source ID is not canonically derived",
        )?;
    }
    for (output_index, &actual) in block.phase_node_ids.iter().enumerate() {
        let expected = covariance_identified_id(
            b"phase",
            &block.burst_id,
            block.source_manifest_digest,
            block.source_model_version_digest,
            block.block_id,
            0,
            block.output_grid,
            output_index,
        )?;
        ensure_valid(
            actual == expected,
            "replayable covariance phase-node ID is not canonically derived",
        )?;
    }
    for (native_index, &actual) in block.compressed_node_ids.iter().enumerate() {
        let expected = covariance_identified_id(
            b"compressed",
            &block.burst_id,
            block.source_manifest_digest,
            block.source_model_version_digest,
            block.block_id,
            0,
            block.native_grid,
            native_index,
        )?;
        ensure_valid(
            actual == expected,
            "replayable covariance compressed-node ID is not canonically derived",
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CovarianceSourceLocation {
    block_id: u64,
    native_index: usize,
}

#[derive(Debug, Clone)]
struct CovarianceChainIndex {
    burst_id: String,
    native_grid: CovarianceOperatorGrid,
    block_ids_by_generation: BTreeMap<u32, u64>,
}

impl CovarianceChainIndex {
    fn from_state(state: &CovarianceTopologyState) -> Result<Self> {
        let first = state
            .blocks
            .values()
            .next()
            .ok_or_else(|| invalid("covariance tile chain is empty"))?;
        let mut block_ids_by_generation = BTreeMap::new();
        for block in state.blocks.values() {
            ensure_valid(
                block.burst_id == first.burst_id && block.native_grid == first.native_grid,
                "covariance tile chain changes burst or native grid",
            )?;
            ensure_valid(
                block_ids_by_generation
                    .insert(block.generation, block.block_id)
                    .is_none(),
                "covariance tile chain repeats a generation",
            )?;
        }
        Ok(Self {
            burst_id: first.burst_id.clone(),
            native_grid: first.native_grid,
            block_ids_by_generation,
        })
    }
}

#[derive(Debug)]
struct CovarianceBurstInvariant {
    output_stride: (u32, u32),
    rect_support: CovarianceRectSupport,
    estimator_branch: CovarianceEstimatorBranch,
    branch_tolerance_bits: u64,
    dates_by_generation: BTreeMap<u32, Vec<u32>>,
    expected_tiles: Vec<CovarianceTilePlan>,
    future_min_native_rows: Vec<u64>,
    next_tile_index: usize,
    last_owned_origin: (u64, u64),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CovarianceIdentityRecord {
    kind: u8,
    id: u64,
    fingerprint: [u8; 16],
}

impl CovarianceIdentityRecord {
    const fn same_key(self, other: Self) -> bool {
        self.kind == other.kind && self.id == other.id
    }
}

#[derive(Debug)]
struct CovarianceIdentityRun {
    path: PathBuf,
    record_count: u64,
}

impl CovarianceIdentityRun {
    fn bytes(&self) -> Result<u64> {
        self.record_count
            .checked_mul(IDENTITY_RECORD_BYTES)
            .ok_or_else(|| invalid("covariance identity-index byte count overflow"))
    }
}

impl Drop for CovarianceIdentityRun {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

struct CovarianceIdentityRunReader {
    reader: BufReader<File>,
    remaining: u64,
}

impl CovarianceIdentityRunReader {
    fn open(run: &CovarianceIdentityRun) -> Result<Self> {
        let file =
            File::open(&run.path).map_err(|error| identity_io("opening a sorted run", error))?;
        Ok(Self {
            reader: BufReader::new(file),
            remaining: run.record_count,
        })
    }

    fn next(&mut self) -> Result<Option<CovarianceIdentityRecord>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let mut encoded = [0_u8; IDENTITY_RECORD_BYTES as usize];
        self.reader
            .read_exact(&mut encoded)
            .map_err(|error| identity_io("reading a sorted run", error))?;
        self.remaining -= 1;
        let kind = encoded[0];
        ensure_valid(
            matches!(kind, SOURCE_IDENTITY_KIND | NODE_IDENTITY_KIND),
            "covariance identity index contains an unknown kind",
        )?;
        ensure_valid(
            encoded[1..8].iter().all(|byte| *byte == 0),
            "covariance identity index contains nonzero reserved bytes",
        )?;
        let id = u64::from_le_bytes(
            encoded[8..16]
                .try_into()
                .map_err(|_| invalid("covariance identity ID width changed"))?,
        );
        let fingerprint = encoded[16..32]
            .try_into()
            .map_err(|_| invalid("covariance identity fingerprint width changed"))?;
        Ok(Some(CovarianceIdentityRecord {
            kind,
            id,
            fingerprint,
        }))
    }
}

#[derive(Debug)]
struct CovarianceIdentityIndex {
    directory: PathBuf,
    disk_cap_bytes: u64,
    levels: Vec<Option<CovarianceIdentityRun>>,
    current_disk_bytes: u64,
    peak_disk_bytes: u64,
    bytes_read: u64,
    bytes_written: u64,
    merge_count: u64,
    peak_block_records: usize,
}

impl CovarianceIdentityIndex {
    fn create(scratch_path: &Path, disk_cap_bytes: u64) -> Result<Self> {
        ensure_valid(
            disk_cap_bytes > 0,
            "covariance identity-index disk cap must be positive",
        )?;
        let directory = identity_workspace_path(scratch_path)?;
        std::fs::create_dir(&directory).map_err(|error| {
            identity_io(
                "creating the exclusive workspace; recover the exact incomplete scratch before retrying",
                error,
            )
        })?;
        Ok(Self {
            directory,
            disk_cap_bytes,
            levels: Vec::new(),
            current_disk_bytes: 0,
            peak_disk_bytes: 0,
            bytes_read: 0,
            bytes_written: 0,
            merge_count: 0,
            peak_block_records: 0,
        })
    }

    fn add_block(&mut self, block: &CovarianceBlockTopology) -> Result<()> {
        let records = covariance_identity_records(block)?;
        self.peak_block_records = self.peak_block_records.max(records.len());
        let maximum_run_bytes = u64::try_from(records.len())
            .ok()
            .and_then(|count| count.checked_mul(IDENTITY_RECORD_BYTES))
            .ok_or_else(|| invalid("covariance identity-index block byte count overflow"))?;
        self.ensure_disk_room(maximum_run_bytes)?;
        let run = write_identity_run(&self.directory, records)?;
        let run_bytes = run.bytes()?;
        self.add_written_run(run_bytes)?;
        self.insert_run(run, 0)
    }

    fn finish(&mut self) -> Result<()> {
        let mut carry: Option<CovarianceIdentityRun> = None;
        for level in 0..self.levels.len() {
            let Some(run) = self.levels[level].take() else {
                continue;
            };
            carry = Some(match carry.take() {
                None => run,
                Some(prior) => self.merge_runs(prior, run)?,
            });
        }
        self.levels.clear();
        if let Some(run) = carry {
            self.levels.push(Some(run));
        }
        Ok(())
    }

    fn insert_run(&mut self, mut carry: CovarianceIdentityRun, mut level: usize) -> Result<()> {
        loop {
            if self.levels.len() <= level {
                self.levels.resize_with(level + 1, || None);
            }
            let Some(prior) = self.levels[level].take() else {
                self.levels[level] = Some(carry);
                return Ok(());
            };
            carry = self.merge_runs(prior, carry)?;
            level = level
                .checked_add(1)
                .ok_or_else(|| invalid("covariance identity-index level overflow"))?;
        }
    }

    fn merge_runs(
        &mut self,
        left: CovarianceIdentityRun,
        right: CovarianceIdentityRun,
    ) -> Result<CovarianceIdentityRun> {
        let left_bytes = left.bytes()?;
        let right_bytes = right.bytes()?;
        let maximum_merged_bytes = left_bytes
            .checked_add(right_bytes)
            .ok_or_else(|| invalid("covariance identity-index merge byte count overflow"))?;
        self.ensure_disk_room(maximum_merged_bytes)?;
        let merged = merge_identity_runs(&self.directory, &left, &right)?;
        let merged_bytes = merged.bytes()?;
        self.bytes_read = self
            .bytes_read
            .checked_add(left_bytes)
            .and_then(|bytes| bytes.checked_add(right_bytes))
            .ok_or_else(|| invalid("covariance identity-index read count overflow"))?;
        self.merge_count = self
            .merge_count
            .checked_add(1)
            .ok_or_else(|| invalid("covariance identity-index merge count overflow"))?;
        self.add_written_run(merged_bytes)?;
        self.current_disk_bytes = self
            .current_disk_bytes
            .checked_sub(left_bytes)
            .and_then(|bytes| bytes.checked_sub(right_bytes))
            .ok_or_else(|| invalid("covariance identity-index disk accounting underflow"))?;
        drop(left);
        drop(right);
        Ok(merged)
    }

    fn add_written_run(&mut self, bytes: u64) -> Result<()> {
        self.bytes_written = self
            .bytes_written
            .checked_add(bytes)
            .ok_or_else(|| invalid("covariance identity-index write count overflow"))?;
        self.current_disk_bytes = self
            .current_disk_bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid("covariance identity-index disk count overflow"))?;
        self.peak_disk_bytes = self.peak_disk_bytes.max(self.current_disk_bytes);
        Ok(())
    }

    fn ensure_disk_room(&self, additional_bytes: u64) -> Result<()> {
        let required = self
            .current_disk_bytes
            .checked_add(additional_bytes)
            .ok_or_else(|| invalid("covariance identity-index disk count overflow"))?;
        if required > self.disk_cap_bytes {
            return Err(invalid(format!(
                "covariance identity index requires {required} temporary bytes but its cap is {}",
                self.disk_cap_bytes
            )));
        }
        Ok(())
    }
}

impl Drop for CovarianceIdentityIndex {
    fn drop(&mut self) {
        self.levels.clear();
        let _ = std::fs::remove_dir(&self.directory);
    }
}

fn covariance_identity_records(
    block: &CovarianceBlockTopology,
) -> Result<Vec<CovarianceIdentityRecord>> {
    let count = block
        .source_ids
        .len()
        .checked_add(block.phase_node_ids.len())
        .and_then(|count| count.checked_add(block.compressed_node_ids.len()))
        .ok_or_else(|| invalid("covariance block identity count overflow"))?;
    let mut records = Vec::with_capacity(count);
    for (native_index, &id) in block.source_ids.iter().enumerate() {
        records.push(CovarianceIdentityRecord {
            kind: SOURCE_IDENTITY_KIND,
            id,
            fingerprint: source_identity_fingerprint(block, native_index),
        });
    }
    records.extend(
        block
            .phase_node_ids
            .iter()
            .chain(block.compressed_node_ids.iter())
            .map(|&id| CovarianceIdentityRecord {
                kind: NODE_IDENTITY_KIND,
                id,
                fingerprint: [0; 16],
            }),
    );
    Ok(records)
}

fn source_identity_fingerprint(block: &CovarianceBlockTopology, native_index: usize) -> [u8; 16] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:covariance_source_identity:v2");
    digest.update((block.burst_id.len() as u64).to_le_bytes());
    digest.update(block.burst_id.as_bytes());
    digest.update(block.generation.to_le_bytes());
    digest.update((block.source_date_indices.len() as u64).to_le_bytes());
    for date in &block.source_date_indices {
        digest.update(date.to_le_bytes());
    }
    let (row, column) = native_coordinate(block.native_grid, native_index);
    digest.update(row.to_le_bytes());
    digest.update(column.to_le_bytes());
    digest.update(source_digest(block, native_index));
    digest.update(source_factor_digest(block, native_index));
    digest.finalize()[..16]
        .try_into()
        .expect("SHA-256 has at least 16 bytes")
}

fn write_identity_run(
    directory: &Path,
    mut records: Vec<CovarianceIdentityRecord>,
) -> Result<CovarianceIdentityRun> {
    records.sort_unstable();
    let (path, file) = create_identity_run_file(directory)?;
    let mut run = CovarianceIdentityRun {
        path,
        record_count: 0,
    };
    let mut writer = BufWriter::new(file);
    let mut prior = None;
    for record in records {
        if prior.is_some_and(|value: CovarianceIdentityRecord| value.same_key(record)) {
            identity_duplicate_allowed(prior.expect("prior identity exists"), record)?;
            continue;
        }
        write_identity_record(&mut writer, record)?;
        run.record_count = run
            .record_count
            .checked_add(1)
            .ok_or_else(|| invalid("covariance identity-index record count overflow"))?;
        prior = Some(record);
    }
    writer
        .flush()
        .map_err(|error| identity_io("flushing a sorted run", error))?;
    Ok(run)
}

fn merge_identity_runs(
    directory: &Path,
    left: &CovarianceIdentityRun,
    right: &CovarianceIdentityRun,
) -> Result<CovarianceIdentityRun> {
    let mut left_reader = CovarianceIdentityRunReader::open(left)?;
    let mut right_reader = CovarianceIdentityRunReader::open(right)?;
    let (path, file) = create_identity_run_file(directory)?;
    let mut run = CovarianceIdentityRun {
        path,
        record_count: 0,
    };
    let mut writer = BufWriter::new(file);
    let mut left_record = left_reader.next()?;
    let mut right_record = right_reader.next()?;
    while left_record.is_some() || right_record.is_some() {
        let (record, advance_left, advance_right) = match (left_record, right_record) {
            (Some(left), Some(right)) if left.same_key(right) => {
                identity_duplicate_allowed(left, right)?;
                (left, true, true)
            }
            (Some(left), Some(right)) if left < right => (left, true, false),
            (Some(_), Some(right)) => (right, false, true),
            (Some(left), None) => (left, true, false),
            (None, Some(right)) => (right, false, true),
            (None, None) => break,
        };
        write_identity_record(&mut writer, record)?;
        run.record_count = run
            .record_count
            .checked_add(1)
            .ok_or_else(|| invalid("covariance identity-index record count overflow"))?;
        if advance_left {
            left_record = left_reader.next()?;
        }
        if advance_right {
            right_record = right_reader.next()?;
        }
    }
    writer
        .flush()
        .map_err(|error| identity_io("flushing a merged run", error))?;
    Ok(run)
}

fn identity_duplicate_allowed(
    left: CovarianceIdentityRecord,
    right: CovarianceIdentityRecord,
) -> Result<()> {
    if left.kind == SOURCE_IDENTITY_KIND && left.fingerprint == right.fingerprint {
        return Ok(());
    }
    let message = match left.kind {
        SOURCE_IDENTITY_KIND => "one covariance source ID identifies different primitive sources",
        _ => "covariance phase/compressed node ID is not globally unique",
    };
    Err(invalid(message))
}

fn write_identity_record(
    writer: &mut BufWriter<File>,
    record: CovarianceIdentityRecord,
) -> Result<()> {
    let mut encoded = [0_u8; IDENTITY_RECORD_BYTES as usize];
    encoded[0] = record.kind;
    encoded[8..16].copy_from_slice(&record.id.to_le_bytes());
    encoded[16..32].copy_from_slice(&record.fingerprint);
    writer
        .write_all(&encoded)
        .map_err(|error| identity_io("writing a sorted run", error))
}

fn create_identity_run_file(directory: &Path) -> Result<(PathBuf, File)> {
    for _ in 0..1024 {
        let nonce = IDENTITY_RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!("{nonce:016x}.run"));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(identity_io("creating a sorted run", error)),
        }
    }
    Err(invalid(
        "covariance identity index exhausted temporary run names",
    ))
}

fn identity_workspace_path(scratch_path: &Path) -> Result<PathBuf> {
    let file_name = scratch_path
        .file_name()
        .ok_or_else(|| invalid("covariance scratch path has no file name"))?;
    let mut workspace_name = file_name.to_os_string();
    workspace_name.push(".identity-index");
    Ok(scratch_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(workspace_name))
}

/// Remove one explicitly named incomplete scratch artifact and its identity workspace.
///
/// The path must end in `.scratch`; committed operator paths are rejected.
///
/// # Errors
/// Returns an error for a non-scratch path or a filesystem cleanup failure.
pub fn recover_incomplete_covariance_operator(scratch_path: impl AsRef<Path>) -> Result<()> {
    let scratch_path = scratch_path.as_ref();
    let name = scratch_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid("covariance scratch path has no UTF-8 file name"))?;
    ensure_valid(
        name.ends_with(".scratch"),
        "covariance recovery refuses a path that does not end in .scratch",
    )?;
    let workspace = identity_workspace_path(scratch_path)?;
    match std::fs::remove_dir_all(&workspace) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(identity_io("removing the identity workspace", error)),
    }
    match std::fs::remove_file(scratch_path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(identity_io("removing the incomplete HDF5 scratch", error)),
    }
}

fn identity_io(operation: &str, error: std::io::Error) -> IoError {
    invalid(format!(
        "covariance identity index failed while {operation}: {error}"
    ))
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
    source_manifest_digest: [u8; 32],
    source_model_version_digest: [u8; 32],
    expected_generations: BTreeMap<String, BTreeMap<u32, Vec<u32>>>,
    expected_tiles: BTreeMap<String, Vec<CovarianceTilePlan>>,
    identity_index: CovarianceIdentityIndex,
    poisoned: bool,
    current_chain: CovarianceTopologyState,
    overlap_frontier: Vec<CovarianceChainIndex>,
    active_overlap_chains: Vec<CovarianceChainIndex>,
    burst_invariants: BTreeMap<String, CovarianceBurstInvariant>,
    active_burst: Option<String>,
    block_count: u64,
    overlap_topology_reads: u64,
    overlap_topology_bytes: u64,
    peak_retained_topology_blocks: usize,
    peak_frontier_chains: usize,
}

/// Resource receipt returned after an operator scratch file is sealed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CovarianceOperatorWriteReceipt {
    /// Exact cap required by the header-only metadata validation pass.
    pub metadata_validation_bytes: u64,
    /// Prior overlap blocks streamed from HDF5 during cross-tile validation.
    pub overlap_topology_reads: u64,
    /// Logical topology bytes streamed for those overlap checks.
    pub overlap_topology_bytes: u64,
    /// Maximum full topology blocks retained in memory by the active tile chain.
    pub peak_retained_topology_blocks: usize,
    /// Maximum lightweight tile descriptors retained in the spatial frontier.
    pub peak_frontier_chains: usize,
    /// Sorted-run identity-index bytes read while enforcing global uniqueness.
    pub identity_index_bytes_read: u64,
    /// Sorted-run identity-index bytes written while enforcing global uniqueness.
    pub identity_index_bytes_written: u64,
    /// Maximum temporary disk occupied by the bounded identity index.
    pub peak_identity_index_disk_bytes: u64,
    /// Maximum source plus node identity records allocated for one block.
    pub peak_identity_block_records: usize,
    /// Number of bounded sorted-run merges performed by the identity index.
    pub identity_index_merges: u64,
    /// Fail-closed cap applied to the temporary disk-backed identity index.
    pub identity_index_disk_cap_bytes: u64,
    /// HDF5 byte count sealed after the complete marker was flushed.
    pub sealed_hdf5_bytes: u64,
    /// Lowercase SHA-256 digest sealed after the complete marker was flushed.
    pub sealed_hdf5_sha256: String,
}

impl CovarianceOperatorWriter {
    /// Create an incomplete scratch artifact and persist its checked registries.
    pub fn create(
        path: impl AsRef<Path>,
        metadata: &CovarianceOperatorMetadata,
        plan: &CovarianceOperatorPlan,
    ) -> Result<Self> {
        Self::create_with_identity_index_disk_cap(path, metadata, plan, u64::MAX)
    }

    /// Create a scratch artifact with an explicit temporary identity-index disk cap.
    pub fn create_with_identity_index_disk_cap(
        path: impl AsRef<Path>,
        metadata: &CovarianceOperatorMetadata,
        plan: &CovarianceOperatorPlan,
        identity_index_disk_cap_bytes: u64,
    ) -> Result<Self> {
        metadata.validate()?;
        plan.validate(metadata)?;
        let path = path.as_ref();
        let identity_index = CovarianceIdentityIndex::create(path, identity_index_disk_cap_bytes)?;
        let file = hdf5::File::create(path)?;
        write_metadata(&file, metadata)?;
        write_registries(&file)?;
        file.create_group("blocks")?;
        file.new_attr::<u8>().create("complete")?.write_scalar(&0)?;
        file.flush()?;
        Ok(Self {
            file,
            metadata: metadata.clone(),
            source_manifest_digest: plan.source_manifest_digest,
            source_model_version_digest: plan.source_model_version_digest,
            expected_generations: plan.expected_generations()?,
            expected_tiles: plan.expected_tiles(),
            identity_index,
            poisoned: false,
            current_chain: CovarianceTopologyState::default(),
            overlap_frontier: Vec::new(),
            active_overlap_chains: Vec::new(),
            burst_invariants: BTreeMap::new(),
            active_burst: None,
            block_count: 0,
            overlap_topology_reads: 0,
            overlap_topology_bytes: 0,
            peak_retained_topology_blocks: 0,
            peak_frontier_chains: 0,
        })
    }

    /// Append one validated block without constructing expanded incidence tensors.
    pub fn write_block(&mut self, block: &CovarianceOperatorBlock) -> Result<()> {
        ensure_valid(
            !self.poisoned,
            "covariance operator writer is poisoned by an earlier rejected block",
        )?;
        let result = self.write_block_inner(block);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn write_block_inner(&mut self, block: &CovarianceOperatorBlock) -> Result<()> {
        block.validate(self.metadata.gauge_date_index)?;
        let topology = CovarianceBlockTopology::from(block);
        let group_name = format!("blocks/{:020}", block.block_id);
        if self.file.link_exists(&group_name) {
            return Err(invalid(format!(
                "duplicate covariance operator block {}",
                block.block_id
            )));
        }
        if topology.generation == 0 {
            self.begin_tile_chain(&topology)?;
        }
        ensure_valid(
            topology.source_manifest_digest == self.source_manifest_digest
                && topology.source_model_version_digest == self.source_model_version_digest,
            "covariance block source namespace differs from the capture plan",
        )?;
        ensure_valid(
            self.current_chain
                .blocks
                .values()
                .all(|prior| prior.generation != topology.generation),
            "covariance tile chain repeats a generation",
        )?;
        self.validate_burst_invariant(&topology)?;
        self.current_chain
            .validate(&topology, self.metadata.stitched_status)?;
        self.validate_active_overlaps(&topology)?;
        if self.metadata.replay_status == CovarianceReplayStatus::Replayable {
            validate_replayable_ids(&topology)?;
        }
        self.identity_index.add_block(&topology)?;
        let group = self.file.create_group(&group_name)?;
        write_block(&group, block)?;
        inspect_block_layout(&group)?;
        self.file.flush()?;
        self.current_chain.insert(topology);
        self.peak_retained_topology_blocks = self
            .peak_retained_topology_blocks
            .max(self.current_chain.blocks.len());
        self.block_count = self
            .block_count
            .checked_add(1)
            .ok_or_else(|| invalid("covariance operator block count overflow"))?;
        Ok(())
    }

    fn begin_tile_chain(&mut self, topology: &CovarianceBlockTopology) -> Result<()> {
        if !self.current_chain.blocks.is_empty() {
            self.validate_current_chain_complete()?;
            self.overlap_frontier
                .push(CovarianceChainIndex::from_state(&self.current_chain)?);
            self.current_chain = CovarianceTopologyState::default();
        }
        if self.active_burst.as_deref() != Some(topology.burst_id.as_str()) {
            ensure_valid(
                !self.burst_invariants.contains_key(&topology.burst_id),
                "covariance burst tile chains are not contiguous",
            )?;
            self.overlap_frontier.clear();
            self.active_burst = Some(topology.burst_id.clone());
        }
        let future_min_native_row = match self.burst_invariants.get(&topology.burst_id) {
            Some(invariant) => invariant
                .future_min_native_rows
                .get(invariant.next_tile_index)
                .copied(),
            None => self
                .expected_tiles
                .get(&topology.burst_id)
                .and_then(|tiles| future_min_native_rows(tiles).first().copied()),
        }
        .ok_or_else(|| invalid("covariance operator writes an unplanned tile chain"))?;
        self.overlap_frontier.retain(|chain| {
            chain.burst_id == topology.burst_id
                && grid_stop(chain.native_grid.row_start, chain.native_grid.rows)
                    .is_ok_and(|stop| stop > future_min_native_row)
        });
        self.active_overlap_chains = self
            .overlap_frontier
            .iter()
            .filter(|chain| grids_overlap(chain.native_grid, topology.native_grid))
            .cloned()
            .collect();
        self.peak_frontier_chains = self.peak_frontier_chains.max(self.overlap_frontier.len());
        Ok(())
    }

    fn validate_burst_invariant(&mut self, topology: &CovarianceBlockTopology) -> Result<()> {
        if !self.burst_invariants.contains_key(&topology.burst_id) {
            let dates_by_generation = self
                .expected_generations
                .get(&topology.burst_id)
                .ok_or_else(|| invalid("covariance block burst is absent from the capture plan"))?
                .clone();
            let expected_tiles = self
                .expected_tiles
                .get(&topology.burst_id)
                .ok_or_else(|| invalid("covariance block burst is absent from the tile plan"))?
                .clone();
            let future_min_native_rows = future_min_native_rows(&expected_tiles);
            ensure_valid(
                self.burst_invariants.is_empty()
                    || self.metadata.stitched_status
                        == StitchedCovarianceStatus::UnsupportedSeamCovariance,
                "multiple covariance bursts require unsupported-seam stitched status",
            )?;
            self.burst_invariants.insert(
                topology.burst_id.clone(),
                CovarianceBurstInvariant {
                    output_stride: (topology.output_grid.stride_y, topology.output_grid.stride_x),
                    rect_support: topology.rect_support,
                    estimator_branch: topology.estimator_branch,
                    branch_tolerance_bits: topology.branch_tolerance_bits,
                    dates_by_generation,
                    expected_tiles,
                    future_min_native_rows,
                    next_tile_index: 0,
                    last_owned_origin: (
                        topology.owned_output_grid.row_start,
                        topology.owned_output_grid.col_start,
                    ),
                },
            );
        }
        let invariant = self
            .burst_invariants
            .get_mut(&topology.burst_id)
            .ok_or_else(|| invalid("covariance burst invariant is missing"))?;
        ensure_valid(
            invariant.output_stride
                == (topology.output_grid.stride_y, topology.output_grid.stride_x)
                && invariant.rect_support == topology.rect_support
                && invariant.estimator_branch == topology.estimator_branch
                && invariant.branch_tolerance_bits == topology.branch_tolerance_bits,
            "covariance records in one burst differ in geometry, estimator branch, or tolerance",
        )?;
        if topology.generation == 0 {
            let owned_origin = (
                topology.owned_output_grid.row_start,
                topology.owned_output_grid.col_start,
            );
            if invariant.next_tile_index > 0 {
                ensure_valid(
                    owned_origin > invariant.last_owned_origin,
                    "covariance tile chains are not written in strict row-major order",
                )?;
            }
            let planned_tile = invariant
                .expected_tiles
                .get(invariant.next_tile_index)
                .ok_or_else(|| invalid("covariance operator writes an unplanned tile chain"))?;
            ensure_valid(
                planned_tile.native_grid == topology.native_grid
                    && planned_tile.output_grid == topology.output_grid
                    && planned_tile.owned_output_grid == topology.owned_output_grid,
                "covariance tile grids differ from the capture plan",
            )?;
            invariant.next_tile_index = invariant
                .next_tile_index
                .checked_add(1)
                .ok_or_else(|| invalid("covariance tile-plan index overflow"))?;
            invariant.last_owned_origin = owned_origin;
        }
        match invariant.dates_by_generation.get(&topology.generation) {
            Some(dates) => ensure_valid(
                dates == &topology.source_date_indices,
                "covariance tiles in one generation date map differ",
            ),
            None => Err(invalid(
                "covariance block generation is absent from the capture plan",
            )),
        }
    }

    fn validate_current_chain_complete(&self) -> Result<()> {
        let Some(first) = self.current_chain.blocks.values().next() else {
            return Ok(());
        };
        let expected = self
            .burst_invariants
            .get(&first.burst_id)
            .ok_or_else(|| invalid("covariance burst invariant is missing"))?;
        let actual = self
            .current_chain
            .blocks
            .values()
            .map(|block| block.generation)
            .collect::<BTreeSet<_>>();
        let planned = expected
            .dates_by_generation
            .keys()
            .copied()
            .collect::<BTreeSet<_>>();
        ensure_valid(
            actual == planned && self.current_chain.blocks.len() == planned.len(),
            "covariance tile chain omits a planned generation",
        )
    }

    fn validate_active_overlaps(&mut self, topology: &CovarianceBlockTopology) -> Result<()> {
        let block_ids = self
            .active_overlap_chains
            .iter()
            .map(|chain| {
                chain
                    .block_ids_by_generation
                    .get(&topology.generation)
                    .copied()
                    .ok_or_else(|| invalid("overlapping covariance tile omits a generation"))
            })
            .collect::<Result<Vec<_>>>()?;
        for block_id in block_ids {
            let name = format!("blocks/{block_id:020}");
            let group = self.file.group(&name)?;
            let topology_bytes = inspect_topology_layout(&group)?;
            let prior = read_topology_header(&group)?;
            validate_topology_header(
                &prior,
                self.metadata.gauge_date_index,
                usize::try_from(self.block_count)
                    .map_err(|_| invalid("covariance operator block count exceeds usize"))?,
            )?;
            let mut state = CovarianceTopologyState::default();
            state.insert(prior);
            validate_cross_record_topology(topology, &state, self.metadata.stitched_status)?;
            self.overlap_topology_reads = self
                .overlap_topology_reads
                .checked_add(1)
                .ok_or_else(|| invalid("covariance overlap topology-read count overflow"))?;
            self.overlap_topology_bytes =
                self.overlap_topology_bytes
                    .checked_add(topology_bytes)
                    .ok_or_else(|| invalid("covariance overlap topology byte count overflow"))?;
        }
        Ok(())
    }

    /// Number of topology blocks retained for rolling validation.
    ///
    /// The writer keeps only the current spatial tile chain. Cross-tile checks
    /// read only prior blocks whose native grids overlap the current tile.
    #[must_use]
    pub fn retained_topology_block_count(&self) -> usize {
        self.current_chain.blocks.len()
    }

    /// Mark the artifact complete, flush all block data, and return the exact
    /// header-allocation cap and the sealed HDF5 byte receipt.
    pub fn finish(mut self) -> Result<CovarianceOperatorWriteReceipt> {
        ensure_valid(
            !self.poisoned,
            "covariance operator writer is poisoned by an earlier rejected block",
        )?;
        ensure_valid(
            self.block_count > 0,
            "covariance operator requires at least one block",
        )?;
        self.validate_current_chain_complete()?;
        ensure_valid(
            self.burst_invariants.len() == self.expected_generations.len()
                && self
                    .expected_generations
                    .keys()
                    .all(|burst| self.burst_invariants.contains_key(burst)),
            "covariance operator omits a planned burst",
        )?;
        ensure_valid(
            self.burst_invariants
                .values()
                .all(|invariant| invariant.next_tile_index == invariant.expected_tiles.len()),
            "covariance operator omits a planned tile chain",
        )?;
        self.identity_index.finish()?;
        validate_root_schema(&self.file)?;
        inspect_metadata_layout(&self.file)?;
        validate_registries(&self.file)?;
        ensure_valid(
            read_metadata(&self.file)? == self.metadata,
            "covariance operator metadata changed before finalization",
        )?;
        let blocks = self.file.group("blocks")?;
        ensure_valid(
            blocks.len() == self.block_count,
            "covariance operator block index changed before finalization",
        )?;
        drop(blocks);
        let metadata_validation_bytes = inspect_metadata_layout(&self.file)?;
        self.file.attr("complete")?.write_scalar(&1u8)?;
        self.file.flush()?;
        let filename = self.file.filename();
        self.file.close()?;
        let (sealed_hdf5_sha256, sealed_hdf5_bytes) = sha256_path(Path::new(&filename))?;
        Ok(CovarianceOperatorWriteReceipt {
            metadata_validation_bytes,
            overlap_topology_reads: self.overlap_topology_reads,
            overlap_topology_bytes: self.overlap_topology_bytes,
            peak_retained_topology_blocks: self.peak_retained_topology_blocks,
            peak_frontier_chains: self.peak_frontier_chains,
            identity_index_bytes_read: self.identity_index.bytes_read,
            identity_index_bytes_written: self.identity_index.bytes_written,
            peak_identity_index_disk_bytes: self.identity_index.peak_disk_bytes,
            peak_identity_block_records: self.identity_index.peak_block_records,
            identity_index_merges: self.identity_index.merge_count,
            identity_index_disk_cap_bytes: self.identity_index.disk_cap_bytes,
            sealed_hdf5_bytes,
            sealed_hdf5_sha256,
        })
    }
}

fn sha256_path(path: &Path) -> Result<(String, u64)> {
    let mut reader = BufReader::new(
        File::open(path).map_err(|error| identity_io("opening the sealed HDF5", error))?,
    );
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut byte_count = 0_u64;
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| identity_io("hashing the sealed HDF5", error))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
        byte_count = byte_count
            .checked_add(count as u64)
            .ok_or_else(|| invalid("covariance HDF5 byte count overflow"))?;
    }
    Ok((format!("{:x}", digest.finalize()), byte_count))
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

/// Read and validate only the committed operator header under an allocation cap.
///
/// This streams the block-link names to reject noncanonical or non-hard links,
/// but does not retain those names or load block topology. Production replay
/// validates each selected block against the caller's planned topology.
pub fn read_covariance_operator_header_with_byte_cap(
    path: impl AsRef<Path>,
    byte_cap: u64,
) -> Result<CovarianceOperatorMetadata> {
    let file = hdf5::File::open(path)?;
    let mut budget = ReadBudget::new(byte_cap);
    let metadata = read_checked_metadata(&file, &mut budget)?;
    let blocks = file.group("blocks")?;
    validate_exact_schema(
        &blocks,
        None,
        &[],
        "covariance blocks schema contains unexpected attributes",
    )?;
    ensure_valid(
        !blocks.is_empty(),
        "covariance operator requires at least one block",
    )?;
    validate_block_links_streaming(&blocks)?;
    Ok(metadata)
}

/// One validated block plus the logical bytes retained in its cache payload.
#[derive(Debug, Clone, PartialEq)]
pub struct CovarianceOperatorBlockRead {
    /// Validated persisted replay block.
    pub block: CovarianceOperatorBlock,
    /// Sum of selected-block dataset payload bytes inspected before allocation.
    pub logical_payload_bytes: u64,
}

/// Reusable reader whose block-link index is validated once at open.
#[derive(Debug)]
pub struct CovarianceOperatorBlockReader {
    file: hdf5::File,
    metadata: CovarianceOperatorMetadata,
}

impl CovarianceOperatorBlockReader {
    /// Open an operator and validate its header plus every block link once.
    ///
    /// # Errors
    /// Returns an error for invalid metadata, a missing block set, non-hard or
    /// noncanonical block links, or a metadata allocation above `byte_cap`.
    pub fn open(path: impl AsRef<Path>, byte_cap: u64) -> Result<Self> {
        let file = hdf5::File::open(path)?;
        let mut budget = ReadBudget::new(byte_cap);
        let metadata = read_checked_metadata(&file, &mut budget)?;
        let blocks = file.group("blocks")?;
        validate_exact_schema(
            &blocks,
            None,
            &[],
            "covariance blocks schema contains unexpected attributes",
        )?;
        ensure_valid(
            !blocks.is_empty(),
            "covariance operator requires at least one block",
        )?;
        validate_block_links_streaming(&blocks)?;
        Ok(Self { file, metadata })
    }

    /// Checked operator metadata loaded at open.
    #[must_use]
    pub const fn metadata(&self) -> &CovarianceOperatorMetadata {
        &self.metadata
    }

    /// Read one block without rescanning the already validated block-link set.
    ///
    /// # Errors
    /// Returns an error for a missing/malformed block or allocation above `byte_cap`.
    pub fn read_block_with_receipt(
        &self,
        block_id: u64,
        byte_cap: u64,
    ) -> Result<CovarianceOperatorBlockRead> {
        read_covariance_operator_block_from_file(
            &self.file,
            &self.metadata,
            block_id,
            byte_cap,
            false,
        )
    }
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
    Ok(read_covariance_operator_block_with_receipt(path, block_id, byte_cap)?.block)
}

/// Read one validated block and return its exact logical payload-byte receipt.
///
/// Header metadata, the selected block name, selected payload, and selected
/// topology workspace are covered by `byte_cap`; unrelated block topology is
/// not loaded. `logical_payload_bytes` is the block retained by a one-block
/// replay cache.
pub fn read_covariance_operator_block_with_receipt(
    path: impl AsRef<Path>,
    block_id: u64,
    byte_cap: u64,
) -> Result<CovarianceOperatorBlockRead> {
    let file = hdf5::File::open(path)?;
    let mut budget = ReadBudget::new(byte_cap);
    let metadata = read_checked_metadata(&file, &mut budget)?;
    read_covariance_operator_block_from_file(&file, &metadata, block_id, byte_cap, true)
}

fn read_covariance_operator_block_from_file(
    file: &hdf5::File,
    metadata: &CovarianceOperatorMetadata,
    block_id: u64,
    byte_cap: u64,
    validate_link: bool,
) -> Result<CovarianceOperatorBlockRead> {
    let mut budget = ReadBudget::new(byte_cap);
    budget.charge(inspect_metadata_layout(file)?)?;
    let blocks = file.group("blocks")?;
    validate_exact_schema(
        &blocks,
        None,
        &[],
        "covariance blocks schema contains unexpected attributes",
    )?;
    let name = format!("{block_id:020}");
    budget.charge(
        u64::try_from(name.len())
            .ok()
            .and_then(|length| length.checked_add(BLOCK_NAME_BUDGET_BYTES))
            .ok_or_else(|| invalid("covariance block-name byte count overflow"))?,
    )?;
    if validate_link {
        validate_selected_block_link(&blocks, &name)?;
    }
    let group = blocks.group(&name)?;
    let logical_payload_bytes = inspect_block_layout(&group)?;
    let topology_bytes = inspect_topology_layout(&group)?;
    budget.charge(logical_payload_bytes)?;
    budget.charge(topology_workspace_bytes(topology_bytes)?)?;
    let block = read_block(&group)?;
    block.validate(metadata.gauge_date_index)?;
    let topology = CovarianceBlockTopology::from(&block);
    validate_block_namespace(&topology, metadata)?;
    if metadata.replay_status == CovarianceReplayStatus::Replayable {
        validate_replayable_ids(&topology)?;
    }
    ensure_valid(
        block.block_id == block_id,
        "covariance block group ID mismatch",
    )?;
    Ok(CovarianceOperatorBlockRead {
        block,
        logical_payload_bytes,
    })
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

fn validate_block_links_streaming(blocks: &Group) -> Result<()> {
    let error = blocks.iter_visit_default(None, |group, name, info, error| {
        let checked = (|| {
            ensure_valid(
                info.link_type == LinkType::Hard,
                "covariance block entry is not a hard link",
            )?;
            let parsed = name
                .parse::<u64>()
                .map_err(|_| invalid("invalid covariance block group name"))?;
            ensure_valid(
                name == format!("{parsed:020}"),
                "covariance block group is not a canonical padded ID",
            )?;
            group.group(name)?;
            Ok(())
        })();
        match checked {
            Ok(()) => true,
            Err(cause) => {
                *error = Some(cause);
                false
            }
        }
    })?;
    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn validate_selected_block_link(blocks: &Group, selected: &str) -> Result<()> {
    struct SelectedLink {
        found: bool,
        error: Option<IoError>,
    }
    let result = blocks.iter_visit_default(
        SelectedLink {
            found: false,
            error: None,
        },
        |group, name, info, selected_link| {
            if name != selected {
                return true;
            }
            selected_link.found = true;
            let checked = (|| {
                ensure_valid(
                    info.link_type == LinkType::Hard,
                    "covariance block entry is not a hard link",
                )?;
                group.group(name)?;
                Ok(())
            })();
            if let Err(error) = checked {
                selected_link.error = Some(error);
            }
            false
        },
    )?;
    if let Some(error) = result.error {
        return Err(error);
    }
    ensure_valid(result.found, "covariance block is missing")
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
    add_exact_dataset::<u8>(group, "source_manifest_digest", &[32], &mut bytes)?;
    add_exact_dataset::<u8>(group, "source_model_version_digest", &[32], &mut bytes)?;

    let (date_shape, date_bytes) = inspect_dataset::<u32>(group, "source_date_indices", None)?;
    ensure_valid(
        date_shape.len() == 1 && date_shape[0] > 0,
        "source_date_indices shape is not a nonempty vector",
    )?;
    checked_add_bytes(&mut bytes, date_bytes)?;
    add_exact_dataset::<u32>(group, "ordered_date_indices", &date_shape, &mut bytes)?;
    add_exact_dataset::<u64>(group, "source_ids", &[native_area], &mut bytes)?;
    add_exact_dataset::<u8>(
        group,
        "source_content_digests",
        &[native_area, 32],
        &mut bytes,
    )?;
    add_exact_dataset::<u8>(
        group,
        "source_factor_digests",
        &[native_area, 32],
        &mut bytes,
    )?;
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
        ("source_manifest_digest", 1),
        ("source_model_version_digest", 1),
        ("source_date_indices", 4),
        ("ordered_date_indices", 4),
        ("source_ids", 8),
        ("source_content_digests", 1),
        ("source_factor_digests", 1),
        ("native_validity_bits", 1),
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
        let entry = read_topology_header(&group)?;
        ensure_valid(
            *name == format!("{:020}", entry.block_id),
            "covariance block group ID mismatch",
        )?;
        validate_block_namespace(&entry, metadata)?;
        validate_topology_header(&entry, metadata.gauge_date_index, names.len())?;
        if metadata.replay_status == CovarianceReplayStatus::Replayable {
            validate_replayable_ids(&entry)?;
        }
        topology.validate(&entry, metadata.stitched_status)?;
        topology.insert(entry);
    }
    Ok(())
}

fn validate_block_namespace(
    block: &CovarianceBlockTopology,
    metadata: &CovarianceOperatorMetadata,
) -> Result<()> {
    ensure_valid(
        metadata
            .source
            .manifest_digest
            .as_deref()
            .and_then(sha256_digest_bytes)
            == Some(block.source_manifest_digest),
        "covariance block source manifest differs from operator metadata",
    )?;
    ensure_valid(
        metadata
            .source
            .model_version_digest
            .as_deref()
            .and_then(sha256_digest_bytes)
            == Some(block.source_model_version_digest),
        "covariance block source-model identity differs from operator metadata",
    )
}

fn read_topology_header(group: &Group) -> Result<CovarianceBlockTopology> {
    Ok(CovarianceBlockTopology {
        block_id: read_scalar_attr(group, "block_id")?,
        generation: read_scalar_attr(group, "generation")?,
        burst_id: read_string(group, "burst_id")?,
        source_manifest_digest: read_digest_dataset(group, "source_manifest_digest")?,
        source_model_version_digest: read_digest_dataset(group, "source_model_version_digest")?,
        native_grid: read_grid(group, "native_grid")?,
        output_grid: read_grid(group, "output_grid")?,
        owned_output_grid: read_grid(group, "owned_output_grid")?,
        rect_support: read_rect_support(group)?,
        reference_date_index: read_scalar_attr(group, "reference_date_index")?,
        source_date_indices: group.dataset("source_date_indices")?.read_raw()?,
        ordered_date_indices: group.dataset("ordered_date_indices")?.read_raw()?,
        source_ids: group.dataset("source_ids")?.read_raw()?,
        source_content_digests: group.dataset("source_content_digests")?.read_raw()?,
        source_factor_digests: group.dataset("source_factor_digests")?.read_raw()?,
        native_validity_bits: group.dataset("native_validity_bits")?.read_raw()?,
        estimator_branch: CovarianceEstimatorBranch::from_code(read_scalar_attr(
            group,
            "estimator_branch",
        )?)?,
        branch_tolerance_bits: read_scalar_attr::<f64>(group, "branch_tolerance")?.to_bits(),
        phase_node_ids: group.dataset("phase_node_ids")?.read_raw()?,
        compressed_node_ids: group.dataset("compressed_node_ids")?.read_raw()?,
        carry_parent_ids: group.dataset("carry_parent_ids")?.read_raw()?,
        phase_components: read_phase_components(group)?,
    })
}

fn read_digest_dataset(group: &Group, name: &str) -> Result<[u8; 32]> {
    let bytes: Vec<u8> = group.dataset(name)?.read_raw()?;
    bytes
        .try_into()
        .map_err(|_| invalid(format!("{name} is not a 32-byte digest")))
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
        "model_version_digest",
        metadata.source.model_version_digest.as_deref(),
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
            model_version_digest: read_optional_string(&source, "model_version_digest")?,
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

#[allow(clippy::too_many_lines)]
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
    write_chunked_1d(
        group,
        "source_manifest_digest",
        &block.source_manifest_digest,
    )?;
    write_chunked_1d(
        group,
        "source_model_version_digest",
        &block.source_model_version_digest,
    )?;
    write_grid(group, "native_grid", block.native_grid)?;
    write_grid(group, "output_grid", block.output_grid)?;
    write_grid(group, "owned_output_grid", block.owned_output_grid)?;
    write_rect_support(group, block.rect_support)?;
    write_chunked_1d(group, "source_date_indices", &block.source_date_indices)?;
    write_chunked_1d(group, "ordered_date_indices", &block.ordered_date_indices)?;
    write_chunked_1d(group, "source_ids", &block.source_ids)?;
    write_chunked_2d(
        group,
        "source_content_digests",
        (block.source_ids.len(), 32),
        &block.source_content_digests,
    )?;
    write_chunked_2d(
        group,
        "source_factor_digests",
        (block.source_ids.len(), 32),
        &block.source_factor_digests,
    )?;
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
        source_manifest_digest: read_digest_dataset(group, "source_manifest_digest")?,
        source_model_version_digest: read_digest_dataset(group, "source_model_version_digest")?,
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
        source_content_digests: group.dataset("source_content_digests")?.read_raw()?,
        source_factor_digests: group.dataset("source_factor_digests")?.read_raw()?,
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

/// HDF5 schema version for bounded reference-specific covariance factors.
pub const SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION: u16 = 4;
const SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION: u16 = 2;
const SPATIAL_REFERENCE_COVARIANCE_PREVIOUS_SCHEMA_VERSION: u16 = 3;
/// Stable method identity for issue #54 reference-specific factors.
pub const SPATIAL_REFERENCE_COVARIANCE_METHOD: &str = "reference_specific_influence_v1";
/// Persisted sentinel for an approximation bound that is unavailable or has not
/// passed the frozen validation scope.
pub const SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE: f64 = f64::NAN;
/// Persisted source-burst index for a masked, failed, mixed, or ambiguous target
/// that cannot claim one source-burst owner.
pub const SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE: u32 = u32::MAX;

const SPATIAL_ROOT_MEMBERS: &[&str] = &["metadata", "full_grid", "blocks"];
const SPATIAL_ROOT_ATTRIBUTES: &[&str] = &[
    "schema_version",
    "method_version",
    "gauge_date_index",
    "calibration_scope",
    "maximum_block_bytes",
    "complete",
];
const SPATIAL_METADATA_MEMBERS_V2: &[&str] = &[
    "method",
    "crate_version",
    "producer_commit",
    "burst_id",
    "crs",
    "units",
    "ordered_date_indices",
    "mask_digest",
    "source_replay_digest",
    "l2_map_digest",
    "reference_signature_digest",
    "approximation_receipt_digest",
    "resource_receipt_digest",
    "review_receipt_digest",
    "method_manifest_digest",
    "calibration_scope_digest",
    "source_model_digest",
    "effective_looks_digest",
    "support_method",
    "support_digest",
    "correction_order_digest",
    "unwrap_branch_digest",
    "burst_ownership_digest",
    "source_burst_ids",
];
const SPATIAL_METADATA_MEMBERS_V3: &[&str] = SPATIAL_METADATA_MEMBERS_V2;
const SPATIAL_METADATA_MEMBERS_V4: &[&str] = &[
    "method",
    "crate_version",
    "producer_commit",
    "burst_id",
    "crs",
    "units",
    "geotransform",
    "ordered_date_indices",
    "acquisition_days",
    "mask_digest",
    "source_replay_digest",
    "l2_map_digest",
    "reference_signature_digest",
    "approximation_receipt_digest",
    "resource_receipt_digest",
    "runtime_resource_receipt_digest",
    "review_receipt_digest",
    "method_manifest_digest",
    "calibration_scope_digest",
    "source_model_digest",
    "effective_looks_digest",
    "support_method",
    "support_digest",
    "correction_order_digest",
    "unwrap_branch_digest",
    "burst_ownership_digest",
    "source_burst_ids",
    "aggregate_byte_cap",
    "factor_block_high_water_bytes",
    "serialization_high_water_bytes",
    "fixed_l2_workspace_bytes",
    "replay_reservation_high_water_bytes",
    "provider_peak_count",
    "provider_peak_bytes",
    "aggregate_high_water_bytes",
];
const SPATIAL_BLOCK_MEMBERS_V2: &[&str] = &[
    "target_grid",
    "rank_by_target",
    "status",
    "source_burst_index_by_target",
    "difference_factor",
    "approximation_error_bound",
    "source_factor_digest",
];
const SPATIAL_BLOCK_MEMBERS_V3: &[&str] = SPATIAL_BLOCK_MEMBERS_V2;
const SPATIAL_BLOCK_MEMBERS_V4: &[&str] = &[
    "target_grid",
    "rank_by_target",
    "status",
    "source_burst_index_by_target",
    "difference_factor",
    "approximation_error_bound",
    "effective_looks_fraction",
    "support_union_count",
    "effective_looks_receipt",
    "resource_high_water_bytes",
    "source_factor_digest",
];
const SPATIAL_BLOCK_ATTRIBUTES: &[&str] = &["block_id", "maximum_rank"];
const SPATIAL_RUNTIME_RESOURCE_MEMBERS: &[&str] = &[
    "aggregate_byte_cap",
    "factor_block_high_water_bytes",
    "serialization_high_water_bytes",
    "fixed_l2_workspace_bytes",
    "replay_reservation_high_water_bytes",
    "provider_peak_count",
    "provider_peak_bytes",
    "aggregate_high_water_bytes",
];

/// Calibration scope bound to a persisted reference-specific factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialReferenceCalibrationScope {
    /// No approximation and independent-review receipt authorizes inference.
    Uncalibrated,
    /// The exact factor configuration matches a successful immutable receipt.
    CalibratedScopeMatch,
}

impl SpatialReferenceCalibrationScope {
    const fn code(self) -> u16 {
        match self {
            Self::Uncalibrated => 0,
            Self::CalibratedScopeMatch => 1,
        }
    }

    fn from_code(code: u16) -> Result<Self> {
        match code {
            0 => Ok(Self::Uncalibrated),
            1 => Ok(Self::CalibratedScopeMatch),
            _ => Err(invalid(format!(
                "unknown spatial reference calibration scope {code}"
            ))),
        }
    }
}

/// Per-target disposition stored with a reference-specific factor block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum SpatialReferenceCovarianceStatus {
    /// The target factor is finite and evaluable for the declared scope.
    Valid = 0,
    /// The target or selected reference is invalid.
    InvalidReference = 1,
    /// Legacy coarse status for phase-link or sequential replay failure.
    ReplayUnsupported = 2,
    /// The fixed-valid-observation L2 map is rank deficient.
    L2RankDeficient = 3,
    /// An input identity or calibration scope does not match.
    ScopeMismatch = 4,
    /// Target/reference ownership is mixed, ambiguous, or date-varying.
    UnsupportedMultiburstReference = 5,
    /// The target is excluded by the production validity mask.
    MaskedTarget = 6,
    /// The exact temporal propagation factor cannot be constructed.
    TemporalFactorInvalid = 7,
    /// The persisted replay source is unavailable.
    ReplayUnavailable = 8,
    /// The persisted replay identity differs from the requested scope.
    ReplayMismatch = 9,
    /// The target/reference influence contraction is invalid.
    InfluenceInvalid = 10,
    /// The selected phase estimator is on a nondifferentiable branch.
    NondifferentiableEstimator = 11,
    /// Adaptive support changed relative to its frozen realization.
    UnstableAdaptiveSupport = 12,
    /// L1 inversion has no fixed production L2 propagation map.
    UnsupportedL1 = 13,
    /// Phase-bias correction is outside the supported covariance scope.
    UnsupportedPhaseBias = 14,
    /// The configured correction/reference order is unsupported.
    UnsupportedCorrectionOrder = 15,
    /// The EVD branch has a tied selected eigenvalue.
    TiedEigenvalue = 16,
    /// The target or reference has no usable source support.
    EmptySupport = 17,
    /// A primitive proper-complex source contains a non-finite sample.
    NonfiniteSource = 18,
    /// The requested downstream estimator/model is outside the supported scope.
    UnsupportedModel = 19,
    /// The production factor or solve exceeded its frozen condition limit.
    IllConditioned = 20,
    /// Realized support identity differs from the frozen support receipt.
    SupportIdentityMismatch = 21,
}

impl SpatialReferenceCovarianceStatus {
    const fn code(self) -> u16 {
        self as u16
    }

    fn from_code(code: u16, schema_version: u16) -> Result<Self> {
        let status = match code {
            0 => Ok(Self::Valid),
            1 => Ok(Self::InvalidReference),
            2 => Ok(Self::ReplayUnsupported),
            3 => Ok(Self::L2RankDeficient),
            4 => Ok(Self::ScopeMismatch),
            5 => Ok(Self::UnsupportedMultiburstReference),
            6 => Ok(Self::MaskedTarget),
            7 => Ok(Self::TemporalFactorInvalid),
            8 => Ok(Self::ReplayUnavailable),
            9 => Ok(Self::ReplayMismatch),
            10 => Ok(Self::InfluenceInvalid),
            11 => Ok(Self::NondifferentiableEstimator),
            12 => Ok(Self::UnstableAdaptiveSupport),
            13 => Ok(Self::UnsupportedL1),
            14 => Ok(Self::UnsupportedPhaseBias),
            15 => Ok(Self::UnsupportedCorrectionOrder),
            16 => Ok(Self::TiedEigenvalue),
            17 => Ok(Self::EmptySupport),
            18 => Ok(Self::NonfiniteSource),
            19 => Ok(Self::UnsupportedModel),
            20 => Ok(Self::IllConditioned),
            21 => Ok(Self::SupportIdentityMismatch),
            _ => Err(invalid(format!(
                "unknown spatial reference covariance status {code}"
            ))),
        }?;
        ensure_valid(
            schema_version != SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION || code <= 5,
            "legacy spatial reference covariance status is outside schema v2",
        )?;
        Ok(status)
    }
}

/// Stable reference-specific covariance status registry.
///
/// Codes `0..=5` retain the schema-v2 meanings already written by older
/// producers. New producers should prefer the detailed statuses over the legacy
/// [`SpatialReferenceCovarianceStatus::ReplayUnsupported`] catch-all.
pub const SPATIAL_REFERENCE_COVARIANCE_STATUS_REGISTRY: &[CovarianceRegistryEntry] = &[
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::Valid as u16,
        name: "valid",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::InvalidReference as u16,
        name: "invalid_reference",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::ReplayUnsupported as u16,
        name: "replay_unsupported",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::L2RankDeficient as u16,
        name: "l2_rank_deficient",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::ScopeMismatch as u16,
        name: "scope_mismatch",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::UnsupportedMultiburstReference as u16,
        name: "unsupported_multiburst_reference",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::MaskedTarget as u16,
        name: "masked_target",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::TemporalFactorInvalid as u16,
        name: "temporal_factor_invalid",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::ReplayUnavailable as u16,
        name: "replay_unavailable",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::ReplayMismatch as u16,
        name: "replay_mismatch",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::InfluenceInvalid as u16,
        name: "influence_invalid",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::NondifferentiableEstimator as u16,
        name: "nondifferentiable_estimator",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::UnstableAdaptiveSupport as u16,
        name: "unstable_adaptive_support",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::UnsupportedL1 as u16,
        name: "unsupported_l1",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::UnsupportedPhaseBias as u16,
        name: "unsupported_phase_bias",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::UnsupportedCorrectionOrder as u16,
        name: "unsupported_correction_order",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::TiedEigenvalue as u16,
        name: "tied_eigenvalue",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::EmptySupport as u16,
        name: "empty_support",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::NonfiniteSource as u16,
        name: "nonfinite_source",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::UnsupportedModel as u16,
        name: "unsupported_model",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::IllConditioned as u16,
        name: "ill_conditioned",
    },
    CovarianceRegistryEntry {
        code: SpatialReferenceCovarianceStatus::SupportIdentityMismatch as u16,
        name: "support_identity_mismatch",
    },
];

/// Artifact-level identity for bounded reference-specific covariance blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialReferenceRuntimeResourceReceipt {
    /// Single aggregate production resident-byte cap.
    pub aggregate_byte_cap: u64,
    /// Largest resident in-progress factor block.
    pub factor_block_high_water_bytes: u64,
    /// Largest HDF5 serialization reservation for one factor block.
    pub serialization_high_water_bytes: u64,
    /// Dynamically projected fixed-L2 propagation workspace.
    pub fixed_l2_workspace_bytes: u64,
    /// Largest admitted replay reservation after all other components.
    pub replay_reservation_high_water_bytes: u64,
    /// Observed maximum simultaneously live replay providers.
    pub provider_peak_count: u64,
    /// Observed maximum resident bytes held by live replay providers.
    pub provider_peak_bytes: u64,
    /// Exact sum of the admitted component high-water bounds.
    pub aggregate_high_water_bytes: u64,
}

impl SpatialReferenceRuntimeResourceReceipt {
    fn validate(&self, maximum_block_bytes: u64) -> Result<()> {
        let composed = self
            .factor_block_high_water_bytes
            .checked_add(self.serialization_high_water_bytes)
            .and_then(|bytes| bytes.checked_add(self.fixed_l2_workspace_bytes))
            .and_then(|bytes| bytes.checked_add(self.replay_reservation_high_water_bytes))
            .ok_or_else(|| invalid("spatial reference resource receipt overflows u64"))?;
        ensure_valid(
            self.aggregate_byte_cap == maximum_block_bytes
                && self.aggregate_byte_cap > 0
                && self.factor_block_high_water_bytes > 0
                && self.serialization_high_water_bytes > 0
                && self.fixed_l2_workspace_bytes > 0
                && self.provider_peak_count <= 2
                && self.provider_peak_bytes <= self.replay_reservation_high_water_bytes
                && self.aggregate_high_water_bytes == composed
                && self.aggregate_high_water_bytes <= self.aggregate_byte_cap,
            "spatial reference runtime resource receipt is inconsistent",
        )
    }
}

/// Digest the exact aggregate resource receipt committed by a production run.
#[must_use]
pub fn spatial_reference_runtime_resource_receipt_digest(
    receipt: SpatialReferenceRuntimeResourceReceipt,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:production-runtime-resource-receipt:v1");
    for value in [
        receipt.aggregate_byte_cap,
        receipt.factor_block_high_water_bytes,
        receipt.serialization_high_water_bytes,
        receipt.fixed_l2_workspace_bytes,
        receipt.replay_reservation_high_water_bytes,
        receipt.provider_peak_count,
        receipt.provider_peak_bytes,
        receipt.aggregate_high_water_bytes,
    ] {
        digest.update(value.to_le_bytes());
    }
    format!("sha256:{:x}", digest.finalize())
}

/// Artifact-level identity for bounded reference-specific covariance blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialReferenceCovarianceMetadata {
    /// HDF5 schema version.
    pub schema_version: u16,
    /// Stable factor method.
    pub method: String,
    /// Numeric method version.
    pub method_version: u16,
    /// Producing crate version.
    pub crate_version: String,
    /// Producing Git commit, when supplied by the build.
    pub producer_commit: Option<String>,
    /// Source burst owning both target and selected reference.
    pub burst_id: String,
    /// Exact output CRS identity.
    pub crs: String,
    /// Factor output units.
    pub units: String,
    /// Exact GDAL affine geotransform for the emitted output grid.
    pub geotransform: Option<[f64; 6]>,
    /// Complete output grid covered by the artifact.
    pub full_grid: CovarianceOperatorGrid,
    /// Selected reference row in the full output grid.
    pub reference_row: u64,
    /// Selected reference column in the full output grid.
    pub reference_col: u64,
    /// Exact temporal gauge date index.
    pub gauge_date_index: u32,
    /// Fixed output date map, including the gauge date.
    pub ordered_date_indices: Vec<u32>,
    /// Exact acquisition day coordinates, relative to acquisition zero.
    pub acquisition_days: Option<Vec<f64>>,
    /// Native/output mask identity.
    pub mask_digest: String,
    /// Persisted #52 replay/source identity.
    pub source_replay_digest: String,
    /// Fixed-valid-observation L2 map identity.
    pub l2_map_digest: String,
    /// Selected-reference and grid signature.
    pub reference_signature_digest: String,
    /// Frozen approximation-validation receipt identity.
    pub approximation_receipt_digest: String,
    /// Frozen resource receipt identity.
    pub resource_receipt_digest: String,
    /// Digest of the machine-readable runtime aggregate receipt.
    pub runtime_resource_receipt_digest: String,
    /// Machine-readable aggregate production resource receipt. Legacy schema
    /// v2 and v3 artifacts explicitly lack this runtime evidence.
    pub runtime_resource_receipt: Option<SpatialReferenceRuntimeResourceReceipt>,
    /// Independent-review receipt authorizing this exact scope, when calibrated.
    pub review_receipt_digest: String,
    /// Immutable method manifest binding code, configuration, and evidence.
    pub method_manifest_digest: String,
    /// Digest of the exact calibrated metadata scope, excluding this field.
    pub calibration_scope_digest: String,
    /// Proper-complex primitive source-model identity.
    pub source_model_digest: String,
    /// Effective-look rule and parameter identity.
    pub effective_looks_digest: String,
    /// Realized fixed support method (`rect`, `glrt_frozen`, or `ks_frozen`).
    pub support_method: String,
    /// Digest of the realized native support masks.
    pub support_digest: String,
    /// Digest proving corrections were applied before spatial subtraction.
    pub correction_order_digest: String,
    /// Fixed unwrap/estimator branch identity.
    pub unwrap_branch_digest: String,
    /// Source-burst ownership and seam-leveling identity.
    pub burst_ownership_digest: String,
    /// Ordered source bursts represented by per-target ownership indices.
    pub source_burst_ids: Vec<String>,
    /// Source-burst index of the selected reference.
    pub reference_source_burst_index: u32,
    /// Calibration scope authorized by matching receipts.
    pub calibration_scope: SpatialReferenceCalibrationScope,
    /// Maximum logical numeric bytes allowed in one persisted block.
    pub maximum_block_bytes: u64,
}

impl SpatialReferenceCovarianceMetadata {
    #[allow(clippy::too_many_lines)]
    fn validate(&self) -> Result<()> {
        ensure_valid(
            matches!(
                self.schema_version,
                SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION
                    | SPATIAL_REFERENCE_COVARIANCE_PREVIOUS_SCHEMA_VERSION
                    | SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION
            ),
            "unsupported spatial reference covariance schema",
        )?;
        ensure_valid(
            self.method == SPATIAL_REFERENCE_COVARIANCE_METHOD && self.method_version == 1,
            "unsupported spatial reference covariance method",
        )?;
        ensure_valid(
            !self.crate_version.is_empty()
                && !self.burst_id.is_empty()
                && !self.crs.is_empty()
                && matches!(self.units.as_str(), "radians" | "meters" | "millimeters"),
            "spatial reference covariance identity or units are invalid",
        )?;
        ensure_valid(
            self.producer_commit
                .as_ref()
                .is_none_or(|value| !value.is_empty()),
            "spatial reference producer commit is empty",
        )?;
        self.full_grid.area()?;
        let row_stop = self
            .full_grid
            .row_start
            .checked_add(u64::from(self.full_grid.rows))
            .ok_or_else(|| invalid("spatial reference grid row extent overflow"))?;
        let col_stop = self
            .full_grid
            .col_start
            .checked_add(u64::from(self.full_grid.cols))
            .ok_or_else(|| invalid("spatial reference grid column extent overflow"))?;
        ensure_valid(
            self.reference_row >= self.full_grid.row_start
                && self.reference_row < row_stop
                && self.reference_col >= self.full_grid.col_start
                && self.reference_col < col_stop,
            "selected spatial reference is outside the full grid",
        )?;
        ensure_valid(
            self.gauge_date_index == 0
                && self.ordered_date_indices.first() == Some(&0)
                && consecutive(&self.ordered_date_indices),
            "spatial reference date map or gauge is invalid",
        )?;
        match self.schema_version {
            SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION
            | SPATIAL_REFERENCE_COVARIANCE_PREVIOUS_SCHEMA_VERSION => ensure_valid(
                self.geotransform.is_none()
                    && self.acquisition_days.is_none()
                    && self.runtime_resource_receipt.is_none(),
                "legacy spatial reference covariance cannot claim unavailable coordinates",
            )?,
            SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION => {
                let geotransform = self.geotransform.as_ref().ok_or_else(|| {
                    invalid("current spatial reference covariance requires a geotransform")
                })?;
                let acquisition_days = self.acquisition_days.as_ref().ok_or_else(|| {
                    invalid("current spatial reference covariance requires acquisition days")
                })?;
                let determinant =
                    geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4];
                ensure_valid(
                    acquisition_days.len() == self.ordered_date_indices.len()
                        && acquisition_days.first() == Some(&0.0)
                        && acquisition_days.iter().all(|day| day.is_finite())
                        && acquisition_days.windows(2).all(|pair| pair[1] > pair[0]),
                    "spatial reference acquisition days are invalid",
                )?;
                ensure_valid(
                    geotransform.iter().all(|value| value.is_finite())
                        && determinant.is_finite()
                        && determinant != 0.0,
                    "spatial reference affine geotransform is invalid",
                )?;
                self.runtime_resource_receipt
                    .ok_or_else(|| {
                        invalid("current spatial reference covariance requires a resource receipt")
                    })?
                    .validate(self.maximum_block_bytes)?;
            }
            _ => unreachable!("schema version was validated above"),
        }
        ensure_valid(
            self.maximum_block_bytes > 0,
            "spatial reference maximum block bytes must be positive",
        )?;
        for value in [
            &self.mask_digest,
            &self.source_replay_digest,
            &self.l2_map_digest,
            &self.reference_signature_digest,
            &self.approximation_receipt_digest,
            &self.resource_receipt_digest,
            &self.source_model_digest,
            &self.effective_looks_digest,
            &self.support_digest,
            &self.correction_order_digest,
            &self.unwrap_branch_digest,
            &self.burst_ownership_digest,
        ] {
            ensure_valid(
                is_sha256_digest(value),
                "spatial reference identity is not a strong SHA-256 digest",
            )?;
        }
        if self.schema_version == SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION {
            ensure_valid(
                self.runtime_resource_receipt_digest
                    == spatial_reference_runtime_resource_receipt_digest(
                        self.runtime_resource_receipt
                            .expect("current schema receipt was validated above"),
                    ),
                "spatial reference runtime resource digest does not bind its receipt",
            )?;
        } else {
            ensure_valid(
                self.runtime_resource_receipt_digest.is_empty(),
                "historical spatial reference schema cannot claim a runtime resource receipt",
            )?;
        }
        ensure_valid(
            matches!(
                self.support_method.as_str(),
                "rect" | "glrt_frozen" | "ks_frozen"
            ),
            "spatial reference support method is unsupported",
        )?;
        ensure_valid(
            !self.source_burst_ids.is_empty()
                && self
                    .source_burst_ids
                    .iter()
                    .all(|burst| !burst.is_empty() && !burst.contains('\n'))
                && self.source_burst_ids.iter().collect::<BTreeSet<_>>().len()
                    == self.source_burst_ids.len()
                && usize::try_from(self.reference_source_burst_index)
                    .is_ok_and(|index| index < self.source_burst_ids.len()),
            "spatial reference burst ownership registry is invalid",
        )?;
        match self.calibration_scope {
            SpatialReferenceCalibrationScope::Uncalibrated => ensure_valid(
                self.review_receipt_digest.is_empty()
                    && self.method_manifest_digest.is_empty()
                    && self.calibration_scope_digest.is_empty(),
                "uncalibrated spatial reference scope cannot carry promotion receipts",
            )?,
            SpatialReferenceCalibrationScope::CalibratedScopeMatch => {
                let required = match self.schema_version {
                    SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION => vec![
                        &self.review_receipt_digest,
                        &self.method_manifest_digest,
                        &self.calibration_scope_digest,
                    ],
                    _ => vec![
                        &self.review_receipt_digest,
                        &self.method_manifest_digest,
                        &self.calibration_scope_digest,
                        &self.mask_digest,
                        &self.source_replay_digest,
                        &self.l2_map_digest,
                        &self.reference_signature_digest,
                        &self.approximation_receipt_digest,
                        &self.resource_receipt_digest,
                        &self.source_model_digest,
                        &self.effective_looks_digest,
                        &self.support_digest,
                        &self.correction_order_digest,
                        &self.unwrap_branch_digest,
                        &self.burst_ownership_digest,
                    ],
                };
                for value in required {
                    ensure_valid(
                        is_nonzero_sha256_digest(value),
                        "calibrated spatial reference scope requires strong promotion receipts",
                    )?;
                }
                if self.schema_version != SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION {
                    ensure_valid(
                        self.producer_commit
                            .as_deref()
                            .is_some_and(is_exact_producer_code_identity),
                        "calibrated spatial reference scope requires an exact producer code identity",
                    )?;
                }
                ensure_valid(
                    self.calibration_scope_digest
                        == spatial_reference_calibration_scope_digest(self),
                    "calibrated spatial reference scope digest is stale or mismatched",
                )?;
            }
        }
        Ok(())
    }

    fn validate_for_write(&self) -> Result<()> {
        ensure_valid(
            self.schema_version == SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION,
            "new spatial reference covariance writes require the current schema version",
        )?;
        self.validate()
    }
}

/// Compute the immutable calibrated scope identity, excluding promotion state.
#[must_use]
pub fn spatial_reference_calibration_scope_digest(
    metadata: &SpatialReferenceCovarianceMetadata,
) -> String {
    let mut digest = Sha256::new();
    let legacy = metadata.schema_version == SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION;
    digest.update(match metadata.schema_version {
        SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION => {
            b"dolphinrust:spatial-reference-calibration-scope:v1".as_slice()
        }
        SPATIAL_REFERENCE_COVARIANCE_PREVIOUS_SCHEMA_VERSION => {
            b"dolphinrust:spatial-reference-calibration-scope:v2".as_slice()
        }
        _ => b"dolphinrust:spatial-reference-calibration-scope:v3".as_slice(),
    });
    for value in [
        metadata.method.as_str(),
        metadata.crate_version.as_str(),
        metadata.burst_id.as_str(),
        metadata.crs.as_str(),
        metadata.units.as_str(),
        metadata.mask_digest.as_str(),
        metadata.source_replay_digest.as_str(),
        metadata.l2_map_digest.as_str(),
        metadata.reference_signature_digest.as_str(),
        metadata.approximation_receipt_digest.as_str(),
        metadata.resource_receipt_digest.as_str(),
        metadata.source_model_digest.as_str(),
        metadata.effective_looks_digest.as_str(),
        metadata.support_method.as_str(),
        metadata.support_digest.as_str(),
        metadata.correction_order_digest.as_str(),
        metadata.unwrap_branch_digest.as_str(),
        metadata.burst_ownership_digest.as_str(),
    ] {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value.as_bytes());
    }
    if !legacy {
        let producer_identity = metadata.producer_commit.as_deref().unwrap_or_default();
        digest.update((producer_identity.len() as u64).to_le_bytes());
        digest.update(producer_identity.as_bytes());
        digest.update(metadata.schema_version.to_le_bytes());
    }
    if metadata.schema_version == SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION {
        digest.update((metadata.runtime_resource_receipt_digest.len() as u64).to_le_bytes());
        digest.update(metadata.runtime_resource_receipt_digest.as_bytes());
    }
    digest.update(metadata.method_version.to_le_bytes());
    digest.update(metadata.full_grid.row_start.to_le_bytes());
    digest.update(metadata.full_grid.col_start.to_le_bytes());
    digest.update(metadata.full_grid.rows.to_le_bytes());
    digest.update(metadata.full_grid.cols.to_le_bytes());
    digest.update(metadata.full_grid.stride_y.to_le_bytes());
    digest.update(metadata.full_grid.stride_x.to_le_bytes());
    digest.update(metadata.reference_row.to_le_bytes());
    digest.update(metadata.reference_col.to_le_bytes());
    digest.update(metadata.gauge_date_index.to_le_bytes());
    if let Some(geotransform) = metadata.geotransform {
        for value in geotransform {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    for date in &metadata.ordered_date_indices {
        digest.update(date.to_le_bytes());
    }
    if let Some(acquisition_days) = &metadata.acquisition_days {
        for day in acquisition_days {
            digest.update(day.to_bits().to_le_bytes());
        }
    }
    for burst in &metadata.source_burst_ids {
        digest.update((burst.len() as u64).to_le_bytes());
        digest.update(burst.as_bytes());
    }
    digest.update(metadata.reference_source_burst_index.to_le_bytes());
    digest.update(metadata.maximum_block_bytes.to_le_bytes());
    format!("sha256:{:x}", digest.finalize())
}

/// One bounded target block of a reference-specific difference factor.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialReferenceCovarianceBlock {
    /// Stable block identifier.
    pub block_id: u64,
    /// Non-overlapping target grid represented by this block.
    pub target_grid: CovarianceOperatorGrid,
    /// Padded factor rank dimension.
    pub maximum_rank: u32,
    /// Realized factor rank for each target.
    pub rank_by_target: Vec<u32>,
    /// Stable disposition for each target.
    pub status: Vec<SpatialReferenceCovarianceStatus>,
    /// Source-burst registry index for every target. Non-valid targets may use
    /// [`SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE`].
    pub source_burst_index_by_target: Vec<u32>,
    /// Target-major, date-major, rank-minor difference factor.
    pub difference_factor: Vec<f64>,
    /// Validated approximation-error bound for each target. Uncalibrated and
    /// non-valid targets use [`SPATIAL_REFERENCE_APPROXIMATION_ERROR_UNAVAILABLE`].
    pub approximation_error_bound: Vec<f64>,
    /// Exact effective-look fraction applied for each target.
    pub effective_looks_fraction: Option<Vec<f64>>,
    /// Exact target/reference realized support-union cardinality.
    pub support_union_count: Option<Vec<u64>>,
    /// Target-major 32-byte effective-look realization receipts.
    pub effective_looks_receipt: Option<Vec<u8>>,
    /// Conservative replay resource high-water bound for each target.
    pub resource_high_water_bytes: Option<Vec<u64>>,
    /// Digest of the exact replayed source factors represented by this block.
    pub source_factor_digest: String,
}

impl SpatialReferenceCovarianceBlock {
    fn logical_payload_bytes(&self) -> Result<u64> {
        let factor = u64::try_from(self.difference_factor.len())
            .ok()
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| invalid("spatial reference factor byte count overflow"))?;
        let rank = u64::try_from(self.rank_by_target.len())
            .ok()
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| invalid("spatial reference rank byte count overflow"))?;
        let status = u64::try_from(self.status.len())
            .ok()
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| invalid("spatial reference status byte count overflow"))?;
        let bounds = u64::try_from(self.approximation_error_bound.len())
            .ok()
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| invalid("spatial reference bound byte count overflow"))?;
        let ownership = u64::try_from(self.source_burst_index_by_target.len())
            .ok()
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| invalid("spatial reference ownership byte count overflow"))?;
        let effective = u64::try_from(self.effective_looks_fraction.as_ref().map_or(0, Vec::len))
            .ok()
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| invalid("effective-look fraction byte count overflow"))?;
        let support = u64::try_from(self.support_union_count.as_ref().map_or(0, Vec::len))
            .ok()
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| invalid("support-union count byte count overflow"))?;
        let receipts = u64::try_from(self.effective_looks_receipt.as_ref().map_or(0, Vec::len))
            .map_err(|_| invalid("effective-look receipt byte count overflow"))?;
        let resource = u64::try_from(self.resource_high_water_bytes.as_ref().map_or(0, Vec::len))
            .ok()
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| invalid("resource high-water byte count overflow"))?;
        factor
            .checked_add(rank)
            .and_then(|value| value.checked_add(status))
            .and_then(|value| value.checked_add(ownership))
            .and_then(|value| value.checked_add(bounds))
            .and_then(|value| value.checked_add(effective))
            .and_then(|value| value.checked_add(support))
            .and_then(|value| value.checked_add(receipts))
            .and_then(|value| value.checked_add(resource))
            .ok_or_else(|| invalid("spatial reference block byte count overflow"))
    }

    #[allow(clippy::too_many_lines)]
    fn validate(&self, metadata: &SpatialReferenceCovarianceMetadata) -> Result<()> {
        let targets = self.target_grid.area()?;
        ensure_valid(
            metadata.full_grid.contains(self.target_grid),
            "spatial reference target block is outside the full grid",
        )?;
        let dates = metadata.ordered_date_indices.len();
        let rank = usize::try_from(self.maximum_rank)
            .map_err(|_| invalid("spatial reference maximum rank exceeds usize"))?;
        ensure_valid(
            rank > 0 && rank <= dates,
            "spatial reference maximum rank is invalid",
        )?;
        ensure_valid(
            self.rank_by_target.len() == targets
                && self.status.len() == targets
                && self.source_burst_index_by_target.len() == targets
                && self.approximation_error_bound.len() == targets
                && self.difference_factor.len()
                    == targets
                        .checked_mul(dates)
                        .and_then(|value| value.checked_mul(rank))
                        .ok_or_else(|| invalid("spatial reference factor dimensions overflow"))?,
            "spatial reference block shapes do not match",
        )?;
        let realization = match metadata.schema_version {
            SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION
            | SPATIAL_REFERENCE_COVARIANCE_PREVIOUS_SCHEMA_VERSION => {
                ensure_valid(
                    self.effective_looks_fraction.is_none()
                        && self.support_union_count.is_none()
                        && self.effective_looks_receipt.is_none()
                        && self.resource_high_water_bytes.is_none(),
                    "legacy spatial reference block cannot claim unavailable realization receipts",
                )?;
                None
            }
            SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION => {
                let effective = self.effective_looks_fraction.as_ref().ok_or_else(|| {
                    invalid("current spatial reference block requires effective-look fractions")
                })?;
                let support = self.support_union_count.as_ref().ok_or_else(|| {
                    invalid("current spatial reference block requires support-union counts")
                })?;
                let receipts = self.effective_looks_receipt.as_ref().ok_or_else(|| {
                    invalid("current spatial reference block requires effective-look receipts")
                })?;
                let resource = self.resource_high_water_bytes.as_ref().ok_or_else(|| {
                    invalid("current spatial reference block requires resource high-water bounds")
                })?;
                ensure_valid(
                    effective.len() == targets
                        && support.len() == targets
                        && receipts.len() == targets * 32
                        && resource.len() == targets,
                    "spatial reference realization receipt shapes do not match",
                )?;
                Some((effective, support, receipts, resource))
            }
            _ => unreachable!("schema version was validated above"),
        };
        ensure_valid(
            is_sha256_digest(&self.source_factor_digest),
            "spatial reference source factor digest is not strong",
        )?;
        ensure_valid(
            self.difference_factor.iter().all(|value| value.is_finite()),
            "spatial reference factor contains non-finite values",
        )?;
        if metadata.schema_version == SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION {
            ensure_valid(
                self.approximation_error_bound
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0),
                "legacy spatial reference approximation bounds are invalid",
            )?;
        }
        for target in 0..targets {
            let realized = usize::try_from(self.rank_by_target[target])
                .map_err(|_| invalid("spatial reference rank exceeds usize"))?;
            ensure_valid(
                realized <= rank,
                "spatial reference target rank exceeds maximum",
            )?;
            let legacy =
                metadata.schema_version == SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION;
            ensure_valid(
                if legacy {
                    (self.status[target] == SpatialReferenceCovarianceStatus::Valid)
                        == (realized > 0)
                } else {
                    self.status[target] == SpatialReferenceCovarianceStatus::Valid || realized == 0
                },
                "spatial reference status and target rank disagree",
            )?;
            if let Some((effective, support, receipts, resource)) = realization {
                let receipt = &receipts[target * 32..(target + 1) * 32];
                ensure_valid(
                    match self.status[target] == SpatialReferenceCovarianceStatus::Valid {
                        true => {
                            effective[target].is_finite()
                                && effective[target] > 0.0
                                && effective[target] <= 1.0
                                && support[target] > 0
                                && receipt.iter().any(|byte| *byte != 0)
                                && resource[target] > 0
                        }
                        false => {
                            effective[target].is_nan()
                                && support[target] == 0
                                && receipt.iter().all(|byte| *byte == 0)
                        }
                    },
                    "spatial reference effective-look/resource receipt disagrees with target status",
                )?;
            }
            if !legacy {
                let approximation_bound = self.approximation_error_bound[target];
                let validated_bound_required = self.status[target]
                    == SpatialReferenceCovarianceStatus::Valid
                    && metadata.calibration_scope
                        == SpatialReferenceCalibrationScope::CalibratedScopeMatch;
                ensure_valid(
                    match validated_bound_required {
                        true => approximation_bound.is_finite() && approximation_bound >= 0.0,
                        false => approximation_bound.is_nan(),
                    },
                    "spatial reference approximation bound disagrees with status or calibration scope",
                )?;
            }
            let source_burst_index = self.source_burst_index_by_target[target];
            let source_burst_unavailable =
                source_burst_index == SPATIAL_REFERENCE_SOURCE_BURST_UNAVAILABLE;
            let source_burst = usize::try_from(source_burst_index).ok();
            ensure_valid(
                (!legacy && source_burst_unavailable)
                    || source_burst.is_some_and(|index| index < metadata.source_burst_ids.len()),
                "spatial reference source-burst index is outside the registry",
            )?;
            ensure_valid(
                (legacy && realized == 0)
                    || self.status[target] != SpatialReferenceCovarianceStatus::Valid
                    || source_burst_index == metadata.reference_source_burst_index,
                "valid spatial reference factor crosses source-burst ownership",
            )?;
            ensure_valid(
                legacy
                    || self.status[target]
                        != SpatialReferenceCovarianceStatus::UnsupportedMultiburstReference
                    || source_burst_unavailable,
                "unsupported multiburst target cannot claim one source-burst owner",
            )?;
            for date in 0..dates {
                for component in 0..rank {
                    let index = (target * dates + date) * rank + component;
                    ensure_valid(
                        (date != 0 && component < realized) || self.difference_factor[index] == 0.0,
                        "spatial reference factor violates gauge or rank padding",
                    )?;
                }
            }
        }
        ensure_valid(
            self.logical_payload_bytes()? <= metadata.maximum_block_bytes,
            "spatial reference block exceeds maximum block bytes",
        )
    }
}

/// Sealed HDF5 receipt returned after a bounded factor artifact is closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpatialReferenceCovarianceWriteReceipt {
    /// Number of committed factor blocks.
    pub block_count: usize,
    /// Final HDF5 byte count.
    pub hdf5_bytes: u64,
    /// Lowercase SHA-256 digest of the final HDF5 bytes.
    pub hdf5_sha256: String,
}

/// One checked bounded factor block plus its logical numeric byte count.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialReferenceCovarianceBlockRead {
    /// Validated persisted factor block.
    pub block: SpatialReferenceCovarianceBlock,
    /// Numeric payload bytes charged before allocation.
    pub logical_payload_bytes: u64,
}

/// Streaming writer for one bounded reference-specific factor artifact.
pub struct SpatialReferenceCovarianceWriter {
    file: Option<hdf5::File>,
    path: PathBuf,
    metadata: SpatialReferenceCovarianceMetadata,
    block_ids: BTreeSet<u64>,
    target_grids: Vec<CovarianceOperatorGrid>,
}

impl SpatialReferenceCovarianceWriter {
    /// Create an incomplete artifact and persist its immutable metadata.
    ///
    /// # Errors
    /// Returns an error for invalid metadata or HDF5/I/O failure.
    pub fn create(
        path: impl AsRef<Path>,
        metadata: &SpatialReferenceCovarianceMetadata,
    ) -> Result<Self> {
        metadata.validate_for_write()?;
        let path = path.as_ref().to_owned();
        let file = hdf5::File::create(&path)?;
        write_spatial_reference_metadata(&file, metadata)?;
        Ok(Self {
            file: Some(file),
            path,
            metadata: metadata.clone(),
            block_ids: BTreeSet::new(),
            target_grids: Vec::new(),
        })
    }

    /// Append one validated non-overlapping target block.
    ///
    /// # Errors
    /// Returns an error for invalid/duplicate blocks, byte-cap violations, or
    /// HDF5/I/O failure.
    pub fn write_block(&mut self, block: &SpatialReferenceCovarianceBlock) -> Result<()> {
        block.validate(&self.metadata)?;
        ensure_valid(
            self.block_ids.insert(block.block_id),
            "duplicate spatial reference block ID",
        )?;
        ensure_valid(
            self.target_grids
                .iter()
                .all(|grid| !grids_overlap(*grid, block.target_grid)),
            "spatial reference target blocks overlap",
        )?;
        let file = self
            .file
            .as_ref()
            .ok_or_else(|| invalid("spatial reference writer is already finished"))?;
        let block_group = file.group("blocks")?;
        let group = block_group.create_group(&format!("{:020}", block.block_id))?;
        write_spatial_reference_block(&group, &self.metadata, block)?;
        self.target_grids.push(block.target_grid);
        Ok(())
    }

    /// Replace the preflight resource receipt with the exact observed runtime
    /// receipt before sealing the artifact.
    ///
    /// # Errors
    /// Returns an error for an inconsistent receipt or HDF5 update failure.
    pub fn seal_runtime_resource_receipt(
        &mut self,
        receipt: SpatialReferenceRuntimeResourceReceipt,
    ) -> Result<SpatialReferenceCovarianceMetadata> {
        receipt.validate(self.metadata.maximum_block_bytes)?;
        ensure_valid(
            self.metadata.calibration_scope == SpatialReferenceCalibrationScope::Uncalibrated,
            "calibrated spatial reference resource evidence is immutable",
        )?;
        let digest = spatial_reference_runtime_resource_receipt_digest(receipt);
        let file = self
            .file
            .as_ref()
            .ok_or_else(|| invalid("spatial reference writer is already finished"))?;
        let identity = file.group("metadata")?;
        write_runtime_resource_receipt_datasets(&identity, receipt, true)?;
        identity
            .dataset("runtime_resource_receipt_digest")?
            .write_raw(digest.as_bytes())?;
        self.metadata.runtime_resource_receipt = Some(receipt);
        self.metadata.runtime_resource_receipt_digest = digest;
        self.metadata.validate()?;
        Ok(self.metadata.clone())
    }

    /// Mark the artifact complete, validate its exact schema, and return its digest.
    ///
    /// # Errors
    /// Returns an error for an empty artifact or HDF5/I/O failure.
    pub fn finish(mut self) -> Result<SpatialReferenceCovarianceWriteReceipt> {
        ensure_valid(
            !self.block_ids.is_empty(),
            "spatial reference artifact has no blocks",
        )?;
        let covered_targets = self.target_grids.iter().try_fold(0_usize, |total, grid| {
            total
                .checked_add(grid.area()?)
                .ok_or_else(|| invalid("spatial reference covered area overflow"))
        })?;
        ensure_valid(
            covered_targets == self.metadata.full_grid.area()?,
            "spatial reference target blocks do not exactly cover the full grid",
        )?;
        let file = self
            .file
            .take()
            .ok_or_else(|| invalid("spatial reference writer is already finished"))?;
        file.attr("complete")?.write_scalar(&1_u8)?;
        validate_spatial_root_schema(&file)?;
        file.flush()?;
        file.close()?;
        let (hdf5_sha256, hdf5_bytes) = sha256_path(&self.path)?;
        Ok(SpatialReferenceCovarianceWriteReceipt {
            block_count: self.block_ids.len(),
            hdf5_bytes,
            hdf5_sha256,
        })
    }
}

#[allow(clippy::too_many_lines)]
fn write_spatial_reference_metadata(
    file: &hdf5::File,
    metadata: &SpatialReferenceCovarianceMetadata,
) -> Result<()> {
    write_scalar_attr(file, "schema_version", metadata.schema_version)?;
    write_scalar_attr(file, "method_version", metadata.method_version)?;
    write_scalar_attr(file, "gauge_date_index", metadata.gauge_date_index)?;
    write_scalar_attr(file, "calibration_scope", metadata.calibration_scope.code())?;
    write_scalar_attr(file, "maximum_block_bytes", metadata.maximum_block_bytes)?;
    write_scalar_attr(file, "complete", 0_u8)?;
    let identity = file.create_group("metadata")?;
    let source_burst_ids = metadata.source_burst_ids.join("\n");
    for (name, value) in [
        ("method", metadata.method.as_str()),
        ("crate_version", metadata.crate_version.as_str()),
        (
            "producer_commit",
            metadata.producer_commit.as_deref().unwrap_or_default(),
        ),
        ("burst_id", metadata.burst_id.as_str()),
        ("crs", metadata.crs.as_str()),
        ("units", metadata.units.as_str()),
        ("mask_digest", metadata.mask_digest.as_str()),
        (
            "source_replay_digest",
            metadata.source_replay_digest.as_str(),
        ),
        ("l2_map_digest", metadata.l2_map_digest.as_str()),
        (
            "reference_signature_digest",
            metadata.reference_signature_digest.as_str(),
        ),
        (
            "approximation_receipt_digest",
            metadata.approximation_receipt_digest.as_str(),
        ),
        (
            "resource_receipt_digest",
            metadata.resource_receipt_digest.as_str(),
        ),
        (
            "runtime_resource_receipt_digest",
            metadata.runtime_resource_receipt_digest.as_str(),
        ),
        (
            "review_receipt_digest",
            metadata.review_receipt_digest.as_str(),
        ),
        (
            "method_manifest_digest",
            metadata.method_manifest_digest.as_str(),
        ),
        (
            "calibration_scope_digest",
            metadata.calibration_scope_digest.as_str(),
        ),
        ("source_model_digest", metadata.source_model_digest.as_str()),
        (
            "effective_looks_digest",
            metadata.effective_looks_digest.as_str(),
        ),
        ("support_method", metadata.support_method.as_str()),
        ("support_digest", metadata.support_digest.as_str()),
        (
            "correction_order_digest",
            metadata.correction_order_digest.as_str(),
        ),
        (
            "unwrap_branch_digest",
            metadata.unwrap_branch_digest.as_str(),
        ),
        (
            "burst_ownership_digest",
            metadata.burst_ownership_digest.as_str(),
        ),
        ("source_burst_ids", source_burst_ids.as_str()),
    ] {
        write_string(&identity, name, value)?;
    }
    write_chunked_1d(
        &identity,
        "ordered_date_indices",
        &metadata.ordered_date_indices,
    )?;
    write_chunked_1d(
        &identity,
        "acquisition_days",
        metadata
            .acquisition_days
            .as_ref()
            .ok_or_else(|| invalid("current schema acquisition days are missing"))?,
    )?;
    write_chunked_1d(
        &identity,
        "geotransform",
        metadata
            .geotransform
            .as_ref()
            .ok_or_else(|| invalid("current schema geotransform is missing"))?,
    )?;
    write_scalar_attr(&identity, "reference_row", metadata.reference_row)?;
    write_scalar_attr(&identity, "reference_col", metadata.reference_col)?;
    write_scalar_attr(
        &identity,
        "reference_source_burst_index",
        metadata.reference_source_burst_index,
    )?;
    write_runtime_resource_receipt_datasets(
        &identity,
        metadata
            .runtime_resource_receipt
            .ok_or_else(|| invalid("current schema runtime resource receipt is missing"))?,
        false,
    )?;
    write_grid(file, "full_grid", metadata.full_grid)?;
    file.create_group("blocks")?;
    drop(identity);
    Ok(())
}

fn write_runtime_resource_receipt_datasets(
    group: &Group,
    receipt: SpatialReferenceRuntimeResourceReceipt,
    overwrite: bool,
) -> Result<()> {
    macro_rules! write_receipt_value {
        ($name:literal, $value:expr) => {
            if overwrite {
                group.dataset($name)?.write_raw(&[$value])?;
            } else {
                write_chunked_1d(group, $name, &[$value])?;
            }
        };
    }
    write_receipt_value!("aggregate_byte_cap", receipt.aggregate_byte_cap);
    write_receipt_value!(
        "factor_block_high_water_bytes",
        receipt.factor_block_high_water_bytes
    );
    write_receipt_value!(
        "serialization_high_water_bytes",
        receipt.serialization_high_water_bytes
    );
    write_receipt_value!("fixed_l2_workspace_bytes", receipt.fixed_l2_workspace_bytes);
    write_receipt_value!(
        "replay_reservation_high_water_bytes",
        receipt.replay_reservation_high_water_bytes
    );
    write_receipt_value!("provider_peak_count", receipt.provider_peak_count);
    write_receipt_value!("provider_peak_bytes", receipt.provider_peak_bytes);
    write_receipt_value!(
        "aggregate_high_water_bytes",
        receipt.aggregate_high_water_bytes
    );
    Ok(())
}

fn write_spatial_reference_block(
    group: &Group,
    metadata: &SpatialReferenceCovarianceMetadata,
    block: &SpatialReferenceCovarianceBlock,
) -> Result<()> {
    write_scalar_attr(group, "block_id", block.block_id)?;
    write_scalar_attr(group, "maximum_rank", block.maximum_rank)?;
    write_grid(group, "target_grid", block.target_grid)?;
    write_chunked_1d(group, "rank_by_target", &block.rank_by_target)?;
    write_chunked_1d(
        group,
        "status",
        &block
            .status
            .iter()
            .map(|status| status.code())
            .collect::<Vec<_>>(),
    )?;
    write_chunked_1d(
        group,
        "source_burst_index_by_target",
        &block.source_burst_index_by_target,
    )?;
    let target_count = block.target_grid.area()?;
    let date_count = metadata.ordered_date_indices.len();
    let rank = block.maximum_rank as usize;
    let view = ArrayView3::from_shape((target_count, date_count, rank), &block.difference_factor)
        .map_err(|error| invalid(format!("spatial reference factor shape: {error}")))?;
    group
        .new_dataset_builder()
        .with_data(view)
        .chunk((1, date_count.min(32), rank.min(32)))
        .create("difference_factor")?;
    write_chunked_1d(
        group,
        "approximation_error_bound",
        &block.approximation_error_bound,
    )?;
    write_chunked_1d(
        group,
        "effective_looks_fraction",
        block
            .effective_looks_fraction
            .as_ref()
            .ok_or_else(|| invalid("current block effective-look fractions are missing"))?,
    )?;
    write_chunked_1d(
        group,
        "support_union_count",
        block
            .support_union_count
            .as_ref()
            .ok_or_else(|| invalid("current block support-union counts are missing"))?,
    )?;
    let effective_looks_receipt = block
        .effective_looks_receipt
        .as_ref()
        .ok_or_else(|| invalid("current block effective-look receipts are missing"))?;
    let receipt_view = ndarray::ArrayView2::from_shape((target_count, 32), effective_looks_receipt)
        .map_err(|error| invalid(format!("effective-look receipt shape: {error}")))?;
    group
        .new_dataset_builder()
        .with_data(receipt_view)
        .chunk((target_count.min(64), 32))
        .create("effective_looks_receipt")?;
    write_chunked_1d(
        group,
        "resource_high_water_bytes",
        block
            .resource_high_water_bytes
            .as_ref()
            .ok_or_else(|| invalid("current block resource high-water bounds are missing"))?,
    )?;
    write_string(group, "source_factor_digest", &block.source_factor_digest)
}

/// Write and seal a bounded reference-specific HDF5 factor artifact.
///
/// # Errors
/// Returns an error for invalid metadata, invalid/duplicate blocks, byte-cap
/// violations, or HDF5/I/O failures.
pub fn write_spatial_reference_covariance(
    path: impl AsRef<Path>,
    metadata: &SpatialReferenceCovarianceMetadata,
    blocks: &[SpatialReferenceCovarianceBlock],
) -> Result<SpatialReferenceCovarianceWriteReceipt> {
    let mut writer = SpatialReferenceCovarianceWriter::create(path, metadata)?;
    for block in blocks {
        writer.write_block(block)?;
    }
    writer.finish()
}

/// Read and validate only bounded-factor metadata under an allocation cap.
///
/// # Errors
/// Returns an error for malformed/incomplete schema, invalid identities, or a
/// metadata allocation above `byte_cap`.
pub fn read_spatial_reference_covariance_header(
    path: impl AsRef<Path>,
    byte_cap: u64,
) -> Result<SpatialReferenceCovarianceMetadata> {
    let file = hdf5::File::open(path)?;
    validate_spatial_root_schema(&file)?;
    let schema_version: u16 = read_scalar_attr(&file, "schema_version")?;
    let mut budget = ReadBudget::new(byte_cap);
    let identity = file.group("metadata")?;
    let metadata_members = match schema_version {
        SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION => SPATIAL_METADATA_MEMBERS_V2,
        SPATIAL_REFERENCE_COVARIANCE_PREVIOUS_SCHEMA_VERSION => SPATIAL_METADATA_MEMBERS_V3,
        SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION => SPATIAL_METADATA_MEMBERS_V4,
        _ => return Err(invalid("unsupported spatial reference covariance schema")),
    };
    for name in metadata_members.iter().copied().filter(|name| {
        !matches!(
            *name,
            "ordered_date_indices" | "acquisition_days" | "geotransform"
        ) && !SPATIAL_RUNTIME_RESOURCE_MEMBERS.contains(name)
    }) {
        let (_, bytes) = inspect_dataset::<u8>(&identity, name, None)?;
        budget.charge(bytes)?;
    }
    let (_, date_bytes) = inspect_dataset::<u32>(&identity, "ordered_date_indices", None)?;
    budget.charge(date_bytes)?;
    if schema_version == SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION {
        let (_, acquisition_bytes) = inspect_dataset::<f64>(&identity, "acquisition_days", None)?;
        budget.charge(acquisition_bytes)?;
        let (_, geotransform_bytes) =
            inspect_dataset::<f64>(&identity, "geotransform", Some(&[6]))?;
        budget.charge(geotransform_bytes)?;
        for name in SPATIAL_RUNTIME_RESOURCE_MEMBERS {
            let (_, bytes) = inspect_dataset::<u64>(&identity, name, Some(&[1]))?;
            budget.charge(bytes)?;
        }
    }
    read_spatial_metadata(&file)
}

/// Read one bounded factor block after shape checks and byte-cap admission.
///
/// # Errors
/// Returns an error for a missing/malformed block or numeric allocation above
/// `byte_cap`.
#[allow(clippy::too_many_lines)]
pub fn read_spatial_reference_covariance_block(
    path: impl AsRef<Path>,
    block_id: u64,
    byte_cap: u64,
) -> Result<SpatialReferenceCovarianceBlockRead> {
    let file = hdf5::File::open(path)?;
    validate_spatial_root_schema(&file)?;
    let metadata = read_spatial_metadata(&file)?;
    let blocks = file.group("blocks")?;
    let name = format!("{block_id:020}");
    validate_selected_block_link(&blocks, &name)?;
    let group = blocks.group(&name)?;
    let block_members = match metadata.schema_version {
        SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION => SPATIAL_BLOCK_MEMBERS_V2,
        SPATIAL_REFERENCE_COVARIANCE_PREVIOUS_SCHEMA_VERSION => SPATIAL_BLOCK_MEMBERS_V3,
        SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION => SPATIAL_BLOCK_MEMBERS_V4,
        _ => return Err(invalid("unsupported spatial reference covariance schema")),
    };
    validate_exact_schema(
        &group,
        Some(block_members),
        SPATIAL_BLOCK_ATTRIBUTES,
        "spatial reference block schema contains missing or unexpected members",
    )?;
    let target_grid = read_grid(&group, "target_grid")?;
    let targets = target_grid.area()?;
    let maximum_rank: u32 = read_scalar_attr(&group, "maximum_rank")?;
    let rank = usize::try_from(maximum_rank)
        .map_err(|_| invalid("spatial reference maximum rank exceeds usize"))?;
    let dates = metadata.ordered_date_indices.len();
    let mut logical_payload_bytes = 0_u64;
    add_exact_dataset::<u32>(
        &group,
        "rank_by_target",
        &[targets],
        &mut logical_payload_bytes,
    )?;
    add_exact_dataset::<u16>(&group, "status", &[targets], &mut logical_payload_bytes)?;
    add_exact_dataset::<u32>(
        &group,
        "source_burst_index_by_target",
        &[targets],
        &mut logical_payload_bytes,
    )?;
    add_exact_dataset::<f64>(
        &group,
        "difference_factor",
        &[targets, dates, rank],
        &mut logical_payload_bytes,
    )?;
    add_exact_dataset::<f64>(
        &group,
        "approximation_error_bound",
        &[targets],
        &mut logical_payload_bytes,
    )?;
    if metadata.schema_version == SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION {
        add_exact_dataset::<f64>(
            &group,
            "effective_looks_fraction",
            &[targets],
            &mut logical_payload_bytes,
        )?;
        add_exact_dataset::<u64>(
            &group,
            "support_union_count",
            &[targets],
            &mut logical_payload_bytes,
        )?;
        add_exact_dataset::<u8>(
            &group,
            "effective_looks_receipt",
            &[targets, 32],
            &mut logical_payload_bytes,
        )?;
        add_exact_dataset::<u64>(
            &group,
            "resource_high_water_bytes",
            &[targets],
            &mut logical_payload_bytes,
        )?;
    }
    ensure_valid(
        logical_payload_bytes <= metadata.maximum_block_bytes,
        "spatial reference block exceeds embedded byte cap",
    )?;
    let mut budget = ReadBudget::new(byte_cap);
    budget.charge(logical_payload_bytes)?;
    let status = group
        .dataset("status")?
        .read_raw::<u16>()?
        .into_iter()
        .map(|code| SpatialReferenceCovarianceStatus::from_code(code, metadata.schema_version))
        .collect::<Result<Vec<_>>>()?;
    let block = SpatialReferenceCovarianceBlock {
        block_id: read_scalar_attr(&group, "block_id")?,
        target_grid,
        maximum_rank,
        rank_by_target: group.dataset("rank_by_target")?.read_raw()?,
        status,
        source_burst_index_by_target: group.dataset("source_burst_index_by_target")?.read_raw()?,
        difference_factor: group.dataset("difference_factor")?.read_raw()?,
        approximation_error_bound: group.dataset("approximation_error_bound")?.read_raw()?,
        effective_looks_fraction: read_current_block_dataset(
            &group,
            metadata.schema_version,
            "effective_looks_fraction",
        )?,
        support_union_count: read_current_block_dataset(
            &group,
            metadata.schema_version,
            "support_union_count",
        )?,
        effective_looks_receipt: read_current_block_dataset(
            &group,
            metadata.schema_version,
            "effective_looks_receipt",
        )?,
        resource_high_water_bytes: read_current_block_dataset(
            &group,
            metadata.schema_version,
            "resource_high_water_bytes",
        )?,
        source_factor_digest: read_string(&group, "source_factor_digest")?,
    };
    ensure_valid(
        block.block_id == block_id,
        "spatial reference block ID mismatch",
    )?;
    block.validate(&metadata)?;
    Ok(SpatialReferenceCovarianceBlockRead {
        block,
        logical_payload_bytes,
    })
}

fn read_current_block_dataset<T: hdf5::H5Type>(
    group: &hdf5::Group,
    schema_version: u16,
    name: &str,
) -> Result<Option<Vec<T>>> {
    match schema_version {
        SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION
        | SPATIAL_REFERENCE_COVARIANCE_PREVIOUS_SCHEMA_VERSION => Ok(None),
        SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION => Ok(Some(group.dataset(name)?.read_raw()?)),
        _ => Err(invalid("unsupported spatial reference covariance schema")),
    }
}

fn validate_spatial_root_schema(file: &hdf5::File) -> Result<()> {
    validate_exact_schema(
        file,
        Some(SPATIAL_ROOT_MEMBERS),
        SPATIAL_ROOT_ATTRIBUTES,
        "spatial reference root schema contains missing or unexpected members",
    )?;
    ensure_valid(
        read_scalar_attr::<u8>(file, "complete")? == 1,
        "spatial reference artifact is incomplete",
    )?;
    let schema_version: u16 = read_scalar_attr(file, "schema_version")?;
    let metadata_members = match schema_version {
        SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION => SPATIAL_METADATA_MEMBERS_V2,
        SPATIAL_REFERENCE_COVARIANCE_PREVIOUS_SCHEMA_VERSION => SPATIAL_METADATA_MEMBERS_V3,
        SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION => SPATIAL_METADATA_MEMBERS_V4,
        _ => return Err(invalid("unsupported spatial reference covariance schema")),
    };
    let identity = file.group("metadata")?;
    validate_exact_schema(
        &identity,
        Some(metadata_members),
        &[
            "reference_row",
            "reference_col",
            "reference_source_burst_index",
        ],
        "spatial reference metadata schema contains missing or unexpected members",
    )?;
    let blocks = file.group("blocks")?;
    ensure_valid(
        !blocks.is_empty(),
        "spatial reference artifact has no blocks",
    )
}

#[allow(clippy::too_many_lines)]
fn read_spatial_metadata(file: &hdf5::File) -> Result<SpatialReferenceCovarianceMetadata> {
    let identity = file.group("metadata")?;
    let schema_version: u16 = read_scalar_attr(file, "schema_version")?;
    let metadata = SpatialReferenceCovarianceMetadata {
        schema_version,
        method: read_string(&identity, "method")?,
        method_version: read_scalar_attr(file, "method_version")?,
        crate_version: read_string(&identity, "crate_version")?,
        producer_commit: read_optional_string(&identity, "producer_commit")?,
        burst_id: read_string(&identity, "burst_id")?,
        crs: read_string(&identity, "crs")?,
        units: read_string(&identity, "units")?,
        geotransform: match schema_version {
            SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION
            | SPATIAL_REFERENCE_COVARIANCE_PREVIOUS_SCHEMA_VERSION => None,
            SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION => Some(
                identity
                    .dataset("geotransform")?
                    .read_raw::<f64>()?
                    .try_into()
                    .map_err(|_| {
                        invalid("spatial reference geotransform must contain six values")
                    })?,
            ),
            _ => return Err(invalid("unsupported spatial reference covariance schema")),
        },
        full_grid: read_grid(file, "full_grid")?,
        reference_row: read_scalar_attr(&identity, "reference_row")?,
        reference_col: read_scalar_attr(&identity, "reference_col")?,
        gauge_date_index: read_scalar_attr(file, "gauge_date_index")?,
        ordered_date_indices: identity.dataset("ordered_date_indices")?.read_raw()?,
        acquisition_days: match schema_version {
            SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION
            | SPATIAL_REFERENCE_COVARIANCE_PREVIOUS_SCHEMA_VERSION => None,
            SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION => {
                Some(identity.dataset("acquisition_days")?.read_raw()?)
            }
            _ => return Err(invalid("unsupported spatial reference covariance schema")),
        },
        mask_digest: read_string(&identity, "mask_digest")?,
        source_replay_digest: read_string(&identity, "source_replay_digest")?,
        l2_map_digest: read_string(&identity, "l2_map_digest")?,
        reference_signature_digest: read_string(&identity, "reference_signature_digest")?,
        approximation_receipt_digest: read_string(&identity, "approximation_receipt_digest")?,
        resource_receipt_digest: read_string(&identity, "resource_receipt_digest")?,
        runtime_resource_receipt_digest: match schema_version {
            SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION
            | SPATIAL_REFERENCE_COVARIANCE_PREVIOUS_SCHEMA_VERSION => String::new(),
            SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION => {
                read_string(&identity, "runtime_resource_receipt_digest")?
            }
            _ => return Err(invalid("unsupported spatial reference covariance schema")),
        },
        runtime_resource_receipt: match schema_version {
            SPATIAL_REFERENCE_COVARIANCE_LEGACY_SCHEMA_VERSION
            | SPATIAL_REFERENCE_COVARIANCE_PREVIOUS_SCHEMA_VERSION => None,
            SPATIAL_REFERENCE_COVARIANCE_SCHEMA_VERSION => {
                Some(SpatialReferenceRuntimeResourceReceipt {
                    aggregate_byte_cap: read_runtime_resource_value(
                        &identity,
                        "aggregate_byte_cap",
                    )?,
                    factor_block_high_water_bytes: read_runtime_resource_value(
                        &identity,
                        "factor_block_high_water_bytes",
                    )?,
                    serialization_high_water_bytes: read_runtime_resource_value(
                        &identity,
                        "serialization_high_water_bytes",
                    )?,
                    fixed_l2_workspace_bytes: read_runtime_resource_value(
                        &identity,
                        "fixed_l2_workspace_bytes",
                    )?,
                    replay_reservation_high_water_bytes: read_runtime_resource_value(
                        &identity,
                        "replay_reservation_high_water_bytes",
                    )?,
                    provider_peak_count: read_runtime_resource_value(
                        &identity,
                        "provider_peak_count",
                    )?,
                    provider_peak_bytes: read_runtime_resource_value(
                        &identity,
                        "provider_peak_bytes",
                    )?,
                    aggregate_high_water_bytes: read_runtime_resource_value(
                        &identity,
                        "aggregate_high_water_bytes",
                    )?,
                })
            }
            _ => return Err(invalid("unsupported spatial reference covariance schema")),
        },
        review_receipt_digest: read_string(&identity, "review_receipt_digest")?,
        method_manifest_digest: read_string(&identity, "method_manifest_digest")?,
        calibration_scope_digest: read_string(&identity, "calibration_scope_digest")?,
        source_model_digest: read_string(&identity, "source_model_digest")?,
        effective_looks_digest: read_string(&identity, "effective_looks_digest")?,
        support_method: read_string(&identity, "support_method")?,
        support_digest: read_string(&identity, "support_digest")?,
        correction_order_digest: read_string(&identity, "correction_order_digest")?,
        unwrap_branch_digest: read_string(&identity, "unwrap_branch_digest")?,
        burst_ownership_digest: read_string(&identity, "burst_ownership_digest")?,
        source_burst_ids: read_string(&identity, "source_burst_ids")?
            .split('\n')
            .map(str::to_owned)
            .collect(),
        reference_source_burst_index: read_scalar_attr(&identity, "reference_source_burst_index")?,
        calibration_scope: SpatialReferenceCalibrationScope::from_code(read_scalar_attr(
            file,
            "calibration_scope",
        )?)?,
        maximum_block_bytes: read_scalar_attr(file, "maximum_block_bytes")?,
    };
    metadata.validate()?;
    Ok(metadata)
}

fn read_runtime_resource_value(group: &Group, name: &str) -> Result<u64> {
    let (shape, _) = inspect_dataset::<u64>(group, name, Some(&[1]))?;
    ensure_valid(
        shape == [1],
        "spatial reference runtime resource field is not scalar",
    )?;
    group
        .dataset(name)?
        .read_raw::<u64>()?
        .into_iter()
        .next()
        .ok_or_else(|| invalid("spatial reference runtime resource field is empty"))
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

#[allow(clippy::too_many_lines)]
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
            && consecutive(&block.source_date_indices),
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
            "source_content_digests",
            block.source_content_digests.len(),
            native_area
                .checked_mul(32)
                .ok_or_else(|| invalid("source digest dimensions overflow usize"))?,
        ),
        (
            "source_factor_digests",
            block.source_factor_digests.len(),
            native_area
                .checked_mul(32)
                .ok_or_else(|| invalid("source factor digest dimensions overflow usize"))?,
        ),
        (
            "compressed_node_ids",
            block.compressed_node_ids.len(),
            native_area,
        ),
        ("phase_node_ids", block.phase_node_ids.len(), output_area),
        (
            "native_validity_bits",
            block.native_validity_bits.len(),
            native_area.div_ceil(8),
        ),
    ] {
        check_len(name, actual, expected)?;
    }
    ensure_valid(
        block
            .source_content_digests
            .chunks_exact(32)
            .all(|digest| digest.iter().any(|byte| *byte != 0)),
        "primitive source has an all-zero content digest",
    )?;
    ensure_valid(
        packed_trailing_bits_are_zero(&block.native_validity_bits, native_area),
        "native validity sets bits outside the native grid",
    )?;
    ensure_valid(
        f64::from_bits(block.branch_tolerance_bits).is_finite()
            && f64::from_bits(block.branch_tolerance_bits) > 0.0,
        "branch tolerance must be finite and positive",
    )?;
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
                && parent.owned_output_grid == block.owned_output_grid
                && parent.rect_support == block.rect_support
                && parent.estimator_branch == block.estimator_branch
                && parent.branch_tolerance_bits == block.branch_tolerance_bits,
            "covariance parent burst, geometry, estimator branch, or tolerance differs from child",
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
            ensure_valid(
                source_digest(block, native_index) == source_digest(block, prior_index),
                "one covariance source ID has different content digests",
            )?;
            ensure_valid(
                source_factor_digest(block, native_index)
                    == source_factor_digest(block, prior_index),
                "one covariance source ID has different numeric factor receipts",
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
            ensure_valid(
                source_digest(block, native_index) == source_digest(prior, location.native_index),
                "one covariance source ID has different content digests",
            )?;
            ensure_valid(
                source_factor_digest(block, native_index)
                    == source_factor_digest(prior, location.native_index),
                "one covariance source ID has different numeric factor receipts",
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
        ensure_valid(
            prior.output_grid.stride_y == block.output_grid.stride_y
                && prior.output_grid.stride_x == block.output_grid.stride_x
                && prior.rect_support == block.rect_support
                && prior.estimator_branch == block.estimator_branch
                && prior.branch_tolerance_bits == block.branch_tolerance_bits,
            "covariance records in one burst differ in geometry, estimator branch, or tolerance",
        )?;
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
                && prior.rect_support == block.rect_support
                && prior.estimator_branch == block.estimator_branch
                && prior.branch_tolerance_bits == block.branch_tolerance_bits,
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

fn source_digest(block: &CovarianceBlockTopology, native_index: usize) -> &[u8] {
    &block.source_content_digests[native_index * 32..(native_index + 1) * 32]
}

fn source_factor_digest(block: &CovarianceBlockTopology, native_index: usize) -> &[u8] {
    &block.source_factor_digests[native_index * 32..(native_index + 1) * 32]
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
            ensure_valid(
                source_digest(left, left_index) == source_digest(right, right_index),
                "shared native source has different content digests across tiles",
            )?;
            ensure_valid(
                source_factor_digest(left, left_index) == source_factor_digest(right, right_index),
                "shared native source has different numeric factor receipts across tiles",
            )?;
            ensure_valid(
                packed_bit(&left.native_validity_bits, left_index)
                    == packed_bit(&right.native_validity_bits, right_index),
                "shared native source has different validity across tiles",
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

fn future_min_native_rows(tiles: &[CovarianceTilePlan]) -> Vec<u64> {
    let mut minimum = u64::MAX;
    let mut rows = tiles
        .iter()
        .rev()
        .map(|tile| {
            minimum = minimum.min(tile.native_grid.row_start);
            minimum
        })
        .collect::<Vec<_>>();
    rows.reverse();
    rows
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

fn consecutive(values: &[u32]) -> bool {
    !values.is_empty()
        && values
            .windows(2)
            .all(|pair| pair[0].checked_add(1) == Some(pair[1]))
}

fn is_sha256_digest(value: &str) -> bool {
    let hex = value.strip_prefix("sha256:").unwrap_or(value);
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_nonzero_sha256_digest(value: &str) -> bool {
    is_sha256_digest(value)
        && value
            .strip_prefix("sha256:")
            .unwrap_or(value)
            .bytes()
            .any(|byte| byte != b'0')
}

fn is_exact_producer_code_identity(value: &str) -> bool {
    let digest = value.strip_prefix("sha256:").unwrap_or(value);
    matches!(digest.len(), 40 | 64) && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_digest_bytes(value: &str) -> Option<[u8; 32]> {
    let hexadecimal = value.strip_prefix("sha256:").unwrap_or(value).as_bytes();
    if hexadecimal.len() != 64 {
        return None;
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in hexadecimal.chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0])?;
        let low = hex_nibble(pair[1])?;
        digest[index] = (high << 4) | low;
    }
    Some(digest)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn invalid(message: impl Into<String>) -> IoError {
    IoError::Shape(message.into())
}

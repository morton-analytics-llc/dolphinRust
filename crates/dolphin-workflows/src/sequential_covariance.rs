//! Topology and byte-capped replay queries for sequential source influence.
//!
//! The persistent representation is implicit: blocks retain deterministic
//! node/source IDs and carried-block ancestry. [`InfluenceDag`] remains the
//! lower-level analytic algebra; production queries regenerate and immediately
//! contract local JVPs. No expanded spatial incidence, ancestry coefficient,
//! or full-frame influence arrays are constructed here.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use dolphin_core::config::{CompressedSlcPlan, ShpMethod};
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_io::{
    read_covariance_operator_block, read_covariance_operator_metadata_with_byte_cap,
    CovarianceEstimatorBranch, CovarianceOperatorBlock, CovarianceOperatorGrid,
    CovarianceOperatorStatus, CovariancePhaseComponent, CovariancePhaseComponentKind,
    CovarianceRectSupport, CovarianceReplayStatus, CovarianceSupportOrdering,
};
use dolphin_phaselink::{
    compress_pixel_jvp, phase_angle_jvp, process_coherence_matrix,
    rect_source_values_coherence_jvp, replay_rect_source_values, CompressionJvpError,
    CompressionReplayGrid, CompressionReplayStatus, CovarianceReplayError, EstimatorJvpError,
    FixedBranchStatus, FixedEstimatorBranch, InfluenceDag, InfluenceError, NativeSourcePixel,
    NodeId, PhaseReplayGrid, ProperComplexFactor, RectPixelReplay, RectReplayDescriptor, SourceId,
    TemporalCoordinate,
};
use dolphin_stack::{MiniStack, MiniStackPlanner};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use sha2::{Digest, Sha256};

use crate::sequential::SequentialConfig;

/// Stable method name for the replayable global covariance operator.
pub const SEQUENTIAL_SOURCE_DAG_METHOD: &str = "sequential_source_dag_v1";

const NODE_KIND_SHIFT: u32 = 62;
const NODE_MAJOR_LIMIT: u32 = 1 << 30;

/// Strong namespace used to derive tile-independent source and node IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayIdNamespace {
    /// Immutable burst identity.
    pub burst_id: String,
    /// Ordered source-manifest digest.
    pub source_manifest_digest: [u8; 32],
    /// Frozen source-model version digest.
    pub source_model_version_digest: [u8; 32],
    /// Global native-grid origin for this bounded replay block.
    pub native_origin: (u64, u64),
    /// Global output-grid origin for this bounded replay block.
    pub output_origin: (u64, u64),
    /// Global origin of the public output rectangle owned by this record.
    pub owned_output_origin: (u64, u64),
    /// Shape of the public output rectangle owned by this record.
    pub owned_output_shape: (usize, usize),
}

/// Strong identity and global grids for streaming captured operator blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct SequentialCovarianceCaptureRequest {
    /// Immutable burst identity.
    pub burst_id: String,
    /// Ordered source-manifest digest.
    pub source_manifest_digest: [u8; 32],
    /// Frozen source-model version digest.
    pub source_model_version_digest: [u8; 32],
    /// Native replay grid for this whole or bounded call.
    pub native_grid: CovarianceOperatorGrid,
    /// Full local phase replay grid, including tile halo.
    pub output_grid: CovarianceOperatorGrid,
    /// Public owner/write rectangle contained in `output_grid`.
    pub owned_output_grid: CovarianceOperatorGrid,
    /// Fixed-branch tolerance used by phase/compression replay capture.
    pub branch_tolerance: f64,
}

/// Immutable identity asserted by a bounded replay provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequentialSourceProviderIdentity {
    /// Ordered source-manifest digest whose members the provider verifies.
    pub source_manifest_digest: [u8; 32],
    /// Frozen source-model version digest.
    pub source_model_version_digest: [u8; 32],
    /// Digest of the ordered proper-complex factor model.
    pub source_model_hash: [u8; 32],
}

/// Verified raw values and proper-complex factor for one primitive source.
#[derive(Debug, Clone)]
pub struct ResolvedPrimitiveSource {
    /// Consumer-independent source ID.
    pub id: SourceId,
    /// Ordered raw complex values for the block's new real dates.
    pub samples: Array1<Cf64>,
    /// Validated lower factor and ordered component/model identity.
    pub factor: ProperComplexFactor,
    /// Digest of the resolved immutable raw bytes.
    pub content_digest: [u8; 32],
}

/// Captured fixed-branch phase state for one block/output pixel.
#[derive(Debug, Clone)]
pub struct ResolvedPhaseReplay {
    /// Record-specific phase node ID.
    pub id: NodeId,
    /// Combined carried-plus-real linked phasors in production order.
    pub linked_phase: Array1<Cf64>,
    /// Captured selected eigenvalue.
    pub selected_eigenvalue: f64,
    /// Captured selected eigengap.
    pub selected_eigengap: f64,
    /// Per-node captured validity.
    pub status: CovarianceOperatorStatus,
    /// Persisted fixed estimator branch.
    pub estimator_branch: CovarianceEstimatorBranch,
    /// Persisted positive branch-separation tolerance.
    pub branch_tolerance: f64,
}

/// Captured compressed state for one block/native pixel.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedCompressionReplay {
    /// Record-specific compressed node ID.
    pub id: NodeId,
    /// Production compressed complex value.
    pub value: Cf64,
    /// Captured projection accumulator.
    pub projection: Cf64,
    /// Captured mean amplitude.
    pub mean_amplitude: f64,
    /// Per-node captured validity.
    pub status: CovarianceOperatorStatus,
}

/// Byte-bounded resolver used to regenerate local production influence edges.
pub trait SequentialSourceReplayProvider {
    /// Verified immutable provider/model identity.
    fn identity(&self) -> &SequentialSourceProviderIdentity;

    /// Maximum provider-internal resident bytes during one resolver call.
    ///
    /// Returned source/phase/compression values are counted by the workflow
    /// estimate; this bound covers additional provider buffers, including a
    /// capped HDF5 block read.
    fn maximum_resident_bytes(&self) -> u64;

    /// Resolve one current-block primitive source and factor.
    ///
    /// Implementations must verify the member content digest against their
    /// ordered source manifest before returning.
    fn resolve_source(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError>;

    /// Resolve one captured phase node from the persisted operator.
    fn resolve_phase(
        &mut self,
        block: &SequentialReplayBlock,
        output_index: usize,
    ) -> Result<ResolvedPhaseReplay, SequentialReplayError>;

    /// Resolve one captured compressed node from the persisted operator.
    fn resolve_compression(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedCompressionReplay, SequentialReplayError>;
}

/// Immutable raw-source/model resolver used by the capped artifact provider.
pub trait SequentialPrimitiveSourceResolver {
    /// Verified source manifest and proper-complex model identity.
    fn identity(&self) -> &SequentialSourceProviderIdentity;

    /// Maximum resolver-internal resident bytes beyond the returned source.
    fn maximum_resident_bytes(&self) -> u64;

    /// Resolve and verify one raw source plus its proper-complex factor.
    fn resolve_source(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError>;
}

/// Read/cache metrics for a capped artifact replay provider.
///
/// The IO reader does not currently return the admitted logical payload size,
/// so payload-byte fields are `None`; `block_reservation_bytes` is the exact
/// bound reserved by [`SequentialSourceReplayProvider::maximum_resident_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CovarianceArtifactReplayMetrics {
    /// Physical capped single-block reads completed after provider admission.
    pub operator_block_reads: u64,
    /// Logical block resolutions served from the one-block cache.
    pub operator_block_cache_hits: u64,
    /// Logical block bytes read, unavailable from the current IO receipt.
    pub logical_block_bytes_read: Option<u64>,
    /// Wall-clock time spent in physical capped block reads.
    pub operator_block_read_elapsed: Duration,
    /// Current cached logical payload bytes, unavailable from the IO receipt.
    pub current_cached_payload_bytes: Option<u64>,
    /// Peak cached logical payload bytes, unavailable from the IO receipt.
    pub peak_cached_payload_bytes: Option<u64>,
    /// Configured cap reserved for metadata and one cached block read.
    pub block_reservation_bytes: u64,
    /// Persistent ID of the currently cached operator block.
    pub cached_block_id: Option<u64>,
}

/// Production replay provider backed by capped single-block HDF5 reads.
///
/// Operator blocks are never eagerly loaded as a complete artifact. Every
/// phase/compression resolver call loads at most `operator_block_byte_cap`
/// payload bytes through [`read_covariance_operator_block`] and discards the
/// block after extracting the requested state.
pub struct CovarianceArtifactReplayProvider<R> {
    operator_path: PathBuf,
    operator_block_byte_cap: u64,
    identity: SequentialSourceProviderIdentity,
    source_resolver: R,
    cached_block: Option<CovarianceOperatorBlock>,
    operator_block_reads: u64,
    operator_block_cache_hits: u64,
    operator_block_read_elapsed: Duration,
}

impl<R> CovarianceArtifactReplayProvider<R>
where
    R: SequentialPrimitiveSourceResolver,
{
    /// Open and metadata-check a completed operator without loading its numeric blocks.
    ///
    /// # Errors
    /// Returns an explicit status for a zero cap, non-replayable metadata, or
    /// source manifest/model receipt mismatch.
    pub fn open(
        operator_path: impl AsRef<Path>,
        operator_block_byte_cap: u64,
        source_resolver: R,
    ) -> Result<Self, SequentialReplayError> {
        if operator_block_byte_cap == 0 {
            return Err(SequentialReplayError::Invalid(
                "operator block byte cap must be positive",
            ));
        }
        let metadata = read_covariance_operator_metadata_with_byte_cap(
            operator_path.as_ref(),
            operator_block_byte_cap,
        )
        .map_err(|_| {
            SequentialReplayError::Provider(
                ReplayStatus::SourceUnavailable,
                "capped covariance operator metadata read failed",
            )
        })?;
        if metadata.replay_status != CovarianceReplayStatus::Replayable {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceUnavailable,
                "covariance operator metadata is not replayable",
            ));
        }
        let identity = source_resolver.identity().clone();
        if !metadata
            .source
            .manifest_digest
            .as_deref()
            .is_some_and(|value| digest_matches(value, identity.source_manifest_digest))
            || !metadata
                .source
                .model_version
                .as_deref()
                .is_some_and(|value| digest_matches(value, identity.source_model_version_digest))
            || !metadata
                .source
                .model_receipt_digest
                .as_deref()
                .is_some_and(|value| digest_matches(value, identity.source_model_hash))
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceIdentityMismatch,
                "artifact manifest/model receipt differs from the source resolver",
            ));
        }
        Ok(Self {
            operator_path: operator_path.as_ref().to_path_buf(),
            operator_block_byte_cap,
            identity,
            source_resolver,
            cached_block: None,
            operator_block_reads: 0,
            operator_block_cache_hits: 0,
            operator_block_read_elapsed: Duration::ZERO,
        })
    }

    /// Return physical-read, cache-hit, timing, and resident-cap metrics.
    #[must_use]
    pub fn metrics(&self) -> CovarianceArtifactReplayMetrics {
        CovarianceArtifactReplayMetrics {
            operator_block_reads: self.operator_block_reads,
            operator_block_cache_hits: self.operator_block_cache_hits,
            logical_block_bytes_read: None,
            operator_block_read_elapsed: self.operator_block_read_elapsed,
            current_cached_payload_bytes: None,
            peak_cached_payload_bytes: None,
            block_reservation_bytes: self.operator_block_byte_cap,
            cached_block_id: self.cached_block.as_ref().map(|block| block.block_id),
        }
    }

    fn read_block(
        &mut self,
        block: &SequentialReplayBlock,
    ) -> Result<&CovarianceOperatorBlock, SequentialReplayError> {
        if self
            .cached_block
            .as_ref()
            .is_some_and(|stored| stored.block_id == block.id.get())
        {
            self.operator_block_cache_hits = self.operator_block_cache_hits.saturating_add(1);
        } else {
            self.cached_block = None;
            let started = Instant::now();
            let stored = read_covariance_operator_block(
                &self.operator_path,
                block.id.get(),
                self.operator_block_byte_cap,
            )
            .map_err(|_| {
                SequentialReplayError::Provider(
                    ReplayStatus::SourceUnavailable,
                    "capped covariance operator block read failed",
                )
            })?;
            self.operator_block_read_elapsed = self
                .operator_block_read_elapsed
                .saturating_add(started.elapsed());
            self.operator_block_reads = self.operator_block_reads.saturating_add(1);
            self.cached_block = Some(stored);
        }
        let stored = self
            .cached_block
            .as_ref()
            .ok_or(SequentialReplayError::Provider(
                ReplayStatus::SourceUnavailable,
                "capped covariance operator block cache is empty",
            ))?;
        let expected_dates = (block.real_date_start.get()
            ..block
                .real_date_start
                .get()
                .checked_add(u32::try_from(block.num_real_dates).map_err(|_| {
                    SequentialReplayError::Invalid("replay block date count exceeds u32")
                })?)
                .ok_or(SequentialReplayError::Invalid(
                    "replay block date range overflows u32",
                ))?)
            .collect::<Vec<_>>();
        if stored.generation != block.generation
            || stored.source_date_indices != expected_dates
            || stored.carry_parent_ids
                != block
                    .carried_parent_ids
                    .iter()
                    .map(|parent| parent.get())
                    .collect::<Vec<_>>()
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::ReplayStateMismatch,
                "operator block topology differs from the planned replay",
            ));
        }
        Ok(stored)
    }
}

impl<R> SequentialSourceReplayProvider for CovarianceArtifactReplayProvider<R>
where
    R: SequentialPrimitiveSourceResolver,
{
    fn identity(&self) -> &SequentialSourceProviderIdentity {
        &self.identity
    }

    fn maximum_resident_bytes(&self) -> u64 {
        self.operator_block_byte_cap
            .saturating_add(self.source_resolver.maximum_resident_bytes())
    }

    fn resolve_source(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError> {
        if self.source_resolver.identity() != &self.identity {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceIdentityMismatch,
                "source resolver identity changed after artifact admission",
            ));
        }
        let stored_source_id = self
            .read_block(block)?
            .source_ids
            .get(native_index)
            .copied();
        let source = self.source_resolver.resolve_source(block, native_index)?;
        if stored_source_id != Some(source.id.get()) {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceIdentityMismatch,
                "artifact source ID differs from the immutable source resolver",
            ));
        }
        Ok(source)
    }

    fn resolve_phase(
        &mut self,
        block: &SequentialReplayBlock,
        output_index: usize,
    ) -> Result<ResolvedPhaseReplay, SequentialReplayError> {
        let stored = self.read_block(block)?;
        let width = stored.phase_components.len();
        let start = output_index
            .checked_mul(width)
            .ok_or(SequentialReplayError::Invalid(
                "phase replay offset overflows usize",
            ))?;
        let angles = stored.phase_angles.get(start..start + width).ok_or(
            SequentialReplayError::Provider(
                ReplayStatus::ReplayStateMismatch,
                "phase output index is outside the capped operator block",
            ),
        )?;
        Ok(ResolvedPhaseReplay {
            id: NodeId::new(*stored.phase_node_ids.get(output_index).ok_or(
                SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "phase node ID is missing from the capped operator block",
                ),
            )?),
            linked_phase: Array1::from_iter(
                angles.iter().map(|&angle| Cf64::from_polar(1.0, angle)),
            ),
            selected_eigenvalue: *stored.selected_eigenvalue.get(output_index).ok_or(
                SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "phase eigenvalue is missing from the capped operator block",
                ),
            )?,
            selected_eigengap: *stored.eigen_gap.get(output_index).ok_or(
                SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "phase eigengap is missing from the capped operator block",
                ),
            )?,
            status: *stored
                .status
                .get(output_index)
                .ok_or(SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "phase status is missing from the capped operator block",
                ))?,
            estimator_branch: stored.estimator_branch,
            branch_tolerance: stored.branch_tolerance,
        })
    }

    fn resolve_compression(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedCompressionReplay, SequentialReplayError> {
        let stored = self.read_block(block)?;
        Ok(ResolvedCompressionReplay {
            id: NodeId::new(*stored.compressed_node_ids.get(native_index).ok_or(
                SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "compressed node ID is missing from the capped operator block",
                ),
            )?),
            value: *stored.compressed_raster.get(native_index).ok_or(
                SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "compressed value is missing from the capped operator block",
                ),
            )?,
            projection: *stored.projection_accumulator.get(native_index).ok_or(
                SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "compression projection is missing from the capped operator block",
                ),
            )?,
            mean_amplitude: *stored.mean_amplitude.get(native_index).ok_or(
                SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "compression amplitude is missing from the capped operator block",
                ),
            )?,
            status: *stored.compressed_status.get(native_index).ok_or(
                SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "compression status is missing from the capped operator block",
                ),
            )?,
        })
    }
}

impl ReplayIdNamespace {
    fn validate(&self) -> Result<(), SequentialReplayError> {
        if self.burst_id.is_empty()
            || self.source_manifest_digest.iter().all(|byte| *byte == 0)
            || self
                .source_model_version_digest
                .iter()
                .all(|byte| *byte == 0)
            || self.owned_output_shape.0 == 0
            || self.owned_output_shape.1 == 0
        {
            return Err(SequentialReplayError::Unsupported(
                ReplayStatus::UnsupportedSourceIdentity,
            ));
        }
        Ok(())
    }
}

impl SequentialCovarianceCaptureRequest {
    pub(crate) fn namespace_for(
        &self,
        native_shape: (usize, usize),
        output_shape: (usize, usize),
        strides: Strides,
    ) -> Result<ReplayIdNamespace, SequentialReplayError> {
        if !self.branch_tolerance.is_finite() || self.branch_tolerance <= 0.0 {
            return Err(SequentialReplayError::Invalid(
                "covariance capture branch tolerance is invalid",
            ));
        }
        let native_matches = usize::try_from(self.native_grid.rows).ok() == Some(native_shape.0)
            && usize::try_from(self.native_grid.cols).ok() == Some(native_shape.1)
            && self.native_grid.stride_y == 1
            && self.native_grid.stride_x == 1;
        let output_matches = usize::try_from(self.output_grid.rows).ok() == Some(output_shape.0)
            && usize::try_from(self.output_grid.cols).ok() == Some(output_shape.1)
            && usize::try_from(self.output_grid.stride_y).ok() == Some(strides.y)
            && usize::try_from(self.output_grid.stride_x).ok() == Some(strides.x);
        if !native_matches
            || !output_matches
            || self.owned_output_grid.stride_y != self.output_grid.stride_y
            || self.owned_output_grid.stride_x != self.output_grid.stride_x
            || !grid_contains(self.output_grid, self.owned_output_grid)
        {
            return Err(SequentialReplayError::Invalid(
                "covariance capture grids do not match the sequential call",
            ));
        }
        let owned_output_shape = (
            usize::try_from(self.owned_output_grid.rows).map_err(|_| {
                SequentialReplayError::Invalid("owned output row count exceeds usize")
            })?,
            usize::try_from(self.owned_output_grid.cols).map_err(|_| {
                SequentialReplayError::Invalid("owned output column count exceeds usize")
            })?,
        );
        let namespace = ReplayIdNamespace {
            burst_id: self.burst_id.clone(),
            source_manifest_digest: self.source_manifest_digest,
            source_model_version_digest: self.source_model_version_digest,
            native_origin: (self.native_grid.row_start, self.native_grid.col_start),
            output_origin: (self.output_grid.row_start, self.output_grid.col_start),
            owned_output_origin: (
                self.owned_output_grid.row_start,
                self.owned_output_grid.col_start,
            ),
            owned_output_shape,
        };
        namespace.validate()?;
        Ok(namespace)
    }
}

/// Global zero-based ministack identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalBlockId(u64);

impl GlobalBlockId {
    /// Construct an identifier from its persistent value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the persistent integer value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Global zero-based real-acquisition identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalDateId(u32);

impl GlobalDateId {
    /// Construct an identifier from its persistent value.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the persistent integer value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Backend used by the replay-producing phase-link call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayBackend {
    /// CPU/f64 production kernels, the only version-1 supported backend.
    CpuF64,
    /// GPU or hybrid GPU execution.
    Gpu,
}

/// Execution facts that determine whether version-1 replay is supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayExecutionScope {
    /// Whether the covariance-operator path was explicitly requested.
    pub enabled: bool,
    /// Backend actually selected for the phase-link call.
    pub backend: ReplayBackend,
    /// Whether EMI fell back to a different estimator branch.
    pub estimator_fallback: bool,
    /// Whether phase-bias correction changes the linked phase.
    pub phase_bias_correction: bool,
    /// Whether source keys are backed by immutable strong identities.
    pub strong_source_identity: bool,
    /// Number of stitched bursts represented by the requested graph.
    pub stitched_burst_count: usize,
}

/// Stable support or query disposition for a replay operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayStatus {
    /// The frozen version-1 branch is evaluable.
    Valid,
    /// Operator capture was not requested.
    Disabled,
    /// The compressed-reference plan is not `AlwaysFirst`.
    UnsupportedReferencePlan,
    /// The public output reference is not acquisition zero.
    UnsupportedOutputReference,
    /// The selected execution backend is not CPU/f64.
    UnsupportedBackend,
    /// The realized support is adaptive rather than rectangular.
    UnsupportedShpMethod,
    /// The fixed phase-link estimator branch was not retained.
    UnsupportedEstimatorFallback,
    /// Phase-bias correction is not part of the frozen derivative contract.
    UnsupportedPhaseBiasCorrection,
    /// Source identities are weak or mutable.
    UnsupportedSourceIdentity,
    /// A stitched multiburst graph lacks seam covariance.
    UnsupportedSeamCovariance,
    /// Planner, grid, identifier, or query metadata is invalid.
    InvalidTopology,
    /// The numeric local influence graph is inconsistent with the topology.
    InvalidReplayGraph,
    /// Immutable raw source bytes could not be resolved.
    SourceUnavailable,
    /// A proper-complex source factor could not be resolved.
    SourceModelUnavailable,
    /// Provider, source, component, digest, or model identity did not match.
    SourceIdentityMismatch,
    /// Recomputed estimator/compression state differs from the captured branch.
    ReplayStateMismatch,
    /// The selected local information state is singular.
    SingularLocalInformation,
    /// A required captured or recomputed state is non-finite.
    NonFiniteReplayState,
    /// A selected fixed branch is not differentiable at the captured state.
    NondifferentiableNode,
    /// A required compressed node is not differentiable.
    InvalidCompression,
    /// A selected or internally required node is masked.
    MaskedNode,
    /// The topology-only query estimate exceeds its byte cap.
    DependencyConeExceedsBudget,
}

impl ReplayStatus {
    /// Stable serialized status name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Valid => "valid",
            Self::Disabled => "disabled",
            Self::UnsupportedReferencePlan => "unsupported_reference_plan",
            Self::UnsupportedOutputReference => "unsupported_output_reference",
            Self::UnsupportedBackend => "unsupported_backend",
            Self::UnsupportedShpMethod => "unsupported_shp_method",
            Self::UnsupportedEstimatorFallback => "unsupported_estimator_fallback",
            Self::UnsupportedPhaseBiasCorrection => "unsupported_phase_bias_correction",
            Self::UnsupportedSourceIdentity => "unsupported_source_identity",
            Self::UnsupportedSeamCovariance => "unsupported_seam_covariance",
            Self::InvalidTopology => "invalid_topology",
            Self::InvalidReplayGraph => "invalid_replay_graph",
            Self::SourceUnavailable => "source_unavailable",
            Self::SourceModelUnavailable => "source_model_unavailable",
            Self::SourceIdentityMismatch => "source_identity_mismatch",
            Self::ReplayStateMismatch => "replay_state_mismatch",
            Self::SingularLocalInformation => "singular_local_information",
            Self::NonFiniteReplayState => "nonfinite_replay_state",
            Self::NondifferentiableNode => "nondifferentiable_node",
            Self::InvalidCompression => "invalid_compression",
            Self::MaskedNode => "masked_node",
            Self::DependencyConeExceedsBudget => "dependency_cone_exceeds_budget",
        }
    }
}

/// One implicit block in the global replay topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequentialReplayBlock {
    /// Global block identifier.
    pub id: GlobalBlockId,
    /// Sequential generation/ministack number.
    pub generation: u32,
    /// Oldest real acquisition in this block.
    pub real_date_start: GlobalDateId,
    /// Number of new real acquisitions in this block.
    pub num_real_dates: usize,
    /// Ordered compressed-block parents after carry-cap eviction.
    pub carried_parent_ids: Vec<GlobalBlockId>,
    /// Local phase-vector dimension after removing its reference coordinate.
    pub phase_dimension: usize,
}

/// Topology-only memory estimate produced before replay allocations or source reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyConeEstimate {
    /// Blocks visited in deterministic reverse-frontier order.
    pub block_ids: Vec<GlobalBlockId>,
    /// Bytes for active source/node adjoints in the requested microbatch.
    pub frontier_bytes: u64,
    /// Bytes for one raw-complex source window and its real source factor.
    pub source_window_bytes: u64,
    /// Peak bytes for one streamed JVP plus topology/adjoint control records.
    /// Regenerated coefficients are contracted and discarded immediately.
    pub operator_bytes: u64,
    /// Bytes for the largest recomputed covariance/derivative baseline buffer.
    pub baseline_bytes: u64,
    /// Bytes for the realized-support bit set over the dependency cone.
    pub support_bytes: u64,
    /// Bytes for the selected dense temporal covariance result.
    pub covariance_bytes: u64,
    /// Provider-internal bytes reserved before the first source/artifact read.
    pub provider_bytes: u64,
    /// Conservative total of all query allocations above.
    pub total_bytes: u64,
}

/// Byte-capped query parameters for one covariance replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DependencyConeQuery {
    /// Rank of the caller-supplied real source factor at each native source.
    pub source_rank: usize,
    /// Number of output pixels replayed together.
    pub microbatch: usize,
    /// Maximum admitted query allocation in bytes.
    pub byte_cap: u64,
}

/// Successful byte-capped temporal covariance replay.
#[derive(Debug)]
pub struct TemporalCovarianceReplay {
    /// Acquisition-0-gauge covariance in selection order.
    pub covariance: Array2<f64>,
    /// Topology-only allocation receipt checked before graph contraction.
    pub dependency_cone: DependencyConeEstimate,
}

struct SpatialQueryCone {
    active_outputs: Vec<BTreeSet<usize>>,
    active_sources: Vec<BTreeSet<usize>>,
    required_compressed: Vec<BTreeSet<usize>>,
    selected_dates: Vec<BTreeSet<(GlobalDateId, usize)>>,
}

struct StreamingAdjoints {
    phase: Vec<BTreeMap<usize, Array2<f64>>>,
    compressed: Vec<BTreeMap<usize, Array2<f64>>>,
}

struct PhaseWindowReplay {
    source_pixels: Vec<NativeSourcePixel>,
    source_values: Array2<Cf64>,
    replay: RectPixelReplay,
}

/// Replay topology, support, budget, or graph failure.
#[derive(Debug)]
pub enum SequentialReplayError {
    /// A frozen version-1 support condition was not met.
    Unsupported(ReplayStatus),
    /// Topology or query metadata is invalid.
    Invalid(&'static str),
    /// Query byte cap is smaller than the conservative estimate.
    Budget(DependencyConeEstimate),
    /// The supplied numeric influence graph failed validation or replay.
    Influence(InfluenceError),
    /// The production capture path or streaming sink failed.
    Execution(&'static str),
    /// External source/model resolution or captured-state verification failed.
    Provider(ReplayStatus, &'static str),
}

impl SequentialReplayError {
    /// Stable status corresponding to this failure.
    #[must_use]
    pub const fn status(&self) -> ReplayStatus {
        match self {
            Self::Unsupported(status) => *status,
            Self::Invalid(_) => ReplayStatus::InvalidTopology,
            Self::Budget(_) => ReplayStatus::DependencyConeExceedsBudget,
            Self::Influence(_) => ReplayStatus::InvalidReplayGraph,
            Self::Execution(_) => ReplayStatus::InvalidReplayGraph,
            Self::Provider(status, _) => *status,
        }
    }

    /// Return the dependency-cone receipt for a budget rejection.
    #[must_use]
    pub const fn estimate(&self) -> Option<&DependencyConeEstimate> {
        match self {
            Self::Budget(estimate) => Some(estimate),
            _ => None,
        }
    }
}

impl Display for SequentialReplayError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(status) => write!(f, "{}", status.as_str()),
            Self::Invalid(message) => write!(f, "{message}"),
            Self::Budget(estimate) => {
                write!(f, "dependency cone requires {} bytes", estimate.total_bytes)
            }
            Self::Influence(error) => Display::fmt(error, f),
            Self::Execution(message) => f.write_str(message),
            Self::Provider(_, message) => f.write_str(message),
        }
    }
}

impl Error for SequentialReplayError {}

impl From<InfluenceError> for SequentialReplayError {
    fn from(value: InfluenceError) -> Self {
        Self::Influence(value)
    }
}

/// Deterministic implicit topology for one sequential source-influence graph.
#[derive(Debug, Clone)]
pub struct SequentialReplayTopology {
    blocks: Vec<SequentialReplayBlock>,
    num_real_dates: usize,
    native_area: usize,
    output_area: usize,
    native_shape: (usize, usize),
    output_shape: (usize, usize),
    half_window: HalfWindow,
    strides: Strides,
    native_validity: Vec<bool>,
    id_namespace: Option<ReplayIdNamespace>,
    estimator_branch: FixedEstimatorBranch,
}

impl SequentialReplayTopology {
    /// Validate the version-1 support boundary and plan its global block graph.
    ///
    /// Unsupported cases return their stable status before a block topology is
    /// allocated. `support_slots_per_output` is the fixed rectangular window
    /// area whose realized validity bits are recorded for each output cell.
    ///
    /// # Errors
    /// Returns an explicit unsupported status or invalid topology error.
    pub fn plan(
        num_real_dates: usize,
        native_shape: (usize, usize),
        output_shape: (usize, usize),
        support_slots_per_output: usize,
        cfg: &SequentialConfig,
        scope: ReplayExecutionScope,
    ) -> Result<Self, SequentialReplayError> {
        let native_validity = Array2::from_elem(native_shape, true);
        Self::plan_impl(
            num_real_dates,
            native_shape,
            output_shape,
            support_slots_per_output,
            native_validity.view(),
            cfg,
            scope,
            None,
        )
    }

    /// Plan a graph with one immutable native validity mask.
    ///
    /// # Errors
    /// Returns an explicit unsupported status, mask mismatch, or invalid
    /// topology error.
    pub fn plan_masked(
        num_real_dates: usize,
        native_shape: (usize, usize),
        output_shape: (usize, usize),
        support_slots_per_output: usize,
        native_validity: ArrayView2<bool>,
        cfg: &SequentialConfig,
        scope: ReplayExecutionScope,
    ) -> Result<Self, SequentialReplayError> {
        Self::plan_impl(
            num_real_dates,
            native_shape,
            output_shape,
            support_slots_per_output,
            native_validity,
            cfg,
            scope,
            None,
        )
    }

    /// Plan a strongly identified whole or bounded graph with fixed validity.
    ///
    /// IDs include immutable source/model/burst identity plus global grid
    /// coordinates, so overlapping bounded consumers resolve the same source
    /// key independently of their local tile index.
    ///
    /// # Errors
    /// Returns an explicit unsupported identity/scope status or invalid
    /// topology error.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_identified(
        num_real_dates: usize,
        native_shape: (usize, usize),
        output_shape: (usize, usize),
        support_slots_per_output: usize,
        native_validity: ArrayView2<bool>,
        cfg: &SequentialConfig,
        scope: ReplayExecutionScope,
        id_namespace: ReplayIdNamespace,
    ) -> Result<Self, SequentialReplayError> {
        id_namespace.validate()?;
        Self::plan_impl(
            num_real_dates,
            native_shape,
            output_shape,
            support_slots_per_output,
            native_validity,
            cfg,
            scope,
            Some(id_namespace),
        )
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn plan_impl(
        num_real_dates: usize,
        native_shape: (usize, usize),
        output_shape: (usize, usize),
        support_slots_per_output: usize,
        native_validity: ArrayView2<bool>,
        cfg: &SequentialConfig,
        scope: ReplayExecutionScope,
        id_namespace: Option<ReplayIdNamespace>,
    ) -> Result<Self, SequentialReplayError> {
        assess_support(cfg, scope)?;
        if num_real_dates == 0 {
            return Err(SequentialReplayError::Invalid(
                "sequential replay requires at least one real acquisition",
            ));
        }
        let native_area = checked_area(native_shape)?;
        let output_area = checked_area(output_shape)?;
        let window_rows = cfg
            .half_window
            .y
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(SequentialReplayError::Invalid(
                "replay window dimensions overflow usize",
            ))?;
        let window_cols = cfg
            .half_window
            .x
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(SequentialReplayError::Invalid(
                "replay window dimensions overflow usize",
            ))?;
        let window_area =
            window_rows
                .checked_mul(window_cols)
                .ok_or(SequentialReplayError::Invalid(
                    "replay support area overflows usize",
                ))?;
        if native_validity.dim() != native_shape
            || cfg.strides.y == 0
            || cfg.strides.x == 0
            || output_shape != cfg.strides.out_shape(native_shape)
            || window_rows > native_shape.0
            || window_cols > native_shape.1
            || support_slots_per_output != window_area
            || native_area > u32::MAX as usize
            || output_area > u32::MAX as usize
            || num_real_dates >= NODE_MAJOR_LIMIT as usize
        {
            return Err(SequentialReplayError::Invalid(
                "replay grid, support, or date identifier exceeds its supported range",
            ));
        }
        if let Some(namespace) = id_namespace.as_ref() {
            let output_row_stop = namespace.output_origin.0.checked_add(output_shape.0 as u64);
            let output_col_stop = namespace.output_origin.1.checked_add(output_shape.1 as u64);
            let owned_row_stop = namespace
                .owned_output_origin
                .0
                .checked_add(namespace.owned_output_shape.0 as u64);
            let owned_col_stop = namespace
                .owned_output_origin
                .1
                .checked_add(namespace.owned_output_shape.1 as u64);
            let contained = match (
                output_row_stop,
                output_col_stop,
                owned_row_stop,
                owned_col_stop,
            ) {
                (Some(output_row), Some(output_col), Some(owned_row), Some(owned_col)) => {
                    namespace.owned_output_origin.0 >= namespace.output_origin.0
                        && namespace.owned_output_origin.1 >= namespace.output_origin.1
                        && owned_row <= output_row
                        && owned_col <= output_col
                }
                _ => false,
            };
            if !contained {
                return Err(SequentialReplayError::Invalid(
                    "owned replay output is outside the full replay grid",
                ));
            }
        }
        let planner = MiniStackPlanner {
            num_slc: num_real_dates,
            max_num_compressed: cfg.max_num_compressed,
            output_reference_idx: isize::try_from(cfg.output_reference_idx).map_err(|_| {
                SequentialReplayError::Invalid("output reference index exceeds isize")
            })?,
            compressed_slc_plan: cfg.compressed_slc_plan,
        };
        let planned = planner
            .plan(cfg.ministack_size)
            .map_err(SequentialReplayError::Invalid)?;
        if planned.len() >= NODE_MAJOR_LIMIT as usize {
            return Err(SequentialReplayError::Invalid(
                "replay block identifier exceeds its supported range",
            ));
        }
        if id_namespace.is_some() && planned.len() > (u16::MAX as usize + 1) {
            return Err(SequentialReplayError::Invalid(
                "identified replay exceeds the 16-bit block generation range",
            ));
        }
        let mut blocks: Vec<SequentialReplayBlock> = Vec::with_capacity(planned.len());
        for block in &planned {
            let generation = u32::try_from(block.block_id).map_err(|_| {
                SequentialReplayError::Invalid("replay block generation exceeds u32")
            })?;
            let id = match id_namespace.as_ref() {
                Some(namespace) => GlobalBlockId::new(record_block_id(
                    namespace,
                    generation,
                    native_shape,
                    output_shape,
                )),
                None => GlobalBlockId::new(u64::from(generation)),
            };
            let real_date_start =
                GlobalDateId::new(u32::try_from(block.real_start).map_err(|_| {
                    SequentialReplayError::Invalid("replay source date index exceeds u32")
                })?);
            let carried_parent_ids = block
                .carried_parent_ids()
                .map(|parent| blocks[parent].id)
                .collect();
            blocks.push(SequentialReplayBlock {
                id,
                generation,
                real_date_start,
                num_real_dates: block.num_real,
                carried_parent_ids,
                phase_dimension: block.size() - 1,
            });
        }
        Ok(Self {
            blocks,
            num_real_dates,
            native_area,
            output_area,
            native_shape,
            output_shape,
            half_window: cfg.half_window,
            strides: cfg.strides,
            native_validity: native_validity.iter().copied().collect(),
            id_namespace,
            estimator_branch: match cfg.use_evd {
                true => FixedEstimatorBranch::Evd,
                false => FixedEstimatorBranch::Emi {
                    beta: cfg.beta,
                    zero_correlation_threshold: cfg.zero_correlation_threshold,
                },
            },
        })
    }

    /// Supported status of a successfully constructed topology.
    #[must_use]
    pub const fn status(&self) -> ReplayStatus {
        ReplayStatus::Valid
    }

    /// Planned blocks in global order.
    #[must_use]
    pub fn blocks(&self) -> &[SequentialReplayBlock] {
        &self.blocks
    }

    /// Deterministic primitive source ID for one block/native pixel.
    ///
    /// # Errors
    /// Returns `Err` for an unknown block or native pixel.
    pub fn source_id(
        &self,
        block: GlobalBlockId,
        native_index: usize,
    ) -> Result<SourceId, SequentialReplayError> {
        self.validate_block_local(block, native_index, self.native_area)?;
        let definition = self.block(block)?;
        let value = match &self.id_namespace {
            Some(_) => self.identified_id(
                b"source",
                u64::from(definition.generation),
                (u64::from(definition.real_date_start.get()) << 32)
                    | definition.num_real_dates as u64,
                native_index,
                true,
            ),
            None => (block.get() << 32) | native_index as u64,
        };
        Ok(SourceId::new(value))
    }

    /// Deterministic phase-node ID for one block/output pixel.
    ///
    /// # Errors
    /// Returns `Err` for an unknown block or output pixel.
    pub fn phase_node_id(
        &self,
        block: GlobalBlockId,
        output_index: usize,
    ) -> Result<NodeId, SequentialReplayError> {
        self.validate_block_local(block, output_index, self.output_area)?;
        Ok(NodeId::new(match &self.id_namespace {
            Some(_) => self.identified_id(b"phase", block.get(), 0, output_index, false),
            None => pack_node_id(0, block.get() as u32, output_index as u32).get(),
        }))
    }

    /// Deterministic compressed-node ID for one block/native pixel.
    ///
    /// # Errors
    /// Returns `Err` for an unknown block or native pixel.
    pub fn compressed_node_id(
        &self,
        block: GlobalBlockId,
        native_index: usize,
    ) -> Result<NodeId, SequentialReplayError> {
        self.validate_block_local(block, native_index, self.native_area)?;
        Ok(NodeId::new(match &self.id_namespace {
            Some(_) => self.identified_id(b"compressed", block.get(), 0, native_index, true),
            None => pack_node_id(1, block.get() as u32, native_index as u32).get(),
        }))
    }

    /// Deterministic date-node ID for one retained date/output pixel.
    ///
    /// Acquisition zero is the exact gauge and therefore has no stochastic
    /// date node.
    ///
    /// # Errors
    /// Returns `Err` for acquisition zero, an unknown date, or output pixel.
    pub fn date_node_id(
        &self,
        date: GlobalDateId,
        output_index: usize,
    ) -> Result<NodeId, SequentialReplayError> {
        self.validate_date_output(date, output_index)?;
        if date.get() == 0 {
            return Err(SequentialReplayError::Invalid(
                "acquisition zero is an exact gauge and has no date node",
            ));
        }
        let block = self.block_for_date(date)?;
        Ok(NodeId::new(match &self.id_namespace {
            Some(_) => self.identified_id(
                b"date",
                block.id.get(),
                u64::from(date.get()),
                output_index,
                false,
            ),
            None => pack_node_id(2, date.get(), output_index as u32).get(),
        }))
    }

    /// Map a public real date to the shared graph-algebra coordinate.
    ///
    /// # Errors
    /// Returns `Err` for an unknown date or output pixel.
    pub fn temporal_coordinate(
        &self,
        date: GlobalDateId,
        output_index: usize,
    ) -> Result<TemporalCoordinate, SequentialReplayError> {
        self.validate_date_output(date, output_index)?;
        match date.get() {
            0 => Ok(TemporalCoordinate::Gauge),
            _ => Ok(TemporalCoordinate::node(
                self.date_node_id(date, output_index)?,
                0,
            )),
        }
    }

    /// Enumerate the reverse block frontier, newest block first.
    ///
    /// Shared ancestors appear once even when several carried paths reach them.
    ///
    /// # Errors
    /// Returns `Err` when a target block does not exist.
    pub fn reverse_frontier(
        &self,
        targets: &[GlobalBlockId],
    ) -> Result<Vec<GlobalBlockId>, SequentialReplayError> {
        let mut pending = BTreeSet::new();
        for &target in targets {
            self.block(target)?;
            pending.insert(target);
        }
        let mut visited = BTreeSet::new();
        let mut frontier = Vec::new();
        while let Some(&block_id) = pending.iter().next_back() {
            pending.remove(&block_id);
            if !visited.insert(block_id) {
                continue;
            }
            frontier.push(block_id);
            for &parent in &self.block(block_id)?.carried_parent_ids {
                if !visited.contains(&parent) {
                    pending.insert(parent);
                }
            }
        }
        Ok(frontier)
    }

    fn spatial_query_cone(
        &self,
        selection: &[(GlobalDateId, usize)],
        microbatch: usize,
    ) -> Result<SpatialQueryCone, SequentialReplayError> {
        let mut selected_dates = vec![BTreeSet::new(); self.blocks.len()];
        let mut active_outputs = vec![BTreeSet::new(); self.blocks.len()];
        let mut distinct_outputs = BTreeSet::new();
        for &(date, output_index) in selection {
            self.validate_date_output(date, output_index)?;
            let block = self.block_for_date(date)?;
            let block_index = block.generation as usize;
            active_outputs[block_index].insert(output_index);
            selected_dates[block_index].insert((date, output_index));
            distinct_outputs.insert(output_index);
        }
        if distinct_outputs.len() != microbatch {
            return Err(SequentialReplayError::Invalid(
                "dependency-cone microbatch must equal the selected output-pixel count",
            ));
        }

        let mut required_compressed = vec![BTreeSet::new(); self.blocks.len()];
        let mut active_sources = vec![BTreeSet::new(); self.blocks.len()];
        for block_index in (0..self.blocks.len()).rev() {
            let block = &self.blocks[block_index];
            for &native in &required_compressed[block_index] {
                active_sources[block_index].insert(native);
                active_outputs[block_index].insert(self.nearest_output_index(native)?);
            }
            for output in active_outputs[block_index].clone() {
                for native in self.native_support_indices(output)? {
                    active_sources[block_index].insert(native);
                    for &parent in &block.carried_parent_ids {
                        let parent_generation = self.block(parent)?.generation as usize;
                        required_compressed[parent_generation].insert(native);
                    }
                }
            }
        }
        Ok(SpatialQueryCone {
            active_outputs,
            active_sources,
            required_compressed,
            selected_dates,
        })
    }

    /// Estimate a query from topology and dimensions only.
    ///
    /// This performs no source resolution and allocates no numeric graph
    /// adjoints. The bound includes all active node coordinates, the bounded
    /// raw/source-factor working set, one local JVP/baseline reservation,
    /// fixed support bits, and the requested result.
    ///
    /// # Errors
    /// Returns `Err` for an empty/invalid selection, zero rank/microbatch, or
    /// checked arithmetic overflow.
    #[allow(clippy::too_many_lines)]
    pub fn estimate_dependency_cone(
        &self,
        selection: &[(GlobalDateId, usize)],
        source_rank: usize,
        microbatch: usize,
    ) -> Result<DependencyConeEstimate, SequentialReplayError> {
        if selection.is_empty() || source_rank == 0 || microbatch == 0 {
            return Err(SequentialReplayError::Invalid(
                "dependency-cone selection, source rank, and microbatch must be nonzero",
            ));
        }
        let cone = self.spatial_query_cone(selection, microbatch)?;
        let block_ids: Vec<_> = (0..self.blocks.len())
            .rev()
            .filter(|&index| {
                !cone.active_outputs[index].is_empty()
                    || !cone.active_sources[index].is_empty()
                    || !cone.required_compressed[index].is_empty()
            })
            .map(|index| self.blocks[index].id)
            .collect();

        let mut frontier_coordinates = 0_u64;
        let mut max_real = 0_u64;
        let mut max_combined = 0_u64;
        let mut max_phase_dimension = 0_u64;
        let mut support_bits = 0_u64;
        let mut resident_control_records = 0_u64;
        let mut baseline_bytes = 0_u64;
        for &block_id in &block_ids {
            let block = self.block(block_id)?;
            let block_index = block.generation as usize;
            let native = cone.active_sources[block_index].len() as u64;
            let output = cone.active_outputs[block_index].len() as u64;
            let real = block.num_real_dates as u64;
            let compressed = cone.required_compressed[block_index].len() as u64;
            let selected_dates = cone.selected_dates[block_index].len() as u64;
            let source_coordinates = checked_mul(checked_mul(2, real)?, native)?;
            let phase_coordinates = checked_mul(block.phase_dimension as u64, output)?;
            let compressed_coordinates = checked_mul(2, compressed)?;
            let date_coordinates = cone.selected_dates[block_index]
                .iter()
                .filter(|(date, _)| date.get() != 0)
                .count() as u64;
            frontier_coordinates = checked_add(
                frontier_coordinates,
                checked_add(
                    checked_add(source_coordinates, phase_coordinates)?,
                    checked_add(compressed_coordinates, date_coordinates)?,
                )?,
            )?;
            max_real = max_real.max(real);
            max_phase_dimension = max_phase_dimension.max(block.phase_dimension as u64);
            let combined = checked_add(real, block.carried_parent_ids.len() as u64)?;
            max_combined = max_combined.max(combined);

            let cone_records = checked_add(
                1,
                checked_add(
                    checked_add(native, output)?,
                    checked_add(compressed, selected_dates)?,
                )?,
            )?;
            let adjoint_records = checked_add(checked_add(native, output)?, compressed)?;
            resident_control_records = checked_add(
                resident_control_records,
                checked_add(checked_add(cone_records, adjoint_records)?, 7)?,
            )?;
            for &output_index in &cone.active_outputs[block_index] {
                let support = self.native_support_indices(output_index)?.len() as u64;
                support_bits = checked_add(support_bits, support)?;
                let source_values = checked_mul(checked_mul(support, combined)?, 16)?;
                let matrix_buffers = checked_mul(checked_mul(combined, combined)?, 16 * 6)?;
                let vector_buffers = checked_mul(combined, 16 * 4)?;
                baseline_bytes = baseline_bytes.max(checked_add(
                    source_values,
                    checked_add(matrix_buffers, vector_buffers)?,
                )?);
            }
        }

        let frontier_bytes = checked_mul(
            checked_mul(frontier_coordinates, selection.len() as u64)?,
            8,
        )?;
        let lower_factor_bytes = checked_mul(checked_mul(max_real, max_real)?, 16)?;
        let component_id_bytes = checked_mul(max_real, 8)?;
        let real_embedding_bytes =
            checked_mul(checked_mul(source_rank as u64, source_rank as u64)?, 8)?;
        let raw_vector_bytes = checked_mul(max_real, 16 * 2)?;
        let source_window_bytes = checked_add(
            checked_add(lower_factor_bytes, component_id_bytes)?,
            checked_add(real_embedding_bytes, raw_vector_bytes)?,
        )?;
        let one_jvp_bytes = checked_add(
            checked_add(checked_mul(max_real, 16)?, checked_mul(max_combined, 16)?)?,
            checked_add(
                checked_add(
                    checked_mul(max_combined, 8)?,
                    checked_mul(max_phase_dimension, 8)?,
                )?,
                16,
            )?,
        )?;
        let operator_bytes =
            checked_add(checked_mul(resident_control_records, 64)?, one_jvp_bytes)?;
        let support_bytes = checked_add(support_bits, 7)? / 8;
        let selected = selection.len() as u64;
        let covariance_bytes = checked_mul(checked_mul(selected, selected)?, 8)?;
        let total_bytes = checked_add(
            checked_add(
                checked_add(frontier_bytes, source_window_bytes)?,
                checked_add(operator_bytes, baseline_bytes)?,
            )?,
            checked_add(support_bytes, covariance_bytes)?,
        )?;
        Ok(DependencyConeEstimate {
            block_ids,
            frontier_bytes,
            source_window_bytes,
            operator_bytes,
            baseline_bytes,
            support_bytes,
            covariance_bytes,
            provider_bytes: 0,
            total_bytes,
        })
    }

    /// Enforce the dependency-cone byte cap before numeric replay.
    ///
    /// # Errors
    /// Returns `DependencyConeExceedsBudget` with the full estimate when the
    /// cap is too small, or an invalid-topology error from estimation.
    pub fn preflight_dependency_cone(
        &self,
        selection: &[(GlobalDateId, usize)],
        source_rank: usize,
        microbatch: usize,
        byte_cap: u64,
    ) -> Result<DependencyConeEstimate, SequentialReplayError> {
        let estimate = self.estimate_dependency_cone(selection, source_rank, microbatch)?;
        if estimate.total_bytes > byte_cap {
            return Err(SequentialReplayError::Budget(estimate));
        }
        Ok(estimate)
    }

    /// Replay selected temporal covariance with the shared source graph algebra.
    ///
    /// Byte preflight completes before coordinates are resolved in the numeric
    /// graph. Every source path is combined by [`InfluenceDag`] before root
    /// contraction, and acquisition zero is inserted as a literal zero row and
    /// column.
    ///
    /// # Errors
    /// Returns a topology, byte-cap, or influence-graph error.
    pub fn replay_temporal_covariance<F>(
        &self,
        selection: &[(GlobalDateId, usize)],
        query: DependencyConeQuery,
        build_graph: F,
    ) -> Result<TemporalCovarianceReplay, SequentialReplayError>
    where
        F: FnOnce(&DependencyConeEstimate) -> Result<InfluenceDag, InfluenceError>,
    {
        let dependency_cone = self.preflight_dependency_cone(
            selection,
            query.source_rank,
            query.microbatch,
            query.byte_cap,
        )?;
        let coordinates = selection
            .iter()
            .map(|&(date, output)| self.temporal_coordinate(date, output))
            .collect::<Result<Vec<_>, _>>()?;
        let dag = build_graph(&dependency_cone)?;
        let covariance = dag.temporal_covariance(&coordinates)?;
        Ok(TemporalCovarianceReplay {
            covariance,
            dependency_cone,
        })
    }

    /// Stream a production fixed-branch replay from immutable captured state.
    ///
    /// The byte cap is enforced before the provider is called. Local phase and
    /// compression Jacobian-vector products are regenerated one at a time,
    /// immediately contracted into reverse adjoints, and discarded; no dense
    /// influence DAG or full-frame influence raster is materialized.
    ///
    /// # Errors
    /// Returns an explicit budget, source/model identity, captured-state, or
    /// fixed-branch status.
    #[allow(clippy::too_many_lines)]
    pub fn replay_temporal_covariance_from_provider<P>(
        &self,
        selection: &[(GlobalDateId, usize)],
        query: DependencyConeQuery,
        branch_tolerance: f64,
        provider: &mut P,
    ) -> Result<TemporalCovarianceReplay, SequentialReplayError>
    where
        P: SequentialSourceReplayProvider + ?Sized,
    {
        if !branch_tolerance.is_finite() || branch_tolerance <= 0.0 {
            return Err(SequentialReplayError::Invalid(
                "replay branch tolerance must be finite and positive",
            ));
        }
        self.validate_provider_identity(provider.identity())?;
        let cone = self.spatial_query_cone(selection, query.microbatch)?;
        let mut dependency_cone =
            self.estimate_dependency_cone(selection, query.source_rank, query.microbatch)?;
        let expected_rank = dependency_cone
            .block_ids
            .iter()
            .map(|&block| self.block(block).map(|item| 2 * item.num_real_dates))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or(SequentialReplayError::Invalid(
                "dependency cone contains no source block",
            ))?;
        if query.source_rank != expected_rank {
            return Err(SequentialReplayError::Invalid(
                "declared source rank does not match the active block factors",
            ));
        }
        dependency_cone.provider_bytes = provider.maximum_resident_bytes();
        dependency_cone.total_bytes =
            checked_add(dependency_cone.total_bytes, dependency_cone.provider_bytes)?;
        if dependency_cone.total_bytes > query.byte_cap {
            return Err(SequentialReplayError::Budget(dependency_cone));
        }

        let selected = selection.len();
        let mut adjoints = StreamingAdjoints {
            phase: vec![BTreeMap::new(); self.blocks.len()],
            compressed: vec![BTreeMap::new(); self.blocks.len()],
        };
        for (column, &(date, output)) in selection.iter().enumerate() {
            if date.get() == 0 {
                continue;
            }
            let block = self.block_for_date(date)?;
            let block_index = block.generation as usize;
            let full_component = block.carried_parent_ids.len()
                + (date.get() - block.real_date_start.get()) as usize;
            let reduced_component =
                full_component
                    .checked_sub(1)
                    .ok_or(SequentialReplayError::Invalid(
                        "non-gauge date selected the gauge component",
                    ))?;
            let phase = adjoints.phase[block_index]
                .entry(output)
                .or_insert_with(|| Array2::zeros((block.phase_dimension, selected)));
            phase[(reduced_component, column)] += 1.0;
        }

        let mut covariance = Array2::zeros((selected, selected));
        for block_index in (0..self.blocks.len()).rev() {
            let block = &self.blocks[block_index];
            let mut source_adjoints: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
            for &native in &cone.required_compressed[block_index] {
                let compressed = adjoints.compressed[block_index]
                    .remove(&native)
                    .unwrap_or_else(|| Array2::zeros((2, selected)));
                self.propagate_compression_adjoint(
                    block,
                    native,
                    compressed.view(),
                    query.source_rank,
                    branch_tolerance,
                    provider,
                    &mut adjoints.phase[block_index],
                    &mut source_adjoints,
                )?;
            }
            for &output in &cone.active_outputs[block_index] {
                let phase = adjoints.phase[block_index]
                    .remove(&output)
                    .unwrap_or_else(|| Array2::zeros((block.phase_dimension, selected)));
                self.propagate_phase_adjoint(
                    block,
                    output,
                    phase.view(),
                    query.source_rank,
                    branch_tolerance,
                    provider,
                    &mut adjoints.compressed,
                    &mut source_adjoints,
                )?;
            }
            for root in source_adjoints.values() {
                covariance += &root.t().dot(root);
            }
        }
        if covariance.iter().any(|value| !value.is_finite()) {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::NonFiniteReplayState,
                "streamed covariance contraction is non-finite",
            ));
        }
        Ok(TemporalCovarianceReplay {
            covariance,
            dependency_cone,
        })
    }

    fn validate_provider_identity(
        &self,
        identity: &SequentialSourceProviderIdentity,
    ) -> Result<(), SequentialReplayError> {
        let namespace = self
            .id_namespace
            .as_ref()
            .ok_or(SequentialReplayError::Unsupported(
                ReplayStatus::UnsupportedSourceIdentity,
            ))?;
        if identity.source_model_hash.iter().all(|byte| *byte == 0) {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceModelUnavailable,
                "source provider has no proper-complex model digest",
            ));
        }
        if identity.source_manifest_digest != namespace.source_manifest_digest
            || identity.source_model_version_digest != namespace.source_model_version_digest
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceIdentityMismatch,
                "source provider identity does not match the replay namespace",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn propagate_compression_adjoint<P>(
        &self,
        block: &SequentialReplayBlock,
        native_index: usize,
        child_adjoint: ArrayView2<f64>,
        source_rank: usize,
        branch_tolerance: f64,
        provider: &mut P,
        phase_adjoints: &mut BTreeMap<usize, Array2<f64>>,
        source_adjoints: &mut BTreeMap<usize, Array2<f64>>,
    ) -> Result<(), SequentialReplayError>
    where
        P: SequentialSourceReplayProvider + ?Sized,
    {
        let output_index = self.nearest_output_index(native_index)?;
        let phase = self.resolve_phase_checked(block, output_index, branch_tolerance, provider)?;
        let compression = self.resolve_compression_checked(block, native_index, provider)?;
        let source = self.resolve_source_checked(block, native_index, source_rank, provider)?;
        let first_real = block.carried_parent_ids.len();
        let linked_phase = Array1::from_iter(phase.linked_phase.iter().skip(first_real).copied());
        let zero_samples = Array1::zeros(block.num_real_dates);
        let zero_phase = Array1::zeros(block.num_real_dates);
        let baseline = compress_pixel_jvp(
            source.samples.view(),
            linked_phase.view(),
            zero_samples.view(),
            zero_phase.view(),
            branch_tolerance,
        )
        .map_err(compression_jvp_error)?;
        if !complex_close(baseline.value, compression.value, branch_tolerance)
            || !complex_close(
                baseline.projection,
                compression.projection,
                branch_tolerance,
            )
            || !scalar_close(
                baseline.mean_amplitude,
                compression.mean_amplitude,
                branch_tolerance,
            )
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::ReplayStateMismatch,
                "recomputed compression state differs from the captured block",
            ));
        }

        let selected = child_adjoint.ncols();
        let embedding = source.factor.real_embedding();
        let root = source_adjoints
            .entry(native_index)
            .or_insert_with(|| Array2::zeros((source_rank, selected)));
        for basis in 0..embedding.ncols() {
            let direction = complex_factor_direction(&embedding, block.num_real_dates, basis);
            let jvp = compress_pixel_jvp(
                source.samples.view(),
                linked_phase.view(),
                direction.view(),
                zero_phase.view(),
                branch_tolerance,
            )
            .map_err(compression_jvp_error)?;
            let output_direction = Array1::from_vec(vec![jvp.direction.re, jvp.direction.im]);
            accumulate_basis(root, basis, output_direction.view(), child_adjoint);
        }

        let phase_root = phase_adjoints
            .entry(output_index)
            .or_insert_with(|| Array2::zeros((block.phase_dimension, selected)));
        for reduced_component in 0..block.phase_dimension {
            let full_component = reduced_component + 1;
            if full_component < first_real {
                continue;
            }
            let real_component = full_component - first_real;
            let mut direction = Array1::zeros(block.num_real_dates);
            direction[real_component] = 1.0;
            let jvp = compress_pixel_jvp(
                source.samples.view(),
                linked_phase.view(),
                zero_samples.view(),
                direction.view(),
                branch_tolerance,
            )
            .map_err(compression_jvp_error)?;
            let output_direction = Array1::from_vec(vec![jvp.direction.re, jvp.direction.im]);
            accumulate_basis(
                phase_root,
                reduced_component,
                output_direction.view(),
                child_adjoint,
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn propagate_phase_adjoint<P>(
        &self,
        block: &SequentialReplayBlock,
        output_index: usize,
        child_adjoint: ArrayView2<f64>,
        source_rank: usize,
        branch_tolerance: f64,
        provider: &mut P,
        compressed_adjoints: &mut [BTreeMap<usize, Array2<f64>>],
        source_adjoints: &mut BTreeMap<usize, Array2<f64>>,
    ) -> Result<(), SequentialReplayError>
    where
        P: SequentialSourceReplayProvider + ?Sized,
    {
        let window =
            self.load_phase_window(block, output_index, branch_tolerance, source_rank, provider)?;
        let selected = child_adjoint.ncols();
        let first_real = block.carried_parent_ids.len();
        for (support_index, &source_pixel) in window.source_pixels.iter().enumerate() {
            let native_index = source_pixel.row * self.native_shape.1 + source_pixel.column;
            let source = self.resolve_source_checked(block, native_index, source_rank, provider)?;
            let embedding = source.factor.real_embedding();
            let root = source_adjoints
                .entry(native_index)
                .or_insert_with(|| Array2::zeros((source_rank, selected)));
            for basis in 0..embedding.ncols() {
                let raw_direction =
                    complex_factor_direction(&embedding, block.num_real_dates, basis);
                let mut combined_direction = Array1::zeros(window.source_values.nrows());
                for (date, value) in raw_direction.iter().copied().enumerate() {
                    combined_direction[first_real + date] = value;
                }
                let coherence_direction = rect_source_values_coherence_jvp(
                    window.source_values.view(),
                    &window.replay,
                    source_pixel,
                    combined_direction.view(),
                    branch_tolerance,
                )
                .map_err(covariance_replay_error)?;
                let phase_direction = phase_angle_jvp(
                    window.replay.coherence.view(),
                    coherence_direction.view(),
                    self.estimator_branch,
                    0,
                    branch_tolerance,
                )
                .map_err(estimator_jvp_error)?;
                let reduced = Array1::from_iter(phase_direction.iter().skip(1).copied());
                accumulate_basis(root, basis, reduced.view(), child_adjoint);
            }

            for (parent_component, &parent_id) in block.carried_parent_ids.iter().enumerate() {
                let parent = self.block(parent_id)?;
                let parent_index = parent.generation as usize;
                let parent_root = compressed_adjoints[parent_index]
                    .entry(native_index)
                    .or_insert_with(|| Array2::zeros((2, selected)));
                for basis in 0..2 {
                    let mut direction = Array1::zeros(window.source_values.nrows());
                    direction[parent_component] = match basis {
                        0 => Cf64::new(1.0, 0.0),
                        _ => Cf64::new(0.0, 1.0),
                    };
                    let coherence_direction = rect_source_values_coherence_jvp(
                        window.source_values.view(),
                        &window.replay,
                        window.source_pixels[support_index],
                        direction.view(),
                        branch_tolerance,
                    )
                    .map_err(covariance_replay_error)?;
                    let phase_direction = phase_angle_jvp(
                        window.replay.coherence.view(),
                        coherence_direction.view(),
                        self.estimator_branch,
                        0,
                        branch_tolerance,
                    )
                    .map_err(estimator_jvp_error)?;
                    let reduced = Array1::from_iter(phase_direction.iter().skip(1).copied());
                    accumulate_basis(parent_root, basis, reduced.view(), child_adjoint);
                }
            }
        }
        Ok(())
    }

    fn load_phase_window<P>(
        &self,
        block: &SequentialReplayBlock,
        output_index: usize,
        branch_tolerance: f64,
        source_rank: usize,
        provider: &mut P,
    ) -> Result<PhaseWindowReplay, SequentialReplayError>
    where
        P: SequentialSourceReplayProvider + ?Sized,
    {
        let phase = self.resolve_phase_checked(block, output_index, branch_tolerance, provider)?;
        let native_indices = self.native_support_indices(output_index)?;
        let source_pixels = native_indices
            .iter()
            .map(|&native| {
                NativeSourcePixel::new(native / self.native_shape.1, native % self.native_shape.1)
            })
            .collect::<Vec<_>>();
        let combined = block.carried_parent_ids.len() + block.num_real_dates;
        let mut source_values = Array2::zeros((combined, native_indices.len()));
        for (component, &parent) in block.carried_parent_ids.iter().enumerate() {
            let parent = self.block(parent)?;
            for (support_index, &native_index) in native_indices.iter().enumerate() {
                source_values[(component, support_index)] = self
                    .resolve_compression_checked(parent, native_index, provider)?
                    .value;
            }
        }
        for (support_index, &native_index) in native_indices.iter().enumerate() {
            let source = self.resolve_source_checked(block, native_index, source_rank, provider)?;
            for (date, value) in source.samples.iter().copied().enumerate() {
                source_values[(block.carried_parent_ids.len() + date, support_index)] = value;
            }
        }
        let descriptor =
            RectReplayDescriptor::new(self.native_shape, self.half_window, self.strides)
                .map_err(covariance_replay_error)?;
        let output = (
            output_index / self.output_shape.1,
            output_index % self.output_shape.1,
        );
        let replay =
            replay_rect_source_values(descriptor, output, &source_pixels, source_values.view())
                .map_err(covariance_replay_error)?;
        self.validate_phase_baseline(&phase, &replay, branch_tolerance)?;
        Ok(PhaseWindowReplay {
            source_pixels,
            source_values,
            replay,
        })
    }

    fn validate_phase_baseline(
        &self,
        phase: &ResolvedPhaseReplay,
        replay: &RectPixelReplay,
        branch_tolerance: f64,
    ) -> Result<(), SequentialReplayError> {
        let (use_evd, beta, zero_correlation_threshold, estimator) = match self.estimator_branch {
            FixedEstimatorBranch::Evd => (true, 0.0, 0.0, 0),
            FixedEstimatorBranch::Emi {
                beta,
                zero_correlation_threshold,
            } => (false, beta, zero_correlation_threshold, 1),
        };
        let baseline = process_coherence_matrix(
            replay.coherence.view(),
            use_evd,
            beta,
            zero_correlation_threshold,
            0,
        );
        let phase_matches = baseline.phase.len() == phase.linked_phase.len()
            && baseline
                .phase
                .iter()
                .zip(phase.linked_phase.iter())
                .all(|(&left, &right)| {
                    left.norm() > branch_tolerance
                        && complex_close(left / left.norm(), right, branch_tolerance)
                });
        if baseline.estimator != estimator
            || !phase_matches
            || !scalar_close(
                baseline.eigenvalue,
                phase.selected_eigenvalue,
                branch_tolerance,
            )
            || !scalar_close(baseline.eigengap, phase.selected_eigengap, branch_tolerance)
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::ReplayStateMismatch,
                "recomputed phase estimator differs from the captured block",
            ));
        }
        Ok(())
    }

    fn resolve_source_checked<P>(
        &self,
        block: &SequentialReplayBlock,
        native_index: usize,
        source_rank: usize,
        provider: &mut P,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError>
    where
        P: SequentialSourceReplayProvider + ?Sized,
    {
        let source = provider.resolve_source(block, native_index)?;
        let expected_id = self.source_id(block.id, native_index)?;
        let date_count = u32::try_from(block.num_real_dates).map_err(|_| {
            SequentialReplayError::Invalid("replay block source-date count exceeds u32")
        })?;
        let date_stop = block.real_date_start.get().checked_add(date_count).ok_or(
            SequentialReplayError::Invalid("replay block source-date range overflows u32"),
        )?;
        let expected_components = (block.real_date_start.get()..date_stop)
            .map(u64::from)
            .collect::<Vec<_>>();
        let expected_rank = 2 * block.num_real_dates;
        let identity = provider.identity();
        if source.id != expected_id
            || source.factor.source() != expected_id
            || source.samples.len() != block.num_real_dates
            || source.factor.component_ids() != expected_components
            || source.factor.model_hash() != &identity.source_model_hash
            || source.factor.real_embedding().dim() != (expected_rank, expected_rank)
            || expected_rank > source_rank
            || source.content_digest.iter().all(|byte| *byte == 0)
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceIdentityMismatch,
                "resolved source or proper-complex factor identity does not match the block",
            ));
        }
        if source.samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::NonFiniteReplayState,
                "resolved source contains a non-finite sample",
            ));
        }
        Ok(source)
    }

    fn resolve_phase_checked<P>(
        &self,
        block: &SequentialReplayBlock,
        output_index: usize,
        branch_tolerance: f64,
        provider: &mut P,
    ) -> Result<ResolvedPhaseReplay, SequentialReplayError>
    where
        P: SequentialSourceReplayProvider + ?Sized,
    {
        let phase = provider.resolve_phase(block, output_index)?;
        ensure_operator_status(phase.status, "required phase node is not replayable")?;
        if phase.id != self.phase_node_id(block.id, output_index)?
            || phase.linked_phase.len() != block.carried_parent_ids.len() + block.num_real_dates
            || phase.estimator_branch
                != match self.estimator_branch {
                    FixedEstimatorBranch::Evd => CovarianceEstimatorBranch::Evd,
                    FixedEstimatorBranch::Emi { .. } => CovarianceEstimatorBranch::Emi,
                }
            || phase.branch_tolerance != branch_tolerance
            || phase.linked_phase.iter().any(|value| !value.is_finite())
            || !phase.selected_eigenvalue.is_finite()
            || !phase.selected_eigengap.is_finite()
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::ReplayStateMismatch,
                "captured phase node does not match its topology or fixed branch",
            ));
        }
        Ok(phase)
    }

    fn resolve_compression_checked<P>(
        &self,
        block: &SequentialReplayBlock,
        native_index: usize,
        provider: &mut P,
    ) -> Result<ResolvedCompressionReplay, SequentialReplayError>
    where
        P: SequentialSourceReplayProvider + ?Sized,
    {
        let compression = provider.resolve_compression(block, native_index)?;
        ensure_operator_status(
            compression.status,
            "required compressed node is not replayable",
        )?;
        if compression.id != self.compressed_node_id(block.id, native_index)?
            || !compression.value.is_finite()
            || !compression.projection.is_finite()
            || !compression.mean_amplitude.is_finite()
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::ReplayStateMismatch,
                "captured compressed node does not match its topology",
            ));
        }
        Ok(compression)
    }

    fn validate_block_local(
        &self,
        block: GlobalBlockId,
        local_index: usize,
        local_len: usize,
    ) -> Result<(), SequentialReplayError> {
        self.block(block)?;
        if local_index >= local_len {
            return Err(SequentialReplayError::Invalid(
                "replay node local index is outside its grid",
            ));
        }
        Ok(())
    }

    fn validate_date_output(
        &self,
        date: GlobalDateId,
        output_index: usize,
    ) -> Result<(), SequentialReplayError> {
        if date.get() as usize >= self.num_real_dates
            || output_index >= self.output_area
            || !self.is_owned_output(output_index)
        {
            return Err(SequentialReplayError::Invalid(
                "replay date or output index is outside its owned grid",
            ));
        }
        Ok(())
    }

    fn is_owned_output(&self, output_index: usize) -> bool {
        let Some(namespace) = self.id_namespace.as_ref() else {
            return true;
        };
        let row = output_index / self.output_shape.1;
        let column = output_index % self.output_shape.1;
        let owned_row_start =
            (namespace.owned_output_origin.0 - namespace.output_origin.0) as usize;
        let owned_col_start =
            (namespace.owned_output_origin.1 - namespace.output_origin.1) as usize;
        (owned_row_start..owned_row_start + namespace.owned_output_shape.0).contains(&row)
            && (owned_col_start..owned_col_start + namespace.owned_output_shape.1).contains(&column)
    }

    fn block(&self, id: GlobalBlockId) -> Result<&SequentialReplayBlock, SequentialReplayError> {
        self.blocks
            .iter()
            .find(|block| block.id == id)
            .ok_or(SequentialReplayError::Invalid(
                "replay block identifier does not exist",
            ))
    }

    fn block_for_date(
        &self,
        date: GlobalDateId,
    ) -> Result<&SequentialReplayBlock, SequentialReplayError> {
        self.blocks
            .iter()
            .find(|block| {
                let start = block.real_date_start.get() as usize;
                let date = date.get() as usize;
                (start..start + block.num_real_dates).contains(&date)
            })
            .ok_or(SequentialReplayError::Invalid(
                "replay date is not assigned to a block",
            ))
    }

    /// Row-major valid native indices in one output pixel's fixed Rect support.
    ///
    /// # Errors
    /// Returns `Err` for an output index outside the topology grid.
    pub fn native_support_indices(
        &self,
        output_index: usize,
    ) -> Result<Vec<usize>, SequentialReplayError> {
        if output_index >= self.output_area {
            return Err(SequentialReplayError::Invalid(
                "replay output index is outside its grid",
            ));
        }
        let (row_start, col_start) = self.window_origin(output_index);
        let window_rows = 2 * self.half_window.y + 1;
        let window_cols = 2 * self.half_window.x + 1;
        Ok((row_start..row_start + window_rows)
            .flat_map(|row| {
                (col_start..col_start + window_cols).filter_map(move |col| {
                    let index = row * self.native_shape.1 + col;
                    self.native_validity[index].then_some(index)
                })
            })
            .collect())
    }

    fn support_slots_per_output(&self) -> usize {
        (2 * self.half_window.y + 1) * (2 * self.half_window.x + 1)
    }

    fn support_slot_validity(
        &self,
        output_index: usize,
    ) -> Result<Vec<bool>, SequentialReplayError> {
        if output_index >= self.output_area {
            return Err(SequentialReplayError::Invalid(
                "replay output index is outside its grid",
            ));
        }
        let (row_start, col_start) = self.window_origin(output_index);
        let window_rows = 2 * self.half_window.y + 1;
        let window_cols = 2 * self.half_window.x + 1;
        Ok((row_start..row_start + window_rows)
            .flat_map(|row| {
                (col_start..col_start + window_cols)
                    .map(move |col| self.native_validity[row * self.native_shape.1 + col])
            })
            .collect())
    }

    fn window_origin(&self, output_index: usize) -> (usize, usize) {
        let output_row = output_index / self.output_shape.1;
        let output_col = output_index % self.output_shape.1;
        let window_rows = 2 * self.half_window.y + 1;
        let window_cols = 2 * self.half_window.x + 1;
        let center_row = self.strides.y / 2 + output_row * self.strides.y;
        let center_col = self.strides.x / 2 + output_col * self.strides.x;
        (
            center_row
                .saturating_sub(self.half_window.y)
                .min(self.native_shape.0 - window_rows),
            center_col
                .saturating_sub(self.half_window.x)
                .min(self.native_shape.1 - window_cols),
        )
    }

    /// Output index repeated onto one native compression pixel.
    ///
    /// # Errors
    /// Returns `Err` for a native index outside the topology grid.
    pub fn nearest_output_index(
        &self,
        native_index: usize,
    ) -> Result<usize, SequentialReplayError> {
        if native_index >= self.native_area {
            return Err(SequentialReplayError::Invalid(
                "replay native index is outside its grid",
            ));
        }
        let native_row = native_index / self.native_shape.1;
        let native_col = native_index % self.native_shape.1;
        let row_looks = (self.native_shape.0 / self.output_shape.0).max(1);
        let col_looks = (self.native_shape.1 / self.output_shape.1).max(1);
        let output_row = (native_row / row_looks).min(self.output_shape.0 - 1);
        let output_col = (native_col / col_looks).min(self.output_shape.1 - 1);
        Ok(output_row * self.output_shape.1 + output_col)
    }

    fn identified_id(
        &self,
        kind: &[u8],
        major: u64,
        secondary: u64,
        local: usize,
        native: bool,
    ) -> u64 {
        let namespace = self
            .id_namespace
            .as_ref()
            .expect("identified IDs require a validated namespace");
        let shape = match native {
            true => self.native_shape,
            false => self.output_shape,
        };
        let origin = match native {
            true => namespace.native_origin,
            false => namespace.output_origin,
        };
        let row = origin.0 + (local / shape.1) as u64;
        let column = origin.1 + (local % shape.1) as u64;
        let mut digest = Sha256::new();
        digest.update(SEQUENTIAL_SOURCE_DAG_METHOD.as_bytes());
        digest.update(kind);
        digest.update(namespace.burst_id.as_bytes());
        digest.update(namespace.source_manifest_digest);
        digest.update(namespace.source_model_version_digest);
        digest.update(major.to_le_bytes());
        digest.update(secondary.to_le_bytes());
        digest.update(row.to_le_bytes());
        digest.update(column.to_le_bytes());
        u64::from_le_bytes(digest.finalize()[..8].try_into().expect("SHA-256 prefix"))
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn build_covariance_operator_block(
    topology: &SequentialReplayTopology,
    request: &SequentialCovarianceCaptureRequest,
    ministack: MiniStack,
    linked_phase: ArrayView3<dolphin_core::Cf64>,
    phase_replay: &PhaseReplayGrid,
    compression_replay: &CompressionReplayGrid,
    use_evd: bool,
) -> Result<CovarianceOperatorBlock, SequentialReplayError> {
    let block = topology
        .blocks
        .get(ministack.block_id)
        .ok_or(SequentialReplayError::Invalid(
            "captured ministack has no replay topology block",
        ))?;
    if block.generation as usize != ministack.block_id
        || linked_phase.dim().0 != ministack.size()
        || linked_phase.dim().1 * linked_phase.dim().2 != topology.output_area
        || phase_replay.branch_status.len() != topology.output_area
        || compression_replay.compressed.len() != topology.native_area
    {
        return Err(SequentialReplayError::Invalid(
            "captured replay state does not match its block topology",
        ));
    }

    let source_date_indices: Vec<u32> = (ministack.real_start
        ..ministack.real_start + ministack.num_real)
        .map(|date| {
            u32::try_from(date).map_err(|_| {
                SequentialReplayError::Invalid("captured source date index exceeds u32")
            })
        })
        .collect::<Result<_, _>>()?;
    let mut phase_components = block
        .carried_parent_ids
        .iter()
        .map(|parent| CovariancePhaseComponent {
            kind: CovariancePhaseComponentKind::CompressedParent,
            id: parent.get(),
        })
        .collect::<Vec<_>>();
    phase_components.extend(
        source_date_indices
            .iter()
            .map(|&date| CovariancePhaseComponent {
                kind: match date {
                    0 => CovariancePhaseComponentKind::GaugeDate,
                    _ => CovariancePhaseComponentKind::RetainedDate,
                },
                id: u64::from(date),
            }),
    );
    if phase_components.len() != ministack.size() {
        return Err(SequentialReplayError::Invalid(
            "captured phase component map differs from the combined ministack",
        ));
    }

    let source_ids = (0..topology.native_area)
        .map(|native| topology.source_id(block.id, native).map(SourceId::get))
        .collect::<Result<Vec<_>, _>>()?;
    let compressed_node_ids = (0..topology.native_area)
        .map(|native| {
            topology
                .compressed_node_id(block.id, native)
                .map(NodeId::get)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let phase_node_ids = (0..topology.output_area)
        .map(|output| topology.phase_node_id(block.id, output).map(NodeId::get))
        .collect::<Result<Vec<_>, _>>()?;
    let nearest_output_map = (0..topology.native_area)
        .map(|native| {
            topology.nearest_output_index(native).and_then(|output| {
                u32::try_from(output).map_err(|_| {
                    SequentialReplayError::Invalid("nearest replay output index exceeds u32")
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let bytes_per_support = topology.support_slots_per_output().div_ceil(8);
    let mut support_bits = Vec::with_capacity(topology.output_area * bytes_per_support);
    for output in 0..topology.output_area {
        support_bits.extend(pack_bits(&topology.support_slot_validity(output)?));
    }
    let native_validity_bits = pack_bits(&topology.native_validity);

    let phase_angles = (0..topology.output_area)
        .flat_map(|output| {
            let row = output / topology.output_shape.1;
            let column = output % topology.output_shape.1;
            (0..ministack.size()).map(move |component| linked_phase[(component, row, column)].arg())
        })
        .collect();

    let half_window_rows = u32::try_from(topology.half_window.y).map_err(|_| {
        SequentialReplayError::Invalid("captured Rect half-window row count exceeds u32")
    })?;
    let half_window_cols = u32::try_from(topology.half_window.x).map_err(|_| {
        SequentialReplayError::Invalid("captured Rect half-window column count exceeds u32")
    })?;
    let support_bits_per_output = u32::try_from(topology.support_slots_per_output())
        .map_err(|_| SequentialReplayError::Invalid("captured Rect support count exceeds u32"))?;
    Ok(CovarianceOperatorBlock {
        burst_id: request.burst_id.clone(),
        block_id: block.id.get(),
        generation: block.generation,
        native_grid: request.native_grid,
        output_grid: request.output_grid,
        owned_output_grid: request.owned_output_grid,
        rect_support: CovarianceRectSupport {
            half_window_rows,
            half_window_cols,
            ordering: CovarianceSupportOrdering::RowMajorInwardClampV1,
        },
        branch_tolerance: request.branch_tolerance,
        reference_date_index: 0,
        source_date_indices: source_date_indices.clone(),
        ordered_date_indices: source_date_indices,
        source_ids,
        phase_node_ids,
        compressed_node_ids,
        carry_parent_ids: block
            .carried_parent_ids
            .iter()
            .map(|parent| parent.get())
            .collect(),
        nearest_output_map,
        phase_components,
        phase_angles,
        compressed_raster: compression_replay.compressed.iter().copied().collect(),
        compressed_status: compression_replay
            .status
            .iter()
            .copied()
            .map(compression_status)
            .collect(),
        projection_accumulator: compression_replay.projection.iter().copied().collect(),
        mean_amplitude: compression_replay.mean_amplitude.iter().copied().collect(),
        support_bits_per_output,
        support_bits,
        native_validity_bits,
        estimator_branch: match use_evd {
            true => CovarianceEstimatorBranch::Evd,
            false => CovarianceEstimatorBranch::Emi,
        },
        selected_eigenvalue: phase_replay.selected_eigenvalue.iter().copied().collect(),
        eigen_gap: phase_replay.selected_eigengap.iter().copied().collect(),
        status: phase_replay
            .branch_status
            .iter()
            .copied()
            .map(phase_status)
            .collect(),
    })
}

fn ensure_operator_status(
    status: CovarianceOperatorStatus,
    message: &'static str,
) -> Result<(), SequentialReplayError> {
    let replay_status = match status {
        CovarianceOperatorStatus::Valid => return Ok(()),
        CovarianceOperatorStatus::Masked | CovarianceOperatorStatus::NoContributor => {
            ReplayStatus::MaskedNode
        }
        CovarianceOperatorStatus::SingularLocalInformation => {
            ReplayStatus::SingularLocalInformation
        }
        CovarianceOperatorStatus::NonfiniteState | CovarianceOperatorStatus::NonfiniteJacobian => {
            ReplayStatus::NonFiniteReplayState
        }
        CovarianceOperatorStatus::Nondifferentiable => ReplayStatus::NondifferentiableNode,
        CovarianceOperatorStatus::InvalidCompression => ReplayStatus::InvalidCompression,
    };
    Err(SequentialReplayError::Provider(replay_status, message))
}

fn covariance_replay_error(error: CovarianceReplayError) -> SequentialReplayError {
    let status = match error {
        CovarianceReplayError::NonFiniteSource
        | CovarianceReplayError::NonFiniteDirection
        | CovarianceReplayError::NonFiniteDerivative => ReplayStatus::NonFiniteReplayState,
        CovarianceReplayError::AmplitudeFloorBoundary => ReplayStatus::NondifferentiableNode,
        _ => ReplayStatus::ReplayStateMismatch,
    };
    SequentialReplayError::Provider(status, "rectangular covariance replay failed")
}

fn estimator_jvp_error(error: EstimatorJvpError) -> SequentialReplayError {
    let status = match error {
        EstimatorJvpError::EigenvalueTie => ReplayStatus::SingularLocalInformation,
        EstimatorJvpError::NonFiniteState | EstimatorJvpError::NonFiniteDerivative => {
            ReplayStatus::NonFiniteReplayState
        }
        EstimatorJvpError::EmiFallback => ReplayStatus::UnsupportedEstimatorFallback,
        EstimatorJvpError::ZeroMagnitudeBranch
        | EstimatorJvpError::ThresholdBoundary
        | EstimatorJvpError::VanishingReference => ReplayStatus::NondifferentiableNode,
        EstimatorJvpError::MatrixShapeMismatch | EstimatorJvpError::ReferenceOutOfBounds => {
            ReplayStatus::ReplayStateMismatch
        }
    };
    SequentialReplayError::Provider(status, "fixed estimator replay failed")
}

fn compression_jvp_error(error: CompressionJvpError) -> SequentialReplayError {
    let status = match error {
        CompressionJvpError::NonFiniteState | CompressionJvpError::NonFiniteDerivative => {
            ReplayStatus::NonFiniteReplayState
        }
        CompressionJvpError::ZeroIncludedAmplitude
        | CompressionJvpError::ZeroProjection
        | CompressionJvpError::NodataBranch => ReplayStatus::InvalidCompression,
        CompressionJvpError::ShapeMismatch => ReplayStatus::ReplayStateMismatch,
    };
    SequentialReplayError::Provider(status, "fixed compression replay failed")
}

fn complex_factor_direction(
    embedding: &Array2<f64>,
    complex_dimension: usize,
    basis: usize,
) -> Array1<Cf64> {
    Array1::from_shape_fn(complex_dimension, |component| {
        Cf64::new(
            embedding[(component, basis)],
            embedding[(complex_dimension + component, basis)],
        )
    })
}

fn accumulate_basis(
    parent: &mut Array2<f64>,
    parent_component: usize,
    child_direction: ArrayView1<f64>,
    child_adjoint: ArrayView2<f64>,
) {
    for selected in 0..child_adjoint.ncols() {
        parent[(parent_component, selected)] += child_direction
            .iter()
            .zip(child_adjoint.column(selected).iter())
            .map(|(direction, adjoint)| direction * adjoint)
            .sum::<f64>();
    }
}

fn scalar_close(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance * (1.0 + left.abs().max(right.abs()))
}

fn complex_close(left: Cf64, right: Cf64, tolerance: f64) -> bool {
    (left - right).norm() <= tolerance * (1.0 + left.norm().max(right.norm()))
}

fn digest_hex(digest: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn digest_matches(encoded: &str, digest: [u8; 32]) -> bool {
    let expected = digest_hex(digest);
    encoded == expected || encoded.strip_prefix("sha256:") == Some(expected.as_str())
}

fn phase_status(status: FixedBranchStatus) -> CovarianceOperatorStatus {
    match status {
        FixedBranchStatus::Evd | FixedBranchStatus::Emi => CovarianceOperatorStatus::Valid,
        FixedBranchStatus::Masked => CovarianceOperatorStatus::Masked,
        FixedBranchStatus::NonFiniteState => CovarianceOperatorStatus::NonfiniteState,
        FixedBranchStatus::EigenvalueTie => CovarianceOperatorStatus::SingularLocalInformation,
        FixedBranchStatus::UnsupportedEmiFallback
        | FixedBranchStatus::Nondifferentiable
        | FixedBranchStatus::AmplitudeFloorBoundary
        | FixedBranchStatus::VanishingReference
        | FixedBranchStatus::InvalidEstimator => CovarianceOperatorStatus::Nondifferentiable,
    }
}

fn compression_status(status: CompressionReplayStatus) -> CovarianceOperatorStatus {
    match status {
        CompressionReplayStatus::Valid => CovarianceOperatorStatus::Valid,
        CompressionReplayStatus::Masked => CovarianceOperatorStatus::Masked,
        CompressionReplayStatus::NonFiniteState => CovarianceOperatorStatus::NonfiniteState,
        CompressionReplayStatus::ZeroIncludedAmplitude
        | CompressionReplayStatus::ZeroProjection
        | CompressionReplayStatus::NodataBranch => CovarianceOperatorStatus::InvalidCompression,
    }
}

fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut packed = vec![0_u8; bits.len().div_ceil(8)];
    for (index, &value) in bits.iter().enumerate() {
        if value {
            packed[index / 8] |= 1 << (index % 8);
        }
    }
    packed
}

fn assess_support(
    cfg: &SequentialConfig,
    scope: ReplayExecutionScope,
) -> Result<(), SequentialReplayError> {
    let status = if !scope.enabled {
        ReplayStatus::Disabled
    } else if cfg.compressed_slc_plan != CompressedSlcPlan::AlwaysFirst {
        ReplayStatus::UnsupportedReferencePlan
    } else if cfg.output_reference_idx != 0 {
        ReplayStatus::UnsupportedOutputReference
    } else if scope.backend != ReplayBackend::CpuF64 {
        ReplayStatus::UnsupportedBackend
    } else if cfg.shp_method != ShpMethod::Rect {
        ReplayStatus::UnsupportedShpMethod
    } else if scope.estimator_fallback {
        ReplayStatus::UnsupportedEstimatorFallback
    } else if scope.phase_bias_correction {
        ReplayStatus::UnsupportedPhaseBiasCorrection
    } else if !scope.strong_source_identity {
        ReplayStatus::UnsupportedSourceIdentity
    } else if scope.stitched_burst_count != 1 {
        ReplayStatus::UnsupportedSeamCovariance
    } else {
        ReplayStatus::Valid
    };
    match status {
        ReplayStatus::Valid => Ok(()),
        _ => Err(SequentialReplayError::Unsupported(status)),
    }
}

fn checked_area(shape: (usize, usize)) -> Result<usize, SequentialReplayError> {
    if shape.0 == 0 || shape.1 == 0 {
        return Err(SequentialReplayError::Invalid(
            "replay grids must have positive dimensions",
        ));
    }
    shape
        .0
        .checked_mul(shape.1)
        .ok_or(SequentialReplayError::Invalid(
            "replay grid area overflows usize",
        ))
}

fn pack_node_id(kind: u64, major: u32, local: u32) -> NodeId {
    NodeId::new((kind << NODE_KIND_SHIFT) | (u64::from(major) << 32) | u64::from(local))
}

fn record_block_id(
    namespace: &ReplayIdNamespace,
    generation: u32,
    native_shape: (usize, usize),
    output_shape: (usize, usize),
) -> u64 {
    let mut digest = Sha256::new();
    digest.update(SEQUENTIAL_SOURCE_DAG_METHOD.as_bytes());
    digest.update(b"record_block");
    digest.update(namespace.burst_id.as_bytes());
    digest.update(namespace.source_manifest_digest);
    digest.update(namespace.source_model_version_digest);
    digest.update(namespace.native_origin.0.to_le_bytes());
    digest.update(namespace.native_origin.1.to_le_bytes());
    digest.update((native_shape.0 as u64).to_le_bytes());
    digest.update((native_shape.1 as u64).to_le_bytes());
    digest.update(namespace.output_origin.0.to_le_bytes());
    digest.update(namespace.output_origin.1.to_le_bytes());
    digest.update((output_shape.0 as u64).to_le_bytes());
    digest.update((output_shape.1 as u64).to_le_bytes());
    digest.update(namespace.owned_output_origin.0.to_le_bytes());
    digest.update(namespace.owned_output_origin.1.to_le_bytes());
    digest.update((namespace.owned_output_shape.0 as u64).to_le_bytes());
    digest.update((namespace.owned_output_shape.1 as u64).to_le_bytes());
    let tile_hash = u64::from_le_bytes(digest.finalize()[..8].try_into().expect("SHA-256 prefix"))
        & ((1_u64 << 48) - 1);
    (u64::from(generation) << 48) | tile_hash
}

fn grid_contains(outer: CovarianceOperatorGrid, inner: CovarianceOperatorGrid) -> bool {
    let outer_row_stop = outer.row_start.checked_add(u64::from(outer.rows));
    let outer_col_stop = outer.col_start.checked_add(u64::from(outer.cols));
    let inner_row_stop = inner.row_start.checked_add(u64::from(inner.rows));
    let inner_col_stop = inner.col_start.checked_add(u64::from(inner.cols));
    match (
        outer_row_stop,
        outer_col_stop,
        inner_row_stop,
        inner_col_stop,
    ) {
        (Some(outer_row), Some(outer_col), Some(inner_row), Some(inner_col)) => {
            inner.rows > 0
                && inner.cols > 0
                && inner.row_start >= outer.row_start
                && inner.col_start >= outer.col_start
                && inner_row <= outer_row
                && inner_col <= outer_col
        }
        _ => false,
    }
}

fn checked_add(left: u64, right: u64) -> Result<u64, SequentialReplayError> {
    left.checked_add(right)
        .ok_or(SequentialReplayError::Invalid(
            "dependency-cone byte estimate overflowed u64",
        ))
}

fn checked_mul(left: u64, right: u64) -> Result<u64, SequentialReplayError> {
    left.checked_mul(right)
        .ok_or(SequentialReplayError::Invalid(
            "dependency-cone byte estimate overflowed u64",
        ))
}

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
use std::mem::size_of;
use std::path::Path;
use std::time::{Duration, Instant};

use dolphin_core::config::{CompressedSlcPlan, ShpMethod};
use dolphin_core::{Cf64, HalfWindow, Strides};
use dolphin_io::{
    covariance_content_bound_source_id, covariance_identified_id, covariance_record_block_id,
    CovarianceBurstPlan, CovarianceEstimatorBranch, CovarianceOperatorBlock,
    CovarianceOperatorBlockReader, CovarianceOperatorGrid, CovarianceOperatorPlan,
    CovarianceOperatorStatus, CovariancePhaseComponent, CovariancePhaseComponentKind,
    CovarianceRectSupport, CovarianceReplayStatus, CovarianceSupportOrdering, CovarianceTilePlan,
};
use dolphin_phaselink::{
    compress_pixel_jvp, phase_angle_jvp, phase_angle_jvp_workspace_bytes, process_coherence_matrix,
    rect_source_values_coherence_jvp, replay_rect_source_values, CompressionJvpError,
    CompressionReplayGrid, CompressionReplayStatus, CovarianceReplayError, EstimatorJvpError,
    FixedBranchStatus, FixedEstimatorBranch, InfluenceDag, InfluenceError, NativeSourcePixel,
    NodeId, PhaseReplayGrid, ProperComplexFactor, RectPixelReplay, RectReplayDescriptor, SourceId,
    TemporalCoordinate,
};
use dolphin_stack::{MiniStack, MiniStackPlanner};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use sha2::{Digest, Sha256};

use crate::covariance_artifact::{
    acquire_covariance_artifact_read_lock, read_covariance_artifact_manifest_with_byte_cap,
    CovarianceArtifactReadLock, COVARIANCE_OPERATOR_FILENAME,
};
use crate::sequential::SequentialConfig;

/// Stable method name for the replayable global covariance operator.
pub const SEQUENTIAL_SOURCE_DAG_METHOD: &str = "sequential_source_dag_v1";
/// Versioned identity of the production derivative and contraction kernels.
pub const SEQUENTIAL_SOURCE_DAG_KERNEL_ID: &str = "dolphinrust:sequential_source_dag_v1:kernel_v1";

const NODE_KIND_SHIFT: u32 = 62;
const NODE_MAJOR_LIMIT: u32 = 1 << 30;
// `std` does not expose B-tree node occupancy. Reserve one whole node for each
// logical record using the current standard-library node capacities. Actual
// node count cannot exceed record count, so this remains conservative for a
// sparsely populated tree and independent of allocator reuse.
const BTREE_NODE_HEADER_BYTES: u64 = 64;
const BTREE_NODE_KEY_VALUE_SLOTS: u64 = 11;
const BTREE_NODE_CHILD_POINTERS: u64 = 12;

fn btree_record_reservation_bytes<K, V>() -> u64 {
    BTREE_NODE_HEADER_BYTES
        + BTREE_NODE_KEY_VALUE_SLOTS * (size_of::<K>() + size_of::<V>()) as u64
        + BTREE_NODE_CHILD_POINTERS * size_of::<usize>() as u64
}

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
    /// Resolver/provider name persisted with the artifact.
    pub provider: String,
    /// Resolver/provider version persisted with the artifact.
    pub provider_version: String,
    /// Proper-complex source-model name persisted with the artifact.
    pub model: String,
    /// Proper-complex source-model version persisted with the artifact.
    pub model_version: String,
    /// Digest of the four ordered provider/model identity strings.
    pub source_model_version_digest: [u8; 32],
    /// Digest of the ordered proper-complex factor model.
    pub source_model_hash: [u8; 32],
}

/// Reviewed code/config identity required before an HDF5 operator can replay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SequentialReplayBuildIdentity {
    /// Digest of the normalized producer configuration expected by the caller.
    pub normalized_config_digest: [u8; 32],
    /// Digest of the fixed derivative/replay kernel expected by the caller.
    pub kernel_digest: [u8; 32],
    /// Exact branch tolerance used during capture and replay.
    pub branch_tolerance: f64,
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

/// Bind an exact empirical support/content receipt to its numeric factor receipt.
#[must_use]
pub fn empirical_source_factor_receipt_digest(
    exact_receipt_digest: [u8; 32],
    numeric_factor_digest: [u8; 32],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:available_empirical_source_factor_receipt:v1");
    digest.update(exact_receipt_digest);
    digest.update(numeric_factor_digest);
    digest.finalize().into()
}

fn available_source_factor_receipt_digest(
    exact_receipt_digest: [u8; 32],
    source: &ResolvedPrimitiveSource,
) -> [u8; 32] {
    empirical_source_factor_receipt_digest(
        exact_receipt_digest,
        source.factor.numeric_receipt_digest(),
    )
}

fn masked_source_factor_receipt_digest(
    block: &SequentialReplayBlock,
    source_id: u64,
    content_digest: [u8; 32],
    identity: &SequentialSourceProviderIdentity,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:masked_empirical_source_factor_receipt:v1");
    digest.update(source_id.to_le_bytes());
    digest.update(content_digest);
    digest.update(block.real_date_start.get().to_le_bytes());
    digest.update((block.num_real_dates as u64).to_le_bytes());
    digest.update(identity.source_manifest_digest);
    digest.update(identity.source_model_version_digest);
    digest.update(identity.source_model_hash);
    digest.finalize().into()
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
    /// Exact row-major realized support within the fixed clamped window.
    pub realized_support: Vec<bool>,
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

struct QuerySourceCache<'a, P: ?Sized> {
    provider: &'a mut P,
    sources: BTreeMap<(GlobalBlockId, usize), ResolvedPrimitiveSource>,
    current_payload_bytes: u64,
    peak_payload_bytes: u64,
    source_factor_receipt: Sha256,
    support_receipt: Sha256,
}

impl<'a, P: ?Sized> QuerySourceCache<'a, P> {
    fn new(provider: &'a mut P) -> Self {
        let mut source_factor_receipt = Sha256::new();
        source_factor_receipt.update(b"dolphinrust:query-source-factor-receipt:v1");
        let mut support_receipt = Sha256::new();
        support_receipt.update(b"dolphinrust:query-realized-support-receipt:v1");
        Self {
            provider,
            sources: BTreeMap::new(),
            current_payload_bytes: 0,
            peak_payload_bytes: 0,
            source_factor_receipt,
            support_receipt,
        }
    }

    fn clear_block(&mut self) {
        self.sources.clear();
        self.current_payload_bytes = 0;
    }

    const fn peak_payload_bytes(&self) -> u64 {
        self.peak_payload_bytes
    }

    const fn current_payload_bytes(&self) -> u64 {
        self.current_payload_bytes
    }

    fn source_factor_receipt(&self) -> [u8; 32] {
        self.source_factor_receipt.clone().finalize().into()
    }

    fn support_receipt(&self) -> [u8; 32] {
        self.support_receipt.clone().finalize().into()
    }
}

fn resolved_source_payload_bytes(
    source: &ResolvedPrimitiveSource,
) -> Result<u64, SequentialReplayError> {
    let samples = u64::try_from(source.samples.len())
        .map_err(|_| SequentialReplayError::Invalid("source sample count exceeds u64"))?;
    let factor = u64::try_from(source.factor.lower().len())
        .map_err(|_| SequentialReplayError::Invalid("source factor size exceeds u64"))?;
    let components = u64::try_from(source.factor.component_ids().len())
        .map_err(|_| SequentialReplayError::Invalid("source component count exceeds u64"))?;
    checked_add(
        checked_add(checked_mul(samples, 16)?, checked_mul(factor, 16)?)?,
        checked_add(checked_mul(components, 8)?, 32)?,
    )
}

impl<P> SequentialSourceReplayProvider for QuerySourceCache<'_, P>
where
    P: SequentialSourceReplayProvider + ?Sized,
{
    fn identity(&self) -> &SequentialSourceProviderIdentity {
        self.provider.identity()
    }

    fn maximum_resident_bytes(&self) -> u64 {
        self.provider.maximum_resident_bytes()
    }

    fn resolve_source(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedPrimitiveSource, SequentialReplayError> {
        let key = (block.id, native_index);
        if let Some(source) = self.sources.get(&key) {
            return Ok(source.clone());
        }
        let source = self.provider.resolve_source(block, native_index)?;
        self.source_factor_receipt
            .update(block.generation.to_le_bytes());
        self.source_factor_receipt
            .update((native_index as u64).to_le_bytes());
        self.source_factor_receipt
            .update(source.id.get().to_le_bytes());
        self.source_factor_receipt.update(source.content_digest);
        self.source_factor_receipt
            .update(source.factor.numeric_receipt_digest());
        self.source_factor_receipt
            .update(source.factor.model_hash());
        for &component in source.factor.component_ids() {
            self.source_factor_receipt.update(component.to_le_bytes());
        }
        self.current_payload_bytes = checked_add(
            self.current_payload_bytes,
            resolved_source_payload_bytes(&source)?,
        )?;
        self.peak_payload_bytes = self.peak_payload_bytes.max(self.current_payload_bytes);
        self.sources.insert(key, source.clone());
        Ok(source)
    }

    fn resolve_phase(
        &mut self,
        block: &SequentialReplayBlock,
        output_index: usize,
    ) -> Result<ResolvedPhaseReplay, SequentialReplayError> {
        let phase = self.provider.resolve_phase(block, output_index)?;
        self.support_receipt.update(block.generation.to_le_bytes());
        self.support_receipt
            .update((output_index as u64).to_le_bytes());
        self.support_receipt.update(phase.id.get().to_le_bytes());
        self.support_receipt
            .update((phase.realized_support.len() as u64).to_le_bytes());
        for chunk in phase.realized_support.chunks(8) {
            let mut packed = 0_u8;
            for (bit, &value) in chunk.iter().enumerate() {
                packed |= u8::from(value) << bit;
            }
            self.support_receipt.update([packed]);
        }
        Ok(phase)
    }

    fn resolve_compression(
        &mut self,
        block: &SequentialReplayBlock,
        native_index: usize,
    ) -> Result<ResolvedCompressionReplay, SequentialReplayError> {
        self.provider.resolve_compression(block, native_index)
    }
}

/// Immutable raw-source/model resolver used by the capped artifact provider.
pub trait SequentialPrimitiveSourceResolver {
    /// Verified source manifest and proper-complex model identity.
    fn identity(&self) -> &SequentialSourceProviderIdentity;

    /// Generation-specific identity for one replay block.
    ///
    /// Batch resolvers use their artifact-wide identity. NRT resolver bundles
    /// override this with the exact sealed/open generation member receipt.
    fn identity_for_block(
        &self,
        _block: &SequentialReplayBlock,
    ) -> Result<&SequentialSourceProviderIdentity, SequentialReplayError> {
        Ok(self.identity())
    }

    /// Maximum resolver-internal resident bytes beyond the returned source.
    fn maximum_resident_bytes(&self) -> u64;

    /// Exact empirical support/content/configuration receipt for the last source.
    ///
    /// Synthetic resolvers retain their numeric receipt as the exact receipt;
    /// production resolvers override this method with the empirical receipt.
    fn factor_receipt_digest(
        &self,
        source: &ResolvedPrimitiveSource,
    ) -> Result<[u8; 32], SequentialReplayError> {
        Ok(source.factor.numeric_receipt_digest())
    }

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
    /// Cumulative selected-block logical payload bytes loaded from HDF5.
    pub logical_block_bytes_read: Option<u64>,
    /// Wall-clock time spent in physical capped block reads.
    pub operator_block_read_elapsed: Duration,
    /// Current cached selected-block logical payload bytes.
    pub current_cached_payload_bytes: Option<u64>,
    /// Peak cached selected-block logical payload bytes.
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
pub struct CovarianceArtifactReplayProvider<'a, R> {
    _artifact_read_lock: CovarianceArtifactReadLock,
    operator_reader: CovarianceOperatorBlockReader,
    operator_block_byte_cap: u64,
    topology: &'a SequentialReplayTopology,
    build_identity: SequentialReplayBuildIdentity,
    identity: SequentialSourceProviderIdentity,
    source_resolver: R,
    cached_block: Option<CovarianceOperatorBlock>,
    operator_block_reads: u64,
    operator_block_cache_hits: u64,
    logical_block_bytes_read: u64,
    current_cached_payload_bytes: u64,
    peak_cached_payload_bytes: u64,
    operator_block_read_elapsed: Duration,
}

impl<'a, R> CovarianceArtifactReplayProvider<'a, R>
where
    R: SequentialPrimitiveSourceResolver,
{
    /// Open a manifest-committed operator directory without loading numeric blocks.
    ///
    /// # Errors
    /// Returns an explicit status for a zero cap, non-replayable metadata, or
    /// source manifest/model receipt mismatch.
    #[allow(clippy::too_many_lines)]
    pub fn open(
        artifact_directory: impl AsRef<Path>,
        operator_block_byte_cap: u64,
        topology: &'a SequentialReplayTopology,
        build_identity: SequentialReplayBuildIdentity,
        source_resolver: R,
    ) -> Result<Self, SequentialReplayError> {
        if operator_block_byte_cap == 0 {
            return Err(SequentialReplayError::Invalid(
                "operator block byte cap must be positive",
            ));
        }
        let artifact_directory = artifact_directory.as_ref();
        let artifact_read_lock = acquire_covariance_artifact_read_lock(artifact_directory)
            .map_err(|_| {
                SequentialReplayError::Provider(
                    ReplayStatus::SourceUnavailable,
                    "covariance artifact is not stable for replay",
                )
            })?;
        read_covariance_artifact_manifest_with_byte_cap(
            artifact_directory,
            operator_block_byte_cap,
        )
        .map_err(|_| {
            SequentialReplayError::Provider(
                ReplayStatus::SourceUnavailable,
                "capped covariance artifact manifest verification failed",
            )
        })?;
        let operator_path = artifact_directory.join(COVARIANCE_OPERATOR_FILENAME);
        let operator_reader =
            CovarianceOperatorBlockReader::open(&operator_path, operator_block_byte_cap).map_err(
                |_| {
                    SequentialReplayError::Provider(
                        ReplayStatus::SourceUnavailable,
                        "capped covariance operator metadata read failed",
                    )
                },
            )?;
        let metadata = operator_reader.metadata();
        if let Some(status) = persisted_replay_failure(metadata.replay_status) {
            return Err(SequentialReplayError::Provider(
                status,
                "covariance operator metadata is not replayable",
            ));
        }
        if build_identity
            .normalized_config_digest
            .iter()
            .all(|byte| *byte == 0)
            || build_identity.kernel_digest.iter().all(|byte| *byte == 0)
            || !build_identity.branch_tolerance.is_finite()
            || build_identity.branch_tolerance <= 0.0
            || build_identity.normalized_config_digest != topology.normalized_config_digest()
            || build_identity.kernel_digest != sequential_replay_kernel_digest()
            || !digest_matches(
                &metadata.normalized_config_digest,
                build_identity.normalized_config_digest,
            )
            || !digest_matches(&metadata.kernel_digest, build_identity.kernel_digest)
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::ReplayStateMismatch,
                "artifact kernel or normalized configuration differs from the reviewed replay identity",
            ));
        }
        let identity = source_resolver.identity().clone();
        let identity_strings_are_valid = [
            identity.provider.as_str(),
            identity.provider_version.as_str(),
            identity.model.as_str(),
            identity.model_version.as_str(),
        ]
        .iter()
        .all(|value| !value.is_empty());
        let identity_mismatch = |message| {
            SequentialReplayError::Provider(ReplayStatus::SourceIdentityMismatch, message)
        };
        if !metadata
            .source
            .manifest_digest
            .as_deref()
            .is_some_and(|value| digest_matches(value, identity.source_manifest_digest))
        {
            return Err(identity_mismatch(
                "artifact source manifest differs from the source resolver",
            ));
        }
        if !identity_strings_are_valid
            || metadata.source.provider.as_deref() != Some(identity.provider.as_str())
            || metadata.source.provider_version.as_deref()
                != Some(identity.provider_version.as_str())
            || metadata.source.model.as_deref() != Some(identity.model.as_str())
            || metadata.source.model_version.as_deref() != Some(identity.model_version.as_str())
        {
            return Err(identity_mismatch(
                "artifact provider/model name or version differs from the source resolver",
            ));
        }
        if identity.source_model_version_digest
            != sequential_source_model_identity_digest(
                &identity.provider,
                &identity.provider_version,
                &identity.model,
                &identity.model_version,
            )
        {
            return Err(identity_mismatch(
                "source resolver provider/model identity digest is inconsistent",
            ));
        }
        if !metadata
            .source
            .model_receipt_digest
            .as_deref()
            .is_some_and(|value| digest_matches(value, identity.source_model_hash))
        {
            return Err(identity_mismatch(
                "artifact source-model receipt differs from the source resolver",
            ));
        }
        topology.validate_provider_identity(&identity)?;
        for block in topology.blocks() {
            let block_identity = source_resolver.identity_for_block(block)?;
            topology.validate_provider_identity_for_block(block, block_identity)?;
            if block_identity.provider != identity.provider
                || block_identity.provider_version != identity.provider_version
                || block_identity.model != identity.model
                || block_identity.model_version != identity.model_version
                || block_identity.source_model_version_digest
                    != identity.source_model_version_digest
                || block_identity.source_model_hash != identity.source_model_hash
            {
                return Err(identity_mismatch(
                    "artifact generation provider/model identity differs from the complete revision resolver",
                ));
            }
        }
        Ok(Self {
            _artifact_read_lock: artifact_read_lock,
            operator_reader,
            operator_block_byte_cap,
            topology,
            build_identity,
            identity,
            source_resolver,
            cached_block: None,
            operator_block_reads: 0,
            operator_block_cache_hits: 0,
            logical_block_bytes_read: 0,
            current_cached_payload_bytes: 0,
            peak_cached_payload_bytes: 0,
            operator_block_read_elapsed: Duration::ZERO,
        })
    }

    /// Return physical-read, cache-hit, timing, and resident-cap metrics.
    #[must_use]
    pub fn metrics(&self) -> CovarianceArtifactReplayMetrics {
        CovarianceArtifactReplayMetrics {
            operator_block_reads: self.operator_block_reads,
            operator_block_cache_hits: self.operator_block_cache_hits,
            logical_block_bytes_read: Some(self.logical_block_bytes_read),
            operator_block_read_elapsed: self.operator_block_read_elapsed,
            current_cached_payload_bytes: self
                .cached_block
                .as_ref()
                .map(|_| self.current_cached_payload_bytes),
            peak_cached_payload_bytes: (self.operator_block_reads > 0)
                .then_some(self.peak_cached_payload_bytes),
            block_reservation_bytes: self.operator_block_byte_cap,
            cached_block_id: self.cached_block.as_ref().map(|block| block.block_id),
        }
    }

    pub(crate) const fn source_resolver(&self) -> &R {
        &self.source_resolver
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
            self.operator_block_cache_hits = self.operator_block_cache_hits.checked_add(1).ok_or(
                SequentialReplayError::Invalid("operator block cache-hit metric overflow"),
            )?;
        } else {
            self.cached_block = None;
            self.current_cached_payload_bytes = 0;
            let started = Instant::now();
            let receipt = self
                .operator_reader
                .read_block_with_receipt(block.id.get(), self.operator_block_byte_cap)
                .map_err(|_| {
                    SequentialReplayError::Provider(
                        ReplayStatus::SourceUnavailable,
                        "capped covariance operator block read failed",
                    )
                })?;
            self.topology.validate_operator_block_contract(
                block,
                &receipt.block,
                self.build_identity.branch_tolerance,
            )?;
            for native_index in 0..receipt.block.source_ids.len() {
                if packed_bit_value(&receipt.block.native_validity_bits, native_index) {
                    continue;
                }
                let start = native_index * 32;
                let content: [u8; 32] = receipt.block.source_content_digests[start..start + 32]
                    .try_into()
                    .map_err(|_| {
                        SequentialReplayError::Provider(
                            ReplayStatus::ReplayStateMismatch,
                            "artifact masked-source content receipt is malformed",
                        )
                    })?;
                let stored = &receipt.block.source_factor_digests[start..start + 32];
                let block_identity = self.source_resolver.identity_for_block(block)?;
                let expected = masked_source_factor_receipt_digest(
                    block,
                    receipt.block.source_ids[native_index],
                    content,
                    block_identity,
                );
                if stored != expected {
                    return Err(SequentialReplayError::Provider(
                        ReplayStatus::SourceIdentityMismatch,
                        "artifact masked-source receipt differs from captured status",
                    ));
                }
            }
            self.operator_block_read_elapsed = self
                .operator_block_read_elapsed
                .checked_add(started.elapsed())
                .ok_or(SequentialReplayError::Invalid(
                    "operator block read duration overflow",
                ))?;
            self.operator_block_reads =
                self.operator_block_reads
                    .checked_add(1)
                    .ok_or(SequentialReplayError::Invalid(
                        "operator block read metric overflow",
                    ))?;
            self.logical_block_bytes_read = self
                .logical_block_bytes_read
                .checked_add(receipt.logical_payload_bytes)
                .ok_or(SequentialReplayError::Invalid(
                    "operator block payload metric overflow",
                ))?;
            self.current_cached_payload_bytes = receipt.logical_payload_bytes;
            self.peak_cached_payload_bytes = self
                .peak_cached_payload_bytes
                .max(receipt.logical_payload_bytes);
            self.cached_block = Some(receipt.block);
        }
        let stored = self
            .cached_block
            .as_ref()
            .ok_or(SequentialReplayError::Provider(
                ReplayStatus::SourceUnavailable,
                "capped covariance operator block cache is empty",
            ))?;
        Ok(stored)
    }
}

const fn persisted_replay_failure(status: CovarianceReplayStatus) -> Option<ReplayStatus> {
    match status {
        CovarianceReplayStatus::Replayable => None,
        CovarianceReplayStatus::SourceManifestMissing
        | CovarianceReplayStatus::SourceManifestMismatch => {
            Some(ReplayStatus::SourceIdentityMismatch)
        }
        CovarianceReplayStatus::SupportNotFrozen => Some(ReplayStatus::ReplayStateMismatch),
        CovarianceReplayStatus::UnsupportedBackend => Some(ReplayStatus::UnsupportedBackend),
        CovarianceReplayStatus::SourceUnavailable => Some(ReplayStatus::SourceUnavailable),
        CovarianceReplayStatus::SourceModelUnavailable => {
            Some(ReplayStatus::SourceModelUnavailable)
        }
    }
}

impl<R> SequentialSourceReplayProvider for CovarianceArtifactReplayProvider<'_, R>
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
        self.topology.validate_provider_identity_for_block(
            block,
            self.source_resolver.identity_for_block(block)?,
        )?;
        let stored = self.read_block(block)?;
        let stored_source_id = stored.source_ids.get(native_index).copied();
        let digest_start = native_index
            .checked_mul(32)
            .ok_or(SequentialReplayError::Invalid(
                "source content digest offset overflows usize",
            ))?;
        let digest = stored
            .source_content_digests
            .get(digest_start..digest_start + 32)
            .ok_or(SequentialReplayError::Provider(
                ReplayStatus::ReplayStateMismatch,
                "artifact source content digest is missing",
            ))?;
        let mut stored_content_digest = [0_u8; 32];
        stored_content_digest.copy_from_slice(digest);
        let factor_digest = stored
            .source_factor_digests
            .get(digest_start..digest_start + 32)
            .ok_or(SequentialReplayError::Provider(
                ReplayStatus::ReplayStateMismatch,
                "artifact source factor receipt is missing",
            ))?;
        let mut stored_factor_digest = [0_u8; 32];
        stored_factor_digest.copy_from_slice(factor_digest);
        if !packed_bit_value(&stored.native_validity_bits, native_index) {
            let expected = masked_source_factor_receipt_digest(
                block,
                stored_source_id.ok_or(SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "artifact masked source ID is missing",
                ))?,
                stored_content_digest,
                self.source_resolver.identity_for_block(block)?,
            );
            if stored_factor_digest != expected {
                return Err(SequentialReplayError::Provider(
                    ReplayStatus::SourceIdentityMismatch,
                    "artifact masked-source receipt differs from captured status",
                ));
            }
            return Err(SequentialReplayError::Provider(
                ReplayStatus::MaskedNode,
                "masked primitive source has no empirical factor",
            ));
        }
        let source = self.source_resolver.resolve_source(block, native_index)?;
        let exact_factor_receipt = self.source_resolver.factor_receipt_digest(&source)?;
        if stored_source_id != Some(source.id.get())
            || stored_content_digest != source.content_digest
            || stored_factor_digest
                != available_source_factor_receipt_digest(exact_factor_receipt, &source)
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceIdentityMismatch,
                "artifact source, content, or numeric factor differs from the immutable resolver",
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
            realized_support: {
                let bits = usize::try_from(stored.support_bits_per_output).map_err(|_| {
                    SequentialReplayError::Provider(
                        ReplayStatus::ReplayStateMismatch,
                        "phase support width exceeds usize",
                    )
                })?;
                let bytes = bits.div_ceil(8);
                let start =
                    output_index
                        .checked_mul(bytes)
                        .ok_or(SequentialReplayError::Invalid(
                            "phase support offset overflows usize",
                        ))?;
                let packed = stored.support_bits.get(start..start + bytes).ok_or(
                    SequentialReplayError::Provider(
                        ReplayStatus::ReplayStateMismatch,
                        "phase support is missing from the capped operator block",
                    ),
                )?;
                (0..bits)
                    .map(|slot| packed_bit_value(packed, slot))
                    .collect()
            },
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
    /// Legacy status retained for version-1 receipt compatibility.
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
    /// Target/reference selections do not share a reproducible common date axis.
    InvalidReference,
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
            Self::InvalidReference => "invalid_reference",
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
    /// Peak retained per-source influence matrices and local/global map records.
    pub source_influence_bytes: u64,
    /// Query-local source-correlation matrix and two streamed multiply buffers.
    pub source_correlation_workspace_bytes: u64,
    /// Bytes for query-cached raw sources/factors plus one factor working copy.
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

/// Execution mode for a reference-specific covariance replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceSpecificExecutionMode {
    /// Whole, tiled, or bounded batch execution.
    Batch,
    /// Resumable or near-real-time execution.
    Nrt,
}

/// Version-1 execution scope for a reference-specific covariance replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceSpecificReplayScope {
    execution_mode: ReferenceSpecificExecutionMode,
    stitched_burst_count: usize,
}

impl ReferenceSpecificReplayScope {
    /// Construct a scope from the actual execution mode and stitched burst count.
    #[must_use]
    pub const fn new(
        execution_mode: ReferenceSpecificExecutionMode,
        stitched_burst_count: usize,
    ) -> Self {
        Self {
            execution_mode,
            stitched_burst_count,
        }
    }

    /// Return the stable version-1 disposition without performing numeric replay.
    #[must_use]
    pub const fn disposition(self) -> SpatialCovarianceStatus {
        if matches!(self.execution_mode, ReferenceSpecificExecutionMode::Nrt) {
            return SpatialCovarianceStatus::UnsupportedNrtReplay;
        }
        if self.stitched_burst_count != 1 {
            return SpatialCovarianceStatus::UnsupportedMultiburstReference;
        }
        SpatialCovarianceStatus::Valid
    }
}

/// Stable version-1 disposition for reference-specific spatial covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialCovarianceStatus {
    /// One single-burst batch target/reference query is eligible for replay.
    Valid,
    /// Artifact-backed sealed/open replay is not supported for NRT.
    UnsupportedNrtReplay,
    /// Stitched multiburst covariance is not modeled.
    UnsupportedMultiburstReference,
}

impl SpatialCovarianceStatus {
    /// Stable serialized status name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Valid => "valid",
            Self::UnsupportedNrtReplay => "unsupported_nrt_replay",
            Self::UnsupportedMultiburstReference => "unsupported_multiburst_reference",
        }
    }
}

/// Successful byte-capped temporal covariance replay.
#[derive(Debug)]
pub struct TemporalCovarianceReplay {
    /// Acquisition-0-gauge covariance in selection order.
    pub covariance: Array2<f64>,
    /// Topology-only allocation receipt checked before graph contraction.
    pub dependency_cone: DependencyConeEstimate,
    /// Peak logical raw-source and factor payload retained by the query cache.
    pub source_cache_peak_bytes: u64,
}

/// Successful joint target/reference covariance replay.
#[derive(Debug)]
pub struct ReferenceDifferenceCovarianceReplay {
    /// Target marginal covariance in the requested common-date order.
    pub target_covariance: Array2<f64>,
    /// Reference marginal covariance in the requested common-date order.
    pub reference_covariance: Array2<f64>,
    /// Target/reference cross covariance in the requested common-date order.
    pub target_reference_covariance: Array2<f64>,
    /// Covariance of target minus reference, with the exact gauge retained.
    pub difference_covariance: Array2<f64>,
    /// Topology-only allocation receipt checked before graph contraction.
    pub dependency_cone: DependencyConeEstimate,
    /// Stable digest of the ordered target/reference selection pair.
    pub reference_signature: [u8; 32],
    /// Peak logical source/factor payload retained during provider replay.
    pub source_cache_peak_bytes: u64,
    /// Digest of every exact primitive source/factor resolved by the query.
    pub source_factor_receipt: [u8; 32],
    /// Digest of every exact realized phase support resolved by the query.
    pub support_receipt: [u8; 32],
    /// Exact support-union effective-look scaling applied by production replay.
    pub effective_looks: Option<EffectiveLooksReplay>,
    /// Successful target replay disposition.
    pub target_disposition: ReplayStatus,
    /// Successful reference replay disposition.
    pub reference_disposition: ReplayStatus,
}

/// Exact effective-look realization applied to one target/reference pair.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectiveLooksReplay {
    /// Frozen source-factor scaling model.
    pub model: &'static str,
    /// Frozen exponential spatial-correlation distance scale in pixels.
    pub distance_scale_pixels: f64,
    /// Number of unique global native source coordinates in the pair union.
    pub support_union_count: usize,
    /// `n / (1^T R 1)` for the exact sorted support union.
    pub fraction: f64,
    /// Strong receipt over the model, scale, exact coordinates, and fraction.
    pub receipt: [u8; 32],
}

/// Spatial correlation applied between primitive source influence operators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SourceCorrelationModel {
    /// Distinct global source coordinates are independent; exact coordinate
    /// overlap remains one shared primitive source.
    Identity,
    /// Isotropic exponential correlation on global native-grid coordinates.
    ExponentialEuclidean {
        /// Positive finite correlation distance scale in native pixels.
        distance_scale_pixels: f64,
    },
}

impl SourceCorrelationModel {
    /// Stable machine-readable model identity.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Identity => "identity_v1",
            Self::ExponentialEuclidean { .. } => "exponential_euclidean_v1",
        }
    }

    const fn distance_scale_pixels(self) -> f64 {
        match self {
            Self::Identity => 0.0,
            Self::ExponentialEuclidean {
                distance_scale_pixels,
            } => distance_scale_pixels,
        }
    }

    fn validate(self) -> Result<(), SequentialReplayError> {
        match self {
            Self::Identity => Ok(()),
            Self::ExponentialEuclidean {
                distance_scale_pixels,
            } if distance_scale_pixels.is_finite() && distance_scale_pixels > 0.0 => Ok(()),
            Self::ExponentialEuclidean { .. } => Err(SequentialReplayError::Invalid(
                "source correlation distance scale must be finite and positive",
            )),
        }
    }

    fn correlation(self, left: (u64, u64), right: (u64, u64)) -> f64 {
        match self {
            Self::Identity => f64::from(left == right),
            Self::ExponentialEuclidean {
                distance_scale_pixels,
            } => {
                let row = left.0.abs_diff(right.0) as f64;
                let column = left.1.abs_diff(right.1) as f64;
                (-(row.hypot(column)) / distance_scale_pixels).exp()
            }
        }
    }
}

/// Global production query routed across captured phase-link tile topologies.
#[derive(Debug, Clone, Copy)]
pub struct GlobalReferenceCovarianceQuery<'a> {
    /// Exact burst owning both output pixels.
    pub burst_id: &'a str,
    /// Global output-grid target row and column.
    pub target: (u64, u64),
    /// Global output-grid reference row and column.
    pub reference: (u64, u64),
    /// Exact increasing acquisition order, including acquisition zero first.
    pub ordered_dates: &'a [GlobalDateId],
    /// Maximum real source-factor rank across active blocks.
    pub source_rank: usize,
    /// Explicit spatial correlation for the primitive source support union.
    pub source_correlation: SourceCorrelationModel,
    /// Total admitted bytes including routing selections and the returned joint matrix.
    pub byte_cap: u64,
    /// Exact fixed-branch tolerance used by capture.
    pub branch_tolerance: f64,
}

/// One topology and its separately opened, tile-scoped replay provider.
pub struct SequentialTileReplayProvider<'a> {
    topology: &'a SequentialReplayTopology,
    provider: &'a mut (dyn SequentialSourceReplayProvider + 'a),
}

impl<'a> SequentialTileReplayProvider<'a> {
    /// Bind one captured tile topology to its provider lifetime.
    #[must_use]
    pub fn new<P>(topology: &'a SequentialReplayTopology, provider: &'a mut P) -> Self
    where
        P: SequentialSourceReplayProvider + 'a,
    {
        Self { topology, provider }
    }
}

/// Global routed replay with an explicit full joint covariance and high-water receipt.
#[derive(Debug)]
pub struct GlobalReferenceDifferenceCovarianceReplay {
    /// Full target-then-reference `2N x 2N` joint phase covariance.
    pub joint_phase_covariance: Array2<f64>,
    /// Exact marginals, cross block, difference, receipts, and dispositions.
    pub replay: ReferenceDifferenceCovarianceReplay,
    /// Conservative query high-water including routing and joint-result allocations.
    pub resource_high_water_bytes: u64,
}

/// Pure global routing/dependency resource estimate produced before source reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalReferenceCovarianceResourceEstimate {
    /// Topology, operator, source-cache, covariance, and provider reservation.
    pub replay_bytes: u64,
    /// Global routing selections and returned joint covariance allocation.
    pub wrapper_bytes: u64,
    /// Exact conservative sum admitted by the global replay entry point.
    pub total_bytes: u64,
}

struct SpatialQueryCone {
    active_outputs: Vec<BTreeSet<usize>>,
    active_sources: Vec<BTreeSet<usize>>,
    required_compressed: Vec<BTreeSet<usize>>,
    selected_dates: Vec<BTreeSet<(GlobalDateId, usize)>>,
}

struct ReferenceDifferenceQueryPlan {
    selection: Vec<(GlobalDateId, usize)>,
    cone: SpatialQueryCone,
    estimate: DependencyConeEstimate,
}

struct StreamingAdjoints {
    phase: Vec<BTreeMap<usize, Array2<f64>>>,
    compressed: Vec<BTreeMap<usize, Array2<f64>>>,
}

struct PhaseWindowReplay {
    source_values: Array2<Cf64>,
    replay: RectPixelReplay,
}

fn reference_selection_signature(
    target_selection: &[(GlobalDateId, usize)],
    reference_selection: &[(GlobalDateId, usize)],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:reference-selection:v1");
    for &(date, output) in target_selection.iter().chain(reference_selection) {
        digest.update(date.get().to_le_bytes());
        digest.update((output as u64).to_le_bytes());
    }
    digest.finalize().into()
}

fn cross_topology_reference_selection_signature(
    target_namespace: &ReplayIdNamespace,
    target_selection: &[(GlobalDateId, usize)],
    target_output: (u64, u64),
    reference_namespace: &ReplayIdNamespace,
    reference_selection: &[(GlobalDateId, usize)],
    reference_output: (u64, u64),
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:cross-topology-reference-selection:v1");
    digest.update(target_namespace.burst_id.as_bytes());
    digest.update(target_namespace.source_manifest_digest);
    digest.update(target_namespace.source_model_version_digest);
    for (namespace, selection, output) in [
        (target_namespace, target_selection, target_output),
        (reference_namespace, reference_selection, reference_output),
    ] {
        digest.update(namespace.native_origin.0.to_le_bytes());
        digest.update(namespace.native_origin.1.to_le_bytes());
        digest.update(namespace.output_origin.0.to_le_bytes());
        digest.update(namespace.output_origin.1.to_le_bytes());
        digest.update(output.0.to_le_bytes());
        digest.update(output.1.to_le_bytes());
        for &(date, _) in selection {
            digest.update(date.get().to_le_bytes());
        }
    }
    digest.finalize().into()
}

fn empty_query_receipt(domain: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.finalize().into()
}

fn combined_query_receipt(domain: &[u8], left: [u8; 32], right: [u8; 32]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(left);
    digest.update(right);
    digest.finalize().into()
}

fn source_correlation_workspace_bytes(
    support: u64,
    selected: u64,
    model: SourceCorrelationModel,
) -> Result<u64, SequentialReplayError> {
    model.validate()?;
    if matches!(model, SourceCorrelationModel::Identity) {
        return Ok(0);
    }
    let correlation = checked_mul(checked_mul(support, support)?, 8)?;
    let multiply_buffers = checked_mul(checked_mul(checked_mul(2, support)?, selected)?, 8)?;
    let coordinate_buffer = checked_mul(support, size_of::<(u64, u64)>() as u64)?;
    checked_add(
        correlation,
        checked_add(multiply_buffers, coordinate_buffer)?,
    )
}

fn global_source_adjoints(
    topology: &SequentialReplayTopology,
    source_adjoints: BTreeMap<usize, Array2<f64>>,
) -> Result<BTreeMap<(u64, u64), Array2<f64>>, SequentialReplayError> {
    let mut global = BTreeMap::new();
    for (native, root) in source_adjoints {
        let coordinate = topology.global_native_coordinate(native)?;
        match global.entry(coordinate) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(root);
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                if entry.get().dim() != root.dim() {
                    return Err(SequentialReplayError::Provider(
                        ReplayStatus::ReplayStateMismatch,
                        "shared global source influence dimensions differ",
                    ));
                }
                *entry.get_mut() += &root;
            }
        }
    }
    Ok(global)
}

fn merge_global_source_adjoints(
    mut left: BTreeMap<(u64, u64), Array2<f64>>,
    right: BTreeMap<(u64, u64), Array2<f64>>,
) -> Result<BTreeMap<(u64, u64), Array2<f64>>, SequentialReplayError> {
    for (coordinate, root) in right {
        match left.entry(coordinate) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(root);
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                if entry.get().dim() != root.dim() {
                    return Err(SequentialReplayError::Provider(
                        ReplayStatus::ReplayStateMismatch,
                        "shared global source influence dimensions differ",
                    ));
                }
                *entry.get_mut() += &root;
            }
        }
    }
    Ok(left)
}

fn contract_source_adjoints(
    covariance: &mut Array2<f64>,
    source_adjoints: &BTreeMap<(u64, u64), Array2<f64>>,
    model: SourceCorrelationModel,
) -> Result<(), SequentialReplayError> {
    model.validate()?;
    let Some(first) = source_adjoints.values().next() else {
        return Ok(());
    };
    if first.ncols() != covariance.nrows() || covariance.nrows() != covariance.ncols() {
        return Err(SequentialReplayError::Provider(
            ReplayStatus::ReplayStateMismatch,
            "source influence dimensions do not match the covariance query",
        ));
    }
    if matches!(model, SourceCorrelationModel::Identity) {
        for root in source_adjoints.values() {
            if root.dim() != first.dim() {
                return Err(SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "source influence dimensions differ",
                ));
            }
            for row in 0..covariance.nrows() {
                for column in 0..covariance.ncols() {
                    covariance[(row, column)] += (0..root.nrows())
                        .map(|basis| root[(basis, row)] * root[(basis, column)])
                        .sum::<f64>();
                }
            }
        }
        return Ok(());
    }
    let coordinates = source_adjoints.keys().copied().collect::<Vec<_>>();
    let support = coordinates.len();
    let selected = covariance.nrows();
    let correlation = Array2::from_shape_fn((support, support), |(left, right)| {
        model.correlation(coordinates[left], coordinates[right])
    });
    let mut basis_influences = Array2::zeros((support, selected));
    for basis in 0..first.nrows() {
        for (source, root) in source_adjoints.values().enumerate() {
            if root.dim() != first.dim() {
                return Err(SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "source influence dimensions differ",
                ));
            }
            for column in 0..selected {
                basis_influences[(source, column)] = root[(basis, column)];
            }
        }
        let correlated = correlation.dot(&basis_influences);
        for row in 0..selected {
            for column in 0..selected {
                covariance[(row, column)] += (0..support)
                    .map(|source| basis_influences[(source, row)] * correlated[(source, column)])
                    .sum::<f64>();
            }
        }
    }
    Ok(())
}

fn attach_source_correlation_receipt(
    replay: &mut ReferenceDifferenceCovarianceReplay,
    support_union: &BTreeSet<(u64, u64)>,
    model: SourceCorrelationModel,
) -> Result<(), SequentialReplayError> {
    model.validate()?;
    if support_union.is_empty() {
        return Err(SequentialReplayError::Provider(
            ReplayStatus::ReplayStateMismatch,
            "source correlation requires a nonempty realized support union",
        ));
    }
    let denominator = support_union
        .iter()
        .flat_map(|left| {
            support_union
                .iter()
                .map(move |right| model.correlation(*left, *right))
        })
        .sum::<f64>();
    let fraction = support_union.len() as f64 / denominator;
    if !denominator.is_finite()
        || denominator <= 0.0
        || !fraction.is_finite()
        || fraction <= 0.0
        || fraction > 1.0 + 16.0 * f64::EPSILON
    {
        return Err(SequentialReplayError::Provider(
            ReplayStatus::NonFiniteReplayState,
            "source support correlation is invalid",
        ));
    }
    let mut receipt = Sha256::new();
    receipt.update(b"dolphinrust:source-correlation-realization:v1");
    receipt.update(model.as_str().as_bytes());
    receipt.update(model.distance_scale_pixels().to_bits().to_le_bytes());
    receipt.update((support_union.len() as u64).to_le_bytes());
    for &(row, column) in support_union {
        receipt.update(row.to_le_bytes());
        receipt.update(column.to_le_bytes());
    }
    receipt.update(fraction.to_bits().to_le_bytes());
    receipt.update(replay.source_factor_receipt);
    receipt.update(replay.support_receipt);
    replay.effective_looks = Some(EffectiveLooksReplay {
        model: model.as_str(),
        distance_scale_pixels: model.distance_scale_pixels(),
        support_union_count: support_union.len(),
        fraction,
        receipt: receipt.finalize().into(),
    });
    Ok(())
}

#[cfg(test)]
mod source_correlation_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn factorized_source_contraction_matches_direct_ordered_pairs() {
        let roots = BTreeMap::from([
            ((0, 0), array![[1.0, -0.5], [0.25, 0.75]]),
            ((0, 2), array![[-0.4, 0.8], [1.2, -0.3]]),
        ]);
        let model = SourceCorrelationModel::ExponentialEuclidean {
            distance_scale_pixels: 1.5,
        };
        let mut expected = Array2::<f64>::zeros((2, 2));
        for (&left_coordinate, left) in &roots {
            for (&right_coordinate, right) in &roots {
                let correlation = model.correlation(left_coordinate, right_coordinate);
                for row in 0..2 {
                    for column in 0..2 {
                        expected[(row, column)] += correlation
                            * (0..left.nrows())
                                .map(|basis| left[(basis, row)] * right[(basis, column)])
                                .sum::<f64>();
                    }
                }
            }
        }
        let mut actual = Array2::<f64>::zeros((2, 2));
        contract_source_adjoints(&mut actual, &roots, model).unwrap();
        for (&left, &right) in actual.iter().zip(expected.iter()) {
            assert!((left - right).abs() <= 16.0 * f64::EPSILON);
        }
    }
}

fn global_reference_selection_signature(query: GlobalReferenceCovarianceQuery<'_>) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:global-reference-selection:v1");
    digest.update((query.burst_id.len() as u64).to_le_bytes());
    digest.update(query.burst_id.as_bytes());
    for coordinate in [query.target, query.reference] {
        digest.update(coordinate.0.to_le_bytes());
        digest.update(coordinate.1.to_le_bytes());
    }
    for date in query.ordered_dates {
        digest.update(date.get().to_le_bytes());
    }
    digest.update(query.source_correlation.as_str().as_bytes());
    digest.update(
        query
            .source_correlation
            .distance_scale_pixels()
            .to_bits()
            .to_le_bytes(),
    );
    digest.finalize().into()
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
    /// Persisted operator I/O or schema validation failed.
    Io(dolphin_io::IoError),
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
            Self::Io(_) => ReplayStatus::InvalidReplayGraph,
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
            Self::Io(error) => Display::fmt(error, f),
        }
    }
}

impl Error for SequentialReplayError {}

impl From<InfluenceError> for SequentialReplayError {
    fn from(value: InfluenceError) -> Self {
        Self::Influence(value)
    }
}

impl From<dolphin_io::IoError> for SequentialReplayError {
    fn from(value: dolphin_io::IoError) -> Self {
        Self::Io(value)
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
    generation_source_manifest_digests: Option<Vec<[u8; 32]>>,
    estimator_branch: FixedEstimatorBranch,
    normalized_config_digest: [u8; 32],
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
            None,
        )
    }

    /// Plan a strongly identified graph whose generations use exact member receipts.
    ///
    /// The complete revision digest remains in `id_namespace`; block, source,
    /// phase, date, and compressed IDs use the corresponding generation digest.
    ///
    /// # Errors
    /// Returns an error when the digest list is missing, weak, or does not match
    /// the complete ministack plan.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_identified_generations(
        num_real_dates: usize,
        native_shape: (usize, usize),
        output_shape: (usize, usize),
        support_slots_per_output: usize,
        native_validity: ArrayView2<bool>,
        cfg: &SequentialConfig,
        scope: ReplayExecutionScope,
        id_namespace: ReplayIdNamespace,
        generation_source_manifest_digests: Vec<[u8; 32]>,
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
            Some(generation_source_manifest_digests),
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
        generation_source_manifest_digests: Option<Vec<[u8; 32]>>,
    ) -> Result<Self, SequentialReplayError> {
        assess_support(cfg, scope)?;
        if num_real_dates == 0 {
            return Err(SequentialReplayError::Invalid(
                "sequential replay requires at least one real acquisition",
            ));
        }
        if cfg.max_num_compressed == 0 && num_real_dates > cfg.ministack_size {
            return Err(SequentialReplayError::Invalid(
                "multi-ministack covariance replay requires at least one carried compressed SLC",
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
        if generation_source_manifest_digests
            .as_ref()
            .is_some_and(|digests| {
                digests.len() != planned.len()
                    || digests
                        .iter()
                        .any(|digest| digest.iter().all(|byte| *byte == 0))
            })
        {
            return Err(SequentialReplayError::Unsupported(
                ReplayStatus::UnsupportedSourceIdentity,
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
                    generation_source_manifest_digests
                        .as_ref()
                        .map_or(namespace.source_manifest_digest, |digests| {
                            digests[block.block_id]
                        }),
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
            generation_source_manifest_digests,
            estimator_branch: match cfg.use_evd {
                true => FixedEstimatorBranch::Evd,
                false => FixedEstimatorBranch::Emi {
                    beta: cfg.beta,
                    zero_correlation_threshold: cfg.zero_correlation_threshold,
                },
            },
            normalized_config_digest: sequential_replay_config_digest(cfg),
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

    /// Build the complete writer plan for this topology and one burst identity.
    ///
    /// # Errors
    /// Returns an error for an empty burst ID or overflowing date range.
    pub fn covariance_operator_plan(
        &self,
        burst_id: &str,
    ) -> Result<CovarianceOperatorPlan, SequentialReplayError> {
        if burst_id.is_empty() {
            return Err(SequentialReplayError::Invalid(
                "covariance writer plan requires a burst ID",
            ));
        }
        let namespace = self
            .id_namespace
            .as_ref()
            .ok_or(SequentialReplayError::Invalid(
                "covariance writer plan requires strongly identified grids",
            ))?;
        let grid = |origin: (u64, u64),
                    shape: (usize, usize),
                    strides: (usize, usize)|
         -> Result<CovarianceOperatorGrid, SequentialReplayError> {
            Ok(CovarianceOperatorGrid {
                row_start: origin.0,
                col_start: origin.1,
                rows: u32::try_from(shape.0).map_err(|_| {
                    SequentialReplayError::Invalid("covariance writer grid rows exceed u32")
                })?,
                cols: u32::try_from(shape.1).map_err(|_| {
                    SequentialReplayError::Invalid("covariance writer grid columns exceed u32")
                })?,
                stride_y: u32::try_from(strides.0).map_err(|_| {
                    SequentialReplayError::Invalid("covariance writer row stride exceeds u32")
                })?,
                stride_x: u32::try_from(strides.1).map_err(|_| {
                    SequentialReplayError::Invalid("covariance writer column stride exceeds u32")
                })?,
            })
        };
        let source_dates_by_generation = self
            .blocks
            .iter()
            .map(|block| {
                let count = u32::try_from(block.num_real_dates).map_err(|_| {
                    SequentialReplayError::Invalid("covariance writer plan date count exceeds u32")
                })?;
                let stop = block.real_date_start.get().checked_add(count).ok_or(
                    SequentialReplayError::Invalid("covariance writer plan date range overflows"),
                )?;
                Ok((block.real_date_start.get()..stop).collect())
            })
            .collect::<Result<Vec<Vec<u32>>, SequentialReplayError>>()?;
        Ok(CovarianceOperatorPlan {
            source_manifest_digest: namespace.source_manifest_digest,
            source_model_version_digest: namespace.source_model_version_digest,
            bursts: vec![CovarianceBurstPlan {
                burst_id: burst_id.to_owned(),
                source_dates_by_generation,
                tiles: vec![CovarianceTilePlan {
                    native_grid: grid(namespace.native_origin, self.native_shape, (1, 1))?,
                    output_grid: grid(
                        namespace.output_origin,
                        self.output_shape,
                        (self.strides.y, self.strides.x),
                    )?,
                    owned_output_grid: grid(
                        namespace.owned_output_origin,
                        namespace.owned_output_shape,
                        (self.strides.y, self.strides.x),
                    )?,
                }],
            }],
        })
    }

    /// Digest of every sequential configuration field bound by this topology.
    #[must_use]
    pub const fn normalized_config_digest(&self) -> [u8; 32] {
        self.normalized_config_digest
    }

    /// Exact source-member receipt used by one generation namespace.
    ///
    /// # Errors
    /// Returns an error for an unknown block.
    pub fn generation_source_manifest_digest(
        &self,
        block: GlobalBlockId,
    ) -> Result<[u8; 32], SequentialReplayError> {
        let definition = self.block(block)?;
        Ok(self
            .generation_source_manifest_digests
            .as_ref()
            .map_or_else(
                || {
                    self.id_namespace
                        .as_ref()
                        .map_or([0; 32], |namespace| namespace.source_manifest_digest)
                },
                |digests| digests[definition.generation as usize],
            ))
    }

    /// Deterministic source-locator ID before raw-content binding.
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
                self.generation_source_manifest_digest(block)?,
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

    /// Deterministic primitive source ID bound to its raw-content digest.
    ///
    /// # Errors
    /// Returns `Err` for an unknown source coordinate or an all-zero digest.
    pub fn source_id_for_content_digest(
        &self,
        block: GlobalBlockId,
        native_index: usize,
        content_digest: &[u8; 32],
    ) -> Result<SourceId, SequentialReplayError> {
        if content_digest.iter().all(|byte| *byte == 0) {
            return Err(SequentialReplayError::Unsupported(
                ReplayStatus::UnsupportedSourceIdentity,
            ));
        }
        let locator = self.source_id(block, native_index)?;
        covariance_content_bound_source_id(locator.get(), content_digest)
            .map(SourceId::new)
            .map_err(|_| {
                SequentialReplayError::Unsupported(ReplayStatus::UnsupportedSourceIdentity)
            })
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
        let source_manifest_digest = self.generation_source_manifest_digest(block)?;
        Ok(NodeId::new(match &self.id_namespace {
            Some(_) => self.identified_id(
                b"phase",
                source_manifest_digest,
                block.get(),
                0,
                output_index,
                false,
            ),
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
        let source_manifest_digest = self.generation_source_manifest_digest(block)?;
        Ok(NodeId::new(match &self.id_namespace {
            Some(_) => self.identified_id(
                b"compressed",
                source_manifest_digest,
                block.get(),
                0,
                native_index,
                true,
            ),
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
                self.generation_source_manifest_digest(block.id)?,
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
        maximum_output_pixels: usize,
    ) -> Result<SpatialQueryCone, SequentialReplayError> {
        if microbatch != 1 {
            return Err(SequentialReplayError::Invalid(
                "version-1 temporal covariance replay requires one output pixel",
            ));
        }
        let mut selected_dates = vec![BTreeSet::new(); self.blocks.len()];
        let mut active_outputs = vec![BTreeSet::new(); self.blocks.len()];
        let mut distinct_outputs = BTreeSet::new();
        for &(date, output_index) in selection {
            self.validate_date_output(date, output_index)?;
            distinct_outputs.insert(output_index);
            if date.get() != 0 {
                let block = self.block_for_date(date)?;
                let block_index = block.generation as usize;
                active_outputs[block_index].insert(output_index);
                selected_dates[block_index].insert((date, output_index));
            }
        }
        if maximum_output_pixels == 1 && distinct_outputs.len() != 1 {
            return Err(SequentialReplayError::Invalid(
                "version-1 temporal covariance selection must use one output pixel",
            ));
        }
        if maximum_output_pixels == 2 && (distinct_outputs.is_empty() || distinct_outputs.len() > 2)
        {
            return Err(SequentialReplayError::Invalid(
                "reference-specific covariance selection must use one target and one reference",
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
        let cone = self.spatial_query_cone(selection, microbatch, 1)?;
        self.estimate_dependency_cone_for_spatial_query(selection, source_rank, &cone)
    }

    #[allow(clippy::too_many_lines)]
    fn estimate_dependency_cone_for_spatial_query(
        &self,
        selection: &[(GlobalDateId, usize)],
        source_rank: usize,
        cone: &SpatialQueryCone,
    ) -> Result<DependencyConeEstimate, SequentialReplayError> {
        let mut block_ids = Vec::with_capacity(self.blocks.len());
        for index in (0..self.blocks.len()).rev() {
            if !cone.active_outputs[index].is_empty()
                || !cone.active_sources[index].is_empty()
                || !cone.required_compressed[index].is_empty()
            {
                block_ids.push(self.blocks[index].id);
            }
        }

        let mut frontier_coordinates = 0_u64;
        let mut source_influence_bytes = 0_u64;
        let mut max_real = 0_u64;
        let mut max_combined = 0_u64;
        let mut max_phase_dimension = 0_u64;
        let mut support_bits = 0_u64;
        let mut resident_control_bytes = 0_u64;
        let mut baseline_bytes = 0_u64;
        let mut source_cache_bytes = 0_u64;
        let set_record_bytes = btree_record_reservation_bytes::<usize, ()>();
        let selected_date_record_bytes =
            btree_record_reservation_bytes::<(GlobalDateId, usize), ()>();
        let adjoint_record_bytes = btree_record_reservation_bytes::<usize, Array2<f64>>();
        let source_cache_record_bytes =
            btree_record_reservation_bytes::<(GlobalBlockId, usize), ResolvedPrimitiveSource>();
        for &block_id in &block_ids {
            let block = self.block(block_id)?;
            let block_index = block.generation as usize;
            let native = cone.active_sources[block_index].len() as u64;
            let output = cone.active_outputs[block_index].len() as u64;
            let real = block.num_real_dates as u64;
            let compressed = cone.required_compressed[block_index].len() as u64;
            let selected_dates = cone.selected_dates[block_index].len() as u64;
            // Production replay pads every active root adjoint to the declared
            // maximum source rank so blocks can share one query contract. Charge
            // that actual allocation even when the final ministack is partial.
            let phase_coordinates = checked_mul(block.phase_dimension as u64, output)?;
            let compressed_coordinates = checked_mul(2, compressed)?;
            let date_coordinates = cone.selected_dates[block_index]
                .iter()
                .filter(|(date, _)| date.get() != 0)
                .count() as u64;
            frontier_coordinates = checked_add(
                frontier_coordinates,
                checked_add(
                    phase_coordinates,
                    checked_add(compressed_coordinates, date_coordinates)?,
                )?,
            )?;
            let source_payload_bytes = checked_mul(
                checked_mul(
                    checked_mul(source_rank as u64, native)?,
                    selection.len() as u64,
                )?,
                8,
            )?;
            source_influence_bytes = checked_add(source_influence_bytes, source_payload_bytes)?;
            max_real = max_real.max(real);
            max_phase_dimension = max_phase_dimension.max(block.phase_dimension as u64);
            let combined = checked_add(real, block.carried_parent_ids.len() as u64)?;
            max_combined = max_combined.max(combined);
            let cached_source_bytes = checked_add(
                checked_mul(checked_mul(real, real)?, 16)?,
                checked_add(checked_mul(real, 24)?, source_cache_record_bytes)?,
            )?;
            source_cache_bytes = source_cache_bytes.max(checked_mul(native, cached_source_bytes)?);

            let cone_control_bytes = checked_add(
                checked_mul(
                    checked_add(checked_add(native, output)?, compressed)?,
                    set_record_bytes,
                )?,
                checked_mul(selected_dates, selected_date_record_bytes)?,
            )?;
            let adjoint_control_bytes = checked_mul(
                checked_add(checked_add(native, output)?, compressed)?,
                adjoint_record_bytes,
            )?;
            resident_control_bytes = checked_add(
                resident_control_bytes,
                checked_add(cone_control_bytes, adjoint_control_bytes)?,
            )?;
            for &output_index in &cone.active_outputs[block_index] {
                let support = self.native_support_indices(output_index)?.len() as u64;
                support_bits = checked_add(support_bits, support)?;
                let source_values = checked_mul(checked_mul(support, combined)?, 16)?;
                let native_index_buffer = checked_mul(
                    self.support_slots_per_output() as u64,
                    size_of::<usize>() as u64,
                )?;
                let source_pixel_buffers = checked_mul(
                    checked_mul(support, 2)?,
                    size_of::<NativeSourcePixel>() as u64,
                )?;
                let source_index_buffers = checked_add(native_index_buffer, source_pixel_buffers)?;
                let complex_matrix = checked_mul(checked_mul(combined, combined)?, 16)?;
                let replay_and_rect_jvp_matrices = checked_mul(complex_matrix, 5)?;
                let combined_direction = checked_mul(combined, 16)?;
                let estimator_workspace = phase_angle_jvp_workspace_bytes(
                    usize::try_from(combined).map_err(|_| {
                        SequentialReplayError::Invalid("phase dimension exceeds usize")
                    })?,
                    self.estimator_branch,
                )
                .ok_or(SequentialReplayError::Invalid(
                    "phase estimator workspace estimate overflowed",
                ))?;
                baseline_bytes = baseline_bytes.max(checked_add(
                    checked_add(source_values, source_index_buffers)?,
                    checked_add(
                        replay_and_rect_jvp_matrices,
                        checked_add(combined_direction, estimator_workspace)?,
                    )?,
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
            source_cache_bytes,
            checked_add(
                checked_add(lower_factor_bytes, component_id_bytes)?,
                checked_add(
                    checked_add(real_embedding_bytes, raw_vector_bytes)?,
                    checked_add(
                        size_of::<ResolvedPrimitiveSource>() as u64,
                        checked_add(
                            size_of::<BTreeMap<(GlobalBlockId, usize), ResolvedPrimitiveSource>>()
                                as u64,
                            checked_mul(2, size_of::<Sha256>() as u64)?,
                        )?,
                    )?,
                )?,
            )?,
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
        let per_block_collection_headers = (4 * size_of::<BTreeSet<usize>>()
            + 2 * size_of::<BTreeMap<usize, Array2<f64>>>())
            as u64;
        let collection_header_bytes =
            checked_mul(self.blocks.len() as u64, per_block_collection_headers)?;
        let block_id_bytes = checked_mul(
            block_ids.capacity() as u64,
            size_of::<GlobalBlockId>() as u64,
        )?;
        let operator_bytes = checked_add(
            checked_add(
                resident_control_bytes,
                checked_add(collection_header_bytes, block_id_bytes)?,
            )?,
            one_jvp_bytes,
        )?;
        let support_bytes = checked_add(support_bits, 7)? / 8;
        let selected = selection.len() as u64;
        let covariance_bytes = checked_mul(checked_mul(selected, selected)?, 8)?;
        let total_bytes = checked_add(
            checked_add(
                checked_add(
                    checked_add(frontier_bytes, source_influence_bytes)?,
                    source_window_bytes,
                )?,
                checked_add(operator_bytes, baseline_bytes)?,
            )?,
            checked_add(support_bytes, covariance_bytes)?,
        )?;
        Ok(DependencyConeEstimate {
            block_ids,
            frontier_bytes,
            source_influence_bytes,
            source_correlation_workspace_bytes: 0,
            source_window_bytes,
            operator_bytes,
            baseline_bytes,
            support_bytes,
            covariance_bytes,
            provider_bytes: 0,
            total_bytes,
        })
    }

    fn effective_support_reservation_bytes(
        &self,
        cone: &SpatialQueryCone,
    ) -> Result<u64, SequentialReplayError> {
        let records = cone
            .active_sources
            .iter()
            .zip(&cone.required_compressed)
            .try_fold(0_u64, |total, (sources, compressed)| {
                checked_add(
                    total,
                    checked_add(sources.len() as u64, compressed.len() as u64)?,
                )
            })?;
        checked_add(
            size_of::<BTreeSet<(u64, u64)>>() as u64,
            checked_mul(records, btree_record_reservation_bytes::<(u64, u64), ()>())?,
        )
    }

    fn source_correlation_workspace_reservation_bytes(
        &self,
        cone: &SpatialQueryCone,
        selected: usize,
        model: SourceCorrelationModel,
    ) -> Result<u64, SequentialReplayError> {
        let support = cone
            .active_sources
            .iter()
            .map(BTreeSet::len)
            .max()
            .unwrap_or(0) as u64;
        source_correlation_workspace_bytes(support, selected as u64, model)
    }

    fn global_source_map_control_reservation_bytes(
        &self,
        cone: &SpatialQueryCone,
    ) -> Result<u64, SequentialReplayError> {
        let support = cone
            .active_sources
            .iter()
            .map(BTreeSet::len)
            .max()
            .unwrap_or(0) as u64;
        checked_add(
            size_of::<BTreeMap<(u64, u64), Array2<f64>>>() as u64,
            checked_mul(
                support,
                btree_record_reservation_bytes::<(u64, u64), Array2<f64>>(),
            )?,
        )
    }

    fn plan_reference_difference_query(
        &self,
        target_selection: &[(GlobalDateId, usize)],
        reference_selection: &[(GlobalDateId, usize)],
        query: DependencyConeQuery,
        source_correlation: SourceCorrelationModel,
    ) -> Result<ReferenceDifferenceQueryPlan, SequentialReplayError> {
        let selection = target_selection
            .iter()
            .chain(reference_selection)
            .copied()
            .collect::<Vec<_>>();
        let cone = self.spatial_query_cone(&selection, query.microbatch, 2)?;
        let mut estimate =
            self.estimate_dependency_cone_for_spatial_query(&selection, query.source_rank, &cone)?;
        let dates = target_selection.len() as u64;
        let output_matrices = checked_mul(checked_mul(checked_mul(4, dates)?, dates)?, 8)?;
        estimate.covariance_bytes = checked_add(estimate.covariance_bytes, output_matrices)?;
        estimate.total_bytes = checked_add(estimate.total_bytes, output_matrices)?;
        let selection_bytes = checked_mul(
            selection.capacity() as u64,
            size_of::<(GlobalDateId, usize)>() as u64,
        )?;
        estimate.operator_bytes = checked_add(estimate.operator_bytes, selection_bytes)?;
        estimate.total_bytes = checked_add(estimate.total_bytes, selection_bytes)?;
        if selection.iter().any(|(date, _)| date.get() != 0) {
            let effective_support_bytes = self.effective_support_reservation_bytes(&cone)?;
            estimate.support_bytes = checked_add(estimate.support_bytes, effective_support_bytes)?;
            estimate.total_bytes = checked_add(estimate.total_bytes, effective_support_bytes)?;
            let global_source_map_control_bytes =
                self.global_source_map_control_reservation_bytes(&cone)?;
            estimate.source_influence_bytes = checked_add(
                estimate.source_influence_bytes,
                global_source_map_control_bytes,
            )?;
            estimate.total_bytes =
                checked_add(estimate.total_bytes, global_source_map_control_bytes)?;
            let source_correlation_workspace_bytes = self
                .source_correlation_workspace_reservation_bytes(
                    &cone,
                    selection.len(),
                    source_correlation,
                )?;
            estimate.source_correlation_workspace_bytes = source_correlation_workspace_bytes;
            estimate.total_bytes =
                checked_add(estimate.total_bytes, source_correlation_workspace_bytes)?;
        }
        Ok(ReferenceDifferenceQueryPlan {
            selection,
            cone,
            estimate,
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
            source_cache_peak_bytes: 0,
        })
    }

    /// Replay one target/reference pair against a shared source graph.
    ///
    /// The target and reference selections must contain the same unique,
    /// increasing dates and one output pixel each. The method plans one union
    /// dependency cone, performs one graph contraction, and then applies the
    /// exact target-minus-reference contrast. It does not call the single-pixel
    /// temporal replay twice.
    ///
    /// # Errors
    /// Returns a topology, byte-cap, or influence-graph error.
    #[allow(clippy::too_many_lines)]
    pub fn replay_reference_difference_covariance<F>(
        &self,
        target_selection: &[(GlobalDateId, usize)],
        reference_selection: &[(GlobalDateId, usize)],
        query: DependencyConeQuery,
        build_graph: F,
    ) -> Result<ReferenceDifferenceCovarianceReplay, SequentialReplayError>
    where
        F: FnOnce(&DependencyConeEstimate) -> Result<InfluenceDag, InfluenceError>,
    {
        if target_selection.is_empty()
            || target_selection.len() != reference_selection.len()
            || query.source_rank == 0
            || query.microbatch != 1
        {
            return Err(SequentialReplayError::Invalid(
                "reference-specific covariance requires aligned selections, nonzero source rank, and one target",
            ));
        }
        let target_outputs = target_selection
            .iter()
            .map(|&(_, output)| output)
            .collect::<BTreeSet<_>>();
        let reference_outputs = reference_selection
            .iter()
            .map(|&(_, output)| output)
            .collect::<BTreeSet<_>>();
        if target_outputs.len() != 1 || reference_outputs.len() != 1 {
            return Err(SequentialReplayError::Invalid(
                "reference-specific covariance requires one target and one reference pixel",
            ));
        }
        let target_dates = target_selection
            .iter()
            .map(|&(date, _)| date)
            .collect::<Vec<_>>();
        let reference_dates = reference_selection
            .iter()
            .map(|&(date, _)| date)
            .collect::<Vec<_>>();
        if target_dates != reference_dates
            || target_dates.first().is_none_or(|date| date.get() != 0)
            || !target_dates
                .windows(2)
                .all(|pair| pair[0].get() < pair[1].get())
        {
            return Err(SequentialReplayError::Invalid(
                "reference-specific covariance requires identical increasing dates with acquisition zero first",
            ));
        }

        let selection = target_selection
            .iter()
            .chain(reference_selection)
            .copied()
            .collect::<Vec<_>>();
        let cone = self.spatial_query_cone(&selection, query.microbatch, 2)?;
        let dependency_cone =
            self.estimate_dependency_cone_for_spatial_query(&selection, query.source_rank, &cone)?;
        if dependency_cone.total_bytes > query.byte_cap {
            return Err(SequentialReplayError::Budget(dependency_cone));
        }
        let coordinates = selection
            .iter()
            .map(|&(date, output)| self.temporal_coordinate(date, output))
            .collect::<Result<Vec<_>, _>>()?;
        let dag = build_graph(&dependency_cone)?;
        let joint = dag.temporal_covariance(&coordinates)?;
        let dates = target_selection.len();
        let target_covariance =
            Array2::from_shape_fn((dates, dates), |(row, column)| joint[(row, column)]);
        let reference_covariance = Array2::from_shape_fn((dates, dates), |(row, column)| {
            joint[(dates + row, dates + column)]
        });
        let target_reference_covariance =
            Array2::from_shape_fn((dates, dates), |(row, column)| joint[(row, dates + column)]);
        let coincident = target_selection == reference_selection;
        let difference_covariance = Array2::from_shape_fn((dates, dates), |(row, column)| {
            if coincident {
                return 0.0;
            }
            target_covariance[(row, column)] + reference_covariance[(row, column)]
                - target_reference_covariance[(row, column)]
                - target_reference_covariance[(column, row)]
        });
        Ok(ReferenceDifferenceCovarianceReplay {
            target_covariance,
            reference_covariance,
            target_reference_covariance,
            difference_covariance,
            dependency_cone,
            reference_signature: reference_selection_signature(
                target_selection,
                reference_selection,
            ),
            source_cache_peak_bytes: 0,
            source_factor_receipt: [0; 32],
            support_receipt: [0; 32],
            effective_looks: None,
            target_disposition: ReplayStatus::Valid,
            reference_disposition: ReplayStatus::Valid,
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
        let cone = self.spatial_query_cone(selection, query.microbatch, 1)?;
        let mut dependency_cone =
            self.estimate_dependency_cone_for_spatial_query(selection, query.source_rank, &cone)?;
        if selection.iter().all(|(date, _)| date.get() == 0) {
            if dependency_cone.total_bytes > query.byte_cap {
                return Err(SequentialReplayError::Budget(dependency_cone));
            }
            return Ok(TemporalCovarianceReplay {
                covariance: Array2::zeros((selection.len(), selection.len())),
                dependency_cone,
                source_cache_peak_bytes: 0,
            });
        }
        self.validate_provider_identity(provider.identity())?;
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
        let mut provider = QuerySourceCache::new(provider);

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

        let mut covariance = Array2::<f64>::zeros((selected, selected));
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
                    &mut provider,
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
                    &mut provider,
                    &mut adjoints.compressed,
                    &mut source_adjoints,
                )?;
            }
            for root in source_adjoints.values() {
                for row in 0..selected {
                    for column in 0..selected {
                        covariance[(row, column)] += (0..root.nrows())
                            .map(|basis| root[(basis, row)] * root[(basis, column)])
                            .sum::<f64>();
                    }
                }
            }
            provider.clear_block();
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
            source_cache_peak_bytes: provider.peak_payload_bytes(),
        })
    }

    /// Replay one target/reference pair from the persisted sequential operator.
    ///
    /// Both paths are seeded into one reverse pass.  The source cache is shared
    /// across the pair and evicted at each block boundary, so carried parents
    /// are composed once and shared-source cross covariance is retained.  The
    /// reference signature is part of the receipt and prevents a cached
    /// reference result being reused for another output/date selection.
    #[allow(clippy::too_many_lines)]
    pub fn replay_reference_difference_covariance_from_provider_with_source_correlation<P>(
        &self,
        target_selection: &[(GlobalDateId, usize)],
        reference_selection: &[(GlobalDateId, usize)],
        query: DependencyConeQuery,
        source_correlation: SourceCorrelationModel,
        branch_tolerance: f64,
        provider: &mut P,
    ) -> Result<ReferenceDifferenceCovarianceReplay, SequentialReplayError>
    where
        P: SequentialSourceReplayProvider + ?Sized,
    {
        source_correlation.validate()?;
        if target_selection.is_empty()
            || target_selection.len() != reference_selection.len()
            || query.source_rank == 0
            || query.microbatch == 0
            || !branch_tolerance.is_finite()
            || branch_tolerance <= 0.0
        {
            return Err(SequentialReplayError::Invalid(
                "reference provider replay requires aligned selections, positive rank/microbatch, and branch tolerance",
            ));
        }
        let target_outputs = target_selection
            .iter()
            .map(|&(_, output)| output)
            .collect::<BTreeSet<_>>();
        let reference_outputs = reference_selection
            .iter()
            .map(|&(_, output)| output)
            .collect::<BTreeSet<_>>();
        if target_outputs.len() != 1 || reference_outputs.len() != 1 {
            return Err(SequentialReplayError::Invalid(
                "reference provider replay requires one target and one reference pixel",
            ));
        }
        let target_dates = target_selection
            .iter()
            .map(|&(date, _)| date)
            .collect::<Vec<_>>();
        let reference_dates = reference_selection
            .iter()
            .map(|&(date, _)| date)
            .collect::<Vec<_>>();
        if target_dates != reference_dates
            || target_dates.first().is_none_or(|date| date.get() != 0)
            || !target_dates
                .windows(2)
                .all(|pair| pair[0].get() < pair[1].get())
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::InvalidReference,
                "reference provider replay requires identical increasing dates with acquisition zero first",
            ));
        }

        let ReferenceDifferenceQueryPlan {
            selection,
            cone,
            estimate: mut dependency_cone,
        } = self.plan_reference_difference_query(
            target_selection,
            reference_selection,
            query,
            source_correlation,
        )?;
        if selection.iter().all(|(date, _)| date.get() == 0) {
            let dates = target_selection.len();
            return Ok(ReferenceDifferenceCovarianceReplay {
                target_covariance: Array2::zeros((dates, dates)),
                reference_covariance: Array2::zeros((dates, dates)),
                target_reference_covariance: Array2::zeros((dates, dates)),
                difference_covariance: Array2::zeros((dates, dates)),
                dependency_cone,
                reference_signature: reference_selection_signature(
                    target_selection,
                    reference_selection,
                ),
                source_cache_peak_bytes: 0,
                source_factor_receipt: empty_query_receipt(
                    b"dolphinrust:empty-source-factor-query:v1",
                ),
                support_receipt: empty_query_receipt(b"dolphinrust:empty-support-query:v1"),
                effective_looks: None,
                target_disposition: ReplayStatus::Valid,
                reference_disposition: ReplayStatus::Valid,
            });
        }
        self.validate_provider_identity(provider.identity())?;
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

        let mut provider = QuerySourceCache::new(provider);
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

        let mut covariance = Array2::<f64>::zeros((selected, selected));
        let mut effective_support = BTreeSet::new();
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
                    &mut provider,
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
                    &mut provider,
                    &mut adjoints.compressed,
                    &mut source_adjoints,
                )?;
            }
            let source_adjoints = global_source_adjoints(self, source_adjoints)?;
            effective_support.extend(source_adjoints.keys().copied());
            contract_source_adjoints(&mut covariance, &source_adjoints, source_correlation)?;
            provider.clear_block();
        }
        if covariance.iter().any(|value| !value.is_finite()) {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::NonFiniteReplayState,
                "streamed reference covariance contraction is non-finite",
            ));
        }
        let dates = target_selection.len();
        let target_covariance = covariance.slice(ndarray::s![..dates, ..dates]).to_owned();
        let reference_covariance = covariance.slice(ndarray::s![dates.., dates..]).to_owned();
        let target_reference_covariance =
            covariance.slice(ndarray::s![..dates, dates..]).to_owned();
        let difference_covariance = if target_selection == reference_selection {
            Array2::zeros((dates, dates))
        } else {
            &target_covariance + &reference_covariance
                - &target_reference_covariance
                - target_reference_covariance.t()
        };
        let mut replay = ReferenceDifferenceCovarianceReplay {
            target_covariance,
            reference_covariance,
            target_reference_covariance,
            difference_covariance,
            dependency_cone,
            reference_signature: reference_selection_signature(
                target_selection,
                reference_selection,
            ),
            source_cache_peak_bytes: provider.peak_payload_bytes(),
            source_factor_receipt: provider.source_factor_receipt(),
            support_receipt: provider.support_receipt(),
            effective_looks: None,
            target_disposition: ReplayStatus::Valid,
            reference_disposition: ReplayStatus::Valid,
        };
        attach_source_correlation_receipt(&mut replay, &effective_support, source_correlation)?;
        Ok(replay)
    }

    /// Replay one target/reference pair with the production-default
    /// exponential source correlation.
    #[allow(clippy::too_many_lines)]
    pub fn replay_reference_difference_covariance_from_provider<P>(
        &self,
        target_selection: &[(GlobalDateId, usize)],
        reference_selection: &[(GlobalDateId, usize)],
        query: DependencyConeQuery,
        branch_tolerance: f64,
        provider: &mut P,
    ) -> Result<ReferenceDifferenceCovarianceReplay, SequentialReplayError>
    where
        P: SequentialSourceReplayProvider + ?Sized,
    {
        self.replay_reference_difference_covariance_from_provider_with_source_correlation(
            target_selection,
            reference_selection,
            query,
            SourceCorrelationModel::ExponentialEuclidean {
                distance_scale_pixels: 1.5,
            },
            branch_tolerance,
            provider,
        )
    }

    /// Jointly replay one target/reference pair captured in separate tile topologies.
    ///
    /// The two reverse graphs retain independent record nodes but contract
    /// overlapping global primitive-source coordinates together. Provider and
    /// topology memory are admitted before the first source read, and both
    /// per-block source caches are discarded after their shared-source cross
    /// terms are accumulated.
    ///
    /// # Errors
    /// Returns a fail-closed identity, topology, reference, replay-state, or
    /// byte-budget error.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn replay_cross_topology_reference_difference_covariance_from_providers_with_source_correlation<
        T,
        R,
    >(
        &self,
        target_selection: &[(GlobalDateId, usize)],
        target_provider: &mut T,
        reference_topology: &Self,
        reference_selection: &[(GlobalDateId, usize)],
        reference_provider: &mut R,
        query: DependencyConeQuery,
        source_correlation: SourceCorrelationModel,
        branch_tolerance: f64,
    ) -> Result<ReferenceDifferenceCovarianceReplay, SequentialReplayError>
    where
        T: SequentialSourceReplayProvider + ?Sized,
        R: SequentialSourceReplayProvider + ?Sized,
    {
        source_correlation.validate()?;
        if target_provider.identity() != reference_provider.identity() {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceIdentityMismatch,
                "cross-topology providers do not have one exact source/model identity",
            ));
        }
        self.validate_provider_identity(target_provider.identity())?;
        reference_topology.validate_provider_identity(reference_provider.identity())?;
        self.validate_cross_topology(reference_topology)?;
        self.validate_cross_reference_selections(target_selection, reference_selection, query)?;
        if self.same_replay_graph(reference_topology) {
            let ReferenceDifferenceQueryPlan {
                estimate: mut aggregate,
                ..
            } = self.plan_reference_difference_query(
                target_selection,
                reference_selection,
                query,
                source_correlation,
            )?;
            let dates = target_selection.len() as u64;
            let retained_output_bytes =
                checked_mul(checked_mul(checked_mul(4, dates)?, dates)?, 8)?;
            aggregate.covariance_bytes =
                checked_add(aggregate.covariance_bytes, retained_output_bytes)?;
            aggregate.total_bytes = checked_add(aggregate.total_bytes, retained_output_bytes)?;
            let retained_block_ids = checked_mul(
                aggregate.block_ids.capacity() as u64,
                size_of::<GlobalBlockId>() as u64,
            )?;
            aggregate.operator_bytes = checked_add(aggregate.operator_bytes, retained_block_ids)?;
            aggregate.total_bytes = checked_add(aggregate.total_bytes, retained_block_ids)?;
            aggregate.provider_bytes = checked_add(
                target_provider.maximum_resident_bytes(),
                reference_provider.maximum_resident_bytes(),
            )?;
            aggregate.total_bytes = checked_add(aggregate.total_bytes, aggregate.provider_bytes)?;
            if aggregate.total_bytes > query.byte_cap {
                return Err(SequentialReplayError::Budget(aggregate));
            }
            let replay_query = DependencyConeQuery {
                byte_cap: aggregate.total_bytes,
                ..query
            };
            let mut target_replay = self
                .replay_reference_difference_covariance_from_provider_with_source_correlation(
                    target_selection,
                    reference_selection,
                    replay_query,
                    source_correlation,
                    branch_tolerance,
                    target_provider,
                )?;
            let reference_replay = self
                .replay_reference_difference_covariance_from_provider_with_source_correlation(
                    target_selection,
                    reference_selection,
                    replay_query,
                    source_correlation,
                    branch_tolerance,
                    reference_provider,
                )?;
            if target_replay.target_covariance != reference_replay.target_covariance
                || target_replay.reference_covariance != reference_replay.reference_covariance
                || target_replay.target_reference_covariance
                    != reference_replay.target_reference_covariance
                || target_replay.difference_covariance != reference_replay.difference_covariance
                || target_replay.reference_signature != reference_replay.reference_signature
                || target_replay.source_factor_receipt != reference_replay.source_factor_receipt
                || target_replay.support_receipt != reference_replay.support_receipt
                || target_replay.effective_looks != reference_replay.effective_looks
                || target_replay.target_disposition != reference_replay.target_disposition
                || target_replay.reference_disposition != reference_replay.reference_disposition
            {
                return Err(SequentialReplayError::Provider(
                    ReplayStatus::SourceIdentityMismatch,
                    "same-topology providers do not have exact replay artifact receipts",
                ));
            }
            target_replay.dependency_cone = aggregate;
            return Ok(target_replay);
        }
        if !branch_tolerance.is_finite() || branch_tolerance <= 0.0 {
            return Err(SequentialReplayError::Invalid(
                "cross-topology replay requires a positive finite branch tolerance",
            ));
        }

        let target_cone = self.spatial_query_cone(target_selection, query.microbatch, 1)?;
        let reference_cone =
            reference_topology.spatial_query_cone(reference_selection, query.microbatch, 1)?;
        let mut target_estimate = self.estimate_dependency_cone_for_spatial_query(
            target_selection,
            query.source_rank,
            &target_cone,
        )?;
        let mut reference_estimate = reference_topology
            .estimate_dependency_cone_for_spatial_query(
                reference_selection,
                query.source_rank,
                &reference_cone,
            )?;
        let selected =
            target_selection
                .len()
                .checked_mul(2)
                .ok_or(SequentialReplayError::Invalid(
                    "cross-topology selection size overflows usize",
                ))?;
        let selected_u64 = u64::try_from(selected).map_err(|_| {
            SequentialReplayError::Invalid("cross-topology selection size exceeds u64")
        })?;
        let joint_covariance_bytes = checked_mul(checked_mul(selected_u64, selected_u64)?, 8)?;
        for estimate in [&mut target_estimate, &mut reference_estimate] {
            let doubled_frontier = checked_mul(estimate.frontier_bytes, 2)?;
            estimate.total_bytes = estimate
                .total_bytes
                .checked_sub(estimate.frontier_bytes)
                .and_then(|value| value.checked_sub(estimate.covariance_bytes))
                .ok_or(SequentialReplayError::Invalid(
                    "cross-topology dependency estimate underflowed",
                ))?;
            estimate.frontier_bytes = doubled_frontier;
            estimate.covariance_bytes = joint_covariance_bytes;
            estimate.total_bytes = checked_add(
                estimate.total_bytes,
                checked_add(doubled_frontier, joint_covariance_bytes)?,
            )?;
        }
        let target_effective_support_bytes =
            self.effective_support_reservation_bytes(&target_cone)?;
        target_estimate.support_bytes = checked_add(
            target_estimate.support_bytes,
            target_effective_support_bytes,
        )?;
        target_estimate.total_bytes =
            checked_add(target_estimate.total_bytes, target_effective_support_bytes)?;
        let reference_effective_support_bytes =
            reference_topology.effective_support_reservation_bytes(&reference_cone)?;
        reference_estimate.support_bytes = checked_add(
            reference_estimate.support_bytes,
            reference_effective_support_bytes,
        )?;
        reference_estimate.total_bytes = checked_add(
            reference_estimate.total_bytes,
            reference_effective_support_bytes,
        )?;
        let combined_support = target_cone
            .active_sources
            .iter()
            .zip(&reference_cone.active_sources)
            .map(|(target, reference)| target.len().saturating_add(reference.len()))
            .max()
            .unwrap_or(0) as u64;
        let source_correlation_workspace_bytes =
            source_correlation_workspace_bytes(combined_support, selected_u64, source_correlation)?;
        let target_global_map_control_bytes =
            self.global_source_map_control_reservation_bytes(&target_cone)?;
        let reference_global_map_control_bytes =
            reference_topology.global_source_map_control_reservation_bytes(&reference_cone)?;
        let merged_global_map_control_bytes = checked_add(
            size_of::<BTreeMap<(u64, u64), Array2<f64>>>() as u64,
            checked_mul(
                combined_support,
                btree_record_reservation_bytes::<(u64, u64), Array2<f64>>(),
            )?,
        )?;
        let global_source_map_control_bytes = checked_add(
            checked_add(
                target_global_map_control_bytes,
                reference_global_map_control_bytes,
            )?,
            merged_global_map_control_bytes,
        )?;
        let mut block_ids = target_estimate.block_ids.clone();
        block_ids.extend(reference_estimate.block_ids.iter().copied());
        block_ids.sort_unstable_by_key(|block| block.get());
        block_ids.dedup();
        let provider_bytes = checked_add(
            target_provider.maximum_resident_bytes(),
            reference_provider.maximum_resident_bytes(),
        )?;
        let mut dependency_cone = DependencyConeEstimate {
            block_ids,
            frontier_bytes: checked_add(
                target_estimate.frontier_bytes,
                reference_estimate.frontier_bytes,
            )?,
            source_influence_bytes: checked_add(
                checked_add(
                    target_estimate.source_influence_bytes,
                    reference_estimate.source_influence_bytes,
                )?,
                global_source_map_control_bytes,
            )?,
            source_correlation_workspace_bytes,
            source_window_bytes: checked_add(
                target_estimate.source_window_bytes,
                reference_estimate.source_window_bytes,
            )?,
            operator_bytes: checked_add(
                target_estimate.operator_bytes,
                reference_estimate.operator_bytes,
            )?,
            baseline_bytes: checked_add(
                target_estimate.baseline_bytes,
                reference_estimate.baseline_bytes,
            )?,
            support_bytes: checked_add(
                target_estimate.support_bytes,
                reference_estimate.support_bytes,
            )?,
            covariance_bytes: checked_add(
                target_estimate.covariance_bytes,
                reference_estimate.covariance_bytes,
            )?,
            provider_bytes,
            total_bytes: 0,
        };
        dependency_cone.total_bytes = checked_add(
            checked_add(
                checked_add(target_estimate.total_bytes, reference_estimate.total_bytes)?,
                checked_add(
                    source_correlation_workspace_bytes,
                    global_source_map_control_bytes,
                )?,
            )?,
            provider_bytes,
        )?;
        if dependency_cone.total_bytes > query.byte_cap {
            return Err(SequentialReplayError::Budget(dependency_cone));
        }
        if target_selection.iter().any(|(date, _)| date.get() != 0) {
            let expected_target_rank = self.expected_source_rank(&target_estimate.block_ids)?;
            let expected_reference_rank =
                reference_topology.expected_source_rank(&reference_estimate.block_ids)?;
            if query.source_rank != expected_target_rank
                || query.source_rank != expected_reference_rank
            {
                return Err(SequentialReplayError::Invalid(
                    "declared source rank does not match both cross-topology block factors",
                ));
            }
        }

        let dates = target_selection.len();
        let target_output = target_selection[0].1;
        let reference_output = reference_selection[0].1;
        let target_global_output = self.global_output_coordinate(target_output)?;
        let reference_global_output =
            reference_topology.global_output_coordinate(reference_output)?;
        let target_namespace =
            self.id_namespace
                .as_ref()
                .ok_or(SequentialReplayError::Unsupported(
                    ReplayStatus::UnsupportedSourceIdentity,
                ))?;
        let reference_namespace =
            reference_topology
                .id_namespace
                .as_ref()
                .ok_or(SequentialReplayError::Unsupported(
                    ReplayStatus::UnsupportedSourceIdentity,
                ))?;
        let reference_signature = cross_topology_reference_selection_signature(
            target_namespace,
            target_selection,
            target_global_output,
            reference_namespace,
            reference_selection,
            reference_global_output,
        );
        if target_selection.iter().all(|(date, _)| date.get() == 0) {
            return Ok(ReferenceDifferenceCovarianceReplay {
                target_covariance: Array2::zeros((dates, dates)),
                reference_covariance: Array2::zeros((dates, dates)),
                target_reference_covariance: Array2::zeros((dates, dates)),
                difference_covariance: Array2::zeros((dates, dates)),
                dependency_cone,
                reference_signature,
                source_cache_peak_bytes: 0,
                source_factor_receipt: empty_query_receipt(
                    b"dolphinrust:empty-cross-source-factor-query:v1",
                ),
                support_receipt: empty_query_receipt(b"dolphinrust:empty-cross-support-query:v1"),
                effective_looks: None,
                target_disposition: ReplayStatus::Valid,
                reference_disposition: ReplayStatus::Valid,
            });
        }

        let mut target_adjoints = StreamingAdjoints {
            phase: vec![BTreeMap::new(); self.blocks.len()],
            compressed: vec![BTreeMap::new(); self.blocks.len()],
        };
        let mut reference_adjoints = StreamingAdjoints {
            phase: vec![BTreeMap::new(); reference_topology.blocks.len()],
            compressed: vec![BTreeMap::new(); reference_topology.blocks.len()],
        };
        self.seed_cross_topology_adjoints(target_selection, 0, selected, &mut target_adjoints)?;
        reference_topology.seed_cross_topology_adjoints(
            reference_selection,
            dates,
            selected,
            &mut reference_adjoints,
        )?;
        let mut target_provider = QuerySourceCache::new(target_provider);
        let mut reference_provider = QuerySourceCache::new(reference_provider);
        let mut covariance = Array2::<f64>::zeros((selected, selected));
        let mut source_cache_peak_bytes = 0_u64;
        let mut effective_support = BTreeSet::new();
        for block_index in (0..self.blocks.len()).rev() {
            let target_block = &self.blocks[block_index];
            let reference_block = &reference_topology.blocks[block_index];
            let mut target_roots: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
            let mut reference_roots: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
            for &native in &target_cone.required_compressed[block_index] {
                let compressed = target_adjoints.compressed[block_index]
                    .remove(&native)
                    .unwrap_or_else(|| Array2::zeros((2, selected)));
                self.propagate_compression_adjoint(
                    target_block,
                    native,
                    compressed.view(),
                    query.source_rank,
                    branch_tolerance,
                    &mut target_provider,
                    &mut target_adjoints.phase[block_index],
                    &mut target_roots,
                )?;
            }
            for &output in &target_cone.active_outputs[block_index] {
                let phase = target_adjoints.phase[block_index]
                    .remove(&output)
                    .unwrap_or_else(|| Array2::zeros((target_block.phase_dimension, selected)));
                self.propagate_phase_adjoint(
                    target_block,
                    output,
                    phase.view(),
                    query.source_rank,
                    branch_tolerance,
                    &mut target_provider,
                    &mut target_adjoints.compressed,
                    &mut target_roots,
                )?;
            }
            for &native in &reference_cone.required_compressed[block_index] {
                let compressed = reference_adjoints.compressed[block_index]
                    .remove(&native)
                    .unwrap_or_else(|| Array2::zeros((2, selected)));
                reference_topology.propagate_compression_adjoint(
                    reference_block,
                    native,
                    compressed.view(),
                    query.source_rank,
                    branch_tolerance,
                    &mut reference_provider,
                    &mut reference_adjoints.phase[block_index],
                    &mut reference_roots,
                )?;
            }
            for &output in &reference_cone.active_outputs[block_index] {
                let phase = reference_adjoints.phase[block_index]
                    .remove(&output)
                    .unwrap_or_else(|| Array2::zeros((reference_block.phase_dimension, selected)));
                reference_topology.propagate_phase_adjoint(
                    reference_block,
                    output,
                    phase.view(),
                    query.source_rank,
                    branch_tolerance,
                    &mut reference_provider,
                    &mut reference_adjoints.compressed,
                    &mut reference_roots,
                )?;
            }

            for &target_native in target_roots.keys() {
                let global = self.global_native_coordinate(target_native)?;
                let Some(reference_native) = reference_topology.native_index_for_global(global)?
                else {
                    continue;
                };
                if !reference_roots.contains_key(&reference_native) {
                    continue;
                }
                let target_source = self.resolve_source_checked(
                    target_block,
                    target_native,
                    query.source_rank,
                    &mut target_provider,
                )?;
                let reference_source = reference_topology.resolve_source_checked(
                    reference_block,
                    reference_native,
                    query.source_rank,
                    &mut reference_provider,
                )?;
                if target_source.id != reference_source.id
                    || target_source.content_digest != reference_source.content_digest
                    || target_source.samples != reference_source.samples
                    || target_source.factor.component_ids()
                        != reference_source.factor.component_ids()
                    || target_source.factor.model_hash() != reference_source.factor.model_hash()
                    || target_source.factor.numeric_receipt_digest()
                        != reference_source.factor.numeric_receipt_digest()
                    || target_source.factor.lower() != reference_source.factor.lower()
                {
                    return Err(SequentialReplayError::Provider(
                        ReplayStatus::SourceIdentityMismatch,
                        "overlapping cross-topology primitive source factors differ",
                    ));
                }
            }
            let target_roots = global_source_adjoints(self, target_roots)?;
            let reference_roots = global_source_adjoints(reference_topology, reference_roots)?;
            let source_adjoints = merge_global_source_adjoints(target_roots, reference_roots)?;
            effective_support.extend(source_adjoints.keys().copied());
            contract_source_adjoints(&mut covariance, &source_adjoints, source_correlation)?;
            source_cache_peak_bytes = source_cache_peak_bytes.max(checked_add(
                target_provider.current_payload_bytes(),
                reference_provider.current_payload_bytes(),
            )?);
            target_provider.clear_block();
            reference_provider.clear_block();
        }
        if covariance.iter().any(|value| !value.is_finite()) {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::NonFiniteReplayState,
                "cross-topology covariance contraction is non-finite",
            ));
        }
        let target_covariance = covariance.slice(ndarray::s![..dates, ..dates]).to_owned();
        let mut reference_covariance = covariance.slice(ndarray::s![dates.., dates..]).to_owned();
        let mut target_reference_covariance =
            covariance.slice(ndarray::s![..dates, dates..]).to_owned();
        let coincident = target_global_output == reference_global_output;
        if coincident {
            if target_covariance
                .iter()
                .zip(reference_covariance.iter())
                .chain(
                    target_covariance
                        .iter()
                        .zip(target_reference_covariance.iter()),
                )
                .any(|(&left, &right)| !scalar_close(left, right, branch_tolerance))
            {
                return Err(SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "globally coincident tile outputs do not replay identically",
                ));
            }
            reference_covariance = target_covariance.clone();
            target_reference_covariance = target_covariance.clone();
        }
        let difference_covariance = if coincident {
            Array2::zeros((dates, dates))
        } else {
            &target_covariance + &reference_covariance
                - &target_reference_covariance
                - target_reference_covariance.t()
        };
        let mut replay = ReferenceDifferenceCovarianceReplay {
            target_covariance,
            reference_covariance,
            target_reference_covariance,
            difference_covariance,
            dependency_cone,
            reference_signature,
            source_cache_peak_bytes,
            source_factor_receipt: combined_query_receipt(
                b"dolphinrust:cross-source-factor-query:v1",
                target_provider.source_factor_receipt(),
                reference_provider.source_factor_receipt(),
            ),
            support_receipt: combined_query_receipt(
                b"dolphinrust:cross-support-query:v1",
                target_provider.support_receipt(),
                reference_provider.support_receipt(),
            ),
            effective_looks: None,
            target_disposition: ReplayStatus::Valid,
            reference_disposition: ReplayStatus::Valid,
        };
        attach_source_correlation_receipt(&mut replay, &effective_support, source_correlation)?;
        Ok(replay)
    }

    /// Replay a cross-topology target/reference pair with the
    /// production-default exponential source correlation.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn replay_cross_topology_reference_difference_covariance_from_providers<T, R>(
        &self,
        target_selection: &[(GlobalDateId, usize)],
        target_provider: &mut T,
        reference_topology: &Self,
        reference_selection: &[(GlobalDateId, usize)],
        reference_provider: &mut R,
        query: DependencyConeQuery,
        branch_tolerance: f64,
    ) -> Result<ReferenceDifferenceCovarianceReplay, SequentialReplayError>
    where
        T: SequentialSourceReplayProvider + ?Sized,
        R: SequentialSourceReplayProvider + ?Sized,
    {
        self.replay_cross_topology_reference_difference_covariance_from_providers_with_source_correlation(
            target_selection,
            target_provider,
            reference_topology,
            reference_selection,
            reference_provider,
            query,
            SourceCorrelationModel::ExponentialEuclidean {
                distance_scale_pixels: 1.5,
            },
            branch_tolerance,
        )
    }

    fn same_replay_graph(&self, other: &Self) -> bool {
        self.blocks == other.blocks
            && self.num_real_dates == other.num_real_dates
            && self.native_shape == other.native_shape
            && self.output_shape == other.output_shape
            && self.half_window == other.half_window
            && self.strides == other.strides
            && self.native_validity == other.native_validity
            && self.id_namespace == other.id_namespace
            && self.estimator_branch == other.estimator_branch
            && self.normalized_config_digest == other.normalized_config_digest
    }

    fn validate_cross_reference_selections(
        &self,
        target_selection: &[(GlobalDateId, usize)],
        reference_selection: &[(GlobalDateId, usize)],
        query: DependencyConeQuery,
    ) -> Result<(), SequentialReplayError> {
        if target_selection.is_empty()
            || target_selection.len() != reference_selection.len()
            || query.source_rank == 0
            || query.microbatch == 0
            || target_selection
                .iter()
                .map(|(_, output)| output)
                .collect::<BTreeSet<_>>()
                .len()
                != 1
            || reference_selection
                .iter()
                .map(|(_, output)| output)
                .collect::<BTreeSet<_>>()
                .len()
                != 1
        {
            return Err(SequentialReplayError::Invalid(
                "cross-topology reference replay requires aligned nonempty single-pixel selections and positive rank/microbatch",
            ));
        }
        let target_dates = target_selection
            .iter()
            .map(|(date, _)| *date)
            .collect::<Vec<_>>();
        let reference_dates = reference_selection
            .iter()
            .map(|(date, _)| *date)
            .collect::<Vec<_>>();
        if target_dates != reference_dates
            || target_dates.first().is_none_or(|date| date.get() != 0)
            || !target_dates
                .windows(2)
                .all(|pair| pair[0].get() < pair[1].get())
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::InvalidReference,
                "cross-topology reference replay requires identical increasing dates with acquisition zero first",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn validate_cross_topology(&self, other: &Self) -> Result<(), SequentialReplayError> {
        let left = self
            .id_namespace
            .as_ref()
            .ok_or(SequentialReplayError::Unsupported(
                ReplayStatus::UnsupportedSourceIdentity,
            ))?;
        let right = other
            .id_namespace
            .as_ref()
            .ok_or(SequentialReplayError::Unsupported(
                ReplayStatus::UnsupportedSourceIdentity,
            ))?;
        if left.burst_id != right.burst_id
            || left.source_manifest_digest != right.source_manifest_digest
            || left.source_model_version_digest != right.source_model_version_digest
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceIdentityMismatch,
                "cross-topology source namespace identity differs",
            ));
        }
        if self.num_real_dates != other.num_real_dates
            || self.blocks.len() != other.blocks.len()
            || self.half_window != other.half_window
            || self.strides != other.strides
            || self.estimator_branch != other.estimator_branch
            || self.normalized_config_digest != other.normalized_config_digest
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::ReplayStateMismatch,
                "cross-topology date, estimator, or support configuration differs",
            ));
        }
        for (left_block, right_block) in self.blocks.iter().zip(&other.blocks) {
            let left_parents = left_block
                .carried_parent_ids
                .iter()
                .map(|parent| self.block(*parent).map(|block| block.generation))
                .collect::<Result<Vec<_>, _>>()?;
            let right_parents = right_block
                .carried_parent_ids
                .iter()
                .map(|parent| other.block(*parent).map(|block| block.generation))
                .collect::<Result<Vec<_>, _>>()?;
            if left_block.generation != right_block.generation
                || left_block.real_date_start != right_block.real_date_start
                || left_block.num_real_dates != right_block.num_real_dates
                || left_block.phase_dimension != right_block.phase_dimension
                || left_parents != right_parents
            {
                return Err(SequentialReplayError::Provider(
                    ReplayStatus::ReplayStateMismatch,
                    "cross-topology sequential block graphs differ",
                ));
            }
        }
        let left_row_stop = left
            .native_origin
            .0
            .checked_add(self.native_shape.0 as u64)
            .ok_or(SequentialReplayError::Invalid(
                "cross-topology left native row extent overflows u64",
            ))?;
        let left_col_stop = left
            .native_origin
            .1
            .checked_add(self.native_shape.1 as u64)
            .ok_or(SequentialReplayError::Invalid(
                "cross-topology left native column extent overflows u64",
            ))?;
        let right_row_stop = right
            .native_origin
            .0
            .checked_add(other.native_shape.0 as u64)
            .ok_or(SequentialReplayError::Invalid(
                "cross-topology right native row extent overflows u64",
            ))?;
        let right_col_stop = right
            .native_origin
            .1
            .checked_add(other.native_shape.1 as u64)
            .ok_or(SequentialReplayError::Invalid(
                "cross-topology right native column extent overflows u64",
            ))?;
        for row in
            left.native_origin.0.max(right.native_origin.0)..left_row_stop.min(right_row_stop)
        {
            for column in
                left.native_origin.1.max(right.native_origin.1)..left_col_stop.min(right_col_stop)
            {
                let left_index = self.native_index_for_global((row, column))?.ok_or(
                    SequentialReplayError::Invalid("overlap coordinate is outside left tile"),
                )?;
                let right_index = other.native_index_for_global((row, column))?.ok_or(
                    SequentialReplayError::Invalid("overlap coordinate is outside right tile"),
                )?;
                if self.native_validity[left_index] != other.native_validity[right_index] {
                    return Err(SequentialReplayError::Provider(
                        ReplayStatus::ReplayStateMismatch,
                        "cross-topology native validity masks differ on their overlap",
                    ));
                }
            }
        }
        Ok(())
    }

    fn expected_source_rank(
        &self,
        block_ids: &[GlobalBlockId],
    ) -> Result<usize, SequentialReplayError> {
        block_ids
            .iter()
            .map(|&block| self.block(block).map(|item| 2 * item.num_real_dates))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or(SequentialReplayError::Invalid(
                "dependency cone contains no source block",
            ))
    }

    fn seed_cross_topology_adjoints(
        &self,
        selection: &[(GlobalDateId, usize)],
        column_offset: usize,
        selected: usize,
        adjoints: &mut StreamingAdjoints,
    ) -> Result<(), SequentialReplayError> {
        for (selection_column, &(date, output)) in selection.iter().enumerate() {
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
            phase[(reduced_component, column_offset + selection_column)] += 1.0;
        }
        Ok(())
    }

    fn global_native_coordinate(
        &self,
        native_index: usize,
    ) -> Result<(u64, u64), SequentialReplayError> {
        if native_index >= self.native_area {
            return Err(SequentialReplayError::Invalid(
                "cross-topology native index is outside its grid",
            ));
        }
        let namespace = self
            .id_namespace
            .as_ref()
            .ok_or(SequentialReplayError::Unsupported(
                ReplayStatus::UnsupportedSourceIdentity,
            ))?;
        Ok((
            namespace
                .native_origin
                .0
                .checked_add((native_index / self.native_shape.1) as u64)
                .ok_or(SequentialReplayError::Invalid(
                    "cross-topology native row coordinate overflows u64",
                ))?,
            namespace
                .native_origin
                .1
                .checked_add((native_index % self.native_shape.1) as u64)
                .ok_or(SequentialReplayError::Invalid(
                    "cross-topology native column coordinate overflows u64",
                ))?,
        ))
    }

    fn native_index_for_global(
        &self,
        coordinate: (u64, u64),
    ) -> Result<Option<usize>, SequentialReplayError> {
        let namespace = self
            .id_namespace
            .as_ref()
            .ok_or(SequentialReplayError::Unsupported(
                ReplayStatus::UnsupportedSourceIdentity,
            ))?;
        let Some(row) = coordinate.0.checked_sub(namespace.native_origin.0) else {
            return Ok(None);
        };
        let Some(column) = coordinate.1.checked_sub(namespace.native_origin.1) else {
            return Ok(None);
        };
        if row >= self.native_shape.0 as u64 || column >= self.native_shape.1 as u64 {
            return Ok(None);
        }
        Ok(Some(row as usize * self.native_shape.1 + column as usize))
    }

    fn global_output_coordinate(
        &self,
        output_index: usize,
    ) -> Result<(u64, u64), SequentialReplayError> {
        if output_index >= self.output_area {
            return Err(SequentialReplayError::Invalid(
                "cross-topology output index is outside its grid",
            ));
        }
        let namespace = self
            .id_namespace
            .as_ref()
            .ok_or(SequentialReplayError::Unsupported(
                ReplayStatus::UnsupportedSourceIdentity,
            ))?;
        Ok((
            namespace
                .output_origin
                .0
                .checked_add((output_index / self.output_shape.1) as u64)
                .ok_or(SequentialReplayError::Invalid(
                    "cross-topology output row coordinate overflows u64",
                ))?,
            namespace
                .output_origin
                .1
                .checked_add((output_index % self.output_shape.1) as u64)
                .ok_or(SequentialReplayError::Invalid(
                    "cross-topology output column coordinate overflows u64",
                ))?,
        ))
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

    fn validate_provider_identity_for_block(
        &self,
        block: &SequentialReplayBlock,
        identity: &SequentialSourceProviderIdentity,
    ) -> Result<(), SequentialReplayError> {
        if identity.source_model_hash.iter().all(|byte| *byte == 0) {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceModelUnavailable,
                "generation source provider has no proper-complex model digest",
            ));
        }
        if identity.source_manifest_digest != self.generation_source_manifest_digest(block.id)?
            || self.id_namespace.as_ref().is_none_or(|namespace| {
                identity.source_model_version_digest != namespace.source_model_version_digest
            })
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceIdentityMismatch,
                "source provider generation identity does not match the replay block namespace",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn validate_operator_block_contract(
        &self,
        block: &SequentialReplayBlock,
        stored: &CovarianceOperatorBlock,
        branch_tolerance: f64,
    ) -> Result<(), SequentialReplayError> {
        let mismatch = || {
            SequentialReplayError::Provider(
                ReplayStatus::ReplayStateMismatch,
                "operator block geometry, support, or component identity differs from the planned replay",
            )
        };
        let namespace = self
            .id_namespace
            .as_ref()
            .ok_or(SequentialReplayError::Unsupported(
                ReplayStatus::UnsupportedSourceIdentity,
            ))?;
        let native_grid = CovarianceOperatorGrid {
            row_start: namespace.native_origin.0,
            col_start: namespace.native_origin.1,
            rows: u32::try_from(self.native_shape.0).map_err(|_| mismatch())?,
            cols: u32::try_from(self.native_shape.1).map_err(|_| mismatch())?,
            stride_y: 1,
            stride_x: 1,
        };
        let output_grid = CovarianceOperatorGrid {
            row_start: namespace.output_origin.0,
            col_start: namespace.output_origin.1,
            rows: u32::try_from(self.output_shape.0).map_err(|_| mismatch())?,
            cols: u32::try_from(self.output_shape.1).map_err(|_| mismatch())?,
            stride_y: u32::try_from(self.strides.y).map_err(|_| mismatch())?,
            stride_x: u32::try_from(self.strides.x).map_err(|_| mismatch())?,
        };
        let owned_output_grid = CovarianceOperatorGrid {
            row_start: namespace.owned_output_origin.0,
            col_start: namespace.owned_output_origin.1,
            rows: u32::try_from(namespace.owned_output_shape.0).map_err(|_| mismatch())?,
            cols: u32::try_from(namespace.owned_output_shape.1).map_err(|_| mismatch())?,
            stride_y: output_grid.stride_y,
            stride_x: output_grid.stride_x,
        };
        let rect_support = CovarianceRectSupport {
            half_window_rows: u32::try_from(self.half_window.y).map_err(|_| mismatch())?,
            half_window_cols: u32::try_from(self.half_window.x).map_err(|_| mismatch())?,
            ordering: CovarianceSupportOrdering::RowMajorInwardClampV1,
        };
        let estimator_branch = match self.estimator_branch {
            FixedEstimatorBranch::Evd => CovarianceEstimatorBranch::Evd,
            FixedEstimatorBranch::Emi { .. } => CovarianceEstimatorBranch::Emi,
        };
        if stored.burst_id != namespace.burst_id
            || stored.block_id != block.id.get()
            || stored.native_grid != native_grid
            || stored.output_grid != output_grid
            || stored.owned_output_grid != owned_output_grid
            || stored.rect_support != rect_support
            || stored.reference_date_index != 0
            || stored.estimator_branch != estimator_branch
            || stored.branch_tolerance.to_bits() != branch_tolerance.to_bits()
            || usize::try_from(stored.support_bits_per_output)
                != Ok(self.support_slots_per_output())
        {
            return Err(mismatch());
        }
        let expected_date_stop = block
            .real_date_start
            .get()
            .checked_add(u32::try_from(block.num_real_dates).map_err(|_| mismatch())?)
            .ok_or_else(mismatch)?;
        if !stored
            .source_date_indices
            .iter()
            .copied()
            .eq(block.real_date_start.get()..expected_date_stop)
            || stored.ordered_date_indices != stored.source_date_indices
            || !stored
                .carry_parent_ids
                .iter()
                .copied()
                .eq(block.carried_parent_ids.iter().map(|parent| parent.get()))
        {
            return Err(mismatch());
        }
        if stored.phase_components.len() != block.carried_parent_ids.len() + block.num_real_dates {
            return Err(mismatch());
        }
        for (index, component) in stored.phase_components.iter().enumerate() {
            let expected = if index < block.carried_parent_ids.len() {
                (
                    CovariancePhaseComponentKind::CompressedParent,
                    block.carried_parent_ids[index].get(),
                )
            } else {
                let date = block.real_date_start.get()
                    + u32::try_from(index - block.carried_parent_ids.len())
                        .map_err(|_| mismatch())?;
                (
                    match date {
                        0 => CovariancePhaseComponentKind::GaugeDate,
                        _ => CovariancePhaseComponentKind::RetainedDate,
                    },
                    u64::from(date),
                )
            };
            if (component.kind, component.id) != expected {
                return Err(mismatch());
            }
        }
        for native_index in 0..self.native_area {
            let digest_start = native_index * 32;
            let content_digest: &[u8; 32] = stored.source_content_digests
                [digest_start..digest_start + 32]
                .try_into()
                .map_err(|_| mismatch())?;
            if stored.source_ids[native_index]
                != self
                    .source_id_for_content_digest(block.id, native_index, content_digest)?
                    .get()
                || stored.compressed_node_ids[native_index]
                    != self.compressed_node_id(block.id, native_index)?.get()
                || usize::try_from(stored.nearest_output_map[native_index])
                    != Ok(self.nearest_output_index(native_index)?)
                || packed_bit_value(&stored.native_validity_bits, native_index)
                    != self.native_validity[native_index]
            {
                return Err(mismatch());
            }
        }
        let support_bytes = self.support_slots_per_output().div_ceil(8);
        for output_index in 0..self.output_area {
            if stored.phase_node_ids[output_index]
                != self.phase_node_id(block.id, output_index)?.get()
            {
                return Err(mismatch());
            }
            let start = output_index * support_bytes;
            let actual = &stored.support_bits[start..start + support_bytes];
            let fixed = self.support_slot_validity(output_index)?;
            if fixed
                .iter()
                .enumerate()
                .any(|(slot, valid)| packed_bit_value(actual, slot) && !*valid)
            {
                return Err(mismatch());
            }
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
        for (support_index, &source_pixel) in window.replay.source_pixels.iter().enumerate() {
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
                        window.replay.source_pixels[support_index],
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
        let native_indices =
            self.native_support_indices_for_realized(output_index, &phase.realized_support)?;
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
        let expected_id =
            self.source_id_for_content_digest(block.id, native_index, &source.content_digest)?;
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
        let recomputed_content_digest =
            primitive_source_content_digest(source.samples.iter().copied());
        if source.id != expected_id
            || source.factor.source() != expected_id
            || source.samples.len() != block.num_real_dates
            || source.factor.component_ids() != expected_components
            || source.factor.model_hash() != &identity.source_model_hash
            || source.factor.real_embedding().dim() != (expected_rank, expected_rank)
            || expected_rank > source_rank
            || source.content_digest.iter().all(|byte| *byte == 0)
            || source.content_digest != recomputed_content_digest
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
            || phase.realized_support.len() != self.support_slots_per_output()
            || phase
                .realized_support
                .iter()
                .zip(self.support_slot_validity(output_index)?.iter())
                .any(|(realized, fixed)| *realized && !*fixed)
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
        let mut indices = Vec::with_capacity(self.support_slots_per_output());
        for row in row_start..row_start + window_rows {
            for col in col_start..col_start + window_cols {
                let index = row * self.native_shape.1 + col;
                if self.native_validity[index] {
                    indices.push(index);
                }
            }
        }
        Ok(indices)
    }

    fn native_support_indices_for_realized(
        &self,
        output_index: usize,
        realized: &[bool],
    ) -> Result<Vec<usize>, SequentialReplayError> {
        let fixed = self.support_slot_validity(output_index)?;
        if realized.len() != fixed.len()
            || realized
                .iter()
                .zip(fixed.iter())
                .any(|(selected, valid)| *selected && !*valid)
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::ReplayStateMismatch,
                "captured phase support is outside fixed native validity",
            ));
        }
        let (row_start, col_start) = self.window_origin(output_index);
        let window_cols = 2 * self.half_window.x + 1;
        Ok(realized
            .iter()
            .enumerate()
            .filter(|(_, selected)| **selected)
            .map(|(slot, _)| {
                (row_start + slot / window_cols) * self.native_shape.1
                    + col_start
                    + slot % window_cols
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
        source_manifest_digest: [u8; 32],
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
        covariance_identified_id(
            kind,
            &namespace.burst_id,
            source_manifest_digest,
            namespace.source_model_version_digest,
            major,
            secondary,
            CovarianceOperatorGrid {
                row_start: origin.0,
                col_start: origin.1,
                rows: u32::try_from(shape.0).expect("validated replay rows fit u32"),
                cols: u32::try_from(shape.1).expect("validated replay columns fit u32"),
                stride_y: 1,
                stride_x: 1,
            },
            local,
        )
        .expect("validated replay ID coordinate")
    }
}

/// Route and jointly replay a global target/reference query across tile-scoped providers.
///
/// # Errors
/// Returns a fail-closed routing, identity, topology, replay-state, or byte-budget error.
#[allow(clippy::too_many_lines)]
pub fn replay_global_reference_difference_covariance_from_provider_bundle(
    tiles: &mut [SequentialTileReplayProvider<'_>],
    query: GlobalReferenceCovarianceQuery<'_>,
) -> Result<GlobalReferenceDifferenceCovarianceReplay, SequentialReplayError> {
    if tiles.is_empty()
        || query.burst_id.is_empty()
        || query.ordered_dates.is_empty()
        || query.source_rank == 0
        || !query.branch_tolerance.is_finite()
        || query.branch_tolerance <= 0.0
    {
        return Err(SequentialReplayError::Invalid(
            "global reference replay query is empty or invalid",
        ));
    }
    query.source_correlation.validate()?;
    let (selected, wrapper_bytes, joint_bytes) = global_reference_wrapper_bytes(query)?;
    if wrapper_bytes > query.byte_cap {
        return Err(SequentialReplayError::Budget(DependencyConeEstimate {
            block_ids: Vec::new(),
            frontier_bytes: 0,
            source_influence_bytes: 0,
            source_correlation_workspace_bytes: 0,
            source_window_bytes: 0,
            operator_bytes: 0,
            baseline_bytes: 0,
            support_bytes: 0,
            covariance_bytes: joint_bytes,
            provider_bytes: 0,
            total_bytes: wrapper_bytes,
        }));
    }

    let locate = |coordinate: (u64, u64)| -> Result<(usize, usize), SequentialReplayError> {
        let mut found = None;
        for (index, tile) in tiles.iter().enumerate() {
            let Some(namespace) = tile.topology.id_namespace.as_ref() else {
                continue;
            };
            if namespace.burst_id != query.burst_id {
                continue;
            }
            let owned_row_stop = namespace
                .owned_output_origin
                .0
                .checked_add(namespace.owned_output_shape.0 as u64)
                .ok_or(SequentialReplayError::Invalid(
                    "owned output row extent overflows u64",
                ))?;
            let owned_col_stop = namespace
                .owned_output_origin
                .1
                .checked_add(namespace.owned_output_shape.1 as u64)
                .ok_or(SequentialReplayError::Invalid(
                    "owned output column extent overflows u64",
                ))?;
            if coordinate.0 < namespace.owned_output_origin.0
                || coordinate.0 >= owned_row_stop
                || coordinate.1 < namespace.owned_output_origin.1
                || coordinate.1 >= owned_col_stop
            {
                continue;
            }
            let local_row = coordinate.0.checked_sub(namespace.output_origin.0).ok_or(
                SequentialReplayError::Invalid("owned output precedes tile output row origin"),
            )?;
            let local_col = coordinate.1.checked_sub(namespace.output_origin.1).ok_or(
                SequentialReplayError::Invalid("owned output precedes tile output column origin"),
            )?;
            if local_row >= tile.topology.output_shape.0 as u64
                || local_col >= tile.topology.output_shape.1 as u64
            {
                return Err(SequentialReplayError::Invalid(
                    "owned global output is outside its tile replay grid",
                ));
            }
            let local = local_row as usize * tile.topology.output_shape.1 + local_col as usize;
            if found.replace((index, local)).is_some() {
                return Err(SequentialReplayError::Provider(
                    ReplayStatus::InvalidReference,
                    "global output has multiple owning replay tiles",
                ));
            }
        }
        found.ok_or(SequentialReplayError::Provider(
            ReplayStatus::InvalidReference,
            "global output has no owning replay tile",
        ))
    };
    let (target_tile, target_output) = locate(query.target)?;
    let (reference_tile, reference_output) = locate(query.reference)?;
    let target_selection = query
        .ordered_dates
        .iter()
        .map(|&date| (date, target_output))
        .collect::<Vec<_>>();
    let reference_selection = query
        .ordered_dates
        .iter()
        .map(|&date| (date, reference_output))
        .collect::<Vec<_>>();
    let dependency_query = DependencyConeQuery {
        source_rank: query.source_rank,
        microbatch: 1,
        byte_cap: query.byte_cap - wrapper_bytes,
    };
    let mut replay = if target_tile == reference_tile {
        let tile = &mut tiles[target_tile];
        tile.topology
            .replay_reference_difference_covariance_from_provider_with_source_correlation(
                &target_selection,
                &reference_selection,
                dependency_query,
                query.source_correlation,
                query.branch_tolerance,
                tile.provider,
            )?
    } else if target_tile < reference_tile {
        let (left, right) = tiles.split_at_mut(reference_tile);
        let target = &mut left[target_tile];
        let reference = &mut right[0];
        target
            .topology
            .replay_cross_topology_reference_difference_covariance_from_providers_with_source_correlation(
                &target_selection,
                target.provider,
                reference.topology,
                &reference_selection,
                reference.provider,
                dependency_query,
                query.source_correlation,
                query.branch_tolerance,
            )?
    } else {
        let (left, right) = tiles.split_at_mut(target_tile);
        let reference = &mut left[reference_tile];
        let target = &mut right[0];
        target
            .topology
            .replay_cross_topology_reference_difference_covariance_from_providers_with_source_correlation(
                &target_selection,
                target.provider,
                reference.topology,
                &reference_selection,
                reference.provider,
                dependency_query,
                query.source_correlation,
                query.branch_tolerance,
            )?
    };
    replay.reference_signature = global_reference_selection_signature(query);
    let dates = query.ordered_dates.len();
    let mut joint_phase_covariance = Array2::zeros((selected, selected));
    joint_phase_covariance
        .slice_mut(ndarray::s![..dates, ..dates])
        .assign(&replay.target_covariance);
    joint_phase_covariance
        .slice_mut(ndarray::s![dates.., dates..])
        .assign(&replay.reference_covariance);
    joint_phase_covariance
        .slice_mut(ndarray::s![..dates, dates..])
        .assign(&replay.target_reference_covariance);
    joint_phase_covariance
        .slice_mut(ndarray::s![dates.., ..dates])
        .assign(&replay.target_reference_covariance.t());
    let resource_high_water_bytes = checked_add(replay.dependency_cone.total_bytes, wrapper_bytes)?;
    Ok(GlobalReferenceDifferenceCovarianceReplay {
        joint_phase_covariance,
        replay,
        resource_high_water_bytes,
    })
}

/// Estimate the exact conservative global replay reservation without numeric
/// primitive-source reads.
///
/// The existing replay preflight is invoked with zero bytes remaining after
/// the wrapper reservation, so every non-empty dependency cone returns its
/// complete provider-inclusive estimate before source resolution. A gauge-only
/// query may complete with zero source reads and returns its normal bound.
///
/// # Errors
/// Returns the same routing, identity, and topology errors as global replay.
pub fn estimate_global_reference_difference_covariance_from_provider_bundle(
    tiles: &mut [SequentialTileReplayProvider<'_>],
    query: GlobalReferenceCovarianceQuery<'_>,
) -> Result<GlobalReferenceCovarianceResourceEstimate, SequentialReplayError> {
    let (_, wrapper_bytes, _) = global_reference_wrapper_bytes(query)?;
    let preflight_query = GlobalReferenceCovarianceQuery {
        byte_cap: wrapper_bytes,
        ..query
    };
    match replay_global_reference_difference_covariance_from_provider_bundle(tiles, preflight_query)
    {
        Err(SequentialReplayError::Budget(estimate)) => {
            let total_bytes = checked_add(estimate.total_bytes, wrapper_bytes)?;
            Ok(GlobalReferenceCovarianceResourceEstimate {
                replay_bytes: estimate.total_bytes,
                wrapper_bytes,
                total_bytes,
            })
        }
        Ok(replay) => Ok(GlobalReferenceCovarianceResourceEstimate {
            replay_bytes: replay
                .resource_high_water_bytes
                .checked_sub(wrapper_bytes)
                .ok_or(SequentialReplayError::Invalid(
                    "global replay resource receipt underflows its wrapper",
                ))?,
            wrapper_bytes,
            total_bytes: replay.resource_high_water_bytes,
        }),
        Err(error) => Err(error),
    }
}

fn global_reference_wrapper_bytes(
    query: GlobalReferenceCovarianceQuery<'_>,
) -> Result<(usize, u64, u64), SequentialReplayError> {
    let selected =
        query
            .ordered_dates
            .len()
            .checked_mul(2)
            .ok_or(SequentialReplayError::Invalid(
                "global reference selection size overflows usize",
            ))?;
    let selected_u64 = u64::try_from(selected).map_err(|_| {
        SequentialReplayError::Invalid("global reference selection size exceeds u64")
    })?;
    let selection_bytes = checked_mul(selected_u64, size_of::<(GlobalDateId, usize)>() as u64)?;
    let joint_bytes = checked_mul(checked_mul(selected_u64, selected_u64)?, 8)?;
    Ok((
        selected,
        checked_add(selection_bytes, joint_bytes)?,
        joint_bytes,
    ))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn build_covariance_operator_block(
    topology: &SequentialReplayTopology,
    request: &SequentialCovarianceCaptureRequest,
    ministack: MiniStack,
    combined_source: ArrayView3<dolphin_core::Cf64>,
    linked_phase: ArrayView3<dolphin_core::Cf64>,
    phase_replay: &PhaseReplayGrid,
    compression_replay: &CompressionReplayGrid,
    use_evd: bool,
    mut source_resolver: Option<&mut dyn SequentialPrimitiveSourceResolver>,
) -> Result<CovarianceOperatorBlock, SequentialReplayError> {
    let block = topology
        .blocks
        .get(ministack.block_id)
        .ok_or(SequentialReplayError::Invalid(
            "captured ministack has no replay topology block",
        ))?;
    let generation_source_manifest_digest = topology.generation_source_manifest_digest(block.id)?;
    if block.generation as usize != ministack.block_id
        || combined_source.dim().0 != ministack.size()
        || combined_source.dim().1 * combined_source.dim().2 != topology.native_area
        || linked_phase.dim().0 != ministack.size()
        || linked_phase.dim().1 * linked_phase.dim().2 != topology.output_area
        || phase_replay.branch_status.len() != topology.output_area
        || compression_replay.compressed.len() != topology.native_area
    {
        return Err(SequentialReplayError::Invalid(
            "captured replay state does not match its block topology",
        ));
    }

    let source_date_indices = (0..ministack.num_real)
        .map(|offset| {
            u32::try_from(offset)
                .ok()
                .and_then(|value| block.real_date_start.get().checked_add(value))
                .ok_or(SequentialReplayError::Invalid(
                    "captured source date index exceeds u32",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
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

    let source_digest_bytes =
        topology
            .native_area
            .checked_mul(32)
            .ok_or(SequentialReplayError::Invalid(
                "captured source digest dimensions overflow usize",
            ))?;
    let mut source_content_digests = Vec::with_capacity(source_digest_bytes);
    for native in 0..topology.native_area {
        let row = native / topology.native_shape.1;
        let column = native % topology.native_shape.1;
        let digest = primitive_source_content_digest(
            (ministack.num_compressed..ministack.size())
                .map(|component| combined_source[(component, row, column)]),
        );
        source_content_digests.extend_from_slice(&digest);
    }
    let source_ids = source_content_digests
        .chunks_exact(32)
        .enumerate()
        .map(|(native, digest)| {
            let content_digest: &[u8; 32] = digest.try_into().map_err(|_| {
                SequentialReplayError::Invalid("captured source digest width is not SHA-256")
            })?;
            topology
                .source_id_for_content_digest(block.id, native, content_digest)
                .map(SourceId::get)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut source_factor_digests = Vec::with_capacity(source_digest_bytes);
    if let Some(resolver) = source_resolver.as_mut() {
        let identity = resolver.identity_for_block(block)?.clone();
        if identity.source_manifest_digest != generation_source_manifest_digest
            || resolver.identity().source_model_version_digest
                != request.source_model_version_digest
            || identity.source_model_hash.iter().all(|byte| *byte == 0)
            || [
                identity.provider.as_str(),
                identity.provider_version.as_str(),
                identity.model.as_str(),
                identity.model_version.as_str(),
            ]
            .iter()
            .any(|value| value.is_empty())
        {
            return Err(SequentialReplayError::Provider(
                ReplayStatus::SourceIdentityMismatch,
                "capture source resolver identity differs from the operator request",
            ));
        }
        let expected_components = source_date_indices
            .iter()
            .copied()
            .map(u64::from)
            .collect::<Vec<_>>();
        for native in 0..topology.native_area {
            let expected_content: &[u8; 32] = source_content_digests
                [native * 32..(native + 1) * 32]
                .try_into()
                .map_err(|_| {
                    SequentialReplayError::Invalid("captured source digest width is not SHA-256")
                })?;
            if !topology.native_validity[native] {
                let receipt = masked_source_factor_receipt_digest(
                    block,
                    source_ids[native],
                    *expected_content,
                    &identity,
                );
                source_factor_digests.extend_from_slice(&receipt);
                continue;
            }
            let source = resolver.resolve_source(block, native)?;
            let exact_factor_receipt = resolver.factor_receipt_digest(&source)?;
            let row = native / topology.native_shape.1;
            let column = native % topology.native_shape.1;
            let expected_samples = (ministack.num_compressed..ministack.size())
                .map(|component| combined_source[(component, row, column)])
                .collect::<Vec<_>>();
            if source.id.get() != source_ids[native]
                || source.factor.source() != source.id
                || source.content_digest != *expected_content
                || source.samples.as_slice() != Some(expected_samples.as_slice())
                || source.factor.component_ids() != expected_components
                || source.factor.model_hash() != &resolver.identity().source_model_hash
                || exact_factor_receipt.iter().all(|byte| *byte == 0)
            {
                return Err(SequentialReplayError::Provider(
                    ReplayStatus::SourceIdentityMismatch,
                    "capture source bytes or factor identity differ from the production tile",
                ));
            }
            let factor_digest =
                available_source_factor_receipt_digest(exact_factor_receipt, &source);
            if factor_digest.iter().all(|byte| *byte == 0) {
                return Err(SequentialReplayError::Provider(
                    ReplayStatus::SourceModelUnavailable,
                    "capture source factor has no numeric receipt",
                ));
            }
            source_factor_digests.extend_from_slice(&factor_digest);
        }
    } else {
        source_factor_digests.resize(source_digest_bytes, 0);
    }
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

    if phase_replay.realized_support.dim()
        != (
            topology.output_shape.0,
            topology.output_shape.1,
            2 * topology.half_window.y + 1,
            2 * topology.half_window.x + 1,
        )
    {
        return Err(SequentialReplayError::Invalid(
            "captured realized support shape differs from replay topology",
        ));
    }
    let bytes_per_support = topology.support_slots_per_output().div_ceil(8);
    let mut support_bits = Vec::with_capacity(topology.output_area * bytes_per_support);
    for output in 0..topology.output_area {
        let output_row = output / topology.output_shape.1;
        let output_col = output % topology.output_shape.1;
        let realized = phase_replay
            .realized_support
            .slice(ndarray::s![output_row, output_col, .., ..])
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let fixed = topology.support_slot_validity(output)?;
        if realized
            .iter()
            .zip(fixed.iter())
            .any(|(selected, valid)| *selected && !*valid)
        {
            return Err(SequentialReplayError::Invalid(
                "captured realized support is outside fixed native validity",
            ));
        }
        support_bits.extend(pack_bits(&realized));
    }
    let native_validity_bits = pack_bits(&topology.native_validity);

    let phase_angles = (0..topology.output_area)
        .flat_map(|output| {
            let row = output / topology.output_shape.1;
            let column = output % topology.output_shape.1;
            (0..ministack.size()).map(move |component| {
                persisted_phase_angle(component, linked_phase[(component, row, column)])
            })
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
        source_manifest_digest: generation_source_manifest_digest,
        source_model_version_digest: request.source_model_version_digest,
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
        source_content_digests,
        source_factor_digests,
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

/// Digest of the versioned production replay kernel identity.
#[must_use]
pub fn sequential_replay_kernel_digest() -> [u8; 32] {
    Sha256::digest(SEQUENTIAL_SOURCE_DAG_KERNEL_ID.as_bytes()).into()
}

/// Digest the ordered resolver and source-model identity used in source IDs.
#[must_use]
pub fn sequential_source_model_identity_digest(
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

/// Digest of the normalized sequential producer configuration used by replay.
#[must_use]
pub fn sequential_replay_config_digest(cfg: &SequentialConfig) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"dolphinrust:sequential_source_dag_config:v1");
    for value in [
        cfg.ministack_size,
        cfg.max_num_compressed,
        cfg.half_window.y,
        cfg.half_window.x,
        cfg.strides.y,
        cfg.strides.x,
        cfg.output_reference_idx,
    ] {
        digest.update((value as u64).to_le_bytes());
    }
    digest.update(cfg.beta.to_bits().to_le_bytes());
    digest.update(cfg.zero_correlation_threshold.to_bits().to_le_bytes());
    digest.update(cfg.shp_alpha.to_bits().to_le_bytes());
    digest.update([
        u8::from(cfg.use_evd),
        u8::from(cfg.compute_crlb),
        u8::from(cfg.compute_closure_phase),
        u8::from(cfg.compute_average_coherence),
        match cfg.compressed_slc_plan {
            CompressedSlcPlan::AlwaysFirst => 0,
            CompressedSlcPlan::FirstPerMinistack => 1,
            CompressedSlcPlan::LastPerMinistack => 2,
        },
        match cfg.shp_method {
            ShpMethod::Glrt => 0,
            ShpMethod::Ks => 1,
            ShpMethod::Rect => 2,
        },
    ]);
    digest.finalize().into()
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

pub(crate) fn primitive_source_content_digest(samples: impl IntoIterator<Item = Cf64>) -> [u8; 32] {
    let mut digest = Sha256::new();
    for sample in samples {
        digest.update(sample.re.to_le_bytes());
        digest.update(sample.im.to_le_bytes());
    }
    digest.finalize().into()
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

fn packed_bit_value(bits: &[u8], index: usize) -> bool {
    bits[index / 8] & (1 << (index % 8)) != 0
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
    source_manifest_digest: [u8; 32],
    generation: u32,
    native_shape: (usize, usize),
    output_shape: (usize, usize),
) -> u64 {
    let grid = |origin: (u64, u64), shape: (usize, usize), strides: (usize, usize)| {
        CovarianceOperatorGrid {
            row_start: origin.0,
            col_start: origin.1,
            rows: u32::try_from(shape.0).expect("validated replay rows fit u32"),
            cols: u32::try_from(shape.1).expect("validated replay columns fit u32"),
            stride_y: u32::try_from(strides.0).expect("validated replay row stride fits u32"),
            stride_x: u32::try_from(strides.1).expect("validated replay column stride fits u32"),
        }
    };
    let output_strides = (
        native_shape.0 / output_shape.0,
        native_shape.1 / output_shape.1,
    );
    covariance_record_block_id(
        &namespace.burst_id,
        source_manifest_digest,
        namespace.source_model_version_digest,
        generation,
        grid(namespace.native_origin, native_shape, (1, 1)),
        grid(namespace.output_origin, output_shape, output_strides),
        grid(
            namespace.owned_output_origin,
            namespace.owned_output_shape,
            output_strides,
        ),
    )
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

fn persisted_phase_angle(component: usize, phase: Cf64) -> f64 {
    match component {
        0 => 0.0,
        _ => phase.arg(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_reference_phase_canonicalizes_floating_rotation_residue() {
        let shifted = (1..100_000)
            .find_map(|step| {
                let source = Cf64::from_polar(1.234_567, step as f64 / 997.0);
                let shifted = source * Cf64::from_polar(1.0, -source.arg());
                (shifted.arg() != 0.0).then_some(shifted)
            })
            .expect("fixture scan must find a nonzero floating phase residue");
        assert_ne!(shifted.arg(), 0.0);
        assert_eq!(persisted_phase_angle(0, shifted), 0.0);
        assert_eq!(persisted_phase_angle(1, shifted), shifted.arg());
    }
}

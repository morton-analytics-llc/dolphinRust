//! Pipeline orchestration — port of `dolphin/workflows/`.
//!
//! The displacement pipeline (`displacement.py`) in execution order:
//! prepare/group inputs → per-burst wrapped_phase (mask → PS → SHP →
//! covariance → phase-link → compress → ifg network) → stitch bursts →
//! unwrap → timeseries inversion → velocity. Owns the YAML config models
//! (`config/`) and the burst-parallel executor.
#![warn(missing_docs)]

pub mod burst;
pub mod corrections;
pub mod covariance_artifact;
pub mod crop;
pub mod cslc_covariance_source;
pub mod dates;
pub mod displacement;
pub mod fixed_cube;
pub mod provenance;
pub mod sequential;
pub mod sequential_covariance;
pub mod spatial_covariance_artifact;
pub mod spatial_covariance_validation;
mod spatial_reference_covariance_output;
pub mod tiling;
pub mod unwrap_backend;

pub use covariance_artifact::{
    admit_covariance_artifact_disk, admit_covariance_artifact_disk_with_identity_index,
    covariance_artifact_disk_bytes, covariance_artifact_disk_bytes_with_identity_index,
    finalize_covariance_artifact, preflight_covariance_artifact_disk,
    preflight_covariance_artifact_disk_with_identity_index, read_covariance_artifact_manifest,
    read_covariance_artifact_manifest_with_byte_cap, CovarianceArtifactDiskAdmission,
    CovarianceArtifactManifest, CovarianceArtifactTransaction, COVARIANCE_OPERATOR_FILENAME,
    COVARIANCE_OPERATOR_MANIFEST_FILENAME,
};
pub use crop::{BoundsError, ProcessingBoundsProvenance};
pub use cslc_covariance_source::{
    empirical_factor_config, CslcCovarianceManifest, CslcCovarianceResolverMetrics,
    CslcCovarianceSourceResolver, CslcCovarianceValidityReader, CslcManifestResourceEstimate,
    CSLC_COVARIANCE_SOURCE_MODEL, CSLC_COVARIANCE_SOURCE_MODEL_VERSION,
    CSLC_COVARIANCE_SOURCE_PROVIDER, CSLC_COVARIANCE_SOURCE_PROVIDER_VERSION,
};
pub use displacement::{
    run_displacement, run_displacement_resumable, run_displacement_with_output_policy,
    update_displacement, DisplacementOutput, DisplacementOutputPolicy, DisplacementState,
    VelocityEstimator,
};
pub use fixed_cube::{write_fixed_cube_bundle, FixedCubeReceipt};
pub use provenance::{
    assemble_geometry_provenance, assemble_geometry_provenance_with_bounds,
    assemble_geometry_provenance_with_coverage, write_geometry_provenance, BurstCoverageProvenance,
    FieldProvenance, GeometryProvenance, InputCoverageProvenance, GEOMETRY_PROVENANCE_FILENAME,
    INPUT_COVERAGE_POLICY_VERSION,
};
pub use sequential::{
    run_sequential, run_sequential_masked, run_sequential_masked_with_covariance_capture,
    run_sequential_masked_with_covariance_capture_and_source_factors, run_sequential_resumable,
    run_sequential_resumable_masked, run_sequential_with_covariance_capture,
    run_sequential_with_covariance_capture_and_source_factors, update_sequential,
    update_sequential_masked, SequentialConfig, SequentialOutput, SequentialState,
};
pub use sequential_covariance::{
    empirical_source_factor_receipt_digest,
    estimate_global_reference_difference_covariance_from_provider_bundle,
    replay_global_reference_difference_covariance_from_provider_bundle,
    sequential_replay_config_digest, sequential_replay_kernel_digest,
    sequential_source_model_identity_digest, CovarianceArtifactReplayMetrics,
    CovarianceArtifactReplayProvider, DependencyConeEstimate, DependencyConeQuery,
    EffectiveLooksReplay, GlobalBlockId, GlobalDateId, GlobalReferenceCovarianceQuery,
    GlobalReferenceCovarianceResourceEstimate, GlobalReferenceDifferenceCovarianceReplay,
    ReferenceDifferenceCovarianceReplay, ReferenceSpecificExecutionMode,
    ReferenceSpecificReplayScope, ReplayBackend, ReplayExecutionScope, ReplayIdNamespace,
    ReplayStatus, ResolvedCompressionReplay, ResolvedPhaseReplay, ResolvedPrimitiveSource,
    SequentialCovarianceCaptureRequest, SequentialPrimitiveSourceResolver, SequentialReplayBlock,
    SequentialReplayBuildIdentity, SequentialReplayError, SequentialReplayTopology,
    SequentialSourceProviderIdentity, SequentialSourceReplayProvider, SequentialTileReplayProvider,
    SourceCorrelationModel, SpatialCovarianceStatus, TemporalCovarianceReplay,
    SEQUENTIAL_SOURCE_DAG_KERNEL_ID, SEQUENTIAL_SOURCE_DAG_METHOD,
};
pub use spatial_covariance_artifact::{
    finalize_spatial_reference_covariance_artifact,
    read_spatial_reference_covariance_artifact_manifest,
    SpatialReferenceCovarianceArtifactManifest, SpatialReferenceCovarianceArtifactTransaction,
    SPATIAL_REFERENCE_COVARIANCE_FILENAME, SPATIAL_REFERENCE_COVARIANCE_MANIFEST_FILENAME,
};
pub use unwrap_backend::{NativeUnwrapBackend, SnaphuBackend, TophuBackend, UnwrapBackend};

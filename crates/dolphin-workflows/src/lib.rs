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
pub mod dates;
pub mod displacement;
pub mod provenance;
pub mod sequential;
pub mod sequential_covariance;
pub mod tiling;
pub mod unwrap_backend;

pub use covariance_artifact::{
    admit_covariance_artifact_disk, covariance_artifact_disk_bytes, finalize_covariance_artifact,
    preflight_covariance_artifact_disk, read_covariance_artifact_manifest,
    CovarianceArtifactDiskAdmission, CovarianceArtifactManifest, COVARIANCE_OPERATOR_FILENAME,
    COVARIANCE_OPERATOR_MANIFEST_FILENAME,
};
pub use crop::{BoundsError, ProcessingBoundsProvenance};
pub use displacement::{
    run_displacement, run_displacement_resumable, run_displacement_with_output_policy,
    update_displacement, DisplacementOutput, DisplacementOutputPolicy, DisplacementState,
    VelocityEstimator,
};
pub use provenance::{
    assemble_geometry_provenance, assemble_geometry_provenance_with_bounds,
    assemble_geometry_provenance_with_coverage, write_geometry_provenance, BurstCoverageProvenance,
    FieldProvenance, GeometryProvenance, InputCoverageProvenance, GEOMETRY_PROVENANCE_FILENAME,
    INPUT_COVERAGE_POLICY_VERSION,
};
pub use sequential::{
    run_sequential, run_sequential_masked, run_sequential_masked_with_covariance_capture,
    run_sequential_resumable, run_sequential_resumable_masked,
    run_sequential_with_covariance_capture, update_sequential, update_sequential_masked,
    SequentialConfig, SequentialOutput, SequentialState,
};
pub use sequential_covariance::{
    CovarianceArtifactReplayMetrics, CovarianceArtifactReplayProvider, DependencyConeEstimate,
    DependencyConeQuery, GlobalBlockId, GlobalDateId, ReplayBackend, ReplayExecutionScope,
    ReplayIdNamespace, ReplayStatus, ResolvedCompressionReplay, ResolvedPhaseReplay,
    ResolvedPrimitiveSource, SequentialCovarianceCaptureRequest, SequentialPrimitiveSourceResolver,
    SequentialReplayBlock, SequentialReplayError, SequentialReplayTopology,
    SequentialSourceProviderIdentity, SequentialSourceReplayProvider, TemporalCovarianceReplay,
    SEQUENTIAL_SOURCE_DAG_METHOD,
};
pub use unwrap_backend::{NativeUnwrapBackend, SnaphuBackend, TophuBackend, UnwrapBackend};
